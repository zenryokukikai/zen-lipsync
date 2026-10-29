// lipf.c - ZEN-LipSync frame model on the CPU in float32: the WebGPU frame model's arithmetic (src/runtime/wgsl.ts,
// buildFrame) on the same weights (frame.i8.wgw dequantised once at load, or frame.wgw), one thread per call.
// Built twice by native/node/build.sh: with -mavx512f (lipf_avx512_*) and with -mavx2 -mfma (lipf_avx2_*).
// Activations are NHWC. Each 3x3 conv reads a zero-bordered copy of its input with the input activation applied
// (none / SiLU / ReLU) and writes its raw output (+ the residual), or for the up-sampling layers DepthToSpace(2, CRD)
// directly: their output channels are reordered at load so that a vector holds one sub-pixel of consecutive channels.
// The last layer (8 -> 3 channels, times 64 times the window) runs on 8 planes, vectorised along x.
// Weights are read only after create(); the scratch buffers are per thread, so run() may be called from several
// threads at once, also on different models.
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__)
#define VW 16
typedef __m512 V;
#define VLD(p) _mm512_loadu_ps(p)
#define VST(p, v) _mm512_storeu_ps(p, v)
#define VSET1(x) _mm512_set1_ps(x)
#define VZERO() _mm512_setzero_ps()
#define VFMA(a, b, c) _mm512_fmadd_ps(a, b, c)
#define VADD(a, b) _mm512_add_ps(a, b)
#define VSUB(a, b) _mm512_sub_ps(a, b)
#define VMUL(a, b) _mm512_mul_ps(a, b)
#define VDIV(a, b) _mm512_div_ps(a, b)
#define VMAX(a, b) _mm512_max_ps(a, b)
#define VMIN(a, b) _mm512_min_ps(a, b)
#define VFLOOR(a) _mm512_roundscale_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
#define VPOW2(n) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvttps_epi32(n), _mm512_set1_epi32(127)), 23))
#define KNAME(x) lipf_avx512_##x
#elif defined(__AVX2__) && defined(__FMA__)
#define VW 8
typedef __m256 V;
#define VLD(p) _mm256_loadu_ps(p)
#define VST(p, v) _mm256_storeu_ps(p, v)
#define VSET1(x) _mm256_set1_ps(x)
#define VZERO() _mm256_setzero_ps()
#define VFMA(a, b, c) _mm256_fmadd_ps(a, b, c)
#define VADD(a, b) _mm256_add_ps(a, b)
#define VSUB(a, b) _mm256_sub_ps(a, b)
#define VMUL(a, b) _mm256_mul_ps(a, b)
#define VDIV(a, b) _mm256_div_ps(a, b)
#define VMAX(a, b) _mm256_max_ps(a, b)
#define VMIN(a, b) _mm256_min_ps(a, b)
#define VFLOOR(a) _mm256_floor_ps(a)
#define VPOW2(n) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127)), 23))
#define KNAME(x) lipf_avx2_##x
#else
#error "build with -mavx512f, or -mavx2 -mfma"
#endif
#define EXPORT __attribute__((visibility("default")))

enum { A_NONE, A_SILU, A_RELU };
enum { M_STORE, M_D2S };
#define NCONV 14   // the 3x3 convs before the last layer
#define SLACK 26   // zero columns right of each padded row: a tile of up to 24 pixels reads past the image
#define BOX_H 160
#define BOX_W 352
#define FPW (BOX_W + 2 + 2 * 16)

// inp, body.0-5, mid, up.0.0, up.0.1, up.1.0, up.1.1, up.2.0, up.2.1, out (src/runtime/wgsl.ts buildFrame)
static const struct { int C, O, H, W, act, mode; } SPEC[NCONV + 1] = {
  { 40, 64, 20, 44, A_NONE, M_STORE },
  { 64, 64, 20, 44, A_SILU, M_STORE }, { 64, 64, 20, 44, A_SILU, M_STORE }, { 64, 64, 20, 44, A_SILU, M_STORE },
  { 64, 64, 20, 44, A_SILU, M_STORE }, { 64, 64, 20, 44, A_SILU, M_STORE }, { 64, 64, 20, 44, A_SILU, M_STORE },
  { 64, 64, 20, 44, A_SILU, M_STORE },
  { 64, 64, 20, 44, A_NONE, M_STORE }, { 64, 128, 20, 44, A_RELU, M_D2S },
  { 32, 32, 40, 88, A_NONE, M_STORE }, { 32, 64, 40, 88, A_RELU, M_D2S },
  { 16, 16, 80, 176, A_NONE, M_STORE }, { 16, 32, 80, 176, A_RELU, M_D2S },
  { 8, 3, 160, 352, A_RELU, 0 },
};

typedef struct { int C, O, H, W, mode; float *wp, *b; } Conv;  // wp[(tap * C + c) * O + o'], b[o']
typedef struct {
  float *g0w, *g0b, *g2w, *g2b;  // [I][O] (transposed from the file's [O][I]), bias [O]
  Conv cv[NCONV];
  float wout[3 * 8 * 9], bout[3], *win64;
} LipF;

static void *amalloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, (n + 63) / 64 * 64)) return NULL; return p; }

/** Floats of the weights blob: zfc0.w, zfc0.b, zfc2.w, zfc2.b, then w and b of each conv in SPEC order, then win. */
static size_t blob_size(void) {
  size_t n = 256 * 640 + 256 + 7040 * 256 + 7040;
  for (int l = 0; l <= NCONV; l++) { size_t o4 = (SPEC[l].O + 3) / 4; n += o4 * SPEC[l].C * 9 * 4 + o4 * 4; }
  return n + BOX_H * BOX_W;
}

EXPORT void KNAME(destroy)(LipF *m) {
  if (!m) return;
  free(m->g0w); free(m->g0b); free(m->g2w); free(m->g2b); free(m->win64);
  for (int l = 0; l < NCONV; l++) { free(m->cv[l].wp); free(m->cv[l].b); }
  free(m);
}

static float *gemm_pack(const float *w, int O, int I) {
  float *t = amalloc((size_t)O * I * 4);
  if (t) for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) t[(size_t)i * O + o] = w[(size_t)o * I + i];
  return t;
}

/**
 * blob: n floats as blob_size() lists, in the layout of the WebGPU buffers (gemm w [O][I]; conv w [O4][C][9][4] with
 * output channel o4 * 4 + j in lane j; conv b [O4 * 4]). NULL when n is not the expected size or memory runs out.
 */
EXPORT LipF *KNAME(create)(const float *blob, size_t n) {
  if (n != blob_size()) return NULL;
  LipF *m = calloc(1, sizeof *m);
  if (!m) return NULL;
  const float *p = blob;
  m->g0w = gemm_pack(p, 256, 640); p += 256 * 640;
  m->g0b = amalloc(256 * 4); if (m->g0b) memcpy(m->g0b, p, 256 * 4); p += 256;
  m->g2w = gemm_pack(p, 7040, 256); p += 7040 * 256;
  m->g2b = amalloc(7040 * 4); if (m->g2b) memcpy(m->g2b, p, 7040 * 4); p += 7040;
  if (!m->g0w || !m->g0b || !m->g2w || !m->g2b) { KNAME(destroy)(m); return NULL; }
  for (int l = 0; l <= NCONV; l++) {
    int C = SPEC[l].C, O = SPEC[l].O, O4 = (O + 3) / 4;
    const float *w = p, *b = p + (size_t)O4 * C * 36;
    p = b + O4 * 4;
    if (l == NCONV) {  // last layer: 3 output channels, plain arrays
      for (int o = 0; o < 3; o++) {
        m->bout[o] = b[o];
        for (int c = 0; c < 8; c++) for (int t = 0; t < 9; t++) m->wout[(o * 8 + c) * 9 + t] = w[(c * 9 + t) * 4 + o];
      }
      break;
    }
    Conv *L = &m->cv[l];
    L->C = C; L->O = O; L->H = SPEC[l].H; L->W = SPEC[l].W; L->mode = SPEC[l].mode;
    L->wp = amalloc((size_t)9 * C * O * 4); L->b = amalloc((size_t)O * 4);
    if (!L->wp || !L->b) { KNAME(destroy)(m); return NULL; }
    for (int oc = 0; oc < O; oc++) {
      // DepthToSpace(2, CRD): output channel oc = cc * 4 + j is sub-pixel j of channel cc; stored at j * (O / 4) + cc
      int op = L->mode == M_D2S ? (oc % 4) * (O / 4) + oc / 4 : oc;
      L->b[op] = b[oc];
      for (int c = 0; c < C; c++) for (int t = 0; t < 9; t++) L->wp[((size_t)t * C + c) * O + op] = w[(((size_t)(oc / 4) * C + c) * 9 + t) * 4 + oc % 4];
    }
  }
  m->win64 = amalloc(BOX_H * BOX_W * 4);
  if (!m->win64) { KNAME(destroy)(m); return NULL; }
  for (int i = 0; i < BOX_H * BOX_W; i++) m->win64[i] = p[i] * 64.0f;
  return m;
}

// ---- per-thread scratch ----
typedef struct {
  float *P40, *P64, *P32, *P16;  // zero-bordered activated inputs (20x44x40, 20x44x64, 40x88x32, 80x176x16)
  float *X0, *B0, *E, *G, *J, *L, *O2, *M, *U00, *D1, *U10, *D2, *U20, *D3, *F;
  float *racc;  // partial sums of one output row (widest: 176 + 24 pixels x 64 channels)
  float g0[256], g1[7040];
} Scratch;
static __thread Scratch *S;

static size_t padded(int H, int W, int C) { return (size_t)(H + 2) * (W + 2 + SLACK) * C; }
static float *zeros(size_t n) { float *p = amalloc(n * 4); if (p) memset(p, 0, n * 4); return p; }
static Scratch *scratch(void) {
  if (S) return S;
  Scratch *s = calloc(1, sizeof *s);
  if (!s) return NULL;
  s->P40 = zeros(padded(20, 44, 40)); s->P64 = zeros(padded(20, 44, 64)); s->P32 = zeros(padded(40, 88, 32)); s->P16 = zeros(padded(80, 176, 16));
  float **raw64[] = { &s->X0, &s->B0, &s->E, &s->G, &s->J, &s->L, &s->O2, &s->M, &s->U00 };
  for (unsigned i = 0; i < sizeof raw64 / sizeof *raw64; i++) *raw64[i] = zeros(880 * 64);
  s->D1 = zeros(3520 * 32); s->U10 = zeros(3520 * 32); s->D2 = zeros(14080 * 16); s->U20 = zeros(14080 * 16); s->D3 = zeros(56320 * 8);
  s->F = zeros((size_t)8 * (BOX_H + 2) * FPW);
  s->racc = zeros((size_t)(176 + 24) * 128);
  float *all[] = { s->P40, s->P64, s->P32, s->P16, s->X0, s->B0, s->E, s->G, s->J, s->L, s->O2, s->M, s->U00, s->D1, s->U10, s->D2, s->U20, s->D3, s->F, s->racc };
  for (unsigned i = 0; i < sizeof all / sizeof *all; i++) if (!all[i]) return NULL;  // out of memory: run() reports it
  return S = s;
}

// exp (Cephes polynomial, |x| clamped to 88: relative error ~1e-7)
static inline V vexp(V x) {
  x = VMAX(VMIN(x, VSET1(88.0f)), VSET1(-87.0f));
  V fx = VFLOOR(VFMA(x, VSET1(1.44269504088896341f), VSET1(0.5f)));
  x = VSUB(x, VMUL(fx, VSET1(0.693359375f)));
  x = VSUB(x, VMUL(fx, VSET1(-2.12194440e-4f)));
  V y = VSET1(1.9875691500e-4f);
  y = VFMA(y, x, VSET1(1.3981999507e-3f));
  y = VFMA(y, x, VSET1(8.3334519073e-3f));
  y = VFMA(y, x, VSET1(4.1665795894e-2f));
  y = VFMA(y, x, VSET1(1.6666665459e-1f));
  y = VFMA(y, x, VSET1(5.0000001201e-1f));
  y = VFMA(y, VMUL(x, x), VADD(x, VSET1(1.0f)));
  return VMUL(y, VPOW2(fx));
}
static inline V act(V v, int a) {
  if (a == A_SILU) return VDIV(v, VADD(VSET1(1.0f), vexp(VSUB(VZERO(), v))));
  if (a == A_RELU) return VMAX(v, VZERO());
  return v;
}

/** src NHWC [H][W][C] -> the interior of the zero-bordered dst, with the activation applied. */
static void act_pad(const float *src, int H, int W, int C, int a, float *dst) {
  int PW = W + 2 + SLACK, n = W * C;
  for (int y = 0; y < H; y++) {
    const float *s = src + (size_t)y * n; float *d = dst + ((size_t)(y + 1) * PW + 1) * C;
    for (int i = 0; i < n; i += VW) VST(d + i, act(VLD(s + i), a));
  }
}

/** y[o] = act(scale * sum_i x[i] w[i][o] + b[o]); O a multiple of 8 vectors. */
static void gemm(const float *w, const float *b, int I, int O, const float *x, float *y, float scale, int a) {
  for (int o0 = 0; o0 < O; o0 += 8 * VW) {
    V acc[8];
    for (int n = 0; n < 8; n++) acc[n] = VZERO();
    for (int i = 0; i < I; i++) {
      V xv = VSET1(x[i]); const float *wr = w + (size_t)i * O + o0;
      for (int n = 0; n < 8; n++) acc[n] = VFMA(xv, VLD(wr + n * VW), acc[n]);
    }
    for (int n = 0; n < 8; n++) VST(y + o0 + n * VW, act(VFMA(acc[n], VSET1(scale), VLD(b + o0 + n * VW)), a));
  }
}

/** Bias, residual and store of one tile: vector t * NB + n of acc = pixel x0 + t, output vector nb0 + n. */
static inline void epilogue(const Conv *L, int y, int x0, int nv, int nb0, int NB, const float *acc, float *dst, const float *res) {
  int O = L->O, W = L->W;
  for (int t = 0; t < nv; t++) for (int n = 0; n < NB; n++) {
    int o = (nb0 + n) * VW, x = x0 + t;
    V v = VADD(VLD(acc + (t * NB + n) * VW), VLD(L->b + o));
    if (L->mode == M_STORE) {
      size_t i = ((size_t)y * W + x) * O + o;
      if (res) v = VADD(v, VLD(res + i));
      VST(dst + i, v);
    } else {
      float tmp[VW] __attribute__((aligned(64))); VST(tmp, v);
      int Cd = O / 4, run = Cd < VW ? Cd : VW;
      for (int r = 0; r < VW; r += run) {
        int oo = o + r, j = oo / Cd, cc = oo % Cd;
        memcpy(dst + (((size_t)(2 * y + (j >> 1)) * (2 * W) + 2 * x + (j & 1)) * Cd + cc), tmp + r, (size_t)run * 4);
      }
    }
  }
}

// One output row, NBc output vectors from nb0, TXc pixels per tile. The 9 taps run one after another over the whole
// row: a tap's weights (C x NBc vectors) stay in L1 while every tile of the row uses them, and the partial sums of the
// row wait in racc (L1) between taps. 24 accumulators (AVX-512) or 12 (AVX2) per tile.
#define CONV_ROW(NBc, TXc)                                                                                          \
  for (int tap = 0; tap < 9; tap++) {                                                                               \
    const float *wb = L->wp + (size_t)tap * C * O + nb0 * VW;                                                       \
    for (int x0 = 0; x0 < W; x0 += TXc) {                                                                           \
      V acc[TXc * NBc];                                                                                             \
      float *ra = racc + (size_t)x0 * NBc * VW;                                                                     \
      if (tap == 0) { _Pragma("GCC unroll 32") for (int i = 0; i < TXc * NBc; i++) acc[i] = VZERO(); }              \
      else { _Pragma("GCC unroll 32") for (int i = 0; i < TXc * NBc; i++) acc[i] = VLD(ra + i * VW); }              \
      const float *ip = in + ((size_t)(y + tap / 3) * PW + x0 + tap % 3) * C;                                       \
      for (int c = 0; c < C; c++) {                                                                                 \
        V w[NBc];                                                                                                   \
        _Pragma("GCC unroll 4") for (int n = 0; n < NBc; n++) w[n] = VLD(wb + (size_t)c * O + n * VW);              \
        _Pragma("GCC unroll 32") for (int t = 0; t < TXc; t++) {                                                    \
          V b = VSET1(ip[(size_t)t * C + c]);                                                                       \
          _Pragma("GCC unroll 4") for (int n = 0; n < NBc; n++) acc[t * NBc + n] = VFMA(b, w[n], acc[t * NBc + n]); \
        }                                                                                                           \
      }                                                                                                             \
      _Pragma("GCC unroll 32") for (int i = 0; i < TXc * NBc; i++) VST(ra + i * VW, acc[i]);                        \
      if (tap == 8) epilogue(L, y, x0, W - x0 < TXc ? W - x0 : TXc, nb0, NBc, ra, dst, res);                         \
    }                                                                                                               \
  }

// Narrow layers (all 9 taps' weights fit L1): the 9 taps inside the tile, accumulators kept in registers throughout.
#define CONV_TILE(NBc, TXc)                                                                                         \
  for (int x0 = 0; x0 < W; x0 += TXc) {                                                                             \
    V acc[TXc * NBc];                                                                                               \
    _Pragma("GCC unroll 32") for (int i = 0; i < TXc * NBc; i++) acc[i] = VZERO();                                  \
    for (int tap = 0; tap < 9; tap++) {                                                                             \
      const float *ip = in + ((size_t)(y + tap / 3) * PW + x0 + tap % 3) * C;                                       \
      const float *wb = L->wp + (size_t)tap * C * O + nb0 * VW;                                                     \
      for (int c = 0; c < C; c++) {                                                                                 \
        V w[NBc];                                                                                                   \
        _Pragma("GCC unroll 4") for (int n = 0; n < NBc; n++) w[n] = VLD(wb + (size_t)c * O + n * VW);              \
        _Pragma("GCC unroll 32") for (int t = 0; t < TXc; t++) {                                                    \
          V b = VSET1(ip[(size_t)t * C + c]);                                                                       \
          _Pragma("GCC unroll 4") for (int n = 0; n < NBc; n++) acc[t * NBc + n] = VFMA(b, w[n], acc[t * NBc + n]); \
        }                                                                                                           \
      }                                                                                                             \
    }                                                                                                               \
    _Pragma("GCC unroll 32") for (int i = 0; i < TXc * NBc; i++) VST(racc + i * VW, acc[i]);                        \
    epilogue(L, y, x0, W - x0 < TXc ? W - x0 : TXc, nb0, NBc, racc, dst, res);                                      \
  }

/** in: the zero-bordered activated input; dst: raw NHWC output (or the DepthToSpace image); res: added (STORE only). */
static void conv(const Conv *L, const float *in, float *dst, const float *res, float *racc) {
  int H = L->H, W = L->W, C = L->C, O = L->O, PW = W + 2 + SLACK, NBt = O / VW;
  if ((size_t)9 * C * O * 4 <= 20 * 1024) {
    for (int y = 0; y < H; y++)
      for (int nb0 = 0; nb0 < NBt;) {
#if VW == 16
        if (NBt - nb0 >= 2) { CONV_TILE(2, 12) nb0 += 2; }
        else { CONV_TILE(1, 24) nb0 += 1; }
#else
        if (NBt - nb0 >= 2) { CONV_TILE(2, 6) nb0 += 2; }
        else { CONV_TILE(1, 12) nb0 += 1; }
#endif
      }
    return;
  }
  for (int y = 0; y < H; y++)
    for (int nb0 = 0; nb0 < NBt;) {
#if VW == 16
      if (NBt - nb0 >= 4) { CONV_ROW(4, 6) nb0 += 4; }
      else if (NBt - nb0 >= 2) { CONV_ROW(2, 11) nb0 += 2; }
      else { CONV_ROW(1, 22) nb0 += 1; }
#else
      if (NBt - nb0 >= 2) { CONV_ROW(2, 6) nb0 += 2; }
      else { CONV_ROW(1, 12) nb0 += 1; }
#endif
    }
}

/** Last layer: ReLU(D3) as 8 zero-bordered planes, 3x3 conv to 3 channels + bias, times 64 times the window; NCHW out. */
static void tail(const LipF *m, const float *D3, float *F, float *out) {
  for (int y = 0; y < BOX_H; y++)
    for (int x = 0; x < BOX_W; x++) {
      const float *s = D3 + ((size_t)y * BOX_W + x) * 8;
      for (int c = 0; c < 8; c++) F[(size_t)c * (BOX_H + 2) * FPW + (size_t)(y + 1) * FPW + x + 1] = s[c] > 0 ? s[c] : 0;
    }
  for (int y = 0; y < BOX_H; y++)
    for (int x0 = 0; x0 < BOX_W; x0 += 2 * VW) {
      V a[3][2];
      for (int o = 0; o < 3; o++) a[o][0] = a[o][1] = VSET1(m->bout[o]);
      for (int c = 0; c < 8; c++) for (int ky = 0; ky < 3; ky++) {
        const float *r = F + (size_t)c * (BOX_H + 2) * FPW + (size_t)(y + ky) * FPW + x0;
        _Pragma("GCC unroll 3") for (int kx = 0; kx < 3; kx++) {
          V v0 = VLD(r + kx), v1 = VLD(r + kx + VW);
          _Pragma("GCC unroll 3") for (int o = 0; o < 3; o++) {
            V w = VSET1(m->wout[(o * 8 + c) * 9 + ky * 3 + kx]);
            a[o][0] = VFMA(v0, w, a[o][0]); a[o][1] = VFMA(v1, w, a[o][1]);
          }
        }
      }
      const float *wr = m->win64 + y * BOX_W + x0;
      for (int o = 0; o < 3; o++) {
        float *d = out + (size_t)o * BOX_H * BOX_W + y * BOX_W + x0;
        VST(d, VMUL(a[o][0], VLD(wr))); VST(d + VW, VMUL(a[o][1], VLD(wr + VW)));
      }
    }
}

/** One frame: feat [32,20,44], z [640] -> out [3,160,352] (as WgslLips.frame). 0, or -1 when out of memory. */
EXPORT int KNAME(run)(const LipF *m, const float *feat, const float *z, float *out) {
  Scratch *s = scratch();
  if (!s) return -1;
  gemm(m->g0w, m->g0b, 640, 256, z, s->g0, 20.0f, A_SILU);
  gemm(m->g2w, m->g2b, 256, 7040, s->g0, s->g1, 1.0f, A_NONE);
  // concat (feat: 32 channels, zfc2: 8 channels) into the zero-bordered NHWC input of inp
  for (int p = 0; p < 880; p++) {
    float *d = s->P40 + ((size_t)(p / 44 + 1) * (44 + 2 + SLACK) + p % 44 + 1) * 40;
    for (int c = 0; c < 32; c++) d[c] = feat[c * 880 + p];
    for (int c = 0; c < 8; c++) d[32 + c] = s->g1[c * 880 + p];
  }
  const Conv *cv = m->cv;
  conv(&cv[0], s->P40, s->X0, NULL, s->racc);
  act_pad(s->X0, 20, 44, 64, A_SILU, s->P64); conv(&cv[1], s->P64, s->B0, NULL, s->racc);
  act_pad(s->B0, 20, 44, 64, A_SILU, s->P64); conv(&cv[2], s->P64, s->E, s->X0, s->racc);
  act_pad(s->E, 20, 44, 64, A_SILU, s->P64); conv(&cv[3], s->P64, s->G, NULL, s->racc);
  act_pad(s->G, 20, 44, 64, A_SILU, s->P64); conv(&cv[4], s->P64, s->J, s->E, s->racc);
  act_pad(s->J, 20, 44, 64, A_SILU, s->P64); conv(&cv[5], s->P64, s->L, NULL, s->racc);
  act_pad(s->L, 20, 44, 64, A_SILU, s->P64); conv(&cv[6], s->P64, s->O2, s->J, s->racc);
  act_pad(s->O2, 20, 44, 64, A_SILU, s->P64); conv(&cv[7], s->P64, s->M, NULL, s->racc);
  act_pad(s->M, 20, 44, 64, A_NONE, s->P64); conv(&cv[8], s->P64, s->U00, NULL, s->racc);
  act_pad(s->U00, 20, 44, 64, A_RELU, s->P64); conv(&cv[9], s->P64, s->D1, NULL, s->racc);
  act_pad(s->D1, 40, 88, 32, A_NONE, s->P32); conv(&cv[10], s->P32, s->U10, NULL, s->racc);
  act_pad(s->U10, 40, 88, 32, A_RELU, s->P32); conv(&cv[11], s->P32, s->D2, NULL, s->racc);
  act_pad(s->D2, 80, 176, 16, A_NONE, s->P16); conv(&cv[12], s->P16, s->U20, NULL, s->racc);
  act_pad(s->U20, 80, 176, 16, A_RELU, s->P16); conv(&cv[13], s->P16, s->D3, NULL, s->racc);
  tail(m, s->D3, s->F, out);
  return 0;
}

/** Floats the weights blob of create() must have. */
EXPORT size_t KNAME(blob_size)(void) { return blob_size(); }
