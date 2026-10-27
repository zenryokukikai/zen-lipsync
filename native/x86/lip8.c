// lip8.c - ZEN-LipSync frame model on the CPU: int8 with AVX-512 VNNI, single thread
// (per-channel s8 weights, u8 activations, the last conv + *64 + *window in fp32).
// Single thread. NHWC u8 activations with a 1-pixel border holding the zero point, so the zero-point term is a
// per-output constant (bias - zx*sum(w)); vpdpbusd over 4 input channels x 16 output channels per instruction.
// DepthToSpace is fused into the conv stores; SiLU (QLinearSigmoid*QLinearMul) is a 256-entry table (VBMI permute);
// the residual adds are QLinearAdd in float; the tail (DQ, D2S, ReLU, 8->3 conv, *win) is fp32 AVX-512.
// build: gcc -O3 -march=icelake-server -o lip8 lip8.c -lm   (bench / tool)
// library (Node addon, native/node/build.sh): -DLIP8_LIB, exports lip8_avx512_init / lip8_avx512_run. The scratch
// buffers are per thread (LIP_TLS), the packed weights are shared and read only after init.
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef LIP8_LIB
#define LIP_TLS __thread
#else
#define LIP_TLS
#endif

typedef struct { int O, C, zx, zy, K, OB, C4; float sx, sy; int8_t *w; float *sw; int32_t *bq;
                 uint8_t *wp; int32_t *biasc; float *mult; } L8;
static L8 Ls[16];
static uint8_t lut[8][256] __attribute__((aligned(64)));
static float addr[3][2]; static int32_t addz[3][3];
static float zsz, zcval, zsmq, fsf, fscc; static int32_t zzz, zzmq, fzf, fzcc;
static float Wout[3 * 8 * 9], Bout[3], *win;
static uint8_t featlut[256];

static const uint8_t *rp, *rend; static int rerr;
static void rd(void *p, size_t n) { if (rp + n > rend) { rerr = 1; memset(p, 0, n); return; } memcpy(p, rp, n); rp += n; }
static void *amalloc(size_t n) { void *p = aligned_alloc(64, (n + 63) / 64 * 64); if (!p) exit(2); return p; }

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) exit(2); return p; }
/** Reads frame.lip8 from memory; 0 or a negative error code. */
static int load(const uint8_t *buf, size_t n) {
  rp = buf; rend = buf + n; rerr = 0;
  char mg[4]; int32_t ver; rd(mg, 4); rd(&ver, 4); if (memcmp(mg, "LIP8", 4) || ver != 2) return -1;
  for (int l = 0; l < 16; l++) {
    L8 *L = &Ls[l]; int32_t h[3]; rd(h, 12); L->O = h[0]; L->C = h[1]; L->zx = h[2]; rd(&L->sx, 4);
    L->K = l < 14 ? L->C * 9 : L->C;
    if (rerr || L->O <= 0 || L->C <= 0 || (size_t)L->O * L->K > (size_t)(rend - rp)) return -3;
    L->w = xmalloc((size_t)L->O * L->K); rd(L->w, (size_t)L->O * L->K);
    L->sw = xmalloc(L->O * 4); rd(L->sw, L->O * 4); L->bq = xmalloc(L->O * 4); rd(L->bq, L->O * 4);
    rd(&L->zy, 4); rd(&L->sy, 4);
  }
  for (int i = 0; i < 8; i++) rd(lut[i], 256);
  for (int i = 0; i < 3; i++) { rd(addr[i], 8); rd(addz[i], 12); }
  rd(&zsz, 4); rd(&zzz, 4); rd(&zcval, 4); rd(&zsmq, 4); rd(&zzmq, 4);
  rd(&fsf, 4); rd(&fzf, 4); rd(&fscc, 4); rd(&fzcc, 4);
  rd(Wout, sizeof Wout); rd(Bout, sizeof Bout); win = amalloc(160 * 352 * 4); rd(win, 160 * 352 * 4);
  char e[4]; rd(e, 4); if (memcmp(e, "END8", 4)) return -2;
  return rerr ? -3 : 0;
}

// conv weights: wp[((tap*C4 + c4)*OB + ob)*64 + j*4 + k] = w[ob*16+j][(c4*4+k)*9 + tap]
// gemm weights: wp[(i4*O + o)*4 + k] = w[o][i4*4+k]
static int pack(void) {
  for (int l = 0; l < 16; l++) {
    L8 *L = &Ls[l]; int O = L->O, C = L->C; L->OB = O / 16; L->C4 = C / 4;
    if (O % 16 || C % 4) return -10 - l;
    L->biasc = amalloc(O * 4); L->mult = amalloc(O * 4);
    for (int o = 0; o < O; o++) {
      long s = 0; for (int k = 0; k < L->K; k++) s += L->w[(size_t)o * L->K + k];
      L->biasc[o] = L->bq[o] - L->zx * (int32_t)s; L->mult[o] = L->sx * L->sw[o] / L->sy;
    }
    size_t n = (size_t)O * L->K; L->wp = amalloc(n);
    if (l < 14) {
      for (int tap = 0; tap < 9; tap++) for (int c4 = 0; c4 < L->C4; c4++) for (int ob = 0; ob < L->OB; ob++)
        for (int j = 0; j < 16; j++) for (int k = 0; k < 4; k++)
          L->wp[(((size_t)tap * L->C4 + c4) * L->OB + ob) * 64 + j * 4 + k] = (uint8_t)L->w[(size_t)(ob * 16 + j) * L->K + (c4 * 4 + k) * 9 + tap];
    } else {
      for (int i4 = 0; i4 < L->C4; i4++) for (int o = 0; o < O; o++) for (int k = 0; k < 4; k++)
        L->wp[((size_t)i4 * O + o) * 4 + k] = (uint8_t)L->w[(size_t)o * L->K + i4 * 4 + k];
    }
  }
  for (int v = 0; v < 256; v++) {
    float x = (float)(v - fzf) * fsf; float q = nearbyintf(x / fscc) + (float)fzcc;
    featlut[v] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
  }
  return 0;
}

typedef struct { uint8_t *p; int H, W, C, PW, zp; } Act;
#define SLACK 26
static Act mkact(int H, int W, int C, int zp) {
  Act a = { 0, H, W, C, W + 2 + SLACK, zp }; size_t n = (size_t)(H + 2) * a.PW * C + 4096;
  a.p = amalloc(n); memset(a.p, zp, n); return a;
}
static inline uint8_t *px(const Act *a, int y, int x) { return a->p + ((size_t)(y + 1) * a->PW + (x + 1)) * a->C; }

// requantise 16 int32 accumulators (output channels o..o+15) to u8
static inline __m128i requant(const L8 *L, int o, __m512i acc) {
  __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(acc, _mm512_load_si512(L->biasc + o))), _mm512_load_ps(L->mult + o));
  f = _mm512_max_ps(f, _mm512_set1_ps((float)-L->zy)); f = _mm512_min_ps(f, _mm512_set1_ps((float)(255 - L->zy)));
  __m512i i = _mm512_add_epi32(_mm512_cvtps_epi32(f), _mm512_set1_epi32(L->zy));
  return _mm512_cvtusepi32_epi8(i);
}

enum { E_STORE, E_D2S, E_LAST };
static LIP_TLS float *Fpl; static LIP_TLS int FPW;  // tail input: 8 fp32 planes, (160+2) x FPW, border 0

static inline void epilogue(const L8 *L, int mode, Act *dst, int y, int x0, int nv, int nb0, int NB, const int32_t *tmp) {
  for (int t = 0; t < nv; t++) for (int n = 0; n < NB; n++) {
    int o = (nb0 + n) * 16; __m128i q = requant(L, o, _mm512_load_si512(tmp + (t * NB + n) * 16)); int x = x0 + t;
    if (mode == E_STORE) _mm_storeu_si128((__m128i *)(px(dst, y, x) + o), q);
    else if (mode == E_D2S || mode == E_LAST) {
      // channels o..o+15 = 4 output channels (o/4..o/4+3) x 4 subpixels (CRD: in ch = c*4 + a*2 + b)
      __m128i s = _mm_shuffle_epi8(q, _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
      int c0 = o / 4; int32_t v[4]; _mm_storeu_si128((__m128i *)v, s);
      for (int sq = 0; sq < 4; sq++) memcpy(px(dst, 2 * y + sq / 2, 2 * x + sq % 2) + c0, &v[sq], 4);
    }
  }
}

#define CONV_LOOP(NBc, TXc)                                                                                    \
  for (int y = 0; y < H; y++)                                                                                   \
    for (int x0 = 0; x0 < W; x0 += TXc) {                                                                       \
      __m512i acc[TXc][NBc];                                                                                    \
      _Pragma("GCC unroll 32") for (int t = 0; t < TXc; t++) _Pragma("GCC unroll 4") for (int n = 0; n < NBc; n++) acc[t][n] = _mm512_setzero_si512(); \
      for (int ky = 0; ky < 3; ky++) {                                                                          \
        const uint8_t *row = in->p + ((size_t)(y + ky) * PW + x0) * C;                                        \
        for (int kx = 0; kx < 3; kx++) {                                                                        \
          const uint8_t *base = row + kx * C;                                                                   \
          const uint8_t *wb = L->wp + ((size_t)((ky * 3 + kx) * C4) * OB + nb0) * 64;                          \
          for (int c4 = 0; c4 < C4; c4++) {                                                                     \
            const uint8_t *wq = wb + (size_t)c4 * OB * 64; __m512i w[NBc];                                       \
            _Pragma("GCC unroll 4") for (int n = 0; n < NBc; n++) w[n] = _mm512_loadu_si512(wq + n * 64);       \
            _Pragma("GCC unroll 32") for (int t = 0; t < TXc; t++) {                                             \
              __m512i b = _mm512_set1_epi32(*(const int32_t *)(base + (size_t)t * C + c4 * 4));                 \
              _Pragma("GCC unroll 4") for (int n = 0; n < NBc; n++) acc[t][n] = _mm512_dpbusd_epi32(acc[t][n], b, w[n]); \
            }                                                                                                   \
          }                                                                                                     \
        }                                                                                                       \
      }                                                                                                         \
      int32_t tmp[TXc * NBc * 16] __attribute__((aligned(64)));                                                \
      _Pragma("GCC unroll 32") for (int t = 0; t < TXc; t++) _Pragma("GCC unroll 4") for (int n = 0; n < NBc; n++) _mm512_store_si512(tmp + (t * NBc + n) * 16, acc[t][n]); \
      epilogue(L, mode, dst, y, x0, W - x0 < TXc ? W - x0 : TXc, nb0, NBc, tmp);                               \
    }

static void conv8(const L8 *L, const Act *in, int mode, Act *dst) {
  int H = in->H, W = in->W, C = in->C, PW = in->PW, C4 = L->C4, OB = L->OB;
  if (C != L->C || in->zp != L->zx) { fprintf(stderr, "conv input mismatch C %d/%d zp %d/%d\n", C, L->C, in->zp, L->zx); exit(1); }
  for (int nb0 = 0; nb0 < OB; nb0 += 4) {
    int nb = OB - nb0 < 4 ? OB - nb0 : 4;
    if (nb == 4) { CONV_LOOP(4, 6) } else if (nb == 2) { CONV_LOOP(2, 12) } else if (nb == 1) { CONV_LOOP(1, 24) } else { CONV_LOOP(1, 24) }
  }
}

static inline __m512i lut512(__m512i x, const uint8_t *T) {
  __m512i t0 = _mm512_load_si512(T), t1 = _mm512_load_si512(T + 64), t2 = _mm512_load_si512(T + 128), t3 = _mm512_load_si512(T + 192);
  __m512i lo = _mm512_permutex2var_epi8(t0, x, t1), hi = _mm512_permutex2var_epi8(t2, x, t3);
  return _mm512_mask_blend_epi8(_mm512_movepi8_mask(x), lo, hi);
}
static void lut_pass(const Act *s, Act *d, const uint8_t *T) {
  int n = s->W * s->C;
  for (int y = 0; y < s->H; y++) {
    const uint8_t *a = px(s, y, 0); uint8_t *b = px(d, y, 0);
    for (int i = 0; i < n; i += 64) { __mmask64 m = n - i >= 64 ? ~0ULL : ((1ULL << (n - i)) - 1);
      _mm512_mask_storeu_epi8(b + i, m, lut512(_mm512_maskz_loadu_epi8(m, a + i), T)); }
  }
}
static void add_pass(const Act *a, const Act *b, Act *d, int k) {
  int n = a->W * a->C; __m512 ra = _mm512_set1_ps(addr[k][0]), rb = _mm512_set1_ps(addr[k][1]);
  __m512i za = _mm512_set1_epi32(addz[k][0]), zb = _mm512_set1_epi32(addz[k][1]), zo = _mm512_set1_epi32(addz[k][2]);
  __m512 lo = _mm512_set1_ps((float)-addz[k][2]), hi = _mm512_set1_ps((float)(255 - addz[k][2]));
  for (int y = 0; y < a->H; y++) {
    const uint8_t *pa = px(a, y, 0), *pb = px(b, y, 0); uint8_t *pd = px(d, y, 0);
    for (int i = 0; i < n; i += 16) {
      __m512 fa = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(pa + i))), za));
      __m512 fb = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(pb + i))), zb));
      __m512 f = _mm512_fmadd_ps(fb, rb, _mm512_mul_ps(fa, ra)); f = _mm512_min_ps(_mm512_max_ps(f, lo), hi);
      _mm_storeu_si128((__m128i *)(pd + i), _mm512_cvtusepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(f), zo)));
    }
  }
}
static void gemm8(const L8 *L, const uint8_t *x, uint8_t *y) {
  int O = L->O, I4 = L->C4;
  for (int ob0 = 0; ob0 < O / 16; ob0 += 8) {
    int nb = O / 16 - ob0 < 8 ? O / 16 - ob0 : 8; __m512i acc[8];
    for (int n = 0; n < 8; n++) acc[n] = _mm512_setzero_si512();
    for (int i4 = 0; i4 < I4; i4++) {
      __m512i b = _mm512_set1_epi32(*(const int32_t *)(x + i4 * 4)); const uint8_t *wq = L->wp + ((size_t)i4 * O + ob0 * 16) * 4;
      if (nb == 8) { _Pragma("GCC unroll 8") for (int n = 0; n < 8; n++) acc[n] = _mm512_dpbusd_epi32(acc[n], b, _mm512_loadu_si512(wq + n * 64)); }
      else for (int n = 0; n < nb; n++) acc[n] = _mm512_dpbusd_epi32(acc[n], b, _mm512_loadu_si512(wq + n * 64));
    }
    for (int n = 0; n < nb; n++) _mm_storeu_si128((__m128i *)(y + (ob0 + n) * 16), requant(L, (ob0 + n) * 16, acc[n]));
  }
}

// ---- the whole frame model ----
static LIP_TLS Act Fq; static LIP_TLS Act A_in, X0, S1, B0, S2, Dd, Ee, S3, Gg, S4, Ii, Jj, S5, Ll, S6, Nn, O2, S7, Qq, Rr, U1, Tt, U2, Vv;
static LIP_TLS uint8_t zq[640] __attribute__((aligned(64))), g0[256] __attribute__((aligned(64))), g0s[256] __attribute__((aligned(64))), z8[7040] __attribute__((aligned(64)));
static double tacc[8];
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int prof;
#define T(k) do { if (prof) { double t2 = now(); tacc[k] += t2 - t1; t1 = t2; } } while (0)

static void setup(void) {
  A_in = mkact(20, 44, 40, Ls[0].zx); X0 = mkact(20, 44, 64, 0); S1 = mkact(20, 44, 64, Ls[1].zx); B0 = mkact(20, 44, 64, 0);
  S2 = mkact(20, 44, 64, Ls[2].zx); Dd = mkact(20, 44, 64, 0); Ee = mkact(20, 44, 64, 0); S3 = mkact(20, 44, 64, Ls[3].zx);
  Gg = mkact(20, 44, 64, 0); S4 = mkact(20, 44, 64, Ls[4].zx); Ii = mkact(20, 44, 64, 0); Jj = mkact(20, 44, 64, 0);
  S5 = mkact(20, 44, 64, Ls[5].zx); Ll = mkact(20, 44, 64, 0); S6 = mkact(20, 44, 64, Ls[6].zx); Nn = mkact(20, 44, 64, 0);
  O2 = mkact(20, 44, 64, 0); S7 = mkact(20, 44, 64, Ls[7].zx); Qq = mkact(20, 44, 64, Ls[8].zx); Rr = mkact(20, 44, 64, Ls[9].zx);
  U1 = mkact(40, 88, 32, Ls[10].zx); Tt = mkact(40, 88, 32, Ls[11].zx); U2 = mkact(80, 176, 16, Ls[12].zx); Vv = mkact(80, 176, 16, Ls[13].zx);
  FPW = 352 + 2 + 32; Fpl = amalloc((size_t)8 * 162 * FPW * 4); memset(Fpl, 0, (size_t)8 * 162 * FPW * 4);
  Fq = mkact(160, 352, 8, Ls[13].zy);
}

static void frame8(const float *feat, const float *z, float *out) {
  double t1 = prof ? now() : 0;
  // z -> QuantizeLinear -> QLinearMul(const) -> zfc0 (QGemm) -> SiLU table -> zfc2 (QGemm) = concat channels 32..39
  float rz = zsz * zcval / zsmq;
  for (int i = 0; i < 640; i++) {
    float q = nearbyintf(z[i] / zsz) + (float)zzz; q = q < 0 ? 0 : q > 255 ? 255 : q;
    float m = nearbyintf((q - (float)zzz) * rz) + (float)zzmq; zq[i] = (uint8_t)(m < 0 ? 0 : m > 255 ? 255 : m);
  }
  gemm8(&Ls[14], zq, g0); for (int i = 0; i < 256; i++) g0s[i] = lut[0][g0[i]];
  gemm8(&Ls[15], g0s, z8);
  // concat input: feat (QuantizeLinear, then QLinearConcat requant table) + z8, NHWC
  for (int p = 0; p < 880; p++) {
    uint8_t *d = px(&A_in, p / 44, p % 44);
    for (int c = 0; c < 32; c++) {
      float q = nearbyintf(feat[c * 880 + p] / fsf) + (float)fzf; q = q < 0 ? 0 : q > 255 ? 255 : q; d[c] = featlut[(int)q];
    }
    for (int c = 0; c < 8; c++) d[32 + c] = z8[c * 880 + p];
  }
  T(0);
  conv8(&Ls[0], &A_in, E_STORE, &X0); lut_pass(&X0, &S1, lut[1]);
  conv8(&Ls[1], &S1, E_STORE, &B0); lut_pass(&B0, &S2, lut[2]);
  conv8(&Ls[2], &S2, E_STORE, &Dd); add_pass(&Dd, &X0, &Ee, 0); lut_pass(&Ee, &S3, lut[3]);
  conv8(&Ls[3], &S3, E_STORE, &Gg); lut_pass(&Gg, &S4, lut[4]);
  conv8(&Ls[4], &S4, E_STORE, &Ii); add_pass(&Ii, &Ee, &Jj, 1); lut_pass(&Jj, &S5, lut[5]);
  conv8(&Ls[5], &S5, E_STORE, &Ll); lut_pass(&Ll, &S6, lut[6]);
  conv8(&Ls[6], &S6, E_STORE, &Nn); add_pass(&Nn, &Jj, &O2, 2); lut_pass(&O2, &S7, lut[7]);
  conv8(&Ls[7], &S7, E_STORE, &Qq);
  conv8(&Ls[8], &Qq, E_STORE, &Rr);
  T(1);
  conv8(&Ls[9], &Rr, E_D2S, &U1);
  conv8(&Ls[10], &U1, E_STORE, &Tt);
  conv8(&Ls[11], &Tt, E_D2S, &U2);
  T(2);
  conv8(&Ls[12], &U2, E_STORE, &Vv);
  T(3);
  conv8(&Ls[13], &Vv, E_LAST, &Fq);
  { // DQ + ReLU: NHWC u8 (8 ch) -> 8 fp32 planes (zero border kept), 16 pixels per step
    const __m512 sc = _mm512_set1_ps(Ls[13].sy), zero = _mm512_setzero_ps(); const __m512i zp = _mm512_set1_epi32(Ls[13].zy);
    __m512i sel[8];
    for (int c = 0; c < 8; c++) { uint8_t ix[64]; for (int t = 0; t < 64; t++) ix[t] = (uint8_t)(t < 16 ? c + 8 * t : 0); sel[c] = _mm512_loadu_si512(ix); }
    for (int y = 0; y < 160; y++) {
      const uint8_t *r = px(&Fq, y, 0);
      for (int x0 = 0; x0 < 352; x0 += 16) {
        __m512i v0 = _mm512_loadu_si512(r + x0 * 8), v1 = _mm512_loadu_si512(r + x0 * 8 + 64);
        for (int c = 0; c < 8; c++) {
          __m128i b = _mm512_castsi512_si128(_mm512_permutex2var_epi8(v0, sel[c], v1));
          __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(b), zp)), sc);
          _mm512_storeu_ps(Fpl + (size_t)c * 162 * FPW + (size_t)(y + 1) * FPW + x0 + 1, _mm512_max_ps(f, zero));
        }
      }
    }
  }
  T(5);
  // tail: 8 -> 3 conv (x64 fused) + bias, * win, fp32, NCHW out
  for (int y = 0; y < 160; y++)
    for (int x0 = 0; x0 < 352; x0 += 32) {
      __m512 a[3][2];
      for (int o = 0; o < 3; o++) { a[o][0] = _mm512_set1_ps(Bout[o]); a[o][1] = a[o][0]; }
      for (int c = 0; c < 8; c++) for (int ky = 0; ky < 3; ky++) {
        const float *r = Fpl + (size_t)c * 162 * FPW + (size_t)(y + ky) * FPW + x0;
        _Pragma("GCC unroll 3") for (int kx = 0; kx < 3; kx++) {
          __m512 v0 = _mm512_loadu_ps(r + kx), v1 = _mm512_loadu_ps(r + kx + 16);
          _Pragma("GCC unroll 3") for (int o = 0; o < 3; o++) { __m512 w = _mm512_set1_ps(Wout[((o * 8 + c) * 3 + ky) * 3 + kx]);
            a[o][0] = _mm512_fmadd_ps(v0, w, a[o][0]); a[o][1] = _mm512_fmadd_ps(v1, w, a[o][1]); }
        }
      }
      __m512 w0 = _mm512_loadu_ps(win + y * 352 + x0), w1 = _mm512_loadu_ps(win + y * 352 + x0 + 16);
      for (int o = 0; o < 3; o++) { float *d = out + (size_t)o * 160 * 352 + y * 352 + x0;
        _mm512_storeu_ps(d, _mm512_mul_ps(a[o][0], w0)); _mm512_storeu_ps(d + 16, _mm512_mul_ps(a[o][1], w1)); }
    }
  T(4);
}

#ifdef LIP8_LIB
/** Loads and packs the weights (once per process; not thread safe, the addon serialises it). 0 or an error code. */
int lip8_avx512_init(const uint8_t *buf, int n) { int r = load(buf, (size_t)n); if (r) return r; return pack(); }
/** One frame: feat [32,20,44], z [640] -> out [3,160,352]. The calling thread's scratch buffers are made on first use. */
void lip8_avx512_run(const float *feat, const float *z, float *out) { if (!A_in.p) setup(); frame8(feat, z, out); }
#else
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 200; const char *wf = getenv("W8") ? getenv("W8") : "frame.lip8";
  int N = getenv("IN_N") ? atoi(getenv("IN_N")) : 203;
  const char *ff = getenv("FEAT") ? getenv("FEAT") : "feat.f32", *zf = getenv("ZIN") ? getenv("ZIN") : "z.f32";
  { FILE *wfp = fopen(wf, "rb"); if (!wfp) { perror(wf); return 1; } fseek(wfp, 0, SEEK_END); long wn = ftell(wfp); fseek(wfp, 0, SEEK_SET);
    uint8_t *wb = xmalloc(wn); if (fread(wb, 1, wn, wfp) != (size_t)wn) { fprintf(stderr, "short read\n"); return 1; } fclose(wfp);
    int r = load(wb, wn); if (!r) r = pack(); if (r) { fprintf(stderr, "frame.lip8: error %d\n", r); return 1; } free(wb); }
  setup(); prof = getenv("PROF") != NULL;
  float *feat = malloc((size_t)N * 32 * 880 * 4), *z = malloc((size_t)N * 640 * 4);
  FILE *f = fopen(ff, "rb"); if (!f || fread(feat, 4, (size_t)N * 32 * 880, f) != (size_t)N * 32 * 880) { fprintf(stderr, "feat\n"); return 1; } fclose(f);
  f = fopen(zf, "rb"); if (!f || fread(z, 4, (size_t)N * 640, f) != (size_t)N * 640) { fprintf(stderr, "z\n"); return 1; } fclose(f);
  size_t on = 3 * 160 * 352; float *out = malloc(on * 4);
  if (getenv("OUT")) { FILE *fo = fopen(getenv("OUT"), "wb");
    for (int i = 0; i < N; i++) { frame8(feat + (size_t)i * 32 * 880, z + i * 640, out); fwrite(out, 4, on, fo); } fclose(fo); }
  for (int i = 0; i < 10; i++) frame8(feat + (size_t)(i % N) * 32 * 880, z + (i % N) * 640, out);
  memset(tacc, 0, sizeof tacc);
  double *ts = malloc(reps * sizeof(double));
  for (int r = 0; r < reps; r++) { double t = now(); frame8(feat + (size_t)(r % N) * 32 * 880, z + (r % N) * 640, out); ts[r] = now() - t; }
  qsort(ts, reps, sizeof(double), cmpd);
  printf("{\"mode\":\"lip8\",\"reps\":%d,\"ms_med\":%.3f,\"ms_min\":%.3f,\"ms_p90\":%.3f", reps, ts[reps / 2], ts[0], ts[(int)(reps * 0.9)]);
  if (prof) printf(",\"prof\":{\"zfc_concat\":%.3f,\"body\":%.3f,\"up01\":%.3f,\"up2\":%.3f,\"up21\":%.3f,\"tail\":%.3f}", tacc[0] / reps, tacc[1] / reps, tacc[2] / reps, tacc[3] / reps, tacc[5] / reps, tacc[4] / reps);
  printf("}\n");
  return 0;
}
#endif
