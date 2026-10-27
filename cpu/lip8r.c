// lip8r.c - ZEN-LipSync frame model on the CPU in WebAssembly relaxed SIMD, single thread. Same weights file
// (frame.lip8) and the same integer arithmetic as native/x86/lip8.c: u8 activations x s8 weights accumulated exactly in
// int32 (i32x4.dot_i16x8_s on i16-widened values), the same float requantisation (mul, clamp, round-to-nearest-even).
// The fp32 tail (8->3 conv, *window) uses separate mul+add (differences around 1e-7 of the delta).
// build: scripts/build-cpu-wasm.sh (Emscripten)
// relaxed SIMD: u8 activations split x = lo + 128*hi for the i7 operand (exact).
#include <wasm_simd128.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emscripten/emscripten.h>

typedef struct { int O, C, zx, zy, K, OV, C2; float sx, sy; const int8_t *w; const float *sw; const int32_t *bq;
                 int16_t *wp; int8_t *wq8; int C4; int32_t *biasc; float *mult; } L8;
static L8 Ls[16];
static uint8_t lut[8][256];
static float addr[3][2]; static int32_t addz[3][3];
static float zsz, zcval, zsmq, fsf, fscc; static int32_t zzz, zzmq, fzf, fzcc;
static float Wout[3 * 8 * 9], Bout[3], *win;
static uint8_t featlut[256];

static const uint8_t *rp; static const uint8_t *rend; static int rerr;
static void rd(void *p, size_t n) { if (rp + n > rend) { rerr = 1; return; } memcpy(p, rp, n); rp += n; }
static const void *rdp(size_t n) { if (rp + n > rend) { rerr = 1; return rp; } const void *q = rp; rp += n; return q; }
static void *amalloc(size_t n) { void *p = aligned_alloc(16, (n + 15) / 16 * 16); if (!p) abort(); return p; }

static int load(const uint8_t *buf, int n) {
  rp = buf; rend = buf + n; rerr = 0;
  char mg[4]; int32_t ver; rd(mg, 4); rd(&ver, 4); if (memcmp(mg, "LIP8", 4) || ver != 2) return -1;
  for (int l = 0; l < 16; l++) {
    L8 *L = &Ls[l]; int32_t h[3]; rd(h, 12); L->O = h[0]; L->C = h[1]; L->zx = h[2]; rd(&L->sx, 4);
    L->K = l < 14 ? L->C * 9 : L->C;
    // copies: the weights buffer is freed by the caller after init
    int8_t *w = malloc((size_t)L->O * L->K); rd(w, (size_t)L->O * L->K); L->w = w;
    float *sw = malloc(L->O * 4); rd(sw, L->O * 4); L->sw = sw;
    int32_t *bq = malloc(L->O * 4); rd(bq, L->O * 4); L->bq = bq;
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

// conv weights (i16): wp[((tap*C2 + c2)*OV + ov)*8 + j*2 + k] = w[ov*4+j][(c2*2+k)*9 + tap]
// gemm weights (i16): wp[(i2*OV + ov)*8 + j*2 + k] = w[ov*4+j][i2*2+k]
static int pack(void) {
  for (int l = 0; l < 16; l++) {
    L8 *L = &Ls[l]; int O = L->O, C = L->C; L->OV = O / 4; L->C2 = C / 2;
    if (O % 4 || C % 2) return -10 - l;
    L->biasc = amalloc(O * 4); L->mult = amalloc(O * 4);
    for (int o = 0; o < O; o++) {
      long s = 0; for (int k = 0; k < L->K; k++) s += L->w[(size_t)o * L->K + k];
      L->biasc[o] = L->bq[o] - L->zx * (int32_t)s; L->mult[o] = L->sx * L->sw[o] / L->sy;
    }
    size_t n = (size_t)O * L->K; L->wp = amalloc(n * 2);
    if (l < 14) {
      if (C % 4) return -30 - l;
      L->C4 = C / 4; L->wq8 = amalloc(n);
      // relaxed dot: wq8[((tap*C4 + c4)*OV + ov)*16 + j*4 + k] = w[ov*4+j][(c4*4+k)*9 + tap]
      for (int tap = 0; tap < 9; tap++) for (int c4 = 0; c4 < L->C4; c4++) for (int ov = 0; ov < L->OV; ov++)
        for (int j = 0; j < 4; j++) for (int k = 0; k < 4; k++)
          L->wq8[(((size_t)tap * L->C4 + c4) * L->OV + ov) * 16 + j * 4 + k] = L->w[(size_t)(ov * 4 + j) * L->K + (c4 * 4 + k) * 9 + tap];
    } else {
      for (int i2 = 0; i2 < L->C2; i2++) for (int ov = 0; ov < L->OV; ov++) for (int j = 0; j < 4; j++) for (int k = 0; k < 2; k++)
        L->wp[((size_t)i2 * L->OV + ov) * 8 + j * 2 + k] = L->w[(size_t)(ov * 4 + j) * L->K + i2 * 2 + k];
    }
    free((void *)L->w); free((void *)L->sw); free((void *)L->bq); L->w = 0; L->sw = 0; L->bq = 0;
  }
  for (int v = 0; v < 256; v++) {
    float x = (float)(v - fzf) * fsf; float q = nearbyintf(x / fscc) + (float)fzcc;
    featlut[v] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
  }
  return 0;
}

// activations: i16 NHWC with a 1-pixel border holding the zero point (plus right-side slack for the last tile)
typedef struct { uint8_t *p; int H, W, C, PW, zp; } Act;
#define SLACK 8
static Act mkact(int H, int W, int C, int zp) {
  Act a = { 0, H, W, C, W + 2 + SLACK, zp }; size_t n = (size_t)(H + 2) * a.PW * C + 256;
  a.p = amalloc(n); memset(a.p, zp, n); return a;
}
static inline uint8_t *px(const Act *a, int y, int x) { return a->p + ((size_t)(y + 1) * a->PW + (x + 1)) * a->C; }

// requantise 4 int32 accumulators (output channels o..o+3) to integers in [0,255]
static inline v128_t requant(const L8 *L, int o, v128_t acc) {
  v128_t f = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_add(acc, wasm_v128_load(L->biasc + o))), wasm_v128_load(L->mult + o));
  f = wasm_f32x4_max(f, wasm_f32x4_splat((float)-L->zy)); f = wasm_f32x4_min(f, wasm_f32x4_splat((float)(255 - L->zy)));
  return wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f)), wasm_i32x4_splat(L->zy));
}

enum { E_STORE, E_D2S, E_LAST };
static float *Fpl; static int FPW;  // tail input: 8 fp32 planes, (160+2) x FPW, border 0

static inline void store_out(const L8 *L, int mode, Act *dst, int y, int x, int ov, v128_t acc) {
  v128_t q = requant(L, ov * 4, acc);
  int32_t v[4]; wasm_v128_store(v, q);
  if (mode == E_STORE) {
    uint8_t *d = px(dst, y, x) + ov * 4; d[0] = (uint8_t)v[0]; d[1] = (uint8_t)v[1]; d[2] = (uint8_t)v[2]; d[3] = (uint8_t)v[3];
  } else if (mode == E_D2S) {
    // conv output channel o = c*4 + a*2 + b (CRD) -> pixel (2y+a, 2x+b), channel c; here c = ov, sq = lane
    for (int sq = 0; sq < 4; sq++) px(dst, 2 * y + sq / 2, 2 * x + sq % 2)[ov] = (uint8_t)v[sq];
  } else { // E_LAST: D2S into 8 channels, then DQ + ReLU into the fp32 planes (same arithmetic as lip8.c)
    for (int sq = 0; sq < 4; sq++) {
      float f = (float)(v[sq] - L->zy) * L->sy; f = f > 0.f ? f : 0.f;
      Fpl[(size_t)ov * 162 * FPW + (size_t)(2 * y + sq / 2 + 1) * FPW + 2 * x + sq % 2 + 1] = f;
    }
  }
}

#ifdef __wasm_relaxed_simd__
#define DOT(acc, w, x) wasm_i32x4_relaxed_dot_i8x16_i7x16_add((w), (x), (acc))
#define CONV_LOOP(NV, TX)                                                                                        \
  for (int y = 0; y < H; y++)                                                                                   \
    for (int x0 = 0; x0 < W; x0 += TX) {                                                                        \
      v128_t al[TX][NV], ah[TX][NV];                                                                            \
      for (int t = 0; t < TX; t++) for (int n = 0; n < NV; n++) { al[t][n] = wasm_i32x4_splat(0); ah[t][n] = al[t][n]; } \
      for (int ky = 0; ky < 3; ky++) {                                                                          \
        const uint8_t *row = in->p + ((size_t)(y + ky) * PW + x0) * C;                                         \
        for (int kx = 0; kx < 3; kx++) {                                                                        \
          const uint8_t *base = row + kx * C;                                                                   \
          const int8_t *wb = L->wq8 + ((size_t)((ky * 3 + kx) * C4) * OV + ov0) * 16;                          \
          for (int c4 = 0; c4 < C4; c4++) {                                                                     \
            const int8_t *wq = wb + (size_t)c4 * OV * 16; v128_t w[NV];                                          \
            for (int n = 0; n < NV; n++) w[n] = wasm_v128_load(wq + n * 16);                                    \
            for (int t = 0; t < TX; t++) {                                                                      \
              v128_t v = wasm_v128_load32_splat(base + (size_t)t * C + c4 * 4);                                 \
              v128_t lo = wasm_v128_and(v, wasm_i8x16_splat(0x7f)), hi = wasm_u8x16_shr(v, 7);                  \
              for (int n = 0; n < NV; n++) { al[t][n] = DOT(al[t][n], w[n], lo); ah[t][n] = DOT(ah[t][n], w[n], hi); } \
            }                                                                                                   \
          }                                                                                                     \
        }                                                                                                       \
      }                                                                                                         \
      int nv = W - x0 < TX ? W - x0 : TX;                                                                       \
      for (int t = 0; t < nv; t++) for (int n = 0; n < NV; n++)                                                 \
        store_out(L, mode, dst, y, x0 + t, ov0 + n, wasm_i32x4_add(al[t][n], wasm_i32x4_shl(ah[t][n], 7)));      \
    }
#else
#error "build with -mrelaxed-simd"
#endif

static void conv8(const L8 *L, const Act *in, int mode, Act *dst) {
  int H = in->H, W = in->W, C = in->C, PW = in->PW, C4 = L->C4, OV = L->OV;
  for (int ov0 = 0; ov0 < OV; ov0 += 4) {
    int nb = OV - ov0 < 4 ? OV - ov0 : 4;
    if (nb == 4) { CONV_LOOP(4, 2) } else if (nb == 2) { CONV_LOOP(2, 4) } else { for (; ov0 < OV; ov0++) { CONV_LOOP(1, 8) } break; }
  }
}

static void lut_pass(const Act *s, Act *d, const uint8_t *T) {
  int n = s->W * s->C;
  for (int y = 0; y < s->H; y++) { const uint8_t *a = px(s, y, 0); uint8_t *b = px(d, y, 0); for (int i = 0; i < n; i++) b[i] = T[a[i]]; }
}
// QLinearAdd: lip8.c computes fma(fb, rb, fa*ra) in float; double gives the same single rounding here
static void add_pass(const Act *a, const Act *b, Act *d, int k) {
  int n = a->W * a->C; double ra = addr[k][0], rb = addr[k][1]; int za = addz[k][0], zb = addz[k][1], zo = addz[k][2];
  float lo = (float)-zo, hi = (float)(255 - zo);
  for (int y = 0; y < a->H; y++) {
    const uint8_t *pa = px(a, y, 0), *pb = px(b, y, 0); uint8_t *pd = px(d, y, 0);
    for (int i = 0; i < n; i++) {
      float fa = (float)(pa[i] - za), fb = (float)(pb[i] - zb);
      float p = (float)(fa * (float)ra);
      float f = (float)((double)fb * rb + (double)p);
      f = f < lo ? lo : f > hi ? hi : f;
      pd[i] = (uint8_t)((int)nearbyintf(f) + zo);
    }
  }
}
static void gemm8(const L8 *L, const int16_t *x, int16_t *y) {
  int OV = L->OV, I2 = L->C2;
  for (int ov = 0; ov < OV; ov += 4) {
    int nb = OV - ov < 4 ? OV - ov : 4; v128_t acc[4] = { wasm_i32x4_splat(0), wasm_i32x4_splat(0), wasm_i32x4_splat(0), wasm_i32x4_splat(0) };
    for (int i2 = 0; i2 < I2; i2++) {
      v128_t b = wasm_v128_load32_splat(x + i2 * 2); const int16_t *wq = L->wp + ((size_t)i2 * OV + ov) * 8;
      for (int n = 0; n < nb; n++) acc[n] = wasm_i32x4_add(acc[n], wasm_i32x4_dot_i16x8(b, wasm_v128_load(wq + n * 8)));
    }
    for (int n = 0; n < nb; n++) { int32_t v[4]; wasm_v128_store(v, requant(L, (ov + n) * 4, acc[n])); for (int j = 0; j < 4; j++) y[(ov + n) * 4 + j] = (int16_t)v[j]; }
  }
}

static Act A_in, X0, S1, B0, S2, Dd, Ee, S3, Gg, S4, Ii, Jj, S5, Ll, S6, Nn, O2, S7, Qq, Rr, U1, Tt, U2, Vv;
static int16_t zq[640], g0[256], g0s[256], z8[7040];
static Act Dummy;

static void setup(void) {
  A_in = mkact(20, 44, 40, Ls[0].zx); X0 = mkact(20, 44, 64, 0); S1 = mkact(20, 44, 64, Ls[1].zx); B0 = mkact(20, 44, 64, 0);
  S2 = mkact(20, 44, 64, Ls[2].zx); Dd = mkact(20, 44, 64, 0); Ee = mkact(20, 44, 64, 0); S3 = mkact(20, 44, 64, Ls[3].zx);
  Gg = mkact(20, 44, 64, 0); S4 = mkact(20, 44, 64, Ls[4].zx); Ii = mkact(20, 44, 64, 0); Jj = mkact(20, 44, 64, 0);
  S5 = mkact(20, 44, 64, Ls[5].zx); Ll = mkact(20, 44, 64, 0); S6 = mkact(20, 44, 64, Ls[6].zx); Nn = mkact(20, 44, 64, 0);
  O2 = mkact(20, 44, 64, 0); S7 = mkact(20, 44, 64, Ls[7].zx); Qq = mkact(20, 44, 64, Ls[8].zx); Rr = mkact(20, 44, 64, Ls[9].zx);
  U1 = mkact(40, 88, 32, Ls[10].zx); Tt = mkact(40, 88, 32, Ls[11].zx); U2 = mkact(80, 176, 16, Ls[12].zx); Vv = mkact(80, 176, 16, Ls[13].zx);
  FPW = 352 + 2 + 8; Fpl = amalloc((size_t)8 * 162 * FPW * 4); memset(Fpl, 0, (size_t)8 * 162 * FPW * 4);
}

EMSCRIPTEN_KEEPALIVE int lip_init(const uint8_t *w, int n) { int r = load(w, n); if (r) return r; r = pack(); if (r) return r; setup(); return 0; }

EMSCRIPTEN_KEEPALIVE void lip_run(const float *feat, const float *z, float *out) {
  float rz = zsz * zcval / zsmq;
  for (int i = 0; i < 640; i++) {
    float q = nearbyintf(z[i] / zsz) + (float)zzz; q = q < 0 ? 0 : q > 255 ? 255 : q;
    float m = nearbyintf((q - (float)zzz) * rz) + (float)zzmq; zq[i] = (int16_t)(m < 0 ? 0 : m > 255 ? 255 : m);
  }
  gemm8(&Ls[14], zq, g0); for (int i = 0; i < 256; i++) g0s[i] = lut[0][g0[i]];
  gemm8(&Ls[15], g0s, z8);
  for (int p = 0; p < 880; p++) {
    uint8_t *d = px(&A_in, p / 44, p % 44);
    for (int c = 0; c < 32; c++) {
      float q = nearbyintf(feat[c * 880 + p] / fsf) + (float)fzf; q = q < 0 ? 0 : q > 255 ? 255 : q; d[c] = featlut[(int)q];
    }
    for (int c = 0; c < 8; c++) d[32 + c] = (uint8_t)z8[c * 880 + p];
  }
  conv8(&Ls[0], &A_in, E_STORE, &X0); lut_pass(&X0, &S1, lut[1]);
  conv8(&Ls[1], &S1, E_STORE, &B0); lut_pass(&B0, &S2, lut[2]);
  conv8(&Ls[2], &S2, E_STORE, &Dd); add_pass(&Dd, &X0, &Ee, 0); lut_pass(&Ee, &S3, lut[3]);
  conv8(&Ls[3], &S3, E_STORE, &Gg); lut_pass(&Gg, &S4, lut[4]);
  conv8(&Ls[4], &S4, E_STORE, &Ii); add_pass(&Ii, &Ee, &Jj, 1); lut_pass(&Jj, &S5, lut[5]);
  conv8(&Ls[5], &S5, E_STORE, &Ll); lut_pass(&Ll, &S6, lut[6]);
  conv8(&Ls[6], &S6, E_STORE, &Nn); add_pass(&Nn, &Jj, &O2, 2); lut_pass(&O2, &S7, lut[7]);
  conv8(&Ls[7], &S7, E_STORE, &Qq);
  conv8(&Ls[8], &Qq, E_STORE, &Rr);
  conv8(&Ls[9], &Rr, E_D2S, &U1);
  conv8(&Ls[10], &U1, E_STORE, &Tt);
  conv8(&Ls[11], &Tt, E_D2S, &U2);
  conv8(&Ls[12], &U2, E_STORE, &Vv);
  conv8(&Ls[13], &Vv, E_LAST, &Dummy);
  // tail: 8 -> 3 conv (x64 fused into the weights) + bias, * win, fp32, NCHW out (same order as lip8.c, mul+add)
  for (int y = 0; y < 160; y++)
    for (int x0 = 0; x0 < 352; x0 += 8) {
      v128_t a[3][2];
      for (int o = 0; o < 3; o++) { a[o][0] = wasm_f32x4_splat(Bout[o]); a[o][1] = a[o][0]; }
      for (int c = 0; c < 8; c++) for (int ky = 0; ky < 3; ky++) {
        const float *r = Fpl + (size_t)c * 162 * FPW + (size_t)(y + ky) * FPW + x0;
        for (int kx = 0; kx < 3; kx++) {
          v128_t v0 = wasm_v128_load(r + kx), v1 = wasm_v128_load(r + kx + 4);
          for (int o = 0; o < 3; o++) { v128_t w = wasm_f32x4_splat(Wout[((o * 8 + c) * 3 + ky) * 3 + kx]);
            a[o][0] = wasm_f32x4_add(a[o][0], wasm_f32x4_mul(v0, w)); a[o][1] = wasm_f32x4_add(a[o][1], wasm_f32x4_mul(v1, w)); }
        }
      }
      v128_t w0 = wasm_v128_load(win + y * 352 + x0), w1 = wasm_v128_load(win + y * 352 + x0 + 4);
      for (int o = 0; o < 3; o++) { float *d = out + (size_t)o * 160 * 352 + y * 352 + x0;
        wasm_v128_store(d, wasm_f32x4_mul(a[o][0], w0)); wasm_v128_store(d + 4, wasm_f32x4_mul(a[o][1], w1)); }
    }
}
