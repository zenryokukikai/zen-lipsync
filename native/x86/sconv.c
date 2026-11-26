// The sentence model's dilated conv1d on the CPU (kernel 3, same padding, 256 outputs), for the Node addon: the part
// of CpuSentenceModel (src/core/sentence-model.ts) that takes nearly all its time. Same order of float32 operations as
// the TypeScript (bias, then inputs c = 0.., taps k = 0..2, a multiply then an add each: built with -ffp-contract=off),
// so the latents equal the TypeScript ones.
// Built twice (-mavx512f / -mavx2 -mfma) with NAME set; the addon picks one at run time. Also built to WebAssembly
// SIMD128 (scripts/build-cpu-wasm.sh, cpu/sconvw) for Node where the addon is not available.
#ifndef NAME
#define NAME sconv_generic
#endif

/* Output rows o..o+3 over times t in [ta, tb): bias, then each input and tap (a multiply then an add each). */
static void rows4(const float *x, const float *w0, const float *w1, const float *w2, const float *w3, const float *b, int o, int C, int T, int dil, int ta, int tb,
                  float *y0, float *y1, float *y2, float *y3) {
  for (int t = ta; t < tb; t++) { y0[t] = b[o]; y1[t] = b[o + 1]; y2[t] = b[o + 2]; y3[t] = b[o + 3]; }
  for (int c = 0; c < C; c++) {
    const float *xc = x + (long)c * T;
    for (int k = 0; k < 3; k++) {
      const int ck = c * 3 + k, sh = (k - 1) * dil;
      const float a0 = w0[ck], a1 = w1[ck], a2 = w2[ck], a3 = w3[ck];
      int t0 = sh < 0 ? -sh : 0, t1 = sh > 0 ? T - sh : T;
      if (t0 < ta) t0 = ta;
      if (t1 > tb) t1 = tb;
      for (int t = t0; t < t1; t++) {
        const float v = xc[t + sh];
        y0[t] += a0 * v; y1[t] += a1 * v; y2[t] += a2 * v; y3[t] += a3 * v;
      }
    }
  }
}

/* Times are done in tiles of TILE so the 4 output rows stay in L1 while every input row passes over them. */
#define TILE 256

void NAME(const float *x, const float *w, const float *b, int C, int T, int dil, const float *r, float *y) {
  for (int o = 0; o < 256; o += 4) {
    float *y0 = y + (long)o * T, *y1 = y0 + T, *y2 = y1 + T, *y3 = y2 + T;
    const float *w0 = w + (long)o * C * 3, *w1 = w0 + C * 3, *w2 = w1 + C * 3, *w3 = w2 + C * 3;
    for (int ta = 0; ta < T; ta += TILE) rows4(x, w0, w1, w2, w3, b, o, C, T, dil, ta, ta + TILE < T ? ta + TILE : T, y0, y1, y2, y3);
    if (r) for (int j = 0; j < 4; j++) { float *yj = y + (long)(o + j) * T; const float *rj = r + (long)(o + j) * T; for (int t = 0; t < T; t++) yj[t] += rj[t]; }
  }
}
