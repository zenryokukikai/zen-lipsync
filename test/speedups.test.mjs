// CPU speedups that must not change any output (the Node frames depend on these values):
// - resample() with the per-phase weight table gives the same samples as the formula per output sample;
// - logMel() gives the same values as before the twiddle table and scratch reuse (reference FFT below, written as the
//   original loop: cos/sin per element, fresh scratch arrays);
// - the native sentence conv (addon sconv) gives latents within float32 rounding of the TypeScript conv (C rounds each
//   product to float32, V8 multiplies in float64; the sums are in the same order). Skips without the linux-x64 addon.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resample, logMel, MEL_SAMPLE_RATE, CpuSentenceModel, nativeAddonPath } from '../dist/node/index.js';

function rng(seed) { let s = seed >>> 0; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296 - 0.5); }
function noise(n, seed) { const r = rng(seed); const x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = r() * 0.6 + 0.3 * Math.sin(i * 0.013); return x; }

/** resample as one formula per output sample (the code before the phase table). */
function resampleRef(x, from, to) {
  const ratio = from / to, n = Math.floor(x.length / ratio), out = new Float32Array(n), cutoff = Math.min(1, to / from), taps = 16;
  for (let i = 0; i < n; i++) {
    const c = i * ratio, i0 = Math.floor(c);
    let s = 0, wsum = 0;
    for (let k = i0 - taps + 1; k <= i0 + taps; k++) {
      if (k < 0 || k >= x.length) continue;
      const t = c - k;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t * cutoff) / (Math.PI * t * cutoff);
      const win = 0.5 + 0.5 * Math.cos((Math.PI * t) / (taps + 1));
      const w = sinc * win;
      s += x[k] * w; wsum += w;
    }
    out[i] = wsum ? s / wsum : 0;
  }
  return out;
}

test('resample: the per-phase table gives the per-sample formula\'s samples (short clips: exactly; minutes: within one float32 step)', () => {
  for (const from of [44100, 48000, 24000, 22050, 32000, 8000, 11025]) {
    const x = noise(Math.round(from * 0.37) + 13, from);
    const a = resample(x, from, MEL_SAMPLE_RATE), b = resampleRef(x, from, MEL_SAMPLE_RATE);
    assert.equal(a.length, b.length, `${from}: length`);
    let max = 0;
    for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
    assert.equal(max, 0, `${from} -> 16000: max difference ${max}`);
  }
  // 3 minutes of 44.1 kHz: i * ratio of the per-sample form rounds differently far into the clip
  const long = noise(44100 * 180, 2);
  const a = resample(long, 44100, MEL_SAMPLE_RATE), b = resampleRef(long, 44100, MEL_SAMPLE_RATE);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  assert.ok(max <= 6e-8, `3 min at 44.1 kHz: max difference ${max}`);
});

// ---- log-mel reference: mel.ts as it was before the twiddle table and the scratch reuse (cos/sin per element) ----
const N_FFT = 800, HOP = 200, N_MELS = 80, BINS = N_FFT / 2 + 1;
function fftRef(re, im, n, stride, off, outRe, outIm, outOff) {
  if (n === 1) { outRe[outOff] = re[off]; outIm[outOff] = im[off]; return; }
  const r = n % 2 === 0 ? 2 : n % 5 === 0 ? 5 : n, m = n / r;
  for (let q = 0; q < r; q++) fftRef(re, im, m, stride * r, off + q * stride, outRe, outIm, outOff + q * m);
  const tr = new Float64Array(r), ti = new Float64Array(r);
  for (let k = 0; k < m; k++) {
    for (let q = 0; q < r; q++) {
      const a = (-2 * Math.PI * q * k) / n;
      const xr = outRe[outOff + q * m + k], xi = outIm[outOff + q * m + k];
      tr[q] = xr * Math.cos(a) - xi * Math.sin(a);
      ti[q] = xr * Math.sin(a) + xi * Math.cos(a);
    }
    for (let s2 = 0; s2 < r; s2++) {
      let sr = 0, si = 0;
      for (let q = 0; q < r; q++) {
        const a = (-2 * Math.PI * q * s2) / r;
        sr += tr[q] * Math.cos(a) - ti[q] * Math.sin(a);
        si += tr[q] * Math.sin(a) + ti[q] * Math.cos(a);
      }
      outRe[outOff + s2 * m + k] = sr; outIm[outOff + s2 * m + k] = si;
    }
  }
}
function hzToMel(f) { const sp = 200 / 3; return f < 1000 ? f / sp : 1000 / sp + Math.log(f / 1000) / (Math.log(6.4) / 27); }
function melToHz(m) { const sp = 200 / 3, ml = 1000 / sp; return m < ml ? m * sp : 1000 * Math.exp((Math.log(6.4) / 27) * (m - ml)); }
function logMelRef(pcm) {
  const lo = hzToMel(55), hi = hzToMel(7600);
  const melF = Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (N_MELS + 1)));
  const fk = Array.from({ length: BINS }, (_, k) => (k * MEL_SAMPLE_RATE) / N_FFT);
  const filters = [];
  for (let i = 0; i < N_MELS; i++) {
    const w = new Float64Array(BINS), d0 = melF[i + 1] - melF[i], d1 = melF[i + 2] - melF[i + 1], en = 2 / (melF[i + 2] - melF[i]);
    for (let k = 0; k < BINS; k++) w[k] = Math.max(0, Math.min((fk[k] - melF[i]) / d0, (melF[i + 2] - fk[k]) / d1)) * en;
    filters.push(w);
  }
  const ranges = filters.map((w) => { let a = 0; while (a < BINS && w[a] === 0) a++; let b = BINS; while (b > a && w[b - 1] === 0) b--; return [a, b]; });
  const win = new Float64Array(N_FFT); for (let n = 0; n < N_FFT; n++) win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  const T = 1 + Math.floor(pcm.length / HOP), mel = new Float32Array(T * N_MELS);
  const frame = new Float64Array(N_FFT), zero = new Float64Array(N_FFT), oRe = new Float64Array(N_FFT), oIm = new Float64Array(N_FFT), mag = new Float64Array(BINS);
  for (let t = 0; t < T; t++) {
    const st = t * HOP - N_FFT / 2;
    for (let n = 0; n < N_FFT; n++) { const i = st + n; frame[n] = i >= 0 && i < pcm.length ? pcm[i] * win[n] : 0; }
    fftRef(frame, zero, N_FFT, 1, 0, oRe, oIm, 0);
    for (let k = 0; k < BINS; k++) mag[k] = Math.hypot(oRe[k], oIm[k]);
    for (let m = 0; m < N_MELS; m++) { const [a, b] = ranges[m]; let s = 0; for (let k = a; k < b; k++) s += filters[m][k] * mag[k]; mel[t * N_MELS + m] = Math.log(Math.max(s, 1e-5)); }
  }
  return mel;
}

test('logMel: the FFT with the twiddle table and reused scratch gives the same values as the original loop', () => {
  const x = noise(Math.round(MEL_SAMPLE_RATE * 1.7), 5);
  const a = logMel(x).mel, b = logMelRef(x);
  assert.equal(a.length, b.length);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  assert.equal(max, 0, `max difference ${max}`);
  assert.deepEqual(logMel(x).mel, a, 'the same on a second call (the reused scratch holds no state)');
});

const addonPath = nativeAddonPath();
const addon = addonPath ? createRequire(import.meta.url)(addonPath) : null;
const noSconv = !addon || typeof addon.sconv !== 'function' || !addon.sconvKernel?.() ? 'native addon with sconv not built, or this CPU has neither AVX-512F nor AVX2' : false;

/** Random sentence-model weights under the names CpuSentenceModel reads (conv w packed [O/4][C][3][4], b [O]). */
function randomSentenceWeights() {
  const r = rng(11), W = {};
  const t = (name, n, scale, base = 0) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = base + r() * scale; W[name] = a; };
  t('in.w', 256 * 208 * 3, 2 / Math.sqrt(208 * 3)); t('in.b', 256, 0.2);
  for (let i = 0; i < 7; i++) for (const j of [1, 2]) {
    t(`b${i}.conv${j}.w`, 256 * 256 * 3, 2 / Math.sqrt(256 * 3)); t(`b${i}.conv${j}.b`, 256, 0.2);
    t(`b${i}.norm${j}.g`, 256, 0.5, 1); t(`b${i}.norm${j}.b`, 256, 0.2);
  }
  t('out_norm.g', 256, 0.5, 1); t('out_norm.b', 256, 0.2);
  t('head.w', 256 * 128, 2 / Math.sqrt(256)); t('head.b', 128, 0.2); t('scale', 128, 0.5, 1); t('shift', 128, 0.5);
  return W;
}

/** Latents of conv vs the TypeScript conv: within a few float32 ulps of the largest latent (15 convs of 3 x 256 terms). */
function assertCloseLatents(W, conv, T, seed) {
  const mel = noise(T * 80, seed), ref = noise(T * 128, seed + 1);
  const ts = new CpuSentenceModel(W).run(mel, ref, T);
  for (let i = 0; i < ts.length; i++) assert.ok(Number.isFinite(ts[i]), 'finite latents');
  for (let rep = 0; rep < 2; rep++) {
    // twice: a conv that keeps a work block reuses it
    const got = new CpuSentenceModel(W, conv).run(mel, ref, T);
    assert.equal(got.length, ts.length);
    let max = 0, mag = 0;
    for (let i = 0; i < ts.length; i++) { max = Math.max(max, Math.abs(got[i] - ts[i])); mag = Math.max(mag, Math.abs(ts[i])); }
    assert.ok(max <= 4e-6 * Math.max(1, mag), `T ${T}: max difference ${max} (largest latent ${mag})`);
  }
}

test('sentence model: the native conv gives the TypeScript latents within float32 rounding', { skip: noSconv }, () => {
  assertCloseLatents(randomSentenceWeights(), addon.sconv.bind(addon), 157, 3);
});

test('sentence model: the WebAssembly conv (cpu/sconvw) gives the TypeScript latents within float32 rounding, also past its first memory', async () => {
  const { wasmSentenceConv } = await import('../dist/node/frame-model.js');
  const conv = await wasmSentenceConv();
  const W = randomSentenceWeights();
  assertCloseLatents(W, conv, 211, 8);
  // about 60 s of speech: the work block (~25 MB) is past the module's first 16 MB, so its memory grows
  assertCloseLatents(W, conv, 4801, 21);
  // and a short one again on the grown memory
  assertCloseLatents(W, conv, 9, 30);
});

test('sentence model: sentenceConv() picks a conv (native addon, else WebAssembly)', async () => {
  const { sentenceConv } = await import('../dist/node/frame-model.js');
  assert.ok(await sentenceConv(), 'cpu/sconvw.wasm is in the package, so there is always one');
});

test('resample: a sample rate that is not a positive number is refused (not silently NaN samples)', () => {
  const x = noise(100, 1);
  for (const [f, t] of [[0, 16000], [44100, 0], [-8000, 16000], [NaN, 16000], [Infinity, 16000]]) assert.throws(() => resample(x, f, t), RangeError, `${f} -> ${t}`);
});
