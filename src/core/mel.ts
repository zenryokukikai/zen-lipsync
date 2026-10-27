// Log-mel features of the speech for the sentence model: 16 kHz, n_fft 800, hop 200, periodic Hann window of 800,
// centred frames with zero padding, 80 mel bands from 55 to 7600 Hz (Slaney mel scale and area normalisation),
// magnitude (power 1), log(max(mel, 1e-5)). 80 mel frames per second; video frame i (25 fps) uses mel column
// round(i / 25 * 80).

export const MEL_SAMPLE_RATE = 16000;
const N_FFT = 800;
const HOP = 200;
const N_MELS = 80;
const BINS = N_FFT / 2 + 1;
const VIDEO_FPS = 25;
const MEL_FPS = 80;

function hzToMel(f: number): number {
  const sp = 200 / 3;
  if (f < 1000) return f / sp;
  return 1000 / sp + Math.log(f / 1000) / (Math.log(6.4) / 27);
}

function melToHz(m: number): number {
  const sp = 200 / 3;
  const minLogMel = 1000 / sp;
  if (m < minLogMel) return m * sp;
  return 1000 * Math.exp((Math.log(6.4) / 27) * (m - minLogMel));
}

function melFilters(): Float64Array[] {
  const lo = hzToMel(55);
  const hi = hzToMel(7600);
  const melF = Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (N_MELS + 1)));
  const fft = Array.from({ length: BINS }, (_, k) => (k * MEL_SAMPLE_RATE) / N_FFT);
  const out: Float64Array[] = [];
  for (let i = 0; i < N_MELS; i++) {
    const w = new Float64Array(BINS);
    const d0 = melF[i + 1]! - melF[i]!;
    const d1 = melF[i + 2]! - melF[i + 1]!;
    const enorm = 2 / (melF[i + 2]! - melF[i]!);
    for (let k = 0; k < BINS; k++) {
      const lower = (fft[k]! - melF[i]!) / d0;
      const upper = (melF[i + 2]! - fft[k]!) / d1;
      w[k] = Math.max(0, Math.min(lower, upper)) * enorm;
    }
    out.push(w);
  }
  return out;
}

/** Recursive mixed-radix complex FFT (800 = 2^5 * 5^2), in place on re/im of length n. */
function fft(re: Float64Array, im: Float64Array, n: number, stride: number, off: number, outRe: Float64Array, outIm: Float64Array, outOff: number): void {
  if (n === 1) { outRe[outOff] = re[off]!; outIm[outOff] = im[off]!; return; }
  const r = n % 2 === 0 ? 2 : n % 5 === 0 ? 5 : n;
  const m = n / r;
  for (let q = 0; q < r; q++) fft(re, im, m, stride * r, off + q * stride, outRe, outIm, outOff + q * m);
  const tr = new Float64Array(r);
  const ti = new Float64Array(r);
  for (let k = 0; k < m; k++) {
    for (let q = 0; q < r; q++) {
      const a = (-2 * Math.PI * q * k) / n;
      const xr = outRe[outOff + q * m + k]!;
      const xi = outIm[outOff + q * m + k]!;
      tr[q] = xr * Math.cos(a) - xi * Math.sin(a);
      ti[q] = xr * Math.sin(a) + xi * Math.cos(a);
    }
    for (let s2 = 0; s2 < r; s2++) {
      let sr = 0;
      let si = 0;
      for (let q = 0; q < r; q++) {
        const a = (-2 * Math.PI * q * s2) / r;
        sr += tr[q]! * Math.cos(a) - ti[q]! * Math.sin(a);
        si += tr[q]! * Math.sin(a) + ti[q]! * Math.cos(a);
      }
      outRe[outOff + s2 * m + k] = sr;
      outIm[outOff + s2 * m + k] = si;
    }
  }
}

let tables: { win: Float64Array; filters: Float64Array[]; ranges: [number, number][] } | null = null;

function getTables(): NonNullable<typeof tables> {
  if (tables) return tables;
  const win = new Float64Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  const filters = melFilters();
  // only the bins where a filter is non-zero
  const ranges = filters.map((w): [number, number] => {
    let a = 0;
    while (a < BINS && w[a] === 0) a++;
    let b = BINS;
    while (b > a && w[b - 1] === 0) b--;
    return [a, b];
  });
  tables = { win, filters, ranges };
  return tables;
}

export interface LogMel {
  /** float32 [melFrames, 80], row major */
  mel: Float32Array;
  melFrames: number;
  /** Video frames (25 fps) the audio spans. */
  frames: number;
  /** Mel column of every video frame. */
  gather: number[];
}

/** pcm: float samples at 16 kHz in [-1, 1]. */
export function logMel(pcm: Float32Array): LogMel {
  const { win, filters, ranges } = getTables();
  const pad = N_FFT / 2;
  const T = 1 + Math.floor(pcm.length / HOP);
  const mel = new Float32Array(T * N_MELS);
  const frame = new Float64Array(N_FFT);
  const zero = new Float64Array(N_FFT);
  const outRe = new Float64Array(N_FFT);
  const outIm = new Float64Array(N_FFT);
  const mag = new Float64Array(BINS);
  for (let t = 0; t < T; t++) {
    const start = t * HOP - pad;
    for (let n = 0; n < N_FFT; n++) {
      const i = start + n;
      frame[n] = i >= 0 && i < pcm.length ? pcm[i]! * win[n]! : 0;
    }
    fft(frame, zero, N_FFT, 1, 0, outRe, outIm, 0);
    for (let k = 0; k < BINS; k++) mag[k] = Math.hypot(outRe[k]!, outIm[k]!);
    for (let m = 0; m < N_MELS; m++) {
      const w = filters[m]!;
      const [a, b] = ranges[m]!;
      let s = 0;
      for (let k = a; k < b; k++) s += w[k]! * mag[k]!;
      mel[t * N_MELS + m] = Math.log(Math.max(s, 1e-5));
    }
  }
  const frames = Math.floor((T / MEL_FPS) * VIDEO_FPS);
  const gather = Array.from({ length: frames }, (_, i) => Math.min(Math.max(Math.round((i / VIDEO_FPS) * MEL_FPS), 0), T - 1));
  return { mel, melFrames: T, frames, gather };
}
