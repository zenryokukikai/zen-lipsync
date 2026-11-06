// Resampling of speech audio to the 16 kHz the audio features use (shared by both entries).
import { yieldToPage } from './mel.js';

/** Simple resampler (windowed sinc, 16 taps each side): good enough for 16 kHz log-mel features. */
export function resample(x: Float32Array, from: number, to: number): Float32Array {
  const out = new Float32Array(Math.floor(x.length / (from / to)));
  resampleRange(x, from, to, out, 0, out.length);
  return out;
}

/** resample in parts of RESAMPLE_PART output samples, with yieldToPage() between them (see logMelYielding). */
export async function resampleYielding(x: Float32Array, from: number, to: number): Promise<Float32Array> {
  const out = new Float32Array(Math.floor(x.length / (from / to)));
  for (let i = 0; i < out.length; i += RESAMPLE_PART) {
    resampleRange(x, from, to, out, i, Math.min(out.length, i + RESAMPLE_PART));
    await yieldToPage();
  }
  return out;
}

/** Output samples per part of resampleYielding (about 1 ms of work on a laptop). */
const RESAMPLE_PART = 2000;

function resampleRange(x: Float32Array, from: number, to: number, out: Float32Array, i0: number, i1: number): void {
  const ratio = from / to;
  const cutoff = Math.min(1, to / from);
  const taps = 16;
  for (let i = i0; i < i1; i++) {
    const c = i * ratio;
    const k0 = Math.floor(c);
    let s = 0;
    let wsum = 0;
    for (let k = k0 - taps + 1; k <= k0 + taps; k++) {
      if (k < 0 || k >= x.length) continue;
      const t = c - k;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t * cutoff) / (Math.PI * t * cutoff);
      const win = 0.5 + 0.5 * Math.cos((Math.PI * t) / (taps + 1));
      const w = sinc * win;
      s += x[k]! * w;
      wsum += w;
    }
    out[i] = wsum ? s / wsum : 0;
  }
}
