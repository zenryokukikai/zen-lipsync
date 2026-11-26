// Resampling of speech audio to the 16 kHz the audio features use (shared by both entries).
import { yieldToPage } from './mel.js';

/** Taps each side of the windowed sinc. */
const TAPS = 16;

/**
 * Weights per output phase: when from / to is a ratio of small integers (44100 / 16000 = 441 / 160), the fractional
 * position of output i repeats every `den` outputs, so the 2 * TAPS weights are computed once per phase instead of a
 * sin and a cos per tap per output. The table's phase positions are exact (ph * ratio, ph < den); the per-output form
 * took i * ratio, whose rounding grows with i: on a few minutes of 44.1 kHz audio about 0.6% of the outputs differ
 * by one float32 step (3e-8), none on 48 kHz.
 */
const tables = new Map<string, { den: number; num: number; w: Float64Array } | null>();
function phaseTable(from: number, to: number): { den: number; num: number; w: Float64Array } | null {
  const key = `${from}/${to}`;
  if (tables.has(key)) return tables.get(key)!;
  const g = gcd(Math.round(from), Math.round(to));
  const num = Math.round(from) / g, den = Math.round(to) / g;
  if (from !== Math.round(from) || to !== Math.round(to) || den > 4096) { tables.set(key, null); return null; }
  const ratio = from / to;
  const cutoff = Math.min(1, to / from);
  const w = new Float64Array(den * 2 * TAPS);
  for (let ph = 0; ph < den; ph++) {
    const c = ph * ratio;
    const i0 = Math.floor(c);
    for (let j = 0; j < 2 * TAPS; j++) {
      const k = i0 - TAPS + 1 + j;
      const t = c - k;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t * cutoff) / (Math.PI * t * cutoff);
      const win = 0.5 + 0.5 * Math.cos((Math.PI * t) / (TAPS + 1));
      w[ph * 2 * TAPS + j] = sinc * win;
    }
  }
  const r = { den, num, w };
  tables.set(key, r);
  return r;
}
function gcd(a: number, b: number): number { while (b) [a, b] = [b, a % b]; return a; }

/** Simple resampler (windowed sinc, 16 taps each side): good enough for 16 kHz log-mel features. */
export function resample(x: Float32Array, from: number, to: number): Float32Array {
  const out = new Float32Array(outputLength(x, from, to));
  resampleRange(x, from, to, out, 0, out.length);
  return out;
}

/** resample in parts of RESAMPLE_PART output samples, with yieldToPage() between them (see logMelYielding). */
export async function resampleYielding(x: Float32Array, from: number, to: number): Promise<Float32Array> {
  const out = new Float32Array(outputLength(x, from, to));
  for (let i = 0; i < out.length; i += RESAMPLE_PART) {
    resampleRange(x, from, to, out, i, Math.min(out.length, i + RESAMPLE_PART));
    await yieldToPage();
  }
  return out;
}

/** Output samples per part of resampleYielding (about 1 ms of work on a laptop). */
const RESAMPLE_PART = 2000;

function outputLength(x: Float32Array, from: number, to: number): number {
  if (!(from > 0 && to > 0 && Number.isFinite(from) && Number.isFinite(to))) throw new RangeError(`resample: sample rates must be positive (${from} -> ${to})`);
  return Math.floor(x.length / (from / to));
}

function resampleRange(x: Float32Array, from: number, to: number, out: Float32Array, i0: number, i1: number): void {
  const ratio = from / to;
  const tab = phaseTable(from, to);
  if (tab) {
    // output i = q * den + ph sits at input q * num + ph * ratio: the same taps as below, weights from the table
    for (let i = i0; i < i1; i++) {
      const ph = i % tab.den;
      const q = (i - ph) / tab.den;
      const k0 = q * tab.num + Math.floor(ph * ratio);
      const wo = ph * 2 * TAPS;
      let s = 0;
      let wsum = 0;
      for (let j = 0; j < 2 * TAPS; j++) {
        const k = k0 - TAPS + 1 + j;
        if (k < 0 || k >= x.length) continue;
        const w = tab.w[wo + j]!;
        s += x[k]! * w;
        wsum += w;
      }
      out[i] = wsum ? s / wsum : 0;
    }
    return;
  }
  const cutoff = Math.min(1, to / from);
  for (let i = i0; i < i1; i++) {
    const c = i * ratio;
    const k0 = Math.floor(c);
    let s = 0;
    let wsum = 0;
    for (let k = k0 - TAPS + 1; k <= k0 + TAPS; k++) {
      if (k < 0 || k >= x.length) continue;
      const t = c - k;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t * cutoff) / (Math.PI * t * cutoff);
      const win = 0.5 + 0.5 * Math.cos((Math.PI * t) / (TAPS + 1));
      const w = sinc * win;
      s += x[k]! * w;
      wsum += w;
    }
    out[i] = wsum ? s / wsum : 0;
  }
}
