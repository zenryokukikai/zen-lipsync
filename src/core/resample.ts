// Resampling of speech audio to the 16 kHz the audio features use (shared by both entries).

/** Simple resampler (windowed sinc, 16 taps each side): good enough for 16 kHz log-mel features. */
export function resample(x: Float32Array, from: number, to: number): Float32Array {
  const ratio = from / to;
  const n = Math.floor(x.length / ratio);
  const out = new Float32Array(n);
  const cutoff = Math.min(1, to / from);
  const taps = 16;
  for (let i = 0; i < n; i++) {
    const c = i * ratio;
    const i0 = Math.floor(c);
    let s = 0;
    let wsum = 0;
    for (let k = i0 - taps + 1; k <= i0 + taps; k++) {
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
  return out;
}
