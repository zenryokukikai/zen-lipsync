// Strength of the mouth change drawn per frame of a clip (browser player, Node renderer and the inference engine).

/**
 * Silent frames right after speech over which the model's change fades out (its own output on those frames, scaled by
 * 5/6 .. 1/6), drawn on the source frame each was made for. Without it the patch came off in one frame when speech
 * stopped and the face jumped back to the idle video (the skin around the mouth brightened at once).
 * Speech frames are always drawn at full strength: nothing fades in or out before the sound does.
 */
const FADE_OUT_FRAMES = 5;

/** Per frame: 1 on speech, (FADE_OUT_FRAMES + 1 - k) / (FADE_OUT_FRAMES + 1) on the k-th silent frame after speech, else 0. */
export function patchStrength(speech: readonly boolean[]): Float32Array {
  const out = new Float32Array(speech.length);
  let since = Infinity;
  for (let f = 0; f < speech.length; f++) {
    if (speech[f]) { out[f] = 1; since = 0; continue; }
    since++;
    out[f] = since <= FADE_OUT_FRAMES ? (FADE_OUT_FRAMES + 1 - since) / (FADE_OUT_FRAMES + 1) : 0;
  }
  return out;
}


/** Scales the change of frames [f, f+n) of a batch (pred: float32 [n, ...]) by their strength (frames at 0 or 1 untouched). */
export function applyStrength(pred: Float32Array, strength: Float32Array, f: number, n: number): void {
  const plane = pred.length / n;
  for (let i = 0; i < n; i++) {
    const k = strength[f + i]!;
    if (k > 0 && k < 1) { const d = pred.subarray(i * plane, (i + 1) * plane); for (let j = 0; j < d.length; j++) d[j]! *= k; }
  }
}
