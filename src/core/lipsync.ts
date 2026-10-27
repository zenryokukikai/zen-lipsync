// The lip-sync computation around the two models, shared by both entries: the sentence model's inputs and the
// per-frame latents (fp16), and the frame model's inputs for a batch of video frames.

/** The two models on one backend: WebGPU (browser) or the CPU (Node). */
export interface LipModels {
  readonly backend: 'webgpu' | 'cpu';
  /** Bytes of the weight files downloaded. */
  readonly weightBytes: number;
  /** Sentence model: mel [T,80] and ref [T,128] -> latent [T,128]. */
  sentence(mel: Float32Array, ref: Float32Array, T: number): Promise<Float32Array>;
  /** Frame model: feat [n,32,20,44], z [n,640] -> delta [n,3,160,352]. */
  frame(feat: Float32Array, z: Float32Array, n: number): Promise<Float32Array>;
}

const F16 = new Float32Array(65536);
for (let x = 0; x < 65536; x++) {
  const s = x & 0x8000 ? -1 : 1;
  const e = (x >> 10) & 0x1f;
  const f = x & 0x3ff;
  F16[x] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
}
const hf = new Float32Array(1);
const hi = new Uint32Array(hf.buffer);
/** float32 -> fp16 bits, round to nearest. */
function toHalf(v: number): number {
  hf[0] = v;
  const x = hi[0]!;
  const sgn = (x >>> 16) & 0x8000;
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e >= 31) return sgn | 0x7c00 | (Number.isNaN(v) ? 0x200 : 0);
  if (e <= 0) {
    if (e < -10) return sgn;
    m |= 0x800000;
    const sh = 14 - e;
    return sgn | ((m + (1 << (sh - 1))) >> sh);
  }
  return (sgn | (e << 10) | (m >> 13)) + ((m >> 12) & 1);
}

/** What the lip-sync needs of a loaded avatar pack. */
export interface LipAvatar {
  readonly meta: { readonly latent_window: number };
  /** fp16 bits [source frames, featSize]. */
  readonly feat: Uint16Array;
  readonly featSize: number;
  /** float32 [source frames, 128]. */
  readonly refLatent: Float32Array;
  /** Source frame shown at walk step (the walk loops). */
  walk(step: number): number;
}

export interface SentenceInput {
  /** Video frame (walk step) where the sentence starts. */
  startStep: number;
  frames: number;
  melFrames: number;
  gather: number[];
  mel: Float32Array;
}

/** The sentence model over a sentence: the mouth-motion latent of every video frame, fp16 bits [frames,128]. */
export async function sentenceLatents(lips: LipModels, avatar: LipAvatar, s: SentenceInput): Promise<Uint16Array> {
  const T = s.melFrames;
  const us = Array.from({ length: s.frames }, (_, f) => avatar.walk(s.startStep + f));
  // reference input: the source frame's latent at the video frame nearest to each mel column
  const ref = new Float32Array(T * 128);
  let j = 0;
  for (let t = 0; t < T; t++) {
    while (j + 1 < s.frames && Math.abs(t - s.gather[j + 1]!) < Math.abs(t - s.gather[j]!)) j++;
    ref.set(avatar.refLatent.subarray(us[j]! * 128, us[j]! * 128 + 128), t * 128);
  }
  const z = await lips.sentence(s.mel, ref, T);
  const zl = new Uint16Array(s.frames * 128);
  for (let f = 0; f < s.frames; f++) {
    const g = s.gather[f]!;
    for (let k = 0; k < 128; k++) zl[f * 128 + k] = toHalf(z[g * 128 + k]!);
  }
  return zl;
}

function halfToFloat(d: Uint16Array): Float32Array {
  const out = new Float32Array(d.length);
  for (let i = 0; i < d.length; i++) out[i] = F16[d[i]!]!;
  return out;
}

/**
 * A log-mel value this low is silence: logMel() clamps every band to log(1e-5) = -11.51, so a frame whose loudest band
 * stays under log(1e-5) + 0.5 carried no sound (quiet room, the padding before/after a line).
 */
const SILENT_BELOW = Math.log(1e-5) + 0.5;

/**
 * Which of a sentence's video frames have sound under them (false: silence). The frame model is trained on speech and
 * still predicts a mouth shape on silent input; drawing it over the idle face would double the lips. Silent frames
 * therefore keep the idle frame unchanged.
 */
export function speechMask(mel: Float32Array, melFrames: number, gather: number[], frames: number): boolean[] {
  const out = new Array<boolean>(frames);
  for (let f = 0; f < frames; f++) {
    const t = Math.min(melFrames - 1, Math.max(0, gather[f] ?? 0));
    let loudest = -Infinity;
    for (let c = 0; c < 80; c++) {
      const v = mel[t * 80 + c]!;
      if (v > loudest) loudest = v;
    }
    out[f] = loudest >= SILENT_BELOW;
  }
  return out;
}

/**
 * Mouth changes of frames [f, f+n) of a sentence (zl: its latents, fp16 bits [frames,128]) for source frames us:
 * float32 [n,3,boxH,boxW] in 0..255 units. Input: each frame's features and the latents of frames t-2..t+2 (clamped).
 * silent: per-sentence speechMask(); the changes of silent frames are zeroed so those frames stay as they are.
 */
export async function inferFrames(lips: LipModels, avatar: LipAvatar, us: number[], zl: Uint16Array, f: number, n: number, silent?: boolean[]): Promise<Float32Array> {
  const frames = zl.length / 128;
  const win = avatar.meta.latent_window;
  const half = Math.floor(win / 2);
  const z = new Uint16Array(n * win * 128);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < win; k++) {
      const j = Math.min(frames - 1, Math.max(0, f + i + k - half));
      z.set(zl.subarray(j * 128, j * 128 + 128), (i * win + k) * 128);
    }
  }
  const feat = new Uint16Array(n * avatar.featSize);
  us.forEach((u, i) => feat.set(avatar.feat.subarray(u * avatar.featSize, (u + 1) * avatar.featSize), i * avatar.featSize));
  const delta = await lips.frame(halfToFloat(feat), halfToFloat(z), n);
  if (silent) {
    const plane = delta.length / (n * 3);
    for (let i = 0; i < n; i++) {
      if (silent[f + i]) continue;
      delta.fill(0, i * 3 * plane, (i + 1) * 3 * plane);
    }
  }
  return delta;
}

