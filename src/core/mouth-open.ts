// How wide the mouth opens: the sentence model's mouth-motion latents scaled toward the closed mouth. The one place, shared
// by the browser player, the browser renderer and Node, so a setting means the same everywhere.
//
// z_closed: the source frame's own latent (its closed / idle mouth) = ref_latent.f32 (the pack's normalized reference)
// mapped back to the raw latent space with the sentence model's scale and shift (sentence.wgw `scale`, `shift`): the
// sentence model outputs raw latents (z * scale + shift), and ref_latent is (z_self - shift) / scale.
// direction: mouth_open_dir.f32 in the pack (128 float32, unit length), the direction along which the latent opens the
// mouth (built from the avatar's training data, shipped in the avatar pack). Only mode 'opening' needs it.
//
//   'opening'  p = max(0, (z - z_closed) . d);  z' = z + (scale - 1) * p * d   (lip shape and teeth kept)
//   'all'      z' = z_closed + scale * (z - z_closed)                          (every mouth move scaled)
//
// scale 1 leaves the latents untouched (the same output bytes as without the setting).
import { OptionsError } from './errors.js';

export type MouthOpenMode = 'opening' | 'all';
export const MOUTH_OPEN_MODES: readonly MouthOpenMode[] = ['opening', 'all'];

/** The setting: scale in (0, 2], 1 = as trained. */
export interface MouthOpen { scale: number; mode?: MouthOpenMode }

/** Pack file of the opening direction (optional; packs without it support mode 'all' only). */
export const MOUTH_OPEN_DIR_FILE = 'mouth_open_dir.f32';

const LATENT = 128;

/** Checks a setting; returns it with the default mode. Throws OptionsError for a bad one (where: who asked). */
export function checkMouthOpen(v: unknown, where: string): Required<MouthOpen> {
  if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new OptionsError(`${where}: mouthOpen must be { scale, mode? }`);
  const o = v as Record<string, unknown>;
  for (const k of Object.keys(o)) if (k !== 'scale' && k !== 'mode') throw new OptionsError(`${where}: mouthOpen: unknown key "${k}"`);
  const s = o['scale'];
  if (typeof s !== 'number' || !Number.isFinite(s) || s <= 0 || s > 2) throw new OptionsError(`${where}: mouthOpen.scale must be a number in (0, 2]`);
  const mode = o['mode'] ?? 'opening';
  if (!MOUTH_OPEN_MODES.includes(mode as MouthOpenMode)) throw new OptionsError(`${where}: mouthOpen.mode must be ${MOUTH_OPEN_MODES.join(' or ')}`);
  return { scale: s, mode: mode as MouthOpenMode };
}

/**
 * A setting for an avatar that can do `modes`: checked (OptionsError when invalid or the mode is not available), null
 * for none / scale 1 (as trained). The one check of create() and setMouthOpen().
 */
export function resolveMouthOpen(v: unknown, modes: readonly MouthOpenMode[], where: string): Required<MouthOpen> | null {
  if (v === undefined || v === null) return null;
  const m = checkMouthOpen(v, where);
  if (m.scale === 1) return null;
  if (!modes.includes(m.mode)) throw new OptionsError(`${where}: this avatar cannot do mouthOpen mode "${m.mode}" (it can: ${modes.join(', ') || 'none'})`);
  return m;
}

/** What an avatar can do: 'all' needs only the closed-mouth latents (every pack has them), 'opening' also the direction. */
export function mouthOpenModes(hasDirection: boolean): MouthOpenMode[] {
  return hasDirection ? ['opening', 'all'] : ['all'];
}

/** The direction file's float32 values, checked (128 values, unit length). null for no file. */
export function checkMouthOpenDirection(buf: ArrayBuffer | null, where: string): Float32Array | null {
  if (!buf) return null;
  if (buf.byteLength !== LATENT * 4) throw new OptionsError(`${where}: ${MOUTH_OPEN_DIR_FILE}: ${buf.byteLength} bytes, expected ${LATENT} float32 values`);
  const d = new Float32Array(buf.slice(0));
  let n = 0;
  for (const x of d) { if (!Number.isFinite(x)) throw new OptionsError(`${where}: ${MOUTH_OPEN_DIR_FILE}: not finite`); n += x * x; }
  if (Math.abs(Math.sqrt(n) - 1) > 1e-3) throw new OptionsError(`${where}: ${MOUTH_OPEN_DIR_FILE}: not unit length (${Math.sqrt(n).toFixed(4)})`);
  return d;
}

/**
 * Scales the latents of a sentence in place. z: raw latents [frames,128] (the sentence model's output at each video
 * frame), refs: each frame's normalized reference latent [frames,128] (ref_latent rows of the frame's source frame),
 * scale / shift: the sentence model's (sentence.wgw), direction: the pack's (null when absent).
 * Throws OptionsError when the mode needs the direction and the pack has none.
 */
export function scaleMouthOpen(z: Float32Array, refs: Float32Array, frames: number, scale: Float32Array, shift: Float32Array,
  direction: Float32Array | null, setting: Required<MouthOpen>): void {
  if (setting.scale === 1) return;
  if (setting.mode === 'opening' && !direction) throw new OptionsError('mouthOpen: mode "opening" needs the pack\'s mouth_open_dir.f32 (this avatar has none); use mode "all"');
  const s = setting.scale;
  const closed = new Float32Array(LATENT);
  for (let f = 0; f < frames; f++) {
    const o = f * LATENT;
    for (let k = 0; k < LATENT; k++) closed[k] = refs[o + k]! * scale[k]! + shift[k]!;
    if (setting.mode === 'all') {
      for (let k = 0; k < LATENT; k++) z[o + k] = closed[k]! + s * (z[o + k]! - closed[k]!);
      continue;
    }
    let p = 0;
    for (let k = 0; k < LATENT; k++) p += (z[o + k]! - closed[k]!) * direction![k]!;
    if (p <= 0) continue;
    const a = (s - 1) * p;
    for (let k = 0; k < LATENT; k++) z[o + k] = z[o + k]! + a * direction![k]!;
  }
}
