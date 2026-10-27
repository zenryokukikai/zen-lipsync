// Avatar packs (docs/AVATAR-PACK.md): the formats, and loading them through a PackSource so the browser (fetch +
// Cache Storage) and Node (files or URLs) share the same reading and checks.
import { AvatarPackError } from './errors.js';

export interface Geometry {
  /** Patch rectangle on the video frame: x, y, width, height. */
  rect: [number, number, number, number];
  /** 2x3 matrix from the 512x512 face crop to the patch rectangle. */
  matrix: [[number, number, number], [number, number, number]];
  abs_x: number;
  abs_y: number;
  /** Part of the 512 crop stored in the crop_ files: x0, y0, x1, y1. */
  crop_box: [number, number, number, number];
}

/** manifest.json of one avatar pack (one idle loop). */
export interface PackMeta {
  format: 'zen-lipsync-avatar';
  version: 1;
  name: string;
  /** Video frame size [width, height] and frame rate. */
  frame_size: [number, number];
  fps: number;
  /** Source frames (crops, features) and the walk through them: video frame k shows source frame walk[k % walk.length]. */
  source_frames: number;
  walk: number[];
  geo: Record<string, Geometry>;
  /** Mouth box in the 512 crop (rows y0..y1, columns x0..x1). */
  box: { y0: number; y1: number; x0: number; x1: number };
  /** Frame model input shapes: feat [C,H,W], latent window frames. */
  feat_dims: [number, number, number];
  latent_window: number;
  /** Idle video files, best first (the first one the player can play is used). */
  video: { src: string; type: string }[];
  /** Files with size and sha256, relative to the pack (the model files are verified). */
  files: Record<string, { bytes: number; sha256: string }>;
  /** Directory of the model weights relative to the pack (default: the pack itself); packs of one avatar share it. */
  models?: string;
  license?: string;
  credit?: string;
}

/** avatar.json: one avatar with one or more idle loops (each an avatar pack directory), sharing its models. */
export interface AvatarSet {
  format: 'zen-lipsync-avatar-set';
  version: 1;
  id: string;
  name: Record<string, string>;
  /** The avatar's model set version, e.g. 'marine-1.0' (optional; informational). */
  model?: string;
  sources: Record<string, string>;
  default: string;
  idleSwitch?: string;
  license?: string;
  credit?: string;
}

/** Colour tables of a pack (tone.json): per channel, 256 entries each way. */
export interface ToneTables { to_video: number[][]; from_video: number[][] }

/** Where avatar files come from. Paths are joined with join(); a path may contain '..'. */
export interface PackSource {
  /** Joins a directory and a relative path ('a/b' + '../c' -> 'a/c'). */
  join(dir: string, rel: string): string;
  /** The bytes of a file; sha256 (hex) when known, then the data must match it. */
  bytes(path: string, sha256?: string): Promise<ArrayBuffer>;
  /** A JSON file. */
  json<T>(path: string): Promise<T>;
}

/** One idle loop of an avatar: its pack directory and manifest. */
export interface PackRef { name: string; dir: string; meta: PackMeta }

/** An avatar as listed by its avatar.json (or a single pack), not yet loaded. */
export interface AvatarListing {
  /** The avatar.json, or null for a single pack. */
  set: AvatarSet | null;
  /** The idle loops, the first one shown first. */
  packs: PackRef[];
  /** Directory of the avatar (avatar.json's directory, or the pack). */
  dir: string;
}

/** The data of one pack the lip-sync needs (besides the frame images and the video). */
export interface PackData {
  /** fp16 bits [source frames, C*H*W]. */
  feat: Uint16Array;
  /** float32 [source frames, 128]. */
  refLatent: Float32Array;
  tone: ToneTables;
  /** Directory of the model weights. */
  modelDir: string;
}

export const MODEL_FILES = { sentence: 'sentence.wgw', frameI8: 'frame.i8.wgw', frameF32: 'frame.wgw', frameCpu: 'frame.lip8' } as const;

/** Checks a manifest.json (shape and sizes the runtime relies on). */
export function checkPackMeta(meta: unknown, where: string): PackMeta {
  const m = meta as Partial<PackMeta> | null;
  if (!m || m.format !== 'zen-lipsync-avatar' || m.version !== 1) throw new AvatarPackError(`${where}: not a ZEN-LipSync avatar pack (version 1)`);
  const need = <K extends keyof PackMeta>(k: K, ok: (v: unknown) => boolean): void => {
    if (!ok(m[k])) throw new AvatarPackError(`${where}: "${String(k)}" is missing or invalid`);
  };
  const isInt = (v: unknown): boolean => Number.isInteger(v) && (v as number) > 0;
  need('frame_size', (v) => Array.isArray(v) && v.length === 2 && v.every(isInt));
  need('fps', (v) => typeof v === 'number' && v > 0);
  need('source_frames', isInt);
  need('walk', (v) => Array.isArray(v) && v.length > 0 && v.every((u) => Number.isInteger(u) && u >= 0 && u < m.source_frames!));
  need('geo', (v) => typeof v === 'object' && v !== null);
  need('box', (v) => typeof v === 'object' && v !== null);
  need('feat_dims', (v) => Array.isArray(v) && v.length === 3 && v.every(isInt));
  need('latent_window', isInt);
  need('video', (v) => Array.isArray(v) && v.length > 0);
  need('files', (v) => typeof v === 'object' && v !== null);
  return m as PackMeta;
}

/** Checks an avatar.json. */
export function checkAvatarSet(set: unknown, where: string): AvatarSet {
  const s = set as Partial<AvatarSet> | null;
  if (!s || s.format !== 'zen-lipsync-avatar-set' || s.version !== 1) throw new AvatarPackError(`${where}: not a ZEN-LipSync avatar (avatar.json version 1)`);
  if (!s.sources || typeof s.sources !== 'object' || Object.keys(s.sources).length === 0) throw new AvatarPackError(`${where}: "sources" is missing`);
  if (typeof s.default !== 'string' || !(s.default in s.sources)) throw new AvatarPackError(`${where}: "default" must name one of "sources"`);
  return s as AvatarSet;
}

/**
 * Lists an avatar: root is an avatar.json (path or URL ending in .json) or a pack directory. first: the idle loop to
 * put first (default: the avatar's default).
 */
export async function listAvatar(src: PackSource, root: string, first: string | null = null): Promise<AvatarListing> {
  if (root.endsWith('.json')) {
    const set = checkAvatarSet(await src.json(root), root);
    const dir = src.join(root, '..');
    const start = first ?? set.default;
    if (!(start in set.sources)) throw new AvatarPackError(`${root}: no idle loop "${start}" (loops: ${Object.keys(set.sources).join(', ')})`);
    const names = [start, ...Object.keys(set.sources).filter((k) => k !== start)];
    const packs = await Promise.all(names.map(async (name) => {
      const d = src.join(dir, set.sources[name]!);
      const mf = src.join(d, 'manifest.json');
      return { name, dir: d, meta: checkPackMeta(await src.json(mf), mf) };
    }));
    return { set, packs, dir };
  }
  if (first !== null && first !== 'default') throw new AvatarPackError(`${root}: a single pack has one idle loop ("default")`);
  const mf = src.join(root, 'manifest.json');
  return { set: null, packs: [{ name: 'default', dir: root, meta: checkPackMeta(await src.json(mf), mf) }], dir: root };
}

/** sha256 of a pack file, when the manifest gives a valid one. */
export function fileHash(meta: PackMeta, rel: string): string | undefined {
  const h = meta.files[rel]?.sha256;
  return h && /^[a-f0-9]{64}$/.test(h) ? h : undefined;
}

/** The model weights' directory of a pack, and the manifest name of a model file in it. */
export function modelPath(src: PackSource, pack: PackRef, file: string): { path: string; rel: string } {
  const rel = pack.meta.models ? `${pack.meta.models.replace(/\/+$/, '')}/${file}` : file;
  return { path: src.join(pack.dir, rel), rel };
}

/** Reads a pack's features, reference latents and colour tables, checking their sizes. */
export async function loadPackData(src: PackSource, pack: PackRef): Promise<PackData> {
  const m = pack.meta;
  const [feat, ref, tone] = await Promise.all([
    src.bytes(src.join(pack.dir, 'features.f16'), fileHash(m, 'features.f16')),
    src.bytes(src.join(pack.dir, 'ref_latent.f32'), fileHash(m, 'ref_latent.f32')),
    src.json<ToneTables>(src.join(pack.dir, 'tone.json')),
  ]);
  const [c, h, w] = m.feat_dims;
  const featSize = c * h * w;
  if (feat.byteLength !== m.source_frames * featSize * 2) throw new AvatarPackError(`${pack.dir}/features.f16: ${feat.byteLength} bytes, expected ${m.source_frames} x ${featSize} fp16 values`);
  if (ref.byteLength !== m.source_frames * 128 * 4) throw new AvatarPackError(`${pack.dir}/ref_latent.f32: ${ref.byteLength} bytes, expected ${m.source_frames} x 128 float32 values`);
  if (!Array.isArray(tone.to_video) || tone.to_video.length !== 3 || !Array.isArray(tone.from_video) || tone.from_video.length !== 3) throw new AvatarPackError(`${pack.dir}/tone.json: expected 3 colour tables each way`);
  const modelDir = m.models ? src.join(pack.dir, m.models) : pack.dir;
  return { feat: new Uint16Array(feat), refLatent: new Float32Array(ref), tone, modelDir };
}

/** sha256 (hex) of bytes, with the Web Crypto API (browsers and Node 20+). */
export async function sha256Hex(data: ArrayBuffer): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', data))].map((v) => v.toString(16).padStart(2, '0')).join('');
}
