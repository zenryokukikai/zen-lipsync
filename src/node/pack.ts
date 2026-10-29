// One idle loop of an avatar on Node: its manifest, features, colour tables, the source frames' crops and alpha masks
// (version 1: PNG files, kept compressed and decoded on first use; version 2: one frames.pack of WebP images, decoded
// when the pack loads; either way kept decoded: about 65 MB per loop of 251 frames) and the idle video (ffmpeg).
import type { LipAvatar } from '../core/lipsync.js';
import { fileHash, loadPackData, type PackRef, type PackSource, type PackMeta } from '../core/pack.js';
import { AvatarPackError } from '../core/errors.js';
import { decodePng, type DecodedImage } from './png.js';
import { decodeWebp } from './webp.js';
import { readFramesPackIndex } from '../core/frames-pack.js';
import type { SourceImages } from './compositor.js';
import { IdleVideo, type IdleCache } from './idle-video.js';

export class NodePack implements LipAvatar {
  readonly meta: PackMeta;
  readonly feat: Uint16Array;
  readonly featSize: number;
  readonly refLatent: Float32Array;
  readonly fromVideo: Float32Array[];
  readonly toVideo: Float32Array[];
  readonly modelDir: string;
  readonly video: IdleVideo;
  private readonly files: Map<string, Uint8Array>;
  private readonly decoded = new Map<number, SourceImages>();

  private constructor(readonly name: string, readonly dir: string, pack: PackRef, data: Awaited<ReturnType<typeof loadPackData>>, files: Map<string, Uint8Array>, video: IdleVideo) {
    this.meta = pack.meta;
    this.feat = data.feat;
    this.refLatent = data.refLatent;
    const [c, h, w] = pack.meta.feat_dims;
    this.featSize = c * h * w;
    this.fromVideo = data.tone.from_video.map((t) => Float32Array.from(t));
    this.toVideo = data.tone.to_video.map((t) => Float32Array.from(t));
    this.modelDir = data.modelDir;
    this.files = files;
    this.video = video;
  }

  static async load(src: PackSource, pack: PackRef, opts: { ffmpeg: string; idleCache: IdleCache; localDir: (p: string) => Promise<string> }): Promise<NodePack> {
    const m = pack.meta;
    const data = await loadPackData(src, pack);
    const files = new Map<string, Uint8Array>();
    const decoded = new Map<number, [DecodedImage, DecodedImage]>();
    if (m.frames) {
      const all = new Uint8Array(await src.bytes(src.join(pack.dir, m.frames.pack), fileHash(m, m.frames.pack)));
      const index = readFramesPackIndex(all, `${pack.dir}/${m.frames.pack}`, m.source_frames);
      for (let u = 0; u < m.source_frames; u++) {
        const [co, cl, ao, al] = index.frames[u]!;
        decoded.set(u, [await decodeWebp(all.subarray(co, co + cl), 3), await decodeWebp(all.subarray(ao, ao + al), 1)]);
      }
    } else {
      const names: string[] = [];
      for (let u = 0; u < m.source_frames; u++) names.push(`frames/crop_${String(u).padStart(3, '0')}.png`, `frames/alpha_${String(u).padStart(3, '0')}.png`);
      let next = 0;
      const worker = async (): Promise<void> => {
        while (next < names.length) {
          const n = names[next++]!;
          files.set(n, new Uint8Array(await src.bytes(src.join(pack.dir, n), fileHash(m, n))));
        }
      };
      await Promise.all([worker(), worker(), worker(), worker()]);
    }
    const webm = m.video.find((v) => v.src.endsWith('.webm')) ?? m.video[0];
    if (!webm) throw new AvatarPackError(`${pack.dir}: no idle video`);
    const file = await opts.localDir(src.join(pack.dir, webm.src));
    const video = new IdleVideo(file, m.frame_size[0], m.frame_size[1], m.walk.length, opts.ffmpeg, opts.idleCache);
    await video.open();
    const p = new NodePack(pack.name, pack.dir, pack, data, files, video);
    for (const [u, [crop, alpha]] of decoded) p.keep(u, crop, alpha);
    return p;
  }

  walk(step: number): number {
    const n = this.meta.walk.length;
    return this.meta.walk[((step % n) + n) % n]!;
  }

  /** Crop (RGB) and alpha of source frame u. */
  source(u: number): SourceImages {
    const s = this.decoded.get(u);
    if (s) { this.decoded.delete(u); this.decoded.set(u, s); return s; }
    const n = String(u).padStart(3, '0');
    return this.keep(u, decodePng(this.files.get(`frames/crop_${n}.png`)!), decodePng(this.files.get(`frames/alpha_${n}.png`)!));
  }

  private keep(u: number, crop: DecodedImage, alpha: DecodedImage): SourceImages {
    const geo = this.meta.geo[String(u)];
    if (!geo) throw new AvatarPackError(`${this.dir}: source frame ${u} has no geometry`);
    const rgb = crop.channels === 3 ? crop.data : toRgb(crop.data, crop.channels);
    const a = alpha.channels === 1 ? alpha.data : toGrey(alpha.data, alpha.channels);
    const s = { geo, crop: rgb, cropW: crop.width, cropH: crop.height, alpha: a };
    this.decoded.set(u, s);
    return s;
  }

  /** Bytes held: compressed frames, features, decoded images, the idle video's frames. */
  get bytes(): number {
    let b = this.feat.byteLength + this.refLatent.byteLength + this.video.bytes;
    for (const f of this.files.values()) b += f.length;
    for (const s of this.decoded.values()) b += s.crop.length + s.alpha.length;
    return b;
  }

  close(): void { this.video.close(); this.decoded.clear(); }
}

function toRgb(d: Uint8Array, ch: number): Uint8Array {
  const n = d.length / ch; const o = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) { const v = d[i * ch]!; if (ch === 1) { o[i * 3] = v; o[i * 3 + 1] = v; o[i * 3 + 2] = v; } else { o[i * 3] = v; o[i * 3 + 1] = d[i * ch + 1]!; o[i * 3 + 2] = d[i * ch + 2]!; } }
  return o;
}
function toGrey(d: Uint8Array, ch: number): Uint8Array {
  const n = d.length / ch; const o = new Uint8Array(n);
  for (let i = 0; i < n; i++) o[i] = d[i * ch]!;
  return o;
}
