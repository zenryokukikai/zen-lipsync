// An avatar pack: the idle video's per-frame geometry, the precomputed features of every source frame, the colour
// tables, the mouth crops and alpha masks, and the two models' weights. See docs/AVATAR-PACK.md for the format.
import { WebGPUUnavailableError } from '../core/errors.js';
import { listAvatar, loadPackData, type Geometry, type PackMeta, type PackRef } from '../core/pack.js';
import type { LipWeights } from '../core/types.js';
import { browserPackSource, registerPackHashes } from './pack-source.js';
import { WgslLips } from './wgsl.js';
import type { LipAvatar, LipModels } from '../core/lipsync.js';

export { inferFrames, sentenceLatents, speechMask, type LipModels, type SentenceInput } from '../core/lipsync.js';

export type { Geometry, PackMeta } from '../core/pack.js';

export interface SourceFrame {
  geo: Geometry;
  /** RGBA of the stored part of the 512 crop. */
  crop: ImageData;
  /** Patch alpha as the alpha channel of an image of the rect's size. */
  alpha: ImageBitmap;
}

/** Columns of the video frame that are shown (the page may crop the frame to a portrait). */
export interface View {
  x: number;
  width: number;
  height: number;
}

/** Decoded source frames kept at once (most recently used). The rest stay compressed as Blobs. */
const DECODED_FRAMES = 24;

export class Avatar implements LipAvatar {
  private readonly blobs = new Map<string, Promise<Blob>>();
  private readonly decoded = new Map<number, Promise<SourceFrame>>();
  /** fp16 bits [source frames, featSize]. */
  readonly feat: Uint16Array;
  readonly featSize: number;
  /** Per channel colour tables: fromVideo maps a video pixel value to the model's colour, toVideo back (256 entries each). */
  readonly fromVideo: Float32Array[];
  readonly toVideo: Float32Array[];
  bytesLoaded = 0;
  /** URL of the model weights' directory. */
  readonly modelBase: string;

  readonly base: string;
  readonly meta: PackMeta;
  readonly refLatent: Float32Array;

  private constructor(pack: PackRef, data: Awaited<ReturnType<typeof loadPackData>>) {
    this.base = pack.dir;
    this.meta = pack.meta;
    this.feat = data.feat;
    this.refLatent = data.refLatent;
    const [c, h, w] = pack.meta.feat_dims;
    this.featSize = c * h * w;
    this.modelBase = data.modelDir;
    this.fromVideo = data.tone.from_video.map((t) => Float32Array.from(t));
    this.toVideo = data.tone.to_video.map((t) => Float32Array.from(t));
  }

  /** One listed pack (core/pack.ts listAvatar) with its features, reference latents and colour tables. */
  static async fromPack(pack: PackRef, onBytes: (n: number) => void = () => undefined): Promise<Avatar> {
    registerPackHashes(pack.dir, pack.meta.files);
    return new Avatar(pack, await loadPackData(browserPackSource(onBytes), pack));
  }

  /** base: the avatar pack's URL (a directory with manifest.json). */
  static async load(base: string, onBytes: (n: number) => void = () => undefined): Promise<Avatar> {
    const listing = await listAvatar(browserPackSource(onBytes), base.replace(/\/+$/, ''));
    return Avatar.fromPack(listing.packs[0]!, onBytes);
  }

  /** Source frame shown at video frame step (the walk loops). */
  walk(step: number): number {
    const n = this.meta.walk.length;
    return this.meta.walk[((step % n) + n) % n]!;
  }

  /** Decoded source frame u; at most DECODED_FRAMES stay decoded (the oldest is closed). */
  frame(u: number): Promise<SourceFrame> {
    let f = this.decoded.get(u);
    if (f) {
      this.decoded.delete(u);
      this.decoded.set(u, f);
      return f;
    }
    f = this.decodeFrame(u);
    this.decoded.set(u, f);
    while (this.decoded.size > DECODED_FRAMES) {
      const [oldest, old] = this.decoded.entries().next().value as [number, Promise<SourceFrame>];
      this.decoded.delete(oldest);
      void old.then((fr) => fr.alpha.close(), () => undefined);
    }
    return f;
  }

  /** Downloads the compressed crops and masks of every source frame (no decoding), 4 at a time. */
  async fetchAllFrames(): Promise<void> {
    const names: string[] = [];
    for (let u = 0; u < this.meta.source_frames; u++) names.push(fileName('crop', u), fileName('alpha', u));
    let next = 0;
    const worker = async (): Promise<void> => { while (next < names.length) await this.blob(names[next++]!); };
    await Promise.all([worker(), worker(), worker(), worker()]);
  }

  private blob(name: string): Promise<Blob> {
    let b = this.blobs.get(name);
    if (!b) {
      b = fetch(`${this.base}/frames/${name}`).then((r) => { if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`); return r.blob(); })
        .then((x) => { this.bytesLoaded += x.size; return x; });
      b.catch(() => this.blobs.delete(name));
      this.blobs.set(name, b);
    }
    return b;
  }

  private async decodeFrame(u: number): Promise<SourceFrame> {
    const geo = this.meta.geo[String(u)];
    if (!geo) throw new Error(`source frame ${u} has no geometry`);
    const [cropBlob, alphaBlob] = await Promise.all([this.blob(fileName('crop', u)), this.blob(fileName('alpha', u))]);
    const pixels = async (blob: Blob): Promise<ImageData> => {
      const bm = await createImageBitmap(blob);
      const c = new OffscreenCanvas(bm.width, bm.height);
      const g = c.getContext('2d', { willReadFrequently: true })!;
      g.drawImage(bm, 0, 0);
      bm.close();
      return g.getImageData(0, 0, c.width, c.height);
    };
    const [crop, alphaData] = await Promise.all([pixels(cropBlob), pixels(alphaBlob)]);
    // the alpha file is grey: its value becomes the alpha of a mask image
    const al = alphaData.data;
    for (let i = 0; i < al.length; i += 4) al[i + 3] = al[i]!;
    const alpha = await createImageBitmap(alphaData, { premultiplyAlpha: 'premultiply' });
    return { geo, crop, alpha };
  }
}

function fileName(kind: 'crop' | 'alpha', u: number): string {
  return `${kind}_${String(u).padStart(3, '0')}.png`;
}

export interface ModelOptions {
  /** Frame model weights: 'i8' (default, int8 restored to float32 once at load), 'f32' or 'i8q'. */
  weights?: LipWeights;
}

/** The GPU adapter, or WebGPUUnavailableError (the browser entry has no CPU fallback). */
export async function requestGpuAdapter(): Promise<{ requestDevice(): Promise<unknown> }> {
  const gpu = (globalThis.navigator as (Navigator & { gpu?: { requestAdapter(): Promise<{ requestDevice(): Promise<unknown> } | null> } }) | undefined)?.gpu;
  if (!gpu) throw new WebGPUUnavailableError('no-navigator-gpu');
  const adapter = await gpu.requestAdapter().catch(() => null);
  if (!adapter) throw new WebGPUUnavailableError('no-adapter');
  return adapter;
}

/** Both models on WebGPU. Rejects with WebGPUUnavailableError when WebGPU is missing. */
export async function loadModels(avatar: Avatar, onBytes: (n: number) => void = () => undefined, opts: ModelOptions = {}): Promise<LipModels> {
  const adapter = await requestGpuAdapter();
  const lips = await WgslLips.create(await adapter.requestDevice(), avatar.modelBase, opts.weights ?? 'i8');
  onBytes(lips.weightBytes);
  return lips;
}

/** The mouth patch of one video frame and where it goes on the canvas. */
/** One 1x1 transparent bitmap, shared by every silent frame. */
let EMPTY_PATCH: Promise<ImageBitmap> | null = null;

export interface LipPatch {
  bitmap: ImageBitmap;
  x: number;
  y: number;
}

export class Compositor {
  private readonly mc = new OffscreenCanvas(1, 1);
  private readonly mg: OffscreenCanvasRenderingContext2D;
  private readonly pc = new OffscreenCanvas(1, 1);
  private readonly pg: OffscreenCanvasRenderingContext2D;

  /** x of the video frame maps to x - shown.x on the canvas. */
  constructor(private readonly shown: View, private readonly avatar: Avatar) {
    this.mg = this.mc.getContext('2d', { willReadFrequently: true })!;
    this.pg = this.pc.getContext('2d')!;
  }

  /** A patch that draws nothing: the frames with no sound under them are shown exactly as the idle video has them. */
  async emptyPatch(x: number, y: number): Promise<LipPatch> {
    if (!EMPTY_PATCH) EMPTY_PATCH = createImageBitmap(new ImageData(1, 1));
    return { bitmap: await EMPTY_PATCH, x, y };
  }

  /** Mouth patch of the index-th frame of a batch: the change added to the source crop, warped into the patch with its alpha. */
  frame(f: SourceFrame, pred: Float32Array, index: number): LipPatch {
    const [x0, y0, x1, y1] = f.geo.crop_box;
    const img = new ImageData(new Uint8ClampedArray(f.crop.data), x1 - x0, y1 - y0);
    this.addChange(img, f, pred, index);
    return this.warp(f, img);
  }

  /** The box's change added in the model's colour (video value -> model colour, + change, -> video). Its box edges are already 0. */
  private addChange(img: ImageData, f: SourceFrame, pred: Float32Array, index: number): void {
    const a = this.avatar;
    const box = a.meta.box;
    const [x0, y0, x1, y1] = f.geo.crop_box;
    const cw = x1 - x0;
    const bw = box.x1 - box.x0;
    const plane = (box.y1 - box.y0) * bw;
    const offset = index * 3 * plane;
    const d = img.data;
    const ya = Math.max(y0, box.y0);
    const yb = Math.min(y1, box.y1);
    const xa = Math.max(x0, box.x0);
    const xb = Math.min(x1, box.x1);
    for (let y = ya; y < yb; y++) {
      const brow = (y - box.y0) * bw - box.x0;
      const crow = (y - y0) * cw - x0;
      for (let x = xa; x < xb; x++) {
        const o = (crow + x) * 4;
        for (let c = 0; c < 3; c++) {
          const delta = pred[offset + c * plane + brow + x]!;
          if (delta > -0.5 && delta < 0.5) continue;
          const v = a.fromVideo[c]![d[o + c]!]! + delta;
          d[o + c] = a.toVideo[c]![v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v)]!;
        }
      }
    }
  }

  /** Crop warped into the patch rect, cut by the patch alpha (canvas operations, no pixel read-back). */
  private warp(f: SourceFrame, img: ImageData): LipPatch {
    const [x0, y0] = f.geo.crop_box;
    if (this.mc.width !== img.width || this.mc.height !== img.height) { this.mc.width = img.width; this.mc.height = img.height; }
    this.mg.putImageData(img, 0, 0);
    const [, , rw, rh] = f.geo.rect;
    const mt = f.geo.matrix;
    if (this.pc.width !== rw || this.pc.height !== rh) { this.pc.width = rw; this.pc.height = rh; } else this.pg.clearRect(0, 0, rw, rh);
    this.pg.setTransform(mt[0][0], mt[1][0], mt[0][1], mt[1][1], mt[0][2], mt[1][2]);
    this.pg.drawImage(this.mc, x0, y0);
    this.pg.setTransform(1, 0, 0, 1, 0, 0);
    this.pg.globalCompositeOperation = 'destination-in';
    this.pg.drawImage(f.alpha, 0, 0);
    this.pg.globalCompositeOperation = 'source-over';
    return { bitmap: this.pc.transferToImageBitmap(), x: f.geo.abs_x - this.shown.x, y: f.geo.abs_y };
  }
}
