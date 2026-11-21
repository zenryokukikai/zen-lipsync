// createRenderer(): the browser's offline renderer. Speech in, video frames out, one at a time and as fast as the GPU
// allows (no audio clock, no <video>, no DOM: it runs in a Web Worker too). The same frames as the Node entry's
// frames() (25 fps, frame k on the idle video's frame walk(step + k), chunks laid end to end, idle frames before and
// after), drawn the way the real-time player draws them: the idle video frame, then the frame's mouth patch made by
// the same Compositor, with the same speech mask and patch strength. The idle video's frames come from WebCodecs by
// index (runtime/idle-frames.ts), so frame k is exactly video frame k.
import { OptionsError } from '../core/errors.js';
import { logMelYielding, MEL_SAMPLE_RATE } from '../core/mel.js';
import { framesOfMs, OutputTimeline, speechChunks } from '../core/output-timeline.js';
import { listAvatar } from '../core/pack.js';
import { applyStrength, patchStrength } from '../core/patch-strength.js';
import { resampleYielding } from '../core/resample.js';
import { resolveOptions } from '../core/options.js';
import { resolveMouthOpen, type MouthOpen } from '../core/mouth-open.js';
import type { CreateOptions, SpeechAudio } from '../core/types.js';
import { Avatar, Compositor, inferFrames, loadModels, requestGpuAdapter, sentenceLatents, speechMask, type LipModels } from '../runtime/avatar.js';
import { IdleFrames } from '../runtime/idle-frames.js';
import { MatteOutput } from '../runtime/matte.js';
import { browserPackSource } from '../runtime/pack-source.js';
import { WGSL_MAX_BATCH } from '../runtime/wgsl.js';

export interface RendererOptions extends CreateOptions {
  /**
   * Cut-out output: the file name of a stacked matte video in the idle loop's pack directory (top half: the frame with
   * the background's colour taken out, bottom half: the person's matte; same frames and frame rate as the idle video, as
   * create({ matte })). The frames then have a transparent background.
   */
  matte?: string;
  /** Output width in pixels (the height keeps the aspect ratio); default: the avatar's frame size. */
  width?: number;
}

/** One output frame. */
export interface RenderedFrame {
  /** Frame number in this output (25 fps), and its time. */
  index: number;
  timeMs: number;
  /**
   * The picture. The caller closes it (it can go straight to a VideoEncoder). With a matte its alpha is the person's
   * matte; drawn on a canvas (drawImage) it composites as any transparent image, read back (copyTo / getImageData) its
   * colour is straight (not premultiplied).
   */
  image: VideoFrame;
  /** True on the frames of a speech chunk (as the Node entry's frames()); idle frames before and after are false. */
  speaking: boolean;
  /** This frame's audio (mono float32 at sampleRate; silence on idle frames). */
  audio: Float32Array;
  sampleRate: number;
}

export interface RenderFramesOptions {
  /** Idle frames before the speech (ms, default 0) and after it (default 0). */
  idleBeforeMs?: number;
  idleAfterMs?: number;
  /** Sample rate of the output audio (default: the first chunk's rate). */
  audioSampleRate?: number;
}

const FPS = 25;

interface Prepared {
  audio: Float32Array;
  rate: number;
  frames: number;
  walkStart: number;
  zl: Uint16Array;
  speech: boolean[];
  strength: Float32Array;
}

export class LipSyncRenderer {
  /** Walk position of the next frame (the idle loop goes on across outputs, as on Node). */
  private step = 0;
  private busy = false;
  private destroyed = false;
  private readonly comp: Compositor;
  /** The frame (and its patch) is composed here: the idle frame size, or the stacked matte video's (twice as tall). */
  private readonly canvas: OffscreenCanvas;
  private readonly g: OffscreenCanvasRenderingContext2D;
  private readonly matte: MatteOutput | null;
  private readonly matteCanvas: OffscreenCanvas | null;
  private readonly scaled: { canvas: OffscreenCanvas; g: OffscreenCanvasRenderingContext2D } | null;

  /** How wide the mouth opens (create()'s mouthOpen, core/mouth-open.ts); null = unchanged. */
  private mouthOpen: MouthOpen | null = null;
  /** True when the mouth opening can be scaled (the pack has mouth_open_dir.f32). */
  get mouthOpenAvailable(): boolean { return this.avatar.mouthOpenDir !== null; }
  /** Changes how wide the mouth opens for the next frames() (null / scale 1 = unchanged). OptionsError when this avatar cannot do the mode. */
  setMouthOpen(mouthOpen: MouthOpen | null): void { this.mouthOpen = resolveMouthOpen(mouthOpen, this.mouthOpenAvailable, 'setMouthOpen()'); }
  /** @internal use createRenderer() */
  constructor(readonly avatar: Avatar, readonly lips: LipModels, private readonly video: IdleFrames, readonly idleLoop: string, matte: boolean, width: number | undefined) {
    const [W, H] = avatar.meta.frame_size;
    if (matte ? video.width !== W || video.height !== 2 * H : video.width !== W || video.height !== H) {
      throw new Error(`ZEN-LipSync: the ${matte ? 'matte' : 'idle'} video is ${video.width}x${video.height}, the pack's frames are ${W}x${H}${matte ? ' (a matte video is the frame size stacked twice)' : ''}`);
    }
    if (video.length < avatar.meta.walk.length) throw new Error(`ZEN-LipSync: the ${matte ? 'matte' : 'idle'} video has ${video.length} frames, the pack's loop ${avatar.meta.walk.length}`);
    this.comp = new Compositor({ x: 0, width: W, height: H }, avatar);
    this.canvas = new OffscreenCanvas(video.width, video.height);
    this.g = this.canvas.getContext('2d', { alpha: false })!;
    this.matteCanvas = matte ? new OffscreenCanvas(W, H) : null;
    this.matte = matte ? new MatteOutput(this.matteCanvas!, this.canvas) : null;
    if (width !== undefined && width !== W) {
      const c = new OffscreenCanvas(width, Math.round((H * width) / W / 2) * 2);
      const g = c.getContext('2d', { alpha: matte })!;
      g.imageSmoothingQuality = 'high';
      this.scaled = { canvas: c, g };
    } else this.scaled = null;
  }

  /** Output frame size. */
  get width(): number { return this.scaled ? this.scaled.canvas.width : this.avatar.meta.frame_size[0]; }
  get height(): number { return this.scaled ? this.scaled.canvas.height : this.avatar.meta.frame_size[1]; }

  private live(): void { if (this.destroyed) throw new Error('ZEN-LipSync: the renderer was destroyed'); }

  /**
   * Frames of speech (one clip, or chunks as they arrive), 25 per second, each with its share of the audio. Each frame
   * is made when it is asked for (no real-time clock). One output at a time per renderer.
   */
  async *frames(input: SpeechAudio | AsyncIterable<SpeechAudio> | Iterable<SpeechAudio>, opts: RenderFramesOptions = {}): AsyncGenerator<RenderedFrame> {
    this.live();
    if (this.busy) throw new OptionsError('frames(): this renderer is already producing an output (one at a time; create another renderer for parallel outputs)');
    this.busy = true;
    try { yield* this.produce(speechChunks(input), opts); } finally { this.busy = false; }
  }

  /** n idle frames (the loop goes on from where the last output left it; silence at audioSampleRate, default 48000). */
  async *idle(n: number, opts: { audioSampleRate?: number } = {}): AsyncGenerator<RenderedFrame> {
    this.live();
    if (!Number.isInteger(n) || n < 0) throw new OptionsError('idle(): n must be a whole number of frames');
    if (this.busy) throw new OptionsError('idle(): this renderer is already producing an output');
    this.busy = true;
    try {
      const tl = new OutputTimeline(opts.audioSampleRate ?? 0);
      for (let i = 0; i < n; i++) yield await this.out(tl, null, false);
    } finally { this.busy = false; }
  }

  private async *produce(chunks: AsyncIterable<SpeechAudio>, opts: RenderFramesOptions): AsyncGenerator<RenderedFrame> {
    const tl = new OutputTimeline(opts.audioSampleRate ?? 0);
    const it = chunks[Symbol.asyncIterator]();
    const prepare = async (c: SpeechAudio, startStep: number): Promise<Prepared> => {
      if (!(c?.pcm instanceof Float32Array) || !(c.sampleRate > 0)) throw new OptionsError('audio must be { pcm: Float32Array, sampleRate > 0 }');
      const pcm16 = c.sampleRate === MEL_SAMPLE_RATE ? c.pcm : await resampleYielding(c.pcm, c.sampleRate, MEL_SAMPLE_RATE);
      const mel = await logMelYielding(pcm16);
      const zl = await sentenceLatents(this.lips, this.avatar, { startStep, frames: mel.frames, melFrames: mel.melFrames, gather: mel.gather, mel: mel.mel }, this.mouthOpen);
      const speech = speechMask(mel.mel, mel.melFrames, mel.gather, mel.frames);
      return { audio: c.pcm, rate: c.sampleRate, frames: mel.frames, walkStart: startStep, zl, speech, strength: patchStrength(speech) };
    };
    type Next = Prepared | null;
    const fetchNext = (startStep: number): Promise<Next> => it.next().then((r) => (r.done ? null : prepare(r.value, startStep)));

    const before = framesOfMs(opts.idleBeforeMs ?? 0);
    let pending = fetchNext(this.step + before);
    pending.catch(() => undefined);
    for (let i = 0; i < before; i++) yield await this.out(tl, null, false);
    for (;;) {
      const p = await pending;
      if (!p) break;
      const n = tl.addChunk(p.audio, p.rate);
      // the next chunk is prepared while this one is drawn, planned to start right after it
      pending = fetchNext(this.step + n);
      pending.catch(() => undefined);
      // the model's change on every frame with a strength (speech and the fade after it); frames past the chunk's last
      // mel frame use its last latents (inferFrames), as on Node
      const drawn = Array.from(p.strength, (k) => k > 0);
      for (let f = 0; f < n; f += WGSL_MAX_BATCH) {
        const m = Math.min(WGSL_MAX_BATCH, n - f);
        const us = Array.from({ length: m }, (_, i) => this.avatar.walk(p.walkStart + f + i));
        const strength = Float32Array.from({ length: m }, (_, i) => p.strength[f + i] ?? 0);
        const any = strength.some((k) => k > 0);
        const pred = any ? await inferFrames(this.lips, this.avatar, us, p.zl, f, m, drawn) : null;
        if (pred) applyStrength(pred, strength, 0, m);
        for (let i = 0; i < m; i++) {
          let patch = null;
          if (pred && strength[i]! > 0) patch = this.comp.frame(await this.avatar.frame(us[i]!), pred, i);
          yield await this.out(tl, patch, true);
        }
      }
    }
    const after = tl.framesAfter(opts.idleAfterMs ?? 0);
    for (let i = 0; i < after; i++) yield await this.out(tl, null, false);
  }

  /** The next output frame: the idle video's frame at this.step with the patch drawn on it; advances the step. */
  private async out(tl: OutputTimeline, patch: { bitmap: ImageBitmap; x: number; y: number } | null, speaking: boolean): Promise<RenderedFrame> {
    this.live();
    const vf = await this.video.frame(this.step++);
    try {
      this.g.drawImage(vf, 0, 0, this.canvas.width, this.canvas.height);
    } finally { vf.close(); }
    if (patch) { this.g.drawImage(patch.bitmap, patch.x, patch.y); patch.bitmap.close(); }
    let src: OffscreenCanvas = this.canvas;
    if (this.matte) { this.matte.render(); src = this.matteCanvas!; }
    if (this.scaled) {
      const s = this.scaled;
      s.g.clearRect(0, 0, s.canvas.width, s.canvas.height);
      s.g.drawImage(src, 0, 0, src.width, this.matte ? src.height : this.avatar.meta.frame_size[1], 0, 0, s.canvas.width, s.canvas.height);
      src = s.canvas;
    }
    const t = tl.next();
    const image = new VideoFrame(src, { timestamp: Math.round(t.timeMs * 1000), duration: 1_000_000 / FPS, alpha: this.matte ? 'keep' : 'discard' });
    return { index: t.index, timeMs: t.timeMs, image, speaking, audio: t.audio, sampleRate: t.sampleRate };
  }

  /** Frees the GPU buffers, the video decoder and the matte's WebGL context. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.video.close();
    this.matte?.destroy();
    (this.lips as { destroy?: () => void }).destroy?.();
  }
}

/**
 * Loads an avatar for offline rendering (no DOM: also in a Web Worker). Rejects with WebGPUUnavailableError without
 * WebGPU, OptionsError on invalid options, AvatarPackError when the avatar's files are missing or invalid, and an Error
 * when the browser cannot decode the idle video with WebCodecs.
 */
export async function createRenderer(options: RendererOptions): Promise<LipSyncRenderer> {
  const { matte, width, ...rest } = (options ?? {}) as Partial<RendererOptions>;
  if (matte !== undefined && (typeof matte !== 'string' || !/^[\w.-]+$/.test(matte))) throw new OptionsError('createRenderer(): "matte" must be a file name in the idle loop\'s directory');
  if (width !== undefined && !(Number.isInteger(width) && width >= 16 && width <= 8192)) throw new OptionsError('createRenderer(): "width" must be a whole number of pixels (16 .. 8192)');
  const o = resolveOptions(rest);
  await requestGpuAdapter();
  let loaded = 0;
  const onBytes = (n: number): void => { loaded += n; o.onProgress({ loadedBytes: loaded, file: '' }); };
  const src = browserPackSource(onBytes);
  const listing = await listAvatar(src, o.avatar, o.idle);
  const pack = listing.packs[0]!;
  const avatar = await Avatar.fromPack(pack, onBytes);
  const file = matte ?? pack.meta.video.find((v) => /webm/i.test(v.type) || v.src.endsWith('.webm'))?.src;
  if (!file) throw new Error(`ZEN-LipSync: ${pack.dir} has no WebM idle video (the renderer decodes it with WebCodecs)`);
  const videoUrl = src.join(pack.dir, file);
  const [lips, bytes] = await Promise.all([
    loadModels(avatar, onBytes, { weights: o.weights }),
    fetch(videoUrl).then(async (r) => { if (!r.ok) throw new Error(`${videoUrl}: HTTP ${r.status}`); const b = await r.arrayBuffer(); onBytes(b.byteLength); return b; }),
  ]);
  let video: IdleFrames;
  try { video = await IdleFrames.open(bytes); } catch (e) { (lips as { destroy?: () => void }).destroy?.(); throw e; }
  let r: LipSyncRenderer;
  try { r = new LipSyncRenderer(avatar, lips, video, pack.name, matte !== undefined, width); } catch (e) { video.close(); (lips as { destroy?: () => void }).destroy?.(); throw e; }
  try { r.setMouthOpen(o.mouthOpen); } catch (e) { r.destroy(); throw e; }
  return r;
}
