// The Node avatar (CPU): speech audio in, video frames out. The idle loop's frames come from its idle video (ffmpeg),
// the mouth patch of every speech frame is drawn on them (compositor.ts), like the browser player draws its canvas
// over the <video>. Output: frames() (raw RGB), render() (a video file), stream() (a byte stream or JPEG frames).
// Speech can come as one clip or as a stream of chunks: each chunk is prepared (features, sentence model) while the
// previous one is drawn, and its frames follow the previous chunk's without a gap.
import { Readable } from 'node:stream';
import { unlink } from 'node:fs/promises';
import type { ChildProcess } from 'node:child_process';
import { chooseIdlePair, type IdleSwitchFile } from '../core/idle-switch.js';
import { inferFrames, sentenceLatents, speechMask, type LipModels } from '../core/lipsync.js';
import { resample } from '../core/resample.js';
import { OptionsError } from '../core/errors.js';
import { Emitter } from '../core/events.js';
import type { AvatarHandle, SpeechAudio } from '../core/types.js';
import type { AvatarSet } from '../core/pack.js';
import { NodeCompositor } from './compositor.js';
import type { NodePack } from './pack.js';
import { spawnEncoder } from './ffmpeg.js';
import type { FrameKernel } from './frame-model.js';
import type { SpeechPrep } from './index.js';

/** One output frame. */
export interface NodeFrame {
  /** Frame number in this output (25 fps). */
  index: number;
  timeMs: number;
  width: number;
  height: number;
  /** width x height x 3 bytes (RGB). Owned by the caller. */
  rgb: Buffer;
  /** True when a mouth patch is drawn (speech). */
  speaking: boolean;
  /** The output's audio for this frame (mono float32 at sampleRate; silence for idle frames). */
  audio: Float32Array;
  sampleRate: number;
}

export interface FramesOptions {
  /** Idle frames before the speech (ms, default 0) and after it (default 0). */
  idleBeforeMs?: number;
  idleAfterMs?: number;
  /**
   * Produce frames at real time (25 per second, by the clock) and fill with idle frames (and silence) while the next
   * speech chunk is not ready: for live delivery. Default false (as fast as the CPU allows).
   */
  realtime?: boolean;
  /**
   * realtime only: the output clock starts this much after the first frame is asked for (ms, default 600), so the
   * first chunk's preparation (about 0.3 s of CPU per second of speech on one core) does not make frames late. The
   * viewer's delay grows by the same amount.
   */
  startDelayMs?: number;
  /** Sample rate of the output audio (default: the first chunk's rate). */
  audioSampleRate?: number;
}

export type VideoContainer = 'mp4' | 'webm';
/** How far a render has got (see RenderOptions.onProgress). */
export interface RenderProgress {
  /** Frames written so far (25 fps). */
  frames: number;
  /** Their time in the video (ms). */
  timeMs: number;
  /** Frames the whole render will write, when the caller knows it (an audio file); else null. */
  totalFrames: number | null;
}
export interface RenderOptions extends Omit<FramesOptions, 'realtime'> {
  /** Output file (.mp4: H.264 + AAC; .webm: VP9 + Opus). */
  out: string;
  /** Encoder quality: lower is better (default 20 for H.264, 32 for VP9). */
  crf?: number;
  /** Called about once a second of video, and once at the end. */
  onProgress?: (p: RenderProgress) => void;
  /** Aborting stops the render and removes the partial output file. */
  signal?: AbortSignal;
  /** Frames the whole render will write: only used to fill RenderProgress.totalFrames. */
  expectedFrames?: number;
}
export interface StreamOptions extends FramesOptions {
  /**
   * 'fmp4': fragmented MP4 (H.264 + AAC) for MSE / HTTP; 'webm': WebM (VP9 + Opus) for MSE / HTTP;
   * 'jpeg': an object stream of { type: 'video', index, timeMs, jpeg } and { type: 'audio', timeMs, pcm, sampleRate }.
   */
  format: 'fmp4' | 'webm' | 'jpeg';
  /** Output width (height keeps the aspect ratio; default: the avatar's frame size). */
  width?: number;
  crf?: number;
}

export interface NodeTiming {
  frames: number;
  /** Frame model ms per frame, and all work (model + drawing) ms per frame. */
  inferMsPerFrame: number;
  totalMsPerFrame: number;
}

const FPS = 25;
const FADE_FRAMES = 8;
/** Frames of the frame model run per step (the native kernel runs one at a time; batching only groups the copies). */
const BATCH = 4;

interface Prepared {
  audio: Float32Array;
  rate: number;
  frames: number;
  walkStart: number;
  zl: Uint16Array;
  /** Per frame of this chunk: false when nothing sounds under it (its mouth patch is left off). */
  speech: boolean[];
}

export class NodeAvatar implements AvatarHandle {
  readonly backend = 'cpu' as const;
  private readonly events = new Emitter();
  private destroyed = false;
  /** Walk position of the next frame of the shown loop. */
  private step = 0;
  private shown: NodePack;
  private fade: { from: NodePack; fromStep: number; left: number } | null = null;
  private busy = false;
  private readonly comps = new Map<NodePack, NodeCompositor>();
  private readonly timing = { frames: 0, inferMs: 0, totalMs: 0 };

  constructor(private readonly packs: Map<string, NodePack>, readonly lips: LipModels, readonly kernel: FrameKernel, private readonly prep: SpeechPrep,
    private readonly pairs: IdleSwitchFile | null, readonly set: AvatarSet | null, private readonly ffmpeg: string, loadMs: number, loadedBytes: number) {
    this.shown = packs.values().next().value as NodePack;
    for (const p of packs.values()) this.comps.set(p, new NodeCompositor(p.meta, p.fromVideo, p.toVideo));
    queueMicrotask(() => this.events.emit('ready', { backend: 'cpu', loadMs, loadedBytes }));
  }

  get idle(): string { return this.shown.name; }
  get idleLoops(): readonly string[] { return [...this.packs.keys()]; }
  get speaking(): boolean { return this.busy; }
  /** Frame size of the output. */
  get width(): number { return this.shown.meta.frame_size[0]; }
  get height(): number { return this.shown.meta.frame_size[1]; }

  private live(): void { if (this.destroyed) throw new Error('ZEN-LipSync: the avatar was destroyed'); }

  /**
   * Frames of speech (one clip, or chunks as they arrive), 25 per second, each with its share of the audio.
   * One output at a time per avatar.
   */
  async *frames(input: SpeechAudio | AsyncIterable<SpeechAudio> | Iterable<SpeechAudio>, opts: FramesOptions = {}): AsyncGenerator<NodeFrame> {
    this.live();
    if (this.busy) throw new OptionsError('frames(): this avatar is already producing an output (one at a time; create another avatar for parallel outputs)');
    this.busy = true;
    try { yield* this.produce(toChunks(input), opts); } finally { this.busy = false; }
  }

  private async *produce(chunks: AsyncIterable<SpeechAudio>, opts: FramesOptions): AsyncGenerator<NodeFrame> {
    let rate = opts.audioSampleRate ?? 0;
    let index = 0;
    // the output audio: speech segments placed on the output timeline (samples at rate); idle time is silence.
    // Chunks follow each other without a gap (each starts where the previous one's audio ended).
    const segs: { start: number; data: Float32Array }[] = [];
    let outSamples = 0; // samples given to frames so far
    let audioEnd = 0; // end of the last speech segment
    const realtime = opts.realtime ?? false;
    const t0 = performance.now() + (realtime ? (opts.startDelayMs ?? 600) : 0);
    const pace = async (): Promise<void> => {
      if (!realtime) return;
      const due = t0 + (index * 1000) / FPS - performance.now();
      if (due > 1) await new Promise((r) => setTimeout(r, due));
    };
    const frameAudio = (): Float32Array => {
      const end = Math.round(((index + 1) * rate) / FPS);
      const a = new Float32Array(Math.max(0, end - outSamples));
      for (const s of segs) {
        const from = Math.max(outSamples, s.start), to = Math.min(end, s.start + s.data.length);
        for (let j = from; j < to; j++) a[j - outSamples] = s.data[j - s.start]!;
      }
      while (segs.length && segs[0]!.start + segs[0]!.data.length <= end) segs.shift();
      outSamples = end;
      return a;
    };
    const frame = (rgb: Buffer, speaking: boolean): NodeFrame => {
      const f: NodeFrame = { index, timeMs: (index * 1000) / FPS, width: this.width, height: this.height, rgb, speaking, audio: frameAudio(), sampleRate: rate };
      index++;
      return f;
    };

    // each chunk is prepared (resampled, log-mel, sentence model) while the previous one is drawn
    const it = chunks[Symbol.asyncIterator]();
    const prepare = async (c: SpeechAudio, startStep: number): Promise<Prepared> => {
      if (!(c?.pcm instanceof Float32Array) || !(c.sampleRate > 0)) throw new OptionsError('audio must be { pcm: Float32Array, sampleRate > 0 }');
      const mel = await this.prep.mel(c.pcm.slice(), c.sampleRate);
      const zl = await sentenceLatents(this.lips, this.shown, { startStep, frames: mel.frames, melFrames: mel.melFrames, gather: mel.gather, mel: mel.mel });
      const speech = speechMask(mel.mel, mel.melFrames, mel.gather, mel.frames);
      return { audio: c.pcm, rate: c.sampleRate, frames: mel.frames, walkStart: startStep, zl, speech };
    };
    type Next = { c: SpeechAudio; p: Prepared } | null;
    const fetchNext = (startStep: number): Promise<Next> => it.next().then(async (r) => (r.done ? null : { c: r.value, p: await prepare(r.value, startStep) }));

    const before = Math.round(((opts.idleBeforeMs ?? 0) * FPS) / 1000);
    let pending: Promise<Next> = fetchNext(this.step + before);
    pending.catch(() => undefined);
    for (let i = 0; i < before; i++) { if (!rate) rate = 48000; await pace(); yield frame(await this.baseImage(), false); }
    this.events.emit('speechstart', { frames: 0 });
    let interrupted = true;
    try {
      for (;;) {
        let got: Next;
        if (realtime) {
          // keep the video going with idle frames until the chunk is ready
          let ready = false;
          void pending.then(() => { ready = true; }, () => { ready = true; });
          await new Promise((r) => setImmediate(r));
          while (!ready) {
            if (!rate) rate = 48000;
            await pace();
            if (ready) break;
            yield frame(await this.baseImage(), false);
          }
          got = await pending;
          // idle frames came in between: the chunk was planned for an earlier walk position. Its sentence latents
          // depend on the walk position only through the reference latents (a small effect), so it is kept and drawn
          // from where the walk is now (re-running the sentence model here would stall the realtime output)
          if (got) got.p.walkStart = this.step;
        } else got = await pending;
        if (!got) break;
        const p = got.p;
        if (!rate) rate = p.rate;
        const audio = p.rate === rate ? p.audio : resample(p.audio, p.rate, rate);
        const startSample = Math.max(audioEnd, Math.round((index * rate) / FPS));
        segs.push({ start: startSample, data: audio });
        audioEnd = startSample + audio.length;
        // frames of this chunk: up to the end of its audio on the output timeline (no drift over many chunks)
        const n = Math.max(0, Math.round((audioEnd * FPS) / rate) - index);
        // the next chunk is prepared now, planned to start right after this one
        pending = fetchNext(this.step + n);
        pending.catch(() => undefined);
        const pack = this.shown;
        const comp = this.comps.get(pack)!;
        const batch = realtime ? 1 : BATCH; // realtime: one frame model run per frame, so frames go out evenly
        for (let f = 0; f < n; f += batch) {
          const m = Math.min(batch, n - f);
          const us = Array.from({ length: m }, (_, i) => pack.walk(p.walkStart + f + i));
          const tb = performance.now();
          // latents past the chunk's last mel frame are clamped to it (inferFrames)
          const pred = await inferFrames(this.lips, pack, us, p.zl, f, m, p.speech);
          const ti = performance.now() - tb;
          for (let i = 0; i < m; i++) {
            await pace();
            const td = performance.now();
            const rgb = await this.baseImage();
            // nothing sounds under this frame: show it just as the idle video has it (a patch would repaint the mouth)
            if (p.speech[f + i]) comp.draw(rgb, this.width, this.height, pack.source(us[i]!), pred, i);
            this.timing.totalMs += performance.now() - td + ti / m;
            yield frame(rgb, true);
          }
          this.timing.inferMs += ti;
          this.timing.frames += m;
        }
      }
      interrupted = false;
    } finally {
      this.events.emit('speechend', { interrupted });
    }
    if (!rate) rate = 48000;
    const after = Math.max(Math.round(((opts.idleAfterMs ?? 0) * FPS) / 1000), Math.ceil(((audioEnd - outSamples) * FPS) / rate));
    for (let i = 0; i < after; i++) { await pace(); yield frame(await this.baseImage(), false); }
  }

  /** The shown loop's idle frame at this.step (a copy), crossfaded while a switch runs; advances the step. */
  private async baseImage(): Promise<Buffer> {
    const k = this.step++;
    const img = Buffer.from(await this.shown.video.frame(k));
    const fd = this.fade;
    if (fd) {
      const old = await fd.from.video.frame(fd.fromStep++);
      const p = 1 - (fd.left - 1) / FADE_FRAMES; // new loop's weight: 1/8 .. 1
      for (let i = 0; i < img.length; i++) img[i] = Math.round(img[i]! * p + old[i]! * (1 - p));
      if (--fd.left <= 0) this.fade = null;
    }
    return img;
  }

  /** A video file of the speech: H.264 + AAC (.mp4) or VP9 + Opus (.webm), with ffmpeg. Resolves when written. */
  async render(input: SpeechAudio | AsyncIterable<SpeechAudio> | Iterable<SpeechAudio>, opts: RenderOptions): Promise<{ file: string; frames: number; durationMs: number }> {
    this.live();
    if (!opts || typeof opts.out !== 'string') throw new OptionsError('render(): { out: "file.mp4" | "file.webm" } is required');
    const container: VideoContainer = opts.out.toLowerCase().endsWith('.webm') ? 'webm' : 'mp4';
    const { out, crf, onProgress, signal, expectedFrames, ...fo } = opts;
    const enc = new Encoder(this.ffmpeg, container, this.width, this.height, { file: out, crf });
    let frames = 0;
    const report = (timeMs: number): void => { onProgress?.({ frames, timeMs, totalFrames: expectedFrames ?? null }); };
    try {
      for await (const f of this.frames(input, fo)) {
        signal?.throwIfAborted();
        await enc.write(f);
        frames++;
        if (onProgress && frames % FPS === 0) report(f.timeMs + 1000 / FPS);
      }
      await enc.end();
      report((frames * 1000) / FPS);
    } catch (e) {
      enc.kill();
      // an aborted render leaves no half-written file behind (a failure keeps it: ffmpeg's message says why)
      if (signal?.aborted) await unlink(out).catch(() => undefined);
      throw e;
    }
    return { file: out, frames, durationMs: (frames * 1000) / FPS };
  }

  /** The speech as a stream for delivery (WebSocket / HTTP). See StreamOptions.format. */
  stream(input: SpeechAudio | AsyncIterable<SpeechAudio> | Iterable<SpeechAudio>, opts: StreamOptions): Readable {
    this.live();
    const fmt = opts?.format;
    if (fmt !== 'fmp4' && fmt !== 'webm' && fmt !== 'jpeg') throw new OptionsError('stream(): format must be fmp4, webm or jpeg');
    const { format, width, crf, ...fo } = opts;
    if (format === 'jpeg') {
      // ffmpeg's JPEG encoder holds about one frame back, so frames are written as they are made and each JPEG is
      // given out (with its frame's audio) when it comes back; the audio goes first, the JPEG follows
      const out = new Readable({ objectMode: true, read: () => undefined });
      const enc = new JpegEncoder(this.ffmpeg, this.width, this.height, width);
      const meta: { index: number; timeMs: number }[] = [];
      enc.onJpeg = (jpeg) => { const m = meta.shift(); if (m) out.push({ type: 'video', index: m.index, timeMs: m.timeMs, jpeg }); };
      void (async () => {
        try {
          for await (const f of this.frames(input, fo)) {
            if (out.destroyed) break;
            out.push({ type: 'audio', timeMs: f.timeMs, pcm: f.audio, sampleRate: f.sampleRate });
            meta.push({ index: f.index, timeMs: f.timeMs });
            await enc.write(f.rgb);
          }
          await enc.end();
          out.push(null);
        } catch (e) { enc.kill(); out.destroy(e as Error); }
      })();
      out.on('close', () => enc.kill());
      return out;
    }
    const enc = new Encoder(this.ffmpeg, format === 'fmp4' ? 'fmp4' : 'webm-live', this.width, this.height, { crf, width });
    void (async () => {
      try { for await (const f of this.frames(input, fo)) await enc.write(f); await enc.end(); } catch (e) { enc.output.destroy(e as Error); enc.kill(); }
    })();
    return enc.output;
  }

  /**
   * Switches the idle loop (between outputs, or between idle frames of a realtime output): the loop goes to the other
   * loop's frame most like a frame the shown loop reaches soon, with a 0.3 s crossfade on the next frames.
   */
  async setIdle(name: string): Promise<boolean> {
    this.live();
    const to = this.packs.get(name);
    const from = this.shown;
    const table = this.pairs?.pairs[`${from.name}->${name}`];
    if (!to || to === from || !table) return false;
    const L = from.meta.walk.length;
    const c = chooseIdlePair(this.step, from.meta.walk, table, 0, Math.min(L - 1, 75));
    const rising = c.fromWalk < L / 2;
    const n = to.meta.walk.length;
    const tw = rising || c.toSource === 0 ? c.toSource : n - c.toSource;
    // the switch happens now (no waiting): the old loop continues from its pair frame under the fade
    this.fade = { from, fromStep: c.fromWalk, left: FADE_FRAMES };
    this.shown = to;
    this.step = tw;
    this.events.emit('idlechange', { from: from.name, to: name });
    return true;
  }

  /** On Node, speech is turned into frames with frames() / render() / stream(); speak() is not used. */
  speak(_audio: SpeechAudio): Promise<void> { return Promise.reject(new OptionsError('zen-lipsync/node: use frames(), render() or stream() (speak() is for the browser player)')); }
  speakStream(_chunks: AsyncIterable<SpeechAudio>): Promise<void> { return this.speak({ pcm: new Float32Array(0), sampleRate: 1 }); }

  on: AvatarHandle['on'] = (event, listener) => this.events.on(event, listener);
  stop(): void { /* outputs end when their consumer stops reading */ }
  pause(): void { /* nothing plays on Node */ }
  resume(): void { /* nothing plays on Node */ }

  frameTiming(): { frames: number; inferMsPerFrame: number; totalMsPerFrame: number } {
    const t = this.timing;
    return { frames: t.frames, inferMsPerFrame: t.frames ? t.inferMs / t.frames : 0, totalMsPerFrame: t.frames ? t.totalMs / t.frames : 0 };
  }

  /** Bytes the avatar holds (packs, decoded images, idle video buffers), besides the model weights. */
  memoryBytes(): number { let b = 0; for (const p of this.packs.values()) b += p.bytes; return b; }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const p of this.packs.values()) p.close();
    this.events.clear();
  }
}

async function* toChunks(input: SpeechAudio | AsyncIterable<SpeechAudio> | Iterable<SpeechAudio>): AsyncGenerator<SpeechAudio> {
  if (input && (input as SpeechAudio).pcm instanceof Float32Array) { yield input as SpeechAudio; return; }
  if (input && typeof (input as AsyncIterable<SpeechAudio>)[Symbol.asyncIterator] === 'function') { yield* input as AsyncIterable<SpeechAudio>; return; }
  if (input && typeof (input as Iterable<SpeechAudio>)[Symbol.iterator] === 'function') { yield* input as Iterable<SpeechAudio>; return; }
  throw new OptionsError('audio must be { pcm: Float32Array, sampleRate } or an (async) iterable of them');
}

/** ffmpeg with video (rgb24, stdin) and audio (f32le mono, fd 3). */
class Encoder {
  private p: ChildProcess | null = null;
  private rate = 0;
  readonly output: Readable;
  private done: Promise<void>;
  private err = '';

  constructor(private readonly ffmpeg: string, private readonly kind: 'mp4' | 'webm' | 'fmp4' | 'webm-live', private readonly W: number, private readonly H: number,
    private readonly o: { file?: string; crf?: number | undefined; width?: number | undefined }) {
    const out = new Readable({ read: () => { this.p?.stdout?.resume(); } });
    this.output = out;
    this.done = Promise.resolve();
  }

  private start(rate: number): void {
    this.rate = rate;
    const k = this.kind;
    const vf = [`scale=${this.o.width ? `${this.o.width}:-2` : 'iw:ih'}:out_color_matrix=bt709:out_range=tv`];
    const h264 = ['-c:v', 'libx264', '-preset', k === 'fmp4' ? 'veryfast' : 'medium', '-crf', String(this.o.crf ?? 20), '-pix_fmt', 'yuv420p', '-g', '50',
      '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv', '-c:a', 'aac', '-b:a', '128k'];
    const vp9 = ['-c:v', 'libvpx-vp9', '-crf', String(this.o.crf ?? 32), '-b:v', '0', '-deadline', 'realtime', '-cpu-used', '8', '-row-mt', '1', '-g', '50', '-pix_fmt', 'yuv420p',
      '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv', '-c:a', 'libopus', '-b:a', '96k'];
    // no probing of the raw inputs: ffmpeg probes its inputs one after the other, and a probe reading a lot of the
    // video would wait for audio that is only written along with the next frames (a deadlock)
    const raw = ['-probesize', '32', '-analyzeduration', '0'];
    const args = [...raw, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${this.W}x${this.H}`, '-r', String(FPS), '-thread_queue_size', '64', '-i', 'pipe:0',
      ...raw, '-f', 'f32le', '-ar', String(rate), '-ac', '1', '-thread_queue_size', '1024', '-i', 'pipe:3', '-vf', vf.join(','), '-shortest'];
    if (k === 'mp4') args.push(...h264, '-movflags', '+faststart', '-y', this.o.file!);
    else if (k === 'webm') args.push(...vp9, '-y', this.o.file!);
    else if (k === 'fmp4') args.push(...h264, '-tune', 'zerolatency', '-movflags', 'frag_keyframe+empty_moov+default_base_moof', '-f', 'mp4', 'pipe:1');
    else args.push(...vp9, '-f', 'webm', '-cluster_time_limit', '500', 'pipe:1');
    const p = spawnEncoder(this.ffmpeg, args);
    this.p = p;
    p.stderr!.on('data', (d: Buffer) => { this.err = (this.err + d.toString()).slice(-2000); });
    p.stdout!.on('data', (d: Buffer) => { if (!this.output.push(d)) p.stdout!.pause(); });
    this.done = new Promise((resolve, reject) => {
      p.on('error', reject);
      p.on('close', (code) => { this.output.push(null); if (code === 0) resolve(); else reject(new Error(`ffmpeg exited with ${code}: ${this.err}`)); });
    });
    this.done.catch(() => undefined);
  }

  async write(f: NodeFrame): Promise<void> {
    if (!this.p) this.start(f.sampleRate);
    const p = this.p!;
    const a = p.stdio[3] as NodeJS.WritableStream;
    a.write(Buffer.from(f.audio.buffer, f.audio.byteOffset, f.audio.byteLength));
    if (!p.stdin!.write(f.rgb)) {
      await new Promise<void>((resolve, reject) => {
        const onDrain = (): void => { p.off('close', onClose); resolve(); };
        const onClose = (): void => { p.stdin!.off('drain', onDrain); reject(new Error(`ffmpeg stopped: ${this.err}`)); };
        p.stdin!.once('drain', onDrain);
        p.once('close', onClose);
      });
    }
  }

  async end(): Promise<void> {
    if (!this.p) return;
    (this.p.stdio[3] as NodeJS.WritableStream).end();
    this.p.stdin!.end();
    await this.done;
  }

  kill(): void { try { this.p?.kill('SIGKILL'); } catch { /* gone */ } }
}

/** rgb24 frames -> JPEG with one ffmpeg (image2pipe); a JPEG ends at FF D9 (the encoder writes no other FF D9). */
class JpegEncoder {
  private readonly p: ChildProcess;
  private buf: Buffer = Buffer.alloc(0);
  onJpeg: (jpeg: Buffer) => void = () => undefined;
  private readonly closed: Promise<void>;
  constructor(ffmpeg: string, W: number, H: number, width?: number) {
    this.p = spawnEncoder(ffmpeg, ['-probesize', '32', '-analyzeduration', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${W}x${H}`, '-r', String(FPS), '-i', 'pipe:0',
      ...(width ? ['-vf', `scale=${width}:-2`] : []), '-c:v', 'mjpeg', '-q:v', '4', '-pix_fmt', 'yuvj420p', '-f', 'image2pipe', 'pipe:1']);
    this.p.stdout!.on('data', (d: Buffer) => {
      this.buf = this.buf.length ? Buffer.concat([this.buf, d]) : d;
      for (;;) {
        const e = this.buf.indexOf(EOI);
        if (e < 0) break;
        this.onJpeg(Buffer.from(this.buf.subarray(0, e + 2)));
        this.buf = this.buf.subarray(e + 2);
      }
    });
    this.closed = new Promise((r) => this.p.on('close', () => r()));
  }
  async write(rgb: Buffer): Promise<void> {
    if (!this.p.stdin!.write(rgb)) await new Promise<void>((r) => this.p.stdin!.once('drain', r));
  }
  async end(): Promise<void> { this.p.stdin!.end(); await this.closed; }
  kill(): void { try { this.p.kill('SIGKILL'); } catch { /* gone */ } }
}

const EOI = Buffer.from([0xff, 0xd9]);
