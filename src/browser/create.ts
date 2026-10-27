// create() for the browser: an AvatarHandle on the WebGPU player (no CPU fallback in the browser).
import { OptionsError } from '../core/errors.js';
import { Emitter } from '../core/events.js';
import { resolveOptions } from '../core/options.js';
import type { AvatarHandle, CreateOptions, SpeechAudio } from '../core/types.js';
import { requestGpuAdapter } from '../runtime/avatar.js';
import { LipSyncPlayer } from '../runtime/player.js';

/** How the picture fills its box: 'contain' shows the whole frame (letterboxed when the box has another shape), 'cover' fills the box and crops. */
export type AvatarFit = 'contain' | 'cover';

export interface BrowserCreateOptions extends CreateOptions {
  /**
   * Element the avatar is put in. The SDK adds and styles its own elements inside (no CSS needed from the page): a box as
   * wide as the container with the video's aspect ratio, the idle <video> and the lip-sync <canvas> stacked in it. If the
   * container has a fixed height, the box keeps within it and the picture is fitted inside (see `fit`); the lip-sync
   * stays on the face for any size, resize, rotation and device pixel ratio.
   */
  container: HTMLElement;
  /** Default 'contain'. */
  fit?: AvatarFit;
}

/**
 * The avatar's own elements: the box, and the style every layer (each idle <video> and the <canvas>) shares. Every layer
 * fills the box with the same object-fit / object-position over its own intrinsic size, which is the video frame size for
 * all of them (the canvas is sized to the frame), so a patch drawn at frame pixel (x, y) lands on video pixel (x, y).
 */
export function avatarLayout(fit: AvatarFit, frameW: number, frameH: number): { box: string; layer: string } {
  return {
    box: `position:relative;display:block;width:100%;max-width:100%;height:auto;max-height:100%;aspect-ratio:${frameW} / ${frameH};overflow:hidden;margin:0;padding:0;border:0;box-sizing:border-box`,
    layer: `position:absolute;left:0;top:0;width:100%;height:100%;margin:0;padding:0;border:0;display:block;object-fit:${fit};object-position:50% 50%;background:transparent`,
  };
}

/**
 * Loads an avatar and starts its idle loop in container. Rejects with WebGPUUnavailableError when the browser has no
 * WebGPU, OptionsError on invalid options, AvatarPackError when the avatar's files are missing or invalid.
 */
export async function create(options: BrowserCreateOptions): Promise<AvatarHandle> {
  const { container, fit = 'contain', ...rest } = (options ?? {}) as Partial<BrowserCreateOptions>;
  const o = resolveOptions(rest);
  if (!container || typeof (container as HTMLElement).appendChild !== 'function') throw new OptionsError('create(): "container" must be an HTML element');
  if (fit !== 'contain' && fit !== 'cover') throw new OptionsError('create(): "fit" must be \'contain\' or \'cover\'');
  await requestGpuAdapter();
  const t0 = performance.now();
  // the frame size is not known before the pack loads: a 16:9 box until then, set exactly once the player has it
  let layout = avatarLayout(fit, 16, 9);
  const box = document.createElement('div');
  box.setAttribute('data-zen-lipsync', '');
  box.style.cssText = layout.box;
  const video = document.createElement('video');
  video.style.cssText = layout.layer;
  // extra idle loops are clones of this video (same style), inserted before the canvas; the canvas stays on top
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `${layout.layer};pointer-events:none`;
  box.append(video, canvas);
  container.append(box);
  let loaded = 0;
  let player: LipSyncPlayer;
  try {
    player = await LipSyncPlayer.create(o.avatar, video, canvas, {
      weights: o.weights, idle: o.idle,
      onBytes: (n) => { loaded += n; o.onProgress({ loadedBytes: loaded, file: '' }); },
    });
  } catch (e) {
    box.remove();
    throw e;
  }
  const [W, H] = player.avatar.meta.frame_size;
  layout = avatarLayout(fit, W, H);
  box.style.cssText = layout.box;
  return new BrowserAvatar(player, box, performance.now() - t0, loaded);
}

class BrowserAvatar implements AvatarHandle {
  readonly backend = 'webgpu' as const;
  private readonly events = new Emitter();
  private destroyed = false;
  private queue: Promise<void> = Promise.resolve();
  private streamGen = 0;

  constructor(private readonly player: LipSyncPlayer, private readonly box: HTMLElement, loadMs: number, loadedBytes: number) {
    queueMicrotask(() => this.events.emit('ready', { backend: 'webgpu', loadMs, loadedBytes }));
  }

  get idle(): string { return this.player.idle; }
  get idleLoops(): readonly string[] { return this.player.idleLoops; }
  get speaking(): boolean { return this.player.speaking; }

  private live(): void { if (this.destroyed) throw new Error('ZEN-LipSync: the avatar was destroyed'); }

  async speak(audio: SpeechAudio): Promise<void> {
    this.live();
    if (!(audio?.pcm instanceof Float32Array) || !(audio.sampleRate > 0)) throw new OptionsError('speak(): audio must be { pcm: Float32Array, sampleRate > 0 }');
    const frames = Math.floor((audio.pcm.length / audio.sampleRate) * 25);
    this.events.emit('speechstart', { frames });
    let interrupted = false;
    const stopped = this.events.once('_stop').then(() => { interrupted = true; });
    try { await this.player.speak(audio.pcm, audio.sampleRate); } catch (e) {
      this.events.emit('error', { error: e instanceof Error ? e : new Error(String(e)) });
      throw e;
    } finally {
      void stopped;
      this.events.emit('speechend', { interrupted });
    }
  }

  async speakStream(chunks: AsyncIterable<SpeechAudio>): Promise<void> {
    this.live();
    // the chunks play back to back: each is prepared while the previous one plays (player.speakChunks).
    // stop() and destroy() end the stream: chunks that arrive after it are not spoken.
    const gen = ++this.streamGen;
    const self = this;
    const it = chunks[Symbol.asyncIterator]();
    const checked: AsyncIterator<{ pcm: Float32Array; sampleRate: number }> = {
      async next() {
        if (self.destroyed || gen !== self.streamGen) return { done: true, value: undefined };
        const r = await it.next();
        if (r.done || self.destroyed || gen !== self.streamGen) return { done: true, value: undefined };
        const a = r.value;
        if (!(a?.pcm instanceof Float32Array) || !(a.sampleRate > 0)) throw new OptionsError('speakStream(): every chunk must be { pcm: Float32Array, sampleRate > 0 }');
        return { done: false, value: a };
      },
    };
    this.events.emit('speechstart', { frames: 0 });
    let interrupted = false;
    const stopped = this.events.once('_stop').then(() => { interrupted = true; });
    const run = this.queue.then(() => this.player.speakChunks(checked));
    this.queue = run.catch(() => undefined);
    try { await run; } catch (e) {
      this.events.emit('error', { error: e instanceof Error ? e : new Error(String(e)) });
      throw e;
    } finally {
      void stopped;
      this.events.emit('speechend', { interrupted });
    }
  }

  async setIdle(name: string): Promise<boolean> {
    this.live();
    const from = this.player.idle;
    const done = await this.player.switchIdle(name);
    if (done) this.events.emit('idlechange', { from, to: name });
    return done;
  }

  on: AvatarHandle['on'] = (event, listener) => this.events.on(event, listener);

  stop(): void { this.streamGen++; this.events.emit('_stop', {}); this.player.stop(); }
  pause(): void { this.live(); this.player.pause(); }
  resume(): void { this.live(); this.player.resume(); }
  frameTiming(): ReturnType<AvatarHandle['frameTiming']> { return this.player.frameTiming(); }

  destroy(): void {
    if (this.destroyed) return;
    this.stop();
    this.destroyed = true;
    this.player.destroy();
    this.box.remove();
    this.events.clear();
  }
}
