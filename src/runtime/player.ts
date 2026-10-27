// A small player: the idle video loops in a <video> (25 fps; video frame k shows source frame walk[k]); a canvas on
// top draws, for every presented video frame, the mouth patch made for that frame while speech plays.
// speak(pcm) renders the clip's first patches, starts the sound when the video reaches the clip's first frame, renders
// the rest ahead of the sound, and draws each patch on the video frame it was made for (requestVideoFrameCallback,
// else animation frames). An avatar with several idle loops (avatar.json) can switch loops while idle (switchIdle).
import { Avatar, Compositor, inferFrames, loadModels, sentenceLatents, speechMask, type LipModels, type LipPatch, type ModelOptions } from './avatar.js';
import { chooseIdlePair, type IdleSwitchFile } from '../core/idle-switch.js';
import { logMel, MEL_SAMPLE_RATE } from '../core/mel.js';
import { listAvatar, type AvatarSet } from '../core/pack.js';
import { resample } from '../core/resample.js';
import { browserPackSource } from './pack-source.js';
import { WGSL_MAX_BATCH } from './wgsl.js';

export interface FrameTiming {
  /** Frames drawn so far and the inference time per frame (ms, batch time / frames). */
  frames: number;
  inferMsPerFrame: number;
}

export type { AvatarSet } from '../core/pack.js';

interface Prepared {
  pcm: Float32Array;
  sampleRate: number;
  /** Walk position of frame 0, frames, sentence latents, patches made so far (f of them), video frames to frame 0. */
  walkStart: number;
  frames: number;
  zl: Uint16Array;
  /** Per frame of this clip: false when nothing sounds under it (its mouth patch is left off). */
  speech: boolean[];
  patches: Map<number, LipPatch>;
  f: number;
  ahead: number;
}

interface Clip {
  /** AudioContext time at which the sound starts (frame 0 of the clip), and its length (s). */
  soundAt: number;
  duration: number;
  frames: number;
  walkStart: number;
  patches: Map<number, LipPatch>;
  source: AudioBufferSourceNode;
  prep: Prepared;
  ended: Promise<void>;
}

interface Loop { name: string; avatar: Avatar; comp: Compositor; video: HTMLVideoElement }

type VideoWithCallback = HTMLVideoElement & { requestVideoFrameCallback?: (cb: (now: number, md: { mediaTime: number }) => void) => number };

/** Frames rendered before the sound starts (the rest render while it plays, kept ahead of the sound). */
const FIRST_FRAMES = 8;
/** Least lead (video frames) between now and the clip's first frame. */
const MIN_LEAD = 8;
/** Idle switch: the pair is looked for 3 to 75 frames ahead (0.1 to 3 s), then a crossfade of this length. */
const SWITCH_MIN_WAIT = 3;
const SWITCH_MAX_WAIT = 75;
const FADE_MS = 300;

export class LipSyncPlayer {
  readonly lips: LipModels;
  private readonly loops = new Map<string, Loop>();
  private shownLoop: Loop;
  private readonly ctx: AudioContext;
  private clip: Clip | null = null;
  private shown: { x: number; y: number; w: number; h: number } | null = null;
  private lastStep = -1;
  /** Lead used for the next clip: the last measured time to the first frames + a margin (video frames). */
  private lead = MIN_LEAD;
  private readonly timing = { frames: 0, inferMs: 0 };
  private switching = false;
  /** Clips of the current utterance whose sound has been replaced by the next chunk's but may still be playing. */
  private prevClips: Clip[] = [];
  private speakGen = 0;
  private watchGen = 0;

  private constructor(loops: Loop[], lips: LipModels, readonly canvas: HTMLCanvasElement, readonly set: AvatarSet | null, private readonly pairs: IdleSwitchFile | null) {
    this.lips = lips;
    for (const l of loops) this.loops.set(l.name, l);
    this.shownLoop = loops[0]!;
    const [W, H] = this.shownLoop.avatar.meta.frame_size;
    canvas.width = W;
    canvas.height = H;
    this.ctx = new AudioContext();
  }

  /** The avatar pack of the loop shown now. */
  get avatar(): Avatar { return this.shownLoop.avatar; }
  /** The idle loop shown now, and the loops there are. */
  get idle(): string { return this.shownLoop.name; }
  get idleLoops(): string[] { return [...this.loops.keys()]; }

  /**
   * url: an avatar (avatar.json, one or more idle loops) or an avatar pack directory (one loop). video: the <video>
   * element for the first loop; the other loops get their own <video> elements, inserted after it.
   */
  static async create(url: string, video: HTMLVideoElement, canvas: HTMLCanvasElement, opts: ModelOptions & { onBytes?: (n: number) => void; idle?: string | null } = {}): Promise<LipSyncPlayer> {
    const src = browserPackSource(opts.onBytes);
    const listing = await listAvatar(src, url.replace(/\/+$/, ''), opts.idle ?? null);
    const set: AvatarSet | null = listing.set;
    const packs: [string, string][] = listing.packs.map((p) => [p.name, p.dir]);
    const avatars = await Promise.all(listing.packs.map((p) => Avatar.fromPack(p, opts.onBytes)));
    const [lips] = await Promise.all([loadModels(avatars[0]!, opts.onBytes, opts), ...avatars.map((a) => a.fetchAllFrames())]);
    let pairs: IdleSwitchFile | null = null;
    if (set?.idleSwitch && avatars.length > 1) pairs = await src.json<IdleSwitchFile>(src.join(listing.dir, set.idleSwitch)).catch(() => null);
    // every layer (idle videos, canvas) shares one box and fit, so all idle loops must have the same frame size
    const [W0, H0] = avatars[0]!.meta.frame_size;
    for (const a of avatars) if (a.meta.frame_size[0] !== W0 || a.meta.frame_size[1] !== H0) throw new Error(`ZEN-LipSync: idle loops have different frame sizes (${a.meta.frame_size.join('x')} vs ${W0}x${H0})`);
    const loops: Loop[] = avatars.map((avatar, i) => {
      // the clones keep the first video's style (the SDK's layer: absolute, same fit), so they stack exactly on it
      const v = i === 0 ? video : (video.cloneNode(false) as HTMLVideoElement);
      if (i > 0) { v.removeAttribute('id'); v.style.opacity = '0'; video.after(v); }
      const [W, H] = avatar.meta.frame_size;
      return { name: packs[i]![0], avatar, comp: new Compositor({ x: 0, width: W, height: H }, avatar), video: v };
    });
    const p = new LipSyncPlayer(loops, lips, canvas, set, pairs);
    for (const l of loops) await p.prepareVideo(l, l === loops[0]);
    p.watchVideo();
    return p;
  }

  private async prepareVideo(l: Loop, play: boolean): Promise<void> {
    const v = l.video;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.preload = 'auto';
    v.replaceChildren(...l.avatar.meta.video.map((s) => Object.assign(document.createElement('source'), { src: `${l.avatar.base}/${s.src}`, type: s.type })));
    v.load();
    if (play) await v.play().catch(() => undefined);
  }

  /** Plays the shown idle video and any sound (after a user gesture, for browsers that hold autoplay). */
  resume(): void {
    if (this.shownLoop.video.paused) void this.shownLoop.video.play().catch(() => undefined);
    if (this.ctx.state === 'suspended') void this.ctx.resume().catch(() => undefined);
  }

  /** Pauses the shown idle video and the sound. */
  pause(): void {
    this.shownLoop.video.pause();
    if (this.ctx.state === 'running') void this.ctx.suspend().catch(() => undefined);
  }

  /** Stops everything, removes the extra idle <video> elements and closes the audio. */
  destroy(): void {
    this.stop();
    this.watchGen++;
    let first = true;
    for (const l of this.loops.values()) { l.video.pause(); l.video.removeAttribute('src'); l.video.replaceChildren(); if (!first) l.video.remove(); first = false; }
    void this.ctx.close().catch(() => undefined);
    (this.lips as { destroy?: () => void }).destroy?.();
  }

  private watchVideo(): void {
    const gen = ++this.watchGen;
    const loop = this.shownLoop;
    const vc = loop.video as VideoWithCallback;
    const fps = loop.avatar.meta.fps;
    if (typeof vc.requestVideoFrameCallback === 'function') {
      const onFrame = (_now: number, md: { mediaTime: number }): void => {
        if (gen !== this.watchGen) return;
        this.onVideoFrame(Math.round(md.mediaTime * fps));
        vc.requestVideoFrameCallback!(onFrame);
      };
      vc.requestVideoFrameCallback(onFrame);
    } else {
      const tick = (): void => { if (gen !== this.watchGen) return; this.onVideoFrame(Math.floor(vc.currentTime * fps + 1e-3)); requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
    }
  }

  /** Walk position the shown video shows now (the idle video loops the walk). */
  private videoStep(): number {
    const a = this.avatar;
    return Math.floor(this.shownLoop.video.currentTime * a.meta.fps + 1e-3) % a.meta.walk.length;
  }

  /**
   * Resumes the AudioContext and waits until its clock moves: some browsers start the audio device a moment (up to
   * about 2 s) after resume(), and the sound's start time is planned on that clock against the video's.
   */
  private async audioRunning(): Promise<void> {
    if (this.ctx.state !== 'running') await this.ctx.resume();
    const t = this.ctx.currentTime;
    for (let i = 0; i < 100 && this.ctx.currentTime === t; i++) await new Promise((r) => setTimeout(r, 20));
  }

  /** Speak a clip: pcm at sampleRate (any rate; resampled to 16 kHz for the features). Resolves when its sound ends. */
  async speak(pcm: Float32Array, sampleRate: number): Promise<void> {
    await this.speakChunks([{ pcm, sampleRate }][Symbol.iterator]());
  }

  /**
   * Speaks chunks one after the other without a gap: while a chunk plays, the next one is prepared (features, sentence
   * model and first frames, planned for the video frame where the playing chunk ends) and its sound is scheduled to
   * start exactly when the previous one ends on the audio clock. Resolves when the last sound ends (or on stop()).
   */
  async speakChunks(chunks: Iterator<{ pcm: Float32Array; sampleRate: number }> | AsyncIterator<{ pcm: Float32Array; sampleRate: number }>): Promise<void> {
    await this.audioRunning();
    while (this.switching) await new Promise((r) => setTimeout(r, 20));
    this.stop();
    const gen = ++this.speakGen;
    const loop = this.shownLoop;
    const avatar = loop.avatar;
    const fps = avatar.meta.fps;
    const walkLen = avatar.meta.walk.length;
    const nextChunk = async (): Promise<{ pcm: Float32Array; sampleRate: number } | null> => {
      const r = await chunks.next();
      return r.done ? null : r.value;
    };
    // the first chunk: planned `lead` frames ahead of the video
    let c = await nextChunk();
    if (!c || gen !== this.speakGen) return;
    let prepared = await this.prepareFirst(c.pcm, c.sampleRate, loop);
    if (!prepared || gen !== this.speakGen) return;
    let clip = this.startClip(prepared, loop, null);
    let lastEnded: Promise<void> = clip.ended;
    for (;;) {
      // prepare the next chunk while this one plays: it starts where this one's sound ends
      const nextP = nextChunk().then(async (n) => {
        if (!n || gen !== this.speakGen) return null;
        const startStep = (clip.walkStart + clip.frames) % walkLen;
        return this.prepareAt(n.pcm, n.sampleRate, loop, startStep);
      });
      nextP.catch(() => undefined);
      const [, nx] = await Promise.all([this.renderRest(clip, loop), nextP]);
      if (!nx || gen !== this.speakGen) break;
      clip = this.startClip(nx, loop, clip);
      lastEnded = clip.ended;
    }
    await lastEnded;
    if (gen === this.speakGen && this.clip === clip) this.stop();
    void fps;
  }

  /** The first chunk: features and sentence latents planned ahead of the video, its first frames rendered. */
  private async prepareFirst(pcm: Float32Array, sampleRate: number, loop: Loop): Promise<Prepared | null> {
    const avatar = loop.avatar;
    const fps = avatar.meta.fps;
    const walkLen = avatar.meta.walk.length;
    // the clip's frame 0 goes on a video frame `lead` frames ahead of the one shown now; positions are taken on the
    // walk (the idle video loops it: video time t shows walk position floor(t * fps) mod walkLen). The sentence model's
    // input depends on that position, so when the first frames are not ready before the video gets there, plan again
    // further ahead (lead = the measured time to the first frames + a margin).
    let lead = this.lead;
    let p: Prepared | null = null;
    for (let attempt = 0; ; attempt++) {
      const t0 = performance.now();
      const walkStart = (this.videoStep() + lead) % walkLen;
      if (p) for (const q of p.patches.values()) q.bitmap.close();
      p = await this.prepareAt(pcm, sampleRate, loop, walkStart);
      const took = Math.ceil(((performance.now() - t0) / 1000) * fps);
      this.lead = Math.max(MIN_LEAD, took + 3);
      // frames the video still has to play until it shows walkStart (about a lap when it has just passed it)
      let ahead = (((walkStart - this.videoStep()) % walkLen) + walkLen) % walkLen;
      if (ahead >= 1 && ahead <= lead + 2) { p.ahead = ahead; break; }
      if (attempt >= 2) { ahead = Math.max(1, Math.min(ahead, lead)); p.ahead = ahead; break; }
      lead = this.lead;
    }
    return p;
  }

  /** Features, sentence latents and the first frames of a chunk whose frame 0 is shown at walk position walkStart. */
  private async prepareAt(pcm: Float32Array, sampleRate: number, loop: Loop, walkStart: number): Promise<Prepared> {
    const avatar = loop.avatar;
    const pcm16 = sampleRate === MEL_SAMPLE_RATE ? pcm : resample(pcm, sampleRate, MEL_SAMPLE_RATE);
    const mel = logMel(pcm16);
    const zl = await sentenceLatents(this.lips, avatar, { startStep: walkStart, frames: mel.frames, melFrames: mel.melFrames, gather: mel.gather, mel: mel.mel });
    const speech = speechMask(mel.mel, mel.melFrames, mel.gather, mel.frames);
    const p: Prepared = { pcm, sampleRate, walkStart, frames: mel.frames, zl, speech, patches: new Map(), f: 0, ahead: 0 };
    await this.batch(p, loop, Math.min(FIRST_FRAMES, mel.frames));
    return p;
  }

  private async batch(p: Prepared, loop: Loop, n: number): Promise<void> {
    const avatar = loop.avatar;
    const us = Array.from({ length: n }, (_, i) => avatar.walk(p.walkStart + p.f + i));
    const tb = performance.now();
    const [pred, srcs] = await Promise.all([inferFrames(this.lips, avatar, us, p.zl, p.f, n, p.speech), Promise.all(us.map((u) => avatar.frame(u)))]);
    this.timing.inferMs += performance.now() - tb;
    this.timing.frames += n;
    const silent = await Promise.all(us.map((_, i) => (p.speech[p.f + i] ? null : loop.comp.emptyPatch(0, 0))));
    srcs.forEach((src, i) => p.patches.set(p.f + i, silent[i] ?? loop.comp.frame(src, pred, i)));
    p.f += n;
  }

  /**
   * Starts a prepared chunk's sound: the first chunk when the video reaches its first frame, a following chunk right
   * when the previous one's sound ends (prev). Its patches are drawn from then on.
   */
  private startClip(p: Prepared, loop: Loop, prev: Clip | null): Clip {
    const fps = loop.avatar.meta.fps;
    const buf = this.ctx.createBuffer(1, p.pcm.length, p.sampleRate);
    buf.copyToChannel(p.pcm as Float32Array<ArrayBuffer>, 0);
    const source = this.ctx.createBufferSource();
    source.buffer = buf;
    source.connect(this.ctx.destination);
    let soundAt: number;
    if (prev) soundAt = Math.max(this.ctx.currentTime, prev.soundAt + prev.duration);
    else {
      // sound when the video reaches that frame (the video and the audio clock both run at real time)
      const vt = loop.video.currentTime * fps;
      const frac = vt - Math.floor(vt + 1e-3);
      soundAt = this.ctx.currentTime + Math.max(0, p.ahead - frac) / fps;
    }
    const clip: Clip = { soundAt, duration: p.pcm.length / p.sampleRate, frames: p.frames, walkStart: p.walkStart, patches: p.patches, source, prep: p,
      ended: new Promise<void>((r) => { source.onended = () => r(); }) };
    if (prev) this.prevClips.push(prev);
    this.clip = clip;
    source.start(soundAt);
    return clip;
  }

  /** The rest of a clip's frames, staying at most 2 batches ahead of its sound. */
  private async renderRest(clip: Clip, loop: Loop): Promise<void> {
    const p = clip.prep;
    while (p.f < p.frames && this.alive(clip)) {
      while (this.alive(clip) && p.f - this.clipFrame(clip) > WGSL_MAX_BATCH * 2) await new Promise((r) => setTimeout(r, 15));
      if (!this.alive(clip)) break;
      await this.batch(p, loop, Math.min(WGSL_MAX_BATCH, p.frames - p.f));
    }
  }

  /** The clip is playing or still to play (not stopped). */
  private alive(c: Clip): boolean { return this.clip === c || this.prevClips.includes(c); }

  /** Stops the sound and clears the mouth patches. */
  stop(): void {
    this.speakGen++;
    const all = [...this.prevClips, ...(this.clip ? [this.clip] : [])];
    this.prevClips = [];
    if (!all.length) return;
    this.clip = null;
    for (const c of all) {
      try { c.source.stop(); } catch { /* not started */ }
      for (const p of c.patches.values()) p.bitmap.close();
    }
    this.clearPatch();
  }

  /** Speech is being prepared or played (the idle loop is not switched then). */
  get speaking(): boolean { return this.clip !== null || this.prevClips.length > 0; }

  /**
   * Switches to another idle loop while idle: waits (up to 3 s) for the shown loop to reach a frame with a similar frame
   * in the other loop, seeks the other loop there and crossfades (0.3 s). Resolves false when there is nothing to do.
   */
  async switchIdle(to: string): Promise<boolean> {
    const target = this.loops.get(to);
    const from = this.shownLoop;
    const table = this.pairs?.pairs[`${from.name}->${to}`];
    if (!target || target === from || !table || this.switching || this.clip) return false;
    this.switching = true;
    try {
      const fps = from.avatar.meta.fps;
      const L = from.avatar.meta.walk.length;
      const c = chooseIdlePair(this.videoStep(), from.avatar.meta.walk, table, SWITCH_MIN_WAIT, Math.min(L - 1, SWITCH_MAX_WAIT));
      // the other loop's walk position of the pair's frame, moving the same way as the shown loop (ping-pong walks)
      const n = target.avatar.meta.walk.length;
      const rising = c.fromWalk < L / 2;
      const tw = rising || c.toSource === 0 ? c.toSource : n - c.toSource;
      await seekVideo(target.video, (tw + 0.1) / fps);
      // wait until the shown loop presents the pair's frame
      const left = (((c.fromWalk - this.videoStep()) % L) + L) % L;
      await new Promise((r) => setTimeout(r, (left * 1000) / fps));
      await target.video.play().catch(() => undefined);
      // crossfade: the target fades in over the shown loop, then becomes the shown loop
      const t0 = performance.now();
      await new Promise<void>((resolve) => {
        const step = (): void => {
          const p = Math.min(1, (performance.now() - t0) / FADE_MS);
          target.video.style.opacity = String(p);
          if (p < 1) requestAnimationFrame(step); else resolve();
        };
        requestAnimationFrame(step);
      });
      from.video.style.opacity = '0';
      from.video.pause();
      this.shownLoop = target;
      this.lastStep = -1;
      this.watchVideo();
      return true;
    } finally { this.switching = false; }
  }

  /** Frame of the clip the sound is at now (negative before it starts). */
  private clipFrame(c: Clip): number {
    return (this.ctx.currentTime - c.soundAt) * this.avatar.meta.fps;
  }

  /** Inference time per frame since the start (ms). */
  frameTiming(): FrameTiming {
    const t = this.timing;
    return { frames: t.frames, inferMsPerFrame: t.frames ? t.inferMs / t.frames : 0 };
  }

  private clearPatch(): void {
    const s = this.shown;
    if (s) this.canvas.getContext('2d')!.clearRect(s.x, s.y, s.w, s.h);
    this.shown = null;
  }

  /** The video presents step: draw the patch made for it (or nothing). The step counts loops of the video. */
  private onVideoFrame(rawStep: number): void {
    const walkLen = this.avatar.meta.walk.length;
    if (rawStep === this.lastStep) return;
    this.lastStep = rawStep;
    this.clearPatch();
    // the clip whose sound plays now: a previous chunk until its sound ends, then the next one (chunks follow back to back)
    let c = this.clip;
    const now = this.ctx.currentTime;
    for (const pc of this.prevClips) if (now < pc.soundAt + pc.duration) { c = pc; break; }
    this.prevClips = this.prevClips.filter((pc) => { if (now < pc.soundAt + pc.duration + 0.2) return true; for (const q of pc.patches.values()) q.bitmap.close(); return false; });
    if (!c) return;
    // the clip frame of this video frame: its walk position relative to the clip's start, on the lap nearest to where
    // the sound is (the video loops the walk, so a position repeats every walkLen frames). Nothing before the sound.
    const cf = this.clipFrame(c);
    if (cf < -0.5) return;
    const w = ((rawStep % walkLen) + walkLen) % walkLen;
    const base = (((w - c.walkStart) % walkLen) + walkLen) % walkLen;
    const f = base + walkLen * Math.round((cf - base) / walkLen);
    for (const [i, p] of c.patches) if (i < f) { p.bitmap.close(); c.patches.delete(i); }
    const p = c.patches.get(f);
    if (!p) return;
    this.canvas.getContext('2d')!.drawImage(p.bitmap, p.x, p.y);
    this.shown = { x: p.x, y: p.y, w: p.bitmap.width, h: p.bitmap.height };
  }
}

function seekVideo(v: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve) => {
    const done = (): void => { v.removeEventListener('seeked', done); resolve(); };
    v.addEventListener('seeked', done);
    v.currentTime = t;
    setTimeout(done, 1500);
  });
}

export { resample };
