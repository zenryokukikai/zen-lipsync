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
  /** Per frame of this clip: false when nothing sounds under it. */
  speech: boolean[];
  /** Per frame: strength of the model's change drawn (patchStrength: 1 on speech, fading over the silence after it, 0 = no patch). */
  strength: Float32Array;
  /**
   * Patches made so far, per clip frame; null for a frame with no patch (strength 0: the idle video is shown as it is).
   * Every bitmap here is the clip's own, closed once when it is dropped (closePatches).
   */
  patches: Map<number, LipPatch | null>;
  f: number;
  ahead: number;
}

interface Clip {
  /** AudioContext time at which the sound starts (frame 0 of the clip), and its length (s). */
  soundAt: number;
  duration: number;
  frames: number;
  walkStart: number;
  patches: Map<number, LipPatch | null>;
  source: AudioBufferSourceNode;
  prep: Prepared;
  ended: Promise<void>;
}

interface Loop { name: string; avatar: Avatar; comp: Compositor }
/** One of the player's two <video> elements and the loop whose idle video it holds (null: none loaded yet). */
interface Screen { video: HTMLVideoElement; loop: Loop | null }

type VideoWithCallback = HTMLVideoElement & { requestVideoFrameCallback?: (cb: (now: number, md: { mediaTime: number; presentedFrames?: number; expectedDisplayTime?: number }) => void) => number };

/** What the player drew on the last presented video frame (for a debug overlay). */
export interface DrawInfo {
  /** Video frame index from the frame callback (mediaTime x fps, rounded) and the walk position it shows. */
  videoFrame: number;
  walk: number;
  /** Clip frame drawn (null: no patch), its walk position, and the audio clock's clip frame at that moment. */
  patchFrame: number | null;
  patchWalk: number | null;
  audioFrame: number | null;
  /** Video frames the callback skipped since the last one (presentedFrames gap - 1). */
  skipped: number;
  /** The pairing asked for, and the one used on this frame ('auto' resolves to 'video' or 'next'). */
  sync: PatchSync;
  pairing: 'video' | 'next' | 'canvas';
  /** Phase of this frame: idle, start (first FIRST_FRAMES frames of a run), speech, end (the fade after speech). */
  phase: DrawPhase;
  /** Time of drawImage(video) and of the patch draw on this frame (ms). */
  videoDrawMs: number;
  patchDrawMs: number;
  /**
   * Canvas draw done minus the video frame's expected display time (ms). Measured on the frame callback's own clock: the
   * time the draw took (performance.now() around it) minus the time left at the callback (expectedDisplayTime - now),
   * so a browser whose callback timestamps are on another clock than performance.now() is measured right. Positive: the
   * canvas is drawn after the video frame is due on screen, so it shows with a later refresh. null: no patch drawn, no
   * expectedDisplayTime, or expectedDisplayTime more than CLOCK_MISMATCH_MS away from now (not on the same clock).
   */
  lateMs: number | null;
  /** expectedDisplayTime - now at the callback (ms, raw; for the overlay), and now - performance.now() there. */
  leadMs: number | null;
  nowSkewMs: number | null;
}

export type DrawPhase = 'idle' | 'start' | 'speech' | 'end';

/**
 * How the mouth patch is paired with the presented video frame (the frame callback's mediaTime):
 *  'video' — the patch of that frame alone, on a canvas over the <video> (the default; as 1.0.0);
 *  'next'  — the patch of the following frame (for a browser that presents the canvas one video frame late);
 *  'canvas'— the video frame and its patch drawn together on the canvas, every frame (the <video> stays underneath);
 *  'auto'  — 'video' or 'next', chosen on the device from how late the patch draws are (AutoPairing; the default).
 */
export type PatchSync = 'video' | 'next' | 'canvas' | 'auto';

/**
 * 'auto': on the device, once AUTO_WINDOW patch draws are measured, the last AUTO_WINDOW are checked against their video frame's expected
 * display time; when at least AUTO_NEXT_AT of them were late, the patch of the next frame is drawn ('next'), and back to
 * the frame's own patch ('video') only when at most AUTO_VIDEO_AT were late (no flipping on a borderline browser).
 * Measured: Safari on iPhone was late on every frame and the next frame's patch matched; Chromium on a desktop was late
 * on about two thirds (draws finishing right at the display time) and its own frame matched.
 */
const AUTO_WINDOW = 25;
const AUTO_NEXT_AT = 0.9;
const AUTO_VIDEO_AT = 0.5;
/** expectedDisplayTime further than this from the callback's now is taken to be on another clock (not measured). */
const CLOCK_MISMATCH_MS = 250;
/**
 * When a browser's expectedDisplayTime is not on the callback's clock (nothing can be measured), 'auto' draws the next
 * frame's patch once this many video frame callbacks in a row (idle ones too, so it is decided before the first speech)
 * showed it, and none was ever measured: the browser seen doing that (Safari on iPhone, whose expectedDisplayTime is
 * far off performance.now()) presented the patch canvas one video frame late every time.
 */
const AUTO_UNMEASURED_NEXT_AFTER = 2;

/** The 'auto' pairing decision from the frame callbacks and patch draws seen so far (see AUTO_WINDOW). */
export class AutoPairing {
  pairing: 'video' | 'next' = 'video';
  private readonly window: boolean[] = [];
  private measured = 0;
  private unmeasuredRun = 0;

  /** A video frame callback with an expectedDisplayTime: on the callback's clock (sameClock) or on another one. */
  clock(sameClock: boolean): void {
    if (sameClock) { this.unmeasuredRun = 0; return; }
    this.unmeasuredRun++;
    if (this.measured === 0 && this.unmeasuredRun >= AUTO_UNMEASURED_NEXT_AFTER) this.pairing = 'next';
  }

  /** lateMs: a patch draw done minus its frame's display time (measured: the display time was on the callback's clock). */
  add(lateMs: number): void {
    this.measured++;
    this.window.push(lateMs > 0);
    if (this.window.length > AUTO_WINDOW) this.window.shift();
    if (this.window.length < AUTO_WINDOW) return;
    const frac = this.window.filter(Boolean).length / this.window.length;
    if (this.pairing === 'video' && frac >= AUTO_NEXT_AT) this.pairing = 'next';
    else if (this.pairing === 'next' && frac <= AUTO_VIDEO_AT) this.pairing = 'video';
  }
}

/** Frames rendered before the sound starts (the rest render while it plays, kept ahead of the sound). */
const FIRST_FRAMES = 8;
/** Least lead (video frames) between now and the clip's first frame. */
const MIN_LEAD = 8;
/** Idle switch: the pair is looked for 3 to 75 frames ahead (0.1 to 3 s), then a crossfade of this length. */
const SWITCH_MIN_WAIT = 3;
const SWITCH_MAX_WAIT = 75;
const FADE_MS = 300;
/**
 * Idle switch: the other loop must present a frame of its own within this time after play(), or the switch is given up
 * and the shown loop goes on. iOS Safari can refuse play() outside a user gesture (Low Power Mode: even a muted inline
 * video) without any error on the page; swapping to that video froze the picture and its frame callbacks never came,
 * so the mouth stopped while the sound went on.
 */
const SWITCH_FRAME_MS = 1000;
/** Idle switch: the standby video loading another loop's file must be able to show a frame within this time. */
const SWITCH_LOAD_MS = 4000;
/** The shown loop is played again when no frame callback came for this long while it should play (watchdog). */
const WATCHDOG_MS = 700;
/** How often the watchdog looks (ms). */
const WATCHDOG_CHECK_MS = 250;

/** Why an idle switch was given up (switchFailures.reasons). */
export type SwitchFailure = 'cancelled' | 'load-timeout' | 'play-refused' | 'no-frame' | 'seek-timeout';
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

export class LipSyncPlayer {
  readonly lips: LipModels;
  private readonly loops = new Map<string, Loop>();
  private shownLoop: Loop;
  /**
   * The two <video> elements, however many idle loops there are: the one shown, and the standby one (null with one loop)
   * that loads, seeks and plays the next loop hidden before the crossfade, after which the two swap roles. Both are
   * started in the user's gesture (resume()), so the browser lets them play later outside a gesture (iOS Safari in Low
   * Power Mode refuses play() otherwise, with no error on the page).
   */
  private shown: Screen;
  private standby: Screen | null;
  /** The <video> the page gave to create() (kept on destroy()). */
  private readonly pageVideo: HTMLVideoElement;
  private readonly ctx: AudioContext;
  private clip: Clip | null = null;
  private shownPatch: { x: number; y: number; w: number; h: number } | null = null;
  private lastStep = -1;
  /** Lead used for the next clip: the last measured time to the first frames + a margin (video frames). */
  private lead = MIN_LEAD;
  private readonly timing = { frames: 0, inferMs: 0 };
  private switching = false;
  /** Clips of the current utterance whose sound has been replaced by the next chunk's but may still be playing. */
  private prevClips: Clip[] = [];
  private speakGen = 0;
  private watchGen = 0;
  /**
   * Pairing of the patch with the presented video frame (PatchSync). 'canvas': the canvas gets the presented video frame
   * and then its mouth patch, drawn together in the video's frame callback, so the two are always shown as one picture;
   * with the patch alone over the <video>, a browser that presents the canvas a display refresh after the video frame
   * shows the patch on the next frame of a moving head (on a nod the nose looks short).
   */
  private sync: PatchSync = 'auto';
  /** 'auto': the pairing chosen from the patch draws seen. */
  private readonly auto = new AutoPairing();
  /** Frames measured and frames whose expectedDisplayTime was on another clock (not measured), since create(). */
  readonly lateCounts = { measured: 0, late: 0, clockMismatch: 0 };
  /** Errors thrown while drawing a presented frame (the frame callbacks go on), and the last one's text. */
  readonly drawErrors = { count: 0, last: '' };
  /** Idle switches done and given up (the shown loop went on), by reason; the last failure's detail. */
  readonly switchCounts = { done: 0, failed: 0, reasons: { cancelled: 0, 'load-timeout': 0, 'play-refused': 0, 'no-frame': 0, 'seek-timeout': 0 } as Record<SwitchFailure, number>, last: '' };
  /** Watchdog: times the shown loop was played again after its frame callbacks stopped. */
  readonly watchdog = { kicks: 0, last: '' };
  /** Frame callbacks while speaking at which the standby <video> was playing too (two decoders: it should stay 0). */
  readonly standbyPlaying = { speechFrames: 0 };
  /** performance.now() of the shown loop's last frame callback, and whether the page paused it (pause()). */
  private lastFrameAt = 0;
  /** A frame callback came at least once (the watchdog waits for it: before a gesture the browser may hold autoplay). */
  private framesSeen = false;
  private pausedByUser = false;
  /** Set by speech arriving during an idle switch: the switch is dropped unless its crossfade has begun. */
  private cancelSwitch: (() => void) | null = null;
  private watchdogTimer: ReturnType<typeof setInterval> | null = null;
  /** performance.now() of the watchdog's last check (a check that itself came late is not counted: main thread busy). */
  private lastCheckAt = 0;
  /** Idle crossfade on a composed canvas: the target loop and its start (performance.now). */
  private fade: { video: HTMLVideoElement; t0: number } | null = null;
  private lastPresented = -1;
  /** The last presented frame's draw (debug overlay), and counters since create(). */
  lastDraw: DrawInfo | null = null;
  readonly drawCounts = { frames: 0, patches: 0, skipped: 0, skippedBy: { idle: 0, start: 0, speech: 0, end: 0 } as Record<DrawPhase, number> };
  /** Recent per-frame draw times (ms, about 2 s at 25 fps), for the debug overlay's p90. */
  readonly drawTimes = { video: [] as number[], patch: [] as number[] };

  private constructor(loops: Loop[], lips: LipModels, readonly canvas: HTMLCanvasElement, readonly set: AvatarSet | null, private readonly pairs: IdleSwitchFile | null, video: HTMLVideoElement, standby: HTMLVideoElement | null) {
    this.lips = lips;
    for (const l of loops) this.loops.set(l.name, l);
    this.shownLoop = loops[0]!;
    this.shown = { video, loop: loops[0]! };
    this.pageVideo = video;
    this.standby = standby ? { video: standby, loop: null } : null;
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
  static async create(url: string, video: HTMLVideoElement, canvas: HTMLCanvasElement, opts: ModelOptions & { onBytes?: (n: number) => void; idle?: string | null; sync?: PatchSync } = {}): Promise<LipSyncPlayer> {
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
      const [W, H] = avatar.meta.frame_size;
      return { name: packs[i]![0], avatar, comp: new Compositor({ x: 0, width: W, height: H }, avatar) };
    });
    // one standby <video> for every other loop (a clone keeps the first video's style: the SDK's layer, absolute, same
    // fit, so it stacks exactly on it); a single loop needs none
    let standby: HTMLVideoElement | null = null;
    if (loops.length > 1) {
      standby = video.cloneNode(false) as HTMLVideoElement;
      standby.removeAttribute('id');
      standby.style.opacity = '0';
      video.after(standby);
      initVideo(standby);
    }
    const p = new LipSyncPlayer(loops, lips, canvas, set, pairs, video, standby);
    p.sync = opts.sync ?? 'auto';
    initVideo(video);
    loadLoop(video, loops[0]!);
    await video.play().catch(() => undefined);
    p.watchVideo();
    return p;
  }

  /**
   * Plays the shown idle video and any sound; call it from a user gesture (a tap). The other idle loops are played there
   * too and keep playing hidden, so an idle switch later (not in a gesture) swaps to a video already playing: iOS Safari
   * may refuse play() outside a gesture (Low Power Mode), and loads a video only once it has played.
   */
  resume(): void {
    this.pausedByUser = false;
    if (this.shown.video.paused) void this.shown.video.play().catch(() => undefined);
    // the standby video is started here too (in the gesture) and paused again: from then on the browser lets it play
    // outside a gesture, with any loop's file in it
    const sb = this.standby;
    if (sb && sb.video.paused && !this.switching) {
      if (!sb.loop) { sb.loop = this.otherLoop(); loadLoop(sb.video, sb.loop); }
      const v = sb.video;
      const p = v.play();
      if (p) p.then(() => { if (this.standby === sb && !this.switching) v.pause(); }, () => undefined);
    }
    if (this.ctx.state === 'suspended') void this.ctx.resume().catch(() => undefined);
  }

  /** A loop other than the shown one (the first such loop: the one a switch most likely goes to). */
  private otherLoop(): Loop {
    for (const l of this.loops.values()) if (l !== this.shownLoop) return l;
    return this.shownLoop;
  }

  /** Pauses the shown idle video and the sound. */
  pause(): void {
    this.pausedByUser = true;
    this.shown.video.pause();
    this.standby?.video.pause();
    if (this.ctx.state === 'running') void this.ctx.suspend().catch(() => undefined);
  }

  /** Stops everything, removes the extra idle <video> elements and closes the audio. */
  destroy(): void {
    this.stop();
    this.watchGen++;
    if (this.watchdogTimer) clearInterval(this.watchdogTimer);
    this.watchdogTimer = null;
    // the page's own <video> stays (emptied); the one the SDK added is removed (either may be the shown one by now)
    for (const sc of [this.shown, this.standby]) { if (!sc) continue; sc.video.pause(); sc.video.removeAttribute('src'); sc.video.replaceChildren(); if (sc.video !== this.pageVideo) sc.video.remove(); }
    void this.ctx.close().catch(() => undefined);
    (this.lips as { destroy?: () => void }).destroy?.();
  }

  private watchVideo(): void {
    const gen = ++this.watchGen;
    const loop = this.shownLoop;
    const vc = this.shown.video as VideoWithCallback;
    const fps = loop.avatar.meta.fps;
    // presentedFrames counts per element: after a switch the other element's count must not be read as skipped frames
    this.lastPresented = -1;
    this.lastFrameAt = performance.now();
    if (!this.watchdogTimer) { this.lastCheckAt = performance.now(); this.watchdogTimer = setInterval(() => this.checkWatchdog(), WATCHDOG_CHECK_MS); }
    if (typeof vc.requestVideoFrameCallback === 'function') {
      const onFrame = (now: number, md: { mediaTime: number; presentedFrames?: number; expectedDisplayTime?: number }): void => {
        if (gen !== this.watchGen) return;
        this.lastFrameAt = performance.now();
        this.framesSeen = true;
        if (this.clip && this.standby && !this.standby.video.paused) this.standbyPlaying.speechFrames++;
        // the next callback first: an error in this frame's drawing must not end the frame callbacks (the mouth would stop
        // for good while the sound goes on)
        vc.requestVideoFrameCallback!(onFrame);
        try { this.presentFrame(now, md, fps); } catch (e) { this.noteDrawError(e); }
      };
      vc.requestVideoFrameCallback(onFrame);
    } else {
      const tick = (): void => {
        if (gen !== this.watchGen) return;
        this.lastFrameAt = performance.now();
        this.framesSeen = true;
        requestAnimationFrame(tick);
        try { this.onVideoFrame(Math.floor(vc.currentTime * fps + 1e-3)); } catch (e) { this.noteDrawError(e); }
      };
      requestAnimationFrame(tick);
    }
  }

  /**
   * Watchdog of the shown loop: when no frame callback came for WATCHDOG_MS while the video should play (not paused by
   * the page, the page visible), the video is played again and its frame callbacks registered anew.
   */
  private checkWatchdog(): void {
    const now = performance.now(), sinceCheck = now - this.lastCheckAt;
    this.lastCheckAt = now;
    // the timer itself came late: the main thread was busy (the frame callbacks were held by the same work), not the video
    if (sinceCheck > 2 * WATCHDOG_CHECK_MS) { this.lastFrameAt = now; return; }
    if (!this.framesSeen || this.pausedByUser || this.switching || (typeof document !== 'undefined' && document.visibilityState === 'hidden')) { this.lastFrameAt = performance.now(); return; }
    if (performance.now() - this.lastFrameAt < WATCHDOG_MS) return;
    const v = this.shown.video;
    this.watchdog.kicks++;
    this.watchdog.last = `${this.shownLoop.name} paused=${v.paused} readyState=${v.readyState}`;
    if (v.paused) void v.play().catch(() => undefined);
    this.lastStep = -1;
    this.watchVideo();
  }

  /** The draw of one presented video frame (watchVideo's frame callback). */
  private presentFrame(now: number, md: { mediaTime: number; presentedFrames?: number; expectedDisplayTime?: number }, fps: number): void {
    const t0 = performance.now();
    const skew = now - t0;
    const pf = md.presentedFrames ?? -1;
    const skipped = pf >= 0 && this.lastPresented >= 0 ? Math.max(0, pf - this.lastPresented - 1) : 0;
    this.lastPresented = pf;
    this.drawCounts.skipped += skipped;
    this.onVideoFrame(Math.round(md.mediaTime * fps), skipped);
    const d = this.lastDraw;
    if (d) d.nowSkewMs = skew;
    if (md.expectedDisplayTime === undefined) return;
    const lead = md.expectedDisplayTime - now;
    const sameClock = Math.abs(lead) <= CLOCK_MISMATCH_MS;
    // every callback (idle ones too): a browser whose display time is on another clock is known before the first speech
    if (this.sync === 'auto') this.auto.clock(sameClock);
    if (d && d.patchFrame !== null) this.measureLate(d, lead, sameClock, performance.now() - t0);
  }

  /** An error while drawing a frame: counted and kept (debug overlay); the next frame is drawn as usual. */
  private noteDrawError(e: unknown): void {
    this.drawErrors.count++;
    this.drawErrors.last = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  }

  /** A patch draw against its frame's display time (lead: time left at the callback; took: callback to draw done). */
  private measureLate(d: DrawInfo, lead: number, sameClock: boolean, took: number): void {
    d.leadMs = lead;
    if (!sameClock) { this.lateCounts.clockMismatch++; return; }
    d.lateMs = took - lead;
    this.lateCounts.measured++;
    if (d.lateMs > 0) this.lateCounts.late++;
    if (this.sync === 'auto') this.auto.add(d.lateMs);
  }

  /** The pairing used now ('auto' resolved). */
  get pairing(): 'video' | 'next' | 'canvas' { return this.sync === 'auto' ? this.auto.pairing : this.sync; }

  /** Walk position the shown video shows now (the idle video loops the walk). */
  private videoStep(): number {
    const a = this.avatar;
    return Math.floor(this.shown.video.currentTime * a.meta.fps + 1e-3) % a.meta.walk.length;
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
    // a switch still preparing is dropped; one already crossfading ends within FADE_MS
    this.cancelSwitch?.();
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
      if (p) closePatches(p.patches);
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
    const strength = patchStrength(speech);
    const p: Prepared = { pcm, sampleRate, walkStart, frames: mel.frames, zl, speech, strength, patches: new Map(), f: 0, ahead: 0 };
    await this.batch(p, loop, Math.min(FIRST_FRAMES, mel.frames));
    return p;
  }

  private async batch(p: Prepared, loop: Loop, n: number): Promise<void> {
    const avatar = loop.avatar;
    const us = Array.from({ length: n }, (_, i) => avatar.walk(p.walkStart + p.f + i));
    const tb = performance.now();
    // the model's change is kept on every frame with a strength (speech and the fade after it); the others are zeroed
    const drawn = Array.from(p.strength, (k) => k > 0);
    const [pred, srcs] = await Promise.all([inferFrames(this.lips, avatar, us, p.zl, p.f, n, drawn), Promise.all(us.map((u) => avatar.frame(u)))]);
    this.timing.inferMs += performance.now() - tb;
    this.timing.frames += n;
    const plane = pred.length / n;
    for (let i = 0; i < n; i++) {
      const k = p.strength[p.f + i]!;
      if (k > 0 && k < 1) { const d = pred.subarray(i * plane, (i + 1) * plane); for (let j = 0; j < d.length; j++) d[j]! *= k; }
    }
    // frames with no patch hold null (nothing drawn): a shared empty bitmap here was closed with the first clip dropped,
    // and every later silent frame then threw on drawImage (InvalidStateError), which ended the frame callbacks and the
    // mouth stopped while the sound went on
    srcs.forEach((src, i) => p.patches.set(p.f + i, p.strength[p.f + i]! > 0 ? loop.comp.frame(src, pred, i) : null));
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
      const vt = this.shown.video.currentTime * fps;
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
      closePatches(c.patches);
    }
    this.clearPatch();
  }

  /** Speech is being prepared or played (the idle loop is not switched then). */
  get speaking(): boolean { return this.clip !== null || this.prevClips.length > 0; }

  /**
   * Switches to another idle loop while idle: waits (up to 3 s) for the shown loop to reach a frame with a similar frame
   * in the other loop, seeks the other loop there and crossfades (0.3 s). Resolves false when there is nothing to do, or
   * when the other loop's video does not play (play() refused, no frame within SWITCH_FRAME_MS, the seek never done):
   * the shown loop then goes on as it was and the reason is counted (switchCounts).
   */
  async switchIdle(to: string): Promise<boolean> {
    const target = this.loops.get(to);
    const from = this.shownLoop;
    const table = this.pairs?.pairs[`${from.name}->${to}`];
    const sb = this.standby;
    if (!target || target === from || !table || !sb || this.switching || this.clip) return false;
    this.switching = true;
    const v = sb.video;
    // speech arriving before the crossfade: every wait below ends at once (the switch is dropped as 'cancelled')
    let cancelled = false;
    const cancelP = new Promise<'cancelled'>((r) => { this.cancelSwitch = () => { cancelled = true; r('cancelled'); }; });
    const orCancel = <T>(x: Promise<T>): Promise<T | 'cancelled'> => Promise.race([x, cancelP]);
    const fail = (why: SwitchFailure, detail: string): false => {
      this.switchCounts.failed++;
      this.switchCounts.reasons[why]++;
      this.switchCounts.last = `${from.name}->${to} ${why}: ${detail}`;
      this.fade = null;
      v.style.opacity = '0';
      v.pause();
      if (this.shown.video.paused && !this.pausedByUser) void this.shown.video.play().catch(() => undefined);
      return false;
    };
    try {
      const fps = from.avatar.meta.fps;
      const L = from.avatar.meta.walk.length;
      // 1. the standby video holds the target loop's file (loaded now when it holds another loop's)
      if (sb.loop !== target) { sb.loop = target; loadLoop(v, target); }
      // (it may also still be loading the file put in it at the gesture)
      if (v.readyState < 2) {
        const ready = await orCancel(videoReady(v, SWITCH_LOAD_MS));
        if (ready === 'cancelled') return fail('cancelled', 'speech while loading');
        if (!ready) { sb.loop = null; return fail('load-timeout', `readyState=${v.readyState} networkState=${v.networkState}`); }
      }
      const c = chooseIdlePair(this.videoStep(), from.avatar.meta.walk, table, SWITCH_MIN_WAIT, Math.min(L - 1, SWITCH_MAX_WAIT));
      // the other loop's walk position of the pair's frame, moving the same way as the shown loop (ping-pong walks)
      const n = target.avatar.meta.walk.length;
      const rising = c.fromWalk < L / 2;
      const tw = rising || c.toSource === 0 ? c.toSource : n - c.toSource;
      // 2. seeked to the pair's frame
      const seeked = await orCancel(seekVideo(v, (tw + 0.1) / fps));
      if (seeked === 'cancelled') return fail('cancelled', 'speech while seeking');
      if (!seeked) return fail('seek-timeout', `readyState=${v.readyState}`);
      // wait until the shown loop presents the pair's frame
      const left = (((c.fromWalk - this.videoStep()) % L) + L) % L;
      await orCancel(new Promise((r) => setTimeout(r, (left * 1000) / fps)));
      if (cancelled) return fail('cancelled', 'speech while waiting for the pair frame');
      // the standby takes the shown element's layout (the page may set object-position / fit on its <video> after create(),
      // and the canvas follows that): else the patches land off the face once it is shown
      copyLayout(this.shown.video, v);
      // 3-4. playing, and a frame of its own presented, before it is shown (a video that does not play would freeze the
      // picture and end the frame callbacks)
      const played = await orCancel(playVideo(v, SWITCH_FRAME_MS));
      if (played === 'cancelled') return fail('cancelled', 'speech while starting the video');
      if (played !== 'ok') return fail(played === 'refused' ? 'play-refused' : 'no-frame', `paused=${v.paused} readyState=${v.readyState}`);
      // 5. crossfade: the standby fades in over the shown video, then the two swap roles (speech now waits for its end)
      this.cancelSwitch = null;
      const t0 = performance.now();
      // composed canvas: the shown loop's frame callback draws both videos (the standby over it at the fade's opacity)
      if (this.pairing === 'canvas') this.fade = { video: v, t0 };
      await new Promise<void>((resolve) => {
        // by timers, not animation frames: a browser that holds animation frames must not hold the switch (and the speech
        // waiting for it) beyond the fade
        const step = (): void => {
          const p = Math.min(1, (performance.now() - t0) / FADE_MS);
          v.style.opacity = String(p);
          if (p < 1) setTimeout(step, 16); else resolve();
        };
        step();
      });
      this.fade = null;
      const old = this.shown;
      old.video.style.opacity = '0';
      old.video.pause();
      this.shown = sb;
      this.standby = old;
      this.shownLoop = target;
      this.lastStep = -1;
      this.watchVideo();
      this.switchCounts.done++;
      return true;
    } finally { this.switching = false; this.cancelSwitch = null; }
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
    const s = this.shownPatch;
    if (s) this.canvas.getContext('2d')!.clearRect(s.x, s.y, s.w, s.h);
    this.shownPatch = null;
  }

  /** The video presents step: draw the patch made for it (or nothing). The step counts loops of the video. */
  private onVideoFrame(rawStep: number, skipped = 0): void {
    const walkLen = this.avatar.meta.walk.length;
    if (rawStep === this.lastStep) return;
    this.lastStep = rawStep;
    const w = ((rawStep % walkLen) + walkLen) % walkLen;
    const info: DrawInfo = { videoFrame: rawStep, walk: w, patchFrame: null, patchWalk: null, audioFrame: null, skipped, sync: this.sync, pairing: this.pairing, phase: 'idle', videoDrawMs: 0, patchDrawMs: 0, lateMs: null, leadMs: null, nowSkewMs: null };
    this.lastDraw = info;
    this.drawCounts.frames++;
    const g = this.canvas.getContext('2d')!;
    if (info.pairing === 'canvas') {
      // this callback's video frame first, then its patch: one picture on the canvas. Drawn on every frame, idle too: a
      // canvas switched on at speech and off after it is shown a display refresh late at either switch, so the picture
      // would step back one video frame there.
      const t0 = performance.now();
      g.drawImage(this.shown.video, 0, 0, this.canvas.width, this.canvas.height);
      const fd = this.fade;
      if (fd) {
        g.globalAlpha = Math.min(1, (performance.now() - fd.t0) / FADE_MS);
        g.drawImage(fd.video, 0, 0, this.canvas.width, this.canvas.height);
        g.globalAlpha = 1;
      }
      info.videoDrawMs = performance.now() - t0;
      pushRecent(this.drawTimes.video, info.videoDrawMs);
      this.shownPatch = null;
    } else this.clearPatch();
    const done = (): void => { this.drawCounts.skippedBy[info.phase] += skipped; };
    // the clip whose sound plays now: a previous chunk until its sound ends, then the next one (chunks follow back to back)
    let c = this.clip;
    const now = this.ctx.currentTime;
    for (const pc of this.prevClips) if (now < pc.soundAt + pc.duration) { c = pc; break; }
    this.prevClips = this.prevClips.filter((pc) => { if (now < pc.soundAt + pc.duration + 0.2) return true; closePatches(pc.patches); return false; });
    if (!c) { done(); return; }
    // the clip frame of this video frame (the one presented now, from the frame callback: after a skipped frame, the patch
    // of the frame actually shown): its walk position relative to the clip's start, on the lap nearest to where the sound
    // is (the video loops the walk, so a position repeats every walkLen frames). Nothing before the sound.
    const cf = this.clipFrame(c);
    if (cf < -0.5) { done(); return; }
    // 'next': the patch of the following video frame (the walk position one on)
    const next = info.pairing === 'next' ? 1 : 0;
    const pw = (w + next) % walkLen;
    const base = (((pw - c.walkStart) % walkLen) + walkLen) % walkLen;
    const f = base + walkLen * Math.round((cf + next - base) / walkLen);
    info.audioFrame = cf;
    info.phase = f < FIRST_FRAMES && !this.prevClips.length && c === this.clip ? 'start' : c.prep.speech[f] ? 'speech' : (c.prep.strength[f] ?? 0) > 0 ? 'end' : 'idle';
    done();
    for (const [i, q] of c.patches) if (i < f) { q?.bitmap.close(); c.patches.delete(i); }
    const p = c.patches.get(f);
    if (!p) return;
    const t1 = performance.now();
    g.drawImage(p.bitmap, p.x, p.y);
    info.patchDrawMs = performance.now() - t1;
    pushRecent(this.drawTimes.patch, info.patchDrawMs);
    info.patchFrame = f;
    info.patchWalk = (c.walkStart + f) % walkLen;
    this.drawCounts.patches++;
    if (info.pairing !== 'canvas') this.shownPatch = { x: p.x, y: p.y, w: p.bitmap.width, h: p.bitmap.height };
  }
}

/** Closes a clip's patch bitmaps and empties its map (each bitmap is the clip's own: closed once). */
function closePatches(m: Map<number, LipPatch | null>): void {
  for (const q of m.values()) q?.bitmap.close();
  m.clear();
}

/** Keeps the last 50 values (about 2 s of frames). */
function pushRecent(a: number[], v: number): void {
  a.push(v);
  if (a.length > 50) a.shift();
}

/** Copies the inline style of from onto to, except its opacity (the crossfade's). */
export function copyLayout(from: HTMLVideoElement, to: HTMLVideoElement): void {
  const fs = from.style, ts = to.style;
  for (let i = 0; i < fs.length; i++) {
    const k = fs.item(i);
    if (k === 'opacity') continue;
    ts.setProperty(k, fs.getPropertyValue(k), fs.getPropertyPriority(k));
  }
}

/** Settings every idle <video> gets (muted inline loop: the browser may play it without sound permission). */
function initVideo(v: HTMLVideoElement): void {
  v.muted = true;
  v.loop = true;
  v.playsInline = true;
  v.preload = 'auto';
}

/** Puts a loop's idle video file(s) in v and loads them. */
function loadLoop(v: HTMLVideoElement, l: Loop): void {
  v.replaceChildren(...l.avatar.meta.video.map((s) => Object.assign(document.createElement('source'), { src: `${l.avatar.base}/${s.src}`, type: s.type })));
  v.load();
}

/** Resolves true once v can show a frame (loadeddata), false on an error or after ms. */
function videoReady(v: HTMLVideoElement, ms: number): Promise<boolean> {
  return new Promise((resolve) => {
    if (v.readyState >= 2) { resolve(true); return; }
    let timer: ReturnType<typeof setTimeout> | null = null;
    const done = (ok: boolean): void => { v.removeEventListener('loadeddata', onOk); v.removeEventListener('error', onErr, true); if (timer) clearTimeout(timer); resolve(ok); };
    const onOk = (): void => done(true);
    const onErr = (): void => done(false);
    v.addEventListener('loadeddata', onOk);
    v.addEventListener('error', onErr, true);
    timer = setTimeout(() => done(v.readyState >= 2), ms);
  });
}

/** Seeks v to t; resolves true on 'seeked', false when it did not come within 1.5 s. */
function seekVideo(v: HTMLVideoElement, t: number): Promise<boolean> {
  return new Promise((resolve) => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const done = (ok: boolean): void => { v.removeEventListener('seeked', onSeeked); if (timer) clearTimeout(timer); resolve(ok); };
    const onSeeked = (): void => done(true);
    v.addEventListener('seeked', onSeeked);
    v.currentTime = t;
    timer = setTimeout(() => done(false), 1500);
  });
}

/**
 * Plays v (if paused) and waits for it to present a frame: 'ok', 'refused' (play() rejected) or 'no-frame' (nothing
 * presented within ms). A frame callback where the browser has one, else the clock moving while not paused.
 */
export function playVideo(v: HTMLVideoElement, ms: number): Promise<'ok' | 'refused' | 'no-frame'> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (r: 'ok' | 'refused' | 'no-frame'): void => { if (settled) return; settled = true; clearTimeout(timer); clearInterval(poll); resolve(r); };
    const timer = setTimeout(() => finish('no-frame'), ms);
    const vc = v as VideoWithCallback;
    if (typeof vc.requestVideoFrameCallback === 'function') vc.requestVideoFrameCallback(() => finish('ok'));
    const t0 = v.currentTime;
    const poll = setInterval(() => { if (!v.paused && v.currentTime !== t0 && typeof vc.requestVideoFrameCallback !== 'function') finish('ok'); }, 30);
    if (v.paused) {
      const p = v.play();
      if (p) p.catch(() => finish('refused'));
    }
  });
}

export { resample };
