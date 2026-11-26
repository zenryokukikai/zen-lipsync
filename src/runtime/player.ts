// A small player: the idle video loops in a <video> (25 fps; video frame k shows source frame walk[k]); a canvas on
// top draws, for every presented video frame, the mouth patch made for that frame while speech plays.
// speak(pcm) renders the clip's first patches, starts the sound when the video reaches the clip's first frame, renders
// the rest ahead of the sound, and draws each patch on the video frame it was made for (requestVideoFrameCallback,
// else animation frames). prepareSpeech(pcm) + play(plan) start the sound at once instead (the work that does not
// depend on the idle video's position done ahead); the mouth then joins on the first video frame its patch is ready for. An avatar with several idle loops (avatar.json) can switch loops while idle (switchIdle).
import { Avatar, Compositor, inferFrames, loadModels, sentenceLatents, speechMask, type LipModels, type LipPatch, type ModelOptions } from './avatar.js';
import { chooseIdlePair, type IdleSwitchFile } from '../core/idle-switch.js';
import { type MouthOpen } from '../core/mouth-open.js';
import { logMelYielding, MEL_SAMPLE_RATE, yieldToPage, type LogMel } from '../core/mel.js';
import { listAvatar, type AvatarSet } from '../core/pack.js';
import { melGather } from '../core/mel.js';
import { resample, resampleYielding } from '../core/resample.js';
import { browserPackSource } from './pack-source.js';
import { WGSL_MAX_BATCH } from './wgsl.js';
import { applyStrength, patchStrength } from '../core/patch-strength.js';
import { MatteOutput } from './matte.js';

export interface FrameTiming {
  /** Frames drawn so far and the inference time per frame (ms, batch time / frames). */
  frames: number;
  inferMsPerFrame: number;
}

export type { AvatarSet } from '../core/pack.js';

/**
 * Speech analysed ahead of playing (prepareSpeech): the parts that do not depend on where the idle video is when it
 * plays. The sentence model's input does (the reference latents of the walk positions under the speech), so the latents
 * and patches are made when it starts.
 */
export interface SpeechPlan {
  readonly pcm: Float32Array;
  readonly sampleRate: number;
  readonly mel: LogMel;
  /** Per frame: false when nothing sounds under it. */
  readonly speech: boolean[];
  /** Per frame: strength of the model's change drawn (patchStrength: 1 on speech, fading over the silence after it, 0 = no patch). */
  readonly strength: Float32Array;
  /** The sound, ready to play on the player's AudioContext. */
  readonly buffer: AudioBuffer;
}

/** When a speech's sound starts: the player's AudioContext time, and the performance.now() time it is heard. */
export interface SpeechStartTime {
  audioTime: number;
  performanceTime: number;
  /** This device's audio output delay at that moment (ms): performanceTime minus the time the sound is due on the AudioContext. */
  outputDelayMs: number;
}

interface Prepared {
  plan: SpeechPlan;
  /** Walk position of frame 0, frames, sentence latents (empty until made), patches made so far (f of them). */
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
  /** Video frames from the sound's start to the start of frame 0 on screen (0: speak(); 0..2 or so: play()). */
  shift: number;
}

interface Clip {
  /** AudioContext time at which the sound starts, and its length (s). Frame 0 of the clip is shown prep.shift frames later. */
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
 *  'video' — the patch of that frame alone, on a canvas over the <video> (as 1.0.0);
 *  'next'  — the patch of the following frame (for a browser that presents the canvas one video frame late);
 *  'canvas'— the video frame and its patch drawn together on the canvas, every frame, idle and crossfades too; the
 *            <video> elements play hidden (opacity 0) as the decode source only (since 1.1.2);
 *  'auto'  — 'video' or 'next', chosen on the device from how late the patch draws are (AutoPairing; the default).
 * 'video', 'next' and 'auto' show the face from the <video> element and only the mouth from the canvas. Chrome on Windows
 * can present the <video> element's picture through the GPU's video processing, which a canvas does not get: the mouth
 * patch then shows in other colours than the face around it (seen: the lower face bluish while speaking; the exact
 * processing step not identified). 'canvas' draws both from the same decoded frame; a page can choose it
 * on Windows. Elsewhere the default keeps the <video> shown: the canvas alone showed short stalls right at speech starts
 * that the <video> did not (Mac Chrome).
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
/** play(): frames rendered before the rest goes on in batches (the first must be ready within about a video frame). */
const PLAY_FIRST_FRAMES = 2;
/** play(): margin (ms) added to the measured time from play() to the first patch, to choose the first frame with a mouth. */
const PLAY_MARGIN_MS = 8;
/**
 * prepareSpeech() warms the models (one run of the sentence model and the first frames on this player, discarded) when
 * they have not run for this long: the first run on a device pays for first-time GPU work, and a GPU left idle may
 * have slowed down; play() then starts on models as fast as on the following lines.
 */
const WARM_AFTER_IDLE_MS = 5000;
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
  /** play(): measured time (ms) from the call to its first patch (sentence model + first frames), the last few kept. */
  private readonly playFirstMs: number[] = [];
  /** performance.now() of the last run of the models (batch); -Infinity before the first. */
  private lastModelRunAt = -Infinity;
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
  /** Cut-out output (create({ matte })): the idle loops play their stacked matte video and this draws the cut-out. */
  private matte: MatteOutput | null = null;
  /** File name of the stacked matte video in each idle loop's pack directory (null: the pack's own idle video). */
  private matteFile: string | null = null;
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
  /** The shown loop's last frame callback: its mediaTime (s) and the callback's timestamp on performance.now()'s clock (null: none yet, or the rAF path). */
  private lastPresentedMedia: { mediaTime: number; at: number } | null = null;
  /** mediaTime (ms) minus the callback's timestamp, of the last frame callbacks (the video plays at real time: a constant). */
  private videoPhase: number[] = [];
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

  /** How wide the mouth opens (core/mouth-open.ts); null = unchanged. Set by the handle (create() / setMouthOpen). */
  mouthOpen: MouthOpen | null = null;
  /** True when the mouth opening can be scaled: every loop's pack has the direction. */
  get mouthOpenAvailable(): boolean { return [...this.loops.values()].every((l) => l.avatar.mouthOpenDir !== null); }

  /** The avatar pack of the loop shown now. */
  get avatar(): Avatar { return this.shownLoop.avatar; }
  /** The idle loop shown now, and the loops there are. */
  get idle(): string { return this.shownLoop.name; }
  get idleLoops(): string[] { return [...this.loops.keys()]; }

  /**
   * url: an avatar (avatar.json, one or more idle loops) or an avatar pack directory (one loop). video: the <video>
   * element for the first loop; the other loops get their own <video> elements, inserted after it.
   */
  static async create(url: string, video: HTMLVideoElement, canvas: HTMLCanvasElement, opts: ModelOptions & { onBytes?: (n: number) => void; idle?: string | null; sync?: PatchSync; matte?: { file: string; output: HTMLCanvasElement } } = {}): Promise<LipSyncPlayer> {
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
    if (opts.matte) {
      // the matte needs the frame and its patch composed on the canvas first: 'canvas' pairing, and a canvas as tall as
      // the stacked video (the patch goes to the top half at the same frame coordinates)
      p.sync = 'canvas';
      p.matteFile = opts.matte.file;
      canvas.height = canvas.height * 2;
      p.matte = new MatteOutput(opts.matte.output, canvas);
      if (standby) standby.dataset['zenMatte'] = opts.matte.file;
      video.dataset['zenMatte'] = opts.matte.file;
    }
    // 'canvas': every picture is the canvas's (idle, speech, crossfades); the <video> elements are only decoded from, never
    // shown (opacity 0 still plays and gets frame callbacks)
    if (p.sync === 'canvas') video.style.opacity = '0';
    initVideo(video);
    loadLoop(video, loops[0]!);
    await video.play().catch(() => undefined);
    p.watchVideo();
    // matte: show a frame even when the video does not play yet (autoplay held until a gesture)
    if (p.matte && video.paused) p.drawStill();
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
    this.drawStill();
  }

  /**
   * Matte output: draws the frame the paused video shows (no frame callbacks come while paused, and before the first one
   * the cut-out canvas would stay empty). Without a matte the <video> itself shows the paused frame.
   */
  private drawStill(): void {
    if (!this.matte) return;
    const v = this.shown.video;
    if (v.readyState < 2) { v.addEventListener('loadeddata', () => { if (v.paused) this.drawStill(); }, { once: true }); return; }
    try {
      this.lastStep = -1;
      this.onVideoFrame(Math.floor(v.currentTime * this.avatar.meta.fps + 1e-3));
      this.matte.render();
    } catch (e) { this.noteDrawError(e); }
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
    this.matte?.destroy();
    (this.lips as { destroy?: () => void }).destroy?.();
  }

  private watchVideo(): void {
    const gen = ++this.watchGen;
    const loop = this.shownLoop;
    const vc = this.shown.video as VideoWithCallback;
    const fps = loop.avatar.meta.fps;
    // presentedFrames counts per element: after a switch the other element's count must not be read as skipped frames
    this.lastPresented = -1;
    this.lastPresentedMedia = null;
    this.videoPhase = [];
    this.lastFrameAt = performance.now();
    if (!this.watchdogTimer) { this.lastCheckAt = performance.now(); this.watchdogTimer = setInterval(() => this.checkWatchdog(), WATCHDOG_CHECK_MS); }
    if (typeof vc.requestVideoFrameCallback === 'function') {
      const onFrame = (now: number, md: { mediaTime: number; presentedFrames?: number; expectedDisplayTime?: number }): void => {
        if (gen !== this.watchGen) return;
        this.lastFrameAt = performance.now();
        // anchored at the callback's own timestamp (the frame's time) when it is on performance.now()'s clock: the callback
        // can run late (a busy main thread on a slower phone), and the time it ran then misplaces the video
        const shownAt = Math.abs(now - this.lastFrameAt) <= CLOCK_MISMATCH_MS ? now : this.lastFrameAt;
        this.lastPresentedMedia = { mediaTime: md.mediaTime, at: shownAt };
        pushPhase(this.videoPhase, md.mediaTime * 1000 - shownAt);
        this.framesSeen = true;
        if (this.clip && this.standby && !this.standby.video.paused) this.standbyPlaying.speechFrames++;
        // the next callback first: an error in this frame's drawing must not end the frame callbacks (the mouth would stop
        // for good while the sound goes on)
        vc.requestVideoFrameCallback!(onFrame);
        try { this.presentFrame(now, md, fps); } catch (e) { this.noteDrawError(e); }
        try { this.matte?.render(); } catch (e) { this.noteDrawError(e); }
      };
      vc.requestVideoFrameCallback(onFrame);
    } else {
      const tick = (): void => {
        if (gen !== this.watchGen) return;
        this.lastFrameAt = performance.now();
        this.framesSeen = true;
        requestAnimationFrame(tick);
        try { this.onVideoFrame(Math.floor(vc.currentTime * fps + 1e-3)); } catch (e) { this.noteDrawError(e); }
        try { this.matte?.render(); } catch (e) { this.noteDrawError(e); }
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
  async speak(pcm: Float32Array, sampleRate: number, onStart?: (t: SpeechStartTime) => void): Promise<void> {
    await this.speakChunks([{ pcm, sampleRate }][Symbol.iterator](), onStart);
  }

  /**
   * Speaks chunks one after the other without a gap: while a chunk plays, the next one is prepared (features, sentence
   * model and first frames, planned for the video frame where the playing chunk ends) and its sound is scheduled to
   * start exactly when the previous one ends on the audio clock. Resolves when the last sound ends (or on stop()).
   * onStart: called when the first chunk's sound is scheduled, with the time it is heard.
   */
  async speakChunks(chunks: Iterator<{ pcm: Float32Array; sampleRate: number }> | AsyncIterator<{ pcm: Float32Array; sampleRate: number }>, onStart?: (t: SpeechStartTime) => void): Promise<void> {
    await this.audioRunning();
    const gen = await this.takeOver();
    const loop = this.shownLoop;
    const walkLen = loop.avatar.meta.walk.length;
    const nextChunk = async (): Promise<SpeechPlan | null> => {
      const r = await chunks.next();
      return r.done || gen !== this.speakGen ? null : this.prepareSpeech(r.value.pcm, r.value.sampleRate, { warm: false });
    };
    // the first chunk: planned `lead` frames ahead of the video
    const plan = await nextChunk();
    if (!plan || gen !== this.speakGen) return;
    const first = await this.prepareFirst(plan, loop);
    if (!first || gen !== this.speakGen) return;
    let clip = this.startClip(first.p, this.ctx.currentTime + this.videoFramesToSound(first.ahead), null);
    onStart?.(this.heardAt(clip.soundAt));
    await this.continueClip(clip, loop, gen, async () => {
      // prepare the next chunk while this one plays: it starts where this one's sound ends
      const n = await nextChunk();
      if (!n || gen !== this.speakGen) return null;
      return this.prepareAt(n, loop, (clip.walkStart + clip.frames) % walkLen, 0, FIRST_FRAMES);
    }, (c) => { clip = c; });
  }

  /**
   * The work on a speech that does not depend on where the idle video is: resampling, log-mel, and the sound as an
   * AudioBuffer of this player's AudioContext. warm (default true): starts the audio device (some browsers take up to
   * about 2 s) and, when the models have not run for WARM_AFTER_IDLE_MS (the first speech of this avatar, or after a
   * pause), runs them once on this speech (discarded) and measures that time for play(): a first play() on the device
   * is then as fast as the following ones. Can run while the avatar speaks.
   */
  async prepareSpeech(pcm: Float32Array, sampleRate: number, opts: { warm?: boolean } = {}): Promise<SpeechPlan> {
    // in parts that let the page draw in between: the canvas shows every picture, and these took 50-70 ms at once
    const pcm16 = sampleRate === MEL_SAMPLE_RATE ? pcm : await resampleYielding(pcm, sampleRate, MEL_SAMPLE_RATE);
    const mel = await logMelYielding(pcm16);
    const buffer = this.ctx.createBuffer(1, pcm.length, sampleRate);
    buffer.copyToChannel(pcm as Float32Array<ArrayBuffer>, 0);
    const speech = speechMask(mel.mel, mel.melFrames, mel.gather, mel.frames);
    const plan: SpeechPlan = { pcm, sampleRate, mel, speech, strength: patchStrength(speech), buffer };
    if (opts.warm ?? true) {
      // the sound must start at once on play(): the audio device is started here (some browsers take up to about 2 s)
      await this.audioRunning();
      if (performance.now() - this.lastModelRunAt > WARM_AFTER_IDLE_MS) {
        const loop = this.shownLoop;
        // twice: the first run on a device pays for first-time GPU work; the second is the time play() will take
        for (let i = 0; i < 2; i++) {
          const t0 = performance.now();
          const p = await this.prepareAt(plan, loop, (this.videoStep() + 2) % loop.avatar.meta.walk.length, 0.5, PLAY_FIRST_FRAMES);
          closePatches(p.patches);
          if (i === 1) this.notePlayFirst(performance.now() - t0);
        }
      }
    }
    return plan;
  }

  /**
   * Starts a prepared speech's sound now (on this player's AudioContext: within its next render quantum) and draws the
   * mouth from the first video frame its patch can be made for: the sentence model and the first frames run after the
   * sound is started, planned for the video frame the measured time from play() to the first patch (+ PLAY_MARGIN_MS)
   * reaches. That frame and every one after it show the mouth of the sound under them. Like speak(), stops any speech
   * playing. Resolves when the sound ends (or on stop()). onStart: called when the sound is scheduled.
   */
  async play(plan: SpeechPlan, onStart?: (t: SpeechStartTime) => void): Promise<void> {
    if (this.ctx.state !== 'running') void this.ctx.resume().catch(() => undefined);
    const gen = await this.takeOver();
    const t0 = performance.now();
    const loop = this.shownLoop;
    const fps = loop.avatar.meta.fps;
    const walkLen = loop.avatar.meta.walk.length;
    const soundAt = this.ctx.currentTime;
    // the video frame shown now and how far into it the video is; frame k from now starts (k - frac) frames after the sound
    const vt = this.presentedVideoTime() * fps;
    const now = Math.floor(vt + 1e-3);
    const frac = Math.max(0, vt - now);
    const k = Math.max(1, Math.ceil(frac + ((this.playLeadMs() / 1000) * fps)));
    const p = this.planClip(plan, (now + k) % walkLen, k - frac);
    const clip = this.startClip(p, soundAt, null);
    onStart?.(this.heardAt(soundAt));
    p.zl = await sentenceLatents(this.lips, loop.avatar, { startStep: p.walkStart, frames: p.frames, melFrames: plan.mel.melFrames, gather: p.gather, mel: plan.mel.mel }, this.mouthOpen);
    if (gen !== this.speakGen) return;
    // frames already on screen (or due before a patch could be drawn) are skipped: the mouth joins on the next one
    p.f = Math.min(p.frames, Math.max(0, Math.ceil(this.clipFrame(clip) + 0.5)));
    await this.batch(p, loop, Math.min(PLAY_FIRST_FRAMES, p.frames - p.f));
    this.notePlayFirst(performance.now() - t0, p.f > 0);
    await this.continueClip(clip, loop, gen, async () => null, () => undefined);
  }

  /** speak() / play(): drops a switch still preparing (one already crossfading ends within FADE_MS), stops any speech. */
  private async takeOver(): Promise<number> {
    this.cancelSwitch?.();
    while (this.switching) await new Promise((r) => setTimeout(r, 20));
    this.stop();
    return ++this.speakGen;
  }

  /** Renders a started clip's frames while it plays, starts the chunks next() gives back to back, resolves at the last sound's end. */
  private async continueClip(clip: Clip, loop: Loop, gen: number, next: () => Promise<Prepared | null>, onClip: (c: Clip) => void): Promise<void> {
    let lastEnded: Promise<void> = clip.ended;
    for (;;) {
      const nextP = next();
      nextP.catch(() => undefined);
      const [, nx] = await Promise.all([this.renderRest(clip, loop), nextP]);
      if (!nx || gen !== this.speakGen) break;
      clip = this.startClip(nx, Math.max(this.ctx.currentTime, clip.soundAt + clip.duration), clip);
      onClip(clip);
      lastEnded = clip.ended;
    }
    await lastEnded;
    if (gen === this.speakGen && this.clip === clip) this.stop();
  }

  /** play(): the time (ms) planned from the call to the first patch: the slowest of the last measured + PLAY_MARGIN_MS. */
  private playLeadMs(): number {
    return (this.playFirstMs.length ? Math.max(...this.playFirstMs) : 40) + PLAY_MARGIN_MS;
  }

  /** A measured time from play() (or its warm-up) to the first patch; late: the first planned frame was missed. */
  private notePlayFirst(ms: number, late = false): void {
    this.playFirstMs.push(late ? ms * 1.5 : ms);
    if (this.playFirstMs.length > 5) this.playFirstMs.shift();
  }

  /**
   * Video time (s) of the picture on screen now, from the recent frame callbacks (the frames the mouth is drawn on):
   * each gives a frame's mediaTime and its timestamp, so now + their median phase; else the element's currentTime.
   * Chrome's currentTime ran about 0.4 video frames ahead of the presented frame (Mac), which planned play()'s mouth that
   * much early. The time since the last callback alone placed it wrong by up to 2-3 frames where the callbacks come
   * late (a busy main thread on a slower phone): play()'s mouth was then off its sound by that much for the whole line.
   */
  private presentedVideoTime(): number {
    const m = this.lastPresentedMedia, v = this.shown.video;
    if (!m || v.paused || (v.playbackRate || 1) !== 1) return v.currentTime;
    const now = performance.now();
    if (now - m.at > 0.3 * 1000) return v.currentTime;
    // the median of the recent callbacks' phase: one callback that came late (a busy main thread) moves it by nothing
    const phase = medianPhase(this.videoPhase);
    return phase === null ? m.mediaTime + (now - m.at) / 1000 : (now + phase) / 1000;
  }

  /** Seconds from now until the video shows the frame `ahead` frames on (the video and the audio clock both run at real time). */
  private videoFramesToSound(ahead: number): number {
    const fps = this.avatar.meta.fps;
    const vt = this.shown.video.currentTime * fps;
    const frac = vt - Math.floor(vt + 1e-3);
    return Math.max(0, ahead - frac) / fps;
  }

  /**
   * This device's audio output delay now (ms): from a sound started at the AudioContext's current time to it being heard,
   * measured from the context's output timestamp where the browser gives one, else its reported output and base latency.
   */
  outputDelayMs(): number {
    return this.heardAt(this.ctx.currentTime).outputDelayMs;
  }

  /**
   * When the sound starting at AudioContext time t is heard: t itself, and on performance.now()'s clock (the context's
   * output timestamp where the browser gives one, which counts the output latency; else now + the time left + the
   * context's reported output and base latency). The output delay is never negative: a sound cannot be heard before it
   * is due. iOS Safari's output timestamp put it 0-2 ms before (the timestamp is not ahead of the context time there);
   * such a timestamp is not used and the reported latencies are.
   */
  private heardAt(t: number): SpeechStartTime {
    const now = performance.now();
    const due = now + (t - this.ctx.currentTime) * 1000;
    const ts = typeof this.ctx.getOutputTimestamp === 'function' ? this.ctx.getOutputTimestamp() : null;
    // the output timestamp: the context time being heard now (performanceTime), so t is heard (t - contextTime) later
    const byTimestamp = ts && ts.contextTime !== undefined && ts.performanceTime !== undefined && ts.performanceTime > 0
      ? ts.performanceTime + (t - ts.contextTime) * 1000 - due : null;
    const byLatency = ((this.ctx.outputLatency || 0) + (this.ctx.baseLatency || 0)) * 1000;
    const delay = outputDelay(byTimestamp, byLatency);
    return { audioTime: t, performanceTime: due + delay, outputDelayMs: delay };
  }

  /** The first chunk: features and sentence latents planned ahead of the video, its first frames rendered. */
  private async prepareFirst(plan: SpeechPlan, loop: Loop): Promise<{ p: Prepared; ahead: number } | null> {
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
      p = await this.prepareAt(plan, loop, walkStart, 0, FIRST_FRAMES);
      const took = Math.ceil(((performance.now() - t0) / 1000) * fps);
      this.lead = Math.max(MIN_LEAD, took + 3);
      // frames the video still has to play until it shows walkStart (about a lap when it has just passed it)
      const ahead = (((walkStart - this.videoStep()) % walkLen) + walkLen) % walkLen;
      if (ahead >= 1 && ahead <= lead + 2) return { p, ahead };
      if (attempt >= 2) return { p, ahead: Math.max(1, Math.min(ahead, lead)) };
      lead = this.lead;
    }
  }

  /**
   * A clip of plan whose frame 0 is shown at walk position walkStart, `shift` video frames after the sound starts: every
   * frame takes the mel row and the speech mask of the sound under it. No latents yet (zl empty), no patches.
   */
  private planClip(plan: SpeechPlan, walkStart: number, shift: number): Prepared & { gather: number[] } {
    const m = plan.mel;
    const { frames, gather } = shift === 0 ? m : melGather(m.melFrames, shift);
    const speech = shift === 0 ? plan.speech : speechMask(m.mel, m.melFrames, gather, frames);
    const strength = shift === 0 ? plan.strength : patchStrength(speech);
    return { plan, walkStart, frames, gather, zl: new Uint16Array(0), speech, strength, patches: new Map(), f: 0, shift };
  }

  /** Sentence latents and the first n frames of a clip whose frame 0 is shown at walk position walkStart (planClip). */
  private async prepareAt(plan: SpeechPlan, loop: Loop, walkStart: number, shift: number, n: number): Promise<Prepared> {
    const p = this.planClip(plan, walkStart, shift);
    p.zl = await sentenceLatents(this.lips, loop.avatar, { startStep: walkStart, frames: p.frames, melFrames: plan.mel.melFrames, gather: p.gather, mel: plan.mel.mel }, this.mouthOpen);
    await this.batch(p, loop, Math.min(n, p.frames));
    return p;
  }

  private async batch(p: Prepared, loop: Loop, n: number): Promise<void> {
    if (n <= 0) return;
    const avatar = loop.avatar;
    const us = Array.from({ length: n }, (_, i) => avatar.walk(p.walkStart + p.f + i));
    const tb = performance.now();
    this.lastModelRunAt = tb;
    // the model's change is kept on every frame with a strength (speech and the fade after it); the others are zeroed
    const drawn = Array.from(p.strength, (k) => k > 0);
    const [pred, srcs] = await Promise.all([inferFrames(this.lips, avatar, us, p.zl, p.f, n, drawn), Promise.all(us.map((u) => avatar.frame(u)))]);
    this.timing.inferMs += performance.now() - tb;
    this.timing.frames += n;
    // 'delta' packs: the change is scaled by the strength here; 'absolute' packs: the strength weights the painted mouth
    if (!loop.comp.absolute) applyStrength(pred, p.strength, p.f, n);
    // frames with no patch hold null (nothing drawn): a shared empty bitmap here was closed with the first clip dropped,
    // and every later silent frame then threw on drawImage (InvalidStateError), which ended the frame callbacks and the
    // mouth stopped while the sound went on
    // one patch at a time with the page let in between (each is 1.5-5 ms of canvas and pixel work on the main thread)
    for (let i = 0; i < srcs.length; i++) {
      const k = p.strength[p.f + i]!;
      p.patches.set(p.f + i, k > 0 ? loop.comp.frame(srcs[i]!, pred, i, k) : null);
      await yieldToPage();
    }
    p.f += n;
  }

  /** Starts a prepared chunk's sound at AudioContext time soundAt (a following chunk: prev, still playing until then). Its patches are drawn from then on. */
  private startClip(p: Prepared, soundAt: number, prev: Clip | null): Clip {
    const source = this.ctx.createBufferSource();
    source.buffer = p.plan.buffer;
    source.connect(this.ctx.destination);
    const clip: Clip = { soundAt, duration: p.plan.buffer.duration, frames: p.frames, walkStart: p.walkStart, patches: p.patches, source, prep: p,
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
          if (this.pairing !== 'canvas') v.style.opacity = String(p);
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

  /** Frame of the clip the sound is at now (negative before frame 0: before the sound, or within its shift). */
  private clipFrame(c: Clip): number {
    return (this.ctx.currentTime - c.soundAt) * this.avatar.meta.fps - c.prep.shift;
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

/**
 * A device's audio output delay (ms) from the output timestamp's measure (null: none) and the context's reported output
 * + base latency: the timestamp's when it is positive, else the reported latency (0 when the browser reports none).
 */
export function outputDelay(byTimestamp: number | null, byLatency: number): number {
  if (byTimestamp !== null && Number.isFinite(byTimestamp) && byTimestamp > 0) return byTimestamp;
  return Number.isFinite(byLatency) && byLatency > 0 ? byLatency : 0;
}

/** Frame callbacks kept for the video's phase (about 0.6 s). */
const PHASE_WINDOW = 15;

/** Keeps the last PHASE_WINDOW phases (ms); a jump of more than PHASE_JUMP_MS (the loop wrapped, a seek) starts anew. */
function pushPhase(a: number[], v: number): void {
  if (a.length && Math.abs(v - a[a.length - 1]!) > PHASE_JUMP_MS) a.length = 0;
  a.push(v);
  if (a.length > PHASE_WINDOW) a.shift();
}

/** A phase change this large is a wrap or a seek, not a late callback (a late one is off by a fraction of a second at most). */
const PHASE_JUMP_MS = 400;

/**
 * The video's phase (mediaTime ms minus callback timestamp) from the recent callbacks: their median, which a few late callbacks
 * do not move. null with fewer than 3.
 */
export function medianPhase(a: readonly number[]): number | null {
  if (a.length < 3) return null;
  const s = [...a].sort((x, y) => x - y);
  return s[s.length >> 1]!;
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
  // create({ matte }): the loop's stacked matte video (same frames and timing as its idle video) instead of the idle video
  const matte = v.dataset?.['zenMatte'];
  const srcs = matte ? [{ src: matte, type: 'video/webm; codecs="vp9"' }] : l.avatar.meta.video;
  v.replaceChildren(...srcs.map((s) => Object.assign(document.createElement('source'), { src: `${l.avatar.base}/${s.src}`, type: s.type })));
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
export { patchStrength };
