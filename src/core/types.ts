// The SDK's public API shape, shared by the browser entry (zen-lipsync, WebGPU) and the Node entry (zen-lipsync/node, CPU).

/** Frame model weights on WebGPU: 'i8' (default, int8 restored to float32 once at load), 'f32', or 'i8q' (int8 kept, restored on every use). */
import type { MouthOpen, MouthOpenMode } from './mouth-open.js';
export type LipWeights = 'f32' | 'i8' | 'i8q';

/** Options of create(). */
export interface CreateOptions {
  /**
   * The avatar: an avatar.json URL or path (one or more idle loops) or an avatar pack directory (one loop).
   * Example: 'https://example.com/avatars/marine/avatar.json'.
   */
  avatar: string;
  /**
   * Base URL (or directory on Node) that a relative `avatar` is resolved against, e.g. where the zen-lipsync-avatars
   * files are served: { assetsBaseUrl: 'https://cdn.example.com/zen-lipsync-avatars/', avatar: 'avatars/marine/avatar.json' }.
   * Default: none (`avatar` is used as given, relative to the page / the working directory).
   */
  assetsBaseUrl?: string;
  /** Frame model weights (WebGPU only; the Node CPU path always uses int8). Default 'i8'. */
  weights?: LipWeights;
  /** Idle loop shown first (a key of avatar.json "sources"); default: the avatar's "default". */
  idle?: string;
  /** Called with the bytes of every file loaded (models, features, frames). */
  onProgress?: (p: LoadProgress) => void;
  /**
   * How wide the mouth opens: { scale, mode? }. scale in (0, 2], 1 = as trained (default); below 1 the mouth opens less
   * (a small child). mode 'opening' (default) scales only the opening (lip shape kept, needs the pack's
   * mouth_open_dir.f32), 'all' every mouth move. The modes an avatar can do: handle.mouthOpenModes. Also setMouthOpen().
   */
  mouthOpen?: MouthOpen;
}

export interface LoadProgress {
  /** Bytes loaded so far. */
  loadedBytes: number;
  /** The file just loaded (relative to the avatar). */
  file: string;
}

/** Speech audio: mono samples in [-1, 1] at any sample rate. */
export interface SpeechAudio {
  pcm: Float32Array;
  sampleRate: number;
}

export interface FrameTiming {
  /** Frames rendered so far and the inference time per frame (ms). */
  frames: number;
  inferMsPerFrame: number;
}

/** An utterance made ready by AvatarHandle.prepare(): play it once with play(), or dispose() it when not needed. */
export interface PreparedSpeech {
  /** Video frames (25 fps) and length (ms) of the sound. */
  readonly frames: number;
  readonly durationMs: number;
  /** False once played or disposed. */
  readonly playable: boolean;
  /**
   * This device's audio output delay (ms) measured when prepare() finished: play() starts the sound at once on the
   * AudioContext, and it is heard about this much later.
   */
  readonly outputDelayMs: number;
  /** Drops the prepared data without playing it (nothing to do after play()). */
  dispose(): void;
}

/** Events of an AvatarHandle. */
export interface AvatarEvents {
  /** Loading finished; the idle loop plays. */
  ready: { backend: 'webgpu' | 'cpu'; loadMs: number; loadedBytes: number };
  /**
   * The sound of an utterance started / ended (ended also on stop()). speechstart comes when the sound is scheduled, with
   * the time it is heard: `audioTime` on the avatar's AudioContext clock and `performanceTime` on performance.now()'s,
   * which includes `outputDelayMs`, this device's audio output delay measured then (the AudioContext's output timestamp,
   * else its reported latencies). They are missing when the utterance ended before its sound was scheduled (stop(), an
   * error); speechend follows then at once.
   */
  speechstart: { frames: number; audioTime?: number; performanceTime?: number; outputDelayMs?: number };
  speechend: { interrupted: boolean };
  /** The idle loop changed (setIdle). */
  idlechange: { from: string; to: string };
  /** Any error after create() resolved (load errors reject create() instead). */
  error: { error: Error };
}

export type AvatarEventName = keyof AvatarEvents;

/** A loaded avatar: speaks audio with lip-sync and switches its idle loops. */
export interface AvatarHandle {
  /** Backend the models run on: 'webgpu' in the browser, 'cpu' on Node. */
  readonly backend: 'webgpu' | 'cpu';
  /** The idle loop shown now, and the loops the avatar has. */
  readonly idle: string;
  readonly idleLoops: readonly string[];
  /** True while an utterance is prepared or plays. */
  readonly speaking: boolean;
  /** Speaks one clip; resolves when its sound ends (or it is stopped). */
  speak(audio: SpeechAudio): Promise<void>;
  /** Speaks audio that arrives in chunks (e.g. streamed speech synthesis), each spoken as soon as it is ready, in order. */
  speakStream(chunks: AsyncIterable<SpeechAudio>): Promise<void>;
  /**
   * Does the work of an utterance ahead of time (audio features, the sound as audio data, the audio device started, and a
   * warm-up run of the models on this device when they have not run for a while, e.g. this avatar's first utterance), so
   * that play() starts its sound at once. Can be called while the avatar speaks. Browser only.
   */
  prepare(audio: SpeechAudio): Promise<PreparedSpeech>;
  /**
   * Starts a prepared utterance's sound now (within one video frame, a near-constant delay) and moves the mouth with it
   * from the next video frame. Like speak(), it stops what the avatar is saying. Resolves when the sound ends (or it is
   * stopped). Each PreparedSpeech plays once, on the avatar that prepared it.
   */
  play(prepared: PreparedSpeech): Promise<void>;
  /** Switches the idle loop while idle (a short crossfade at a similar frame); false when nothing was done. */
  setIdle(name: string): Promise<boolean>;
  /** The mouthOpen modes this avatar can do (every pack: 'all'; with mouth_open_dir.f32 in every loop also 'opening'). */
  readonly mouthOpenModes: readonly MouthOpenMode[];
  /** How wide the mouth opens from the next utterance on (create() option mouthOpen); null = as trained. */
  setMouthOpen(mouthOpen: MouthOpen | null): void;
  /** Subscribes to an event; returns the function that unsubscribes. */
  on<K extends AvatarEventName>(event: K, listener: (e: AvatarEvents[K]) => void): () => void;
  /** Stops the sound and the lip-sync now (the idle loop keeps playing). */
  stop(): void;
  /** Pauses / resumes the idle loop and any sound. */
  pause(): void;
  resume(): void;
  /** Inference time per frame since load. */
  frameTiming(): FrameTiming;
  /** Releases the models, the elements it added and the audio. The handle cannot be used afterwards. */
  destroy(): void;
}
