// The SDK's public API shape, shared by the browser entry (zen-lipsync, WebGPU) and the Node entry (zen-lipsync/node, CPU).

/** Frame model weights on WebGPU: 'i8' (default, int8 restored to float32 once at load), 'f32', or 'i8q' (int8 kept, restored on every use). */
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

/** Events of an AvatarHandle. */
export interface AvatarEvents {
  /** Loading finished; the idle loop plays. */
  ready: { backend: 'webgpu' | 'cpu'; loadMs: number; loadedBytes: number };
  /** The sound of an utterance started / ended (ended also on stop()). */
  speechstart: { frames: number };
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
  /** Switches the idle loop while idle (a short crossfade at a similar frame); false when nothing was done. */
  setIdle(name: string): Promise<boolean>;
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
