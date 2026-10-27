// Long-form rendering from audio: an audio file (WAV) in, one video file out. The file is read in bounded chunks and
// handed to the avatar's render() path, which prepares the next chunk while drawing the current one, so a 30-second or
// 3-minute recording keeps flat memory (one chunk of samples at a time, not the whole file).
import { OptionsError } from '../core/errors.js';
import type { SpeechAudio } from '../core/types.js';
import type { NodeAvatar, RenderProgress } from './avatar.js';
import { openWav, type WavReader } from './wav.js';
import type { IdleCache } from './idle-video.js';
import type { NodeBackend } from './frame-model.js';
import { create } from './index.js';

/** Chunks of the WAV's samples (mono, the file's own sample rate); the reader is closed at the end or on abort. */
export async function* wavAudio(reader: WavReader, opts: { chunkSeconds?: number; signal?: AbortSignal } = {}): AsyncGenerator<SpeechAudio> {
  const chunkSeconds = opts.chunkSeconds ?? 8;
  if (!(chunkSeconds > 0)) throw new OptionsError('chunkSeconds must be > 0');
  const per = Math.max(1, Math.round(chunkSeconds * reader.sampleRate));
  try {
    for (let at = 0; at < reader.frames; at += per) {
      opts.signal?.throwIfAborted();
      const pcm = await reader.read(at, per);
      if (pcm.length === 0) break;
      yield { pcm, sampleRate: reader.sampleRate };
    }
  } finally {
    await reader.close();
  }
}

export interface RenderAudioOptions {
  /** Output file (.mp4: H.264 + AAC; .webm: VP9 + Opus). */
  out: string;
  /** Encoder quality: lower is better (default 20 for H.264, 32 for VP9). */
  crf?: number;
  /** Idle frames before the speech (ms, default 0) and after it (default 0). */
  idleAfterMs?: number;
  idleBeforeMs?: number;
  /** Aborting stops the render and removes the partial output file. */
  signal?: AbortSignal;
  /** Called about once a second of video (and once at the end). */
  onProgress?: (p: RenderAudioProgress) => void;
}

export interface RenderAudioProgress extends RenderProgress {
  /** 0..1 when the total is known (an audio file), else null. */
  ratio: number | null;
  /** Seconds of video written so far, and the rate they were made at. */
  wallMs: number;
  fps: number;
}

export interface RenderAudioResult {
  file: string;
  /** Frames written (25 fps). */
  frames: number;
  /** Length of the video (ms). */
  durationMs: number;
  /** Length of the speech (ms), when the input's length is known (an audio file). */
  audioMs: number | null;
  /** Wall time of the whole render (ms), and the rate it ran at. */
  wallMs: number;
  fps: number;
  /** wallMs / durationMs: below 1 is faster than real time. */
  rtf: number;
}

/** Renders speech chunks (or one clip) to a video file, with progress and an AbortSignal. */
export async function renderAudio(avatar: NodeAvatar, input: SpeechAudio | AsyncIterable<SpeechAudio> | Iterable<SpeechAudio>,
  opts: RenderAudioOptions & { audioMs?: number; expectedFrames?: number }): Promise<RenderAudioResult> {
  if (typeof opts.out !== 'string' || opts.out === '') throw new OptionsError('renderAudio(): { out: "file.mp4" | "file.webm" } is required');
  const t0 = performance.now();
  const total = opts.expectedFrames ?? null;
  const onProgress = opts.onProgress;
  const wrapped = onProgress
    ? (p: RenderProgress): void => {
      const wallMs = performance.now() - t0;
      const fps = p.timeMs > 0 ? (p.frames * 1000) / wallMs : 0;
      onProgress({ ...p, ratio: p.totalFrames ? Math.min(1, p.frames / p.totalFrames) : null, wallMs, fps });
    }
    : undefined;
  const r = await avatar.render(input, {
    out: opts.out,
    ...(opts.crf !== undefined ? { crf: opts.crf } : {}),
    ...(opts.idleBeforeMs !== undefined ? { idleBeforeMs: opts.idleBeforeMs } : {}),
    ...(opts.idleAfterMs !== undefined ? { idleAfterMs: opts.idleAfterMs } : {}),
    ...(opts.signal !== undefined ? { signal: opts.signal } : {}),
    ...(wrapped !== undefined ? { onProgress: wrapped } : {}),
    ...(total !== null ? { expectedFrames: total } : {}),
  });
  const wallMs = performance.now() - t0;
  return {
    file: r.file,
    frames: r.frames,
    durationMs: r.durationMs,
    audioMs: opts.audioMs ?? null,
    wallMs,
    fps: wallMs > 0 ? (r.frames * 1000) / wallMs : 0,
    rtf: r.durationMs > 0 ? wallMs / r.durationMs : 0,
  };
}

export interface RenderAudioFileOptions extends RenderAudioOptions {
  /** The avatar to show: a name under assetsPath ("marine"), or an avatar.json path or URL (as create()). */
  avatar: string;
  /** Directory of the avatar files (a clone of zen-lipsync-avatars). */
  assetsPath?: string;
  /** Base URL of the avatar files, instead of assetsPath. */
  assetsBaseUrl?: string;
  /** 'auto' (default): the native frame model when it loads, else WebAssembly. */
  backend?: NodeBackend;
  /** Idle loop shown first (default: the avatar's default). */
  idle?: string;
  /** The ffmpeg program (default: ZEN_LIPSYNC_FFMPEG, else "ffmpeg" on PATH). */
  ffmpegPath?: string;
  /** 'stream' (default) or 'memory' (see create()). */
  idleCache?: IdleCache;
  /** The WAV file to speak (mono or stereo; 8/16/24/32-bit PCM or 32/64-bit float). */
  audio: string;
  /** Length of each chunk given to the avatar (seconds, default 8; smaller uses less memory, larger is slightly faster). */
  chunkSeconds?: number;
}

/**
 * Audio file -> video file in one call: loads the avatar, streams the WAV through it and writes the video.
 * For a long recording this is the entry to use (the SDK creates and releases the avatar for you); call create()
 * and renderAudio() yourself to reuse one avatar for several videos.
 */
export async function renderAudioFile(opts: RenderAudioFileOptions): Promise<RenderAudioResult> {
  if (typeof opts?.audio !== 'string' || opts.audio === '') throw new OptionsError('renderAudioFile(): { audio: "in.wav" } is required');
  if (!/\.wav$/i.test(opts.audio)) throw new OptionsError(`renderAudioFile(): "${opts.audio}" is not a .wav file. Convert it first: ffmpeg -i in.mp3 -ac 1 -ar 48000 out.wav`);
  const reader = await openWav(opts.audio);
  const before = Math.round(((opts.idleBeforeMs ?? 0) * 25) / 1000);
  const after = Math.round(((opts.idleAfterMs ?? 0) * 25) / 1000);
  const expectedFrames = Math.round((reader.durationSec * 25)) + before + after;
  const { audio, chunkSeconds, out, crf, idleBeforeMs, idleAfterMs, signal, onProgress, ...avatarOpts } = opts;
  void audio;
  const avatar = await create(avatarOpts);
  try {
    return await renderAudio(avatar, wavAudio(reader, { ...(chunkSeconds !== undefined ? { chunkSeconds } : {}), ...(opts.signal ? { signal: opts.signal } : {}) }), {
      out: opts.out,
      ...(opts.crf !== undefined ? { crf: opts.crf } : {}),
      ...(opts.idleBeforeMs !== undefined ? { idleBeforeMs: opts.idleBeforeMs } : {}),
      ...(opts.idleAfterMs !== undefined ? { idleAfterMs: opts.idleAfterMs } : {}),
      ...(opts.signal !== undefined ? { signal: opts.signal } : {}),
      ...(opts.onProgress !== undefined ? { onProgress: opts.onProgress } : {}),
      audioMs: reader.durationSec * 1000,
      expectedFrames,
    });
  } finally {
    avatar.destroy();
    await reader.close().catch(() => undefined);
  }
}
