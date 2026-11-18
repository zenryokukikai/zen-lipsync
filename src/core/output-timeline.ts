// The timeline of a frames() output (Node and browser renderers): frames at the avatar's 25 fps, speech chunks laid on
// the output audio end to end (each starts where the previous one's audio ended, or at the current frame after idle
// frames), every frame carrying its share of the audio (silence where nothing sounds).
import { OptionsError } from './errors.js';
import { resample } from './resample.js';
import type { SpeechAudio } from './types.js';

const FPS = 25;
/** Audio rate of an output whose first frames come before any speech told it (no audioSampleRate given). */
const DEFAULT_RATE = 48000;

export interface TimelineFrame {
  /** Frame number in this output (25 fps). */
  index: number;
  timeMs: number;
  /** This frame's audio (mono float32 at sampleRate; silence for idle frames). */
  audio: Float32Array;
  sampleRate: number;
}

export class OutputTimeline {
  private rate: number;
  private index = 0;
  /** Speech segments placed on the output timeline (samples at rate); idle time is silence. */
  private readonly segs: { start: number; data: Float32Array }[] = [];
  /** Samples given to frames so far, and the end of the last speech segment. */
  private outSamples = 0;
  private audioEnd = 0;

  /** rate: the output's audio rate (audioSampleRate); 0 = the first chunk's rate. */
  constructor(rate = 0) { this.rate = rate; }

  get frameIndex(): number { return this.index; }
  get sampleRate(): number { return this.rate; }

  /** The next frame's index, time and audio (the output audio rate is fixed from here on). */
  next(): TimelineFrame {
    if (!this.rate) this.rate = DEFAULT_RATE;
    const rate = this.rate;
    const end = Math.round(((this.index + 1) * rate) / FPS);
    const a = new Float32Array(Math.max(0, end - this.outSamples));
    for (const s of this.segs) {
      const from = Math.max(this.outSamples, s.start), to = Math.min(end, s.start + s.data.length);
      for (let j = from; j < to; j++) a[j - this.outSamples] = s.data[j - s.start]!;
    }
    while (this.segs.length && this.segs[0]!.start + this.segs[0]!.data.length <= end) this.segs.shift();
    this.outSamples = end;
    const f: TimelineFrame = { index: this.index, timeMs: (this.index * 1000) / FPS, audio: a, sampleRate: rate };
    this.index++;
    return f;
  }

  /**
   * Places a speech chunk on the audio (right after the previous chunk's audio, or at the next frame when idle frames
   * came in between) and returns how many frames it takes: the whole frames up to the end of its audio, so many chunks
   * do not drift and the audio has no gaps.
   */
  addChunk(pcm: Float32Array, sampleRate: number): number {
    if (!this.rate) this.rate = sampleRate;
    const rate = this.rate;
    const audio = sampleRate === rate ? pcm : resample(pcm, sampleRate, rate);
    const start = Math.max(this.audioEnd, Math.round((this.index * rate) / FPS));
    this.segs.push({ start, data: audio });
    this.audioEnd = start + audio.length;
    // whole frames only: a part frame at the end goes with the next chunk's first frame (or the frames after the speech),
    // so a following chunk's audio starts right where this one's ends, with no silence put in between
    const n = Math.max(0, Math.floor((this.audioEnd * FPS) / rate + 1e-9) - this.index);
    return n;
  }

  /**
   * Frames after the speech: the rest of the speech audio not yet given to frames (its last part frame included, so N s
   * of speech take ceil(N x 25) frames in all), then idleAfterMs of idle frames.
   */
  framesAfter(idleAfterMs: number): number {
    if (!this.rate) this.rate = DEFAULT_RATE;
    return Math.max(0, Math.ceil(((this.audioEnd - this.outSamples) * FPS) / this.rate)) + Math.round((idleAfterMs * FPS) / 1000);
  }
}

/** Idle frames for a duration (ms) at 25 fps. */
export function framesOfMs(ms: number): number { return Math.round((ms * FPS) / 1000); }

/** frames() input: one clip, or chunks (an async or plain iterable), as one stream of chunks. */
export async function* speechChunks(input: SpeechAudio | AsyncIterable<SpeechAudio> | Iterable<SpeechAudio>): AsyncGenerator<SpeechAudio> {
  if (input && (input as SpeechAudio).pcm instanceof Float32Array) { yield input as SpeechAudio; return; }
  if (input && typeof (input as AsyncIterable<SpeechAudio>)[Symbol.asyncIterator] === 'function') { yield* input as AsyncIterable<SpeechAudio>; return; }
  if (input && typeof (input as Iterable<SpeechAudio>)[Symbol.iterator] === 'function') { yield* input as Iterable<SpeechAudio>; return; }
  throw new OptionsError('audio must be { pcm: Float32Array, sampleRate } or an (async) iterable of them');
}
