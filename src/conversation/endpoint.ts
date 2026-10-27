// End of a spoken turn from the microphone's frames, without a model: the level goes over the threshold, and the
// turn ends after this much quiet. Built in so the SDK runs without a model; an app can drive the same two decisions
// from a VAD / turn-end model instead (`isSpeech`, `end()`).

export interface EndpointOptions {
  /** Frame length in ms (the microphone's frames: 32 ms at 16 kHz, 512 samples). */
  frameMs?: number;
  /** Quiet for this long ends the turn (ms, default 700). */
  endSilenceMs?: number;
  /** Frames this loud (RMS, dBFS) count as speech (default -45). */
  thresholdDbfs?: number;
  /** Speech starts after this many loud frames in a row (default 2). */
  startFrames?: number;
}

export type EndpointEvent = { type: 'speech_start' } | { type: 'speech_end'; speakingMs: number };

/** Level meter + hangover: push() one frame, get the events it causes. */
export class EndPointer {
  private readonly frameMs: number;
  private readonly endSilenceMs: number;
  private readonly threshold: number;
  private readonly startFrames: number;
  private speaking = false;
  private run = 0;
  private quietMs = 0;
  private speakingMs = 0;
  /** Last frame's level (dBFS), for a level meter. */
  lastDbfs = -120;

  constructor(o: EndpointOptions = {}) {
    this.frameMs = o.frameMs ?? 32;
    this.endSilenceMs = o.endSilenceMs ?? 700;
    this.threshold = o.thresholdDbfs ?? -45;
    this.startFrames = o.startFrames ?? 2;
  }

  /** Level of one frame in dBFS (silence is -120, not -inf). */
  static dbfs(frame: Float32Array): number {
    let sq = 0;
    for (let i = 0; i < frame.length; i++) sq += frame[i]! * frame[i]!;
    const rms = Math.sqrt(sq / Math.max(1, frame.length));
    return rms > 0 ? Math.max(-120, 20 * Math.log10(rms)) : -120;
  }

  push(frame: Float32Array): EndpointEvent[] {
    const db = EndPointer.dbfs(frame);
    this.lastDbfs = db;
    return this.pushDecision(db >= this.threshold);
  }

  /** The same hangover, on a decision made elsewhere (a VAD model: speech / not speech for this frame). */
  pushDecision(speech: boolean): EndpointEvent[] {
    const events: EndpointEvent[] = [];
    if (!this.speaking) {
      this.run = speech ? this.run + 1 : 0;
      if (this.run >= this.startFrames) {
        this.speaking = true;
        this.run = 0;
        this.quietMs = 0;
        // the frames that started the speech count toward its length
        this.speakingMs = this.startFrames * this.frameMs;
        events.push({ type: 'speech_start' });
      }
      return events;
    }
    this.speakingMs += this.frameMs;
    if (speech) {
      this.quietMs = 0;
    } else {
      this.quietMs += this.frameMs;
      if (this.quietMs >= this.endSilenceMs) {
        this.speaking = false;
        const ms = Math.round(this.speakingMs - this.quietMs);
        this.quietMs = 0;
        events.push({ type: 'speech_end', speakingMs: Math.max(0, ms) });
      }
    }
    return events;
  }

  /** Ends the turn now (a button, a cap): an event when something was being said. */
  endNow(): EndpointEvent[] {
    if (!this.speaking) return [];
    this.speaking = false;
    this.quietMs = 0;
    return [{ type: 'speech_end', speakingMs: Math.round(this.speakingMs) }];
  }

  get inSpeech(): boolean { return this.speaking; }
}
