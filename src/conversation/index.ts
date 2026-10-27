// A spoken dialogue on top of an avatar: the user's turn in, the reply's sound out.
//
// The SDK supplies the parts that must not drift apart — when a turn starts and ends, when to stop the sound
// (barge-in), when to switch the idle loop on emotion, and when each moment happened — and the app supplies the
// models: `reply()` (any LLM) and `synthesize()` (any TTS; ZEN-TTS streams, so the first sentence sounds while the
// next one is made). Nothing here is tied to a provider, a framework or a DOM: the browser entry adds the microphone
// (see ./mic.ts), and `sendText()` drives the same machine without one.
import { OptionsError } from '../core/errors.js';
import { Emitter } from '../core/events.js';
import type { AvatarHandle, SpeechAudio } from '../core/types.js';
import { initialTurnData, turnStep, type TurnAction, type TurnData, type TurnEvent, type TurnState } from './machine.js';
import { PushQueue } from './queue.js';

export * from './machine.js';
export { PushQueue } from './queue.js';
export { EndPointer, type EndpointEvent, type EndpointOptions } from './endpoint.js';
export { openMic, type Mic, type MicOptions } from './mic.js';

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ReplyContext {
  /** The messages so far, ending with the user's line. */
  messages: readonly ConversationMessage[];
  /** The utterance number this reply answers. */
  utt: number;
  signal: AbortSignal;
}

export interface SynthesisContext {
  /** The piece to say now (one text piece of the reply, as `reply()` yielded it). */
  piece: string;
  /** The reply's text so far, piece included (for a TTS that wants the sentence's context). */
  text: string;
  index: number;
  utt: number;
  signal: AbortSignal;
  /** 'reply' for the answer, 'aizuchi' for a short backchannel. */
  kind: 'reply' | 'aizuchi';
}

/** The emotion of a reply as the app knows it: only the angry weight moves the idle loop (the packs have normal + angry). */
export interface EmotionReading {
  angry: number;
}

export interface ConversationOptions {
  /** The avatar to speak through (the browser or Node create()). */
  avatar: Pick<AvatarHandle, 'speakStream' | 'stop' | 'setIdle' | 'idle' | 'idleLoops'>;
  /**
   * The reply: text pieces as the model produces them (yield a piece as soon as it is ready; the first one is the
   * latency that matters). The conversation aborts the signal on barge-in and stop().
   */
  reply(ctx: ReplyContext): AsyncIterable<string>;
  /** Speech for one piece of text. Yield audio as it is made (ZEN-TTS streams), not one clip per sentence. */
  synthesize(ctx: SynthesisContext): AsyncIterable<SpeechAudio>;
  /** Backchannel played right after the user stops speaking (optional). */
  aizuchi?: {
    /** Text to say, or null for none this turn. */
    pick(ctx: { messages: readonly ConversationMessage[]; utt: number; signal: AbortSignal }): string | null | Promise<string | null>;
  };
  /** The emotion of a reply's text, when the idle loop should follow it (optional). */
  emotion?(text: string): EmotionReading | null | Promise<EmotionReading | null>;
  /** Angry weight that switches the idle loop to 'angry', and back to 'normal' below the second (default 0.5 / 0.4). */
  idleAngry?: { up: number; down: number };
  /** Clock (tests). Default performance.now(). */
  now?(): number;
}

export interface TurnLatency {
  utt: number;
  /** User's turn end (the VAD's speech end) -> the app's final text. */
  speechEndToFinalMs: number | null;
  /** User's turn end -> the reply's first text piece. */
  speechEndToTokenMs: number | null;
  /** User's turn end -> the reply's first audio handed to the avatar. */
  speechEndToSoundMs: number | null;
  /** User's turn end -> the reply's text complete. */
  speechEndToReplyDoneMs: number | null;
  /** Reply's text complete -> the sound over. */
  replyDoneToPlaybackEndMs: number | null;
  /** A barge-in: when the user started speaking, and how long the sound took to stop (ms). */
  bargeIn: { atMs: number; stoppedInMs: number } | null;
  aizuchi: boolean;
  aborted: boolean;
}

export interface ConversationEvents {
  /** The user's speech started (barge-in applied). */
  speechstart: { utt: number; sounding: boolean };
  /** End of the user's turn was detected (before the app's final text). */
  speechend: { utt: number };
  /** A reply was asked for. */
  replystart: { utt: number; text: string };
  /** The reply's first audio is playing. */
  replysound: { utt: number; speechEndToSoundMs: number | null };
  /** The reply's text is complete. */
  replydone: { utt: number; text: string };
  /** The reply ended (sound over, or aborted). */
  turnend: { utt: number; aborted: boolean };
  /** The reply could not be made. */
  error: { utt: number; error: Error };
  /** A turn's numbers, once it is over. */
  latency: TurnLatency;
}

export interface ConversationStats {
  turns: number;
  /** Median of each latency over the turns that measured it (ms), null when nothing was measured. */
  medians: { speechEndToFinalMs: number | null; speechEndToTokenMs: number | null; speechEndToSoundMs: number | null; speechEndToReplyDoneMs: number | null };
  bargeIns: number;
  /** Largest time to stop the sound after a barge-in (ms). */
  maxBargeInStopMs: number | null;
  errors: number;
}

export class Conversation {
  private readonly o: ConversationOptions;
  private readonly events = new Emitter();
  private readonly now: () => number;
  private state: TurnState = 'idle';
  private data: TurnData = initialTurnData();
  private readonly history: ConversationMessage[] = [];
  private readonly latencies: TurnLatency[] = [];
  private ac: AbortController | null = null;
  private queue: PushQueue<SpeechAudio> | null = null;
  private playing: Promise<void> | null = null;
  /** When the user's turn ended (VAD), and the pending final of that turn. */
  private turnEndAt: number | null = null;
  private current: TurnLatency | null = null;
  private errors = 0;
  private destroyed = false;

  constructor(options: ConversationOptions) {
    this.o = options ?? ({} as ConversationOptions);
    if (!this.o.avatar || typeof this.o.avatar.speakStream !== 'function') throw new OptionsError('Conversation: "avatar" must be an avatar from create()');
    if (typeof this.o.reply !== 'function') throw new OptionsError('Conversation: "reply" must be a function returning an AsyncIterable of text');
    if (typeof this.o.synthesize !== 'function') throw new OptionsError('Conversation: "synthesize" must be a function returning an AsyncIterable of { pcm, sampleRate }');
    this.now = this.o.now ?? (() => performance.now());
  }

  get turnState(): TurnState { return this.state; }
  get messages(): readonly ConversationMessage[] { return this.history; }

  on<K extends keyof ConversationEvents>(event: K, listener: (e: ConversationEvents[K]) => void): () => void {
    return this.events.on(event, listener);
  }

  /** Starts listening (the machine leaves 'idle'). */
  start(): void {
    this.send({ type: 'start' });
  }

  /** Ends the conversation now: the sound stops and any reply in flight is dropped. */
  stop(): void {
    this.send({ type: 'stop' });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.stop();
    this.destroyed = true;
    this.events.clear();
  }

  /** The user started speaking (the microphone's VAD). Stops the avatar's sound: barge-in. */
  userSpeechStart(sounding = false): void {
    const before = this.now();
    const wasSounding = this.state === 'speaking' || this.playing !== null;
    // barge-in: the sound stops and the reply still being made is dropped (its audio never reaches the avatar)
    const hadReply = this.data.replyPending || wasSounding;
    this.send({ type: 'user_speech_start' });
    if (hadReply) {
      this.abortReply();
      this.stopSound();
    }
    const stoppedIn = this.now() - before;
    this.events.emit('speechstart', { utt: this.data.utt, sounding: wasSounding });
    if (wasSounding && this.current) this.current.bargeIn = { atMs: before, stoppedInMs: stoppedIn };
  }

  /** End of the user's turn was detected (the VAD's speech end, before the app's final text). */
  speechEnd(): void {
    if (this.state === 'idle') return;
    this.turnEndAt = this.now();
    this.current = { utt: this.data.utt + 1, speechEndToFinalMs: null, speechEndToTokenMs: null, speechEndToSoundMs: null, speechEndToReplyDoneMs: null, replyDoneToPlaybackEndMs: null, bargeIn: null, aizuchi: false, aborted: false };
    this.send({ type: 'speech_end' });
    this.events.emit('speechend', { utt: this.data.utt + 1 });
    void this.maybeAizuchi();
  }

  /** The app's final text for the user's turn (its own ASR): the reply is asked for from here. */
  userFinal(text: string): void {
    const t = (text ?? '').trim();
    const at = this.now();
    const utt = this.data.utt + 1;
    if (t === '') return;
    if (this.turnEndAt === null) this.turnEndAt = at;
    if (this.current) this.current.speechEndToFinalMs = Math.round(at - this.turnEndAt);
    this.history.push({ role: 'user', content: t });
    const step = this.send({ type: 'final', utt, text: t });
    if (step.actions.some((a) => typeof a !== 'string' && a.kind === 'request_reply')) this.startReply(utt, t);
  }

  /**
   * No microphone: one line of the user, final at once (the same machine, so latency hooks and barge-in behave as
   * they do with a mic). Useful for tests, keyboards and scripted demos.
   */
  sendText(text: string): void {
    if (this.state === 'idle') this.start();
    this.userSpeechStart();
    this.speechEnd();
    this.userFinal(text);
  }

  stats(): ConversationStats {
    const med = (f: (l: TurnLatency) => number | null): number | null => {
      const v = this.latencies.map(f).filter((x): x is number => x !== null).sort((a, b) => a - b);
      if (v.length === 0) return null;
      const mid = Math.floor(v.length / 2);
      return v.length % 2 ? v[mid]! : Math.round((v[mid - 1]! + v[mid]!) / 2);
    };
    const stops = this.latencies.map((l) => l.bargeIn?.stoppedInMs).filter((x): x is number => x !== undefined);
    return {
      turns: this.latencies.length,
      medians: {
        speechEndToFinalMs: med((l) => l.speechEndToFinalMs),
        speechEndToTokenMs: med((l) => l.speechEndToTokenMs),
        speechEndToSoundMs: med((l) => l.speechEndToSoundMs),
        speechEndToReplyDoneMs: med((l) => l.speechEndToReplyDoneMs),
      },
      bargeIns: stops.length,
      maxBargeInStopMs: stops.length ? Math.max(...stops) : null,
      errors: this.errors,
    };
  }

  /** The per-turn numbers so far (in turn order). */
  turnLatencies(): readonly TurnLatency[] { return this.latencies; }

  // --- internals ---------------------------------------------------------------------------------------------

  private send(e: TurnEvent): { state: TurnState; data: TurnData; actions: TurnAction[] } {
    if (this.destroyed && e.type !== 'stop') return { state: this.state, data: this.data, actions: [] };
    const step = turnStep(this.state, this.data, e);
    this.state = step.state;
    this.data = step.data;
    for (const a of step.actions) {
      if (a === 'stop_audio') this.stopSound();
    }
    if (e.type === 'stop') this.abortReply();
    return step;
  }

  private stopSound(): void {
    this.queue?.close();
    this.queue = null;
    this.o.avatar.stop();
    this.playing = null;
  }

  private abortReply(): void {
    this.ac?.abort();
    this.ac = null;
  }

  private startReply(utt: number, text: string): void {
    const ac = new AbortController();
    this.ac = ac;
    this.events.emit('replystart', { utt, text });
    void this.runReply(utt, text, ac.signal);
  }

  private async runReply(utt: number, text: string, signal: AbortSignal): Promise<void> {
    const queue = new PushQueue<SpeechAudio>();
    this.queue = queue;
    const t0 = this.turnEndAt ?? this.now();
    let firstToken = true;
    let firstSound = true;
    let replyText = '';
    let index = 0;
    // playback is started with the queue: the avatar pulls the audio as the reply is synthesised
    const playing = this.o.avatar.speakStream(queue);
    this.playing = playing;
    const ended = playing.then(() => true, (e: unknown) => { void e; return false; });
    try {
      for await (const piece of this.o.reply({ messages: [...this.history], utt, signal })) {
        if (signal.aborted) break;
        if (piece === '') continue;
        if (firstToken) {
          firstToken = false;
          if (this.current) this.current.speechEndToTokenMs = Math.round(this.now() - t0);
        }
        replyText += piece;
        for await (const chunk of this.o.synthesize({ piece, text: replyText, index: index++, utt, signal, kind: 'reply' })) {
          if (signal.aborted) break;
          if (firstSound) {
            firstSound = false;
            const at = this.now();
            if (this.current) this.current.speechEndToSoundMs = Math.round(at - t0);
            this.send({ type: 'reply_audio', utt });
            this.events.emit('replysound', { utt, speechEndToSoundMs: this.current?.speechEndToSoundMs ?? null });
          }
          queue.push(chunk);
        }
        if (signal.aborted) break;
      }
      if (signal.aborted) { queue.close(); this.finishTurn(utt, true); return; }
      this.history.push({ role: 'assistant', content: replyText });
      if (this.current) this.current.speechEndToReplyDoneMs = Math.round(this.now() - t0);
      this.events.emit('replydone', { utt, text: replyText });
      queue.end();
      this.send({ type: 'reply_done', utt });
      await ended;
      if (signal.aborted) { this.finishTurn(utt, true); return; }
      if (this.current && this.current.speechEndToReplyDoneMs !== null) {
        this.current.replyDoneToPlaybackEndMs = Math.round(this.now() - t0 - this.current.speechEndToReplyDoneMs);
      }
      this.send({ type: 'playback_end', utt });
      this.finishTurn(utt, false);
      await this.applyIdle(replyText);
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      queue.fail(err);
      if (signal.aborted) { this.finishTurn(utt, true); return; }
      this.errors++;
      this.send({ type: 'reply_failed', utt, error: err });
      this.events.emit('error', { utt, error: err });
      this.finishTurn(utt, false);
    } finally {
      if (this.queue === queue) this.queue = null;
      if (this.current && replyText === '' && this.current.speechEndToTokenMs === null) {
        // nothing was said: still report the turn's numbers
        this.latencies.push(this.current);
      }
    }
  }

  private finishTurn(utt: number, aborted: boolean): void {
    this.playing = null;
    this.events.emit('turnend', { utt, aborted });
    if (this.current && this.current.utt === utt) {
      this.current.aborted = aborted;
      this.latencies.push(this.current);
      this.events.emit('latency', this.current);
      this.current = null;
    }
    if (this.ac && this.ac.signal.aborted) this.ac = null;
  }

  private async maybeAizuchi(): Promise<void> {
    const az = this.o.aizuchi;
    if (!az) return;
    const utt = this.data.utt + 1;
    const ac = new AbortController();
    try {
      const text = await az.pick({ messages: [...this.history], utt, signal: ac.signal });
      if (!text || ac.signal.aborted) return;
      const queue = new PushQueue<SpeechAudio>();
      if (this.current) this.current.aizuchi = true;
      void this.o.avatar.speakStream(queue).catch(() => undefined);
      for await (const chunk of this.o.synthesize({ piece: text, text, index: 0, utt, signal: ac.signal, kind: 'aizuchi' })) queue.push(chunk);
      queue.end();
      this.send({ type: 'aizuchi_end', utt });
    } catch {
      // an aizuchi is optional: its failure never breaks the turn
    }
  }

  /** The idle loop follows the reply's emotion (only when the avatar has an angry loop and the app gives an emotion). */
  private async applyIdle(text: string): Promise<void> {
    const f = this.o.emotion;
    if (!f) return;
    const loops = this.o.avatar.idleLoops ?? [];
    if (!loops.includes('angry') || !loops.includes('normal')) return;
    const thresholds = this.o.idleAngry ?? { up: 0.5, down: 0.4 };
    let reading: EmotionReading | null = null;
    try { reading = await f(text); } catch { return; }
    if (!reading) return;
    const want = this.o.avatar.idle === 'angry' ? (reading.angry <= thresholds.down ? 'normal' : 'angry') : (reading.angry >= thresholds.up ? 'angry' : 'normal');
    if (want === this.o.avatar.idle) return;
    try { await this.o.avatar.setIdle(want); } catch { /* the next turn tries again */ }
  }
}
