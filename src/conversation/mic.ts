// The microphone for a conversation: 16 kHz frames (512 samples, 32 ms) from an AudioWorklet, end-pointed by
// EndPointer, into Conversation.userSpeechStart() / speechEnd(). Browser only, no dependency.
//
// Echo cancellation stays on: with the avatar's voice in the room, the mic must not take it for the user (barge-in
// depends on it). A VAD model (Silero, a turn-end model) is an app-level choice: pass
// `isSpeech` to use one instead of the level threshold, and call `end()` from your own turn-end decision.
import type { Conversation } from './index.js';
import { EndPointer, type EndpointOptions } from './endpoint.js';

export interface MicOptions extends EndpointOptions {
  /** The conversation to drive. */
  conversation: Pick<Conversation, 'userSpeechStart' | 'speechEnd'>;
  /** Speech probability of one frame (0..1) from your own VAD; with it, the level threshold is not used. */
  isSpeech?(frame: Float32Array): boolean | Promise<boolean>;
  /** Called with every frame (level meters, recordings, your own ASR). */
  onFrame?(frame: Float32Array, index: number): void;
  /** The audio context to record on (e.g. the page's, for iOS: play and record in one session). Default: a 16 kHz one. */
  context?: AudioContext;
  /** Sample rate of the frames (default 16000). */
  sampleRate?: number;
}

const WORKLET = (frame: number, target: number) => `class P extends AudioWorkletProcessor{constructor(){super();this.b=new Float32Array(${frame});this.n=0;this.r=sampleRate/${target};this.ph=0;this.acc=0;this.cnt=0}
process(i){const x=i[0]&&i[0][0];if(!x)return true;for(let k=0;k<x.length;k++){this.acc+=x[k];this.cnt++;this.ph+=1;if(this.ph>=this.r){this.ph-=this.r;this.b[this.n++]=this.acc/this.cnt;this.acc=0;this.cnt=0;if(this.n===${frame}){this.port.postMessage(this.b.slice(0));this.n=0}}}return true}}
registerProcessor('zen-lipsync-mic',P)`;

export interface Mic {
  /** Stops the microphone and closes the context it opened. */
  stop(): void;
  /** Ends the turn now (a cap, or a button), as if the quiet had lasted. */
  end(): void;
  readonly inSpeech: boolean;
  /** Last frame's level in dBFS. */
  readonly levelDbfs: number;
}

/**
 * Opens the microphone and drives the conversation: `userSpeechStart()` when speech begins (which stops the avatar's
 * sound: barge-in) and `speechEnd()` after the configured quiet. Rejects when the user does not allow the microphone.
 */
export async function openMic(o: MicOptions): Promise<Mic> {
  const frameSamples = Math.round(((o.sampleRate ?? 16000) * (o.frameMs ?? 32)) / 1000);
  const end = new EndPointer(o);
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 } });
  const own = !o.context;
  const ctx = o.context ?? new AudioContext({ sampleRate: o.sampleRate ?? 16000 });
  const url = URL.createObjectURL(new Blob([WORKLET(frameSamples, o.sampleRate ?? 16000)], { type: 'text/javascript' }));
  let node: AudioWorkletNode | null = null;
  let index = 0;
  let chain: Promise<void> = Promise.resolve();
  let stopped = false;
  try {
    await ctx.audioWorklet.addModule(url);
    node = new AudioWorkletNode(ctx, 'zen-lipsync-mic');
    node.port.onmessage = (e: MessageEvent) => {
      if (stopped) return;
      const frame = e.data as Float32Array;
      const i = index++;
      o.onFrame?.(frame, i);
      chain = chain.then(async () => {
        if (stopped) return;
        const events = o.isSpeech ? end.pushDecision(await o.isSpeech(frame)) : end.push(frame);
        for (const ev of events) {
          if (ev.type === 'speech_start') o.conversation.userSpeechStart(true);
          else o.conversation.speechEnd();
        }
      }).catch(() => undefined);
    };
    ctx.createMediaStreamSource(stream).connect(node);
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    if (own) void ctx.close();
    throw e;
  }
  return {
    stop() {
      if (stopped) return;
      stopped = true;
      stream.getTracks().forEach((t) => t.stop());
      node?.disconnect();
      URL.revokeObjectURL(url);
      if (own) void ctx.close();
    },
    end() {
      for (const ev of end.endNow()) if (ev.type === 'speech_end') o.conversation.speechEnd();
    },
    get inSpeech() { return end.inSpeech; },
    get levelDbfs() { return end.lastDbfs; },
  };
}
