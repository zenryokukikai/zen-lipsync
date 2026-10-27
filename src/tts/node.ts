// zen-lipsync/tts/node: text -> the avatar speaking it, on Node, with ZEN-TTS on Node (zen-tts/node; an optional peer:
// pass a loaded ZenTtsNode, this module does not import it).
//
//   import { create } from '@zenryokukikai/zen-lipsync/node';
//   import { renderText, streamText } from '@zenryokukikai/zen-lipsync/tts/node';
//   import { ZenTtsNode } from 'zen-tts/node';
//   const tts = await ZenTtsNode.load({ modelsPath: './tts-models' });
//   await renderText(avatar, tts, 'こんにちは。全力マリンです。', { out: 'marine.mp4' });
//
// Sentences are synthesised one at a time and given to the avatar as speech chunks; the avatar prepares each chunk while
// the previous one is drawn, so the sentences follow without a gap. audioSampleRate is 44100 (ZEN-TTS) unless set.
import type { Readable } from 'node:stream';
import type { FramesOptions, NodeAvatar, NodeFrame, RenderOptions, StreamOptions } from '../node/avatar.js';
import { OptionsError } from '../core/errors.js';
import { TTS_SAMPLE_RATE, ttsChunks, type SpeakTextOptions, type TtsEngine } from './index.js';

export { splitSentences, ttsChunks, TTS_SAMPLE_RATE, type SpeakTextOptions, type TtsEngine, type TtsSpeakOptions } from './index.js';

/** Options of ZEN-TTS (emotion, strength, ...) and of the avatar's output together. */
export interface TextOutputOptions {
  /** ZEN-TTS speak() options and onSentence. */
  tts?: SpeakTextOptions;
}

function checkAvatar(avatar: unknown): asserts avatar is NodeAvatar {
  if (!avatar || typeof (avatar as NodeAvatar).frames !== 'function') throw new OptionsError('pass an avatar from zen-lipsync/node create()');
}

/** The avatar's frames (RGB + audio, 25 fps) speaking the text: NodeAvatar.frames() over the ZEN-TTS sentences. */
export function speakTextFrames(avatar: NodeAvatar, tts: TtsEngine, text: string, opts: FramesOptions & TextOutputOptions = {}): AsyncGenerator<NodeFrame> {
  checkAvatar(avatar);
  const { tts: to, ...fo } = opts;
  return avatar.frames(ttsChunks(tts, text, to, true), { audioSampleRate: TTS_SAMPLE_RATE, ...fo });
}

/** A video file of the avatar speaking the text (.mp4: H.264 + AAC, .webm: VP9 + Opus). */
export function renderText(avatar: NodeAvatar, tts: TtsEngine, text: string, opts: RenderOptions & TextOutputOptions): Promise<{ file: string; frames: number; durationMs: number }> {
  checkAvatar(avatar);
  const { tts: to, ...ro } = opts ?? ({} as RenderOptions);
  return avatar.render(ttsChunks(tts, text, to, true), { audioSampleRate: TTS_SAMPLE_RATE, ...ro });
}

/** The avatar speaking the text as a stream for delivery (fmp4 / webm / jpeg objects; see StreamOptions). */
export function streamText(avatar: NodeAvatar, tts: TtsEngine, text: string, opts: StreamOptions & TextOutputOptions): Readable {
  checkAvatar(avatar);
  const { tts: to, ...so } = opts ?? ({} as StreamOptions);
  return avatar.stream(ttsChunks(tts, text, to, true), { audioSampleRate: TTS_SAMPLE_RATE, ...so });
}
