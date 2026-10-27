// zen-lipsync/tts (browser): speaking text with ZEN-TTS on the avatar. zen-tts is an optional peer dependency: pass a
// loaded ZenTts (`import { ZenTts } from 'zen-tts'`); this module does not import it.
//
//   const tts = await ZenTts.load({ modelsUrl: './tts-models', backend: 'webgpu', device });
//   await speakText(avatar, tts, '今日はいい天気ですね。散歩に行きましょう。', { emotion: { happy: 60, neutral: 40 } });
//
// The text is split into sentences; each sentence is synthesised and handed to avatar.speakStream as soon as it is
// ready, so the first sentence plays while the next is made, and the sentences follow without a gap.
// On Node use zen-lipsync/tts/node with zen-tts/node.
import { OptionsError } from '../core/errors.js';
import type { AvatarHandle, SpeechAudio } from '../core/types.js';

/** ZEN-TTS's sample rate (44.1 kHz). */
export const TTS_SAMPLE_RATE = 44100;

/** The ZEN-TTS speak() options (emotion, change, strength, dict, settings), passed through as they are. */
export type TtsSpeakOptions = Record<string, unknown>;

/** What speakText needs of a ZEN-TTS engine (zen-tts ZenTts, or zen-tts/node ZenTtsNode). */
export interface TtsEngine {
  speak(text: string, o?: TtsSpeakOptions): Promise<{ audio: Float32Array; sampleRate: number }>;
}

export interface SpeakTextOptions extends TtsSpeakOptions {
  /** Called with each sentence when its audio is ready (index, text, synthesis ms). */
  onSentence?: (s: { index: number; text: string; synthMs: number; seconds: number }) => void;
}

/**
 * Text -> sentences: split after 。！？!? (kept with their sentence) and at line breaks; a piece longer than maxChars is
 * cut after its last 、，, or space before the limit (else at the limit). The same rule as zen-tts splitSentences.
 */
export function splitSentences(text: string, maxChars = 120): string[] {
  const out: string[] = [];
  for (const line of text.split(/\r?\n+/)) {
    for (const m of line.matchAll(/[^。！？!?]+[。！？!?]*|[。！？!?]+/gu)) {
      let s = m[0].trim();
      while (s.length > maxChars) {
        const head = s.slice(0, maxChars);
        const cut = Math.max(head.lastIndexOf('、'), head.lastIndexOf('，'), head.lastIndexOf(','), head.lastIndexOf(' '));
        const at = cut > 0 ? cut + 1 : maxChars;
        out.push(s.slice(0, at).trim());
        s = s.slice(at).trim();
      }
      if (s && /[^\s。！？!?、，,]/u.test(s)) out.push(s);
    }
  }
  return out;
}

function checkEngine(tts: unknown): asserts tts is TtsEngine {
  if (!tts || typeof (tts as TtsEngine).speak !== 'function') throw new OptionsError('speakText(): pass a loaded ZEN-TTS engine (ZenTts from "zen-tts", or ZenTtsNode from "zen-tts/node")');
}

/**
 * The text's sentences as speech chunks, synthesised one after another (each when the consumer asks for the next).
 * eager: the first sentence is started now (before the consumer asks), so it is made while the consumer gets ready.
 */
export function ttsChunks(tts: TtsEngine, text: string, options: SpeakTextOptions = {}, eager = false): AsyncGenerator<SpeechAudio> {
  checkEngine(tts);
  if (typeof text !== 'string') throw new OptionsError('speakText(): text must be a string');
  const { onSentence, ...o } = options;
  const list = splitSentences(text);
  const make = (sentence: string): Promise<{ r: { audio: Float32Array; sampleRate: number }; ms: number }> => {
    const t0 = performance.now();
    return tts.speak(sentence, o).then((r) => ({ r, ms: performance.now() - t0 }));
  };
  const first = eager && list.length ? make(list[0]!) : null;
  first?.catch(() => undefined);
  return (async function* chunks(): AsyncGenerator<SpeechAudio> {
    for (const [index, sentence] of list.entries()) {
      const { r, ms } = index === 0 && first ? await first : await make(sentence);
      const sampleRate = r.sampleRate || TTS_SAMPLE_RATE;
      onSentence?.({ index, text: sentence, synthMs: ms, seconds: r.audio.length / sampleRate });
      yield { pcm: r.audio, sampleRate };
    }
  })();
}

/** Synthesises the text with ZEN-TTS, sentence by sentence, and speaks it on the avatar (browser). Resolves at the end of the sound. */
export async function speakText(avatar: AvatarHandle, tts: TtsEngine, text: string, options: SpeakTextOptions = {}): Promise<void> {
  if (!avatar || typeof avatar.speakStream !== 'function') throw new OptionsError('speakText(): pass an avatar from create()');
  checkEngine(tts);
  if (avatar.backend === 'cpu') throw new OptionsError('speakText(): on Node use zen-lipsync/tts/node (speakTextFrames / renderText / streamText)');
  // the first sentence is made while the player gets its audio clock running; the next ones while the previous plays
  await avatar.speakStream(ttsChunks(tts, text, options, true));
}
