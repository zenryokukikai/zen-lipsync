// The turn machine of a spoken conversation, apart from the browser so that node --test can drive it.
//
// States: idle (no conversation) -> listening -> awaiting_reply -> speaking -> listening ...
// A turn is one user utterance and the reply to it. The machine owns only what the replies and the audio must agree
// on: whether a reply is in flight, whether sound is playing, and which turn a late event belongs to.

export type TurnState = 'idle' | 'listening' | 'awaiting_reply' | 'speaking';

export type TurnEvent =
  | { type: 'start' }
  | { type: 'user_speech_start' }
  | { type: 'speech_end' }
  | { type: 'final'; utt: number; text: string }
  | { type: 'reply_token'; utt: number }
  | { type: 'reply_audio'; utt: number }
  | { type: 'reply_done'; utt: number }
  | { type: 'reply_failed'; utt: number; error: Error }
  | { type: 'playback_end'; utt: number }
  | { type: 'aizuchi_end'; utt: number }
  | { type: 'stop' };

/** What the owner of the machine does in response. */
export type TurnAction =
  /** The user started speaking while something sounded: stop it now (barge-in). */
  | 'stop_audio'
  /** The user's line is final: ask the app for the reply (exactly once per turn). */
  | { kind: 'request_reply'; utt: number; text: string }
  /** The reply's first audio is ready: hand it to the avatar. */
  | { kind: 'play_reply'; utt: number }
  /** The turn is over and the app may listen again. */
  | { kind: 'listen'; utt: number }
  /** The reply could not be made: tell the app, then listen again. */
  | { kind: 'reply_failed'; utt: number; error: Error };

export interface TurnData {
  /** Turns taken so far (a turn is counted when its user line is final). */
  utt: number;
  /** True from the request until the reply's audio is handed over. */
  replyPending: boolean;
  /** True while the reply's sound plays. */
  playing: boolean;
  /** The utterance whose reply failed last (for the app's error display). */
  failed: number;
}

export interface TurnStep {
  state: TurnState;
  data: TurnData;
  actions: TurnAction[];
}

const initialData: TurnData = Object.freeze({ utt: 0, replyPending: false, playing: false, failed: 0 }) as TurnData;

export function turnStep(state: TurnState, data: TurnData, e: TurnEvent): TurnStep {
  const same = (actions: TurnAction[] = []): TurnStep => ({ state, data, actions });
  switch (e.type) {
    case 'start':
      return state === 'idle' ? { state: 'listening', data: { ...data }, actions: [] } : same();
    case 'stop':
      return { state: 'idle', data: { ...data, replyPending: false, playing: false }, actions: data.playing || data.replyPending ? ['stop_audio'] : [] };
    case 'user_speech_start':
      // barge-in: the user's voice stops the reply's sound and drops the reply that is still being made
      if (state === 'idle') return same();
      if (data.playing || state === 'speaking') return { state: 'listening', data: { ...data, playing: false, replyPending: false }, actions: ['stop_audio'] };
      return { state: 'listening', data: { ...data, replyPending: false }, actions: [] };
    case 'speech_end':
      // end of turn detection: the app's own ASR final may still be on its way (it calls `final`)
      return state === 'listening' ? same() : same();
    case 'final': {
      if (state === 'idle') return same();
      // one reply per line: a second final for the same utterance, or a final while a reply is in flight, is ignored
      if (e.utt <= data.utt || data.replyPending) return same();
      const utt = e.utt;
      return { state: 'awaiting_reply', data: { ...data, utt, replyPending: true }, actions: [{ kind: 'request_reply', utt, text: e.text }] };
    }
    case 'reply_token':
      return e.utt === data.utt ? same() : same();
    case 'reply_audio':
      if (e.utt !== data.utt || !data.replyPending) return same();
      return { state: 'speaking', data: { ...data, replyPending: false, playing: true }, actions: [{ kind: 'play_reply', utt: e.utt }] };
    case 'reply_done':
      if (e.utt !== data.utt) return same();
      // the text is done but its sound may still be playing: the turn ends at playback_end then
      if (state === 'speaking' || data.playing) return same();
      return { state: 'listening', data: { ...data, replyPending: false, playing: false }, actions: [{ kind: 'listen', utt: e.utt }] };
    case 'reply_failed':
      if (e.utt !== data.utt) return same();
      return { state: 'listening', data: { ...data, replyPending: false, playing: false, failed: e.utt }, actions: [{ kind: 'reply_failed', utt: e.utt, error: e.error }, { kind: 'listen', utt: e.utt }] };
    case 'playback_end':
      if (e.utt !== data.utt) return same();
      if (data.replyPending) return same();
      return { state: 'listening', data: { ...data, playing: false }, actions: [{ kind: 'listen', utt: e.utt }] };
    case 'aizuchi_end':
      // an aizuchi is not a turn: it never ends one, and it never starts one
      return same();
    default:
      return same();
  }
}

/** The data a conversation starts with (utt 0, nothing in flight). */
export function initialTurnData(): TurnData {
  return { ...initialData };
}
