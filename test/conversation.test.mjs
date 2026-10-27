// The conversation kit: the turn machine, barge-in, abort and the latency hooks (node --test, after `pnpm build`).
// No browser, no model, no provider: the avatar, the reply and the speech are stand-ins.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Conversation, EndPointer, PushQueue, initialTurnData, turnStep } from '../dist/conversation/index.js';

/** Let the microtask queue and the pending promise chains run. */
const tick = () => new Promise((r) => setTimeout(r, 0));

/** An avatar stand-in: speakStream() pulls the chunks and finishes after `playMs` per chunk. */
function fakeAvatar({ loops = ['normal', 'angry'], playMs = 5 } = {}) {
  const spoken = [];
  const state = { idle: 'normal', stopped: 0, streams: 0, idles: [], speaking: false };
  return {
    state,
    spoken,
    get idle() { return state.idle; },
    idleLoops: loops,
    async speakStream(chunks) {
      state.streams++;
      state.speaking = true;
      const got = [];
      spoken.push(got);
      for await (const c of chunks) { got.push(c.pcm.length); await new Promise((r) => setTimeout(r, playMs)); }
      state.speaking = false;
      return undefined;
    },
    stop() { state.stopped++; state.speaking = false; },
    async setIdle(name) { state.idle = name; state.idles.push(name); return true; },
  };
}

/** Speech of `n` chunks, each 1/25 s at 16 kHz. */
async function* say(text, n = 2) {
  for (let i = 0; i < n; i++) yield { pcm: new Float32Array(640).fill(0.01), sampleRate: 16000 };
  void text;
}

/** A reply of `pieces` tokens, each arriving after `ms`. */
async function* tokens(pieces, ms = 2) {
  for (const p of pieces) { await new Promise((r) => setTimeout(r, ms)); yield p; }
}

test('turn machine: one reply per line, no double request', () => {
  let s = 'idle';
  let d = initialTurnData();
  const step = (e) => { const r = turnStep(s, d, e); s = r.state; d = r.data; return r.actions; };
  assert.deepEqual(step({ type: 'start' }), []);
  assert.equal(s, 'listening');
  step({ type: 'user_speech_start' });
  step({ type: 'speech_end' });
  const a1 = step({ type: 'final', utt: 1, text: 'こんにちは' });
  assert.deepEqual(a1, [{ kind: 'request_reply', utt: 1, text: 'こんにちは' }]);
  // a second final for the same line, or another final while the reply is in flight: ignored
  assert.deepEqual(step({ type: 'final', utt: 1, text: 'こんにちは' }), []);
  assert.deepEqual(step({ type: 'final', utt: 2, text: 'もう一度' }), []);
  assert.equal(s, 'awaiting_reply');
  assert.deepEqual(step({ type: 'reply_audio', utt: 1 }), [{ kind: 'play_reply', utt: 1 }]);
  assert.equal(s, 'speaking');
  step({ type: 'reply_done', utt: 1 });
  assert.equal(s, 'speaking', 'the text ending while its sound plays keeps the turn open');
  assert.deepEqual(step({ type: 'playback_end', utt: 1 }), [{ kind: 'listen', utt: 1 }]);
  assert.equal(s, 'listening');
});

test('turn machine: barge-in stops the sound and drops the reply', () => {
  let s = 'listening';
  let d = { ...initialTurnData(), utt: 1, replyPending: true };
  const r = turnStep(s, d, { type: 'user_speech_start' });
  assert.equal(r.state, 'listening');
  assert.equal(r.data.replyPending, false);
  assert.deepEqual(r.actions, []);
  s = 'speaking';
  d = { ...initialTurnData(), utt: 1, playing: true };
  const r2 = turnStep(s, d, { type: 'user_speech_start' });
  assert.deepEqual(r2.actions, ['stop_audio']);
  assert.equal(r2.data.playing, false);
});

test('turn machine: a stale event of an older turn changes nothing', () => {
  const s = 'awaiting_reply';
  const d = { ...initialTurnData(), utt: 2, replyPending: true };
  assert.deepEqual(turnStep(s, d, { type: 'reply_audio', utt: 1 }).actions, []);
  assert.deepEqual(turnStep(s, d, { type: 'playback_end', utt: 1 }).actions, []);
  assert.equal(turnStep(s, d, { type: 'reply_audio', utt: 1 }).state, 'awaiting_reply');
});

test('conversation: a turn speaks the reply and reports the three latencies', async () => {
  const avatar = fakeAvatar();
  const seen = [];
  const c = new Conversation({
    avatar,
    async *reply() { yield* tokens(['こんにちは。', '今日は', 'いい天気ですね。'], 3); },
    synthesize: (ctx) => say(ctx.text),
    now: () => performance.now(),
  });
  c.on('latency', (l) => seen.push(l));
  c.start();
  c.sendText('やあ');
  for (let i = 0; i < 40 && seen.length === 0; i++) await tick();
  assert.equal(seen.length, 1, 'one turn reported');
  const l = seen[0];
  assert.equal(l.utt, 1);
  assert.ok(l.speechEndToFinalMs !== null && l.speechEndToFinalMs >= 0);
  assert.ok(l.speechEndToTokenMs !== null && l.speechEndToTokenMs >= 0);
  assert.ok(l.speechEndToSoundMs !== null && l.speechEndToSoundMs >= l.speechEndToTokenMs, 'the sound cannot precede the first token');
  assert.equal(l.aborted, false);
  assert.equal(c.turnState, 'listening');
  assert.equal(avatar.state.streams, 1, 'one stream for the reply');
  const said = avatar.spoken[0];
  assert.equal(said.length, 6, "three text pieces of two chunks each, all played");
  assert.deepEqual(c.messages.map((m) => m.role), ['user', 'assistant']);
  assert.equal(c.stats().turns, 1);
  assert.equal(c.stats().errors, 0);
  c.destroy();
});

test('conversation: barge-in stops the sound and a second reply is not started', async () => {
  const avatar = fakeAvatar({ playMs: 40 });
  let replies = 0;
  const c = new Conversation({
    avatar,
    async *reply(ctx) {
      replies++;
      yield* tokens(['長い返事を', 'しています。'], 5);
      // a well-behaved producer watches the signal
      while (!ctx.signal.aborted) await new Promise((r) => setTimeout(r, 5));
      if (!ctx.signal.aborted) yield 'まだ続きます。';
    },
    synthesize: (ctx) => say(ctx.text, 4),
  });
  c.start();
  c.sendText('話しかける');
  for (let i = 0; i < 30 && (avatar.state.streams === 0 || replies === 0); i++) await tick();
  assert.equal(replies, 1);
  // the user speaks over the reply
  c.userSpeechStart();
  assert.ok(avatar.state.stopped >= 1, 'the avatar was stopped');
  assert.equal(c.turnState, 'listening');
  // the reply's producer sees the abort on its next step: give it a moment
  for (let i = 0; i < 20 && c.stats().bargeIns === 0; i++) await tick();
  const stats = c.stats();
  assert.equal(stats.bargeIns, 1);
  assert.ok(stats.maxBargeInStopMs !== null && stats.maxBargeInStopMs < 50, `stop was immediate (${stats.maxBargeInStopMs} ms)`);
  // the next turn works normally: exactly one more reply
  c.sendText('もう一度');
  for (let i = 0; i < 40 && replies < 2; i++) await tick();
  assert.equal(replies, 2, 'no reply was made twice');
  c.destroy();
});

test('conversation: stop() aborts the reply in flight and drops its audio', async () => {
  const avatar = fakeAvatar({ playMs: 30 });
  let aborted = false;
  const c = new Conversation({
    avatar,
    async *reply(ctx) {
      ctx.signal.addEventListener('abort', () => { aborted = true; });
      yield* tokens(['途中まで', '話します。'], 10);
      await new Promise((r) => setTimeout(r, 50));
    },
    synthesize: (ctx) => say(ctx.text, 4),
  });
  c.start();
  c.sendText('やあ');
  for (let i = 0; i < 20 && avatar.state.streams === 0; i++) await tick();
  c.stop();
  for (let i = 0; i < 10; i++) await tick();
  assert.equal(aborted, true, 'the reply saw the abort');
  assert.ok(avatar.state.stopped >= 1);
  assert.equal(c.turnState, 'idle');
  c.destroy();
});

test('conversation: the idle loop follows the emotion when the avatar has an angry one', async () => {
  const avatar = fakeAvatar();
  const c = new Conversation({
    avatar,
    // the reply echoes the line, so the emotion follows what the user said
    async *reply(ctx) { yield ctx.messages.at(-1).content; },
    synthesize: (ctx) => say(ctx.text, 1),
    emotion: (text) => ({ angry: text.includes('むか') ? 0.8 : 0.1 }),
  });
  c.start();
  c.sendText('むか');
  for (let i = 0; i < 100 && avatar.state.idles.length === 0; i++) await tick();
  assert.deepEqual(avatar.state.idles, ['angry']);
  c.sendText('しずか');
  for (let i = 0; i < 200 && avatar.state.idles.length < 2; i++) await tick();
  assert.deepEqual(avatar.state.idles, ['angry', 'normal'], 'it comes back below the lower threshold');
  c.destroy();
});

test('conversation: the aizuchi plays at the turn end, before the reply', async () => {
  const avatar = fakeAvatar();
  const kinds = [];
  const c = new Conversation({
    avatar,
    async *reply() { yield '返事です。'; },
    synthesize: async function* (ctx) { kinds.push(ctx.kind); yield { pcm: new Float32Array(640), sampleRate: 16000 }; },
    aizuchi: { pick: () => 'はい。' },
  });
  c.start();
  c.sendText('ねえ');
  for (let i = 0; i < 40 && kinds.length < 2; i++) await tick();
  assert.deepEqual(kinds, ['aizuchi', 'reply']);
  c.destroy();
});

test('conversation: a failing reply is reported and the conversation listens again', async () => {
  const avatar = fakeAvatar();
  const errors = [];
  const c = new Conversation({
    avatar,
    async *reply() { throw new Error('model down'); },
    synthesize: (ctx) => say(ctx.text),
  });
  c.on('error', (e) => errors.push(e.error.message));
  c.start();
  c.sendText('やあ');
  for (let i = 0; i < 20 && errors.length === 0; i++) await tick();
  assert.deepEqual(errors, ['model down']);
  assert.equal(c.turnState, 'listening');
  assert.equal(c.stats().errors, 1);
  c.destroy();
});

test('end pointer: speech starts after two loud frames, the turn ends after the quiet', () => {
  const e = new EndPointer({ frameMs: 32, endSilenceMs: 96, thresholdDbfs: -45, startFrames: 2 });
  const loud = new Float32Array(512).fill(0.2);
  const quiet = new Float32Array(512);
  assert.deepEqual(e.push(loud), []);
  assert.deepEqual(e.push(loud), [{ type: 'speech_start' }]);
  assert.equal(e.inSpeech, true);
  const events = [];
  for (let i = 0; i < 4; i++) events.push(...e.push(quiet));
  assert.deepEqual(events.map((x) => x.type), ['speech_end']);
  assert.equal(events[0].speakingMs, 64, 'the two loud frames');
  assert.equal(EndPointer.dbfs(quiet), -120);
});

test('end pointer: a VAD decision from outside drives the same hangover', () => {
  const e = new EndPointer({ frameMs: 32, endSilenceMs: 64, startFrames: 1 });
  assert.deepEqual(e.pushDecision(true), [{ type: 'speech_start' }]);
  assert.deepEqual(e.pushDecision(false), []);
  assert.deepEqual(e.pushDecision(false), [{ type: 'speech_end', speakingMs: 32 }]);
});

test('push queue: end, fail and close', async () => {
  const q = new PushQueue();
  q.push(1); q.push(2); q.end();
  const got = [];
  for await (const v of q) got.push(v);
  assert.deepEqual(got, [1, 2]);
  const f = new PushQueue();
  f.push(1); f.fail(new Error('boom'));
  const first = await f[Symbol.asyncIterator]().next();
  assert.deepEqual(first, { done: false, value: 1 });
  await assert.rejects(f[Symbol.asyncIterator]().next(), /boom/);
  const c = new PushQueue();
  c.push(1); c.close(); c.push(2);
  assert.equal(c.closed, true);
  assert.equal(c.dropped, 2);
});
