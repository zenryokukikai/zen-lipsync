// prepare() / play() (1.3.0): the sound starts at once and the mouth joins on the video frame its patch is made for;
// a PreparedSpeech plays once, on the avatar that prepared it; speechstart carries the time the sound is heard.
// Run without a browser: a LipSyncPlayer on fake audio / video / models, and create() on a fake player.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { melGather, logMel, OptionsError } from '../dist/index.js';

test('melGather: shift 0 is logMel\'s own gather; a shift moves every frame later by that many video frames', () => {
  const mel = logMel(new Float32Array(16000));
  const g0 = melGather(mel.melFrames, 0);
  assert.equal(g0.frames, mel.frames);
  assert.deepEqual(g0.gather, mel.gather);
  // 1.5 frames later: frame i takes the mel row of video time (i + 1.5) / 25 s (80 rows per second), one or two frames fewer
  const g = melGather(mel.melFrames, 1.5);
  assert.equal(g.frames, mel.frames - 2);
  assert.deepEqual(g.gather.slice(0, 3), [5, 8, 11].map((_, i) => Math.round(((i + 1.5) / 25) * 80)));
  assert.ok(g.gather.every((r) => r >= 0 && r < mel.melFrames));
});

/** A LipSyncPlayer on fakes: audio clock set by the test, models that answer when the test lets them. */
async function fakePlayer() {
  const { LipSyncPlayer } = await import('../dist/runtime/player.js');
  globalThis.performance ??= { now: () => Date.now() };
  let releaseSentence = null;
  const starts = [];
  const ctx = {
    currentTime: 10, state: 'running', outputLatency: 0.02,
    createBuffer: (ch, n, rate) => ({ duration: n / rate, copyToChannel() {} }),
    createBufferSource: () => { const s = { connect() {}, start(t) { starts.push({ t, at: ctx.currentTime }); }, stop() {}, onended: null }; return s; },
    resume: async () => {},
    destination: {},
  };
  const avatar = { meta: { fps: 25, walk: { length: 100 }, latent_window: 5 }, feat: new Uint16Array(100 * 4), featSize: 4, refLatent: new Float32Array(100 * 128), walk: (s) => ((s % 100) + 100) % 100, frame: async () => ({}) };
  const lips = {
    sentence: (mel, ref, T) => new Promise((r) => { releaseSentence = () => r(new Float32Array(T * 128)); }),
    frame: async (feat, z, n) => new Float32Array(n * 3 * 4),
  };
  const p = Object.create(LipSyncPlayer.prototype);
  Object.assign(p, {
    shownLoop: { avatar, comp: { frame: () => ({ bitmap: { width: 1, height: 1, close() {} }, x: 0, y: 0 }) } },
    shown: { video: { currentTime: 4.012 } }, lips, ctx, prevClips: [], clip: null, speakGen: 0, switching: false, cancelSwitch: null,
    canvas: { getContext: () => ({ clearRect() {} }) }, shownPatch: null, timing: { frames: 0, inferMs: 0 }, playFirstMs: [], lead: 8,
  });
  return { p, ctx, starts, sentenceRan: () => releaseSentence !== null, release: () => releaseSentence?.() };
}

test('play(): the sound is started at the audio clock\'s now, before the models run; the mouth is planned a few frames on', async () => {
  const { p, ctx, starts, sentenceRan, release } = await fakePlayer();
  const pcm = new Float32Array(16000).map((_, i) => Math.sin(i / 5) * 0.3);
  const plan = await p.prepareSpeech(pcm, 16000, { warm: false });
  let heard = null;
  const done = p.play(plan, (t) => { heard = t; });
  // let play() reach the sentence model
  for (let i = 0; i < 20 && !sentenceRan(); i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(starts.length, 1, 'the sound is started');
  assert.equal(starts[0].t, 10, 'at the AudioContext time of the call (no wait for the models)');
  assert.ok(sentenceRan(), 'the sentence model runs after the sound is started');
  assert.equal(heard.audioTime, 10);
  assert.ok(Number.isFinite(heard.performanceTime));
  // no output timestamp on the fake context: the reported output latency (20 ms) is the device's delay
  assert.ok(Math.abs(heard.outputDelayMs - 20) < 1e-6);
  // video frame 100 (4.012 s, 0.3 into it) is on screen: frame 0 of the clip is a later video frame, `shift` frames after the sound
  const clip = p.clip;
  assert.ok(clip.prep.shift > 0 && clip.prep.shift < 3, `shift ${clip.prep.shift}`);
  assert.equal((clip.walkStart - 100 + 100) % 100, Math.round(clip.prep.shift + 0.3));
  release();
  ctx.currentTime = 11.5; // the sound has ended
  clip.source?.onended?.();
  p.stop();
  await done;
});

test('play(): a stop() while the models run ends play() without drawing or a second sound', async () => {
  const { p, starts, sentenceRan, release } = await fakePlayer();
  const plan = await p.prepareSpeech(new Float32Array(8000), 16000, { warm: false });
  const done = p.play(plan);
  for (let i = 0; i < 20 && !sentenceRan(); i++) await new Promise((r) => setTimeout(r, 0));
  p.stop();
  release();
  await done;
  assert.equal(starts.length, 1);
  assert.equal(p.clip, null);
});

// ---- create(): the public handle, on a fake player ----

async function fakeAvatar(name) {
  const { LipSyncPlayer } = await import('../dist/runtime/player.js');
  const { create } = await import('../dist/browser/create.js');
  const el = () => ({ style: { cssText: '' }, setAttribute() {}, append() {}, appendChild() {}, remove() {} });
  Object.defineProperty(globalThis, 'document', { value: { createElement: el }, configurable: true });
  Object.defineProperty(globalThis, 'navigator', { value: { gpu: { requestAdapter: async () => ({ requestDevice: async () => ({}) }) } }, configurable: true });
  const calls = [];
  const fake = {
    avatar: { meta: { frame_size: [8, 8] } }, idle: 'normal', idleLoops: ['normal'], speaking: false,
    prepareSpeech: async (pcm, sampleRate) => ({ pcm, sampleRate, mel: { frames: Math.floor((pcm.length / sampleRate) * 25) } }),
    play: async (plan, onStart) => { calls.push(['play', name, plan]); onStart({ audioTime: 3, performanceTime: 4000, outputDelayMs: 20 }); },
    speak: async (pcm, sr, onStart) => { calls.push(['speak', name]); onStart({ audioTime: 5, performanceTime: 6000, outputDelayMs: 20 }); },
    stop() {}, destroy() {}, outputDelayMs: () => 20,
  };
  const orig = LipSyncPlayer.create;
  LipSyncPlayer.create = async () => fake;
  try { return { a: await create({ container: el(), avatar: `/a/${name}/avatar.json` }), calls }; } finally { LipSyncPlayer.create = orig; }
}

test('prepare()/play(): plays once; a disposed or another avatar\'s PreparedSpeech is refused; speechstart has the heard time', async () => {
  const { a, calls } = await fakeAvatar('one');
  const { a: b } = await fakeAvatar('two');
  const audio = { pcm: new Float32Array(48000), sampleRate: 48000 };
  const events = [];
  a.on('speechstart', (e) => events.push(['start', e]));
  a.on('speechend', (e) => events.push(['end', e]));
  const prep = await a.prepare(audio);
  assert.equal(prep.frames, 25);
  assert.equal(prep.durationMs, 1000);
  assert.equal(prep.playable, true);
  assert.equal(prep.outputDelayMs, 20);
  await assert.rejects(b.play(prep), OptionsError, 'another avatar');
  await a.play(prep);
  assert.equal(prep.playable, false);
  assert.deepEqual(events, [['start', { frames: 25, audioTime: 3, performanceTime: 4000, outputDelayMs: 20 }], ['end', { interrupted: false }]]);
  await assert.rejects(a.play(prep), /already played/);
  const prep2 = await a.prepare(audio);
  prep2.dispose();
  assert.equal(prep2.playable, false);
  await assert.rejects(a.play(prep2), /already disposed/);
  await assert.rejects(a.prepare({ pcm: [], sampleRate: 1 }), OptionsError);
  // speak() is unchanged and reports the heard time too
  await a.speak(audio);
  assert.deepEqual(events.at(-2), ['start', { frames: 25, audioTime: 5, performanceTime: 6000, outputDelayMs: 20 }]);
  assert.deepEqual(calls.map((c) => c[0]), ['play', 'speak']);
  a.destroy(); b.destroy();
});

test('Node: prepare() and play() are for the browser player (OptionsError)', async () => {
  const { NodeAvatar } = await import('../dist/node/avatar.js');
  const n = Object.create(NodeAvatar.prototype);
  await assert.rejects(n.prepare({ pcm: new Float32Array(1), sampleRate: 1 }), OptionsError);
  await assert.rejects(n.play({}), OptionsError);
});
