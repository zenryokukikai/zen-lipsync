// Unit tests of the shared core (node --test). Run after `pnpm build` (they import dist/).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { resolveOptions, OptionsError, WebGPUUnavailableError, AvatarPackError, NotImplementedError, logMel, MEL_SAMPLE_RATE, listAvatar, loadPackData, nodePackSource, speechMask, inferFrames } from '../dist/node/index.js';
import { requestGpuAdapter } from '../dist/runtime/avatar.js';

test('options: defaults and validation', () => {
  const o = resolveOptions({ avatar: './avatars/marine/avatar.json/' });
  assert.equal(o.avatar, './avatars/marine/avatar.json');
  assert.equal(o.weights, 'i8');
  assert.equal(o.idle, null);
  assert.equal(typeof o.onProgress, 'function');
  assert.equal(resolveOptions({ avatar: 'a', weights: 'f32', idle: 'angry' }).weights, 'f32');
  assert.equal(resolveOptions({ avatar: 'avatars/marine/avatar.json', assetsBaseUrl: 'https://cdn.example.com/za/' }).avatar, 'https://cdn.example.com/za/avatars/marine/avatar.json');
  assert.equal(resolveOptions({ avatar: './avatars/marine', assetsBaseUrl: '/assets' }).avatar, '/assets/avatars/marine');
  assert.equal(resolveOptions({ avatar: 'https://x.example/a.json', assetsBaseUrl: '/assets' }).avatar, 'https://x.example/a.json');
  for (const bad of [null, 1, [], {}, { avatar: '' }, { avatar: 'a', weights: 'fp16' }, { avatar: 'a', idle: '' }, { avatar: 'a', onProgress: 1 }, { avatar: 'a', backend: 'cpu' }, { avatar: 'a', assetsBaseUrl: '' }]) {
    assert.throws(() => resolveOptions(bad), OptionsError, JSON.stringify(bad));
  }
});

test('WebGPUUnavailableError: no navigator.gpu, and no adapter', async () => {
  const had = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  try {
    Object.defineProperty(globalThis, 'navigator', { value: {}, configurable: true });
    await assert.rejects(requestGpuAdapter(), (e) => e instanceof WebGPUUnavailableError && e.reason === 'no-navigator-gpu' && e.name === 'WebGPUUnavailableError');
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: { requestAdapter: async () => null } }, configurable: true });
    await assert.rejects(requestGpuAdapter(), (e) => e instanceof WebGPUUnavailableError && e.reason === 'no-adapter');
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: { requestAdapter: async () => { throw new Error('denied'); } } }, configurable: true });
    await assert.rejects(requestGpuAdapter(), WebGPUUnavailableError);
  } finally {
    if (had) Object.defineProperty(globalThis, 'navigator', had); else delete globalThis.navigator;
  }
});

/** A tiny avatar: avatar.json with two packs sharing ../models (2 source frames, feat 2x2x2). */
async function fixture() {
  const root = await mkdtemp(path.join(tmpdir(), 'zl-pack-'));
  const sha = (b) => createHash('sha256').update(b).digest('hex');
  const feat = Buffer.alloc(2 * 8 * 2);
  const ref = Buffer.from(new Float32Array(2 * 128).fill(0.5).buffer);
  const tone = JSON.stringify({ to_video: [[0], [0], [0]], from_video: [[0], [0], [0]] });
  for (const name of ['normal', 'angry']) {
    const d = path.join(root, name);
    await mkdir(d);
    await writeFile(path.join(d, 'features.f16'), feat);
    await writeFile(path.join(d, 'ref_latent.f32'), ref);
    await writeFile(path.join(d, 'tone.json'), tone);
    await writeFile(path.join(d, 'manifest.json'), JSON.stringify({
      format: 'zen-lipsync-avatar', version: 1, name, frame_size: [8, 8], fps: 25, source_frames: 2, walk: [0, 1, 1, 0],
      geo: {}, box: { y0: 0, y1: 1, x0: 0, x1: 1 }, feat_dims: [2, 2, 2], latent_window: 5,
      video: [{ src: 'idle.webm', type: 'video/webm' }], models: '../models',
      files: { 'features.f16': { bytes: feat.length, sha256: sha(feat) }, 'ref_latent.f32': { bytes: ref.length, sha256: sha(ref) } },
    }));
  }
  await mkdir(path.join(root, 'models'));
  await writeFile(path.join(root, 'avatar.json'), JSON.stringify({ format: 'zen-lipsync-avatar-set', version: 1, id: 't', name: { en: 'T' }, sources: { normal: 'normal', angry: 'angry' }, default: 'normal' }));
  return root;
}

test('pack loader (Node source): avatar.json, idle order, data sizes, models directory', async () => {
  const root = await fixture();
  let bytes = 0;
  const src = nodePackSource((n) => { bytes += n; });
  const l = await listAvatar(src, path.join(root, 'avatar.json'));
  assert.deepEqual(l.packs.map((p) => p.name), ['normal', 'angry']);
  assert.deepEqual((await listAvatar(src, path.join(root, 'avatar.json'), 'angry')).packs.map((p) => p.name), ['angry', 'normal']);
  const d = await loadPackData(src, l.packs[0]);
  assert.equal(d.feat.length, 16);
  assert.equal(d.refLatent.length, 256);
  assert.equal(d.modelDir, path.join(root, 'models'));
  assert.equal(bytes, 32 + 1024);
  const single = await listAvatar(src, path.join(root, 'normal'));
  assert.deepEqual(single.packs.map((p) => p.name), ['default']);
  await assert.rejects(listAvatar(src, path.join(root, 'avatar.json'), 'sad'), AvatarPackError);
  await assert.rejects(listAvatar(src, path.join(root, 'missing')), AvatarPackError);
  await writeFile(path.join(root, 'normal', 'features.f16'), Buffer.alloc(32, 1));
  await assert.rejects(loadPackData(src, l.packs[0]), /SHA-256 mismatch/);
});

test('audio features: log-mel shape for 1 s at 16 kHz', () => {
  const pcm = new Float32Array(MEL_SAMPLE_RATE);
  for (let i = 0; i < pcm.length; i++) pcm[i] = 0.3 * Math.sin((2 * Math.PI * 220 * i) / MEL_SAMPLE_RATE);
  const m = logMel(pcm);
  assert.equal(m.melFrames, 81);
  assert.equal(m.mel.length, 81 * 80);
  assert.equal(m.frames, 25);
  assert.equal(m.gather.length, 25);
  assert.ok(m.gather.every((g, i) => g === Math.round((i / 25) * 80)));
  assert.ok(m.mel.every(Number.isFinite));
});

test('speechMask: silence under a frame is not drawn, and a silent frame keeps its delta at zero', async () => {
  // silence: every band clamped to log(1e-5); a tone: well above it
  const silent = logMel(new Float32Array(MEL_SAMPLE_RATE));
  const loud = new Float32Array(MEL_SAMPLE_RATE);
  for (let i = 0; i < loud.length; i++) loud[i] = 0.3 * Math.sin((2 * Math.PI * 220 * i) / MEL_SAMPLE_RATE);
  const spoken = logMel(loud);
  const silentMask = speechMask(silent.mel, silent.melFrames, silent.gather, silent.frames);
  assert.ok(silentMask.every((v) => v === false), 'a silent sentence has no speech frames');
  const spokenMask = speechMask(spoken.mel, spoken.melFrames, spoken.gather, spoken.frames);
  assert.ok(spokenMask.some((v) => v === true), 'a tone has speech frames');
  // inferFrames: the deltas of the frames the mask says have no sound stay 0
  const featSize = 32 * 20 * 44;
  const avatar = { meta: { latent_window: 5 }, feat: new Uint16Array(featSize * 2), featSize, walk: (s) => s % 2, refLatent: new Float32Array(2 * 128) };
  const lips = { backend: 'cpu', weightBytes: 0, sentence: async () => new Float32Array(0), frame: async (f, z, n) => new Float32Array(n * 3 * 160 * 352).fill(7) };
  const zl = new Uint16Array(2 * 128);
  const all = await inferFrames(lips, avatar, [0, 1], zl, 0, 2, [false, true]);
  const plane = 3 * 160 * 352;
  assert.ok(all.subarray(0, plane).every((v) => v === 0), 'the silent frame is left as it is');
  assert.ok(all.subarray(plane, 2 * plane).every((v) => v === 7), 'the frame with sound keeps its delta');
});

test('tts: sentences are synthesised in order and spoken as one stream at 44.1 kHz', async () => {
  const { speakText, splitSentences, ttsChunks } = await import('../dist/tts/index.js');
  assert.deepEqual(splitSentences('はい。そうです！\n次です？'), ['はい。', 'そうです！', '次です？']);
  const said = [];
  const tts = { speak: async (t, o) => { said.push([t, o.emotion?.happy]); return { audio: new Float32Array(441 * t.length), sampleRate: 44100 }; } };
  const got = [];
  const avatar = { backend: 'webgpu', speakStream: async (it) => { for await (const c of it) got.push(c); } };
  const seen = [];
  await speakText(avatar, tts, 'こんにちは。全力マリンです。', { emotion: { happy: 1 }, onSentence: (s) => seen.push(s.text) });
  assert.deepEqual(said, [['こんにちは。', 1], ['全力マリンです。', 1]]);
  assert.deepEqual(seen, ['こんにちは。', '全力マリンです。']);
  assert.equal(got.length, 2);
  assert.ok(got.every((c) => c.sampleRate === 44100 && c.pcm instanceof Float32Array));
  // lazily: nothing is synthesised before the first chunk is asked for; eager: the first one starts at once
  said.length = 0;
  const it = ttsChunks(tts, 'あ。い。');
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(said.length, 0);
  await it.next();
  assert.equal(said.length, 1);
  said.length = 0;
  const eager = ttsChunks(tts, 'う。え。', {}, true);
  assert.deepEqual(said.map((x) => x[0]), ['う。']); // started before the first next()
  await eager.next();
  assert.deepEqual(said.map((x) => x[0]), ['う。']);
  await eager.next();
  assert.deepEqual(said.map((x) => x[0]), ['う。', 'え。']);
  await assert.rejects(speakText(avatar, {}, 'x'), OptionsError);
  await assert.rejects(speakText({ backend: 'cpu', speakStream: async () => {} }, tts, 'x'), OptionsError);
});

test('tts/node: frames / render / stream get the sentences as chunks with audioSampleRate 44100', async () => {
  const { speakTextFrames, renderText, streamText } = await import('../dist/tts/node.js');
  const tts = { speak: async (t) => ({ audio: new Float32Array(100), sampleRate: 44100 }) };
  const calls = [];
  const fake = (name) => (input, opts) => { calls.push([name, opts]); return (async function* () { for await (const c of input) yield c; })(); };
  const avatar = { frames: fake('frames'), render: async (input, opts) => { calls.push(['render', opts]); let n = 0; for await (const _ of input) n++; return { file: opts.out, frames: n, durationMs: 0 }; }, stream: fake('stream') };
  let n = 0; for await (const _ of speakTextFrames(avatar, tts, 'あ。い。う。', { realtime: true })) n++;
  assert.equal(n, 3);
  const r = await renderText(avatar, tts, 'あ。い。', { out: 'x.mp4', tts: { emotion: { sad: 1 } } });
  assert.equal(r.frames, 2);
  streamText(avatar, tts, 'あ。', { format: 'jpeg' });
  assert.deepEqual(calls.map((c) => [c[0], c[1].audioSampleRate]), [['frames', 44100], ['render', 44100], ['stream', 44100]]);
  assert.equal(calls[0][1].realtime, true);
  assert.equal('tts' in calls[1][1], false);
  assert.throws(() => speakTextFrames({}, tts, 'x'), OptionsError);
});

test('patchStrength: full on speech, nothing before it, fades over the 5 silent frames after it', async () => {
  const { patchStrength } = await import('../dist/runtime/player.js');
  const s = (str) => [...str].map((c) => c === '1');
  const r = (str) => [...patchStrength(s(str))].map((v) => +v.toFixed(3));
  // speech starts at full strength on its first frame (no fade-in); silence before it has no patch
  assert.deepEqual(r('0011'), [0, 0, 1, 1]);
  // after speech: 5/6 .. 1/6 on the following silent frames, then nothing
  assert.deepEqual(r('11000000'), [1, 1, 0.833, 0.667, 0.5, 0.333, 0.167, 0]);
  // a short pause inside speech: the frames before the pause stay at full strength
  assert.deepEqual(r('110011'), [1, 1, 0.833, 0.667, 1, 1]);
});

test('AutoPairing: next when nearly every patch draw is late, back to video only well below, display time on another clock -> next after 2 callbacks', async () => {
  const { AutoPairing } = await import('../dist/runtime/player.js');
  // late on every frame (iPhone Safari): next after enough samples
  const a = new AutoPairing();
  for (let i = 0; i < 24; i++) a.add(5);
  assert.equal(a.pairing, 'video');
  a.add(5);
  assert.equal(a.pairing, 'next');
  // late on about two thirds (Chromium on a desktop): stays on its own frame
  const b = new AutoPairing();
  for (let i = 0; i < 100; i++) b.add(i % 25 < 17 ? 0.3 : -0.3);
  assert.equal(b.pairing, 'video');
  // hysteresis: from next, 80 % late keeps next; at most 50 % late goes back
  for (let i = 0; i < 25; i++) a.add(i % 5 === 0 ? -1 : 1);
  assert.equal(a.pairing, 'next');
  for (let i = 0; i < 25; i++) a.add(i % 2 === 0 ? -1 : 1);
  assert.equal(a.pairing, 'video');
  // expectedDisplayTime on another clock (iPhone Safari): next after 2 callbacks in a row, idle ones count
  const c = new AutoPairing();
  c.clock(false);
  assert.equal(c.pairing, 'video');
  c.clock(false);
  assert.equal(c.pairing, 'next');
  // one stray callback on another clock between same-clock ones: no switch
  const e = new AutoPairing();
  for (let i = 0; i < 50; i++) e.clock(i % 3 !== 0);
  assert.equal(e.pairing, 'video');
  // a browser measured once keeps the 25-frame rule: later unmeasurable callbacks do not switch it
  const f = new AutoPairing();
  f.clock(true); f.add(-1);
  for (let i = 0; i < 10; i++) f.clock(false);
  assert.equal(f.pairing, 'video');
});

test('LipSyncPlayer: silent frames draw nothing and every patch bitmap is closed once (no drawImage of a closed bitmap)', async () => {
  const { LipSyncPlayer } = await import('../dist/runtime/player.js');
  // a bitmap that throws when drawn after close(), as browsers do (InvalidStateError)
  let closes = 0;
  const bitmap = () => { let closed = false; return { width: 4, height: 4, get closed() { return closed; }, close() { if (closed) throw new Error('closed twice'); closed = true; closes++; } }; };
  const drawn = [];
  const g = { drawImage(b) { if (b.closed) throw new Error('InvalidStateError: drawImage of a closed bitmap'); drawn.push(b); }, clearRect() {} };
  const walkLen = 100;
  const p = Object.create(LipSyncPlayer.prototype);
  Object.assign(p, {
    shownLoop: { avatar: { meta: { walk: { length: walkLen }, fps: 25 } } },
    canvas: { getContext: () => g, width: 10, height: 10 },
    ctx: { currentTime: 0 },
    prevClips: [], lastStep: -1, shownPatch: null, sync: 'video', lastDraw: null, fade: null,
    drawCounts: { frames: 0, patches: 0, skipped: 0, skippedBy: { idle: 0, start: 0, speech: 0, end: 0 } },
    drawTimes: { video: [], patch: [] },
  });
  // speech, a pause, speech, a pause (patch strength 0 on the long silent runs)
  const speech = [...Array(40)].map((_, f) => !((f >= 10 && f < 22) || f >= 32));
  const { patchStrength } = await import('../dist/runtime/player.js');
  const strength = patchStrength(speech);
  const patches = new Map();
  for (let f = 0; f < 40; f++) patches.set(f, strength[f] > 0 ? { bitmap: bitmap(), x: 0, y: 0 } : null);
  const clip = { soundAt: 0, duration: 40 / 25, frames: 40, walkStart: 0, patches, prep: { speech, strength } };
  p.clip = clip;
  for (let f = 0; f < 40; f++) { p.ctx.currentTime = f / 25; p.onVideoFrame(f); }
  assert.equal(drawn.length, [...strength].filter((k) => k > 0).length);
  // every patch drawn is closed once when the video passes it (the clip ends on silent frames, so all of them are passed)
  assert.equal(closes, drawn.length);
});

// ---- idle switch with two <video> elements (shown + standby), on fake elements (no DOM) ----

/**
 * A fake <video>: play() resolves only after start() (a user gesture) on this element, as iOS Safari in Low Power Mode;
 * once started it plays later outside a gesture too, with any file loaded. A playing element presents frames (rVFC)
 * every 10 ms. opts: never load (no loadeddata), no 'seeked', play rejects always.
 */
/** An inline style with the CSSStyleDeclaration calls copyLayout uses (opacity kept as a property, as the player sets it). */
function fakeStyle(init = {}) {
  const props = new Map(Object.entries(init));
  const st = {
    get length() { return props.size; },
    item(i) { return [...props.keys()][i]; },
    getPropertyValue(k) { return props.get(k) ?? ''; },
    getPropertyPriority() { return ''; },
    setProperty(k, v) { props.set(k, v); },
  };
  Object.defineProperty(st, 'opacity', { get() { return props.get('opacity') ?? ''; }, set(v) { props.set('opacity', v); } });
  return st;
}

function fakeVideo(name, opts = {}) {
  const ls = new Map();
  const v = {
    name, paused: true, readyState: 0, networkState: 0, currentTime: 0, style: fakeStyle({ opacity: '1' }), muted: false, loop: false, playsInline: false, preload: '',
    activated: false, inGesture: false, src: null, cbs: [], plays: 0,
    addEventListener(t, f) { if (!ls.has(t)) ls.set(t, new Set()); ls.get(t).add(f); },
    removeEventListener(t, f) { ls.get(t)?.delete(f); },
    emit(t) { for (const f of [...(ls.get(t) ?? [])]) f(); },
    replaceChildren(...c) { v.src = c.map((x) => x.src).join(','); },
    removeAttribute() {}, remove() {},
    load() { v.readyState = 0; v.paused = true; if (!opts.neverLoads) setTimeout(() => { v.readyState = 4; v.emit('loadeddata'); }, 5); },
    play() {
      v.plays++;
      if (opts.alwaysRefuse || !(v.activated || v.inGesture)) return Promise.reject(Object.assign(new Error('not allowed'), { name: 'NotAllowedError' }));
      v.activated = true; v.paused = false; return Promise.resolve();
    },
    pause() { v.paused = true; },
    requestVideoFrameCallback(cb) { v.cbs.push(cb); return v.cbs.length; },
  };
  Object.defineProperty(v, 'currentTime', { get() { return v._t ?? 0; }, set(t) { v._t = t; if (!opts.noSeeked) setTimeout(() => v.emit('seeked'), 3); }, configurable: true });
  v.ticker = setInterval(() => { if (!v.paused && v.readyState >= 2) { v._t = (v._t ?? 0) + 0.04; const cbs = v.cbs; v.cbs = []; for (const cb of cbs) cb(performance.now(), { mediaTime: v._t, presentedFrames: 0 }); } }, 10);
  return v;
}

async function fakePlayer(names, opts = {}) {
  const { LipSyncPlayer } = await import('../dist/runtime/player.js');
  globalThis.document ??= { createElement: () => ({}), visibilityState: 'visible' };
  globalThis.performance ??= { now: () => Date.now() };
  const loops = names.map((name) => ({ name, avatar: { base: `/a/${name}`, meta: { fps: 25, walk: { length: 50 }, frame_size: [8, 8], video: [{ src: `${name}.webm`, type: 'video/webm' }] } }, comp: {} }));
  const page = fakeVideo('page');
  // the page fits its <video> after create() (sets object-position on it and on the canvas)
  page.style.setProperty('object-position', '46.999% 18%');
  page.style.setProperty('object-fit', 'cover');
  const standby = names.length > 1 ? fakeVideo('standby', opts.standby ?? {}) : null;
  const pairs = { version: 1, sources: names, pairs: {} };
  const table = { to: [...Array(50).keys()], dist: Array(50).fill(0) };
  for (const a of names) for (const b of names) if (a !== b) pairs.pairs[`${a}->${b}`] = table;
  const p = Object.create(LipSyncPlayer.prototype);
  for (const [k, val] of Object.entries(new (class {
    constructor() {
      Object.assign(this, { loops: new Map(loops.map((l) => [l.name, l])), shownLoop: loops[0], shown: { video: page, loop: loops[0] }, standby: standby ? { video: standby, loop: null } : null, pageVideo: page,
        canvas: { getContext: () => ({ drawImage() {}, clearRect() {} }), width: 8, height: 8 }, ctx: { currentTime: 0, state: 'running', resume: async () => {} }, pairs, set: null,
        prevClips: [], clip: null, lastStep: -1, shownPatch: null, sync: 'video', lastDraw: null, fade: null, switching: false, watchGen: 0, speakGen: 0,
        drawCounts: { frames: 0, patches: 0, skipped: 0, skippedBy: { idle: 0, start: 0, speech: 0, end: 0 } }, drawTimes: { video: [], patch: [] }, drawErrors: { count: 0, last: '' },
        lateCounts: { measured: 0, late: 0, clockMismatch: 0 }, auto: { pairing: 'video', clock() {}, add() {} },
        switchCounts: { done: 0, failed: 0, reasons: { cancelled: 0, 'load-timeout': 0, 'play-refused': 0, 'no-frame': 0, 'seek-timeout': 0 }, last: '' },
        watchdog: { kicks: 0, last: '' }, standbyPlaying: { speechFrames: 0 }, lastCheckAt: 0, lastFrameAt: 0, framesSeen: false, pausedByUser: false, watchdogTimer: null, cancelSwitch: null, timing: { frames: 0, inferMs: 0 } });
    }
  })())) p[k] = val;
  page.replaceChildren({ src: `/a/${names[0]}/${names[0]}.webm` }); page.load();
  page.inGesture = true; await page.play(); page.inGesture = false;
  p.watchVideo();
  const gesture = () => { for (const v of [page, standby]) if (v) v.inGesture = true; p.resume(); for (const v of [page, standby]) if (v) v.inGesture = false; };
  const cleanup = () => { clearInterval(page.ticker); if (standby) clearInterval(standby.ticker); p.destroy?.call({ ...p, stop() {}, ctx: { close: async () => {} }, lips: {} }); if (p.watchdogTimer) clearInterval(p.watchdogTimer); };
  return { p, page, standby, gesture, cleanup };
}

const frameNow = async (v) => { const n = v.cbs.length; for (let i = 0; i < 50 && v.paused === false && v.cbs.length === n; i++) await new Promise((r) => setTimeout(r, 10)); };

test('idle switch: two <video> elements whatever the loop count; started in the gesture, a switch outside it plays (Low Power Mode); cycles 3 loops', async () => {
  const { p, page, standby, gesture, cleanup } = await fakePlayer(['normal', 'angry', 'sad']);
  try {
    gesture();
    await new Promise((r) => setTimeout(r, 40));
    assert.equal(standby.activated, true);
    assert.equal(standby.paused, true, 'the standby is paused again after the gesture');
    const order = ['angry', 'normal', 'sad', 'angry', 'sad', 'normal', 'angry'];
    let shownVideos = new Set();
    for (const to of order) {
      assert.equal(await p.switchIdle(to), true, `switch to ${to}: ${p.switchCounts.last}`);
      assert.equal(p.idle, to);
      // the patches and the walk come from the loop shown now, and its <video> has the page's layout (else the patch
      // lands off the face)
      assert.equal(p.avatar.base, `/a/${to}`);
      assert.equal(p.shown.loop.name, to);
      assert.equal(p.shown.video.style.getPropertyValue('object-position'), '46.999% 18%');
      assert.equal(p.shown.video.style.getPropertyValue('object-fit'), 'cover');
      shownVideos.add(p.shown.video.name);
      assert.equal(p.shown.video.paused, false, 'the shown video plays');
      assert.equal(p.standby.video.paused, true, 'the old one is paused as standby');
      assert.equal(p.shown.video.style.opacity, '1');
      assert.equal(p.standby.video.style.opacity, '0');
      // the frame callbacks follow the shown video
      const before = p.lastFrameAt; await new Promise((r) => setTimeout(r, 40)); assert.ok(p.lastFrameAt > before, 'frame callbacks on the shown video');
    }
    assert.deepEqual([...shownVideos].sort(), ['page', 'standby']);
    assert.equal(p.switchCounts.done, order.length);
    assert.equal(p.switchCounts.failed, 0);
  } finally { cleanup(); }
});

test('idle switch: no gesture (Low Power Mode refuses play): given up as play-refused, the shown loop goes on, its frame callbacks too', async () => {
  const { p, page, standby, cleanup } = await fakePlayer(['normal', 'angry']);
  try {
    assert.equal(await p.switchIdle('angry'), false);
    assert.equal(p.switchCounts.reasons['play-refused'], 1);
    assert.equal(p.idle, 'normal');
    assert.equal(p.shown.video, page);
    assert.equal(page.paused, false);
    assert.equal(standby.style.opacity, '0');
    const before = p.lastFrameAt; await new Promise((r) => setTimeout(r, 40)); assert.ok(p.lastFrameAt > before);
    assert.equal(p.switching, false);
  } finally { cleanup(); }
});

test('idle switch: a loop that never loads, a seek that never ends, a video that never presents: each given up with its reason within its bound', async () => {
  for (const [opt, reason] of [[{ neverLoads: true }, 'load-timeout'], [{ noSeeked: true }, 'seek-timeout']]) {
    const { p, gesture, cleanup } = await fakePlayer(['normal', 'angry'], { standby: opt });
    try {
      gesture();
      await new Promise((r) => setTimeout(r, 20));
      const t0 = Date.now();
      assert.equal(await p.switchIdle('angry'), false);
      assert.equal(p.switchCounts.reasons[reason], 1, JSON.stringify(p.switchCounts));
      assert.ok(Date.now() - t0 < 5000);
      assert.equal(p.idle, 'normal');
    } finally { cleanup(); }
  }
  // started, but never presents a frame (decoder held): no-frame after about 1 s
  const { p, standby, gesture, cleanup } = await fakePlayer(['normal', 'angry']);
  try {
    gesture();
    await new Promise((r) => setTimeout(r, 20));
    standby.requestVideoFrameCallback = () => 0;
    clearInterval(standby.ticker);
    assert.equal(await p.switchIdle('angry'), false);
    assert.equal(p.switchCounts.reasons['no-frame'], 1);
    assert.equal(p.idle, 'normal');
  } finally { cleanup(); }
});

test('idle switch: speech arriving before the crossfade drops the switch at once (cancelled), not after its waits', async () => {
  const { p, gesture, cleanup } = await fakePlayer(['normal', 'angry'], { standby: { noSeeked: true } });
  try {
    gesture();
    await new Promise((r) => setTimeout(r, 20));
    const sw = p.switchIdle('angry');
    await new Promise((r) => setTimeout(r, 30));
    const t0 = Date.now();
    p.cancelSwitch?.();
    assert.equal(await sw, false);
    assert.ok(Date.now() - t0 < 100, 'dropped at once');
    assert.equal(p.switchCounts.reasons.cancelled, 1);
  } finally { cleanup(); }
});

test('watchdog: the shown video stopped by the browser (no frame callbacks) is played again', async () => {
  const { p, page, cleanup } = await fakePlayer(['normal']);
  try {
    assert.equal(p.standby, null, 'one loop: no standby video');
    await new Promise((r) => setTimeout(r, 50));
    page.paused = true;
    await new Promise((r) => setTimeout(r, 1200));
    assert.ok(p.watchdog.kicks >= 1);
    assert.equal(page.paused, false);
    const before = p.lastFrameAt; await new Promise((r) => setTimeout(r, 40)); assert.ok(p.lastFrameAt > before);
  } finally { cleanup(); }
});
