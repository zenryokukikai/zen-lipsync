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
