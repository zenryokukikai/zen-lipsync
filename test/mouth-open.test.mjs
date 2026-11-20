// How wide the mouth opens (core/mouth-open.ts): node --test, after `pnpm build` (imports dist/).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { checkMouthOpen, scaleMouthOpen, OptionsError, AvatarPackError, resolveOptions, listAvatar, loadPackData, nodePackSource, MOUTH_OPEN_DIR_FILE } from '../dist/node/index.js';
import { resolveMouthOpen } from '../dist/core/mouth-open.js';
import { sentenceLatents } from '../dist/core/lipsync.js';

const N = 128;
const unit = (k) => { const d = new Float32Array(N); d[k] = 1; return d; };
const ones = () => new Float32Array(N).fill(1);
const zeros = () => new Float32Array(N);

test('mouthOpen: scale 1 leaves the latents byte for byte', () => {
  const z = Float32Array.from({ length: 3 * N }, (_, i) => Math.sin(i));
  const before = Buffer.from(z.buffer.slice(0));
  scaleMouthOpen(z, new Float32Array(3 * N), 3, ones(), zeros(), unit(0), { scale: 1 });
  assert.ok(Buffer.from(z.buffer).equals(before));
  scaleMouthOpen(z, new Float32Array(3 * N), 3, ones(), zeros(), null, { scale: 1 });
  assert.ok(Buffer.from(z.buffer).equals(before));
});

test('mouthOpen: below 1 only the opening shrinks; the rest of the move and a closing frame are kept', () => {
  // closed mouth at 0 (ref 0, shift 0); the move: +2 along the opening direction (k=0), +1 elsewhere (k=1)
  const z1 = new Float32Array(N); z1[0] = 2; z1[1] = 1;
  scaleMouthOpen(z1, new Float32Array(N), 1, ones(), zeros(), unit(0), { scale: 0.5 });
  assert.equal(z1[0], 1); assert.equal(z1[1], 1);
  // a frame closing (negative along the opening) is left as it is
  const z3 = new Float32Array(N); z3[0] = -1;
  scaleMouthOpen(z3, new Float32Array(N), 1, ones(), zeros(), unit(0), { scale: 0.5 });
  assert.equal(z3[0], -1);
  // the closed mouth is the frame's reference in raw space (ref * scale + shift): ref 0.5, scale 2, shift 1 -> 2
  const z4 = new Float32Array(N).fill(3);
  scaleMouthOpen(z4, new Float32Array(N).fill(0.5), 1, new Float32Array(N).fill(2), new Float32Array(N).fill(1), unit(0), { scale: 0.5 });
  assert.equal(z4[0], 2.5); assert.equal(z4[1], 3);
});

test('mouthOpen: settings are checked; a mode and an avatar without the direction are refused (not ignored)', () => {
  assert.deepEqual(checkMouthOpen({ scale: 0.6 }, 'x'), { scale: 0.6 });
  for (const bad of [null, 1, [], {}, { scale: 0 }, { scale: 2.5 }, { scale: NaN }, { scale: Infinity }, { scale: '0.5' }, { scale: 0.5, mode: 'opening' }, { scale: 0.5, mode: 'all' }, { scale: 0.5, extra: 1 }]) {
    assert.throws(() => checkMouthOpen(bad, 'x'), OptionsError, JSON.stringify(bad));
  }
  assert.equal(resolveMouthOpen({ scale: 1 }, false, 'x'), null);
  assert.equal(resolveMouthOpen(null, false, 'x'), null);
  assert.throws(() => resolveMouthOpen({ scale: 0.5 }, false, 'x'), /cannot scale the mouth/);
  assert.deepEqual(resolveMouthOpen({ scale: 0.5 }, true, 'x'), { scale: 0.5 });
  assert.throws(() => scaleMouthOpen(new Float32Array(N), new Float32Array(N), 1, ones(), zeros(), null, { scale: 0.5 }), OptionsError);
  // create() option
  assert.deepEqual(resolveOptions({ avatar: 'a', mouthOpen: { scale: 0.7 } }).mouthOpen, { scale: 0.7 });
  assert.equal(resolveOptions({ avatar: 'a', mouthOpen: { scale: 1 } }).mouthOpen, null);
  assert.throws(() => resolveOptions({ avatar: 'a', mouthOpen: { scale: 3 } }), OptionsError);
  assert.throws(() => resolveOptions({ avatar: 'a', mouthOpen: { scale: 0.7, mode: 'all' } }), OptionsError);
});

async function pack({ dir }) {
  const root = await mkdtemp(path.join(tmpdir(), 'zl-mo-'));
  const sha = (b) => createHash('sha256').update(b).digest('hex');
  const feat = Buffer.alloc(2 * 8 * 2);
  const ref = Buffer.from(new Float32Array(2 * N).buffer);
  const files = { 'features.f16': { bytes: feat.length, sha256: sha(feat) }, 'ref_latent.f32': { bytes: ref.length, sha256: sha(ref) } };
  await writeFile(path.join(root, 'features.f16'), feat);
  await writeFile(path.join(root, 'ref_latent.f32'), ref);
  if (dir) { await writeFile(path.join(root, MOUTH_OPEN_DIR_FILE), dir); files[MOUTH_OPEN_DIR_FILE] = { bytes: dir.length, sha256: sha(dir) }; }
  await writeFile(path.join(root, 'manifest.json'), JSON.stringify({
    format: 'zen-lipsync-avatar', version: 1, name: 'p', frame_size: [8, 8], fps: 25, source_frames: 2, walk: [0, 1],
    geo: {}, box: { y0: 0, y1: 1, x0: 0, x1: 1 }, feat_dims: [2, 2, 2], latent_window: 5, video: [{ src: 'idle.webm', type: 'video/webm' }], files,
  }));
  return root;
}

test('pack: mouth_open_dir.f32 is optional; read and checked when there', async () => {
  const src = nodePackSource();
  const none = await loadPackData(src, (await listAvatar(src, await pack({}))).packs[0]);
  assert.equal(none.mouthOpenDir, null);
  const ok = await loadPackData(src, (await listAvatar(src, await pack({ dir: Buffer.from(unit(5).buffer) }))).packs[0]);
  assert.equal(ok.mouthOpenDir[5], 1);
  await assert.rejects(loadPackData(src, (await listAvatar(src, await pack({ dir: Buffer.from(ones().buffer) }))).packs[0]), AvatarPackError);
  await assert.rejects(loadPackData(src, (await listAvatar(src, await pack({ dir: Buffer.alloc(12) }))).packs[0]), AvatarPackError);
});

test('sentenceLatents: the setting changes the latents the frame model gets; none / scale 1 keeps them', async () => {
  const lips = { backend: 'cpu', weightBytes: 0, latent: { scale: ones(), shift: zeros() }, frame: async () => new Float32Array(0),
    sentence: async (_mel, _ref, T) => { const z = new Float32Array(T * N); for (let t = 0; t < T; t++) z[t * N] = 1; return z; } };
  const avatar = { meta: { latent_window: 5 }, feat: new Uint16Array(0), featSize: 0, refLatent: new Float32Array(2 * N), mouthOpenDir: unit(0), walk: (s) => s % 2 };
  const s = { startStep: 0, frames: 2, melFrames: 2, gather: [0, 1], mel: new Float32Array(2 * 80) };
  const plain = await sentenceLatents(lips, avatar, s);
  assert.deepEqual(await sentenceLatents(lips, avatar, s, { scale: 1 }), plain);
  const half = await sentenceLatents(lips, avatar, s, { scale: 0.5 });
  assert.equal(half[0], 0x3800);   // fp16 0.5 (was 1.0 = 0x3c00)
  assert.equal(plain[0], 0x3c00);
});
