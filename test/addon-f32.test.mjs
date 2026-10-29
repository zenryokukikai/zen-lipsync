// The native float32 frame model (native/x86/lipf.c through the addon): its output equals a plain JavaScript float32
// implementation of the WebGPU frame model (src/runtime/wgsl.ts buildFrame and its shaders) on random weights, on both
// kernels, and it gives the same bytes when several threads run it at once. Needs no avatar files; skips where the
// linux-x64 addon is not built or the CPU has neither kernel.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { Worker } from 'node:worker_threads';
import { nativeAddonPath } from '../dist/node/index.js';

const path = nativeAddonPath();
const addon = path ? createRequire(import.meta.url)(path) : null;
const skip = !addon || typeof addon.createF32 !== 'function' ? 'native addon with the float32 frame model not built' : false;

// inp, body.0-5, mid, up.0.0, up.0.1, up.1.0, up.1.1, up.2.0, up.2.1, out: [C, O, H, W, input act (0/1 SiLU/2 ReLU), mode]
const CONVS = [[40, 64, 20, 44, 0, 0], [64, 64, 20, 44, 1, 0], [64, 64, 20, 44, 1, 0], [64, 64, 20, 44, 1, 0], [64, 64, 20, 44, 1, 0],
  [64, 64, 20, 44, 1, 0], [64, 64, 20, 44, 1, 0], [64, 64, 20, 44, 1, 0], [64, 64, 20, 44, 0, 0], [64, 128, 20, 44, 2, 1],
  [32, 32, 40, 88, 0, 0], [32, 64, 40, 88, 2, 1], [16, 16, 80, 176, 0, 0], [16, 32, 80, 176, 2, 1], [8, 3, 160, 352, 2, 2]];

function rng(seed) { let s = seed >>> 0; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296 - 0.5); }

/** Random weights in the blob layout (zfc0.w [256][640], zfc0.b, zfc2.w [7040][256], zfc2.b, per conv w [O4][C][9][4] and b [O4*4], win). */
function randomBlob(n) {
  const r = rng(7), b = new Float32Array(n);
  let at = 0;
  const fill = (len, scale) => { for (let i = 0; i < len; i++) b[at++] = r() * scale; };
  fill(256 * 640, 0.02); fill(256, 0.2); fill(7040 * 256, 0.1); fill(7040, 0.2);
  for (const [C, O] of CONVS) { const O4 = Math.ceil(O / 4); fill(O4 * C * 36, 2 / Math.sqrt(C * 9)); fill(O4 * 4, 0.2); }
  for (let i = 0; i < 160 * 352; i++) b[at++] = 0.5 + r();
  assert.equal(at, n);
  return b;
}

const act = (v, a) => (a === 1 ? v / (1 + Math.exp(-v)) : a === 2 ? Math.max(v, 0) : v);

/** The WebGPU frame model in plain JavaScript (NCHW, float64 sums): one frame. */
function reference(blob, feat, z) {
  let at = 0;
  const take = (len) => blob.subarray(at, (at += len));
  const g0w = take(256 * 640), g0b = take(256), g2w = take(7040 * 256), g2b = take(7040);
  const g0 = new Float64Array(256), g1 = new Float64Array(7040);
  for (let o = 0; o < 256; o++) { let s = 0; for (let i = 0; i < 640; i++) s += g0w[o * 640 + i] * z[i]; g0[o] = act(s * 20 + g0b[o], 1); }
  for (let o = 0; o < 7040; o++) { let s = 0; for (let i = 0; i < 256; i++) s += g2w[o * 256 + i] * g0[i]; g1[o] = s + g2b[o]; }
  let x = new Float64Array(40 * 880);
  for (let i = 0; i < 32 * 880; i++) x[i] = feat[i];
  for (let i = 0; i < 8 * 880; i++) x[32 * 880 + i] = g1[i];
  const keep = {};
  const residual = { 2: 'X0', 4: 'E', 6: 'J' }, save = { 0: 'X0', 2: 'E', 4: 'J' };
  const win = blob.subarray(blob.length - 160 * 352);
  let out = null;
  CONVS.forEach(([C, O, H, W, a, mode], l) => {
    const O4 = Math.ceil(O / 4), w = take(O4 * C * 36), b = take(O4 * 4), HW = H * W;
    const xa = x.map((v) => act(v, a));
    const y = new Float64Array(mode === 2 ? 3 * HW : O * HW);
    for (let oc = 0; oc < O; oc++) {
      const o4 = oc >> 2, j = oc & 3;
      for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
        let s = b[oc];
        for (let c = 0; c < C; c++) for (let ky = 0; ky < 3; ky++) {
          const iy = py + ky - 1; if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < 3; kx++) {
            const ix = px + kx - 1; if (ix < 0 || ix >= W) continue;
            s += xa[c * HW + iy * W + ix] * w[((o4 * C + c) * 9 + ky * 3 + kx) * 4 + j];
          }
        }
        if (mode === 0) y[oc * HW + py * W + px] = s + (residual[l] ? keep[residual[l]][oc * HW + py * W + px] : 0);
        else if (mode === 1) y[o4 * HW * 4 + (py * 2 + (j >> 1)) * W * 2 + px * 2 + (j & 1)] = s;
        else y[oc * HW + py * W + px] = s * win[py * W + px] * 64;
      }
    }
    if (save[l]) keep[save[l]] = y;
    if (mode === 2) out = y; else x = y;
  });
  return out;
}

function inputs(seed) {
  const r = rng(seed);
  return { feat: Float32Array.from({ length: 28160 }, () => r() * 2), z: Float32Array.from({ length: 640 }, () => r() * 2) };
}

const kernels = () => {
  const k = [];
  for (const name of ['avx512', 'avx2']) { try { addon.createF32(new Float32Array(addon.blobSizeF32()), name); k.push(name); } catch { /* not on this CPU */ } }
  return k;
};

test('float32 frame model = the WebGPU arithmetic in JavaScript (random weights, every kernel of this CPU)', { skip }, () => {
  const blob = randomBlob(addon.blobSizeF32());
  const { feat, z } = inputs(11);
  const ref = reference(blob, feat, z);
  let peak = 0; for (const v of ref) peak = Math.max(peak, Math.abs(v));
  assert.ok(peak > 1, `reference output too small to compare (${peak})`);
  const ks = kernels();
  assert.ok(ks.length > 0, 'no float32 kernel on this CPU');
  for (const k of ks) {
    const m = addon.createF32(blob, k);
    assert.equal(m.kernel, k);
    const out = new Float32Array(168960);
    addon.runF32(m.model, feat, z, out);
    let worst = 0; for (let i = 0; i < out.length; i++) worst = Math.max(worst, Math.abs(out[i] - ref[i]));
    assert.ok(worst <= 1e-4 * peak, `${k}: max |diff| ${worst} against peak ${peak}`);
  }
});

test('float32 frame model: wrong weights size and bad arguments are refused', { skip }, () => {
  assert.throws(() => addon.createF32(new Float32Array(10)), /wrong size/);
  assert.throws(() => addon.createF32(new Float32Array(addon.blobSizeF32()), 'sse41'), /auto, avx512 or avx2/);
  const m = addon.createF32(new Float32Array(addon.blobSizeF32()));
  assert.throws(() => addon.runF32(m.model, new Float32Array(5), new Float32Array(640), new Float32Array(168960)), /runF32/);
});

test('float32 frame model: several threads and several models at once give the bytes of one thread alone', { skip }, async () => {
  const blob = randomBlob(addon.blobSizeF32());
  const frames = [inputs(21), inputs(22), inputs(23)];
  const m = addon.createF32(blob);
  const other = addon.createF32(randomBlob(addon.blobSizeF32()).map((v) => v * 0.5));
  const alone = frames.map(({ feat, z }) => { const o = new Float32Array(168960); addon.runF32(m.model, feat, z, o); return Buffer.from(o.buffer).toString('base64'); });
  // the same thread: two models interleaved
  const scratch = new Float32Array(168960);
  frames.forEach(({ feat, z }, i) => {
    addon.runF32(other.model, feat, z, scratch);
    const o = new Float32Array(168960); addon.runF32(m.model, feat, z, o);
    assert.equal(Buffer.from(o.buffer).toString('base64'), alone[i]);
  });
  // 4 worker threads, each with its own model of the same weights, 20 frames each at the same time
  const code = `
    const { parentPort, workerData } = require('node:worker_threads');
    const a = require(workerData.path);
    const m = a.createF32(new Float32Array(workerData.blob));
    const out = [];
    for (let r = 0; r < 20; r++) { const f = workerData.frames[r % 3]; const o = new Float32Array(168960);
      a.runF32(m.model, new Float32Array(f.feat), new Float32Array(f.z), o); out.push(Buffer.from(o.buffer).toString('base64')); }
    parentPort.postMessage(out);`;
  const data = { path, blob: blob.buffer.slice(0), frames: frames.map((f) => ({ feat: f.feat.buffer.slice(0), z: f.z.buffer.slice(0) })) };
  const results = await Promise.all([0, 1, 2, 3].map(() => new Promise((res, rej) => {
    const w = new Worker(code, { eval: true, workerData: data });
    w.once('message', res); w.once('error', rej);
  })));
  for (const r of results) r.forEach((b64, i) => assert.equal(b64, alone[i % 3]));
});
