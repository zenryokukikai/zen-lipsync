// 'absolute' packs (core/frame-output.ts): the frame model paints the mouth, and every compositor draws the painted
// mouth over the idle video as it is (video * (1 - A) + painted * A): the pack's crops take no part. Packs without
// frame_output are read and drawn as before.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const F = await import('../dist/core/frame-output.js');
const { NodeCompositor } = await import('../dist/node/compositor.js');
const { checkPackMeta, loadPackData, listAvatar, nodePackSource, AvatarPackError } = await import('../dist/node/index.js');

const BH = 160, BW = 352, PLANE = BH * BW;
const box = { y0: 312, y1: 312 + BH, x0: 72, x1: 72 + BW };
const ID = [0, 1, 2].map(() => Float32Array.from({ length: 256 }, (_, v) => v));
/** The model output that paints p(j) at box pixel j (model colour): K = (p - 127.5) * win. */
const painting = (p) => { const win = F.boxWindow(BH, BW); return new Float32Array(3 * PLANE).map((_, i) => (p(i % PLANE, Math.floor(i / PLANE)) - 127.5) * win[i % PLANE]); };

test('frame output: the painted mouth is the painted pixel, weighted by mouth alpha * edge window * strength', () => {
  const alpha = new Uint8Array(PLANE);
  const inside = 80 * BW + 176, edge = 80 * BW + 3;
  alpha[inside] = 255; alpha[edge] = 128;
  const win = F.boxWindow(BH, BW);
  const pm = F.paintMouth(painting(() => 200), 0, alpha, box, ID, F.paintedMouth(PLANE), 0.5);
  for (let c = 0; c < 3; c++) assert.ok(Math.abs(pm.rgb[c * PLANE + inside] - 200) < 1e-3 && Math.abs(pm.rgb[c * PLANE + edge] - 200) < 1e-3);
  assert.ok(Math.abs(pm.weight[inside] - 0.5) < 1e-6);
  assert.ok(Math.abs(pm.weight[edge] - (128 / 255) * win[edge] * 0.5) < 1e-6);
  assert.equal(pm.weight[5 * BW + 5], 0, 'no weight outside the mouth alpha');
});

test('frame output: the painted pixel is shown in the video\'s colour, clamped to 0..255', () => {
  const darker = [0, 1, 2].map(() => Float32Array.from({ length: 256 }, (_, v) => v - 15)); // the video is 15 brighter
  const alpha = new Uint8Array(PLANE).fill(255), j = 80 * BW + 176;
  assert.ok(Math.abs(F.paintMouth(painting(() => 60), 0, alpha, box, darker.map((t) => F.inverseColour(t)), F.paintedMouth(PLANE)).rgb[j] - 75) < 1e-3);
  assert.equal(F.paintMouth(painting(() => 400), 0, alpha, box, ID, F.paintedMouth(PLANE)).rgb[j], 255);
  assert.equal(F.paintMouth(painting(() => -50), 0, alpha, box, ID, F.paintedMouth(PLANE)).rgb[j], 0);
});

test('frame output: the painted pixel goes back through model_colour, so channels whose video/model gap differs stay in step', () => {
  // video -> model: R as is, G and B 20 darker above 180 (a bright-teeth gap that differs per channel)
  const table = [0, 1, 2].map((c) => Array.from({ length: 256 }, (_, v) => (c > 0 && v > 180 ? v - 20 : v)));
  const pm = F.paintMouth(painting(() => 200), 0, new Uint8Array(PLANE).fill(255), box, table.map((t) => F.inverseColour(t)), F.paintedMouth(PLANE));
  const j = 80 * BW + 176, out = [0, 1, 2].map((c) => pm.rgb[c * PLANE + j]);
  assert.ok(Math.abs(out[0] - 200) < 1e-3 && Math.abs(out[1] - 220) < 1e-3 && Math.abs(out[2] - 220) < 1e-3, `video colour of model 200: ${out}`);
});

/** A small 'absolute' pack's source frame: the patch rect is the box itself (identity warp), every patch alpha 255. */
function absFrame(bw, bh, crop, mouth) {
  const geo = { rect: [0, 0, bw, bh], matrix: [[1, 0, 0], [0, 1, 0]], abs_x: 3, abs_y: 2, crop_box: [0, 0, bw, bh] };
  return { geo, crop, cropW: bw, cropH: bh, alpha: new Uint8Array(bw * bh).fill(255), mouth };
}

test('NodeCompositor (absolute pack): under a full mouth alpha the frame shows the painted pixels, whatever the crop (a crop that differs from the idle video never shows through)', () => {
  // a box of 28 x 26 (the 12-pixel edge window leaves its centre at 1)
  const bw = 28, bh = 26, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  const W = 40, H = 34;
  const video = Uint8Array.from({ length: W * H * 3 }, (_, i) => 60 + (i % 7));        // the idle video frame
  const paint = (j) => 150 + (j % 5) * 10;                                               // a textured painted mouth
  const win = F.boxWindow(bh, bw);
  const pred = new Float32Array(3 * plane).map((_, i) => (paint(i % plane) - 127.5) * win[i % plane]);
  const mouth = new Uint8Array(plane).fill(255);
  const comp = new NodeCompositor({ box: mbox }, ID, ID, ID);
  const crops = [new Uint8Array(plane * 3).fill(60), new Uint8Array(plane * 3).map((_, i) => (i * 37) % 256)]; // same as the video / another export
  const out = crops.map((crop) => { const f = Uint8Array.from(video); comp.draw(f, W, H, absFrame(bw, bh, crop, mouth), pred, 0); return f; });
  assert.deepEqual(out[0], out[1], 'the crop takes no part in the drawn frame');
  const c = 13 * bw + 14, px = ((2 + 13) * W + 3 + 14) * 3; // box centre (window 1) on the frame
  for (let k = 0; k < 3; k++) assert.equal(out[1][px + k], Math.round(paint(c)), 'the painted pixel, not idle + painted - crop');
  // at the box edge the window blends the painted pixel with the video pixel
  const e = 13 * bw + 0, pe = ((2 + 13) * W + 3) * 3, A = win[e];
  assert.ok(Math.abs(out[1][pe] - (video[pe] * (1 - A) + paint(e) * A)) <= 1);
  // outside the patch rect nothing changes
  assert.equal(out[1][0], video[0]);
});

test('NodeCompositor (absolute pack): a fading frame blends the painted mouth by its strength; strength 0 leaves the video as it is', () => {
  const bw = 28, bh = 26, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  const W = 40, H = 34, video = new Uint8Array(W * H * 3).fill(40);
  const win = F.boxWindow(bh, bw);
  const pred = new Float32Array(3 * plane).map((_, i) => (240 - 127.5) * win[i % plane]);
  const comp = new NodeCompositor({ box: mbox }, ID, ID, ID);
  const src = absFrame(bw, bh, new Uint8Array(plane * 3), new Uint8Array(plane).fill(255));
  const half = Uint8Array.from(video); comp.draw(half, W, H, src, pred, 0, 0.5);
  const px = ((2 + 13) * W + 3 + 14) * 3;
  assert.equal(half[px], 140); // 40 * 0.5 + 240 * 0.5
  const none = Uint8Array.from(video); comp.draw(none, W, H, src, pred, 0, 0);
  assert.deepEqual(none, video);
});

test('NodeCompositor (absolute pack): at the box edge only taps inside the box count, weighted by their weight (no crop, no zero colour pulled in)', () => {
  // the patch rect is 2 px larger than the box on every side; the box is all one painted colour with a full mouth alpha
  const bw = 28, bh = 26, plane = bw * bh, mbox = { y0: 2, y1: 2 + bh, x0: 2, x1: 2 + bw };
  const geo = { rect: [0, 0, bw + 4, bh + 4], matrix: [[1, 0, 0], [0, 1, 0]], abs_x: 0, abs_y: 0, crop_box: [0, 0, bw + 4, bh + 4] };
  const win = F.boxWindow(bh, bw);
  const pred = new Float32Array(3 * plane).map((_, i) => (200 - 127.5) * win[i % plane]);
  const src = { geo, crop: new Uint8Array((bw + 4) * (bh + 4) * 3), cropW: bw + 4, cropH: bh + 4, alpha: new Uint8Array((bw + 4) * (bh + 4)).fill(255), mouth: new Uint8Array(plane).fill(255) };
  const W = bw + 4, H = bh + 4, frame = new Uint8Array(W * H * 3).fill(50);
  new NodeCompositor({ box: mbox }, ID, ID, ID).draw(frame, W, H, src, pred, 0);
  const at = (x, y) => frame[(y * W + x) * 3];
  assert.equal(at(0, 15), 50, 'outside the box: the video pixel');
  assert.equal(at(1, 15), 50, 'outside the box: the video pixel');
  const A = win[13 * bw + 0];
  assert.ok(Math.abs(at(2, 15) - (50 * (1 - A) + 200 * A)) <= 1, `first box column: ${at(2, 15)}`);
  assert.equal(at(16, 15), 200, 'inside: the painted colour');
});

async function pack(extra, files = {}) {
  const d = await mkdtemp(path.join(tmpdir(), 'zl-abs-'));
  const sha = (b) => createHash('sha256').update(b).digest('hex');
  const all = { 'features.f16': Buffer.alloc(2 * 8 * 2), 'ref_latent.f32': Buffer.alloc(2 * 128 * 4), ...files };
  for (const [n, b] of Object.entries(all)) await writeFile(path.join(d, n), b);
  await writeFile(path.join(d, 'manifest.json'), JSON.stringify({
    format: 'zen-lipsync-avatar', version: extra.frame_output === 'absolute' ? 3 : 1, name: 'p', frame_size: [8, 8], fps: 25, source_frames: 2, walk: [0, 1],
    geo: {}, box: { y0: 0, y1: 2, x0: 0, x1: 3 }, feat_dims: [2, 2, 2], latent_window: 5, video: [{ src: 'idle.webm', type: 'video/webm' }],
    ...(extra.frame_output === 'absolute' ? { frames: { pack: 'frames.pack', type: 'image/webp' } } : {}),
    files: Object.fromEntries(Object.entries({ ...all, ...(extra.frame_output === 'absolute' ? { 'frames.pack': Buffer.alloc(1) } : {}) }).map(([n, b]) => [n, { bytes: b.length, sha256: sha(b) }])), ...extra,
  }));
  const src = nodePackSource();
  return loadPackData(src, (await listAvatar(src, d)).packs[0]);
}

test('pack loader: a pack without frame_output has no mouth alpha (drawn as before); an absolute pack loads its alphas', async () => {
  assert.equal((await pack({})).mouthAlpha, null);
  const a = Buffer.from(Array.from({ length: 12 }, (_, i) => i));
  const colour = [0, 1, 2].map(() => Array.from({ length: 256 }, (_, v) => v));
  const d = await pack({ frame_output: 'absolute', model_colour: colour }, { [F.MOUTH_ALPHA_FILE]: a });
  assert.deepEqual([...d.mouthAlpha.of(1)], [6, 7, 8, 9, 10, 11]);
  await assert.rejects(pack({ frame_output: 'absolute', model_colour: colour }, { [F.MOUTH_ALPHA_FILE]: Buffer.alloc(5) }), AvatarPackError);
  await assert.rejects(pack({ frame_output: 'absolute' }, { [F.MOUTH_ALPHA_FILE]: a }), /model_colour/);
});

/** The manifest fields of a version 3 pack (version 2's frames.pack; 'absolute' models only). */
const V3 = { version: 3, frames: { pack: 'frames.pack', type: 'image/webp' }, files: { 'frames.pack': { bytes: 1, sha256: 'c' } } };

test('pack check: an absolute model goes with manifest version 3 and version 3 with an absolute model (earlier SDKs, which accept 1 and 2 only, refuse it)', () => {
  const base = { format: 'zen-lipsync-avatar', frame_size: [8, 8], fps: 25, source_frames: 1, walk: [0], geo: {}, box: {}, feat_dims: [1, 1, 1], latent_window: 5, video: [{}] };
  const colour = [0, 1, 2].map(() => Array.from({ length: 256 }, (_, v) => v));
  const abs = { frame_output: 'absolute', model_colour: colour };
  const files = { ...V3.files, 'mouth_alpha.u8': { bytes: 1, sha256: 'a' } };
  assert.doesNotThrow(() => checkPackMeta({ ...base, ...V3, ...abs, files }, 'm'));
  assert.throws(() => checkPackMeta({ ...base, ...V3, version: 2, ...abs, files }, 'm'), /version 3/);
  assert.throws(() => checkPackMeta({ ...base, ...V3, files }, 'm'), /version 3/);
  assert.throws(() => checkPackMeta({ ...base, ...V3, frame_output: 'delta', files }, 'm'), /version 3/);
  assert.throws(() => checkPackMeta({ ...base, ...V3, frames: undefined, ...abs, files }, 'm'), /"frames"/);
  assert.throws(() => checkPackMeta({ ...base, version: 4, files: {} }, 'm'), /version 1, 2 or 3/);
});

test('pack check: frame_output must be delta or absolute, and an absolute pack needs its mouth alpha', () => {
  const m = { format: 'zen-lipsync-avatar', version: 1, frame_size: [8, 8], fps: 25, source_frames: 1, walk: [0], geo: {}, box: {}, feat_dims: [1, 1, 1], latent_window: 5, video: [{}], files: {} };
  assert.doesNotThrow(() => checkPackMeta({ ...m }, 'm'));
  assert.throws(() => checkPackMeta({ ...m, frame_output: 'pixels' }, 'm'), AvatarPackError);
  assert.throws(() => checkPackMeta({ ...m, ...V3, frame_output: 'absolute', files: V3.files }, 'm'), /mouth_alpha/);
});

test('frame output: an absolute pack without a valid model_colour is refused when its manifest is checked', () => {
  const base = { format: 'zen-lipsync-avatar', frame_size: [8, 8], fps: 25, source_frames: 1, walk: [0], geo: {}, box: {}, feat_dims: [1, 1, 1], latent_window: 5, video: [{}],
    ...V3, frame_output: 'absolute', files: { ...V3.files, 'mouth_alpha.u8': { bytes: 1, sha256: 'a' } } };
  const ok = [0, 1, 2].map(() => Array.from({ length: 256 }, (_, v) => v));
  assert.doesNotThrow(() => checkPackMeta({ ...base, model_colour: ok }, 'pack'));
  for (const model_colour of [undefined, [[1, 2]], [0, 1, 2].map(() => Array(256).fill(Number.NaN))])
    assert.throws(() => checkPackMeta({ ...base, model_colour }, 'pack'), (e) => e instanceof AvatarPackError && /model_colour/.test(e.message));
});

test('pack check: an absolute pack carries no tone.json', () => {
  const m = { format: 'zen-lipsync-avatar', frame_size: [8, 8], fps: 25, source_frames: 1, walk: [0], geo: {}, box: {}, feat_dims: [1, 1, 1], latent_window: 5, video: [{}],
    ...V3, frame_output: 'absolute', model_colour: [0, 1, 2].map(() => Array.from({ length: 256 }, (_, v) => v)), files: { ...V3.files, 'mouth_alpha.u8': { bytes: 1, sha256: 'a' }, 'tone.json': { bytes: 1, sha256: 'b' } } };
  assert.throws(() => checkPackMeta(m, 'm'), /tone\.json/);
});

test('frame output: inverse of a non-unit-slope table, fractional painted values and the ends', () => {
  const t = Array.from({ length: 256 }, (_, v) => 20 + v * 0.8); // video v -> model 20 + 0.8 v (model range 20..224)
  const inv = F.inverseColour(t);
  assert.ok(Math.abs(inv[100] - 100) < 1e-3, 'model 100 <- video 100');
  assert.ok(Math.abs(inv[60] - 50) < 1e-3, 'model 60 <- video 50');
  assert.equal(inv[0], 0, 'below the table: kept at 0');
  assert.equal(inv[250], 255, 'above the table: kept at 255');
  const alpha = new Uint8Array(PLANE).fill(255), win = F.boxWindow(BH, BW), j = 80 * BW + 176, e = 0; // middle and the corner (win 1/169)
  const pred = new Float32Array(3 * PLANE).map((_, i) => (60.4 - 127.5) * win[i % PLANE]); // paints 60.4 (fractional)
  const pm = F.paintMouth(pred, 0, alpha, box, [inv, inv, inv], F.paintedMouth(PLANE));
  assert.ok(Math.abs(pm.rgb[j] - 50.5) < 1e-3, 'fractional painted value -> 50.5');
  assert.ok(Math.abs(pm.rgb[e] - 50.5) < 1e-3, 'corner pixel: the same painted value');
  assert.ok(Math.abs(pm.weight[e] - win[e]) < 1e-6, 'its weight is the window there');
});

// The painted mouth is put back into the frame bicubic, with OpenCV INTER_CUBIC's kernel (a = -0.75).
/** warpPainted's out with every pixel outside the part it wrote (x0..x1, y0..y1) set to "not drawn" (all 0). */
const warpOut = (w, rw, rh) => { const o = new Float32Array(rw * rh * 4); for (let y = w.y0; y <= w.y1; y++) for (let x = w.x0; x <= w.x1; x++) for (let k = 0; k < 4; k++) o[(y * rw + x) * 4 + k] = w.out[(y * rw + x) * 4 + k]; return o; };
const warp = (pm, mbox, geo, alpha, work) => { const w = F.warpPainted(pm, mbox, geo, alpha, work); return { ...w, raw: w, out: warpOut(w, geo.rect[2], geo.rect[3]) }; };

const cubic = (x) => { const A = -0.75, t = Math.abs(x); return t <= 1 ? ((A + 2) * t - (A + 3)) * t * t + 1 : t < 2 ? ((A * t - 5 * A) * t + 8 * A) * t - 4 * A : 0; };
test('warpPainted: a sharp painted box is sampled with the INTER_CUBIC kernel (a = -0.75), as a reference computes it', () => {
  const bw = 24, bh = 20, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  // stripes 1 px wide (the sharpest detail), full weight everywhere
  const pm = F.paintedMouth(plane);
  for (let j = 0; j < plane; j++) { const v = (j % bw) % 2 ? 230 : 30; pm.rgb[j] = v; pm.rgb[plane + j] = 255 - v; pm.rgb[2 * plane + j] = 128; pm.weight[j] = 1; }
  // a slight scale, rotation and shift (as a pack's matrix: 512 crop -> patch rect)
  const geo = { rect: [0, 0, 20, 16], matrix: [[0.83, 0.05, -1.7], [-0.04, 0.81, -1.2]], abs_x: 0, abs_y: 0, crop_box: [0, 0, bw, bh] };
  const w = warp(pm, mbox, geo, null);
  const m = geo.matrix, det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  let checked = 0, maxErr = 0;
  for (let py = 0; py < 16; py++) for (let px = 0; px < 20; px++) {
    const vx = px + 0.5 - m[0][2], vy = py + 0.5 - m[1][2];
    const sx = (m[1][1] * vx - m[0][1] * vy) / det - 0.5, sy = (-m[1][0] * vx + m[0][0] * vy) / det - 0.5;
    const xf = Math.floor(sx), yf = Math.floor(sy);
    if (xf < 1 || yf < 1 || xf + 2 >= bw || yf + 2 >= bh) continue; // all 16 taps inside: the plain kernel
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let ty = -1; ty <= 2; ty++) for (let tx = -1; tx <= 2; tx++) s += cubic(sx - (xf + tx)) * cubic(sy - (yf + ty)) * pm.rgb[c * plane + (yf + ty) * bw + xf + tx];
      const want = Math.max(0, Math.min(255, s));
      maxErr = Math.max(maxErr, Math.abs(w.out[(py * 20 + px) * 4 + c] - want));
    }
    assert.ok(Math.abs(w.out[(py * 20 + px) * 4 + 3] - 1) < 1e-4, 'full weight inside');
    checked++;
  }
  assert.ok(checked > 100, `${checked} pixels checked`);
  assert.ok(maxErr < 1e-2, `max error ${maxErr}`);
});

test('warpPainted: an identity matrix gives the painted pixels back exactly (the cubic kernel interpolates)', () => {
  const bw = 16, bh = 12, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  const pm = F.paintedMouth(plane);
  for (let j = 0; j < plane; j++) { pm.rgb[j] = (j * 37) % 256; pm.rgb[plane + j] = (j * 11) % 256; pm.rgb[2 * plane + j] = 200; pm.weight[j] = 1; }
  const geo = { rect: [0, 0, bw, bh], matrix: [[1, 0, 0], [0, 1, 0]], abs_x: 0, abs_y: 0, crop_box: [0, 0, bw, bh] };
  const w = warp(pm, mbox, geo, null);
  for (let j = 0; j < plane; j++) for (let c = 0; c < 3; c++) assert.ok(Math.abs(w.out[j * 4 + c] - pm.rgb[c * plane + j]) < 1e-3);
});

test('warpPainted: keeps a 1-px edge sharper than bilinear would (a 0.75x downscale of a step keeps its contrast)', () => {
  const bw = 32, bh = 8, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  const pm = F.paintedMouth(plane);
  for (let j = 0; j < plane; j++) { const v = (j % bw) % 4 < 2 ? 220 : 40; pm.rgb[j] = v; pm.rgb[plane + j] = v; pm.rgb[2 * plane + j] = v; pm.weight[j] = 1; }
  // 512 crop -> frame at 0.75: the frame's pixel spans 1.33 box pixels
  const geo = { rect: [0, 0, 24, 6], matrix: [[0.75, 0, 0], [0, 0.75, 0]], abs_x: 0, abs_y: 0, crop_box: [0, 0, bw, bh] };
  const w = warp(pm, mbox, geo, null);
  const row = 3, vals = []; for (let x = 3; x < 21; x++) vals.push(w.out[(row * 24 + x) * 4]);
  const cubicRange = Math.max(...vals) - Math.min(...vals);
  // the same samples taken bilinearly
  const step = (x) => (((x % 4) + 4) % 4 < 2 ? 220 : 40);
  const bil = []; for (let x = 3; x < 21; x++) { const sx = (x + 0.5) / 0.75 - 0.5, xf = Math.floor(sx), t = sx - xf; bil.push(step(xf) * (1 - t) + step(xf + 1) * t); }
  const bilRange = Math.max(...bil) - Math.min(...bil);
  assert.ok(cubicRange > bilRange, `cubic contrast ${cubicRange.toFixed(1)} > bilinear ${bilRange.toFixed(1)}`);
});

test('warpPainted: taps outside the painted mouth (weight 0) pull in no colour; the edge weight falls off smoothly', () => {
  const bw = 20, bh = 10, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  const pm = F.paintedMouth(plane);
  for (let j = 0; j < plane; j++) { const x = j % bw, v = x < 10 ? 200 : 0; pm.rgb[j] = v; pm.rgb[plane + j] = v; pm.rgb[2 * plane + j] = v; pm.weight[j] = x < 10 ? 1 : 0; }
  const geo = { rect: [0, 0, 26, 10], matrix: [[1.3, 0, 0], [0, 1, 0]], abs_x: 0, abs_y: 0, crop_box: [0, 0, bw, bh] };
  const w = warp(pm, mbox, geo, null);
  for (let x = 0; x < 26; x++) {
    const o = (5 * 26 + x) * 4;
    if (w.out[o + 3] > 0) assert.ok(Math.abs(w.out[o] - 200) < 1e-3, `x=${x}: colour ${w.out[o]} is the painted 200 (no 0 pulled in)`);
  }
  assert.ok(w.out[(5 * 26 + 2) * 4 + 3] > 0.99, 'inside: full weight');
  assert.equal(w.out[(5 * 26 + 25) * 4 + 3], 0, 'far outside: weight 0');
});

test('warpPainted: a feathered weight edge over a colour gradient stays within the painted colours (no overshoot, no ghost)', () => {
  const bw = 30, bh = 12, plane = bw * bh, mbox = { y0: 5, y1: 5 + bh, x0: 7, x1: 7 + bw };
  const pm = F.paintedMouth(plane);
  for (let j = 0; j < plane; j++) {
    const x = j % bw;
    pm.rgb[j] = 20 + x * 7; pm.rgb[plane + j] = 240 - x * 6; pm.rgb[2 * plane + j] = x % 2 ? 250 : 5; // gradient + 1-px stripes
    pm.weight[j] = x < 12 ? 1 : x < 20 ? (20 - x) / 8 : 0;                                            // feathered edge
  }
  const geo = { rect: [0, 0, 40, 14], matrix: [[1.21, 0.03, -7.9], [-0.02, 1.13, -5.4]], abs_x: 0, abs_y: 0, crop_box: [0, 0, 64, 64] };
  const w = warp(pm, mbox, geo, null);
  for (let i = 0; i < 40 * 14; i++) {
    const ws = w.out[i * 4 + 3];
    assert.ok(ws >= 0 && ws <= 1, `weight ${ws}`);
    if (ws === 0) continue;
    assert.ok(w.out[i * 4] >= 20 - 1e-3 && w.out[i * 4] <= 20 + 19 * 7 + 1e-3, `red ${w.out[i * 4]} inside the painted range`);
  }
  // far right of the feathered edge: nothing drawn
  for (let y = 0; y < 14; y++) assert.equal(w.out[(y * 40 + 39) * 4 + 3], 0);
});

test('warpPainted: the box offset, a patch alpha and reused work buffers (a larger, then a smaller rect) give the same pixels as fresh ones', () => {
  const bw = 22, bh = 14, plane = bw * bh, mbox = { y0: 9, y1: 9 + bh, x0: 4, x1: 4 + bw };
  const pm = F.paintedMouth(plane);
  for (let j = 0; j < plane; j++) { pm.rgb[j] = (j * 31) % 256; pm.rgb[plane + j] = (j * 17) % 256; pm.rgb[2 * plane + j] = (j * 5) % 256; pm.weight[j] = 1; }
  // identity on the box: rect pixel (x, y) is box pixel (x, y) only when the box offset is taken off
  const geoA = { rect: [0, 0, bw, bh], matrix: [[1, 0, -4], [0, 1, -9]], abs_x: 0, abs_y: 0, crop_box: [0, 0, 64, 64] };
  const alpha = Uint8Array.from({ length: bw * bh }, (_, i) => (i % 3 === 0 ? 0 : 255));
  const fresh = warp(pm, mbox, geoA, alpha);
  for (let i = 0; i < plane; i++) {
    if (alpha[i] === 0) { assert.equal(fresh.out[i * 4 + 3], 0, 'patch alpha 0: skipped'); continue; }
    for (let c = 0; c < 3; c++) assert.ok(Math.abs(fresh.out[i * 4 + c] - pm.rgb[c * plane + i]) < 1e-3, 'box offset taken off');
  }
  const geoBig = { rect: [0, 0, 40, 30], matrix: [[1.6, 0, -6], [0, 1.6, -14]], abs_x: 0, abs_y: 0, crop_box: [0, 0, 64, 64] };
  const work = warp(pm, mbox, geoBig, null);
  const again = warp(pm, mbox, geoA, alpha, work.raw);
  assert.equal(again.raw.premul, work.raw.premul, 'the box buffer is reused');
  assert.equal(again.raw.out, work.raw.out, 'the larger rect buffer is reused');
  assert.deepEqual(Array.from(again.out.subarray(0, plane * 4)), Array.from(fresh.out.subarray(0, plane * 4)), 'no stale pixels');
  const grown = warp(pm, mbox, geoBig, null, fresh.raw);
  assert.equal(grown.raw.premul, fresh.raw.premul, 'the box buffer is kept when only the rect grows');
  assert.notEqual(grown.raw.out, fresh.raw.out, 'the rect buffer grows');
  assert.deepEqual(Array.from(grown.out), Array.from(warp(pm, mbox, geoBig, null).out));
});

test('warpPainted: a fading mouth (one weight below 1 everywhere) is still bicubic, with that weight', () => {
  const bw = 24, bh = 20, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  const make = (k) => { const pm = F.paintedMouth(plane); for (let j = 0; j < plane; j++) { const v = (j % bw) % 2 ? 230 : 30; pm.rgb[j] = v; pm.rgb[plane + j] = 255 - v; pm.rgb[2 * plane + j] = 90; pm.weight[j] = k; } return pm; };
  const geo = { rect: [0, 0, 20, 16], matrix: [[0.83, 0.05, -1.7], [-0.04, 0.81, -1.2]], abs_x: 0, abs_y: 0, crop_box: [0, 0, bw, bh] };
  const full = warp(make(1), mbox, geo, null), fade = warp(make(0.4), mbox, geo, null);
  let n = 0;
  for (let i = 0; i < 20 * 16; i++) {
    if (full.out[i * 4 + 3] !== 1) continue;
    for (let c = 0; c < 3; c++) assert.ok(Math.abs(fade.out[i * 4 + c] - full.out[i * 4 + c]) < 1e-3, 'the same colour');
    assert.ok(Math.abs(fade.out[i * 4 + 3] - 0.4) < 1e-6, 'the fade weight'); n++;
  }
  assert.ok(n > 100);
});

test('warpPainted: the separable path of an axis-aligned matrix (every pack) gives the general path\'s pixels', () => {
  const bw = 40, bh = 24, plane = bw * bh, mbox = { y0: 6, y1: 6 + bh, x0: 3, x1: 3 + bw };
  const pm = F.paintedMouth(plane);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const j = y * bw + x, r = Math.hypot((x - bw / 2) / 14, (y - bh / 2) / 8);
    pm.rgb[j] = (j * 37) % 256; pm.rgb[plane + j] = (x % 2) * 220 + 10; pm.rgb[2 * plane + j] = (y * 9) % 256;
    pm.weight[j] = r < 0.7 ? 1 : r < 1.2 ? (1.2 - r) / 0.5 : 0;   // full centre, feathered ring, 0 outside
  }
  for (const s of [0.6, 1.1, 1.19]) {
    const rw = Math.round(bw * s) + 6, rh = Math.round(bh * s) + 6;
    const geo = (off) => ({ rect: [0, 0, rw, rh], matrix: [[s, off, -3 * s + 2.3], [off, s, -6 * s + 1.7]], abs_x: 0, abs_y: 0, crop_box: [0, 0, 64, 64] });
    const alpha = Uint8Array.from({ length: rw * rh }, (_, i) => (i % 11 === 0 ? 0 : 255));
    const sep = warp(pm, mbox, geo(0), alpha).out, gen = warp(pm, mbox, geo(1e-12), alpha).out;
    let max = 0;
    for (let i = 0; i < rw * rh * 4; i++) max = Math.max(max, Math.abs(sep[i] - gen[i]) * (i % 4 === 3 ? 255 : 1));
    assert.ok(max < 0.01, `scale ${s}: max difference ${max}`);
  }
});

test('warpPainted: a reused work buffer after a frame with a larger painted mouth gives the same pixels as a fresh one', () => {
  const bw = 40, bh = 24, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  const mk = (rad) => { const pm = F.paintedMouth(plane); for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) { const j = y * bw + x, r = Math.hypot((x - 20) / rad, (y - 12) / (rad * 0.6)); pm.rgb[j] = (x % 2) * 200 + 20; pm.rgb[plane + j] = 90; pm.rgb[2 * plane + j] = (y * 13) % 256; pm.weight[j] = r < 0.8 ? 1 : r < 1 ? (1 - r) * 5 : 0; } return pm; };
  const geo = { rect: [0, 0, 44, 27], matrix: [[1.1, 0, 0.4], [0, 1.1, 0.2]], abs_x: 0, abs_y: 0, crop_box: [0, 0, 64, 64] };
  const big = warp(mk(18), mbox, geo, null);
  const reused = warp(mk(7), mbox, geo, null, big.raw), fresh = warp(mk(7), mbox, geo, null);
  assert.deepEqual(Array.from(reused.out.subarray(0, 44 * 27 * 4)), Array.from(fresh.out.subarray(0, 44 * 27 * 4)));
});

test('warpPainted: inside a feathered ring (every tap painted, weights differ) the colour is still bicubic and the weight bilinear', () => {
  const bw = 24, bh = 20, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  const pm = F.paintedMouth(plane);
  for (let j = 0; j < plane; j++) { const x = j % bw, v = x % 2 ? 230 : 30; pm.rgb[j] = v; pm.rgb[plane + j] = 255 - v; pm.rgb[2 * plane + j] = 128; pm.weight[j] = 0.2 + 0.8 * (x / (bw - 1)); }
  const geo = { rect: [0, 0, 20, 16], matrix: [[0.83, 0.05, -1.7], [-0.04, 0.81, -1.2]], abs_x: 0, abs_y: 0, crop_box: [0, 0, bw, bh] };
  const w = warp(pm, mbox, geo, null);
  const m = geo.matrix, det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  let n = 0;
  for (let py = 0; py < 16; py++) for (let px = 0; px < 20; px++) {
    const vx = px + 0.5 - m[0][2], vy = py + 0.5 - m[1][2];
    const sx = (m[1][1] * vx - m[0][1] * vy) / det - 0.5, sy = (-m[1][0] * vx + m[0][0] * vy) / det - 0.5;
    const xf = Math.floor(sx), yf = Math.floor(sy), tx = sx - xf, ty = sy - yf;
    if (xf < 1 || yf < 1 || xf + 2 >= bw || yf + 2 >= bh) continue;
    let s = 0;
    for (let a = -1; a <= 2; a++) for (let b = -1; b <= 2; b++) s += cubic(sx - (xf + b)) * cubic(sy - (yf + a)) * pm.rgb[(yf + a) * bw + xf + b];
    assert.ok(Math.abs(w.out[(py * 20 + px) * 4] - Math.max(0, Math.min(255, s))) < 1e-2, 'colour: the cubic reference');
    const wl = (x, y) => pm.weight[y * bw + x];
    const wb = (1 - ty) * ((1 - tx) * wl(xf, yf) + tx * wl(xf + 1, yf)) + ty * ((1 - tx) * wl(xf, yf + 1) + tx * wl(xf + 1, yf + 1));
    assert.ok(Math.abs(w.out[(py * 20 + px) * 4 + 3] - wb) < 1e-5, 'weight: bilinear');
    n++;
  }
  assert.ok(n > 100);
});

test('warpPainted: a mirrored axis-aligned matrix (negative scale) is drawn (the general path), not dropped', () => {
  const bw = 20, bh = 12, plane = bw * bh, mbox = { y0: 0, y1: bh, x0: 0, x1: bw };
  const pm = F.paintedMouth(plane);
  for (let j = 0; j < plane; j++) { pm.rgb[j] = 100; pm.rgb[plane + j] = 100; pm.rgb[2 * plane + j] = 100; pm.weight[j] = 1; }
  const geo = { rect: [0, 0, bw, bh], matrix: [[-1, 0, bw], [0, 1, 0]], abs_x: 0, abs_y: 0, crop_box: [0, 0, bw, bh] };
  const w = warp(pm, mbox, geo, null);
  assert.ok(Math.abs(w.out[(6 * bw + 10) * 4] - 100) < 1e-3 && w.out[(6 * bw + 10) * 4 + 3] === 1);
});
