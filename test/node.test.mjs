// Unit tests of the Node entry's API surface (node --test, after `pnpm build`). Needs no avatar files, ffmpeg or
// native addon: option checks, the PNG decoder, the compositor, and an avatar built on a small fixture with stub models.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deflateSync } from 'node:zlib';
import { create, OptionsError, FfmpegMissingError, nativeAddonPath, NodeAvatar } from '../dist/node/index.js';
import { decodePng } from '../dist/node/png.js';
import { NodeCompositor } from '../dist/node/compositor.js';

function png(w, h, ch, data) {
  const ct = ch === 1 ? 0 : ch === 3 ? 2 : 6;
  const raw = Buffer.alloc(h * (w * ch + 1));
  for (let y = 0; y < h; y++) { raw[y * (w * ch + 1)] = 1; for (let x = 0; x < w * ch; x++) { const v = data[y * w * ch + x], l = x >= ch ? data[y * w * ch + x - ch] : 0; raw[y * (w * ch + 1) + 1 + x] = (v - l) & 255; } }
  const chunk = (type, body) => { const b = Buffer.alloc(12 + body.length); b.writeUInt32BE(body.length); b.write(type, 4); body.copy(b, 8); return b; };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = ct;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

test('create(): option checks (before any file is read)', async () => {
  await assert.rejects(create(null), OptionsError);
  await assert.rejects(create({ avatar: 'marine', nope: 1 }), /unknown option "nope"/);
  await assert.rejects(create({ avatar: 'marine', assetsPath: '/a', assetsBaseUrl: 'https://x/' }), /assetsPath or assetsBaseUrl/);
  await assert.rejects(create({ avatar: 'marine', assetsPath: '/a', backend: 'gpu' }), /backend/);
  await assert.rejects(create({ avatar: 'marine', assetsPath: '/a', idleCache: 'disk' }), /idleCache/);
  await assert.rejects(create({ avatar: '' }), OptionsError);
});

test('create(): a missing ffmpeg is reported as FfmpegMissingError', async () => {
  await assert.rejects(create({ avatar: 'marine', assetsPath: '/nonexistent', ffmpegPath: '/nonexistent/ffmpeg' }), FfmpegMissingError);
});

test('nativeAddonPath() is a path or null', () => {
  const p = nativeAddonPath();
  assert.ok(p === null || p.endsWith('zen_lipsync.node'));
});

test('decodePng: grey, RGB and RGBA with filters', () => {
  const rgb = Uint8Array.from({ length: 5 * 3 * 3 }, (_, i) => (i * 37) & 255);
  const d = decodePng(png(5, 3, 3, rgb));
  assert.deepEqual([d.width, d.height, d.channels], [5, 3, 3]);
  assert.deepEqual([...d.data], [...rgb]);
  const g = Uint8Array.from({ length: 4 * 2 }, (_, i) => i * 30);
  assert.deepEqual([...decodePng(png(4, 2, 1, g)).data], [...g]);
  assert.throws(() => decodePng(Buffer.from('nope')), /not a PNG/);
});

test('NodeCompositor: adds the change to the video pixel and cuts it by the patch alpha (a change of zero leaves the video pixel as it is)', () => {
  const ident = Float32Array.from({ length: 256 }, (_, i) => i);
  const meta = { box: { y0: 0, y1: 4, x0: 0, x1: 4 } };
  const comp = new NodeCompositor(meta, [ident, ident, ident], [ident, ident, ident]);
  const W = 8, H = 8;
  const frame = new Uint8Array(W * H * 3).fill(10);
  const geo = { rect: [0, 0, 4, 4], matrix: [[1, 0, 0], [0, 1, 0]], abs_x: 2, abs_y: 3, crop_box: [0, 0, 4, 4] };
  const src = { geo, crop: new Uint8Array(4 * 4 * 3).fill(100), cropW: 4, cropH: 4, alpha: new Uint8Array(16).fill(255) };
  src.alpha[0] = 0;
  const pred = new Float32Array(3 * 4 * 4); pred[5] = 20; // change at box (1,1), channel 0
  comp.draw(frame, W, H, src, pred, 0);
  const px = (x, y) => [...frame.subarray((y * W + x) * 3, (y * W + x) * 3 + 3)];
  assert.deepEqual(px(0, 0), [10, 10, 10]);      // outside the patch
  assert.deepEqual(px(2, 3), [10, 10, 10]);      // alpha 0
  assert.deepEqual(px(3, 3), [10, 10, 10]);      // patch, no change: the video pixel stays (no resampled crop on top)
  assert.deepEqual(px(3, 4), [30, 10, 10]);      // patch with the change at box (1,1): 10 + 20
});

test('NodeAvatar: frames() of chunked speech, audio laid end to end, one output at a time', async () => {
  const W = 8, H = 8, L = 6;
  const meta = { frame_size: [W, H], fps: 25, walk: [0, 1, 2, 2, 1, 0], latent_window: 5, box: { y0: 0, y1: 160, x0: 0, x1: 352 }, geo: {} };
  const video = { frame: async (k) => Buffer.alloc(W * H * 3, k % L), bytes: 0, close() {} };
  const pack = { name: 'normal', meta, feat: new Uint16Array(3 * 28160), featSize: 28160, refLatent: new Float32Array(3 * 128), fromVideo: [], toVideo: [], video,
    walk: (s) => meta.walk[((s % L) + L) % L], source: () => ({ geo: { rect: [0, 0, 1, 1], matrix: [[1, 0, 0], [0, 1, 0]], abs_x: 0, abs_y: 0, crop_box: [0, 0, 1, 1] }, crop: new Uint8Array(3), cropW: 1, cropH: 1, alpha: new Uint8Array([0]) }),
    bytes: 0, close() {} };
  const lips = { backend: 'cpu', weightBytes: 0, sentence: async (_m, _r, T) => new Float32Array(T * 128), frame: async (_f, _z, n) => new Float32Array(n * 168960) };
  const prep = { mel: async (pcm, rate) => { const frames = Math.floor((pcm.length / rate) * 25); const melFrames = Math.floor((pcm.length / rate) * 80) + 1; return { mel: new Float32Array(melFrames * 80), melFrames, frames, gather: Array.from({ length: frames }, (_, i) => Math.min(Math.round((i / 25) * 80), melFrames - 1)) }; }, sentence: lips.sentence };
  const a = new NodeAvatar(new Map([['normal', pack]]), lips, 'wasm', prep, null, null, 'ffmpeg', 0, 0);
  const chunk = (sec) => ({ pcm: new Float32Array(24000 * sec).fill(0.5), sampleRate: 24000 });
  async function* chunks() { yield chunk(0.4); yield chunk(0.6); }
  const got = [];
  const it = a.frames(chunks(), { idleAfterMs: 80 });
  const first = await it.next();
  await assert.rejects(a.frames(chunk(1)).next(), /one at a time/);
  got.push(first.value);
  for await (const f of it) got.push(f);
  const speech = got.filter((f) => f.speaking);
  assert.equal(speech.length, 25);                       // 1.0 s of speech at 25 fps, no gap between the chunks
  assert.equal(got.length, 25 + 2);                      // + 80 ms of idle after
  assert.deepEqual(got.map((f) => f.index), got.map((_, i) => i));
  const audio = got.reduce((n, f) => n + f.audio.length, 0);
  assert.equal(audio, 27 * 960);                         // the output audio covers every frame
  const speechAudio = got.reduce((n, f) => n + f.audio.filter((v) => v === 0.5).length, 0);
  assert.equal(speechAudio, 24000);                      // all of the speech, once
  assert.equal(got[0].rgb.length, W * H * 3);
  assert.equal(await a.setIdle('angry'), false);
  await assert.rejects(a.speak(chunk(1)), OptionsError);
  a.destroy();
  await assert.rejects(a.frames(chunk(1)).next(), /destroyed/);
});
