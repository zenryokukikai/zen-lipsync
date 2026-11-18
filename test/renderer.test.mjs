// Unit tests of the offline renderer's parts that run without a browser: the WebM reader (frame-exact idle frames) and
// the output timeline shared by the Node and browser frames() (frame count, audio laid end to end).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readWebm, webCodecsCodec } from '../dist/runtime/webm.js';
import { OutputTimeline, framesOfMs, speechChunks } from '../dist/core/output-timeline.js';

// EBML element: id bytes + size (8-byte vint, or unknown) + body
const el = (id, body, unknown = false) => {
  const idb = []; for (let x = id; x > 0; x = Math.floor(x / 256)) idb.unshift(x & 255);
  const size = unknown ? [0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff] : [0x01, 0, 0, 0, (body.length >>> 24) & 255, (body.length >>> 16) & 255, (body.length >>> 8) & 255, body.length & 255];
  return Uint8Array.from([...idb, ...size, ...body]);
};
const cat = (...a) => { const o = new Uint8Array(a.reduce((n, x) => n + x.length, 0)); let p = 0; for (const x of a) { o.set(x, p); p += x.length; } return o; };
const uint = (v, n) => Array.from({ length: n }, (_, i) => (v >>> (8 * (n - 1 - i))) & 255);
const simple = (track, rel, key, payload) => el(0xa3, [0x80 | track, (rel >> 8) & 255, rel & 255, key ? 0x80 : 0, ...payload]);

function webm({ unknownCluster = false } = {}) {
  const ebml = el(0x1a45dfa3, [...el(0x4282, [...new TextEncoder().encode('webm')])]);
  const info = el(0x1549a966, [...el(0x2ad7b1, uint(1000000, 3))]);
  const video = el(0xe0, [...el(0xb0, uint(1264, 2)), ...el(0xba, uint(720, 2))]);
  const audioTrack = el(0xae, [...el(0xd7, [2]), ...el(0x83, [2]), ...el(0x86, [...new TextEncoder().encode('A_OPUS')])]);
  const videoTrack = el(0xae, [...el(0xd7, [1]), ...el(0x83, [1]), ...el(0x86, [...new TextEncoder().encode('V_VP9')]), ...video]);
  const tracks = el(0x1654ae6b, [...audioTrack, ...videoTrack]);
  const c1 = el(0x1f43b675, [...el(0xe7, [0]), ...simple(1, 0, true, [1, 1, 1]), ...simple(2, 0, true, [9]), ...simple(1, 40, false, [2, 2]), ...simple(1, 80, false, [3])], unknownCluster);
  const bg = el(0xa0, [...el(0xa1, [0x81, 0, 0, 0, 5, 5]), ...el(0xfb, [0xd8])]); // a BlockGroup with a reference: a delta frame
  const c2 = el(0x1f43b675, [...el(0xe7, uint(120, 2)), ...simple(1, 0, true, [4, 4, 4, 4]), ...bg]);
  return cat(ebml, el(0x18538067, [...info, ...tracks, ...c1, ...c2], true));
}

test('readWebm: the video track, its frames in order with key flags, times and byte ranges (audio blocks skipped)', () => {
  const buf = webm();
  const v = readWebm(buf);
  assert.deepEqual(v.track, { codecId: 'V_VP9', codecPrivate: null, width: 1264, height: 720 });
  assert.equal(webCodecsCodec(v.track), 'vp09.00.10.08');
  assert.deepEqual(v.frames.map((f) => [f.key, f.timeMs, f.size]), [[true, 0, 3], [false, 40, 2], [false, 80, 1], [true, 120, 4], [false, 120, 2]]);
  assert.deepEqual([...buf.subarray(v.frames[3].offset, v.frames[3].offset + v.frames[3].size)], [4, 4, 4, 4]);
  assert.deepEqual([...buf.subarray(v.frames[4].offset, v.frames[4].offset + v.frames[4].size)], [5, 5]);
});

test('readWebm: refuses what it cannot index frame by frame', () => {
  assert.throws(() => readWebm(new Uint8Array([1, 2, 3, 4])), /not a WebM/);
  const laced = webm(); const v = readWebm(laced); laced[v.frames[1].offset - 1] = 0x02; // Xiph lacing flag
  assert.throws(() => readWebm(laced), /laced/);
  assert.throws(() => webCodecsCodec({ codecId: 'V_MPEG4/ISO/AVC', codecPrivate: null, width: 1, height: 1 }), /not supported/);
});

test('OutputTimeline: N s of speech take ceil(N x 25) frames, idle frames before and after, chunks laid end to end', () => {
  const rate = 24000;
  const tl = new OutputTimeline(rate);
  const before = framesOfMs(400), after = 600;
  const frames = [];
  for (let i = 0; i < before; i++) frames.push(tl.next());
  const chunks = [0.31, 0.5, 0.123].map((s, j) => new Float32Array(Math.round(s * rate)).fill(j + 1));
  for (const c of chunks) { const n = tl.addChunk(c, rate); for (let i = 0; i < n; i++) frames.push(tl.next()); }
  const tail = tl.framesAfter(after);
  for (let i = 0; i < tail; i++) frames.push(tl.next());
  const N = chunks.reduce((n, c) => n + c.length, 0) / rate;
  assert.equal(frames.length, Math.ceil(N * 25) + 10 + 15);
  const audio = new Float32Array(frames.reduce((n, f) => n + f.audio.length, 0));
  let o = 0; for (const f of frames) { audio.set(f.audio, o); o += f.audio.length; }
  assert.equal(audio.length, (frames.length * rate) / 25);
  const speech = audio.subarray(0.4 * rate, 0.4 * rate + N * rate);
  assert.deepEqual([...new Set(speech.subarray(0, chunks[0].length))], [1]);
  assert.deepEqual([...new Set(speech.subarray(chunks[0].length, chunks[0].length + chunks[1].length))], [2]);
  assert.equal(speech[speech.length - 1], 3);
  assert.equal(audio[0.4 * rate + N * rate], 0);
  assert.deepEqual(frames.map((f) => f.index), frames.map((_, i) => i));
});

test('OutputTimeline: chunks at another rate are resampled to the output rate; speechChunks takes a clip or (async) iterables', async () => {
  const tl = new OutputTimeline(48000);
  const n = tl.addChunk(new Float32Array(16000).fill(0.5), 16000);
  assert.equal(n, 25);
  const got = [];
  for await (const c of speechChunks({ pcm: new Float32Array(1), sampleRate: 1 })) got.push(c);
  for await (const c of speechChunks([{ pcm: new Float32Array(2), sampleRate: 1 }])) got.push(c);
  assert.deepEqual(got.map((c) => c.pcm.length), [1, 2]);
  await assert.rejects((async () => { for await (const _ of speechChunks(42)) { /* nothing */ } })(), /audio must be/);
});
