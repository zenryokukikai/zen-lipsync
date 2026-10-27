// Tests of the audio-file entry (WAV reader, chunking, progress/abort wiring). Needs no avatar files or ffmpeg: the
// avatar is a stub with the same shape as NodeAvatar.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { openWav, WavFormatError, wavAudio, renderAudio, OptionsError } from '../dist/node/index.js';

/** A WAV file in memory: fmt (tag 1 pcm / 3 float / 0xFFFE extensible) + data. */
function wav({ sampleRate = 24000, channels = 1, bits = 16, format = 'pcm', frames = 100, extensible = false, value = () => 0, declaredBytes = null }) {
  const bytes = bits / 8;
  const blockAlign = bytes * channels;
  const data = Buffer.alloc(frames * blockAlign);
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < channels; c++) {
      const v = value(i, c);
      const o = i * blockAlign + c * bytes;
      if (format === 'float') { if (bits === 32) data.writeFloatLE(v, o); else data.writeDoubleLE(v, o); }
      else if (bits === 8) data.writeUInt8(Math.round(v * 128 + 128), o);
      else if (bits === 16) data.writeInt16LE(Math.round(v * 32767), o);
      else if (bits === 24) data.writeIntLE(Math.round(v * 8388607), o, 3);
      else data.writeInt32LE(Math.round(v * 2147483647), o);
    }
  }
  const tag = format === 'float' ? 3 : 1;
  const fmtSize = extensible ? 40 : 16;
  const fmt = Buffer.alloc(8 + fmtSize);
  fmt.write('fmt ', 0);
  fmt.writeUInt32LE(fmtSize, 4);
  fmt.writeUInt16LE(extensible ? 0xfffe : tag, 8);
  fmt.writeUInt16LE(channels, 10);
  fmt.writeUInt32LE(sampleRate, 12);
  fmt.writeUInt32LE(sampleRate * blockAlign, 16);
  fmt.writeUInt16LE(blockAlign, 20);
  fmt.writeUInt16LE(bits, 22);
  if (extensible) {
    fmt.writeUInt16LE(22, 24);           // cbSize
    fmt.writeUInt16LE(bits, 26);         // valid bits
    fmt.writeUInt32LE(0, 28);            // channel mask
    fmt.writeUInt16LE(tag, 32);          // sub-format GUID, first two bytes
    fmt.writeUInt16LE(0x0000, 34); fmt.write('000010008000aa00389b71', 36, 'hex');
  }
  const head = Buffer.alloc(12 + fmt.length + 8);
  head.write('RIFF', 0);
  head.writeUInt32LE(4 + fmt.length + 8 + data.length, 4);
  head.write('WAVE', 8);
  fmt.copy(head, 12);
  head.write('data', 12 + fmt.length);
  head.writeUInt32LE(declaredBytes ?? data.length, 12 + fmt.length + 4);
  return Buffer.concat([head, data]);
}

async function file(buf) {
  const dir = await mkdtemp(path.join(tmpdir(), 'zl-wav-'));
  const p = path.join(dir, 'in.wav');
  await writeFile(p, buf);
  return p;
}

test('openWav: 16-bit mono and stereo, values and length', async () => {
  const mono = await openWav(await file(wav({ frames: 480, value: (i) => (i % 2 ? 0.5 : -0.5) })));
  assert.deepEqual([mono.sampleRate, mono.channels, mono.bitsPerSample, mono.format], [24000, 1, 16, 'pcm']);
  assert.equal(mono.frames, 480);
  assert.equal(mono.durationSec, 0.02);
  assert.equal(mono.truncated, false);
  const a = await mono.read(0, 4);
  assert.equal(a.length, 4);
  assert.ok(Math.abs(a[0] - -0.5) < 0.001 && Math.abs(a[1] - 0.5) < 0.001);
  assert.equal((await mono.read(478, 100)).length, 2);   // short at the end
  assert.equal((await mono.read(480, 10)).length, 0);
  await mono.close();
  await assert.rejects(mono.read(0, 1), /closed/);

  const stereo = await openWav(await file(wav({ channels: 2, frames: 10, value: (i, c) => (c === 0 ? 0.4 : -0.4) })));
  assert.equal(stereo.channels, 2);
  const s = await stereo.read(0, 2);
  assert.ok(Math.abs(s[0]) < 0.001, 'channels are averaged');   // (0.4 + -0.4) / 2
  await stereo.close();
});

test('openWav: 8, 24 and 32-bit PCM, 32 and 64-bit float, and WAVE_FORMAT_EXTENSIBLE', async () => {
  const cases = [
    { bits: 8, format: 'pcm' }, { bits: 24, format: 'pcm' }, { bits: 32, format: 'pcm' },
    { bits: 32, format: 'float' }, { bits: 64, format: 'float' },
    { bits: 16, format: 'pcm', extensible: true }, { bits: 32, format: 'float', extensible: true },
  ];
  for (const c of cases) {
    const r = await openWav(await file(wav({ ...c, frames: 32, value: (i) => (i === 3 ? 0.25 : 0) })));
    assert.deepEqual([r.bitsPerSample, r.format], [c.bits, c.format], JSON.stringify(c));
    const got = await r.read(0, 8);
    assert.equal(got.length, 8, JSON.stringify(c));
    assert.ok(Math.abs(got[3] - 0.25) < 0.01, `${JSON.stringify(c)}: ${got[3]}`);
    assert.equal(got[4], 0);
    await r.close();
  }
});

test('openWav: a data chunk shorter than declared is truncated, a non-WAV is refused', async () => {
  const buf = wav({ frames: 100, declaredBytes: 400 });
  const cut = buf.subarray(0, buf.length - 100);              // 100 bytes of samples missing
  const r = await openWav(await file(cut));
  assert.equal(r.truncated, true);
  assert.equal(r.frames, 50);                                 // what is actually there
  assert.equal((await r.read(0, 1000)).length, 50);
  await r.close();

  await assert.rejects(openWav(await file(Buffer.from('ID3 not a wav file at all'))), WavFormatError);
  await assert.rejects(openWav(await file(Buffer.from('RIFF'))), WavFormatError);
  const bad = wav({ frames: 10 });
  bad.writeUInt16LE(7, 20);                                   // format tag 7 (mu-law): not supported
  await assert.rejects(openWav(await file(bad)), /unsupported WAV encoding/);
});

test('wavAudio: chunks cover every sample once, in order', async () => {
  const r = await openWav(await file(wav({ sampleRate: 1000, frames: 2500, value: (i) => (i / 2500) - 0.5 })));
  const chunks = [];
  for await (const c of wavAudio(r, { chunkSeconds: 0.4 })) chunks.push(c);
  assert.deepEqual(chunks.map((c) => c.pcm.length), [400, 400, 400, 400, 400, 400, 100]);
  assert.ok(chunks.every((c) => c.sampleRate === 1000));
  const total = chunks.reduce((n, c) => n + c.pcm.length, 0);
  assert.equal(total, 2500);
  const first = chunks[0].pcm[0], last = chunks.at(-1).pcm.at(-1);
  assert.ok(Math.abs(first - -0.5) < 0.01 && Math.abs(last - 0.5) < 0.01);
  // no sample is repeated or skipped across a chunk join
  const joined = Float32Array.from(chunks.flatMap((c) => [...c.pcm]));
  for (let i = 1; i < joined.length; i++) assert.ok(joined[i] > joined[i - 1], `monotonic at ${i}`);
});

test('wavAudio: an aborted signal stops the chunks', async () => {
  const r = await openWav(await file(wav({ sampleRate: 1000, frames: 5000 })));
  const ac = new AbortController();
  const it = wavAudio(r, { chunkSeconds: 0.5, signal: ac.signal });
  assert.equal((await it.next()).done, false);
  ac.abort();
  await assert.rejects(it.next(), (e) => e.name === 'AbortError');
  assert.ok(ac.signal.aborted);
});

test('renderAudio: passes the signal, the progress callback and the frame total through, and reports the rate', async () => {
  const seen = [];
  const avatar = {
    async render(input, opts) {
      const got = [];
      for await (const c of input) got.push(c.pcm.length);
      seen.push({ got, opts });
      opts.onProgress({ frames: 25, timeMs: 1000, totalFrames: opts.expectedFrames });
      return { file: opts.out, frames: 50, durationMs: 2000 };
    },
  };
  const r = await openWav(await file(wav({ sampleRate: 1000, frames: 2000 })));
  const progress = [];
  const res = await renderAudio(avatar, wavAudio(r, { chunkSeconds: 0.5 }), {
    out: '/tmp/x.mp4', crf: 18, idleBeforeMs: 500, expectedFrames: 75, audioMs: 2000,
    onProgress: (p) => progress.push(p),
  });
  assert.deepEqual(seen[0].got, [500, 500, 500, 500]);
  assert.equal(seen[0].opts.out, '/tmp/x.mp4');
  assert.equal(seen[0].opts.crf, 18);
  assert.equal(seen[0].opts.idleBeforeMs, 500);
  assert.equal(seen[0].opts.expectedFrames, 75);
  assert.equal(typeof seen[0].opts.onProgress, 'function');
  assert.equal(progress.length, 1);
  assert.equal(progress[0].ratio, 25 / 75);
  assert.equal(res.audioMs, 2000);
  assert.equal(res.frames, 50);
  assert.equal(res.rtf > 0, true);
  await assert.rejects(renderAudio(avatar, [], {}), OptionsError);
});

test('renderAudio: an abort removes the partial file and throws AbortError', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'zl-abort-'));
  const out = path.join(dir, 'out.mp4');
  await writeFile(out, 'partial');
  const ac = new AbortController();
  const avatar = {
    async render(_input, opts) {
      opts.signal.throwIfAborted();
      return { file: opts.out, frames: 0, durationMs: 0 };
    },
  };
  ac.abort();
  await assert.rejects(renderAudio(avatar, [], { out, signal: ac.signal }), (e) => e.name === 'AbortError');
});
