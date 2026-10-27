// A WAV (RIFF/WAVE) reader for long recordings: the header is read once, then the samples are read straight from the
// file in bounded chunks, so a 3-minute (or 1-hour) recording is never held in memory as a whole. Mono float32 out
// (channels averaged), which is what the avatar takes.
import { open, type FileHandle } from 'node:fs/promises';

/** The file is not a WAV the reader understands (or not a WAV at all). */
export class WavFormatError extends Error {
  override readonly name = 'WavFormatError';
}

export interface WavInfo {
  /** Samples per second (e.g. 44100). */
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  /** 'pcm': signed integer samples; 'float': IEEE float samples. */
  format: 'pcm' | 'float';
  /** Sample frames (one sample per channel). */
  frames: number;
  durationSec: number;
  /** Bytes per sample frame (all channels). */
  blockAlign: number;
  /** True when the data chunk claims more bytes than the file holds (the tail is missing). */
  truncated: boolean;
}

export interface WavReader extends WavInfo {
  path: string;
  /** Up to `frames` sample frames from `startFrame`, as mono float32 (channels averaged); shorter at the end. */
  read(startFrame: number, frames: number): Promise<Float32Array>;
  close(): Promise<void>;
}

const RIFF = 0x46464952; // 'RIFF'
const WAVE = 0x45564157; // 'WAVE'
const FMT = 0x20746d66; // 'fmt '
const DATA = 0x61746164; // 'data'
/** WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of the sub-format GUID. */
const EXTENSIBLE = 0xfffe;

/** Opens a WAV file: reads the header only, then seeks to the samples on each read(). */
export async function openWav(path: string): Promise<WavReader> {
  const fh = await open(path, 'r');
  try {
    return await readHeader(fh, path);
  } catch (e) {
    await fh.close();
    throw e;
  }
}

async function readHeader(fh: FileHandle, path: string): Promise<WavReader> {
  const size = (await fh.stat()).size;
  const head = await readExactly(fh, 0, Math.min(size, 12));
  if (head.length < 12) throw new WavFormatError(`${path}: not a WAV file (too short)`);
  if (head.readUInt32LE(0) !== RIFF || head.readUInt32LE(8) !== WAVE) {
    throw new WavFormatError(`${path}: not a WAV file (RIFF/WAVE header missing). Convert it first, e.g. ffmpeg -i in.mp3 -ac 1 -ar 48000 out.wav`);
  }
  let at = 12;
  let fmt: { format: 'pcm' | 'float'; channels: number; sampleRate: number; bitsPerSample: number } | null = null;
  let data: { offset: number; bytes: number; declared: number } | null = null;
  while (at + 8 <= size) {
    const h = await readExactly(fh, at, 8);
    if (h.length < 8) break;
    const id = h.readUInt32LE(0);
    const len = h.readUInt32LE(4);
    const body = at + 8;
    if (id === FMT) {
      const b = await readExactly(fh, body, Math.min(len, 40));
      if (b.length < 16) throw new WavFormatError(`${path}: broken fmt chunk`);
      let tag = b.readUInt16LE(0);
      const channels = b.readUInt16LE(2);
      const sampleRate = b.readUInt32LE(4);
      const bitsPerSample = b.readUInt16LE(14);
      if (tag === EXTENSIBLE && b.length >= 26) tag = b.readUInt16LE(24);
      if (tag !== 1 && tag !== 3) throw new WavFormatError(`${path}: unsupported WAV encoding (format tag ${tag}); 8/16/24/32-bit PCM and 32/64-bit float are supported`);
      if (channels < 1) throw new WavFormatError(`${path}: ${channels} channels`);
      if (!(sampleRate > 0)) throw new WavFormatError(`${path}: sample rate ${sampleRate}`);
      fmt = { format: tag === 3 ? 'float' : 'pcm', channels, sampleRate, bitsPerSample };
    } else if (id === DATA) {
      // the data chunk runs to the end of the file when its length was left at 0 (streamed WAVs)
      const declared = len === 0 ? size - body : len;
      data = { offset: body, bytes: Math.min(declared, size - body), declared };
      if (fmt) break; // everything needed is known
    }
    at = body + len + (len & 1); // chunks are word-aligned
  }
  if (!fmt) throw new WavFormatError(`${path}: no fmt chunk`);
  if (!data) throw new WavFormatError(`${path}: no data chunk`);
  const bits = fmt.bitsPerSample;
  const ok = fmt.format === 'pcm' ? [8, 16, 24, 32].includes(bits) : [32, 64].includes(bits);
  if (!ok) throw new WavFormatError(`${path}: unsupported sample size (${fmt.format} ${bits} bit)`);
  const blockAlign = (bits / 8) * fmt.channels;
  const frames = Math.floor(data.bytes / blockAlign);
  const truncated = data.bytes < data.declared;
  const info: WavInfo = {
    sampleRate: fmt.sampleRate,
    channels: fmt.channels,
    bitsPerSample: bits,
    format: fmt.format,
    frames,
    durationSec: frames / fmt.sampleRate,
    blockAlign,
    truncated,
  };
  let closed = false;
  return {
    ...info,
    path,
    async read(startFrame: number, n: number): Promise<Float32Array> {
      if (closed) throw new WavFormatError(`${path}: the reader is closed`);
      const from = Math.max(0, Math.min(Math.floor(startFrame), frames));
      const count = Math.max(0, Math.min(Math.floor(n), frames - from));
      if (count === 0) return new Float32Array(0);
      const raw = await readExactly(fh, data!.offset + from * blockAlign, count * blockAlign);
      return toMono(raw, fmt!, blockAlign);
    },
    async close(): Promise<void> {
      if (closed) return;
      closed = true;
      await fh.close();
    },
  };
}

/** Reads exactly `n` bytes at `at`, looping over short reads; fewer bytes only at the end of the file. */
async function readExactly(fh: FileHandle, at: number, n: number): Promise<Buffer> {
  const buf = Buffer.allocUnsafe(n);
  let got = 0;
  while (got < n) {
    const r = await fh.read(buf, got, n - got, at + got);
    if (r.bytesRead === 0) break;
    got += r.bytesRead;
  }
  return got === n ? buf : buf.subarray(0, got);
}

/** Interleaved samples -> mono float32 in -1..1 (channels averaged). */
function toMono(raw: Buffer, fmt: { format: 'pcm' | 'float'; channels: number; bitsPerSample: number }, blockAlign: number): Float32Array {
  const n = Math.floor(raw.length / blockAlign);
  const ch = fmt.channels;
  const out = new Float32Array(n);
  const bits = fmt.bitsPerSample;
  const inv = ch;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let c = 0; c < ch; c++) {
      const o = i * blockAlign + c * (bits / 8);
      sum += sampleAt(raw, o, fmt.format, bits);
    }
    out[i] = sum / inv;
  }
  return out;
}

function sampleAt(b: Buffer, o: number, format: 'pcm' | 'float', bits: number): number {
  if (format === 'float') return bits === 32 ? b.readFloatLE(o) : b.readDoubleLE(o);
  switch (bits) {
    case 8: return (b.readUInt8(o) - 128) / 128; // 8-bit WAV is unsigned
    case 16: return b.readInt16LE(o) / 32768;
    case 24: return b.readIntLE(o, 3) / 8388608;
    default: return b.readInt32LE(o) / 2147483648;
  }
}
