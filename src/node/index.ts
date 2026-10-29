// zen-lipsync/node (Node.js >= 26.12.1): the CPU runtime for servers. create() loads an avatar (files or URLs) and the two
// models on the CPU (the frame model native: AVX-512 VNNI > AVX2 > SSE4.1 on linux-x64, else WebAssembly); the
// avatar turns speech into video frames: frames() (raw RGB), render() (an MP4 / WebM file), stream() (fMP4 / WebM /
// JPEG for delivery). ffmpeg (an external program) decodes the idle video and encodes the outputs.
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { OptionsError } from '../core/errors.js';
import type { IdleSwitchFile } from '../core/idle-switch.js';
import type { LipModels } from '../core/lipsync.js';
import { listAvatar, modelPath, fileHash, MODEL_FILES, type PackRef } from '../core/pack.js';
import { resolveOptions } from '../core/options.js';
import { Worker } from 'node:worker_threads';
import { CpuSentenceModel, parseSentenceWeights } from '../core/sentence-model.js';
import { logMel, MEL_SAMPLE_RATE } from '../core/mel.js';
import { resample } from '../core/resample.js';
import { NodeAvatar } from './avatar.js';
import { resolveFfmpeg } from './ffmpeg.js';
import { frameF32Blob, loadFrameModel, loadFrameModelF32, type FrameKernel, type FrameModel, type FramePrecision, type NodeBackend } from './frame-model.js';
import type { IdleCache } from './idle-video.js';
import { NodePack } from './pack.js';
import { nodePackSource } from './pack-source.js';

export * from '../core/index.js';
export { nodePackSource } from './pack-source.js';
export { NodeAvatar, type NodeFrame, type FramesOptions, type RenderOptions, type RenderProgress, type StreamOptions, type VideoContainer } from './avatar.js';
export { renderAudio, renderAudioFile, wavAudio, type RenderAudioOptions, type RenderAudioFileOptions, type RenderAudioProgress, type RenderAudioResult } from './render-audio.js';
export { openWav, WavFormatError, type WavInfo, type WavReader } from './wav.js';
export { FfmpegMissingError } from './ffmpeg.js';
export { nativeAddonPath, type NodeBackend, type FrameKernel, type FramePrecision } from './frame-model.js';
export type { IdleCache } from './idle-video.js';

/** Options of create() on Node. */
export interface NodeCreateOptions {
  /**
   * The avatar: an avatar.json path or URL, or an avatar pack directory. A name without a path ("marine") means
   * avatars/marine/avatar.json under assetsPath / assetsBaseUrl.
   */
  avatar: string;
  /** Directory of the avatar files (e.g. a clone of zen-lipsync-avatars). */
  assetsPath?: string;
  /** Base URL of the avatar files (downloaded on create; the idle videos go to a temporary directory). */
  assetsBaseUrl?: string;
  /** 'auto' (default): the native frame model when it loads, else WebAssembly. */
  backend?: NodeBackend;
  /**
   * Arithmetic of the frame model. 'float32': the same output as the browser (frame.i8.wgw, else frame.wgw); native
   * only (AVX-512F or AVX2 with FMA, linux-x64). 'int8': frame.lip8, fastest (AVX-512 VNNI), its rounding can show as
   * fine grain around the mouth. Not given (default): float32 when the native float32 frame model loads, else int8
   * (backend 'cpu-wasm', no native addon, or a CPU without AVX2); avatar.kernel tells which. Given 'float32' with
   * backend 'cpu-wasm' is refused, and fails when the native float32 frame model does not load.
   */
  precision?: FramePrecision;
  /** Idle loop shown first (default: the avatar's default). */
  idle?: string;
  /** The ffmpeg program (default: ZEN_LIPSYNC_FFMPEG, else "ffmpeg" on PATH). */
  ffmpegPath?: string;
  /**
   * Idle video frames: 'stream' (default) decodes them as they are drawn (an ffmpeg process per idle loop, a few MB);
   * 'memory' decodes each loop once and keeps every frame (about 1.4 GB for Marine's two 500-frame loops at 1264x720).
   */
  idleCache?: IdleCache;
  onProgress?: (p: { loadedBytes: number; file: string }) => void;
}

/** The models of one avatar on the CPU, shared by every avatar of that model directory in this process. */
const modelCache = new Map<string, Promise<{ lips: LipModels; kernel: FrameKernel; prep: SpeechPrep }>>();

/** Loads an avatar and its models on the CPU. */
export async function create(options: NodeCreateOptions): Promise<NodeAvatar> {
  const t0 = performance.now();
  if (typeof options !== 'object' || options === null) throw new OptionsError('create(): options must be an object');
  const known = ['avatar', 'assetsPath', 'assetsBaseUrl', 'backend', 'precision', 'idle', 'ffmpegPath', 'idleCache', 'onProgress'];
  for (const k of Object.keys(options)) if (!known.includes(k)) throw new OptionsError(`create(): unknown option "${k}"`);
  const { assetsPath, assetsBaseUrl, backend = 'auto', precision, ffmpegPath, idleCache = 'stream' } = options;
  if (assetsPath !== undefined && assetsBaseUrl !== undefined) throw new OptionsError('create(): give assetsPath or assetsBaseUrl, not both');
  if (!['auto', 'cpu-native', 'cpu-wasm'].includes(backend)) throw new OptionsError('create(): "backend" must be auto, cpu-native or cpu-wasm');
  if (precision !== undefined && precision !== 'int8' && precision !== 'float32') throw new OptionsError('create(): "precision" must be int8 or float32');
  if (precision === 'float32' && backend === 'cpu-wasm') throw new OptionsError('create(): precision "float32" runs on the native frame model only (backend auto or cpu-native)');
  if (idleCache !== 'stream' && idleCache !== 'memory') throw new OptionsError('create(): "idleCache" must be stream or memory');
  let avatar = options.avatar;
  if (typeof avatar === 'string' && /^[a-z0-9_-]+$/i.test(avatar) && (assetsPath || assetsBaseUrl)) avatar = `avatars/${avatar}/avatar.json`;
  const base = assetsPath ?? assetsBaseUrl;
  const o = resolveOptions({ avatar, ...(base !== undefined ? { assetsBaseUrl: base } : {}), ...(options.idle !== undefined ? { idle: options.idle } : {}),
    ...(options.onProgress ? { onProgress: options.onProgress } : {}) });
  const ffmpeg = resolveFfmpeg(ffmpegPath);
  let loaded = 0;
  const src = nodePackSource((n) => { loaded += n; o.onProgress({ loadedBytes: loaded, file: '' }); });
  const listing = await listAvatar(src, o.avatar, o.idle);
  let tmp: string | null = null;
  const localDir = async (p: string): Promise<string> => {
    if (!/^https?:\/\//.test(p)) return p;
    tmp ??= await mkdtemp(path.join(tmpdir(), 'zen-lipsync-'));
    const f = path.join(tmp, `${Math.random().toString(36).slice(2)}${path.extname(new URL(p).pathname)}`);
    await writeFile(f, new Uint8Array(await src.bytes(p)));
    return f;
  };
  const first = listing.packs[0]!;
  const models = precision !== undefined ? loadModels(src, first, backend, precision)
    : backend === 'cpu-wasm' ? loadModels(src, first, backend, 'int8')
    : loadModels(src, first, backend, 'float32').catch(() => loadModels(src, first, backend, 'int8'));
  const packs = await Promise.all(listing.packs.map((p) => NodePack.load(src, p, { ffmpeg, idleCache, localDir })));
  const { lips, kernel, prep } = await models;
  let pairs: IdleSwitchFile | null = null;
  if (listing.set?.idleSwitch && packs.length > 1) pairs = await src.json<IdleSwitchFile>(src.join(listing.dir, listing.set.idleSwitch)).catch(() => null);
  return new NodeAvatar(new Map(packs.map((p) => [p.name, p])), lips, kernel, prep, pairs, listing.set, ffmpeg, performance.now() - t0, loaded);
}

/** The float32 frame model's weights: frame.i8.wgw (as the browser), else frame.wgw. */
async function loadF32(src: ReturnType<typeof nodePackSource>, pack: PackRef): Promise<{ frame: FrameModel; bytes: number }> {
  const i8 = modelPath(src, pack, MODEL_FILES.frameI8);
  let file: ArrayBuffer, format: 'i8' | 'f32' = 'i8';
  try { file = await src.bytes(i8.path, fileHash(pack.meta, i8.rel)); } catch {
    const f = modelPath(src, pack, MODEL_FILES.frameF32);
    file = await src.bytes(f.path, fileHash(pack.meta, f.rel));
    format = 'f32';
  }
  return { frame: loadFrameModelF32(frameF32Blob(file, format)), bytes: file.byteLength };
}

function loadModels(src: ReturnType<typeof nodePackSource>, pack: PackRef, backend: NodeBackend, precision: FramePrecision): Promise<{ lips: LipModels; kernel: FrameKernel; prep: SpeechPrep }> {
  const sm = modelPath(src, pack, MODEL_FILES.sentence);
  const fm = modelPath(src, pack, precision === 'float32' ? MODEL_FILES.frameI8 : MODEL_FILES.frameCpu);
  const key = `${fm.path}|${backend}|${precision}`;
  let m = modelCache.get(key);
  if (!m) {
    m = (async () => {
      const sbP = src.bytes(sm.path, fileHash(pack.meta, sm.rel));
      const fP = precision === 'float32' ? loadF32(src, pack)
        : src.bytes(fm.path, fileHash(pack.meta, fm.rel)).then(async (fb) => ({ frame: await loadFrameModel(fb, backend), bytes: fb.byteLength }));
      const [sb, { frame, bytes: fbBytes }] = await Promise.all([sbP, fP]);
      const prep = speechPrep(sb);
      const out = new Float32Array(3 * 160 * 352);
      const lips: LipModels = {
        backend: 'cpu',
        weightBytes: sb.byteLength + fbBytes,
        sentence: (mel, ref, T) => prep.sentence(mel, ref, T),
        frame: async (feat, z, n) => {
          const res = new Float32Array(n * out.length);
          for (let i = 0; i < n; i++) {
            frame.run(feat.subarray(i * 28160, (i + 1) * 28160), z.subarray(i * 640, (i + 1) * 640), out);
            res.set(out, i * out.length);
          }
          return res;
        },
      };
      return { lips, kernel: frame.kernel, prep };
    })();
    m.catch(() => modelCache.delete(key));
    modelCache.set(key, m);
  }
  return m;
}

/** Speech preparation off the main thread: log-mel features and the sentence model (see sentence-worker.ts). */
export interface SpeechPrep {
  mel(pcm: Float32Array, sampleRate: number): Promise<{ mel: Float32Array; melFrames: number; frames: number; gather: number[] }>;
  sentence(mel: Float32Array, ref: Float32Array, T: number): Promise<Float32Array>;
}

/**
 * The speech preparation in a worker thread (about 0.35 s of CPU per second of speech; the frames go on being drawn
 * meanwhile), else in this thread when workers are not available. The worker does not keep the process alive.
 */
function speechPrep(weights: ArrayBuffer): SpeechPrep {
  let w: Worker | null = null;
  try {
    w = new Worker(new URL('./sentence-worker.js', import.meta.url), { workerData: { weights: weights.slice(0) } });
    w.unref();
  } catch { w = null; }
  if (!w) {
    const m = new CpuSentenceModel(parseSentenceWeights(weights));
    return {
      mel: async (pcm, rate) => logMel(rate === MEL_SAMPLE_RATE ? pcm : resample(pcm, rate, MEL_SAMPLE_RATE)),
      sentence: async (mel, ref, T) => m.run(mel, ref, T),
    };
  }
  const worker = w;
  let seq = 0;
  const waiting = new Map<number, { resolve: (v: never) => void; reject: (e: Error) => void }>();
  worker.on('message', (m: { id: number; error?: string }) => {
    const p = waiting.get(m.id);
    if (!p) return;
    waiting.delete(m.id);
    if (waiting.size === 0) worker.unref();
    if (m.error) p.reject(new Error(m.error)); else p.resolve(m as never);
  });
  worker.on('error', (e) => { for (const p of waiting.values()) p.reject(e); waiting.clear(); });
  const call = <T>(msg: object): Promise<T> => new Promise<T>((resolve, reject) => {
    const id = ++seq;
    waiting.set(id, { resolve: resolve as (v: never) => void, reject });
    worker.ref();
    worker.postMessage({ id, ...msg });
  });
  return {
    mel: async (pcm, sampleRate) => call({ kind: 'mel', pcm, sampleRate }),
    sentence: async (mel, ref, T) => (await call<{ z: Float32Array }>({ kind: 'sentence', mel, ref, T })).z,
  };
}
