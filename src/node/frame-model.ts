// The frame model on the CPU for Node, two precisions:
// - 'float32': the WebGPU frame model's arithmetic on its weights (frame.i8.wgw dequantised at load, as the browser,
//   or frame.wgw), native only (native/x86/lipf.c: AVX-512F > AVX2+FMA, linux-x64). Same output as the browser.
// - 'int8': frame.lip8 (int8 weights and int8 activations), native (AVX-512 VNNI > AVX2 > SSE4.1) or the WebAssembly
//   SIMD128 build (cpu/lip8w.wasm, any platform). Faster; its rounding shows as fine grain on some avatars.
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as path from 'node:path';
import { CpuFrameModel } from './cpu-frame.js';
import type { NativeSentenceConv } from '../core/sentence-model.js';
import { dequantFrameI8, parseF32Weights, parseI8Weights } from '../core/frame-weights.js';

export type NodeBackend = 'auto' | 'cpu-native' | 'cpu-wasm';
/** Arithmetic of the frame model: 'float32' (as the browser) or 'int8' (faster, coarser). */
export type FramePrecision = 'float32' | 'int8';
/** The kernel actually used: float32 'avx512-f32' | 'avx2-f32'; int8 'avx512' | 'avx2' | 'sse41' (native) or 'wasm'. */
export type FrameKernel = 'avx512-f32' | 'avx2-f32' | 'avx512' | 'avx2' | 'sse41' | 'wasm';

const FEAT = 32 * 20 * 44;
const Z = 640;
const OUT = 3 * 160 * 352;

interface Addon {
  init(w: Uint8Array, kernel?: string): FrameKernel;
  run(feat: Float32Array, z: Float32Array, out: Float32Array): void;
  supported(): string[];
  createF32(blob: Float32Array, kernel?: string): { model: unknown; kernel: 'avx512' | 'avx2' };
  runF32(model: unknown, feat: Float32Array, z: Float32Array, out: Float32Array): void;
  blobSizeF32(): number;
  sconv?(x: Float32Array, w: Float32Array, b: Float32Array, C: number, T: number, dil: number, r: Float32Array | null, y: Float32Array): void;
  /** The kernel sconv runs ('avx512' | 'avx2'), '' when the CPU has neither. */
  sconvKernel?(): string;
}

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

/** The native addon: ZEN_LIPSYNC_NATIVE, else native/node/build/zen_lipsync.node of the package, else prebuilds/<platform>-<arch>/. */
export function nativeAddonPath(): string | null {
  const cands = [process.env['ZEN_LIPSYNC_NATIVE'], path.join(ROOT, 'native', 'node', 'build', 'zen_lipsync.node'),
    path.join(ROOT, 'prebuilds', `${process.platform}-${process.arch}`, 'zen_lipsync.node')].filter((p): p is string => !!p);
  return cands.find((p) => existsSync(p)) ?? null;
}

export interface FrameModel {
  readonly kernel: FrameKernel;
  /** One frame: feat [28160], z [640] -> out [168960] (out is reused by the caller). */
  run(feat: Float32Array, z: Float32Array, out: Float32Array): void;
}

class NativeFrame implements FrameModel {
  constructor(private readonly a: Addon, readonly kernel: FrameKernel) {}
  run(feat: Float32Array, z: Float32Array, out: Float32Array): void { this.a.run(feat, z, out); }
}

class NativeF32Frame implements FrameModel {
  constructor(private readonly a: Addon, private readonly m: unknown, readonly kernel: FrameKernel) {}
  run(feat: Float32Array, z: Float32Array, out: Float32Array): void { this.a.runF32(this.m, feat, z, out); }
}

class WasmFrame implements FrameModel {
  readonly kernel = 'wasm' as const;
  constructor(private readonly m: CpuFrameModel) {}
  run(feat: Float32Array, z: Float32Array, out: Float32Array): void { out.set(this.m.frame(feat, z, 1)); }
}

/** The frame model's float32 tensors in the order native/x86/lipf.c reads them (the WebGPU frame model's layout). */
const F32_ORDER = ['zfc0.w', 'zfc0.b', 'zfc2.w', 'zfc2.b', ...['inp', 'body.0', 'body.1', 'body.2', 'body.3', 'body.4', 'body.5', 'mid',
  'up.0.0', 'up.0.1', 'up.1.0', 'up.1.1', 'up.2.0', 'up.2.1', 'out'].flatMap((n) => [`${n}.w`, `${n}.b`]), 'win'];

/** frame.i8.wgw (dequantised as the browser does) or frame.wgw -> one Float32Array in F32_ORDER. */
export function frameF32Blob(file: ArrayBuffer, format: 'i8' | 'f32'): Float32Array {
  const w = format === 'i8' ? dequantFrameI8(parseI8Weights(file)) : parseF32Weights(file);
  const parts = F32_ORDER.map((k) => {
    const v = w[k];
    if (!(v instanceof Float32Array)) throw new Error(`frame model weights: ${k} missing`);
    return v;
  });
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

function addon(): Addon {
  const p = nativeAddonPath();
  if (!p) throw new Error('the native addon is not built (run native/node/build.sh, or use backend: "cpu-wasm")');
  return createRequire(import.meta.url)(p) as Addon;
}

/**
 * The float32 frame model (native only). blob: frameF32Blob(). ZEN_LIPSYNC_KERNEL=avx512|avx2 forces a kernel.
 */
export function loadFrameModelF32(blob: Float32Array): FrameModel {
  const a = addon();
  if (typeof a.createF32 !== 'function') throw new Error('the native addon has no float32 frame model (rebuild it: native/node/build.sh)');
  const want = process.env['ZEN_LIPSYNC_KERNEL'];
  const r = a.createF32(blob, want === 'avx512' || want === 'avx2' ? want : 'auto');
  return new NativeF32Frame(a, r.model, `${r.kernel}-f32`);
}

/** The native conv1d of the sentence model, when the addon has it and this CPU can run it (else null). */
export function nativeSentenceConv(): NativeSentenceConv | null {
  try {
    const a = addon();
    return typeof a.sconv === 'function' && typeof a.sconvKernel === 'function' && a.sconvKernel() ? a.sconv.bind(a) : null;
  } catch { return null; }
}

/**
 * The sentence model's conv1d for this process: the addon's (AVX-512 / AVX2), else the same C built to WebAssembly
 * SIMD128 (cpu/sconvw.wasm: about 4x the TypeScript loop on Apple silicon), else null (the TypeScript loop). The
 * latents of all three agree within float32 rounding (each product rounded to float32 in C, kept in float64 in V8).
 */
export async function sentenceConv(): Promise<NativeSentenceConv | null> {
  const n = nativeSentenceConv();
  if (n) return n;
  try { return await wasmSentenceConv(); } catch { return null; }
}

interface SconvModule { HEAPF32: Float32Array; _malloc(n: number): number; _free(p: number): void; _sconv(x: number, w: number, b: number, C: number, T: number, dil: number, r: number, y: number): void }

/**
 * The conv on cpu/sconvw.wasm: inputs copied into its memory (one block, grown as needed and kept for the next call:
 * at most one sentence's worth, about 2 MB per 10 s of speech), the output copied out.
 */
export async function wasmSentenceConv(): Promise<NativeSentenceConv> {
  const create = (await import(`${new URL('../../cpu', import.meta.url).href}/sconvw.mjs`)).default as () => Promise<SconvModule>;
  const m = await create();
  let base = 0, cap = 0;
  const conv: NativeSentenceConv = (x, w, b, C, T, dil, r, y) => {
    if (!(C > 0 && T > 0 && dil > 0) || x.length < C * T || w.length < 256 * C * 3 || b.length < 256 || y.length < 256 * T || (r && r.length < 256 * T)) throw new Error('sconv: sizes');
    const n = C * T + 256 * C * 3 + 256 + (r ? 256 * T : 0) + 256 * T;
    if (n * 4 > cap) {
      if (base) m._free(base);
      base = 0; cap = 0;
      const p = m._malloc(n * 4);
      if (!p) throw new Error('sconv: out of WebAssembly memory');
      base = p; cap = n * 4;
    }
    const H = m.HEAPF32; // read after the malloc: a grown memory has a new view
    let o = base >>> 2;
    const px = o; H.set(x.subarray(0, C * T), o); o += C * T;
    const pw = o; H.set(w.subarray(0, 256 * C * 3), o); o += 256 * C * 3;
    const pb = o; H.set(b.subarray(0, 256), o); o += 256;
    let pr = 0;
    if (r) { pr = o; H.set(r.subarray(0, 256 * T), o); o += 256 * T; }
    m._sconv(px * 4, pw * 4, pb * 4, C, T, dil, pr * 4, o * 4);
    y.set(m.HEAPF32.subarray(o, o + 256 * T));
  };
  return conv;
}

/**
 * Loads the int8 frame model (frame.lip8). 'cpu-native' (or ZEN_LIPSYNC_KERNEL=avx512|avx2|sse41 to force a kernel)
 * fails when the addon is missing; 'auto' falls back to WebAssembly.
 */
export async function loadFrameModel(weights: ArrayBuffer, backend: NodeBackend): Promise<FrameModel> {
  if (backend !== 'cpu-wasm') {
    try {
      const a = addon();
      const kernel = a.init(new Uint8Array(weights), process.env['ZEN_LIPSYNC_KERNEL'] ?? 'auto');
      return new NativeFrame(a, kernel);
    } catch (e) {
      if (backend === 'cpu-native') throw e;
    }
  }
  const cpuBase = new URL('../../cpu', import.meta.url).href;
  // the relaxed-SIMD build checks out slower on x86 (54.6 vs 27.7 ms per frame in Chromium); SIMD128 is used on Node
  const create = (await import(`${cpuBase}/lip8w.mjs`)).default as () => Promise<unknown>;
  return new WasmFrame(await CpuFrameModel.fromFactory(create, weights));
}

export const FRAME_SIZES = { FEAT, Z, OUT } as const;
