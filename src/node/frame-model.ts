// The frame model on the CPU for Node: the native addon (native/node, AVX-512 VNNI > AVX2 > SSE4.1, linux-x64) or the
// WebAssembly SIMD128 build (cpu/lip8w.wasm, any platform). Same weights file (frame.lip8) and outputs.
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as path from 'node:path';
import { CpuFrameModel } from './cpu-frame.js';

export type NodeBackend = 'auto' | 'cpu-native' | 'cpu-wasm';
/** The kernel actually used: 'avx512' | 'avx2' | 'sse41' (native) or 'wasm'. */
export type FrameKernel = 'avx512' | 'avx2' | 'sse41' | 'wasm';

const FEAT = 32 * 20 * 44;
const Z = 640;
const OUT = 3 * 160 * 352;

interface Addon { init(w: Uint8Array, kernel?: string): FrameKernel; run(feat: Float32Array, z: Float32Array, out: Float32Array): void; supported(): string[] }

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

class WasmFrame implements FrameModel {
  readonly kernel = 'wasm' as const;
  constructor(private readonly m: CpuFrameModel) {}
  run(feat: Float32Array, z: Float32Array, out: Float32Array): void { out.set(this.m.frame(feat, z, 1)); }
}

/**
 * Loads the frame model. 'cpu-native' (or ZEN_LIPSYNC_KERNEL=avx512|avx2|sse41 to force a kernel) fails when the
 * addon is missing; 'auto' falls back to WebAssembly.
 */
export async function loadFrameModel(weights: ArrayBuffer, backend: NodeBackend): Promise<FrameModel> {
  if (backend !== 'cpu-wasm') {
    const p = nativeAddonPath();
    try {
      if (!p) throw new Error('the native addon is not built (run native/node/build.sh, or use backend: "cpu-wasm")');
      const a = createRequire(import.meta.url)(p) as Addon;
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
