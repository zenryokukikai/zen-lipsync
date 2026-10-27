// Frame model on the CPU (for the Node entry, zen-lipsync/node): WebAssembly (int8, cpu/lip8r.wasm or cpu/lip8w.wasm,
// single thread). Same inputs and outputs as the WebGPU frame model. The browser entry is WebGPU only.

interface Lip8Module {
  HEAPU8: Uint8Array;
  HEAPF32: Float32Array;
  _malloc(n: number): number;
  _free(p: number): void;
  _lip_init(p: number, n: number): number;
  _lip_run(feat: number, z: number, out: number): void;
}

const FEAT_SIZE = 32 * 20 * 44;
const Z_IN = 640;
const OUT_SIZE = 3 * 160 * 352;

/** Frame model on the CPU: frame.lip8 weights (int8), one frame per call. */
export class CpuFrameModel {
  private constructor(private readonly m: Lip8Module, private readonly fp: number, private readonly zp: number, private readonly op: number) {}

  /** cpuBase: the directory with lip8r.mjs / lip8w.mjs; weights: the frame.lip8 file. relaxed SIMD is tried first. */
  static async create(cpuBase: string, weights: ArrayBuffer): Promise<CpuFrameModel> {
    let create: () => Promise<Lip8Module>;
    try { create = (await import(`${cpuBase}/lip8r.mjs`)).default; await create(); } catch {
      create = (await import(`${cpuBase}/lip8w.mjs`)).default;
    }
    return CpuFrameModel.fromFactory(create, weights);
  }

  /** create: an Emscripten module factory of cpu/lip8w.mjs or cpu/lip8r.mjs. */
  static async fromFactory(create: () => Promise<unknown>, weights: ArrayBuffer): Promise<CpuFrameModel> {
    const m = (await create()) as Lip8Module;
    const w = new Uint8Array(weights);
    const wp = m._malloc(w.length);
    m.HEAPU8.set(w, wp);
    const rc = m._lip_init(wp, w.length);
    m._free(wp);
    if (rc) throw new Error(`frame.lip8: init error ${rc}`);
    return new CpuFrameModel(m, m._malloc(FEAT_SIZE * 4), m._malloc(Z_IN * 4), m._malloc(OUT_SIZE * 4));
  }

  /** feat [n,32,20,44], z [n,640] -> delta [n,3,160,352] (same as WgslLips.frame). */
  frame(feat: Float32Array, z: Float32Array, n: number): Float32Array {
    const out = new Float32Array(n * OUT_SIZE);
    for (let i = 0; i < n; i++) {
      this.m.HEAPF32.set(feat.subarray(i * FEAT_SIZE, (i + 1) * FEAT_SIZE), this.fp >> 2);
      this.m.HEAPF32.set(z.subarray(i * Z_IN, (i + 1) * Z_IN), this.zp >> 2);
      this.m._lip_run(this.fp, this.zp, this.op);
      out.set(this.m.HEAPF32.subarray(this.op >> 2, (this.op >> 2) + OUT_SIZE), i * OUT_SIZE);
    }
    return out;
  }
}
