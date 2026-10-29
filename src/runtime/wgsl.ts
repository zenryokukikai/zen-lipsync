import { fetchModel } from './model-cache.js';
import { dequantFrameI8, parseF32Weights, parseI8Weights, type I8File, type WeightFile } from '../core/frame-weights.js';
export { dequantFrameI8 } from '../core/frame-weights.js';
// ZEN-LipSync WebGPU runtime: the two lip-sync models in hand-written WGSL (no ML framework on the page).
//   frame model:    per video frame. Inputs: the avatar's precomputed features of that source frame [32,20,44] and
//                   5 frames of mouth-motion latents [640]. Output: the RGB change of the mouth box [3,160,352].
//   sentence model: per sentence. Inputs: log-mel [T,80] and the avatar's reference latent per mel frame [T,128].
//                   Output: the mouth-motion latent per mel frame [T,128].
// Every GPU buffer is created once (the sentence model's grow when a longer sentence comes); an inference creates none.
// Weights (in the avatar pack): frame.i8.wgw (int8 per output channel, default; dequantised once at load, then the
// float32 kernels run) or frame.wgw (float32), and sentence.wgw (float32).

/* eslint-disable @typescript-eslint/no-explicit-any */
type GpuDevice = any;
type GpuBuffer = any;
type GpuBindGroup = any;
type GpuPipeline = any;

const STORAGE = 0x80;
const UNIFORM = 0x40;
const COPY_DST = 0x08;
const COPY_SRC = 0x04;
const MAP_READ = 0x01;
const WG = 64;

/** Frame model: max frames per run. */
export const WGSL_MAX_BATCH = 8;
const FEAT = [32, 20, 44] as const;
const FEAT_SIZE = 32 * 20 * 44;
const Z_IN = 640;
const BOX_H = 160;
const BOX_W = 352;

const ACT = `fn act(v: f32, a: u32) -> f32 {
  if (a == 1u) { return v / (1.0 + exp(-v)); }
  if (a == 2u) { return max(v, 0.0); }
  return v;
}`;

/** Frame model weights: int8 dequantised once at load ('i8', default), float32 ('f32'), or int8 dequantised in the shader at every use ('i8q'). */
import type { LipWeights } from '../core/types.js';
export type { LipWeights } from '../core/types.js';

/** The client log / bench label of a LipWeights. */
export const lipWeightsLabel = (w: LipWeights): string => (w === 'i8' ? 'int8' : w === 'i8q' ? 'int8q' : 'float32');

/** 4 signed bytes of a u32 word (byte j = element j) as float32. */
const UNPACK_I8 = `fn i8x4(word: u32) -> vec4<f32> {
  let u = bitcast<i32>(word);
  return vec4<f32>(f32((u << 24u) >> 24u), f32((u << 16u) >> 24u), f32((u << 8u) >> 24u), f32(u >> 24u));
}`;

/** y[b,o] = act(scale * sum_i W[o,i] x[b,i] + bias[o]) */
const GEMM = `${ACT}
struct P { B: u32, I: u32, O: u32, act: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  if (n >= p.B * p.O) { return; }
  let b = n / p.O;
  let o = n % p.O;
  var acc = 0.0;
  for (var i = 0u; i < p.I; i++) { acc += w[o * p.I + i] * x[b * p.I + i]; }
  y[n] = act(acc * p.scale + bias[o], p.act);
}`;

/** GEMM with int8 weights: w[o, i/4] holds inputs i..i+3 in its bytes; ws[o] the row scale. */
const GEMM_I8 = `${ACT}
${UNPACK_I8}
struct P { B: u32, I: u32, O: u32, act: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> ws: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  if (n >= p.B * p.O) { return; }
  let b = n / p.O;
  let o = n % p.O;
  let I4 = p.I / 4u;
  var acc = 0.0;
  for (var i = 0u; i < I4; i++) { acc += dot(i8x4(w[o * I4 + i]), x[b * I4 + i]); }
  y[n] = act(acc * ws[o] * p.scale + bias[o], p.act);
}`;

/** x[b] = concat(feat[b] (32 channels), h[b] (8 channels)) over 20x44 */
const CONCAT = `
struct P { B: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> h: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  if (n >= p.B * 40u * 880u) { return; }
  let b = n / (40u * 880u);
  let r = n % (40u * 880u);
  let c = r / 880u;
  let s = r % 880u;
  if (c < 32u) { y[n] = feat[(b * 32u + c) * 880u + s]; } else { y[n] = h[(b * 8u + c - 32u) * 880u + s]; }
}`;

/**
 * 3x3 conv, pad 1, one thread = 4 output channels at one pixel. Input activation act (0 none, 1 SiLU, 2 ReLU).
 * mode 0: y = conv (+ r if res), 1: DepthToSpace(2, CRD) of the output, 2: the frame model's last layer: 3 channels
 * times 64 times the window r[y,x].
 */
const conv2dShader = (i8: boolean): string => `${ACT}
${i8 ? UNPACK_I8 : ''}
struct P { B: u32, C: u32, O4: u32, H: u32, W: u32, act: u32, mode: u32, res: u32, cOut: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${i8 ? 'u32' : 'vec4<f32>'}>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> r: array<f32>;
${i8 ? '@group(0) @binding(6) var<storage, read> ws: array<vec4<f32>>;' : ''}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let HW = p.H * p.W;
  let n = g.x + g.y * 65535u * ${WG}u;
  if (n >= p.B * p.O4 * HW) { return; }
  let b = n / (p.O4 * HW);
  let o4 = (n / HW) % p.O4;
  let yy = i32((n % HW) / p.W);
  let xx = i32(n % p.W);
  var acc = ${i8 ? 'vec4<f32>(0.0)' : 'bias[o4]'};
  for (var c = 0u; c < p.C; c++) {
    let xb = (b * p.C + c) * HW;
    let wb = (o4 * p.C + c) * 9u;
    for (var ky = 0; ky < 3; ky++) {
      let iy = yy + ky - 1;
      if (iy < 0 || iy >= i32(p.H)) { continue; }
      for (var kx = 0; kx < 3; kx++) {
        let ix = xx + kx - 1;
        if (ix < 0 || ix >= i32(p.W)) { continue; }
        let v = act(x[xb + u32(iy) * p.W + u32(ix)], p.act);
        acc += v * ${i8 ? 'i8x4(w[wb + u32(ky * 3 + kx)])' : 'w[wb + u32(ky * 3 + kx)]'};
      }
    }
  }
  ${i8 ? 'acc = acc * ws[o4] + bias[o4];' : ''}
  let py = u32(yy);
  let px = u32(xx);
  if (p.mode == 0u) {
    for (var j = 0u; j < 4u; j++) {
      let oc = o4 * 4u + j;
      if (oc >= p.cOut) { break; }
      let i = (b * p.cOut + oc) * HW + py * p.W + px;
      var v = acc[j];
      if (p.res == 1u) { v += r[i]; }
      y[i] = v;
    }
  } else if (p.mode == 1u) {
    let W2 = p.W * 2u;
    let base = (b * p.O4 + o4) * HW * 4u;
    for (var j = 0u; j < 4u; j++) {
      y[base + (py * 2u + j / 2u) * W2 + px * 2u + j % 2u] = acc[j];
    }
  } else {
    let win = r[py * p.W + px] * 64.0;
    for (var j = 0u; j < 3u; j++) { y[(b * 3u + j) * HW + py * p.W + px] = acc[j] * win; }
  }
}`;

/** GroupNorm (8 groups) statistics at each time step of x [C][T]: stats[(g*T+t)*2] = mean, 1/sqrt(var+eps). */
const GN_STATS = `
struct P { C: u32, T: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  if (n >= 8u * p.T) { return; }
  let grp = n / p.T;
  let t = n % p.T;
  let cg = p.C / 8u;
  var m = 0.0;
  for (var c = 0u; c < cg; c++) { m += x[(grp * cg + c) * p.T + t]; }
  m /= f32(cg);
  var v = 0.0;
  for (var c = 0u; c < cg; c++) { let d = x[(grp * cg + c) * p.T + t] - m; v += d * d; }
  v /= f32(cg);
  stats[n * 2u] = m;
  stats[n * 2u + 1u] = 1.0 / sqrt(v + 1e-5);
}`;

/** Normalised and SiLU'd x[c,t] (GroupNorm 8 groups; gb = gamma then beta). */
const GN_READ = `fn gnSilu(c: u32, t: u32) -> f32 {
  let s = ((c / (p.C / 8u)) * p.T + t) * 2u;
  let v = (x[c * p.T + t] - stats[s]) * stats[s + 1u] * gb[c] + gb[p.C + c];
  return v / (1.0 + exp(-v));
}`;

/** Dilated conv1d, kernel 3, "same" padding, on [C][T]; input raw (gn 0) or GroupNorm+SiLU (gn 1); + r if res. */
const CONV1D = `
struct P { C: u32, O4: u32, T: u32, dil: u32, gn: u32, res: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read> gb: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(7) var<storage, read> r: array<f32>;
${GN_READ}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  if (n >= p.O4 * p.T) { return; }
  let o4 = n / p.T;
  let t = i32(n % p.T);
  var acc = bias[o4];
  for (var c = 0u; c < p.C; c++) {
    for (var k = 0; k < 3; k++) {
      let tt = t + (k - 1) * i32(p.dil);
      if (tt < 0 || tt >= i32(p.T)) { continue; }
      var v: f32;
      if (p.gn == 1u) { v = gnSilu(c, u32(tt)); } else { v = x[c * p.T + u32(tt)]; }
      acc += v * w[(o4 * p.C + c) * 3u + u32(k)];
    }
  }
  for (var j = 0u; j < 4u; j++) {
    let i = (o4 * 4u + j) * p.T + u32(t);
    var v = acc[j];
    if (p.res == 1u) { v += r[i]; }
    y[i] = v;
  }
}`;

/** z[t,k] = (sum_c gnSilu(c,t) W[c,k] + b[k]) * scale[k] + shift[k]; hss = b, scale, shift. */
const HEAD = `
struct P { C: u32, T: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> gb: array<f32>;
@group(0) @binding(4) var<storage, read> w: array<f32>;
@group(0) @binding(5) var<storage, read> hss: array<f32>;
@group(0) @binding(6) var<storage, read_write> z: array<f32>;
${GN_READ}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  if (n >= p.T * 128u) { return; }
  let t = n / 128u;
  let k = n % 128u;
  var acc = 0.0;
  for (var c = 0u; c < p.C; c++) { acc += gnSilu(c, t) * w[c * 128u + k]; }
  z[n] = (acc + hss[k]) * hss[128u + k] + hss[256u + k];
}`;

let loadedBytes = 0;

async function loadWeightsI8(url: string): Promise<I8File> {
  const res = await fetchModel(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  loadedBytes += buf.byteLength;
  return parseI8Weights(buf);
}

async function loadWeights(url: string): Promise<WeightFile> {
  const res = await fetchModel(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  loadedBytes += buf.byteLength;
  return parseF32Weights(buf);
}

interface Step { pipeline: GpuPipeline; bind: GpuBindGroup; uniform: GpuBuffer; params: () => ArrayBuffer; threads: () => number }

/** u32 fields (and f32 ones where given as {f: value}) of a uniform struct, padded to 16 bytes. */
function uniformData(fields: (number | { f: number })[]): ArrayBuffer {
  const n = Math.ceil(fields.length / 4) * 4;
  const b = new ArrayBuffer(n * 4);
  const u = new Uint32Array(b);
  const f = new Float32Array(b);
  fields.forEach((v, i) => { if (typeof v === 'number') u[i] = v; else f[i] = v.f; });
  return b;
}

/** Both lip-sync models on one GPUDevice. */
export class WgslLips {
  readonly backend = 'webgpu' as const;
  private readonly pipelines: Record<string, GpuPipeline> = {};
  private readonly weights = new Map<string, GpuBuffer>();
  private frameSteps: Step[] = [];
  private sentenceSteps: Step[] = [];
  private batch = 1;
  private T = 0;
  private capT = 0;
  private sentenceBuffers: GpuBuffer[] = [];
  private sentenceIn!: GpuBuffer;
  private sentenceOut!: GpuBuffer;
  private sentenceRead!: GpuBuffer;
  private featIn!: GpuBuffer;
  private zIn!: GpuBuffer;
  private frameOut!: GpuBuffer;
  private frameRead!: GpuBuffer;
  private dummy!: GpuBuffer;
  /** Bytes of every buffer created here (weights + scratch). */
  bytes = 0;
  /** Bytes of the two weight files. */
  weightBytes = 0;
  /** Bytes of the frame model's weight buffers on the GPU. */
  frameWeightBytes = 0;
  /** Load breakdown: both weight files fetched (ms), and the int8 -> float32 dequant at load ('i8' only, ms). */
  downloadMs = 0;
  dequantMs = 0;

  /** The weights requested (requested) and the ones loaded (format: 'f32' when the pack has no frame.i8.wgw). */
  readonly requested: LipWeights;
  i8Fallback: string | null = null;

  private constructor(readonly device: GpuDevice, public format: LipWeights) { this.requested = format; }

  /**
   * base: the avatar pack (frame.i8.wgw or frame.wgw, sentence.wgw). An int8 request on a pack without frame.i8.wgw
   * (or a failed int8 fetch) loads frame.wgw instead (format becomes 'f32', i8Fallback the reason).
   */
  static async create(device: GpuDevice, base: string, format: LipWeights = 'i8'): Promise<WgslLips> {
    const m = new WgslLips(device, format);
    const before = loadedBytes;
    const tf = performance.now();
    const frameFile = async (): Promise<WeightFile | I8File> => {
      if (m.format === 'f32') return loadWeights(`${base}/frame.wgw`);
      try { return await loadWeightsI8(`${base}/frame.i8.wgw`); } catch (e) {
        m.i8Fallback = e instanceof Error ? e.message : String(e);
        m.format = 'f32';
        return loadWeights(`${base}/frame.wgw`);
      }
    };
    const [sf, bw] = await Promise.all([frameFile(), loadWeights(`${base}/sentence.wgw`)]);
    format = m.format;
    m.downloadMs = performance.now() - tf;
    m.weightBytes = loadedBytes - before;
    let sw: WeightFile;
    if (format === 'f32') sw = sf as WeightFile;
    else if (format === 'i8') { const td = performance.now(); sw = dequantFrameI8(sf as I8File); m.dequantMs = performance.now() - td; }
    else sw = (sf as I8File).w;
    const perUse = format === 'i8q';
    const shaders = { gemm: perUse ? GEMM_I8 : GEMM, concat: CONCAT, conv2d: conv2dShader(perUse), gnStats: GN_STATS, conv1d: CONV1D, head: HEAD };
    for (const [name, code] of Object.entries(shaders)) {
      m.pipelines[name] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    }
    const sBytes = m.bytes;
    for (const [k, v] of Object.entries(sw)) m.weights.set(`s:${k}`, m.upload(v));
    m.frameWeightBytes = m.bytes - sBytes;
    for (const [k, v0] of Object.entries(bw)) {
      const v = v0 as Float32Array;
      if (/\.g$/.test(k)) {
        const beta = bw[k.replace(/\.g$/, '.b')]! as Float32Array;
        const gb = new Float32Array(v.length * 2);
        gb.set(v);
        gb.set(beta, v.length);
        m.weights.set(`b:${k.replace(/\.g$/, '.gb')}`, m.upload(gb));
      } else if (!/norm\d?\.b$|out_norm\.b$/.test(k)) m.weights.set(`b:${k}`, m.upload(v));
    }
    const hss = new Float32Array(384);
    hss.set(bw['head.b']! as Float32Array);
    hss.set(bw['scale']! as Float32Array, 128);
    hss.set(bw['shift']! as Float32Array, 256);
    m.weights.set('b:hss', m.upload(hss));
    m.dummy = m.buffer(16, STORAGE);
    m.buildFrame();
    m.growSentence(256);
    return m;
  }

  private buffer(size: number, usage: number): GpuBuffer {
    this.bytes += size;
    return this.device.createBuffer({ size: Math.ceil(size / 16) * 16, usage });
  }

  private upload(data: Float32Array | Uint8Array): GpuBuffer {
    const b = this.buffer(data.byteLength, STORAGE | COPY_DST);
    this.device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }

  private step(name: string, buffers: GpuBuffer[], params: () => ArrayBuffer, threads: () => number): Step {
    const pipeline = this.pipelines[name];
    const uniform = this.buffer(64, UNIFORM | COPY_DST);
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [uniform, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    return { pipeline, bind, uniform, params, threads };
  }

  private w(name: string): GpuBuffer {
    const b = this.weights.get(name);
    if (!b) throw new Error(`wgsl weight ${name} missing`);
    return b;
  }

  private buildFrame(): void {
    const B = WGSL_MAX_BATCH;
    const big = 8 * BOX_H * BOX_W * B * 4;
    const P = [this.buffer(big, STORAGE), this.buffer(big, STORAGE), this.buffer(big, STORAGE)];
    this.featIn = this.buffer(B * FEAT_SIZE * 4, STORAGE | COPY_DST);
    this.zIn = this.buffer(B * Z_IN * 4, STORAGE | COPY_DST);
    const g0 = this.buffer(B * 256 * 4, STORAGE);
    const g1 = this.buffer(B * 7040 * 4, STORAGE);
    const outBytes = B * 3 * BOX_H * BOX_W * 4;
    this.frameOut = this.buffer(outBytes, STORAGE | COPY_SRC);
    this.frameRead = this.buffer(outBytes, MAP_READ | COPY_DST);
    const b = (): number => this.batch;
    const i8 = this.format === 'i8q';
    const gemm = (x: GpuBuffer, wn: string, y: GpuBuffer, I: number, O: number, act: number, scale: number): Step =>
      this.step('gemm', [x, this.w(`s:${wn}.w`), this.w(`s:${wn}.b`), y, ...(i8 ? [this.w(`s:${wn}.ws`)] : [])], () => uniformData([b(), I, O, act, { f: scale }]), () => b() * O);
    const conv = (x: GpuBuffer, n: string, y: GpuBuffer, C: number, O: number, H: number, W: number, act: number, mode: number, r: GpuBuffer | null): Step =>
      this.step('conv2d', [x, this.w(`s:${n}.w`), this.w(`s:${n}.b`), y, r ?? this.dummy, ...(i8 ? [this.w(`s:${n}.ws`)] : [])],
        () => uniformData([b(), C, Math.ceil(O / 4), H, W, act, mode, r && mode === 0 ? 1 : 0, O]), () => b() * Math.ceil(O / 4) * H * W);
    const [H0, W0] = [FEAT[1], FEAT[2]];
    const s: Step[] = [
      gemm(this.zIn, 'zfc0', g0, Z_IN, 256, 1, 20),
      gemm(g0, 'zfc2', g1, 256, 7040, 0, 1),
      this.step('concat', [this.featIn, g1, P[0]], () => uniformData([b()]), () => b() * 40 * 880),
      conv(P[0], 'inp', P[1], 40, 64, H0, W0, 0, 0, null),
      conv(P[1], 'body.0', P[2], 64, 64, H0, W0, 1, 0, null),
      conv(P[2], 'body.1', P[0], 64, 64, H0, W0, 1, 0, P[1]),
      conv(P[0], 'body.2', P[2], 64, 64, H0, W0, 1, 0, null),
      conv(P[2], 'body.3', P[1], 64, 64, H0, W0, 1, 0, P[0]),
      conv(P[1], 'body.4', P[2], 64, 64, H0, W0, 1, 0, null),
      conv(P[2], 'body.5', P[0], 64, 64, H0, W0, 1, 0, P[1]),
      conv(P[0], 'mid', P[1], 64, 64, H0, W0, 1, 0, null),
      conv(P[1], 'up.0.0', P[2], 64, 64, H0, W0, 0, 0, null),
      conv(P[2], 'up.0.1', P[0], 64, 128, H0, W0, 2, 1, null),
      conv(P[0], 'up.1.0', P[1], 32, 32, H0 * 2, W0 * 2, 0, 0, null),
      conv(P[1], 'up.1.1', P[2], 32, 64, H0 * 2, W0 * 2, 2, 1, null),
      conv(P[2], 'up.2.0', P[0], 16, 16, H0 * 4, W0 * 4, 0, 0, null),
      conv(P[0], 'up.2.1', P[1], 16, 32, H0 * 4, W0 * 4, 2, 1, null),
      conv(P[1], 'out', this.frameOut, 8, 3, BOX_H, BOX_W, 2, 2, this.w('s:win')),
    ];
    this.frameSteps = s;
  }

  /** The sentence model's buffers for sentences of up to capT mel frames (the old ones are destroyed). */
  private growSentence(capT: number): void {
    for (const b of this.sentenceBuffers) { b.destroy(); this.bytes -= b.size; }
    for (const st of this.sentenceSteps) { st.uniform.destroy(); this.bytes -= 64; }
    const keep: GpuBuffer[] = [];
    const mk = (size: number, usage: number): GpuBuffer => { const b = this.buffer(size, usage); keep.push(b); return b; };
    this.capT = capT;
    this.sentenceIn = mk(208 * capT * 4, STORAGE | COPY_DST);
    const X = [mk(256 * capT * 4, STORAGE), mk(256 * capT * 4, STORAGE), mk(256 * capT * 4, STORAGE)];
    const stats = mk(16 * capT * 4, STORAGE);
    this.sentenceOut = mk(128 * capT * 4, STORAGE | COPY_SRC);
    this.sentenceRead = mk(128 * capT * 4, MAP_READ | COPY_DST);
    this.sentenceBuffers = keep;
    const T = (): number => this.T;
    const gnStats = (x: GpuBuffer): Step => this.step('gnStats', [x, stats], () => uniformData([256, T()]), () => 8 * T());
    const conv = (x: GpuBuffer, n: string, y: GpuBuffer, C: number, dil: number, gn: string | null, r: GpuBuffer | null): Step =>
      this.step('conv1d', [x, this.w(`b:${n}.w`), this.w(`b:${n}.b`), stats, gn ? this.w(`b:${gn}.gb`) : this.dummy, y, r ?? this.dummy],
        () => uniformData([C, 64, T(), dil, gn ? 1 : 0, r ? 1 : 0]), () => 64 * T());
    const s: Step[] = [conv(this.sentenceIn, 'in', X[0], 208, 1, null, null)];
    let cur = 0;
    [1, 2, 4, 8, 4, 2, 1].forEach((d, i) => {
      const h = (cur + 1) % 3;
      const nx = (cur + 2) % 3;
      s.push(gnStats(X[cur]), conv(X[cur], `b${i}.conv1`, X[h], 256, d, `b${i}.norm1`, null));
      s.push(gnStats(X[h]), conv(X[h], `b${i}.conv2`, X[nx], 256, d, `b${i}.norm2`, X[cur]));
      cur = nx;
    });
    s.push(gnStats(X[cur]));
    s.push(this.step('head', [X[cur], stats, this.w('b:out_norm.gb'), this.w('b:head.w'), this.w('b:hss'), this.sentenceOut], () => uniformData([256, T()]), () => 128 * T()));
    this.sentenceSteps = s;
  }

  /** Frees every GPU buffer and the device (the models are not usable after). */
  destroy(): void {
    this.device.destroy();
    this.bytes = 0;
  }

  /**
   * One model run at a time: frame() and sentence() write the shared input buffers and map one read buffer each, so two
   * runs in flight (the next chunk prepared while the current one renders, speakChunks) would overwrite each other's
   * inputs or map the read buffer twice. Runs are chained here in call order; each run's result is unchanged.
   */
  private queue: Promise<unknown> = Promise.resolve();

  private run(which: 'frame' | 'sentence', bytes: number, write: () => void): Promise<Float32Array> {
    // the steps and buffers are read after write(): growSentence() (in write) replaces them
    const r = this.queue.then(() => {
      write();
      return which === 'frame' ? this.runNow(this.frameSteps, this.frameOut, this.frameRead, bytes) : this.runNow(this.sentenceSteps, this.sentenceOut, this.sentenceRead, bytes);
    });
    this.queue = r.catch(() => undefined);
    return r;
  }

  private async runNow(steps: Step[], out: GpuBuffer, read: GpuBuffer, bytes: number): Promise<Float32Array> {
    const enc = this.device.createCommandEncoder();
    for (const st of steps) this.device.queue.writeBuffer(st.uniform, 0, st.params());
    const pass = enc.beginComputePass();
    for (const st of steps) {
      const groups = Math.ceil(st.threads() / WG);
      pass.setPipeline(st.pipeline);
      pass.setBindGroup(0, st.bind);
      pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
    }
    pass.end();
    enc.copyBufferToBuffer(out, 0, read, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(1, 0, bytes);
    const r = new Float32Array(read.getMappedRange(0, bytes).slice(0));
    read.unmap();
    return r;
  }

  /** Sentence model: mel [T,80] and ref [T,128] (row per mel frame) -> latent [T,128]. */
  async sentence(mel: Float32Array, ref: Float32Array, T: number): Promise<Float32Array> {
    const x = new Float32Array(208 * T);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < 80; c++) x[c * T + t] = mel[t * 80 + c]!;
      for (let c = 0; c < 128; c++) x[(80 + c) * T + t] = ref[t * 128 + c]!;
    }
    // the buffers are sized and written when this run's turn comes (a run before it may still use them)
    return this.run('sentence', T * 128 * 4, () => {
      if (T > this.capT) this.growSentence(Math.ceil(T / 256) * 256);
      this.T = T;
      this.device.queue.writeBuffer(this.sentenceIn, 0, x.buffer, 0, x.byteLength);
    });
  }

  /** Frame model: feat [n,32,20,44], z [n,640] (float32) -> delta [n,3,160,352]. */
  async frame(feat: Float32Array, z: Float32Array, n: number): Promise<Float32Array> {
    if (n < 1 || n > WGSL_MAX_BATCH) throw new Error(`frame model batch ${n}`);
    return this.run('frame', n * 3 * BOX_H * BOX_W * 4, () => {
      this.batch = n;
      this.device.queue.writeBuffer(this.featIn, 0, feat.buffer, feat.byteOffset, n * FEAT_SIZE * 4);
      this.device.queue.writeBuffer(this.zIn, 0, z.buffer, z.byteOffset, n * Z_IN * 4);
    });
  }
}
