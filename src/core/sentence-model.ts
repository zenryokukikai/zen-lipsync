// Sentence model on the CPU (float32, TypeScript; it runs once per sentence), shared by the Node entry and tests.
// Same graph and outputs as the WebGPU sentence model.

export type SentenceWeights = Record<string, Float32Array>;

/** sentence.wgw (uint32 header length, JSON {"tensors": {name: {offset (floats), shape}}}, float32 data). */
export function parseSentenceWeights(buf: ArrayBuffer): SentenceWeights {
  const hl = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))) as { tensors: Record<string, { offset: number; shape: number[] }> };
  const data = new Float32Array(buf, 4 + hl);
  const out: SentenceWeights = {};
  for (const [name, t] of Object.entries(head.tensors)) out[name] = data.subarray(t.offset, t.offset + t.shape.reduce((a, b) => a * b, 1));
  return out;
}

/**
 * Sentence model on the CPU: mel [T,80] + ref [T,128] -> latent [T,128]. The same graph as the WGSL kernels:
 * an input conv (208 -> 256), 7 residual blocks of two dilated convs (kernel 3, GroupNorm 8 + SiLU before each,
 * dilations 1,2,4,8,4,2,1), GroupNorm + SiLU, a 256 -> 128 head with scale and shift.
 * Weight layout as uploaded by wgsl.ts: conv w [O/4][C][3][4] (4 output channels packed), b [O].
 */
export class CpuSentenceModel {
  constructor(private readonly w: SentenceWeights) {}

  run(mel: Float32Array, ref: Float32Array, T: number): Float32Array {
    const x = new Float32Array(208 * T);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < 80; c++) x[c * T + t] = mel[t * 80 + c]!;
      for (let c = 0; c < 128; c++) x[(80 + c) * T + t] = ref[t * 128 + c]!;
    }
    let cur = this.conv(x, 'in', 208, T, 1, null, null);
    [1, 2, 4, 8, 4, 2, 1].forEach((d, i) => {
      const h = this.conv(cur, `b${i}.conv1`, 256, T, d, `b${i}.norm1`, null);
      cur = this.conv(h, `b${i}.conv2`, 256, T, d, `b${i}.norm2`, cur);
    });
    const a = this.gnSilu(cur, 256, T, 'out_norm');
    const hw = this.w['head.w']!, hb = this.w['head.b']!, sc = this.w['scale']!, sh = this.w['shift']!;
    const z = new Float32Array(T * 128);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < 128; k++) {
        let acc = 0;
        for (let c = 0; c < 256; c++) acc += a[c * T + t]! * hw[c * 128 + k]!;
        z[t * 128 + k] = (acc + hb[k]!) * sc[k]! + sh[k]!;
      }
    }
    return z;
  }

  /** GroupNorm (8 groups over channels at each time step, eps 1e-5) then SiLU, on x [C][T]. */
  private gnSilu(x: Float32Array, C: number, T: number, norm: string): Float32Array {
    const g = this.w[`${norm}.g`]!, b = this.w[`${norm}.b`]!;
    const y = new Float32Array(C * T);
    const cg = C / 8;
    for (let grp = 0; grp < 8; grp++) {
      for (let t = 0; t < T; t++) {
        let m = 0;
        for (let c = 0; c < cg; c++) m += x[(grp * cg + c) * T + t]!;
        m /= cg;
        let v = 0;
        for (let c = 0; c < cg; c++) { const dd = x[(grp * cg + c) * T + t]! - m; v += dd * dd; }
        const inv = 1 / Math.sqrt(v / cg + 1e-5);
        for (let c = 0; c < cg; c++) {
          const ch = grp * cg + c;
          const u = (x[ch * T + t]! - m) * inv * g[ch]! + b[ch]!;
          y[ch * T + t] = u / (1 + Math.exp(-u));
        }
      }
    }
    return y;
  }

  /** Weights of a conv as [O][C][3] (from the packed [O/4][C][3][4] layout), made once. */
  private readonly unpacked = new Map<string, Float32Array>();
  private convWeights(name: string, C: number): Float32Array {
    let u = this.unpacked.get(name);
    if (!u) {
      const w = this.w[`${name}.w`]!;
      u = new Float32Array(256 * C * 3);
      for (let o = 0; o < 256; o++) for (let c = 0; c < C; c++) for (let k = 0; k < 3; k++) u[(o * C + c) * 3 + k] = w[(((o >> 2) * C + c) * 3 + k) * 4 + (o & 3)]!;
      this.unpacked.set(name, u);
    }
    return u;
  }

  /** Dilated conv1d (kernel 3, same padding) on x [C][T] -> [256][T]; input GroupNorm+SiLU'd when norm is given; + r. */
  private conv(x: Float32Array, name: string, C: number, T: number, dil: number, norm: string | null, r: Float32Array | null): Float32Array {
    const inp = norm ? this.gnSilu(x, C, T, norm) : x;
    const w = this.convWeights(name, C), bias = this.w[`${name}.b`]!;
    const y = new Float32Array(256 * T);
    // fixed summation order (per output: bias, then c = 0.., k = 0..2, then + r), done for 4 outputs
    // at once so each input row is read once per 4 outputs (about 2.5x faster in V8)
    for (let o = 0; o < 256; o += 4) {
      const r0 = y.subarray(o * T, (o + 1) * T), r1 = y.subarray((o + 1) * T, (o + 2) * T);
      const r2 = y.subarray((o + 2) * T, (o + 3) * T), r3 = y.subarray((o + 3) * T, (o + 4) * T);
      r0.fill(bias[o]!); r1.fill(bias[o + 1]!); r2.fill(bias[o + 2]!); r3.fill(bias[o + 3]!);
      const w0 = o * C * 3, w1 = w0 + C * 3, w2 = w1 + C * 3, w3 = w2 + C * 3;
      for (let c = 0; c < C; c++) {
        const xb = c * T;
        for (let k = 0; k < 3; k++) {
          const ck = c * 3 + k;
          const a0 = w[w0 + ck]!, a1 = w[w1 + ck]!, a2 = w[w2 + ck]!, a3 = w[w3 + ck]!;
          const sh = (k - 1) * dil;
          const t0 = Math.max(0, -sh), t1 = Math.min(T, T - sh);
          for (let t = t0; t < t1; t++) {
            const v = inp[xb + t + sh]!;
            r0[t]! += a0 * v; r1[t]! += a1 * v; r2[t]! += a2 * v; r3[t]! += a3 * v;
          }
        }
      }
      if (r) for (let j = 0; j < 4; j++) { const row = y.subarray((o + j) * T, (o + j + 1) * T); const rb = (o + j) * T; for (let t = 0; t < T; t++) row[t]! += r[rb + t]!; }
    }
    return y;
  }
}
