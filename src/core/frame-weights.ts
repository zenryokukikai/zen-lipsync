// The frame model's weight files (frame.i8.wgw, frame.wgw), shared by the WebGPU runtime and the Node float32 kernel.

export interface WeightFile { [name: string]: Float32Array | Uint8Array }

/** int8 .wgw: uint32 header length, JSON {"tensors": {name: {offset (bytes), shape, dtype "f32" | "i8"}}}, data. */
export interface I8File { w: WeightFile; shapes: Record<string, number[]> }
export function parseI8Weights(buf: ArrayBuffer): I8File {
  const hl = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))) as { tensors: Record<string, { offset: number; shape: number[]; dtype: 'f32' | 'i8' }> };
  const at0 = 4 + hl;
  const out: WeightFile = {};
  const shapes: Record<string, number[]> = {};
  for (const [name, t] of Object.entries(head.tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    out[name] = t.dtype === 'i8' ? new Uint8Array(buf, at0 + t.offset, n) : new Float32Array(buf, at0 + t.offset, n);
    shapes[name] = t.shape;
  }
  return { w: out, shapes };
}

/** float32 .wgw: uint32 header length, JSON {"tensors": {name: {offset (floats), shape}}}, data. */
export function parseF32Weights(buf: ArrayBuffer): WeightFile {
  const hl = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))) as { tensors: Record<string, { offset: number; shape: number[] }> };
  const data = new Float32Array(buf, 4 + hl);
  const out: WeightFile = {};
  for (const [name, t] of Object.entries(head.tensors)) out[name] = data.subarray(t.offset, t.offset + t.shape.reduce((a, b) => a * b, 1));
  return out;
}

/**
 * frame.i8.wgw -> the float32 tensors of frame.wgw (same names and layout, no .ws): w = q * scale of its
 * output channel, rounded once to float32 (the product of an int8 and a float32 is exact in a double).
 *   gemm "<n>.w" [O, I/4, 4] -> [O, I] (row o, scale ws[o]);
 *   conv "<n>.w" [O4, C, 9, 4] -> the same packed [O4, C, 9, 4] (byte j = output channel o4*4+j, scale ws[o4*4+j]).
 */
export function dequantFrameI8(f: I8File): WeightFile {
  const out: WeightFile = {};
  for (const [k, v] of Object.entries(f.w)) {
    if (k.endsWith('.ws')) continue;
    if (!(v instanceof Uint8Array)) { out[k] = v; continue; }
    const s = f.w[`${k.slice(0, -2)}.ws`] as Float32Array;
    const shape = f.shapes[k]!;
    const q = new Int8Array(v.buffer, v.byteOffset, v.length);
    const y = new Float32Array(q.length);
    if (shape.length === 3) {
      const per = q.length / shape[0]!;
      let i = 0;
      for (let o = 0; o < shape[0]!; o++) { const so = s[o]!; for (const e = i + per; i < e; i++) y[i] = q[i]! * so; }
    } else {
      const O4 = shape[0]!, per = q.length / O4;
      let i = 0;
      for (let o4 = 0; o4 < O4; o4++) {
        const s0 = s[o4 * 4]!, s1 = s[o4 * 4 + 1]!, s2 = s[o4 * 4 + 2]!, s3 = s[o4 * 4 + 3]!;
        for (const e = i + per; i < e; i += 4) { y[i] = q[i]! * s0; y[i + 1] = q[i + 1]! * s1; y[i + 2] = q[i + 2]! * s2; y[i + 3] = q[i + 3]! * s3; }
      }
    }
    out[k] = y;
  }
  return out;
}
