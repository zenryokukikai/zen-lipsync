// What the frame model's output means (the pack's manifest `frame_output`), and the one function that turns an
// 'absolute' output into the painted mouth every compositor draws.
//
// 'delta' (packs without the key, every pack before): the output is the change to add to the source frame's box, as is.
// 'absolute': the model paints the mouth itself. Its output K = 64 * o * win (win: the model's fixed edge window) is the
// painted pixel p = 127.5 + 64 * o (clamped to 0..255) in the model's colour; the manifest's
// `model_colour` ([3][256], video value -> model colour, monotone per channel) takes it back to the video's colour.
// Its weight is the frame's mouth alpha (mouth_alpha.u8, one [boxH, boxW] mask per source frame) times win, times the
// strength of the frame (the fade after speech). Every compositor draws it over the idle video frame as it is:
//     out = frame * (1 - A) + painted * A,   A = patch alpha * weight (both warped into the frame)
// The pack's crops take no part, so a crop that differs from the idle video under it never shows through.
// The painted mouth is taken from the model's box into the frame by warpPainted (bicubic inside the mouth, the kernel of
// OpenCV's INTER_CUBIC, a = -0.75), which keeps the teeth and lip edges sharp.
import { AvatarPackError } from './errors.js';
import type { Geometry } from './pack.js';

export type FrameOutput = 'delta' | 'absolute';

/** Pack file of the 'absolute' packs: uint8 [source frames, boxH, boxW], the mouth alpha in the model's box. */
export const MOUTH_ALPHA_FILE = 'mouth_alpha.u8';
/** The middle grey the painted pixels are centred on (0..255 units). */
const GREY = 127.5;

export function checkFrameOutput(v: unknown, where: string): FrameOutput {
  if (v === undefined || v === 'delta') return 'delta';
  if (v === 'absolute') return 'absolute';
  throw new AvatarPackError(`${where}: "frame_output" must be "delta" or "absolute"`);
}

const WINDOWS = new Map<string, Float32Array>();

/** The frame model's fixed edge window (its `win` tensor): 1 inside, a 12-pixel linear ramp to 0 at each box edge. */
export function boxWindow(bh: number, bw: number): Float32Array {
  const key = `${bh}x${bw}`;
  let w = WINDOWS.get(key);
  if (w) return w;
  const ramp = (n: number): Float32Array => {
    const r = new Float32Array(n).fill(1);
    for (let i = 0; i < 12; i++) { const t = (i + 1) / 13; r[i] = t; r[n - 1 - i] = t; }
    return r;
  };
  const ry = ramp(bh), rx = ramp(bw);
  w = new Float32Array(bh * bw);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) w[y * bw + x] = ry[y]! * rx[x]!;
  WINDOWS.set(key, w);
  return w;
}

/** The painted mouth of one frame of an 'absolute' pack, in the model's box. */
export interface PaintedMouth {
  /** [3, boxH, boxW]: the painted pixel in the video's colour (0..255). */
  rgb: Float32Array;
  /** [boxH, boxW]: how much of it is drawn over the frame (0..1): mouth alpha * edge window * strength. */
  weight: Float32Array;
}

/** Buffers for paintMouth of a box of `plane` pixels. */
export function paintedMouth(plane: number): PaintedMouth {
  return { rgb: new Float32Array(3 * plane), weight: new Float32Array(plane) };
}

/**
 * The index-th frame of an 'absolute' pack's output (pred: float32 [n, 3, bh, bw]) as its painted mouth, written to out.
 * alpha: the frame's mouth alpha ([bh * bw] bytes); box: the pack's model box; toVideo: per channel, model colour ->
 * video colour (inverseColour of model_colour); strength: the frame's strength (1 on speech, the fade after it).
 */
export function paintMouth(pred: Float32Array, index: number, alpha: Uint8Array, box: { y0: number; y1: number; x0: number; x1: number },
  toVideo: readonly Float32Array[], out: PaintedMouth, strength = 1): PaintedMouth {
  const plane = (box.x1 - box.x0) * (box.y1 - box.y0);
  const win = boxWindow(box.y1 - box.y0, box.x1 - box.x0);
  const o = index * 3 * plane, k = strength / 255;
  for (let j = 0; j < plane; j++) {
    const wj = win[j]!, inv = 1 / wj;
    out.weight[j] = alpha[j]! * k * wj;
    if (out.weight[j] === 0) { out.rgb[j] = 0; out.rgb[plane + j] = 0; out.rgb[2 * plane + j] = 0; continue; } // not drawn
    for (let c = 0; c < 3; c++) {
      // the painted pixel in the model colour, clamped to 0..255, then in the video's colour
      const pm = Math.max(0, Math.min(255, GREY + pred[o + c * plane + j]! * inv));
      out.rgb[c * plane + j] = lookup(toVideo[c]!, pm);
    }
  }
  return out;
}

/** OpenCV INTER_CUBIC's kernel (a = -0.75) at the 4 taps around a sample whose fractional offset is t, written to w. */
function cubicTaps(t: number, w: Float32Array): void {
  const A = -0.75;
  const x0 = t + 1, x1 = t, x2 = 1 - t;
  w[0] = ((A * x0 - 5 * A) * x0 + 8 * A) * x0 - 4 * A;
  w[1] = ((A + 2) * x1 - (A + 3)) * x1 * x1 + 1;
  w[2] = ((A + 2) * x2 - (A + 3)) * x2 * x2 + 1;
  w[3] = 1 - w[0]! - w[1]! - w[2]!;
}

/** Work buffers of warpPainted for a box of `plane` pixels and a patch rect of `rectPixels` pixels. */
export interface PaintedWarp {
  /** [boxH * boxW * 4]: per box pixel the painted colour times its weight (r, g, b) and the weight. */
  premul: Float32Array;
  /**
   * [rect pixels * 4]: per patch pixel the painted colour (r, g, b, 0..255) and its weight (0..1); weight 0 = not drawn.
   * Only the rect pixels x0..x1, y0..y1 (inclusive) are written; every pixel outside them is not drawn (x1 < x0: none).
   */
  out: Float32Array;
  x0: number; x1: number; y0: number; y1: number;
  /** Axis-aligned matrices (every pack's): the horizontal cubic pass [boxH * rect width * 3] and the 4x4-uniform flags [box pixels]. */
  rows?: Float32Array;
  flat?: Uint8Array;
}

const WX = new Float32Array(4), WY = new Float32Array(4);

/**
 * The painted mouth (in the model's box of the 512 crop) warped into the frame's patch rect through geo.matrix.
 * The colour is bicubic with OpenCV INTER_CUBIC's kernel (a = -0.75), which keeps the teeth and lip edges,
 * wherever all 4x4 taps are painted (weight > 0), clamped to 0..255 as cv2 does; the weight is always bilinear (never
 * negative, no ghost outside the mouth). At the outer fringe, where some taps are not painted, the colour is bilinear
 * weighted by the painted weight, so nothing outside the painted mouth is pulled in. alpha (the patch alpha, rect bytes) given: pixels where it is 0 are skipped (weight 0).
 * Per patch pixel out gets the colour (0..255) and the weight (0..1) the compositor draws with:
 * out = video * (1 - patch alpha * weight) + colour * patch alpha * weight. work is reused (each buffer grown when small).
 */
export function warpPainted(pm: PaintedMouth, box: { y0: number; y1: number; x0: number; x1: number }, geo: Geometry,
  alpha: Uint8Array | null, work?: PaintedWarp): PaintedWarp {
  const bw = box.x1 - box.x0, bh = box.y1 - box.y0, plane = bw * bh;
  const [, , rw, rh] = geo.rect;
  const premul = work && work.premul.length >= plane * 4 ? work.premul : new Float32Array(plane * 4);
  const out = work && work.out.length >= rw * rh * 4 ? work.out : new Float32Array(rw * rh * 4);
  const w: PaintedWarp = work && premul === work.premul && out === work.out ? work
    : { premul, out, x0: 0, x1: -1, y0: 0, y1: -1, ...(work?.rows ? { rows: work.rows } : {}), ...(work?.flat ? { flat: work.flat } : {}) };
  const rgb = pm.rgb, wt = pm.weight;
  // the painted part of the box (weight > 0): x0..x1, y0..y1 inclusive; nothing painted: nothing to warp
  let bx0 = bw, bx1 = -1, by0 = bh, by1 = -1;
  for (let y = 0; y < bh; y++) {
    const r0 = y * bw;
    let x = 0;
    while (x < bw && wt[r0 + x] === 0) x++;
    if (x === bw) continue;
    let xe = bw - 1;
    while (wt[r0 + xe] === 0) xe--;
    if (x < bx0) bx0 = x; if (xe > bx1) bx1 = xe; if (y < by0) by0 = y; by1 = y;
  }
  const m = geo.matrix;
  const a = m[0][0], b = m[0][1], c = m[0][2], d = m[1][0], e = m[1][1], f = m[1][2];
  w.x0 = 0; w.x1 = -1; w.y0 = 0; w.y1 = -1;
  if (bx1 < 0) return w;
  if (b === 0 && d === 0 && a > 0 && e > 0) { warpAxisAligned(w, pm, bw, bh, box, rw, rh, a, c, e, f, alpha, bx0, bx1, by0, by1); return w; }
  w.x0 = 0; w.x1 = rw - 1; w.y0 = 0; w.y1 = rh - 1;
  out.fill(0, 0, rw * rh * 4);
  for (let j = 0; j < plane; j++) {
    const k = wt[j]!, o = j * 4;
    premul[o] = rgb[j]! * k; premul[o + 1] = rgb[plane + j]! * k; premul[o + 2] = rgb[2 * plane + j]! * k; premul[o + 3] = k;
  }
  const det = a * e - b * d;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  for (let py = 0; py < rh; py++) {
    const vy = py + 0.5 - f;
    for (let px = 0; px < rw; px++) {
      if (alpha && alpha[py * rw + px] === 0) continue;
      const vx = px + 0.5 - c;
      // position in box pixels, relative to pixel centres
      const sx = ia * vx + ib * vy - box.x0 - 0.5;
      const sy = id * vx + ie * vy - box.y0 - 0.5;
      const xf = Math.floor(sx), yf = Math.floor(sy);
      if (xf < -1 || yf < -1 || xf >= bw || yf >= bh) continue; // the 2x2 around the point is outside the box: nothing painted
      const tx = sx - xf, ty = sy - yf;
      let r = 0, g = 0, bl = 0, ws = 0;
      if (xf >= 1 && yf >= 1 && xf + 2 < bw && yf + 2 < bh && painted4x4(wt, (yf - 1) * bw + xf - 1, bw)) {
        // every tap painted: the colour cubic (not premultiplied), the weight bilinear
        cubicTaps(tx, WX); cubicTaps(ty, WY);
        const w0 = WX[0]!, w1 = WX[1]!, w2 = WX[2]!, w3 = WX[3]!;
        for (let k = 0; k < 4; k++) {
          const q = (yf - 1 + k) * bw + xf - 1, wy = WY[k]!;
          r += wy * (w0 * rgb[q]! + w1 * rgb[q + 1]! + w2 * rgb[q + 2]! + w3 * rgb[q + 3]!);
          g += wy * (w0 * rgb[plane + q]! + w1 * rgb[plane + q + 1]! + w2 * rgb[plane + q + 2]! + w3 * rgb[plane + q + 3]!);
          bl += wy * (w0 * rgb[2 * plane + q]! + w1 * rgb[2 * plane + q + 1]! + w2 * rgb[2 * plane + q + 2]! + w3 * rgb[2 * plane + q + 3]!);
        }
        const j = yf * bw + xf;
        ws = (1 - ty) * ((1 - tx) * wt[j]! + tx * wt[j + 1]!) + ty * ((1 - tx) * wt[j + bw]! + tx * wt[j + bw + 1]!);
        const o = (py * rw + px) * 4;
        out[o] = r <= 0 ? 0 : r >= 255 ? 255 : r;
        out[o + 1] = g <= 0 ? 0 : g >= 255 ? 255 : g;
        out[o + 2] = bl <= 0 ? 0 : bl >= 255 ? 255 : bl;
        out[o + 3] = ws >= 1 ? 1 : ws;
        continue;
      } else {
        for (let t = 0; t < 4; t++) {
          const x = xf + (t & 1), y = yf + (t >> 1);
          if (x < 0 || y < 0 || x >= bw || y >= bh) continue;
          const q = (y * bw + x) * 4, k = ((t & 1) ? tx : 1 - tx) * ((t >> 1) ? ty : 1 - ty);
          r += k * premul[q]!; g += k * premul[q + 1]!; bl += k * premul[q + 2]!; ws += k * premul[q + 3]!;
        }
      }
      if (!(ws > 0)) continue;
      const o = (py * rw + px) * 4, inv = 1 / ws;
      const cr = r * inv, cg = g * inv, cb = bl * inv;
      out[o] = cr <= 0 ? 0 : cr >= 255 ? 255 : cr;
      out[o + 1] = cg <= 0 ? 0 : cg >= 255 ? 255 : cg;
      out[o + 2] = cb <= 0 ? 0 : cb >= 255 ? 255 : cb;
      out[o + 3] = ws >= 1 ? 1 : ws;
    }
  }
  return w;
}

/** Scratch of warpAxisAligned, grown as needed (one per module instance: per worker / page). */
let SX = { n: 0, xf: new Int32Array(0), tx: new Float32Array(0), cw: new Float32Array(0) };

/**
 * warpPainted for an axis-aligned matrix (no rotation; every pack so far): the same pixels, computed separably and only
 * over the painted part of the box. Per rect column the taps and the 4 cubic weights are found once; where all 4x4 taps
 * are painted a horizontal cubic pass over the box rows, then a vertical one; at the outer fringe bilinear weighted by
 * the painted weight, straight from the painted planes.
 */
function warpAxisAligned(w: PaintedWarp, pm: PaintedMouth, bw: number, bh: number, box: { y0: number; x0: number }, rw: number, rh: number,
  sxScale: number, c: number, syScale: number, f: number, alpha: Uint8Array | null, bx0: number, bx1: number, by0: number, by1: number): void {
  const out = w.out, plane = bw * bh, wt = pm.weight, R = pm.rgb;
  if (!w.flat || w.flat.length < plane) w.flat = new Uint8Array(plane);
  if (!w.rows || w.rows.length < bh * rw * 3) w.rows = new Float32Array(bh * rw * 3);
  const flat = w.flat, rows = w.rows;
  // flat[j] = 5: the 4x4 box pixels from j (its top-left) are all painted (weight > 0). First pass: per row, how many
  // painted pixels from j rightwards (up to 4); then 4 rows of such runs. Only the painted part is looked at; flags of
  // an earlier frame are cleared first.
  flat.fill(0, 0, plane);
  for (let y = by0; y <= by1; y++) {
    const r0 = y * bw;
    let n = 0;
    for (let x = bx1; x >= bx0; x--) { n = wt[r0 + x]! > 0 ? (n < 4 ? n + 1 : 4) : 0; flat[r0 + x] = n; }
  }
  for (let y = by0; y + 3 <= by1; y++) {
    const r0 = y * bw;
    for (let x = bx0; x + 3 <= bx1; x++) {
      const j = r0 + x;
      if (flat[j] === 4 && flat[j + bw] === 4 && flat[j + 2 * bw] === 4 && flat[j + 3 * bw] === 4) flat[j] = 5;
    }
  }
  const ixs = 1 / sxScale, iys = 1 / syScale;
  // rect columns / rows whose taps can reach the painted part
  const pxA = Math.max(0, Math.floor((bx0 - 2 + box.x0 + 0.5) * sxScale + c - 0.5)), pxB = Math.min(rw - 1, Math.ceil((bx1 + 2 + box.x0 + 0.5) * sxScale + c - 0.5));
  const pyA = Math.max(0, Math.floor((by0 - 2 + box.y0 + 0.5) * syScale + f - 0.5)), pyB = Math.min(rh - 1, Math.ceil((by1 + 2 + box.y0 + 0.5) * syScale + f - 0.5));
  if (pxA > pxB || pyA > pyB) return;
  w.x0 = pxA; w.x1 = pxB; w.y0 = pyA; w.y1 = pyB;
  for (let py = pyA; py <= pyB; py++) out.fill(0, (py * rw + pxA) * 4, (py * rw + pxB + 1) * 4);
  if (SX.n < rw) SX = { n: rw, xf: new Int32Array(rw), tx: new Float32Array(rw), cw: new Float32Array(rw * 4) };
  const XF = SX.xf, TX = SX.tx, CW = SX.cw;
  for (let px = pxA; px <= pxB; px++) {
    const sx = (px + 0.5 - c) * ixs - box.x0 - 0.5, xf = Math.floor(sx);
    XF[px] = xf; TX[px] = sx - xf;
    cubicTaps(sx - xf, WX); CW[px * 4] = WX[0]!; CW[px * 4 + 1] = WX[1]!; CW[px * 4 + 2] = WX[2]!; CW[px * 4 + 3] = WX[3]!;
  }
  // horizontal cubic pass over the painted rows: rows[(y * rw + px) * 3 ..] = the colour (where all 16 taps are painted
  // the colour is interpolated as it is, not premultiplied; the weight is bilinear)
  const cxA = Math.max(pxA, 0), cxB = pxB;
  for (let y = Math.max(0, by0); y <= by1; y++) {
    const rb = y * bw;
    let o = (y * rw + cxA) * 3;
    for (let px = cxA; px <= cxB; px++, o += 3) {
      const xf = XF[px]!;
      if (xf < 1 || xf + 2 >= bw) continue;
      const q = rb + xf - 1, k = px * 4;
      const w0 = CW[k]!, w1 = CW[k + 1]!, w2 = CW[k + 2]!, w3 = CW[k + 3]!;
      rows[o] = w0 * R[q]! + w1 * R[q + 1]! + w2 * R[q + 2]! + w3 * R[q + 3]!;
      rows[o + 1] = w0 * R[plane + q]! + w1 * R[plane + q + 1]! + w2 * R[plane + q + 2]! + w3 * R[plane + q + 3]!;
      rows[o + 2] = w0 * R[2 * plane + q]! + w1 * R[2 * plane + q + 1]! + w2 * R[2 * plane + q + 2]! + w3 * R[2 * plane + q + 3]!;
    }
  }
  const st = rw * 3;
  for (let py = pyA; py <= pyB; py++) {
    const sy = (py + 0.5 - f) * iys - box.y0 - 0.5, yf = Math.floor(sy), ty = sy - yf;
    if (yf < -1 || yf >= bh) continue;
    const cubicRow = yf >= 1 && yf + 2 < bh;
    cubicTaps(ty, WY);
    const y0 = WY[0]!, y1 = WY[1]!, y2 = WY[2]!, y3 = WY[3]!;
    const fr = (yf - 1) * bw, rc = (yf - 1) * st;
    const yaIn = yf >= 0, ybIn = yf + 1 < bh, ja = yf * bw, jb = (yf + 1) * bw;
    for (let px = pxA; px <= pxB; px++) {
      if (alpha && alpha[py * rw + px] === 0) continue;
      const xf = XF[px]!;
      let r: number, g: number, bl: number, ws: number;
      if (cubicRow && xf >= 1 && xf + 2 < bw && flat[fr + xf - 1] === 5) {
        const q = rc + px * 3;
        r = y0 * rows[q]! + y1 * rows[q + st]! + y2 * rows[q + 2 * st]! + y3 * rows[q + 3 * st]!;
        g = y0 * rows[q + 1]! + y1 * rows[q + st + 1]! + y2 * rows[q + 2 * st + 1]! + y3 * rows[q + 3 * st + 1]!;
        bl = y0 * rows[q + 2]! + y1 * rows[q + st + 2]! + y2 * rows[q + 2 * st + 2]! + y3 * rows[q + 3 * st + 2]!;
        const tx = TX[px]!, j = ja + xf;
        ws = (1 - ty) * ((1 - tx) * wt[j]! + tx * wt[j + 1]!) + ty * ((1 - tx) * wt[j + bw]! + tx * wt[j + bw + 1]!);
        const o = (py * rw + px) * 4;
        out[o] = r <= 0 ? 0 : r >= 255 ? 255 : r;
        out[o + 1] = g <= 0 ? 0 : g >= 255 ? 255 : g;
        out[o + 2] = bl <= 0 ? 0 : bl >= 255 ? 255 : bl;
        out[o + 3] = ws >= 1 ? 1 : ws;
        continue;
      }
      // bilinear, weighted by the painted weight (taps outside the box count 0)
      const tx = TX[px]!, xaIn = xf >= 0 && xf < bw, xbIn = xf + 1 >= 0 && xf + 1 < bw;
      r = 0; g = 0; bl = 0; ws = 0;
      if (yaIn) {
        if (xaIn) { const j = ja + xf, k = (1 - tx) * (1 - ty) * wt[j]!; r += k * R[j]!; g += k * R[plane + j]!; bl += k * R[2 * plane + j]!; ws += k; }
        if (xbIn) { const j = ja + xf + 1, k = tx * (1 - ty) * wt[j]!; r += k * R[j]!; g += k * R[plane + j]!; bl += k * R[2 * plane + j]!; ws += k; }
      }
      if (ybIn) {
        if (xaIn) { const j = jb + xf, k = (1 - tx) * ty * wt[j]!; r += k * R[j]!; g += k * R[plane + j]!; bl += k * R[2 * plane + j]!; ws += k; }
        if (xbIn) { const j = jb + xf + 1, k = tx * ty * wt[j]!; r += k * R[j]!; g += k * R[plane + j]!; bl += k * R[2 * plane + j]!; ws += k; }
      }
      if (!(ws > 0)) continue;
      const o = (py * rw + px) * 4, inv = 1 / ws;
      const cr = r * inv, cg = g * inv, cb = bl * inv;
      out[o] = cr <= 0 ? 0 : cr >= 255 ? 255 : cr;
      out[o + 1] = cg <= 0 ? 0 : cg >= 255 ? 255 : cg;
      out[o + 2] = cb <= 0 ? 0 : cb >= 255 ? 255 : cb;
      out[o + 3] = ws >= 1 ? 1 : ws;
    }
  }
}

/** True when the 4x4 box pixels from index j0 (row stride bw) are all painted (weight > 0). */
function painted4x4(wt: Float32Array, j0: number, bw: number): boolean {
  for (let y = 0; y < 4; y++) {
    const r = j0 + y * bw;
    if (!(wt[r]! > 0 && wt[r + 1]! > 0 && wt[r + 2]! > 0 && wt[r + 3]! > 0)) return false;
  }
  return true;
}

/** t[x] at a fractional x in 0..255 (linear between entries). */
function lookup(t: Float32Array, x: number): number {
  const i = Math.min(254, Math.floor(x)), f = x - i;
  return t[i]! + f * (t[i + 1]! - t[i]!);
}

/**
 * The inverse of one model_colour table (video value -> model colour, made monotone): for each model value 0..255 the
 * video value that maps to it (linear between the table's entries; 0 and 255 beyond its ends).
 */
export function inverseColour(table: ArrayLike<number>): Float32Array {
  if (table.length !== 256) throw new AvatarPackError(`model_colour: a table has ${table.length} entries, expected 256`);
  const f = new Float32Array(256);
  let m = -Infinity;
  for (let v = 0; v < 256; v++) { m = Math.max(m, Number(table[v])); f[v] = m; } // monotone
  const out = new Float32Array(256);
  let v = 0;
  for (let y = 0; y < 256; y++) {
    if (y <= f[0]!) { out[y] = 0; continue; }
    if (y >= f[255]!) { out[y] = 255; continue; }
    while (v < 254 && f[v + 1]! < y) v++;
    const d = f[v + 1]! - f[v]!;
    out[y] = d > 0 ? v + (y - f[v]!) / d : v;
  }
  return out;
}

/** model_colour of an 'absolute' pack: 3 tables of 256 finite numbers (idle-video value -> model value, per channel). */
export function checkModelColour(colour: unknown, where: string): number[][] {
  if (!Array.isArray(colour) || colour.length !== 3 || !colour.every((t) => Array.isArray(t) && t.length === 256 && t.every((v) => typeof v === 'number' && Number.isFinite(v))))
    throw new AvatarPackError(`${where}: an "absolute" pack needs "model_colour" (3 tables of 256 numbers)`);
  return colour as number[][];
}

/** What an 'absolute' pack adds: the mouth alphas of its source frames ([bh * bw] bytes each) and, from model_colour, its inverse. */
export class MouthAlpha {
  /** Per channel, model colour -> video colour (the inverse of model_colour), what paintMouth takes. */
  readonly colour: Float32Array[];
  constructor(private readonly data: Uint8Array, readonly plane: number, frames: number, colour: unknown, where: string) {
    if (data.length !== frames * plane) throw new AvatarPackError(`${where}/${MOUTH_ALPHA_FILE}: ${data.length} bytes, expected ${frames} x ${plane}`);
    this.colour = checkModelColour(colour, where).map((t) => inverseColour(t));
  }

  of(u: number): Uint8Array { return this.data.subarray(u * this.plane, (u + 1) * this.plane); }

}
