// Draws a mouth patch on an idle video frame (rgb24) without a canvas, at (abs_x, abs_y), cut by the patch alpha, through
// the pack's 2x3 matrix (bilinear; 'absolute' packs bicubic, core/frame-output.ts warpPainted).
// 'delta' packs: the model's change is sampled through the same warp as the source crop and that *difference* is added
// to the video pixel, so the video pixel stays untouched where the change is zero.
// 'absolute' packs (core/frame-output.ts): the painted mouth itself is drawn over the video pixel,
// out = video * (1 - A) + painted * A; the crop takes no part, so a crop that differs from the video never shows.
import type { Geometry, PackMeta } from '../core/pack.js';
import { DeltaCleaner } from '../core/delta-clean.js';
import { paintMouth, paintedMouth, warpPainted, type PaintedMouth, type PaintedWarp } from '../core/frame-output.js';

export interface SourceImages {
  geo: Geometry;
  /** RGB of the stored part of the 512 crop (crop_box), 3 bytes per pixel. */
  crop: Uint8Array;
  cropW: number;
  cropH: number;
  /** Patch alpha, rect width x height, 1 byte per pixel. */
  alpha: Uint8Array;
  /** 'absolute' packs: the frame's mouth alpha in the model's box (mouth_alpha.u8), [boxH * boxW] bytes. */
  mouth?: Uint8Array;
}

export class NodeCompositor {
  private work = new Uint8Array(0);
  private readonly cleaner: DeltaCleaner;
  private readonly painted: PaintedMouth | null;
  private warped: PaintedWarp | undefined;

  /**
   * paintColour: 'absolute' packs, per channel model colour -> video colour (MouthAlpha.colour); null for 'delta' packs.
   */
  constructor(private readonly meta: PackMeta, private readonly fromVideo: Float32Array[], private readonly toVideo: Float32Array[],
    private readonly paintColour: readonly Float32Array[] | null = null) {
    this.cleaner = new DeltaCleaner(meta.box.x1 - meta.box.x0, meta.box.y1 - meta.box.y0);
    this.painted = paintColour ? paintedMouth((meta.box.x1 - meta.box.x0) * (meta.box.y1 - meta.box.y0)) : null;
  }

  /**
   * Draws the index-th frame of pred (float32 [n,3,boxH,boxW]) for source frame src onto frame (rgb24, W x H) in place.
   * strength: 'absolute' packs, the frame's strength (the fade after speech); a 'delta' pack's caller scales pred itself.
   */
  draw(frame: Uint8Array, W: number, H: number, src: SourceImages, pred: Float32Array, index: number, strength = 1): void {
    if (this.painted) this.sampleAbsolute(src, this.paint(src, pred, index, strength), frame, W, H);
    else this.sample(src, this.withChange(src, pred, index), frame, W, H);
  }

  private paint(src: SourceImages, pred: Float32Array, index: number, strength: number): PaintedMouth {
    if (!src.mouth) throw new Error('an absolute pack\'s source frame has no mouth alpha');
    return paintMouth(pred, index, src.mouth, this.meta.box, this.paintColour!, this.painted!, strength);
  }

  /**
   * 'absolute' packs: the painted mouth warped into the patch rect (core/frame-output.ts warpPainted: bicubic, weighted by
   * the painted weight, nothing outside the box), and out = frame * (1 - A) + painted * A, A = patch alpha * weight.
   */
  private sampleAbsolute(src: SourceImages, pm: PaintedMouth, frame: Uint8Array, W: number, H: number): void {
    const { geo } = src;
    const [, , rw, rh] = geo.rect;
    const w = this.warped = warpPainted(pm, this.meta.box, geo, src.alpha, this.warped);
    const pw = w.out;
    // only the part the warp wrote can be drawn; values are 0..255, so (q + 0.5) | 0 rounds as Math.round
    for (let py = w.y0; py <= w.y1; py++) {
      const oy = geo.abs_y + py;
      if (oy < 0 || oy >= H) continue;
      for (let px = w.x0; px <= w.x1; px++) {
        const i = (py * rw + px) * 4, ws = pw[i + 3]!;
        if (ws === 0) continue;
        const ox = geo.abs_x + px;
        if (ox < 0 || ox >= W) continue;
        const A = src.alpha[py * rw + px]! * (1 / 255) * ws;
        const o = (oy * W + ox) * 3, B = 1 - A;
        frame[o] = (frame[o]! * B + pw[i]! * A + 0.5) | 0;
        frame[o + 1] = (frame[o + 1]! * B + pw[i + 1]! * A + 0.5) | 0;
        frame[o + 2] = (frame[o + 2]! * B + pw[i + 2]! * A + 0.5) | 0;
      }
    }
  }

  /** 'delta' packs: the changed crop (img) and the plain crop through the warp; their difference, weighted by the alpha, is added to frame. */
  private sample(src: SourceImages, img: Uint8Array, frame: Uint8Array, W: number, H: number): void {
    const crop = src.crop;
    const { geo } = src;
    const [, , rw, rh] = geo.rect;
    const m = geo.matrix;
    // patch pixel centre (px+.5, py+.5) -> crop coordinates (inverse of the matrix); the crop image starts at crop_box x0,y0
    const a = m[0][0], b = m[0][1], c = m[0][2], d = m[1][0], e = m[1][1], f = m[1][2];
    const det = a * e - b * d;
    const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
    const [x0, y0] = geo.crop_box;
    const cw = src.cropW, ch = src.cropH;
    for (let py = 0; py < rh; py++) {
      const oy = geo.abs_y + py;
      if (oy < 0 || oy >= H) continue;
      const vy = py + 0.5 - f;
      for (let px = 0; px < rw; px++) {
        const al = src.alpha[py * rw + px]!;
        if (al === 0) continue;
        const ox = geo.abs_x + px;
        if (ox < 0 || ox >= W) continue;
        const vx = px + 0.5 - c;
        // source position in crop image pixels, relative to pixel centres
        const sx = ia * vx + ib * vy - x0 - 0.5;
        const sy = id * vx + ie * vy - y0 - 0.5;
        const xf = Math.floor(sx), yf = Math.floor(sy);
        const tx = sx - xf, ty = sy - yf;
        const xa = xf < 0 ? 0 : xf >= cw ? cw - 1 : xf, xb = xf + 1 < 0 ? 0 : xf + 1 >= cw ? cw - 1 : xf + 1;
        const ya = yf < 0 ? 0 : yf >= ch ? ch - 1 : yf, yb = yf + 1 < 0 ? 0 : yf + 1 >= ch ? ch - 1 : yf + 1;
        const p00 = (ya * cw + xa) * 3, p01 = (ya * cw + xb) * 3, p10 = (yb * cw + xa) * 3, p11 = (yb * cw + xb) * 3;
        const al1 = al / 255;
        const o = (oy * W + ox) * 3;
        for (let k = 0; k < 3; k++) {
          const top = img[p00 + k]! + (img[p01 + k]! - img[p00 + k]!) * tx;
          const bot = img[p10 + k]! + (img[p11 + k]! - img[p10 + k]!) * tx;
          const changed = top + (bot - top) * ty;
          const t0 = crop[p00 + k]! + (crop[p01 + k]! - crop[p00 + k]!) * tx;
          const b0 = crop[p10 + k]! + (crop[p11 + k]! - crop[p10 + k]!) * tx;
          const plain = t0 + (b0 - t0) * ty;
          const q = frame[o + k]! + (changed - plain) * al1;
          frame[o + k] = q <= 0 ? 0 : q >= 255 ? 255 : Math.round(q);
        }
      }
    }
  }

  /** The crop with the box's change, cleaned (DeltaCleaner: edges faded out), added in the model's colour (video value -> model colour, + change, -> video). */
  private withChange(src: SourceImages, pred: Float32Array, index: number): Uint8Array {
    const n = src.crop.length;
    if (this.work.length < n) this.work = new Uint8Array(n);
    const img = this.work.subarray(0, n);
    img.set(src.crop);
    const box = this.meta.box;
    const [x0, y0, x1, y1] = src.geo.crop_box;
    const cw = x1 - x0;
    const bw = box.x1 - box.x0;
    const plane = (box.y1 - box.y0) * bw;
    const ya = Math.max(y0, box.y0), yb = Math.min(y1, box.y1), xa = Math.max(x0, box.x0), xb = Math.min(x1, box.x1);
    const delta0 = this.cleaner.clean(pred, index, [xa - box.x0, ya - box.y0, xb - box.x0, yb - box.y0]);
    for (let y = ya; y < yb; y++) {
      const brow = (y - box.y0) * bw - box.x0;
      const crow = (y - y0) * cw - x0;
      for (let x = xa; x < xb; x++) {
        const o = (crow + x) * 3;
        for (let k = 0; k < 3; k++) {
          const delta = delta0[k * plane + brow + x]!;
          if (delta > -0.5 && delta < 0.5) continue;
          const v = this.fromVideo[k]![img[o + k]!]! + delta;
          img[o + k] = this.toVideo[k]![v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v)]!;
        }
      }
    }
    return img;
  }
}
