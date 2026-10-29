// Clean-up of the frame model's change (delta) before it is added to the mouth crop, shared by the browser and Node
// compositors so both draw the same pixels.
//
// 1. Skin tint: on some avatars the model also tints the skin around the mouth while speaking (e.g. green lowered more
//    than red and blue: a magenta cast that grows over an utterance). On skin-like pixels (small deltas: not lips or
//    teeth) the delta keeps its brightness (the mean of the three channels) and loses its chroma, so a neutral darkening
//    such as the shadow of an open mouth stays while the colour cast goes.
// 2. Box edges: the model's box has hard edges; whatever delta remains on an edge row shows as a straight line where the
//    box ends inside the patch (the top edge, between nose and upper lip). The delta on each edge, smoothed along the
//    edge, is faded out over a few pixels into the box, so the delta reaches 0 at the edge.

/** Deltas below this (max |delta| over the channels) count fully as skin; at twice this they are not skin at all. */
const SKIN_DELTA = 10;
/** Rows (top, bottom) and columns (left, right) over which an edge's delta is faded out to 0. */
const EDGE_FEATHER = 16;
/** Half-width of the smoothing along an edge (pixels), so the fade follows the edge's low-frequency value only. */
const EDGE_SMOOTH = 7;

export class DeltaCleaner {
  private work = new Float32Array(0);
  private edge = new Float32Array(0);
  private sum = new Float32Array(0);

  /** Box width and height of the model output. */
  constructor(private readonly bw: number, private readonly bh: number) {}

  /**
   * The cleaned delta of the index-th frame of pred (float32 [n,3,bh,bw]); the returned array is reused by the next call.
   * win: the part of the box that is drawn (box coordinates [x0, y0, x1, y1], the box cut by the crop); its edges are
   * the ones faded out.
   */
  clean(pred: Float32Array, index: number, win: readonly [number, number, number, number]): Float32Array {
    const { bw, bh } = this;
    const plane = bw * bh;
    if (this.work.length !== 3 * plane) this.work = new Float32Array(3 * plane);
    const d = this.work;
    d.set(pred.subarray(index * 3 * plane, (index + 1) * 3 * plane));
    this.removeSkinTint(d, plane);
    for (let c = 0; c < 3; c++) this.featherEdges(d, c * plane, win);
    return d;
  }

  /**
   * Each skin pixel's delta keeps only its brightness: the chroma part (channel minus the mean of the three) is taken
   * out, fully below SKIN_DELTA, fading to none at 2 x SKIN_DELTA (lips and teeth keep their colour).
   */
  private removeSkinTint(d: Float32Array, plane: number): void {
    const lo = SKIN_DELTA, hi = 2 * SKIN_DELTA;
    for (let i = 0; i < plane; i++) {
      const r = d[i]!, g = d[plane + i]!, b = d[2 * plane + i]!;
      let m = r < 0 ? -r : r;
      const ga = g < 0 ? -g : g, ba = b < 0 ? -b : b;
      if (ga > m) m = ga;
      if (ba > m) m = ba;
      if (m >= hi) continue;
      const k = m <= lo ? 1 : (hi - m) / lo;
      const l = (r + g + b) / 3;
      d[i] = r - (r - l) * k; d[plane + i] = g - (g - l) * k; d[2 * plane + i] = b - (b - l) * k;
    }
  }

  /** Fades the smoothed value of each edge of the window, one channel (at offset o), to 0 over EDGE_FEATHER pixels. */
  private featherEdges(d: Float32Array, o: number, win: readonly [number, number, number, number]): void {
    const bw = this.bw;
    const [x0, y0, x1, y1] = win;
    const w = x1 - x0, h = y1 - y0;
    if (w < 4 || h < 4) return;
    const n = EDGE_FEATHER;
    // top and bottom: value per column = mean of the two edge rows, smoothed along the row
    for (const top of [true, false]) {
      const r0 = top ? y0 : y1 - 2;
      const e = this.smoothed(w, (i) => (d[o + r0 * bw + x0 + i]! + d[o + (r0 + 1) * bw + x0 + i]!) / 2);
      for (let k = 0; k < n && k < h; k++) {
        const f = 1 - k / n, row = o + (top ? y0 + k : y1 - 1 - k) * bw + x0;
        for (let i = 0; i < w; i++) d[row + i] = d[row + i]! - e[i]! * f;
      }
    }
    for (const left of [true, false]) {
      const c0 = left ? x0 : x1 - 2;
      const e = this.smoothed(h, (i) => (d[o + (y0 + i) * bw + c0]! + d[o + (y0 + i) * bw + c0 + 1]!) / 2);
      for (let i = 0; i < h; i++) {
        const ei = e[i]!, row = o + (y0 + i) * bw;
        for (let k = 0; k < n && k < w; k++) { const x = left ? x0 + k : x1 - 1 - k; d[row + x] = d[row + x]! - ei * (1 - k / n); }
      }
    }
  }

  /** v(0..len-1) smoothed by a box of half-width EDGE_SMOOTH (ends repeated). */
  private smoothed(len: number, v: (i: number) => number): Float32Array {
    if (this.edge.length < len) { this.edge = new Float32Array(len); this.sum = new Float32Array(len + 1); }
    const s = this.sum, e = this.edge, r = EDGE_SMOOTH;
    s[0] = 0;
    for (let i = 0; i < len; i++) s[i + 1] = s[i]! + v(i);
    for (let i = 0; i < len; i++) {
      const a = i - r, b = i + r;
      // repeated ends: the part beyond either end counts as the end value
      let t = s[Math.min(len, b + 1)]! - s[Math.max(0, a)]!;
      if (a < 0) t += -a * (s[1]! - s[0]!);
      if (b >= len) t += (b - len + 1) * (s[len]! - s[len - 1]!);
      e[i] = t / (2 * r + 1);
    }
    return e;
  }
}
