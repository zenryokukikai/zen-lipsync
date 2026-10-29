// One frames.pack (core/frames-pack.ts) in the browser: one download, read as it arrives, so the first frames are usable
// long before the whole file is in. The finished file is checked against the manifest's SHA-256 and kept in Cache Storage
// with the model files (a later page load reads it from there at once).
import { framesPackFrameEnd, framesPackHeadLength, readFramesPackIndex, type FramesPackIndex } from '../core/frames-pack.js';
import { MODEL_CACHE } from './model-cache.js';

async function sha256Hex(data: Uint8Array): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', data as Uint8Array<ArrayBuffer>))].map((v) => v.toString(16).padStart(2, '0')).join('');
}

export class FramesPackStream {
  /** The bytes that have arrived (data[0..received)). */
  private data = new Uint8Array(0);
  private received = 0;
  private index: FramesPackIndex | null = null;
  private failed: Error | null = null;
  private waiters: { need: () => number; resolve: () => void; reject: (e: Error) => void }[] = [];
  /** Resolves when the whole file is in and verified. */
  readonly done: Promise<void>;

  constructor(private readonly url: string, private readonly sourceFrames: number, private readonly type: string, private readonly sha256: string | undefined, private readonly onBytes: (n: number) => void) {
    this.done = this.load().catch((e: unknown) => {
      this.failed = e instanceof Error ? e : new Error(String(e));
      for (const w of this.waiters.splice(0)) w.reject(this.failed);
      throw this.failed;
    });
    this.done.catch(() => undefined);
  }

  /** The crop and mask of source frame u as Blobs, as soon as their bytes have arrived. */
  async frame(u: number): Promise<[Blob, Blob]> {
    await this.until(() => (this.index ? framesPackFrameEnd(this.index, u) : Infinity));
    const f = this.index!.frames[u];
    if (!f) throw new Error(`${this.url}: no source frame ${u}`);
    const slice = (o: number, n: number): Blob => new Blob([this.data.slice(o, o + n) as Uint8Array<ArrayBuffer>], { type: this.type });
    return [slice(f[0], f[1]), slice(f[2], f[3])];
  }

  private until(need: () => number): Promise<void> {
    if (this.failed) return Promise.reject(this.failed);
    if (this.received >= need()) return Promise.resolve();
    return new Promise((resolve, reject) => { this.waiters.push({ need, resolve, reject }); });
  }

  private arrived(chunk: Uint8Array): void {
    if (this.received + chunk.length > this.data.length) {
      const grown = new Uint8Array(Math.max(this.received + chunk.length, this.data.length * 2, 1 << 20));
      grown.set(this.data.subarray(0, this.received));
      this.data = grown;
    }
    this.data.set(chunk, this.received);
    this.received += chunk.length;
    this.onBytes(chunk.length);
    if (!this.index) {
      const head = framesPackHeadLength(this.data.subarray(0, this.received));
      if (head > 0 && this.received >= head) this.index = readFramesPackIndex(this.data.subarray(0, head), this.url, this.sourceFrames);
    }
    if (this.index) this.waiters = this.waiters.filter((w) => (this.received >= w.need() ? (w.resolve(), false) : true));
  }

  private async load(): Promise<void> {
    const url = new URL(this.url, globalThis.location?.href).href;
    let cache: Cache | null = null;
    try { cache = await caches.open(MODEL_CACHE); } catch { /* no Cache Storage (http: or private mode): plain fetch */ }
    const hit = await cache?.match(url).catch(() => undefined);
    if (hit) {
      const data = new Uint8Array(await hit.arrayBuffer());
      if (!this.sha256 || (await sha256Hex(data)) === this.sha256) { this.finish(data); return; }
      await cache?.delete(url);
    }
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new Error(`${this.url}: HTTP ${res.status}`);
    const total = Number(res.headers.get('content-length') ?? 0);
    if (total > 0) this.data = new Uint8Array(total);
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value?.length) this.arrived(value);
    }
    const all = this.data.subarray(0, this.received);
    if (this.sha256 && (await sha256Hex(all)) !== this.sha256) throw new Error(`${this.url}: SHA-256 mismatch`);
    this.finish(all);
    try { await cache?.put(url, new Response(all.slice(0) as Uint8Array<ArrayBuffer>, { headers: { 'Content-Type': 'application/octet-stream' } })); } catch { /* quota: keep going */ }
  }

  private finish(all: Uint8Array): void {
    if (this.received === 0) { this.data = new Uint8Array(0); this.arrived(all); }
    if (!this.index) throw new Error(`${this.url}: frames.pack is too short`);
    const last = framesPackFrameEnd(this.index, this.sourceFrames - 1);
    if (this.received < last) throw new Error(`${this.url}: frames.pack is truncated`);
    for (const w of this.waiters.splice(0)) w.resolve();
  }
}
