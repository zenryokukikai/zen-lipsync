// An idle loop's video on Node: video frame k (k = 0 .. walk length - 1) as rgb24, decoded with ffmpeg (BT.709 limited
// range -> full-range RGB, as browsers show it). 'stream' (default) keeps one ffmpeg decoder reading the loop in
// order (a few MB); 'memory' decodes the loop once and keeps every frame (width x height x 3 bytes per frame).
import { spawn, type ChildProcessByStdio } from 'node:child_process';
import type { Readable } from 'node:stream';
import { ffmpegOutput } from './ffmpeg.js';

export type IdleCache = 'stream' | 'memory';

const SCALE = 'scale=in_color_matrix=bt709:in_range=tv:out_range=full,format=rgb24';

export class IdleVideo {
  private all: Buffer | null = null;
  private proc: ChildProcessByStdio<null, Readable, null> | null = null;
  private chunks: Buffer[] = [];
  private have = 0;
  private next = -1;
  private waiter: (() => void) | null = null;
  private failed: Error | null = null;
  readonly frameBytes: number;

  constructor(private readonly file: string, readonly width: number, readonly height: number, readonly frames: number, private readonly ffmpeg: string, private readonly cache: IdleCache) {
    this.frameBytes = width * height * 3;
  }

  /** Decodes the whole loop now ('memory'), or checks that it decodes ('stream'). */
  async open(): Promise<void> {
    if (this.cache === 'memory') {
      const b = await ffmpegOutput(this.ffmpeg, ['-i', this.file, '-vf', SCALE, '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']);
      if (b.length < this.frames * this.frameBytes) throw new Error(`${this.file}: ${Math.floor(b.length / this.frameBytes)} frames decoded, the pack has ${this.frames}`);
      // exactly the loop's frames (ffmpeg may decode a few more); a copy so the decoder's chunk buffers are freed
      this.all = Buffer.from(b.subarray(0, this.frames * this.frameBytes));
    } else await this.frame(0);
  }

  /** rgb24 of video frame k (mod the loop). The buffer is only valid until the next call. */
  async frame(k: number): Promise<Buffer> {
    k = ((k % this.frames) + this.frames) % this.frames;
    if (this.all) return this.all.subarray(k * this.frameBytes, (k + 1) * this.frameBytes);
    if (k !== this.next || !this.proc) this.start(k);
    while (this.have < this.frameBytes) {
      if (this.failed) throw this.failed;
      await new Promise<void>((r) => { this.waiter = r; });
    }
    const out = Buffer.allocUnsafe(this.frameBytes);
    let off = 0;
    while (off < this.frameBytes) {
      const c = this.chunks[0]!;
      const n = Math.min(c.length, this.frameBytes - off);
      c.copy(out, off, 0, n);
      off += n;
      if (n === c.length) this.chunks.shift(); else this.chunks[0] = c.subarray(n);
    }
    this.have -= this.frameBytes;
    if (this.have < 2 * this.frameBytes) this.proc?.stdout.resume();
    this.next = (k + 1) % this.frames;
    return out;
  }

  private start(k: number): void {
    this.stop();
    const args = ['-hide_banner', '-loglevel', 'error', '-stream_loop', '-1'];
    if (k > 0) args.push('-ss', ((k - 0.5) / 25).toFixed(4));
    args.push('-i', this.file, '-vf', SCALE, '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1');
    const p = spawn(this.ffmpeg, args, { stdio: ['ignore', 'pipe', 'ignore'] });
    this.proc = p;
    this.chunks = []; this.have = 0; this.next = k; this.failed = null;
    p.stdout.on('data', (d: Buffer) => {
      if (this.proc !== p) return;
      this.chunks.push(d); this.have += d.length;
      if (this.have >= 3 * this.frameBytes) p.stdout.pause();
      this.wake();
    });
    p.on('error', (e) => { if (this.proc === p) { this.failed = e; this.wake(); } });
    p.on('close', () => { if (this.proc === p && this.have < this.frameBytes) { this.failed = new Error(`${this.file}: the ffmpeg decoder stopped`); this.wake(); } });
  }

  private wake(): void { const w = this.waiter; this.waiter = null; w?.(); }

  private stop(): void {
    const p = this.proc;
    this.proc = null;
    if (p) { p.stdout.removeAllListeners('data'); p.kill('SIGKILL'); }
  }

  /** Bytes held (decoded frames). */
  get bytes(): number { return this.all ? this.all.length : this.have; }

  close(): void { this.stop(); this.all = null; this.chunks = []; this.have = 0; }
}
