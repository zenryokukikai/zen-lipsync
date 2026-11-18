// Frame-exact idle video frames for the offline renderer: video frame k of a pack's WebM, decoded with WebCodecs'
// VideoDecoder from the key frame at or before k (a <video> seek lands on whatever frame the browser picks, which is
// what put the mouth on the wrong frame when a playback was recorded). Frames are decoded in order a few ahead of the
// one asked for; nothing else is kept, so a whole loop is never expanded in memory.
import { readWebm, webCodecsCodec, type WebmVideo } from './webm.js';

/** Frames fed to the decoder ahead of the one asked for (a decoder may hold a few before it outputs one). */
const AHEAD = 4;
/** Fed further ahead, one at a time, while a frame does not come out (a decoder holding more than AHEAD). */
const MAX_EXTRA = 24;
const STALL_MS = 40;

export class IdleFrames {
  private dec: VideoDecoder | null = null;
  /** Next frame index to feed, and the lowest index still wanted (outputs below it are closed). */
  private fed = 0;
  private want = 0;
  private flushed = false;
  private readonly ready = new Map<number, VideoFrame>();
  private tick: (() => void) | null = null;
  private failed: Error | null = null;
  private closed = false;

  private constructor(private readonly buf: Uint8Array, readonly video: WebmVideo, private readonly codec: string) {}

  /** The WebM file's bytes; throws when it is not a WebM with a video track or the browser cannot decode its codec. */
  static async open(bytes: ArrayBuffer): Promise<IdleFrames> {
    if (typeof VideoDecoder === 'undefined') throw new Error('ZEN-LipSync: the renderer needs WebCodecs (VideoDecoder)');
    const buf = new Uint8Array(bytes);
    const video = readWebm(buf);
    const codec = webCodecsCodec(video.track);
    const s = await VideoDecoder.isConfigSupported({ codec, codedWidth: video.track.width, codedHeight: video.track.height });
    if (!s.supported) throw new Error(`ZEN-LipSync: this browser cannot decode the idle video (${codec}) with WebCodecs`);
    return new IdleFrames(buf, video, codec);
  }

  /** Frames in the video (the loop). */
  get length(): number { return this.video.frames.length; }
  get width(): number { return this.video.track.width; }
  get height(): number { return this.video.track.height; }

  /** Video frame k (mod the loop's length). The caller owns the frame and closes it. */
  async frame(k: number): Promise<VideoFrame> {
    if (this.closed) throw new Error('ZEN-LipSync: the idle video was closed');
    const n = this.length;
    k = ((k % n) + n) % n;
    this.drop(k);
    if (!this.ready.has(k) && (!this.dec || k < this.want || this.keyAtOrBefore(k) > this.fed)) this.restart(k);
    this.want = k;
    let extra = 0;
    while (!this.ready.has(k)) {
      if (this.failed) throw this.failed;
      const target = Math.min(n, k + 1 + AHEAD + extra);
      while (this.fed < target) this.feed();
      if (this.fed >= n && !this.flushed) { this.flushed = true; await this.dec!.flush().catch((e: unknown) => { this.failed ??= e as Error; }); continue; }
      const came = await new Promise<boolean>((r) => { this.tick = () => r(true); setTimeout(() => r(false), STALL_MS); });
      this.tick = null;
      if (!came && extra < MAX_EXTRA) extra++;
    }
    const f = this.ready.get(k)!;
    this.ready.delete(k);
    this.want = k + 1;
    // keep the decoder a few frames ahead of the next frame asked for
    while (this.fed < Math.min(n, k + 1 + AHEAD)) this.feed();
    return f;
  }

  private keyAtOrBefore(k: number): number {
    const frames = this.video.frames;
    let kf = k;
    while (kf > 0 && !frames[kf]!.key) kf--;
    return kf;
  }

  /** Starts decoding again from the key frame at or before k (going back, or a jump past the next key frame). */
  private restart(k: number): void {
    const kf = this.keyAtOrBefore(k);
    for (const f of this.ready.values()) f.close();
    this.ready.clear();
    if (this.dec && this.dec.state !== 'closed') this.dec.close();
    this.dec = new VideoDecoder({
      output: (f) => {
        const i = f.timestamp;
        if (this.closed || i < this.want || this.ready.has(i)) { f.close(); return; }
        this.ready.set(i, f);
        this.tick?.();
      },
      error: (e) => { this.failed ??= e instanceof Error ? e : new Error(String(e)); this.tick?.(); },
    });
    const t = this.video.track;
    this.dec.configure({ codec: this.codec, codedWidth: t.width, codedHeight: t.height, ...(t.codecPrivate && t.codecId === 'V_AV1' ? { description: t.codecPrivate } : {}) });
    this.fed = kf;
    this.want = kf;
    this.flushed = false;
  }

  private feed(): void {
    const i = this.fed++;
    const fr = this.video.frames[i]!;
    // the timestamp carries the frame index: the output frame says which one it is
    this.dec!.decode(new EncodedVideoChunk({ type: fr.key ? 'key' : 'delta', timestamp: i, duration: 1, data: this.buf.subarray(fr.offset, fr.offset + fr.size) }));
  }

  /** Closes decoded frames before k (asked for out of order: they will not be wanted). */
  private drop(k: number): void {
    for (const [i, f] of this.ready) if (i < k) { f.close(); this.ready.delete(i); }
  }

  close(): void {
    this.closed = true;
    for (const f of this.ready.values()) f.close();
    this.ready.clear();
    if (this.dec && this.dec.state !== 'closed') this.dec.close();
    this.dec = null;
  }
}
