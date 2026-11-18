// A small WebM (Matroska) reader for the idle videos: the first video track's codec, size and every frame's bytes in
// file order (= decode order), with its key-frame flag. Enough for VideoDecoder (WebCodecs) to decode frame k exactly
// by index, which a <video> seek cannot promise. Lacing (never used for video by the encoders we use) is refused.

export interface WebmTrack {
  /** Matroska codec id, e.g. 'V_VP9', 'V_AV1', 'V_VP8'. */
  codecId: string;
  /** CodecPrivate (AV1: the av1C box body), if any. */
  codecPrivate: Uint8Array | null;
  width: number;
  height: number;
}

export interface WebmFrame {
  /** Byte range of the frame in the file. */
  offset: number;
  size: number;
  key: boolean;
  /** Presentation time (ms). */
  timeMs: number;
}

export interface WebmVideo {
  track: WebmTrack;
  /** The video track's frames, in presentation order (the idle videos have no reordering: also decode order). */
  frames: WebmFrame[];
}

const ID = {
  Segment: 0x18538067, Info: 0x1549a966, TimecodeScale: 0x2ad7b1, Tracks: 0x1654ae6b, TrackEntry: 0xae, TrackNumber: 0xd7,
  TrackType: 0x83, CodecID: 0x86, CodecPrivate: 0x63a2, Video: 0xe0, PixelWidth: 0xb0, PixelHeight: 0xba, Cluster: 0x1f43b675,
  Timecode: 0xe7, SimpleBlock: 0xa3, BlockGroup: 0xa0, Block: 0xa1, ReferenceBlock: 0xfb,
} as const;
/** Master elements read in place (their children follow); every other element is skipped as a whole. */
const DESCEND = new Set<number>([ID.Segment, ID.Info, ID.Tracks, ID.TrackEntry, ID.Video, ID.Cluster, ID.BlockGroup]);

/** Reads the first video track of a WebM file. Throws on a file that is not WebM or has no video track. */
export function readWebm(buf: Uint8Array): WebmVideo {
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 0;
  const vint = (keepMarker: boolean): { value: number; len: number; unknown: boolean } => {
    const b = buf[pos]!;
    let len = 1;
    while (len <= 8 && !(b & (0x80 >> (len - 1)))) len++;
    if (len > 8) throw new Error('webm: bad element header');
    let value = keepMarker ? b : b & (0xff >> len);
    let allOnes = (b & (0xff >> len)) === 0xff >> len;
    for (let i = 1; i < len; i++) { const x = buf[pos + i]!; value = value * 256 + x; if (x !== 0xff) allOnes = false; }
    pos += len;
    return { value, len, unknown: !keepMarker && allOnes };
  };
  const uint = (at: number, size: number): number => { let x = 0; for (let i = 0; i < size; i++) x = x * 256 + buf[at + i]!; return x; };

  if (buf.length < 4 || v.getUint32(0) !== 0x1a45dfa3) throw new Error('webm: not a WebM/Matroska file');
  let scaleNs = 1_000_000;
  const tracks: (WebmTrack & { number: number; type: number })[] = [];
  let entry: (WebmTrack & { number: number; type: number }) | null = null;
  let clusterTime = 0;
  let videoTrack = -1;
  const frames: WebmFrame[] = [];
  const groupBlocks: { offset: number; end: number }[] = [];
  let groupHasRef = false;
  let groupEnd = -1;

  const block = (start: number, end: number, simple: boolean, keyHint: boolean): void => {
    pos = start;
    const track = vint(false).value;
    const rel = v.getInt16(pos); pos += 2;
    const flags = buf[pos++]!;
    if (videoTrack < 0) videoTrack = (tracks.find((t) => t.type === 1) ?? { number: -2 }).number;
    if (track !== videoTrack) return;
    if (flags & 0x06) throw new Error('webm: laced video blocks are not supported');
    frames.push({ offset: pos, size: end - pos, key: simple ? (flags & 0x80) !== 0 : keyHint, timeMs: ((clusterTime + rel) * scaleNs) / 1e6 });
  };
  const closeGroup = (): void => {
    for (const b of groupBlocks) block(b.offset, b.end, false, !groupHasRef);
    groupBlocks.length = 0; groupHasRef = false; groupEnd = -1;
  };

  pos = 0;
  while (pos < buf.length) {
    if (groupEnd >= 0 && pos >= groupEnd) { const keep = pos; closeGroup(); pos = keep; }
    const id = vint(true).value;
    const sz = vint(false);
    const start = pos;
    const end = sz.unknown ? buf.length : Math.min(buf.length, start + sz.value);
    if (DESCEND.has(id)) {
      if (id === ID.TrackEntry) { entry = { number: 0, type: 0, codecId: '', codecPrivate: null, width: 0, height: 0 }; tracks.push(entry); }
      if (id === ID.BlockGroup) { if (groupEnd >= 0) closeGroup(); groupEnd = end; }
      continue; // children follow
    }
    switch (id) {
      case ID.TimecodeScale: scaleNs = uint(start, end - start); break;
      case ID.TrackNumber: if (entry) entry.number = uint(start, end - start); break;
      case ID.TrackType: if (entry) entry.type = uint(start, end - start); break;
      case ID.CodecID: if (entry) entry.codecId = new TextDecoder().decode(buf.subarray(start, end)).replace(/\0+$/, ''); break;
      case ID.CodecPrivate: if (entry) entry.codecPrivate = buf.slice(start, end); break;
      case ID.PixelWidth: if (entry) entry.width = uint(start, end - start); break;
      case ID.PixelHeight: if (entry) entry.height = uint(start, end - start); break;
      case ID.Timecode: clusterTime = uint(start, end - start); break;
      case ID.SimpleBlock: block(start, end, true, false); break;
      case ID.Block: groupBlocks.push({ offset: start, end }); break;
      case ID.ReferenceBlock: groupHasRef = true; break;
      default: break;
    }
    pos = end;
  }
  if (groupEnd >= 0) closeGroup();
  const t = tracks.find((x) => x.type === 1);
  if (!t) throw new Error('webm: no video track');
  frames.sort((a, b) => a.timeMs - b.timeMs);
  if (!frames.length || !frames[0]!.key) throw new Error('webm: the video does not start with a key frame');
  return { track: { codecId: t.codecId, codecPrivate: t.codecPrivate, width: t.width, height: t.height }, frames };
}

/** WebCodecs codec string for a Matroska video codec id (8-bit 4:2:0, the idle videos' format). */
export function webCodecsCodec(t: WebmTrack): string {
  switch (t.codecId) {
    case 'V_VP9': return 'vp09.00.10.08';
    case 'V_VP8': return 'vp8';
    case 'V_AV1': return 'av01.0.08M.08';
    default: throw new Error(`webm: codec ${t.codecId} is not supported for frame-exact decoding`);
  }
}
