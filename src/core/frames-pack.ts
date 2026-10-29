// frames.pack (pack version 2): the mouth crops and patch masks of every source frame of one idle loop in one file, so
// a loop is one download instead of one request per image. Layout (little-endian):
//   bytes 0..7   magic "ZLFPACK1"
//   bytes 8..11  uint32 index length L (bytes of the JSON that follows)
//   bytes 12..   JSON index (UTF-8, L bytes):
//                { "type": "image/webp", "frames": [[cropOffset, cropLength, alphaOffset, alphaLength], ...] }
//                one entry per source frame u, offsets counted from the start of the file
//   then the images, in the order u = 0, 1, 2, ... (crop then alpha), so a frame is complete as soon as the file has
//   arrived up to its alpha's end.
// The crop is the RGB mouth crop (crop_box of the 512 face crop), the alpha the greyscale patch mask (rect size): the
// same pictures as frames/crop_NNN.png and frames/alpha_NNN.png of a version 1 pack.

export const FRAMES_PACK_MAGIC = 'ZLFPACK1';

export interface FramesPackIndex {
  /** MIME type of every image in the pack. */
  type: string;
  /** Per source frame: [crop offset, crop length, alpha offset, alpha length] in bytes from the start of the file. */
  frames: [number, number, number, number][];
}

/** Bytes before the images: the magic, the index length and the index. 0 when the head has not fully arrived. */
export function framesPackHeadLength(head: Uint8Array): number {
  if (head.length < 12) return 0;
  return 12 + new DataView(head.buffer, head.byteOffset, head.byteLength).getUint32(8, true);
}

/** The index of a frames.pack from its first bytes (at least framesPackHeadLength of them). */
export function readFramesPackIndex(head: Uint8Array, where: string, sourceFrames: number): FramesPackIndex {
  const magic = new TextDecoder().decode(head.subarray(0, 8));
  if (magic !== FRAMES_PACK_MAGIC) throw new Error(`${where}: not a frames.pack`);
  const end = framesPackHeadLength(head);
  if (end === 0 || head.length < end) throw new Error(`${where}: frames.pack header is incomplete`);
  let index: FramesPackIndex;
  try { index = JSON.parse(new TextDecoder().decode(head.subarray(12, end))) as FramesPackIndex; } catch { throw new Error(`${where}: frames.pack index is not valid JSON`); }
  if (!Array.isArray(index.frames) || index.frames.length !== sourceFrames) throw new Error(`${where}: frames.pack has ${index.frames?.length ?? 0} frames, the pack has ${sourceFrames}`);
  for (const f of index.frames) {
    if (!Array.isArray(f) || f.length !== 4 || !f.every((v) => Number.isInteger(v) && v >= 0) || f[0] < end || f[2] < end) throw new Error(`${where}: frames.pack index entry is invalid`);
  }
  return index;
}

/** Bytes of the file needed before frame u can be decoded (the end of its last image). */
export function framesPackFrameEnd(index: FramesPackIndex, u: number): number {
  const f = index.frames[u]!;
  return Math.max(f[0] + f[1], f[2] + f[3]);
}
