// A small PNG decoder for the avatar pack images (8-bit greyscale, RGB or RGBA, not interlaced), with node:zlib.
import { inflateSync } from 'node:zlib';

export interface DecodedImage {
  width: number;
  height: number;
  /** 1 (grey), 3 (RGB) or 4 (RGBA). */
  channels: 1 | 3 | 4;
  data: Uint8Array;
}

const SIG = [137, 80, 78, 71, 13, 10, 26, 10];

export function decodePng(buf: Uint8Array): DecodedImage {
  for (let i = 0; i < 8; i++) if (buf[i] !== SIG[i]) throw new Error('not a PNG file');
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let p = 8, width = 0, height = 0, channels: 1 | 3 | 4 = 3;
  const idat: Uint8Array[] = [];
  while (p + 8 <= buf.length) {
    const len = dv.getUint32(p);
    const type = String.fromCharCode(buf[p + 4]!, buf[p + 5]!, buf[p + 6]!, buf[p + 7]!);
    const body = buf.subarray(p + 8, p + 8 + len);
    if (type === 'IHDR') {
      width = dv.getUint32(p + 8); height = dv.getUint32(p + 12);
      const depth = body[8], ct = body[9], interlace = body[12];
      if (depth !== 8 || interlace !== 0) throw new Error('PNG: only 8-bit, not interlaced');
      channels = ct === 0 ? 1 : ct === 2 ? 3 : ct === 6 ? 4 : (() => { throw new Error(`PNG: colour type ${ct} not supported`); })();
    } else if (type === 'IDAT') idat.push(body);
    else if (type === 'IEND') break;
    p += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = new Uint8Array(height * stride);
  const bpp = channels;
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)]!;
    const src = y * (stride + 1) + 1;
    const dst = y * stride;
    const up = dst - stride;
    for (let x = 0; x < stride; x++) {
      const v = raw[src + x]!;
      const a = x >= bpp ? out[dst + x - bpp]! : 0;
      const b = y > 0 ? out[up + x]! : 0;
      const c = x >= bpp && y > 0 ? out[up + x - bpp]! : 0;
      let r: number;
      switch (f) {
        case 0: r = v; break;
        case 1: r = v + a; break;
        case 2: r = v + b; break;
        case 3: r = v + ((a + b) >> 1); break;
        case 4: { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); r = v + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c); break; }
        default: throw new Error(`PNG: bad filter ${f}`);
      }
      out[dst + x] = r & 255;
    }
  }
  return { width, height, channels, data: out };
}
