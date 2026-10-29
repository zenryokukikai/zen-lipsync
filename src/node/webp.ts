// WebP decoding on Node for the frames.pack images: libwebp's decoder as WebAssembly (cpu/webp/, built by jSquash),
// loaded once from the package itself, so nothing needs to be installed.
import { readFile } from 'node:fs/promises';
import type { DecodedImage } from './png.js';

interface WebpModule { decode(data: Uint8Array): { data: Uint8ClampedArray; width: number; height: number } | null }
let mod: Promise<WebpModule> | null = null;

function load(): Promise<WebpModule> {
  mod ??= (async () => {
    const base = new URL('../../cpu/webp/', import.meta.url);
    const [factory, wasm] = await Promise.all([import(new URL('webp_dec.mjs', base).href), readFile(new URL('webp_dec.wasm', base))]);
    return (await (factory.default as (o: object) => Promise<WebpModule>)({ wasmBinary: wasm, noInitialRun: true }));
  })();
  mod.catch(() => { mod = null; });
  return mod;
}

/** Loads the decoder (once); decodeWebp needs it loaded. */
export async function initWebp(): Promise<void> { await load(); }

let ready: WebpModule | null = null;
/** Decodes a WebP image to RGB (channels 3) or grey (channels 1, the red channel). Call initWebp() first. */
export async function decodeWebp(data: Uint8Array, channels: 1 | 3): Promise<DecodedImage> {
  ready ??= await load();
  const r = ready.decode(data);
  if (!r) throw new Error('WebP: decoding failed');
  const n = r.width * r.height;
  const out = new Uint8Array(n * channels);
  const s = r.data;
  if (channels === 3) for (let i = 0; i < n; i++) { out[i * 3] = s[i * 4]!; out[i * 3 + 1] = s[i * 4 + 1]!; out[i * 3 + 2] = s[i * 4 + 2]!; }
  else for (let i = 0; i < n; i++) out[i] = s[i * 4]!;
  return { width: r.width, height: r.height, channels, data: out };
}
