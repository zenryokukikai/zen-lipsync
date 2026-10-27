// Node PackSource: an avatar from the filesystem (paths) or from http(s) URLs. SHA-256 is checked when the manifest
// gives one. Uses only node:fs/promises and node:path (plus the global fetch of Node 20+ for URLs).
import { readFile } from 'node:fs/promises';
import * as path from 'node:path';
import { AvatarPackError } from '../core/errors.js';
import { sha256Hex, type PackSource } from '../core/pack.js';

const isUrl = (p: string): boolean => /^https?:\/\//.test(p);

async function read(p: string): Promise<ArrayBuffer> {
  if (isUrl(p)) {
    const res = await fetch(p);
    if (!res.ok) throw new AvatarPackError(`${p}: HTTP ${res.status}`);
    return res.arrayBuffer();
  }
  try {
    const b = await readFile(p);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  } catch (e) {
    throw new AvatarPackError(`${p}: ${(e as NodeJS.ErrnoException).code === 'ENOENT' ? 'file not found' : (e as Error).message}`);
  }
}

/** onBytes: called with the size of every binary file read. */
export function nodePackSource(onBytes: (n: number) => void = () => undefined): PackSource {
  return {
    join: (dir, rel) => (isUrl(dir) ? new URL(rel, dir.endsWith('/') ? dir : `${dir}/`).href.replace(/\/+$/, '') : path.join(dir, rel)),
    async bytes(p, sha256) {
      const data = await read(p);
      if (sha256 && (await sha256Hex(data)) !== sha256) throw new AvatarPackError(`${p}: SHA-256 mismatch`);
      onBytes(data.byteLength);
      return data;
    },
    async json<T>(p: string): Promise<T> {
      const data = await read(p);
      try { return JSON.parse(new TextDecoder().decode(data)) as T; } catch { throw new AvatarPackError(`${p}: not valid JSON`); }
    },
  };
}
