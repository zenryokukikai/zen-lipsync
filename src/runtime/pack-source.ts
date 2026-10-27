// Browser PackSource: URLs resolved against the page, binary files through the model cache (fetch + Cache Storage,
// SHA-256 checked when the manifest gives one), JSON files fetched fresh.
import { AvatarPackError } from '../core/errors.js';
import type { PackSource } from '../core/pack.js';
import { fetchModel, setModelHashes } from './model-cache.js';

const dirUrl = (dir: string): string => new URL(dir.endsWith('/') ? dir : `${dir}/`, globalThis.location?.href).href;

/** onBytes: called with the size of every binary file loaded. */
export function browserPackSource(onBytes: (n: number) => void = () => undefined): PackSource {
  return {
    join: (dir, rel) => new URL(rel, dirUrl(dir)).href.replace(/\/+$/, ''),
    async bytes(path, sha256) {
      const url = new URL(path, globalThis.location?.href).href;
      if (sha256) setModelHashes(url.slice(0, url.lastIndexOf('/')), { [url.slice(url.lastIndexOf('/') + 1)]: { sha256 } });
      const buf = await (await fetchModel(url)).arrayBuffer();
      onBytes(buf.byteLength);
      return buf;
    },
    async json<T>(path: string): Promise<T> {
      const res = await fetch(path);
      if (!res.ok) throw new AvatarPackError(`${path}: HTTP ${res.status}`);
      return (await res.json()) as T;
    },
  };
}

/** Registers the manifest's sha256 of every file of a pack (the model files are verified when fetched). */
export function registerPackHashes(dir: string, files: Record<string, { sha256: string }>): void {
  setModelHashes(new URL(dir, globalThis.location?.href).href.replace(/\/+$/, ''), files);
}
