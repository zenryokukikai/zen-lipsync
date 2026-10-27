// Model files: fetched once and kept in Cache Storage, verified by SHA-256 when the avatar pack's manifest gives one.
// Images and the idle video are left to the browser's normal loaders and HTTP cache.

export const MODEL_CACHE = 'zen-lipsync-models-v1';

/** sha256 (hex) per file URL, from the avatar pack's manifest.json (setModelHashes); unknown files are not verified. */
const hashes = new Map<string, string>();
const pending = new Map<string, Promise<ArrayBuffer>>();

export function setModelHashes(base: string, files: Record<string, { sha256: string }>): void {
  for (const [name, f] of Object.entries(files)) {
    if (/^[a-f0-9]{64}$/.test(f.sha256)) hashes.set(new URL(name, new URL(`${base}/`, location.href)).href, f.sha256);
  }
}

async function sha256Hex(data: ArrayBuffer): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', data))].map((v) => v.toString(16).padStart(2, '0')).join('');
}

async function load(url: string): Promise<ArrayBuffer> {
  const want = hashes.get(url);
  let cache: Cache | null = null;
  try { cache = await caches.open(MODEL_CACHE); } catch { /* no Cache Storage (http: or private mode): plain fetch */ }
  const hit = await cache?.match(url).catch(() => undefined);
  if (hit) {
    const data = await hit.arrayBuffer();
    if (!want || (await sha256Hex(data)) === want) return data;
    await cache?.delete(url);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const data = await res.arrayBuffer();
  if (want && (await sha256Hex(data)) !== want) throw new Error(`${url}: SHA-256 mismatch`);
  try { await cache?.put(url, new Response(data.slice(0), { headers: { 'Content-Type': 'application/octet-stream' } })); } catch { /* quota: keep going */ }
  return data;
}

/** A model file as a Response (a fresh body for every caller; one network fetch per URL at a time). */
export async function fetchModel(path: string): Promise<Response> {
  const url = new URL(path, location.href).href;
  // '..' segments are resolved, so a shared models directory has one cache entry
  let p = pending.get(url);
  if (!p) {
    p = load(url);
    pending.set(url, p);
    void p.finally(() => pending.delete(url)).catch(() => undefined);
  }
  return new Response((await p).slice(0));
}
