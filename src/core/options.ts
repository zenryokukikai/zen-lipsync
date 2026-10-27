// Validation of create() options, shared by both entries.
import { OptionsError } from './errors.js';
import type { CreateOptions, LipWeights } from './types.js';

const WEIGHTS: readonly LipWeights[] = ['i8', 'f32', 'i8q'];

export interface ResolvedOptions {
  avatar: string;
  weights: LipWeights;
  idle: string | null;
  onProgress: NonNullable<CreateOptions['onProgress']>;
}

/** Checks create() options (unknown keys are rejected so typos are not silently ignored) and fills the defaults. */
export function resolveOptions(opts: unknown): ResolvedOptions {
  if (typeof opts !== 'object' || opts === null || Array.isArray(opts)) throw new OptionsError('create(): options must be an object');
  const o = opts as Record<string, unknown>;
  for (const k of Object.keys(o)) if (!['avatar', 'assetsBaseUrl', 'weights', 'idle', 'onProgress'].includes(k)) throw new OptionsError(`create(): unknown option "${k}"`);
  if (typeof o['avatar'] !== 'string' || o['avatar'].trim() === '') throw new OptionsError('create(): "avatar" must be the URL or path of avatar.json or of an avatar pack directory');
  const weights = o['weights'] ?? 'i8';
  if (!WEIGHTS.includes(weights as LipWeights)) throw new OptionsError(`create(): "weights" must be one of ${WEIGHTS.join(', ')}`);
  const idle = o['idle'] ?? null;
  if (idle !== null && (typeof idle !== 'string' || idle === '')) throw new OptionsError('create(): "idle" must be the name of an idle loop');
  const base = o['assetsBaseUrl'] ?? null;
  if (base !== null && (typeof base !== 'string' || base.trim() === '')) throw new OptionsError('create(): "assetsBaseUrl" must be a URL or a directory');
  const onProgress = o['onProgress'] ?? (() => undefined);
  if (typeof onProgress !== 'function') throw new OptionsError('create(): "onProgress" must be a function');
  return { avatar: joinAssets(base as string | null, o['avatar']).replace(/\/+$/, ''), weights: weights as LipWeights, idle: idle as string | null, onProgress: onProgress as ResolvedOptions['onProgress'] };
}

/** avatar resolved against assetsBaseUrl: absolute URLs and absolute paths are kept; else base + '/' + avatar. */
export function joinAssets(base: string | null, avatar: string): string {
  if (base === null || /^[a-z][a-z0-9+.-]*:/i.test(avatar) || avatar.startsWith('/')) return avatar;
  return `${base.replace(/\/+$/, '')}/${avatar.replace(/^\.\/+/, '')}`;
}
