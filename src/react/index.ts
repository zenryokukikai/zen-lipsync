// zen-lipsync/react: <ZenLipSync> puts an avatar in the page and hands its AvatarHandle to onReady. React is an
// optional peer dependency; this file imports only the React API it uses (createElement, useEffect, useRef).
import { createElement, useEffect, useRef, type CSSProperties, type ReactElement } from 'react';
import { create, type BrowserCreateOptions } from '../browser/create.js';
import type { AvatarHandle } from '../core/types.js';

export interface ZenLipSyncProps extends Omit<BrowserCreateOptions, 'container'> {
  /** Called once the avatar is loaded (also with WebGPUUnavailableError and load errors through onError). */
  onReady?: (avatar: AvatarHandle) => void;
  onError?: (error: Error) => void;
  className?: string;
  style?: CSSProperties;
}

/** The avatar in a <div>. Changing `avatar`, `assetsBaseUrl`, `weights` or `idle` loads it again; unmount destroys it. */
export function ZenLipSync(props: ZenLipSyncProps): ReactElement {
  const ref = useRef<HTMLDivElement | null>(null);
  const cb = useRef(props);
  cb.current = props;
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    let handle: AvatarHandle | null = null;
    let gone = false;
    const { avatar, assetsBaseUrl, weights, idle, onProgress, fit } = cb.current;
    const opts: BrowserCreateOptions = { container: el, avatar, ...(assetsBaseUrl ? { assetsBaseUrl } : {}), ...(weights ? { weights } : {}), ...(idle ? { idle } : {}), ...(onProgress ? { onProgress } : {}), ...(fit ? { fit } : {}) };
    create(opts).then((h) => {
      if (gone) { h.destroy(); return; }
      handle = h;
      cb.current.onReady?.(h);
    }, (e: unknown) => { if (!gone) cb.current.onError?.(e instanceof Error ? e : new Error(String(e))); });
    return () => { gone = true; handle?.destroy(); };
  }, [props.avatar, props.assetsBaseUrl, props.weights, props.idle, props.fit]);
  return createElement('div', { ref, className: props.className, style: props.style });
}

export type { AvatarHandle } from '../core/types.js';
export { WebGPUUnavailableError } from '../core/errors.js';
