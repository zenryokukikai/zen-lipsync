// A small typed event emitter for AvatarHandle (no DOM dependency, so Node shares it).
import type { AvatarEvents } from './types.js';

type Listener = (e: never) => void;

export class Emitter {
  private readonly map = new Map<string, Set<Listener>>();

  on<K extends keyof AvatarEvents>(event: K, listener: (e: AvatarEvents[K]) => void): () => void;
  on(event: string, listener: (e: never) => void): () => void;
  on(event: string, listener: Listener): () => void {
    let s = this.map.get(event);
    if (!s) this.map.set(event, (s = new Set()));
    s.add(listener);
    return () => { s!.delete(listener); };
  }

  /** Resolves on the next emit of event. */
  once(event: string): Promise<unknown> {
    return new Promise((resolve) => { const off = this.on(event, (e: never) => { off(); resolve(e); }); });
  }

  emit(event: string, payload: unknown): void {
    for (const l of [...(this.map.get(event) ?? [])]) {
      try { (l as (e: unknown) => void)(payload); } catch (err) { if (event !== 'error') this.emit('error', { error: err instanceof Error ? err : new Error(String(err)) }); }
    }
  }

  clear(): void { this.map.clear(); }
}
