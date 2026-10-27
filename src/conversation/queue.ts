// A push queue as an AsyncIterable: the reply's audio is pushed in as it is synthesised, and the avatar pulls it
// through speakStream(). end() finishes the stream, fail() makes the consumer throw, close() drops what is left
// (barge-in: the avatar is stopped and the producer's next push is ignored).

export class PushQueue<T> implements AsyncIterable<T> {
  private readonly items: T[] = [];
  private readonly waiters: ((r: IteratorResult<T>) => void)[] = [];
  private done = false;
  private failure: Error | null = null;
  /** Values dropped because the queue was closed (barge-in). */
  dropped = 0;

  push(v: T): void {
    if (this.done) { this.dropped++; return; }
    const w = this.waiters.shift();
    if (w) w({ done: false, value: v });
    else this.items.push(v);
  }

  end(): void {
    if (this.done) return;
    this.done = true;
    for (const w of this.waiters.splice(0)) w({ done: true, value: undefined });
  }

  fail(error: Error): void {
    if (this.done) return;
    this.failure = error;
    this.end();
  }

  close(): void {
    this.dropped += this.items.length;
    this.items.length = 0;
    this.end();
  }

  get closed(): boolean { return this.done; }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: () => {
        const v = this.items.shift();
        if (v !== undefined) return Promise.resolve({ done: false as const, value: v });
        if (this.failure) return Promise.reject(this.failure);
        if (this.done) return Promise.resolve({ done: true as const, value: undefined });
        return new Promise((resolve) => { this.waiters.push(resolve as (r: IteratorResult<T>) => void); });
      },
      return: () => Promise.resolve({ done: true as const, value: undefined }),
    };
  }
}
