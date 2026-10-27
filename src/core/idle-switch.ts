// Switching between idle loops of one avatar (e.g. a calm and an angry idle video) while the avatar is idle:
// the pair of most similar frames of the two loops that the shown loop reaches soon, the other loop seeked to its
// frame, then a short crossfade; the other loop goes on from there and its pack draws the next speech.

/** to[u] = the other loop's source frame most like source frame u of this loop; dist[u] = their difference (mean |RGB|). */
export interface IdlePairTable { readonly to: readonly number[]; readonly dist: readonly number[] }

/** idle-switch.json of an avatar: tables keyed "<from>-><to>". */
export interface IdleSwitchFile { version: 1; sources: string[]; pairs: Record<string, IdlePairTable> }

export interface IdlePairChoice {
  /** Video frames from now until the shown loop is at the pair's frame. */
  readonly waitFrames: number;
  /** Walk position of the shown loop where the fade starts, and its source frame. */
  readonly fromWalk: number;
  readonly fromSource: number;
  /** Source frame of the other loop the fade goes to. */
  readonly toSource: number;
  readonly dist: number;
}

/** Pairs at most this far apart (mean |RGB|, 0..255) are taken at once; else the closest pair in the window. */
export const IDLE_GOOD_PAIR_DIST = 4;

/** The first walk position in [now + minWait, now + maxWait] with a good pair, else the closest pair there. */
export function chooseIdlePair(walkNow: number, walk: readonly number[], table: IdlePairTable, minWait: number, maxWait: number, goodDist = IDLE_GOOD_PAIR_DIST): IdlePairChoice {
  const L = walk.length;
  let best: IdlePairChoice | null = null;
  for (let k = minWait; k <= Math.max(minWait, maxWait); k++) {
    const w = (((walkNow + k) % L) + L) % L;
    const u = walk[w]!;
    const d = table.dist[u]!;
    const c = { waitFrames: k, fromWalk: w, fromSource: u, toSource: table.to[u]!, dist: d };
    if (d <= goodDist) return c;
    if (!best || d < best.dist) best = c;
  }
  return best!;
}
