/** A saved K/V prefix: the tokens it holds, how far its cache was written, and its storage. */
export interface SavedPrefix<T> {
  readonly tokens: readonly number[];
  /** Positions written when it was saved (ring layers need it; see `LlamaModel.minReuseStart`). */
  readonly written: number;
  readonly bytes: number;
  readonly data: T;
  used: number;
}

/** Length of the common prefix of a and b. */
export function commonPrefix(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

/**
 * Saved K/V prefixes under a byte budget, least recently used evicted first. A prefix that another
 * saved one extends is redundant (any prompt it serves, the longer one serves at least as well) and
 * is dropped when the longer one arrives.
 */
export class PrefixStore<T> {
  private readonly entries: SavedPrefix<T>[] = [];
  private clock = 0;

  constructor(
    public budget: number,
    private readonly free: (data: T) => void,
  ) {}

  get size(): number {
    return this.entries.length;
  }

  get bytes(): number {
    return this.entries.reduce((n, e) => n + e.bytes, 0);
  }

  /**
   * The saved prefix sharing the most tokens with `prompt`, if it shares more than `atLeast`. Reuse
   * stops one token short of the prompt (its last token is always prefilled for logits), and below
   * `minStart(written)` it is not possible at all.
   */
  best(prompt: readonly number[], atLeast: number, minStart: (written: number) => number): { entry: SavedPrefix<T>; length: number } | undefined {
    let found: { entry: SavedPrefix<T>; length: number } | undefined;
    for (const entry of this.entries) {
      const length = Math.min(commonPrefix(entry.tokens, prompt), prompt.length - 1);
      if (length <= (found?.length ?? atLeast) || length < minStart(entry.written)) continue;
      found = { entry, length };
    }
    if (found) found.entry.used = ++this.clock;
    return found;
  }

  /** True when a saved prefix already holds `tokens` (equal to it or extending it). */
  covers(tokens: readonly number[]): boolean {
    return this.entries.some((e) => e.tokens.length >= tokens.length && commonPrefix(e.tokens, tokens) === tokens.length);
  }

  /**
   * Saves a prefix made by `make` (called only if it fits the budget), dropping saved prefixes it
   * extends and then the least recently used until the total fits. Returns false if it was not kept.
   */
  add(tokens: readonly number[], written: number, bytes: number, make: () => T): boolean {
    if (bytes > this.budget || this.covers(tokens)) return false;
    for (let i = this.entries.length - 1; i >= 0; i--) {
      const e = this.entries[i]!;
      if (e.tokens.length <= tokens.length && commonPrefix(e.tokens, tokens) === e.tokens.length) this.drop(i);
    }
    while (this.bytes + bytes > this.budget) {
      let lru = 0;
      for (let i = 1; i < this.entries.length; i++) if (this.entries[i]!.used < this.entries[lru]!.used) lru = i;
      this.drop(lru);
    }
    this.entries.push({ tokens: [...tokens], written, bytes, data: make(), used: ++this.clock });
    return true;
  }

  clear(): void {
    while (this.entries.length) this.drop(this.entries.length - 1);
  }

  private drop(i: number): void {
    this.free(this.entries[i]!.data);
    this.entries.splice(i, 1);
  }
}
