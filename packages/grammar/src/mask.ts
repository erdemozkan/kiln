import { accepts, push, start, stepAll, type Node, type Stack } from "./matcher";

/**
 * Every token's bytes in a prefix tree, flattened into typed arrays. Tokens that share a prefix share
 * the matcher steps for it, and a whole subtree is skipped as soon as its prefix is rejected.
 */
export class TokenTrie {
  /** Per node: first child edge and number of edges. */
  private readonly edgeStart: Int32Array;
  private readonly edgeCount: Int32Array;
  /** Per edge: byte and child node. */
  private readonly edgeByte: Uint8Array;
  private readonly edgeNode: Int32Array;
  /** Per node: tokens whose bytes end here, as a range of `tokIds`. */
  private readonly tokStart: Int32Array;
  private readonly tokCount: Int32Array;
  private readonly tokIds: Int32Array;

  /** `tokens[id]` is the token's bytes, or undefined for ids text can never produce (special tokens). */
  constructor(tokens: readonly (Uint8Array | undefined)[]) {
    const children: Map<number, number>[] = [new Map()];
    const ends: number[][] = [[]];
    tokens.forEach((bytes, id) => {
      if (!bytes?.length) return;
      let n = 0;
      for (const b of bytes) {
        let c = children[n]!.get(b);
        if (c === undefined) {
          c = children.length;
          children.push(new Map());
          ends.push([]);
          children[n]!.set(b, c);
        }
        n = c;
      }
      ends[n]!.push(id);
    });
    const N = children.length;
    this.edgeStart = new Int32Array(N);
    this.edgeCount = new Int32Array(N);
    this.edgeByte = new Uint8Array(N);
    this.edgeNode = new Int32Array(N);
    this.tokStart = new Int32Array(N);
    this.tokCount = new Int32Array(N);
    this.tokIds = new Int32Array(ends.reduce((n, e) => n + e.length, 0));
    let e = 0;
    let t = 0;
    for (let n = 0; n < N; n++) {
      this.edgeStart[n] = e;
      this.edgeCount[n] = children[n]!.size;
      for (const [b, c] of [...children[n]!].sort((x, y) => x[0] - y[0])) {
        this.edgeByte[e] = b;
        this.edgeNode[e++] = c;
      }
      this.tokStart[n] = t;
      this.tokCount[n] = ends[n]!.length;
      for (const id of ends[n]!) this.tokIds[t++] = id;
    }
  }

  get nodes(): number {
    return this.edgeStart.length;
  }

  /** Sets the bit of every token whose bytes `step` accepts in full from `states`. */
  collect<S>(states: S, step: (s: S, b: number) => S | undefined, mask: Uint32Array): void {
    const walk = (n: number, s: S) => {
      const e0 = this.edgeStart[n]!;
      for (let e = e0; e < e0 + this.edgeCount[n]!; e++) {
        const next = step(s, this.edgeByte[e]!);
        if (next === undefined) continue;
        const c = this.edgeNode[e]!;
        const t0 = this.tokStart[c]!;
        for (let t = t0; t < t0 + this.tokCount[c]!; t++) {
          const id = this.tokIds[t]!;
          mask[id >>> 5]! |= 1 << (id & 31);
        }
        walk(c, next);
      }
    };
    walk(0, states);
  }
}

/** Distinct masks kept per matcher; generation revisits the same grammar positions all the time. */
const CACHE = 256;

/**
 * Tracks a grammar over generated tokens and gives the set of tokens allowed next, as a bitmask of
 * u32 words (bit id % 32 of word id / 32).
 */
export class GrammarMatcher {
  private states: Stack[];
  private readonly cache = new Map<string, Uint32Array>();

  constructor(
    grammar: Node,
    private readonly trie: TokenTrie,
    private readonly tokens: readonly (Uint8Array | undefined)[],
    /** Mask length in tokens (the model's vocabulary, which may exceed the tokenizer's). */
    private readonly vocabSize: number,
    /** Tokens that end generation; allowed exactly when the text so far is a complete match. */
    private readonly stopIds: readonly number[],
  ) {
    this.states = [push(null, start(grammar))];
  }

  /** Whether the text so far is a complete match. */
  get complete(): boolean {
    return this.states.some(accepts);
  }

  /** Whether nothing but a stop token can follow. */
  get finished(): boolean {
    return this.states.every((s) => s === null);
  }

  /** Advances past token `id`. Returns false (and leaves the state unchanged) if the grammar rejects it. */
  accept(id: number): boolean {
    const bytes = this.tokens[id];
    if (!bytes?.length) return this.stopIds.includes(id) && this.complete;
    let s = this.states;
    for (const b of bytes) {
      s = stepAll(s, b);
      if (!s.length) return false;
    }
    this.states = s;
    return true;
  }

  /** Bitmask of the tokens allowed next. The returned array is shared; copy it before keeping it. */
  mask(): Uint32Array {
    const key = this.states.map((s) => s?.key ?? "").sort().join("\n");
    let m = this.cache.get(key);
    if (m) {
      // Refresh its place in the LRU order.
      this.cache.delete(key);
      this.cache.set(key, m);
      return m;
    }
    m = new Uint32Array(Math.ceil(this.vocabSize / 32));
    this.trie.collect(this.states, (s, b) => {
      const next = stepAll(s, b);
      return next.length ? next : undefined;
    }, m);
    if (this.complete) for (const id of this.stopIds) if (id < this.vocabSize) m[id >>> 5]! |= 1 << (id & 31);
    if (this.cache.size >= CACHE) this.cache.delete(this.cache.keys().next().value!);
    this.cache.set(key, m);
    return m;
  }
}

