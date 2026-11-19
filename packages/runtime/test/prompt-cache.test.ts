import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CpuBackend } from "@kiln/kernels";
import { LlamaModel, configFromHF } from "@kiln/models";
import { Tokenizer } from "@kiln/tokenizers";
import { Engine } from "../src/engine";
import { loadPromptCache, loadRecurrentStates, promptCacheName, savePromptCache, saveRecurrentStates, type PromptCacheSource } from "../src/prompt-cache";

/** In-memory OPFS: the directory calls and sync-handle methods the cache files use. */
const files = new Map<string, Uint8Array>();
class FakeHandle {
  constructor(private readonly name: string) {}
  truncate(n: number) {
    files.set(this.name, files.get(this.name)!.slice(0, n));
  }
  write(buf: Uint8Array, { at }: { at: number }) {
    const old = files.get(this.name)!;
    const next = new Uint8Array(Math.max(old.length, at + buf.length));
    next.set(old);
    next.set(buf, at);
    files.set(this.name, next);
    return buf.length;
  }
  flush() {}
  close() {}
}
const dir = {
  getDirectoryHandle: async () => dir,
  removeEntry: async (name: string) => {
    if (!files.delete(name)) throw new Error("NotFoundError");
  },
  getFileHandle: async (name: string, o?: { create?: boolean }) => {
    if (!files.has(name)) {
      if (!o?.create) throw new Error("NotFoundError");
      files.set(name, new Uint8Array(0));
    }
    return {
      createSyncAccessHandle: async () => new FakeHandle(name),
      getFile: async () => new Blob([files.get(name)!.slice()]),
    };
  },
};

beforeEach(() => {
  files.clear();
  vi.stubGlobal("FileSystemSyncAccessHandle", FakeHandle);
  vi.stubGlobal("navigator", { storage: { getDirectory: async () => dir } });
});
afterEach(() => {
  vi.unstubAllGlobals();
});

/**
 * A stand-in for the engine: layer 0 keeps every position (12 slots), layer 1 is a ring of 3. Each
 * row is two bytes that name its layer, token and position, so a misplaced row shows.
 */
class FakeEngine implements PromptCacheSource {
  readonly cacheShape: { kvDtype: string; layers: number; rowBytes: number; slots: number[] };
  cacheTokens: number[] = [];
  reads = 0;
  private k: Uint8Array[];
  private v: Uint8Array[];
  /** A layer of 0 slots keeps no keys and values (linear attention). */
  constructor(slots = [12, 3]) {
    this.cacheShape = { kvDtype: "f16", layers: 2, rowBytes: 2, slots };
    this.k = slots.map((s) => new Uint8Array(s * 2));
    this.v = slots.map((s) => new Uint8Array(s * 2));
  }
  set(tokens: number[]) {
    this.cacheTokens = tokens;
    tokens.forEach((t, p) =>
      this.cacheShape.slots.forEach((slots, l) => {
        if (!slots) return;
        this.k[l]!.set([(t * 7 + l) & 255, p], (p % slots) * 2);
        this.v[l]!.set([(t * 7 + l + 100) & 255, p], (p % slots) * 2);
      }),
    );
  }
  async exportCacheFrom(from: number) {
    this.reads++;
    const to = this.cacheTokens.length;
    const parts: Uint8Array[] = [];
    this.cacheShape.slots.forEach((slots, l) => {
      for (const buf of [this.k[l]!, this.v[l]!]) for (let p = Math.max(from, to - slots); p < to; p++) parts.push(buf.slice((p % slots) * 2, (p % slots) * 2 + 2));
    });
    return { tokens: [...this.cacheTokens], parts };
  }
  /** What `LlamaModel.readKv` would return. */
  snapshot() {
    const n = this.cacheTokens.length;
    const out: number[] = [];
    this.cacheShape.slots.forEach((slots, l) => out.push(...this.k[l]!.subarray(0, Math.min(n, slots) * 2), ...this.v[l]!.subarray(0, Math.min(n, slots) * 2)));
    return { tokens: [...this.cacheTokens], kvDtype: "f16", layers: 2, rowBytes: 2, data: Uint8Array.from(out) };
  }
}

describe("prompt cache", () => {
  it("round-trips a cache, ring layers included", async () => {
    const name = promptCacheName("org/model", { quant: "q4" });
    const e = new FakeEngine();
    e.set([5, 6, 7, 8, 9]);
    const state = await savePromptCache(name, e);
    expect(state?.tokens).toEqual([5, 6, 7, 8, 9]);
    const back = await loadPromptCache(name);
    expect(back?.snapshot).toEqual(e.snapshot());
    expect(back?.state).toEqual(state);
  });

  it("appends what a turn added and leaves the rest of the file alone", async () => {
    const name = promptCacheName("org/model", {});
    const e = new FakeEngine();
    e.set([1, 2, 3, 4]);
    const first = await savePromptCache(name, e);
    const before = files.get(name)!.slice();
    e.set([1, 2, 3, 4, 5, 6]);
    const second = await savePromptCache(name, e, first);
    expect(second?.segments.map((s) => s.to)).toEqual([4, 6]);
    expect(files.get(name)!.slice(0, before.length)).toEqual(before);
    // Two positions' rows (K and V, two layers, two bytes each) and a small header: not the cache again.
    expect(files.get(name)!.length - before.length).toBeLessThan(120);
    expect((await loadPromptCache(name))?.snapshot).toEqual(e.snapshot());
    // Nothing changed: nothing is read back or written.
    const reads = e.reads;
    expect(await savePromptCache(name, e, second)).toBe(second);
    expect(e.reads).toBe(reads);
  });

  it("rewrites from the last segment both token lists share when they part", async () => {
    const name = promptCacheName("org/model", {});
    const e = new FakeEngine();
    let state = undefined as Awaited<ReturnType<typeof savePromptCache>>;
    for (const tokens of [[1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 9, 9, 9, 9, 9, 9], [1, 2, 3], [1, 2], [3, 3, 3, 3]]) {
      e.set(tokens);
      state = await savePromptCache(name, e, state);
      expect(state?.tokens, String(tokens)).toEqual(tokens);
      const back = await loadPromptCache(name);
      expect(back?.snapshot, String(tokens)).toEqual(e.snapshot());
      expect(back?.state, String(tokens)).toEqual(state);
    }
  });

  it("starts the file over when another tab rewrote it since this one last saved", async () => {
    const name = promptCacheName("org/model", {});
    const mine = new FakeEngine();
    mine.set([1, 2, 3, 4]);
    const first = await savePromptCache(name, mine);
    // Another tab, another conversation of the same length: the file's segment ends where mine did.
    const other = new FakeEngine();
    other.set([9, 8, 7, 6]);
    await savePromptCache(name, other);
    expect(files.get(name)!.length).toBe(first!.segments[0]!.end);
    mine.set([1, 2, 3, 4, 5, 6]);
    const second = await savePromptCache(name, mine, first);
    // Not an append onto the other tab's rows: one segment with all of mine.
    expect(second?.segments.map((s) => s.to)).toEqual([6]);
    expect((await loadPromptCache(name))?.snapshot).toEqual(mine.snapshot());
    // And a file that is gone is written whole as well.
    files.delete(name);
    mine.set([1, 2, 3, 4, 5, 6, 7]);
    expect((await savePromptCache(name, mine, second))?.segments.map((s) => s.to)).toEqual([7]);
    expect((await loadPromptCache(name))?.snapshot).toEqual(mine.snapshot());
  });

  it("starts the file over when the state is unknown or was made for another shape", async () => {
    const name = promptCacheName("org/model", {});
    const e = new FakeEngine();
    e.set([1, 2, 3]);
    const state = await savePromptCache(name, e);
    e.set([1, 2, 3, 4]);
    const other = await savePromptCache(name, e, { ...state!, shape: { ...state!.shape, slots: [12, 4] } });
    expect(other?.segments.map((s) => s.to)).toEqual([4]);
    expect((await loadPromptCache(name))?.snapshot).toEqual(e.snapshot());
  });

  it("removes the files earlier layouts left for the same model", async () => {
    const name = promptCacheName("org/model", { quant: "q4" });
    const old = [name.replace("v3.kv", "v2.kv"), name.replace("v3.kv", "v1.kv")];
    const other = promptCacheName("org/other", {}).replace("v3.kv", "v2.kv");
    for (const n of [...old, other]) files.set(n, new Uint8Array(8));
    expect(await loadPromptCache(name)).toBeUndefined();
    for (const n of old) expect(files.has(n), n).toBe(false);
    expect(files.has(other)).toBe(true);
  });

  it("keys on everything that changes the cached values", () => {
    const base = promptCacheName("org/model", {});
    expect(promptCacheName("org/model", { revision: "main", quant: "q4" })).toBe(base);
    for (const other of [promptCacheName("org/other", {}), promptCacheName("org/model", { revision: "v2" }), promptCacheName("org/model", { quant: "f32" }), promptCacheName("org/model", { f16Math: true })]) {
      expect(other).not.toBe(base);
    }
  });

  it("ignores missing and foreign files, and keeps what came before a write cut short", async () => {
    const name = promptCacheName("org/model", {});
    expect(await loadPromptCache(name)).toBeUndefined();
    const e = new FakeEngine();
    e.set([1, 2, 3]);
    const first = await savePromptCache(name, e);
    const whole = e.snapshot();
    e.set([1, 2, 3, 4, 5]);
    await savePromptCache(name, e, first);
    files.set(name, files.get(name)!.slice(0, -5)); // a worker killed mid-write
    const back = await loadPromptCache(name);
    expect(back?.snapshot).toEqual(whole);
    expect(back?.state).toEqual(first);
    files.set(name, files.get(name)!.slice(0, first!.segments[0]!.end - 5));
    expect(await loadPromptCache(name)).toBeUndefined();
    files.set(name, new TextEncoder().encode("not a cache file at all"));
    expect(await loadPromptCache(name)).toBeUndefined();
  });

  it("keeps recurrent states beside the cache, for the tokens they were taken on", async () => {
    const name = promptCacheName("org/recurrent", {});
    const tokens = [4, 5, 6, 7, 8, 9];
    const states = [
      { tokens: 3, data: Uint8Array.from({ length: 40 }, (_, i) => i) },
      { tokens: 5, data: Uint8Array.from({ length: 40 }, (_, i) => 200 - i) },
    ];
    expect(await loadRecurrentStates(name, tokens)).toEqual([]);
    expect(await saveRecurrentStates(name, { tokens, states })).toBe(true);
    expect(await loadRecurrentStates(name, tokens)).toEqual(states);
    // A cache that goes on differently before the last state, or is shorter than it, does not own them.
    expect(await loadRecurrentStates(name, [4, 5, 6, 7, 1, 9])).toEqual([]);
    expect(await loadRecurrentStates(name, [4, 5, 6])).toEqual([]);
    // One that only differs after it does.
    expect(await loadRecurrentStates(name, [4, 5, 6, 7, 8, 1, 2])).toEqual(states);
    const file = `${name}.state`;
    files.set(file, files.get(file)!.slice(0, -3)); // cut short
    expect(await loadRecurrentStates(name, tokens)).toEqual([]);
  });

  it("round-trips a cache whose layers partly keep no keys and values", async () => {
    const name = promptCacheName("org/mixed", {});
    const e = new FakeEngine([0, 12]);
    e.set([5, 6, 7, 8]);
    let state = await savePromptCache(name, e);
    e.set([5, 6, 7, 8, 9, 10]);
    state = await savePromptCache(name, e, state);
    const back = await loadPromptCache(name);
    expect(back?.snapshot.tokens).toEqual([5, 6, 7, 8, 9, 10]);
    expect(back?.snapshot).toEqual(e.snapshot());
    expect(back?.state).toEqual(state);
  });
});

describe("prompt cache of a model whose layers differ in head size", () => {
  // Gemma 4's shape: sliding layers of 32-wide heads, full-attention layers of 64, the last layers
  // reading earlier ones' keys and values. Only the K/V matter here, so the model has no layer weights.
  const cfg = configFromHF({
    model_type: "gemma4_text",
    num_hidden_layers: 10,
    hidden_size: 64,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_dim: 32,
    global_head_dim: 64,
    intermediate_size: 128,
    vocab_size: 96,
    sliding_window: 16,
    num_kv_shared_layers: 4,
    layer_types: ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
  } as never);
  const engine = (be: CpuBackend) => {
    const emb = be.alloc(96 * 64 * 4, "storage", "embed");
    const m = new LlamaModel(be, cfg, new Map([["model.embed_tokens.weight", { shape: [96, 64], dtype: "f32", buf: emb }]]), { maxSeq: 512, kvDtype: "f16" });
    return new Engine(be, m, Tokenizer.fromJSON({ model: { type: "BPE", vocab: { a: 1 }, merges: [] }, pre_tokenizer: { type: "ByteLevel" } as never } as never));
  };
  /**
   * Puts `n` tokens and their K/V in the engine's cache, as a load of a snapshot would. A row's bytes
   * depend on its layer, K or V, and position only, so a longer cache extends a shorter one.
   */
  const fill = (e: Engine, n: number) => {
    const s = e.cacheShape;
    const out: number[] = [];
    e.model.slots.forEach((slots, l) => {
      const held = Math.min(n, slots);
      s.layerRowBytes![l]!.forEach((bytes, part) => {
        for (let j = 0; j < held; j++) {
          // Slot j holds the last position written there.
          const p = j + Math.floor((n - 1 - j) / slots) * slots;
          for (let b = 0; b < bytes; b++) out.push((l * 31 + part * 17 + p * 7 + b) & 255);
        }
      });
    });
    const data = Uint8Array.from(out);
    expect(data.length).toBe(e.model.kvSnapshotBytes(n));
    const tokens = Array.from({ length: n }, (_, i) => 1 + ((i * 7) % 90));
    expect(e.importCache({ tokens, kvDtype: s.kvDtype, layers: s.layers, rowBytes: s.rowBytes, ...(s.valueRowBytes ? { valueRowBytes: s.valueRowBytes } : {}), layerRowBytes: s.layerRowBytes, data })).toBe(true);
  };

  it("saves, appends and restores the cache", async () => {
    const be = new CpuBackend({ threads: 1 });
    const e = engine(be);
    expect(e.cacheShape.layerRowBytes?.map(([k]) => k)).toEqual([64, 64, 64, 64, 128, 64, 64, 64, 64, 128]);
    const name = promptCacheName("org/gemma4", {});
    fill(e, 260);
    let state = await savePromptCache(name, e);
    expect(state?.tokens.length).toBe(260);
    // A turn that adds tokens appends a segment; the ring layers' rows wrap past their 272 slots.
    fill(e, 300);
    const after = await e.exportCache();
    state = await savePromptCache(name, e, state);
    expect(state?.segments.length).toBe(2);
    const back = await loadPromptCache(name);
    expect(back?.state).toEqual(state);
    const f = engine(be);
    expect(f.importCache(back!.snapshot)).toBe(true);
    expect((await f.exportCache()).data).toEqual(after.data);
    // Files of models whose layers are alike keep their old header: nothing per layer in it.
    expect(new FakeEngine().cacheShape).not.toHaveProperty("layerRowBytes");
  });

  it("does not take a snapshot made with other per-layer rows", async () => {
    const be = new CpuBackend({ threads: 1 });
    const e = engine(be);
    fill(e, 20);
    const snap = await e.exportCache();
    const f = engine(be);
    expect(f.importCache({ ...snap, layerRowBytes: snap.layerRowBytes!.map(([k, v], l) => (l === 4 ? [k / 2, v / 2] : [k, v])) })).toBe(false);
    expect(f.importCache({ ...snap, layerRowBytes: undefined })).toBe(false);
    expect(f.importCache(snap)).toBe(true);
  });
});
