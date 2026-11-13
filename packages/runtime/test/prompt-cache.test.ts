import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadPromptCache, promptCacheName, savePromptCache, type PromptCacheSource } from "../src/prompt-cache";

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
  readonly cacheShape = { kvDtype: "f16", layers: 2, rowBytes: 2, slots: [12, 3] };
  cacheTokens: number[] = [];
  reads = 0;
  private k = this.cacheShape.slots.map((s) => new Uint8Array(s * 2));
  private v = this.cacheShape.slots.map((s) => new Uint8Array(s * 2));
  set(tokens: number[]) {
    this.cacheTokens = tokens;
    tokens.forEach((t, p) =>
      this.cacheShape.slots.forEach((slots, l) => {
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
});
