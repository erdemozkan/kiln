import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadPromptCache, promptCacheName, savePromptCache } from "../src/prompt-cache";

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
      getFile: async () => ({ arrayBuffer: async () => files.get(name)!.slice().buffer }),
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

const snap = { tokens: [5, 6, 7], kvDtype: "f16", layers: 2, rowBytes: 4, data: Uint8Array.from({ length: 2 * 2 * 3 * 4 }, (_, i) => i) };

describe("prompt cache", () => {
  it("round-trips a snapshot", async () => {
    const name = promptCacheName("org/model", { quant: "q4" });
    expect(await savePromptCache(name, snap)).toBe(true);
    expect(await loadPromptCache(name)).toEqual(snap);
  });

  it("keys on everything that changes the cached values", () => {
    const base = promptCacheName("org/model", {});
    expect(promptCacheName("org/model", { revision: "main", quant: "q4" })).toBe(base);
    for (const other of [promptCacheName("org/other", {}), promptCacheName("org/model", { revision: "v2" }), promptCacheName("org/model", { quant: "f32" }), promptCacheName("org/model", { f16Math: true })]) {
      expect(other).not.toBe(base);
    }
  });

  it("ignores missing, truncated and foreign files", async () => {
    const name = promptCacheName("org/model", {});
    expect(await loadPromptCache(name)).toBeUndefined();
    await savePromptCache(name, snap);
    files.set(name, files.get(name)!.slice(0, -5)); // a worker killed mid-write
    expect(await loadPromptCache(name)).toBeUndefined();
    files.set(name, new TextEncoder().encode("not a cache file at all"));
    expect(await loadPromptCache(name)).toBeUndefined();
  });
});
