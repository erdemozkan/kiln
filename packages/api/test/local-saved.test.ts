import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { localModel, savedLocalModels } from "../src/local";

/** The slice of IndexedDB the handle store uses: one object store, requests that settle on a later tick. */
function fakeIndexedDB(): IDBFactory {
  const data = new Map<string, unknown>();
  const request = <T>(value: () => T) => {
    const req = { result: undefined as T, error: null, onsuccess: null as null | (() => void), onerror: null as null | (() => void) };
    queueMicrotask(() => {
      req.result = value();
      req.onsuccess?.();
    });
    return req;
  };
  const store = {
    put: (v: unknown, k: string) => request(() => void data.set(k, v)),
    delete: (k: string) => request(() => void data.delete(k)),
    getAllKeys: () => request(() => [...data.keys()]),
    getAll: () => request(() => [...data.values()]),
  };
  const db = { createObjectStore: () => store, transaction: () => ({ objectStore: () => store }), close: () => undefined };
  return {
    open: () => {
      const req = request(() => db) as ReturnType<typeof request> & { onupgradeneeded?: () => void };
      return req;
    },
  } as unknown as IDBFactory;
}

const fileHandle = (name: string, permission: { state: string; asked: number }) => ({
  kind: "file" as const,
  name,
  getFile: async () => new File(["x"], name, { lastModified: 1 }),
  queryPermission: async () => permission.state,
  requestPermission: async () => (permission.asked++, (permission.state = permission.state === "prompt" ? "granted" : permission.state)),
});

describe("remembered local models", () => {
  const g = globalThis as { indexedDB?: IDBFactory };
  beforeEach(() => (g.indexedDB = fakeIndexedDB()));
  afterEach(() => delete g.indexedDB);

  it("saves a handle, lists it, reopens it and forgets it", async () => {
    const permission = { state: "granted", asked: 0 };
    const id = await localModel(fileHandle("tiny-Q4_K_M.gguf", permission) as unknown as FileSystemHandle, { remember: true });
    const saved = await savedLocalModels();
    expect(saved.map((s) => [s.name, s.kind])).toEqual([["tiny-Q4_K_M.gguf", "file"]]);
    // The same file gives the same id, so its caches are found again.
    expect(await saved[0]!.open()).toBe(id);
    expect(permission.asked).toBe(0);
    await saved[0]!.forget();
    expect(await savedLocalModels()).toEqual([]);
  });

  it("asks for permission when the browser wants a prompt, and gives up when it is denied", async () => {
    const permission = { state: "prompt", asked: 0 };
    await localModel(fileHandle("a.gguf", permission) as unknown as FileSystemHandle, { remember: true, name: "mine" });
    const [saved] = await savedLocalModels();
    expect(saved!.name).toBe("mine");
    expect(await saved!.open()).toMatch(/^@local\/mine-/);
    expect(permission.asked).toBe(1);
    permission.state = "denied";
    expect(await saved!.open()).toBeUndefined();
  });

  it("keeps a copy of picked files where there are no handles, and reopens it to the same id", async () => {
    const files = [new File(["{}"], "config.json", { lastModified: 3 }), new File(["w"], "model.safetensors", { lastModified: 4 })].map((f) => Object.assign(f, { webkitRelativePath: `bert-tiny/${f.name}` }));
    const id = await localModel(files, { remember: "copy" });
    const [saved] = await savedLocalModels();
    expect([saved!.name, saved!.kind, saved!.copied]).toEqual(["bert-tiny", "directory", true]);
    expect(await saved!.open()).toBe(id);
    const gguf = await localModel(new File(["x"], "tiny.gguf", { lastModified: 1 }), { remember: "copy" });
    const again = (await savedLocalModels()).find((s) => s.name === "tiny")!;
    expect([again.kind, await again.open()]).toEqual(["file", gguf]);
    await saved!.forget();
    expect((await savedLocalModels()).map((s) => s.name)).toEqual(["tiny"]);
  });

  it("cannot remember what is not a handle, and lists nothing without IndexedDB", async () => {
    await expect(localModel(new File(["x"], "a.gguf"), { remember: true })).rejects.toThrow("only a file or directory handle");
    delete g.indexedDB;
    expect(await savedLocalModels()).toEqual([]);
  });
});
