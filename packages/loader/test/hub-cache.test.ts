import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/** In-memory OPFS: the directory, file and sync-handle calls the small-file cache uses. */
const disk = new Map<string, { bytes: Uint8Array }>();
class FakeHandle {
  constructor(private readonly file: { bytes: Uint8Array }) {}
  write(buf: Uint8Array, { at }: { at: number }): number {
    if (at + buf.length > this.file.bytes.length) this.truncate(at + buf.length);
    this.file.bytes.set(buf, at);
    return buf.length;
  }
  truncate(n: number): void {
    const b = new Uint8Array(n);
    b.set(this.file.bytes.subarray(0, n));
    this.file.bytes = b;
  }
  flush(): void {}
  close(): void {}
}
const dir = {
  getDirectoryHandle: async () => dir,
  getFileHandle: async (name: string, o?: { create?: boolean }) => {
    if (!disk.has(name)) {
      if (!o?.create) throw new Error("NotFoundError");
      disk.set(name, { bytes: new Uint8Array(0) });
    }
    const file = disk.get(name)!;
    return { createSyncAccessHandle: async () => new FakeHandle(file), getFile: async () => ({ arrayBuffer: async () => file.bytes.slice().buffer }) };
  },
};

let routes: Record<string, string | number | Error> = {};
const requests: string[] = [];
const settle = () => new Promise((r) => setTimeout(r, 5));
/** A fresh copy of the loader: what a new page or worker sees, with the cache as the last one left it. */
const session = async () => {
  await settle();
  vi.resetModules();
  requests.length = 0;
  return import("../src");
};

beforeEach(() => {
  disk.clear();
  routes = {};
  vi.stubGlobal("FileSystemSyncAccessHandle", FakeHandle);
  vi.stubGlobal("navigator", { storage: { getDirectory: async () => dir } });
  vi.stubGlobal("fetch", async (url: string) => {
    requests.push(url);
    const hit = Object.entries(routes).find(([suffix]) => url.endsWith(suffix));
    if (!hit) return new Response("", { status: 404 });
    if (hit[1] instanceof Error) throw hit[1];
    return typeof hit[1] === "number" ? new Response("", { status: hit[1] }) : new Response(hit[1]);
  });
});
afterEach(() => {
  vi.unstubAllGlobals();
});

const url = "https://huggingface.co/org/model/resolve/main/config.json";

describe("small repo files are cached", () => {
  it("answers from the cache without the network, under the repo's cache key", async () => {
    routes = { "config.json": '{"a":1}' };
    let hub = await session();
    expect(await hub.fetchJSON(url)).toEqual({ a: 1 });
    await settle();
    expect([...disk.keys()]).toEqual([`${encodeURIComponent("org/model@main/config.json")}.hub`]);

    hub = await session();
    routes = { "config.json": new TypeError("Failed to fetch") };
    expect(await hub.fetchJSON(url)).toEqual({ a: 1 });
    expect(await hub.fetchText(url)).toBe('{"a":1}');
    expect(new TextDecoder().decode(await hub.fetchBytes(url))).toBe('{"a":1}');
    // One background comparison per session, however often the file is asked for.
    expect(requests).toEqual([url]);
  });

  it("picks up a changed file on the load after the one that noticed it", async () => {
    routes = { "config.json": '{"a":1}' };
    let hub = await session();
    await hub.fetchJSON(url);

    routes = { "config.json": '{"a":2}' };
    hub = await session();
    expect(await hub.fetchJSON(url)).toEqual({ a: 1 });

    hub = await session();
    expect(await hub.fetchJSON(url)).toEqual({ a: 2 });
  });

  it("keeps a missing optional file as missing, and notices when it appears", async () => {
    let hub = await session();
    expect(await hub.fetchJSON(url, true)).toBeUndefined();

    hub = await session();
    routes = { "config.json": '{"a":1}' };
    expect(await hub.fetchJSON(url, true)).toBeUndefined();

    hub = await session();
    expect(await hub.fetchJSON(url, true)).toEqual({ a: 1 });
  });

  it("does not store errors, or a 404 for a required file", async () => {
    routes = { "config.json": 401 };
    const hub = await session();
    await expect(hub.fetchJSON(url)).rejects.toThrow("HTTP 401 (gated or private model?)");
    routes = {};
    await expect(hub.fetchJSON(url)).rejects.toThrow("HTTP 404");
    await settle();
    expect(disk.size).toBe(0);
  });

  it("keeps the cached file when the server fails later", async () => {
    routes = { "config.json": '{"a":1}' };
    let hub = await session();
    await hub.fetchJSON(url);
    routes = { "config.json": 500 };
    hub = await session();
    expect(await hub.fetchJSON(url)).toEqual({ a: 1 });
    hub = await session();
    expect(await hub.fetchJSON(url)).toEqual({ a: 1 });
  });

  it("refetches an entry that was cut short", async () => {
    routes = { "config.json": '{"a":1}' };
    let hub = await session();
    await hub.fetchJSON(url);
    await settle();
    const file = [...disk.values()][0]!;
    file.bytes = file.bytes.subarray(0, file.bytes.length - 2);
    hub = await session();
    routes = { "config.json": '{"a":3}' };
    expect(await hub.fetchJSON(url)).toEqual({ a: 3 });
  });

  it("caches the file listing per server, and never a failed one", async () => {
    routes = { "siblings": JSON.stringify({ siblings: [{ rfilename: "config.json" }] }) };
    let hub = await session();
    const ref = hub.parseModelId("org/model");
    expect(await hub.listRepoFiles(ref)).toEqual(new Set(["config.json"]));
    expect(await hub.listRepoFiles(ref, "https://mirror.example")).toEqual(new Set(["config.json"]));
    await settle();
    expect([...disk.keys()].sort()).toEqual([`${encodeURIComponent("https://mirror.example|org/model@main/?siblings")}.hub`, `${encodeURIComponent("org/model@main/?siblings")}.hub`].sort());

    hub = await session();
    routes = { "siblings": 500 };
    expect(await hub.listRepoFiles(ref)).toEqual(new Set(["config.json"]));
    expect(await hub.listRepoFiles(hub.parseModelId("org/other"))).toBeUndefined();
    await settle();
    expect(disk.size).toBe(2);
  });

  it("works without a cache", async () => {
    vi.stubGlobal("navigator", {});
    routes = { "config.json": '{"a":1}' };
    const hub = await session();
    expect(await hub.fetchJSON(url)).toEqual({ a: 1 });
    expect(await hub.fetchJSON(url)).toEqual({ a: 1 });
    expect(requests).toHaveLength(2);
  });
});
