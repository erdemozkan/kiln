import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CachedHttpSource } from "../src";

/** In-memory OPFS: just enough of the directory, file and sync-handle API for the cached source. */
class FakeHandle {
  /** Simulated storage failures: a read that throws or returns a wrapped error code, a write that stops landing. */
  static readFails: "throw" | number | undefined;
  static writeFailsAt: number | undefined;
  private open = true;
  constructor(private readonly file: { bytes: Uint8Array }) {}
  read(buf: Uint8Array, { at }: { at: number }): number {
    if (FakeHandle.readFails === "throw") throw new Error("InvalidStateError: Failed to read the content");
    if (FakeHandle.readFails !== undefined) return FakeHandle.readFails;
    const n = Math.max(0, Math.min(buf.length, this.file.bytes.length - at));
    buf.set(this.file.bytes.subarray(at, at + n));
    return n;
  }
  write(buf: Uint8Array, { at }: { at: number }): number {
    if (!this.open) throw new Error("InvalidStateError: handle closed");
    if (FakeHandle.writeFailsAt !== undefined && at >= FakeHandle.writeFailsAt) return 0;
    if (at + buf.length > this.file.bytes.length) this.resize(at + buf.length);
    this.file.bytes.set(buf, at);
    return buf.length;
  }
  getSize(): number {
    return this.file.bytes.length;
  }
  truncate(size: number): void {
    this.resize(size);
  }
  flush(): void {}
  close(): void {
    this.open = false;
  }
  private resize(n: number) {
    const b = new Uint8Array(n);
    b.set(this.file.bytes.subarray(0, n));
    this.file.bytes = b;
  }
}

const files = new Map<string, { bytes: Uint8Array }>();
const fileHandle = (name: string) => ({
  createSyncAccessHandle: async () => new FakeHandle(files.get(name)!),
  getFile: async () => ({ text: async () => new TextDecoder().decode(files.get(name)!.bytes) }),
});
const dir = {
  getFileHandle: async (name: string, opts?: { create?: boolean }) => {
    if (!files.has(name)) {
      if (!opts?.create) throw new Error("NotFoundError");
      files.set(name, { bytes: new Uint8Array(0) });
    }
    return fileHandle(name);
  },
  getDirectoryHandle: async () => dir,
};

const BODY = Uint8Array.from({ length: 300 }, (_, i) => (i * 7 + 3) & 255);
const requests: string[] = [];

/** Serves BODY with Range support, in 64-byte chunks. */
function fakeFetch(_url: string, init?: { headers?: Record<string, string> }) {
  const range = init?.headers?.Range ?? "";
  requests.push(range);
  const m = /bytes=(\d+)-(\d*)/.exec(range);
  const from = m ? Number(m[1]) : 0;
  const to = m?.[2] ? Number(m[2]) + 1 : BODY.length;
  const part = BODY.subarray(from, to);
  let pos = 0;
  return Promise.resolve({
    status: m ? 206 : 200,
    ok: true,
    headers: { get: (h: string) => (h === "Content-Range" ? `bytes ${from}-${to - 1}/${BODY.length}` : String(part.length)) },
    arrayBuffer: async () => part.slice().buffer,
    body: {
      getReader: () => ({
        read: async () => (pos >= part.length ? { done: true } : { done: false, value: part.slice(pos, (pos += 64)) }),
        cancel: async () => void (pos = part.length),
      }),
      cancel: async () => undefined,
    },
  });
}

const KEY = "repo@main/model.safetensors";
const NAME = encodeURIComponent(KEY);
const setMeta = (meta: object) => files.set(`${NAME}.meta.json`, { bytes: new TextEncoder().encode(JSON.stringify(meta)) });
const getMeta = () => JSON.parse(new TextDecoder().decode(files.get(`${NAME}.meta.json`)!.bytes));

async function readAll(src: { read(o: number, n: number): Promise<Uint8Array>; close?(): void }) {
  const out = await src.read(0, BODY.length);
  await new Promise((r) => setTimeout(r, 0)); // let the completion metadata land
  src.close?.();
  return out;
}

beforeEach(() => {
  files.clear();
  requests.length = 0;
  FakeHandle.readFails = undefined;
  FakeHandle.writeFailsAt = undefined;
  vi.stubGlobal("FileSystemSyncAccessHandle", FakeHandle);
  vi.stubGlobal("navigator", { storage: { getDirectory: async () => dir } });
  vi.stubGlobal("fetch", fakeFetch);
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CachedHttpSource", () => {
  it("downloads, then serves a complete cache without the network", async () => {
    expect(await readAll(await CachedHttpSource.open("u", KEY))).toEqual(BODY);
    expect(getMeta()).toMatchObject({ complete: true, flushed: BODY.length });
    requests.length = 0;
    expect(await readAll(await CachedHttpSource.open("u", KEY))).toEqual(BODY);
    expect(requests).toEqual([]);
  });

  it("cuts off bytes past the flush watermark before resuming", async () => {
    // A killed worker: 40 bytes confirmed, the rest of a full-size file is garbage.
    const stale = new Uint8Array(BODY.length);
    stale.set(BODY.subarray(0, 40));
    files.set(NAME, { bytes: stale });
    setMeta({ size: BODY.length, complete: false, flushed: 40 });
    const src = await CachedHttpSource.open("u", KEY);
    expect(files.get(NAME)!.bytes.length).toBeLessThanOrEqual(BODY.length);
    expect(requests.at(-1)).toBe("bytes=40-");
    expect(await readAll(src)).toEqual(BODY);
  });

  it("does not take a full-size file with an unconfirmed tail for a complete one", async () => {
    // Before the fix, an interrupted resume left the file at full size; once anything marked the
    // metadata complete, the zeros were served as model bytes.
    files.set(NAME, { bytes: new Uint8Array(BODY.length) });
    setMeta({ size: BODY.length, complete: false, flushed: 0 });
    const src = await CachedHttpSource.open("u", KEY);
    expect(files.get(NAME)!.bytes.length).toBe(0);
    expect(await readAll(src)).toEqual(BODY);
  });

  it("fresh re-downloads a cache marked complete whose bytes are wrong", async () => {
    files.set(NAME, { bytes: new Uint8Array(BODY.length) });
    setMeta({ size: BODY.length, complete: true, flushed: BODY.length });
    expect(await readAll(await CachedHttpSource.open("u", KEY))).not.toEqual(BODY);
    expect(await readAll(await CachedHttpSource.open("u", KEY, { fresh: true }))).toEqual(BODY);
    expect(getMeta()).toMatchObject({ complete: true });
  });

  it.each([["returns a wrapped error code", 4294967292], ["throws", "throw" as const]])(
    "a cached read that %s is reported as corrupt and queued for download",
    async (_, failure) => {
      await readAll(await CachedHttpSource.open("u", KEY));
      FakeHandle.readFails = failure;
      const src = await CachedHttpSource.open("u", KEY);
      await expect(src.read(0, 8)).rejects.toMatchObject({ name: "CorruptCacheError" });
      await new Promise((r) => setTimeout(r, 0));
      src.close?.();
      expect(getMeta()).toMatchObject({ complete: false, flushed: 0 });
      FakeHandle.readFails = undefined;
      expect(await readAll(await CachedHttpSource.open("u", KEY))).toEqual(BODY);
    },
  );

  it("a write that does not land switches the rest of the file to the network instead of caching a hole", async () => {
    FakeHandle.writeFailsAt = 128;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const src = await CachedHttpSource.open("u", KEY);
    expect(await src.read(200, 50)).toEqual(BODY.subarray(200, 250));
    expect(await src.read(0, 64)).toEqual(BODY.subarray(0, 64)); // cached part, no request
    expect(requests).toEqual(["bytes=0-0", "bytes=0-", "bytes=200-249"]); // size probe, download, network read
    expect(await src.read(0, BODY.length)).toEqual(BODY);
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/cache write of 64 bytes at 128 returned 0/));
    src.close?.();
    expect(getMeta()).toMatchObject({ complete: false, flushed: 128 });
    // The next open resumes after the cached part.
    FakeHandle.writeFailsAt = undefined;
    requests.length = 0;
    expect(await readAll(await CachedHttpSource.open("u", KEY))).toEqual(BODY);
    expect(requests).toEqual(["bytes=128-"]);
    warn.mockRestore();
  });

  describe("segments", () => {
    const seg = { segment: 100 };
    const sizes = () => [NAME, `${NAME}.part1`, `${NAME}.part2`].map((n) => files.get(n)?.bytes.length);

    it("stores a file larger than one segment across segment files", async () => {
      const src = await CachedHttpSource.open("u", KEY, seg);
      expect(await src.read(90, 120)).toEqual(BODY.subarray(90, 210)); // spans three segments
      expect(await readAll(src)).toEqual(BODY);
      expect(sizes()).toEqual([100, 100, 100]);
      expect(getMeta()).toMatchObject({ complete: true, flushed: BODY.length });
      requests.length = 0;
      expect(await readAll(await CachedHttpSource.open("u", KEY, seg))).toEqual(BODY);
      expect(requests).toEqual([]);
    });

    it("resumes inside a later segment and drops unconfirmed bytes in every segment", async () => {
      files.set(NAME, { bytes: BODY.slice(0, 100) });
      files.set(`${NAME}.part1`, { bytes: BODY.slice(100, 200) });
      files.set(`${NAME}.part2`, { bytes: new Uint8Array(100) }); // garbage past the watermark
      setMeta({ size: BODY.length, complete: false, flushed: 150 });
      const src = await CachedHttpSource.open("u", KEY, seg);
      expect(requests.at(-1)).toBe("bytes=150-");
      expect(await readAll(src)).toEqual(BODY);
      expect(sizes()).toEqual([100, 100, 100]);
    });

    it("does not count bytes after a short segment as cached", async () => {
      // A hole: segment 0 is short, so segment 1's bytes cannot be placed and are cut.
      files.set(NAME, { bytes: BODY.slice(0, 60) });
      files.set(`${NAME}.part1`, { bytes: BODY.slice(100, 200) });
      setMeta({ size: BODY.length, complete: false, flushed: 200 });
      const src = await CachedHttpSource.open("u", KEY, seg);
      expect(requests.at(-1)).toBe("bytes=60-");
      expect(await readAll(src)).toEqual(BODY);
    });

    it("re-downloads a file cached as one oversized segment", async () => {
      files.set(NAME, { bytes: BODY.slice() });
      setMeta({ size: BODY.length, complete: true, flushed: BODY.length });
      const src = await CachedHttpSource.open("u", KEY, seg);
      expect(requests.at(-1)).toBe("bytes=0-");
      expect(await readAll(src)).toEqual(BODY);
      expect(sizes()).toEqual([100, 100, 100]);
    });
  });

  it("closing twice is harmless", async () => {
    const src = await CachedHttpSource.open("u", KEY);
    await readAll(src);
    expect(() => src.close?.()).not.toThrow();
  });
});
