import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CachedHttpSource } from "../src";

/** In-memory OPFS: just enough of the directory, file and sync-handle API for the cached source. */
class FakeHandle {
  private open = true;
  constructor(private readonly file: { bytes: Uint8Array }) {}
  read(buf: Uint8Array, { at }: { at: number }): number {
    const n = Math.max(0, Math.min(buf.length, this.file.bytes.length - at));
    buf.set(this.file.bytes.subarray(at, at + n));
    return n;
  }
  write(buf: Uint8Array, { at }: { at: number }): number {
    if (!this.open) throw new Error("InvalidStateError: handle closed");
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
    body: {
      getReader: () => ({
        read: async () => (pos >= part.length ? { done: true } : { done: false, value: part.slice(pos, (pos += 64)) }),
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

  it("closing twice is harmless", async () => {
    const src = await CachedHttpSource.open("u", KEY);
    await readAll(src);
    expect(() => src.close?.()).not.toThrow();
  });
});
