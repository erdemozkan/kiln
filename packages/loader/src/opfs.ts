import { HttpSource, parseContentRangeTotal, type ByteSource, checkRange } from "./source";

/** Minimal typing for the worker-only OPFS sync handle (not in lib.dom). */
interface SyncHandle {
  read(buffer: Uint8Array, opts: { at: number }): number;
  write(buffer: Uint8Array, opts: { at: number }): number;
  getSize(): number;
  truncate(size: number): void;
  flush(): void;
  close(): void;
}
type SyncFileHandle = FileSystemFileHandle & { createSyncAccessHandle(): Promise<SyncHandle> };

const DIR = "kiln-cache";
/** Bytes written between flushes; the resume point only advances past flushed bytes. */
const FLUSH_EVERY = 32 * 2 ** 20;
/**
 * Largest single OPFS file. Some stores cap one file (Chrome's in-memory storage, used by embedded
 * browsers and incognito, fails writes at 2000 MiB) while holding more in total, so a cached file is
 * kept as segments of this size: `name`, `name.part1`, `name.part2`, ...
 */
const SEGMENT = 2 ** 30;

const partName = (name: string, i: number) => (i === 0 ? name : `${name}.part${i}`);

/** One logical file over segment files of `seg` bytes each (the last may be shorter). */
class Segments implements SyncHandle {
  constructor(
    private readonly parts: SyncHandle[],
    private readonly seg: number,
  ) {}

  /** Splits [at, at + n) by segment; stops and returns the first count that is not the full piece. */
  private each(at: number, n: number, op: (p: SyncHandle, from: number, len: number, off: number) => number): number {
    let done = 0;
    while (done < n) {
      const i = Math.floor((at + done) / this.seg);
      const off = at + done - i * this.seg;
      const len = Math.min(n - done, this.seg - off);
      const p = this.parts[i];
      if (!p) return done;
      const got = op(p, done, len, off);
      if (got !== len) return got;
      done += len;
    }
    return done;
  }

  read(buffer: Uint8Array, { at }: { at: number }): number {
    return this.each(at, buffer.length, (p, from, len, off) => p.read(buffer.subarray(from, from + len), { at: off }));
  }

  write(buffer: Uint8Array, { at }: { at: number }): number {
    return this.each(at, buffer.length, (p, from, len, off) => p.write(buffer.subarray(from, from + len), { at: off }));
  }

  /** Bytes up to the first segment that is not full: a later segment's bytes have no defined place. */
  getSize(): number {
    let total = 0;
    for (const p of this.parts) {
      const n = p.getSize();
      total += Math.min(n, this.seg);
      if (n < this.seg) break;
    }
    return total;
  }

  truncate(size: number): void {
    this.parts.forEach((p, i) => p.truncate(Math.max(0, Math.min(this.seg, size - i * this.seg))));
  }

  flush(): void {
    for (const p of this.parts) p.flush();
  }

  close(): void {
    for (const p of this.parts) p.close();
  }
}

/** True inside a dedicated worker with OPFS sync access (the only place the cached source works). */
export function opfsAvailable(): boolean {
  const g = globalThis as { FileSystemSyncAccessHandle?: unknown; navigator?: Navigator };
  return typeof g.FileSystemSyncAccessHandle === "function" && !!g.navigator?.storage?.getDirectory;
}

/**
 * The download *is* the cache: one streaming fetch writes into an OPFS file, and reads are served
 * from that file as soon as the bytes they need have landed. An interrupted download resumes from
 * the last flushed byte recorded in the metadata, never from the file size: a killed worker can leave
 * the file longer than the bytes that actually reached disk, so that tail is cut off before resuming.
 * Peak memory is one network chunk.
 */
/** How far past the downloaded bytes a read must start to be fetched on its own instead of awaited. */
const READ_AHEAD = 64 * 2 ** 20;
/** Reads up to this size are fetched on their own once they start this far past the download. */
const SMALL_READ = 2 ** 20;

export class CachedHttpSource implements ByteSource {
  private have = 0;
  private complete = false;
  private failure: Error | undefined;
  private waiters: (() => void)[] = [];
  private closed = false;
  /** Set once the cache stopped taking bytes: reads past `have` then go straight to the network. */
  private spill: HttpSource | undefined;
  /** Range reads of bytes far past the download (a zip's directory at the end of pytorch_model.bin). */
  private ahead: HttpSource | undefined;
  private url = "";
  private readAhead = READ_AHEAD;

  private constructor(
    readonly name: string,
    private readonly handle: SyncHandle,
    private readonly total: number,
    private readonly saveMeta: (meta: Meta) => Promise<void>,
    private readonly retryMs: number,
  ) {}

  /**
   * `key` should pin the revision (e.g. repo@commit/file) so a new upload never reuses stale bytes.
   * `fresh` discards whatever is cached and downloads from the start.
   */
  static async open(
    url: string,
    key: string,
    { fresh = false, segment = SEGMENT, retryMs = 1000, readAhead = READ_AHEAD }: { fresh?: boolean; segment?: number; retryMs?: number; readAhead?: number } = {},
  ): Promise<ByteSource> {
    if (!opfsAvailable()) return new HttpSource(url);
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(DIR, { create: true });
    const fileName = encodeURIComponent(key);
    const meta = fresh ? undefined : await readMeta(dir, fileName);
    const total = meta?.size ?? (await new HttpSource(url).size());
    const parts: SyncHandle[] = [];
    try {
      for (let i = 0; i < Math.max(1, Math.ceil(total / segment)); i++) {
        const file = (await dir.getFileHandle(partName(fileName, i), { create: true })) as SyncFileHandle;
        parts.push(await file.createSyncAccessHandle());
      }
    } catch (err) {
      for (const p of parts) p.close();
      if ((err as { name?: string }).name === "NoModificationAllowedError") {
        throw new Error(`${key} is already open in another tab or worker; close it and try again`);
      }
      throw err;
    }
    // A file cached before segments existed is one oversized segment; its bytes are re-downloaded.
    if (parts[0]!.getSize() > segment) parts[0]!.truncate(0);
    const handle = new Segments(parts, segment);
    const src = new CachedHttpSource(key, handle, total, (m) => writeMeta(dir, fileName, m), retryMs);
    const onDisk = Math.min(handle.getSize(), total);
    if (meta?.complete && onDisk === total) {
      src.have = total;
      src.complete = true;
    } else {
      src.have = Math.min(onDisk, meta?.complete ? 0 : meta?.flushed ?? 0);
      // Nothing past the resume point is trusted. Leaving it would let the file reach full size with
      // bytes that were never confirmed, and a later open would take that for a complete download.
      handle.truncate(src.have);
      handle.flush();
      await writeMeta(dir, fileName, { size: total, complete: false, flushed: src.have });
      src.url = url;
      src.readAhead = readAhead;
      void src.download(url);
    }
    return src;
  }

  async size(): Promise<number> {
    return this.total;
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    checkRange(this.name, offset, length);
    if (offset + length > this.total) throw new RangeError(`${this.name}: read past end`);
    // Ahead of the download: fetch just those bytes rather than wait for everything before them, for
    // small reads anywhere past it and large ones far past it (a pytorch_model.bin is a zip whose
    // directory sits at the end and whose every entry has a small header to read first).
    const gap = length <= SMALL_READ ? Math.min(SMALL_READ, this.readAhead) : this.readAhead;
    if (!this.complete && !this.spill && this.url && offset >= this.have + gap) {
      return (this.ahead ??= new HttpSource(this.url, this.name, this.retryMs)).read(offset, length);
    }
    while (!this.complete && !this.spill && this.have < offset + length) {
      if (this.failure) throw this.failure;
      await new Promise<void>((r) => this.waiters.push(r));
    }
    if (!this.complete && this.spill && this.have < offset + length) return this.spill.read(offset, length);
    const out = new Uint8Array(length);
    // A short read means the bytes are not on disk after all; returning the zero-filled buffer would
    // hand the parser garbage (a safetensors header length of 0, say). Chrome reports a failed read
    // either as an error code wrapped to a huge count (4294967292 for -4, seen with in-memory storage)
    // or, in newer versions, by throwing.
    let n: number | string;
    try {
      n = this.handle.read(out, { at: offset });
    } catch (err) {
      n = String(err);
    }
    if (n !== length) {
      // Marked incomplete, so the next open downloads the file again instead of failing the same way.
      void this.saveMeta({ size: this.total, complete: false, flushed: 0 }).catch(() => undefined);
      throw new CorruptCacheError(`${this.name}: cached read of ${length} bytes at ${offset} returned ${n}`);
    }
    return out;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.handle.flush();
    this.handle.close();
  }

  /** True once the background download has failed (reads past the downloaded bytes will throw). */
  get failed(): boolean {
    return this.failure !== undefined;
  }

  /** Bytes downloaded so far (for progress bars). */
  get downloaded(): number {
    return this.have;
  }

  /**
   * Downloads the rest of the file. A dropped connection or a server error (5xx, 429) is retried from
   * the bytes already written, after 1, 2, 4, ... seconds; the count resets whenever a retry makes
   * progress, so a slow, flaky link still finishes. Other failures end the download.
   */
  private async download(url: string): Promise<void> {
    let failures = 0;
    for (;;) {
      const before = this.have;
      try {
        if ((await this.fetchRest(url)) !== "retry") break;
      } catch (err) {
        const e = err instanceof Error ? err : new Error(String(err));
        if (!(e instanceof RetryableError) && e.name !== "TypeError") {
          this.failure = e;
          break;
        }
        if (this.have > before) failures = 0;
        if (++failures > MAX_RETRIES || this.closed) {
          this.failure = e;
          break;
        }
        console.warn(`[kiln] ${this.name}: ${e.message}; retrying from byte ${this.have}`);
        await new Promise((r) => setTimeout(r, this.retryMs * 2 ** (failures - 1)));
        if (this.closed) break;
        continue;
      }
    }
    this.wake();
  }

  /** One request for the bytes after `have`. Returns once the file is complete or caching stopped. */
  private async fetchRest(url: string): Promise<"done" | "retry"> {
    const res = await fetch(url, { headers: { Range: `bytes=${this.have}-` } });
    if (res.status === 200 && this.have > 0) this.have = 0; // server ignored Range: restart
    else if (res.status >= 500 || res.status === 429) throw new RetryableError(`${url}: HTTP ${res.status}`);
    else if (res.status !== 206 && res.status !== 200) throw new Error(`${url}: HTTP ${res.status}`);
    const total = parseContentRangeTotal(res.headers.get("Content-Range"));
    if (total !== undefined && total !== this.total) throw new Error(`${url}: size changed (${this.total} → ${total})`);
    const reader = res.body!.getReader();
    let flushed = this.have;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (this.closed) {
        await reader.cancel().catch(() => undefined);
        return "done";
      }
      // Same failure reporting as reads: a short or wrapped count means the bytes did not land.
      const n = this.handle.write(value, { at: this.have });
      if (n !== value.byteLength) {
        await reader.cancel().catch(() => undefined);
        await this.stopCaching(url, `cache write of ${value.byteLength} bytes at ${this.have} returned ${n}`);
        return "done";
      }
      this.have += value.byteLength;
      // The last chunk wakes readers only after completion is recorded: the final reader's caller
      // may close the handle right away, and then the file would never be marked complete.
      if (this.have >= this.total) continue;
      this.wake();
      if (this.have - flushed >= FLUSH_EVERY) {
        // Data first, then the watermark, so the metadata never points past durable bytes.
        this.handle.flush();
        flushed = this.have;
        await this.saveMeta({ size: this.total, complete: false, flushed });
      }
    }
    if (this.have !== this.total) throw new RetryableError(`${url}: download ended at ${this.have} of ${this.total} bytes`);
    this.handle.flush();
    this.complete = true;
    await this.saveMeta({ size: this.total, complete: true, flushed: this.total });
    return "done";
  }

  /**
   * The store took no more bytes (full, or a per-file size limit; Chrome's in-memory storage stops at
   * 2000 MiB). The cached part keeps serving reads, the rest is read over HTTP, and the next open
   * resumes from the cached part.
   */
  private async stopCaching(url: string, why: string): Promise<void> {
    console.warn(`[kiln] ${this.name}: ${why} (storage full?); reading the rest of the file from the network`);
    try {
      this.handle.flush();
      await this.saveMeta({ size: this.total, complete: false, flushed: this.have });
    } catch {
      // The metadata keeps an older watermark, which is still safe.
    }
    this.spill = new HttpSource(url, this.name);
    this.wake();
  }

  private wake(): void {
    const w = this.waiters;
    this.waiters = [];
    for (const r of w) r();
  }
}

/** A download failure worth retrying: a server error or a body that ended early. */
class RetryableError extends Error {}

/** Consecutive failed attempts (without progress) before a download gives up. */
const MAX_RETRIES = 5;

/** The cached bytes are inconsistent with their metadata; opening with `fresh` downloads them again. */
export class CorruptCacheError extends Error {
  override name = "CorruptCacheError";
}

interface Meta {
  size: number;
  complete: boolean;
  /** Bytes known to be flushed to disk; resume point. Missing in old metadata, which then restarts. */
  flushed?: number;
}

async function readMeta(dir: FileSystemDirectoryHandle, name: string): Promise<Meta | undefined> {
  try {
    const f = await dir.getFileHandle(`${name}.meta.json`);
    return JSON.parse(await (await f.getFile()).text()) as Meta;
  } catch {
    return undefined;
  }
}

async function writeMeta(dir: FileSystemDirectoryHandle, name: string, meta: Meta): Promise<void> {
  const f = (await dir.getFileHandle(`${name}.meta.json`, { create: true })) as SyncFileHandle;
  const h = await f.createSyncAccessHandle();
  const bytes = new TextEncoder().encode(JSON.stringify(meta));
  h.truncate(0);
  h.write(bytes, { at: 0 });
  h.flush();
  h.close();
}

/** Reads a small JSON document from the cache directory (undefined if absent or unreadable). */
export async function readCacheJSON<T>(name: string): Promise<T | undefined> {
  try {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(DIR);
    return JSON.parse(await (await (await dir.getFileHandle(name)).getFile()).text()) as T;
  } catch {
    return undefined;
  }
}

/** Writes a small JSON document to the cache directory. Worker-only (sync handle); failures are ignored. */
export async function writeCacheJSON(name: string, value: unknown): Promise<void> {
  if (!opfsAvailable()) return;
  try {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(DIR, { create: true });
    const h = await ((await dir.getFileHandle(name, { create: true })) as SyncFileHandle).createSyncAccessHandle();
    const bytes = new TextEncoder().encode(JSON.stringify(value));
    h.truncate(0);
    h.write(bytes, { at: 0 });
    h.flush();
    h.close();
  } catch {
    // A cache miss next time is harmless.
  }
}

/** Reads a file from the cache directory (undefined if absent or unreadable). */
export async function readCacheFile(name: string): Promise<Uint8Array | undefined> {
  try {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(DIR);
    return new Uint8Array(await (await (await dir.getFileHandle(name)).getFile()).arrayBuffer());
  } catch {
    return undefined;
  }
}

/**
 * Writes a file to the cache directory from parts, replacing it. Worker-only (sync handle); returns
 * false on failure (a full store, say), which a cache can ignore. A short write leaves the file empty
 * rather than truncated, so readers never see half of it.
 */
export async function writeCacheFile(name: string, parts: readonly Uint8Array[]): Promise<boolean> {
  if (!opfsAvailable()) return false;
  let h: SyncHandle | undefined;
  try {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(DIR, { create: true });
    h = await ((await dir.getFileHandle(name, { create: true })) as SyncFileHandle).createSyncAccessHandle();
    h.truncate(0);
    let at = 0;
    for (const p of parts) {
      if (h.write(p, { at }) !== p.length) throw new Error("short write");
      at += p.length;
    }
    h.flush();
    return true;
  } catch {
    try {
      h?.truncate(0);
    } catch {
      // Nothing more to do; the reader rejects what does not parse.
    }
    return false;
  } finally {
    h?.close();
  }
}

/** Deletes one cached file, its segments and its metadata. Silently keeps a file another tab or worker has open. */
export async function evictCached(key: string): Promise<void> {
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle(DIR, { create: true });
  const name = encodeURIComponent(key);
  for (const n of [name, `${name}.meta.json`]) await dir.removeEntry(n).catch(() => undefined);
  for (let i = 1; ; i++) {
    const gone = await dir.removeEntry(partName(name, i)).then(
      () => true,
      (err: unknown) => (err as { name?: string }).name !== "NotFoundError",
    );
    if (!gone) break;
  }
}

/** One cached model file, as `listCache` reports it. */
export interface CachedFile {
  /** The cache key (repo@revision/file). */
  key: string;
  /** Bytes on disk, all segments together. */
  bytes: number;
  /** Full size of the file, from its metadata (undefined for a small JSON or tuning entry). */
  size?: number;
  complete: boolean;
}

/** Lists the cache directory: model files with their segments summed, plus small entries. Works outside workers. */
export async function listCache(): Promise<CachedFile[]> {
  let dir: FileSystemDirectoryHandle;
  try {
    dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(DIR);
  } catch {
    return [];
  }
  const sizes = new Map<string, number>();
  const metas = new Map<string, Meta>();
  for await (const [n, h] of (dir as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries()) {
    if (h.kind !== "file") continue;
    const file = await (h as FileSystemFileHandle).getFile();
    if (n.endsWith(".meta.json")) {
      try {
        metas.set(n.slice(0, -".meta.json".length), JSON.parse(await file.text()) as Meta);
      } catch {
        // An unreadable watermark only means the file is listed as incomplete.
      }
      continue;
    }
    const base = n.replace(/\.part\d+$/, "");
    sizes.set(base, (sizes.get(base) ?? 0) + file.size);
  }
  return [...sizes].map(([n, bytes]) => {
    const meta = metas.get(n);
    return { key: decodeURIComponent(n), bytes, size: meta?.size, complete: meta ? meta.complete : true };
  });
}

/** Deletes every cached model file. */
export async function clearCache(): Promise<void> {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(DIR, { recursive: true }).catch(() => undefined);
}
