import { HttpSource, parseContentRangeTotal, type ByteSource } from "./source";

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

/** True inside a dedicated worker with OPFS sync access (the only place the cached source works). */
export function opfsAvailable(): boolean {
  const g = globalThis as { FileSystemSyncAccessHandle?: unknown; navigator?: Navigator };
  return typeof g.FileSystemSyncAccessHandle === "function" && !!g.navigator?.storage?.getDirectory;
}

/**
 * The download *is* the cache: one streaming fetch writes into an OPFS file, and reads are served
 * from that file as soon as the bytes they need have landed. An interrupted download resumes from
 * the last flushed byte recorded in the metadata, never from the file size: a killed worker can leave
 * the file longer than the bytes that actually reached disk. Peak memory is one network chunk.
 */
export class CachedHttpSource implements ByteSource {
  private have = 0;
  private complete = false;
  private failure: Error | undefined;
  private waiters: (() => void)[] = [];

  private constructor(
    readonly name: string,
    private readonly handle: SyncHandle,
    private readonly total: number,
    private readonly saveMeta: (meta: Meta) => Promise<void>,
  ) {}

  /** `key` should pin the revision (e.g. repo@commit/file) so a new upload never reuses stale bytes. */
  static async open(url: string, key: string): Promise<ByteSource> {
    if (!opfsAvailable()) return new HttpSource(url);
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(DIR, { create: true });
    const fileName = encodeURIComponent(key);
    const meta = await readMeta(dir, fileName);
    const file = (await dir.getFileHandle(fileName, { create: true })) as SyncFileHandle;
    const handle = await file.createSyncAccessHandle().catch((err: unknown) => {
      if ((err as { name?: string }).name === "NoModificationAllowedError") {
        throw new Error(`${key} is already open in another tab or worker; close it and try again`);
      }
      throw err;
    });
    const total = meta?.size ?? (await new HttpSource(url).size());
    const src = new CachedHttpSource(key, handle, total, (m) => writeMeta(dir, fileName, m));
    const onDisk = Math.min(handle.getSize(), total);
    if (meta?.complete && onDisk === total) {
      src.have = total;
      src.complete = true;
    } else {
      src.have = Math.min(onDisk, meta?.complete ? 0 : meta?.flushed ?? 0);
      await writeMeta(dir, fileName, { size: total, complete: false, flushed: src.have });
      void src.download(url);
    }
    return src;
  }

  async size(): Promise<number> {
    return this.total;
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    if (offset + length > this.total) throw new RangeError(`${this.name}: read past end`);
    while (!this.complete && this.have < offset + length) {
      if (this.failure) throw this.failure;
      await new Promise<void>((r) => this.waiters.push(r));
    }
    const out = new Uint8Array(length);
    this.handle.read(out, { at: offset });
    return out;
  }

  close(): void {
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

  private async download(url: string): Promise<void> {
    try {
      const res = await fetch(url, { headers: { Range: `bytes=${this.have}-` } });
      if (res.status === 200 && this.have > 0) this.have = 0; // server ignored Range: restart
      else if (res.status !== 206 && res.status !== 200) throw new Error(`${url}: HTTP ${res.status}`);
      const total = parseContentRangeTotal(res.headers.get("Content-Range"));
      if (total !== undefined && total !== this.total) throw new Error(`${url}: size changed (${this.total} → ${total})`);
      const reader = res.body!.getReader();
      let flushed = this.have;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        this.handle.write(value, { at: this.have });
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
      if (this.have !== this.total) throw new Error(`${url}: download ended at ${this.have} of ${this.total} bytes`);
      this.handle.flush();
      this.complete = true;
      await this.saveMeta({ size: this.total, complete: true, flushed: this.total });
    } catch (err) {
      this.failure = err instanceof Error ? err : new Error(String(err));
    }
    this.wake();
  }

  private wake(): void {
    const w = this.waiters;
    this.waiters = [];
    for (const r of w) r();
  }
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

/** Deletes one cached file and its metadata (e.g. when its contents turn out to be corrupt). */
export async function evictCached(key: string): Promise<void> {
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle(DIR, { create: true });
  const name = encodeURIComponent(key);
  for (const n of [name, `${name}.meta.json`]) await dir.removeEntry(n).catch(() => undefined);
}

/** Deletes every cached model file. */
export async function clearCache(): Promise<void> {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(DIR, { recursive: true }).catch(() => undefined);
}
