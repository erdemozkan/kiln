/** Throws unless [offset, offset + length) is a valid byte range (integers, not negative). */
export function checkRange(name: string, offset: number, length: number): void {
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0) throw new RangeError(`${name}: invalid read of ${length} bytes at ${offset}`);
}

/** Random-access bytes. Loaders read tensors roughly in offset order, which the cached source relies on. */
export interface ByteSource {
  readonly name: string;
  size(): Promise<number>;
  read(offset: number, length: number): Promise<Uint8Array>;
  close?(): void;
}

export class MemorySource implements ByteSource {
  constructor(
    private readonly bytes: Uint8Array,
    readonly name = "memory",
  ) {}
  async size(): Promise<number> {
    return this.bytes.byteLength;
  }
  async read(offset: number, length: number): Promise<Uint8Array> {
    checkRange(this.name, offset, length);
    if (offset + length > this.bytes.byteLength) throw new RangeError(`${this.name}: read past end`);
    return this.bytes.subarray(offset, offset + length);
  }
}

export class BlobSource implements ByteSource {
  constructor(
    private readonly blob: Blob,
    readonly name = "blob",
  ) {}
  async size(): Promise<number> {
    return this.blob.size;
  }
  async read(offset: number, length: number): Promise<Uint8Array> {
    checkRange(this.name, offset, length);
    if (offset + length > this.blob.size) throw new RangeError(`${this.name}: read past end`);
    return new Uint8Array(await this.blob.slice(offset, offset + length).arrayBuffer());
  }
}

/** Plain HTTP range requests, no caching. */
export class HttpSource implements ByteSource {
  private total: number | undefined;
  constructor(
    readonly url: string,
    readonly name = url,
    /** First retry delay; later ones double (see `withRetries`). */
    private readonly retryMs = 1000,
  ) {}

  async size(): Promise<number> {
    if (this.total === undefined) {
      this.total = await withRetries(this.name, this.retryMs, async () => {
        const res = await fetch(this.url, { headers: { Range: "bytes=0-0" } });
        check(this.url, res);
        const total = parseContentRangeTotal(res.headers.get("Content-Range")) ?? Number(res.headers.get("Content-Length"));
        await res.body?.cancel();
        return total;
      });
    }
    return this.total;
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    checkRange(this.name, offset, length);
    // Known size: a read past the end is a bad header, not something to retry.
    if (this.total !== undefined && offset + length > this.total) throw new RangeError(`${this.name}: read past end`);
    return withRetries(this.name, this.retryMs, async () => {
      const res = await fetch(this.url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
      check(this.url, res);
      const buf = new Uint8Array(await res.arrayBuffer());
      // A server that ignores Range sends the whole file.
      const out = res.status === 200 ? buf.subarray(offset, offset + length) : buf;
      if (out.length !== length) throw new TransientError(`${this.url}: got ${out.length} of ${length} bytes`);
      return out;
    });
  }
}

/** A failure worth retrying: the network (fetch's TypeError), a server error, or a short body. */
class TransientError extends Error {}

function check(url: string, res: Response): void {
  if (res.status >= 500 || res.status === 429) throw new TransientError(`${url}: HTTP ${res.status}`);
  if (res.status !== 206 && res.status !== 200) throw new Error(`${url}: HTTP ${res.status}`);
}

/** Runs `f`, retrying transient failures up to 4 times after 1, 2, 4 and 8 × `retryMs`. */
export async function withRetries<T>(name: string, retryMs: number, f: () => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await f();
    } catch (err) {
      const transient = err instanceof TransientError || (err instanceof Error && err.name === "TypeError");
      if (!transient || attempt >= 4) throw err;
      console.warn(`[kiln] ${name}: ${(err as Error).message}; retrying`);
      await new Promise((r) => setTimeout(r, retryMs * 2 ** attempt));
    }
  }
}

export function parseContentRangeTotal(header: string | null): number | undefined {
  const m = header?.match(/\/(\d+)\s*$/);
  return m ? Number(m[1]) : undefined;
}
