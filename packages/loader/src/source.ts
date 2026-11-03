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
    return new Uint8Array(await this.blob.slice(offset, offset + length).arrayBuffer());
  }
}

/** Plain HTTP range requests, no caching. */
export class HttpSource implements ByteSource {
  private total: number | undefined;
  constructor(
    readonly url: string,
    readonly name = url,
  ) {}

  async size(): Promise<number> {
    if (this.total === undefined) {
      const res = await fetch(this.url, { headers: { Range: "bytes=0-0" } });
      if (!res.ok) throw new Error(`${this.url}: HTTP ${res.status}`);
      this.total = parseContentRangeTotal(res.headers.get("Content-Range")) ?? Number(res.headers.get("Content-Length"));
      await res.body?.cancel();
    }
    return this.total;
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    const res = await fetch(this.url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
    if (res.status !== 206 && res.status !== 200) throw new Error(`${this.url}: HTTP ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    // A server that ignores Range sends the whole file.
    return res.status === 200 ? buf.subarray(offset, offset + length) : buf;
  }
}

export function parseContentRangeTotal(header: string | null): number | undefined {
  const m = header?.match(/\/(\d+)\s*$/);
  return m ? Number(m[1]) : undefined;
}
