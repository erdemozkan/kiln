import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Tensor } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import { MemorySource, convertedKey, openConverted, type ByteSource, type TensorEntry } from "@kiln/loader";
import { freeWeights, loadWeights, type WeightFile } from "../src/weights";

/** In-memory OPFS: the directory, file and sync-handle calls the converted-weights cache uses. */
const disk = new Map<string, { bytes: Uint8Array }>();
let writeFailsAt: number | undefined;
let readFails = false;
class FakeHandle {
  constructor(private readonly file: { bytes: Uint8Array }) {}
  read(buf: Uint8Array, { at }: { at: number }): number {
    if (readFails) return 0;
    const n = Math.max(0, Math.min(buf.length, this.file.bytes.length - at));
    buf.set(this.file.bytes.subarray(at, at + n));
    return n;
  }
  write(buf: Uint8Array, { at }: { at: number }): number {
    if (writeFailsAt !== undefined && at + buf.length > writeFailsAt) return 0;
    if (at + buf.length > this.file.bytes.length) this.truncate(at + buf.length);
    this.file.bytes.set(buf, at);
    return buf.length;
  }
  getSize(): number {
    return this.file.bytes.length;
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
    return { createSyncAccessHandle: async () => new FakeHandle(file), getFile: async () => ({ text: async () => new TextDecoder().decode(file.bytes) }) };
  },
};

beforeEach(() => {
  disk.clear();
  writeFailsAt = undefined;
  readFails = false;
  vi.stubGlobal("FileSystemSyncAccessHandle", FakeHandle);
  vi.stubGlobal("navigator", { storage: { getDirectory: async () => dir } });
});
afterEach(() => {
  vi.unstubAllGlobals();
});

/** A small F32 "model": two matrices that quantize, a norm and a bias that stay f32. */
function model(seed = 1): { bytes: Uint8Array; tensors: TensorEntry[] } {
  const shapes: [string, number[]][] = [
    ["layers.0.q_proj.weight", [8, 64]],
    ["layers.0.norm.weight", [64]],
    ["layers.0.mlp.weight", [16, 32]],
    ["layers.0.mlp.bias", [16]],
  ];
  const total = shapes.reduce((s, [, sh]) => s + sh.reduce((a, b) => a * b, 1), 0);
  const values = new Float32Array(total);
  let x = seed;
  for (let i = 0; i < total; i++) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    values[i] = x / 2 ** 31 - 1;
  }
  let offset = 0;
  const tensors = shapes.map(([name, shape]) => {
    const byteLength = shape.reduce((a, b) => a * b, 1) * 4;
    const e: TensorEntry = { name, shape, type: "F32", offset, byteLength };
    offset += byteLength;
    return e;
  });
  return { bytes: new Uint8Array(values.buffer), tensors };
}

/** A source that counts the bytes read from it. */
function counting(bytes: Uint8Array): ByteSource & { read_: number } {
  const inner = new MemorySource(bytes);
  const src = {
    name: "counting",
    read_: 0,
    size: () => inner.size(),
    read: (o: number, n: number) => {
      src.read_ += n;
      return inner.read(o, n);
    },
  };
  return src;
}

async function dump(backend: CpuBackend, weights: Map<string, Tensor>): Promise<Record<string, number[][]>> {
  const out: Record<string, number[][]> = {};
  for (const [name, t] of weights) {
    out[name] = [];
    for (const b of [t.buf, t.scales, t.mins]) if (b) out[name].push([...new Uint8Array(await backend.readback(b, 0, b.size))]);
  }
  return out;
}

const cache = { key: "org/model@main/weights" };
const metaOf = (quant: string) => {
  const f = disk.get(`${encodeURIComponent(convertedKey(`${cache.key}|${quant}`))}.meta.json`);
  return f ? (JSON.parse(new TextDecoder().decode(f.bytes)) as { size: number; complete: boolean; stamp: string }) : undefined;
};

describe("converted-weight cache", () => {
  it("writes the converted weights once, then loads the same bytes without reading the file", async () => {
    const backend = new CpuBackend();
    const m = model();
    const first = counting(m.bytes);
    const a = await loadWeights(backend, [{ name: "model.safetensors", source: first, tensors: m.tensors }], "q4", undefined, cache);
    expect(first.read_).toBe(m.bytes.length);
    const meta = metaOf("q4")!;
    expect(meta.complete).toBe(true);
    // q4: 4 bits a weight plus an f32 scale per 32, and the f32 norm and bias.
    expect(meta.size).toBe((8 * 64 + 16 * 32) / 2 + ((8 * 64 + 16 * 32) / 32) * 4 + (64 + 16) * 4);

    const second = counting(m.bytes);
    const progress: number[] = [];
    const b = await loadWeights(backend, [{ name: "model.safetensors", source: second, tensors: m.tensors }], "q4", (p) => progress.push(p.loaded / p.total), cache);
    expect(second.read_).toBe(0);
    expect(progress.at(-1)).toBe(1);
    expect([...b.keys()]).toEqual([...a.keys()]);
    expect([...b.values()].map((t) => [t.dtype, t.shape])).toEqual([...a.values()].map((t) => [t.dtype, t.shape]));
    expect(await dump(backend, b)).toEqual(await dump(backend, a));
    freeWeights(backend, a);
    freeWeights(backend, b);
  });

  it("keeps what transform and scale did to the weights", async () => {
    const backend = new CpuBackend();
    const m = model();
    const file = (source: ByteSource): WeightFile => ({
      name: "model.safetensors",
      source,
      tensors: m.tensors,
      transform: (name, v) => {
        if (name.includes("norm")) for (let i = 0; i < v.length; i++) v[i] = v[i]! + 1;
      },
      scale: (name) => (name.includes("mlp") ? 0.5 : undefined),
      rename: (name) => (name.endsWith(".bias") ? undefined : `model.${name}`),
    });
    const a = await loadWeights(backend, [file(new MemorySource(m.bytes))], "q8", undefined, cache);
    const src = counting(m.bytes);
    const b = await loadWeights(backend, [file(src)], "q8", undefined, cache);
    expect(src.read_).toBe(0);
    expect([...b.keys()]).toEqual(["model.layers.0.q_proj.weight", "model.layers.0.norm.weight", "model.layers.0.mlp.weight"]);
    expect(await dump(backend, b)).toEqual(await dump(backend, a));
  });

  it("does not cache what is no smaller than the file", async () => {
    const backend = new CpuBackend();
    const m = model();
    await loadWeights(backend, [{ name: "model.safetensors", source: new MemorySource(m.bytes), tensors: m.tensors }], "f32", undefined, cache);
    expect(disk.size).toBe(0);
  });

  it("keeps each setting apart and converts again when the tensors change", async () => {
    const backend = new CpuBackend();
    const m = model();
    const load = (bytes: Uint8Array, tensors: TensorEntry[], quant: "q4" | "q8") => {
      const source = counting(bytes);
      return loadWeights(backend, [{ name: "model.safetensors", source, tensors }], quant, undefined, cache).then(() => source.read_);
    };
    expect(await load(m.bytes, m.tensors, "q4")).toBeGreaterThan(0);
    expect(await load(m.bytes, m.tensors, "q8")).toBeGreaterThan(0);
    expect(await load(m.bytes, m.tensors, "q4")).toBe(0);
    expect(await load(m.bytes, m.tensors, "q8")).toBe(0);
    // The same bytes under another name: a different model as far as the cache can tell.
    const renamed = m.tensors.map((t) => ({ ...t, name: t.name.replace("q_proj", "k_proj") }));
    expect(await load(m.bytes, renamed, "q4")).toBeGreaterThan(0);
    expect(await load(m.bytes, renamed, "q4")).toBe(0);
  });

  it("loads from the file when the store fills up, and tries again next time", async () => {
    const backend = new CpuBackend();
    const m = model();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const load = async () => {
      const source = counting(m.bytes);
      const w = await loadWeights(backend, [{ name: "model.safetensors", source, tensors: m.tensors }], "q4", undefined, cache);
      return { read: source.read_, bytes: await dump(backend, w) };
    };
    writeFailsAt = 300;
    const a = await load();
    expect(a.read).toBe(m.bytes.length);
    expect(metaOf("q4")!.complete).toBe(false);
    const again = await load();
    expect(again.read).toBe(m.bytes.length);
    writeFailsAt = undefined;
    expect((await load()).read).toBe(m.bytes.length);
    const hit = await load();
    expect(hit.read).toBe(0);
    expect(hit.bytes).toEqual(a.bytes);
    warn.mockRestore();
  });

  it("converts again when the cached bytes cannot be read", async () => {
    const backend = new CpuBackend();
    const m = model();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const a = await loadWeights(backend, [{ name: "model.safetensors", source: new MemorySource(m.bytes), tensors: m.tensors }], "q4", undefined, cache);
    readFails = true;
    const source = counting(m.bytes);
    const b = await loadWeights(backend, [{ name: "model.safetensors", source, tensors: m.tensors }], "q4", undefined, cache);
    expect(source.read_).toBe(m.bytes.length);
    readFails = false;
    expect(await dump(backend, b)).toEqual(await dump(backend, a));
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("frees the tensors and leaves no entry when a load fails part-way", async () => {
    const backend = new CpuBackend();
    const m = model();
    const source: ByteSource = { name: "flaky", size: async () => m.bytes.length, read: async (o, n) => (o > 0 ? Promise.reject(new Error("network down")) : m.bytes.subarray(o, o + n)) };
    await expect(loadWeights(backend, [{ name: "model.safetensors", source, tensors: m.tensors }], "q4", undefined, cache)).rejects.toThrow("network down");
    expect(metaOf("q4")!.complete).toBe(false);
    const entry = await openConverted(`${cache.key}|q4`, "other", 8);
    expect(entry?.mode).toBe("write");
    entry?.mode === "write" && entry.abort();
  });
});

describe("openConverted", () => {
  it("round-trips across segments and rejects another stamp or length", async () => {
    const data = Uint8Array.from({ length: 100 }, (_, i) => i * 3);
    const w = await openConverted("k", "s1", 100, { segment: 32 });
    if (w?.mode !== "write") throw new Error("expected a writer");
    expect(w.append(data.subarray(0, 50))).toBe(true);
    expect(w.append(data.subarray(50))).toBe(true);
    expect(await w.commit()).toBe(true);
    const r = await openConverted("k", "s1", 100, { segment: 32 });
    if (r?.mode !== "read") throw new Error("expected a reader");
    expect([...r.read(20, 60)]).toEqual([...data.subarray(20, 80)]);
    expect(() => r.read(90, 20)).toThrow(/cached read/);
    r.close();
    // The failed read dropped the entry.
    expect((await openConverted("k", "s1", 100, { segment: 32 }))?.mode).toBe("write");
  });

  it("is not complete until every byte is written", async () => {
    const w = await openConverted("k", "s", 10);
    if (w?.mode !== "write") throw new Error("expected a writer");
    w.append(new Uint8Array(4));
    expect(await w.commit()).toBe(false);
    expect((await openConverted("k", "s", 10))?.mode).toBe("write");
    expect((await openConverted("k", "s2", 10))?.mode).toBe("write");
  });

  it("is undefined without OPFS", async () => {
    vi.unstubAllGlobals();
    expect(await openConverted("k", "s", 10)).toBeUndefined();
  });
});
