import { describe, expect, it } from "vitest";
import {
  MemorySource,
  UnsupportedTensorError,
  bf16ToF32,
  dequantQ4,
  dequantQ8,
  dequantQ8_0,
  exactPlane,
  exactQ4,
  exactQ8,
  f16ToF32,
  parseGGUF,
  parseModelId,
  quantizeQ4,
  quantizeQ8,
  readGGUFHeader,
  readSafetensorsHeader,
  repackQ4_0,
  repackQ8_0,
  rowBytes,
  toF32,
} from "../src";

function safetensors(header: object, data: Uint8Array): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.length + data.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  out.set(data, 8 + json.length);
  return out;
}

class GGUFWriter {
  private parts: number[] = [];
  u32(v: number) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); this.parts.push(...b); return this; }
  u64(v: number) { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); this.parts.push(...b); return this; }
  str(s: string) { const b = new TextEncoder().encode(s); this.u64(b.length); this.parts.push(...b); return this; }
  bytes() { return Uint8Array.from(this.parts); }
}

function tinyGGUF(tensorType = 0): Uint8Array {
  const w = new GGUFWriter().u32(0x46554747).u32(3).u64(1).u64(2);
  w.str("general.architecture").u32(8).str("llama");
  w.str("general.alignment").u32(4).u32(32);
  w.str("token_embd.weight").u32(2).u64(64).u64(2).u32(tensorType).u64(0);
  const header = w.bytes();
  const out = new Uint8Array(Math.ceil(header.length / 32) * 32 + 512);
  out.set(header);
  return out;
}

describe("safetensors", () => {
  it("parses the header and makes offsets absolute", async () => {
    const data = new Uint8Array(20);
    const bytes = safetensors(
      {
        __metadata__: { format: "pt" },
        b: { dtype: "F16", shape: [2], data_offsets: [16, 20] },
        a: { dtype: "F32", shape: [2, 2], data_offsets: [0, 16] },
      },
      data,
    );
    const h = await readSafetensorsHeader(new MemorySource(bytes));
    const base = bytes.length - 20;
    expect(h.metadata.format).toBe("pt");
    expect(h.tensors.map((t) => [t.name, t.type, t.offset, t.byteLength])).toEqual([
      ["a", "F32", base, 16],
      ["b", "F16", base + 16, 4],
    ]);
  });

  it("rejects unsupported dtypes with a clear error", async () => {
    const bytes = safetensors({ x: { dtype: "I64", shape: [1], data_offsets: [0, 8] } }, new Uint8Array(8));
    await expect(readSafetensorsHeader(new MemorySource(bytes))).rejects.toThrow(/I64 is not supported/);
  });
});

describe("gguf", () => {
  it("parses metadata and reverses dims into row-major shape", () => {
    const g = parseGGUF(tinyGGUF());
    expect(g.version).toBe(3);
    expect(g.metadata.get("general.architecture")).toBe("llama");
    const t = g.tensors[0]!;
    expect(t.shape).toEqual([2, 64]);
    expect(t.offset % 32).toBe(0);
    expect(t.byteLength).toBe(2 * 64 * 4);
  });

  it("grows the prefix until the header fits", async () => {
    const g = await readGGUFHeader(new MemorySource(tinyGGUF()), 16);
    expect(g.tensors).toHaveLength(1);
  });

  it("names unsupported k-quants", () => {
    expect(() => parseGGUF(tinyGGUF(12))).toThrow(UnsupportedTensorError);
    expect(() => parseGGUF(tinyGGUF(12))).toThrow(/Q4_K/);
  });
});

describe("conversion", () => {
  it("converts f16 and bf16", () => {
    expect([...f16ToF32(new Uint8Array(new Uint16Array([0x3c00, 0xc000, 0x3555]).buffer))].map((v) => +v.toFixed(4))).toEqual([1, -2, 0.3333]);
    expect([...bf16ToF32(new Uint8Array(new Uint16Array([0x3f80, 0xc040]).buffer))]).toEqual([1, -3]);
  });

  it("q4 round-trips within half a quantization step", () => {
    const x = Float32Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.7) * (1 + (i % 32) / 8));
    const p = quantizeQ4(x);
    const y = dequantQ4(p);
    for (let i = 0; i < x.length; i++) expect(Math.abs(x[i]! - y[i]!)).toBeLessThanOrEqual(Math.abs(p.scales[i >> 5]!) / 2 + 1e-6);
  });

  it("repacks GGUF Q4_0 blocks into sequential nibbles", () => {
    const block = new Uint8Array(18);
    new DataView(block.buffer).setUint16(0, 0x3c00, true); // scale 1.0
    for (let j = 0; j < 16; j++) block[2 + j] = j | ((15 - j) << 4);
    const y = dequantQ4(repackQ4_0(block, 32));
    expect([...y.subarray(0, 16)]).toEqual(Array.from({ length: 16 }, (_, j) => j - 8));
    expect([...y.subarray(16)]).toEqual(Array.from({ length: 16 }, (_, j) => 7 - j));
  });

  it("repacks many Q4_0 blocks element for element", () => {
    const blocks = 37;
    const src = Uint8Array.from({ length: blocks * 18 }, (_, i) => (i * 97 + 13) & 255);
    for (let b = 0; b < blocks; b++) new DataView(src.buffer).setUint16(b * 18, 0x3c00 + b, true);
    const p = repackQ4_0(src, blocks * 32);
    for (let e = 0; e < blocks * 32; e++) {
      const b = e >> 5;
      const j = e & 31;
      const byte = src[b * 18 + 2 + (j & 15)]!;
      expect((p.nibbles[e >> 3]! >>> ((e & 7) * 4)) & 15).toBe(j < 16 ? byte & 15 : byte >> 4);
    }
    expect(p.scales[1]).toBe(f16ToF32(src.subarray(18, 20))[0]);
  });

  it("q4 picks llama.cpp's codes, including the clamped block maximum", () => {
    const x = Float32Array.from({ length: 128 }, (_, i) => Math.cos(i * 1.3) * (i % 7) - (i === 40 ? 9 : 0));
    // A tie between +3 and -3 in the last block: the first (-3) sets the scale's sign.
    x.subarray(96).fill(0.5);
    x[100] = -3;
    x[110] = 3;
    const p = quantizeQ4(x);
    for (let e = 0; e < x.length; e++) {
      const blk = x.subarray(e & ~31, (e & ~31) + 32);
      const max = blk.reduce((m, v) => (Math.abs(v) > Math.abs(m) ? v : m), 0);
      if (e === 100) expect(p.scales[3]).toBe(Math.fround(3 / 8));
      const want = Math.min(15, Math.max(0, Math.floor(x[e]! * (-8 / max) + 8.5)));
      expect((p.nibbles[e >> 3]! >>> ((e & 7) * 4)) & 15).toBe(want);
    }
  });

  it("q8 round-trips within half a step, with 4 int8 per word in element order", () => {
    const x = Float32Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.7) * (1 + (i % 32) / 8));
    const p = quantizeQ8(x);
    const y = dequantQ8(p);
    for (let i = 0; i < x.length; i++) expect(Math.abs(x[i]! - y[i]!)).toBeLessThanOrEqual(p.scales[i >> 5]! / 2 + 1e-6);
    // The block's largest magnitude maps to ±127; the low byte of word 0 is element 0.
    const one = quantizeQ8(Float32Array.from({ length: 32 }, (_, j) => (j === 0 ? -2 : j === 1 ? 1 : 0)));
    expect(one.scales[0]).toBeCloseTo(2 / 127, 7);
    expect(one.bytes[0]! & 0xff).toBe(0x81); // -127
    expect((one.bytes[0]! >> 8) & 0xff).toBe(64); // round(63.5)
  });

  it("q8 rounds exact ties away from zero", () => {
    // amax 127/128 makes d = 1/128 exactly, so x = (k + 0.5) / 128 is a tie.
    const x = new Float32Array(32);
    x.set([127, 2.5, -2.5, 0.5, -0.5, 126.5, 3.4999].map((v) => v / 128));
    const p = quantizeQ8(x);
    expect(p.scales[0]).toBe(1 / 128);
    expect([...new Int8Array(p.bytes.buffer).subarray(0, 7)]).toEqual([127, 3, -3, 1, -1, 127, 3]);
  });

  it("holds Q4_1/Q4_K as affine q4 and Q5_0/Q5_1/Q5_K as q8, bit for bit as toF32", () => {
    // Random blocks whose f16 scales (the first bytes of each block or super-block) are finite.
    const blocks = (type: "Q4_1" | "Q4_K" | "Q5_0" | "Q5_1" | "Q5_K", n: number) => {
      const bytes = Uint8Array.from({ length: rowBytes(type, n) }, (_, i) => (i * 2654435761) >>> 24);
      const size = rowBytes(type, type.endsWith("K") ? 256 : 32);
      const f16s = type === "Q5_0" ? 1 : 2;
      for (let o = 0; o < bytes.length; o += size) for (let h = 0; h < f16s; h++) bytes[o + 2 * h + 1] = (bytes[o + 2 * h + 1]! & 0x83) | 0x14;
      return bytes;
    };
    const n = 1024;
    for (const type of ["Q4_1", "Q4_K"] as const) {
      expect(exactPlane(type)).toBe("q4");
      expect([...dequantQ4(exactQ4(type, blocks(type, n), n))], type).toEqual([...toF32(type, blocks(type, n), n)]);
    }
    for (const type of ["Q5_0", "Q5_1", "Q5_K"] as const) {
      expect(exactPlane(type)).toBe("q8");
      const p = exactQ8(type, blocks(type, n), n);
      expect(!!p.mins, type).toBe(type !== "Q5_0");
      expect([...dequantQ8(p)], type).toEqual([...toF32(type, blocks(type, n), n)]);
    }
    expect([exactPlane("Q4_0"), exactPlane("Q8_0"), exactPlane("Q6_K"), exactPlane("BF16")]).toEqual(["q4", "q8", undefined, undefined]);
  });

  it("repacks GGUF Q8_0 blocks exactly", () => {
    const blocks = new Uint8Array(68);
    const dv = new DataView(blocks.buffer);
    dv.setUint16(0, 0x3800, true); // 0.5
    dv.setUint16(34, 0x4000, true); // 2.0
    for (let j = 0; j < 32; j++) {
      dv.setInt8(2 + j, j - 16);
      dv.setInt8(36 + j, 127 - j * 8);
    }
    expect([...dequantQ8(repackQ8_0(blocks, 64))]).toEqual([...dequantQ8_0(blocks, 64)]);
  });
});

describe("hub ids", () => {
  it("parses repo and gguf ids", () => {
    expect(parseModelId("org/repo")).toEqual({ repo: "org/repo", revision: "main", gguf: undefined });
    expect(parseModelId("org/repo/sub/m.Q4_0.gguf", "abc").gguf).toBe("sub/m.Q4_0.gguf");
    expect(() => parseModelId("repo")).toThrow();
  });
});
