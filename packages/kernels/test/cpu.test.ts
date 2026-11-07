import { describe, expect, it } from "vitest";
import { Ctx, STEP_BYTES, packStep, type Tensor } from "@kiln/core";
import { dequantQ4, dequantQ8, exactQ4, exactQ8, quantizeQ4, quantizeQ8 } from "@kiln/loader";
import { CpuBackend, ops, ref } from "../src";
import { toF16 } from "../src/cpu";
import { WasmHeap } from "../src/simd";

let seed = 99;
const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32) * 2 - 1;
const randn = (n: number) => Float32Array.from({ length: n }, rand);

/** The CPU backend runs the same ops as WebGPU; the browser kernel page checks all of them (?backend=cpu). */
describe.each([
  ["SIMD", true],
  ["TypeScript", false],
])("CPU backend, matmuls in %s", (_, simd) => {
  const be = new CpuBackend({ simd });
  it("uses the path asked for", () => expect(be.simd).toBe(simd));
  const step = be.alloc(STEP_BYTES);
  const upload = (data: Float32Array | Uint32Array, shape: number[]): Tensor => {
    const buf = be.alloc(data.byteLength);
    be.write(buf, 0, data);
    return { shape, dtype: data instanceof Uint32Array ? "u32" : "f32", buf };
  };
  const run = (pos0: number, build: (ctx: Ctx) => void) => {
    be.write(step, 0, packStep({ pos0, lastRow: 0, seed: 1, temperature: 0 }));
    const ctx = new Ctx(be, step, "t");
    build(ctx);
    be.submit(ctx.dispatches);
  };
  const read = async (t: Tensor, n: number) => new Float32Array(await be.readback(t.buf, 0, n * 4));

  it("matmuls in f32, q4, q8 and affine q4, with the norm prologue and gated epilogue", async () => {
    const [M, N, K] = [3, 10, 64];
    const x = randn(M * K);
    const w = randn(N * K);
    const nw = randn(K);
    const q4 = quantizeQ4(w);
    const q8 = quantizeQ8(w);
    // Affine planes straight from GGUF Q4_1 blocks: f16 d and m, 16 bytes of nibbles per 32 values.
    const blocks = new Uint8Array((N * K) / 32 * 20);
    for (let b = 0; b < (N * K) / 32; b++) {
      const dv = new DataView(blocks.buffer, b * 20);
      dv.setUint16(0, toF16(0.01 + b * 1e-4), true);
      dv.setUint16(2, toF16(-0.1), true);
      for (let j = 0; j < 16; j++) blocks[b * 20 + 4 + j] = (b * 7 + j * 13) & 255;
    }
    const q4a = exactQ4("Q4_1", blocks, N * K);
    // And Q5_1 (f16 d and m, 4 bytes of high bits, 16 of nibbles): affine q8.
    const b5 = new Uint8Array((N * K) / 32 * 24);
    for (let b = 0; b < (N * K) / 32; b++) {
      const dv = new DataView(b5.buffer, b * 24);
      dv.setUint16(0, toF16(0.005 + b * 1e-4), true);
      dv.setUint16(2, toF16(-0.08), true);
      for (let j = 0; j < 20; j++) b5[b * 24 + 4 + j] = (b * 11 + j * 29) & 255;
    }
    const q8a = exactQ8("Q5_1", b5, N * K);
    const weights: [string, Tensor, Float32Array][] = [
      ["f32", upload(w, [N, K]), w],
      ["q4", { ...upload(q4.nibbles, [N, K]), dtype: "q4", scales: upload(q4.scales, [0]).buf }, dequantQ4(q4)],
      ["q8", { ...upload(q8.bytes, [N, K]), dtype: "q8", scales: upload(q8.scales, [0]).buf }, dequantQ8(q8)],
      ["q8a", { ...upload(q8a.bytes, [N, K]), dtype: "q8", scales: upload(q8a.scales, [0]).buf, mins: upload(q8a.mins!, [0]).buf }, dequantQ8(q8a)],
      ["q4a", { ...upload(q4a.nibbles, [N, K]), dtype: "q4", scales: upload(q4a.scales, [0]).buf, mins: upload(q4a.mins!, [0]).buf }, dequantQ4(q4a)],
    ];
    for (const [name, wt, deq] of weights) {
      let y!: Tensor;
      run(0, (ctx) => (y = ops.linear(ctx, upload(x, [M, K]), wt, { norm: { w: upload(nw, [K]), eps: 1e-5 } })));
      const want = ref.matmulRef(ref.rmsnormRef(x, nw, K, 1e-5), deq, M, N, K);
      expect(ref.maxAbsDiff(await read(y, M * N), want), name).toBeLessThan(1e-4);
      run(0, (ctx) => (y = ops.linear(ctx, upload(x, [1, K]), wt, { silu: true })));
      const g = ref.matmulRef(x.subarray(0, K), deq, 1, N, K);
      const gated = Float32Array.from({ length: N / 2 }, (_, i) => (g[i]! / (1 + Math.exp(-g[i]!))) * g[N / 2 + i]!);
      expect(ref.maxAbsDiff(await read(y, N / 2), gated), `${name} silu`).toBeLessThan(1e-4);
    }
  });

  it("attention over an f16 ring cache with a sliding window matches the reference", async () => {
    // A ring of 300 slots for a 270-key window, positions past 300 so it has wrapped.
    const [T, H, Hkv, D, maxSeq, slots, window, pos0] = [2, 4, 2, 16, 512, 300, 270, 400];
    const row = Hkv * D;
    const kf = randn(maxSeq * row);
    const vf = randn(maxSeq * row);
    const f16 = (v: number) => {
      const h = toF16(v);
      const e = (h >> 10) & 31;
      const a = e ? (1024 + (h & 1023)) * 2 ** (e - 25) : (h & 1023) * 2 ** -24;
      return h & 0x8000 ? -a : a;
    };
    const ringK = new Uint32Array((slots * row) / 2);
    const ringV = new Uint32Array((slots * row) / 2);
    for (let j = 0; j < pos0 + T; j++)
      for (let e = 0; e < row; e += 2) {
        const s = ((j % slots) * row + e) / 2;
        ringK[s] = (toF16(kf[j * row + e]!) | (toF16(kf[j * row + e + 1]!) << 16)) >>> 0;
        ringV[s] = (toF16(vf[j * row + e]!) | (toF16(vf[j * row + e + 1]!) << 16)) >>> 0;
      }
    const q = randn(T * H * D);
    let o!: Tensor;
    run(pos0, (ctx) => (o = ops.attention(ctx, upload(q, [T, H * D]), upload(ringK, [0]).buf, upload(ringV, [0]).buf, { T, H, Hkv, D, maxSeq, kv: "f16", window, slots })));
    const want = ref.attentionRef(q, kf.map(f16), vf.map(f16), { T, H, Hkv, D, pos0, window });
    expect(ref.maxAbsDiff(await read(o, T * H * D), want)).toBeLessThan(1e-4);
  });

  it("samples greedily and feeds the token back", async () => {
    const V = 100;
    const logits = randn(V);
    logits[37] = 5;
    const out = be.alloc(4);
    const seen = upload(new Uint32Array(4), [0]).buf;
    const allowed = upload(new Uint32Array(5), [0]).buf;
    run(0, (ctx) => ops.sample(ctx, upload(logits, [1, V]), out, seen, allowed));
    expect(new Uint32Array(await be.readback(out, 0, 4))[0]).toBe(37);
  });
});

describe("WasmHeap", () => {
  it("reuses freed regions, coalesces holes and shrinks the top", () => {
    const h = WasmHeap.create()!;
    const a = h.alloc(100), b = h.alloc(10), c = h.alloc(64);
    expect(a).toBeGreaterThan(0);
    expect([b - a, c - b]).toEqual([112, 16]);
    h.release(a, 100);
    expect(h.alloc(40)).toBe(a); // first fit, inside a's hole
    h.release(a, 40);
    h.release(b, 10); // merges with the hole left of it
    expect(h.alloc(128)).toBe(a);
    h.release(a, 128);
    h.release(c, 64); // at the top: the top falls back past the merged hole too
    expect(h.bytes).toBe(a);
  });

  it("grows the memory and keeps buffer contents across growth", () => {
    const be = new CpuBackend();
    const small = be.alloc(16);
    be.write(small, 0, Float32Array.of(1, 2, 3, 4));
    const before = be.heap!.buffer.byteLength;
    be.alloc(4 * before);
    expect(be.heap!.buffer.byteLength).toBeGreaterThan(before);
    expect([...(small as unknown as { f32: Float32Array }).f32]).toEqual([1, 2, 3, 4]);
  });
});

describe("spanTile", () => {
  it("takes the largest head-dim chunk that fits workgroup memory", async () => {
    const { spanTile } = await import("../src/wgsl");
    expect(spanTile(72, 16384)).toEqual({ RT: 4, DC: 24 });
    expect(spanTile(64, 16384)).toEqual({ RT: 4, DC: 32 });
    // 16 value rows of 512 floats alone are 32 KiB: no tile fits 16 KiB.
    expect(spanTile(512, 16384)).toBeUndefined();
  });
});
