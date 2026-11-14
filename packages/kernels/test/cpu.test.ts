import { describe, expect, it } from "vitest";
import { Ctx, STEP_BYTES, packStep, type Tensor } from "@kiln/core";
import { dequantQ4, dequantQ8, exactQ4, exactQ8, quantizeQ4, quantizeQ8 } from "@kiln/loader";
import { CpuBackend, ops, ref } from "../src";
import { F8, toF16, toF8 } from "../src/cpu";
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
      // SIMD takes q4/q8 dot products with x as int16 per block: about 1e-5 of the outputs (≈10 here).
      const tol = simd && name !== "f32" ? 5e-4 : 1e-4;
      let y!: Tensor;
      run(0, (ctx) => (y = ops.linear(ctx, upload(x, [M, K]), wt, { norm: { w: upload(nw, [K]), eps: 1e-5 } })));
      const want = ref.matmulRef(ref.rmsnormRef(x, nw, K, 1e-5), deq, M, N, K);
      expect(ref.maxAbsDiff(await read(y, M * N), want), name).toBeLessThan(tol);
      run(0, (ctx) => (y = ops.linear(ctx, upload(x, [1, K]), wt, { silu: true })));
      const g = ref.matmulRef(x.subarray(0, K), deq, 1, N, K);
      const gated = Float32Array.from({ length: N / 2 }, (_, i) => (g[i]! / (1 + Math.exp(-g[i]!))) * g[N / 2 + i]!);
      expect(ref.maxAbsDiff(await read(y, N / 2), gated), `${name} silu`).toBeLessThan(tol);
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

  it("attention over an f8 ring cache with a sliding window matches the reference", async () => {
    const [T, H, Hkv, D, maxSeq, slots, window, pos0] = [2, 4, 2, 16, 512, 300, 270, 400];
    const row = Hkv * D;
    // Values across the format's range: most near 1, some tiny, some large.
    const wide = (n: number) => randn(n).map((v, i) => v * (i % 7 === 0 ? 0.01 : i % 11 === 0 ? 40 : 1));
    const kf = wide(maxSeq * row);
    const vf = wide(maxSeq * row);
    const ringK = new Uint32Array((slots * row) / 4);
    const ringV = new Uint32Array((slots * row) / 4);
    const word = (f: Float32Array, at: number) => (toF8(f[at]!) | (toF8(f[at + 1]!) << 8) | (toF8(f[at + 2]!) << 16) | (toF8(f[at + 3]!) << 24)) >>> 0;
    for (let j = 0; j < pos0 + T; j++)
      for (let e = 0; e < row; e += 4) {
        const s = ((j % slots) * row + e) / 4;
        ringK[s] = word(kf, j * row + e);
        ringV[s] = word(vf, j * row + e);
      }
    const q = randn(T * H * D);
    let o!: Tensor;
    run(pos0, (ctx) => (o = ops.attention(ctx, upload(q, [T, H * D]), upload(ringK, [0]).buf, upload(ringV, [0]).buf, { T, H, Hkv, D, maxSeq, kv: "f8", window, slots })));
    const f8 = (v: number) => F8[toF8(v)]!;
    const want = ref.attentionRef(q, kf.map(f8), vf.map(f8), { T, H, Hkv, D, pos0, window });
    expect(ref.maxAbsDiff(await read(o, T * H * D), want)).toBeLessThan(1e-3);
  });

  it("writes an i8 ring cache and attends over it as the reference does over the values it holds", async () => {
    const [T, H, Hkv, D, maxSeq, slots, window] = [3, 4, 2, 16, 512, 300, 270];
    const row = Hkv * D;
    const HW = D / 4 + 1;
    // Heads of very different sizes, and one of zeros: each is scaled on its own.
    const sizes = (n: number) => randn(n).map((v, i) => v * [1, 0.01, 40, 0][Math.floor(i / D) % 4]!);
    const kf = sizes(maxSeq * row);
    const vf = sizes(maxSeq * row);
    // What an i8 cache holds for a head: codes sign · round(127·√|x / max|), each standing for c·|c|·max / 127².
    const held = (f: Float32Array) => {
      const out = new Float32Array(f.length);
      for (let h = 0; h < f.length / D; h++) {
        let max = 0;
        for (let i = 0; i < D; i++) max = Math.max(max, Math.abs(f[h * D + i]!));
        const inv = max > 0 ? Math.fround(1 / max) : 0;
        for (let i = 0; i < D; i++) {
          const x = Math.fround(f[h * D + i]! * inv);
          const c = Math.sign(x) * Math.round(127 * Math.fround(Math.sqrt(Math.min(1, Math.abs(x)))));
          out[h * D + i] = c * Math.abs(c) * Math.fround(max / 16129);
        }
      }
      return out;
    };
    const kc = be.alloc(slots * Hkv * HW * 4);
    const vc = be.alloc(slots * Hkv * HW * 4);
    // Written in calls of T rows, the way prefill does, far enough for the ring to wrap.
    const last = 400;
    for (let pos = last - slots - T; pos < last; pos += T) {
      const n = Math.min(T, last - pos);
      run(pos, (ctx) => ops.kvWrite(ctx, upload(kf.slice(pos * row, (pos + n) * row), [n, row]), upload(vf.slice(pos * row, (pos + n) * row), [n, row]), kc, vc, n, maxSeq, "i8", slots, Hkv));
    }
    // One head's bytes and scale, read back.
    const words = new Uint32Array(await be.readback(kc, 0, slots * Hkv * HW * 4));
    const at = ((last - 1) % slots) * Hkv * HW;
    const scale = new Float32Array(words.buffer)[at + D / 4]!;
    const want0 = held(kf.slice((last - 1) * row, (last - 1) * row + D));
    for (let i = 0; i < D; i++) {
      const c = (words[at + (i >> 2)]! << (24 - 8 * (i & 3))) >> 24;
      expect(c * Math.abs(c) * scale).toBeCloseTo(want0[i]!, 4);
    }
    const pos0 = last - T;
    const q = randn(T * H * D);
    let o!: Tensor;
    run(pos0, (ctx) => (o = ops.attention(ctx, upload(q, [T, H * D]), kc, vc, { T, H, Hkv, D, maxSeq, kv: "i8", window, slots })));
    const want = ref.attentionRef(q, held(kf), held(vf), { T, H, Hkv, D, pos0, window });
    expect(ref.maxAbsDiff(await read(o, T * H * D), want)).toBeLessThan(2e-3);
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

describe("CPU profiling", () => {
  it("times each kernel by label until stopProfile", async () => {
    const be = new CpuBackend();
    const step = be.alloc(STEP_BYTES);
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 1, temperature: 0 }));
    const x = be.alloc(64 * 4);
    const w = be.alloc(64 * 4);
    const ctx = new Ctx(be, step, "prof");
    const t: Tensor = { shape: [1, 64], dtype: "f32", buf: x };
    ops.rmsnorm(ctx, t, { shape: [64], dtype: "f32", buf: w }, 1e-6);
    ops.rmsnorm(ctx, t, { shape: [64], dtype: "f32", buf: w }, 1e-6);
    be.submit(ctx.dispatches);
    be.startProfile();
    be.submit(ctx.dispatches);
    be.submit(ctx.dispatches);
    const p = await be.stopProfile();
    expect(p.entries).toHaveLength(1);
    expect(p.entries[0]!.calls).toBe(4);
    expect(p.totalMs).toBeCloseTo(p.entries[0]!.ms);
    await expect(be.stopProfile()).rejects.toThrow();
  });
});

describe("FP8 E4M3 codes", () => {
  it("decodes every code and encodes it back", () => {
    for (let c = 0; c < 256; c++) {
      if ((c & 0x7f) === 0x7f) continue; // NaN in the format: never written
      expect(toF8(F8[c]!), String(c)).toBe(c);
    }
    expect(F8[0x7e]).toBe(448);
    expect(F8[0x08]).toBe(2 ** -6);
    expect(F8[0x01]).toBe(2 ** -9);
    expect(F8[0x38]).toBe(1);
    expect(F8[0xb8]).toBe(-1);
  });

  it("rounds to the nearest code, keeps tiny values and clamps large ones", () => {
    const finite = Array.from(F8).filter((_, c) => (c & 0x7f) !== 0x7f);
    let s = 7;
    const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32) * 2 - 1;
    for (let i = 0; i < 20000; i++) {
      const x = Math.fround(rnd() * 2 ** Math.floor(rnd() * 12));
      const got = F8[toF8(x)]!;
      const best = finite.reduce((m, v) => Math.min(m, Math.abs(v - x)), Infinity);
      expect(Math.abs(got - x), String(x)).toBeLessThanOrEqual(best * (1 + 1e-6));
      // A value keeps its sign, and is off by 1 part in 16 at most (less near zero in absolute terms).
      if (x !== 0) expect(Math.sign(got) === Math.sign(x) || got === 0).toBe(true);
      if (Math.abs(x) >= 2 ** -6 && Math.abs(x) <= 448) expect(Math.abs(got - x)).toBeLessThanOrEqual(Math.abs(x) / 16 + 1e-9);
    }
    expect(F8[toF8(1e6)]).toBe(448);
    expect(F8[toF8(-1e6)]).toBe(-448);
    expect(F8[toF8(Infinity)]).toBe(448);
    expect(F8[toF8(0)]).toBe(0);
    expect(F8[toF8(0.001)]).toBe(2 ** -9);
    expect(F8[toF8(0.0005)]).toBe(0);
  });
});

describe("partial rotary (CPU)", () => {
  const be = new CpuBackend({ simd: false });
  const step = be.alloc(STEP_BYTES);
  const upload = (data: Float32Array, shape: number[]): Tensor => {
    const buf = be.alloc(data.byteLength);
    be.write(buf, 0, data);
    return { shape, dtype: "f32", buf };
  };
  const read = async (t: { buf: Tensor["buf"] }, n: number) => new Float32Array(await be.readback(t.buf, 0, n * 4));
  const run = (pos0: number, build: (ctx: Ctx) => void) => {
    be.write(step, 0, packStep({ pos0, lastRow: 0, seed: 1, temperature: 0 }));
    const ctx = new Ctx(be, step, "t");
    build(ctx);
    be.submit(ctx.dispatches);
  };

  it.each([[false, 12], [true, 12], [false, 16], [false, 4]] as const)("rotates the first part of each head and leaves the rest (interleaved %s, %i of 16)", async (interleaved, R) => {
    const [T, H, Hkv, D, pos0, maxSeq] = [3, 4, 2, 16, 5, 12];
    const inv = Float32Array.from({ length: R / 2 }, (_, i) => 1 / 10000 ** ((2 * i) / R));
    // In place.
    const x = randn(T * H * D);
    const xt = upload(x, [T, H * D]);
    run(pos0, (ctx) => ops.rope(ctx, xt, T, H, D, upload(inv, [R / 2]), interleaved));
    const want = ref.ropeRef(x, T, H, D, pos0, inv, interleaved);
    const got = await read(xt, x.length);
    expect(ref.maxAbsDiff(got, want)).toBeLessThan(1e-5);
    // What lies past the rotated part is bit for bit the input.
    for (let r = 0; r < T * H; r++) for (let i = R; i < D; i++) expect(got[r * D + i]).toBe(x[r * D + i]);
    // Fused with the cache write: q out, k into the cache, v copied.
    const W = (H + 2 * Hkv) * D;
    const qkv = randn(T * W);
    const part = (from: number, n: number) => {
      const out = new Float32Array(T * n);
      for (let t = 0; t < T; t++) out.set(qkv.subarray(t * W + from, t * W + from + n), t * n);
      return out;
    };
    const kc = upload(new Float32Array(maxSeq * Hkv * D), [0]);
    const vc = upload(new Float32Array(maxSeq * Hkv * D), [0]);
    let qo!: Tensor;
    run(pos0, (ctx) => (qo = ops.ropeKv(ctx, upload(qkv, [T, W]), kc.buf, vc.buf, { T, H, Hkv, D, maxSeq }, upload(inv, [R / 2]), interleaved)));
    expect(ref.maxAbsDiff(await read(qo, T * H * D), ref.ropeRef(part(0, H * D), T, H, D, pos0, inv, interleaved))).toBeLessThan(1e-5);
    const wantK = ref.ropeRef(part(H * D, Hkv * D), T, Hkv, D, pos0, inv, interleaved);
    expect(ref.maxAbsDiff((await read(kc, maxSeq * Hkv * D)).subarray(pos0 * Hkv * D, (pos0 + T) * Hkv * D), wantK)).toBeLessThan(1e-5);
    expect((await read(vc, maxSeq * Hkv * D)).subarray(pos0 * Hkv * D, (pos0 + T) * Hkv * D)).toEqual(part((H + Hkv) * D, Hkv * D));
  });

  it("refuses frequencies that do not fit the head", () => {
    const x = upload(randn(16), [1, 16]);
    expect(() => run(0, (ctx) => ops.rope(ctx, x, 1, 1, 16, upload(randn(9), [9]), false))).toThrow(/do not fit a head of 16/);
  });
});

describe("linear attention kernels (CPU)", () => {
  const be = new CpuBackend({ simd: false });
  const step = be.alloc(STEP_BYTES);
  const upload = (data: Float32Array, shape: number[]): Tensor => {
    const buf = be.alloc(Math.max(4, data.byteLength));
    be.write(buf, 0, data);
    return { shape, dtype: "f32", buf };
  };
  const read = async (buf: Tensor["buf"], n: number) => new Float32Array(await be.readback(buf, 0, n * 4));
  const run = (build: (ctx: Ctx) => void) => {
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 1, temperature: 0 }));
    const ctx = new Ctx(be, step, "t");
    build(ctx);
    be.submit(ctx.dispatches);
  };
  const silu = (v: number) => v / (1 + Math.exp(-v));

  it("causal convolution carries its last inputs across calls", async () => {
    const [C, K, T] = [5, 4, 9];
    const x = randn(T * C);
    const w = randn(C * K);
    // Direct: zero padding on the left, as F.conv1d(padding=K-1)[:T].
    const want = new Float32Array(T * C);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < C; c++) {
        let acc = 0;
        for (let j = 0; j < K; j++) acc += t + j - (K - 1) >= 0 ? w[c * K + j]! * x[(t + j - (K - 1)) * C + c]! : 0;
        want[t * C + c] = silu(acc);
      }
    // In one call, and in calls of 1, 2, 1 and 5 rows (shorter and longer than the state).
    for (const chunks of [[T], [1, 2, 1, 5], [1, 1, 1, 1, 1, 1, 1, 1, 1], [4, 5]]) {
      const state = be.alloc((K - 1) * C * 4);
      const got = new Float32Array(T * C);
      let at = 0;
      for (const n of chunks) {
        let y!: Tensor;
        run((ctx) => (y = ops.causalConv(ctx, upload(x.subarray(at * C, (at + n) * C), [n, C]), upload(w, [C, K]), state)));
        got.set(await read(y.buf, n * C), at * C);
        at += n;
      }
      expect(ref.maxAbsDiff(got, want), String(chunks)).toBeLessThan(1e-5);
    }
  });

  it("the gated delta rule matches the reference recurrence, in one call or token by token", async () => {
    const [T, Hk, Hv, Dk, Dv] = [7, 2, 4, 8, 12];
    const keys = 2 * Hk * Dk;
    const C = keys + Hv * Dv;
    const conv = randn(T * C);
    const [a, b, aLog, dt] = [randn(T * Hv).map((v) => v * 3), randn(T * Hv), randn(Hv), randn(Hv)];
    // Queries and keys normalised per head, as the layer does before the recurrence.
    const qk = new Float32Array(T * keys);
    for (let r = 0; r < (T * keys) / Dk; r++) {
      let ss = 0;
      for (let i = 0; i < Dk; i++) ss += conv[Math.floor(r / (keys / Dk)) * C + (r % (keys / Dk)) * Dk + i]! ** 2;
      for (let i = 0; i < Dk; i++) qk[r * Dk + i] = conv[Math.floor(r / (keys / Dk)) * C + (r % (keys / Dk)) * Dk + i]! / Math.sqrt(ss + 1e-6);
    }
    // transformers' torch_recurrent_gated_delta_rule, head by head: state [Dk, Dv].
    const want = new Float32Array(T * Hv * Dv);
    for (let h = 0; h < Hv; h++) {
      const kh = Math.floor(h / (Hv / Hk));
      const S = Array.from({ length: Dk }, () => new Float64Array(Dv));
      for (let t = 0; t < T; t++) {
        const q = Array.from({ length: Dk }, (_, i) => qk[t * keys + kh * Dk + i]! / Math.sqrt(Dk));
        const k = Array.from({ length: Dk }, (_, i) => qk[t * keys + Hk * Dk + kh * Dk + i]!);
        const v = Array.from({ length: Dv }, (_, j) => conv[t * C + keys + h * Dv + j]!);
        const g = -Math.exp(aLog[h]!) * Math.log1p(Math.exp(a[t * Hv + h]! + dt[h]!));
        const beta = 1 / (1 + Math.exp(-b[t * Hv + h]!));
        for (const row of S) for (let j = 0; j < Dv; j++) row[j]! *= Math.exp(g);
        const mem = Array.from({ length: Dv }, (_, j) => S.reduce((sum, row, i) => sum + row[j]! * k[i]!, 0));
        const delta = v.map((x, j) => (x - mem[j]!) * beta);
        S.forEach((row, i) => delta.forEach((d, j) => (row[j]! += k[i]! * d)));
        for (let j = 0; j < Dv; j++) want[(t * Hv + h) * Dv + j] = S.reduce((sum, row, i) => sum + row[j]! * q[i]!, 0);
      }
    }
    const shape = { keyHeads: Hk, valueHeads: Hv, keyDim: Dk, valueDim: Dv };
    for (const chunks of [[T], [1, 1, 1, 1, 1, 1, 1], [3, 4]]) {
      const state = be.alloc(Hv * Dv * Dk * 4);
      const got = new Float32Array(T * Hv * Dv);
      let at = 0;
      for (const n of chunks) {
        const rows = (f: Float32Array, width: number) => upload(f.subarray(at * width, (at + n) * width), [n, width]);
        let y!: Tensor;
        run((ctx) => (y = ops.gatedDelta(ctx, rows(qk, keys), rows(conv, C), rows(a, Hv), rows(b, Hv), upload(aLog, [Hv]), upload(dt, [Hv]), state, { T: n, ...shape })));
        got.set(await read(y.buf, n * Hv * Dv), at * Hv * Dv);
        at += n;
      }
      expect(ref.maxAbsDiff(got, want), String(chunks)).toBeLessThan(2e-5);
    }
  });

  it("takes value heads in tiled order (GGUF files) as well as grouped", async () => {
    const [T, Hk, Hv, Dk, Dv] = [5, 2, 6, 8, 4];
    const r = Hv / Hk;
    const keys = 2 * Hk * Dk;
    const C = keys + Hv * Dv;
    const conv = randn(T * C);
    const [a, b, aLog, dt] = [randn(T * Hv), randn(T * Hv), randn(Hv), randn(Hv)];
    const qk = randn(T * keys).map((v) => v / 3);
    const shape = { keyHeads: Hk, valueHeads: Hv, keyDim: Dk, valueDim: Dv };
    const want = ref.gatedDeltaRef(qk, conv, a, b, aLog, dt, { T, ...shape });
    // Grouped head h = kh·r + j sits at tiled position j·Hk + kh.
    const tiledOf = (h: number) => (h % r) * Hk + Math.floor(h / r);
    const heads = (f: Float32Array, rows: number, width: number, base = 0, stride = Hv * width) => {
      const out = f.slice();
      for (let t = 0; t < rows; t++) for (let h = 0; h < Hv; h++) for (let i = 0; i < width; i++) out[t * stride + base + tiledOf(h) * width + i] = f[t * stride + base + h * width + i]!;
      return out;
    };
    const state = be.alloc(Hv * Dv * Dk * 4);
    let y!: Tensor;
    run((ctx) => (y = ops.gatedDelta(ctx, upload(qk, [T, keys]), upload(heads(conv, T, Dv, keys, C), [T, C]), upload(heads(a, T, 1), [T, Hv]), upload(heads(b, T, 1), [T, Hv]), upload(heads(aLog, 1, 1), [Hv]), upload(heads(dt, 1, 1), [Hv]), state, { T, ...shape, tiled: true })));
    const got = await read(y.buf, T * Hv * Dv);
    expect(ref.maxAbsDiff(got, heads(want, T, Dv))).toBeLessThan(2e-5);
  });

  it("splits heads into blocks and gates in place", async () => {
    const [T, H, D] = [2, 3, 4];
    const x = randn(T * H * 2 * D);
    let q!: Tensor, g!: Tensor;
    run((ctx) => {
      const xt = upload(x, [T, H * 2 * D]);
      q = ops.headSlices(ctx, xt, { H, D, blocks: 2, pick: 0 });
      g = ops.headSlices(ctx, xt, { H, D, blocks: 2, pick: 1 });
    });
    const [gq, gg] = [await read(q.buf, T * H * D), await read(g.buf, T * H * D)];
    for (let r = 0; r < T * H; r++)
      for (let i = 0; i < D; i++) {
        expect(gq[r * D + i]).toBe(x[r * 2 * D + i]);
        expect(gg[r * D + i]).toBe(x[r * 2 * D + D + i]);
      }
    const y = randn(8);
    const gate = randn(8).map((v) => v * 4);
    const yt = upload(y, [1, 8]);
    run((ctx) => ops.sigmoidMul(ctx, yt, upload(gate, [1, 8])));
    expect(ref.maxAbsDiff(await read(yt.buf, 8), y.map((v, i) => v / (1 + Math.exp(-gate[i]!))))).toBeLessThan(1e-6);
  });
});
