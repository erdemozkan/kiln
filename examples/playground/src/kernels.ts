import { createWebGPUBackend } from "@kiln/backend-webgpu";
import { Ctx, packStep, type Backend, type Buf, type StepParams, type Tensor } from "@kiln/core";
import { CpuBackend, gpuConverts, ops, ref } from "@kiln/kernels";
import { dequantQ4, dequantQ8, f16ToF32, MemorySource, quantizeQ4, quantizeQ8, rowBytes, type SrcType } from "@kiln/loader";
import { loadWeights, setGpuConvert } from "@kiln/runtime";

interface Result {
  kernel: string;
  case: string;
  error: number;
  tol: number;
  pass: boolean;
  note?: string;
}

let seed = 1234;
const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32) * 2 - 1;
const randn = (n: number) => Float32Array.from({ length: n }, rand);

/** f32 → f16 bits, round to nearest even (normal range only, enough for test data in [-1, 1]). */
function toHalf(v: number): number {
  const f = new Float32Array([v]);
  const u = new Uint32Array(f.buffer)[0]!;
  const sign = (u >>> 16) & 0x8000;
  const exp = ((u >>> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return sign;
  const mant = u & 0x7fffff;
  let h = sign | (exp << 10) | (mant >>> 13);
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (h & 1))) h++;
  return h;
}
const packHalf = (a: Float32Array) => {
  const out = new Uint32Array(a.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = toHalf(a[2 * i]!) | (toHalf(a[2 * i + 1]!) << 16);
  return out;
};
const unpackHalf = (w: Uint32Array) => f16ToF32(new Uint8Array(w.buffer, w.byteOffset, w.length * 4));

async function main() {
  // ?backend=cpu checks the CPU backend's TypeScript kernels against the same references.
  const onCpu = new URLSearchParams(location.search).get("backend") === "cpu";
  const be: Backend = onCpu ? new CpuBackend() : await createWebGPUBackend();
  document.getElementById("device")!.textContent = `${onCpu ? "CPU backend · " : ""}features: ${[...be.limits.features].join(", ") || "none"}${be.limits.subgroupMinSize ? ` · subgroup size ≥ ${be.limits.subgroupMinSize}` : ""} · max binding ${(be.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`;
  const step = be.alloc(32, "storage", "step");
  const results: Result[] = [];

  const upload = (data: Float32Array | Uint32Array, shape: number[]): Tensor => {
    const buf = be.alloc(data.byteLength, "storage", "in");
    be.write(buf, 0, data);
    return { shape, dtype: data instanceof Uint32Array ? "u32" : "f32", buf };
  };
  const uploadQ4 = (data: Float32Array, shape: number[]): Tensor => {
    const p = quantizeQ4(data);
    const t = upload(p.nibbles, shape);
    return { ...t, dtype: "q4", scales: upload(p.scales, [p.scales.length]).buf };
  };
  const uploadQ8 = (data: Float32Array, shape: number[]): Tensor => {
    const p = quantizeQ8(data);
    const t = upload(p.bytes, shape);
    return { ...t, dtype: "q8", scales: upload(p.scales, [p.scales.length]).buf };
  };
  /** Affine planes (as GGUF Q4_1/Q4_K load into q4, Q5_1/Q5_K into q8): codes 0…15 or 0…31 between each block's min and max. */
  const affine = (wd: "q4a" | "q8a", data: Float32Array) => {
    const top = wd === "q4a" ? 15 : 31;
    const scales = new Float32Array(data.length / 32);
    const mins = new Float32Array(data.length / 32);
    const codes = new Uint8Array(data.length);
    for (let b = 0; b < scales.length; b++) {
      const blk = data.subarray(b * 32, b * 32 + 32);
      const lo = Math.min(...blk);
      const s = Math.fround((Math.max(...blk) - lo) / top);
      scales[b] = s;
      mins[b] = Math.fround(lo);
      blk.forEach((v, j) => (codes[b * 32 + j] = s ? Math.min(top, Math.round((v - lo) / s)) : 0));
    }
    if (wd === "q8a") return { q8: { bytes: new Uint32Array(codes.buffer), scales, mins } };
    const nibbles = new Uint32Array(data.length / 8);
    codes.forEach((c, e) => (nibbles[e >> 3]! |= c << ((e & 7) * 4)));
    return { q4: { nibbles, scales, mins } };
  };
  type WD = "f32" | "q4" | "q8" | "q4a" | "q8a";
  const uploadW = (wd: WD, data: Float32Array, shape: number[]): Tensor => {
    if (wd === "q4a" || wd === "q8a") {
      const a = affine(wd, data);
      const planes = a.q4 ? { data: a.q4.nibbles, scales: a.q4.scales, mins: a.q4.mins } : { data: a.q8!.bytes, scales: a.q8!.scales, mins: a.q8!.mins };
      const t = upload(planes.data, shape);
      return { ...t, dtype: wd === "q4a" ? "q4" : "q8", scales: upload(planes.scales, [planes.scales.length]).buf, mins: upload(planes.mins, [planes.mins.length]).buf };
    }
    return wd === "q4" ? uploadQ4(data, shape) : wd === "q8" ? uploadQ8(data, shape) : upload(data, shape);
  };
  /** The values the GPU should see for weights stored as `wd`. */
  const asStored = (wd: WD, data: Float32Array) => {
    if (wd === "q4a" || wd === "q8a") {
      const a = affine(wd, data);
      return a.q4 ? dequantQ4(a.q4) : dequantQ8(a.q8!);
    }
    return wd === "q4" ? dequantQ4(quantizeQ4(data)) : wd === "q8" ? dequantQ8(quantizeQ8(data)) : data;
  };
  const download = async (buf: Buf, n: number) => new Float32Array(await be.readback(buf, 0, n * 4));
  const run = async (s: StepParams, build: (ctx: Ctx) => void) => {
    be.write(step, 0, packStep(s));
    const ctx = new Ctx(be, step, "check");
    build(ctx);
    be.submit(ctx.dispatches);
    return ctx;
  };
  const check = (kernel: string, name: string, got: Float32Array, want: Float32Array, tol: number) => {
    const error = ref.maxAbsDiff(got, want);
    results.push({ kernel, case: name, error, tol, pass: error <= tol && got.length === want.length });
  };
  const S0: StepParams = { pos0: 0, lastRow: 0, seed: 1, temperature: 0 };

  for (const [M, N, K] of [[1, 70, 96], [5, 33, 256], [19, 61, 512], [9, 50, 576], [1, 45, 1536], [1, 20, 2304], [1, 40000, 64]] as const) {
    const x = randn(M * K);
    const w = randn(N * K);
    let y!: Tensor;
    await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [M, K]), upload(w, [N, K]))));
    check("matmul/f32", `M${M} N${N} K${K}`, await download(y.buf, M * N), ref.matmulRef(x, w, M, N, K), 1e-3 * Math.sqrt(K));

    let yq!: Tensor;
    await run(S0, (ctx) => (yq = ops.linear(ctx, upload(x, [M, K]), uploadQ4(w, [N, K]))));
    check("matmul/q4", `M${M} N${N} K${K}`, await download(yq.buf, M * N), ref.matmulRef(x, dequantQ4(quantizeQ4(w)), M, N, K), 1e-3 * Math.sqrt(K));
    await run(S0, (ctx) => (yq = ops.linear(ctx, upload(x, [M, K]), uploadQ8(w, [N, K]))));
    check("matmul/q8", `M${M} N${N} K${K}`, await download(yq.buf, M * N), ref.matmulRef(x, dequantQ8(quantizeQ8(w)), M, N, K), 1e-3 * Math.sqrt(K));
  }

  // Prefill GEMM (64 rows and up): every weight format and epilogue, with ragged M and N edges.
  for (const [M, N, K] of [[64, 64, 32], [77, 70, 96], [128, 1000, 640], [200, 33, 256]] as const) {
    for (const wd of ["f32", "q4", "q8", "q4a", "q8a"] as const) {
      const x = randn(M * K);
      const w = randn(N * K);
      const deq = asStored(wd, w);
      const tag = `${wd} M${M} N${N} K${K}`;
      const plain = ref.matmulRef(x, deq, M, N, K);
      let y!: Tensor;
      await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [M, K]), uploadW(wd, w, [N, K]))));
      check("gemm", tag, await download(y.buf, M * N), plain, 1e-3 * Math.sqrt(K));
      const res = randn(M * N);
      const bias = randn(N);
      const rt = upload(res, [M, N]);
      await run(S0, (ctx) => ops.linear(ctx, upload(x, [M, K]), uploadW(wd, w, [N, K]), { residual: rt, bias: upload(bias, [N]) }));
      check("gemm+bias+add", tag, await download(rt.buf, M * N), res.map((r, j) => r + plain[j]! + bias[j % N]!), 1e-3 * Math.sqrt(K));
      if (N % 2 === 0) {
        const I = N / 2;
        for (const act of ["silu", "gelu"] as const) {
          const want = new Float32Array(M * I);
          for (let m = 0; m < M; m++) for (let i = 0; i < I; i++) {
            const g = plain[m * N + i]!;
            want[m * I + i] = (act === "gelu" ? ref.geluTanh(g) : g / (1 + Math.exp(-g))) * plain[m * N + I + i]!;
          }
          await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [M, K]), uploadW(wd, w, [N, K]), { [act]: true })));
          check(`gemm+${act}`, tag, await download(y.buf, M * I), want, 2e-3 * Math.sqrt(K));
          const gb = randn(N);
          const wantB = new Float32Array(M * I);
          for (let m = 0; m < M; m++) for (let i = 0; i < I; i++) {
            const g = plain[m * N + i]! + gb[i]!;
            wantB[m * I + i] = (act === "gelu" ? ref.geluTanh(g) : g / (1 + Math.exp(-g))) * (plain[m * N + I + i]! + gb[I + i]!);
          }
          await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [M, K]), uploadW(wd, w, [N, K]), { [act]: true, bias: upload(gb, [N]) })));
          check(`gemm+bias+${act}`, tag, await download(y.buf, M * I), wantB, 2e-3 * Math.sqrt(K));
        }
      }
    }
  }

  // Decode GEMV (one token): every weight format, prologue and epilogue; K past one staged chunk
  // (2048) and ragged in the last one, N ragged against the 64 rows of a workgroup.
  for (const [N, K] of [[70, 96], [1000, 640], [66, 2304], [130, 4128], [16400, 128]] as const) {
    for (const wd of ["f32", "q4", "q8", "q4a", "q8a"] as const) {
      const x = randn(K);
      const w = randn(N * K);
      const nw = randn(K);
      const deq = asStored(wd, w);
      const tag = `${wd} N${N} K${K}`;
      const tol = 1e-3 * Math.sqrt(K);
      const plain = ref.matmulRef(x, deq, 1, N, K);
      const normed = ref.matmulRef(ref.rmsnormRef(x, nw, K, 1e-5), deq, 1, N, K);
      const norm = () => ({ w: upload(nw, [K]), eps: 1e-5 });
      let y!: Tensor;
      await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [1, K]), uploadW(wd, w, [N, K]))));
      check("gemv", tag, await download(y.buf, N), plain, tol);
      await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [1, K]), uploadW(wd, w, [N, K]), { norm: norm() })));
      check("gemv+norm", tag, await download(y.buf, N), normed, tol);
      const res = randn(N);
      const bias = randn(N);
      const rt = upload(res, [1, N]);
      await run(S0, (ctx) => ops.linear(ctx, upload(x, [1, K]), uploadW(wd, w, [N, K]), { residual: rt, bias: upload(bias, [N]) }));
      check("gemv+bias+add", tag, await download(rt.buf, N), res.map((r, j) => r + plain[j]! + bias[j]!), tol);
      const I = N / 2;
      for (const act of ["silu", "gelu"] as const) {
        const f = (g: number) => (act === "gelu" ? ref.geluTanh(g) : g / (1 + Math.exp(-g)));
        const gb = randn(N);
        await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [1, K]), uploadW(wd, w, [N, K]), { [act]: true, norm: norm() })));
        check(`gemv+norm+${act}`, tag, await download(y.buf, I), Float32Array.from({ length: I }, (_, i) => f(normed[i]!) * normed[I + i]!), 2 * tol);
        await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [1, K]), uploadW(wd, w, [N, K]), { [act]: true, bias: upload(gb, [N]) })));
        check(`gemv+bias+${act}`, tag, await download(y.buf, I), Float32Array.from({ length: I }, (_, i) => f(plain[i]! + gb[i]!) * (plain[I + i]! + gb[I + i]!)), 2 * tol);
      }
    }
  }

  // Fused matmul variants: every lanes setting, with norm prologue, SiLU gating, residual add and bias.
  for (const lanes of [8, 16, 32] as const) {
    for (const [M, I, K] of [[1, 40, 96], [5, 24, 256], [19, 70, 576]] as const) {
      for (const wd of ["f32", "q4", "q8", "q4a", "q8a"] as const) {
        const x = randn(M * K);
        const nw = randn(K);
        const wgu = randn(2 * I * K);
        const deq = asStored(wd, wgu);
        const W = (t: Float32Array, shape: number[]) => uploadW(wd, t, shape);
        const xn = ref.rmsnormRef(x, nw, K, 1e-5);
        const gu = ref.matmulRef(xn, deq, M, 2 * I, K);
        const want = new Float32Array(M * I);
        for (let m = 0; m < M; m++) for (let i = 0; i < I; i++) {
          const g = gu[m * 2 * I + i]!;
          want[m * I + i] = (g / (1 + Math.exp(-g))) * gu[m * 2 * I + I + i]!;
        }
        let y!: Tensor;
        await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { norm: { w: upload(nw, [K]), eps: 1e-5 }, silu: true, lanes })));
        const tag = `${wd} l${lanes} M${M} K${K}`;
        check("matmul+norm+silu", tag, await download(y.buf, M * I), want, 2e-3 * Math.sqrt(K));

        let yn!: Tensor;
        await run(S0, (ctx) => (yn = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { norm: { w: upload(nw, [K]), eps: 1e-5 }, lanes })));
        check("matmul+norm", tag, await download(yn.buf, M * 2 * I), gu, 1e-3 * Math.sqrt(K));
        const guPlain = ref.matmulRef(x, deq, M, 2 * I, K);
        const wantS = new Float32Array(M * I);
        for (let m = 0; m < M; m++) for (let i = 0; i < I; i++) {
          const g = guPlain[m * 2 * I + i]!;
          wantS[m * I + i] = (g / (1 + Math.exp(-g))) * guPlain[m * 2 * I + I + i]!;
        }
        let ys!: Tensor;
        await run(S0, (ctx) => (ys = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { silu: true, lanes })));
        check("matmul+silu", tag, await download(ys.buf, M * I), wantS, 2e-3 * Math.sqrt(K));
        // Gemma's GELU (tanh) gate, with and without the fused norm.
        const gated = (gu2: Float32Array) => {
          const out = new Float32Array(M * I);
          for (let m = 0; m < M; m++) for (let i = 0; i < I; i++) out[m * I + i] = ref.geluTanh(gu2[m * 2 * I + i]!) * gu2[m * 2 * I + I + i]!;
          return out;
        };
        let yge!: Tensor;
        await run(S0, (ctx) => (yge = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { norm: { w: upload(nw, [K]), eps: 1e-5 }, gelu: true, lanes })));
        check("matmul+norm+gelu", tag, await download(yge.buf, M * I), gated(gu), 2e-3 * Math.sqrt(K));
        await run(S0, (ctx) => (yge = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { gelu: true, lanes })));
        check("matmul+gelu", tag, await download(yge.buf, M * I), gated(guPlain), 2e-3 * Math.sqrt(K));

        if (lanes <= (be.limits.subgroupMinSize ?? 0)) {
          let yg!: Tensor;
          await run(S0, (ctx) => (yg = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { norm: { w: upload(nw, [K]), eps: 1e-5 }, silu: true, lanes, subgroups: true })));
          check("matmul+norm+silu/sg", tag, await download(yg.buf, M * I), want, 2e-3 * Math.sqrt(K));
          let yp!: Tensor;
          await run(S0, (ctx) => (yp = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { lanes, subgroups: true })));
          check("matmul/sg", tag, await download(yp.buf, M * 2 * I), guPlain, 1e-3 * Math.sqrt(K));
        }
        if (be.limits.features.has("shader-f16")) {
          let yh!: Tensor;
          await run(S0, (ctx) => (yh = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { norm: { w: upload(nw, [K]), eps: 1e-5 }, silu: true, lanes, f16: true })));
          check("matmul+norm+silu/f16", tag, await download(yh.buf, M * I), want, 2e-2 * Math.sqrt(K));
        }
        const res = randn(M * I);
        const bias = randn(I);
        const w2 = randn(I * K);
        const deq2 = asStored(wd, w2);
        const plain = ref.matmulRef(x, deq2, M, I, K);
        const wantAdd = res.map((r, j) => r + plain[j]! + bias[j % I]!);
        const rt = upload(res, [M, I]);
        await run(S0, (ctx) => ops.linear(ctx, upload(x, [M, K]), W(w2, [I, K]), { residual: rt, bias: upload(bias, [I]), lanes }));
        check("matmul+bias+add", tag, await download(rt.buf, M * I), wantAdd, 1e-3 * Math.sqrt(K));
      }
    }
  }

  for (const interleaved of [false, true]) {
    const [T, H, Hkv, D, pos0, maxSeq] = [3, 4, 2, 16, 5, 12];
    const W = (H + 2 * Hkv) * D;
    const qkv = randn(T * W);
    const inv = Float32Array.from({ length: D / 2 }, (_, i) => 1 / 10000 ** ((2 * i) / D));
    const q = new Float32Array(T * H * D);
    const k = new Float32Array(T * Hkv * D);
    const v = new Float32Array(T * Hkv * D);
    for (let t = 0; t < T; t++) {
      q.set(qkv.subarray(t * W, t * W + H * D), t * H * D);
      k.set(qkv.subarray(t * W + H * D, t * W + (H + Hkv) * D), t * Hkv * D);
      v.set(qkv.subarray(t * W + (H + Hkv) * D, (t + 1) * W), t * Hkv * D);
    }
    const wantQ = ref.ropeRef(q, T, H, D, pos0, inv, interleaved);
    const wantK = new Float32Array(maxSeq * Hkv * D);
    wantK.set(ref.ropeRef(k, T, Hkv, D, pos0, inv, interleaved), pos0 * Hkv * D);
    const wantV = new Float32Array(maxSeq * Hkv * D);
    wantV.set(v, pos0 * Hkv * D);
    const kc = upload(new Float32Array(maxSeq * Hkv * D), [0]);
    const vc = upload(new Float32Array(maxSeq * Hkv * D), [0]);
    let qo!: Tensor;
    await run({ ...S0, pos0 }, (ctx) => (qo = ops.ropeKv(ctx, upload(qkv, [T, W]), kc.buf, vc.buf, { T, H, Hkv, D, maxSeq }, upload(inv, [D / 2]), interleaved)));
    const tag = interleaved ? "interleaved" : "rotate-half";
    check("rope_kv", `q ${tag}`, await download(qo.buf, q.length), wantQ, 1e-4);
    check("rope_kv", `k cache ${tag}`, await download(kc.buf, wantK.length), wantK, 1e-4);
    check("rope_kv", `v cache ${tag}`, await download(vc.buf, wantV.length), wantV, 0);
    const kc16 = upload(new Uint32Array((maxSeq * Hkv * D) / 2), [0]);
    const vc16 = upload(new Uint32Array((maxSeq * Hkv * D) / 2), [0]);
    await run({ ...S0, pos0 }, (ctx) => ops.ropeKv(ctx, upload(qkv, [T, W]), kc16.buf, vc16.buf, { T, H, Hkv, D, maxSeq, kv: "f16" }, upload(inv, [D / 2]), interleaved));
    const words = async (b: Buf, n: number) => new Uint32Array(await be.readback(b, 0, n * 4));
    check("rope_kv/f16", `k cache ${tag}`, unpackHalf(await words(kc16.buf, wantK.length / 2)), wantK, 2e-3);
    check("rope_kv/f16", `v cache ${tag}`, unpackHalf(await words(vc16.buf, wantV.length / 2)), wantV, 2e-3);
  }

  // Per-head q/k norm + RoPE + cache write after a stacked qkv matmul (Qwen3, Gemma 3), including a
  // ring cache that wraps (positions 13…15 of 14 slots land in 13, 0, 1).
  for (const [T, H, Hkv, D, pos0, maxSeq, slots, interleaved] of [[3, 4, 2, 64, 5, 12, 12, false], [3, 4, 2, 64, 5, 12, 12, true], [2, 8, 4, 256, 9, 32, 32, false], [3, 2, 1, 256, 13, 64, 14, false]] as const) {
    const W = (H + 2 * Hkv) * D;
    const qkv = randn(T * W);
    const qw = Float32Array.from(randn(D), (x) => 1 + x);
    const kw = Float32Array.from(randn(D), (x) => 1 + x);
    const eps = 1e-6;
    const inv = Float32Array.from({ length: D / 2 }, (_, i) => 1 / 10000 ** ((2 * i) / D));
    const q = new Float32Array(T * H * D);
    const k = new Float32Array(T * Hkv * D);
    const v = new Float32Array(T * Hkv * D);
    for (let t = 0; t < T; t++) {
      q.set(qkv.subarray(t * W, t * W + H * D), t * H * D);
      k.set(qkv.subarray(t * W + H * D, t * W + (H + Hkv) * D), t * Hkv * D);
      v.set(qkv.subarray(t * W + (H + Hkv) * D, (t + 1) * W), t * Hkv * D);
    }
    const wantQ = ref.ropeRef(ref.rmsnormRef(q, qw, D, eps), T, H, D, pos0, inv, interleaved);
    const kr = ref.ropeRef(ref.rmsnormRef(k, kw, D, eps), T, Hkv, D, pos0, inv, interleaved);
    const row = Hkv * D;
    const wantK = new Float32Array(slots * row);
    const wantV = new Float32Array(slots * row);
    for (let t = 0; t < T; t++) {
      wantK.set(kr.subarray(t * row, (t + 1) * row), ((pos0 + t) % slots) * row);
      wantV.set(v.subarray(t * row, (t + 1) * row), ((pos0 + t) % slots) * row);
    }
    const tag = `T${T} H${H}/${Hkv} D${D} pos${pos0} slots${slots} ${interleaved ? "interleaved" : "rotate-half"}`;
    const shape = { T, H, Hkv, D, maxSeq, slots };
    for (const kv of ["f32", "f16"] as const) {
      const kc = upload(kv === "f16" ? new Uint32Array((slots * row) / 2) : new Float32Array(slots * row), [0]);
      const vc = upload(kv === "f16" ? new Uint32Array((slots * row) / 2) : new Float32Array(slots * row), [0]);
      let qo!: Tensor;
      await run({ ...S0, pos0 }, (ctx) => (qo = ops.qkNormRopeKv(ctx, upload(qkv, [T, W]), upload(qw, [D]), upload(kw, [D]), eps, kc.buf, vc.buf, { ...shape, kv }, upload(inv, [D / 2]), interleaved)));
      const cache = async (b: Buf) => (kv === "f16" ? unpackHalf(new Uint32Array(await be.readback(b, 0, slots * row * 2))) : await download(b, slots * row));
      // Normalised values reach about 10 (weights 1 + N(0, 1)), so f32 rounding alone is ~1e-4.
      const tol = kv === "f16" ? 2e-2 : 5e-4;
      check(`qknorm_rope_kv/${kv}`, `q ${tag}`, await download(qo.buf, q.length), wantQ, 5e-4);
      check(`qknorm_rope_kv/${kv}`, `k cache ${tag}`, await cache(kc.buf), wantK, tol);
      check(`qknorm_rope_kv/${kv}`, `v cache ${tag}`, await cache(vc.buf), wantV, kv === "f16" ? 2e-3 : 0);
    }
  }

  {
    const [R, D] = [5, 300];
    const x = randn(R * D);
    const w = randn(D);
    let y!: Tensor;
    await run(S0, (ctx) => (y = ops.rmsnorm(ctx, upload(x, [R, D]), upload(w, [D]), 1e-6)));
    check("rmsnorm", `R${R} D${D}`, await download(y.buf, R * D), ref.rmsnormRef(x, w, D, 1e-6), 1e-4);
  }

  for (const interleaved of [false, true]) {
    const [T, H, D, pos0] = [3, 2, 16, 37];
    const x = randn(T * H * D);
    const inv = Float32Array.from({ length: D / 2 }, (_, i) => 1 / 10000 ** ((2 * i) / D));
    const xt = upload(x, [T, H * D]);
    await run({ ...S0, pos0 }, (ctx) => ops.rope(ctx, xt, T, H, D, upload(inv, [D / 2]), interleaved));
    check("rope", interleaved ? "interleaved" : "rotate-half", await download(xt.buf, x.length), ref.ropeRef(x, T, H, D, pos0, inv, interleaved), 1e-4);
  }

  // Prefill (T > 1) cases run both kernels: tiled (default) and one workgroup per query row.
  for (const [T, H, Hkv, D, pos0, maxSeq] of [[3, 4, 2, 16, 5, 16], [1, 9, 3, 64, 300, 512], [1, 9, 3, 64, 0, 200], [1, 8, 8, 128, 63, 128], [1, 8, 2, 64, 64, 256], [1, 4, 4, 32, 1999, 2048], [4, 4, 1, 128, 61, 128], [5, 9, 3, 64, 600, 1024], [3, 4, 2, 32, 1790, 2048], [2, 2, 2, 256, 255, 512], [64, 9, 3, 64, 1000, 2048], [17, 6, 2, 96, 40, 128], [33, 8, 2, 128, 300, 512], [9, 4, 4, 32, 70, 128]] as const) {
    const q = randn(T * H * D);
    const kc = randn(maxSeq * Hkv * D);
    const vc = randn(maxSeq * Hkv * D);
    const kh = unpackHalf(packHalf(kc));
    const vh = unpackHalf(packHalf(vc));
    const want = ref.attentionRef(q, kc, vc, { T, H, Hkv, D, pos0 });
    const wantH = ref.attentionRef(q, kh, vh, { T, H, Hkv, D, pos0 });
    for (const kind of T > 1 ? (["tiled", "rows"] as const) : (["tiled"] as const)) {
      ops.setPrefillAttention(kind);
      const tag = `T${T} H${H}/${Hkv} D${D} pos${pos0}${T > 1 ? ` ${kind}` : ""}`;
      let o!: Tensor;
      await run({ ...S0, pos0 }, (ctx) => (o = ops.attention(ctx, upload(q, [T, H * D]), upload(kc, [0]).buf, upload(vc, [0]).buf, { T, H, Hkv, D, maxSeq })));
      check("attention", tag, await download(o.buf, q.length), want, 1e-4);
      await run({ ...S0, pos0 }, (ctx) => (o = ops.attention(ctx, upload(q, [T, H * D]), upload(packHalf(kc), [0]).buf, upload(packHalf(vc), [0]).buf, { T, H, Hkv, D, maxSeq, kv: "f16" })));
      check("attention/f16", tag, await download(o.buf, q.length), wantH, 1e-4);
    }
    ops.setPrefillAttention("tiled");
  }

  // Sliding-window attention with an explicit score scale (Gemma 3: window 512, scale 1/16, D 256).
  // Windows that start mid-split and mid-tile, are longer than the context, and decode far past them.
  for (const [T, H, Hkv, D, pos0, maxSeq, window] of [[1, 4, 1, 256, 700, 1024, 512], [1, 4, 1, 256, 100, 1024, 512], [1, 9, 3, 64, 300, 512, 37], [5, 4, 1, 256, 600, 1024, 512], [33, 8, 2, 128, 300, 512, 64], [17, 6, 2, 96, 40, 128, 9], [64, 9, 3, 64, 1000, 2048, 100]] as const) {
    const q = randn(T * H * D);
    const kc = randn(maxSeq * Hkv * D);
    const vc = randn(maxSeq * Hkv * D);
    const scale = 1 / 16;
    const want = ref.attentionRef(q, kc, vc, { T, H, Hkv, D, pos0, scale, window });
    const wantH = ref.attentionRef(q, unpackHalf(packHalf(kc)), unpackHalf(packHalf(vc)), { T, H, Hkv, D, pos0, scale, window });
    for (const kind of T > 1 ? (["tiled", "rows"] as const) : (["tiled"] as const)) {
      ops.setPrefillAttention(kind);
      const tag = `T${T} H${H}/${Hkv} D${D} pos${pos0} win${window}${T > 1 ? ` ${kind}` : ""}`;
      let o!: Tensor;
      await run({ ...S0, pos0 }, (ctx) => (o = ops.attention(ctx, upload(q, [T, H * D]), upload(kc, [0]).buf, upload(vc, [0]).buf, { T, H, Hkv, D, maxSeq, scale, window })));
      check("attention/window", tag, await download(o.buf, q.length), want, 1e-4);
      await run({ ...S0, pos0 }, (ctx) => (o = ops.attention(ctx, upload(q, [T, H * D]), upload(packHalf(kc), [0]).buf, upload(packHalf(vc), [0]).buf, { T, H, Hkv, D, maxSeq, kv: "f16", scale, window })));
      check("attention/window/f16", tag, await download(o.buf, q.length), wantH, 1e-4);
    }
    ops.setPrefillAttention("tiled");
  }

  // Ring caches (sliding layers keep window + slack slots, key j in slot j % slots): filled from the
  // same keys as a full cache at positions far past the ring size, they must give the same output.
  for (const [T, H, Hkv, D, pos0, maxSeq, window, slots] of [[1, 4, 1, 256, 1700, 2048, 512, 640], [1, 9, 3, 64, 3001, 4096, 300, 428], [64, 4, 1, 256, 1500, 2048, 512, 640], [33, 8, 2, 128, 900, 2048, 256, 384], [5, 9, 3, 64, 3000, 4096, 300, 428]] as const) {
    const q = randn(T * H * D);
    const row = Hkv * D;
    const kc = randn(maxSeq * row);
    const vc = randn(maxSeq * row);
    const scale = 1 / 16;
    const ring = (c: Float32Array) => {
      const r = new Float32Array(slots * row);
      for (let j = pos0 + T - slots; j < pos0 + T; j++) r.set(c.subarray(j * row, (j + 1) * row), (j % slots) * row);
      return r;
    };
    const want = ref.attentionRef(q, kc, vc, { T, H, Hkv, D, pos0, scale, window });
    const wantH = ref.attentionRef(q, unpackHalf(packHalf(kc)), unpackHalf(packHalf(vc)), { T, H, Hkv, D, pos0, scale, window });
    for (const kind of T > 1 ? (["tiled", "rows"] as const) : (["tiled"] as const)) {
      ops.setPrefillAttention(kind);
      const tag = `T${T} H${H}/${Hkv} D${D} pos${pos0} win${window} ring${slots}${T > 1 ? ` ${kind}` : ""}`;
      const shape = { T, H, Hkv, D, maxSeq, scale, window, slots };
      let o!: Tensor;
      await run({ ...S0, pos0 }, (ctx) => (o = ops.attention(ctx, upload(q, [T, H * D]), upload(ring(kc), [0]).buf, upload(ring(vc), [0]).buf, shape)));
      check("attention/ring", tag, await download(o.buf, q.length), want, 1e-4);
      await run({ ...S0, pos0 }, (ctx) => (o = ops.attention(ctx, upload(q, [T, H * D]), upload(packHalf(ring(kc)), [0]).buf, upload(packHalf(ring(vc)), [0]).buf, { ...shape, kv: "f16" })));
      check("attention/ring/f16", tag, await download(o.buf, q.length), wantH, 1e-4);
    }
    ops.setPrefillAttention("tiled");
  }

  {
    // A write that wraps around a ring: positions 14…18 of a 16-slot cache land in slots 14, 15, 0, 1, 2.
    const [T, row, pos0, maxSeq, slots] = [5, 8, 14, 64, 16];
    const k = randn(T * row);
    const v = randn(T * row);
    const kc = upload(new Float32Array(slots * row), [slots, row]);
    const vc = upload(new Float32Array(slots * row), [slots, row]);
    await run({ ...S0, pos0 }, (ctx) => ops.kvWrite(ctx, upload(k, [T, row]), upload(v, [T, row]), kc.buf, vc.buf, T, maxSeq, "f32", slots));
    const wantK = new Float32Array(slots * row);
    const wantV = new Float32Array(slots * row);
    for (let t = 0; t < T; t++) {
      wantK.set(k.subarray(t * row, (t + 1) * row), ((pos0 + t) % slots) * row);
      wantV.set(v.subarray(t * row, (t + 1) * row), ((pos0 + t) % slots) * row);
    }
    check("kv_write/ring", "k", await download(kc.buf, wantK.length), wantK, 0);
    check("kv_write/ring", "v", await download(vc.buf, wantV.length), wantV, 0);
  }

  {
    const [T, row, pos0, maxSeq] = [2, 8, 3, 8];
    const k = randn(T * row);
    const v = randn(T * row);
    const kc = upload(new Float32Array(maxSeq * row), [maxSeq, row]);
    const vc = upload(new Float32Array(maxSeq * row), [maxSeq, row]);
    await run({ ...S0, pos0 }, (ctx) => ops.kvWrite(ctx, upload(k, [T, row]), upload(v, [T, row]), kc.buf, vc.buf, T, maxSeq));
    const wantK = new Float32Array(maxSeq * row);
    wantK.set(k, pos0 * row);
    const wantV = new Float32Array(maxSeq * row);
    wantV.set(v, pos0 * row);
    check("kv_write", "k", await download(kc.buf, wantK.length), wantK, 0);
    check("kv_write", "v", await download(vc.buf, wantV.length), wantV, 0);
    const kc16 = upload(new Uint32Array((maxSeq * row) / 2), [0]);
    const vc16 = upload(new Uint32Array((maxSeq * row) / 2), [0]);
    await run({ ...S0, pos0 }, (ctx) => ops.kvWrite(ctx, upload(k, [T, row]), upload(v, [T, row]), kc16.buf, vc16.buf, T, maxSeq, "f16"));
    const words = async (b: Buf, n: number) => new Uint32Array(await be.readback(b, 0, n * 4));
    check("kv_write/f16", "k", unpackHalf(await words(kc16.buf, wantK.length / 2)), wantK, 2e-3);
    check("kv_write/f16", "v", unpackHalf(await words(vc16.buf, wantV.length / 2)), wantV, 2e-3);
  }

  {
    const n = 64 * 70000; // spills into grid.y
    const g = randn(n);
    const u = randn(n);
    let o!: Tensor;
    await run(S0, (ctx) => (o = ops.siluMul(ctx, upload(g, [n]), upload(u, [n]))));
    check("silu_mul", `n=${n} (2-D grid)`, await download(o.buf, n), ref.siluMulRef(g, u), 1e-5);
    await run(S0, (ctx) => (o = ops.siluMul(ctx, upload(g, [n]), upload(u, [n]), "gelu")));
    check("gelu_mul", `n=${n} (2-D grid)`, await download(o.buf, n), ref.geluMulRef(g, u), 1e-5);
  }

  {
    const [R, C] = [4, 24];
    const a = randn(R * C);
    const b = randn(C);
    const at = upload(a, [R, C]);
    await run(S0, (ctx) => ops.addInPlace(ctx, at, upload(b, [C])));
    check("add", "bias broadcast", await download(at.buf, a.length), a.map((v, i) => v + b[i % C]!), 1e-6);
  }

  {
    const [V, D] = [50, 64];
    const table = randn(V * D);
    const ids = Uint32Array.from([3, 0, 49, 3]);
    const want = new Float32Array(ids.length * D);
    const wantQ = new Float32Array(ids.length * D);
    const deq = dequantQ4(quantizeQ4(table));
    ids.forEach((id, t) => {
      want.set(table.subarray(id * D, (id + 1) * D), t * D);
      wantQ.set(deq.subarray(id * D, (id + 1) * D), t * D);
    });
    let o!: Tensor;
    await run(S0, (ctx) => (o = ops.embed(ctx, upload(table, [V, D]), upload(ids, [4]).buf, ids.length)));
    check("embed/f32", `V${V} D${D}`, await download(o.buf, want.length), want, 0);
    await run(S0, (ctx) => (o = ops.embed(ctx, uploadQ4(table, [V, D]), upload(ids, [4]).buf, ids.length)));
    check("embed/q4", `V${V} D${D}`, await download(o.buf, wantQ.length), wantQ, 1e-6);
    await run(S0, (ctx) => (o = ops.embed(ctx, uploadQ4(table, [V, D]), upload(ids, [4]).buf, ids.length, 25.25)));
    check("embed/q4/scaled", `V${V} D${D} ×25.25`, await download(o.buf, wantQ.length), wantQ.map((v) => v * 25.25), 1e-4);
    const deq8 = dequantQ8(quantizeQ8(table));
    const want8 = new Float32Array(ids.length * D);
    ids.forEach((id, t) => want8.set(deq8.subarray(id * D, (id + 1) * D), t * D));
    await run(S0, (ctx) => (o = ops.embed(ctx, uploadQ8(table, [V, D]), upload(ids, [4]).buf, ids.length, 25.25)));
    check("embed/q8/scaled", `V${V} D${D} ×25.25`, await download(o.buf, want8.length), want8.map((v) => v * 25.25), 1e-4);
    for (const wd of ["q4a", "q8a"] as const) {
      const deqA = asStored(wd, table);
      const wantA = new Float32Array(ids.length * D);
      ids.forEach((id, t) => wantA.set(deqA.subarray(id * D, (id + 1) * D), t * D));
      await run(S0, (ctx) => (o = ops.embed(ctx, uploadW(wd, table, [V, D]), upload(ids, [4]).buf, ids.length, 25.25)));
      check(`embed/${wd}/scaled`, `V${V} D${D} ×25.25`, await download(o.buf, wantA.length), wantA.map((v) => v * 25.25), 1e-4);
    }
  }

  {
    const [T, D] = [4, 40];
    const x = randn(T * D);
    let o!: Tensor;
    await run({ ...S0, lastRow: 2 }, (ctx) => (o = ops.copyRow(ctx, upload(x, [T, D]))));
    check("copy_row", "lastRow=2", await download(o.buf, D), x.slice(2 * D, 3 * D), 0);
  }

  {
    const V = 5000;
    const logits = randn(V);
    const out = be.alloc(4, "storage", "out");
    const noneSeen = upload(new Uint32Array(Math.ceil(V / 32)), [0]).buf;
    // Flag word 0 = no grammar mask.
    const anyAllowed = upload(new Uint32Array(1 + Math.ceil(V / 32)), [0]).buf;
    /** A grammar mask allowing only `ids`. */
    const only = (ids: number[], n = V) => {
      const m = new Uint32Array(1 + Math.ceil(n / 32));
      m[0] = 1;
      for (const id of ids) m[1 + (id >>> 5)] |= 1 << (id & 31);
      return upload(m, [0]).buf;
    };
    const lt = upload(logits, [1, V]);
    await run(S0, (ctx) => ops.sample(ctx, lt, out, noneSeen, anyAllowed));
    const got = new Uint32Array(await be.readback(out, 0, 4))[0]!;
    const want = ref.argmaxRef(logits);
    results.push({ kernel: "sample", case: "greedy", error: got === want ? 0 : 1, tol: 0, pass: got === want, note: `${got} (${logits[got]}) vs ${want} (${logits[want]})` });

    // Repetition penalty: with the argmax marked seen and penalty 1e3, greedy picks the runner-up.
    const seenTop = new Uint32Array(Math.ceil(V / 32));
    seenTop[want >>> 5] |= 1 << (want & 31);
    const second = ref.argmaxRef(logits.map((v, i) => (i === want ? -Infinity : v)));
    await run({ ...S0, repetitionPenalty: 1e3 }, (ctx) => ops.sample(ctx, lt, out, upload(seenTop, [0]).buf, anyAllowed));
    const gotRp = new Uint32Array(await be.readback(out, 0, 4))[0]!;
    results.push({ kernel: "sample", case: "repetition penalty (greedy)", error: gotRp === second ? 0 : 1, tol: 0, pass: gotRp === second, note: `${gotRp} vs ${second}` });

    // Grammar mask: greedy picks the best allowed id, even when every allowed logit is far below the max.
    const picks = [7, 1234, 4999, want === 2500 ? 2501 : 2500].filter((i) => i !== want);
    const wantMasked = picks.reduce((a, b) => (logits[b]! > logits[a]! ? b : a));
    await run(S0, (ctx) => ops.sample(ctx, lt, out, noneSeen, only(picks)));
    const gotM = new Uint32Array(await be.readback(out, 0, 4))[0]!;
    results.push({ kernel: "sample", case: "grammar mask (greedy)", error: gotM === wantMasked ? 0 : 1, tol: 0, pass: gotM === wantMasked, note: `${gotM} vs ${wantMasked}` });
    const low = Float32Array.from(logits, (v, i) => (i === 4999 ? -60 : v + 30));
    await run(S0, (ctx) => ops.sample(ctx, upload(low, [1, V]), out, noneSeen, only([4999])));
    const gotLow = new Uint32Array(await be.readback(out, 0, 4))[0]!;
    results.push({ kernel: "sample", case: "grammar mask, single far-below id", error: gotLow === 4999 ? 0 : 1, tol: 0, pass: gotLow === 4999, note: `${gotLow}` });

    // Frequency checks: p over [0, 1, 2, 3] ∝ [1, 1, 1, 3] unless truncated.
    const freqOf = async (logitsArr: number[], params: Partial<StepParams>, token: number, allowedIds?: number[]) => {
      const t = upload(Float32Array.from(logitsArr), [1, logitsArr.length]);
      const allowed = allowedIds ? only(allowedIds, logitsArr.length) : anyAllowed;
      const ctx = await run(S0, (c) => ops.sample(c, t, out, noneSeen, allowed));
      let hits = 0;
      const trials = 400;
      for (let i = 0; i < trials; i++) {
        be.write(step, 0, packStep({ pos0: i, lastRow: 0, seed: 99, temperature: 1, ...params }));
        be.submit(ctx.dispatches);
        if (new Uint32Array(await be.readback(out, 0, 4))[0] === token) hits++;
      }
      return hits / trials;
    };
    const ln3 = Math.log(3);
    const cases: [string, number[], Partial<StepParams>, number, number, number[]?][] = [
      ["gumbel T=1, p=0.5", [0, 0, 0, ln3], {}, 3, 0.5],
      ["top-p 0.4 keeps only the top token", [0, 0, 0, ln3], { topP: 0.4 }, 3, 1],
      ["top-k 1 is greedy", [0, 0, 0, ln3], { topK: 1 }, 3, 1],
      // top-k 2 over ∝ [1, 2, 3, 4] keeps {3, 4}: p(id 3) = 4 / 7.
      ["top-k 2, p=4/7", [0, Math.log(2), ln3, Math.log(4)], { topK: 2 }, 3, 4 / 7],
      // top-p 0.6 over ∝ [1, 2, 3, 4] keeps {4, 3} (mass 0.7): p(id 3) = 4 / 7.
      ["top-p 0.6, p=4/7", [0, Math.log(2), ln3, Math.log(4)], { topP: 0.6 }, 3, 4 / 7],
      // Masking id 2 out of ∝ [1, 1, 1, 3] renormalises over {0, 1, 3}: p(id 3) = 3 / 5.
      ["grammar mask, p=3/5", [0, 0, 0, ln3], {}, 3, 3 / 5, [0, 1, 3]],
      // top-k 2 is applied among allowed ids only: over ∝ [1, 2, 3, 4] with 3 masked out it keeps {1, 2}: p(id 2) = 3 / 5.
      ["grammar mask + top-k 2, p=3/5", [0, Math.log(2), ln3, Math.log(4)], { topK: 2 }, 2, 3 / 5, [0, 1, 2]],
      ["grammar mask + top-p 0.4, p=1", [0, Math.log(2), ln3, Math.log(4)], { topP: 0.4 }, 2, 1, [0, 1, 2]],
    ];
    for (const [name, l, params, token, p, allowedIds] of cases) {
      const f = await freqOf(l, params, token, allowedIds);
      results.push({ kernel: "sample", case: name, error: Math.abs(f - p), tol: 0.08, pass: Math.abs(f - p) <= 0.08, note: `freq ${f.toFixed(3)}` });
    }

    // Per-row sampling (speculative verification): row r must pick what a one-row step at pos0 + r picks.
    const R = 6;
    const rowsLogits = randn(R * V);
    const rowsOut = be.alloc(R * 4, "storage", "rows-out");
    for (const [name, params] of [["greedy", { temperature: 0 }], ["T=1, top-p 0.9", { temperature: 1, topP: 0.9 }], ["T=0.7, top-k 40", { temperature: 0.7, topK: 40 }]] as const) {
      const at = { pos0: 10, lastRow: 0, seed: 7, ...params };
      await run(at, (ctx) => ops.sampleRows(ctx, upload(rowsLogits, [R, V]), rowsOut, noneSeen, anyAllowed));
      const got = [...new Uint32Array(await be.readback(rowsOut, 0, R * 4))];
      const want: number[] = [];
      for (let r = 0; r < R; r++) {
        await run({ ...at, pos0: at.pos0 + r }, (ctx) => ops.sample(ctx, upload(rowsLogits.slice(r * V, (r + 1) * V), [1, V]), out, noneSeen, anyAllowed));
        want.push(new Uint32Array(await be.readback(out, 0, 4))[0]!);
      }
      const same = got.every((id, r) => id === want[r]);
      results.push({ kernel: "sample/rows", case: `${R} rows, ${name}`, error: same ? 0 : 1, tol: 0, pass: same, note: `${got.join(",")} vs ${want.join(",")}` });
    }

    // Per-row grammar masks: row r allows only its own few ids; row 2 is unconstrained.
    const W = 1 + Math.ceil(V / 32);
    // Every id below V: an id past it would set a bit in the next row's mask.
    const rowIds = Array.from({ length: R }, (_, r) => [r * 101 + 3, r * 211 + 17, r * 157 + 4000]);
    const rowMasks = new Uint32Array(R * W);
    for (const [r, ids] of rowIds.entries()) {
      if (r === 2) continue;
      rowMasks[r * W] = 1;
      for (const id of ids) rowMasks[r * W + 1 + (id >>> 5)]! |= 1 << (id & 31);
    }
    for (const [name, params] of [["greedy", { temperature: 0 }], ["T=1", { temperature: 1 }]] as const) {
      const at = { pos0: 3, lastRow: 0, seed: 11, ...params };
      await run(at, (ctx) => ops.sampleRows(ctx, upload(rowsLogits, [R, V]), rowsOut, noneSeen, upload(rowMasks, [0]).buf));
      const got = [...new Uint32Array(await be.readback(rowsOut, 0, R * 4))];
      const want: number[] = [];
      for (let r = 0; r < R; r++) {
        await run({ ...at, pos0: at.pos0 + r }, (ctx) => ops.sample(ctx, upload(rowsLogits.slice(r * V, (r + 1) * V), [1, V]), out, noneSeen, r === 2 ? anyAllowed : only(rowIds[r]!)));
        want.push(new Uint32Array(await be.readback(out, 0, 4))[0]!);
      }
      const same = got.every((id, r) => id === want[r]) && got.every((id, r) => r === 2 || rowIds[r]!.includes(id));
      results.push({ kernel: "sample/rows", case: `${R} rows, per-row masks, ${name}`, error: same ? 0 : 1, tol: 0, pass: same, note: `${got.join(",")} vs ${want.join(",")}` });
    }
  }

  // Load-time conversion to q4 and q8 on the GPU, bit for bit against the CPU converters, over two 8 MB chunks
  // each so the second chunk's offsets are exercised. Scales cover negative, subnormal and zero f16.
  // Float sources mix random values with blocks on a small integer grid, whose codes land exactly on
  // the x / d + 8.5 ties, and all-zero blocks.
  const convertCases: [SrcType, number, number][] = [
    ["Q4_0", 4000, 4128], ["Q6_K", 4500, 2560], ["Q4_K", 3000, 2560], ["Q5_K", 3000, 2560],
    ["Q8_0", 3000, 2560], ["Q4_1", 3000, 2560], ["Q5_0", 3000, 2560], ["Q5_1", 3000, 2560], ["F16", 1700, 2560], ["BF16", 1700, 2560], ["F32", 900, 2560],
  ];
  for (const [type, rows, cols] of convertCases) {
    const bytes = new Uint8Array(rows * rowBytes(type, cols));
    for (let i = 0; i < bytes.length; i++) bytes[i] = (rand() * 128 + 128) & 255;
    const float = type === "F16" || type === "BF16" || type === "F32";
    if (float) {
      const vals = new Float32Array(rows * cols);
      for (let b = 0; b < vals.length / 32; b++) {
        const grid = b % 3 === 0;
        // Power-of-two grid steps keep the grid exact in f16 and bf16.
        const s = grid || b % 3 === 1 ? 2 ** -(5 + (b % 6)) : (b % 7 === 0 ? 1e-3 : 1) * (0.01 + rand() * 0.05);
        for (let l = 0; l < 32; l++) vals[b * 32 + l] = b % 101 === 0 ? 0 : grid ? s * (Math.floor(rand() * 33) - 16) : (rand() - 0.5) * s;
        // q8 ties: amax 127·s makes d = s exactly, and half-steps of s land on x / d = k + 0.5.
        if (b % 3 === 1) {
          for (let l = 0; l < 32; l++) vals[b * 32 + l] = (s * (Math.floor(rand() * 509) - 254)) / 2;
          vals[b * 32 + (b % 32)] = (b % 2 ? -127 : 127) * s;
        }
      }
      if (type === "F32") bytes.set(new Uint8Array(vals.buffer));
      else {
        const dv = new DataView(bytes.buffer);
        for (let i = 0; i < vals.length; i++) dv.setUint16(i * 2, type === "F16" ? toHalf(vals[i]!) : new Uint32Array(Float32Array.of(vals[i]!).buffer)[0]! >>> 16, true);
      }
    }
    const scaleAt: Partial<Record<SrcType, [number, number[]]>> = { Q4_0: [18, [0]], Q6_K: [210, [208]], Q8_0: [34, [0]], Q4_1: [20, [0, 2]], Q5_0: [22, [0]], Q5_1: [24, [0, 2]], Q4_K: [144, [0, 2]], Q5_K: [176, [0, 2]] };
    const [blk, dAts] = scaleAt[type] ?? [0, []];
    for (const dAt of dAts) for (let b = 0, o = dAt; o < bytes.length; b++, o += blk) {
      const h = b % 97 === 0 ? (b % 2 ? 0x8003 : 0) : toHalf((rand() * 0.02) * (b % 13 === 0 ? 1e-3 : 1)) | (b % 3 || dAt === 2 ? 0 : 0x8000);
      bytes[o] = h & 255;
      bytes[o + 1] = h >> 8;
    }
    const files = [{ name: "check", source: new MemorySource(bytes, "check"), tensors: [{ name: "w.weight", shape: [rows, cols], type, offset: 0, byteLength: bytes.length }] }];
    // Under "q4" and "q8" Q4_1/Q4_K load as exact affine q4, and under "q8" Q5_x as exact q8 (convert/…/exact).
    for (const out of ["q4", "q8"] as const) {
      if (!gpuConverts(type, out)) continue;
      const planes = async (gpu: boolean) => {
        setGpuConvert(gpu);
        const t = (await loadWeights(be, files, out)).get("w.weight")!;
        const mins = t.mins ? new Float32Array(await be.readback(t.mins, 0, t.mins.size)) : new Float32Array(0);
        return [new Uint32Array(await be.readback(t.buf, 0, t.buf.size)), new Float32Array(await be.readback(t.scales!, 0, t.scales!.size)), mins, t.dtype] as const;
      };
      const [gn, gs0, gm, gd] = await planes(true);
      const [cn, cs0, cm, cd] = await planes(false);
      // Spread would overflow the stack on 240k scales.
      const join = (a: Float32Array, b: Float32Array) => { const o = new Float32Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; };
      const gs = join(gs0, gm);
      const cs = join(cs0, cm);
      setGpuConvert(true);
      // Codes compared as whole words: 8 nibbles (q4) or 4 bytes (q8) each.
      const per = gd === "q4" ? 8 : 4;
      let codes = 0;
      for (let w = 0; w < cn.length; w++) {
        if (gn[w] === cn[w]) continue;
        for (let e = 0; e < per; e++) if (((gn[w]! >>> (e * (32 / per))) ^ (cn[w]! >>> (e * (32 / per)))) & (per === 8 ? 15 : 255)) codes++;
      }
      const scaleDiffs = gs.reduce((n, v, i) => n + (Object.is(v, cs[i]) ? 0 : 1), 0);
      const pass = scaleDiffs === 0 && codes === 0 && gn.length === cn.length && gs.length === cs.length && gd === cd;
      const path = gm.length ? `${gd}, affine (exact)` : gd;
      results.push({ kernel: `convert/${type}/${out}`, case: `${rows}x${cols}, 2 chunks → ${path}`, error: codes, tol: 0, pass, note: `${codes} of ${cn.length * per} codes differ, ${scaleDiffs} scales/mins` });
    }
  }

  render(results, be);
}

function render(results: Result[], be: Backend & { lostReason?: string }) {
  const rows = document.getElementById("rows")!;
  for (const r of results) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.kernel}</td><td>${r.case}</td><td class="num">${r.error.toExponential(2)} (tol ${r.tol.toExponential(0)})${r.note ? ` · ${r.note}` : ""}</td><td class="${r.pass ? "pass" : "fail"}">${r.pass ? "pass" : "FAIL"}</td>`;
    rows.append(tr);
  }
  const failed = results.filter((r) => !r.pass).length;
  document.getElementById("summary")!.textContent = `${results.length - failed} / ${results.length} passed${be.lostReason ? ` · device lost: ${be.lostReason}` : ""}`;
  (window as unknown as { __kernelResults: unknown }).__kernelResults = { done: true, failed, results };
}

main().catch((err) => {
  const el = document.getElementById("summary")!;
  el.className = "error";
  el.textContent = String(err instanceof Error ? err.stack ?? err.message : err);
  (window as unknown as { __kernelResults: unknown }).__kernelResults = { done: true, error: String(err) };
});
