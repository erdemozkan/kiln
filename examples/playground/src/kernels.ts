import { createWebGPUBackend, type WebGPUBackend } from "@kiln/backend-webgpu";
import { Ctx, packStep, type Buf, type StepParams, type Tensor } from "@kiln/core";
import { ops, ref } from "@kiln/kernels";
import { dequantQ4, f16ToF32, quantizeQ4 } from "@kiln/loader";

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
  const be = await createWebGPUBackend();
  document.getElementById("device")!.textContent = `features: ${[...be.limits.features].join(", ") || "none"} · max binding ${(be.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`;
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
  }

  // Fused matmul variants: every lanes setting, with norm prologue, SiLU gating, residual add and bias.
  for (const lanes of [8, 16, 32] as const) {
    for (const [M, I, K] of [[1, 40, 96], [5, 24, 256], [19, 70, 576]] as const) {
      for (const q4 of [false, true]) {
        const x = randn(M * K);
        const nw = randn(K);
        const wgu = randn(2 * I * K);
        const deq = q4 ? dequantQ4(quantizeQ4(wgu)) : wgu;
        const W = (t: Float32Array, shape: number[]) => (q4 ? uploadQ4(t, shape) : upload(t, shape));
        const xn = ref.rmsnormRef(x, nw, K, 1e-5);
        const gu = ref.matmulRef(xn, deq, M, 2 * I, K);
        const want = new Float32Array(M * I);
        for (let m = 0; m < M; m++) for (let i = 0; i < I; i++) {
          const g = gu[m * 2 * I + i]!;
          want[m * I + i] = (g / (1 + Math.exp(-g))) * gu[m * 2 * I + I + i]!;
        }
        let y!: Tensor;
        await run(S0, (ctx) => (y = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { norm: { w: upload(nw, [K]), eps: 1e-5 }, silu: true, lanes })));
        const tag = `${q4 ? "q4" : "f32"} l${lanes} M${M} K${K}`;
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

        if (be.limits.features.has("shader-f16")) {
          let yh!: Tensor;
          await run(S0, (ctx) => (yh = ops.linear(ctx, upload(x, [M, K]), W(wgu, [2 * I, K]), { norm: { w: upload(nw, [K]), eps: 1e-5 }, silu: true, lanes, f16: true })));
          check("matmul+norm+silu/f16", tag, await download(yh.buf, M * I), want, 2e-2 * Math.sqrt(K));
        }
        const res = randn(M * I);
        const bias = randn(I);
        const w2 = randn(I * K);
        const deq2 = q4 ? dequantQ4(quantizeQ4(w2)) : w2;
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
  }

  render(results, be);
}

function render(results: Result[], be: WebGPUBackend) {
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
