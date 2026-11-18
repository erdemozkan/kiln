import { describe, expect, it } from "vitest";
import { Ctx, STEP_BYTES, packStep, type Tensor } from "@kiln/core";
import { dequantQ4, dequantQ8, quantizeQ4, quantizeQ8 } from "@kiln/loader";
import { CpuBackend, ops } from "../src";

let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32) * 2 - 1;
const randn = (n: number) => Float32Array.from({ length: n }, rand);

/** A token's mixture of experts the plain way: route, run its experts, weigh and add. */
function reference(x: Float32Array, router: Float32Array, gu: Float32Array, down: Float32Array, o: { R: number; H: number; I: number; E: number; k: number; renorm: boolean }) {
  const { R, H, I, E, k } = o;
  const out = new Float32Array(R * H);
  const ids: number[] = [];
  for (let r = 0; r < R; r++) {
    const xr = x.subarray(r * H, (r + 1) * H);
    const logits = Array.from({ length: E }, (_, e) => xr.reduce((s, v, h) => s + v * router[e * H + h]!, 0));
    const order = logits.map((v, e) => [v, e] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k);
    const mx = Math.max(...logits);
    const sum = o.renorm ? order.reduce((s, [v]) => s + Math.exp(v - mx), 0) : logits.reduce((s, v) => s + Math.exp(v - mx), 0);
    for (const [v, e] of order) {
      ids.push(e);
      const wt = Math.exp(v - mx) / sum;
      const m = new Float32Array(I);
      for (let i = 0; i < I; i++) {
        let g = 0;
        let u = 0;
        for (let h = 0; h < H; h++) {
          g += xr[h]! * gu[(e * 2 * I + i) * H + h]!;
          u += xr[h]! * gu[(e * 2 * I + I + i) * H + h]!;
        }
        m[i] = (g / (1 + Math.exp(-g))) * u;
      }
      for (let h = 0; h < H; h++) {
        let d = 0;
        for (let i = 0; i < I; i++) d += m[i]! * down[(e * H + h) * I + i]!;
        out[r * H + h]! += wt * d;
      }
    }
  }
  return { out, ids };
}

describe("mixture of experts (CPU backend)", () => {
  const be = new CpuBackend({ simd: false });
  const step = be.alloc(STEP_BYTES);
  be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 1, temperature: 0 }));
  const upload = (data: Float32Array | Uint32Array, shape: number[]): Tensor => {
    const buf = be.alloc(data.byteLength);
    be.write(buf, 0, data);
    return { shape, dtype: data instanceof Uint32Array ? "u32" : "f32", buf };
  };
  const weight = (w: Float32Array, shape: number[], kind: "f32" | "q4" | "q8"): { t: Tensor; values: Float32Array } => {
    if (kind === "f32") return { t: upload(w, shape), values: w };
    if (kind === "q4") {
      const q = quantizeQ4(w);
      return { t: { ...upload(q.nibbles, shape), dtype: "q4", scales: upload(q.scales, [0]).buf }, values: dequantQ4(q) };
    }
    const q = quantizeQ8(w);
    return { t: { ...upload(q.bytes, shape), dtype: "q8", scales: upload(q.scales, [0]).buf }, values: dequantQ8(q) };
  };

  it.each([
    ["f32", true],
    ["q4", true],
    ["q8", false],
  ] as const)("routes, runs the picked experts (%s) and adds them as the plain loop does (renorm %s)", async (kind, renorm) => {
    const [R, H, I, E, k] = [3, 64, 32, 6, 2];
    const x = randn(R * H);
    const router = randn(E * H);
    const gu = weight(randn(E * 2 * I * H), [E * 2 * I, H], kind);
    const down = weight(randn(E * H * I), [E * H, I], kind);
    const want = reference(x, router, gu.values, down.values, { R, H, I, E, k, renorm });
    const xt = upload(x, [R, H]);
    const res = upload(new Float32Array(R * H), [R, H]);
    const ctx = new Ctx(be, step, "moe");
    const logits = ops.linear(ctx, xt, upload(router, [E, H]));
    const { ids, wts } = ops.moeRoute(ctx, logits, k, renorm);
    const m = ops.gatedAct(ctx, ops.moeGather(ctx, xt, gu.t, ids, { N: 2 * I, k }), "silu");
    ops.moeCombine(ctx, res, ops.moeGather(ctx, m, down.t, ids, { N: H, k, perSlot: true }), wts, k);
    be.submit(ctx.dispatches);
    expect(Array.from(new Uint32Array(await be.readback(ids.buf, 0, R * k * 4)))).toEqual(want.ids);
    const got = new Float32Array(await be.readback(res.buf, 0, R * H * 4));
    for (let i = 0; i < R * H; i++) expect(got[i]).toBeCloseTo(want.out[i]!, 3);
  });
});
