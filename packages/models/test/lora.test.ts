import { describe, expect, it } from "vitest";
import { physical, type Tensor } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import { LORA_MODULES, LlamaModel, type LlamaConfig, type LoraWeights } from "../src";

const CFG: LlamaConfig = {
  family: "llama",
  hiddenSize: 64,
  intermediateSize: 128,
  numLayers: 2,
  numHeads: 2,
  numKvHeads: 1,
  headDim: 32,
  vocabSize: 96,
  rmsEps: 1e-5,
  ropeTheta: 10000,
  ropeInterleaved: false,
  tieEmbeddings: true,
  attentionBias: false,
  qkNorm: false,
  eosTokenIds: [0],
};

/** Deterministic values in [-scale, scale). */
function rand(n: number, seed: number, scale: number): Float32Array {
  let s = seed >>> 0;
  return Float32Array.from({ length: n }, () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s / 2 ** 32 - 0.5) * 2 * scale;
  });
}

const SHAPES = (c: LlamaConfig): Record<string, [number, number]> => ({
  q_proj: [c.numHeads * c.headDim, c.hiddenSize],
  k_proj: [c.numKvHeads * c.headDim, c.hiddenSize],
  v_proj: [c.numKvHeads * c.headDim, c.hiddenSize],
  o_proj: [c.hiddenSize, c.numHeads * c.headDim],
  gate_proj: [c.intermediateSize, c.hiddenSize],
  up_proj: [c.intermediateSize, c.hiddenSize],
  down_proj: [c.hiddenSize, c.intermediateSize],
});

/** A rank-`r` adapter on every projection of every layer (with its scale already in B). */
function adapter(c: LlamaConfig, r: number, seed: number): LoraWeights {
  const out = new Map<string, { rank: number; A: Float32Array; B: Float32Array }>();
  let k = seed;
  for (let l = 0; l < c.numLayers; l++)
    for (const m of LORA_MODULES) {
      const [n, kin] = SHAPES(c)[m]!;
      out.set(`${l}.${m}`, { rank: r, A: rand(r * kin, k++, 0.2), B: rand(n * r, k++, 0.2) });
    }
  return out;
}

/** The model's weights, with `merged`'s updates added (W + B·A) when given. */
function weights(be: CpuBackend, c: LlamaConfig, merged?: LoraWeights): Map<string, Tensor> {
  const w = new Map<string, Tensor>();
  let seed = 1;
  const add = (name: string, shape: number[], data: Float32Array) => {
    const buf = be.alloc(data.byteLength, "storage", name);
    be.write(buf, 0, data);
    w.set(name, { shape, dtype: "f32", buf });
  };
  add("model.embed_tokens.weight", [c.vocabSize, c.hiddenSize], rand(c.vocabSize * c.hiddenSize, seed++, 1));
  add("model.norm.weight", [c.hiddenSize], Float32Array.from({ length: c.hiddenSize }, () => 1));
  for (let l = 0; l < c.numLayers; l++) {
    const p = `model.layers.${l}`;
    add(`${p}.input_layernorm.weight`, [c.hiddenSize], rand(c.hiddenSize, seed++, 0.5).map((v) => 1 + v));
    add(`${p}.post_attention_layernorm.weight`, [c.hiddenSize], rand(c.hiddenSize, seed++, 0.5).map((v) => 1 + v));
    for (const m of LORA_MODULES) {
      const [n, k] = SHAPES(c)[m]!;
      const data = rand(n * k, seed++, 0.15);
      const u = merged?.get(`${l}.${m}`);
      if (u) for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) for (let t = 0; t < u.rank; t++) data[i * k + j]! += u.B[i * u.rank + t]! * u.A[t * k + j]!;
      add(`${p}.${m.endsWith("proj") && ["q_proj", "k_proj", "v_proj", "o_proj"].includes(m) ? "self_attn" : "mlp"}.${m}.weight`, [n, k], data);
    }
  }
  return w;
}

/** Logits of the last prompt token after a prefill of `tokens`, then of one decode step. */
async function run(be: CpuBackend, m: LlamaModel, tokens: number[]): Promise<Float32Array[]> {
  const out: Float32Array[] = [];
  const read = async (t: Tensor) => new Float32Array(await be.readback(physical(t.buf), 0, t.shape.reduce((a, b) => a * b, 1) * 4));
  const pre = m.buildPlan(tokens.length);
  be.write(pre.tokens, 0, Uint32Array.from(tokens));
  m.setStep({ pos0: 0, lastRow: tokens.length - 1, seed: 0, temperature: 0 });
  be.submit(pre.dispatches);
  out.push(await read(pre.logits));
  const dec = m.buildPlan(1);
  be.write(dec.tokens, 0, Uint32Array.of(7));
  m.setStep({ pos0: tokens.length, lastRow: 0, seed: 0, temperature: 0 });
  be.submit(dec.dispatches);
  out.push(await read(dec.logits));
  return out;
}

const maxDiff = (a: Float32Array, b: Float32Array) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]!)), 0);

const tokens = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26, 43, 38, 32, 79, 50, 28];

describe("LoRA slots", () => {
  for (const fuse of [true, false]) {
    it(`match the merged weights, prefill and decode (${fuse ? "fused" : "unfused"})`, async () => {
      const be = new CpuBackend({ threads: 1 });
      const lora = adapter(CFG, 8, 1000);
      const runtime = new LlamaModel(be, CFG, weights(be, CFG), { maxSeq: 64, kvDtype: "f32", fuse });
      runtime.enableLora(8);
      expect(runtime.loraRank).toBe(32);
      runtime.setLora(lora);
      const merged = new LlamaModel(be, CFG, weights(be, CFG, lora), { maxSeq: 64, kvDtype: "f32", fuse });
      const [a, b] = [await run(be, runtime, tokens), await run(be, merged, tokens)];
      // The update must matter, and match the merged weights.
      const base = new LlamaModel(be, CFG, weights(be, CFG), { maxSeq: 64, kvDtype: "f32", fuse });
      const c = await run(be, base, tokens);
      expect(maxDiff(a[0]!, c[0]!)).toBeGreaterThan(0.05);
      expect(maxDiff(a[0]!, b[0]!)).toBeLessThan(1e-4);
      expect(maxDiff(a[1]!, b[1]!)).toBeLessThan(1e-4);
      // Cleared slots give the base model back; another adapter swaps in without a rebuild.
      runtime.setLora(undefined);
      const cleared = await run(be, runtime, tokens);
      expect(maxDiff(cleared[0]!, c[0]!)).toBeLessThan(1e-5);
      const other = adapter(CFG, 16, 5000);
      runtime.setLora(other);
      const otherMerged = new LlamaModel(be, CFG, weights(be, CFG, other), { maxSeq: 64, kvDtype: "f32", fuse });
      expect(maxDiff((await run(be, runtime, tokens))[1]!, (await run(be, otherMerged, tokens))[1]!)).toBeLessThan(1e-4);
      for (const m of [runtime, merged, base, otherMerged]) m.destroy();
    });
  }

  it("switches prepared adapters by GPU copies", async () => {
    const be = new CpuBackend({ threads: 1 });
    const one = adapter(CFG, 8, 1000);
    const two = adapter(CFG, 16, 5000);
    const m = new LlamaModel(be, CFG, weights(be, CFG), { maxSeq: 64, kvDtype: "f32" });
    m.enableLora(16);
    const [p1, p2] = [m.prepareLora(one, 16), m.prepareLora(two, 16)];
    m.setLora(two);
    const want = await run(be, m, tokens);
    m.useLora(p1);
    m.useLora(p2);
    const got = await run(be, m, tokens);
    expect(maxDiff(got[1]!, want[1]!)).toBe(0);
    expect(() => m.useLora(m.prepareLora(one, 64))).toThrow(/rank/);
    m.destroy();
  });

  it("rejects an adapter above the slot rank", () => {
    const be = new CpuBackend({ threads: 1 });
    const m = new LlamaModel(be, CFG, weights(be, CFG), { maxSeq: 16 });
    m.enableLora(16);
    expect(() => m.setLora(adapter(CFG, 40, 1))).toThrow(/rank 40/);
    m.destroy();
  });
});
