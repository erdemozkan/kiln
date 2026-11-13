import { describe, expect, it } from "vitest";
import { physical, type Tensor } from "@kiln/core";
import { CpuBackend, toF8 } from "@kiln/kernels";
import { LlamaModel, type LlamaConfig } from "@kiln/models";
import { Tokenizer } from "@kiln/tokenizers";
import { Engine } from "../src/engine";

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

function rand(n: number, seed: number, scale: number): Float32Array {
  let s = seed >>> 0;
  return Float32Array.from({ length: n }, () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s / 2 ** 32 - 0.5) * 2 * scale;
  });
}

function weights(be: CpuBackend): Map<string, Tensor> {
  const c = CFG;
  const w = new Map<string, Tensor>();
  let seed = 7;
  const add = (name: string, shape: number[], data: Float32Array) => {
    const buf = be.alloc(data.byteLength, "storage", name);
    be.write(buf, 0, data);
    w.set(name, { shape, dtype: "f32", buf });
  };
  add("model.embed_tokens.weight", [c.vocabSize, c.hiddenSize], rand(c.vocabSize * c.hiddenSize, seed++, 1));
  add("model.norm.weight", [c.hiddenSize], new Float32Array(c.hiddenSize).fill(1));
  const q = c.numHeads * c.headDim;
  const kv = c.numKvHeads * c.headDim;
  for (let l = 0; l < c.numLayers; l++) {
    const p = `model.layers.${l}`;
    add(`${p}.input_layernorm.weight`, [c.hiddenSize], new Float32Array(c.hiddenSize).fill(1));
    add(`${p}.post_attention_layernorm.weight`, [c.hiddenSize], new Float32Array(c.hiddenSize).fill(1));
    for (const [n, shape] of [
      ["self_attn.q_proj", [q, c.hiddenSize]],
      ["self_attn.k_proj", [kv, c.hiddenSize]],
      ["self_attn.v_proj", [kv, c.hiddenSize]],
      ["self_attn.o_proj", [c.hiddenSize, q]],
      ["mlp.gate_proj", [c.intermediateSize, c.hiddenSize]],
      ["mlp.up_proj", [c.intermediateSize, c.hiddenSize]],
      ["mlp.down_proj", [c.hiddenSize, c.intermediateSize]],
    ] as const)
      add(`${p}.${n}.weight`, [...shape], rand(shape[0] * shape[1], seed++, 0.2));
  }
  return w;
}

const logSoftmax = (row: Float32Array, i: number) => {
  const max = Math.max(...row);
  return row[i]! - max - Math.log(row.reduce((s, v) => s + Math.exp(v - max), 0));
};

/** Reference: the sequence one token at a time through plain plans, each row's log-probability. */
async function reference(be: CpuBackend, prompt: number[], cont: number[]): Promise<number> {
  const m = new LlamaModel(be, CFG, weights(be), { maxSeq: 64, kvDtype: "f32" });
  const read = async (t: Tensor) => new Float32Array(await be.readback(physical(t.buf), 0, t.shape[0]! * t.shape[1]! * 4));
  const pre = m.buildPlan(prompt.length);
  be.write(pre.tokens, 0, Uint32Array.from(prompt));
  m.setStep({ pos0: 0, lastRow: prompt.length - 1, seed: 0, temperature: 0 });
  be.submit(pre.dispatches);
  let lp = logSoftmax(await read(pre.logits), cont[0]!);
  const dec = m.buildPlan(1);
  for (let t = 1; t < cont.length; t++) {
    be.write(dec.tokens, 0, Uint32Array.of(cont[t - 1]!));
    m.setStep({ pos0: prompt.length + t - 1, lastRow: 0, seed: 0, temperature: 0 });
    be.submit(dec.dispatches);
    lp += logSoftmax(await read(dec.logits), cont[t]!);
  }
  m.destroy();
  return lp;
}

describe("Engine.scoreIds", () => {
  it("sums each continuation's log-probabilities, reusing the prompt prefix", async () => {
    const be = new CpuBackend({ threads: 1 });
    const tok = Tokenizer.fromJSON({ model: { type: "BPE", vocab: { a: 1, b: 2 }, merges: [] }, pre_tokenizer: { type: "ByteLevel" } as never });
    const engine = new Engine(be, new LlamaModel(be, CFG, weights(be), { maxSeq: 64, kvDtype: "f32" }), tok);
    const prompt = [5, 17, 33, 41, 8, 90, 12, 3, 77, 21];
    const conts = [[4], [9, 22, 0], [60, 61, 62, 63, 64, 0]];
    const got = await engine.scoreIds(prompt, conts);
    for (let i = 0; i < conts.length; i++) expect(got[i]).toBeCloseTo(await reference(be, prompt, conts[i]!), 4);
    // Again over the same prompt (the prefix comes from the cache) and after a longer one.
    expect((await engine.scoreIds(prompt, [conts[1]!]))[0]).toBeCloseTo(got[1]!, 5);
    await engine.scoreIds([...prompt, 50, 51], [[1]]);
    expect((await engine.scoreIds(prompt, [conts[2]!]))[0]).toBeCloseTo(got[2]!, 5);
    engine.destroy();
  });
});

describe("f8 KV cache", () => {
  it("stores what the f32 cache holds, as FP8 codes, and still predicts the same tokens", async () => {
    const be = new CpuBackend({ threads: 1 });
    const prompt = [5, 17, 33, 41, 8, 90, 12, 3, 77, 21, 4, 60];
    const run = async (kvDtype: "f32" | "f8") => {
      const m = new LlamaModel(be, CFG, weights(be), { maxSeq: 64, kvDtype });
      const plan = m.buildPlan(prompt.length);
      be.write(plan.tokens, 0, Uint32Array.from(prompt));
      m.setStep({ pos0: 0, lastRow: prompt.length - 1, seed: 0, temperature: 0 });
      be.submit(plan.dispatches);
      const logits = new Float32Array(await be.readback(physical(plan.logits.buf), 0, plan.logits.shape[0]! * plan.logits.shape[1]! * 4));
      const kv = await m.readKv(prompt.length);
      const rowBytes = m.kvRowBytes;
      m.destroy();
      return { logits, kv, rowBytes };
    };
    const exact = await run("f32");
    const small = await run("f8");
    expect(small.rowBytes * 4).toBe(exact.rowBytes);
    expect(small.kv.length * 4).toBe(exact.kv.length);
    // Layer 0's keys and values depend on the tokens only, so its codes are the f32 values' codes.
    const first = 2 * prompt.length * exact.rowBytes;
    const f32 = new Float32Array(exact.kv.buffer, exact.kv.byteOffset, first / 4);
    for (let i = 0; i < f32.length; i++) expect(small.kv[i], String(i)).toBe(toF8(f32[i]!));
    // A random-weight model is a harsh case (no trained margin between tokens), so only the best token is compared.
    const best = (l: Float32Array) => l.indexOf(Math.max(...l));
    expect(best(small.logits)).toBe(best(exact.logits));
  });
});
