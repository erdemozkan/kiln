// A two-layer model with random weights on the CPU backend, for engine tests.
import type { Tensor } from "@kiln/core";
import type { CpuBackend } from "@kiln/kernels";
import type { LlamaConfig } from "@kiln/models";
import { Tokenizer } from "@kiln/tokenizers";

export const TINY: LlamaConfig = {
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
  eosTokenIds: [],
};

export function rand(n: number, seed: number, scale: number): Float32Array {
  let s = seed >>> 0;
  return Float32Array.from({ length: n }, () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s / 2 ** 32 - 0.5) * 2 * scale;
  });
}

/** The same weights on every call (a fixed seed), so two models built from them agree. */
export function tinyWeights(be: CpuBackend, c: LlamaConfig = TINY): Map<string, Tensor> {
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
      add(`${p}.${n}.weight`, [...shape], rand(shape[0] * shape[1], seed++, 0.5));
  }
  return w;
}

/** A byte-level tokenizer of two tokens, plus `added` special tokens (content → id). */
export const tinyTokenizer = (added: Record<string, number> = {}) =>
  Tokenizer.fromJSON({
    model: { type: "BPE", vocab: { a: 1, b: 2 }, merges: [] },
    pre_tokenizer: { type: "ByteLevel" } as never,
    added_tokens: Object.entries(added).map(([content, id]) => ({ id, content, special: true })),
  } as never);
