import { describe, expect, it } from "vitest";
import { byteSize, type Backend, type Buf, type Tensor } from "@kiln/core";
import { LlamaModel, type LlamaConfig } from "../src";

/** SmolLM2-135M shapes. */
const CFG: LlamaConfig = {
  family: "llama",
  hiddenSize: 576,
  intermediateSize: 1536,
  numLayers: 30,
  numHeads: 9,
  numKvHeads: 3,
  headDim: 64,
  vocabSize: 49152,
  rmsEps: 1e-5,
  ropeTheta: 100000,
  ropeInterleaved: false,
  tieEmbeddings: true,
  attentionBias: false,
  qkNorm: false,
  eosTokenIds: [0],
};

function fakeBackend() {
  const live = new Map<Buf, string>();
  const backend: Backend = {
    name: "fake",
    limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 30, maxComputeWorkgroupsPerDimension: 65535, features: new Set() },
    alloc: (size, _u, label = "buf") => {
      const b = { size, label };
      live.set(b, label);
      return b;
    },
    free: (b) => void live.delete(b),
    write: () => {},
    copy: () => {},
    pipeline: (spec) => ({ key: spec.key }),
    bind: (pipeline) => ({ pipeline }),
    submit: () => {},
    readback: async () => new ArrayBuffer(4),
    destroy: () => {},
  };
  /** Bytes of live intermediates (everything a plan allocates except its token/out buffers and uniforms). */
  const actBytes = () => [...live].filter(([, l]) => l.endsWith("/act")).reduce((n, [b]) => n + b.size, 0);
  return { backend, actBytes, live };
}

function weights(be: Backend): Map<string, Tensor> {
  const c = CFG;
  const w = new Map<string, Tensor>();
  const add = (name: string, shape: number[]) => w.set(name, { shape, dtype: "f32", buf: be.alloc(byteSize(shape, "f32"), "storage", name) });
  add("model.embed_tokens.weight", [c.vocabSize, c.hiddenSize]);
  add("model.norm.weight", [c.hiddenSize]);
  const kv = c.numKvHeads * c.headDim;
  for (let l = 0; l < c.numLayers; l++) {
    const p = `model.layers.${l}`;
    add(`${p}.input_layernorm.weight`, [c.hiddenSize]);
    add(`${p}.post_attention_layernorm.weight`, [c.hiddenSize]);
    add(`${p}.self_attn.q_proj.weight`, [c.numHeads * c.headDim, c.hiddenSize]);
    add(`${p}.self_attn.k_proj.weight`, [kv, c.hiddenSize]);
    add(`${p}.self_attn.v_proj.weight`, [kv, c.hiddenSize]);
    add(`${p}.self_attn.o_proj.weight`, [c.hiddenSize, c.numHeads * c.headDim]);
    add(`${p}.mlp.gate_proj.weight`, [c.intermediateSize, c.hiddenSize]);
    add(`${p}.mlp.up_proj.weight`, [c.intermediateSize, c.hiddenSize]);
    add(`${p}.mlp.down_proj.weight`, [c.hiddenSize, c.intermediateSize]);
  }
  return w;
}

describe("plan memory", () => {
  for (const fuse of [true, false]) {
    it(`shares intermediates across all plans (${fuse ? "fused" : "unfused"})`, () => {
      const { backend, actBytes } = fakeBackend();
      const model = new LlamaModel(backend, CFG, weights(backend), { maxSeq: 2048, fuse });
      const plans = [1, 16, 64, 128].map((T) => model.buildPlan(T));
      const total = actBytes();
      // With a size-keyed free list per plan and a [T, H, maxSeq] score buffer this was 19.3 / 21.3 MiB.
      expect(total).toBeLessThan((fuse ? 2 : 4) * 2 ** 20);
      expect(model.memory.activations).toBe(total);
      for (const p of plans) p.destroy();
      model.destroy();
      expect(actBytes()).toBe(0);
    });
  }
});
