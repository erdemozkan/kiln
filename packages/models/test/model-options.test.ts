import { describe, expect, it, vi } from "vitest";
import type { Buf } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import { LlamaModel, checkModelOptions, type LlamaConfig } from "../src";

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
  eosTokenIds: [],
};

describe("model options", () => {
  it("are checked before anything is loaded", () => {
    expect(checkModelOptions(CFG, { maxSeq: 256 })).toBe(256);
    expect(checkModelOptions(CFG)).toBe(2048);
    // Gemma 4's wide full-attention heads.
    const wide: LlamaConfig = { ...CFG, layerHeadDims: [32, 512] };
    expect(() => checkModelOptions(wide, { kvDtype: "i8" })).toThrow(/i8 cache does not hold heads wider than 256/);
    expect(() => checkModelOptions({ ...CFG, layerHeadDims: [32, 1024] }, { kvDtype: "q4" })).toThrow(/q4 cache does not hold heads wider than 512/);
    expect(() => checkModelOptions({ ...CFG, headDim: 48 }, { kvDtype: "q4" })).toThrow(/multiple of 32/);
    expect(checkModelOptions(wide, { maxSeq: 64, kvDtype: "f16" })).toBe(64);
    // LongRoPE's short factors (Phi-3) hold the positions they were loaded for.
    const phi: LlamaConfig = { ...CFG, contextLength: 4096, maxPositions: 4096 };
    expect(() => checkModelOptions(phi, { maxSeq: 8192 })).toThrow(/maxSeq 8192 is past the 4096 tokens/);
    expect(checkModelOptions(phi)).toBe(4096);
  });

  it("free what the model made when its constructor fails", () => {
    const be = new CpuBackend({ threads: 1 });
    const live = new Set<Buf>();
    const alloc = be.alloc.bind(be);
    vi.spyOn(be, "alloc").mockImplementation((...a: Parameters<typeof be.alloc>) => {
      const b = alloc(...a);
      live.add(b);
      return b;
    });
    const free = be.free.bind(be);
    vi.spyOn(be, "free").mockImplementation((b: Buf) => {
      live.delete(b);
      free(b);
    });
    // No embedding: the constructor fails after its caches and step buffers exist.
    expect(() => new LlamaModel(be, CFG, new Map(), { maxSeq: 64, imageTokens: 4 })).toThrow(/embed_tokens/);
    expect([...live]).toEqual([]);
    // Options it cannot take fail before it allocates anything.
    const made = vi.mocked(be.alloc).mock.calls.length;
    expect(() => new LlamaModel(be, { ...CFG, maxPositions: 32 }, new Map(), { maxSeq: 64, imageTokens: 4 })).toThrow(/past the 32 tokens/);
    expect(vi.mocked(be.alloc).mock.calls.length).toBe(made);
  });
});
