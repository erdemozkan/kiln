import { describe, expect, it } from "vitest";
import { UnsupportedModelError, configFromGGUF, ggufToHF, isQuantizable, qwen35GgufValues, ropeInvFreq } from "../src";

// Metadata of unsloth/Qwen3.5-0.8B-GGUF, as far as Kiln reads it.
const meta = (over: [string, unknown][] = []) =>
  new Map<string, unknown>([
    ["general.architecture", "qwen35"],
    ["qwen35.block_count", 24],
    ["qwen35.context_length", 262144],
    ["qwen35.embedding_length", 1024],
    ["qwen35.feed_forward_length", 3584],
    ["qwen35.attention.head_count", 8],
    ["qwen35.attention.head_count_kv", 2],
    ["qwen35.attention.key_length", 256],
    ["qwen35.attention.layer_norm_rms_epsilon", 1e-6],
    ["qwen35.rope.freq_base", 10000000],
    ["qwen35.rope.dimension_count", 64],
    ["qwen35.ssm.conv_kernel", 4],
    ["qwen35.ssm.state_size", 128],
    ["qwen35.ssm.group_count", 16],
    ["qwen35.ssm.time_step_rank", 16],
    ["qwen35.ssm.inner_size", 2048],
    ["qwen35.full_attention_interval", 4],
    ["tokenizer.ggml.tokens", new Array(248320).fill("x")],
    ["tokenizer.ggml.eos_token_id", 248046],
    ...over,
  ]);
const names = ["token_embd.weight", "output_norm.weight", "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_a", "blk.3.attn_q.weight", "blk.3.attn_q_norm.weight"];

describe("Qwen 3.5 from GGUF", () => {
  it("reads the config: which layers are linear attention, their sizes, partial rotary", () => {
    const c = configFromGGUF(meta(), names);
    expect(c).toMatchObject({ family: "qwen35", numLayers: 24, headDim: 256, numKvHeads: 2, qkNorm: true, attnOutputGate: true, rotaryDims: 64, ropeTheta: 10000000, tieEmbeddings: true, ropeInterleaved: false });
    expect(c.linear).toEqual({ keyHeads: 16, valueHeads: 16, keyDim: 128, valueDim: 128, convKernel: 4 });
    expect(c.linearLayers!.map((x, l) => (x ? "" : l)).filter((x) => x !== "")).toEqual([3, 7, 11, 15, 19, 23]);
    expect(ropeInvFreq(c).length).toBe(32);
    // The 4B: twice the value heads, which the converter stores in tiled order.
    const big = configFromGGUF(meta([["qwen35.ssm.time_step_rank", 32], ["qwen35.ssm.inner_size", 4096]]), names);
    expect(big.linear).toEqual({ keyHeads: 16, valueHeads: 32, keyDim: 128, valueDim: 128, convKernel: 4, tiled: true });
    // Layers listed one by one win over the interval.
    const listed = configFromGGUF(meta([["qwen35.attention.recurrent_layers", Array.from({ length: 24 }, (_, l) => l < 20)]]), names);
    expect(listed.linearLayers!.filter(Boolean).length).toBe(20);
  });

  it("refuses layouts it cannot run", () => {
    expect(() => configFromGGUF(meta(), names.filter((n) => n !== "blk.0.attn_gate.weight"))).toThrow(/merged/);
    expect(() => configFromGGUF(meta([["qwen35.ssm.inner_size", 2050]]), names)).toThrow(UnsupportedModelError);
    expect(() => configFromGGUF(meta([["qwen35.ssm.time_step_rank", 24], ["qwen35.ssm.inner_size", 3072]]), names)).toThrow(UnsupportedModelError);
    expect(() => configFromGGUF(meta([["qwen35.rope.dimension_count", 63]]), names)).toThrow(/63 of 256/);
  });

  it("maps llama.cpp's tensor names", () => {
    const hf = (n: string) => ggufToHF(n, "qwen35");
    expect(hf("blk.0.attn_qkv.weight")).toBe("model.layers.0.linear_attn.in_proj_qkv.weight");
    expect(hf("blk.0.attn_gate.weight")).toBe("model.layers.0.linear_attn.in_proj_z.weight");
    expect(hf("blk.0.ssm_alpha.weight")).toBe("model.layers.0.linear_attn.in_proj_a.weight");
    expect(hf("blk.0.ssm_beta.weight")).toBe("model.layers.0.linear_attn.in_proj_b.weight");
    expect(hf("blk.0.ssm_conv1d.weight")).toBe("model.layers.0.linear_attn.conv1d.weight");
    expect(hf("blk.0.ssm_norm.weight")).toBe("model.layers.0.linear_attn.norm.weight");
    expect(hf("blk.0.ssm_out.weight")).toBe("model.layers.0.linear_attn.out_proj.weight");
    expect(hf("blk.0.ssm_a")).toBe("model.layers.0.linear_attn.A_log");
    expect(hf("blk.0.ssm_dt.bias")).toBe("model.layers.0.linear_attn.dt_bias");
    expect(hf("blk.0.attn_norm.weight")).toBe("model.layers.0.input_layernorm.weight");
    expect(hf("blk.0.post_attention_norm.weight")).toBe("model.layers.0.post_attention_layernorm.weight");
    expect(hf("blk.3.attn_q.weight")).toBe("model.layers.3.self_attn.q_proj.weight");
    expect(hf("blk.3.attn_q_norm.weight")).toBe("model.layers.3.self_attn.q_norm.weight");
    expect(hf("blk.24.nextn.eh_proj.weight")).toBeUndefined();
    // Other families are untouched: attn_qkv is Phi-3's stacked attention there, ssm_a nothing.
    expect(ggufToHF("blk.0.attn_qkv.weight", "phi3")).toBe("model.layers.0.self_attn.qkv.weight");
    expect(ggufToHF("blk.0.ssm_a")).toBeUndefined();
    // The decay and write projections stay exact; the rest of a layer may be quantized.
    expect(isQuantizable("model.layers.0.linear_attn.in_proj_a.weight", [16, 1024])).toBe(false);
    expect(isQuantizable("model.layers.0.linear_attn.in_proj_qkv.weight", [6144, 1024])).toBe(true);
    expect(isQuantizable("model.layers.0.linear_attn.conv1d.weight", [6144, 4])).toBe(false);
  });

  it("turns the file's decay rates back into A_log", () => {
    const aLog = [-2.5, 0, 1.2, 2.7];
    const stored = Float32Array.from(aLog, (v) => -Math.exp(v));
    qwen35GgufValues("model.layers.0.linear_attn.A_log", stored);
    aLog.forEach((v, i) => expect(stored[i]).toBeCloseTo(v, 5));
    const other = Float32Array.of(-1, 2);
    qwen35GgufValues("model.layers.0.linear_attn.dt_bias", other);
    expect(Array.from(other)).toEqual([-1, 2]);
    expect(() => qwen35GgufValues("model.layers.0.linear_attn.A_log", Float32Array.of(0.5))).toThrow(/negative/);
  });
});
