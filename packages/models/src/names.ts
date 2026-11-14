/** GGUF tensor names → Hugging Face names, which the model code uses everywhere. */
const BLOCK: Record<string, string> = {
  attn_norm: "input_layernorm",
  attn_q: "self_attn.q_proj",
  attn_k: "self_attn.k_proj",
  attn_v: "self_attn.v_proj",
  attn_output: "self_attn.o_proj",
  attn_q_norm: "self_attn.q_norm",
  attn_k_norm: "self_attn.k_norm",
  ffn_norm: "post_attention_layernorm",
  ffn_gate: "mlp.gate_proj",
  ffn_up: "mlp.up_proj",
  ffn_down: "mlp.down_proj",
};

const GLOBAL: Record<string, string> = {
  "token_embd.weight": "model.embed_tokens.weight",
  "output_norm.weight": "model.norm.weight",
  "output.weight": "lm_head.weight",
};

/** Gemma's sandwich norms: llama.cpp's ffn_norm is the norm before the MLP, not after attention. */
const GEMMA3_BLOCK: Record<string, string> = {
  ...BLOCK,
  post_attention_norm: "post_attention_layernorm",
  ffn_norm: "pre_feedforward_layernorm",
  post_ffw_norm: "post_feedforward_layernorm",
};

/**
 * Gemma 4 adds the per-layer input's gate, projection and norm. The layer's output scale and the
 * per-layer embedding table are not listed: the loader reads them itself (see `GEMMA4_LAYER_SCALE`
 * and `GEMMA4_PER_LAYER_TABLE`).
 */
const GEMMA4_BLOCK: Record<string, string> = {
  ...GEMMA3_BLOCK,
  inp_gate: "per_layer_input_gate",
  proj: "per_layer_projection",
  post_norm: "post_per_layer_input_norm",
};
const GEMMA4_GLOBAL: Record<string, string> = {
  "per_layer_model_proj.weight": "model.per_layer_model_projection.weight",
  "per_layer_proj_norm.weight": "model.per_layer_projection_norm.weight",
};
/** A Gemma 4 layer's output scalar: GGUF `blk.N.layer_output_scale.weight`, checkpoint `model.layers.N.layer_scalar`. Captures the layer. */
export const GEMMA4_LAYER_SCALE = /^(?:blk\.(\d+)\.layer_output_scale\.weight|model\.layers\.(\d+)\.layer_scalar)$/;
/** Gemma 4's per-layer embedding table [vocab, layers · width], in a GGUF file and in a checkpoint (text names). */
export const GEMMA4_PER_LAYER_TABLE = ["per_layer_token_embd.weight", "model.embed_tokens_per_layer.weight"];
/** The table of layer l alone, [vocab, width], as the model reads it. */
export const gemma4PerLayerTable = (l: number): string => `model.embed_tokens_per_layer.${l}.weight`;

/** Phi-3 files keep q/k/v as one tensor and gate/up as one (in `ffn_up`): the stacked weights Kiln runs. */
const PHI3_BLOCK: Record<string, string> = {
  ...BLOCK,
  attn_qkv: "self_attn.qkv",
  ffn_up: "mlp.gate_up",
};

/**
 * Qwen 3.5: llama.cpp's names for linear attention (its `attn_qkv` is the layer's q/k/v projection,
 * not attention's), and the norm after the token mixer under its own name.
 */
const QWEN35_BLOCK: Record<string, string> = {
  ...BLOCK,
  post_attention_norm: "post_attention_layernorm",
  attn_qkv: "linear_attn.in_proj_qkv",
  attn_gate: "linear_attn.in_proj_z",
  ssm_alpha: "linear_attn.in_proj_a",
  ssm_beta: "linear_attn.in_proj_b",
  ssm_conv1d: "linear_attn.conv1d",
  ssm_norm: "linear_attn.norm",
  ssm_out: "linear_attn.out_proj",
};

/** Returns undefined for tensors the model does not use (e.g. rope_freqs). */
export function ggufToHF(name: string, arch = "llama"): string | undefined {
  if (GLOBAL[name]) return GLOBAL[name];
  if (arch === "qwen35") {
    // Two parameters keep no ".weight": the decay rates (see `qwen35GgufValues`) and the time-step bias.
    const p = name.match(/^blk\.(\d+)\.(ssm_a|ssm_dt\.bias)$/);
    if (p) return `model.layers.${p[1]}.linear_attn.${p[2] === "ssm_a" ? "A_log" : "dt_bias"}`;
  }
  if (arch === "gemma4" && GEMMA4_GLOBAL[name]) return GEMMA4_GLOBAL[name];
  const m = name.match(/^blk\.(\d+)\.(\w+)\.(weight|bias)$/);
  if (!m) return undefined;
  const mapped = (arch === "gemma3" ? GEMMA3_BLOCK : arch === "gemma4" ? GEMMA4_BLOCK : arch === "phi3" ? PHI3_BLOCK : arch === "qwen35" ? QWEN35_BLOCK : BLOCK)[m[2]!];
  return mapped ? `model.layers.${m[1]}.${mapped}.${m[3]}` : undefined;
}

/** Linear and embedding matrices can be stored as q4 or q8; norms and biases stay f32. */
export function isQuantizable(name: string, shape: readonly number[]): boolean {
  // Linear attention's decay and write-strength projections are a few rows that steer a recurrent state: kept exact.
  if (/\.linear_attn\.in_proj_[ab]\./.test(name)) return false;
  return shape.length === 2 && name.endsWith(".weight") && !name.includes("norm") && shape[1]! % 32 === 0;
}

/**
 * Values of a Qwen 3.5 GGUF tensor as the model reads them. llama.cpp's converter stores the decay
 * rates as −exp(A_log); the model takes A_log, as the checkpoint has it.
 */
export function qwen35GgufValues(name: string, values: Float32Array): void {
  if (!name.endsWith(".linear_attn.A_log")) return;
  for (let i = 0; i < values.length; i++) {
    if (!(values[i]! < 0)) throw new Error(`${name}: decay rates in a GGUF file are negative, got ${values[i]}`);
    values[i] = Math.log(-values[i]!);
  }
}
