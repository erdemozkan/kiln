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

/** Returns undefined for tensors the model does not use (e.g. rope_freqs). */
export function ggufToHF(name: string): string | undefined {
  if (GLOBAL[name]) return GLOBAL[name];
  const m = name.match(/^blk\.(\d+)\.(\w+)\.(weight|bias)$/);
  if (!m) return undefined;
  const mapped = BLOCK[m[2]!];
  return mapped ? `model.layers.${m[1]}.${mapped}.${m[3]}` : undefined;
}

/** Linear and embedding matrices can be stored as q4; norms and biases stay f32. */
export function isQuantizable(name: string, shape: readonly number[]): boolean {
  return shape.length === 2 && name.endsWith(".weight") && !name.includes("norm") && shape[1]! % 32 === 0;
}
