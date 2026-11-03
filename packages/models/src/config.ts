/** Normalised config for every Llama-shaped decoder (Llama 2/3, Mistral, Qwen2/2.5, Qwen3, SmolLM). */
export interface LlamaConfig {
  family: "llama" | "qwen2" | "qwen3";
  hiddenSize: number;
  intermediateSize: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  vocabSize: number;
  rmsEps: number;
  ropeTheta: number;
  ropeScaling?: Llama3RopeScaling;
  /** GGUF llama checkpoints have Q/K permuted for interleaved (GPT-J style) rotary pairs. */
  ropeInterleaved: boolean;
  tieEmbeddings: boolean;
  attentionBias: boolean;
  /** Per-head RMSNorm on q and k (Qwen3). */
  qkNorm: boolean;
  eosTokenIds: number[];
  bosTokenId?: number;
}

export interface Llama3RopeScaling {
  factor: number;
  lowFreqFactor: number;
  highFreqFactor: number;
  originalMaxPositions: number;
}

export class UnsupportedModelError extends Error {}

const SUPPORTED = ["llama", "mistral", "qwen2", "qwen3"];

type HFConfig = Record<string, unknown>;
const num = (c: HFConfig, k: string, fallback?: number): number => {
  const v = c[k] ?? fallback;
  if (typeof v !== "number") throw new UnsupportedModelError(`config.json is missing "${k}"`);
  return v;
};
const ids = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : typeof v === "number" ? [v] : []);

export function configFromHF(c: HFConfig, generation?: HFConfig): LlamaConfig {
  const type = String(c.model_type);
  if (!SUPPORTED.includes(type)) {
    throw new UnsupportedModelError(`model_type "${type}" is not supported yet (supported: ${SUPPORTED.join(", ")})`);
  }
  const numHeads = num(c, "num_attention_heads");
  const hiddenSize = num(c, "hidden_size");
  const family = type === "qwen2" ? "qwen2" : type === "qwen3" ? "qwen3" : "llama";
  return {
    family,
    hiddenSize,
    intermediateSize: num(c, "intermediate_size"),
    numLayers: num(c, "num_hidden_layers"),
    numHeads,
    numKvHeads: num(c, "num_key_value_heads", numHeads),
    headDim: num(c, "head_dim", hiddenSize / numHeads),
    vocabSize: num(c, "vocab_size"),
    rmsEps: num(c, "rms_norm_eps", 1e-6),
    ropeTheta: num(c, "rope_theta", 10000),
    ropeScaling: ropeScalingFromHF(c.rope_scaling as HFConfig | null | undefined),
    ropeInterleaved: false,
    tieEmbeddings: Boolean(c.tie_word_embeddings ?? false),
    attentionBias: family === "qwen2" ? true : Boolean(c.attention_bias ?? false),
    qkNorm: family === "qwen3",
    eosTokenIds: [...new Set([...ids(c.eos_token_id), ...ids(generation?.eos_token_id)])],
    bosTokenId: typeof c.bos_token_id === "number" ? c.bos_token_id : undefined,
  };
}

function ropeScalingFromHF(s: HFConfig | null | undefined): Llama3RopeScaling | undefined {
  if (!s) return undefined;
  const type = s.rope_type ?? s.type;
  if (type === "default") return undefined;
  if (type !== "llama3") throw new UnsupportedModelError(`rope_scaling type "${String(type)}" is not supported yet (llama3 only)`);
  return {
    factor: num(s, "factor"),
    lowFreqFactor: num(s, "low_freq_factor"),
    highFreqFactor: num(s, "high_freq_factor"),
    originalMaxPositions: num(s, "original_max_position_embeddings"),
  };
}

export function configFromGGUF(meta: Map<string, unknown>, tensorNames: readonly string[]): LlamaConfig {
  const arch = String(meta.get("general.architecture"));
  if (!["llama", "qwen2", "qwen3"].includes(arch)) {
    throw new UnsupportedModelError(`GGUF architecture "${arch}" is not supported yet (llama, qwen2, qwen3)`);
  }
  const get = (k: string, fallback?: number): number => {
    const v = meta.get(`${arch}.${k}`) ?? fallback;
    if (v === undefined) throw new UnsupportedModelError(`GGUF metadata is missing "${arch}.${k}"`);
    return Number(v);
  };
  const numHeads = get("attention.head_count");
  const hiddenSize = get("embedding_length");
  const tokens = meta.get("tokenizer.ggml.tokens") as unknown[] | undefined;
  return {
    family: arch as LlamaConfig["family"],
    hiddenSize,
    intermediateSize: get("feed_forward_length"),
    numLayers: get("block_count"),
    numHeads,
    numKvHeads: get("attention.head_count_kv", numHeads),
    headDim: get("attention.key_length", hiddenSize / numHeads),
    vocabSize: get("vocab_size", tokens?.length ?? 0),
    rmsEps: get("attention.layer_norm_rms_epsilon", 1e-6),
    ropeTheta: get("rope.freq_base", 10000),
    ropeInterleaved: arch === "llama",
    tieEmbeddings: !tensorNames.includes("output.weight"),
    attentionBias: tensorNames.includes("blk.0.attn_q.bias"),
    qkNorm: tensorNames.includes("blk.0.attn_q_norm.weight"),
    eosTokenIds: ids(meta.get("tokenizer.ggml.eos_token_id")),
    bosTokenId: meta.has("tokenizer.ggml.bos_token_id") ? Number(meta.get("tokenizer.ggml.bos_token_id")) : undefined,
  };
}

/** RoPE inverse frequencies, with Llama 3 long-context scaling applied when configured. */
export function ropeInvFreq(cfg: Pick<LlamaConfig, "headDim" | "ropeTheta" | "ropeScaling">): Float32Array {
  const half = cfg.headDim / 2;
  const out = new Float32Array(half);
  const s = cfg.ropeScaling;
  for (let i = 0; i < half; i++) {
    let f = 1 / cfg.ropeTheta ** ((2 * i) / cfg.headDim);
    if (s) {
      const lowWave = s.originalMaxPositions / s.lowFreqFactor;
      const highWave = s.originalMaxPositions / s.highFreqFactor;
      const wave = (2 * Math.PI) / f;
      if (wave > lowWave) f /= s.factor;
      else if (wave >= highWave) {
        const smooth = (s.originalMaxPositions / wave - s.lowFreqFactor) / (s.highFreqFactor - s.lowFreqFactor);
        f = (1 - smooth) * (f / s.factor) + smooth * f;
      }
    }
    out[i] = f;
  }
  return out;
}
