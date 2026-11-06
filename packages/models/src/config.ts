/** Normalised config for every Llama-shaped decoder (Llama 2/3, Mistral, Qwen2/2.5, Qwen3, SmolLM, Gemma 3). */
export interface LlamaConfig {
  family: "llama" | "qwen2" | "qwen3" | "gemma3";
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
  /**
   * Linear rope scaling: positions divided by this factor (inverse frequencies likewise). For Gemma 3
   * it applies to the global layers only; the sliding layers' rope is unscaled.
   */
  ropeLinear?: number;
  /** GGUF llama checkpoints have Q/K permuted for interleaved (GPT-J style) rotary pairs. */
  ropeInterleaved: boolean;
  tieEmbeddings: boolean;
  attentionBias: boolean;
  /** Per-head RMSNorm on q and k (Qwen3). */
  qkNorm: boolean;
  eosTokenIds: number[];
  bosTokenId?: number;
  /** Gated MLP activation (default silu). */
  activation?: "silu" | "gelu";
  /** Embeddings are multiplied by this (Gemma: sqrt(hidden), rounded to the checkpoint's dtype). */
  embedScale?: number;
  /**
   * Gemma: the attention and MLP outputs are normalised before each residual add
   * (post_attention_layernorm, post_feedforward_layernorm), and the MLP input by pre_feedforward_layernorm.
   */
  sandwichNorms?: boolean;
  /** Attention score scale (default 1 / sqrt(headDim)). */
  attnScale?: number;
  /** Keys a sliding-window layer sees, and which layers slide (the others attend to everything). */
  slidingWindow?: number;
  slidingLayers?: boolean[];
  /** RoPE base of the sliding layers, when it differs from `ropeTheta` (Gemma 3: 10k local, 1M global). */
  ropeLocalTheta?: number;
  /** Longest context the model was trained for (max_position_embeddings / GGUF context_length), if stated. */
  contextLength?: number;
}

const optNum = (v: unknown): number | undefined => (typeof v === "number" && v > 0 ? v : typeof v === "bigint" && v > 0n ? Number(v) : undefined);

export interface Llama3RopeScaling {
  factor: number;
  lowFreqFactor: number;
  highFreqFactor: number;
  originalMaxPositions: number;
}

export class UnsupportedModelError extends Error {}

const SUPPORTED = ["llama", "mistral", "qwen2", "qwen3", "gemma3_text"];

type HFConfig = Record<string, unknown>;
const num = (c: HFConfig, k: string, fallback?: number): number => {
  const v = c[k] ?? fallback;
  if (typeof v !== "number") throw new UnsupportedModelError(`config.json is missing "${k}"`);
  return v;
};
const ids = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : typeof v === "number" ? [v] : []);

/** Weights of the multimodal Gemma 3 checkpoints (4B and up) that a text model reads, under their text names. */
export function multimodalTextName(name: string): string | undefined {
  // transformers 4.50–4.51 save "language_model.model.layers…", later versions "model.language_model.layers…".
  if (name.startsWith("language_model.")) return name.slice("language_model.".length);
  if (name.startsWith("model.language_model.")) return `model.${name.slice("model.language_model.".length)}`;
  if (name === "lm_head.weight") return name;
  return undefined; // vision tower and projector
}

/** Gemma3TextConfig's defaults, for configs that leave fields out (the 4B and larger text_config). */
const GEMMA3_DEFAULTS: HFConfig = {
  vocab_size: 262208,
  hidden_size: 2304,
  intermediate_size: 9216,
  num_hidden_layers: 26,
  num_attention_heads: 8,
  num_key_value_heads: 4,
  head_dim: 256,
  rms_norm_eps: 1e-6,
  rope_theta: 1000000,
  rope_local_base_freq: 10000,
  query_pre_attn_scalar: 256,
  sliding_window: 4096,
  max_position_embeddings: 131072,
  sliding_window_pattern: 6,
};

export function configFromHF(c: HFConfig, generation?: HFConfig): LlamaConfig {
  const type = String(c.model_type);
  if (type === "gemma3") {
    // The multimodal checkpoint: run its language model and ignore the vision tower.
    const text = c.text_config as HFConfig | undefined;
    if (!text) throw new UnsupportedModelError('model_type "gemma3" without a text_config');
    const eos = [...ids(c.eos_token_id), ...ids(text.eos_token_id)];
    return gemma3({ ...GEMMA3_DEFAULTS, torch_dtype: c.torch_dtype, ...text, eos_token_id: eos, bos_token_id: text.bos_token_id ?? c.bos_token_id }, generation);
  }
  if (!SUPPORTED.includes(type)) {
    throw new UnsupportedModelError(`model_type "${type}" is not supported yet (supported: ${SUPPORTED.join(", ")})`);
  }
  const numHeads = num(c, "num_attention_heads");
  const hiddenSize = num(c, "hidden_size");
  const family = type === "qwen2" ? "qwen2" : type === "qwen3" ? "qwen3" : type === "gemma3_text" ? "gemma3" : "llama";
  if (family === "gemma3") return gemma3({ ...GEMMA3_DEFAULTS, ...c }, generation);
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
    contextLength: optNum(c.max_position_embeddings),
  };
}

/** Rounds to bfloat16 (round to nearest even), as a bf16 checkpoint's constants are in HF. */
function bf16(x: number): number {
  const u = new Uint32Array(new Float32Array([x]).buffer);
  u[0] = (u[0]! + 0x7fff + ((u[0]! >>> 16) & 1)) & 0xffff0000;
  return new Float32Array(u.buffer)[0]!;
}

function gemma3(c: HFConfig, generation?: HFConfig): LlamaConfig {
  for (const k of ["attn_logit_softcapping", "final_logit_softcapping"]) {
    if (c[k] != null) throw new UnsupportedModelError(`Gemma 3 with ${k} is not supported yet`);
  }
  if (c.use_bidirectional_attention) throw new UnsupportedModelError("Gemma with bidirectional attention (an embedding model) is not supported");
  const act = String(c.hidden_activation ?? c.hidden_act ?? "gelu_pytorch_tanh");
  if (act !== "gelu_pytorch_tanh") throw new UnsupportedModelError(`Gemma activation "${act}" is not supported (gelu_pytorch_tanh only)`);
  const scaling = c.rope_scaling as HFConfig | null | undefined;
  const scalingType = scaling && (scaling.rope_type ?? scaling.type);
  if (scaling && scalingType !== "linear" && scalingType !== "default") {
    throw new UnsupportedModelError(`Gemma 3 with rope_scaling "${String(scalingType)}" is not supported (linear only)`);
  }
  const numLayers = num(c, "num_hidden_layers");
  const hiddenSize = num(c, "hidden_size");
  const numHeads = num(c, "num_attention_heads");
  const headDim = num(c, "head_dim", 256);
  // Newer configs list layer types; older ones give a period: every Nth layer is global.
  const types = Array.isArray(c.layer_types) ? (c.layer_types as string[]) : undefined;
  const period = num(c, "sliding_window_pattern", (c._sliding_window_pattern as number | undefined) ?? 6);
  const slidingLayers = Array.from({ length: numLayers }, (_, l) => (types ? types[l] === "sliding_attention" : (l + 1) % period !== 0));
  const dtype = String(c.torch_dtype ?? c.dtype ?? "bfloat16");
  return {
    family: "gemma3",
    hiddenSize,
    intermediateSize: num(c, "intermediate_size"),
    numLayers,
    numHeads,
    numKvHeads: num(c, "num_key_value_heads", numHeads),
    headDim,
    vocabSize: num(c, "vocab_size"),
    rmsEps: num(c, "rms_norm_eps", 1e-6),
    ropeTheta: num(c, "rope_theta", 1000000),
    ropeLinear: scalingType === "linear" ? num(scaling!, "factor") : undefined,
    ropeLocalTheta: num(c, "rope_local_base_freq", 10000),
    ropeInterleaved: false,
    tieEmbeddings: Boolean(c.tie_word_embeddings ?? true),
    attentionBias: Boolean(c.attention_bias ?? false),
    qkNorm: true,
    eosTokenIds: [...new Set([...ids(c.eos_token_id), ...ids(generation?.eos_token_id)])],
    bosTokenId: typeof c.bos_token_id === "number" ? c.bos_token_id : undefined,
    activation: "gelu",
    embedScale: dtype === "bfloat16" ? bf16(Math.sqrt(hiddenSize)) : Math.fround(Math.sqrt(hiddenSize)),
    sandwichNorms: true,
    attnScale: num(c, "query_pre_attn_scalar", headDim) ** -0.5,
    slidingWindow: num(c, "sliding_window", 512),
    slidingLayers,
    contextLength: optNum(c.max_position_embeddings),
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
  if (!["llama", "qwen2", "qwen3", "gemma3"].includes(arch)) {
    throw new UnsupportedModelError(`GGUF architecture "${arch}" is not supported yet (llama, qwen2, qwen3, gemma3)`);
  }
  const get = (k: string, fallback?: number): number => {
    const v = meta.get(`${arch}.${k}`) ?? fallback;
    if (v === undefined) throw new UnsupportedModelError(`GGUF metadata is missing "${arch}.${k}"`);
    return Number(v);
  };
  const numHeads = get("attention.head_count");
  const hiddenSize = get("embedding_length");
  const tokens = meta.get("tokenizer.ggml.tokens") as unknown[] | undefined;
  if (arch === "gemma3") return gemma3FromGGUF(meta, tensorNames, get, tokens);
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
    contextLength: optNum(meta.get(`${arch}.context_length`)),
  };
}

/**
 * GGUF gemma3, as llama.cpp writes and reads it: norm weights are stored with Gemma's 1 + w already
 * applied, the sliding layers' rope base is 10k and every 6th layer is global (neither is in the
 * metadata), and attention is scaled by head_dim^-0.5, or by (hidden / heads)^-0.5 for the 62-layer 27B.
 */
function gemma3FromGGUF(meta: Map<string, unknown>, tensorNames: readonly string[], get: (k: string, fallback?: number) => number, tokens: unknown[] | undefined): LlamaConfig {
  const scaling = meta.get("gemma3.rope.scaling.type");
  if (scaling !== undefined && scaling !== "none" && scaling !== "linear") {
    throw new UnsupportedModelError(`Gemma 3 with rope scaling "${String(scaling)}" is not supported (linear only)`);
  }
  for (const k of ["attn_logit_softcapping", "final_logit_softcapping"]) {
    if (meta.get(`gemma3.${k}`)) throw new UnsupportedModelError(`Gemma 3 with ${k} is not supported yet`);
  }
  const numLayers = get("block_count");
  const hiddenSize = get("embedding_length");
  const numHeads = get("attention.head_count");
  const headDim = get("attention.key_length", 256);
  const pattern = meta.get("gemma3.attention.sliding_window_pattern");
  const slidingLayers = Array.from({ length: numLayers }, (_, l) =>
    Array.isArray(pattern) ? Boolean(pattern[l]) : (l + 1) % (typeof pattern === "number" ? pattern : 6) !== 0,
  );
  const eos = ids(meta.get("tokenizer.ggml.eos_token_id"));
  // Chat turns end with <end_of_turn>; some files name <eos> as the end-of-sequence token instead.
  const eot = tokens?.indexOf("<end_of_turn>") ?? -1;
  return {
    family: "gemma3",
    hiddenSize,
    intermediateSize: get("feed_forward_length"),
    numLayers,
    numHeads,
    numKvHeads: get("attention.head_count_kv", numHeads),
    headDim,
    vocabSize: get("vocab_size", tokens?.length ?? 0),
    rmsEps: get("attention.layer_norm_rms_epsilon", 1e-6),
    ropeTheta: get("rope.freq_base", 1000000),
    ropeLinear: scaling === "linear" ? get("rope.scaling.factor") : undefined,
    ropeLocalTheta: 10000,
    ropeInterleaved: false,
    tieEmbeddings: !tensorNames.includes("output.weight"),
    attentionBias: false,
    qkNorm: true,
    eosTokenIds: [...new Set([...eos, ...(eot >= 0 ? [eot] : [])])],
    bosTokenId: meta.has("tokenizer.ggml.bos_token_id") ? Number(meta.get("tokenizer.ggml.bos_token_id")) : undefined,
    activation: "gelu",
    // As transformers computes it for the bf16 checkpoints these files come from (llama.cpp skips the rounding).
    embedScale: bf16(Math.sqrt(hiddenSize)),
    sandwichNorms: true,
    attnScale: (numLayers === 62 ? hiddenSize / numHeads : headDim) ** -0.5,
    slidingWindow: get("attention.sliding_window", 512),
    slidingLayers,
    contextLength: optNum(meta.get("gemma3.context_length")),
  };
}

/** RoPE inverse frequencies, with Llama 3 long-context scaling applied when configured. */
export function ropeInvFreq(cfg: Pick<LlamaConfig, "headDim" | "ropeTheta" | "ropeScaling" | "ropeLinear">): Float32Array {
  const half = cfg.headDim / 2;
  const out = new Float32Array(half);
  const s = cfg.ropeScaling;
  for (let i = 0; i < half; i++) {
    let f = 1 / cfg.ropeTheta ** ((2 * i) / cfg.headDim);
    if (cfg.ropeLinear) f /= cfg.ropeLinear;
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
