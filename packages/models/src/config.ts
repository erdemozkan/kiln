/** Normalised config for every Llama-shaped decoder (Llama 2/3, Mistral, Qwen2/2.5, Qwen3, SmolLM, Granite, Gemma 3). */
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
  /**
   * Granite: each attention and MLP output is multiplied by this before its residual add. Folded into
   * the o_proj and down_proj weights at load (see `weightScale`).
   */
  residualScale?: number;
  /**
   * Granite: logits are multiplied by this (1 / logits_scaling). Folded into the output head at load;
   * with a tied head the embedding table is scaled too, and `embedScale` undoes it.
   */
  logitScale?: number;
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

const SUPPORTED = ["llama", "mistral", "qwen2", "qwen3", "granite", "granitemoehybrid", "gemma3_text"];

type HFConfig = Record<string, unknown>;
/**
 * Largest values Kiln accepts for a model's sizes, by config.json key and by GGUF metadata key (after
 * the architecture prefix). Configs come with the repo; these keep a hostile or broken one from
 * sizing loops and GPU buffers in the billions before any weight is read. Real models sit far below.
 */
const SIZE_LIMITS: Record<string, number> = {
  hidden_size: 65536,
  intermediate_size: 262144,
  shared_intermediate_size: 262144,
  num_hidden_layers: 1024,
  num_attention_heads: 1024,
  num_key_value_heads: 1024,
  head_dim: 4096,
  vocab_size: 4194304,
  sliding_window_pattern: 1024,
  embedding_length: 65536,
  feed_forward_length: 262144,
  block_count: 1024,
  "attention.head_count": 1024,
  "attention.head_count_kv": 1024,
  "attention.key_length": 4096,
};

/**
 * A number read from a model's config, checked to be finite and below 2^24: sizes, counts and
 * positions of real models are far below, and nothing larger may size an allocation or a loop.
 */
export function configNumber(k: string, v: unknown): number {
  const n = Number(v);
  if (!Number.isFinite(n) || Math.abs(n) > 2 ** 24) throw new UnsupportedModelError(`config: "${k}" = ${String(v)} is out of range`);
  return n;
}

/** Throws unless a size from the repo is a positive integer within its limit (see SIZE_LIMITS). */
function sized(k: string, v: number): number {
  const max = SIZE_LIMITS[k];
  if (max !== undefined && !(Number.isInteger(v) && v >= 1 && v <= max)) throw new UnsupportedModelError(`config: "${k}" = ${v} is not a size Kiln accepts (1 to ${max})`);
  if (!Number.isFinite(v)) throw new UnsupportedModelError(`config: "${k}" is not a finite number`);
  return v;
}

const num = (c: HFConfig, k: string, fallback?: number): number => {
  const v = c[k] ?? fallback;
  if (typeof v !== "number") throw new UnsupportedModelError(`config.json is missing "${k}"`);
  return sized(k, v);
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
  const base: LlamaConfig = {
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
  return type === "granite" || type === "granitemoehybrid" ? granite(c, base) : base;
}

/**
 * Granite 3 (`granite`) and the attention-only, expert-free Granite 4 models (`granitemoehybrid` with
 * every layer "attention" and no experts, such as granite-4.0-350m and 1b): Llama with four
 * multipliers. Granite 4's MLP is one `shared_mlp.input_linear` (gate rows, then up rows), loaded as
 * the fused gate_up weight (see `graniteName`).
 */
function granite(c: HFConfig, base: LlamaConfig): LlamaConfig {
  if (c.model_type === "granitemoehybrid") {
    const types = Array.isArray(c.layer_types) ? (c.layer_types as string[]) : [];
    if (types.some((t) => t !== "attention")) throw new UnsupportedModelError("hybrid Granite models (Mamba layers) are not supported; the attention-only ones are");
    if (optNum(c.num_local_experts)) throw new UnsupportedModelError("Granite mixture-of-experts models are not supported");
    if ((c.position_embedding_type ?? "rope") !== "rope") throw new UnsupportedModelError(`Granite with position_embedding_type "${String(c.position_embedding_type)}" is not supported`);
  }
  if (c.mlp_bias) throw new UnsupportedModelError("Granite with mlp_bias is not supported");
  const logitScale = 1 / num(c, "logits_scaling", 1);
  const embed = num(c, "embedding_multiplier", 1);
  return {
    ...base,
    // Granite 4's MLP is the shared one; intermediate_size would size the (absent) experts.
    intermediateSize: c.model_type === "granitemoehybrid" ? num(c, "shared_intermediate_size", base.intermediateSize) : base.intermediateSize,
    embedScale: base.tieEmbeddings ? embed / logitScale : embed,
    attnScale: num(c, "attention_multiplier", base.headDim ** -0.5),
    residualScale: num(c, "residual_multiplier", 1),
    logitScale,
  };
}

/** Granite 4 checkpoints' MLP names → the fused gate_up and down_proj that the model reads. */
export function graniteName(name: string): string {
  return name.replace(".shared_mlp.input_linear.", ".mlp.gate_up.").replace(".shared_mlp.output_linear.", ".mlp.down_proj.");
}

/** The factor a weight is multiplied by at load (Granite's residual and logit multipliers), if any. */
export function weightScale(cfg: Pick<LlamaConfig, "residualScale" | "logitScale" | "tieEmbeddings">, name: string): number | undefined {
  if (cfg.residualScale !== undefined && /\.(o_proj|down_proj)\.weight$/.test(name)) return cfg.residualScale;
  if (cfg.logitScale !== undefined && (name === "lm_head.weight" || (cfg.tieEmbeddings && name === "model.embed_tokens.weight"))) return cfg.logitScale;
  return undefined;
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
  if (!["llama", "qwen2", "qwen3", "granite", "gemma3"].includes(arch)) {
    throw new UnsupportedModelError(`GGUF architecture "${arch}" is not supported yet (llama, qwen2, qwen3, granite, gemma3)`);
  }
  const get = (k: string, fallback?: number): number => {
    let v: unknown = meta.get(`${arch}.${k}`) ?? fallback;
    // Per-layer values (Granite 4 lists head_count_kv for every layer): Kiln needs them equal.
    if (Array.isArray(v)) {
      const vs = v as unknown[];
      if (!vs.length || vs.some((x) => Number(x) !== Number(vs[0]))) throw new UnsupportedModelError(`GGUF "${arch}.${k}" differs between layers`);
      v = vs[0];
    }
    if (v === undefined) throw new UnsupportedModelError(`GGUF metadata is missing "${arch}.${k}"`);
    return sized(k, Number(v));
  };
  const numHeads = get("attention.head_count");
  const hiddenSize = get("embedding_length");
  const tokens = meta.get("tokenizer.ggml.tokens") as unknown[] | undefined;
  if (arch === "gemma3") return gemma3FromGGUF(meta, tensorNames, get, tokens);
  if (arch === "granite" && get("expert_count", 0)) throw new UnsupportedModelError("Granite mixture-of-experts models are not supported");
  const base: LlamaConfig = {
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
    // llama.cpp permutes Q/K of llama-converted checkpoints, Granite's included.
    ropeInterleaved: arch === "llama" || arch === "granite",
    tieEmbeddings: !tensorNames.includes("output.weight"),
    attentionBias: tensorNames.includes("blk.0.attn_q.bias"),
    qkNorm: tensorNames.includes("blk.0.attn_q_norm.weight"),
    eosTokenIds: ids(meta.get("tokenizer.ggml.eos_token_id")),
    bosTokenId: meta.has("tokenizer.ggml.bos_token_id") ? Number(meta.get("tokenizer.ggml.bos_token_id")) : undefined,
    contextLength: optNum(meta.get(`${arch}.context_length`)),
  };
  if (arch !== "granite") return base;
  const logitScale = 1 / get("logit_scale", 1);
  const embed = get("embedding_scale", 1);
  return {
    ...base,
    embedScale: base.tieEmbeddings ? embed / logitScale : embed,
    attnScale: get("attention.scale", base.headDim ** -0.5),
    residualScale: get("residual_scale", 1),
    logitScale,
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
