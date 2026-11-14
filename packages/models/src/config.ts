/** Normalised config for every Llama-shaped decoder (Llama 2/3, Mistral, Qwen2/2.5, Qwen3, SmolLM, Granite, Gemma 3, Phi-3). */
export interface LlamaConfig {
  family: "llama" | "qwen2" | "qwen3" | "qwen35" | "gemma3" | "phi3";
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
  /**
   * LongRoPE (Phi-3.5, Phi-3 128k): each rotary frequency is divided by its factor. These are the
   * checkpoint's short factors, which hold up to `maxPositions` tokens.
   */
  ropeFactors?: number[];
  /**
   * Partial rotary (Phi-4 mini): RoPE turns only the first `rotaryDims` elements of each head (pairs
   * i and i + rotaryDims / 2); the rest carry no position. Unset: the whole head.
   */
  rotaryDims?: number;
  /** The longest context this config is valid for; a longer `maxSeq` is refused (LongRoPE's short factors). */
  maxPositions?: number;
  /** GGUF llama checkpoints have Q/K permuted for interleaved (GPT-J style) rotary pairs. */
  ropeInterleaved: boolean;
  tieEmbeddings: boolean;
  attentionBias: boolean;
  /** Per-head RMSNorm on q and k (Qwen3). */
  qkNorm: boolean;
  eosTokenIds: number[];
  bosTokenId?: number;
  /**
   * Qwen 3.5: the layers that mix tokens with linear attention (a gated delta net: a causal
   * convolution, then a recurrent state per head) where the others use attention, and its sizes.
   */
  linearLayers?: boolean[];
  linear?: LinearAttentionConfig;
  /** Qwen 3.5: q_proj computes a gate beside each head's query; the attention output is multiplied by its sigmoid. */
  attnOutputGate?: boolean;
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

/** Sizes of a linear-attention (gated delta net) layer. */
export interface LinearAttentionConfig {
  /** Heads of the queries and keys, and of the values (a multiple of the former: each key head serves that many value heads). */
  keyHeads: number;
  valueHeads: number;
  keyDim: number;
  valueDim: number;
  /** Taps of the causal depthwise convolution over time. */
  convKernel: number;
  /**
   * How value heads map to key heads when there are more of them. Checkpoints group them (value head
   * h uses key head ⌊h / (Hv / Hk)⌋); llama.cpp's converter reorders them so that it is h mod Hk.
   */
  tiled?: boolean;
}

export interface Llama3RopeScaling {
  factor: number;
  lowFreqFactor: number;
  highFreqFactor: number;
  originalMaxPositions: number;
}

export class UnsupportedModelError extends Error {}

/** The tensor a phi3 GGUF keeps LongRoPE's short factors in. */
export const PHI3_ROPE_SHORT = "rope_factors_short.weight";

const SUPPORTED = ["llama", "mistral", "qwen2", "qwen3", "qwen3_5_text", "granite", "granitemoehybrid", "gemma3_text", "phi3"];

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
  linear_num_key_heads: 1024,
  linear_num_value_heads: 1024,
  linear_key_head_dim: 256,
  linear_value_head_dim: 256,
  linear_conv_kernel_dim: 16,
  "ssm.conv_kernel": 16,
  "ssm.state_size": 256,
  "ssm.group_count": 1024,
  "ssm.time_step_rank": 1024,
  "ssm.inner_size": 262144,
  full_attention_interval: 1024,
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
  if (type === "qwen3_5") {
    // The multimodal checkpoint (every Qwen 3.5 is one): run its language model, ignore the vision tower.
    const text = c.text_config as HFConfig | undefined;
    if (!text) throw new UnsupportedModelError('model_type "qwen3_5" without a text_config');
    return configFromHF({ ...text, model_type: "qwen3_5_text", eos_token_id: [...ids(c.eos_token_id), ...ids(text.eos_token_id)], tie_word_embeddings: text.tie_word_embeddings ?? c.tie_word_embeddings }, generation);
  }
  if (!SUPPORTED.includes(type)) {
    throw new UnsupportedModelError(`model_type "${type}" is not supported yet (supported: ${SUPPORTED.join(", ")})`);
  }
  const numHeads = num(c, "num_attention_heads");
  const hiddenSize = num(c, "hidden_size");
  const family = type === "qwen2" ? "qwen2" : type === "qwen3" ? "qwen3" : type === "qwen3_5_text" ? "qwen35" : type === "gemma3_text" ? "gemma3" : type === "phi3" ? "phi3" : "llama";
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
    ropeScaling: type === "phi3" ? undefined : ropeScalingFromHF(c.rope_scaling as HFConfig | null | undefined),
    ropeInterleaved: false,
    tieEmbeddings: Boolean(c.tie_word_embeddings ?? false),
    attentionBias: family === "qwen2" ? true : Boolean(c.attention_bias ?? false),
    qkNorm: family === "qwen3" || family === "qwen35",
    eosTokenIds: [...new Set([...ids(c.eos_token_id), ...ids(generation?.eos_token_id)])],
    bosTokenId: typeof c.bos_token_id === "number" ? c.bos_token_id : undefined,
    contextLength: optNum(c.max_position_embeddings),
  };
  if (type === "phi3") return phi3(c, base);
  if (family === "qwen35") return qwen35(c, base);
  return type === "granite" || type === "granitemoehybrid" ? granite(c, base) : base;
}

/**
 * Qwen 3.5's language model: most layers are linear attention (`linearLayers`), every fourth is
 * attention with a gated output, q/k norms and rotary on the first quarter of each head. Text
 * positions are the same on all three axes of its multimodal rotary, which is then plain rotary.
 */
function qwen35(c: HFConfig, base: LlamaConfig): LlamaConfig {
  const types = c.layer_types;
  if (!Array.isArray(types) || types.length !== base.numLayers || !types.every((t) => t === "linear_attention" || t === "full_attention")) {
    throw new UnsupportedModelError("qwen3_5: layer_types must name every layer as linear_attention or full_attention");
  }
  if (Array.isArray(c.mlp_only_layers) && c.mlp_only_layers.length) throw new UnsupportedModelError("qwen3_5 with mlp_only_layers is not supported yet");
  const rope = (c.rope_parameters ?? {}) as HFConfig;
  if ((rope.rope_type ?? "default") !== "default") throw new UnsupportedModelError(`qwen3_5 rope_type "${String(rope.rope_type)}" is not supported yet`);
  const factor = typeof rope.partial_rotary_factor === "number" ? rope.partial_rotary_factor : 1;
  const linear: LinearAttentionConfig = {
    keyHeads: num(c, "linear_num_key_heads"),
    valueHeads: num(c, "linear_num_value_heads"),
    keyDim: num(c, "linear_key_head_dim"),
    valueDim: num(c, "linear_value_head_dim"),
    convKernel: num(c, "linear_conv_kernel_dim"),
  };
  if (linear.valueHeads % linear.keyHeads !== 0) throw new UnsupportedModelError("qwen3_5: value heads must be a multiple of key heads");
  // The kernels take a head of at most 256, in whole vec4s, and at least two taps.
  if (linear.keyDim % 4 !== 0 || linear.valueDim % 4 !== 0 || linear.convKernel < 2) throw new UnsupportedModelError("qwen3_5: these linear attention sizes are not supported");
  return {
    ...base,
    ropeTheta: typeof rope.rope_theta === "number" ? configNumber("rope_theta", rope.rope_theta) : base.ropeTheta,
    ...(factor !== 1 ? { rotaryDims: partialRotary(base.headDim, Math.floor(base.headDim * factor)) } : {}),
    linearLayers: types.map((t) => t === "linear_attention"),
    linear,
    attnOutputGate: c.attn_output_gate !== false,
  };
}

/**
 * Phi-3 (`phi3`: Phi-3 mini and medium, Phi-3.5 mini): Llama with q/k/v stored as one `qkv_proj` and
 * gate/up as one `gate_up_proj` (see `phi3Name`), which is how Kiln runs them anyway. The 128k
 * variants use LongRoPE: per-frequency factors, a short set up to the original context and a long set
 * past it, and attention inputs scaled by a constant. Kiln applies the short set, so those models run
 * up to their original context (4096 tokens). Phi-4 mini rotates only part of each head (`rotaryDims`).
 */
function phi3(c: HFConfig, plain: LlamaConfig): LlamaConfig {
  // Phi-4 mini rotates the first 3/4 of each head (int(head_dim × factor), as transformers computes it).
  const factor = c.partial_rotary_factor;
  const base = typeof factor === "number" && factor !== 1 ? { ...plain, rotaryDims: partialRotary(plain.headDim, Math.floor(plain.headDim * factor)) } : plain;
  const s = c.rope_scaling as HFConfig | null | undefined;
  if (!s) return base;
  const type = s.rope_type ?? s.type;
  if (type !== "longrope" && type !== "su" && type !== "yarn") throw new UnsupportedModelError(`phi3 rope_scaling type "${String(type)}" is not supported yet`);
  if (type === "yarn") throw new UnsupportedModelError('phi3 rope_scaling type "yarn" is not supported yet (longrope is)');
  const original = num(c, "original_max_position_embeddings");
  return longRope(base, s.short_factor, original, base.contextLength ?? original, optNum(s.attention_factor));
}

/** Checks a partial rotary size: an even number of a head's elements, at least one pair, at most all. */
function partialRotary(headDim: number, dims: number): number {
  if (!Number.isInteger(dims) || dims < 2 || dims > headDim || dims % 2 !== 0) throw new UnsupportedModelError(`rotary embedding on ${dims} of ${headDim} head dimensions is not supported`);
  return dims;
}

/** Applies LongRoPE's short factors and its attention factor (√(1 + ln(scale) / ln(original)) unless the checkpoint states one). */
function longRope(base: LlamaConfig, short: unknown, original: number, trained: number, attnFactor?: number): LlamaConfig {
  const half = (base.rotaryDims ?? base.headDim) / 2;
  if (!Array.isArray(short) || short.length !== half || short.some((v) => typeof v !== "number" || !(v > 0) || !Number.isFinite(v))) {
    throw new UnsupportedModelError(`LongRoPE needs ${half} positive short factors`);
  }
  // These come from the repo and set the context size and a scale on every attention input.
  if (!Number.isInteger(original) || original < 2 || original > 2 ** 20) throw new UnsupportedModelError(`LongRoPE: an original context of ${original} tokens is out of range`);
  if (!(trained >= 1 && trained <= 2 ** 24)) throw new UnsupportedModelError(`LongRoPE: a context of ${trained} tokens is out of range`);
  if (attnFactor !== undefined && !(attnFactor > 0 && attnFactor <= 16)) throw new UnsupportedModelError(`LongRoPE: an attention factor of ${attnFactor} is out of range`);
  const scale = trained / original;
  const factor = attnFactor ?? (scale <= 1 ? 1 : Math.sqrt(1 + Math.log(scale) / Math.log(original)));
  // cos and sin are both multiplied by the factor, so q·k grows by its square.
  return { ...base, ropeFactors: short as number[], attnScale: factor ** 2 * base.headDim ** -0.5, contextLength: original, maxPositions: original };
}

/** Phi-3's stacked projections under the names Kiln's stacked matmuls use. */
export function phi3Name(name: string): string {
  return name.replace(".self_attn.qkv_proj.", ".self_attn.qkv.").replace(".mlp.gate_up_proj.", ".mlp.gate_up.");
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

/**
 * `ropeShort` holds the values of a phi3 file's `rope_factors_short.weight` tensor (LongRoPE), which
 * the caller reads from the file: they are data, not metadata.
 */
export function configFromGGUF(meta: Map<string, unknown>, tensorNames: readonly string[], ropeShort?: ArrayLike<number>): LlamaConfig {
  const arch = String(meta.get("general.architecture"));
  if (!["llama", "qwen2", "qwen3", "qwen35", "granite", "gemma3", "phi3"].includes(arch)) {
    throw new UnsupportedModelError(`GGUF architecture "${arch}" is not supported yet (llama, qwen2, qwen3, qwen35, granite, gemma3, phi3)`);
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
    // Qwen 3.5's first layer is linear attention, so its q/k norms are not in block 0.
    qkNorm: arch === "qwen35" || tensorNames.includes("blk.0.attn_q_norm.weight"),
    eosTokenIds: ids(meta.get("tokenizer.ggml.eos_token_id")),
    bosTokenId: meta.has("tokenizer.ggml.bos_token_id") ? Number(meta.get("tokenizer.ggml.bos_token_id")) : undefined,
    contextLength: optNum(meta.get(`${arch}.context_length`)),
  };
  if (arch === "phi3") {
    // Phi-4 mini: fewer rotated dimensions than the head has.
    const rotary = get("rope.dimension_count", base.headDim);
    const phi = rotary !== base.headDim ? { ...base, rotaryDims: partialRotary(base.headDim, rotary) } : base;
    if (!tensorNames.includes(PHI3_ROPE_SHORT)) return phi;
    if (!ropeShort) throw new UnsupportedModelError(`phi3 with LongRoPE: the values of ${PHI3_ROPE_SHORT} are needed`);
    const original = get("rope.scaling.original_context_length");
    return longRope(phi, Array.from(ropeShort), original, phi.contextLength ?? original, optNum(meta.get("phi3.rope.scaling.attn_factor")));
  }
  if (arch === "qwen35") {
    // Which layers are linear attention: listed, or every layer but each `interval`-th.
    const listed = meta.get("qwen35.attention.recurrent_layers");
    const interval = get("full_attention_interval", 4);
    const linearLayers = Array.from({ length: base.numLayers }, (_, l) => (Array.isArray(listed) ? Boolean(listed[l]) : (l + 1) % interval !== 0));
    const first = linearLayers.indexOf(true);
    // Some conversions merge the q/k/v and gate projections (and the two small ones) into one tensor
    // each; the loader lists their halves under the usual names (see `splitQwen35Merged`).
    if (first >= 0 && !tensorNames.includes(`blk.${first}.attn_gate.weight`) && !tensorNames.includes(`blk.${first}.attn_qkv_gate.weight`)) {
      throw new UnsupportedModelError(`this Qwen 3.5 GGUF has neither blk.${first}.attn_gate nor blk.${first}.attn_qkv_gate: linear attention's projections are missing`);
    }
    const valueHeads = get("ssm.time_step_rank");
    const keyHeads = get("ssm.group_count");
    const keyDim = get("ssm.state_size");
    const inner = get("ssm.inner_size");
    const linear: LinearAttentionConfig = { keyHeads, valueHeads, keyDim, valueDim: inner / valueHeads, convKernel: get("ssm.conv_kernel"), ...(valueHeads !== keyHeads ? { tiled: true } : {}) };
    if (!Number.isInteger(linear.valueDim) || valueHeads % keyHeads !== 0 || keyDim % 4 !== 0 || linear.valueDim % 4 !== 0 || keyDim > 256 || linear.valueDim > 256 || linear.convKernel < 2 || linear.convKernel > 16) {
      throw new UnsupportedModelError("qwen35: these linear attention sizes are not supported");
    }
    const rotary = get("rope.dimension_count", base.headDim);
    return { ...base, ...(rotary !== base.headDim ? { rotaryDims: partialRotary(base.headDim, rotary) } : {}), linearLayers, linear, attnOutputGate: true };
  }
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
export function ropeInvFreq(cfg: Pick<LlamaConfig, "headDim" | "ropeTheta" | "ropeScaling" | "ropeLinear" | "ropeFactors" | "rotaryDims">): Float32Array {
  // One frequency per rotated pair, spread over the rotated part of the head.
  const dims = cfg.rotaryDims ?? cfg.headDim;
  const half = dims / 2;
  const out = new Float32Array(half);
  const s = cfg.ropeScaling;
  for (let i = 0; i < half; i++) {
    let f = 1 / cfg.ropeTheta ** ((2 * i) / dims);
    if (cfg.ropeLinear) f /= cfg.ropeLinear;
    if (cfg.ropeFactors) f /= cfg.ropeFactors[i]!;
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
