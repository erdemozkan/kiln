import { ActivationPool, Ctx, STEP_BYTES, physical, byteSize, isBlockQuant, packStep, q4ScaleBytes, view, type Backend, type Buf, type Dispatch, type StepParams, type Tensor } from "@kiln/core";
import { ATTN_WG, ops, type KvDtype } from "@kiln/kernels";
import { headDimOf, ropeInvFreq, type LlamaConfig } from "./config";
import { gemma4PerLayerTable } from "./names";

/** A recorded forward pass for a fixed number of tokens. */
export interface ForwardPlan {
  readonly T: number;
  /** The forward pass; the last dispatch samples, so it can be submitted separately from the rest. */
  readonly dispatches: readonly Dispatch[];
  /**
   * The layers only, without the output head and sampling: for prefill chunks before the last, whose
   * logits nobody reads (the head is a vocabulary-sized matmul). Same as `dispatches` for verify plans.
   */
  readonly body: readonly Dispatch[];
  /**
   * Gemma 4: the dispatches up to the last layer that writes keys and values. The layers after it
   * read an earlier layer's cache and leave nothing a later token needs, so a prefill chunk whose
   * logits nobody reads stops here. A chunk whose logits are read runs those layers on one row only,
   * `step.lastRow`: the others' outputs there would feed nothing.
   */
  readonly kvBody?: readonly Dispatch[];
  /** u32 [T] token ids, written before each replay. */
  readonly tokens: Buf;
  /** u32 [1] sampled token for row `step.lastRow`; u32 [T], one per row, for a verify plan. */
  readonly out: Buf;
  /** Logits [1, vocab] (or [T, vocab] for a verify plan) before sampling, for inspection and tests. */
  readonly logits: Tensor;
  destroy(): void;
}

/** The projections a LoRA adapter may update (PEFT's target_modules names). */
export const LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] as const;
export type LoraModule = (typeof LORA_MODULES)[number];

/** One projection's low-rank update with its scale folded in: ΔW = B · A, A [rank, in], B [out, rank]. */
export interface LoraUpdate {
  rank: number;
  A: Float32Array;
  B: Float32Array;
}

/** An adapter's updates by "layer.module" ("3.q_proj"). */
export type LoraWeights = ReadonlyMap<string, LoraUpdate>;

/** An adapter laid out for a model's LoRA slots of one rank, in GPU buffers (see `LlamaModel.prepareLora`). */
export interface LoraBuffers {
  rank: number;
  bufs: { A: Buf; B: Buf; aBytes: number; bBytes: number }[];
  bytes: number;
}

/**
 * GPU buffers one adapted matmul adds its update from: A [parts · R, in] and B [out, parts · R],
 * where a stacked matmul (q/k/v, gate/up) takes its parts' A rows stacked and their B blocks on the
 * diagonal. R is the slot rank (a multiple of 32); smaller adapters leave the rest zero.
 */
interface LoraSlot {
  parts: { module: LoraModule; n: number }[];
  k: number;
  A: Tensor;
  B: Tensor;
}

export interface LlamaOptions {
  /** KV cache length in tokens (default `defaultMaxSeq`). */
  maxSeq?: number;
  /**
   * KV cache storage: f16 (default) halves the cache and the attention reads; f32 for exactness; f8
   * (8-bit floats, FP8 E4M3) and i8 (signed bytes with a scale per head) halve it again, so twice
   * the context fits the same memory, at some cost in fidelity: i8's is several times smaller.
   */
  kvDtype?: KvDtype;
  /**
   * Stack q/k/v and gate/up into single matmuls (default true). False keeps the unfused graph; an
   * object chooses each (missing parts default to true).
   */
  fuse?: boolean | { qkv?: boolean; mlp?: boolean };
  /**
   * Rows of image features to hold (Gemma 3 with its vision tower): token ids from vocabSize on read
   * row id - vocabSize of `imageFeatures` instead of the embedding table. Default 0.
   */
  imageTokens?: number;
}

export class MissingWeightError extends Error {}

/** KV cache memory the default context may take, and the longest default context. */
export const KV_BUDGET_BYTES = 512 * 2 ** 20;
export const MAX_DEFAULT_SEQ = 8192;

/**
 * Rows a sliding layer's ring holds beyond its window: keys written ahead of the oldest one a query
 * still needs, up to the largest prefill chunk: 128 rows of text, or one image's 256 tokens, which
 * are prefilled together for their bidirectional attention.
 */
export const RING_SLACK = 256;

/**
 * Cache slots of layer l: maxSeq for full layers; for sliding layers a ring of window + RING_SLACK
 * slots (at least ATTN_WG, which the attention kernels' key runs need), when that is smaller.
 */
export function ringSlots(cfg: LlamaConfig, maxSeq: number, l: number): number {
  // A linear-attention layer keeps a recurrent state, not keys and values.
  if (cfg.linearLayers?.[l]) return 0;
  // Nor does a layer that reads another layer's keys and values (Gemma 4).
  if (cfg.kvFrom?.[l] !== undefined) return 0;
  const win = cfg.slidingLayers?.[l] ? cfg.slidingWindow : undefined;
  return win ? Math.min(maxSeq, Math.max(ATTN_WG, win + RING_SLACK)) : maxSeq;
}

/** Bytes one cached value takes. */
const KV_BYTES: Record<KvDtype, number> = { f32: 4, f16: 2, f8: 1, i8: 1, q4: 0.625 };

/**
 * Bytes of one position's keys (part "k") or values ("v") in layer l's cache: every head's values, and
 * for i8 a scale word per head. A q4 cache holds keys as i8 and values in 5 words per 32 (the scales
 * among them); the other types hold both alike.
 */
export function kvRowBytesOf(cfg: LlamaConfig, kvDtype: KvDtype, l = 0, part: "k" | "v" = "k"): number {
  const as: KvDtype = kvDtype === "q4" && part === "k" ? "i8" : kvDtype;
  return cfg.numKvHeads * (headDimOf(cfg, l) * KV_BYTES[as] + (as === "i8" ? 4 : 0));
}

/** KV cache bytes of a model at a context length: keys and values of every layer, rings included. */
export function kvCacheBytes(cfg: LlamaConfig, maxSeq: number, kvDtype: KvDtype = "f16"): number {
  let bytes = 0;
  for (let l = 0; l < cfg.numLayers; l++) bytes += ringSlots(cfg, maxSeq, l) * (kvRowBytesOf(cfg, kvDtype, l, "k") + kvRowBytesOf(cfg, kvDtype, l, "v"));
  return bytes;
}

/**
 * Default context length: the model's trained context, at most MAX_DEFAULT_SEQ tokens, and no more
 * than fits KV_BUDGET_BYTES of cache (in multiples of 256, never under 2048 unless the model's own
 * context is shorter). Sliding layers count as their rings: Gemma 3 4B takes 320 MB at 8192 tokens,
 * where a full cache per layer would take 1.1 GB.
 */
export function defaultMaxSeq(cfg: LlamaConfig, kvDtype: KvDtype = "f16"): number {
  const top = Math.min(cfg.contextLength ?? 2048, MAX_DEFAULT_SEQ);
  const floor = Math.min(top, 2048);
  let n = top;
  while (n - 256 >= floor && kvCacheBytes(cfg, n, kvDtype) > KV_BUDGET_BYTES) n = Math.ceil(n / 256) * 256 - 256;
  return n;
}

/** Most rows a verify plan can have (the last token plus a draft of up to 31). */
export const MAX_VERIFY_ROWS = 32;

export class LlamaModel {
  readonly maxSeq: number;
  readonly kvDtype: KvDtype;
  readonly step: Buf;
  /** Gemma 4: the step as the layers that run on row `lastRow` alone see it (that row's position, row 0). */
  private readonly tailStep?: Buf;
  /** u32 [maxSeq] sampled ids, appended by `advance` dispatches (slot = step.n). */
  readonly history: Buf;
  /** u32 bitmask [ceil(vocab / 32)] of ids the repetition penalty applies to; set per call, updated by `advance`. */
  readonly seen: Buf;
  /** u32 [1 + ceil(vocab / 32)]: a flag, then the bitmask of ids sampling may choose (see `setAllowed`). */
  readonly allowed: Buf;
  /** The same per verify row, MAX_VERIFY_ROWS rows back to back (see `setAllowedRows`). */
  readonly allowedRows: Buf;
  /** Intermediates of every plan, shared: plans run one after another on the queue. */
  readonly activations: ActivationPool;
  private readonly kv: { k: Buf; v: Buf }[] = [];
  /**
   * Linear-attention layers' state, by layer (Qwen 3.5): `s` the recurrent state [Hv, Dv, Dk], `conv`
   * the convolution's last input rows [K − 1, C]. Both are what the tokens run so far left behind:
   * unlike keys and values they cannot be cut back to an earlier position, only saved and restored whole.
   */
  private readonly linearState = new Map<number, { s: Buf; conv: Buf }>();
  /** L2 normalisation of a key head as an RMSNorm weight (see `buildPlan`). */
  private readonly l2Weight: Tensor | undefined;
  /** Weights of ones by length (see `ones`). */
  private readonly unit = new Map<number, Tensor>();
  /** Cache slots per layer: maxSeq, or a ring for sliding layers (see `ringSlots`). */
  readonly slots: number[] = [];
  private readonly invFreq: Tensor;
  /** RoPE frequencies of sliding-window layers, when their base differs (Gemma 3); else `invFreq`. */
  private readonly localInvFreq: Tensor;
  /** q/k/v stacked into one `qkv` weight; qk-norm models normalise its q and k heads in `qkNormRopeKv`. */
  private readonly fusedQkv: boolean;
  /** gate/up stacked into one `gate_up` weight, consumed by the SiLU-gated matmul. */
  private readonly fusedMlp: boolean;
  /** Image features [imageTokens, hidden], when the model takes images (see `LlamaOptions.imageTokens`). */
  readonly imageFeatures: Buf | undefined;
  /** LoRA slots by "layer.slot" and their rank, once `enableLora` ran; plans built since add their updates. */
  private lora: { rank: number; slots: Map<string, LoraSlot> } | undefined;

  constructor(
    readonly backend: Backend,
    readonly cfg: LlamaConfig,
    private readonly weights: Map<string, Tensor>,
    opts: LlamaOptions = {},
  ) {
    this.kvDtype = opts.kvDtype ?? "f16";
    this.imageFeatures = opts.imageTokens ? backend.alloc(opts.imageTokens * cfg.hiddenSize * 4, "storage", "image/features") : undefined;
    this.maxSeq = opts.maxSeq ?? defaultMaxSeq(cfg, this.kvDtype);
    if (cfg.maxPositions !== undefined && this.maxSeq > cfg.maxPositions) {
      throw new Error(`maxSeq ${this.maxSeq} is past the ${cfg.maxPositions} tokens this model's rotary factors are loaded for`);
    }
    this.step = backend.alloc(STEP_BYTES, "storage", "step");
    if (cfg.kvFrom) this.tailStep = backend.alloc(STEP_BYTES, "storage", "step/tail");
    this.history = backend.alloc(this.maxSeq * 4, "storage", "history");
    this.seen = backend.alloc(Math.ceil(cfg.vocabSize / 32) * 4, "storage", "seen");
    this.allowed = backend.alloc((1 + Math.ceil(cfg.vocabSize / 32)) * 4, "storage", "allowed");
    this.allowedRows = backend.alloc(MAX_VERIFY_ROWS * (1 + Math.ceil(cfg.vocabSize / 32)) * 4, "storage", "allowed/rows");
    this.setAllowed(undefined);
    this.activations = new ActivationPool(backend);
    if (this.kvDtype === "q4" && [cfg.headDim, ...(cfg.layerHeadDims ?? [])].some((d) => d % 32 !== 0)) throw new Error("the q4 cache holds heads of a multiple of 32 values; use f16, f8 or i8 for this model");
    if (this.kvDtype === "q4" && [cfg.headDim, ...(cfg.layerHeadDims ?? [])].some((d) => d > 2 * ATTN_WG)) throw new Error(`the q4 cache does not hold heads wider than ${2 * ATTN_WG}; use f16 or f8 for this model`);
    if (this.kvDtype === "i8" && cfg.layerHeadDims?.some((d) => d > ATTN_WG)) throw new Error(`the i8 cache does not hold heads wider than ${ATTN_WG}; use f16 or f8 for this model`);
    for (let l = 0; l < cfg.numLayers; l++) {
      const slots = ringSlots(cfg, this.maxSeq, l);
      this.slots.push(slots);
      // A layer without a cache (linear attention) keeps one word, so every layer has its buffers.
      this.kv.push({
        k: backend.alloc(Math.max(4, slots * kvRowBytesOf(cfg, this.kvDtype, l, "k")), "storage", `kv/${l}/k`),
        v: backend.alloc(Math.max(4, slots * kvRowBytesOf(cfg, this.kvDtype, l, "v")), "storage", `kv/${l}/v`),
      });
      if (cfg.linearLayers?.[l]) {
        const g = cfg.linear!;
        const channels = 2 * g.keyHeads * g.keyDim + g.valueHeads * g.valueDim;
        this.linearState.set(l, {
          s: backend.alloc(g.valueHeads * g.valueDim * g.keyDim * 4, "storage", `linear/${l}/state`),
          conv: backend.alloc((g.convKernel - 1) * channels * 4, "storage", `linear/${l}/conv`),
        });
      }
    }
    if (cfg.linear) {
      // x · rsqrt(Σx² + 1e-6) is RMSNorm with eps 1e-6 / D and a weight of 1 / √D.
      const D = cfg.linear.keyDim;
      const w = new Float32Array(D).fill(1 / Math.sqrt(D));
      const buf = backend.alloc(w.byteLength, "storage", "linear/l2");
      backend.write(buf, 0, w);
      this.l2Weight = { shape: [D], dtype: "f32", buf };
      this.resetState();
    }
    const freqs = (f: Float32Array, label: string): Tensor => {
      const buf = backend.alloc(f.byteLength, "storage", label);
      backend.write(buf, 0, f);
      return { shape: [f.length], dtype: "f32", buf };
    };
    // With two head sizes (Gemma 4), the full-attention layers' is the one `ropeTheta` and its options describe.
    const fullLayer = cfg.slidingLayers?.indexOf(false) ?? 0;
    this.invFreq = freqs(ropeInvFreq({ ...cfg, headDim: headDimOf(cfg, Math.max(0, fullLayer)) }), "rope/inv_freq");
    this.localInvFreq =
      cfg.ropeLocalTheta !== undefined && (cfg.ropeLocalTheta !== cfg.ropeTheta || cfg.ropeLinear || cfg.layerHeadDims || cfg.ropeProportional !== undefined)
        ? freqs(ropeInvFreq({ headDim: cfg.headDim, ropeTheta: cfg.ropeLocalTheta }), "rope/inv_freq/local")
        : this.invFreq;
    this.w("model.embed_tokens.weight");
    const fuse = typeof opts.fuse === "object" ? { qkv: true, mlp: true, ...opts.fuse } : { qkv: opts.fuse ?? true, mlp: opts.fuse ?? true };
    // Phi-3 checkpoints store qkv already stacked.
    // Qwen 3.5's q_proj carries a gate beside each query, and not every layer has attention: its q/k/v stay apart.
    this.fusedQkv = weights.has("model.layers.0.self_attn.qkv.weight") || (fuse.qkv && !cfg.linearLayers && !cfg.valueNorm && !cfg.kvFrom && !cfg.layerHeadDims && this.fuse("self_attn", ["q_proj", "k_proj", "v_proj"], "qkv", cfg.attentionBias));
    // Granite 4 and Phi-3 checkpoints store gate_up already stacked.
    this.fusedMlp = weights.has("model.layers.0.mlp.gate_up.weight") || (fuse.mlp && this.fuse("mlp", ["gate_proj", "up_proj"], "gate_up", false));
  }

  /**
   * Stacks per-layer weights along N (rows) into one tensor per layer so they run as one matmul,
   * freeing the originals. Skipped (returns false) unless every part shares dtype, K and affine-ness
   * (a GGUF Q4_K_M file mixes Q4_K q and k with a Q6_K v, so its q/k/v stay separate).
   */
  private fuse(module: string, parts: string[], fused: string, withBias: boolean): boolean {
    const names = (l: number, kind: string) => parts.map((p) => `model.layers.${l}.${module}.${p}.${kind}`);
    for (let l = 0; l < this.cfg.numLayers; l++) {
      const ts = names(l, "weight").map((n) => this.weights.get(n));
      if (ts.some((t) => !t || t.dtype !== ts[0]!.dtype || t.shape[1] !== ts[0]!.shape[1] || !t.mins !== !ts[0]!.mins)) return false;
      if (withBias && names(l, "bias").some((n) => !this.weights.has(n))) return false;
    }
    for (let l = 0; l < this.cfg.numLayers; l++) {
      this.stack(names(l, "weight"), `model.layers.${l}.${module}.${fused}.weight`);
      if (withBias) this.stack(names(l, "bias"), `model.layers.${l}.${module}.${fused}.bias`);
    }
    return true;
  }

  private stack(parts: string[], name: string): void {
    const ts = parts.map((n) => this.w(n));
    const { dtype } = ts[0]!;
    const shape = [ts.reduce((n, t) => n + t.shape[0]!, 0), ...ts[0]!.shape.slice(1)];
    const copyPlane = (bytes: (t: Tensor) => number, plane: (t: Tensor) => Buf, label: string) => {
      const dst = this.backend.alloc(ts.reduce((n, t) => n + bytes(t), 0), "storage", label);
      let off = 0;
      for (const t of ts) {
        this.backend.copy(plane(t), 0, dst, off, bytes(t));
        off += bytes(t);
      }
      return dst;
    };
    const buf = copyPlane((t) => byteSize(t.shape, dtype), (t) => t.buf, name);
    const scales = isBlockQuant(dtype) ? copyPlane((t) => q4ScaleBytes(t.shape), (t) => t.scales!, `${name}/scales`) : undefined;
    const mins = ts[0]!.mins ? copyPlane((t) => q4ScaleBytes(t.shape), (t) => t.mins!, `${name}/mins`) : undefined;
    for (const [i, t] of ts.entries()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
      this.weights.delete(parts[i]!);
    }
    this.weights.set(name, { shape, dtype, buf, scales, mins });
  }

  /** GPU bytes held: weights, the KV cache, and the intermediates shared by every plan built so far. */
  get memory(): { weights: number; kv: number; activations: number } {
    let weights = 0;
    for (const t of this.weights.values()) weights += t.buf.size + (t.scales?.size ?? 0) + (t.mins?.size ?? 0);
    const kv = this.kv.reduce((n, { k, v }) => n + k.size + v.size, 0);
    return { weights, kv, activations: this.activations.bytes };
  }

  /** Bytes of one position's K in the first layer's cache: with `slots`, what tells two cache layouts apart. */
  get kvRowBytes(): number {
    return kvRowBytesOf(this.cfg, this.kvDtype);
  }

  /** Bytes of one position's V in the first layer's cache: `kvRowBytes` but for a q4 cache. */
  get kvValueRowBytes(): number {
    return kvRowBytesOf(this.cfg, this.kvDtype, 0, "v");
  }

  /** Bytes of one position's K (or with part "v", V) in layer l's cache (layers may differ in head size). */
  private rowBytes(l: number, part: "k" | "v" = "k"): number {
    return kvRowBytesOf(this.cfg, this.kvDtype, l, part);
  }

  /** Bytes of one position's K and V together in layer l's cache. */
  private pairBytes(l: number): number {
    return this.rowBytes(l, "k") + this.rowBytes(l, "v");
  }

  /** Bytes `readKv(n)` returns: each layer's first min(n, slots) K rows and as many V rows. */
  kvSnapshotBytes(n: number): number {
    return this.slots.reduce((t, slots, l) => t + Math.min(n, slots) * this.pairBytes(l), 0);
  }

  /**
   * Copies the K/V of positions 0…n-1 out of every layer's cache: layer 0's K rows, then its V rows,
   * then layer 1's, and so on. A ring layer gives its first min(n, slots) slots, which hold the last
   * positions at slot = position % slots, the layout `writeKv` restores. The copies are queued at
   * once, so they read the cache as it stands when this is called, even if more work is submitted
   * before they resolve.
   */
  async readKv(n: number): Promise<Uint8Array> {
    const parts = this.kv.flatMap(({ k, v }, l) => {
      const rows = Math.min(n, this.slots[l]!);
      // A layer without a cache (linear attention) has no rows.
      return rows ? [this.backend.readback(k, 0, rows * this.rowBytes(l, "k")), this.backend.readback(v, 0, rows * this.rowBytes(l, "v"))] : [];
    });
    const out = new Uint8Array(this.kvSnapshotBytes(n));
    let at = 0;
    for (const p of await Promise.all(parts)) {
      out.set(new Uint8Array(p), at);
      at += p.byteLength;
    }
    return out;
  }

  /** Positions of from…to-1 that layer l still holds once `to` are written: all, or a ring's last `slots`. */
  private kvRangeStart(l: number, from: number, to: number): number {
    return Math.max(from, to - this.slots[l]!);
  }

  /** Bytes `readKvRange(from, to)` returns in all. */
  kvRangeBytes(from: number, to: number): number {
    return this.slots.reduce((t, slots, l) => t + (slots ? (to - this.kvRangeStart(l, from, to)) * this.pairBytes(l) : 0), 0);
  }

  /**
   * Copies the K/V of positions from…to-1 out of every layer's cache, as parts in order: layer 0's K
   * rows by position, then its V rows, then layer 1's. A ring layer gives only the positions it
   * still holds (its last `slots` up to `to`). Like `readKv`, the copies are queued at once. What a
   * prompt cache appends after a call: the rows of the new tokens, not the whole cache again.
   */
  async readKvRange(from: number, to: number): Promise<Uint8Array[]> {
    if (!(from >= 0 && from <= to && to <= this.maxSeq)) throw new Error(`readKvRange: ${from}…${to} is not inside the cache`);
    const parts = this.kv.flatMap(({ k, v }, l) => {
      const slots = this.slots[l]!;
      // A layer without a cache (linear attention) has no rows.
      if (!slots) return [];
      const start = this.kvRangeStart(l, from, to);
      const count = to - start;
      // Position p sits in slot p % slots, so a range may run off the ring's end and go on at slot 0.
      const first = start % slots;
      const head = Math.min(count, slots - first);
      const runs = (buf: typeof k, row: number) => [this.backend.readback(buf, first * row, head * row), ...(count > head ? [this.backend.readback(buf, 0, (count - head) * row)] : [])];
      return [...runs(k, this.rowBytes(l, "k")), ...runs(v, this.rowBytes(l, "v"))];
    });
    return (await Promise.all(parts)).map((p) => new Uint8Array(p));
  }

  /**
   * GPU copies of the K/V of positions 0…n-1 (per layer: K, then V, as `readKv` lays them out),
   * queued now, so they hold the cache as it stands. `restoreKv` puts them back.
   */
  /** True when some layers carry a state that cannot be cut back to an earlier position (see `saveState`). */
  get recurrent(): boolean {
    return this.linearState.size > 0;
  }

  /** Bytes one `saveState` snapshot takes. */
  get stateBytes(): number {
    let n = 0;
    for (const { s, conv } of this.linearState.values()) n += s.size + conv.size;
    return n;
  }

  /** Empties the linear-attention state: the state before the first token. */
  resetState(): void {
    for (const { s, conv } of this.linearState.values()) {
      this.backend.write(s, 0, new Uint8Array(s.size));
      this.backend.write(conv, 0, new Uint8Array(conv.size));
    }
  }

  /**
   * GPU copies of the linear-attention state, queued now, so they hold it as the work submitted so
   * far leaves it. `restoreState` puts a snapshot back; the caller frees the buffers.
   */
  saveState(into?: readonly Buf[]): Buf[] {
    let i = 0;
    return [...this.linearState.values()].flatMap(({ s, conv }) =>
      [s, conv].map((src) => {
        // `into`: an earlier snapshot's buffers, overwritten (a snapshot taken again and again, as before every draft).
        const dst = into?.[i++] ?? this.backend.alloc(src.size, "storage", "linear/saved");
        this.backend.copy(src, 0, dst, 0, src.size);
        return dst;
      }),
    );
  }

  /** The bytes of a `saveState` snapshot, buffer after buffer (`stateBytes` of them), for a file. */
  async readState(saved: readonly Buf[]): Promise<Uint8Array> {
    const parts = await Promise.all(saved.map((b) => this.backend.readback(b, 0, b.size)));
    const out = new Uint8Array(this.stateBytes);
    let at = 0;
    for (const p of parts) {
      out.set(new Uint8Array(p), at);
      at += p.byteLength;
    }
    return out;
  }

  /** Makes `readState`'s bytes the live state. */
  writeState(data: Uint8Array): void {
    if (data.length !== this.stateBytes) throw new Error(`writeState: ${data.length} bytes are not this model's state (${this.stateBytes})`);
    let at = 0;
    for (const { s, conv } of this.linearState.values())
      for (const b of [s, conv]) {
        this.backend.write(b, 0, data.subarray(at, at + b.size));
        at += b.size;
      }
  }

  restoreState(saved: readonly Buf[]): void {
    let i = 0;
    for (const { s, conv } of this.linearState.values()) {
      this.backend.copy(saved[i++]!, 0, s, 0, s.size);
      this.backend.copy(saved[i++]!, 0, conv, 0, conv.size);
    }
  }

  saveKv(n: number): Buf[] {
    return this.kv.flatMap(({ k, v }, l) => {
      const rows = Math.min(n, this.slots[l]!);
      return [k, v].map((src, i) => {
        const bytes = rows * this.rowBytes(l, i ? "v" : "k");
        // A layer without a cache (linear attention) has nothing to copy, but keeps its place in the list.
        const dst = this.backend.alloc(Math.max(4, bytes), "storage", `kv/saved/${l}/${i ? "v" : "k"}`);
        if (bytes) this.backend.copy(src, 0, dst, 0, bytes);
        return dst;
      });
    });
  }

  /** Copies `saveKv(n)`'s buffers back into the cache. */
  restoreKv(n: number, saved: readonly Buf[]): void {
    for (const [l, { k, v }] of this.kv.entries()) {
      const rows = Math.min(n, this.slots[l]!);
      if (!rows) continue;
      this.backend.copy(saved[2 * l]!, 0, k, 0, rows * this.rowBytes(l, "k"));
      this.backend.copy(saved[2 * l + 1]!, 0, v, 0, rows * this.rowBytes(l, "v"));
    }
  }

  /** Writes K/V for positions 0…n-1 back into every layer's cache, in `readKv`'s layout. */
  writeKv(n: number, data: Uint8Array): void {
    if (n > this.maxSeq || data.length !== this.kvSnapshotBytes(n)) throw new Error(`writeKv: ${data.length} bytes do not hold ${n} positions`);
    let at = 0;
    for (const [l, { k, v }] of this.kv.entries()) {
      const rows = Math.min(n, this.slots[l]!);
      if (!rows) continue;
      const [kb, vb] = [rows * this.rowBytes(l, "k"), rows * this.rowBytes(l, "v")];
      this.backend.write(k, 0, data.subarray(at, at + kb));
      this.backend.write(v, 0, data.subarray(at + kb, at + kb + vb));
      at += kb + vb;
    }
  }

  /**
   * Earliest position a call may resume prefill from when the cache was last written up to `written`
   * positions: a ring has overwritten the keys before written - slots, and the first position
   * prefilled needs the window before it. 0 when no layer is a ring or nothing was overwritten.
   */
  minReuseStart(written: number): number {
    let min = 0;
    for (let l = 0; l < this.slots.length; l++) {
      const slots = this.slots[l]!;
      if (slots === 0 || slots >= this.maxSeq || written <= slots) continue;
      min = Math.max(min, written - slots + (this.cfg.slidingWindow ?? 0) - 1);
    }
    return min;
  }

  /** Marks the ids the repetition penalty starts from (the prompt); generated ids are added on the GPU. */
  setSeen(ids: readonly number[]): void {
    const mask = new Uint32Array(Math.ceil(this.cfg.vocabSize / 32));
    for (const id of ids) if (id < this.cfg.vocabSize) mask[id >>> 5]! |= 1 << (id & 31);
    this.backend.write(this.seen, 0, mask);
  }

  /** Restricts sampling to the ids set in `mask` (ceil(vocab / 32) u32 words), or lifts the restriction. */
  setAllowed(mask: Uint32Array | undefined): void {
    if (!mask) {
      this.backend.write(this.allowed, 0, new Uint32Array(1));
      return;
    }
    const words = Math.ceil(this.cfg.vocabSize / 32);
    if (mask.length !== words) throw new Error(`allowed mask has ${mask.length} words, expected ${words}`);
    const data = new Uint32Array(1 + words);
    data[0] = 1;
    data.set(mask, 1);
    this.backend.write(this.allowed, 0, data);
  }

  /** Per-row `setAllowed` for a verify plan: row r samples under masks[r], or freely where it is undefined. */
  setAllowedRows(masks: readonly (Uint32Array | undefined)[]): void {
    const words = Math.ceil(this.cfg.vocabSize / 32);
    if (masks.length > MAX_VERIFY_ROWS) throw new Error(`setAllowedRows: ${masks.length} rows, at most ${MAX_VERIFY_ROWS}`);
    for (const [r, mask] of masks.entries()) {
      const at = r * (1 + words) * 4;
      if (!mask) {
        this.backend.write(this.allowedRows, at, new Uint32Array(1));
        continue;
      }
      if (mask.length !== words) throw new Error(`allowed mask has ${mask.length} words, expected ${words}`);
      const data = new Uint32Array(1 + words);
      data[0] = 1;
      data.set(mask, 1);
      this.backend.write(this.allowedRows, at, data);
    }
  }

  /** The LoRA slot rank (0 when LoRA is off). */
  get loraRank(): number {
    return this.lora?.rank ?? 0;
  }

  /** The slot rank adapters up to `rank` need: a multiple of 32, as the update matmuls' K must be. */
  static loraSlotRank(rank: number): number {
    return Math.ceil(rank / 32) * 32;
  }

  /** The LoRA slots this model's graph has, in a fixed order: per layer, each adapted matmul and the projections stacked in it. */
  private loraLayout(): { name: string; parts: { module: LoraModule; n: number }[]; k: number }[] {
    const c = this.cfg;
    const out: { name: string; parts: { module: LoraModule; n: number }[]; k: number }[] = [];
    const add = (name: string, parts: LoraModule[], k: number, ns: number[]) => out.push({ name, parts: parts.map((m, i) => ({ module: m, n: ns[i]! })), k });
    const H = c.hiddenSize;
    const q = c.numHeads * c.headDim;
    const kv = c.numKvHeads * c.headDim;
    for (let l = 0; l < c.numLayers; l++) {
      const p = `model.layers.${l}`;
      const I = this.fusedMlp ? this.w(`${p}.mlp.gate_up.weight`).shape[0]! / 2 : this.w(`${p}.mlp.gate_proj.weight`).shape[0]!;
      if (c.linearLayers?.[l]) {
        // Linear attention's five projections (Qwen 3.5), in the order the layer runs them.
        const g = c.linear!;
        const values = g.valueHeads * g.valueDim;
        add(`${l}.in_proj_qkv`, ["in_proj_qkv"], H, [2 * g.keyHeads * g.keyDim + values]);
        add(`${l}.in_proj_z`, ["in_proj_z"], H, [values]);
        add(`${l}.in_proj_a`, ["in_proj_a"], H, [g.valueHeads]);
        add(`${l}.in_proj_b`, ["in_proj_b"], H, [g.valueHeads]);
        add(`${l}.out_proj`, ["out_proj"], values, [H]);
      } else {
        // With an output gate, q_proj also writes the gates: twice the rows.
        const qRows = c.attnOutputGate ? 2 * q : q;
        if (this.fusedQkv) add(`${l}.qkv`, ["q_proj", "k_proj", "v_proj"], H, [qRows, kv, kv]);
        else for (const [m, n] of [["q_proj", qRows], ["k_proj", kv], ["v_proj", kv]] as const) add(`${l}.${m}`, [m], H, [n]);
        add(`${l}.o_proj`, ["o_proj"], q, [H]);
      }
      if (this.fusedMlp) add(`${l}.gate_up`, ["gate_proj", "up_proj"], H, [I, I]);
      else for (const m of ["gate_proj", "up_proj"] as const) add(`${l}.${m}`, [m], H, [I]);
      add(`${l}.down_proj`, ["down_proj"], I, [H]);
    }
    return out;
  }

  /** One slot's A and B contents for an adapter (zeros for projections it leaves out). */
  private loraArrays(slot: { name: string; parts: { module: LoraModule; n: number }[]; k: number }, R: number, weights: LoraWeights | undefined): { A: Float32Array; B: Float32Array } {
    const layer = slot.name.slice(0, slot.name.indexOf("."));
    const P = slot.parts.length;
    const N = slot.parts.reduce((a, b) => a + b.n, 0);
    const A = new Float32Array(P * R * slot.k);
    const B = new Float32Array(N * P * R);
    let row = 0;
    slot.parts.forEach((part, i) => {
      const u = weights?.get(`${layer}.${part.module}`);
      if (u) {
        const r = u.rank;
        if (r > R) throw new Error(`LoRA rank ${r} is above the slot rank ${R}`);
        if (u.A.length !== r * slot.k || u.B.length !== part.n * r) throw new Error(`LoRA ${layer}.${part.module}: expected A [${r}, ${slot.k}] and B [${part.n}, ${r}]`);
        A.set(u.A, i * R * slot.k);
        // Granite's residual multiplier is folded into o_proj and down_proj at load; their updates scale with it.
        const scale = this.cfg.residualScale !== undefined && (part.module === "o_proj" || part.module === "down_proj") ? this.cfg.residualScale : 1;
        for (let o = 0; o < part.n; o++) for (let j = 0; j < r; j++) B[(row + o) * P * R + i * R + j] = u.B[o * r + j]! * scale;
      }
      row += part.n;
    });
    return { A, B };
  }

  /**
   * Allocates LoRA slots for adapters up to `rank` (see `loraSlotRank`) on every projection, all
   * zero; or with rank 0 frees them. Plans built before do not see the change, so the caller
   * rebuilds them.
   */
  enableLora(rank: number): void {
    this.freeLora();
    if (rank <= 0) return;
    if (this.cfg.kvFrom) throw new Error("LoRA adapters are not supported on Gemma 4 yet");
    const R = LlamaModel.loraSlotRank(rank);
    const slots = new Map<string, LoraSlot>();
    for (const s of this.loraLayout()) {
      const { A, B } = this.loraArrays(s, R, undefined);
      const tensor = (data: Float32Array, shape: number[], label: string): Tensor => {
        const buf = this.backend.alloc(data.byteLength, "storage", label);
        this.backend.write(buf, 0, data);
        return { shape, dtype: "f32", buf };
      };
      const N = s.parts.reduce((a, b) => a + b.n, 0);
      slots.set(s.name, { parts: s.parts, k: s.k, A: tensor(A, [s.parts.length * R, s.k], `lora/${s.name}/A`), B: tensor(B, [N, s.parts.length * R], `lora/${s.name}/B`) });
    }
    this.lora = { rank: R, slots };
  }

  /**
   * Writes an adapter's updates into the LoRA slots (projections it leaves out get zeros; undefined
   * clears every slot). Takes effect on the next forward pass, with no plan rebuilt.
   */
  setLora(weights: LoraWeights | undefined): void {
    const L = this.lora;
    if (!L) throw new Error("LoRA is not enabled on this model (enableLora first)");
    for (const s of this.loraLayout()) {
      const slot = L.slots.get(s.name)!;
      const { A, B } = this.loraArrays(s, L.rank, weights);
      this.backend.write(slot.A.buf, 0, A);
      this.backend.write(slot.B.buf, 0, B);
    }
  }

  /**
   * An adapter laid out for slots of `rank` in GPU buffers of its own, so `useLora` can switch to
   * it with GPU copies alone. Free it with `freeLoraBuffers`.
   */
  prepareLora(weights: LoraWeights, rank: number): LoraBuffers {
    const R = LlamaModel.loraSlotRank(rank);
    const bufs = this.loraLayout().map((s) => {
      const { A, B } = this.loraArrays(s, R, weights);
      const a = this.backend.alloc(A.byteLength, "storage", `lora/${s.name}/A/prepared`);
      const b = this.backend.alloc(B.byteLength, "storage", `lora/${s.name}/B/prepared`);
      this.backend.write(a, 0, A);
      this.backend.write(b, 0, B);
      return { A: a, B: b, aBytes: A.byteLength, bBytes: B.byteLength };
    });
    return { rank: R, bufs, bytes: bufs.reduce((n, b) => n + b.aBytes + b.bBytes, 0) };
  }

  /** Copies a prepared adapter into the LoRA slots (GPU to GPU), for the next forward pass. */
  useLora(p: LoraBuffers): void {
    const L = this.lora;
    if (!L || L.rank !== p.rank) throw new Error(`useLora: the slots have rank ${L?.rank ?? 0}, the adapter was prepared for ${p.rank}`);
    this.loraLayout().forEach((s, i) => {
      const slot = L.slots.get(s.name)!;
      const b = p.bufs[i]!;
      this.backend.copy(b.A, 0, slot.A.buf, 0, b.aBytes);
      this.backend.copy(b.B, 0, slot.B.buf, 0, b.bBytes);
    });
  }

  freeLoraBuffers(p: LoraBuffers): void {
    for (const b of p.bufs) {
      this.backend.free(b.A);
      this.backend.free(b.B);
    }
  }

  private freeLora(): void {
    for (const s of this.lora?.slots.values() ?? []) {
      this.backend.free(s.A.buf);
      this.backend.free(s.B.buf);
    }
    this.lora = undefined;
  }

  setStep(s: StepParams): void {
    this.backend.write(this.step, 0, packStep(s));
    // The same step as the one-row layers see it: their only row is row `lastRow`.
    if (this.tailStep) this.backend.write(this.tailStep, 0, packStep({ ...s, pos0: s.pos0 + s.lastRow, lastRow: 0 }));
  }

  /**
   * Records the forward pass for T tokens at positions step.pos0..pos0+T, then sampling. A `verify`
   * plan samples every row (row t as the token after input t) instead of only `step.lastRow`, for
   * checking draft tokens; it needs step.lastRow = 0.
   */
  buildPlan(T: number, verify = false): ForwardPlan {
    if (verify && T > MAX_VERIFY_ROWS) throw new Error(`verify plan of ${T} rows; at most ${MAX_VERIFY_ROWS}`);
    const c = this.cfg;
    const ctx = new Ctx(this.backend, this.step, `${verify ? "V" : "T"}${T}`, this.activations);
    const tokens = ctx.buffer(T * 4, "tokens");
    const out = ctx.buffer((verify ? T : 1) * 4, "out");
    const gelu = c.activation === "gelu";
    const norm = (t: Tensor, name: string) => ops.rmsnorm(ctx, t, this.w(name), c.rmsEps);

    const L = this.lora?.slots;
    /** Adds slot `name`'s LoRA update of input `xin` to `y`: y += B · (A · xin). */
    const delta = (xin: Tensor, y: Tensor, name: string) => {
      const s = L!.get(name)!;
      ops.linear(ctx, ops.linear(ctx, xin, s.A), s.B, { residual: y });
    };
    const features = this.imageFeatures && { buf: this.imageFeatures, vocab: c.vocabSize };
    const x0 = ops.embed(ctx, this.w("model.embed_tokens.weight"), tokens, T, c.embedScale ?? 1, features);
    // Gemma 4's per-layer input, the part every layer shares: a projection of the token embeddings
    // to layers × P values, normalised per layer. The model scales the projection by 1 / √hidden
    // first, which RMSNorm undoes but for eps: the same as eps · hidden without the scaling.
    // An image token's features go through the projection like any embedding; in each layer's
    // table it counts as the padding token.
    const P = c.perLayerInput;
    const asPad = features && { vocab: c.vocabSize };
    const perLayer = P ? ops.rmsnorm(ctx, ops.linear(ctx, x0, this.w("model.per_layer_model_projection.weight")), this.w("model.per_layer_projection_norm.weight"), c.rmsEps * c.hiddenSize, P) : undefined;
    const firstShared = c.kvFrom ? c.kvFrom.findIndex((from) => from !== undefined) : -1;
    let kvEnd: number | undefined;
    // Past the last layer that writes a cache (Gemma 4) only the sampled row matters: the layers
    // from there on run on that row alone, at its position, which the tail step carries.
    let x = x0;
    let rows = T;
    let step = this.step;
    for (let l = 0; l < c.numLayers; l++) {
      const p = `model.layers.${l}`;
      if (l === firstShared) {
        kvEnd = ctx.size;
        if (!verify && T > 1) {
          x = ops.copyRow(ctx, view(x, [T, c.hiddenSize]));
          rows = 1;
          step = this.tailStep!;
        }
      }
      const D = headDimOf(c, l);
      // A layer without keys and values of its own (Gemma 4) reads the cache of an earlier one of its kind.
      const src = c.kvFrom?.[l] ?? l;
      const { k: kc, v: vc } = this.kv[src]!;
      const sliding = !!c.slidingLayers?.[l];
      const invFreq = sliding ? this.localInvFreq : this.invFreq;
      const attnShape = {
        T: rows,
        step,
        H: c.numHeads,
        Hkv: c.numKvHeads,
        D,
        maxSeq: this.maxSeq,
        kv: this.kvDtype,
        scale: c.attnScale,
        window: sliding ? c.slidingWindow : undefined,
        slots: this.slots[src],
      };
      const inputNorm = { w: this.w(`${p}.input_layernorm.weight`), eps: c.rmsEps };
      if (c.linearLayers?.[l]) this.linearAttention(ctx, x, l, rows, L ? delta : undefined);
      else {
      let q: Tensor;
      let gate: Tensor | undefined;
      if (this.fusedQkv) {
        // RMSNorm → one qkv matmul (+bias) → RoPE fused with the cache write.
        let qkv: Tensor;
        if (L) {
          // The update reads the normalised input, so the norm runs on its own.
          const xn = ops.rmsnorm(ctx, x, inputNorm.w, inputNorm.eps);
          qkv = ops.linear(ctx, xn, this.w(`${p}.self_attn.qkv.weight`), { bias: this.bias(`${p}.self_attn.qkv.bias`) });
          delta(xn, qkv, `${l}.qkv`);
        } else qkv = ops.linear(ctx, x, this.w(`${p}.self_attn.qkv.weight`), { norm: inputNorm, bias: this.bias(`${p}.self_attn.qkv.bias`) });
        q = c.qkNorm
          ? ops.qkNormRopeKv(ctx, qkv, this.w(`${p}.self_attn.q_norm.weight`), this.w(`${p}.self_attn.k_norm.weight`), c.rmsEps, kc, vc, attnShape, invFreq, c.ropeInterleaved)
          : ops.ropeKv(ctx, qkv, kc, vc, attnShape, invFreq, c.ropeInterleaved);
      } else {
        ({ q, gate } = this.attentionInputs(ctx, x, l, rows, invFreq, L ? delta : undefined, step));
      }
      const a = ops.attention(ctx, q, kc, vc, attnShape);
      if (gate) ops.sigmoidMul(ctx, a, gate);
      if (c.sandwichNorms) {
        // Gemma: the attention output is normalised before the residual add.
        const o = ops.linear(ctx, a, this.w(`${p}.self_attn.o_proj.weight`));
        if (L) delta(a, o, `${l}.o_proj`);
        ops.addInPlace(ctx, x, norm(o, `${p}.post_attention_layernorm.weight`));
      } else {
        ops.linear(ctx, a, this.w(`${p}.self_attn.o_proj.weight`), { residual: x });
        if (L) delta(a, x, `${l}.o_proj`);
      }
      }

      const mlpNorm = { w: this.w(`${p}.${c.sandwichNorms ? "pre_feedforward_layernorm" : "post_attention_layernorm"}.weight`), eps: c.rmsEps };
      let m: Tensor;
      if (L) {
        // Updates go in before the gated activation: gate and up as plain matmuls on the normalised input.
        const xn = ops.rmsnorm(ctx, x, mlpNorm.w, mlpNorm.eps);
        if (this.fusedMlp) {
          const gu = ops.linear(ctx, xn, this.w(`${p}.mlp.gate_up.weight`));
          delta(xn, gu, `${l}.gate_up`);
          m = ops.gatedAct(ctx, gu, gelu ? "gelu" : "silu");
        } else {
          const h2 = ops.linear(ctx, xn, this.w(`${p}.mlp.gate_proj.weight`));
          delta(xn, h2, `${l}.gate_proj`);
          const u = ops.linear(ctx, xn, this.w(`${p}.mlp.up_proj.weight`));
          delta(xn, u, `${l}.up_proj`);
          m = ops.siluMul(ctx, h2, u, gelu ? "gelu" : "silu");
        }
      } else if (this.fusedMlp) {
        m = ops.linear(ctx, x, this.w(`${p}.mlp.gate_up.weight`), { norm: mlpNorm, silu: !gelu, gelu });
      } else {
        const h2 = ops.linear(ctx, x, this.w(`${p}.mlp.gate_proj.weight`), { norm: mlpNorm });
        const u = ops.linear(ctx, x, this.w(`${p}.mlp.up_proj.weight`), { norm: mlpNorm });
        m = ops.siluMul(ctx, h2, u, gelu ? "gelu" : "silu");
      }
      if (c.sandwichNorms) {
        const d = ops.linear(ctx, m, this.w(`${p}.mlp.down_proj.weight`));
        if (L) delta(m, d, `${l}.down_proj`);
        ops.addInPlace(ctx, x, norm(d, `${p}.post_feedforward_layernorm.weight`));
      } else {
        ops.linear(ctx, m, this.w(`${p}.mlp.down_proj.weight`), { residual: x });
        if (L) delta(m, x, `${l}.down_proj`);
      }
      if (P && perLayer) {
        // This layer's input per token: its row of the second embedding table (times √P) plus its
        // slice of the shared projection. The layer's output gates it, a projection brings it back
        // to the model's width, and its norm goes onto the residual. The model halves the sum's
        // square (× 1 / √2) before the gate; the norm after undoes that but for eps, hence 2 · eps.
        const all = ops.embed(ctx, this.w(gemma4PerLayerTable(l)), tokens, T, Math.sqrt(P), asPad || undefined);
        ops.cols(ctx, all, perLayer, { n: P, srcStart: l * P, accumulate: true });
        const e = rows === T ? all : ops.copyRow(ctx, all);
        const g = ops.linear(ctx, x, this.w(`${p}.per_layer_input_gate.weight`));
        const back = ops.linear(ctx, ops.siluMul(ctx, g, e, "gelu"), this.w(`${p}.per_layer_projection.weight`));
        ops.addInPlace(ctx, x, ops.rmsnorm(ctx, back, this.w(`${p}.post_per_layer_input_norm.weight`), 2 * c.rmsEps));
      }
      const scalar = c.layerScalars?.[l];
      if (scalar !== undefined && scalar !== 1) ops.scaleInPlace(ctx, x, scalar);
    }
    const last = verify || rows < T ? view(x, [rows, c.hiddenSize]) : ops.copyRow(ctx, view(x, [T, c.hiddenSize]));
    const head = this.weights.get("lm_head.weight") ?? this.w("model.embed_tokens.weight");
    const logits = ops.linear(ctx, last, head, { norm: { w: this.w("model.norm.weight"), eps: c.rmsEps } });
    if (c.logitSoftcap) ops.softcap(ctx, logits, c.logitSoftcap);
    if (verify) ops.sampleRows(ctx, logits, out, this.seen, this.allowedRows);
    else ops.sample(ctx, logits, out, this.seen, this.allowed);

    const dispatches = ctx.dispatches;
    // Non-verify plans end (copy_row →) head matmul (→ soft cap) → sample; a plan whose last layers
    // ran on one row has no copy there.
    const copied = rows === T;
    const body = verify ? dispatches : dispatches.slice(0, dispatches.length - 2 - (copied ? 1 : 0) - (c.logitSoftcap ? 1 : 0));
    if (!verify && copied && dispatches[body.length]!.label !== "copy_row") throw new Error("buildPlan: unexpected head layout");
    return { T, dispatches, body, ...(kvEnd !== undefined && !verify ? { kvBody: dispatches.slice(0, kvEnd) } : {}), tokens, out, logits, destroy: () => ctx.destroy() };
  }

  /**
   * Records log-probability scoring of a verify plan's logits: `targets` (u32 [T], the id expected
   * after each row) in, `out` (f32 [T]) the log-probability of each.
   */
  buildScore(plan: ForwardPlan): { targets: Buf; out: Buf; dispatches: readonly Dispatch[]; destroy(): void } {
    const ctx = new Ctx(this.backend, this.step, `S${plan.T}`);
    const targets = ctx.buffer(plan.T * 4, "targets");
    const out = ctx.buffer(plan.T * 4, "logprobs");
    // The logits live in the verify plan's pooled memory: bind its physical buffer, read right after it runs.
    ops.logprobRows(ctx, { ...plan.logits, buf: physical(plan.logits.buf) }, targets, out);
    return { targets, out, dispatches: ctx.dispatches, destroy: () => ctx.destroy() };
  }

  /**
   * Records the step advance that chains decode steps on the GPU: copies the id sampled into `sampled`
   * to `tokens` and the history, and moves the position past the rows just processed.
   */
  buildAdvance(sampled: Buf, tokens: Buf): { dispatches: readonly Dispatch[]; destroy(): void } {
    const ctx = new Ctx(this.backend, this.step, "advance");
    ops.advance(ctx, sampled, tokens, this.history, this.seen);
    return { dispatches: ctx.dispatches, destroy: () => ctx.destroy() };
  }

  destroy(): void {
    for (const { k, v } of this.kv) {
      this.backend.free(k);
      this.backend.free(v);
    }
    for (const t of this.unit.values()) this.backend.free(t.buf);
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    if (this.imageFeatures) this.backend.free(this.imageFeatures);
    this.freeLora();
    this.backend.free(this.invFreq.buf);
    if (this.localInvFreq !== this.invFreq) this.backend.free(this.localInvFreq.buf);
    for (const { s, conv } of this.linearState.values()) {
      this.backend.free(s);
      this.backend.free(conv);
    }
    if (this.l2Weight) this.backend.free(this.l2Weight.buf);
    this.backend.free(this.step);
    if (this.tailStep) this.backend.free(this.tailStep);
    this.backend.free(this.history);
    this.backend.free(this.seen);
    this.backend.free(this.allowed);
    this.backend.free(this.allowedRows);
    this.activations.destroy();
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new MissingWeightError(`weight "${name}" was not loaded`);
    return t;
  }

  /** Unfused attention inputs (qk-norm models): separate q/k/v, per-head norms, RoPE, cache write. */
  /**
   * A linear-attention layer (Qwen 3.5's gated delta net), added to x in place: projections of the
   * normalised input, a causal convolution over time, L2-normalised queries and keys, the recurrence
   * over the layer's state, a gated norm and the output projection.
   */
  private linearAttention(ctx: Ctx, x: Tensor, l: number, T: number, delta?: (xin: Tensor, y: Tensor, name: string) => void): void {
    const c = this.cfg;
    const g = c.linear!;
    const p = `model.layers.${l}.linear_attn`;
    const state = this.linearState.get(l)!;
    const keys = 2 * g.keyHeads * g.keyDim;
    const h = ops.rmsnorm(ctx, x, this.w(`model.layers.${l}.input_layernorm.weight`), c.rmsEps);
    const mixed = ops.linear(ctx, h, this.w(`${p}.in_proj_qkv.weight`));
    const z = ops.linear(ctx, h, this.w(`${p}.in_proj_z.weight`));
    const a = ops.linear(ctx, h, this.w(`${p}.in_proj_a.weight`));
    const b = ops.linear(ctx, h, this.w(`${p}.in_proj_b.weight`));
    if (delta) {
      delta(h, mixed, `${l}.in_proj_qkv`);
      delta(h, z, `${l}.in_proj_z`);
      delta(h, a, `${l}.in_proj_a`);
      delta(h, b, `${l}.in_proj_b`);
    }
    const conv = ops.causalConv(ctx, mixed, this.w(`${p}.conv1d.weight`), state.conv);
    // Queries and keys are the row's first columns: each head normalised to unit length.
    const qk = ctx.alloc([T, keys]);
    ops.cols(ctx, qk, conv, { n: keys });
    const unit = ops.rmsnorm(ctx, qk, this.l2Weight!, 1e-6 / g.keyDim, g.keyDim);
    const out = ops.gatedDelta(ctx, unit, conv, a, b, this.w(`${p}.A_log`), this.w(`${p}.dt_bias`), state.s, { T, ...g });
    // RMSNorm per value head, gated by silu(z).
    const normed = ops.rmsnorm(ctx, out, this.w(`${p}.norm.weight`), c.rmsEps, g.valueDim);
    const gated = ops.siluMul(ctx, z, normed);
    ops.linear(ctx, gated, this.w(`${p}.out_proj.weight`), { residual: x });
    if (delta) delta(gated, x, `${l}.out_proj`);
  }

  private attentionInputs(ctx: Ctx, x: Tensor, l: number, T: number, invFreq: Tensor, delta?: (xin: Tensor, y: Tensor, name: string) => void, step: Buf = this.step): { q: Tensor; gate?: Tensor } {
    const c = this.cfg;
    const D = headDimOf(c, l);
    const p = `model.layers.${l}`;
    const h = ops.rmsnorm(ctx, x, this.w(`${p}.input_layernorm.weight`), c.rmsEps);
    let q = ops.linear(ctx, h, this.w(`${p}.self_attn.q_proj.weight`), this.bias(`${p}.self_attn.q_proj.bias`));
    if (c.kvFrom?.[l] !== undefined) {
      // Only a query: the keys and values are another layer's (Gemma 4), written earlier in this pass.
      if (c.qkNorm) q = ops.rmsnorm(ctx, q, this.w(`${p}.self_attn.q_norm.weight`), c.rmsEps, D);
      ops.rope(ctx, q, T, c.numHeads, D, invFreq, c.ropeInterleaved, step);
      return { q };
    }
    let k = ops.linear(ctx, h, this.w(`${p}.self_attn.k_proj.weight`), this.bias(`${p}.self_attn.k_proj.bias`));
    let v = ops.linear(ctx, h, this.w(`${p}.self_attn.v_proj.weight`), this.bias(`${p}.self_attn.v_proj.bias`));
    if (delta) {
      delta(h, q, `${l}.q_proj`);
      delta(h, k, `${l}.k_proj`);
      delta(h, v, `${l}.v_proj`);
    }
    let gate: Tensor | undefined;
    if (c.attnOutputGate) {
      // q_proj writes each head as [query | gate].
      gate = ops.headSlices(ctx, q, { H: c.numHeads, D, blocks: 2, pick: 1 });
      q = ops.headSlices(ctx, q, { H: c.numHeads, D, blocks: 2, pick: 0 });
    }
    if (c.qkNorm) {
      const qn = ops.rmsnorm(ctx, q, this.w(`${p}.self_attn.q_norm.weight`), c.rmsEps, D);
      const kn = ops.rmsnorm(ctx, k, this.w(`${p}.self_attn.k_norm.weight`), c.rmsEps, D);
      q = qn;
      k = kn;
    }
    // Gemma 4 normalises each value head too, with no learned weight.
    if (c.valueNorm) v = ops.rmsnorm(ctx, v, this.ones(D), c.rmsEps, D);
    ops.rope(ctx, q, T, c.numHeads, D, invFreq, c.ropeInterleaved);
    ops.rope(ctx, k, T, c.numKvHeads, D, invFreq, c.ropeInterleaved);
    ops.kvWrite(ctx, k, v, this.kv[l]!.k, this.kv[l]!.v, T, this.maxSeq, this.kvDtype, this.slots[l], c.numKvHeads);
    return { q, gate };
  }

  /** A weight of D ones: RMSNorm without a learned scale. */
  private ones(D: number): Tensor {
    let t = this.unit.get(D);
    if (!t) {
      const buf = this.backend.alloc(D * 4, "storage", `ones/${D}`);
      this.backend.write(buf, 0, new Float32Array(D).fill(1));
      t = { shape: [D], dtype: "f32", buf };
      this.unit.set(D, t);
    }
    return t;
  }

  private bias(name: string): Tensor | undefined {
    return this.cfg.attentionBias ? this.weights.get(name) : undefined;
  }
}
