import { ActivationPool, Ctx, byteSize, isBlockQuant, packStep, q4ScaleBytes, view, type Backend, type Buf, type Dispatch, type StepParams, type Tensor } from "@kiln/core";
import { ATTN_WG, ops, type KvDtype } from "@kiln/kernels";
import { ropeInvFreq, type LlamaConfig } from "./config";

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
  /** u32 [T] token ids, written before each replay. */
  readonly tokens: Buf;
  /** u32 [1] sampled token for row `step.lastRow`; u32 [T], one per row, for a verify plan. */
  readonly out: Buf;
  destroy(): void;
}

export interface LlamaOptions {
  /** KV cache length in tokens (default `defaultMaxSeq`). */
  maxSeq?: number;
  /** KV cache storage: f16 (default) halves the cache and the attention reads; f32 for exactness. */
  kvDtype?: KvDtype;
  /**
   * Stack q/k/v and gate/up into single matmuls (default true). False keeps the unfused graph; an
   * object chooses each (missing parts default to true).
   */
  fuse?: boolean | { qkv?: boolean; mlp?: boolean };
}

export class MissingWeightError extends Error {}

/** KV cache memory the default context may take, and the longest default context. */
export const KV_BUDGET_BYTES = 512 * 2 ** 20;
export const MAX_DEFAULT_SEQ = 8192;

/**
 * Rows a sliding layer's ring holds beyond its window: keys written ahead of the oldest one a query
 * still needs, up to the largest prefill chunk (128) or verify pass (MAX_VERIFY_ROWS).
 */
export const RING_SLACK = 128;

/**
 * Cache slots of layer l: maxSeq for full layers; for sliding layers a ring of window + RING_SLACK
 * slots (at least ATTN_WG, which the attention kernels' key runs need), when that is smaller.
 */
export function ringSlots(cfg: LlamaConfig, maxSeq: number, l: number): number {
  const win = cfg.slidingLayers?.[l] ? cfg.slidingWindow : undefined;
  return win ? Math.min(maxSeq, Math.max(ATTN_WG, win + RING_SLACK)) : maxSeq;
}

/** KV cache bytes of a model at a context length: keys and values of every layer, rings included. */
export function kvCacheBytes(cfg: LlamaConfig, maxSeq: number, kvDtype: KvDtype = "f16"): number {
  const row = cfg.numKvHeads * cfg.headDim * (kvDtype === "f16" ? 2 : 4);
  let slots = 0;
  for (let l = 0; l < cfg.numLayers; l++) slots += ringSlots(cfg, maxSeq, l);
  return 2 * slots * row;
}

/**
 * Default context length: the model's trained context, at most MAX_DEFAULT_SEQ tokens, and no more
 * than fits KV_BUDGET_BYTES of cache (in multiples of 256, never under 2048 unless the model's own
 * context is shorter). Sliding layers count as their rings: Gemma 3 4B takes 305 MB at 8192 tokens,
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
  /** Cache slots per layer: maxSeq, or a ring for sliding layers (see `ringSlots`). */
  readonly slots: number[] = [];
  private readonly invFreq: Tensor;
  /** RoPE frequencies of sliding-window layers, when their base differs (Gemma 3); else `invFreq`. */
  private readonly localInvFreq: Tensor;
  /** q/k/v stacked into one `qkv` weight; qk-norm models normalise its q and k heads in `qkNormRopeKv`. */
  private readonly fusedQkv: boolean;
  /** gate/up stacked into one `gate_up` weight, consumed by the SiLU-gated matmul. */
  private readonly fusedMlp: boolean;

  constructor(
    readonly backend: Backend,
    readonly cfg: LlamaConfig,
    private readonly weights: Map<string, Tensor>,
    opts: LlamaOptions = {},
  ) {
    this.kvDtype = opts.kvDtype ?? "f16";
    this.maxSeq = opts.maxSeq ?? defaultMaxSeq(cfg, this.kvDtype);
    const kvBytes = this.kvDtype === "f16" ? 2 : 4;
    this.step = backend.alloc(32, "storage", "step");
    this.history = backend.alloc(this.maxSeq * 4, "storage", "history");
    this.seen = backend.alloc(Math.ceil(cfg.vocabSize / 32) * 4, "storage", "seen");
    this.allowed = backend.alloc((1 + Math.ceil(cfg.vocabSize / 32)) * 4, "storage", "allowed");
    this.allowedRows = backend.alloc(MAX_VERIFY_ROWS * (1 + Math.ceil(cfg.vocabSize / 32)) * 4, "storage", "allowed/rows");
    this.setAllowed(undefined);
    this.activations = new ActivationPool(backend);
    const row = cfg.numKvHeads * cfg.headDim;
    for (let l = 0; l < cfg.numLayers; l++) {
      const slots = ringSlots(cfg, this.maxSeq, l);
      this.slots.push(slots);
      this.kv.push({
        k: backend.alloc(slots * row * kvBytes, "storage", `kv/${l}/k`),
        v: backend.alloc(slots * row * kvBytes, "storage", `kv/${l}/v`),
      });
    }
    const freqs = (f: Float32Array, label: string): Tensor => {
      const buf = backend.alloc(f.byteLength, "storage", label);
      backend.write(buf, 0, f);
      return { shape: [f.length], dtype: "f32", buf };
    };
    this.invFreq = freqs(ropeInvFreq(cfg), "rope/inv_freq");
    this.localInvFreq =
      cfg.ropeLocalTheta !== undefined && (cfg.ropeLocalTheta !== cfg.ropeTheta || cfg.ropeLinear)
        ? freqs(ropeInvFreq({ headDim: cfg.headDim, ropeTheta: cfg.ropeLocalTheta }), "rope/inv_freq/local")
        : this.invFreq;
    this.w("model.embed_tokens.weight");
    const fuse = typeof opts.fuse === "object" ? { qkv: true, mlp: true, ...opts.fuse } : { qkv: opts.fuse ?? true, mlp: opts.fuse ?? true };
    this.fusedQkv = fuse.qkv && this.fuse("self_attn", ["q_proj", "k_proj", "v_proj"], "qkv", cfg.attentionBias);
    // Granite 4 checkpoints store gate_up already stacked.
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

  /** Bytes of one position's K (or V) in one layer's cache. */
  get kvRowBytes(): number {
    return this.cfg.numKvHeads * this.cfg.headDim * (this.kvDtype === "f16" ? 2 : 4);
  }

  /** Bytes `readKv(n)` returns: each layer's first min(n, slots) K rows and as many V rows. */
  kvSnapshotBytes(n: number): number {
    return this.slots.reduce((t, slots) => t + 2 * Math.min(n, slots) * this.kvRowBytes, 0);
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
      const bytes = Math.min(n, this.slots[l]!) * this.kvRowBytes;
      return [this.backend.readback(k, 0, bytes), this.backend.readback(v, 0, bytes)];
    });
    const out = new Uint8Array(this.kvSnapshotBytes(n));
    let at = 0;
    for (const p of await Promise.all(parts)) {
      out.set(new Uint8Array(p), at);
      at += p.byteLength;
    }
    return out;
  }

  /** Writes K/V for positions 0…n-1 back into every layer's cache, in `readKv`'s layout. */
  writeKv(n: number, data: Uint8Array): void {
    if (n > this.maxSeq || data.length !== this.kvSnapshotBytes(n)) throw new Error(`writeKv: ${data.length} bytes do not hold ${n} positions`);
    let at = 0;
    for (const [l, { k, v }] of this.kv.entries()) {
      const bytes = Math.min(n, this.slots[l]!) * this.kvRowBytes;
      this.backend.write(k, 0, data.subarray(at, at + bytes));
      this.backend.write(v, 0, data.subarray(at + bytes, at + 2 * bytes));
      at += 2 * bytes;
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
      if (slots >= this.maxSeq || written <= slots) continue;
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

  setStep(s: StepParams): void {
    this.backend.write(this.step, 0, packStep(s));
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
    const D = c.headDim;
    const gelu = c.activation === "gelu";
    const norm = (t: Tensor, name: string) => ops.rmsnorm(ctx, t, this.w(name), c.rmsEps);

    const x = ops.embed(ctx, this.w("model.embed_tokens.weight"), tokens, T, c.embedScale ?? 1);
    for (let l = 0; l < c.numLayers; l++) {
      const p = `model.layers.${l}`;
      const { k: kc, v: vc } = this.kv[l]!;
      const sliding = !!c.slidingLayers?.[l];
      const invFreq = sliding ? this.localInvFreq : this.invFreq;
      const attnShape = {
        T,
        H: c.numHeads,
        Hkv: c.numKvHeads,
        D,
        maxSeq: this.maxSeq,
        kv: this.kvDtype,
        scale: c.attnScale,
        window: sliding ? c.slidingWindow : undefined,
        slots: this.slots[l],
      };
      const inputNorm = { w: this.w(`${p}.input_layernorm.weight`), eps: c.rmsEps };
      let q: Tensor;
      if (this.fusedQkv) {
        // RMSNorm → one qkv matmul (+bias) → RoPE fused with the cache write.
        const qkv = ops.linear(ctx, x, this.w(`${p}.self_attn.qkv.weight`), { norm: inputNorm, bias: this.bias(`${p}.self_attn.qkv.bias`) });
        q = c.qkNorm
          ? ops.qkNormRopeKv(ctx, qkv, this.w(`${p}.self_attn.q_norm.weight`), this.w(`${p}.self_attn.k_norm.weight`), c.rmsEps, kc, vc, attnShape, invFreq, c.ropeInterleaved)
          : ops.ropeKv(ctx, qkv, kc, vc, attnShape, invFreq, c.ropeInterleaved);
      } else {
        q = this.attentionInputs(ctx, x, l, T, invFreq);
      }
      const a = ops.attention(ctx, q, kc, vc, attnShape);
      if (c.sandwichNorms) {
        // Gemma: the attention output is normalised before the residual add.
        const o = ops.linear(ctx, a, this.w(`${p}.self_attn.o_proj.weight`));
        ops.addInPlace(ctx, x, norm(o, `${p}.post_attention_layernorm.weight`));
      } else {
        ops.linear(ctx, a, this.w(`${p}.self_attn.o_proj.weight`), { residual: x });
      }

      const mlpNorm = { w: this.w(`${p}.${c.sandwichNorms ? "pre_feedforward_layernorm" : "post_attention_layernorm"}.weight`), eps: c.rmsEps };
      let m: Tensor;
      if (this.fusedMlp) {
        m = ops.linear(ctx, x, this.w(`${p}.mlp.gate_up.weight`), { norm: mlpNorm, silu: !gelu, gelu });
      } else {
        const h2 = ops.linear(ctx, x, this.w(`${p}.mlp.gate_proj.weight`), { norm: mlpNorm });
        const u = ops.linear(ctx, x, this.w(`${p}.mlp.up_proj.weight`), { norm: mlpNorm });
        m = ops.siluMul(ctx, h2, u, gelu ? "gelu" : "silu");
      }
      if (c.sandwichNorms) {
        const d = ops.linear(ctx, m, this.w(`${p}.mlp.down_proj.weight`));
        ops.addInPlace(ctx, x, norm(d, `${p}.post_feedforward_layernorm.weight`));
      } else {
        ops.linear(ctx, m, this.w(`${p}.mlp.down_proj.weight`), { residual: x });
      }
    }
    const rows = verify ? view(x, [T, c.hiddenSize]) : ops.copyRow(ctx, view(x, [T, c.hiddenSize]));
    const head = this.weights.get("lm_head.weight") ?? this.w("model.embed_tokens.weight");
    const logits = ops.linear(ctx, rows, head, { norm: { w: this.w("model.norm.weight"), eps: c.rmsEps } });
    if (verify) ops.sampleRows(ctx, logits, out, this.seen, this.allowedRows);
    else ops.sample(ctx, logits, out, this.seen, this.allowed);

    const dispatches = ctx.dispatches;
    // Non-verify plans end copy_row → head matmul → sample.
    const body = verify ? dispatches : dispatches.slice(0, dispatches.length - 3);
    if (!verify && dispatches[body.length]!.label !== "copy_row") throw new Error("buildPlan: unexpected head layout");
    return { T, dispatches, body, tokens, out, destroy: () => ctx.destroy() };
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
    for (const t of this.weights.values()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      if (t.mins) this.backend.free(t.mins);
    }
    this.backend.free(this.invFreq.buf);
    if (this.localInvFreq !== this.invFreq) this.backend.free(this.localInvFreq.buf);
    this.backend.free(this.step);
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
  private attentionInputs(ctx: Ctx, x: Tensor, l: number, T: number, invFreq: Tensor): Tensor {
    const c = this.cfg;
    const D = c.headDim;
    const p = `model.layers.${l}`;
    const h = ops.rmsnorm(ctx, x, this.w(`${p}.input_layernorm.weight`), c.rmsEps);
    let q = ops.linear(ctx, h, this.w(`${p}.self_attn.q_proj.weight`), this.bias(`${p}.self_attn.q_proj.bias`));
    let k = ops.linear(ctx, h, this.w(`${p}.self_attn.k_proj.weight`), this.bias(`${p}.self_attn.k_proj.bias`));
    const v = ops.linear(ctx, h, this.w(`${p}.self_attn.v_proj.weight`), this.bias(`${p}.self_attn.v_proj.bias`));
    if (c.qkNorm) {
      const qn = ops.rmsnorm(ctx, q, this.w(`${p}.self_attn.q_norm.weight`), c.rmsEps, D);
      const kn = ops.rmsnorm(ctx, k, this.w(`${p}.self_attn.k_norm.weight`), c.rmsEps, D);
      q = qn;
      k = kn;
    }
    ops.rope(ctx, q, T, c.numHeads, D, invFreq, c.ropeInterleaved);
    ops.rope(ctx, k, T, c.numKvHeads, D, invFreq, c.ropeInterleaved);
    ops.kvWrite(ctx, k, v, this.kv[l]!.k, this.kv[l]!.v, T, this.maxSeq, this.kvDtype, this.slots[l]);
    return q;
  }

  private bias(name: string): Tensor | undefined {
    return this.cfg.attentionBias ? this.weights.get(name) : undefined;
  }
}
