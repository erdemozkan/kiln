import { ActivationPool, Ctx, byteSize, packStep, q4ScaleBytes, view, type Backend, type Buf, type Dispatch, type StepParams, type Tensor } from "@kiln/core";
import { ops, type KvDtype } from "@kiln/kernels";
import { ropeInvFreq, type LlamaConfig } from "./config";

/** A recorded forward pass for a fixed number of tokens. */
export interface ForwardPlan {
  readonly T: number;
  /** The forward pass; the last dispatch samples, so it can be submitted separately from the rest. */
  readonly dispatches: readonly Dispatch[];
  /** u32 [T] token ids, written before each replay. */
  readonly tokens: Buf;
  /** u32 [1] sampled token for row `step.lastRow`; u32 [T], one per row, for a verify plan. */
  readonly out: Buf;
  destroy(): void;
}

export interface LlamaOptions {
  /** KV cache length in tokens. */
  maxSeq?: number;
  /** KV cache storage: f16 (default) halves the cache and the attention reads; f32 for exactness. */
  kvDtype?: KvDtype;
  /** Stack q/k/v and gate/up into single matmuls (default true). False keeps the unfused graph. */
  fuse?: boolean;
}

export class MissingWeightError extends Error {}

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
  private readonly invFreq: Tensor;
  /** RoPE frequencies of sliding-window layers, when their base differs (Gemma 3); else `invFreq`. */
  private readonly localInvFreq: Tensor;
  /** q/k/v stacked into one `qkv` weight (not for qk-norm models, which normalise q and k separately). */
  private readonly fusedQkv: boolean;
  /** gate/up stacked into one `gate_up` weight, consumed by the SiLU-gated matmul. */
  private readonly fusedMlp: boolean;

  constructor(
    readonly backend: Backend,
    readonly cfg: LlamaConfig,
    private readonly weights: Map<string, Tensor>,
    opts: LlamaOptions = {},
  ) {
    this.maxSeq = opts.maxSeq ?? 2048;
    this.kvDtype = opts.kvDtype ?? "f16";
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
      this.kv.push({
        k: backend.alloc(this.maxSeq * row * kvBytes, "storage", `kv/${l}/k`),
        v: backend.alloc(this.maxSeq * row * kvBytes, "storage", `kv/${l}/v`),
      });
    }
    const freqs = (f: Float32Array, label: string): Tensor => {
      const buf = backend.alloc(f.byteLength, "storage", label);
      backend.write(buf, 0, f);
      return { shape: [f.length], dtype: "f32", buf };
    };
    this.invFreq = freqs(ropeInvFreq(cfg), "rope/inv_freq");
    this.localInvFreq =
      cfg.ropeLocalTheta !== undefined && cfg.ropeLocalTheta !== cfg.ropeTheta
        ? freqs(ropeInvFreq({ headDim: cfg.headDim, ropeTheta: cfg.ropeLocalTheta }), "rope/inv_freq/local")
        : this.invFreq;
    this.w("model.embed_tokens.weight");
    const fuse = opts.fuse ?? true;
    this.fusedQkv = fuse && !cfg.qkNorm && this.fuse("self_attn", ["q_proj", "k_proj", "v_proj"], "qkv", cfg.attentionBias);
    this.fusedMlp = fuse && this.fuse("mlp", ["gate_proj", "up_proj"], "gate_up", false);
  }

  /**
   * Stacks per-layer weights along N (rows) into one tensor per layer so they run as one matmul,
   * freeing the originals. Skipped (returns false) unless every part shares dtype and K.
   */
  private fuse(module: string, parts: string[], fused: string, withBias: boolean): boolean {
    const names = (l: number, kind: string) => parts.map((p) => `model.layers.${l}.${module}.${p}.${kind}`);
    for (let l = 0; l < this.cfg.numLayers; l++) {
      const ts = names(l, "weight").map((n) => this.weights.get(n));
      if (ts.some((t) => !t || t.dtype !== ts[0]!.dtype || t.shape[1] !== ts[0]!.shape[1])) return false;
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
    const scales = dtype === "q4" ? copyPlane((t) => q4ScaleBytes(t.shape), (t) => t.scales!, `${name}/scales`) : undefined;
    for (const [i, t] of ts.entries()) {
      this.backend.free(t.buf);
      if (t.scales) this.backend.free(t.scales);
      this.weights.delete(parts[i]!);
    }
    this.weights.set(name, { shape, dtype, buf, scales });
  }

  /** GPU bytes held: weights, the KV cache, and the intermediates shared by every plan built so far. */
  get memory(): { weights: number; kv: number; activations: number } {
    let weights = 0;
    for (const t of this.weights.values()) weights += t.buf.size + (t.scales?.size ?? 0);
    const kv = this.kv.reduce((n, { k, v }) => n + k.size + v.size, 0);
    return { weights, kv, activations: this.activations.bytes };
  }

  /** Bytes of one position's K (or V) in one layer's cache. */
  get kvRowBytes(): number {
    return this.cfg.numKvHeads * this.cfg.headDim * (this.kvDtype === "f16" ? 2 : 4);
  }

  /**
   * Copies the K/V of positions 0…n-1 out of every layer's cache: layer 0's K rows, then its V rows,
   * then layer 1's, and so on. The copies are queued at once, so they read the cache as it stands
   * when this is called, even if more work is submitted before they resolve.
   */
  async readKv(n: number): Promise<Uint8Array> {
    const bytes = n * this.kvRowBytes;
    const parts = this.kv.flatMap(({ k, v }) => [this.backend.readback(k, 0, bytes), this.backend.readback(v, 0, bytes)]);
    const out = new Uint8Array(parts.length * bytes);
    for (const [i, p] of (await Promise.all(parts)).entries()) out.set(new Uint8Array(p), i * bytes);
    return out;
  }

  /** Writes K/V for positions 0…n-1 back into every layer's cache, in `readKv`'s layout. */
  writeKv(n: number, data: Uint8Array): void {
    const bytes = n * this.kvRowBytes;
    if (n > this.maxSeq || data.length !== this.kv.length * 2 * bytes) throw new Error(`writeKv: ${data.length} bytes do not hold ${n} positions`);
    for (const [l, { k, v }] of this.kv.entries()) {
      this.backend.write(k, 0, data.subarray(2 * l * bytes, (2 * l + 1) * bytes));
      this.backend.write(v, 0, data.subarray((2 * l + 1) * bytes, (2 * l + 2) * bytes));
    }
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
      };
      const inputNorm = { w: this.w(`${p}.input_layernorm.weight`), eps: c.rmsEps };
      let q: Tensor;
      if (this.fusedQkv) {
        // RMSNorm → one qkv matmul (+bias) → RoPE fused with the cache write.
        const qkv = ops.linear(ctx, x, this.w(`${p}.self_attn.qkv.weight`), { norm: inputNorm, bias: this.bias(`${p}.self_attn.qkv.bias`) });
        q = ops.ropeKv(ctx, qkv, kc, vc, attnShape, invFreq, c.ropeInterleaved);
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

    return { T, dispatches: ctx.dispatches, tokens, out, destroy: () => ctx.destroy() };
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
    ops.kvWrite(ctx, k, v, this.kv[l]!.k, this.kv[l]!.v, T, this.maxSeq, this.kvDtype);
    return q;
  }

  private bias(name: string): Tensor | undefined {
    return this.cfg.attentionBias ? this.weights.get(name) : undefined;
  }
}
