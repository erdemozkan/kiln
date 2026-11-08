import { f32, isBlockQuant, lastDim, numel, u32, type Buf, type Ctx, type Grid, type Tensor, type WeightDType } from "@kiln/core";
import {
  ATTN_CHUNK,
  ATTN_WG,
  EW,
  MAX_X,
  MM_SLICE,
  addSpec,
  advanceSpec,
  attentionCombineSpec,
  attentionSpec,
  attentionSplitSpec,
  attentionTile,
  attentionTiledSpec,
  attentionSpanSpec,
  spanTile,
  type SpanTile,
  copyRowSpec,
  avgPoolSpec,
  embedSpec,
  actSpec,
  im2col1dSpec,
  channelNormSpec,
  groupConv1dSpec,
  scaleAddSpec,
  type Activation,
  layernormSpec,
  gatedActSpec,
  gemmSpec,
  GEMM_TILE,
  gemvLayout,
  gemvRowsPerGroup,
  gemvSpec,
  kvWriteSpec,
  matmulRowsPerGroup,
  matmulSpec,
  rmsnormSpec,
  qkNormRopeKvSpec,
  ropeKvSpec,
  ropeSpec,
  sampleSpec,
  siluMulSpec,
  type KvDtype,
  type MatmulEpilogue,
  type MatmulLanes,
} from "./wgsl";

/** Launch shape for `threads` elementwise invocations; spills into y past the per-dimension limit. */
export function grid1d(threads: number): Grid {
  const groups = Math.ceil(threads / EW);
  const x = Math.min(groups, MAX_X);
  return [x, Math.ceil(groups / x), 1];
}

function weightBufs(w: Tensor): Buf[] {
  if (isBlockQuant(w.dtype)) {
    if (!w.scales) throw new Error(`${w.dtype} tensor is missing its scales plane`);
    return w.mins ? [w.buf, w.scales, w.mins] : [w.buf, w.scales];
  }
  return [w.buf];
}

function weightDType(w: Tensor): WeightDType {
  if (w.dtype === "u32") throw new Error("a u32 tensor cannot be used as weights");
  return w.dtype;
}

function rowsOf(x: Tensor): number {
  return numel(x.shape) / lastDim(x);
}

export interface LinearOptions {
  bias?: Tensor;
  /** RMSNorm x with this weight before the product (fused into the matmul). */
  norm?: { w: Tensor; eps: number };
  /** Accumulate into this [M, N] tensor instead of allocating the output (fused residual add). */
  residual?: Tensor;
  /** W is [gate; up] stacked along N: return silu(x·gateᵀ) * (x·upᵀ), shape [M, N / 2]. */
  silu?: boolean;
  /** As `silu`, with tanh-approximated GELU (Gemma). */
  gelu?: boolean;
  /** Force a lanes-per-row variant (kernel checks, autotuning); otherwise the tuned choice. */
  lanes?: MatmulLanes;
  /** f16 dot products (needs shader-f16). Defaults to `setMatmulF16`. */
  f16?: boolean;
  /** Subgroup reductions where the device allows them for this variant. Defaults to `setSubgroups`. */
  subgroups?: boolean;
}

let f16Default = false;
/** Use f16 dot products in every matmul from now on (the caller checks for shader-f16). */
export function setMatmulF16(on: boolean): void {
  f16Default = on;
}

/** Rows of x from which prefill matmuls use the tiled GEMM rather than the 8-row kernel. */
let gemmMinRows = 64;
/** Use the tiled GEMM for matmuls with at least this many rows (Infinity turns it off). */
export function setGemmMinRows(rows: number): void {
  gemmMinRows = rows;
}

let gemvOn = true;
/** Use the block-per-lane GEMV for one-token matmuls (default), or the general kernel. */
export function setGemv(on: boolean): void {
  gemvOn = on;
}

let subgroupsDefault = false;
/** Use subgroup reductions in matmuls from now on, where the device supports them. */
export function setSubgroups(on: boolean): void {
  subgroupsDefault = on;
}

/** Tuned lanes per row, keyed by `${wdtype}/mt${mt}/${N}x${K}`; filled by the autotuner. */
const tunedLanes = new Map<string, MatmulLanes>();
export const lanesKey = (wdtype: WeightDType, mt: 1 | 8, N: number, K: number): string => `${wdtype}/mt${mt}/${N}x${K}`;
export function setMatmulLanes(key: string, lanes: MatmulLanes): void {
  tunedLanes.set(key, lanes);
}
export function matmulLanes(): ReadonlyMap<string, MatmulLanes> {
  return tunedLanes;
}

/** x [M, K] · W[N, K]ᵀ → [M, N], with optional fused RMSNorm, bias, residual add or SiLU-gating. */
export function linear(ctx: Ctx, x: Tensor, w: Tensor, opts: LinearOptions | Tensor = {}): Tensor {
  const o: LinearOptions = "buf" in opts ? { bias: opts } : opts;
  const gated = !!(o.silu || o.gelu);
  const K = lastDim(x);
  const M = rowsOf(x);
  const [N, wk] = w.shape as [number, number];
  if (wk !== K) throw new Error(`linear: input has K=${K} but weight is [${w.shape}]`);
  if (K % 32 !== 0) throw new Error(`linear: K must be a multiple of 32 (K=${K})`);
  if (gated && (N % 2 !== 0 || o.residual)) throw new Error("linear: a gated matmul needs an even N and no residual");
  const outN = gated ? N / 2 : N;
  if (o.residual && (rowsOf(o.residual) !== M || lastDim(o.residual) !== outN)) {
    throw new Error(`linear: residual [${o.residual.shape}] does not match output [${M}, ${outN}]`);
  }
  const mt = M === 1 ? 1 : 8;
  if (o.norm && mt === 8) {
    // Prefill has few dispatches to save, and recomputing 8 row norms per workgroup costs more than
    // one separate pass, so the norm is fused only into the decode GEMV.
    const xn = rmsnorm(ctx, x, o.norm.w, o.norm.eps);
    return linear(ctx, xn, w, { ...o, norm: undefined });
  }
  const y = o.residual ?? ctx.alloc([M, outN]);
  const wdtype = weightDType(w);
  const affine = !!w.mins;
  const tag = `${wdtype}${affine ? "a" : ""}`;
  const f16 = o.f16 ?? f16Default;
  if (M >= gemmMinRows && !o.lanes && !o.f16 && gated) {
    // The GEMM's gated epilogue pairs 32 gate rows with 32 up rows per tile and ran 7% slower over a
    // Gemma 3 4B prefill than a plain GEMM over [gate | up] followed by this elementwise pass.
    const gu = linear(ctx, x, w, { bias: o.bias });
    const act = o.gelu ? "gelu" : "silu";
    ctx.run(gatedActSpec(act), [ctx.uniform([u32(M * outN), u32(outN), u32(0), u32(0)]), gu.buf, y.buf], grid1d(M * outN), `gated_${act} ${outN}`);
    return y;
  }
  if (M >= gemmMinRows && !o.lanes && !o.f16) {
    const epi: MatmulEpilogue = o.gelu ? "gelu" : o.silu ? "silu" : o.residual ? "add" : "store";
    const groups = Math.ceil(outN / (gated ? GEMM_TILE / 2 : GEMM_TILE));
    const grid: Grid = [Math.min(groups, MM_SLICE), Math.ceil(M / GEMM_TILE), Math.ceil(groups / MM_SLICE)];
    const bufs = [ctx.uniform([u32(M), u32(N), u32(K), u32(0)]), x.buf, ...weightBufs(w)];
    if (o.bias) bufs.push(o.bias.buf);
    bufs.push(y.buf);
    ctx.run(gemmSpec({ wdtype, affine, bias: !!o.bias, epi }), bufs, grid, `gemm/${tag} ${N}x${K}${gated ? `+${epi}` : ""}${o.residual ? "+add" : ""}`);
    return y;
  }
  const epi: MatmulEpilogue = o.gelu ? "gelu" : o.silu ? "silu" : o.residual ? "add" : "store";
  if (M === 1 && gemvOn && !o.lanes && !f16 && !(o.subgroups ?? subgroupsDefault)) {
    const layout = gemvLayout(outN, K);
    const groups = Math.ceil(outN / gemvRowsPerGroup(epi, layout));
    const grid: Grid = [Math.min(groups, MM_SLICE), 1, Math.ceil(groups / MM_SLICE)];
    const bufs = [ctx.uniform([u32(M), u32(N), u32(K), f32(o.norm?.eps ?? 0)]), x.buf, ...weightBufs(w)];
    if (o.norm) bufs.push(o.norm.w.buf);
    if (o.bias) bufs.push(o.bias.buf);
    bufs.push(y.buf);
    ctx.run(gemvSpec({ wdtype, affine, norm: !!o.norm, bias: !!o.bias, epi, layout }), bufs, grid, `gemv/${tag} ${N}x${K}${o.norm ? "+norm" : ""}${gated ? `+${epi}` : ""}${o.residual ? "+add" : ""}`);
    return y;
  }
  const lanes = o.lanes ?? tunedLanes.get(lanesKey(wdtype, mt, N, K)) ?? 8;
  if (f16 && !ctx.backend.limits.features.has("shader-f16")) throw new Error("linear: f16 math needs the shader-f16 feature");
  // The butterfly needs each row's lanes inside one subgroup.
  const sg = (o.subgroups ?? subgroupsDefault) && lanes <= (ctx.backend.limits.subgroupMinSize ?? 0);
  const spec = matmulSpec({ wdtype, affine, mt, lanes, norm: !!o.norm, bias: !!o.bias, epi, f16, sg });
  const groups = Math.ceil(outN / matmulRowsPerGroup({ lanes, epi }));
  const grid: Grid = [Math.min(groups, MM_SLICE), Math.ceil(M / mt), Math.ceil(groups / MM_SLICE)];
  const bufs = [ctx.uniform([u32(M), u32(N), u32(K), f32(o.norm?.eps ?? 0)]), x.buf, ...weightBufs(w)];
  if (o.norm) bufs.push(o.norm.w.buf);
  if (o.bias) bufs.push(o.bias.buf);
  bufs.push(y.buf);
  ctx.run(spec, bufs, grid, `matmul/${tag}/mt${mt} ${N}x${K}${o.norm ? "+norm" : ""}${gated ? `+${epi}` : ""}${o.residual ? "+add" : ""}`);
  return y;
}

/** RMSNorm over the last axis. Pass `dim` to normalise sub-rows (per-head q/k norm). */
export function rmsnorm(ctx: Ctx, x: Tensor, w: Tensor, eps: number, dim = lastDim(x)): Tensor {
  const rows = numel(x.shape) / dim;
  if (numel(w.shape) !== dim) throw new Error(`rmsnorm: weight [${w.shape}] does not match dim ${dim}`);
  const y = ctx.alloc(x.shape);
  ctx.run(rmsnormSpec(), [ctx.uniform([u32(rows), u32(dim), f32(eps), u32(0)]), x.buf, w.buf, y.buf], [rows, 1, 1]);
  return y;
}

/** In-place RoPE on x [T, H * D]. `invFreq` has D / 2 entries (scaling already applied). */
export function rope(ctx: Ctx, x: Tensor, T: number, H: number, D: number, invFreq: Tensor, interleaved: boolean): void {
  ctx.run(ropeSpec(interleaved), [ctx.uniform([u32(T), u32(H), u32(D), u32(0)]), ctx.step, invFreq.buf, x.buf], grid1d((T * H * D) / 2));
}

/**
 * RoPE on q and k fused with the cache write: qkv [T, (H + 2·Hkv)·D] → rotated q [T, H·D];
 * rotated k and v land in kc / vc at positions step.pos0…
 */
export function ropeKv(ctx: Ctx, qkv: Tensor, kc: Buf, vc: Buf, s: AttentionShape, invFreq: Tensor, interleaved: boolean): Tensor {
  const q = ctx.alloc([s.T, s.H * s.D]);
  const W = (s.H + 2 * s.Hkv) * s.D;
  const params = ctx.uniform([u32(s.T), u32(s.H), u32(s.Hkv), u32(s.D), u32(s.maxSeq), u32(cacheSlots(s)), u32(0), u32(0)]);
  ctx.run(ropeKvSpec(interleaved, s.kv), [params, ctx.step, invFreq.buf, qkv.buf, q.buf, kc, vc], grid1d((s.T * W) / 2));
  return q;
}

/**
 * Per-head q/k RMSNorm, RoPE and the cache write on the output of a stacked qkv matmul (qk-norm
 * models): qkv [T, (H + 2·Hkv)·D] → normalised, rotated q [T, H·D]; k and v land in kc / vc.
 */
export function qkNormRopeKv(ctx: Ctx, qkv: Tensor, qNorm: Tensor, kNorm: Tensor, eps: number, kc: Buf, vc: Buf, s: AttentionShape, invFreq: Tensor, interleaved: boolean): Tensor {
  if (s.D > 256 || s.D % 4 !== 0) throw new Error(`qkNormRopeKv: head dim ${s.D} must be a multiple of 4 and at most 256`);
  const q = ctx.alloc([s.T, s.H * s.D]);
  const params = ctx.uniform([u32(s.T), u32(s.H), u32(s.Hkv), u32(s.D), u32(s.maxSeq), u32(cacheSlots(s)), f32(eps), u32(0)]);
  ctx.run(qkNormRopeKvSpec(interleaved, s.kv), [params, ctx.step, invFreq.buf, qkv.buf, qNorm.buf, kNorm.buf, q.buf, kc, vc], [s.H + 2 * s.Hkv, s.T, 1]);
  return q;
}

/** Writes k/v into the caches at positions pos0…; `slots` < maxSeq makes the cache a ring (slot = position % slots). */
export function kvWrite(ctx: Ctx, k: Tensor, v: Tensor, kc: Buf, vc: Buf, T: number, maxSeq: number, kv: KvDtype = "f32", slots = maxSeq): void {
  const row = numel(k.shape) / T;
  if (kv === "f16" && row % 2 !== 0) throw new Error(`kvWrite: f16 cache needs an even row (row=${row})`);
  const threads = kv === "f16" ? (T * row) / 2 : T * row;
  ctx.run(kvWriteSpec(kv), [ctx.uniform([u32(T), u32(row), u32(maxSeq), u32(slots)]), ctx.step, k.buf, v.buf, kc, vc], grid1d(threads));
}

export type PrefillAttention = "tiled" | "rows";
let prefillAttention: PrefillAttention = "tiled";
/** Prefill attention kernel: K/V tiles shared by a tile of query rows (default), or one workgroup per row. */
export function setPrefillAttention(kind: PrefillAttention): void {
  prefillAttention = kind;
}

export interface AttentionShape {
  T: number;
  H: number;
  Hkv: number;
  D: number;
  maxSeq: number;
  /** Cache storage (default f32). */
  kv?: KvDtype;
  /** Score scale (default 1 / sqrt(D)). */
  scale?: number;
  /** Sliding window: each query sees only its last `window` keys (default 0, all keys). */
  window?: number;
  /**
   * Cache slots (default maxSeq). Fewer makes the cache a ring, position j in slot j % slots: enough
   * for a sliding layer when slots ≥ window + T - 1 and slots ≥ ATTN_WG (see `ringSlots`).
   */
  slots?: number;
  /**
   * Every query row sees the same keys (a bidirectional span covering all of them, as in the vision
   * tower): lets the GPU use span attention, a register-tiled kernel for that case, when the cache is
   * f32 and not a ring.
   */
  span?: boolean;
  /**
   * Step block to read positions and the span from instead of the context's (cross-attention: a step
   * whose span covers every query, so each decoder row sees all encoder keys whatever its position).
   */
  step?: Buf;
  /**
   * T5's learned relative position bias, a table [buckets, H] added to every score by bucket of key
   * minus query position. Runs the per-row kernel.
   */
  relBias?: { table: Tensor; buckets: number; maxDistance: number; bidirectional: boolean };
}

let spanOverride: SpanTile | undefined;
/** Forces the span attention tile (benchmarks and checks); undefined restores the default. */
export function setSpanTile(t: SpanTile | undefined): void {
  spanOverride = t;
}

export const cacheSlots = (s: Pick<AttentionShape, "maxSeq" | "slots">): number => s.slots ?? s.maxSeq;

/**
 * Causal GQA attention over the cache. q [T, H * D] → [T, H * D].
 * Decode (T = 1) uses split-K over ATTN_CHUNK-key splits so the GPU stays busy with few heads;
 * prefill tiles query rows that share a kv head, so each K/V block is read once per tile.
 */
export function attention(ctx: Ctx, q: Tensor, kc: Buf, vc: Buf, s: AttentionShape): Tensor {
  if (s.H % s.Hkv !== 0) throw new Error(`attention: ${s.H} heads not divisible by ${s.Hkv} kv heads`);
  if (s.D > ATTN_WG || s.D % 4 !== 0) throw new Error(`attention: head dim ${s.D} must be a multiple of 4 and at most ${ATTN_WG}`);
  const out = ctx.alloc([s.T, s.H * s.D]);
  const splits = Math.ceil(s.maxSeq / ATTN_CHUNK);
  const slots = cacheSlots(s);
  if (slots < s.maxSeq && (!s.window || slots < s.window + s.T - 1 || slots < ATTN_WG)) {
    throw new Error(`attention: a ${slots}-slot ring cannot hold a ${s.window ?? 0}-key window for ${s.T} rows`);
  }
  const step = s.step ?? ctx.step;
  const params = ctx.uniform([
    u32(s.T), u32(s.H), u32(s.Hkv), u32(s.D), u32(s.maxSeq), f32(s.scale ?? 1 / Math.sqrt(s.D)), u32(splits), u32(s.window ?? 0),
    u32(slots), u32(s.relBias?.buckets ?? 0), u32(s.relBias?.maxDistance ?? 0), u32(s.relBias?.bidirectional ? 1 : 0),
  ]);
  if (s.relBias) {
    ctx.run(attentionSpec(s.kv, true), [params, step, q.buf, kc, vc, s.relBias.table.buf, out.buf], [s.H, s.T, 1]);
    return out;
  }
  if (ctx.backend.name === "cpu") {
    // One CPU implementation serves every attention shape; the split-K pair is a GPU occupancy trick.
    ctx.run(attentionSpec(s.kv), [params, step, q.buf, kc, vc, out.buf], [s.H, s.T, 1]);
    return out;
  }
  if (s.T === 1) {
    const ml = ctx.alloc([s.H, splits, 2]);
    const po = ctx.alloc([s.H, splits, s.D]);
    ctx.run(attentionSplitSpec(s.kv), [params, step, q.buf, kc, vc, ml.buf, po.buf], [s.H, 1, splits]);
    ctx.run(attentionCombineSpec(), [params, step, ml.buf, po.buf, out.buf], [s.H, 1, 1]);
    return out;
  }
  if (s.span && (s.kv ?? "f32") === "f32" && slots === s.maxSeq && !s.window) {
    const t = spanOverride ?? spanTile(s.D, ctx.backend.limits.maxWorkgroupStorage);
    if (t) {
      ctx.run(attentionSpanSpec(s.D, t), [params, step, q.buf, kc, vc, out.buf], [Math.ceil(s.T / (8 * t.RT)), s.H, 1]);
      return out;
    }
  }
  if (prefillAttention === "tiled") {
    const tile = attentionTile(s.D, ctx.backend.limits.maxWorkgroupStorage);
    ctx.run(attentionTiledSpec(s.kv ?? "f32", s.D, tile), [params, step, q.buf, kc, vc, out.buf], [s.Hkv, Math.ceil((s.T * (s.H / s.Hkv)) / tile.R), 1]);
    return out;
  }
  ctx.run(attentionSpec(s.kv), [params, step, q.buf, kc, vc, out.buf], [s.H, s.T, 1]);
  return out;
}

/** act(gate) * up elementwise, SiLU by default. */
export function siluMul(ctx: Ctx, gate: Tensor, up: Tensor, act: "silu" | "gelu" = "silu"): Tensor {
  const n = numel(gate.shape);
  const o = ctx.alloc(gate.shape);
  ctx.run(siluMulSpec(act), [ctx.uniform([u32(n), u32(0), u32(0), u32(0)]), gate.buf, up.buf, o.buf], grid1d(n));
  return o;
}

/** a += b. If b has fewer elements than a it is broadcast over rows (bias). */
export function addInPlace(ctx: Ctx, a: Tensor, b: Tensor): void {
  const n = numel(a.shape);
  ctx.run(addSpec(), [ctx.uniform([u32(n), u32(numel(b.shape)), u32(0), u32(0)]), a.buf, b.buf], grid1d(n));
}

/**
 * Rows of `table` for the token ids, times `scale` (Gemma scales embeddings by sqrt(hidden)). With
 * `features`, ids from `vocab` on take row id - vocab of the features instead, unscaled (image tokens).
 */
export function embed(ctx: Ctx, table: Tensor, tokens: Buf, T: number, scale = 1, features?: { buf: Buf; vocab: number }): Tensor {
  const D = lastDim(table);
  const o = ctx.alloc([T, D]);
  const spec = embedSpec(weightDType(table), !!table.mins, !!features);
  const bufs = [ctx.uniform([u32(T), u32(D), f32(scale), u32(features?.vocab ?? 0)]), tokens, ...weightBufs(table)];
  if (features) bufs.push(features.buf);
  ctx.run(spec, [...bufs, o.buf], grid1d(T * D));
  return o;
}

/** LayerNorm over the last axis, with weight and bias. */
export function layernorm(ctx: Ctx, x: Tensor, w: Tensor, b: Tensor, eps: number): Tensor {
  const D = lastDim(x);
  const rows = numel(x.shape) / D;
  const y = ctx.alloc(x.shape);
  ctx.run(layernormSpec(), [ctx.uniform([u32(rows), u32(D), f32(eps), u32(0)]), x.buf, w.buf, b.buf, y.buf], [Math.min(rows, MAX_X), Math.ceil(rows / MAX_X), 1]);
  return y;
}

/** GELU in place: the tanh approximation, or the exact erf form (BERT's) with `exact`. */
export function gelu(ctx: Ctx, x: Tensor, exact = false): void {
  act(ctx, x, exact ? "gelu" : "gelu_tanh");
}

/** An activation in place: GELU (erf or tanh), SiLU or ReLU. */
export function act(ctx: Ctx, x: Tensor, kind: Activation): void {
  const n = numel(x.shape);
  ctx.run(actSpec(kind), [ctx.uniform([u32(n), u32(0), u32(0), u32(0)]), x.buf], grid1d(n));
}

/**
 * Rows for a 1-D convolution as a matmul (see im2col1dSpec): x [Tin, C] → [Tout, row] with
 * Tout = floor((Tin + 2·pad − k) / stride) + 1 and row = k·C rounded up to a multiple of 32.
 */
export function im2col1d(ctx: Ctx, x: Tensor, k: number, stride: number, pad: number): Tensor {
  const C = lastDim(x);
  const tin = numel(x.shape) / C;
  const tout = Math.floor((tin + 2 * pad - k) / stride) + 1;
  const row = Math.ceil((k * C) / 32) * 32;
  const y = ctx.alloc([tout, row]);
  ctx.run(im2col1dSpec(), [ctx.uniform([u32(tin), u32(tout), u32(C), u32(k), u32(stride), u32(pad), u32(row), u32(0)]), x.buf, y.buf], grid1d(tout * row));
  return y;
}

/** k × k average pooling of a side × side token grid [side², C] → [(side / k)², C]. */
/** GroupNorm with one group per channel: each column of x [T, C] normalized over time, then scaled by w and shifted by b. */
export function channelNorm(ctx: Ctx, x: Tensor, w: Tensor, b: Tensor, eps: number): Tensor {
  const [T, C] = x.shape as [number, number];
  const y = ctx.alloc([T, C]);
  ctx.run(channelNormSpec(), [ctx.uniform([u32(T), u32(C), f32(eps), u32(0)]), x.buf, w.buf, b.buf, y.buf], [C, 1, 1]);
  return y;
}

/**
 * Grouped 1-D convolution of x [Tin, C] with w stored [k, C / groups, O] (see groupConv1dSpec):
 * `tout` output frames (default Tin + 2·pad − k + 1).
 */
export function groupConv1d(ctx: Ctx, x: Tensor, w: Tensor, bias: Tensor, o: { k: number; pad: number; groups: number; tout?: number }): Tensor {
  const [tin, C] = x.shape as [number, number];
  const O = w.shape[w.shape.length - 1]!;
  const tout = o.tout ?? tin + 2 * o.pad - o.k + 1;
  const y = ctx.alloc([tout, O]);
  ctx.run(groupConv1dSpec(), [ctx.uniform([u32(tin), u32(tout), u32(C), u32(O), u32(o.k), u32(o.pad), u32(o.groups), u32(0)]), x.buf, w.buf, bias.buf, y.buf], grid1d(tout * O));
  return y;
}

/** y = a·x, or y += a·x when accumulating (same element count). */
export function scaleAdd(ctx: Ctx, y: Tensor, x: Tensor, a: number, accumulate: boolean): void {
  const n = numel(x.shape);
  ctx.run(scaleAddSpec(), [ctx.uniform([u32(n), u32(accumulate ? 1 : 0), f32(a), u32(0)]), x.buf, y.buf], grid1d(n));
}

export function avgPool(ctx: Ctx, x: Tensor, side: number, k: number): Tensor {
  const C = lastDim(x);
  if (side % k || numel(x.shape) !== side * side * C) throw new Error(`avgPool: [${x.shape}] is not a ${side}×${side} grid poolable by ${k}`);
  const out = side / k;
  const y = ctx.alloc([out * out, C]);
  ctx.run(avgPoolSpec(), [ctx.uniform([u32(side), u32(k), u32(C), u32(0)]), x.buf, y.buf], grid1d(out * out * C));
  return y;
}

/** Row `step.lastRow` of x [T, D] → [1, D]. */
export function copyRow(ctx: Ctx, x: Tensor): Tensor {
  const D = lastDim(x);
  const o = ctx.alloc([1, D]);
  ctx.run(copyRowSpec(), [ctx.uniform([u32(D), u32(0), u32(0), u32(0)]), ctx.step, x.buf, o.buf], grid1d(D));
  return o;
}

/**
 * logits [1, V] → token id (u32 [1]), using the step's temperature, seed, top-k, top-p and repetition
 * penalty. `seen` is a bitmask of ceil(V / 32) u32 marking ids the penalty applies to. `allowed` is a
 * flag word followed by such a bitmask: when the flag is non-zero, only ids set in it can be chosen.
 */
export function sample(ctx: Ctx, logits: Tensor, out: Buf, seen: Buf, allowed: Buf): void {
  const V = lastDim(logits);
  ctx.run(sampleSpec(), [ctx.uniform([u32(V), u32(0), u32(0), u32(0)]), ctx.step, logits.buf, seen, allowed, out], [1, 1, 1]);
}

/**
 * logits [T, V] → T token ids, row t sampled as a single-row step at position pos0 + t would be
 * (same random numbers, so a sampled draft verifies exactly). Needs step.lastRow = 0. `allowed` holds
 * one flag-and-bitmask row per logits row, 1 + ceil(V / 32) words each.
 */
export function sampleRows(ctx: Ctx, logits: Tensor, out: Buf, seen: Buf, allowed: Buf): void {
  const V = lastDim(logits);
  ctx.run(sampleSpec(true), [ctx.uniform([u32(V), u32(0), u32(0), u32(0)]), ctx.step, logits.buf, seen, allowed, out], [1, rowsOf(logits), 1]);
}

/** GPU-side step advance: tokens[0] = sampled, history[step.n++] = sampled, mark seen, pos0 += lastRow + 1. */
export function advance(ctx: Ctx, sampled: Buf, tokens: Buf, history: Buf, seen: Buf): void {
  ctx.run(advanceSpec(), [ctx.step, sampled, tokens, history, seen], [1, 1, 1]);
}
