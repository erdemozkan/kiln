import { f32, lastDim, numel, u32, type Buf, type Ctx, type Grid, type Tensor } from "@kiln/core";
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
  copyRowSpec,
  embedSpec,
  kvWriteSpec,
  matmulRowsPerGroup,
  matmulSpec,
  rmsnormSpec,
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
  if (w.dtype === "q4") {
    if (!w.scales) throw new Error("q4 tensor is missing its scales plane");
    return [w.buf, w.scales];
  }
  return [w.buf];
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

let subgroupsDefault = false;
/** Use subgroup reductions in matmuls from now on, where the device supports them. */
export function setSubgroups(on: boolean): void {
  subgroupsDefault = on;
}

/** Tuned lanes per row, keyed by `${wdtype}/mt${mt}/${N}x${K}`; filled by the autotuner. */
const tunedLanes = new Map<string, MatmulLanes>();
export const lanesKey = (wdtype: "f32" | "q4", mt: 1 | 8, N: number, K: number): string => `${wdtype}/mt${mt}/${N}x${K}`;
export function setMatmulLanes(key: string, lanes: MatmulLanes): void {
  tunedLanes.set(key, lanes);
}
export function matmulLanes(): ReadonlyMap<string, MatmulLanes> {
  return tunedLanes;
}

/** x [M, K] · W[N, K]ᵀ → [M, N], with optional fused RMSNorm, bias, residual add or SiLU-gating. */
export function linear(ctx: Ctx, x: Tensor, w: Tensor, opts: LinearOptions | Tensor = {}): Tensor {
  const o: LinearOptions = "buf" in opts ? { bias: opts } : opts;
  const K = lastDim(x);
  const M = rowsOf(x);
  const [N, wk] = w.shape as [number, number];
  if (wk !== K) throw new Error(`linear: input has K=${K} but weight is [${w.shape}]`);
  if (K % 32 !== 0) throw new Error(`linear: K must be a multiple of 32 (K=${K})`);
  if (o.silu && (N % 2 !== 0 || o.residual)) throw new Error("linear: silu needs an even N and no residual");
  const outN = o.silu ? N / 2 : N;
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
  const wdtype = w.dtype === "q4" ? "q4" : "f32";
  const epi: MatmulEpilogue = o.silu ? "silu" : o.residual ? "add" : "store";
  const lanes = o.lanes ?? tunedLanes.get(lanesKey(wdtype, mt, N, K)) ?? 8;
  const f16 = o.f16 ?? f16Default;
  if (f16 && !ctx.backend.limits.features.has("shader-f16")) throw new Error("linear: f16 math needs the shader-f16 feature");
  // The butterfly needs each row's lanes inside one subgroup.
  const sg = (o.subgroups ?? subgroupsDefault) && lanes <= (ctx.backend.limits.subgroupMinSize ?? 0);
  const spec = matmulSpec({ wdtype, mt, lanes, norm: !!o.norm, bias: !!o.bias, epi, f16, sg });
  const groups = Math.ceil(outN / matmulRowsPerGroup({ lanes, epi }));
  const grid: Grid = [Math.min(groups, MM_SLICE), Math.ceil(M / mt), Math.ceil(groups / MM_SLICE)];
  const bufs = [ctx.uniform([u32(M), u32(N), u32(K), f32(o.norm?.eps ?? 0)]), x.buf, ...weightBufs(w)];
  if (o.norm) bufs.push(o.norm.w.buf);
  if (o.bias) bufs.push(o.bias.buf);
  bufs.push(y.buf);
  ctx.run(spec, bufs, grid, `matmul/${wdtype}/mt${mt} ${N}x${K}${o.norm ? "+norm" : ""}${o.silu ? "+silu" : ""}${o.residual ? "+add" : ""}`);
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
  const params = ctx.uniform([u32(s.T), u32(s.H), u32(s.Hkv), u32(s.D), u32(s.maxSeq), u32(0), u32(0), u32(0)]);
  ctx.run(ropeKvSpec(interleaved, s.kv), [params, ctx.step, invFreq.buf, qkv.buf, q.buf, kc, vc], grid1d((s.T * W) / 2));
  return q;
}

export function kvWrite(ctx: Ctx, k: Tensor, v: Tensor, kc: Buf, vc: Buf, T: number, maxSeq: number, kv: KvDtype = "f32"): void {
  const row = numel(k.shape) / T;
  if (kv === "f16" && row % 2 !== 0) throw new Error(`kvWrite: f16 cache needs an even row (row=${row})`);
  const threads = kv === "f16" ? (T * row) / 2 : T * row;
  ctx.run(kvWriteSpec(kv), [ctx.uniform([u32(T), u32(row), u32(maxSeq), u32(0)]), ctx.step, k.buf, v.buf, kc, vc], grid1d(threads));
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
}

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
  const params = ctx.uniform([u32(s.T), u32(s.H), u32(s.Hkv), u32(s.D), u32(s.maxSeq), f32(1 / Math.sqrt(s.D)), u32(splits), u32(0)]);
  if (s.T === 1) {
    const ml = ctx.alloc([s.H, splits, 2]);
    const po = ctx.alloc([s.H, splits, s.D]);
    ctx.run(attentionSplitSpec(s.kv), [params, ctx.step, q.buf, kc, vc, ml.buf, po.buf], [s.H, 1, splits]);
    ctx.run(attentionCombineSpec(), [params, ctx.step, ml.buf, po.buf, out.buf], [s.H, 1, 1]);
    return out;
  }
  if (prefillAttention === "tiled") {
    const { R } = attentionTile(s.D);
    ctx.run(attentionTiledSpec(s.kv ?? "f32", s.D), [params, ctx.step, q.buf, kc, vc, out.buf], [s.Hkv, Math.ceil((s.T * (s.H / s.Hkv)) / R), 1]);
    return out;
  }
  ctx.run(attentionSpec(s.kv), [params, ctx.step, q.buf, kc, vc, out.buf], [s.H, s.T, 1]);
  return out;
}

export function siluMul(ctx: Ctx, gate: Tensor, up: Tensor): Tensor {
  const n = numel(gate.shape);
  const o = ctx.alloc(gate.shape);
  ctx.run(siluMulSpec(), [ctx.uniform([u32(n), u32(0), u32(0), u32(0)]), gate.buf, up.buf, o.buf], grid1d(n));
  return o;
}

/** a += b. If b has fewer elements than a it is broadcast over rows (bias). */
export function addInPlace(ctx: Ctx, a: Tensor, b: Tensor): void {
  const n = numel(a.shape);
  ctx.run(addSpec(), [ctx.uniform([u32(n), u32(numel(b.shape)), u32(0), u32(0)]), a.buf, b.buf], grid1d(n));
}

export function embed(ctx: Ctx, table: Tensor, tokens: Buf, T: number): Tensor {
  const D = lastDim(table);
  const o = ctx.alloc([T, D]);
  const spec = embedSpec(table.dtype === "q4" ? "q4" : "f32");
  ctx.run(spec, [ctx.uniform([u32(T), u32(D), u32(0), u32(0)]), tokens, ...weightBufs(table), o.buf], grid1d(T * D));
  return o;
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
