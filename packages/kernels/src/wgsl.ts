import { STEP_WGSL, type BindingKind, type KernelSpec } from "@kiln/core";

/**
 * Baseline WGSL kernels. They favour clarity over speed: correct first, then specialised
 * (tiling, subgroups, f16, dp4a) behind the same specs once the autotuner lands.
 */

/** Threads per workgroup for 1-D elementwise kernels. */
export const EW = 64;
/** Max workgroups along x before a 1-D launch spills into y. */
export const MAX_X = 65535;
/** Rows per z-slice for matmul launches; keeps x under the per-dimension limit for large vocabularies. */
export const MM_SLICE = 32768;

/** Flat index for 1-D launches built with `grid1d`. */
const FLAT = /* wgsl */ `let idx = gid.x + gid.y * nw.x * ${EW}u;`;
const EW_ARGS = /* wgsl */ `@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>`;

/** KV-cache storage: f32, or f16 pairs packed into u32 with pack2x16float (core WGSL, no feature needed). */
export type KvDtype = "f32" | "f16";

/** WGSL for reading the cache: kc as 4-element groups, vc element-wise. */
function kvRead(kv: KvDtype): { kDecl: string; vDecl: string; k4: (i: string) => string; v1: (i: string) => string } {
  return kv === "f16"
    ? {
        kDecl: "array<vec2<u32>>",
        vDecl: "array<u32>",
        k4: (i) => `kv4(kc[${i}])`,
        v1: (i) => `unpack2x16float(vc[(${i}) / 2u])[(${i}) % 2u]`,
      }
    : { kDecl: "array<vec4<f32>>", vDecl: "array<f32>", k4: (i) => `kc[${i}]`, v1: (i) => `vc[${i}]` };
}
const KV4_FN = /* wgsl */ `fn kv4(w: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }`;

const cache = new Map<string, KernelSpec>();
function spec(key: string, bindings: KernelSpec["bindings"], code: () => string): KernelSpec {
  let s = cache.get(key);
  if (!s) {
    s = { key, bindings, code: code() };
    cache.set(key, s);
  }
  return s;
}

/** Elements of K staged in workgroup memory per step. */
export const MM_KC = 256;
/** Lanes per weight row a matmul variant may use; rows per workgroup = 256 / lanes. */
export type MatmulLanes = 8 | 16 | 32;
/** What the matmul does with each output after the dot product. */
export type MatmulEpilogue = "store" | "add" | "silu";

export interface MatmulVariant {
  wdtype: "f32" | "q4";
  /** Tokens per workgroup: 1 for decode, 8 for prefill. */
  mt: 1 | 8;
  lanes: MatmulLanes;
  /** RMSNorm x on the fly (each workgroup recomputes the row norms; K is small). */
  norm: boolean;
  bias: boolean;
  /**
   * store: y = acc. add: y += acc (residual). silu: the weight is [gate; up] stacked along N and
   * y[:, i] = silu(gate_i) * up_i, so a workgroup owns matching gate and up rows.
   */
  epi: MatmulEpilogue;
  /** f16 x tile and 4-wide dot products (needs shader-f16); accumulation stays f32. */
  f16?: boolean;
  /**
   * Reduce with subgroup operations instead of workgroup-memory trees (needs `subgroups` and a
   * subgroup size of at least `lanes`): a shuffle butterfly across each row's lanes, and subgroupAdd
   * for the norm prologue. Assumes subgroups are runs of consecutive invocation indices.
   */
  sg?: boolean;
}

/** Rows of the output a workgroup produces for this variant. */
export const matmulRowsPerGroup = (v: Pick<MatmulVariant, "lanes" | "epi">): number => 256 / v.lanes / (v.epi === "silu" ? 2 : 1);

/**
 * y[m, n] = sum_k x[m, k] * W[n, k], for K a multiple of 32, with an optional RMSNorm prologue and a
 * fused epilogue (see MatmulVariant).
 *
 * A 256-thread workgroup owns 256 / lanes weight rows. K is walked in chunks of MM_KC: an [mt, MM_KC]
 * tile of x is staged in workgroup memory once and shared by all rows. Within a 32-element block lane l
 * handles vec4 slot l % 8, and lanes l / 8 = 0, 1, … take every (lanes / 8)-th block, so consecutive lanes
 * read consecutive vec4s of the tile (no bank conflicts) and consecutive bytes of the weights (coalesced).
 * For q4 a block is one scale plus 4 u32 of nibbles; slot s decodes half (s % 2) of word s / 2.
 * More lanes per row means more workgroups for small N, which matters on GPUs with many cores.
 */
export function matmulSpec(v: MatmulVariant): KernelSpec {
  const q4 = v.wdtype === "q4";
  const kc4 = MM_KC / 4;
  const mt = v.mt;
  const rows = 256 / v.lanes;
  const silu = v.epi === "silu";
  const h = !!v.f16;
  const sg = !!v.sg;
  const key = `matmul/${v.wdtype}/mt${mt}/l${v.lanes}${v.norm ? "/norm" : ""}${v.bias ? "/bias" : ""}/${v.epi}${h ? "/f16" : ""}${sg ? "/sg" : ""}`;
  const vt = h ? "vec4<f16>" : "vec4<f32>";
  const bindings: BindingKind[] = q4 ? ["uniform", "read", "read", "read"] : ["uniform", "read", "read"];
  let b = bindings.length;
  const decl: string[] = [];
  if (v.norm) {
    decl.push(`@group(0) @binding(${b}) var<storage, read> nw: array<vec4<f32>>;`);
    bindings.push("read");
    b++;
  }
  if (v.bias) {
    decl.push(`@group(0) @binding(${b}) var<storage, read> bias: array<f32>;`);
    bindings.push("read");
    b++;
  }
  decl.push(`@group(0) @binding(${b}) var<storage, read_write> y: array<f32>;`);
  bindings.push("rw");
  const withBias = (row: string, val: string) => (v.bias ? `(${val} + bias[${row}])` : val);
  const store = silu
    ? `if (r < HALF && i < p.N / 2u) {
      let g = ${withBias("n", `red[lane * 256u + r * LANES]`)};
      let u = ${withBias("n + p.N / 2u", `red[lane * 256u + (r + HALF) * LANES]`)};
      y[(m0 + lane) * (p.N / 2u) + i] = g / (1.0 + exp(-g)) * u;
    }`
    : `if (n < p.N) {
      let val = ${withBias("n", `red[lane * 256u + r * LANES]`)};
      y[(m0 + lane) * p.N + n] ${v.epi === "add" ? "+=" : "="} val;
    }`;
  return spec(key, bindings, () => /* wgsl */ `
${h ? "enable f16;" : ""}
${sg ? "enable subgroups;" : ""}
struct P { M: u32, N: u32, K: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
${q4
  ? `@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> ws: array<f32>;`
  : `@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;`}
${decl.join("\n")}
const MT = ${mt}u;
const KC4 = ${kc4}u;
const LANES = ${v.lanes}u;
const PHASES = ${v.lanes / 8}u;
const HALF = ${rows / 2}u;
var<workgroup> xt: array<${vt}, ${mt * kc4}>;
var<workgroup> red: array<f32, ${256 * mt}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32${sg ? ", @builtin(subgroup_invocation_id) sid: u32, @builtin(subgroup_size) sgs: u32" : ""}) {
  let lane = li % LANES;
  let slot = lane % 8u;
  let phase = lane / 8u;
  let r = li / LANES;
  let grp = wid.x + wid.z * ${MM_SLICE}u;
${silu
  ? `  let i = grp * HALF + r % HALF;
  let n = select(i, p.N / 2u + i, r >= HALF);
  let live = i < p.N / 2u;`
  : `  let n = grp * ${rows}u + r;
  let live = n < p.N;`}
  let m0 = wid.y * MT;
  let blocks = p.K / 32u;
  let xs4 = p.K / 4u;
  let last = arrayLength(&x) - 1u;
  var acc: array<f32, ${mt}>;
${q4 ? `  let shift = (slot % 2u) * 16u;` : ""}
${v.norm && sg
  ? `
  // RMSNorm prologue: sum of squares of each of the MT rows, per subgroup, then across subgroups.
  for (var t = 0u; t < MT; t++) {
    var ss = 0.0;
    let m = m0 + t;
    if (m < p.M) {
      for (var c = li; c < xs4; c += 256u) { let xv = x[m * xs4 + c]; ss += dot(xv, xv); }
    }
    ss = subgroupAdd(ss);
    if (sid == 0u) { red[t * 256u + li / sgs] = ss; }
  }
  workgroupBarrier();
  if (li < MT) {
    var total = 0.0;
    for (var g = 0u; g < 256u / sgs; g++) { total += red[li * 256u + g]; }
    red[li * 256u] = total;
  }
  workgroupBarrier();
  // red[t * 256] now holds row t's sum of squares; it stays untouched until after the K loop.`
  : v.norm
  ? `
  // RMSNorm prologue: sum of squares of each of the MT rows, reduced across the workgroup.
  for (var t = 0u; t < MT; t++) {
    var ss = 0.0;
    let m = m0 + t;
    if (m < p.M) {
      for (var c = li; c < xs4; c += 256u) { let xv = x[m * xs4 + c]; ss += dot(xv, xv); }
    }
    red[t * 256u + li] = ss;
  }
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      for (var t = 0u; t < MT; t++) { red[t * 256u + li] += red[t * 256u + li + s]; }
    }
    workgroupBarrier();
  }
  // red[t * 256] now holds row t's sum of squares; it stays untouched until after the K loop.`
  : ""}

  for (var k4 = 0u; k4 < xs4; k4 += KC4) {
    for (var e = li; e < MT * KC4; e += 256u) {
      let m = m0 + e / KC4;
      let c = k4 + e % KC4;
      let xv = select(vec4<f32>(0.0), x[min(m * xs4 + c, last)], m < p.M && c < xs4);
      xt[e] = ${vt}(${v.norm ? `xv * inverseSqrt(red[(e / KC4) * 256u] / f32(p.K) + p.eps) * nw[min(c, xs4 - 1u)]` : "xv"});
    }
    workgroupBarrier();
    if (live) {
      let nb = min(KC4, xs4 - k4) / 8u;
      let b0 = n * blocks + k4 / 8u;
      for (var bi = phase; bi < nb; bi += PHASES) {
${q4
  ? `        let word = wq[(b0 + bi) * 4u + slot / 2u] >> shift;
        let wv = (${vt}(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(15u)) - ${h ? "8.0h" : "8.0"}) * ${h ? `f16(ws[b0 + bi])` : `ws[b0 + bi]`};`
  : `        let wv = ${h ? `vec4<f16>(w[(b0 + bi) * 8u + slot])` : `w[(b0 + bi) * 8u + slot]`};`}
        for (var t = 0u; t < MT; t++) { acc[t] += ${h ? "f32(" : ""}dot(xt[t * KC4 + bi * 8u + slot], wv)${h ? ")" : ""}; }
      }
    }
    workgroupBarrier();
  }

${sg
  ? `  // Butterfly across the row's lanes, which sit in one subgroup; lane 0 then holds each total.
  for (var t = 0u; t < MT; t++) {
    var v = acc[t];
    for (var s = LANES / 2u; s > 0u; s >>= 1u) { v += subgroupShuffleXor(v, s); }
    if (lane == 0u) { red[t * 256u + li] = v; }
  }
  workgroupBarrier();`
  : `  for (var t = 0u; t < MT; t++) { red[t * 256u + li] = acc[t]; }
  workgroupBarrier();
  for (var s = LANES / 2u; s > 0u; s >>= 1u) {
    if (lane < s) {
      for (var t = 0u; t < MT; t++) { red[t * 256u + li] += red[t * 256u + li + s]; }
    }
    workgroupBarrier();
  }`}
  if (lane < MT && m0 + lane < p.M) {
    ${store}
  }
}`);
}

/** y = x * rsqrt(mean(x^2) + eps) * w, one workgroup per row. */
export function rmsnormSpec(): KernelSpec {
  return spec("rmsnorm", ["uniform", "read", "read", "rw"], () => /* wgsl */ `
struct P { R: u32, D: u32, eps: f32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let o = wid.x * p.D;
  var ss = 0.0;
  for (var i = li; i < p.D; i += 256u) { let v = x[o + i]; ss += v * v; }
  red[li] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { y[o + i] = x[o + i] * inv * w[i]; }
}`);
}

/** In-place rotary embedding on [T, H, D]. Rotate-half (HF) or interleaved pairs (GGUF llama). */
export function ropeSpec(interleaved: boolean): KernelSpec {
  return spec(`rope/${interleaved ? "interleaved" : "half"}`, ["uniform", "read", "read", "rw"], () => /* wgsl */ `
${STEP_WGSL}
struct P { T: u32, H: u32, D: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> invFreq: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  let hd = p.D / 2u;
  if (idx >= p.T * p.H * hd) { return; }
  let i = idx % hd;
  let row = idx / hd;
  let t = row / p.H;
  let a = f32(st.pos0 + t) * invFreq[i];
  let c = cos(a);
  let s = sin(a);
  let base = row * p.D;
  ${interleaved ? "let i0 = base + 2u * i; let i1 = i0 + 1u;" : "let i0 = base + i; let i1 = base + i + hd;"}
  let x0 = x[i0];
  let x1 = x[i1];
  x[i0] = x0 * c - x1 * s;
  x[i1] = x0 * s + x1 * c;
}`);
}

/** Writes k/v rows [T, rowSize] into the caches at positions pos0..pos0+T. */
export function kvWriteSpec(kv: KvDtype = "f32"): KernelSpec {
  const h = kv === "f16";
  return spec(`kv_write/${kv}`, ["uniform", "read", "read", "read", "rw", "rw"], () => /* wgsl */ `
${STEP_WGSL}
struct P { T: u32, row: u32, maxSeq: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> kc: array<${h ? "u32" : "f32"}>;
@group(0) @binding(5) var<storage, read_write> vc: array<${h ? "u32" : "f32"}>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
${h
  ? `  // One thread per element pair.
  let e = idx * 2u;
  if (e >= p.T * p.row) { return; }
  let t = e / p.row;
  if (st.pos0 + t >= p.maxSeq) { return; }
  let dst = ((st.pos0 + t) * p.row + e % p.row) / 2u;
  kc[dst] = pack2x16float(vec2<f32>(k[e], k[e + 1u]));
  vc[dst] = pack2x16float(vec2<f32>(v[e], v[e + 1u]));`
  : `  if (idx >= p.T * p.row) { return; }
  let t = idx / p.row;
  if (st.pos0 + t >= p.maxSeq) { return; }
  let dst = (st.pos0 + t) * p.row + idx % p.row;
  kc[dst] = k[idx];
  vc[dst] = v[idx];`}
}`);
}

/** Threads per attention workgroup; head dims up to this size are supported. */
export const ATTN_WG = 256;

/**
 * Causal attention with GQA: one 256-thread workgroup per (head, query row).
 * q is cached in workgroup memory. Keys are taken ATTN_WG at a time with an online softmax: each block's
 * scores stay in workgroup memory, the running max and exp-sum rescale the partial output, so no scores
 * reach global memory. The weighted sum splits each block's keys across ATTN_WG / D thread groups and
 * reduces their partial outputs at the end, so every thread works even when D is small.
 */
export function attentionSpec(kv: KvDtype = "f32"): KernelSpec {
  const r = kvRead(kv);
  return spec(`attention/online/${kv}`, ["uniform", "read", "read", "read", "read", "rw"], () => /* wgsl */ `
${STEP_WGSL}
${kv === "f16" ? KV4_FN : ""}
struct P { T: u32, H: u32, Hkv: u32, D: u32, maxSeq: u32, scale: f32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: ${r.kDecl};
@group(0) @binding(4) var<storage, read> vc: ${r.vDecl};
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
const WG = ${ATTN_WG}u;
var<workgroup> red: array<f32, ${ATTN_WG}>;
var<workgroup> ss: array<f32, ${ATTN_WG}>;
var<workgroup> qs: array<vec4<f32>, ${ATTN_WG / 4}>;

@compute @workgroup_size(${ATTN_WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wid.x;
  let t = wid.y;
  let kvh = h / (p.H / p.Hkv);
  let n = min(st.pos0 + t + 1u, p.maxSeq);
  let qo = (t * p.H + h) * p.D;
  let stride = p.Hkv * p.D;
  let ko = kvh * p.D;
  let d4 = p.D / 4u;
  if (li < d4) {
    let qb = qo + li * 4u;
    qs[li] = vec4<f32>(q[qb], q[qb + 1u], q[qb + 2u], q[qb + 3u]) * p.scale;
  }
  workgroupBarrier();

  let lane = li % 8u;
  let key = li / 8u;
  let groups = WG / p.D;
  let g = li / p.D;
  let i = li % p.D;
  var m = -3.0e38;
  var l = 0.0;
  var acc = 0.0;
  for (var b0 = 0u; b0 < n; b0 += WG) {
    let bn = min(WG, n - b0);
    // Scores: 8 lanes per key (vec4 slots lane, lane + 8, ...), 32 keys per pass, coalesced cache reads.
    for (var k0 = 0u; k0 < bn; k0 += WG / 8u) {
      let jj = k0 + key;
      var d = 0.0;
      if (jj < bn) {
        let kb = ((b0 + jj) * stride + ko) / 4u;
        for (var c = lane; c < d4; c += 8u) { d += dot(qs[c], ${r.k4("kb + c")}); }
      }
      red[li] = d;
      workgroupBarrier();
      if (lane == 0u && jj < bn) {
        var total = 0.0;
        for (var x = 0u; x < 8u; x++) { total += red[li + x]; }
        ss[jj] = total;
      }
      workgroupBarrier();
    }

    let sc = select(-3.0e38, ss[li], li < bn);
    red[li] = sc;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
      if (li < s) { red[li] = max(red[li], red[li + s]); }
      workgroupBarrier();
    }
    let mNew = max(m, red[0]);
    workgroupBarrier();
    let e = select(0.0, exp(sc - mNew), li < bn);
    ss[li] = e;
    red[li] = e;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
      if (li < s) { red[li] += red[li + s]; }
      workgroupBarrier();
    }
    let corr = exp(m - mNew);
    l = l * corr + red[0];
    m = mNew;
    acc *= corr;
    if (g < groups) {
      for (var jj = g; jj < bn; jj += groups) { acc += ss[jj] * ${r.v1("(b0 + jj) * stride + ko + i")}; }
    }
    workgroupBarrier();
  }

  red[li] = acc;
  workgroupBarrier();
  if (li < p.D) {
    var total = 0.0;
    for (var k = 0u; k < groups; k++) { total += red[k * p.D + li]; }
    o[qo + li] = total / l;
  }
}`);
}

/** Query rows and keys per workgroup tile in tiled prefill attention, sized to fit workgroup memory. */
export function attentionTile(D: number): { R: number; BK: number } {
  const bytes = (R: number, BK: number) => 16 * R * (D / 4) + 16 * BK * (D / 4 + 1) + 4 * R * BK;
  const BK = Math.min(64, Math.floor(2048 / D));
  let R = Math.min(32, Math.floor(1024 / D));
  while (R > 1 && bytes(R, BK) > 16384 - 256) R = Math.floor(R / 2);
  return { R, BK };
}

/**
 * Tiled causal prefill attention with GQA: one 256-thread workgroup per (kv head, tile of R query rows),
 * where the rows are (token, head) pairs of every query head that shares the kv head. Each block of BK
 * keys is staged once in workgroup memory and used by all R rows, then its values likewise, so the cache
 * is read once per tile instead of once per (head, token).
 *
 * Scores: TPR threads per row, each scoring every TPR-th key of the block. Values: thread (row, d4) owns
 * four output columns as a vec4 and folds the online softmax into its pass over the block (block max,
 * rescale, exp, exp-sum), so the row statistics live in registers and need no extra barrier.
 */
export function attentionTiledSpec(kv: KvDtype, D: number): KernelSpec {
  const { R, BK } = attentionTile(D);
  const D4 = D / 4;
  const KP = D4 + 1; // padded row stride (vec4) of the key/value tile, so rows start in different banks
  const TPR = Math.floor(256 / R);
  const load =
    kv === "f16"
      ? { kDecl: "array<vec2<u32>>", vDecl: "array<vec2<u32>>", k: "kv4(kc[i])", v: "kv4(vc[i])" }
      : { kDecl: "array<vec4<f32>>", vDecl: "array<vec4<f32>>", k: "kc[i]", v: "vc[i]" };
  return spec(`attention/tiled/${kv}/D${D}`, ["uniform", "read", "read", "read", "read", "rw"], () => /* wgsl */ `
${STEP_WGSL}
${kv === "f16" ? KV4_FN : ""}
struct P { T: u32, H: u32, Hkv: u32, D: u32, maxSeq: u32, scale: f32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: ${load.kDecl};
@group(0) @binding(4) var<storage, read> vc: ${load.vDecl};
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
const D = ${D}u;
const D4 = ${D4}u;
const KP = ${KP}u;
const R = ${R}u;
const BK = ${BK}u;
const TPR = ${TPR}u;
var<workgroup> qs: array<vec4<f32>, ${R * D4}>;
var<workgroup> tile: array<vec4<f32>, ${BK * KP}>;
var<workgroup> sc: array<f32, ${R * BK}>;

fn K4(i: u32) -> vec4<f32> { return ${load.k}; }
fn V4(i: u32) -> vec4<f32> { return ${load.v}; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let kvh = wid.x;
  let G = p.H / p.Hkv;
  let rows = p.T * G;
  let r0 = wid.y * R;
  let stride4 = p.Hkv * D4;
  let ko4 = kvh * D4;
  for (var e = li; e < R * D4; e += 256u) {
    let rho = r0 + e / D4;
    var v = vec4<f32>(0.0);
    if (rho < rows) {
      let qb = ((rho / G) * p.H + kvh * G + rho % G) * D + (e % D4) * 4u;
      v = vec4<f32>(q[qb], q[qb + 1u], q[qb + 2u], q[qb + 3u]) * p.scale;
    }
    qs[e] = v;
  }
  // Keys visible to the tile's last row; earlier rows mask the tail of the last block.
  let nMax = min(st.pos0 + (min(r0 + R, rows) - 1u) / G + 1u, p.maxSeq);
  // Scoring role: row sr, keys sl, sl + TPR, ...
  let sr = li / TPR;
  let sl = li % TPR;
  let sn = min(st.pos0 + (r0 + sr) / G + 1u, p.maxSeq);
  // Output role: row orow, columns 4 * od .. 4 * od + 3.
  let orow = li / D4;
  let od = li % D4;
  var m = -3.0e38;
  var l = 0.0;
  var acc = vec4<f32>(0.0);
  workgroupBarrier();

  for (var b0 = 0u; b0 < nMax; b0 += BK) {
    for (var e = li; e < BK * D4; e += 256u) {
      let j = b0 + e / D4;
      var v = vec4<f32>(0.0);
      if (j < nMax) { v = K4(j * stride4 + ko4 + e % D4); }
      tile[(e / D4) * KP + e % D4] = v;
    }
    workgroupBarrier();
    if (sr < R) {
      for (var jj = sl; jj < BK; jj += TPR) {
        var d = -3.0e38;
        if (b0 + jj < sn) {
          d = 0.0;
          for (var c = 0u; c < D4; c++) { d += dot(qs[sr * D4 + c], tile[jj * KP + c]); }
        }
        sc[sr * BK + jj] = d;
      }
    }
    workgroupBarrier();
    for (var e = li; e < BK * D4; e += 256u) {
      let j = b0 + e / D4;
      var v = vec4<f32>(0.0);
      if (j < nMax) { v = V4(j * stride4 + ko4 + e % D4); }
      tile[(e / D4) * KP + e % D4] = v;
    }
    workgroupBarrier();
    if (orow < R) {
      var mx = m;
      for (var jj = 0u; jj < BK; jj++) { mx = max(mx, sc[orow * BK + jj]); }
      let corr = exp(m - mx);
      m = mx;
      var bl = 0.0;
      var ba = vec4<f32>(0.0);
      for (var jj = 0u; jj < BK; jj++) {
        let e = exp(sc[orow * BK + jj] - mx);
        bl += e;
        ba += e * tile[jj * KP + od];
      }
      l = l * corr + bl;
      acc = acc * corr + ba;
    }
    workgroupBarrier();
  }

  let rho = r0 + orow;
  if (orow < R && rho < rows) {
    let ob = ((rho / G) * p.H + kvh * G + rho % G) * D + od * 4u;
    let v = acc / l;
    o[ob] = v.x;
    o[ob + 1u] = v.y;
    o[ob + 2u] = v.z;
    o[ob + 3u] = v.w;
  }
}`);
}

/** Keys per split in decode attention. */
export const ATTN_CHUNK = 64;

/**
 * Split-K ("flash-decoding") attention, pass 1. Decode has one query row, so one workgroup per head
 * leaves most of the GPU idle; instead the cache is cut into ATTN_CHUNK-key splits and grid z walks
 * them. Each workgroup writes its local max m, exp-sum l and unnormalised output o for its keys.
 * Scores stay in workgroup memory (no scratch buffer). Splits past the current length exit at once.
 */
export function attentionSplitSpec(kv: KvDtype = "f32"): KernelSpec {
  const r = kvRead(kv);
  return spec(`attention/split/${kv}`, ["uniform", "read", "read", "read", "read", "rw", "rw"], () => /* wgsl */ `
${STEP_WGSL}
${kv === "f16" ? KV4_FN : ""}
struct P { T: u32, H: u32, Hkv: u32, D: u32, maxSeq: u32, scale: f32, S: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: ${r.kDecl};
@group(0) @binding(4) var<storage, read> vc: ${r.vDecl};
@group(0) @binding(5) var<storage, read_write> ml: array<f32>;
@group(0) @binding(6) var<storage, read_write> po: array<f32>;
const CH = ${ATTN_CHUNK}u;
var<workgroup> qs: array<vec4<f32>, ${ATTN_WG / 4}>;
var<workgroup> part: array<f32, 64>;
var<workgroup> sc: array<f32, ${ATTN_CHUNK}>;
var<workgroup> red: array<f32, 64>;
${kv === "f16" ? "var<workgroup> vpart: array<vec2<f32>, 64>;" : ""}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wid.x;
  let t = wid.y;
  let split = wid.z;
  let kvh = h / (p.H / p.Hkv);
  let n = min(st.pos0 + t + 1u, p.maxSeq);
  let j0 = split * CH;
  // Splits past the current length do nothing; combine only reads live splits.
  if (j0 >= n) { return; }
  let cnt = min(CH, n - j0);
  let qo = (t * p.H + h) * p.D;
  let stride = p.Hkv * p.D;
  let ko = kvh * p.D;
  let d4 = p.D / 4u;
  for (var c = li; c < d4; c += 64u) {
    let qb = qo + c * 4u;
    qs[c] = vec4<f32>(q[qb], q[qb + 1u], q[qb + 2u], q[qb + 3u]) * p.scale;
  }
  workgroupBarrier();

  // Scores: 8 lanes per key, 8 keys per pass.
  let lane = li % 8u;
  let key = li / 8u;
  for (var k0 = 0u; k0 < CH; k0 += 8u) {
    let j = k0 + key;
    var d = 0.0;
    if (j < cnt) {
      let kb = ((j0 + j) * stride + ko) / 4u;
      for (var c = lane; c < d4; c += 8u) { d += dot(qs[c], ${r.k4("kb + c")}); }
    }
    part[li] = d;
    workgroupBarrier();
    if (lane == 0u) {
      var total = 0.0;
      for (var l = 0u; l < 8u; l++) { total += part[li + l]; }
      sc[j] = total;
    }
    workgroupBarrier();
  }

  red[li] = select(-3.0e38, sc[li], li < cnt);
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();
  let e = select(0.0, exp(sc[li] - m), li < cnt);
  sc[li] = e;
  red[li] = e;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let row = (t * p.H + h) * p.S + split;
  if (li == 0u) {
    ml[row * 2u] = m;
    ml[row * 2u + 1u] = red[0];
  }
${kv === "f16"
  ? `  // V: one packed pair per thread; when D / 2 < 64, groups of threads split the keys and are summed.
  let pairs = p.D / 2u;
  let groups = max(1u, 64u / pairs);
  let g = select(0u, li / pairs, pairs < 64u);
  let vs = stride / 2u;
  let vo = (j0 * stride + ko) / 2u;
  for (var pi = li % pairs; pi < pairs; pi += 64u) {
    var acc = vec2<f32>(0.0);
    if (g < groups) {
      for (var j = g; j < cnt; j += groups) { acc += sc[j] * unpack2x16float(vc[vo + j * vs + pi]); }
    }
    if (pairs < 64u) {
      vpart[li] = acc;
    } else {
      po[row * p.D + 2u * pi] = acc.x;
      po[row * p.D + 2u * pi + 1u] = acc.y;
    }
  }
  workgroupBarrier();
  if (pairs < 64u && li < pairs) {
    var total = vec2<f32>(0.0);
    for (var k = 0u; k < groups; k++) { total += vpart[k * pairs + li]; }
    po[row * p.D + 2u * li] = total.x;
    po[row * p.D + 2u * li + 1u] = total.y;
  }`
  : `  for (var i = li; i < p.D; i += 64u) {
    var acc = 0.0;
    for (var j = 0u; j < cnt; j++) { acc += sc[j] * ${r.v1("(j0 + j) * stride + ko + i")}; }
    po[row * p.D + i] = acc;
  }`}
}`);
}

/** Split-K attention, pass 2: rescales every split to the global max and normalises. */
export function attentionCombineSpec(): KernelSpec {
  return spec("attention/combine", ["uniform", "read", "read", "read", "rw"], () => /* wgsl */ `
${STEP_WGSL}
struct P { T: u32, H: u32, Hkv: u32, D: u32, maxSeq: u32, scale: f32, S: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> ml: array<f32>;
@group(0) @binding(3) var<storage, read> po: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let base = (wid.y * p.H + wid.x) * p.S;
  let live = (min(st.pos0 + wid.y + 1u, p.maxSeq) + ${ATTN_CHUNK - 1}u) / ${ATTN_CHUNK}u;
  var mx = -3.0e38;
  for (var s = 0u; s < live; s++) { mx = max(mx, ml[(base + s) * 2u]); }
  var den = 0.0;
  for (var s = 0u; s < live; s++) { den += exp(ml[(base + s) * 2u] - mx) * ml[(base + s) * 2u + 1u]; }
  let out = (wid.y * p.H + wid.x) * p.D;
  for (var i = li; i < p.D; i += 64u) {
    var num = 0.0;
    for (var s = 0u; s < live; s++) { num += exp(ml[(base + s) * 2u] - mx) * po[(base + s) * p.D + i]; }
    o[out + i] = num / den;
  }
}`);
}

/** out = silu(gate) * up. */
export function siluMulSpec(): KernelSpec {
  return spec("silu_mul", ["uniform", "read", "read", "rw"], () => /* wgsl */ `
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.n) { return; }
  let a = g[idx];
  o[idx] = a / (1.0 + exp(-a)) * u[idx];
}`);
}

/** a += b, with b broadcast over rows when `cols` < n (bias add). */
export function addSpec(): KernelSpec {
  return spec("add", ["uniform", "rw", "read"], () => /* wgsl */ `
struct P { n: u32, cols: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.n) { return; }
  a[idx] += b[idx % p.cols];
}`);
}

/** Gathers embedding rows for token ids. */
export function embedSpec(wdtype: "f32" | "q4"): KernelSpec {
  const q4 = wdtype === "q4";
  return spec(`embed/${wdtype}`, q4 ? ["uniform", "read", "read", "read", "rw"] : ["uniform", "read", "read", "rw"], () => /* wgsl */ `
struct P { T: u32, D: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
${q4
  ? `@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> ws: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;`
  : `@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;`}

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.T * p.D) { return; }
  let e = tokens[idx / p.D] * p.D + idx % p.D;
${q4
  ? `  let q = f32((wq[e / 8u] >> (4u * (e % 8u))) & 15u) - 8.0;
  o[idx] = q * ws[e / 32u];`
  : `  o[idx] = w[e];`}
}`);
}

/** Copies row `st.lastRow` of x [T, D] into out [1, D]. */
export function copyRowSpec(): KernelSpec {
  return spec("copy_row", ["uniform", "read", "read", "rw"], () => /* wgsl */ `
${STEP_WGSL}
struct P { D: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.D) { return; }
  o[idx] = x[st.lastRow * p.D + idx];
}`);
}

/**
 * Samples one token on the GPU so only 4 bytes come back per step. One workgroup, no sort:
 * - tokens outside the `allowed` bitmask are never chosen, when its word 0 is non-zero (grammars);
 * - repetition penalty from the `seen` bitmask (prompt + generated ids), as in HF;
 * - temperature 0 → argmax;
 * - otherwise top-k / top-p cut at a logit threshold found by bisection (count ≥ k, kept mass ≥ p),
 *   then Gumbel-max over the kept tokens, which is an exact sample of the truncated softmax.
 * With `rows`, workgroup y samples logits row y into out[y] under its own `allowed` row (flag word and
 * bitmask, rows back to back), drawing the random numbers a single-row step at position pos0 + y
 * would (speculative verification).
 */
export function sampleSpec(rows = false): KernelSpec {
  return spec(rows ? "sample/rows" : "sample", ["uniform", "read", "read", "read", "read", "rw"], () => /* wgsl */ `
${STEP_WGSL}
struct P { V: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> logits: array<f32>;
@group(0) @binding(3) var<storage, read> seen: array<u32>;
@group(0) @binding(4) var<storage, read> allowed: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
var<workgroup> rm: array<f32, 256>;
const BISECT = 24u;
/** Offsets of the sampled row in logits and in allowed. */
var<private> base: u32;
var<private> abase: u32;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

/** Whether the grammar mask (if any) lets token i through. */
fn ok(i: u32) -> bool {
  return allowed[abase] == 0u || ((allowed[abase + 1u + i / 32u] >> (i % 32u)) & 1u) == 1u;
}

/** Penalised, temperature-scaled logit. */
fn adj(i: u32) -> f32 {
  var v = logits[base + i];
  if (st.repPenalty != 1.0 && ((seen[i / 32u] >> (i % 32u)) & 1u) == 1u) {
    v = select(v * st.repPenalty, v / st.repPenalty, v > 0.0);
  }
  return select(v, v / st.temperature, st.temperature > 0.0);
}

/** Workgroup max (into bv[0]) and min (into rm[0]) of per-thread values. */
fn reduceMaxMin(li: u32, mx: f32, mn: f32) {
  bv[li] = mx;
  rm[li] = mn;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      bv[li] = max(bv[li], bv[li + s]);
      rm[li] = min(rm[li], rm[li + s]);
    }
    workgroupBarrier();
  }
}

/** Workgroup sums of a count (into bv[0]) and a mass (into rm[0]). */
fn reduceSums(li: u32, c: f32, m: f32) {
  bv[li] = c;
  rm[li] = m;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      bv[li] += bv[li + s];
      rm[li] += rm[li + s];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let row = ${rows ? "wid.y" : "0u"};
  base = row * p.V;
  abase = row * (1u + (p.V + 31u) / 32u);
  let sampling = st.temperature > 0.0;
  let truncate = sampling && (st.topK > 0u || st.topP < 1.0);

  // Largest threshold that still keeps ≥ topK tokens and ≥ topP of the probability mass.
  var lo = -3.0e38;
  if (truncate) {
    var mx = -3.0e38;
    var mn = 3.0e38;
    for (var i = li; i < p.V; i += 256u) { if (!ok(i)) { continue; } let v = adj(i); mx = max(mx, v); mn = min(mn, v); }
    reduceMaxMin(li, mx, mn);
    let vmax = bv[0];
    let vmin = rm[0];
    workgroupBarrier();
    lo = vmin;
    var hi = vmax;
    var total = 1.0;
    var z = 0.0;
    for (var i = li; i < p.V; i += 256u) { if (ok(i)) { z += exp(adj(i) - vmax); } }
    reduceSums(li, 0.0, z);
    total = rm[0];
    workgroupBarrier();
    // Logits this far below the max carry no mass worth keeping for top-p alone.
    if (st.topK == 0u) { lo = max(vmin, vmax - 40.0); }
    for (var it = 0u; it < BISECT; it++) {
      let mid = 0.5 * (lo + hi);
      var c = 0.0;
      var m = 0.0;
      for (var i = li; i < p.V; i += 256u) {
        if (!ok(i)) { continue; }
        let v = adj(i);
        if (v >= mid) { c += 1.0; m += exp(v - vmax); }
      }
      reduceSums(li, c, m);
      let ok = (st.topK == 0u || bv[0] >= f32(st.topK)) && (st.topP >= 1.0 || rm[0] >= st.topP * total);
      workgroupBarrier();
      lo = select(lo, mid, ok);
      hi = select(mid, hi, ok);
    }
  }
  let cut = select(-3.0e38, lo, truncate);

  let salt = pcg(st.seed ^ pcg(st.pos0 + row + st.lastRow * 7919u));
  var best = -3.0e38;
  var bidx = 0u;
  for (var i = li; i < p.V; i += 256u) {
    if (!ok(i)) { continue; }
    var v = adj(i);
    if (v < cut) { continue; }
    if (sampling) {
      let u = (f32(pcg(i ^ salt) >> 8u) + 0.5) / 16777216.0;
      v = v - log(-log(u));
    }
    if (v > best) { best = v; bidx = i; }
  }
  bv[li] = best;
  bi[li] = bidx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    // Ties go to the lower token id, matching a sequential argmax.
    if (li < s && (bv[li + s] > bv[li] || (bv[li + s] == bv[li] && bi[li + s] < bi[li]))) { bv[li] = bv[li + s]; bi[li] = bi[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { out[row] = bi[0]; }
}`);
}

/**
 * Moves the step state forward on the GPU so several decode steps can share one submit:
 * feeds the last sampled id back as the next input, appends it to the history, marks it seen (for the
 * repetition penalty) and advances pos0.
 */
export function advanceSpec(): KernelSpec {
  return spec("advance", ["rw", "read", "rw", "rw", "rw"], () => /* wgsl */ `
${STEP_WGSL}
@group(0) @binding(0) var<storage, read_write> st: Step;
@group(0) @binding(1) var<storage, read> sampled: array<u32>;
@group(0) @binding(2) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> history: array<u32>;
@group(0) @binding(4) var<storage, read_write> seen: array<u32>;

@compute @workgroup_size(1)
fn main() {
  let id = sampled[0];
  tokens[0] = id;
  history[st.n] = id;
  seen[id / 32u] |= 1u << (id % 32u);
  st.n = st.n + 1u;
  st.pos0 = st.pos0 + st.lastRow + 1u;
  st.lastRow = 0u;
}`);
}

/**
 * Rotary embedding fused with the KV-cache write. Reads fused qkv rows [T, (H + 2·Hkv)·D], writes the
 * rotated q to qo [T, H·D] and the rotated k plus v straight into the caches at step.pos0 + t.
 * One thread per output element; each reads its rotation partner, so nothing is updated in place.
 */
export function ropeKvSpec(interleaved: boolean, kv: KvDtype = "f32"): KernelSpec {
  const h = kv === "f16";
  return spec(`rope_kv/${interleaved ? "interleaved" : "half"}/${kv}`, ["uniform", "read", "read", "read", "rw", "rw", "rw"], () => /* wgsl */ `
${STEP_WGSL}
struct P { T: u32, H: u32, Hkv: u32, D: u32, maxSeq: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> invFreq: array<f32>;
@group(0) @binding(3) var<storage, read> qkv: array<f32>;
@group(0) @binding(4) var<storage, read_write> qo: array<f32>;
@group(0) @binding(5) var<storage, read_write> kc: array<${h ? "u32" : "f32"}>;
@group(0) @binding(6) var<storage, read_write> vc: array<${h ? "u32" : "f32"}>;

/** Rotated value of element idx (column i of its head) at position pos. */
fn rot(idx: u32, i: u32, pos: u32) -> f32 {
  let base = idx - i;
${interleaved
  ? `  let j = i / 2u;
  let a = qkv[base + 2u * j];
  let b = qkv[base + 2u * j + 1u];
  let first = (i % 2u) == 0u;`
  : `  let hd = p.D / 2u;
  let j = i % hd;
  let a = qkv[base + j];
  let b = qkv[base + j + hd];
  let first = i < hd;`}
  let ang = f32(pos) * invFreq[j];
  let cs = cos(ang);
  let sn = sin(ang);
  return select(b * cs + a * sn, a * cs - b * sn, first);
}

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  // One thread per element pair (D is a multiple of 4, so pairs never straddle q, k and v).
  let e = idx * 2u;
  let qn = p.H * p.D;
  let kn = p.Hkv * p.D;
  let W = qn + 2u * kn;
  if (e >= p.T * W) { return; }
  let t = e / W;
  let c = e % W;
  let pos = st.pos0 + t;
  var pair: vec2<f32>;
  if (c >= qn + kn) {
    pair = vec2<f32>(qkv[e], qkv[e + 1u]);
  } else {
    let i = c % p.D;
    pair = vec2<f32>(rot(e, i, pos), rot(e + 1u, i + 1u, pos));
  }
  if (c < qn) {
    qo[t * qn + c] = pair.x;
    qo[t * qn + c + 1u] = pair.y;
    return;
  }
  if (pos >= p.maxSeq) { return; }
  let dst = pos * kn + (c - qn) % kn;
${h
  ? `  let packed = pack2x16float(pair);
  if (c < qn + kn) { kc[dst / 2u] = packed; } else { vc[dst / 2u] = packed; }`
  : `  if (c < qn + kn) { kc[dst] = pair.x; kc[dst + 1u] = pair.y; } else { vc[dst] = pair.x; vc[dst + 1u] = pair.y; }`}
}`);
}
