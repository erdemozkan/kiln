import { STEP_WGSL, type BindingKind, type CpuKernel, type KernelSpec, type WeightDType } from "@kiln/core";
import * as cpu from "./cpu";

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
/** A kernel by key: WGSL for the GPU, and the same computation in TypeScript for the CPU backend (if any). */
function spec(key: string, bindings: KernelSpec["bindings"], cpuImpl: CpuKernel | undefined, code: () => string): KernelSpec {
  let s = cache.get(key);
  if (!s) {
    s = { key, bindings, code: code(), cpu: cpuImpl };
    cache.set(key, s);
  }
  return s;
}

/** Elements of K staged in workgroup memory per step. */
export const MM_KC = 256;
/** Lanes per weight row a matmul variant may use; rows per workgroup = 256 / lanes. */
export type MatmulLanes = 8 | 16 | 32;
/** What the matmul does with each output after the dot product. */
export type MatmulEpilogue = "store" | "add" | "silu" | "gelu";

/** Epilogues that read a [gate; up] weight stacked along N and write act(gate) * up. */
export const isGated = (epi: MatmulEpilogue): boolean => epi === "silu" || epi === "gelu";

/**
 * GELU with the tanh approximation (PyTorch's gelu "tanh", Gemma's activation), on f32 x. The tanh
 * argument is clamped: some drivers compute tanh through exp and return NaN past about ±44.
 */
const GELU_WGSL = (x: string) => `(0.5 * ${x} * (1.0 + tanh(clamp(0.7978845608 * (${x} + 0.044715 * ${x} * ${x} * ${x}), -15.0, 15.0))))`;
const actWgsl = (epi: "silu" | "gelu", x: string) => (epi === "gelu" ? GELU_WGSL(x) : `(${x} / (1.0 + exp(-${x})))`);

export interface MatmulVariant {
  wdtype: WeightDType;
  /** Tokens per workgroup: 1 for decode, 8 for prefill. */
  mt: 1 | 8;
  lanes: MatmulLanes;
  /** RMSNorm x on the fly (each workgroup recomputes the row norms; K is small). */
  norm: boolean;
  bias: boolean;
  /**
   * store: y = acc. add: y += acc (residual). silu / gelu: the weight is [gate; up] stacked along N
   * and y[:, i] = act(gate_i) * up_i, so a workgroup owns matching gate and up rows.
   */
  epi: MatmulEpilogue;
  /** f16 x tile and 4-wide dot products (needs shader-f16); accumulation stays f32. */
  f16?: boolean;
  /** Affine q4 / q8: a mins plane after the scales, value = code·scale + min (q4 codes 0…15). */
  affine?: boolean;
  /**
   * Reduce with subgroup operations instead of workgroup-memory trees (needs `subgroups` and a
   * subgroup size of at least `lanes`): a shuffle butterfly across each row's lanes, and subgroupAdd
   * for the norm prologue. Assumes subgroups are runs of consecutive invocation indices.
   */
  sg?: boolean;
}

/** Rows of the output a workgroup produces for this variant. */
export const matmulRowsPerGroup = (v: Pick<MatmulVariant, "lanes" | "epi">): number => 256 / v.lanes / (isGated(v.epi) ? 2 : 1);

/**
 * y[m, n] = sum_k x[m, k] * W[n, k], for K a multiple of 32, with an optional RMSNorm prologue and a
 * fused epilogue (see MatmulVariant).
 *
 * A 256-thread workgroup owns 256 / lanes weight rows. K is walked in chunks of MM_KC: an [mt, MM_KC]
 * tile of x is staged in workgroup memory once and shared by all rows. Within a 32-element block lane l
 * handles vec4 slot l % 8, and lanes l / 8 = 0, 1, … take every (lanes / 8)-th block, so consecutive lanes
 * read consecutive vec4s of the tile (no bank conflicts) and consecutive bytes of the weights (coalesced).
 * For q4 a block is one scale plus 4 u32 of nibbles; slot s decodes half (s % 2) of word s / 2. For q8 a
 * block is one scale plus 8 u32 of int8s, and slot s decodes word s.
 * More lanes per row means more workgroups for small N, which matters on GPUs with many cores.
 */
export function matmulSpec(v: MatmulVariant): KernelSpec {
  const q4 = v.wdtype === "q4";
  const quant = v.wdtype !== "f32";
  const kc4 = MM_KC / 4;
  const mt = v.mt;
  const rows = 256 / v.lanes;
  const silu = isGated(v.epi);
  const h = !!v.f16;
  const sg = !!v.sg;
  const aff = quant && !!v.affine;
  const key = `matmul/${v.wdtype}${aff ? "a" : ""}/mt${mt}/l${v.lanes}${v.norm ? "/norm" : ""}${v.bias ? "/bias" : ""}/${v.epi}${h ? "/f16" : ""}${sg ? "/sg" : ""}`;
  const vt = h ? "vec4<f16>" : "vec4<f32>";
  const bindings: BindingKind[] = quant ? ["uniform", "read", "read", "read"] : ["uniform", "read", "read"];
  if (aff) bindings.push("read");
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
      y[(m0 + lane) * (p.N / 2u) + i] = ${actWgsl(v.epi as "silu" | "gelu", "g")} * u;
    }`
    : `if (n < p.N) {
      let val = ${withBias("n", `red[lane * 256u + r * LANES]`)};
      y[(m0 + lane) * p.N + n] ${v.epi === "add" ? "+=" : "="} val;
    }`;
  return spec(key, bindings, cpu.cpuMatmul(v), () => /* wgsl */ `
${h ? "enable f16;" : ""}
${sg ? "enable subgroups;" : ""}
struct P { M: u32, N: u32, K: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
${quant
  ? `@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> ws: array<f32>;${aff ? "\n@group(0) @binding(4) var<storage, read> wm: array<f32>;" : ""}`
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
        let wv = ${aff
          ? `${vt}(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(15u)) * ${h ? `f16(ws[b0 + bi])` : `ws[b0 + bi]`} + ${h ? `f16(wm[b0 + bi])` : `wm[b0 + bi]`}`
          : `(${vt}(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(15u)) - ${h ? "8.0h" : "8.0"}) * ${h ? `f16(ws[b0 + bi])` : `ws[b0 + bi]`}`};`
  : quant
  ? `        let word = wq[(b0 + bi) * 8u + slot];
        let wv = ${vt}(bitcast<vec4<i32>>(vec4<u32>(word << 24u, word << 16u, word << 8u, word)) >> vec4<u32>(24u)) * ${h ? `f16(ws[b0 + bi])` : `ws[b0 + bi]`}${aff ? ` + ${h ? `f16(wm[b0 + bi])` : `wm[b0 + bi]`}` : ""};`
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

/** 32-element blocks of x the decode GEMV stages per step. */
export const GEMV_KB = 64;

/** Decode GEMV layout: lanes per row group and weight rows per thread (sharing each staged x block). */
export interface GemvLayout {
  lanes: 4 | 8;
  rows: 1 | 2 | 4;
}

/**
 * Measured on Gemma 3 4B and SmolLM2 (Intel Gen9): several rows per thread pay off on large
 * matrices (gate/up and the head 1.1x, down 1.15x) but cost 10-40% on small ones, which have too few
 * workgroups (1024 rows) or too few blocks per lane (K = 576) for 8 lanes.
 */
export const gemvLayout = (outN: number, K: number): GemvLayout =>
  outN >= 8192 ? { lanes: 8, rows: 4 } : K >= 4096 ? { lanes: 8, rows: 2 } : { lanes: 4, rows: 1 };

export interface GemvVariant {
  wdtype: WeightDType;
  /** Affine q4 / q8 (see `MatmulVariant.affine`). */
  affine?: boolean;
  norm: boolean;
  bias: boolean;
  epi: MatmulEpilogue;
  layout: GemvLayout;
}

/** Output rows a GEMV workgroup produces. */
export const gemvRowsPerGroup = (epi: MatmulEpilogue, l: GemvLayout): number => ((256 / l.lanes) * l.rows) / (isGated(epi) ? 2 : 1);

/**
 * Decode matmul (one token), y[n] = sum_k x[k] * W[n, k], with the same prologue and epilogues as
 * `matmulSpec`. Each lane takes whole 32-element blocks: one 16-byte load for a q4 block (two for q8,
 * eight vec4 for f32) and 32 multiply-adds, where the general kernel loads 4 bytes for 4. Neighbouring
 * lanes take neighbouring blocks, so a row's lanes read one contiguous run. The x chunk is staged
 * transposed (slot j of block b at j * KB + b), so those lanes read consecutive vec4s of it, and each
 * thread applies the x block it read to `rows` weight rows: with one row per thread, x reads from
 * workgroup memory were 8x the weight bytes. Gated epilogues give a thread gate rows and the matching
 * up rows; with one row per thread, the first half of the row groups take gate rows and the second
 * half the matching up rows.
 */
export function gemvSpec(v: GemvVariant): KernelSpec {
  const gated = isGated(v.epi);
  const quant = v.wdtype !== "f32";
  const aff = quant && !!v.affine;
  const key = `gemv/${v.wdtype}${aff ? "a" : ""}${v.norm ? "/norm" : ""}${v.bias ? "/bias" : ""}/${v.epi}/l${v.layout.lanes}r${v.layout.rows}`;
  const bindings: BindingKind[] = quant ? ["uniform", "read", "read", "read"] : ["uniform", "read", "read"];
  if (aff) bindings.push("read");
  let b = bindings.length;
  const decl: string[] = [];
  if (v.norm) {
    decl.push(`@group(0) @binding(${b++}) var<storage, read> nw: array<vec4<f32>>;`);
    bindings.push("read");
  }
  if (v.bias) {
    decl.push(`@group(0) @binding(${b++}) var<storage, read> bias: array<f32>;`);
    bindings.push("read");
  }
  decl.push(`@group(0) @binding(${b}) var<storage, read_write> y: array<f32>;`);
  bindings.push("rw");
  const R = v.layout.rows;
  const LANES = v.layout.lanes;
  const groups = 256 / LANES;
  // Gated with one row per thread: row group gi pairs with gi + groups / 2.
  const split = gated && R === 1;
  const withBias = (row: string, val: string) => (v.bias ? `(${val} + bias[${row}])` : val);
  // The dot product of one block (index blk) with the x block in registers x0…x7. Affine blocks add
  // min · (sum of the x block), with the sum shared by the thread's rows.
  const d4 = aff ? "dq4u" : "dq4";
  const tail = (blk: string) => (aff ? ` + xsum * wm[${blk}]` : "");
  const blockDot = (blk: string) =>
    v.wdtype === "q4"
      ? `let q = wq[${blk}];
          acc += (dot(x0, ${d4}(q.x)) + dot(x1, ${d4}(q.x >> 16u)) + dot(x2, ${d4}(q.y)) + dot(x3, ${d4}(q.y >> 16u))
            + dot(x4, ${d4}(q.z)) + dot(x5, ${d4}(q.z >> 16u)) + dot(x6, ${d4}(q.w)) + dot(x7, ${d4}(q.w >> 16u))) * ws[${blk}]${tail(blk)};`
      : quant
        ? `let q0 = wq[${blk} * 2u];
          let q1 = wq[${blk} * 2u + 1u];
          acc += (dot(x0, dq8(q0.x)) + dot(x1, dq8(q0.y)) + dot(x2, dq8(q0.z)) + dot(x3, dq8(q0.w))
            + dot(x4, dq8(q1.x)) + dot(x5, dq8(q1.y)) + dot(x6, dq8(q1.z)) + dot(x7, dq8(q1.w))) * ws[${blk}]${tail(blk)};`
        : `let o = ${blk} * 8u;
          acc += dot(x0, w[o]) + dot(x1, w[o + 1u]) + dot(x2, w[o + 2u]) + dot(x3, w[o + 3u])
            + dot(x4, w[o + 4u]) + dot(x5, w[o + 5u]) + dot(x6, w[o + 6u]) + dot(x7, w[o + 7u]);`;
  const rowsCode = Array.from({ length: R }, (_, r) => `
        if (ok${r}) {
          var acc = acc${r};
          ${blockDot(`(row${r} * blocks + c0 + bi)`)}
          acc${r} = acc;
        }`).join("");
  const rowDecl = split
    ? `  let half = p.N / 2u;
  let i0 = grp * ${groups / 2}u + gi % ${groups / 2}u;
  let row0 = select(i0, half + i0, gi >= ${groups / 2}u);
  let ok0 = i0 < half;`
    : gated
    ? `  let half = p.N / 2u;
  let i0 = grp * ${(groups * R) / 2}u + gi * ${R / 2}u;
${Array.from({ length: R }, (_, r) => `  let row${r} = ${r < R / 2 ? "" : "half + "}i0 + ${r % (R / 2)}u;
  let ok${r} = i0 + ${r % (R / 2)}u < half;`).join("\n")}`
    : `  let n0 = grp * ${groups * R}u + gi * ${R}u;
${Array.from({ length: R }, (_, r) => `  let row${r} = n0 + ${r}u;
  let ok${r} = row${r} < p.N;`).join("\n")}`;
  const store = split
    ? `    if (gi < ${groups / 2}u && ok0) {
      let g = ${withBias("row0", "red[li]")};
      let u = ${withBias("half + i0", `red[li + ${(groups / 2) * LANES}u]`)};
      y[i0] = ${actWgsl(v.epi as "silu" | "gelu", "g")} * u;
    }`
    : gated
    ? Array.from({ length: R / 2 }, (_, j) => `    if (ok${j}) {
      let g = ${withBias(`row${j}`, `red[${j * 256}u + li]`)};
      let u = ${withBias(`row${j + R / 2}`, `red[${(j + R / 2) * 256}u + li]`)};
      y[i0 + ${j}u] = ${actWgsl(v.epi as "silu" | "gelu", "g")} * u;
    }`).join("\n")
    : Array.from({ length: R }, (_, r) => `    if (ok${r}) { y[row${r}] ${v.epi === "add" ? "+=" : "="} ${withBias(`row${r}`, `red[${r * 256}u + li]`)}; }`).join("\n");
  return spec(key, bindings, cpu.cpuMatmul({ ...v, wdtype: v.wdtype }), () => /* wgsl */ `
struct P { M: u32, N: u32, K: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
${quant
  ? `@group(0) @binding(2) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> ws: array<f32>;${aff ? "\n@group(0) @binding(4) var<storage, read> wm: array<f32>;" : ""}`
  : `@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;`}
${decl.join("\n")}
const LANES = ${LANES}u;
const KB = ${GEMV_KB}u;
var<workgroup> xt: array<vec4<f32>, ${GEMV_KB * 8}>;
var<workgroup> red: array<f32, ${256 * R}>;

fn dq4(w: u32) -> vec4<f32> { return vec4<f32>(vec4<u32>(w, w >> 4u, w >> 8u, w >> 12u) & vec4<u32>(15u)) - 8.0; }
fn dq4u(w: u32) -> vec4<f32> { return vec4<f32>(vec4<u32>(w, w >> 4u, w >> 8u, w >> 12u) & vec4<u32>(15u)); }
fn dq8(w: u32) -> vec4<f32> { return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w << 24u, w << 16u, w << 8u, w)) >> vec4<u32>(24u)); }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let lane = li % LANES;
  let gi = li / LANES;
  let grp = wid.x + wid.z * ${MM_SLICE}u;
${rowDecl}
  let blocks = p.K / 32u;
  let xs4 = p.K / 4u;
${v.norm
  ? `  // RMSNorm prologue over the one row of x.
  var ss = 0.0;
  for (var c = li; c < xs4; c += 256u) { let xv = x[c]; ss += dot(xv, xv); }
  red[li] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(p.K) + p.eps);
  workgroupBarrier();`
  : ""}
${Array.from({ length: R }, (_, r) => `  var acc${r} = 0.0;`).join("\n")}
  for (var c0 = 0u; c0 < blocks; c0 += KB) {
    for (var e = li; e < KB * 8u; e += 256u) {
      let c = c0 * 8u + e;
      let xv = select(vec4<f32>(0.0), x[min(c, xs4 - 1u)], c < xs4);
      xt[(e % 8u) * KB + e / 8u] = ${v.norm ? "xv * inv * nw[min(c, xs4 - 1u)]" : "xv"};
    }
    workgroupBarrier();
    let nb = min(KB, blocks - c0);
    for (var bi = lane; bi < nb; bi += LANES) {
      let x0 = xt[bi]; let x1 = xt[KB + bi]; let x2 = xt[2u * KB + bi]; let x3 = xt[3u * KB + bi];
      let x4 = xt[4u * KB + bi]; let x5 = xt[5u * KB + bi]; let x6 = xt[6u * KB + bi]; let x7 = xt[7u * KB + bi];${aff ? "\n      let xsum = dot(((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)), vec4<f32>(1.0));" : ""}${rowsCode}
    }
    workgroupBarrier();
  }
${Array.from({ length: R }, (_, r) => `  red[${r * 256}u + li] = acc${r};`).join("\n")}
  workgroupBarrier();
  for (var s = LANES / 2u; s > 0u; s >>= 1u) {
    if (lane < s) {
${Array.from({ length: R }, (_, r) => `      red[${r * 256}u + li] += red[${r * 256}u + li + s];`).join("\n")}
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
${store}
  }
}`);
}

/** Tokens and weight rows per GEMM workgroup tile. */
export const GEMM_TILE = 64;

export interface GemmVariant {
  wdtype: WeightDType;
  /** Affine q4 / q8 (see `MatmulVariant.affine`). */
  affine?: boolean;
  bias: boolean;
  epi: MatmulEpilogue;
}

/**
 * Prefill matmul, y[m, n] = sum_k x[m, k] * W[n, k], as a tiled GEMM: a 128-thread workgroup owns a
 * 64-token × 64-row output tile and each thread an 8 × 4 block of it. K is walked one 32-element block
 * at a time; the x tile and the dequantized weight tile are staged in workgroup memory, transposed so
 * each k step is three vec4 reads per thread for 32 multiply-adds. On Intel Gen9 this is 1.9x the
 * 256-thread 4 × 4 version, and larger blocks or tiles were slower (register pressure).
 *
 * Each block's global reads are all issued, branch-free (rows past the edge read a clamped row with a
 * zero scale), before anything is written to the tiles: on Gen9 that took Gemma 3 4B's gate/up matmul
 * from 121 to 100 ms and the vision tower's MLP up from 291 to 272 ms against reads guarded by `if`
 * and interleaved with the stores.
 *
 * For gated epilogues the tile's 64 rows are 32 gate rows and the 32 matching up rows, stored so each
 * thread's four columns are gate i, gate i + 1, up i, up i + 1 and it can write act(gate) * up itself.
 */
export function gemmSpec(v: GemmVariant): KernelSpec {
  const gated = isGated(v.epi);
  const quant = v.wdtype !== "f32";
  const aff = quant && !!v.affine;
  const key = `gemm/${v.wdtype}${aff ? "a" : ""}${v.bias ? "/bias" : ""}/${v.epi}`;
  const bindings: BindingKind[] = quant ? ["uniform", "read", "read", "read"] : ["uniform", "read", "read"];
  if (aff) bindings.push("read");
  let b = bindings.length;
  const decl: string[] = [];
  if (v.bias) {
    decl.push(`@group(0) @binding(${b++}) var<storage, read> bias: array<f32>;`);
    bindings.push("read");
  }
  decl.push(`@group(0) @binding(${b}) var<storage, read_write> y: array<f32>;`);
  bindings.push("rw");
  const R8 = [0, 1, 2, 3, 4, 5, 6, 7];

  // x tile: thread j takes tokens 4g…4g+3 (g = j / 8) at vec4 q = j % 8 of the block, so 8
  // neighbouring threads read one row's 128 contiguous bytes.
  const readX = `    for (var c = 0u; c < 4u; c++) {
      let m = m0 + (li / 8u) * 4u + c;
      xr[c] = select(vec4<f32>(0.0), x[min(m, p.M - 1u) * xs4 + kb * 8u + li % 8u], m < p.M);
    }`;
  const storeX = `    for (var e = 0u; e < 4u; e++) { xt[((li % 8u) * 4u + e) * 16u + li / 8u] = vec4<f32>(xr[0][e], xr[1][e], xr[2][e], xr[3][e]); }`;
  // Weight tile: a thread takes the 4 rows of one tile column group and one word of their block
  // (neighbouring threads, neighbouring words), and writes whole vec4s (one per k): threads writing
  // components of one shared vec4 race on some drivers. q4 takes 64 threads, the others all 128.
  const regsW = v.wdtype === "f32" ? "  var wr4: array<vec4<f32>, 4>;" : `  var wr: array<u32, 4>;\n  var wd: array<f32, 4>;${aff ? "\n  var wn: array<f32, 4>;" : ""}`;
  const scales = (i: string) =>
    `        wd[${i}] = select(0.0, ws[r * blocks + kb], row < p.N);${aff ? `\n        wn[${i}] = select(0.0, wm[r * blocks + kb], row < p.N);` : ""}`;
  let readW: string;
  let storeW: string;
  if (v.wdtype === "q4") {
    readW = `    if (li < 64u) {
      for (var c = 0u; c < 4u; c++) {
        let row = slotRow((li / 4u) * 4u + c);
        let r = min(row, p.N - 1u);
        wr[c] = wq[(r * blocks + kb) * 4u + li % 4u];
${scales("c")}
      }
    }`;
    storeW = `    if (li < 64u) {
      var vals: array<vec4<f32>, 8>;
      for (var c = 0u; c < 4u; c++) {
        for (var e = 0u; e < 8u; e++) { vals[e][c] = ${aff ? "f32((wr[c] >> (4u * e)) & 15u) * wd[c] + wn[c]" : "(f32((wr[c] >> (4u * e)) & 15u) - 8.0) * wd[c]"}; }
      }
      for (var e = 0u; e < 8u; e++) { wt[((li % 4u) * 8u + e) * 16u + li / 4u] = vals[e]; }
    }`;
  } else {
    readW = `    for (var c = 0u; c < 4u; c++) {
      let row = slotRow((li / 8u) * 4u + c);
      let r = min(row, p.N - 1u);
${v.wdtype === "q8" ? `        wr[c] = wq[(r * blocks + kb) * 8u + li % 8u];\n${scales("c")}` : "        wr4[c] = select(vec4<f32>(0.0), w[(r * blocks + kb) * 8u + li % 8u], row < p.N);"}
    }`;
    const val =
      v.wdtype === "q8"
        ? `vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(wr[c] << 24u, wr[c] << 16u, wr[c] << 8u, wr[c])) >> vec4<u32>(24u)) * wd[c]${aff ? " + wn[c]" : ""}`
        : "wr4[c]";
    storeW = `    {
      var r4: array<vec4<f32>, 4>;
      for (var c = 0u; c < 4u; c++) { r4[c] = ${val}; }
      for (var e = 0u; e < 4u; e++) { wt[((li % 8u) * 4u + e) * 16u + li / 8u] = vec4<f32>(r4[0][e], r4[1][e], r4[2][e], r4[3][e]); }
    }`;
  }

  const withBias = (row: string, val: string) => (v.bias ? `(${val} + bias[${row}])` : val);
  // Unrolled, with the accumulators as separate variables.
  const store = gated
    ? `  let half = p.N / 2u;
${R8.map((i) => `  if (m0 + tm * 8u + ${i}u < p.M) {
    let m = m0 + tm * 8u + ${i}u;
${[0, 1].map((j) => `    if (grp * 32u + tn * 2u + ${j}u < half) {
      let col = grp * 32u + tn * 2u + ${j}u;
      let g = ${withBias("col", `acc${i}[${j}]`)};
      let u = ${withBias("half + col", `acc${i}[${j + 2}]`)};
      y[m * half + col] = ${actWgsl(v.epi as "silu" | "gelu", "g")} * u;
    }`).join("\n")}
  }`).join("\n")}`
    : R8.map(
        (i) => `  if (m0 + tm * 8u + ${i}u < p.M) {
    let m = m0 + tm * 8u + ${i}u;
${[0, 1, 2, 3].map((j) => `    if (grp * 64u + tn * 4u + ${j}u < p.N) { let n = grp * 64u + tn * 4u + ${j}u; y[m * p.N + n] ${v.epi === "add" ? "+=" : "="} ${withBias("n", `acc${i}[${j}]`)}; }`).join("\n")}
  }`,
      ).join("\n");
  return spec(key, bindings, cpu.cpuMatmul({ ...v, norm: false }), () => /* wgsl */ `
struct P { M: u32, N: u32, K: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
${quant
  ? `@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> ws: array<f32>;${aff ? "\n@group(0) @binding(4) var<storage, read> wm: array<f32>;" : ""}`
  : `@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;`}
${decl.join("\n")}
// [32 k][64 tokens] and [32 k][64 weight rows], as vec4 along the token / row axis.
var<workgroup> xt: array<vec4<f32>, 512>;
var<workgroup> wt: array<vec4<f32>, 512>;
var<private> grp: u32;

/** Weight row staged in column s (0…63) of the tile. */
fn slotRow(s: u32) -> u32 {
${gated
  ? `  // Thread t's four columns are gate rows 2t, 2t + 1 and up rows 2t, 2t + 1 of the group's 32.
  let i = (s / 4u) * 2u + s % 2u;
  return select(grp * 32u + i, p.N / 2u + grp * 32u + i, s % 4u >= 2u);`
  : `  return grp * 64u + s;`}
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  grp = wid.x + wid.z * ${MM_SLICE}u;
  let m0 = wid.y * 64u;
  let tn = li % 16u;
  let tm = li / 16u;
  let blocks = p.K / 32u;
  let xs4 = p.K / 4u;
${R8.map((i) => `  var acc${i} = vec4<f32>();`).join("\n")}
  var xr: array<vec4<f32>, 4>;
${regsW}
  for (var kb = 0u; kb < blocks; kb++) {
${readX}
${readW}
${storeX}
${storeW}
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let a0 = xt[k * 16u + tm * 2u];
      let a1 = xt[k * 16u + tm * 2u + 1u];
      let bv = wt[k * 16u + tn];
${R8.map((i) => `      acc${i} += a${i >> 2}.${"xyzw"[i & 3]} * bv;`).join("\n")}
    }
    workgroupBarrier();
  }
${store}
}`);
}

/** y = x * rsqrt(mean(x^2) + eps) * w, one workgroup per row. */
export function rmsnormSpec(): KernelSpec {
  return spec("rmsnorm", ["uniform", "read", "read", "rw"], cpu.cpuRmsnorm, () => /* wgsl */ `
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
  return spec(`rope/${interleaved ? "interleaved" : "half"}`, ["uniform", "read", "read", "rw"], cpu.cpuRope(interleaved), () => /* wgsl */ `
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

/** Writes k/v rows [T, rowSize] into the caches at positions pos0..pos0+T (slot = position % cap). */
export function kvWriteSpec(kv: KvDtype = "f32"): KernelSpec {
  const h = kv === "f16";
  return spec(`kv_write/${kv}`, ["uniform", "read", "read", "read", "rw", "rw"], cpu.cpuKvWrite(kv === "f16"), () => /* wgsl */ `
${STEP_WGSL}
struct P { T: u32, row: u32, maxSeq: u32, cap: u32 }
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
  let dst = (((st.pos0 + t) % p.cap) * p.row + e % p.row) / 2u;
  kc[dst] = pack2x16float(vec2<f32>(k[e], k[e + 1u]));
  vc[dst] = pack2x16float(vec2<f32>(v[e], v[e + 1u]));`
  : `  if (idx >= p.T * p.row) { return; }
  let t = idx / p.row;
  if (st.pos0 + t >= p.maxSeq) { return; }
  let dst = ((st.pos0 + t) % p.cap) * p.row + idx % p.row;
  kc[dst] = k[idx];
  vc[dst] = v[idx];`}
}`);
}

/** Threads per attention workgroup; head dims up to this size are supported. */
export const ATTN_WG = 256;

/**
 * First key a query sees: with a sliding window of `win` keys (0 = none), a query that sees keys
 * 0…n-1 keeps only the last `win` of them, as Gemma's sliding layers do.
 */
const FIRST_KEY_WGSL = /* wgsl */ `fn firstKey(n: u32, win: u32) -> u32 { return select(0u, n - win, win > 0u && n > win); }
// Keys the query at position pos sees: causally pos + 1, or the whole span [spanLo, spanHi) for a query
// inside it (bidirectional attention: an image's tokens, all of a vision encoder's patches).
fn keys(pos: u32) -> u32 { return min(select(pos + 1u, st.spanHi, pos >= st.spanLo && pos < st.spanHi), p.maxSeq); }
// Cache slot of a key, given base = (first key of a run) % cap and an offset into the run of at most cap.
fn wrap(s: u32) -> u32 { return select(s, s - p.cap, s >= p.cap); }`;

/**
 * Attention parameters. A layer's cache holds cap slots and key j lives in slot j % cap: cap is
 * maxSeq for full layers, and window + slack for sliding layers, whose cache is a ring.
 */
const ATTN_P = (third: string) => `struct P { T: u32, H: u32, Hkv: u32, D: u32, maxSeq: u32, scale: f32, ${third}: u32, win: u32, cap: u32, _x: u32, _y: u32, _z: u32 }`;

/**
 * Causal attention with GQA: one 256-thread workgroup per (head, query row).
 * q is cached in workgroup memory. Keys are taken ATTN_WG at a time with an online softmax: each block's
 * scores stay in workgroup memory, the running max and exp-sum rescale the partial output, so no scores
 * reach global memory. The weighted sum splits each block's keys across ATTN_WG / D thread groups and
 * reduces their partial outputs at the end, so every thread works even when D is small.
 */
export function attentionSpec(kv: KvDtype = "f32", rel = false): KernelSpec {
  const r = kvRead(kv);
  const bindings: BindingKind[] = rel ? ["uniform", "read", "read", "read", "read", "read", "rw"] : ["uniform", "read", "read", "read", "read", "rw"];
  return spec(`attention/online/${kv}${rel ? "/t5bias" : ""}`, bindings, cpu.cpuAttention(kv === "f16", rel), () => /* wgsl */ `
${STEP_WGSL}
${kv === "f16" ? KV4_FN : ""}
${ATTN_P("_a")}
${FIRST_KEY_WGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: ${r.kDecl};
@group(0) @binding(4) var<storage, read> vc: ${r.vDecl};
${rel ? `@group(0) @binding(5) var<storage, read> rel: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
// T5's relative position bucket (see cpu.t5Bucket): _x buckets, _y max distance, _z bidirectional.
fn bucket(r: i32) -> u32 {
  var nb = p._x;
  var ret = 0u;
  var n: i32;
  if (p._z != 0u) {
    nb = nb / 2u;
    ret = select(0u, nb, r > 0);
    n = abs(r);
  } else {
    n = max(-r, 0);
  }
  let exact = nb / 2u;
  if (u32(n) < exact) { return ret + u32(n); }
  let large = exact + u32(log(f32(n) / f32(exact)) / log(f32(p._y) / f32(exact)) * f32(nb - exact));
  return ret + min(large, nb - 1u);
}` : `@group(0) @binding(5) var<storage, read_write> o: array<f32>;`}
const WG = ${ATTN_WG}u;
var<workgroup> red: array<f32, ${ATTN_WG}>;
var<workgroup> ss: array<f32, ${ATTN_WG}>;
var<workgroup> qs: array<vec4<f32>, ${ATTN_WG / 4}>;

@compute @workgroup_size(${ATTN_WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wid.x;
  let t = wid.y;
  let kvh = h / (p.H / p.Hkv);
  let n = keys(st.pos0 + t);
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
  for (var b0 = firstKey(n, p.win); b0 < n; b0 += WG) {
    let bn = min(WG, n - b0);
    let sb = b0 % p.cap;
    // Scores: 8 lanes per key (vec4 slots lane, lane + 8, ...), 32 keys per pass, coalesced cache reads.
    for (var k0 = 0u; k0 < bn; k0 += WG / 8u) {
      let jj = k0 + key;
      var d = 0.0;
      if (jj < bn) {
        let kb = (wrap(sb + jj) * stride + ko) / 4u;
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

    let sc = select(-3.0e38, ss[li]${rel ? " + rel[bucket(i32(b0 + li) - i32(st.pos0 + t)) * p.H + h]" : ""}, li < bn);
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
      for (var jj = g; jj < bn; jj += groups) { acc += ss[jj] * ${r.v1("wrap(sb + jj) * stride + ko + i")}; }
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

export interface AttentionTile {
  /** Query rows per tile. */
  R: number;
  /** Keys per block. */
  BK: number;
  /** Threads per (row, key) dot product. */
  TPP: number;
}

let tileOverride: { R: number; BK: number } | undefined;
/** Forces the tiled attention's rows and keys per tile (for measurements); undefined restores the rule. */
export function setAttentionTile(t: { R: number; BK: number } | undefined): void {
  tileOverride = t;
}

/** Workgroup memory the tiled attention needs for a tile of R rows and BK keys. */
export function attentionTileBytes(D: number, R: number, BK: number): number {
  const tpp = Math.max(1, Math.floor(256 / (R * BK)));
  return 16 * R * (D / 4) + 16 * BK * (D / 4 + 1) + 4 * R * BK + (tpp > 1 ? 1024 : 0);
}

/**
 * Query rows and keys per workgroup tile in tiled prefill attention, sized to fit workgroup memory.
 * Head dim 256 (Gemma 3) takes 8 rows of 4 keys, two output rows per thread: on Gemma 3 4B's shapes
 * (128-token chunk at position 1024, Intel Gen9) 29-31 ms against 35-42 for 4 rows of 8 keys, where
 * more rows (16, 32) or more keys per block (8, 16) were slower and fewer keys (2) too.
 */
export function attentionTile(D: number, wgBytes = 16384): AttentionTile {
  const tpp = (R: number, BK: number) => Math.max(1, Math.floor(256 / (R * BK)));
  if (tileOverride) {
    if (attentionTileBytes(D, tileOverride.R, tileOverride.BK) > wgBytes - 256) throw new Error(`attention tile ${tileOverride.R}x${tileOverride.BK} does not fit ${wgBytes} bytes of workgroup memory`);
    return { ...tileOverride, TPP: tpp(tileOverride.R, tileOverride.BK) };
  }
  const BK = D >= 256 ? 4 : Math.min(64, Math.floor(2048 / D));
  let R = D >= 256 ? 8 : Math.min(32, Math.floor(1024 / D));
  while (R > 1 && attentionTileBytes(D, R, BK) > wgBytes - 256) R = Math.floor(R / 2);
  return { R, BK, TPP: tpp(R, BK) };
}

/** Query rows per thread (4 or 8) and head-dim chunk of span attention, sized to workgroup memory. */
export interface SpanTile {
  RT: 4 | 8;
  DC: number;
}

/** Workgroup vec4s span attention needs: the S-phase Q and K chunks, or the P and V sub-tiles. */
const spanVec4s = (D: number, t: SpanTile) => Math.max(t.DC * (2 * t.RT) + t.DC * 16, 16 * (2 * t.RT) + 16 * (D / 4), 32 * t.RT);

/**
 * 4 rows per thread and the largest head-dim chunk that fits. On the Intel Mac (D = 72, 4096 × 4096
 * span) RT 4 took 883 ms with DC 24 and 934 with DC 12; RT 8 took about 4 s (its 32 score and up to
 * 64 output registers per thread spill).
 */
export function spanTile(D: number, wgBytes = 16384): SpanTile | undefined {
  for (const DC of [32, 24, 16, 12, 8, 4]) {
    if (D % DC !== 0) continue;
    const t = { RT: 4 as const, DC };
    if (spanVec4s(D, t) * 16 <= wgBytes - 256) return t;
  }
  return undefined;
}

/**
 * Attention where every query row sees the same keys (a bidirectional span: the vision tower's 4096
 * patches), as a register-tiled GEMM with the softmax folded in, f32 cache. A 128-thread workgroup
 * takes R = 8·RT query rows of one head and walks the keys 64 at a time:
 *
 *  1. Scores: thread (tm, tn) = (li / 16, li % 16) holds rows tm·RT … + RT − 1 against keys
 *     4·tn … + 3 in registers, accumulated over head-dim chunks of DC staged in workgroup memory
 *     (Q and K transposed so each k step is one vec4 load per 4 rows or 4 keys), as gemm does.
 *  2. Softmax: each thread's per-row block max goes to workgroup memory (whole vec4s, rows as
 *     components); every thread of the row group reads the 16 and takes the same running max, so the
 *     exps (one per score, no redundancy) and each thread's partial exp-sums are consistent, and the
 *     partial sums are added up once at the end.
 *  3. Values: 16 keys at a time, the probabilities (rows as vec4) and the value rows are staged and
 *     thread (tm, tn) adds them into its RT rows × vec4 columns tn, tn + 16, ….
 *
 * Every key slot must be its position (no ring) and no window may cut the span; ops.attention picks
 * it only then. Against the tiled kernel this reuses every staged K and V value across 32 or 64 rows
 * and every staged Q value across 64 keys, with far fewer barriers per key.
 */
export function attentionSpanSpec(D: number, tile: SpanTile): KernelSpec {
  const { RT, DC } = tile;
  if (D % 4 !== 0 || D % DC !== 0 || DC % 4 !== 0) throw new Error(`span attention: head dim ${D} with chunk ${DC}`);
  const D4 = D / 4;
  const RV = RT / 4; // vec4s of rows per thread
  const R = 8 * RT;
  const RQ = R / 4; // vec4s of rows per tile
  const NS = Math.ceil(D4 / 16); // vec4 output columns per thread
  const QT = DC * RQ;
  const PT = 16 * RQ;
  const SMN = spanVec4s(D, tile);
  const unroll = (n: number, f: (i: number) => string) => Array.from({ length: n }, (_, i) => f(i)).join("\n");
  return spec(`attention/span/f32/D${D}/RT${RT}/DC${DC}`, ["uniform", "read", "read", "read", "read", "rw"], cpu.cpuAttention(false), () => /* wgsl */ `
${STEP_WGSL}
${ATTN_P("_a")}
${FIRST_KEY_WGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> kc: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> vc: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> o: array<vec4<f32>>;
const D4 = ${D4}u;
const DC = ${DC}u;
const DC4 = ${DC / 4}u;
const RQ = ${RQ}u;
const QT = ${QT}u;
const PT = ${PT}u;
var<workgroup> sm: array<vec4<f32>, ${SMN}>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wid.y;
  let kvh = h / (p.H / p.Hkv);
  let r0 = wid.x * ${R}u;
  let tm = li / 16u;
  let tn = li % 16u;
  let stride4 = p.Hkv * D4;
  let ko4 = kvh * D4;
  let hq4 = p.H * D4;
  // Every row sees the same keys: the first row's.
  let n = keys(st.pos0 + r0);
  let lo = firstKey(n, p.win);
  var m: array<f32, ${RT}>;
  var l: array<f32, ${RT}>;
  var acc: array<vec4<f32>, ${RT * NS}>;
  for (var r = 0u; r < ${RT}u; r++) { m[r] = -3.0e38; }

  for (var b0 = lo; b0 < n; b0 += 64u) {
    var s: array<vec4<f32>, ${RT}>;
    for (var c0 = 0u; c0 < D4; c0 += DC4) {
      workgroupBarrier();
      // Q chunk, transposed: item (row group g of 4, vec4 dq) writes the 4 dims' row vec4s.
      for (var it = li; it < RQ * DC4; it += 128u) {
        let g = it / DC4;
        let dq = it % DC4;
        var r4: array<vec4<f32>, 4>;
        for (var c = 0u; c < 4u; c++) {
          let row = r0 + g * 4u + c;
          r4[c] = select(vec4<f32>(0.0), q[min(row, p.T - 1u) * hq4 + h * D4 + c0 + dq] * p.scale, row < p.T);
        }
        for (var e = 0u; e < 4u; e++) { sm[(dq * 4u + e) * RQ + g] = vec4<f32>(r4[0][e], r4[1][e], r4[2][e], r4[3][e]); }
      }
      // K chunk likewise, key groups of 4.
      for (var it = li; it < 16u * DC4; it += 128u) {
        let g = it / DC4;
        let dq = it % DC4;
        var r4: array<vec4<f32>, 4>;
        for (var c = 0u; c < 4u; c++) {
          let j = b0 + g * 4u + c;
          r4[c] = select(vec4<f32>(0.0), kc[min(j, n - 1u) * stride4 + ko4 + c0 + dq], j < n);
        }
        for (var e = 0u; e < 4u; e++) { sm[QT + (dq * 4u + e) * 16u + g] = vec4<f32>(r4[0][e], r4[1][e], r4[2][e], r4[3][e]); }
      }
      workgroupBarrier();
      for (var k = 0u; k < DC; k++) {
        let bv = sm[QT + k * 16u + tn];
${unroll(RV, (v) => `        let a${v} = sm[k * RQ + tm * ${RV}u + ${v}u];
        s[${v * 4}] += a${v}.x * bv;
        s[${v * 4 + 1}] += a${v}.y * bv;
        s[${v * 4 + 2}] += a${v}.z * bv;
        s[${v * 4 + 3}] += a${v}.w * bv;`)}
      }
    }
    // Keys past n score nothing; each row's max over this thread's 4 keys.
    let kb = b0 + tn * 4u;
    var pm: array<f32, ${RT}>;
    for (var r = 0u; r < ${RT}u; r++) {
      for (var c = 0u; c < 4u; c++) { if (kb + c >= n) { s[r][c] = -3.0e38; } }
      pm[r] = max(max(s[r].x, s[r].y), max(s[r].z, s[r].w));
    }
    workgroupBarrier();
${unroll(RV, (v) => `    sm[(tm * ${RV}u + ${v}u) * 16u + tn] = vec4<f32>(pm[${v * 4}], pm[${v * 4 + 1}], pm[${v * 4 + 2}], pm[${v * 4 + 3}]);`)}
    workgroupBarrier();
    var al: array<f32, ${RT}>;
    for (var v = 0u; v < ${RV}u; v++) {
      var mx = vec4<f32>(-3.0e38);
      for (var t = 0u; t < 16u; t++) { mx = max(mx, sm[(tm * ${RV}u + v) * 16u + t]); }
      for (var e = 0u; e < 4u; e++) {
        let r = v * 4u + e;
        let mn = max(m[r], mx[e]);
        al[r] = exp(m[r] - mn);
        m[r] = mn;
      }
    }
    for (var r = 0u; r < ${RT}u; r++) {
      s[r] = exp(s[r] - vec4<f32>(m[r]));
      l[r] = l[r] * al[r] + (s[r].x + s[r].y) + (s[r].z + s[r].w);
      for (var t = 0u; t < ${NS}u; t++) { acc[r * ${NS}u + t] *= al[r]; }
    }
    for (var ss = 0u; ss < 4u; ss++) {
      workgroupBarrier();
      // Probabilities of keys 16·ss … + 15, from the 4 threads that hold them, rows as vec4.
      if (tn / 4u == ss) {
        for (var c = 0u; c < 4u; c++) {
          let kk = (tn % 4u) * 4u + c;
${unroll(RV, (v) => `          sm[kk * RQ + tm * ${RV}u + ${v}u] = vec4<f32>(s[${v * 4}][c], s[${v * 4 + 1}][c], s[${v * 4 + 2}][c], s[${v * 4 + 3}][c]);`)}
        }
      }
      for (var it = li; it < 16u * D4; it += 128u) {
        let j = b0 + ss * 16u + it / D4;
        sm[PT + it] = select(vec4<f32>(0.0), vc[min(j, n - 1u) * stride4 + ko4 + it % D4], j < n);
      }
      workgroupBarrier();
      for (var kk = 0u; kk < 16u; kk++) {
${unroll(RV, (v) => `        let a${v} = sm[kk * RQ + tm * ${RV}u + ${v}u];`)}
        for (var t = 0u; t < ${NS}u; t++) {
          let c = tn + 16u * t;
          if (c < D4) {
            let vv = sm[PT + kk * D4 + c];
${unroll(RV, (v) => `            acc[${v * 4} * ${NS} + t] += a${v}.x * vv;
            acc[${v * 4 + 1} * ${NS} + t] += a${v}.y * vv;
            acc[${v * 4 + 2} * ${NS} + t] += a${v}.z * vv;
            acc[${v * 4 + 3} * ${NS} + t] += a${v}.w * vv;`)}
          }
        }
      }
    }
  }
  // Rows' exp-sums: the 16 threads' partial sums added up.
  workgroupBarrier();
${unroll(RV, (v) => `  sm[(tm * ${RV}u + ${v}u) * 16u + tn] = vec4<f32>(l[${v * 4}], l[${v * 4 + 1}], l[${v * 4 + 2}], l[${v * 4 + 3}]);`)}
  workgroupBarrier();
  for (var v = 0u; v < ${RV}u; v++) {
    var ls = vec4<f32>(0.0);
    for (var t = 0u; t < 16u; t++) { ls += sm[(tm * ${RV}u + v) * 16u + t]; }
    for (var e = 0u; e < 4u; e++) {
      let r = v * 4u + e;
      let row = r0 + tm * ${RT}u + r;
      if (row < p.T) {
        for (var t = 0u; t < ${NS}u; t++) {
          let c = tn + 16u * t;
          if (c < D4) { o[(row * p.H + h) * D4 + c] = acc[r * ${NS}u + t] / ls[e]; }
        }
      }
    }
  }
}`);
}

/**
 * Tiled causal prefill attention with GQA: one 256-thread workgroup per (kv head, tile of R query rows),
 * where the rows are (token, head) pairs of every query head that shares the kv head. Each block of BK
 * keys is staged once in workgroup memory and used by all R rows, then its values likewise, so the cache
 * is read once per tile instead of once per (head, token).
 *
 * Scores: TPR threads per row, each scoring every TPR-th key of the block. With large heads the tile has
 * few (row, key) pairs (32 for D = 256), so each dot product is split across TPP threads instead and
 * their partial sums added in workgroup memory. Values: thread (row, d4) owns
 * four output columns as a vec4 and folds the online softmax into its pass over the block (block max,
 * rescale, exp, exp-sum), so the row statistics live in registers and need no extra barrier.
 */
export function attentionTiledSpec(kv: KvDtype, D: number, tile: AttentionTile = attentionTile(D)): KernelSpec {
  const { R, BK, TPP: TPP_ } = tile;
  // Output slots (row, vec4 column) per thread: one while R * D / 4 fits 256 threads, more beyond.
  const ORR = Math.ceil((R * (D / 4)) / 256);
  const split = TPP_ > 1;
  const D4 = D / 4;
  const KP = D4 + 1; // padded row stride (vec4) of the key/value tile, so rows start in different banks
  const TPR = Math.floor(256 / R);
  const load =
    kv === "f16"
      ? { kDecl: "array<vec2<u32>>", vDecl: "array<vec2<u32>>", k: "kv4(kc[i])", v: "kv4(vc[i])" }
      : { kDecl: "array<vec4<f32>>", vDecl: "array<vec4<f32>>", k: "kc[i]", v: "vc[i]" };
  return spec(`attention/tiled/${kv}/D${D}/R${R}/BK${BK}`, ["uniform", "read", "read", "read", "read", "rw"], cpu.cpuAttention(kv === "f16"), () => /* wgsl */ `
${STEP_WGSL}
${kv === "f16" ? KV4_FN : ""}
${ATTN_P("_a")}
${FIRST_KEY_WGSL}
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
${split ? `var<workgroup> part: array<f32, 256>;
const TPP = ${TPP_}u;` : ""}

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
  let nMax = keys(st.pos0 + (min(r0 + R, rows) - 1u) / G);
  // Scoring role: row sr, keys sl, sl + TPR, ...
  let sr = li / TPR;
  let sl = li % TPR;
  let sn = keys(st.pos0 + (r0 + sr) / G);
  let slo = firstKey(sn, p.win);
  // The tile's first row sees the earliest keys (rows are ordered by position).
  let lo = firstKey(keys(st.pos0 + r0 / G), p.win);
  // Output role: slots e = li + 256 k, each row e / D4, columns 4 * (e % D4) .. + 3, with its own
  // running max and exp-sum.
  var m: array<f32, ${ORR}>;
  var l: array<f32, ${ORR}>;
  var acc: array<vec4<f32>, ${ORR}>;
  for (var k = 0u; k < ${ORR}u; k++) { m[k] = -3.0e38; }
  workgroupBarrier();

  for (var b0 = lo; b0 < nMax; b0 += BK) {
    let sb = b0 % p.cap;
    for (var e = li; e < BK * D4; e += 256u) {
      let j = b0 + e / D4;
      var v = vec4<f32>(0.0);
      if (j < nMax) { v = K4(wrap(sb + e / D4) * stride4 + ko4 + e % D4); }
      tile[(e / D4) * KP + e % D4] = v;
    }
    workgroupBarrier();
${split
  ? `    {
      // Thread li adds every TPP-th vec4 of pair li / TPP's dot product.
      let pr = li / TPP;
      var d = 0.0;
      if (pr < R * BK) {
        let rr = pr / BK;
        let jj = pr % BK;
        for (var c = li % TPP; c < D4; c += TPP) { d += dot(qs[rr * D4 + c], tile[jj * KP + c]); }
      }
      part[li] = d;
    }
    workgroupBarrier();
    if (li < R * BK) {
      let rr = li / BK;
      let jj = li % BK;
      let n = keys(st.pos0 + (r0 + rr) / G);
      var d = -3.0e38;
      if (b0 + jj < n && b0 + jj >= firstKey(n, p.win)) {
        d = 0.0;
        for (var t = 0u; t < TPP; t++) { d += part[li * TPP + t]; }
      }
      sc[li] = d;
    }`
  : `    if (sr < R) {
      for (var jj = sl; jj < BK; jj += TPR) {
        var d = -3.0e38;
        if (b0 + jj < sn && b0 + jj >= slo) {
          d = 0.0;
          for (var c = 0u; c < D4; c++) { d += dot(qs[sr * D4 + c], tile[jj * KP + c]); }
        }
        sc[sr * BK + jj] = d;
      }
    }`}
    workgroupBarrier();
    for (var e = li; e < BK * D4; e += 256u) {
      let j = b0 + e / D4;
      var v = vec4<f32>(0.0);
      if (j < nMax) { v = V4(wrap(sb + e / D4) * stride4 + ko4 + e % D4); }
      tile[(e / D4) * KP + e % D4] = v;
    }
    workgroupBarrier();
    for (var k = 0u; k < ${ORR}u; k++) {
      let slot = li + k * 256u;
      if (slot < R * D4) {
        let orow = slot / D4;
        let od = slot % D4;
        var mx = m[k];
        for (var jj = 0u; jj < BK; jj++) { mx = max(mx, sc[orow * BK + jj]); }
        let corr = exp(m[k] - mx);
        m[k] = mx;
        var bl = 0.0;
        var ba = vec4<f32>(0.0);
        for (var jj = 0u; jj < BK; jj++) {
          let e = exp(sc[orow * BK + jj] - mx);
          bl += e;
          ba += e * tile[jj * KP + od];
        }
        l[k] = l[k] * corr + bl;
        acc[k] = acc[k] * corr + ba;
      }
    }
    workgroupBarrier();
  }

  for (var k = 0u; k < ${ORR}u; k++) {
    let slot = li + k * 256u;
    let orow = slot / D4;
    let rho = r0 + orow;
    if (slot < R * D4 && rho < rows) {
      let ob = ((rho / G) * p.H + kvh * G + rho % G) * D + (slot % D4) * 4u;
      let v = acc[k] / l[k];
      o[ob] = v.x;
      o[ob + 1u] = v.y;
      o[ob + 2u] = v.z;
      o[ob + 3u] = v.w;
    }
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
  return spec(`attention/split/${kv}`, ["uniform", "read", "read", "read", "read", "rw", "rw"], undefined, () => /* wgsl */ `
${STEP_WGSL}
${kv === "f16" ? KV4_FN : ""}
${ATTN_P("S")}
${FIRST_KEY_WGSL}
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
  let n = keys(st.pos0 + t);
  // Keys before the sliding window are skipped: splits wholly before it exit, and the first live split
  // starts at the window's first key. Combine reads the same live splits.
  let lo = firstKey(n, p.win);
  let j0s = split * CH;
  if (j0s >= n || j0s + CH <= lo) { return; }
  let j0 = max(j0s, lo);
  let cnt = min(j0s + CH, n) - j0;
  let sb = j0 % p.cap;
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
      let kb = (wrap(sb + j) * stride + ko) / 4u;
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
  let vo = ko / 2u;
  for (var pi = li % pairs; pi < pairs; pi += 64u) {
    var acc = vec2<f32>(0.0);
    if (g < groups) {
      for (var j = g; j < cnt; j += groups) { acc += sc[j] * unpack2x16float(vc[vo + wrap(sb + j) * vs + pi]); }
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
    for (var j = 0u; j < cnt; j++) { acc += sc[j] * ${r.v1("wrap(sb + j) * stride + ko + i")}; }
    po[row * p.D + i] = acc;
  }`}
}`);
}

/** Split-K attention, pass 2: rescales every split to the global max and normalises. */
export function attentionCombineSpec(): KernelSpec {
  return spec("attention/combine", ["uniform", "read", "read", "read", "rw"], undefined, () => /* wgsl */ `
${STEP_WGSL}
${ATTN_P("S")}
${FIRST_KEY_WGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> ml: array<f32>;
@group(0) @binding(3) var<storage, read> po: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let base = (wid.y * p.H + wid.x) * p.S;
  let n = keys(st.pos0 + wid.y);
  let live = (n + ${ATTN_CHUNK - 1}u) / ${ATTN_CHUNK}u;
  let first = firstKey(n, p.win) / ${ATTN_CHUNK}u;
  var mx = -3.0e38;
  for (var s = first; s < live; s++) { mx = max(mx, ml[(base + s) * 2u]); }
  var den = 0.0;
  for (var s = first; s < live; s++) { den += exp(ml[(base + s) * 2u] - mx) * ml[(base + s) * 2u + 1u]; }
  let out = (wid.y * p.H + wid.x) * p.D;
  for (var i = li; i < p.D; i += 64u) {
    var num = 0.0;
    for (var s = first; s < live; s++) { num += exp(ml[(base + s) * 2u] - mx) * po[(base + s) * p.D + i]; }
    o[out + i] = num / den;
  }
}`);
}

/** out = act(gate) * up, with SiLU (default) or tanh-approximated GELU. */
export function siluMulSpec(act: "silu" | "gelu" = "silu"): KernelSpec {
  return spec(`${act}_mul`, ["uniform", "read", "read", "rw"], cpu.cpuSiluMul(act), () => /* wgsl */ `
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
  o[idx] = ${actWgsl(act, "a")} * u[idx];
}`);
}

/** y[r, i] = act(x[r, i]) * x[r, I + i]: a gated activation over [gate | up] rows of width 2I. */
export function gatedActSpec(act: "silu" | "gelu"): KernelSpec {
  return spec(`gated_${act}`, ["uniform", "read", "rw"], cpu.cpuGatedAct(act), () => /* wgsl */ `
struct P { n: u32, I: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.n) { return; }
  let r = idx / p.I;
  let i = idx % p.I;
  let a = x[r * 2u * p.I + i];
  y[idx] = ${actWgsl(act, "a")} * x[r * 2u * p.I + p.I + i];
}`);
}

/** a += b, with b broadcast over rows when `cols` < n (bias add). */
export function addSpec(): KernelSpec {
  return spec("add", ["uniform", "rw", "read"], cpu.cpuAdd, () => /* wgsl */ `
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
export function embedSpec(wdtype: WeightDType, affine = false, features = false): KernelSpec {
  const quant = wdtype !== "f32";
  const aff = quant && affine;
  const bindings: BindingKind[] = quant ? ["uniform", "read", "read", "read"] : ["uniform", "read", "read"];
  if (aff) bindings.push("read");
  const feat = bindings.length;
  if (features) bindings.push("read");
  const out = bindings.length;
  bindings.push("rw");
  return spec(`embed/${wdtype}${aff ? "a" : ""}${features ? "/features" : ""}`, bindings, cpu.cpuEmbed(wdtype, aff, features), () => /* wgsl */ `
struct P { T: u32, D: u32, scale: f32, V: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
${quant
  ? `@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> ws: array<f32>;${aff ? "\n@group(0) @binding(4) var<storage, read> wm: array<f32>;" : ""}`
  : `@group(0) @binding(2) var<storage, read> w: array<f32>;`}
${features ? `@group(0) @binding(${feat}) var<storage, read> feat: array<f32>;` : ""}
@group(0) @binding(${out}) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.T * p.D) { return; }
${features
  ? `  // Ids from V on are rows of the image features (unscaled), as Gemma 3 places them in the prompt.
  let id = tokens[idx / p.D];
  if (id >= p.V) { o[idx] = feat[(id - p.V) * p.D + idx % p.D]; return; }`
  : ""}
  let e = tokens[idx / p.D] * p.D + idx % p.D;
${wdtype === "q4"
  ? aff
    ? `  let q = f32((wq[e / 8u] >> (4u * (e % 8u))) & 15u);
  o[idx] = (q * ws[e / 32u] + wm[e / 32u]) * p.scale;`
    : `  let q = f32((wq[e / 8u] >> (4u * (e % 8u))) & 15u) - 8.0;
  o[idx] = q * ws[e / 32u] * p.scale;`
  : quant
  ? `  let q = f32(bitcast<i32>(wq[e / 4u] << (24u - 8u * (e % 4u))) >> 24u);
  o[idx] = (q * ws[e / 32u]${aff ? " + wm[e / 32u]" : ""}) * p.scale;`
  : `  o[idx] = w[e] * p.scale;`}
}`);
}

/** y = (x - mean) / sqrt(var + eps) * w + b over rows of D (LayerNorm, the vision encoder's norm). */
export function layernormSpec(): KernelSpec {
  return spec("layernorm", ["uniform", "read", "read", "read", "rw"], cpu.cpuLayernorm, () => /* wgsl */ `
struct P { R: u32, D: u32, eps: f32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

fn total(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let t = red[0];
  workgroupBarrier();
  return t;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let o = (wid.x + wid.y * ${MAX_X}u) * p.D;
  if (wid.x + wid.y * ${MAX_X}u >= p.R) { return; }
  var s = 0.0;
  for (var i = li; i < p.D; i += 256u) { s += x[o + i]; }
  let mean = total(li, s) / f32(p.D);
  var v = 0.0;
  for (var i = li; i < p.D; i += 256u) { let d = x[o + i] - mean; v += d * d; }
  let inv = inverseSqrt(total(li, v) / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { y[o + i] = (x[o + i] - mean) * inv * w[i] + b[i]; }
}`);
}

/** x = gelu_tanh(x) in place. */
/** Elementwise activations: GELU exact (erf) or tanh-approximated, SiLU, ReLU, CLIP's quick GELU (x·σ(1.702x)). */
export type Activation = "gelu" | "gelu_tanh" | "silu" | "relu" | "quick_gelu";

/**
 * An activation in place. Exact GELU takes erf from Abramowitz & Stegun 7.1.26 (absolute error under
 * 1.5e-7), the same formula as the CPU kernel.
 */
export function actSpec(kind: Activation): KernelSpec {
  const expr =
    kind === "gelu"
      ? "0.5 * a * (1.0 + erf_as(a * 0.70710678118654752))"
      : kind === "gelu_tanh"
        ? GELU_WGSL("a")
        : kind === "silu"
          ? "a / (1.0 + exp(-a))"
          : kind === "quick_gelu"
            ? "a / (1.0 + exp(-1.702 * a))"
            : "max(a, 0.0)";
  return spec(kind === "gelu_tanh" ? "gelu" : `act/${kind}`, ["uniform", "rw"], cpu.cpuAct(kind), () => /* wgsl */ `
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
${kind === "gelu" ? `
fn erf_as(v: f32) -> f32 {
  let z = abs(v);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return select(-y, y, v >= 0.0);
}
` : ""}
@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.n) { return; }
  let a = x[idx];
  x[idx] = ${expr};
}`);
}

/** GELU in place (tanh approximation, or exact with `exact`); see `actSpec`. */
export const geluSpec = (exact = false): KernelSpec => actSpec(exact ? "gelu" : "gelu_tanh");

/**
 * Average pooling of a square grid of tokens: x [side·side, C] in raster order → y [(side/k)², C],
 * each output the mean of a k × k patch (Gemma 3's projector: 64 × 64 patches → 16 × 16 tokens).
 */
export function avgPoolSpec(): KernelSpec {
  return spec("avg_pool", ["uniform", "read", "rw"], cpu.cpuAvgPool, () => /* wgsl */ `
struct P { side: u32, k: u32, C: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  let out = p.side / p.k;
  if (idx >= out * out * p.C) { return; }
  let c = idx % p.C;
  let t = idx / p.C;
  let y0 = (t / out) * p.k;
  let x0 = (t % out) * p.k;
  var s = 0.0;
  for (var dy = 0u; dy < p.k; dy++) {
    for (var dx = 0u; dx < p.k; dx++) { s += x[((y0 + dy) * p.side + x0 + dx) * p.C + c]; }
  }
  y[idx] = s / f32(p.k * p.k);
}`);
}

/**
 * 1-D convolution inputs as matmul rows (im2col): x [Tin, C] → y [Tout, row], where row t holds the
 * k kernel taps' C channels, tap-major (y[t, j·C + c] = x[t·stride + j − pad, c], zero outside the
 * input), and zeros up to `row` (a multiple of 32 for the matmul kernels).
 */
export function im2col1dSpec(): KernelSpec {
  return spec("im2col1d", ["uniform", "read", "rw"], cpu.cpuIm2col1d, () => /* wgsl */ `
struct P { tin: u32, tout: u32, C: u32, k: u32, stride: u32, pad: u32, row: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.tout * p.row) { return; }
  let t = idx / p.row;
  let r = idx % p.row;
  var v = 0.0;
  if (r < p.k * p.C) {
    let src = i32(t * p.stride + r / p.C) - i32(p.pad);
    if (src >= 0 && u32(src) < p.tin) { v = x[u32(src) * p.C + r % p.C]; }
  }
  y[idx] = v;
}`);
}

/**
 * Per-channel normalization over time (GroupNorm with one group per channel): x [T, C] → y, each
 * column scaled to zero mean and unit variance, then by w[c] and b[c]. One workgroup per channel.
 */
export function channelNormSpec(): KernelSpec {
  return spec("channel_norm", ["uniform", "read", "read", "read", "rw"], cpu.cpuChannelNorm, () => /* wgsl */ `
struct P { T: u32, C: u32, eps: f32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

fn total(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let t = red[0];
  workgroupBarrier();
  return t;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c = wid.x;
  var s = 0.0;
  for (var t = li; t < p.T; t += 256u) { s += x[t * p.C + c]; }
  let mean = total(li, s) / f32(p.T);
  var v = 0.0;
  for (var t = li; t < p.T; t += 256u) { let d = x[t * p.C + c] - mean; v += d * d; }
  let inv = inverseSqrt(total(li, v) / f32(p.T) + p.eps);
  for (var t = li; t < p.T; t += 256u) { y[t * p.C + c] = (x[t * p.C + c] - mean) * inv * w[c] + b[c]; }
}`);
}

/**
 * Grouped 1-D convolution: x [Tin, C] → y [Tout, O], y[t, o] = bias[o] + Σ_j Σ_c x[t + j − pad, g·Cg + c] ·
 * w[(j·Cg + c)·O + o] for o's group g (Cg = C / groups input channels per group), zero outside the input.
 * The weight is stored [k, Cg, O] so neighbouring outputs read neighbouring weights.
 */
export function groupConv1dSpec(): KernelSpec {
  return spec("group_conv1d", ["uniform", "read", "read", "read", "rw"], cpu.cpuGroupConv1d, () => /* wgsl */ `
struct P { tin: u32, tout: u32, C: u32, O: u32, k: u32, pad: u32, groups: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.tout * p.O) { return; }
  let t = idx / p.O;
  let o = idx % p.O;
  let cg = p.C / p.groups;
  let c0 = (o / (p.O / p.groups)) * cg;
  var s = bias[o];
  for (var j = 0u; j < p.k; j++) {
    let src = i32(t + j) - i32(p.pad);
    if (src < 0 || u32(src) >= p.tin) { continue; }
    let xo = u32(src) * p.C + c0;
    let wo = j * cg * p.O + o;
    for (var c = 0u; c < cg; c++) { s += x[xo + c] * w[wo + c * p.O]; }
  }
  y[idx] = s;
}`);
}

/** out[1] = the bits of logits[out[0]]: the value of the token the sampler just chose. */
export function pickLogitSpec(): KernelSpec {
  return spec("pick_logit", ["read", "rw"], cpu.cpuPickLogit, () => /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1)
fn main() {
  out[1] = bitcast<u32>(logits[out[0]]);
}`);
}

/**
 * Attention probabilities of selected heads for one query row: softmax(q_h · K_h^T · scale) over S keys,
 * written to out[(st.pos0 · nSlots + slotBase + i) · S + j] for the i-th head set in `mask`. Whisper's
 * word timestamps align tokens to audio through these cross-attention weights.
 */
export function attnProbsSpec(): KernelSpec {
  return spec("attn_probs", ["uniform", "read", "read", "read", "rw"], cpu.cpuAttnProbs, () => /* wgsl */ `
${STEP_WGSL}
struct P { S: u32, D: u32, row: u32, mask: u32, scale: f32, slotBase: u32, nSlots: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  var h = 0u;
  var seen = 0u;
  for (var b = 0u; b < 32u; b++) {
    if (((p.mask >> b) & 1u) == 1u) {
      if (seen == wg.y) { h = b; }
      seen++;
    }
  }
  let base = (st.pos0 * p.nSlots + p.slotBase + wg.y) * p.S;
  var m = -3.4e38;
  for (var j = lid.x; j < p.S; j += 256u) {
    var s = 0.0;
    for (var d = 0u; d < p.D; d++) { s += q[h * p.D + d] * k[j * p.row + h * p.D + d]; }
    s *= p.scale;
    out[base + j] = s;
    m = max(m, s);
  }
  red[lid.x] = m;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) {
    if (lid.x < w) { red[lid.x] = max(red[lid.x], red[lid.x + w]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lid.x; j < p.S; j += 256u) {
    let e = exp(out[base + j] - mx);
    out[base + j] = e;
    sum += e;
  }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) {
    if (lid.x < w) { red[lid.x] += red[lid.x + w]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var j = lid.x; j < p.S; j += 256u) { out[base + j] *= inv; }
}`);
}

/**
 * Causal attention over a batch of sequences packed as blocks of T rows: q, k, v [B·T, H·D]; row
 * b·T + t sees keys b·T … b·T + t of its own block. One workgroup per (head, row); T ≤ 256.
 */
export function blockAttentionSpec(): KernelSpec {
  return spec("block_attention", ["uniform", "read", "read", "read", "rw"], cpu.cpuBlockAttention, () => /* wgsl */ `
struct P { T: u32, D: u32, row: u32, scale: f32, rows: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let r = wg.y;
  let k0 = r - r % p.T;
  let n = r % p.T + 1u;
  let qo = r * p.row + h * p.D;
  var m = -3.4e38;
  for (var j = lid.x; j < n; j += 64u) {
    let ko = (k0 + j) * p.row + h * p.D;
    var s = 0.0;
    for (var d = 0u; d < p.D; d++) { s += q[qo + d] * k[ko + d]; }
    s *= p.scale;
    sc[j] = s;
    m = max(m, s);
  }
  red[lid.x] = m;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) {
    if (lid.x < w) { red[lid.x] = max(red[lid.x], red[lid.x + w]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lid.x; j < n; j += 64u) {
    let e = exp(sc[j] - mx);
    sc[j] = e;
    sum += e;
  }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) {
    if (lid.x < w) { red[lid.x] += red[lid.x + w]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var d = lid.x; d < p.D; d += 64u) {
    var acc = 0.0;
    for (var j = 0u; j < n; j++) { acc += sc[j] * v[(k0 + j) * p.row + h * p.D + d]; }
    out[qo + d] = acc * inv;
  }
}`);
}

/**
 * 2-D convolution as an implicit GEMM: x [H·W, C] (pixels in raster order, channels last, C a
 * multiple of 4) → y [Ho·Wo, O], y[m, n] = act(bias[n] + Σ_k patch(m)[k] · w[n, k] (+ res[m, n])),
 * where patch row k = (ky·kw + kx)·C + c (tap-major, as the weight [O, K] is stored, K the taps' C
 * channels rounded up to a multiple of 32 with zero columns) and taps outside the image read zero.
 * The tiling is gemm's (64 pixels × 64 channels per 128-thread workgroup, 8 × 4 per thread, K one
 * 32-element block at a time); only the x tile's loads differ, gathering each pixel's patch from the
 * image instead of reading a stored im2col row. Batch-norm is folded into w and bias at load time.
 * With `reluIn` the input goes through ReLU as it is read (pre-activation residual blocks). A batch of
 * N images ([N·H·W, C] → [N·Ho·Wo, O]) runs as one GEMM of N·Ho·Wo rows. Taps are `dil` pixels apart
 * (dilated convolution).
 */
export function conv2dSpec(v: { relu: boolean; residual: boolean; reluIn?: boolean }): KernelSpec {
  const R8 = [0, 1, 2, 3, 4, 5, 6, 7];
  const bindings: BindingKind[] = ["uniform", "read", "read", "read"];
  if (v.residual) bindings.push("read");
  bindings.push("rw");
  const out = (val: string) => (v.relu ? `max(${val}, 0.0)` : val);
  const store = R8.map(
    (i) => `  if (m0 + tm * 8u + ${i}u < p.M) {
    let m = m0 + tm * 8u + ${i}u;
${[0, 1, 2, 3].map((j) => `    if (n0 + tn * 4u + ${j}u < p.O) { let n = n0 + tn * 4u + ${j}u; y[m * p.O + n] = ${out(`acc${i}[${j}] + bias[n]${v.residual ? " + res[m * p.O + n]" : ""}`)}; }`).join("\n")}
  }`,
  ).join("\n");
  return spec(`conv2d${v.reluIn ? "/relu-in" : ""}${v.relu ? "/relu" : ""}${v.residual ? "/res" : ""}`, bindings, cpu.cpuConv2d(v), () => /* wgsl */ `
struct P { H: u32, W: u32, C: u32, O: u32, Ho: u32, Wo: u32, kh: u32, kw: u32, stride: u32, pad: u32, K: u32, M: u32, N: u32, dil: u32, _c: u32, _d: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
${v.residual ? "@group(0) @binding(4) var<storage, read> res: array<f32>;\n@group(0) @binding(5)" : "@group(0) @binding(4)"} var<storage, read_write> y: array<f32>;
var<workgroup> xt: array<vec4<f32>, 512>;
var<workgroup> wt: array<vec4<f32>, 512>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n0 = (wid.x + wid.z * ${MM_SLICE}u) * 64u;
  let m0 = wid.y * 64u;
  let tn = li % 16u;
  let tm = li / 16u;
  let blocks = p.K / 32u;
  let C4 = p.C / 4u;
  let taps = p.kh * p.kw;
  // The top-left input pixel of this thread's 4 staged output pixels.
  var oy: array<i32, 4>;
  var ox: array<i32, 4>;
  var live: array<bool, 4>;
  var base: array<u32, 4>;
  let plane = p.Ho * p.Wo;
  for (var c = 0u; c < 4u; c++) {
    let m = m0 + (li / 8u) * 4u + c;
    let mm = min(m, p.M - 1u);
    let r = mm % plane;
    live[c] = m < p.M;
    base[c] = (mm / plane) * p.H * p.W;
    oy[c] = i32((r / p.Wo) * p.stride) - i32(p.pad);
    ox[c] = i32((r % p.Wo) * p.stride) - i32(p.pad);
  }
${R8.map((i) => `  var acc${i} = vec4<f32>();`).join("\n")}
  var xr: array<vec4<f32>, 4>;
  var wr4: array<vec4<f32>, 4>;
  for (var kb = 0u; kb < blocks; kb++) {
    let k = kb * 32u + (li % 8u) * 4u;
    let tap = k / p.C;
    let c4 = (k % p.C) / 4u;
    let ky = i32((tap / p.kw) * p.dil);
    let kx = i32((tap % p.kw) * p.dil);
    for (var c = 0u; c < 4u; c++) {
      let iy = oy[c] + ky;
      let ix = ox[c] + kx;
      let inside = live[c] && tap < taps && iy >= 0 && iy < i32(p.H) && ix >= 0 && ix < i32(p.W);
      let at = (base[c] + u32(clamp(iy, 0, i32(p.H) - 1)) * p.W + u32(clamp(ix, 0, i32(p.W) - 1))) * C4 + min(c4, C4 - 1u);
      xr[c] = select(vec4<f32>(0.0), ${v.reluIn ? "max(x[at], vec4<f32>(0.0))" : "x[at]"}, inside);
    }
    for (var c = 0u; c < 4u; c++) {
      let row = n0 + (li / 8u) * 4u + c;
      wr4[c] = select(vec4<f32>(0.0), w[(min(row, p.O - 1u) * blocks + kb) * 8u + li % 8u], row < p.O);
    }
    for (var e = 0u; e < 4u; e++) {
      xt[((li % 8u) * 4u + e) * 16u + li / 8u] = vec4<f32>(xr[0][e], xr[1][e], xr[2][e], xr[3][e]);
      wt[((li % 8u) * 4u + e) * 16u + li / 8u] = vec4<f32>(wr4[0][e], wr4[1][e], wr4[2][e], wr4[3][e]);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
      let a0 = xt[kk * 16u + tm * 2u];
      let a1 = xt[kk * 16u + tm * 2u + 1u];
      let bv = wt[kk * 16u + tn];
${R8.map((i) => `      acc${i} += a${i >> 2}.${"xyzw"[i & 3]} * bv;`).join("\n")}
    }
    workgroupBarrier();
  }
${store}
}`);
}

/**
 * Bilinear resize of an image x [H·W, C] (raster order, channels last) to y [Ho·Wo, C], as torch's
 * interpolate(mode="bilinear") does it: with align_corners the corner pixels map onto each other
 * (source = dst · (in − 1) / (out − 1)); without, pixel centers do (source = (dst + 0.5) · in / out
 * − 0.5, clamped at 0). The far neighbour is clamped to the edge.
 */
export function resizeBilinearSpec(): KernelSpec {
  return spec("resize_bilinear", ["uniform", "read", "rw"], cpu.cpuResizeBilinear, () => /* wgsl */ `
struct P { H: u32, W: u32, C: u32, Ho: u32, Wo: u32, align: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

fn src(d: u32, n: u32, o: u32) -> f32 {
  if (p.align == 1u) { return select(0.0, f32(d) * (f32(n - 1u) / f32(o - 1u)), o > 1u); }
  return max((f32(d) + 0.5) * (f32(n) / f32(o)) - 0.5, 0.0);
}

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.Ho * p.Wo * p.C) { return; }
  let c = idx % p.C;
  let t = idx / p.C;
  let sy = src(t / p.Wo, p.H, p.Ho);
  let sx = src(t % p.Wo, p.W, p.Wo);
  let y0 = min(u32(sy), p.H - 1u);
  let x0 = min(u32(sx), p.W - 1u);
  let y1 = min(y0 + 1u, p.H - 1u);
  let x1 = min(x0 + 1u, p.W - 1u);
  let ly = sy - f32(y0);
  let lx = sx - f32(x0);
  let top = (1.0 - lx) * x[(y0 * p.W + x0) * p.C + c] + lx * x[(y0 * p.W + x1) * p.C + c];
  let bot = (1.0 - lx) * x[(y1 * p.W + x0) * p.C + c] + lx * x[(y1 * p.W + x1) * p.C + c];
  y[idx] = (1.0 - ly) * top + ly * bot;
}`);
}

/**
 * Transposed convolution whose kernel equals its stride f (no overlap, no padding): x [H·W, C] →
 * y [(H·f)·(W·f), O], each input pixel spreading into an f × f block, y[(oy, ox), o] = bias[o] +
 * Σ_c x[(oy / f, ox / f), c] · w[((oy % f)·f + ox % f), o, c] (weight stored [f·f, O, C]).
 */
export function convTransposeSpec(): KernelSpec {
  return spec("conv_transpose", ["uniform", "read", "read", "read", "rw"], cpu.cpuConvTranspose, () => /* wgsl */ `
struct P { H: u32, W: u32, C: u32, O: u32, f: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  let Wo = p.W * p.f;
  if (idx >= p.H * p.f * Wo * p.O) { return; }
  let o = idx % p.O;
  let t = idx / p.O;
  let oy = t / Wo;
  let ox = t % Wo;
  let xo = ((oy / p.f) * p.W + ox / p.f) * p.C;
  let wo = (((oy % p.f) * p.f + ox % p.f) * p.O + o) * p.C;
  var s = bias[o];
  for (var c = 0u; c < p.C; c++) { s += x[xo + c] * w[wo + c]; }
  y[idx] = s;
}`);
}

/**
 * GroupNorm over a batch of images x [N, T, C] (T pixels, channels last): for each image and each of
 * G groups of C / G channels, normalized over the group's T · C / G values (biased variance), then
 * scaled by w[c] and shifted by b[c], ReLU after with `relu`. One workgroup per (group, image).
 */
export function groupNormSpec(relu: boolean): KernelSpec {
  return spec(`group_norm${relu ? "/relu" : ""}`, ["uniform", "read", "read", "read", "rw"], cpu.cpuGroupNorm(relu), () => /* wgsl */ `
struct P { T: u32, C: u32, G: u32, eps: f32, N: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

fn total(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let t = red[0];
  workgroupBarrier();
  return t;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let cg = p.C / p.G;
  let c0 = wid.x * cg;
  let o = wid.y * p.T * p.C;
  let n = p.T * cg;
  var s = 0.0;
  for (var i = li; i < n; i += 256u) { s += x[o + (i / cg) * p.C + c0 + i % cg]; }
  let mean = total(li, s) / f32(n);
  var v = 0.0;
  for (var i = li; i < n; i += 256u) { let d = x[o + (i / cg) * p.C + c0 + i % cg] - mean; v += d * d; }
  let inv = inverseSqrt(total(li, v) / f32(n) + p.eps);
  for (var i = li; i < n; i += 256u) {
    let at = o + (i / cg) * p.C + c0 + i % cg;
    let c = c0 + i % cg;
    let r = (x[at] - mean) * inv * w[c] + b[c];
    y[at] = ${relu ? "max(r, 0.0)" : "r"};
  }
}`);
}

/**
 * A batch of images x [N, H·W, C] upsampled by nearest neighbour to Ho × Wo (torch's "nearest":
 * source = floor(dst · in / out)) and added to one shared image f [Ho·Wo, C]: y [N, Ho·Wo, C].
 */
export function nearestAddSpec(): KernelSpec {
  return spec("nearest_add", ["uniform", "read", "read", "rw"], cpu.cpuNearestAdd, () => /* wgsl */ `
struct P { H: u32, W: u32, C: u32, Ho: u32, Wo: u32, N: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> f: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  let plane = p.Ho * p.Wo * p.C;
  if (idx >= p.N * plane) { return; }
  let n = idx / plane;
  let r = idx % plane;
  let c = r % p.C;
  let t = r / p.C;
  let sy = min(u32(floor(f32(t / p.Wo) * (f32(p.H) / f32(p.Ho)))), p.H - 1u);
  let sx = min(u32(floor(f32(t % p.Wo) * (f32(p.W) / f32(p.Wo)))), p.W - 1u);
  y[idx] = f[r] + x[(n * p.H * p.W + sy * p.W + sx) * p.C + c];
}`);
}

/**
 * DETR's box attention map (DetrMHAttentionMap): for each selected query row sel[i] of q [Q, H·D]
 * and every pixel of k [S, H·D], the per-head score (q · scale) · k, softmaxed over all heads and
 * pixels together: out [n, S, H]. One workgroup per selected query.
 */
export function bboxAttentionSpec(): KernelSpec {
  return spec("bbox_attention", ["uniform", "read", "read", "read", "rw"], cpu.cpuBboxAttention, () => /* wgsl */ `
struct P { S: u32, H: u32, D: u32, scale: f32, n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> sel: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = p.H * p.D;
  let qo = sel[wid.x] * row;
  let base = wid.x * p.S * p.H;
  let n = p.S * p.H;
  var m = -3.4e38;
  for (var j = li; j < n; j += 256u) {
    let px = j / p.H;
    let h = j % p.H;
    var s = 0.0;
    for (var d = 0u; d < p.D; d++) { s += (q[qo + h * p.D + d] * p.scale) * k[px * row + h * p.D + d]; }
    o[base + j] = s;
    m = max(m, s);
  }
  red[li] = m;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var z = 0.0;
  for (var j = li; j < n; j += 256u) { let e = exp(o[base + j] - mx); o[base + j] = e; z += e; }
  red[li] = z;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var j = li; j < n; j += 256u) { o[base + j] *= inv; }
}`);
}

/**
 * The same convolution as conv2dSpec for at most 4 output channels (a mask or matte head), where a
 * 64-channel GEMM tile would waste nearly all its work: one thread per output pixel walks the taps
 * and channels as vec4s, with up to 4 accumulators.
 */
export function conv2dNarrowSpec(v: { relu: boolean; residual: boolean }): KernelSpec {
  const bindings: BindingKind[] = ["uniform", "read", "read", "read"];
  if (v.residual) bindings.push("read");
  bindings.push("rw");
  return spec(`conv2d_narrow${v.relu ? "/relu" : ""}${v.residual ? "/res" : ""}`, bindings, cpu.cpuConv2d(v), () => /* wgsl */ `
struct P { H: u32, W: u32, C: u32, O: u32, Ho: u32, Wo: u32, kh: u32, kw: u32, stride: u32, pad: u32, K: u32, M: u32, N: u32, dil: u32, _c: u32, _d: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
${v.residual ? "@group(0) @binding(4) var<storage, read> res: array<f32>;\n@group(0) @binding(5)" : "@group(0) @binding(4)"} var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.M) { return; }
  let plane = p.Ho * p.Wo;
  let r = idx % plane;
  let base = (idx / plane) * p.H * p.W;
  let oy = i32((r / p.Wo) * p.stride) - i32(p.pad);
  let ox = i32((r % p.Wo) * p.stride) - i32(p.pad);
  let C4 = p.C / 4u;
  let K4 = p.K / 4u;
  var acc = vec4<f32>();
  for (var ky = 0u; ky < p.kh; ky++) {
    let iy = oy + i32(ky * p.dil);
    if (iy < 0 || iy >= i32(p.H)) { continue; }
    for (var kx = 0u; kx < p.kw; kx++) {
      let ix = ox + i32(kx * p.dil);
      if (ix < 0 || ix >= i32(p.W)) { continue; }
      let xo = (base + u32(iy) * p.W + u32(ix)) * C4;
      let wo = (ky * p.kw + kx) * C4;
      for (var c = 0u; c < C4; c++) {
        let xv = x[xo + c];
        for (var o = 0u; o < p.O; o++) { acc[o] += dot(xv, w[o * K4 + wo + c]); }
      }
    }
  }
  for (var o = 0u; o < p.O; o++) {
    let val = acc[o] + bias[o]${v.residual ? " + res[idx * p.O + o]" : ""};
    y[idx * p.O + o] = ${v.relu ? "max(val, 0.0)" : "val"};
  }
}`);
}

/**
 * Depthwise 3 × 3 convolution (groups = channels, stride 1, padding 1) of an image x [H·W, C]:
 * y[p, c] = bias[c] + Σ_taps x[p + tap, c] · w[tap, c], weight stored [9, C].
 */
export function dwConv3Spec(): KernelSpec {
  return spec("dwconv3", ["uniform", "read", "read", "read", "rw"], cpu.cpuDwConv3, () => /* wgsl */ `
struct P { H: u32, W: u32, C: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.H * p.W * p.C) { return; }
  let c = idx % p.C;
  let t = idx / p.C;
  let py = i32(t / p.W);
  let px = i32(t % p.W);
  var s = bias[c];
  for (var ky = 0; ky < 3; ky++) {
    let iy = py + ky - 1;
    if (iy < 0 || iy >= i32(p.H)) { continue; }
    for (var kx = 0; kx < 3; kx++) {
      let ix = px + kx - 1;
      if (ix < 0 || ix >= i32(p.W)) { continue; }
      s += x[(u32(iy) * p.W + u32(ix)) * p.C + c] * w[u32(ky * 3 + kx) * p.C + c];
    }
  }
  y[idx] = s;
}`);
}

/**
 * Semantic segmentation's last step, fused: class scores x [H·W, L] resized bilinearly to Ho × Wo
 * as transformers.js's interpolate does it (pixel centers aligned, the near neighbour clamped before
 * the weight is taken), and each output pixel's best class (the first on ties): y [Ho·Wo] as u32.
 */
export function resizeArgmaxSpec(): KernelSpec {
  return spec("resize_argmax", ["uniform", "read", "rw"], cpu.cpuResizeArgmax, () => /* wgsl */ `
struct P { H: u32, W: u32, L: u32, Ho: u32, Wo: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<u32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.Ho * p.Wo) { return; }
  let fx = (f32(idx % p.Wo) + 0.5) / (f32(p.Wo) / f32(p.W)) - 0.5;
  let fy = (f32(idx / p.Wo) + 0.5) / (f32(p.Ho) / f32(p.H)) - 0.5;
  let x2 = u32(min(i32(floor(fx)) + 1, i32(p.W) - 1));
  let y2 = u32(min(i32(floor(fy)) + 1, i32(p.H) - 1));
  let x1 = u32(max(i32(floor(fx)), 0));
  let y1 = u32(max(i32(floor(fy)), 0));
  let s = fx - f32(x1);
  let t = fy - f32(y1);
  let w1 = (1.0 - s) * (1.0 - t);
  let w2 = s * (1.0 - t);
  let w3 = (1.0 - s) * t;
  let w4 = s * t;
  let a = (y1 * p.W + x1) * p.L;
  let b = (y1 * p.W + x2) * p.L;
  let c = (y2 * p.W + x1) * p.L;
  let d = (y2 * p.W + x2) * p.L;
  var best = 0u;
  var top = w1 * x[a] + w2 * x[b] + w3 * x[c] + w4 * x[d];
  for (var l = 1u; l < p.L; l++) {
    let v = w1 * x[a + l] + w2 * x[b + l] + w3 * x[c + l] + w4 * x[d + l];
    if (v > top) { top = v; best = l; }
  }
  y[idx] = best;
}`);
}

/** k × k max pooling with stride and padding (padding never wins) of x [H·W, C] → y [Ho·Wo, C]. */
export function maxPool2dSpec(): KernelSpec {
  return spec("max_pool2d", ["uniform", "read", "rw"], cpu.cpuMaxPool2d, () => /* wgsl */ `
struct P { H: u32, W: u32, C: u32, Ho: u32, Wo: u32, k: u32, stride: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.Ho * p.Wo * p.C) { return; }
  let c = idx % p.C;
  let t = idx / p.C;
  let y0 = i32((t / p.Wo) * p.stride) - i32(p.pad);
  let x0 = i32((t % p.Wo) * p.stride) - i32(p.pad);
  var m = -3.4e38;
  for (var dy = 0; dy < i32(p.k); dy++) {
    for (var dx = 0; dx < i32(p.k); dx++) {
      let iy = y0 + dy;
      let ix = x0 + dx;
      if (iy >= 0 && iy < i32(p.H) && ix >= 0 && ix < i32(p.W)) { m = max(m, x[(u32(iy) * p.W + u32(ix)) * p.C + c]); }
    }
  }
  y[idx] = m;
}`);
}

/** y = a·x (accumulate 0) or y += a·x (accumulate 1), elementwise. */
export function scaleAddSpec(): KernelSpec {
  return spec("scale_add", ["uniform", "read", "rw"], cpu.cpuScaleAdd, () => /* wgsl */ `
struct P { n: u32, accumulate: u32, a: f32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${EW})
fn main(${EW_ARGS}) {
  ${FLAT}
  if (idx >= p.n) { return; }
  let v = p.a * x[idx];
  if (p.accumulate == 1u) { y[idx] += v; } else { y[idx] = v; }
}`);
}

/** Copies row `st.lastRow` of x [T, D] into out [1, D]. */
export function copyRowSpec(): KernelSpec {
  return spec("copy_row", ["uniform", "read", "read", "rw"], cpu.cpuCopyRow, () => /* wgsl */ `
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
  return spec(rows ? "sample/rows" : "sample", ["uniform", "read", "read", "read", "read", "rw"], cpu.cpuSample(rows), () => /* wgsl */ `
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
  return spec("advance", ["rw", "read", "rw", "rw", "rw"], cpu.cpuAdvance, () => /* wgsl */ `
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
 * Per-head RMSNorm on q and k (Qwen3, Gemma 3), RoPE and the cache write, fused: reads qkv rows
 * [T, (H + 2·Hkv)·D] from one stacked matmul, writes the normalised, rotated q to qo [T, H·D] and the
 * normalised, rotated k plus v into the caches at slot (step.pos0 + t) % cap. One workgroup per
 * (head of q, k or v, token), one thread per element (D ≤ 256).
 */
export function qkNormRopeKvSpec(interleaved: boolean, kv: KvDtype = "f32"): KernelSpec {
  const h = kv === "f16";
  return spec(`qknorm_rope_kv/${interleaved ? "interleaved" : "half"}/${kv}`, ["uniform", "read", "read", "read", "read", "read", "rw", "rw", "rw"], cpu.cpuQkNormRopeKv(interleaved, h), () => /* wgsl */ `
${STEP_WGSL}
struct P { T: u32, H: u32, Hkv: u32, D: u32, maxSeq: u32, cap: u32, eps: f32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
@group(0) @binding(2) var<storage, read> invFreq: array<f32>;
@group(0) @binding(3) var<storage, read> qkv: array<f32>;
@group(0) @binding(4) var<storage, read> qw: array<f32>;
@group(0) @binding(5) var<storage, read> kw: array<f32>;
@group(0) @binding(6) var<storage, read_write> qo: array<f32>;
@group(0) @binding(7) var<storage, read_write> kc: array<${h ? "u32" : "f32"}>;
@group(0) @binding(8) var<storage, read_write> vc: array<${h ? "u32" : "f32"}>;
var<workgroup> row: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let hh = wid.x;
  let t = wid.y;
  let D = p.D;
  let base = (t * (p.H + 2u * p.Hkv) + hh) * D;
  let pos = st.pos0 + t;
  let kn = p.Hkv * D;
  let slot = (pos % p.cap) * kn;
  // Writes are guarded rather than returned from: pos comes from storage, and a barrier follows.
  let live = pos < p.maxSeq;
  let x = select(0.0, qkv[base + min(i, D - 1u)], i < D);
  if (hh >= p.H + p.Hkv) {
    // v: copied into the cache as is.
    let c = (hh - p.H - p.Hkv) * D + i;
${h
  ? `    if (live && i < D && i % 2u == 0u) { vc[(slot + c) / 2u] = pack2x16float(vec2<f32>(x, qkv[base + i + 1u])); }`
  : `    if (live && i < D) { vc[slot + c] = x; }`}
    return;
  }
  // RMSNorm over the head.
  red[i] = x * x;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (i < s) { red[i] += red[i + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(D) + p.eps);
  let isQ = hh < p.H;
  let w = select(kw[min(i, D - 1u)], qw[min(i, D - 1u)], isQ);
  row[i] = x * inv * w;
  workgroupBarrier();
  // RoPE from the normalised row.
  var out = 0.0;
  if (i < D) {
${interleaved
  ? `    let j = i / 2u;
    let a = row[2u * j];
    let b = row[2u * j + 1u];
    let first = (i % 2u) == 0u;`
  : `    let hd = D / 2u;
    let j = i % hd;
    let a = row[j];
    let b = row[j + hd];
    let first = i < hd;`}
    let ang = f32(pos) * invFreq[j];
    out = select(b * cos(ang) + a * sin(ang), a * cos(ang) - b * sin(ang), first);
  }
  if (isQ) {
    if (i < D) { qo[(t * p.H + hh) * D + i] = out; }
    return;
  }
  let c = (hh - p.H) * D + i;
${h
  ? `  workgroupBarrier();
  red[i] = out;
  workgroupBarrier();
  if (live && i < D && i % 2u == 0u) { kc[(slot + c) / 2u] = pack2x16float(vec2<f32>(red[i], red[i + 1u])); }`
  : `  if (live && i < D) { kc[slot + c] = out; }`}
}`);
}

/**
 * Rotary embedding fused with the KV-cache write. Reads fused qkv rows [T, (H + 2·Hkv)·D], writes the
 * rotated q to qo [T, H·D] and the rotated k plus v straight into the caches at step.pos0 + t (slot
 * (step.pos0 + t) % cap).
 * One thread per output element; each reads its rotation partner, so nothing is updated in place.
 */
export function ropeKvSpec(interleaved: boolean, kv: KvDtype = "f32"): KernelSpec {
  const h = kv === "f16";
  return spec(`rope_kv/${interleaved ? "interleaved" : "half"}/${kv}`, ["uniform", "read", "read", "read", "rw", "rw", "rw"], cpu.cpuRopeKv(interleaved, h), () => /* wgsl */ `
${STEP_WGSL}
struct P { T: u32, H: u32, Hkv: u32, D: u32, maxSeq: u32, cap: u32, _b: u32, _c: u32 }
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
  let dst = (pos % p.cap) * kn + (c - qn) % kn;
${h
  ? `  let packed = pack2x16float(pair);
  if (c < qn + kn) { kc[dst / 2u] = packed; } else { vc[dst / 2u] = packed; }`
  : `  if (c < qn + kn) { kc[dst] = pair.x; kc[dst + 1u] = pair.y; } else { vc[dst] = pair.x; vc[dst + 1u] = pair.y; }`}
}`);
}
