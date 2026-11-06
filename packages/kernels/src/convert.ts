import type { KernelSpec } from "@kiln/core";

/** Source types the GPU converts to Kiln's q4 or q8 planes at load time. */
export type GpuConvertType = "F32" | "F16" | "BF16" | "Q8_0" | "Q4_0" | "Q4_1" | "Q5_0" | "Q5_1" | "Q4_K" | "Q5_K" | "Q6_K";

export const GPU_CONVERT_TYPES: ReadonlySet<string> = new Set<GpuConvertType>(["F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q4_K", "Q5_K", "Q6_K"]);

/** Planes the convert kernels write. Q4_0 always loads as q4, and Q8_0 to q8 is a plain copy, left to the CPU. */
export type GpuConvertOut = "q4" | "q8";

/** True if the GPU converts `type` to `out`. */
export function gpuConverts(type: string, out: GpuConvertOut): boolean {
  return GPU_CONVERT_TYPES.has(type) && !(out === "q8" && (type === "Q4_0" || type === "Q8_0"));
}

/** Decodes element l of block i into x[l] (as the CPU's toF32 computes it, bit for bit). */
const DECODE: Partial<Record<GpuConvertType, string>> = {
  F32: `x[l] = bitcast<f32>(src[i * 32u + l]);`,
  F16: `x[l] = f16Bits((src[i * 16u + l / 2u] >> ((l & 1u) * 16u)) & 0xffffu);`,
  BF16: `x[l] = bitcast<f32>(((src[i * 16u + l / 2u] >> ((l & 1u) * 16u)) & 0xffffu) << 16u);`,
  Q8_0: `x[l] = f32(i32(rd(i * 34u + 2u + l) << 24u) >> 24u) * d;`,
  // Low nibbles are elements 0–15, high 16–31; Q5's fifth bit of element l is bit l of qh.
  Q4_1: `x[l] = f32((rd(o + 4u + (l & 15u)) >> ((l >> 4u) * 4u)) & 15u) * d + m;`,
  Q5_0: `x[l] = f32(i32(((rd(o + 6u + (l & 15u)) >> ((l >> 4u) * 4u)) & 15u) | (((qh >> l) & 1u) << 4u)) - 16) * d;`,
  Q5_1: `x[l] = f32(((rd(o + 8u + (l & 15u)) >> ((l >> 4u) * 4u)) & 15u) | (((qh >> l) & 1u) << 4u)) * d + m;`,
  // Both products are exact in f32 (f16 times an integer of at most 11 bits), so the one rounding
  // of the difference matches the CPU's double arithmetic.
  Q4_K: `x[l] = d * f32(sm.x * ((rd(o + 16u + (j / 2u) * 32u + l) >> ((j & 1u) * 4u)) & 15u)) - dmin * f32(sm.y);`,
  Q5_K: `x[l] = d * f32(sm.x * (((rd(o + 48u + (j / 2u) * 32u + l) >> ((j & 1u) * 4u)) & 15u) | (((rd(o + 16u + l) >> j) & 1u) << 4u))) - dmin * f32(sm.y);`,
  // d · (scale · q): the integer is exact, so the product is rounded once, as the CPU rounds it.
  Q6_K: `x[l] = d * f32((i32(rd(sc + l / 16u) << 24u) >> 24u) * (i32(((rd(ql + l) >> nib) & 15u) | (((rd(qh + l) >> shift) & 3u) << 4u)) - 32));`,
};

/** Per-block setup before the decode loop. */
const SETUP: Partial<Record<GpuConvertType, string>> = {
  Q8_0: `let d = f16At(i * 34u);`,
  Q4_1: `let o = i * 20u; let d = f16At(o); let m = f16At(o + 2u);`,
  Q5_0: `let o = i * 22u; let d = f16At(o); let qh = rd(o + 2u) | (rd(o + 3u) << 8u) | (rd(o + 4u) << 16u) | (rd(o + 5u) << 24u);`,
  Q5_1: `let o = i * 24u; let d = f16At(o); let m = f16At(o + 2u); let qh = rd(o + 4u) | (rd(o + 5u) << 8u) | (rd(o + 6u) << 16u) | (rd(o + 7u) << 24u);`,
  Q4_K: `let o = (i / 8u) * 144u; let j = i % 8u; let d = f16At(o); let dmin = f16At(o + 2u); let sm = scaleMin(j, o + 4u);`,
  Q5_K: `let o = (i / 8u) * 176u; let j = i % 8u; let d = f16At(o); let dmin = f16At(o + 2u); let sm = scaleMin(j, o + 4u);`,
  // Sub-block i of 8 per 256-element super-block: half (128 elements) and quarter (32) within it.
  Q6_K: `let o = (i / 8u) * 210u; let hf = (i % 8u) / 4u; let quarter = i % 4u;
  let ql = o + hf * 64u + (quarter & 1u) * 32u; let qh = o + 128u + hf * 32u; let sc = o + 192u + hf * 8u + quarter * 2u;
  let d = f16At(o + 208u); let shift = quarter * 2u; let nib = (quarter >> 1u) * 4u;`,
};

/**
 * Exact comparisons in integer math on f32 mantissas (24 bits with the implicit one) and exponents,
 * so no compiler rewriting of float expressions can change a code. Values below the normal range
 * count as zero.
 */
const EXACT_FN = /* wgsl */ `
fn mant(v: f32) -> u32 { return (bitcast<u32>(v) & 0x7fffffu) | 0x800000u; }
fn expo(v: f32) -> i32 { return i32((bitcast<u32>(v) >> 23u) & 255u); }

// Sign of c·|m| - 16·|u| for c in 1..19 and |u| <= |m|.
fn cmp16(c: u32, m: f32, u: f32) -> i32 {
  if (expo(u) == 0) { return 1; }
  let g = expo(m) - expo(u) - 4;
  if (g > 3) { return 1; }
  var l = c * mant(m);
  var r = mant(u);
  if (g >= 0) { l = l << u32(g); } else { r = r << u32(-g); }
  return select(select(0, -1, l < r), 1, l > r);
}`;

/** q4: true when code k is at most floor(x / d + 8.5), d = mx / -8, i.e. 16x vs (17 - 2k)·mx. */
const QUANT_FN = /* wgsl */ `
fn atMost(k: i32, x: f32, mx: f32) -> bool {
  let c = 17 - 2 * k;
  // With u = x·sign(mx): 16u <= c·|mx|.
  let u = select(-x, x, mx > 0.0);
  if (c > 0) { return u <= 0.0 || cmp16(u32(c), mx, u) >= 0; }
  return u < 0.0 && cmp16(u32(-c), mx, u) <= 0;
}`;

const QUANT_BODY = /* wgsl */ `
  // The signed value of largest magnitude, the first on a tie, as quantizeQ4 picks it.
  var mx = 0.0;
  for (var l = 0u; l < 32u; l++) { if (abs(x[l]) > abs(mx)) { mx = x[l]; } }
  let dq = mx / -8.0;
  scales[b] = dq;
  for (var w = 0u; w < 4u; w++) {
    var word = 0u;
    for (var e = 0u; e < 8u; e++) {
      var q = 8;
      if (expo(mx) != 0) {
        let v = x[w * 8u + e];
        q = clamp(i32(floor(v / dq + 8.5)), -1, 16);
        if (atMost(q + 1, v, mx)) { q += 1; } else if (!atMost(q, v, mx)) { q -= 1; }
      }
      word |= u32(clamp(q, 0, 15)) << (4u * e);
    }
    nibbles[b * 4u + w] = word;
  }`;

/**
 * q8. `div127` is amax / 127 correctly rounded (WGSL division may be off by an ulp or two): of the
 * quotient and its neighbours, the one with the smallest |amax - 127·d|, in integers. `atLeast` is
 * true when code k is at most round(a / d) with ties away from zero: (2k - 1)·d <= 2a.
 */
const Q8_FN = /* wgsl */ `
fn div127(amax: f32) -> f32 {
  let d0 = amax / 127.0;
  var best = d0;
  var err = -1.0;
  for (var s = -2; s <= 2; s++) {
    let c = bitcast<f32>(u32(i32(bitcast<u32>(d0)) + s));
    let sh = expo(amax) - expo(c);
    if (expo(c) == 0 || sh < 0 || sh > 8) { continue; }
    let a = mant(amax) << u32(sh);
    let m = 127u * mant(c);
    let e = ldexp(f32(max(a, m) - min(a, m)), expo(c) - 150);
    if (err < 0.0 || e < err) { best = c; err = e; }
  }
  return best;
}

fn atLeast(k: i32, a: f32, d: f32) -> bool {
  if (k <= 0) { return true; }
  let sh = expo(a) - expo(d) + 1;
  if (expo(a) == 0 || sh < 0) { return false; }
  return u32(2 * k - 1) * mant(d) <= mant(a) << u32(sh);
}`;

const Q8_BODY = /* wgsl */ `
  var amax = 0.0;
  for (var l = 0u; l < 32u; l++) { amax = max(amax, abs(x[l])); }
  let dq = select(0.0, div127(amax), expo(amax) != 0);
  scales[b] = dq;
  for (var w = 0u; w < 8u; w++) {
    var word = 0u;
    for (var e = 0u; e < 4u; e++) {
      let v = x[w * 4u + e];
      var k = 0;
      if (dq > 0.0) {
        let a = abs(v);
        k = clamp(i32(round(a / dq)), 0, 127);
        if (k < 127 && atLeast(k + 1, a, dq)) { k += 1; } else if (!atLeast(k, a, dq)) { k -= 1; }
      }
      word |= (u32(select(k, -k, v < 0.0)) & 255u) << (8u * e);
    }
    nibbles[b * 8u + w] = word;
  }`;

/** Threads per workgroup of the convert kernels; each thread makes one 32-element block. */
export const CONVERT_WG = 64;

/**
 * Converts raw source blocks (bytes of one chunk, read as u32 words) into q4 planes (8 nibbles per
 * u32, value + 8, element order) or q8 planes (4 int8 per u32, element order), plus one f32 scale per
 * 32 elements, written from block `dst0` on.
 *
 * Q4_0 to q4 is a bit-exact repack. Everything else decodes as toF32 does and quantizes with
 * quantizeQ4's or quantizeQ8's rule, the rounding decided by exact comparisons, so the planes are
 * the CPU's bits.
 */
export function convertSpec(type: GpuConvertType, out: GpuConvertOut = "q4"): KernelSpec {
  if (!gpuConverts(type, out)) throw new Error(`convertSpec: no GPU conversion from ${type} to ${out}`);
  const decode = DECODE[type];
  const block = decode
    ? `
  ${SETUP[type] ?? ""}
  var x: array<f32, 32>;
  for (var l = 0u; l < 32u; l++) { ${decode} }${out === "q8" ? Q8_BODY : QUANT_BODY}`
    : /* wgsl */ `
  let o = i * 18u;
  scales[b] = f16At(o);
  for (var h = 0u; h < 2u; h++) {
    var lo = 0u;
    var hi = 0u;
    for (var k = 0u; k < 8u; k++) {
      let v = rd(o + 2u + h * 8u + k);
      lo |= (v & 15u) << (4u * k);
      hi |= (v >> 4u) << (4u * k);
    }
    nibbles[b * 4u + h] = lo;
    nibbles[b * 4u + 2u + h] = hi;
  }`;
  return {
    key: `convert/${type}/${out}`,
    bindings: ["uniform", "read", "rw", "rw"],
    code: /* wgsl */ `
struct P { n: u32, dst0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> nibbles: array<u32>;
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;

fn rd(at: u32) -> u32 { return (src[at >> 2u] >> ((at & 3u) * 8u)) & 255u; }

// f16 bits to f32, subnormals included (unpack2x16float may flush them).
fn f16At(at: u32) -> f32 { return f16Bits(rd(at) | (rd(at + 1u) << 8u)); }

fn f16Bits(h: u32) -> f32 {
  let e = i32((h >> 10u) & 31u);
  let m = f32(h & 1023u);
  let v = select(ldexp(m + 1024.0, e - 25), ldexp(m, -24), e == 0);
  return select(v, -v, (h & 0x8000u) != 0u);
}

// 6-bit scale and min of sub-block j from a k-quant's 12-byte table (llama.cpp get_scale_min_k4).
fn scaleMin(j: u32, q: u32) -> vec2<u32> {
  if (j < 4u) { return vec2<u32>(rd(q + j) & 63u, rd(q + j + 4u) & 63u); }
  return vec2<u32>((rd(q + j + 4u) & 15u) | ((rd(q + j - 4u) >> 6u) << 4u), (rd(q + j + 4u) >> 4u) | ((rd(q + j) >> 6u) << 4u));
}
${decode ? EXACT_FN + (out === "q8" ? Q8_FN : QUANT_FN) : ""}

@compute @workgroup_size(${CONVERT_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.x + gid.y * nw.x * ${CONVERT_WG}u;
  if (i >= p.n) { return; }
  let b = p.dst0 + i;${block}
}
`,
  };
}
