import type { KernelSpec } from "@kiln/core";

/** Source types the GPU converts to Kiln's q4 planes at load time. */
export type GpuConvertType = "F32" | "F16" | "BF16" | "Q8_0" | "Q4_0" | "Q4_K" | "Q5_K" | "Q6_K";

export const GPU_CONVERT_TYPES: ReadonlySet<string> = new Set<GpuConvertType>(["F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K"]);

/** Decodes element l of block i into x[l] (as the CPU's toF32 computes it, bit for bit). */
const DECODE: Partial<Record<GpuConvertType, string>> = {
  F32: `x[l] = bitcast<f32>(src[i * 32u + l]);`,
  F16: `x[l] = f16Bits((src[i * 16u + l / 2u] >> ((l & 1u) * 16u)) & 0xffffu);`,
  BF16: `x[l] = bitcast<f32>(((src[i * 16u + l / 2u] >> ((l & 1u) * 16u)) & 0xffffu) << 16u);`,
  Q8_0: `x[l] = f32(i32(rd(i * 34u + 2u + l) << 24u) >> 24u) * d;`,
  // Both products are exact in f32 (f16 times an integer of at most 11 bits), so the one rounding
  // of the difference matches the CPU's double arithmetic.
  Q4_K: `x[l] = d * f32(sm.x * ((rd(o + 16u + (j / 2u) * 32u + l) >> ((j & 1u) * 4u)) & 15u)) - dmin * f32(sm.y);`,
  Q5_K: `x[l] = d * f32(sm.x * (((rd(o + 48u + (j / 2u) * 32u + l) >> ((j & 1u) * 4u)) & 15u) | (((rd(o + 16u + l) >> j) & 1u) << 4u))) - dmin * f32(sm.y);`,
};

/** Per-block setup before the decode loop. */
const SETUP: Partial<Record<GpuConvertType, string>> = {
  Q8_0: `let d = f16At(i * 34u);`,
  Q4_K: `let o = (i / 8u) * 144u; let j = i % 8u; let d = f16At(o); let dmin = f16At(o + 2u); let sm = scaleMin(j, o + 4u);`,
  Q5_K: `let o = (i / 8u) * 176u; let j = i % 8u; let d = f16At(o); let dmin = f16At(o + 2u); let sm = scaleMin(j, o + 4u);`,
};

/**
 * True when code k is at most floor(x / d + 8.5), d = mx / -8: 16x vs (17 - 2k)·mx, compared exactly.
 * mx is split into 12-bit halves so each product is exact; the difference with 16x is exact whenever
 * the two are within a factor of two (Sterbenz), and elsewhere too large for the low half to matter.
 */
const QUANT_FN = /* wgsl */ `
fn atMost(k: i32, x: f32, mx: f32, mh: f32, ml: f32) -> bool {
  let c = f32(17 - 2 * k);
  let lhs = 16.0 * x - c * mh;
  let r = c * ml;
  return select(lhs >= r, lhs <= r, mx > 0.0);
}`;

const QUANT_BODY = /* wgsl */ `
  // The signed value of largest magnitude, the first on a tie, as quantizeQ4 picks it.
  var mx = 0.0;
  for (var l = 0u; l < 32u; l++) { if (abs(x[l]) > abs(mx)) { mx = x[l]; } }
  let dq = mx / -8.0;
  scales[b] = dq;
  let mh = bitcast<f32>(bitcast<u32>(mx) & 0xfffff000u);
  let ml = mx - mh;
  for (var w = 0u; w < 4u; w++) {
    var word = 0u;
    for (var e = 0u; e < 8u; e++) {
      var q = 8;
      if (mx != 0.0) {
        let v = x[w * 8u + e];
        q = clamp(i32(floor(v / dq + 8.5)), -1, 16);
        if (atMost(q + 1, v, mx, mh, ml)) { q += 1; } else if (!atMost(q, v, mx, mh, ml)) { q -= 1; }
      }
      word |= u32(clamp(q, 0, 15)) << (4u * e);
    }
    nibbles[b * 4u + w] = word;
  }`;

/** Threads per workgroup of the convert kernels; each thread makes one 32-element q4 block. */
export const CONVERT_WG = 64;

/**
 * Converts raw source blocks (bytes of one chunk, read as u32 words) into q4 planes: 8 nibbles per u32
 * (value + 8, element order) and one f32 scale per 32 elements, written from block `dst0` on.
 *
 * Q4_0 is a bit-exact repack. Float and other block types decode as toF32 does and quantize with
 * quantizeQ4's rule, compared exactly, so the planes are the CPU's bits. Q6_K is quantized with `quantizeQ4`'s rule (signed max, the first on
 * a tie, q = clamp(floor(x / d + 8.5))) in exact integer math, so it gives the CPU's bits too.
 */
export function convertSpec(type: GpuConvertType): KernelSpec {
  const decode = DECODE[type];
  const block = decode
    ? `
  ${SETUP[type] ?? ""}
  var x: array<f32, 32>;
  for (var l = 0u; l < 32u; l++) { ${decode} }${QUANT_BODY}`
    : type === "Q4_0"
      ? /* wgsl */ `
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
  }`
      : /* wgsl */ `
  // Sub-block i of 8 per 256-element super-block: half (128 elements) and quarter (32) within it.
  let o = (i / 8u) * 210u;
  let hf = (i % 8u) / 4u;
  let quarter = i % 4u;
  let ql = o + hf * 64u + (quarter & 1u) * 32u;
  let qh = o + 128u + hf * 32u;
  let sc = o + 192u + hf * 8u + quarter * 2u;
  let d = f16At(o + 208u);
  let shift = quarter * 2u;
  let nib = (quarter >> 1u) * 4u;
  // Every value is d * t with t = scale * q an integer, so the codes come from exact integer math:
  // floor(x / dq + 8.5) = floor((17 tm - 16 t) / (2 tm)) with tm the t of the signed max.
  var t: array<i32, 32>;
  var tm = 0;
  for (var l = 0u; l < 32u; l++) {
    let q = i32(((rd(ql + l) >> nib) & 15u) | (((rd(qh + l) >> shift) & 3u) << 4u)) - 32;
    let s = i32(rd(sc + l / 16u) << 24u) >> 24u;
    t[l] = s * q;
    if (abs(t[l]) > abs(tm)) { tm = t[l]; }
  }
  let zero = tm == 0 || d == 0.0;
  scales[b] = select(d * f32(tm), 0.0, zero) / -8.0;
  for (var w = 0u; w < 4u; w++) {
    var word = 0u;
    for (var k = 0u; k < 8u; k++) {
      var q = 8;
      if (!zero) {
        var num = 17 * tm - 16 * t[w * 8u + k];
        var den = 2 * tm;
        if (den < 0) { num = -num; den = -den; }
        q = num / den;
        if (num % den != 0 && num < 0) { q -= 1; }
      }
      word |= u32(clamp(q, 0, 15)) << (4u * k);
    }
    nibbles[b * 4u + w] = word;
  }`;
  return {
    key: `convert/${type}`,
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
${decode ? QUANT_FN : ""}

@compute @workgroup_size(${CONVERT_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.x + gid.y * nw.x * ${CONVERT_WG}u;
  if (i >= p.n) { return; }
  let b = p.dst0 + i;${block}
}
`,
  };
}
