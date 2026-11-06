import type { KernelSpec } from "@kiln/core";

/** GGUF block types the GPU converts to Kiln's q4 planes at load time. */
export type GpuConvertType = "Q4_0" | "Q6_K";

/** Threads per workgroup of the convert kernels; each thread makes one 32-element q4 block. */
export const CONVERT_WG = 64;

/**
 * Converts raw GGUF blocks (bytes of one chunk, read as u32 words) into q4 planes: 8 nibbles per u32
 * (value + 8, element order) and one f32 scale per 32 elements, written from block `dst0` on.
 *
 * Q4_0 is a bit-exact repack. Q6_K is quantized with `quantizeQ4`'s rule (signed max, the first on
 * a tie, q = clamp(floor(x / d + 8.5))) in exact integer math, so it gives the CPU's bits too.
 */
export function convertSpec(type: GpuConvertType): KernelSpec {
  const block =
    type === "Q4_0"
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
fn f16At(at: u32) -> f32 {
  let h = rd(at) | (rd(at + 1u) << 8u);
  let e = i32((h >> 10u) & 31u);
  let m = f32(h & 1023u);
  let v = select(ldexp(m + 1024.0, e - 25), ldexp(m, -24), e == 0);
  return select(v, -v, (h & 0x8000u) != 0u);
}

@compute @workgroup_size(${CONVERT_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.x + gid.y * nw.x * ${CONVERT_WG}u;
  if (i >= p.n) { return; }
  let b = p.dst0 + i;${block}
}
`,
  };
}
