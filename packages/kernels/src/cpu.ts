import type { CpuKernel, CpuView, WeightDType } from "@kiln/core";
import { GEMV_F32, GEMV_Q4, GEMV_Q8, type WasmHeap } from "./simd";

/**
 * TypeScript versions of the WGSL kernels, for the CPU backend. Each takes the same bindings, in the
 * same order and with the same uniform layout, as its kernel, and computes the same result (up to
 * float summation order). The kernel-check page runs them against the reference like the GPU ones.
 */

// ---- f16, as pack2x16float / unpack2x16float use it ----

const f32v = new Float32Array(1);
const u32v = new Uint32Array(f32v.buffer);

/** f32 → f16 bits, round to nearest even, overflow to infinity, subnormals kept. */
export function toF16(x: number): number {
  f32v[0] = x;
  const b = u32v[0]!;
  const sign = (b >>> 16) & 0x8000;
  const e = (b >>> 23) & 255;
  let m = b & 0x7fffff;
  if (e === 255) return sign | 0x7c00 | (m ? 0x200 : 0);
  let ex = e - 112;
  if (ex >= 31) return sign | 0x7c00;
  if (ex <= 0) {
    if (ex < -10) return sign;
    m |= 0x800000;
    const shift = 14 - ex;
    const half = 1 << (shift - 1);
    let r = m >>> shift;
    const rest = m & ((1 << shift) - 1);
    if (rest > half || (rest === half && r & 1)) r++;
    return sign | r;
  }
  let r = (ex << 10) | (m >>> 13);
  const rest = m & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && r & 1)) r++;
  return sign | r;
}

const F16 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const e = (h >> 10) & 31;
    const m = h & 1023;
    const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1024 + m) * 2 ** (e - 25);
    t[h] = h & 0x8000 ? -v : v;
  }
  return t;
})();


/** Writes elements e and e + 1 (e even) of a cache. */
/**
 * Writes element pairs (e, e + 1) of a cache: packed halves or two floats. The view is taken once,
 * since a heap buffer's view getter is too slow to call per element.
 */
function cacheWriter(c: CpuView, f16: boolean): (e: number, a: number, b: number) => void {
  if (f16) {
    const u = c.u32;
    return (e, a, b) => void (u[e >>> 1] = (toF16(a) | (toF16(b) << 16)) >>> 0);
  }
  const f = c.f32;
  return (e, a, b) => {
    f[e] = a;
    f[e + 1] = b;
  };
}

const geluTanh = (x: number) => 0.5 * x * (1 + Math.tanh(Math.max(-15, Math.min(15, 0.7978845608 * (x + 0.044715 * x * x * x)))));
const act = (epi: "silu" | "gelu", x: number) => (epi === "gelu" ? geluTanh(x) : x / (1 + Math.exp(-x)));

/** Step buffer fields (STEP_WGSL). */
const step = (s: CpuView) => ({
  pos0: s.u32[0]!, lastRow: s.u32[1]!, seed: s.u32[2]!, temperature: s.f32[3]!, n: s.u32[4]!, topK: s.u32[5]!, topP: s.f32[6]!, rep: s.f32[7]!,
  spanLo: s.u32[8]!, spanHi: s.u32[9]!,
});

// ---- matmul, GEMV and GEMM: one implementation for all three ----

export interface CpuMatmulVariant {
  wdtype: WeightDType;
  affine?: boolean;
  norm: boolean;
  bias: boolean;
  epi: "store" | "add" | "silu" | "gelu";
}

/** Row n of a weight matrix [N, K], dequantized into `out`. */
function weightRow(w: readonly CpuView[], wdtype: WeightDType, affine: boolean, n: number, K: number, out: Float32Array): void {
  const o = n * K;
  if (wdtype === "f32") {
    out.set(w[0]!.f32.subarray(o, o + K));
    return;
  }
  const ws = w[1]!.f32;
  const wm = affine ? w[2]!.f32 : undefined;
  if (wdtype === "q4") {
    const q = w[0]!.u32;
    for (let b = 0; b < K; b += 32) {
      const blk = (o + b) >>> 5;
      const s = ws[blk]!;
      const m = wm ? wm[blk]! : -8 * s;
      for (let j = 0; j < 4; j++) {
        let word = q[((o + b) >>> 3) + j]!;
        for (let e = 0; e < 8; e++, word >>>= 4) out[b + j * 8 + e] = wm ? (word & 15) * s + m : ((word & 15) - 8) * s;
      }
    }
    return;
  }
  const q = w[0]!.i8;
  for (let b = 0; b < K; b += 32) {
    const blk = (o + b) >>> 5;
    const s = ws[blk]!;
    if (wm) {
      const m = wm[blk]!;
      for (let e = 0; e < 32; e++) out[b + e] = q[o + b + e]! * s + m;
    } else for (let e = 0; e < 32; e++) out[b + e] = q[o + b + e]! * s;
  }
}

/** Σ x[xo + k]·w[wo + k] for k < K, with four partial sums. */
function dotF32(x: Float32Array, xo: number, w: Float32Array, wo: number, K: number): number {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (let k = 0; k < K; k += 4) {
    s0 += x[xo + k]! * w[wo + k]!;
    s1 += x[xo + k + 1]! * w[wo + k + 1]!;
    s2 += x[xo + k + 2]! * w[wo + k + 2]!;
    s3 += x[xo + k + 3]! * w[wo + k + 3]!;
  }
  return s0 + s1 + (s2 + s3);
}

/** Σ x·code over one 32-value q4 block (4 words of 8 nibbles, element 8j + e at nibble e of word j). */
function dotQ4(x: Float32Array, xo: number, q: Uint32Array, qo: number): number {
  let t = 0;
  for (let j = 0; j < 4; j++) {
    const w = q[qo + j]!;
    const b = xo + j * 8;
    t += x[b]! * (w & 15) + x[b + 1]! * ((w >>> 4) & 15) + x[b + 2]! * ((w >>> 8) & 15) + x[b + 3]! * ((w >>> 12) & 15)
      + x[b + 4]! * ((w >>> 16) & 15) + x[b + 5]! * ((w >>> 20) & 15) + x[b + 6]! * ((w >>> 24) & 15) + x[b + 7]! * (w >>> 28);
  }
  return t;
}

/** Σ x·code over one 32-value q8 block. */
function dotQ8(x: Float32Array, xo: number, q: Int8Array, qo: number): number {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (let e = 0; e < 32; e += 4) {
    s0 += x[xo + e]! * q[qo + e]!;
    s1 += x[xo + e + 1]! * q[qo + e + 1]!;
    s2 += x[xo + e + 2]! * q[qo + e + 2]!;
    s3 += x[xo + e + 3]! * q[qo + e + 3]!;
  }
  return s0 + s1 + (s2 + s3);
}

/** A view of a buffer in the CPU backend's WASM heap. */
interface HeapView extends CpuView {
  readonly heap?: WasmHeap;
  readonly ptr: number;
}

/**
 * acc[m, n] = Σ_k x[m, k] · W[n, k] in WASM SIMD, written to heap scratch, or to `out` (a heap
 * pointer) when given; returns a view of it, valid until the heap is next used. For f32 weights the x
 * rows (`xs`, already normalised) are copied to scratch; for q4 / q8, `quant_x` reads them from the
 * heap at `xp` and writes them as int16 per block with scales and exact block sums, RMS-normalising
 * them first when `nw` (the norm weights' pointer) is set. Then the kernels run over them, four rows
 * per pass where they can.
 */
const simdScratch = (M: number, N: number, K: number) => M * K * 4 + 2 * Math.ceil((M * (K >>> 5) * 4) / 16) * 16 + M * N * 4;

function simdMatmul(heap: WasmHeap, wdtype: WeightDType, x: { xs?: Float32Array; xp: number; nw: number; eps: number }, w: HeapView[], M: number, N: number, K: number, out = 0): Float32Array {
  const KB = K >>> 5;
  const xBytes = M * K * 4;
  const sumBytes = Math.ceil((M * KB * 4) / 16) * 16;
  const at = heap.scratch(simdScratch(M, N, K)); // reserved by the caller: no growth here
  const xp = at, sp = at + xBytes, scp = sp + sumBytes, yp = out || scp + sumBytes;
  if (wdtype === "f32") {
    new Float32Array(heap.buffer).set(x.xs!.subarray(0, M * K), xp / 4);
    heap.gemm(GEMV_F32, M, N, K, xp, 0, w[0]!.ptr, 0, 0, 0, yp);
  } else {
    const q4 = wdtype === "q4";
    heap.kernels.quant_x(x.xp, K, M, x.nw, x.eps, xp, scp, sp, q4 ? 1 : 0);
    heap.gemm(q4 ? GEMV_Q4 : GEMV_Q8, M, N, K, xp, scp, w[0]!.ptr, w[1]!.ptr, w[2]?.ptr ?? 0, sp, yp);
  }
  return new Float32Array(heap.buffer, yp, M * N);
}

export function cpuMatmul(v: CpuMatmulVariant): CpuKernel {
  const quant = v.wdtype !== "f32";
  const affine = quant && !!v.affine;
  const nw = 2 + (quant ? 2 + (affine ? 1 : 0) : 1);
  return (b) => {
    const p = b[0]!;
    const [M, N, K] = [p.u32[0]!, p.u32[1]!, p.u32[2]!];
    const eps = p.f32[3]!;
    const heap = (b[1] as HeapView).heap;
    // Reserve the SIMD scratch before taking views: growing the heap would detach them.
    heap?.scratch(simdScratch(M, N, K));
    const x = b[1]!.f32;
    const w = b.slice(2, nw);
    let at = nw;
    const norm = v.norm ? b[at++]!.f32 : undefined;
    const bias = v.bias ? b[at++]!.f32 : undefined;
    const y = b[at]!.f32;
    const gated = v.epi === "silu" || v.epi === "gelu";
    // Quantized matmuls on the heap normalise in WASM, and write y directly when nothing follows.
    const wasmX = !!heap && quant;
    const direct = !!heap && !bias && !gated && v.epi !== "add";
    // x rows, normalised on the fly as the GPU kernels do.
    let xs = x.subarray(0, M * K);
    if (norm && !wasmX) {
      xs = new Float32Array(M * K);
      for (let m = 0; m < M; m++) {
        let ss = 0;
        for (let k = 0; k < K; k++) ss += x[m * K + k]! ** 2;
        const inv = 1 / Math.sqrt(Math.fround(ss / K) + eps);
        for (let k = 0; k < K; k++) xs[m * K + k] = x[m * K + k]! * inv * norm[k]!;
      }
    }
    const KB = K >>> 5;
    let acc: Float32Array;
    if (heap) {
      const xv = { xs, xp: (b[1] as HeapView).ptr, nw: norm ? (b[nw] as HeapView).ptr : 0, eps };
      acc = simdMatmul(heap, v.wdtype, xv, w as HeapView[], M, N, K, direct ? (b[at] as HeapView).ptr : 0);
      if (direct) return;
    }
    else if (v.wdtype === "f32") {
      acc = new Float32Array(M * N);
      const wf = w[0]!.f32;
      for (let n = 0; n < N; n++)
        for (let m = 0; m < M; m++) acc[m * N + n] = dotF32(xs, m * K, wf, n * K, K);
    } else {
      acc = new Float32Array(M * N);
      // Dots straight from the codes, one block of 32 at a time: scale · Σ x·code, plus the block's
      // x sum times the min (affine) or times -8·scale (symmetric q4's code offset).
      const xsum = new Float32Array(M * KB);
      for (let m = 0; m < M; m++)
        for (let bl = 0; bl < KB; bl++) {
          let t = 0;
          for (let e = 0, o = m * K + bl * 32; e < 32; e++) t += xs[o + e]!;
          xsum[m * KB + bl] = t;
        }
      const ws = w[1]!.f32;
      const wm = affine ? w[2]!.f32 : undefined;
      const q4 = v.wdtype === "q4";
      const qw = w[0]!.u32;
      const q8 = w[0]!.i8;
      for (let n = 0; n < N; n++)
        for (let m = 0; m < M; m++) {
          let total = 0;
          for (let bl = 0; bl < KB; bl++) {
            const blk = n * KB + bl;
            const xo = m * K + bl * 32;
            const t = q4 ? dotQ4(xs, xo, qw, blk * 4) : dotQ8(xs, xo, q8, blk * 32);
            const sx = xsum[m * KB + bl]!;
            total += wm ? ws[blk]! * t + wm[blk]! * sx : q4 ? ws[blk]! * (t - 8 * sx) : ws[blk]! * t;
          }
          acc[m * N + n] = total;
        }
    }
    if (bias) for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) acc[m * N + n]! += bias[n]!;
    if (gated) {
      const I = N / 2;
      for (let m = 0; m < M; m++) for (let i = 0; i < I; i++) y[m * I + i] = act(v.epi as "silu" | "gelu", acc[m * N + i]!) * acc[m * N + I + i]!;
    } else if (v.epi === "add") for (let j = 0; j < M * N; j++) y[j]! += acc[j]!;
    else y.set(acc);
  };
}

// ---- norms, rope, cache writes ----

export const cpuRmsnorm: CpuKernel = (b) => {
  const [R, D] = [b[0]!.u32[0]!, b[0]!.u32[1]!];
  const eps = b[0]!.f32[2]!;
  const x = b[1]!.f32;
  const w = b[2]!.f32;
  const y = b[3]!.f32;
  for (let r = 0; r < R; r++) {
    let ss = 0;
    for (let i = 0; i < D; i++) ss += x[r * D + i]! ** 2;
    const inv = 1 / Math.sqrt(Math.fround(ss / D) + eps);
    for (let i = 0; i < D; i++) y[r * D + i] = x[r * D + i]! * inv * w[i]!;
  }
};

/** cos and sin of each pair's angle at position pos (the angle rounded to f32, as the kernels do), into cs and sn. */
function angles(pos: number, f: Float32Array, half: number, cs: Float64Array, sn: Float64Array): void {
  for (let j = 0; j < half; j++) {
    const ang = Math.fround(pos * f[j]!);
    cs[j] = Math.cos(ang);
    sn[j] = Math.sin(ang);
  }
}

/** Rotates head `src[base…base+D)` by the angles in cs/sn into out[o…o+D) (src and out may be one array). */
function rotateHead(src: Float32Array, base: number, D: number, cs: Float64Array, sn: Float64Array, interleaved: boolean, out: Float32Array, o: number): void {
  const half = D >>> 1;
  for (let j = 0; j < half; j++) {
    const i0 = interleaved ? 2 * j : j;
    const i1 = interleaved ? 2 * j + 1 : j + half;
    const a = src[base + i0]!, c = src[base + i1]!;
    out[o + i0] = a * cs[j]! - c * sn[j]!;
    out[o + i1] = a * sn[j]! + c * cs[j]!;
  }
}

export function cpuRope(interleaved: boolean): CpuKernel {
  return (b) => {
    const [T, H, D] = [b[0]!.u32[0]!, b[0]!.u32[1]!, b[0]!.u32[2]!];
    const { pos0 } = step(b[1]!);
    const f = b[2]!.f32;
    const x = b[3]!.f32;
    const cs = new Float64Array(D >>> 1), sn = new Float64Array(D >>> 1);
    for (let t = 0; t < T; t++) {
      angles(pos0 + t, f, D >>> 1, cs, sn);
      for (let h = 0; h < H; h++) rotateHead(x, (t * H + h) * D, D, cs, sn, interleaved, x, (t * H + h) * D);
    }
  };
}

export function cpuKvWrite(f16: boolean): CpuKernel {
  return (b) => {
    const [T, row, maxSeq, cap] = [b[0]!.u32[0]!, b[0]!.u32[1]!, b[0]!.u32[2]!, b[0]!.u32[3]!];
    const { pos0 } = step(b[1]!);
    const k = b[2]!.f32;
    const v = b[3]!.f32;
    const kc = cacheWriter(b[4]!, f16), vc = cacheWriter(b[5]!, f16);
    for (let t = 0; t < T; t++) {
      if (pos0 + t >= maxSeq) continue;
      const dst = ((pos0 + t) % cap) * row;
      for (let e = 0; e < row; e += 2) {
        kc(dst + e, k[t * row + e]!, k[t * row + e + 1]!);
        vc(dst + e, v[t * row + e]!, v[t * row + e + 1]!);
      }
    }
  };
}

/** Shared tail of the fused rope + cache-write kernels: q out, rotated k and raw v into the caches. */
function qkvToCache(
  b: readonly CpuView[],
  f16: boolean,
  f: Float32Array,
  interleaved: boolean,
  /** Normalises head hh (at base) into `into` and returns it; undefined rotates the raw values. */
  norm: ((hh: number, base: number, into: Float32Array) => Float32Array) | undefined,
  qo: Float32Array,
  kcView: CpuView,
  vcView: CpuView,
): void {
  const u = b[0]!.u32;
  const [T, H, Hkv, D, maxSeq, cap] = [u[0]!, u[1]!, u[2]!, u[3]!, u[4]!, u[5]!];
  const { pos0 } = step(b[1]!);
  const qkv = b[3]!.f32;
  const kc = cacheWriter(kcView, f16), vc = cacheWriter(vcView, f16);
  const kn = Hkv * D;
  const tmp = new Float32Array(D);
  const normed = new Float32Array(D);
  const cs = new Float64Array(D >>> 1), sn = new Float64Array(D >>> 1);
  for (let t = 0; t < T; t++) {
    const pos = pos0 + t;
    const live = pos < maxSeq;
    const slot = (pos % cap) * kn;
    angles(pos, f, D >>> 1, cs, sn);
    for (let hh = 0; hh < H + 2 * Hkv; hh++) {
      const base = (t * (H + 2 * Hkv) + hh) * D;
      if (hh >= H + Hkv) {
        if (live) for (let i = 0; i < D; i += 2) vc(slot + (hh - H - Hkv) * D + i, qkv[base + i]!, qkv[base + i + 1]!);
        continue;
      }
      if (hh < H) {
        // q straight into its output row.
        if (norm) rotateHead(norm(hh, base, normed), 0, D, cs, sn, interleaved, qo, (t * H + hh) * D);
        else rotateHead(qkv, base, D, cs, sn, interleaved, qo, (t * H + hh) * D);
        continue;
      }
      if (!live) continue;
      if (norm) rotateHead(norm(hh, base, normed), 0, D, cs, sn, interleaved, tmp, 0);
      else rotateHead(qkv, base, D, cs, sn, interleaved, tmp, 0);
      for (let i = 0; i < D; i += 2) kc(slot + (hh - H) * D + i, tmp[i]!, tmp[i + 1]!);
    }
  }
}

export function cpuRopeKv(interleaved: boolean, f16: boolean): CpuKernel {
  return (b) => {
    qkvToCache(b, f16, b[2]!.f32, interleaved, undefined, b[4]!.f32, b[5]!, b[6]!);
  };
}

export function cpuQkNormRopeKv(interleaved: boolean, f16: boolean): CpuKernel {
  return (b) => {
    const u = b[0]!.u32;
    const [H, D] = [u[1]!, u[3]!];
    const eps = b[0]!.f32[6]!;
    const qkv = b[3]!.f32;
    const qw = b[4]!.f32;
    const kw = b[5]!.f32;
    qkvToCache(
      b,
      f16,
      b[2]!.f32,
      interleaved,
      (hh, base, normed) => {
        let ss = 0;
        for (let i = 0; i < D; i++) ss += qkv[base + i]! ** 2;
        const inv = 1 / Math.sqrt(Math.fround(ss / D) + eps);
        const w = hh < H ? qw : kw;
        for (let i = 0; i < D; i++) normed[i] = qkv[base + i]! * inv * w[i]!;
        return normed;
      },
      b[6]!.f32,
      b[7]!,
      b[8]!,
    );
  };
}

// ---- attention ----

/** Causal GQA attention over the cache, with the sliding window and ring slots of ATTN_P. */
/** Row `e / D` (D values from element e) of a cache, decoded into out. */
function kvRow(u: Uint32Array, f: Float32Array, e: number, D: number, f16: boolean, out: Float32Array): void {
  if (!f16) {
    for (let i = 0; i < D; i++) out[i] = f[e + i]!;
    return;
  }
  // e and D are even: whole words, two halves each.
  for (let i = 0, w = e >>> 1; i < D; i += 2, w++) {
    const x = u[w]!;
    out[i] = F16[x & 0xffff]!;
    out[i + 1] = F16[x >>> 16]!;
  }
}

/**
 * T5's relative position bucket of key position minus query position: exact for small distances,
 * logarithmic up to `maxDistance`; both directions get half the buckets when `bidirectional`.
 */
export function t5Bucket(rel: number, bidirectional: boolean, buckets: number, maxDistance: number): number {
  let nb = buckets;
  let ret = 0;
  let n: number;
  if (bidirectional) {
    nb = Math.floor(nb / 2);
    if (rel > 0) ret = nb;
    n = Math.abs(rel);
  } else n = Math.max(-rel, 0);
  const exact = Math.floor(nb / 2);
  if (n < exact) return ret + n;
  const large = exact + Math.trunc(Math.fround(Math.fround(Math.log(Math.fround(n / exact))) / Math.fround(Math.log(Math.fround(maxDistance / exact)))) * (nb - exact));
  return ret + Math.min(large, nb - 1);
}

export function cpuAttention(f16: boolean, rel = false): CpuKernel {
  let scores = new Float32Array(0);
  return (b) => {
    const u = b[0]!.u32;
    const [T, H, Hkv, D, maxSeq] = [u[0]!, u[1]!, u[2]!, u[3]!, u[4]!];
    const scale = b[0]!.f32[5]!;
    const [win, cap] = [u[7]!, u[8]!];
    const [buckets, maxDist, bidir] = [u[9]!, u[10]!, u[11]! !== 0];
    const { pos0, spanLo, spanHi } = step(b[1]!);
    const heap = (b[2] as HeapView).heap;
    // T5's position bias is added to each score in TypeScript below (the WASM path takes the max first).
    const table = rel ? b[5]!.f32 : undefined;
    if (heap && !rel && D % (f16 ? 8 : 4) === 0) {
      // The per-key work in WASM SIMD, on the caches in place, for all the query heads of a KV head at
      // once (each cache row read and widened once); masking and the softmax stay here.
      const G = H / Hkv;
      const at = heap.scratch((2 * G * D + G * maxSeq + G + D) * 4); // before any view: growing detaches them
      const k = heap.kernels;
      const eb = f16 ? 2 : 4;
      const stride = Hkv * D * eb;
      const [kp, vp] = [(b[3] as HeapView).ptr, (b[4] as HeapView).ptr];
      const mem = new Float32Array(heap.buffer);
      const q = b[2]!.f32;
      const o = b[5]!.f32;
      const qs = at, acc = qs + G * D * 4, sc = acc + G * D * 4, mx = sc + G * maxSeq * 4, tmp = mx + G * 4;
      const [qi, ai, si, mi] = [qs >>> 2, acc >>> 2, sc >>> 2, mx >>> 2];
      const ls = new Float64Array(G);
      for (let t = 0; t < T; t++) {
        const pos = pos0 + t;
        const n = Math.min(pos >= spanLo && pos < spanHi ? spanHi : pos + 1, maxSeq);
        const first = win > 0 && n > win ? n - win : 0;
        for (let kvh = 0; kvh < Hkv; kvh++) {
          const qo = (t * H + kvh * G) * D;
          for (let i = 0; i < G * D; i++) mem[qi + i] = q[qo + i]! * scale;
          k.att_scores(qs, kp + kvh * D * eb, stride, cap, first, n, D, sc, G, maxSeq, mx, f16 ? 1 : 0, tmp);
          for (let g = 0; g < G; g++) {
            const m = mem[mi + g]!;
            let l = 0;
            for (let j = si + g * maxSeq + first, end = si + g * maxSeq + n; j < end; j++) l += mem[j] = Math.exp(mem[j]! - m);
            ls[g] = l;
          }
          k.att_wsum(sc, vp + kvh * D * eb, stride, cap, first, n, D, acc, G, maxSeq, f16 ? 1 : 0, tmp);
          for (let g = 0; g < G; g++) for (let i = 0; i < D; i++) o[qo + g * D + i] = mem[ai + g * D + i]! / ls[g]!;
        }
      }
      return;
    }
    const q = b[2]!.f32;
    // Views taken once: a heap buffer's view getter is too slow for the inner loops.
    const [ku, kf, vu, vf] = [b[3]!.u32, b[3]!.f32, b[4]!.u32, b[4]!.f32];
    const o = (rel ? b[6]! : b[5]!).f32;
    const stride = Hkv * D;
    if (scores.length < maxSeq) scores = new Float32Array(maxSeq);
    const qs = new Float32Array(D);
    const row = new Float32Array(D);
    const acc = new Float32Array(D);
    for (let t = 0; t < T; t++)
      for (let h = 0; h < H; h++) {
        const kvh = Math.floor(h / (H / Hkv));
        const pos = pos0 + t;
        // Causal, or the whole bidirectional span for a query inside it (as the kernels' keys()).
        const n = Math.min(pos >= spanLo && pos < spanHi ? spanHi : pos + 1, maxSeq);
        const first = win > 0 && n > win ? n - win : 0;
        const qo = (t * H + h) * D;
        for (let i = 0; i < D; i++) qs[i] = q[qo + i]! * scale;
        let mx = -Infinity;
        for (let j = first; j < n; j++) {
          kvRow(ku, kf, (j % cap) * stride + kvh * D, D, f16, row);
          let d = 0;
          for (let i = 0; i < D; i++) d += qs[i]! * row[i]!;
          if (table) d += table[t5Bucket(j - pos, bidir, buckets, maxDist) * H + h]!;
          scores[j] = d;
          if (d > mx) mx = d;
        }
        let l = 0;
        for (let j = first; j < n; j++) l += scores[j] = Math.exp(scores[j]! - mx);
        acc.fill(0);
        for (let j = first; j < n; j++) {
          kvRow(vu, vf, (j % cap) * stride + kvh * D, D, f16, row);
          const p = scores[j]!;
          for (let i = 0; i < D; i++) acc[i]! += p * row[i]!;
        }
        for (let i = 0; i < D; i++) o[qo + i] = acc[i]! / l;
      }
  };
}

// ---- elementwise ----

export function cpuSiluMul(a: "silu" | "gelu"): CpuKernel {
  return (b) => {
    const n = b[0]!.u32[0]!;
    const g = b[1]!.f32;
    const up = b[2]!.f32;
    const o = b[3]!.f32;
    for (let i = 0; i < n; i++) o[i] = act(a, g[i]!) * up[i]!;
  };
}

export function cpuGatedAct(a: "silu" | "gelu"): CpuKernel {
  return (b) => {
    const [n, I] = [b[0]!.u32[0]!, b[0]!.u32[1]!];
    const x = b[1]!.f32;
    const y = b[2]!.f32;
    for (let idx = 0; idx < n; idx++) {
      const r = Math.floor(idx / I);
      const i = idx % I;
      y[idx] = act(a, x[r * 2 * I + i]!) * x[r * 2 * I + I + i]!;
    }
  };
}

export const cpuAdd: CpuKernel = (b) => {
  const [n, cols] = [b[0]!.u32[0]!, b[0]!.u32[1]!];
  const a = b[1]!.f32;
  const c = b[2]!.f32;
  for (let i = 0; i < n; i++) a[i]! += c[i % cols]!;
};

export function cpuEmbed(wdtype: WeightDType, affine: boolean, features = false): CpuKernel {
  const quant = wdtype !== "f32";
  const aff = quant && affine;
  const nw = quant ? (aff ? 3 : 2) : 1;
  return (b) => {
    const [T, D, , V] = [b[0]!.u32[0]!, b[0]!.u32[1]!, 0, b[0]!.u32[3]!];
    const scale = b[0]!.f32[2]!;
    const tokens = b[1]!.u32;
    const w = b.slice(2, 2 + nw);
    const feat = features ? b[2 + nw]!.f32 : undefined;
    const o = b[2 + nw + (features ? 1 : 0)]!.f32;
    const row = new Float32Array(D);
    for (let t = 0; t < T; t++) {
      const id = tokens[t]!;
      if (feat && id >= V) {
        o.set(feat.subarray((id - V) * D, (id - V + 1) * D), t * D);
        continue;
      }
      weightRow(w, wdtype, aff, id, D, row);
      for (let i = 0; i < D; i++) o[t * D + i] = row[i]! * scale;
    }
  };
}

export const cpuLayernorm: CpuKernel = (b) => {
  const [R, D] = [b[0]!.u32[0]!, b[0]!.u32[1]!];
  const eps = b[0]!.f32[2]!;
  const x = b[1]!.f32;
  const w = b[2]!.f32;
  const bias = b[3]!.f32;
  const y = b[4]!.f32;
  for (let r = 0; r < R; r++) {
    let s = 0;
    for (let i = 0; i < D; i++) s += x[r * D + i]!;
    const mean = s / D;
    let v = 0;
    for (let i = 0; i < D; i++) v += (x[r * D + i]! - mean) ** 2;
    const inv = 1 / Math.sqrt(v / D + eps);
    for (let i = 0; i < D; i++) y[r * D + i] = (x[r * D + i]! - mean) * inv * w[i]! + bias[i]!;
  }
};

export const cpuGelu: CpuKernel = (b) => {
  const n = b[0]!.u32[0]!;
  const x = b[1]!.f32;
  for (let i = 0; i < n; i++) x[i] = geluTanh(x[i]!);
};

/** erf by Abramowitz & Stegun 7.1.26 (absolute error under 1.5e-7), as the WGSL kernel computes it. */
export function erfAS(v: number): number {
  const z = Math.abs(v);
  const t = 1 / (1 + 0.3275911 * z);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-z * z);
  return v >= 0 ? y : -y;
}

export const cpuGeluErf: CpuKernel = (b) => {
  const n = b[0]!.u32[0]!;
  const x = b[1]!.f32;
  for (let i = 0; i < n; i++) x[i] = 0.5 * x[i]! * (1 + erfAS(x[i]! * Math.SQRT1_2));
};

/** An activation in place (see actSpec). */
export function cpuAct(kind: "gelu" | "gelu_tanh" | "silu" | "relu"): CpuKernel {
  if (kind === "gelu") return cpuGeluErf;
  if (kind === "gelu_tanh") return cpuGelu;
  return (b) => {
    const n = b[0]!.u32[0]!;
    const x = b[1]!.f32;
    if (kind === "silu") for (let i = 0; i < n; i++) x[i] = x[i]! / (1 + Math.exp(-x[i]!));
    else for (let i = 0; i < n; i++) x[i] = Math.max(0, x[i]!);
  };
}

export const cpuIm2col1d: CpuKernel = (b) => {
  const [tin, tout, C, k, stride, pad, row] = Array.from(b[0]!.u32.subarray(0, 7));
  const x = b[1]!.f32;
  const y = b[2]!.f32;
  for (let t = 0; t < tout!; t++) {
    const o = t * row!;
    y.fill(0, o, o + row!);
    for (let j = 0; j < k!; j++) {
      const src = t * stride! + j - pad!;
      if (src < 0 || src >= tin!) continue;
      y.set(x.subarray(src * C!, (src + 1) * C!), o + j * C!);
    }
  }
};

export const cpuAvgPool: CpuKernel = (b) => {
  const [side, k, C] = [b[0]!.u32[0]!, b[0]!.u32[1]!, b[0]!.u32[2]!];
  const x = b[1]!.f32;
  const y = b[2]!.f32;
  const out = side / k;
  for (let t = 0; t < out * out; t++) {
    const y0 = Math.floor(t / out) * k;
    const x0 = (t % out) * k;
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) s += x[((y0 + dy) * side + x0 + dx) * C + c]!;
      y[t * C + c] = s / (k * k);
    }
  }
};

export const cpuCopyRow: CpuKernel = (b) => {
  const D = b[0]!.u32[0]!;
  const { lastRow } = step(b[1]!);
  b[3]!.f32.set(b[2]!.f32.subarray(lastRow * D, lastRow * D + D));
};

// ---- sampling ----

const pcg = (v: number): number => {
  const s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
};

/**
 * The sample kernel's algorithm, sequentially: mask, penalty, temperature, top-k / top-p cut, Gumbel-max.
 * The adjusted logits are computed once (masked ids as -Infinity). The cut is the GPU kernel's 24-step
 * bisection, but each step only scans the ids still between its bounds: the rest are known to be in
 * (counted with their mass) or out.
 */
export function cpuSample(rows: boolean): CpuKernel {
  let vals = new Float32Array(0);
  let cand = new Int32Array(0);
  let mass = new Float64Array(0);
  return (b, grid) => {
    const V = b[0]!.u32[0]!;
    const st = step(b[1]!);
    const logits = b[2]!.f32;
    const seen = b[3]!.u32;
    const allowed = b[4]!.u32;
    const out = b[5]!.u32;
    const R = rows ? grid[1] : 1;
    const words = 1 + Math.ceil(V / 32);
    if (vals.length < V) {
      vals = new Float32Array(V);
      cand = new Int32Array(V);
      mass = new Float64Array(V);
    }
    const sampling = st.temperature > 0;
    const truncate = sampling && (st.topK > 0 || st.topP < 1);
    for (let row = 0; row < R; row++) {
      const base = row * V;
      const abase = row * words;
      // Out-of-range words read as 0, as WebGPU's robust buffer access gives them: no mask.
      const masked = (allowed[abase] ?? 0) !== 0;
      let vmax = -3e38;
      let vmin = 3e38;
      for (let i = 0; i < V; i++) {
        if (masked && (((allowed[abase + 1 + (i >>> 5)] ?? 0) >>> (i & 31)) & 1) === 0) {
          vals[i] = -Infinity;
          continue;
        }
        let v = logits[base + i]!;
        if (st.rep !== 1 && ((seen[i >>> 5]! >>> (i & 31)) & 1) === 1) v = v > 0 ? v / st.rep : v * st.rep;
        if (sampling) v /= st.temperature;
        vals[i] = v;
        v = vals[i]!;
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
      }
      let cut = -3e38;
      if (truncate) {
        let total = 0;
        let nc = 0;
        let lo = st.topK === 0 ? Math.max(vmin, vmax - 40) : vmin;
        let hi = vmax;
        for (let i = 0; i < V; i++) {
          const v = vals[i]!;
          if (v === -Infinity) continue;
          const e = Math.exp(v - vmax);
          total += e;
          if (v >= lo) {
            cand[nc] = i;
            mass[nc++] = e;
          }
        }
        // Ids at or above hi (always kept by later steps); candidates are those in [lo, hi).
        let aboveC = 0;
        let aboveM = 0;
        for (let it = 0; it < 24; it++) {
          const mid = Math.fround(0.5 * (lo + hi));
          let c = aboveC;
          let m = aboveM;
          for (let j = 0; j < nc; j++) if (vals[cand[j]!]! >= mid) { c++; m += mass[j]!; }
          const keep = (st.topK === 0 || c >= st.topK) && (st.topP >= 1 || m >= st.topP * total);
          let k = 0;
          if (keep) {
            lo = mid;
            for (let j = 0; j < nc; j++) if (vals[cand[j]!]! >= mid) { cand[k] = cand[j]!; mass[k++] = mass[j]!; }
          } else {
            hi = mid;
            for (let j = 0; j < nc; j++) {
              if (vals[cand[j]!]! >= mid) { aboveC++; aboveM += mass[j]!; }
              else { cand[k] = cand[j]!; mass[k++] = mass[j]!; }
            }
          }
          nc = k;
        }
        cut = lo;
      }
      const salt = pcg((st.seed ^ pcg((st.pos0 + row + Math.imul(st.lastRow, 7919)) >>> 0)) >>> 0);
      let best = -3e38;
      let bidx = 0;
      for (let i = 0; i < V; i++) {
        let v = vals[i]!;
        if (v === -Infinity || v < cut) continue;
        if (sampling) {
          const uu = ((pcg((i ^ salt) >>> 0) >>> 8) + 0.5) / 16777216;
          v = v - Math.log(-Math.log(uu));
        }
        if (v > best) {
          best = v;
          bidx = i;
        }
      }
      out[row] = bidx;
    }
  };
}

export const cpuAdvance: CpuKernel = (b) => {
  const st = b[0]!.u32;
  const id = b[1]!.u32[0]!;
  b[2]!.u32[0] = id;
  b[3]!.u32[st[4]!] = id;
  b[4]!.u32[id >>> 5]! |= 1 << (id & 31);
  st[4] = st[4]! + 1;
  st[0] = st[0]! + st[1]! + 1;
  st[1] = 0;
};
