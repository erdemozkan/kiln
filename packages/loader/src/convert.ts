import type { SrcType } from "./entry";

const f16Table = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    t[h] = e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024);
  }
  return t;
})();

const u16 = (b: Uint8Array) => new Uint16Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));

export function f16ToF32(bytes: Uint8Array): Float32Array {
  const src = u16(bytes);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = f16Table[src[i]!]!;
  return out;
}

export function bf16ToF32(bytes: Uint8Array): Float32Array {
  const src = u16(bytes);
  const bits = new Uint32Array(src.length);
  for (let i = 0; i < src.length; i++) bits[i] = src[i]! << 16;
  return new Float32Array(bits.buffer);
}

export function dequantQ8_0(bytes: Uint8Array, n: number): Float32Array {
  const out = new Float32Array(n);
  const q = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let b = 0; b < n / 32; b++) {
    const o = b * 34;
    const d = f16Table[bytes[o]! | (bytes[o + 1]! << 8)]!;
    for (let j = 0; j < 32; j++) out[b * 32 + j] = q[o + 2 + j]! * d;
  }
  return out;
}

export function toF32(type: Exclude<SrcType, "Q4_0">, bytes: Uint8Array, n: number): Float32Array {
  switch (type) {
    case "F32":
      return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + n * 4));
    case "F16":
      return f16ToF32(bytes);
    case "BF16":
      return bf16ToF32(bytes);
    case "Q8_0":
      return dequantQ8_0(bytes, n);
    case "Q4_1":
      return dequantQ4_1(bytes, n);
    case "Q5_0":
      return dequantQ5(bytes, n, false);
    case "Q5_1":
      return dequantQ5(bytes, n, true);
    case "Q4_K":
      return dequantQ4_K(bytes, n);
    case "Q5_K":
      return dequantQ5_K(bytes, n);
    case "Q6_K":
      return dequantQ6_K(bytes, n);
  }
}

/** Q4_1: d, m (f16), 16 bytes of nibbles; x = q * d + m. Low nibbles are elements 0–15, high 16–31. */
export function dequantQ4_1(bytes: Uint8Array, n: number): Float32Array {
  const out = new Float32Array(n);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let b = 0; b < n / 32; b++) {
    const o = b * 20;
    const d = f16Table[dv.getUint16(o, true)]!;
    const m = f16Table[dv.getUint16(o + 2, true)]!;
    for (let j = 0; j < 16; j++) {
      const q = bytes[o + 4 + j]!;
      out[b * 32 + j] = (q & 15) * d + m;
      out[b * 32 + j + 16] = (q >> 4) * d + m;
    }
  }
  return out;
}

/** Q5_0 (d, qh u32, nibbles; x = (q - 16) * d) and Q5_1 (d, m, qh, nibbles; x = q * d + m). */
export function dequantQ5(bytes: Uint8Array, n: number, withMin: boolean): Float32Array {
  const out = new Float32Array(n);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const size = withMin ? 24 : 22;
  for (let b = 0; b < n / 32; b++) {
    const o = b * size;
    const d = f16Table[dv.getUint16(o, true)]!;
    const m = withMin ? f16Table[dv.getUint16(o + 2, true)]! : 0;
    const h = withMin ? 4 : 2;
    const qh = dv.getUint32(o + h, true);
    for (let j = 0; j < 16; j++) {
      const q = bytes[o + h + 4 + j]!;
      const x0 = (q & 15) | (((qh >>> j) << 4) & 16);
      const x1 = (q >> 4) | ((qh >>> (j + 12)) & 16);
      out[b * 32 + j] = withMin ? x0 * d + m : (x0 - 16) * d;
      out[b * 32 + j + 16] = withMin ? x1 * d + m : (x1 - 16) * d;
    }
  }
  return out;
}

/** 6-bit scale and min j of a k-quant super-block's 12-byte scale table (llama.cpp get_scale_min_k4). */
function scaleMinK4(j: number, q: Uint8Array, o: number): [number, number] {
  if (j < 4) return [q[o + j]! & 63, q[o + j + 4]! & 63];
  return [(q[o + j + 4]! & 15) | ((q[o + j - 4]! >> 6) << 4), (q[o + j + 4]! >> 4) | ((q[o + j]! >> 6) << 4)];
}

/** Q4_K: 256-element super-blocks of 8 × 32, each with a 6-bit scale and min; x = d·sc·q − dmin·m. */
export function dequantQ4_K(bytes: Uint8Array, n: number): Float32Array {
  const out = new Float32Array(n);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let b = 0; b < n / 256; b++) {
    const o = b * 144;
    const d = f16Table[dv.getUint16(o, true)]!;
    const dmin = f16Table[dv.getUint16(o + 2, true)]!;
    let q = o + 16;
    let y = b * 256;
    for (let j = 0, is = 0; j < 256; j += 64, is += 2, q += 32, y += 64) {
      const [sc1, m1] = scaleMinK4(is, bytes, o + 4);
      const [sc2, m2] = scaleMinK4(is + 1, bytes, o + 4);
      for (let l = 0; l < 32; l++) {
        out[y + l] = d * sc1 * (bytes[q + l]! & 15) - dmin * m1;
        out[y + 32 + l] = d * sc2 * (bytes[q + l]! >> 4) - dmin * m2;
      }
    }
  }
  return out;
}

/** Q5_K: Q4_K plus a fifth bit per element from qh[32]. */
export function dequantQ5_K(bytes: Uint8Array, n: number): Float32Array {
  const out = new Float32Array(n);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let b = 0; b < n / 256; b++) {
    const o = b * 176;
    const d = f16Table[dv.getUint16(o, true)]!;
    const dmin = f16Table[dv.getUint16(o + 2, true)]!;
    const qh = o + 16;
    let ql = o + 48;
    let y = b * 256;
    let u1 = 1;
    let u2 = 2;
    for (let j = 0, is = 0; j < 256; j += 64, is += 2, ql += 32, y += 64, u1 <<= 2, u2 <<= 2) {
      const [sc1, m1] = scaleMinK4(is, bytes, o + 4);
      const [sc2, m2] = scaleMinK4(is + 1, bytes, o + 4);
      for (let l = 0; l < 32; l++) {
        const h = bytes[qh + l]!;
        out[y + l] = d * sc1 * ((bytes[ql + l]! & 15) + (h & u1 ? 16 : 0)) - dmin * m1;
        out[y + 32 + l] = d * sc2 * ((bytes[ql + l]! >> 4) + (h & u2 ? 16 : 0)) - dmin * m2;
      }
    }
  }
  return out;
}

/** Q6_K: 256 elements as 4 low bits (ql) + 2 high bits (qh), 16 int8 scales, one f16 d; x = d·sc·(q − 32). */
export function dequantQ6_K(bytes: Uint8Array, n: number): Float32Array {
  const out = new Float32Array(n);
  const i8 = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let b = 0; b < n / 256; b++) {
    const o = b * 210;
    const d = f16Table[bytes[o + 208]! | (bytes[o + 209]! << 8)]!;
    let ql = o;
    let qh = o + 128;
    let sc = o + 192;
    let y = b * 256;
    for (let half = 0; half < 2; half++, ql += 64, qh += 32, sc += 8, y += 128) {
      for (let l = 0; l < 32; l++) {
        const is = sc + (l >> 4);
        const h = bytes[qh + l]!;
        const a = bytes[ql + l]!;
        const c = bytes[ql + l + 32]!;
        out[y + l] = d * i8[is]! * (((a & 15) | ((h & 3) << 4)) - 32);
        out[y + l + 32] = d * i8[is + 2]! * (((c & 15) | (((h >> 2) & 3) << 4)) - 32);
        out[y + l + 64] = d * i8[is + 4]! * (((a >> 4) | (((h >> 4) & 3) << 4)) - 32);
        out[y + l + 96] = d * i8[is + 6]! * (((c >> 4) | (((h >> 6) & 3) << 4)) - 32);
      }
    }
  }
  return out;
}

/** Kiln's GPU q4 layout: 8 sequential nibbles per u32 (value + 8) and one f32 scale per 32 values. */
export interface Q4Planes {
  nibbles: Uint32Array;
  scales: Float32Array;
}

/**
 * llama.cpp's Q4_0 rule: d = (signed max magnitude) / -8, q = clamp(floor(x / d + 8.5)), with the
 * division done exactly rather than as a multiply by 1 / d.
 */
export function quantizeQ4(x: Float32Array): Q4Planes {
  if (x.length % 32 !== 0) throw new Error(`quantizeQ4: length ${x.length} is not a multiple of 32`);
  const nibbles = new Uint32Array(x.length / 8);
  const scales = new Float32Array(x.length / 32);
  for (let b = 0; b < scales.length; b++) {
    const o = b * 32;
    let hi = 0;
    let lo = 0;
    for (let j = 0; j < 32; j++) {
      const v = x[o + j]!;
      if (v > hi) hi = v;
      if (v < lo) lo = v;
    }
    // The signed value of largest magnitude; on a tie, the first one, as llama.cpp picks it.
    let max = -lo > hi ? lo : hi;
    if (-lo === hi && hi !== 0) for (let j = 0; j < 32; j++) if (x[o + j] === hi || x[o + j] === lo) { max = x[o + j]!; break; }
    const d = max / -8;
    scales[b] = d;
    if (!d) {
      nibbles.fill(0x88888888, b * 4, b * 4 + 4);
      continue;
    }
    for (let w = 0; w < 4; w++) {
      let word = 0;
      for (let k = 7; k >= 0; k--) {
        // x / d in double is exact enough that floor never lands on the wrong side of an integer (a
        // multiply by 1 / d does on ties, which block-quantized sources hit often). Truncation equals
        // floor here: anything below 0 clamps to 0 either way.
        const v = x[o + w * 8 + k]! / d + 8.5;
        word = (word << 4) | (v >= 15 ? 15 : v > 0 ? v | 0 : 0);
      }
      nibbles[b * 4 + w] = word;
    }
  }
  return { nibbles, scales };
}

/** Reorders GGUF Q4_0 blocks (low nibbles = 0..15, high = 16..31, f16 scale) into Kiln planes. */
export function repackQ4_0(bytes: Uint8Array, n: number): Q4Planes {
  const nibbles = new Uint32Array(n / 8);
  const scales = new Float32Array(n / 32);
  for (let b = 0; b < n / 32; b++) {
    const o = b * 18;
    scales[b] = f16Table[bytes[o]! | (bytes[o + 1]! << 8)]!;
    // Words 0-1 take the low nibbles of bytes 0-7 and 8-15 (elements 0-15), words 2-3 the high ones.
    for (let h = 0; h < 2; h++) {
      let lo = 0;
      let hi = 0;
      for (let k = 7; k >= 0; k--) {
        const byte = bytes[o + 2 + h * 8 + k]!;
        lo = (lo << 4) | (byte & 15);
        hi = (hi << 4) | (byte >> 4);
      }
      nibbles[b * 4 + h] = lo;
      nibbles[b * 4 + 2 + h] = hi;
    }
  }
  return { nibbles, scales };
}

export function dequantQ4(p: Q4Planes): Float32Array {
  const out = new Float32Array(p.nibbles.length * 8);
  for (let e = 0; e < out.length; e++) out[e] = (((p.nibbles[e >> 3]! >>> ((e & 7) * 4)) & 15) - 8) * p.scales[e >> 5]!;
  return out;
}

/** Kiln's GPU q8 layout: 4 sequential int8 per u32 (little-endian bytes) and one f32 scale per 32 values. */
export interface Q8Planes {
  bytes: Uint32Array;
  scales: Float32Array;
}

/**
 * llama.cpp's Q8_0 rule, rounded exactly: d = amax / 127 (as an f32), q = x / d rounded to nearest
 * with ties away from zero. The tie is decided by (2k - 1)·d against 2|x|, both exact in double, not
 * by a rounded quotient, so the GPU converter can give the same bits.
 */
export function quantizeQ8(x: Float32Array): Q8Planes {
  if (x.length % 32 !== 0) throw new Error(`quantizeQ8: length ${x.length} is not a multiple of 32`);
  const bytes = new Uint32Array(x.length / 4);
  const q = new Int8Array(bytes.buffer);
  const scales = new Float32Array(x.length / 32);
  for (let b = 0; b < scales.length; b++) {
    let amax = 0;
    for (let j = 0; j < 32; j++) amax = Math.max(amax, Math.abs(x[b * 32 + j]!));
    const d = Math.fround(amax / 127);
    scales[b] = d;
    if (!d) continue;
    for (let j = 0; j < 32; j++) {
      const v = x[b * 32 + j]!;
      const a = Math.abs(v);
      let k = Math.min(127, Math.round(a / d));
      if (k < 127 && (2 * k + 1) * d <= 2 * a) k++;
      else if ((2 * k - 1) * d > 2 * a) k--;
      q[b * 32 + j] = v < 0 ? -k : k;
    }
  }
  return { bytes, scales };
}

/** GGUF Q8_0 blocks (f16 scale, 32 int8) into Kiln planes; exact. */
export function repackQ8_0(src: Uint8Array, n: number): Q8Planes {
  const bytes = new Uint32Array(n / 4);
  const q = new Uint8Array(bytes.buffer);
  const scales = new Float32Array(n / 32);
  const dv = new DataView(src.buffer, src.byteOffset, src.byteLength);
  for (let b = 0; b < n / 32; b++) {
    const o = b * 34;
    scales[b] = f16Table[dv.getUint16(o, true)]!;
    q.set(src.subarray(o + 2, o + 34), b * 32);
  }
  return { bytes, scales };
}

export function dequantQ8(p: Q8Planes): Float32Array {
  const q = new Int8Array(p.bytes.buffer, p.bytes.byteOffset, p.bytes.byteLength);
  const out = new Float32Array(q.length);
  for (let e = 0; e < out.length; e++) out[e] = q[e]! * p.scales[e >> 5]!;
  return out;
}
