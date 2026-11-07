/**
 * Plain TypeScript reference implementations. The kernel-check page compares every WGSL kernel
 * against these on random inputs; keep them obviously correct rather than fast.
 */

export function matmulRef(x: Float32Array, w: Float32Array, M: number, N: number, K: number): Float32Array {
  const y = new Float32Array(M * N);
  for (let m = 0; m < M; m++)
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += x[m * K + k]! * w[n * K + k]!;
      y[m * N + n] = acc;
    }
  return y;
}

export function rmsnormRef(x: Float32Array, w: Float32Array, D: number, eps: number): Float32Array {
  const y = new Float32Array(x.length);
  for (let r = 0; r < x.length / D; r++) {
    let ss = 0;
    for (let i = 0; i < D; i++) ss += x[r * D + i]! ** 2;
    const inv = 1 / Math.sqrt(ss / D + eps);
    for (let i = 0; i < D; i++) y[r * D + i] = x[r * D + i]! * inv * w[i]!;
  }
  return y;
}

export function ropeRef(x: Float32Array, T: number, H: number, D: number, pos0: number, invFreq: Float32Array, interleaved: boolean): Float32Array {
  const y = x.slice();
  const half = D / 2;
  for (let t = 0; t < T; t++)
    for (let h = 0; h < H; h++)
      for (let i = 0; i < half; i++) {
        const base = (t * H + h) * D;
        const [i0, i1] = interleaved ? [base + 2 * i, base + 2 * i + 1] : [base + i, base + i + half];
        const a = (pos0 + t) * invFreq[i]!;
        const x0 = x[i0]!;
        const x1 = x[i1]!;
        y[i0] = x0 * Math.cos(a) - x1 * Math.sin(a);
        y[i1] = x0 * Math.sin(a) + x1 * Math.cos(a);
      }
  return y;
}

/** q [T, H, D]; caches [maxSeq, Hkv, D] already containing positions 0..pos0+T-1. */
export function attentionRef(
  q: Float32Array,
  kc: Float32Array,
  vc: Float32Array,
  s: { T: number; H: number; Hkv: number; D: number; pos0: number; scale?: number; window?: number; span?: [number, number] },
): Float32Array {
  const { T, H, Hkv, D, pos0 } = s;
  const scale = s.scale ?? 1 / Math.sqrt(D);
  const out = new Float32Array(T * H * D);
  const group = H / Hkv;
  for (let t = 0; t < T; t++)
    for (let h = 0; h < H; h++) {
      const kvh = Math.floor(h / group);
      const pos = pos0 + t;
      // Causal, or the whole span for a query inside a bidirectional span.
      const n = s.span && pos >= s.span[0] && pos < s.span[1] ? s.span[1] : pos + 1;
      // Keys lo…n-1: all of them, or the last `window` (HF masks q - k >= window).
      const lo = s.window ? Math.max(0, n - s.window) : 0;
      const scores = new Float64Array(n);
      for (let j = lo; j < n; j++) {
        let d = 0;
        for (let i = 0; i < D; i++) d += q[(t * H + h) * D + i]! * kc[(j * Hkv + kvh) * D + i]!;
        scores[j] = d * scale;
      }
      const mx = Math.max(...scores.subarray(lo));
      let sum = 0;
      for (let j = lo; j < n; j++) sum += scores[j] = Math.exp(scores[j]! - mx);
      for (let i = 0; i < D; i++) {
        let acc = 0;
        for (let j = lo; j < n; j++) acc += scores[j]! * vc[(j * Hkv + kvh) * D + i]!;
        out[(t * H + h) * D + i] = acc / sum;
      }
    }
  return out;
}

export function siluMulRef(g: Float32Array, u: Float32Array): Float32Array {
  return g.map((a, i) => (a / (1 + Math.exp(-a))) * u[i]!);
}

/** GELU, tanh approximation (PyTorch `gelu(approximate="tanh")`). */
export function geluTanh(a: number): number {
  return 0.5 * a * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (a + 0.044715 * a ** 3)));
}

export function geluMulRef(g: Float32Array, u: Float32Array): Float32Array {
  return g.map((a, i) => geluTanh(a) * u[i]!);
}

export function argmaxRef(x: Float32Array): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i]! > x[best]!) best = i;
  return best;
}

export function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}
