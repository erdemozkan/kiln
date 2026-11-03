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
  s: { T: number; H: number; Hkv: number; D: number; pos0: number },
): Float32Array {
  const { T, H, Hkv, D, pos0 } = s;
  const out = new Float32Array(T * H * D);
  const group = H / Hkv;
  for (let t = 0; t < T; t++)
    for (let h = 0; h < H; h++) {
      const kvh = Math.floor(h / group);
      const n = pos0 + t + 1;
      const scores = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        let d = 0;
        for (let i = 0; i < D; i++) d += q[(t * H + h) * D + i]! * kc[(j * Hkv + kvh) * D + i]!;
        scores[j] = d / Math.sqrt(D);
      }
      const mx = Math.max(...scores);
      let sum = 0;
      for (let j = 0; j < n; j++) sum += scores[j] = Math.exp(scores[j]! - mx);
      for (let i = 0; i < D; i++) {
        let acc = 0;
        for (let j = 0; j < n; j++) acc += scores[j]! * vc[(j * Hkv + kvh) * D + i]!;
        out[(t * H + h) * D + i] = acc / sum;
      }
    }
  return out;
}

export function siluMulRef(g: Float32Array, u: Float32Array): Float32Array {
  return g.map((a, i) => (a / (1 + Math.exp(-a))) * u[i]!);
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
