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
  // One frequency per rotated pair: D / 2 of them, or fewer when only the head's first part is rotated.
  const half = invFreq.length;
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
  s: { T: number; H: number; Hkv: number; D: number; pos0: number; scale?: number; window?: number; span?: [number, number]; bias?: (h: number, qpos: number, kpos: number) => number },
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
        scores[j] = d * scale + (s.bias ? s.bias(h, pos, j) : 0);
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

/** erf by its Taylor series (to double precision for |x| <= 5; ±1 beyond). */
export function erf(x: number): number {
  if (Math.abs(x) > 5) return Math.sign(x);
  let term = x, sum = x;
  for (let n = 1; n < 200 && Math.abs(term) > 1e-17 * Math.abs(sum); n++) {
    term *= (-x * x) / n;
    sum += term / (2 * n + 1);
  }
  return (2 / Math.sqrt(Math.PI)) * sum;
}

/** GELU in its exact form, x · Φ(x). */
export function geluErf(a: number): number {
  return 0.5 * a * (1 + erf(a / Math.SQRT2));
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

/** Causal depthwise convolution with SiLU over x [T, C], weights [C, K], zeros before the first row (F.conv1d with padding K − 1, cut to T). */
export function causalConvRef(x: Float32Array, w: Float32Array, T: number, C: number, K: number): Float32Array {
  const y = new Float32Array(T * C);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < C; c++) {
      let acc = 0;
      for (let j = 0; j < K; j++) if (t + j - (K - 1) >= 0) acc += w[c * K + j]! * x[(t + j - (K - 1)) * C + c]!;
      y[t * C + c] = acc / (1 + Math.exp(-acc));
    }
  return y;
}

/**
 * Linear attention's gated delta rule as transformers' torch_recurrent_gated_delta_rule runs it, one
 * head at a time with its state [Dk, Dv]. qk [T, 2·Hk·Dk] (queries then keys, normalised), values in
 * the last Hv·Dv columns of conv [T, C], a and b [T, Hv]. Returns [T, Hv·Dv].
 */
export function gatedDeltaRef(
  qk: Float32Array,
  conv: Float32Array,
  a: Float32Array,
  b: Float32Array,
  aLog: Float32Array,
  dt: Float32Array,
  s: { T: number; keyHeads: number; valueHeads: number; keyDim: number; valueDim: number },
): Float32Array {
  const { T, keyHeads: Hk, valueHeads: Hv, keyDim: Dk, valueDim: Dv } = s;
  const keys = 2 * Hk * Dk;
  const C = keys + Hv * Dv;
  const out = new Float32Array(T * Hv * Dv);
  for (let h = 0; h < Hv; h++) {
    const kh = Math.floor(h / (Hv / Hk));
    const S = Array.from({ length: Dk }, () => new Float64Array(Dv));
    for (let t = 0; t < T; t++) {
      const q = Array.from({ length: Dk }, (_, i) => qk[t * keys + kh * Dk + i]! / Math.sqrt(Dk));
      const k = Array.from({ length: Dk }, (_, i) => qk[t * keys + Hk * Dk + kh * Dk + i]!);
      const g = -Math.exp(aLog[h]!) * Math.log1p(Math.exp(a[t * Hv + h]! + dt[h]!));
      const beta = 1 / (1 + Math.exp(-b[t * Hv + h]!));
      for (const row of S) for (let j = 0; j < Dv; j++) row[j]! *= Math.exp(g);
      for (let j = 0; j < Dv; j++) {
        let mem = 0;
        for (let i = 0; i < Dk; i++) mem += S[i]![j]! * k[i]!;
        const delta = (conv[t * C + keys + h * Dv + j]! - mem) * beta;
        let o = 0;
        for (let i = 0; i < Dk; i++) {
          S[i]![j]! += k[i]! * delta;
          o += S[i]![j]! * q[i]!;
        }
        out[(t * Hv + h) * Dv + j] = o;
      }
    }
  }
  return out;
}
