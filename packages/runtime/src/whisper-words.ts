/**
 * Whisper word-level timestamps, as transformers.js computes them: the decoder's cross-attention
 * weights on the model's alignment heads are normalized over tokens, median-filtered over time,
 * averaged over heads and aligned to the audio frames by dynamic time warping; each token starts
 * where the warping path first reaches it. Tokens are then grouped into words.
 */

/** Median filter with mirrored edges (transformers.js's medianFilter). */
export function medianFilter(data: Float32Array, width: number): Float32Array {
  if (width % 2 === 0 || width <= 0) throw new Error("median filter width must be a positive odd number");
  const out = new Float32Array(data.length);
  const buf = new Float32Array(width);
  const half = width >> 1;
  for (let i = 0; i < data.length; i++) {
    let n = 0;
    for (let j = -half; j <= half; j++) {
      let k = i + j;
      if (k < 0) k = -k;
      else if (k >= data.length) k = 2 * (data.length - 1) - k;
      buf[n++] = data[k]!;
    }
    buf.sort();
    out[i] = buf[half]!;
  }
  return out;
}

/**
 * Minimum-cost monotonic path through a cost matrix [rows][cols] from (0, 0) to the far corner, as
 * [row indices, column indices] along the path (transformers.js's dynamic_time_warping, same ties).
 */
export function dynamicTimeWarping(matrix: ArrayLike<number>, rows: number, cols: number): [number[], number[]] {
  const W = cols + 1;
  const cost = new Float64Array((rows + 1) * W).fill(Infinity);
  const trace = new Int8Array((rows + 1) * W).fill(-1);
  cost[0] = 0;
  for (let j = 1; j <= cols; j++) {
    for (let i = 1; i <= rows; i++) {
      const c0 = cost[(i - 1) * W + j - 1]!;
      const c1 = cost[(i - 1) * W + j]!;
      const c2 = cost[i * W + j - 1]!;
      let c: number;
      let t: number;
      if (c0 < c1 && c0 < c2) [c, t] = [c0, 0];
      else if (c1 < c0 && c1 < c2) [c, t] = [c1, 1];
      else [c, t] = [c2, 2];
      cost[i * W + j] = matrix[(i - 1) * cols + j - 1]! + c;
      trace[i * W + j] = t;
    }
  }
  for (let j = 0; j <= cols; j++) trace[j] = 2;
  for (let i = 0; i <= rows; i++) trace[i * W] = 1;
  let i = rows;
  let j = cols;
  const ti: number[] = [];
  const tj: number[] = [];
  while (i > 0 || j > 0) {
    ti.push(i - 1);
    tj.push(j - 1);
    const t = trace[i * W + j];
    if (t === 0) { i--; j--; }
    else if (t === 1) i--;
    else if (t === 2) j--;
    else throw new Error(`dynamic time warping: unexpected trace at [${i}, ${j}]`);
  }
  return [ti.reverse(), tj.reverse()];
}

/**
 * Start time of every decoder position from its cross-attention weights `weights` [rows, heads, S]
 * (rows = positions fed to the decoder; the first `nInit` are the prompt): transformers.js's
 * _extract_token_timestamps with float32 intermediates. Returns rows + 1 times (prompt positions 0,
 * the last repeated for the final token), in seconds.
 */
export function whisperTokenTimestamps(
  weights: Float32Array,
  o: { rows: number; heads: number; S: number; frames: number; nInit: number; medianWidth?: number; timePrecision?: number },
): number[] {
  const { rows, heads, S, nInit } = o;
  const F = Math.max(0, Math.min(o.frames, S));
  const width = o.medianWidth ?? 7;
  const prec = o.timePrecision ?? 0.02;
  const fr = Math.fround;
  const at = (r: number, h: number, f: number) => weights[(r * heads + h) * S + f]!;
  const n = rows - nInit;
  // Head-averaged normalized, filtered weights of the generated positions [n, F].
  const avg = new Float32Array(n * F);
  const row = new Float32Array(F);
  for (let h = 0; h < heads; h++) {
    const mean = new Float32Array(F);
    const std = new Float32Array(F);
    for (let r = 0; r < rows; r++) for (let f = 0; f < F; f++) mean[f] = fr(mean[f]! + at(r, h, f));
    if (rows !== 1) for (let f = 0; f < F; f++) mean[f] = fr(mean[f]! / rows);
    for (let r = 0; r < rows; r++) for (let f = 0; f < F; f++) std[f] = fr(std[f]! + (at(r, h, f) - mean[f]!) ** 2);
    for (let f = 0; f < F; f++) std[f] = fr(Math.sqrt(std[f]! / rows));
    for (let r = nInit; r < rows; r++) {
      for (let f = 0; f < F; f++) row[f] = (at(r, h, f) - mean[f]!) / std[f]!;
      const m = medianFilter(row, width);
      const base = (r - nInit) * F;
      for (let f = 0; f < F; f++) avg[base + f] = fr(avg[base + f]! + m[f]!);
    }
  }
  const neg = new Float64Array(n * F);
  for (let i = 0; i < n * F; i++) neg[i] = -(heads !== 1 ? fr(avg[i]! / heads) : avg[i]!);
  const out: number[] = new Array(nInit).fill(0);
  if (n > 0 && F > 0) {
    const [ti, tj] = dynamicTimeWarping(neg, n, F);
    for (let k = 0; k < ti.length; k++) if (k === 0 || ti[k] !== ti[k - 1]) out.push(fr(tj[k]! * prec));
  }
  if (out.length > nInit) out.push(out.at(-1)!);
  while (out.length < rows + 1) out.push(0);
  return out;
}

/** A word with its start and end in seconds. */
export interface WhisperWord {
  text: string;
  timestamp: [number, number | null];
}

const PUNCTUATION = "\\p{P}\\u0021-\\u002F\\u003A-\\u0040\\u005B-\\u0060\\u007B-\\u007E";
/** transformers.js's punctuation test; global, so a match makes the next test start past it, as there. */
export const PUNCTUATION_ONLY = new RegExp(`^[${PUNCTUATION}]+$`, "gu");

/** Groups tokens into pieces that decode to whole characters. */
function splitOnUnicode(tokens: number[], decode: (ids: number[]) => string): [string[], number[][]] {
  const full = decode(tokens);
  const words: string[] = [];
  const indices: number[][] = [];
  let cur: number[] = [];
  let curIdx: number[] = [];
  let offset = 0;
  for (let i = 0; i < tokens.length; i++) {
    cur.push(tokens[i]!);
    curIdx.push(i);
    const d = decode(cur);
    if (!d.includes("�") || full[offset + d.indexOf("�")] === "�") {
      words.push(d);
      indices.push(curIdx);
      cur = [];
      curIdx = [];
      offset += d.length;
    }
  }
  return [words, indices];
}

/** Groups tokens into words at spaces and punctuation. */
function splitOnSpaces(tokens: number[], decode: (ids: number[]) => string, eos: number): [string[], number[][]] {
  const [subs, subIdx] = splitOnUnicode(tokens, decode);
  const words: string[] = [];
  const indices: number[][] = [];
  subs.forEach((s, i) => {
    const special = tokens[subIdx[i]![0]!]! >= eos;
    if (special || s.startsWith(" ") || PUNCTUATION_ONLY.test(s.trim()) || !words.length) {
      words.push(s);
      indices.push([...subIdx[i]!]);
    } else {
      words[words.length - 1] += s;
      indices[indices.length - 1]!.push(...subIdx[i]!);
    }
  });
  return [words, indices];
}

/** Attaches leading and trailing punctuation to its neighbouring word. */
function mergePunctuation(words: string[], indices: number[][]): [string[], number[][]] {
  const w = [...words];
  const ix = indices.map((x) => [...x]);
  const prepended = "\"'“¡¿([{-";
  const appended = "\"'.。,，!！?？:：”)]}、";
  for (let i = w.length - 2, j = w.length - 1; i >= 0; i--) {
    if (w[i]!.startsWith(" ") && prepended.includes(w[i]!.trim())) {
      w[j] = w[i]! + w[j]!;
      ix[j] = [...ix[i]!, ...ix[j]!];
      w[i] = "";
      ix[i] = [];
    } else j = i;
  }
  for (let i = 0, j = 1; j < w.length; j++) {
    if (!w[i]!.endsWith(" ") && appended.includes(w[j]!)) {
      w[i] += w[j]!;
      ix[i] = [...ix[i]!, ...ix[j]!];
      w[j] = "";
      ix[j] = [];
    } else i = j;
  }
  return [w.filter((x) => x), ix.filter((x) => x.length)];
}

const NO_SPACES = new Set(["zh", "ja", "th", "lo", "my"]);

/** Words of a token list with times from per-token [start, end] (transformers.js's collateWordTimestamps). */
export function collateWords(
  tokens: number[],
  times: [number, number | null][],
  o: { decode: (ids: number[]) => string; eos: number; language?: string },
): WhisperWord[] {
  const [words0, idx0] = o.language && NO_SPACES.has(o.language) ? splitOnUnicode(tokens, o.decode) : splitOnSpaces(tokens, o.decode, o.eos);
  const [words, idx] = mergePunctuation(words0, idx0);
  return words.map((text, i) => ({ text, timestamp: [times[idx[i]![0]!]![0], times[idx[i]!.at(-1)!]![1]] }));
}
