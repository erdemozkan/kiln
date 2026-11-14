/** A scalar field of a WGSL uniform struct. Only 4-byte scalars, so there is no padding to think about. */
export type Field = readonly ["u32" | "i32" | "f32", number];

export const u32 = (v: number): Field => ["u32", v];
export const i32 = (v: number): Field => ["i32", v];
export const f32 = (v: number): Field => ["f32", v];

/** Packs fields into a uniform-sized buffer (multiple of 16 bytes). */
export function packUniform(fields: readonly Field[]): Uint8Array {
  const bytes = new Uint8Array(Math.max(16, Math.ceil((fields.length * 4) / 16) * 16));
  const dv = new DataView(bytes.buffer);
  fields.forEach(([type, v], i) => {
    if (type === "f32") dv.setFloat32(i * 4, v, true);
    else if (type === "i32") dv.setInt32(i * 4, v, true);
    else dv.setUint32(i * 4, v >>> 0, true);
  });
  return bytes;
}

/**
 * Per-step values shared by every kernel in a plan. Changing these between replays is how a single
 * recorded dispatch list serves every decode position. They live in a small storage buffer so the
 * GPU can advance them itself between chained decode steps.
 */
export interface StepParams {
  /** Absolute position of the first token in this step. */
  pos0: number;
  /** Row of the activations whose logits get sampled (last real token when the bucket is padded). */
  lastRow: number;
  seed: number;
  /** 0 means greedy. */
  temperature: number;
  /** Next free slot in the sampled-token history (advanced on the GPU by the `advance` kernel). */
  n?: number;
  /** Keep only the k most likely tokens (0 = off). */
  topK?: number;
  /** Keep the smallest set of tokens whose probability reaches p (1 = off). */
  topP?: number;
  /** Divide positive / multiply negative logits of already-seen tokens by this (1 = off). */
  repetitionPenalty?: number;
  /**
   * Bidirectional span [spanLo, spanHi) of positions: a query inside it sees every key of the span, not
   * only earlier ones (Gemma 3's image tokens, a vision encoder). Default none.
   */
  spanLo?: number;
  spanHi?: number;
  /**
   * Rotary positions on three axes (Qwen 3.5 with images): a token's position is its index minus
   * `posShift`, and the tokens at indices [imgLo, imgHi) are one image, `imgW` tokens to a row, whose
   * time position is `imgPos` and whose height and width positions are `imgPos` plus the token's row
   * and column (see `ROPE_POS_WGSL`). All 0 by default: the position is the index.
   */
  posShift?: number;
  imgLo?: number;
  imgHi?: number;
  imgW?: number;
  imgPos?: number;
}

export const STEP_WGSL = /* wgsl */ `struct Step { pos0: u32, lastRow: u32, seed: u32, temperature: f32, n: u32, topK: u32, topP: f32, repPenalty: f32, spanLo: u32, spanHi: u32, posShift: u32, imgLo: u32, imgHi: u32, imgW: u32, imgPos: u32, _a: u32 }`;

/**
 * The rotary position of the token at index `idx` for frequency `i` (needs `st: Step`). Text: the
 * index minus the shift, on every axis. Inside the image span, frequencies go to the axes in turn
 * (i % 3: time, height, width), as Qwen's interleaved M-RoPE lays them out.
 */
export const ROPE_POS_WGSL = /* wgsl */ `fn ropePos(idx: u32, i: u32) -> u32 {
  if (idx >= st.imgLo && idx < st.imgHi) {
    let j = idx - st.imgLo;
    let w = max(st.imgW, 1u);
    let ax = i % 3u;
    return st.imgPos + select(0u, select(j % w, j / w, ax == 1u), ax != 0u);
  }
  return idx - st.posShift;
}`;

/** `ROPE_POS_WGSL` on the CPU. */
export function ropePos(s: { posShift: number; imgLo: number; imgHi: number; imgW: number; imgPos: number }, idx: number, i: number): number {
  if (idx >= s.imgLo && idx < s.imgHi) {
    const j = idx - s.imgLo;
    const w = Math.max(s.imgW, 1);
    const ax = i % 3;
    return s.imgPos + (ax === 0 ? 0 : ax === 1 ? Math.floor(j / w) : j % w);
  }
  return idx - s.posShift;
}

/** Bytes of the step buffer. */
export const STEP_BYTES = 64;

export function packStep(s: StepParams): Uint8Array {
  return packUniform([
    u32(s.pos0), u32(s.lastRow), u32(s.seed), f32(s.temperature), u32(s.n ?? 0), u32(s.topK ?? 0), f32(s.topP ?? 1), f32(s.repetitionPenalty ?? 1),
    u32(s.spanLo ?? 0), u32(s.spanHi ?? 0), u32(s.posShift ?? 0), u32(s.imgLo ?? 0),
    u32(s.imgHi ?? 0), u32(s.imgW ?? 0), u32(s.imgPos ?? 0), u32(0),
  ]);
}
