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
 * recorded dispatch list serves every decode position. They live in a 32-byte storage buffer so the
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
}

export const STEP_WGSL = /* wgsl */ `struct Step { pos0: u32, lastRow: u32, seed: u32, temperature: f32, n: u32, topK: u32, topP: f32, repPenalty: f32, spanLo: u32, spanHi: u32, _a: u32, _b: u32 }`;

/** Bytes of the step buffer. */
export const STEP_BYTES = 48;

export function packStep(s: StepParams): Uint8Array {
  return packUniform([
    u32(s.pos0), u32(s.lastRow), u32(s.seed), f32(s.temperature), u32(s.n ?? 0), u32(s.topK ?? 0), f32(s.topP ?? 1), f32(s.repetitionPenalty ?? 1),
    u32(s.spanLo ?? 0), u32(s.spanHi ?? 0), u32(0), u32(0),
  ]);
}
