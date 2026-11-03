import { UnsupportedTensorError, rowBytes, type SrcType, type TensorEntry } from "./entry";
import type { ByteSource } from "./source";

export type GGUFValue = number | bigint | boolean | string | GGUFValue[];

export interface GGUFFile {
  version: number;
  metadata: Map<string, GGUFValue>;
  tensors: TensorEntry[];
}

const GGML_TYPES: Record<number, SrcType | undefined> = {
  0: "F32",
  1: "F16",
  2: "Q4_0",
  3: "Q4_1",
  6: "Q5_0",
  7: "Q5_1",
  8: "Q8_0",
  12: "Q4_K",
  13: "Q5_K",
  14: "Q6_K",
  30: "BF16",
};
const GGML_NAMES: Record<number, string> = { 10: "Q2_K", 11: "Q3_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL" };

class NeedMore extends Error {}

class Reader {
  pos = 0;
  private readonly dv: DataView;
  private readonly text = new TextDecoder();
  constructor(private readonly bytes: Uint8Array) {
    this.dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  private need(n: number): void {
    if (this.pos + n > this.bytes.byteLength) throw new NeedMore();
  }
  u8() { this.need(1); return this.dv.getUint8(this.pos++); }
  i8() { this.need(1); return this.dv.getInt8(this.pos++); }
  u16() { this.need(2); const v = this.dv.getUint16(this.pos, true); this.pos += 2; return v; }
  i16() { this.need(2); const v = this.dv.getInt16(this.pos, true); this.pos += 2; return v; }
  u32() { this.need(4); const v = this.dv.getUint32(this.pos, true); this.pos += 4; return v; }
  i32() { this.need(4); const v = this.dv.getInt32(this.pos, true); this.pos += 4; return v; }
  f32() { this.need(4); const v = this.dv.getFloat32(this.pos, true); this.pos += 4; return v; }
  f64() { this.need(8); const v = this.dv.getFloat64(this.pos, true); this.pos += 8; return v; }
  u64() { this.need(8); const v = this.dv.getBigUint64(this.pos, true); this.pos += 8; return v; }
  i64() { this.need(8); const v = this.dv.getBigInt64(this.pos, true); this.pos += 8; return v; }
  /** u64 that must fit in a JS number (counts, sizes, offsets). */
  size() { return Number(this.u64()); }
  str() {
    const n = this.size();
    this.need(n);
    const s = this.text.decode(this.bytes.subarray(this.pos, this.pos + n));
    this.pos += n;
    return s;
  }
  value(type: number): GGUFValue {
    switch (type) {
      case 0: return this.u8();
      case 1: return this.i8();
      case 2: return this.u16();
      case 3: return this.i16();
      case 4: return this.u32();
      case 5: return this.i32();
      case 6: return this.f32();
      case 7: return this.u8() !== 0;
      case 8: return this.str();
      case 9: {
        const inner = this.u32();
        const n = this.size();
        const out: GGUFValue[] = new Array(n);
        for (let i = 0; i < n; i++) out[i] = this.value(inner);
        return out;
      }
      case 10: return this.u64();
      case 11: return this.i64();
      case 12: return this.f64();
      default: throw new Error(`GGUF: unknown metadata value type ${type}`);
    }
  }
}

/** Parses metadata and the tensor table from a prefix of the file. Throws NeedMore if the prefix is too short. */
export function parseGGUF(bytes: Uint8Array): GGUFFile {
  const r = new Reader(bytes);
  if (r.u32() !== 0x46554747) throw new Error("not a GGUF file (bad magic)");
  const version = r.u32();
  if (version < 2) throw new Error(`GGUF version ${version} is not supported (need 2+)`);
  const tensorCount = r.size();
  const kvCount = r.size();
  const metadata = new Map<string, GGUFValue>();
  for (let i = 0; i < kvCount; i++) {
    const key = r.str();
    metadata.set(key, r.value(r.u32()));
  }
  const raw: { name: string; dims: number[]; type: number; offset: number }[] = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = r.str();
    const nd = r.u32();
    const dims: number[] = [];
    for (let d = 0; d < nd; d++) dims.push(r.size());
    raw.push({ name, dims, type: r.u32(), offset: r.size() });
  }
  const alignment = Number(metadata.get("general.alignment") ?? 32);
  const dataStart = Math.ceil(r.pos / alignment) * alignment;
  const tensors = raw.map(({ name, dims, type, offset }): TensorEntry => {
    const src = GGML_TYPES[type];
    if (!src) {
      throw new UnsupportedTensorError(
        `${name}: GGML type ${GGML_NAMES[type] ?? type} is not supported (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K).`,
      );
    }
    const shape = [...dims].reverse();
    const cols = shape[shape.length - 1]!;
    const rows = shape.slice(0, -1).reduce((a, b) => a * b, 1);
    return { name, shape, type: src, offset: dataStart + offset, byteLength: rows * rowBytes(src, cols) };
  });
  tensors.sort((a, b) => a.offset - b.offset);
  return { version, metadata, tensors };
}

/** Reads a growing prefix until the header fits (tokenizer vocabularies make headers several MB). */
export async function readGGUFHeader(src: ByteSource, initial = 4 * 2 ** 20): Promise<GGUFFile> {
  const total = await src.size();
  for (let n = Math.min(initial, total); ; n = Math.min(n * 2, total)) {
    try {
      return parseGGUF(await src.read(0, n));
    } catch (err) {
      if (!(err instanceof NeedMore) || n === total) throw err;
    }
  }
}
