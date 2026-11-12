import { UnsupportedTensorError, type SrcType, type TensorEntry, checkEntries } from "./entry";
import type { ByteSource } from "./source";

const DTYPES: Record<string, SrcType> = { F32: "F32", F16: "F16", BF16: "BF16" };

export interface SafetensorsHeader {
  tensors: TensorEntry[];
  metadata: Record<string, string>;
}

interface RawEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

/** Reads only the JSON header: 8 bytes of length, then the header itself. */
export async function readSafetensorsHeader(src: ByteSource): Promise<SafetensorsHeader> {
  const lenBytes = await src.read(0, 8);
  const len = Number(new DataView(lenBytes.buffer, lenBytes.byteOffset, 8).getBigUint64(0, true));
  if (len > 100 * 2 ** 20) throw new Error(`${src.name}: safetensors header of ${len} bytes is implausible`);
  const json = JSON.parse(new TextDecoder().decode(await src.read(8, len))) as Record<string, unknown>;
  const base = 8 + len;
  const tensors: TensorEntry[] = [];
  let metadata: Record<string, string> = {};
  for (const [name, value] of Object.entries(json)) {
    if (name === "__metadata__") {
      metadata = value as Record<string, string>;
      continue;
    }
    const e = value as RawEntry;
    if (!e || typeof e !== "object") throw new Error(`${src.name}: tensor ${name} has no entry`);
    const type = Object.prototype.hasOwnProperty.call(DTYPES, e.dtype) ? DTYPES[e.dtype] : undefined;
    // Integer and boolean tensors are buffers (BERT's position_ids, attention masks), never weights.
    if (!type && /^(I|U)(8|16|32|64)$|^BOOL$/.test(e.dtype)) continue;
    if (!type) throw new UnsupportedTensorError(`${name}: safetensors dtype ${e.dtype} is not supported (F32, F16, BF16)`);
    const [start, end] = Array.isArray(e.data_offsets) ? e.data_offsets : [NaN, NaN];
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start) throw new Error(`${src.name}: tensor ${name} has invalid data_offsets`);
    tensors.push({ name, shape: e.shape, type, offset: base + start, byteLength: end - start });
  }
  checkEntries(src.name, tensors, await src.size());
  tensors.sort((a, b) => a.offset - b.offset);
  return { tensors, metadata };
}
