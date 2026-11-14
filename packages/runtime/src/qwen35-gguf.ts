import { rowBytes, type TensorEntry } from "@kiln/loader";
import { UnsupportedModelError } from "@kiln/models";

/**
 * Qwen 3.5 GGUFs may store two pairs of linear attention's projections, which read the same input,
 * as one tensor each: `attn_qkv_gate` = [attn_qkv | attn_gate] and `ssm_beta_alpha` =
 * [ssm_beta | ssm_alpha], rows of the first followed by rows of the second. This lists each half
 * under the name files without the merge use. A row is whole quantization blocks, so a half is a
 * byte range of the merged tensor and nothing is read or copied. `gateRows` is the gate
 * projection's output width (value heads × their size); beta and alpha are each half the rows.
 */
export function splitQwen35Merged(tensors: readonly TensorEntry[], gateRows: number): TensorEntry[] {
  const out: TensorEntry[] = [];
  for (const t of tensors) {
    const m = /^(blk\.\d+)\.(attn_qkv_gate|ssm_beta_alpha)\.weight$/.exec(t.name);
    if (!m) {
      out.push(t);
      continue;
    }
    const [rows, cols] = [t.shape[0]!, t.shape[1]!];
    const second = m[2] === "attn_qkv_gate" ? gateRows : rows / 2;
    const first = rows - second;
    const row = rowBytes(t.type, cols);
    if (t.shape.length !== 2 || t.transposed || !Number.isInteger(second) || first <= 0 || second <= 0 || !Number.isInteger(row) || row * rows !== t.byteLength) {
      throw new UnsupportedModelError(`${t.name}: a merged projection of [${t.shape}] ${t.type} cannot be split by rows`);
    }
    const names = m[2] === "attn_qkv_gate" ? ["attn_qkv", "attn_gate"] : ["ssm_beta", "ssm_alpha"];
    out.push({ ...t, name: `${m[1]}.${names[0]}.weight`, shape: [first, cols], byteLength: first * row });
    out.push({ ...t, name: `${m[1]}.${names[1]}.weight`, shape: [second, cols], offset: t.offset + first * row, byteLength: second * row });
  }
  return out;
}
