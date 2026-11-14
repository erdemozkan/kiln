import { describe, expect, it } from "vitest";
import { rowBytes, type TensorEntry } from "@kiln/loader";
import { splitQwen35Merged } from "../src/qwen35-gguf";

describe("Qwen 3.5 GGUF with merged projections", () => {
  // 0.8B's sizes: q, k (16 heads of 128 each) and v (16 heads of 128) are 6144 rows, the gate 2048; hidden 1024.
  const [cols, qkv, gate, heads] = [1024, 6144, 2048, 16];

  it.each(["F32", "F16", "Q8_0", "Q4_0", "Q4_K"] as const)("lists the halves of %s tensors as byte ranges", (type) => {
    const row = rowBytes(type, cols);
    const other: TensorEntry = { name: "blk.0.ssm_out.weight", shape: [cols, gate], type, offset: 64, byteLength: cols * rowBytes(type, gate) };
    const merged: TensorEntry = { name: "blk.0.attn_qkv_gate.weight", shape: [qkv + gate, cols], type, offset: 1000, byteLength: (qkv + gate) * row };
    const ba: TensorEntry = { name: "blk.0.ssm_beta_alpha.weight", shape: [2 * heads, cols], type, offset: merged.offset + merged.byteLength, byteLength: 2 * heads * row };
    const out = splitQwen35Merged([other, merged, ba], gate);
    expect(out.map((t) => t.name)).toEqual(["blk.0.ssm_out.weight", "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"]);
    expect(out[0]).toBe(other);
    expect(out[1]).toMatchObject({ shape: [qkv, cols], offset: 1000, byteLength: qkv * row, type });
    expect(out[2]).toMatchObject({ shape: [gate, cols], offset: 1000 + qkv * row, byteLength: gate * row });
    expect(out[3]).toMatchObject({ shape: [heads, cols], offset: ba.offset, byteLength: heads * row });
    expect(out[4]).toMatchObject({ shape: [heads, cols], offset: ba.offset + heads * row, byteLength: heads * row });
    // The halves cover the merged tensor exactly.
    expect(out[1]!.byteLength + out[2]!.byteLength).toBe(merged.byteLength);
  });

  it("leaves files without the merge as they are", () => {
    const t: TensorEntry[] = [{ name: "blk.0.attn_qkv.weight", shape: [qkv, cols], type: "F16", offset: 0, byteLength: qkv * cols * 2 }];
    expect(splitQwen35Merged(t, gate)).toEqual(t);
  });

  it("refuses a merged tensor it cannot split by rows", () => {
    const bad = (over: Partial<TensorEntry>) => () => splitQwen35Merged([{ name: "blk.1.attn_qkv_gate.weight", shape: [qkv + gate, cols], type: "F16", offset: 0, byteLength: (qkv + gate) * cols * 2, ...over }], gate);
    expect(bad({ byteLength: 12 })).toThrow(/cannot be split/);
    expect(bad({ shape: [gate, cols], byteLength: gate * cols * 2 })).toThrow(/cannot be split/);
    expect(bad({ transposed: true })).toThrow(/cannot be split/);
    // A row that is not whole quantization blocks.
    expect(bad({ type: "Q4_K", shape: [qkv + gate, 1000] })).toThrow(/multiple of/);
    expect(() => splitQwen35Merged([{ name: "blk.1.ssm_beta_alpha.weight", shape: [33, cols], type: "F32", offset: 0, byteLength: 33 * cols * 4 }], gate)).toThrow(/cannot be split/);
  });
});
