import { describe, expect, it } from "vitest";
import type { Tensor } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import { EncoderModel, type EncoderConfig, type MlmHead } from "../src";

const H = 64;
const V = 40;
const CFG: EncoderConfig = {
  family: "bert", modelType: "bert", architectures: ["BertForMaskedLM"], hidden: H, layers: 2, heads: 2, intermediate: 128,
  vocab: V, maxPositions: 32, typeVocab: 2, posOffset: 0, eps: 1e-12, act: "gelu", id2label: undefined, multiLabel: false,
};

let seed = 7;
const rand = (n: number, s = 0.3) => Float32Array.from({ length: n }, () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31 - 0.5) * 2 * s);

function weights(be: CpuBackend): Map<string, Tensor> {
  const w = new Map<string, { shape: number[]; data: Float32Array }>();
  const put = (name: string, shape: number[], data = rand(shape.reduce((a, b) => a * b, 1))) => w.set(name, { shape, data });
  put("emb.word", [V, H]);
  put("emb.pos", [CFG.maxPositions, H]);
  put("emb.type", [2, H]);
  put("emb.ln.weight", [H], Float32Array.from({ length: H }, () => 1));
  put("emb.ln.bias", [H]);
  for (let l = 0; l < CFG.layers; l++) {
    for (const n of ["q", "k", "v", "o"]) (put(`l${l}.${n}.weight`, [H, H]), put(`l${l}.${n}.bias`, [H]));
    put(`l${l}.up.weight`, [CFG.intermediate, H]);
    put(`l${l}.up.bias`, [CFG.intermediate]);
    put(`l${l}.down.weight`, [H, CFG.intermediate]);
    put(`l${l}.down.bias`, [H]);
    for (const n of ["ln1", "ln2"]) (put(`l${l}.${n}.weight`, [H], Float32Array.from({ length: H }, () => 1)), put(`l${l}.${n}.bias`, [H]));
  }
  const out = new Map<string, Tensor>();
  for (const [name, { shape, data }] of w) {
    const buf = be.alloc(data.byteLength, "storage", name);
    be.write(buf, 0, data);
    out.set(name, { shape, dtype: "f32", buf });
  }
  return out;
}

describe("EncoderModel", () => {
  it("maskedLogits: the head on chosen rows on the backend equals the head on the read-back hidden states", async () => {
    const be = new CpuBackend({ threads: 1 });
    const m = new EncoderModel(be, CFG, weights(be));
    const head: MlmHead = { dense: rand(H * H), denseBias: rand(H), lnW: rand(H, 1), lnB: rand(H), eps: 1e-12, exact: true, decoder: undefined, bias: rand(V) };
    const items = [
      { ids: [1, 5, 9, 3, 2], typeIds: [0, 0, 0, 1, 1], rows: [2] },
      { ids: [1, 7, 2], rows: [0, 2] },
    ];
    const got = await m.maskedLogits(items, head);
    const hs = await m.forwardMany(items);
    for (let i = 0; i < items.length; i++) {
      const rows = new Float32Array(items[i]!.rows.length * H);
      items[i]!.rows.forEach((r, j) => rows.set(hs[i]!.subarray(r * H, (r + 1) * H), j * H));
      const want = await m.maskLogits(rows, head);
      expect(got[i]!.length).toBe(items[i]!.rows.length * V);
      expect(Math.max(...Array.from(got[i]!, (v, k) => Math.abs(v - want[k]!)))).toBeLessThan(1e-5);
    }
    await expect(m.maskedLogits([{ ids: [1, 2], rows: [2] }], head)).rejects.toThrow(/outside/);
    m.destroy();
  });
});
