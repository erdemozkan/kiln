import { describe, expect, it } from "vitest";
import type { Tensor } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import { ClipTextModel, clipPoolIndex, type ClipTextConfig } from "../src";

const CFG: ClipTextConfig = { vocab: 50, hidden: 32, intermediate: 64, layers: 2, heads: 2, maxPositions: 12, eps: 1e-5, act: "quick_gelu", eosId: 49, projectionDim: 8 };

function tinyModel(): ClipTextModel {
  const be = new CpuBackend({ simd: false });
  let seed = 3;
  const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32) * 2 - 1;
  const weights = new Map<string, Tensor>();
  const add = (name: string, shape: number[], fill?: number) => {
    const n = shape.reduce((a, b) => a * b, 1);
    const data = Float32Array.from({ length: n }, () => (fill ?? rand() * 0.5));
    const buf = be.alloc(data.byteLength);
    be.write(buf, 0, data);
    weights.set(name, { shape, dtype: "f32", buf });
  };
  const { hidden: H, intermediate: I } = CFG;
  add("tok.weight", [CFG.vocab, H]);
  add("tpos.weight", [CFG.maxPositions, H]);
  for (let l = 0; l < CFG.layers; l++) {
    for (const p of ["q", "k", "v", "o"]) {
      add(`t${l}.${p}.weight`, [H, H]);
      add(`t${l}.${p}.bias`, [H]);
    }
    add(`t${l}.up.weight`, [I, H]);
    add(`t${l}.up.bias`, [I]);
    add(`t${l}.down.weight`, [H, I]);
    add(`t${l}.down.bias`, [H]);
    for (const p of ["ln1", "ln2"]) {
      add(`t${l}.${p}.weight`, [H], 1);
      add(`t${l}.${p}.bias`, [H], 0);
    }
  }
  add("tlnF.weight", [H], 1);
  add("tlnF.bias", [H], 0);
  return new ClipTextModel(be, CFG, weights);
}

describe("ClipTextModel", () => {
  it("a padded batch gives each sequence the embedding it gets alone", async () => {
    const model = tinyModel();
    const seqs = [
      [48, 5, 9, 49],
      [48, 7, 3, 11, 20, 2, 49],
      [48, 49],
    ];
    const rows = seqs.map((s) => clipPoolIndex(s, CFG.eosId));
    const batch = await model.forwardBatch(seqs, rows);
    for (let i = 0; i < seqs.length; i++) {
      const alone = await model.forward(seqs[i]!, rows[i]!);
      expect(batch[i]!.length).toBe(CFG.hidden);
      for (let j = 0; j < CFG.hidden; j++) expect(batch[i]![j]).toBeCloseTo(alone[j]!, 5);
    }
    model.destroy();
  });
});
