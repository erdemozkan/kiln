import { describe, expect, it } from "vitest";
import { isnetConvs } from "../src";

describe("IS-Net (RMBG-1.4)", () => {
  it("lists the checkpoint's convolutions, with the concatenation splits of briarmbg.py", () => {
    const convs = isnetConvs();
    // RMBG-1.4's 798 tensors: 112 conv + batch-norm pairs (7 tensors each), conv_in and side1 … side6.
    expect([...convs.values()].filter((c) => c.bn).length).toBe(112);
    expect(convs.get("conv_in")).toEqual({ splits: [3], bn: false });
    expect(convs.get("side1")).toEqual({ splits: [64], bn: false });
    expect(convs.get("stage1.rebnconv6d")!.splits).toEqual([32, 32]);
    expect(convs.get("stage1.rebnconv1d")!.splits).toEqual([32, 32]);
    expect(convs.get("stage5d.rebnconvin")!.splits).toEqual([512, 512]);
    expect(convs.get("stage1d.rebnconvin")!.splits).toEqual([64, 64]);
    expect(convs.get("stage4d.rebnconvin")!.splits).toEqual([512, 512]);
    expect(convs.get("stage6.rebnconv3d")!.splits).toEqual([256, 256]);
  });
});
