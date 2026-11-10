import { describe, expect, it } from "vitest";
import { donutPixels, donutTaskToken, donutToken2Json, type DonutProcessing } from "../src/donut";

describe("Donut image processing", () => {
  // DonutImageProcessor's steps (transformers v4) run with PIL 8 and numpy at size 40 × 30: resize the
  // shortest edge (bilinear), thumbnail (bicubic, reducing_gap 2), pad around the center, normalize.
  const p: DonutProcessing = { size: [40, 30], resample: 2, alignLongAxis: false, resize: true, thumbnail: true, pad: true, rescale: 0.00392156862745098, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };
  const cases: [number, number, number, number[]][] = [
    [50, 20, -2518.3685945272446, [-1, 0.13725495338439941, -1, -1, -0.05882352590560913, -1, -1, -0.03529411554336548, -1]],
    // 200 × 10 and 10 × 100 shrink enough for PIL to reduce first.
    [200, 10, -3509.8431346416473, [-1, -1, -1, -1, -0.027450978755950928, -1, -1, -1, -1]],
    [10, 100, -3120.2352799773216, [-1, -1, -1, -1, -1, -1, -1, -1, -1]],
    [25, 25, -901.3097369074821, [-1, -0.5607843399047852, -0.5686274766921997, -1, 0.3960784673690796, 0.41960787773132324, -0.6470588445663452, -0.6549019813537598, -1]],
  ];
  it.each(cases)("a %i × %i image", (w, h, sum, probes) => {
    const out = donutPixels({ width: w, height: h, data: Uint8Array.from({ length: w * h * 3 }, (_, i) => (i * 37) % 256) }, p);
    expect(out.length).toBe(3 * 40 * 30);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(sum, 6);
    // Nine evenly spaced values (numpy's linspace over the flat array, truncated).
    const at = Array.from({ length: 9 }, (_, i) => Math.trunc((i * (out.length - 1)) / 8));
    expect(at.map((i) => out[i])).toEqual(probes);
  });
});

describe("Donut's task token and token2json", () => {
  it("finds the task token: the opening tag without a closing one, pre-training tasks left out", () => {
    expect(donutTaskToken(["<s_cord-v2>", "<s_iitcdip>", "<s_synthdog>", "<s_menu>", "</s_menu>", "<sep/>"])).toBe("<s_cord-v2>");
    expect(donutTaskToken(["<s_class>", "</s_class>", "<invoice/>", "<s_rvlcdip>", "<s_iitcdip>"])).toBe("<s_rvlcdip>");
    expect(donutTaskToken(["<s_docvqa>", "<s_question>", "</s_question>", "<s_answer>", "</s_answer>"])).toBe("<s_docvqa>");
    expect(donutTaskToken(["<s_menu>", "</s_menu>"])).toBeUndefined();
  });

  it("parses as transformers' DonutProcessor.token2json does", () => {
    // Expected values printed by token2json (transformers v4) on the same strings.
    const vocab = new Set(["<sep/>", "<invoice/>", "<s_class>", "</s_class>"]);
    expect(donutToken2Json("<s_class><invoice/></s_class>", vocab)).toEqual({ class: "invoice" });
    expect(
      donutToken2Json(
        "<s_menu><s_nm> CINNAMON SUGAR</s_nm><s_unitprice> 17,000</s_unitprice><s_cnt> 1 x</s_cnt><s_price> 17,000</s_price></s_menu><s_sub_total><s_subtotal_price> 17,000</s_subtotal_price></s_sub_total><s_total><s_total_price> 17,000</s_total_price><s_cashprice> 20,000</s_cashprice><s_changeprice> 3,000</s_changeprice></s_total>",
        vocab,
      ),
    ).toEqual({ menu: { nm: "CINNAMON SUGAR", unitprice: "17,000", cnt: "1 x", price: "17,000" }, sub_total: { subtotal_price: "17,000" }, total: { total_price: "17,000", cashprice: "20,000", changeprice: "3,000" } });
    expect(donutToken2Json("<s_menu><s_nm> A</s_nm><s_price> 1</s_price><sep/><s_nm> B</s_nm><s_price> 2</s_price></s_menu>", vocab)).toEqual({ menu: [{ nm: "A", price: "1" }, { nm: "B", price: "2" }] });
    expect(donutToken2Json("<s_nm> x<sep/> y</s_nm>", vocab)).toEqual({ nm: ["x", "y"] });
    expect(donutToken2Json("no tags here", vocab)).toEqual({ text_sequence: "no tags here" });
    expect(donutToken2Json("<s_open> dangling", vocab)).toEqual({ text_sequence: " dangling" });
  });
});
