import { describe, expect, it } from "vitest";
import { Ctx, STEP_BYTES, packStep, type Tensor } from "@kiln/core";
import { CpuBackend, ops } from "@kiln/kernels";
import { detrName, detrSinePositions, foldConvBn } from "../src";

describe("DETR", () => {
  it("sine position embeddings equal DetrSinePositionEmbedding's (torch, 2 × 3 map, d_model 8)", () => {
    // transformers' DetrSinePositionEmbedding(4, normalize=True) on an all-ones mask, as [h·w, d_model].
    const torch = [
      1.3e-6, -1.0, 0.0314107, 0.9995065, 0.8660257, -0.4999994, 0.0209424, 0.9997807, 1.3e-6, -1.0, 0.0314107, 0.9995065, -0.8660247, -0.5000011, 0.0418756, 0.9991228, 1.3e-6, -1.0,
      0.0314107, 0.9995065, -1.7e-6, 1.0, 0.0627905, 0.9980267, -2.7e-6, 1.0, 0.0627905, 0.9980267, 0.8660257, -0.4999994, 0.0209424, 0.9997807, -2.7e-6, 1.0, 0.0627905, 0.9980267,
      -0.8660247, -0.5000011, 0.0418756, 0.9991228, -2.7e-6, 1.0, 0.0627905, 0.9980267, -1.7e-6, 1.0, 0.0627905, 0.9980267,
    ];
    const pos = detrSinePositions(2, 3, 8);
    expect(pos.length).toBe(torch.length);
    torch.forEach((v, i) => expect(Math.abs(pos[i]! - v)).toBeLessThan(2e-6));
  });

  it("a convolution with the batch norm folded in equals the convolution followed by the batch norm", async () => {
    const be = new CpuBackend({ simd: false });
    const step = be.alloc(STEP_BYTES);
    be.write(step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
    let seed = 5;
    const rand = (n: number, lo = -1, hi = 1) => Float32Array.from({ length: n }, () => lo + ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32) * (hi - lo));
    const [C, H, W, O, k] = [3, 6, 5, 7, 3];
    const x = rand(C * H * W);
    const w = rand(O * C * k * k);
    const bn = { weight: rand(O), bias: rand(O), mean: rand(O), var: rand(O, 0.1, 2) };
    const f = foldConvBn(w, [O, C, k, k], bn, 4);
    const upload = (data: Float32Array, shape: number[]): Tensor => {
      const buf = be.alloc(data.byteLength);
      be.write(buf, 0, data);
      return { shape, dtype: "f32", buf };
    };
    const xl = new Float32Array(H * W * 4);
    for (let c = 0; c < C; c++) for (let t = 0; t < H * W; t++) xl[t * 4 + c] = x[c * H * W + t]!;
    const ctx = new Ctx(be, step, "t");
    const out = ops.conv2d(ctx, upload(xl, [H * W, 4]), upload(f.w, [O, f.K]), upload(f.b, [O]), { H, W, k, stride: 1, pad: 1 }).y;
    be.submit(ctx.dispatches);
    const y = new Float32Array(await be.readback(out.buf, 0, H * W * O * 4));
    for (let o = 0; o < O; o++)
      for (let py = 0; py < H; py++)
        for (let px = 0; px < W; px++) {
          let s = 0;
          for (let c = 0; c < C; c++)
            for (let ky = 0; ky < k; ky++)
              for (let kx = 0; kx < k; kx++) {
                const iy = py + ky - 1;
                const ix = px + kx - 1;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) s += x[(c * H + iy) * W + ix]! * w[((o * C + c) * k + ky) * k + kx]!;
              }
          const ref = ((s - bn.mean[o]!) / Math.sqrt(bn.var[o]! + 1e-5)) * bn.weight[o]! + bn.bias[o]!;
          expect(y[(py * W + px) * O + o]).toBeCloseTo(ref, 4);
        }
  });

  it("names the transformer and head tensors, and leaves the backbone to be loaded otherwise", () => {
    expect(detrName("model.encoder.layers.3.self_attn.q_proj.weight")).toBe("e3.q.weight");
    expect(detrName("model.decoder.layers.5.encoder_attn.out_proj.bias")).toBe("d5.co.bias");
    expect(detrName("model.decoder.layers.0.encoder_attn_layer_norm.weight")).toBe("d0.lnC.weight");
    expect(detrName("model.decoder.layernorm.bias")).toBe("dlnF.bias");
    expect(detrName("class_labels_classifier.weight")).toBe("det.cls.0.weight");
    expect(detrName("bbox_predictor.layers.2.bias")).toBe("det.box.2.bias");
    expect(detrName("model.backbone.conv_encoder.model.layer1.0.conv1.weight")).toBeUndefined();
    expect(detrName("model.input_projection.weight")).toBeUndefined();
  });

  it("reads a panoptic checkpoint's names: the detr. prefix and the box attention projections", () => {
    expect(detrName("detr.model.encoder.layers.0.fc1.weight")).toBe("e0.up.weight");
    expect(detrName("detr.class_labels_classifier.bias")).toBe("det.cls.0.bias");
    expect(detrName("bbox_attention.q_linear.weight")).toBe("mask.q.weight");
    expect(detrName("bbox_attention.k_linear.bias")).toBe("mask.k.bias");
    expect(detrName("mask_head.lay1.weight")).toBeUndefined();
  });
});
