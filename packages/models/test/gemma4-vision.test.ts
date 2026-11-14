import { describe, expect, it } from "vitest";
import { type Tensor } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import {
  GEMMA4_CLIPPED,
  Gemma4VisionTower,
  gemma4ImageGrid,
  gemma4Patches,
  gemma4PositionEmbeddings,
  gemma4RotaryAngles,
  gemma4VisionConfig,
  gemma4VisionConfigFromGGUF,
  gemma4VisionName,
  type Gemma4Clip,
  type Gemma4VisionConfig,
} from "../src/gemma4-vision";

const config = {
  text_config: { hidden_size: 1536 },
  vision_soft_tokens_per_image: 280,
  vision_config: { hidden_size: 768, intermediate_size: 3072, num_attention_heads: 12, num_hidden_layers: 16, patch_size: 16, pooling_kernel_size: 3, position_embedding_size: 10240, rms_norm_eps: 1e-6, rope_parameters: { rope_theta: 100, rope_type: "default" }, standardize: false },
};

function rand(n: number, seed: number, scale: number): Float32Array {
  let s = seed >>> 0;
  return Float32Array.from({ length: n }, () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s / 2 ** 32 - 0.5) * 2 * scale;
  });
}

type Weights = Map<string, { shape: number[]; data: Float32Array }>;

/**
 * The tower as the transformers model computes it, in float64, from the image's pixels: patches
 * pixel by pixel (row, column, channel) through `hfPatch`, positions as (x, y), each projection
 * between its clips, pooling by position, the √hidden scale, then the projector's norm and matrix.
 */
function reference(c: Gemma4VisionConfig, w: Weights, clips: Map<string, Gemma4Clip>, hfPatch: Float32Array, posTable: Float32Array, img: { width: number; height: number; data: Uint8Array }): Float64Array {
  const [H, P, D] = [c.hidden, c.patchSize, c.hidden / c.heads];
  const [gh, gw] = [img.height / P, img.width / P];
  const n = gh * gw;
  const W = (name: string) => w.get(name)!;
  const rms = (x: Float64Array, dim: number, eps: number, weight?: Float32Array): Float64Array => {
    const y = new Float64Array(x.length);
    for (let g = 0; g < x.length; g += dim) {
      let ss = 0;
      for (let i = 0; i < dim; i++) ss += x[g + i]! ** 2;
      const r = (ss / dim + eps) ** -0.5;
      for (let i = 0; i < dim; i++) y[g + i] = x[g + i]! * r * (weight ? weight[i]! : 1);
    }
    return y;
  };
  const clipped = (at: string, x: Float64Array): Float64Array => {
    const [a, b, lo, hi] = clips.get(at)!;
    const { shape, data } = W(`${at}.weight`);
    const [N, K] = shape as [number, number];
    const y = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < K; j++) s += data[i * K + j]! * Math.min(Math.max(x[j]!, a), b);
      y[i] = Math.min(Math.max(s, lo), hi);
    }
    return y;
  };
  const gelu = (v: number) => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3)));
  // Rotary: each head's first half turns with x, its second with y; pairs (i, i + D / 4) in a half.
  const rope = (v: Float64Array, px: number, py: number): Float64Array => {
    const y = v.slice();
    const q = D / 4;
    for (let h = 0; h < c.heads; h++)
      for (const [axis, pos] of [[0, px], [1, py]] as const)
        for (let i = 0; i < q; i++) {
          const a = pos * (1 / c.ropeTheta ** ((2 * i) / (D / 2)));
          const i0 = h * D + axis * 2 * q + i;
          const [x0, x1] = [v[i0]!, v[i0 + q]!];
          y[i0] = x0 * Math.cos(a) - x1 * Math.sin(a);
          y[i0 + q] = x1 * Math.cos(a) + x0 * Math.sin(a);
        }
    return y;
  };
  let xs: Float64Array[] = [];
  for (let py = 0; py < gh; py++)
    for (let px = 0; px < gw; px++) {
      const x = new Float64Array(H);
      for (let r = 0; r < H; r++) {
        let s = 0;
        for (let y = 0; y < P; y++)
          for (let xx = 0; xx < P; xx++)
            for (let k = 0; k < 3; k++) s += hfPatch[r * 3 * P * P + (y * P + xx) * 3 + k]! * 2 * (img.data[((py * P + y) * img.width + px * P + xx) * 3 + k]! / 255 - 0.5);
        x[r] = s + posTable[px * H + r]! + posTable[(c.posSize + py) * H + r]!;
      }
      xs.push(x);
    }
  for (let l = 0; l < c.layers; l++) {
    const p = `v.blk.${l}.`;
    const hs = xs.map((x) => rms(x, H, c.eps, W(p + "ln1.weight").data));
    const at = (t: number) => [t % gw, Math.floor(t / gw)] as const;
    const qs = hs.map((h, t) => rope(rms(clipped(p + "attn_q", h), D, c.eps, W(p + "attn_q_norm.weight").data), ...at(t)));
    const ks = hs.map((h, t) => rope(rms(clipped(p + "attn_k", h), D, c.eps, W(p + "attn_k_norm.weight").data), ...at(t)));
    const vs = hs.map((h) => rms(clipped(p + "attn_v", h), D, c.eps));
    xs = xs.map((x, t) => {
      const a = new Float64Array(H);
      for (let h = 0; h < c.heads; h++) {
        const sc = ks.map((k) => {
          let s = 0;
          for (let i = 0; i < D; i++) s += qs[t]![h * D + i]! * k[h * D + i]!;
          return s;
        });
        const m = Math.max(...sc);
        const e = sc.map((s) => Math.exp(s - m));
        const z = e.reduce((p0, v) => p0 + v, 0);
        for (let j = 0; j < n; j++) for (let i = 0; i < D; i++) a[h * D + i] += (e[j]! / z) * vs[j]![h * D + i]!;
      }
      const o = rms(clipped(p + "attn_out", a), H, c.eps, W(p + "attn_post_norm.weight").data);
      const mid = x.map((v, i) => v + o[i]!);
      const h2 = rms(mid, H, c.eps, W(p + "ln2.weight").data);
      const [g, u] = [clipped(p + "ffn_gate", h2), clipped(p + "ffn_up", h2)];
      const d = rms(clipped(p + "ffn_down", g.map((v, i) => gelu(v) * u[i]!)), H, c.eps, W(p + "ffn_post_norm.weight").data);
      return mid.map((v, i) => v + d[i]!);
    });
  }
  // Pooling by position: token (x // k) + (columns // k) · (y // k), the mean of its k² patches.
  const k = c.poolSize;
  const tokens = (gh / k) * (gw / k);
  const proj = W("mm.input_projection.weight");
  const out = new Float64Array(tokens * c.projectionDim);
  for (let t = 0; t < tokens; t++) {
    const pooled = new Float64Array(H);
    xs.forEach((x, r) => {
      if (Math.floor((r % gw) / k) + (gw / k) * Math.floor(Math.floor(r / gw) / k) === t) for (let i = 0; i < H; i++) pooled[i] += x[i]! / (k * k);
    });
    const normed = rms(pooled.map((v) => v * Math.sqrt(H)), H, c.eps);
    for (let i = 0; i < c.projectionDim; i++) {
      let s = 0;
      for (let j = 0; j < H; j++) s += proj.data[i * H + j]! * normed[j]!;
      out[t * c.projectionDim + i] = s;
    }
  }
  return out;
}

describe("Gemma 4 vision", () => {
  const c = gemma4VisionConfig(config, { max_soft_tokens: 280 });

  it("reads the config", () => {
    expect(c).toMatchObject({ hidden: 768, heads: 12, layers: 16, intermediate: 3072, patchSize: 16, poolSize: 3, posSize: 10240, projectionDim: 1536, ropeTheta: 100, maxTokens: 280 });
    expect(() => gemma4VisionConfig({})).toThrow(/vision_config/);
    expect(() => gemma4VisionConfig({ vision_config: { ...config.vision_config, standardize: true } })).toThrow(/standardized/);
    const meta = new Map<string, unknown>([["clip.vision.embedding_length", 768], ["clip.vision.attention.head_count", 12], ["clip.vision.block_count", 16], ["clip.vision.feed_forward_length", 3072], ["clip.vision.patch_size", 16], ["clip.vision.projection_dim", 1536]]);
    expect(gemma4VisionConfigFromGGUF(meta, 10240)).toEqual(c);
  });

  it("picks the token grid as the processor does", () => {
    // 280 tokens of 48 pixels: the largest size of the image's shape within 2520 patches.
    expect(gemma4ImageGrid(1024, 1024, c)).toEqual({ rows: 16, cols: 16 });
    expect(gemma4ImageGrid(640, 480, c)).toEqual({ rows: 14, cols: 19 });
    expect(gemma4ImageGrid(480, 640, c)).toEqual({ rows: 19, cols: 14 });
    // Small images are scaled up to the budget too.
    expect(gemma4ImageGrid(100, 50, c)).toEqual({ rows: 11, cols: 23 });
    expect(gemma4ImageGrid(1024, 1024, c, 70)).toEqual({ rows: 8, cols: 8 });
    // A side that would round to nothing is one token.
    expect(gemma4ImageGrid(3000, 5, c)).toEqual({ rows: 1, cols: 280 });
    expect(gemma4ImageGrid(5, 9000, c)).toEqual({ rows: 280, cols: 1 });
    expect(gemma4ImageGrid(5, 900, c)).toEqual({ rows: 224, cols: 1 });
    for (const [w, h] of [[1920, 1080], [333, 777], [4000, 300]] as const) {
      const g = gemma4ImageGrid(w, h, c);
      expect(g.rows * g.cols).toBeLessThanOrEqual(280);
    }
  });

  it("lays patches out in raster order, values by channel then row then column, scaled to −1…1", () => {
    const [W, H] = [48, 32];
    const data = new Uint8Array(W * H * 3);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        data[(y * W + x) * 3] = Math.floor(x / 16) * 10;
        data[(y * W + x) * 3 + 1] = Math.floor(y / 16) * 10;
        data[(y * W + x) * 3 + 2] = y === 16 && x === 33 ? 255 : 0;
      }
    const p = gemma4Patches(c, { width: W, height: H, data });
    expect(p.length).toBe(6 * 768);
    const byte = (v: number) => Math.round((v / 2 + 0.5) * 255);
    expect([0, 1, 2, 3, 4, 5].map((r) => byte(p[r * 768]!))).toEqual([0, 10, 20, 0, 10, 20]);
    expect([0, 1, 2, 3, 4, 5].map((r) => byte(p[r * 768 + 256]!))).toEqual([0, 0, 0, 10, 10, 10]);
    // The blue pixel at (row 16, column 33) is patch (1, 2), at blue · row 0 · column 1.
    expect(p[5 * 768 + 2 * 256 + 1]).toBeCloseTo(1, 5);
    expect(p[0]).toBe(-1);
    expect(() => gemma4Patches(c, { width: 40, height: 32, data: new Uint8Array(40 * 32 * 3) })).toThrow(/whole patches/);
  });

  it("adds a column's and a row's position embedding, and turns each half of a head with one axis", () => {
    const small: Gemma4VisionConfig = { ...c, hidden: 2, posSize: 4 };
    // First table: row x is [x, 0]; second: row y is [0, 10 · y].
    const table = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      table[i * 2] = i;
      table[(4 + i) * 2 + 1] = 10 * i;
    }
    expect([...gemma4PositionEmbeddings(table, small, 2, 3)]).toEqual([0, 0, 1, 0, 2, 0, 0, 10, 1, 10, 2, 10]);
    expect(() => gemma4PositionEmbeddings(table, small, 5, 1)).toThrow(/positions/);
    const a = gemma4RotaryAngles(c, 2, 4);
    // Heads of 64: 32 pairs, 16 frequencies per axis. Row 6 of the input is patch (row 1, column 2).
    expect(a.length).toBe(8 * 32);
    expect(a[6 * 32]).toBe(2);
    expect(a[6 * 32 + 1]).toBeCloseTo(2 / 100 ** (1 / 16), 6);
    expect(a[6 * 32 + 16]).toBe(1);
  });

  it("matches the reference forward pass on a small tower", async () => {
    const s: Gemma4VisionConfig = { hidden: 64, heads: 2, layers: 2, intermediate: 96, patchSize: 8, poolSize: 3, posSize: 12, projectionDim: 32, eps: 1e-6, ropeTheta: 100, maxTokens: 70 };
    const K = 3 * s.patchSize ** 2;
    const w: Weights = new Map();
    let seed = 11;
    const mat = (name: string, n: number, k: number, scale = 0.25) => w.set(name, { shape: [n, k], data: rand(n * k, seed++, scale) });
    const vec = (name: string, n: number) => w.set(name, { shape: [n], data: rand(n, seed++, 0.4).map((v) => 1 + v) });
    // Limits tight enough to bite: some inputs and outputs are clipped, most are not.
    const clips = new Map<string, Gemma4Clip>();
    for (let l = 0; l < s.layers; l++) {
      const p = `v.blk.${l}.`;
      for (const n of ["ln1", "ln2", "attn_post_norm", "ffn_post_norm"]) vec(`${p}${n}.weight`, s.hidden);
      for (const n of ["attn_q_norm", "attn_k_norm"]) vec(`${p}${n}.weight`, s.hidden / s.heads);
      for (const n of ["attn_q", "attn_k", "attn_v", "attn_out"]) mat(`${p}${n}.weight`, s.hidden, s.hidden);
      mat(`${p}ffn_gate.weight`, s.intermediate, s.hidden);
      mat(`${p}ffn_up.weight`, s.intermediate, s.hidden);
      mat(`${p}ffn_down.weight`, s.hidden, s.intermediate);
      for (const n of GEMMA4_CLIPPED) {
        // q, k and v share their input's limits, as do gate and up; the first layer's down projection has none.
        const inLim = n.startsWith("attn_") && n !== "attn_out" ? 1.1 : n === "attn_out" ? 0.9 : n === "ffn_down" ? 0.6 : 1.3;
        clips.set(p + n, l === 0 && n === "ffn_down" ? [-Infinity, Infinity, -Infinity, Infinity] : [-inLim, inLim * 1.1, -0.8 - 0.1 * l, 0.7]);
      }
    }
    mat("mm.input_projection.weight", s.projectionDim, s.hidden);
    const hfPatch = rand(s.hidden * K, 5, 0.1);
    // What the loader does with a checkpoint's patch projection: columns to channel-major order.
    const pp = s.patchSize ** 2;
    const patchW = new Float32Array(hfPatch.length);
    for (let r = 0; r < s.hidden; r++) for (let k = 0; k < 3; k++) for (let i = 0; i < pp; i++) patchW[r * K + k * pp + i] = hfPatch[r * K + i * 3 + k]!;
    const posTable = rand(2 * s.posSize * s.hidden, 3, 0.5);

    const be = new CpuBackend({ threads: 1 });
    const gpu = new Map<string, Tensor>();
    for (const [name, { shape, data }] of [...w, ["v.patch_embd.weight", { shape: [s.hidden, K], data: patchW }] as const]) {
      const buf = be.alloc(data.byteLength, "storage", name);
      be.write(buf, 0, data);
      gpu.set(name, { shape: [...shape], dtype: "f32", buf });
    }
    const tower = new Gemma4VisionTower(be, s, gpu, clips, posTable, 70);
    expect(tower.grid(300, 200)).toEqual({ rows: 6, cols: 10, width: 240, height: 144 });
    const dst = be.alloc(70 * s.projectionDim * 4, "storage", "features");
    // Two shapes, wide then tall: a plan per grid, and nothing left over from the first in the second.
    for (const [W, H] of [[72, 48], [24, 72]] as const) {
      const img = { width: W, height: H, data: Uint8Array.from(rand(W * H * 3, W, 127.5), (v) => Math.round(v + 127.5)) };
      const want = reference(s, w, clips, hfPatch, posTable, img);
      const g = await tower.encode(img, dst, 0);
      expect(g).toEqual({ rows: H / 24, cols: W / 24 });
      const got = new Float32Array(await be.readback(dst, 0, want.length * 4));
      const diff = got.reduce((m, v, i) => Math.max(m, Math.abs(v - want[i]!)), 0);
      expect(diff).toBeLessThan(2e-3);
      expect(Math.max(...want.map(Math.abs))).toBeGreaterThan(0.5);
    }
    await expect(tower.encode({ width: 40, height: 48, data: new Uint8Array(40 * 48 * 3) }, dst, 0)).rejects.toThrow(/whole tokens/);
    tower.destroy();
  });

  it("names a checkpoint's tensors as the mmproj file does", () => {
    expect(gemma4VisionName("model.vision_tower.encoder.layers.3.self_attn.q_proj.linear.weight")).toBe("v.blk.3.attn_q.weight");
    expect(gemma4VisionName("model.vision_tower.encoder.layers.3.self_attn.o_proj.output_max")).toBe("v.blk.3.attn_out.output_max");
    expect(gemma4VisionName("model.vision_tower.encoder.layers.15.mlp.gate_proj.input_min")).toBe("v.blk.15.ffn_gate.input_min");
    expect(gemma4VisionName("model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight")).toBe("v.blk.0.ffn_down.weight");
    expect(gemma4VisionName("model.vision_tower.encoder.layers.0.pre_feedforward_layernorm.weight")).toBe("v.blk.0.ln2.weight");
    expect(gemma4VisionName("model.vision_tower.encoder.layers.0.post_attention_layernorm.weight")).toBe("v.blk.0.attn_post_norm.weight");
    expect(gemma4VisionName("model.vision_tower.encoder.layers.0.self_attn.k_norm.weight")).toBe("v.blk.0.attn_k_norm.weight");
    expect(gemma4VisionName("model.vision_tower.patch_embedder.input_proj.weight")).toBe("v.patch_embd.weight");
    expect(gemma4VisionName("model.vision_tower.patch_embedder.position_embedding_table")).toBe("v.position_embd.weight");
    expect(gemma4VisionName("model.embed_vision.embedding_projection.weight")).toBe("mm.input_projection.weight");
    expect(gemma4VisionName("model.language_model.layers.0.mlp.up_proj.weight")).toBeUndefined();
    expect(gemma4VisionName("model.audio_tower.layers.0.norm.weight")).toBeUndefined();
  });
});
