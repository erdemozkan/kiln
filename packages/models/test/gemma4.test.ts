import { describe, expect, it } from "vitest";
import { physical, type Tensor } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import { LlamaModel, configFromGGUF, configFromHF, ggufToHF, gemma4PerLayerTable, headDimOf, ropeInvFreq, type LlamaConfig } from "../src";

/** A small Gemma 4: six layers, sliding and full in turn, the last two reading the caches of layers 2 and 3. */
const HF = {
  model_type: "gemma4",
  eos_token_id: [1, 9],
  text_config: {
    model_type: "gemma4_text",
    hidden_size: 64,
    intermediate_size: 96,
    num_hidden_layers: 6,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_dim: 32,
    global_head_dim: 64,
    vocab_size: 50,
    vocab_size_per_layer_input: 50,
    hidden_size_per_layer_input: 32,
    num_kv_shared_layers: 2,
    layer_types: ["sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    sliding_window: 4,
    final_logit_softcapping: 30,
    rms_norm_eps: 1e-6,
    use_double_wide_mlp: true,
    dtype: "float32",
    rope_parameters: { full_attention: { partial_rotary_factor: 0.25, rope_theta: 1000000, rope_type: "proportional" }, sliding_attention: { rope_theta: 10000, rope_type: "default" } },
  },
};

function rand(n: number, seed: number, scale: number): Float32Array {
  let s = seed >>> 0;
  return Float32Array.from({ length: n }, () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s / 2 ** 32 - 0.5) * 2 * scale;
  });
}

/** Random weights by the model's names, as arrays. */
function makeWeights(c: LlamaConfig): Map<string, { shape: number[]; data: Float32Array }> {
  const w = new Map<string, { shape: number[]; data: Float32Array }>();
  let seed = 7;
  const mat = (name: string, n: number, k: number, scale = 0.2) => w.set(name, { shape: [n, k], data: rand(n * k, seed++, scale) });
  const vec = (name: string, n: number) => w.set(name, { shape: [n], data: rand(n, seed++, 0.4).map((v) => 1 + v) });
  const [h, P, L] = [c.hiddenSize, c.perLayerInput!, c.numLayers];
  mat("model.embed_tokens.weight", c.vocabSize, h, 1);
  vec("model.norm.weight", h);
  mat("model.per_layer_model_projection.weight", L * P, h);
  vec("model.per_layer_projection_norm.weight", P);
  for (let l = 0; l < L; l++) {
    const p = `model.layers.${l}`;
    const D = headDimOf(c, l);
    const shared = c.kvFrom?.[l] !== undefined;
    const I = c.intermediateSize * (shared ? 2 : 1);
    mat(gemma4PerLayerTable(l), c.vocabSize, P, 1);
    for (const n of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"]) vec(`${p}.${n}.weight`, h);
    mat(`${p}.self_attn.q_proj.weight`, c.numHeads * D, h);
    vec(`${p}.self_attn.q_norm.weight`, D);
    if (!shared) {
      mat(`${p}.self_attn.k_proj.weight`, c.numKvHeads * D, h);
      mat(`${p}.self_attn.v_proj.weight`, c.numKvHeads * D, h);
      vec(`${p}.self_attn.k_norm.weight`, D);
    }
    mat(`${p}.self_attn.o_proj.weight`, h, c.numHeads * D);
    mat(`${p}.mlp.gate_proj.weight`, I, h);
    mat(`${p}.mlp.up_proj.weight`, I, h);
    mat(`${p}.mlp.down_proj.weight`, h, I);
    mat(`${p}.per_layer_input_gate.weight`, P, h);
    mat(`${p}.per_layer_projection.weight`, h, P);
  }
  return w;
}

/**
 * Gemma 4's forward pass as transformers writes it, in float64 and without a cache: every position
 * is recomputed from the whole sequence. Returns the logits after the last token.
 */
function reference(c: LlamaConfig, w: Map<string, { shape: number[]; data: Float32Array }>, tokens: number[], features?: Float32Array): Float64Array {
  const W = (name: string) => w.get(name)!;
  const [h, P, L, T] = [c.hiddenSize, c.perLayerInput!, c.numLayers, tokens.length];
  const eps = c.rmsEps;
  const matvec = (name: string, x: ArrayLike<number>): Float64Array => {
    const { shape, data } = W(name);
    const [n, k] = shape as [number, number];
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < k; j++) s += data[i * k + j]! * x[j]!;
      y[i] = s;
    }
    return y;
  };
  /** RMSNorm over consecutive groups of `dim` values, with a weight of `dim` values or none. */
  const rms = (x: Float64Array, dim: number, weight?: string): Float64Array => {
    const y = new Float64Array(x.length);
    const wv = weight ? W(weight).data : undefined;
    for (let g = 0; g < x.length; g += dim) {
      let ss = 0;
      for (let i = 0; i < dim; i++) ss += x[g + i]! ** 2;
      const r = (ss / dim + eps) ** -0.5;
      for (let i = 0; i < dim; i++) y[g + i] = x[g + i]! * r * (wv ? wv[i]! : 1);
    }
    return y;
  };
  const gelu = (v: number) => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3)));
  /** Rotate-half rotary of every head of x at position pos. */
  const rope = (x: Float64Array, D: number, pos: number, inv: Float32Array): Float64Array => {
    const y = x.slice();
    const half = D / 2;
    for (let g = 0; g < x.length; g += D)
      for (let i = 0; i < half; i++) {
        const a = pos * (inv[i] ?? 0);
        const [x0, x1] = [x[g + i]!, x[g + i + half]!];
        y[g + i] = x0 * Math.cos(a) - x1 * Math.sin(a);
        y[g + i + half] = x1 * Math.cos(a) + x0 * Math.sin(a);
      }
    return y;
  };
  const embed = W("model.embed_tokens.weight").data;
  const text = (t: number) => Float64Array.from(embed.subarray(t * h, (t + 1) * h), (v) => v * c.embedScale!);
  // An image token (an id from the vocabulary's size on) is its row of the features, unscaled; in
  // the per-layer tables it counts as the padding token, id 0, while the projection takes its features.
  const image = (t: number) => t >= c.vocabSize;
  const xs = tokens.map((t) => (image(t) ? Float64Array.from(features!.subarray((t - c.vocabSize) * h, (t - c.vocabSize + 1) * h)) : text(t)));
  // Per-layer inputs: [T][L · P].
  const per = tokens.map((id, i) => {
    const t = image(id) ? 0 : id;
    const proj = rms(matvec("model.per_layer_model_projection.weight", xs[i]!).map((v) => v * h ** -0.5), P, "model.per_layer_projection_norm.weight");
    const out = new Float64Array(L * P);
    for (let l = 0; l < L; l++) {
      const table = W(gemma4PerLayerTable(l)).data;
      for (let j = 0; j < P; j++) out[l * P + j] = (proj[l * P + j]! + table[t * P + j]! * Math.sqrt(P)) * 2 ** -0.5;
    }
    return out;
  });
  const keys: Float64Array[][] = [];
  const values: Float64Array[][] = [];
  for (let l = 0; l < L; l++) {
    const p = `model.layers.${l}`;
    const D = headDimOf(c, l);
    const sliding = c.slidingLayers![l]!;
    // The frequencies as the model code writes them: default rotary over the whole sliding head; on full layers the first quarter of the pairs.
    const inv = sliding
      ? Float32Array.from({ length: D / 2 }, (_, i) => 1 / c.ropeLocalTheta! ** ((2 * i) / D))
      : Float32Array.from({ length: D / 2 }, (_, i) => (i < Math.floor((0.25 * D) / 2) ? 1 / c.ropeTheta ** ((2 * i) / D) : 0));
    const src = c.kvFrom?.[l];
    const normed = xs.map((x) => rms(x, h, `${p}.input_layernorm.weight`));
    const qs = normed.map((n, t) => rope(rms(matvec(`${p}.self_attn.q_proj.weight`, n), D, `${p}.self_attn.q_norm.weight`), D, t, inv));
    if (src === undefined) {
      keys[l] = normed.map((n, t) => rope(rms(matvec(`${p}.self_attn.k_proj.weight`, n), D, `${p}.self_attn.k_norm.weight`), D, t, inv));
      values[l] = normed.map((n) => rms(matvec(`${p}.self_attn.v_proj.weight`, n), D));
    }
    const [K, V] = [keys[src ?? l]!, values[src ?? l]!];
    const group = c.numHeads / c.numKvHeads;
    for (let t = 0; t < T; t++) {
      const first = sliding ? Math.max(0, t + 1 - c.slidingWindow!) : 0;
      const out = new Float64Array(c.numHeads * D);
      for (let hd = 0; hd < c.numHeads; hd++) {
        const kh = Math.floor(hd / group) * D;
        const scores: number[] = [];
        for (let j = first; j <= t; j++) {
          let s = 0;
          for (let d = 0; d < D; d++) s += qs[t]![hd * D + d]! * K[j]![kh + d]!;
          scores.push(s);
        }
        const mx = Math.max(...scores);
        const e = scores.map((s) => Math.exp(s - mx));
        const sum = e.reduce((a, b) => a + b, 0);
        e.forEach((pr, i) => {
          for (let d = 0; d < D; d++) out[hd * D + d]! += (pr / sum) * V[first + i]![kh + d]!;
        });
      }
      const a = rms(matvec(`${p}.self_attn.o_proj.weight`, out), h, `${p}.post_attention_layernorm.weight`);
      const x = xs[t]!.map((v, i) => v + a[i]!);
      const n2 = rms(x, h, `${p}.pre_feedforward_layernorm.weight`);
      const gate = matvec(`${p}.mlp.gate_proj.weight`, n2);
      const up = matvec(`${p}.mlp.up_proj.weight`, n2);
      const m = rms(matvec(`${p}.mlp.down_proj.weight`, gate.map((g, i) => gelu(g) * up[i]!)), h, `${p}.post_feedforward_layernorm.weight`);
      for (let i = 0; i < h; i++) x[i]! += m[i]!;
      const g2 = matvec(`${p}.per_layer_input_gate.weight`, x).map((g, i) => gelu(g) * per[t]![l * P + i]!);
      const back = rms(matvec(`${p}.per_layer_projection.weight`, g2), h, `${p}.post_per_layer_input_norm.weight`);
      xs[t] = x.map((v, i) => (v + back[i]!) * (c.layerScalars?.[l] ?? 1));
    }
  }
  const last = rms(xs[T - 1]!, h, "model.norm.weight");
  const logits = new Float64Array(c.vocabSize);
  for (let v = 0; v < c.vocabSize; v++) {
    let s = 0;
    for (let i = 0; i < h; i++) s += embed[v * h + i]! * last[i]!;
    logits[v] = c.logitSoftcap! * Math.tanh(s / c.logitSoftcap!);
  }
  return logits;
}

function upload(be: CpuBackend, w: Map<string, { shape: number[]; data: Float32Array }>): Map<string, Tensor> {
  const out = new Map<string, Tensor>();
  for (const [name, { shape, data }] of w) {
    const buf = be.alloc(data.byteLength, "storage", name);
    be.write(buf, 0, data);
    out.set(name, { shape, dtype: "f32", buf });
  }
  return out;
}

const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]!)), 0);
const TOKENS = [3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 32, 38, 46, 26, 43, 38, 32, 7];

describe("Gemma 4", () => {
  const base = configFromHF(HF);
  const cfg: LlamaConfig = { ...base, layerScalars: [0.4, 0.9, 0.7, 0.3, 0.8, 0.6] };

  it("reads the config", () => {
    expect(base).toMatchObject({
      family: "gemma4", hiddenSize: 64, numLayers: 6, numHeads: 2, numKvHeads: 1, headDim: 32, vocabSize: 50, qkNorm: true, valueNorm: true, sandwichNorms: true, attnScale: 1,
      slidingWindow: 4, slidingLayers: [true, false, true, false, true, false], layerHeadDims: [32, 64, 32, 64, 32, 64], kvFrom: [undefined, undefined, undefined, undefined, 2, 3],
      ropeTheta: 1e6, ropeLocalTheta: 1e4, ropeProportional: 8, perLayerInput: 32, logitSoftcap: 30, eosTokenIds: [1, 9], activation: "gelu", embedScale: 8,
    });
    // E2B's own numbers: 35 layers, the last 20 reading layers 13 (sliding) and 14 (full).
    const types = Array.from({ length: 35 }, (_, l) => ((l + 1) % 5 ? "sliding_attention" : "full_attention"));
    const e2b = configFromHF({ model_type: "gemma4", text_config: { ...HF.text_config, hidden_size: 1536, intermediate_size: 6144, num_hidden_layers: 35, num_attention_heads: 8, head_dim: 256, global_head_dim: 512, hidden_size_per_layer_input: 256, vocab_size: 262144, vocab_size_per_layer_input: 262144, num_kv_shared_layers: 20, layer_types: types, sliding_window: 512, dtype: "bfloat16" } });
    expect(e2b.kvFrom!.slice(12, 21)).toEqual([undefined, undefined, undefined, 13, 13, 13, 13, 14, 13]);
    expect(e2b.ropeProportional).toBe(64);
    // √1536 = 39.19, rounded to the checkpoint's bf16 as transformers does.
    expect(e2b.embedScale).toBe(39.25);
    expect(() => configFromHF({ model_type: "gemma4", text_config: { ...HF.text_config, enable_moe_block: true } })).toThrow(/mixture-of-experts/);
    expect(() => configFromHF({ model_type: "gemma4", text_config: { ...HF.text_config, attention_k_eq_v: true } })).toThrow(/attention_k_eq_v/);
    expect(() => configFromHF({ model_type: "gemma4", text_config: { ...HF.text_config, num_kv_shared_layers: 5 } })).toThrow(/no earlier layer/);
    expect(() => configFromHF({ model_type: "gemma4" })).toThrow(/text_config/);
  });

  it("reads the config llama.cpp writes, and its tensor names", () => {
    const meta = new Map<string, unknown>([
      ["general.architecture", "gemma4"],
      ["gemma4.block_count", 6], ["gemma4.context_length", 131072], ["gemma4.embedding_length", 64], ["gemma4.feed_forward_length", [96, 96, 96, 96, 192, 192]],
      ["gemma4.attention.head_count", 2], ["gemma4.attention.head_count_kv", 1], ["gemma4.rope.freq_base", 1000000], ["gemma4.rope.freq_base_swa", 10000],
      ["gemma4.attention.layer_norm_rms_epsilon", 1e-6], ["gemma4.attention.key_length", 64], ["gemma4.attention.key_length_swa", 32], ["gemma4.final_logit_softcapping", 30],
      ["gemma4.attention.sliding_window", 4], ["gemma4.attention.shared_kv_layers", 2], ["gemma4.embedding_length_per_layer_input", 32],
      ["gemma4.attention.sliding_window_pattern", [true, false, true, false, true, false]],
      ["tokenizer.ggml.tokens", [...Array.from({ length: 9 }, (_, i) => `t${i}`), "<turn|>", ...Array.from({ length: 40 }, (_, i) => `u${i}`)]],
      ["tokenizer.ggml.eos_token_id", 1], ["tokenizer.ggml.bos_token_id", 2],
    ]);
    const g = configFromGGUF(meta, ["token_embd.weight", "blk.0.attn_q.weight"]);
    const { embedScale: _a, contextLength: _b, bosTokenId: _c, ...same } = base;
    expect(g).toMatchObject(same);
    expect(g.bosTokenId).toBe(2);
    const hf = (n: string) => ggufToHF(n, "gemma4");
    expect(hf("blk.3.inp_gate.weight")).toBe("model.layers.3.per_layer_input_gate.weight");
    expect(hf("blk.3.proj.weight")).toBe("model.layers.3.per_layer_projection.weight");
    expect(hf("blk.3.post_norm.weight")).toBe("model.layers.3.post_per_layer_input_norm.weight");
    expect(hf("blk.3.ffn_norm.weight")).toBe("model.layers.3.pre_feedforward_layernorm.weight");
    expect(hf("blk.3.post_ffw_norm.weight")).toBe("model.layers.3.post_feedforward_layernorm.weight");
    expect(hf("per_layer_model_proj.weight")).toBe("model.per_layer_model_projection.weight");
    expect(hf("per_layer_proj_norm.weight")).toBe("model.per_layer_projection_norm.weight");
    // Read by the loader itself, or not at all.
    for (const n of ["per_layer_token_embd.weight", "blk.3.layer_output_scale.weight", "rope_freqs.weight"]) expect(hf(n)).toBeUndefined();
    expect(() => configFromGGUF(meta, ["blk.0.ffn_gate_inp.weight"])).toThrow(/mixture-of-experts/);
  });

  it("gives the full-attention layers rotary on a quarter of each head", () => {
    const f = ropeInvFreq({ ...cfg, headDim: 64 });
    expect(f.length).toBe(32);
    expect(f[0]).toBe(1);
    expect(f[7]).toBeCloseTo(1e6 ** (-14 / 64), 6);
    expect([...f.subarray(8)].every((v) => v === 0)).toBe(true);
  });

  it("matches the reference forward pass: prefill at once, in chunks, and token by token", async () => {
    const w = makeWeights(cfg);
    const want = reference(cfg, w, TOKENS);
    const be = new CpuBackend({ threads: 1 });
    const read = async (t: Tensor) => new Float32Array(await be.readback(physical(t.buf), 0, t.shape.reduce((a, b) => a * b, 1) * 4));
    for (const fuse of [true, false]) {
      const m = new LlamaModel(be, cfg, upload(be, w), { maxSeq: 64, kvDtype: "f32", fuse });
      // Only the four layers that make keys and values hold a cache.
      expect(m.slots).toEqual([64, 64, 64, 64, 0, 0]);
      const step = (pos0: number, lastRow: number) => m.setStep({ pos0, lastRow, seed: 0, temperature: 0 });
      // The whole prompt in one pass.
      const whole = m.buildPlan(TOKENS.length);
      be.write(whole.tokens, 0, Uint32Array.from(TOKENS));
      step(0, TOKENS.length - 1);
      be.submit(whole.dispatches);
      expect(maxDiff(await read(whole.logits), want)).toBeLessThan(2e-3);
      // The softcap keeps logits inside ±30, and the answer is not flat.
      expect(Math.max(...want.map(Math.abs))).toBeLessThan(30);
      expect(Math.max(...want) - Math.min(...want)).toBeGreaterThan(1);

      // The layers that share a cache run on the sampled row alone, whichever row that is: the
      // rows after it are padding that later tokens overwrite.
      step(0, 8);
      be.submit(whole.dispatches);
      expect(maxDiff(await read(whole.logits), reference(cfg, w, TOKENS.slice(0, 9)))).toBeLessThan(2e-3);
      // One copy takes that row out of the residual stream, one more per shared layer out of its input.
      expect(whole.dispatches.filter((d) => d.label === "copy_row").length).toBe(3);

      // In chunks: the first stops after the last layer that writes a cache, the second runs everything.
      const [a, b] = [m.buildPlan(11), m.buildPlan(TOKENS.length - 11)];
      expect(a.kvBody!.length).toBeLessThan(a.body.length);
      be.write(a.tokens, 0, Uint32Array.from(TOKENS.slice(0, 11)));
      step(0, 10);
      be.submit(a.kvBody!);
      be.write(b.tokens, 0, Uint32Array.from(TOKENS.slice(11)));
      step(11, TOKENS.length - 12);
      be.submit(b.dispatches);
      expect(maxDiff(await read(b.logits), want)).toBeLessThan(2e-3);

      // Token by token, as decoding does: each step's logits match the reference for that prefix.
      const one = m.buildPlan(1);
      for (let t = 0; t < TOKENS.length; t++) {
        be.write(one.tokens, 0, Uint32Array.of(TOKENS[t]!));
        step(t, 0);
        be.submit(one.dispatches);
        if (t === 6 || t === TOKENS.length - 1) expect(maxDiff(await read(one.logits), reference(cfg, w, TOKENS.slice(0, t + 1)))).toBeLessThan(2e-3);
      }
      // The cache it leaves is what a snapshot holds: four layers' rows, two head sizes.
      expect(m.kvSnapshotBytes(10)).toBe(2 * 10 * 4 * (32 + 64 + 32 + 64));
      m.destroy();
    }
  });

  it("takes image tokens as feature rows, and as padding in the per-layer tables", async () => {
    const w = makeWeights(cfg);
    const be = new CpuBackend({ threads: 1 });
    const read = async (t: Tensor) => new Float32Array(await be.readback(physical(t.buf), 0, t.shape.reduce((a, b) => a * b, 1) * 4));
    const m = new LlamaModel(be, cfg, upload(be, w), { maxSeq: 64, kvDtype: "f32", imageTokens: 6 });
    const features = rand(6 * cfg.hiddenSize, 99, 2);
    be.write(m.imageFeatures!, 0, features);
    // Text, four image tokens (rows 1 to 4 of the features), text; sampled after the last and inside the image.
    const V = cfg.vocabSize;
    const ids = [3, 14, V + 1, V + 2, V + 3, V + 4, 15, 9, 26];
    const plan = m.buildPlan(ids.length);
    be.write(plan.tokens, 0, Uint32Array.from(ids));
    for (const last of [ids.length - 1, 4]) {
      m.setStep({ pos0: 0, lastRow: last, seed: 0, temperature: 0 });
      be.submit(plan.dispatches);
      expect(maxDiff(await read(plan.logits), reference(cfg, w, ids.slice(0, last + 1), features))).toBeLessThan(2e-3);
    }
    // The image changes the answer: the same prompt with padding tokens in its place gives other logits.
    expect(maxDiff(reference(cfg, w, ids, features), reference(cfg, w, ids.map((t) => (t >= V ? 0 : t))))).toBeGreaterThan(0.1);
    m.destroy();
  });

  it("refuses the i8 cache for heads past 256", () => {
    const be = new CpuBackend({ threads: 1 });
    const wide: LlamaConfig = { ...cfg, layerHeadDims: [32, 512, 32, 512, 32, 512] };
    expect(() => new LlamaModel(be, wide, new Map(), { maxSeq: 64, kvDtype: "i8" })).toThrow(/i8 cache/);
  });

  it("runs a LoRA adapter as the reference runs the merged weights, shared layers' k and v left out", async () => {
    const w = makeWeights(cfg);
    // A rank-4 update on every projection the layers have; on the layers that read another's cache,
    // k_proj and v_proj too, as a PEFT adapter may hold them, though they change nothing there.
    const adapter = new Map<string, { rank: number; A: Float32Array; B: Float32Array }>();
    const merged = new Map(w);
    let seed = 500;
    for (let l = 0; l < cfg.numLayers; l++) {
      const D = headDimOf(cfg, l);
      const shared = cfg.kvFrom?.[l] !== undefined;
      const I = cfg.intermediateSize * (shared ? 2 : 1);
      const h = cfg.hiddenSize;
      const shapes: [string, string, number, number][] = [
        ["q_proj", "self_attn", cfg.numHeads * D, h],
        ["k_proj", "self_attn", cfg.numKvHeads * D, h],
        ["v_proj", "self_attn", cfg.numKvHeads * D, h],
        ["o_proj", "self_attn", h, cfg.numHeads * D],
        ["gate_proj", "mlp", I, h],
        ["up_proj", "mlp", I, h],
        ["down_proj", "mlp", h, I],
      ];
      for (const [m, group, n, k] of shapes) {
        const r = 4;
        const A = rand(r * k, seed++, 0.3);
        const B = rand(n * r, seed++, 0.3);
        adapter.set(`${l}.${m}`, { rank: r, A, B });
        const name = `model.layers.${l}.${group}.${m}.weight`;
        const base = w.get(name);
        if (!base) continue;
        const data = base.data.slice();
        for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) for (let t = 0; t < r; t++) data[i * k + j]! += B[i * r + t]! * A[t * k + j]!;
        merged.set(name, { shape: base.shape, data });
      }
    }
    const want = reference(cfg, merged, TOKENS);
    expect(maxDiff(want, reference(cfg, w, TOKENS))).toBeGreaterThan(0.1);
    const be = new CpuBackend({ threads: 1 });
    const read = async (t: Tensor) => new Float32Array(await be.readback(physical(t.buf), 0, t.shape.reduce((a, b) => a * b, 1) * 4));
    const m = new LlamaModel(be, cfg, upload(be, w), { maxSeq: 64, kvDtype: "f32" });
    m.enableLora(4);
    m.setLora(adapter);
    // The whole prompt at once (the shared layers on the last row alone), then token by token.
    const whole = m.buildPlan(TOKENS.length);
    be.write(whole.tokens, 0, Uint32Array.from(TOKENS));
    m.setStep({ pos0: 0, lastRow: TOKENS.length - 1, seed: 0, temperature: 0 });
    be.submit(whole.dispatches);
    expect(maxDiff(await read(whole.logits), want)).toBeLessThan(2e-3);
    const one = m.buildPlan(1);
    for (let t = 0; t < TOKENS.length; t++) {
      be.write(one.tokens, 0, Uint32Array.of(TOKENS[t]!));
      m.setStep({ pos0: t, lastRow: 0, seed: 0, temperature: 0 });
      be.submit(one.dispatches);
    }
    expect(maxDiff(await read(one.logits), want)).toBeLessThan(2e-3);
    m.destroy();
  });
});
