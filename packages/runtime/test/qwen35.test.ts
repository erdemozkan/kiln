import { describe, expect, it } from "vitest";
import { physical, type Tensor } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import { LlamaModel, configFromHF, type LlamaConfig } from "@kiln/models";
import { Tokenizer } from "@kiln/tokenizers";
import { Engine } from "../src/engine";

// A small Qwen 3.5: linear attention in layers 0, 1 and 3, gated attention in layer 2.
const CFG: LlamaConfig = configFromHF({
  model_type: "qwen3_5",
  tie_word_embeddings: true,
  text_config: {
    model_type: "qwen3_5_text",
    hidden_size: 64,
    intermediate_size: 128,
    num_hidden_layers: 4,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_dim: 32,
    vocab_size: 96,
    rms_norm_eps: 1e-6,
    eos_token_id: 0,
    attn_output_gate: true,
    layer_types: ["linear_attention", "linear_attention", "full_attention", "linear_attention"],
    linear_conv_kernel_dim: 4,
    linear_key_head_dim: 8,
    linear_num_key_heads: 2,
    linear_num_value_heads: 4,
    linear_value_head_dim: 8,
    max_position_embeddings: 4096,
    rope_parameters: { rope_type: "default", rope_theta: 10000000, partial_rotary_factor: 0.25, mrope_section: [11, 11, 10] },
  },
});

function rand(n: number, seed: number, scale: number): Float32Array {
  let s = seed >>> 0;
  return Float32Array.from({ length: n }, () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s / 2 ** 32 - 0.5) * 2 * scale;
  });
}

function weights(be: CpuBackend): Map<string, Tensor> {
  const c = CFG;
  const g = c.linear!;
  const w = new Map<string, Tensor>();
  let seed = 11;
  const add = (name: string, shape: number[], data: Float32Array) => {
    const buf = be.alloc(data.byteLength, "storage", name);
    be.write(buf, 0, data);
    w.set(name, { shape, dtype: "f32", buf });
  };
  const ones = (n: number) => new Float32Array(n).fill(1);
  const mat = (name: string, rows: number, cols: number, scale = 0.2) => add(name, [rows, cols], rand(rows * cols, seed++, scale));
  mat("model.embed_tokens.weight", c.vocabSize, c.hiddenSize, 1);
  add("model.norm.weight", [c.hiddenSize], ones(c.hiddenSize));
  const keys = 2 * g.keyHeads * g.keyDim;
  const values = g.valueHeads * g.valueDim;
  for (let l = 0; l < c.numLayers; l++) {
    const p = `model.layers.${l}`;
    add(`${p}.input_layernorm.weight`, [c.hiddenSize], ones(c.hiddenSize));
    add(`${p}.post_attention_layernorm.weight`, [c.hiddenSize], ones(c.hiddenSize));
    mat(`${p}.mlp.gate_proj.weight`, c.intermediateSize, c.hiddenSize);
    mat(`${p}.mlp.up_proj.weight`, c.intermediateSize, c.hiddenSize);
    mat(`${p}.mlp.down_proj.weight`, c.hiddenSize, c.intermediateSize);
    if (c.linearLayers![l]) {
      mat(`${p}.linear_attn.in_proj_qkv.weight`, keys + values, c.hiddenSize, 0.4);
      mat(`${p}.linear_attn.in_proj_z.weight`, values, c.hiddenSize);
      mat(`${p}.linear_attn.in_proj_a.weight`, g.valueHeads, c.hiddenSize);
      mat(`${p}.linear_attn.in_proj_b.weight`, g.valueHeads, c.hiddenSize);
      mat(`${p}.linear_attn.conv1d.weight`, keys + values, g.convKernel, 0.6);
      add(`${p}.linear_attn.A_log`, [g.valueHeads], rand(g.valueHeads, seed++, 1));
      add(`${p}.linear_attn.dt_bias`, [g.valueHeads], rand(g.valueHeads, seed++, 1));
      add(`${p}.linear_attn.norm.weight`, [g.valueDim], ones(g.valueDim));
      mat(`${p}.linear_attn.out_proj.weight`, c.hiddenSize, values);
    } else {
      const q = c.numHeads * c.headDim;
      const kv = c.numKvHeads * c.headDim;
      // q_proj holds a gate beside each head's query.
      mat(`${p}.self_attn.q_proj.weight`, 2 * q, c.hiddenSize);
      mat(`${p}.self_attn.k_proj.weight`, kv, c.hiddenSize);
      mat(`${p}.self_attn.v_proj.weight`, kv, c.hiddenSize);
      mat(`${p}.self_attn.o_proj.weight`, c.hiddenSize, q);
      add(`${p}.self_attn.q_norm.weight`, [c.headDim], ones(c.headDim));
      add(`${p}.self_attn.k_norm.weight`, [c.headDim], ones(c.headDim));
    }
  }
  return w;
}

const tok = () => Tokenizer.fromJSON({ model: { type: "BPE", vocab: { a: 1, b: 2 }, merges: [] }, pre_tokenizer: { type: "ByteLevel" } as never });

describe("Qwen 3.5 (linear attention)", () => {
  it("reads the config", () => {
    expect(CFG).toMatchObject({ family: "qwen35", qkNorm: true, attnOutputGate: true, rotaryDims: 8, ropeTheta: 10000000, tieEmbeddings: true, linearLayers: [true, true, false, true] });
    expect(CFG.linear).toEqual({ keyHeads: 2, valueHeads: 4, keyDim: 8, valueDim: 8, convKernel: 4 });
  });

  it("gives the same logits whether the tokens arrive together or one at a time", async () => {
    const be = new CpuBackend({ threads: 1 });
    const ids = [5, 17, 33, 41, 8, 90, 12, 3, 77, 21];
    const logits = async (chunks: number[]) => {
      const m = new LlamaModel(be, CFG, weights(be), { maxSeq: 64, kvDtype: "f32" });
      expect(m.recurrent).toBe(true);
      let at = 0;
      let last = new Float32Array(0);
      for (const n of chunks) {
        const plan = m.buildPlan(n);
        be.write(plan.tokens, 0, Uint32Array.from(ids.slice(at, at + n)));
        m.setStep({ pos0: at, lastRow: n - 1, seed: 0, temperature: 0 });
        be.submit(plan.dispatches);
        last = new Float32Array(await be.readback(physical(plan.logits.buf), 0, plan.logits.shape[0]! * plan.logits.shape[1]! * 4));
        at += n;
        plan.destroy();
      }
      m.destroy();
      return last;
    };
    const whole = await logits([ids.length]);
    expect(whole.some((v) => !Number.isFinite(v))).toBe(false);
    expect(Math.max(...whole) - Math.min(...whole)).toBeGreaterThan(0.1);
    for (const chunks of [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 7], [9, 1], [2, 1, 4, 3]]) {
      const got = await logits(chunks);
      let diff = 0;
      for (let i = 0; i < whole.length; i++) diff = Math.max(diff, Math.abs(got[i]! - whole[i]!));
      expect(diff, String(chunks)).toBeLessThan(2e-4);
    }
  });

  it("resumes from the state saved after a prompt, and never from a position in between", async () => {
    const be = new CpuBackend({ threads: 1 });
    const fresh = () => new Engine(be, new LlamaModel(be, CFG, weights(be), { maxSeq: 64, kvDtype: "f32" }), tok());
    const run = async (e: Engine, prompt: number[], n: number) => {
      // The engine's id-level entry point and its record of the tokens now cached (prompt, then reply).
      const inner = e as unknown as { decodeIds(ids: number[], o: object): AsyncGenerator<unknown>; cached: number[] };
      for await (const _ of inner.decodeIds(prompt, { maxTokens: n, temperature: 0 })) void _;
      return { out: inner.cached.slice(prompt.length), cached: e.lastStats!.cachedTokens };
    };
    const first = [5, 17, 33, 41, 8, 90, 12, 3];
    const e = fresh();
    const a = await run(e, first, 6);
    expect(a.cached).toBe(0);
    // The next turn repeats the prompt and the reply, then goes on: it resumes at the end of the first prompt.
    const second = [...first, ...a.out, 44, 45, 46];
    const b = await run(e, second, 6);
    expect(b.cached).toBe(first.length);
    expect(b.out).toEqual((await run(fresh(), second, 6)).out);
    // A prompt that parts ways inside the first one cannot reuse its state: all of it is run again.
    const other = [...first.slice(0, 5), 60, 61, 62];
    const c = await run(e, other, 6);
    expect(c.cached).toBe(0);
    expect(c.out).toEqual((await run(fresh(), other, 6)).out);
    // And back to the second conversation, whose prompt state is still kept.
    const again = await run(e, [...second, ...b.out, 50], 4);
    expect(again.cached).toBe(second.length);
    expect(again.out).toEqual((await run(fresh(), [...second, ...b.out, 50], 4)).out);
    // A chat template's last tokens often change in the next turn (the history leaves out what the
    // generation prompt added), so a state from 16 tokens before the end is kept as well.
    const long = Array.from({ length: 30 }, (_, i) => 10 + ((i * 7) % 80));
    await run(e, long, 3);
    const next = [...long.slice(0, 26), 91, 92, 93, 94, 95, 90];
    const d = await run(e, next, 4);
    expect(d.cached).toBe(14);
    expect(d.out).toEqual((await run(fresh(), next, 4)).out);
    // Scoring starts every continuation from the state after the prompt.
    const conts = [[4], [9, 22, 3], [60, 61]];
    const scores = await e.scoreIds(first, conts);
    const alone = await Promise.all(conts.map(async (c2) => (await fresh().scoreIds(first, [c2]))[0]!));
    for (let i = 0; i < conts.length; i++) expect(scores[i]).toBeCloseTo(alone[i]!, 4);
  });
});
