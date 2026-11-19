import { describe, expect, it, vi } from "vitest";

// Drafts as the engine looks them up, except that while `spoilDrafts` is on every other one has a
// token changed (at its first, second or third place in turn), so that part of it must be undone.
// This small random model repeats itself, and would otherwise agree with every draft.
const spoil = globalThis as unknown as { spoilDrafts?: boolean; draftCalls?: number };
vi.mock("../src/speculate", async (original) => {
  const real = await original<typeof import("../src/speculate")>();
  return {
    ...real,
    lookupDraft: (...args: Parameters<typeof real.lookupDraft>) => {
      const d = real.lookupDraft(...args);
      const g = globalThis as unknown as { spoilDrafts?: boolean; draftCalls?: number };
      const call = (g.draftCalls = (g.draftCalls ?? 0) + 1);
      if (!g.spoilDrafts || call % 2 === 0 || !d.length) return d;
      const at = Math.min(d.length - 1, (call >> 1) % 3);
      return d.map((t, i) => (i === at ? (t + 1) % 96 : t));
    },
  };
});
import { physical, type Tensor } from "@kiln/core";
import { CpuBackend } from "@kiln/kernels";
import { LlamaModel, configFromHF, type LlamaConfig, type LoraWeights } from "@kiln/models";
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

/** Rows and columns of every projection an adapter may update, by layer. */
function projections(l: number): [string, string, number, number][] {
  const c = CFG;
  const g = c.linear!;
  const keys = 2 * g.keyHeads * g.keyDim;
  const values = g.valueHeads * g.valueDim;
  const mlp: [string, string, number, number][] = [
    ["mlp", "gate_proj", c.intermediateSize, c.hiddenSize],
    ["mlp", "up_proj", c.intermediateSize, c.hiddenSize],
    ["mlp", "down_proj", c.hiddenSize, c.intermediateSize],
  ];
  if (c.linearLayers![l]) {
    return [...mlp, ["linear_attn", "in_proj_qkv", keys + values, c.hiddenSize], ["linear_attn", "in_proj_z", values, c.hiddenSize], ["linear_attn", "in_proj_a", g.valueHeads, c.hiddenSize], ["linear_attn", "in_proj_b", g.valueHeads, c.hiddenSize], ["linear_attn", "out_proj", c.hiddenSize, values]];
  }
  const q = c.numHeads * c.headDim;
  const kv = c.numKvHeads * c.headDim;
  return [...mlp, ["self_attn", "q_proj", 2 * q, c.hiddenSize], ["self_attn", "k_proj", kv, c.hiddenSize], ["self_attn", "v_proj", kv, c.hiddenSize], ["self_attn", "o_proj", c.hiddenSize, q]];
}

/** A rank-`r` update for every projection of every layer. */
function adapter(r: number, seed: number): LoraWeights {
  const out = new Map<string, { rank: number; A: Float32Array; B: Float32Array }>();
  let k = seed;
  for (let l = 0; l < CFG.numLayers; l++) for (const [, m, n, cols] of projections(l)) out.set(`${l}.${m}`, { rank: r, A: rand(r * cols, k++, 0.2), B: rand(n * r, k++, 0.2) });
  return out;
}

function weights(be: CpuBackend, merged?: LoraWeights): Map<string, Tensor> {
  const c = CFG;
  const g = c.linear!;
  const w = new Map<string, Tensor>();
  let seed = 11;
  const add = (name: string, shape: number[], data: Float32Array) => {
    // W + B·A for a projection the adapter updates: what the slots must match.
    const at = /^model\.layers\.(\d+)\.(?:self_attn|mlp|linear_attn)\.(\w+)\.weight$/.exec(name);
    const u = at && merged?.get(`${at[1]}.${at[2]}`);
    if (u) {
      const [n, cols] = shape as [number, number];
      for (let i = 0; i < n; i++) for (let j = 0; j < cols; j++) for (let t = 0; t < u.rank; t++) data[i * cols + j]! += u.B[i * u.rank + t]! * u.A[t * cols + j]!;
    }
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

  it("carries on in a new session from the saved keys, values and states", async () => {
    const be = new CpuBackend({ threads: 1 });
    const fresh = () => new Engine(be, new LlamaModel(be, CFG, weights(be), { maxSeq: 64, kvDtype: "f32" }), tok());
    const run = async (e: Engine, prompt: number[], n: number) => {
      const inner = e as unknown as { decodeIds(ids: number[], o: object): AsyncGenerator<unknown>; cached: number[] };
      for await (const _ of inner.decodeIds(prompt, { maxTokens: n, temperature: 0 })) void _;
      return { out: inner.cached.slice(prompt.length), cached: e.lastStats!.cachedTokens };
    };
    const first = Array.from({ length: 24 }, (_, i) => 10 + ((i * 11) % 80));
    const a = fresh();
    const reply = (await run(a, first, 5)).out;
    // What a prompt cache file holds: the attention layers' keys and values, and the two newest states.
    const snapshot = await a.exportCache();
    const states = await a.exportStates();
    expect(states.map((s) => s.tokens)).toEqual([first.length - 16, first.length]);
    expect(a.cacheShape.slots).toEqual([0, 0, 64, 0]);

    const b = fresh();
    expect(b.importCache(snapshot)).toBe(true);
    expect(b.importStates(states)).toBe(2);
    // The next turn: the prompt and reply again, then more. It resumes at the state after the prompt.
    const next = [...first, ...reply, 91, 92, 93];
    const resumed = await run(b, next, 5);
    expect(resumed.cached).toBe(first.length);
    expect(resumed.out).toEqual((await run(fresh(), next, 5)).out);

    // States that do not fit are left out: another size, or more tokens than the cache holds.
    const c = fresh();
    expect(c.importCache(snapshot)).toBe(true);
    expect(c.importStates([{ tokens: 8, data: new Uint8Array(12) }, { tokens: 500, data: states[0]!.data }])).toBe(0);
    // Without states the keys and values alone are not resumed from: the whole prompt runs again, correctly.
    const cold = await run(c, next, 5);
    expect(cold.cached).toBe(0);
    expect(cold.out).toEqual(resumed.out);
  });

  it("checks drafts and undoes the wrong ones: the same tokens as plain decoding", async () => {
    const be = new CpuBackend({ threads: 1 });
    const fresh = () => new Engine(be, new LlamaModel(be, CFG, weights(be), { maxSeq: 96, kvDtype: "f32" }), tok());
    const run = async (e: Engine, prompt: number[], opts: object) => {
      const inner = e as unknown as { decodeIds(ids: number[], o: object): AsyncGenerator<unknown>; cached: number[] };
      for await (const _ of inner.decodeIds(prompt, { maxTokens: 24, temperature: 0, ...opts })) void _;
      return { out: inner.cached.slice(prompt.length), spec: e.lastStats!.speculation };
    };
    // Prompts that repeat themselves, so that looking back for the last tokens finds drafts to propose.
    const prompts = [3, 5, 7, 11, 13].map((k) => {
      const unit = Array.from({ length: 8 }, (_, i) => 10 + ((i * k + k) % 80));
      return [...unit, ...unit, ...unit, unit[0]!, unit[1]!];
    });
    spoil.spoilDrafts = true;
    let passes = 0;
    let rejected = 0;
    let accepted = 0;
    let e = fresh();
    let prompt = prompts[0]!;
    let drafted = { out: [] as number[] };
    for (const p of prompts) {
      const plain = await run(fresh(), p, { draftTokens: 0 });
      e = fresh();
      prompt = p;
      const d = await run(e, p, { draftTokens: 7 });
      drafted = d;
      passes += d.spec!.passes;
      accepted += d.spec!.accepted;
      rejected += d.spec!.drafted - d.spec!.accepted;
      expect(d.out, String(p.slice(0, 8))).toEqual(plain.out);
    }
    spoil.spoilDrafts = false;
    // Both ways a pass can end were seen: a draft kept, and one undone by restoring the state.
    expect(passes).toBeGreaterThan(2);
    expect(rejected).toBeGreaterThan(0);
    expect(accepted).toBeGreaterThan(0);
    // And the next turn resumes from the prompt's snapshot as it does without drafts.
    const next = [...prompt, ...drafted.out, 70, 71, 72];
    const after = await run(e, next, { draftTokens: 7 });
    expect(e.lastStats!.cachedTokens).toBe(prompt.length);
    expect(after.out).toEqual((await run(fresh(), next, { draftTokens: 0 })).out);
  });

  it("applies an adapter to every projection, the linear attention ones included, as merged weights would", async () => {
    const be = new CpuBackend({ threads: 1 });
    const ids = [5, 17, 33, 41, 8, 90, 12, 3, 77, 21];
    const logits = async (m: LlamaModel) => {
      const out: Float32Array[] = [];
      const read = async (t: Tensor) => new Float32Array(await be.readback(physical(t.buf), 0, t.shape[0]! * t.shape[1]! * 4));
      m.resetState();
      const pre = m.buildPlan(ids.length);
      be.write(pre.tokens, 0, Uint32Array.from(ids));
      m.setStep({ pos0: 0, lastRow: ids.length - 1, seed: 0, temperature: 0 });
      be.submit(pre.dispatches);
      out.push(await read(pre.logits));
      const dec = m.buildPlan(1);
      be.write(dec.tokens, 0, Uint32Array.of(7));
      m.setStep({ pos0: ids.length, lastRow: 0, seed: 0, temperature: 0 });
      be.submit(dec.dispatches);
      out.push(await read(dec.logits));
      pre.destroy();
      dec.destroy();
      return out;
    };
    const diff = (a: Float32Array, b: Float32Array) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]!)), 0);
    const lora = adapter(4, 900);
    const base = await logits(new LlamaModel(be, CFG, weights(be), { maxSeq: 64, kvDtype: "f32" }));
    const merged = await logits(new LlamaModel(be, CFG, weights(be, lora), { maxSeq: 64, kvDtype: "f32" }));
    const slots = new LlamaModel(be, CFG, weights(be), { maxSeq: 64, kvDtype: "f32" });
    slots.enableLora(4);
    slots.setLora(lora);
    const adapted = await logits(slots);
    expect(diff(merged[0]!, base[0]!)).toBeGreaterThan(0.05);
    expect(diff(adapted[0]!, merged[0]!)).toBeLessThan(2e-4);
    expect(diff(adapted[1]!, merged[1]!)).toBeLessThan(2e-4);
    // Cleared slots give the base model back.
    slots.setLora(undefined);
    expect(diff((await logits(slots))[0]!, base[0]!)).toBeLessThan(1e-5);
  });

  it("keeps the snapshot the next turn resumes from when two of a call's do not fit together", async () => {
    const be = new CpuBackend({ threads: 1 });
    const e = new Engine(be, new LlamaModel(be, CFG, weights(be), { maxSeq: 64, kvDtype: "f32" }), tok());
    const first = Array.from({ length: 24 }, (_, i) => 3 + ((i * 11) % 80));
    // Room for the snapshot after the whole prompt alone, not for it and the one 16 tokens earlier.
    const budget = Engine as unknown as { STATE_BUDGET: number };
    const was = budget.STATE_BUDGET;
    budget.STATE_BUDGET = e.model.stateBytes + e.model.kvSnapshotBytes(first.length);
    try {
      for await (const _ of e.generateTokens(first, { maxTokens: 2, temperature: 0 })) void _;
      // The next turn's template leaves out the prompt's last tokens: it resumes 16 before its end.
      const next = [...first.slice(0, first.length - 4), 50, 51, 52];
      for await (const _ of e.generateTokens(next, { maxTokens: 2, temperature: 0 })) void _;
      expect(e.lastStats?.cachedTokens).toBe(first.length - 16);
    } finally {
      budget.STATE_BUDGET = was;
    }
  });
});
