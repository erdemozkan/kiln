import type { WebGPUBackend } from "@kiln/backend-webgpu";
import { Ctx, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { createEngine } from "@kiln/runtime";

/**
 * Profiles a model in a worker (so weights come from the OPFS cache):
 * whole-step GPU time per plan, per-kernel time by replaying each kernel's dispatches alone,
 * and end-to-end tokens/s through the engine.
 */

export interface KernelTime {
  kernel: string;
  dispatches: number;
  ms: number;
}
export interface PlanProfile {
  T: number;
  pos0: number;
  stepMs: number;
  kernels: KernelTime[];
}
export interface ChainRun {
  stepsPerSubmit: number;
  ttftMs: number;
  decodeTokS: number;
  /** Token ids identical to the stepsPerSubmit = 1 run. */
  sameIds: boolean;
}
export interface GrammarRun {
  decodeTokS: number;
  tokens: number;
  ms: number;
  maskMs: number;
  valid: boolean;
  text: string;
}
export interface BenchResult {
  model: string;
  quant: string;
  plans: PlanProfile[];
  probe?: Record<string, number>;
  chained?: ChainRun[];
  /** Lanes per row chosen by the autotuner, per matmul shape. */
  lanes?: Record<string, number>;
  loadMs?: number;
  /** Prompt + first 64 generated ids as a new prompt: reused tokens, TTFT, and whether the continuation matches. */
  prefixReuse?: { cachedTokens: number; promptTokens: number; ttftMs: number; coldTtftMs: number; matches: boolean };
  /** Greedy story with an f32 KV cache and unfused kernels: tokens matching the default run before the first difference. */
  /** f16 matmul math vs the default f32: decode speed, step time and greedy agreement. */
  f16Math?: { decodeTokS: number; stepMs256: number; matching: number; of: number };
  kvF32Match?: { matching: number; of: number; stepMs1536: number };
  /**
   * Subgroup reductions in matmuls (GPUs with `subgroups` only): decode step time at position 256 with
   * and without, alternating on the same weights; then a separate engine for speed and greedy agreement.
   */
  subgroups?: { minSize: number; stepMsOn: number; stepMsOff: number; decodeTokS: number; matching: number; of: number };
  /** Prefill step time with tiled attention (default) vs one workgroup per query row, alternating. */
  prefillAttention?: { T: number; pos0: number; tiledMs: number; rowsMs: number; tiledAttnMs: number; rowsAttnMs: number }[];
  /** Greedy decode under a JSON schema: speed, CPU mask time, and whether the output parses. */
  /** Default (compact) whitespace. */
  grammar?: GrammarRun;
  /** The same with `whitespace: "pretty"`. */
  grammarPretty?: GrammarRun;
  e2e: { promptTokens: number; generated: number; ttftMs: number; decodeTokS: number; text: string };
}

const scope = self as unknown as {
  postMessage(msg: unknown): void;
  onmessage: ((e: MessageEvent<{ model: string; quant: "q4" | "f32" }>) => void) | null;
};

async function gpuTime(be: WebGPUBackend, dispatches: readonly Dispatch[], reps: number): Promise<number> {
  be.submit(dispatches);
  await be.device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) be.submit(dispatches);
  await be.device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / reps;
}

scope.onmessage = async (e) => {
  try {
    const { model, quant } = e.data;
    const tLoad = performance.now();
    const engine = await createEngine(model, { quant });
    const loadMs = performance.now() - tLoad;
    const be = engine.backend as WebGPUBackend;
    const plans: PlanProfile[] = [];
    for (const [T, pos0] of [[1, 256], [1, 1536], [64, 0], [64, 1024]] as const) {
      const plan = engine.model.buildPlan(T);
      engine.model.setStep({ pos0, lastRow: T - 1, seed: 1, temperature: 0 });
      const stepMs = await gpuTime(be, plan.dispatches, T === 1 ? 40 : 10);
      const groups = new Map<string, Dispatch[]>();
      for (const d of plan.dispatches) groups.set(d.label, [...(groups.get(d.label) ?? []), d]);
      const kernels: KernelTime[] = [];
      for (const [kernel, ds] of groups) kernels.push({ kernel, dispatches: ds.length, ms: await gpuTime(be, ds, T === 1 ? 40 : 10) });
      kernels.sort((a, b) => b.ms - a.ms);
      plans.push({ T, pos0, stepMs, kernels });
      plan.destroy();
    }
    const prefillAttention: NonNullable<BenchResult["prefillAttention"]> = [];
    for (const [T, pos0] of [[16, 0], [16, 1024], [64, 0], [64, 1024], [128, 1536]] as const) {
      const time = async (kind: ops.PrefillAttention) => {
        ops.setPrefillAttention(kind);
        const plan = engine.model.buildPlan(T);
        ops.setPrefillAttention("tiled");
        engine.model.setStep({ pos0, lastRow: T - 1, seed: 1, temperature: 0 });
        const step = await gpuTime(be, plan.dispatches, 10);
        const attn = await gpuTime(be, plan.dispatches.filter((d) => d.label.startsWith("attention")), 10);
        plan.destroy();
        return { step, attn };
      };
      const a = await time("tiled");
      const b = await time("rows");
      const a2 = await time("tiled");
      const b2 = await time("rows");
      prefillAttention.push({ T, pos0, tiledMs: Math.min(a.step, a2.step), rowsMs: Math.min(b.step, b2.step), tiledAttnMs: Math.min(a.attn, a2.attn), rowsAttnMs: Math.min(b.attn, b2.attn) });
    }
    // Probes: per-dispatch overhead, CPU encode cost, a bandwidth ceiling, end-to-end gap.
    const probe: Record<string, number> = {};
    {
      const x = be.alloc(4 * 576, "storage", "px");
      const ctx = new Ctx(be, engine.model.step, "probe");
      let t: Tensor = { shape: [1, 576], dtype: "f32", buf: x };
      for (let i = 0; i < 400; i++) t = ops.copyRow(ctx, t);
      probe.tinyDispatchUs = ((await gpuTime(be, ctx.dispatches, 20)) / 400) * 1000;
      ctx.destroy();
      const plan = engine.model.buildPlan(1);
      probe.decodeDispatches = plan.dispatches.length;
      const c0 = performance.now();
      for (let i = 0; i < 50; i++) be.submit(plan.dispatches);
      probe.encodeSubmitMs = (performance.now() - c0) / 50;
      await be.device.queue.onSubmittedWorkDone();
      plan.destroy();
      const big = be.alloc(64 << 20, "storage", "bw");
      const big2 = be.alloc(64 << 20, "storage", "bw2");
      const copy = async () => {
        const e = be.device.createCommandEncoder();
        for (let i = 0; i < 10; i++) e.copyBufferToBuffer((big as unknown as { gpu: GPUBuffer }).gpu ?? (big as unknown as GPUBuffer), 0, (big2 as unknown as { gpu: GPUBuffer }).gpu ?? (big2 as unknown as GPUBuffer), 0, 64 << 20);
        be.device.queue.submit([e.finish()]);
        await be.device.queue.onSubmittedWorkDone();
      };
      await copy();
      const b0 = performance.now();
      await copy();
      probe.copyGBs = (10 * 2 * 64 * 2 ** 20) / ((performance.now() - b0) / 1000) / 1e9;
      be.free(big);
      be.free(big2);
    }
    const ids = engine.tokenizer.encode(engine.chatFormat!.render([{ role: "user", content: "Write a short story about a lighthouse keeper." }]));
    // Same greedy story at several chain lengths: the ids must match exactly, only the speed may differ.
    const chained: ChainRun[] = [];
    let prefixReuse: BenchResult["prefixReuse"];
    let text = "";
    let base: number[] | undefined;
    for (const k of [1, 4, 8]) {
      const got: number[] = [];
      for await (const id of engine.generateTokens(ids, { temperature: 0, maxTokens: 128, stopTokenIds: [], stepsPerSubmit: k })) got.push(id);
      const s = engine.lastStats!;
      base ??= got;
      if (k === 4) text = engine.tokenizer.decode(got);
      chained.push({
        stepsPerSubmit: k,
        ttftMs: s.ttftMs,
        decodeTokS: s.decodeMs ? ((s.generatedTokens - 1) / s.decodeMs) * 1000 : 0,
        sameIds: got.length === base.length && got.every((v, i) => v === base![i]),
      });
    }
    {
      // Prefix reuse: continue the story from token 64. The cache already holds those tokens.
      const cont = [...ids, ...base!.slice(0, 64)];
      const run = async (reuseCache: boolean) => {
        const got: number[] = [];
        for await (const id of engine.generateTokens(cont, { temperature: 0, maxTokens: 32, stopTokenIds: [], reuseCache })) got.push(id);
        return { got, stats: engine.lastStats! };
      };
      for await (const _ of engine.generateTokens(ids, { temperature: 0, maxTokens: 128, stopTokenIds: [] })) void _;
      const warm = await run(true);
      const cold = await run(false);
      const want = base!.slice(64, 96);
      prefixReuse = {
        cachedTokens: warm.stats.cachedTokens,
        promptTokens: warm.stats.promptTokens,
        ttftMs: warm.stats.ttftMs,
        coldTtftMs: cold.stats.ttftMs,
        matches: warm.got.every((v, i) => v === want[i]) && cold.got.every((v, i) => v === want[i]),
      };
    }
    const runs: Record<"pretty" | "compact", GrammarRun[]> = { pretty: [], compact: [] };
    {
      const schema = {
        type: "object",
        properties: { city: { type: "string" }, country: { type: "string" }, population: { type: "integer" }, landmarks: { type: "array", items: { type: "string" }, maxItems: 4 } },
        required: ["city", "country", "population", "landmarks"],
      } as const;
      const prompt = engine.tokenizer.encode(engine.chatFormat!.render([{ role: "user", content: "Describe Paris as JSON." }]));
      // Alternating, so clock drift hits both alike; the first run also builds the token trie.
      for (const whitespace of ["pretty", "compact", "pretty", "compact"] as const) {
        const got: number[] = [];
        for await (const id of engine.generateTokens(prompt, { temperature: 0, maxTokens: 128, format: { type: "json", schema: schema as never, whitespace } })) got.push(id);
        const gs = engine.lastStats!;
        const out = engine.tokenizer.decode(got);
        let valid = true;
        try {
          JSON.parse(out);
        } catch {
          valid = false;
        }
        runs[whitespace].push({
          decodeTokS: gs.decodeMs ? ((gs.generatedTokens - 1) / gs.decodeMs) * 1000 : 0,
          tokens: gs.generatedTokens,
          ms: gs.decodeMs,
          maskMs: gs.maskMs ?? 0,
          valid,
          text: out.slice(0, 300),
        });
      }
    }
    const fastest = (rs: GrammarRun[]) => rs.reduce((a, b) => (b.decodeTokS > a.decodeTokS ? b : a));
    const grammar = fastest(runs.compact);
    const grammarPretty = fastest(runs.pretty);
    // Intermediates: one pool shared by every plan built so far (decode, prefill buckets, profiling).
    const mem = engine.model.memory;
    probe.weightMB = mem.weights / 2 ** 20;
    probe.kvMB = mem.kv / 2 ** 20;
    probe.activationMB = mem.activations / 2 ** 20;
    const s = engine.lastStats!;
    const k4 = chained.find((c) => c.stepsPerSubmit === 4)!;
    const result: BenchResult = {
      model,
      quant,
      plans,
      probe,
      chained,
      prefixReuse,
      prefillAttention,
      grammar,
      grammarPretty,
      lanes: Object.fromEntries(ops.matmulLanes()),
      loadMs,
      e2e: {
        promptTokens: s.promptTokens,
        generated: s.generatedTokens,
        ttftMs: k4.ttftMs,
        decodeTokS: k4.decodeTokS,
        text: text.slice(0, 300),
      },
    };
    if (be.limits.subgroupMinSize) {
      const step = async (on: boolean) => {
        ops.setSubgroups(on);
        const plan = engine.model.buildPlan(1);
        ops.setSubgroups(false);
        engine.model.setStep({ pos0: 256, lastRow: 0, seed: 1, temperature: 0 });
        const ms = await gpuTime(be, plan.dispatches, 40);
        plan.destroy();
        return ms;
      };
      const on = [await step(true), await step(true)];
      const off = [await step(false), await step(false)];
      on.push(await step(true));
      off.push(await step(false));
      result.subgroups = { minSize: be.limits.subgroupMinSize, stepMsOn: Math.min(...on), stepMsOff: Math.min(...off), decodeTokS: 0, matching: 0, of: base!.length };
    }
    engine.destroy();
    if (result.subgroups) {
      const esg = await createEngine(model, { quant, subgroups: true });
      const got: number[] = [];
      for await (const id of esg.generateTokens(ids, { temperature: 0, maxTokens: 128, stopTokenIds: [], stepsPerSubmit: 8 })) got.push(id);
      const st = esg.lastStats!;
      esg.destroy();
      ops.setSubgroups(false);
      let matching = 0;
      while (matching < got.length && got[matching] === base![matching]) matching++;
      Object.assign(result.subgroups, { decodeTokS: ((st.generatedTokens - 1) / st.decodeMs) * 1000, matching });
    }
    if (be.limits.features.has("shader-f16")) {
      const e16 = await createEngine(model, { quant, f16Math: true });
      const got: number[] = [];
      for await (const id of e16.generateTokens(ids, { temperature: 0, maxTokens: 128, stopTokenIds: [], stepsPerSubmit: 8 })) got.push(id);
      const st16 = e16.lastStats!;
      const p16 = e16.model.buildPlan(1);
      e16.model.setStep({ pos0: 256, lastRow: 0, seed: 1, temperature: 0 });
      const stepMs256 = await gpuTime(e16.backend as WebGPUBackend, p16.dispatches, 40);
      p16.destroy();
      e16.destroy();
      ops.setMatmulF16(false);
      let matching = 0;
      while (matching < got.length && got[matching] === base![matching]) matching++;
      result.f16Math = { decodeTokS: ((st16.generatedTokens - 1) / st16.decodeMs) * 1000, stepMs256, matching, of: base!.length };
    }
    {
      const e32 = await createEngine(model, { quant, kvDtype: "f32", tune: false, fuse: false });
      const got: number[] = [];
      for await (const id of e32.generateTokens(ids, { temperature: 0, maxTokens: 128, stopTokenIds: [], stepsPerSubmit: 8 })) got.push(id);
      const p32 = e32.model.buildPlan(1);
      e32.model.setStep({ pos0: 1536, lastRow: 0, seed: 1, temperature: 0 });
      const stepMs1536 = await gpuTime(e32.backend as WebGPUBackend, p32.dispatches, 40);
      p32.destroy();
      e32.destroy();
      let matching = 0;
      while (matching < got.length && got[matching] === base![matching]) matching++;
      result.kvF32Match = { matching, of: base!.length, stepMs1536 };
    }
    scope.postMessage({ ok: true, result });
  } catch (err) {
    scope.postMessage({ ok: false, error: err instanceof Error ? err.stack ?? err.message : String(err) });
  }
};
