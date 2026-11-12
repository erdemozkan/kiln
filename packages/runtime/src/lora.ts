// PEFT LoRA adapters (adapter_config.json + adapter_model.safetensors) for the decoder LLMs: each
// projection's A and B read as f32, the scale (lora_alpha / r, or / √r with rsLoRA) folded into B.
import { fetchJSON, fileUrl, parseModelId, readSafetensorsHeader, toF32 } from "@kiln/loader";
import { LORA_MODULES, type LoraModule, type LoraUpdate, type LoraWeights } from "@kiln/models";
import { assertLinearRegex } from "@kiln/tokenizers";
import { openParsed } from "./load";

/** adapter_config.json, as far as Kiln reads it. */
export interface LoraConfig {
  peft_type?: string;
  r?: number;
  lora_alpha?: number;
  use_rslora?: boolean;
  use_dora?: boolean;
  fan_in_fan_out?: boolean;
  target_modules?: string[] | string;
  modules_to_save?: string[] | null;
  rank_pattern?: Record<string, number>;
  alpha_pattern?: Record<string, number>;
  base_model_name_or_path?: string;
  [key: string]: unknown;
}

/** A loaded adapter: its updates, the largest rank, and what it adapts. */
export interface LoraAdapter {
  id: string;
  rank: number;
  weights: LoraWeights;
  /** The projections it updates ("q_proj", …). */
  modules: LoraModule[];
  /** The base model the adapter was trained on, as its config names it. */
  base?: string;
}

const TENSOR = /layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)\.lora_([AB])(?:\.default)?\.weight$/;

/** PEFT's pattern lookup: a key applies to a module whose name is the key or ends in ".key" (keys may be regexes). */
function patternValue(patterns: Record<string, number> | undefined, module: string): number | undefined {
  for (const [key, v] of Object.entries(patterns ?? {})) {
    // Keys come from the adapter repo: only patterns that cannot backtrack exponentially are compiled.
    assertLinearRegex(key, "adapter_config.json pattern");
    if (new RegExp(`(^|\\.)(${key})$`).test(module)) return v;
  }
  return undefined;
}

/** Parses an adapter from its config and tensors (by name, f32), scales folded into B. */
export function loraFromTensors(id: string, config: LoraConfig, tensors: Map<string, { shape: number[]; data: Float32Array }>): LoraAdapter {
  if (config.peft_type && config.peft_type !== "LORA") throw new Error(`${id}: ${config.peft_type} adapters are not supported (LoRA is)`);
  if (config.use_dora) throw new Error(`${id}: DoRA adapters are not supported`);
  if (config.fan_in_fan_out) throw new Error(`${id}: fan_in_fan_out adapters (GPT-2 style Conv1D) are not supported`);
  if (config.modules_to_save?.length) throw new Error(`${id}: adapters that also replace whole modules (modules_to_save: ${config.modules_to_save.join(", ")}) are not supported`);
  const parts = new Map<string, { layer: number; group: string; module: LoraModule; A?: { shape: number[]; data: Float32Array }; B?: { shape: number[]; data: Float32Array } }>();
  for (const [name, t] of tensors) {
    if (!/lora_|lora_embedding/.test(name)) continue;
    const m = TENSOR.exec(name);
    if (!m || !(LORA_MODULES as readonly string[]).includes(m[3]!)) throw new Error(`${id}: "${name}" adapts a module Kiln cannot (it adapts ${LORA_MODULES.join(", ")})`);
    const key = `${m[1]}.${m[3]}`;
    const p = parts.get(key) ?? { layer: Number(m[1]), group: m[2]!, module: m[3] as LoraModule };
    p[m[4] as "A" | "B"] = t;
    parts.set(key, p);
  }
  if (!parts.size) throw new Error(`${id} has no LoRA tensors`);
  const weights = new Map<string, LoraUpdate>();
  let rank = 0;
  for (const [key, p] of parts) {
    if (!p.A || !p.B) throw new Error(`${id}: ${key} has lora_${p.A ? "A" : "B"} but not lora_${p.A ? "B" : "A"}`);
    const r = p.A.shape[0]!;
    if (p.B.shape[1] !== r) throw new Error(`${id}: ${key} has A [${p.A.shape}] and B [${p.B.shape}]`);
    const full = `model.layers.${p.layer}.${p.group}.${p.module}`;
    const alpha = patternValue(config.alpha_pattern, full) ?? config.lora_alpha ?? config.r ?? r;
    const scale = config.use_rslora ? alpha / Math.sqrt(r) : alpha / r;
    weights.set(key, { rank: r, A: p.A.data, B: p.B.data.map((v) => v * scale) });
    rank = Math.max(rank, r);
  }
  return { id, rank, weights, modules: [...new Set([...parts.values()].map((p) => p.module))], base: config.base_model_name_or_path };
}

/** Loads a PEFT LoRA adapter repo ("org/repo"): adapter_config.json and adapter_model.safetensors. */
export async function loadLoraAdapter(id: string, opts: { revision?: string; endpoint?: string } = {}): Promise<LoraAdapter> {
  const ref = parseModelId(id, opts.revision);
  const config = await fetchJSON<LoraConfig>(fileUrl(ref, "adapter_config.json", opts.endpoint));
  const { source, header } = await openParsed(ref, "adapter_model.safetensors", opts.endpoint, readSafetensorsHeader).catch((err: unknown) => {
    throw new Error(`${id}: could not read adapter_model.safetensors (${err instanceof Error ? err.message : String(err)}); only safetensors adapters are supported`);
  });
  try {
    const tensors = new Map<string, { shape: number[]; data: Float32Array }>();
    for (const t of header.tensors) {
      if (!t.name.includes("lora_")) continue;
      const count = t.shape.reduce((a, b) => a * b, 1);
      tensors.set(t.name, { shape: t.shape, data: toF32(t.type as Parameters<typeof toF32>[0], await source.read(t.offset, t.byteLength), count) });
    }
    return loraFromTensors(id, config!, tensors);
  } finally {
    source.close?.();
  }
}
