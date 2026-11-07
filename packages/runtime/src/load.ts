import type { Backend } from "@kiln/core";
import { createWebGPUBackend } from "@kiln/backend-webgpu";
import {
  CachedHttpSource,
  cacheKey,
  fetchJSON,
  fetchText,
  fileUrl,
  listRepoFiles,
  parseModelId,
  readGGUFHeader,
  readSafetensorsHeader,
  safetensorsFiles,
  type ByteSource,
  type ModelRef,
} from "@kiln/loader";
import { ops, type KvDtype } from "@kiln/kernels";
import { LlamaModel, configFromGGUF, configFromHF, ggufToHF, graniteName, multimodalTextName, weightScale, type LlamaConfig } from "@kiln/models";
import { Tokenizer, type ChatTemplate, type TokenizerJSON } from "@kiln/tokenizers";
import { Engine } from "./engine";
import { llamaMatmulShapes, tuneMatmuls } from "./tune";
import { loadWeights, type LoadProgress, type Quant, type WeightFile } from "./weights";

export interface LoadOptions {
  /** Weight storage on the GPU: "q4" (default), "q8" or "f32". See `Quant`. */
  quant?: Quant;
  /**
   * KV cache length in tokens. Default: the model's trained context, at most 8192 tokens and 512 MiB
   * of cache (see `defaultMaxSeq`).
   */
  maxSeq?: number;
  /** KV cache storage, f16 (default) or f32. */
  kvDtype?: KvDtype;
  /** f16 dot products in matmuls when the GPU has shader-f16 (default false; accumulation stays f32). */
  f16Math?: boolean;
  /**
   * Subgroup reductions in matmuls where the GPU has `subgroups` (default false until measured on more
   * GPUs). Without the feature this is a no-op.
   */
  subgroups?: boolean;
  /** Fuse q/k/v and gate/up weights into single matmuls (default true). */
  fuse?: boolean | { qkv?: boolean; mlp?: boolean };
  revision?: string;
  /** Hub endpoint, for mirrors or a local server. */
  endpoint?: string;
  backend?: Backend;
  /**
   * Time matmul variants on this GPU and keep the fastest per shape (default true). Runs while the
   * weights download and is cached per device, so later loads only read the cached choice.
   */
  tune?: boolean;
  /** Longest GPU work in one submit, in estimated ms (default 50; see `Engine.maxSubmitMs`). */
  maxSubmitMs?: number;
  onProgress?: (p: LoadProgress) => void;
}

/**
 * Loads a model from the Hub: "org/repo" for safetensors or "org/repo/file.gguf" for GGUF.
 * Weights stream into the GPU while downloading; in a worker they are cached in OPFS as they arrive.
 */
export async function createEngine(modelId: string, opts: LoadOptions = {}): Promise<Engine> {
  const ref = parseModelId(modelId, opts.revision);
  const backend = opts.backend ?? (await createWebGPUBackend());
  const quant = opts.quant ?? "q4";
  ops.setMatmulF16(!!opts.f16Math && backend.limits.features.has("shader-f16"));
  ops.setSubgroups(!!opts.subgroups);
  const { cfg, tokenizer, files } = ref.gguf ? await openGGUF(ref, opts.endpoint) : await openSafetensors(ref, opts.endpoint);
  try {
    const tuning =
      opts.tune === false
        ? Promise.resolve()
        : tuneMatmuls(backend, llamaMatmulShapes(cfg, quant)).then(
            () => undefined,
            (err: unknown) => console.warn("[kiln] matmul tuning failed; using defaults:", err),
          );
    const weights = await loadWeights(backend, files, quant, opts.onProgress);
    await tuning;
    const model = new LlamaModel(backend, cfg, weights, { maxSeq: opts.maxSeq, kvDtype: opts.kvDtype, fuse: opts.fuse });
    const engine = new Engine(backend, model, tokenizer);
    if (opts.maxSubmitMs !== undefined) engine.maxSubmitMs = opts.maxSubmitMs;
    return engine;
  } finally {
    for (const f of files) f.source.close?.();
  }
}

async function open(ref: ModelRef, file: string, endpoint?: string, fresh = false) {
  return CachedHttpSource.open(fileUrl(ref, file, endpoint), cacheKey(ref, file), { fresh });
}

/**
 * Opens a weight file and parses its header. If the header of a cached copy does not parse, the copy
 * is treated as corrupt (an old cache written before flush watermarks, or damaged storage) and
 * downloaded once more over the same file. Network failures are not retried.
 */
async function openParsed<H>(ref: ModelRef, file: string, endpoint: string | undefined, parse: (s: ByteSource) => Promise<H>): Promise<{ source: ByteSource; header: H }> {
  const source = await open(ref, file, endpoint);
  try {
    return { source, header: await parse(source) };
  } catch (err) {
    if (!(source instanceof CachedHttpSource) || source.failed) throw err;
    console.warn(`[kiln] cached ${file} looks corrupt (${String(err)}); downloading it again`);
    source.close();
    // Overwritten in place rather than evicted: removing the file fails quietly while anything still
    // holds it, and the next open would then trust the same bad bytes again.
    const fresh = await open(ref, file, endpoint, true);
    try {
      return { source: fresh, header: await parse(fresh) };
    } catch (err2) {
      fresh.close?.();
      throw err2;
    }
  }
}

async function openSafetensors(ref: ModelRef, endpoint?: string): Promise<{ cfg: LlamaConfig; tokenizer: Tokenizer; files: WeightFile[] }> {
  // Required files start at once. Optional ones wait for the repo's file listing, so files that do
  // not exist are not requested (each would log a 404); without a listing they are probed.
  const listing = listRepoFiles(ref, endpoint);
  const exists = (file: string) => listing.then((files) => !files || files.has(file));
  const optionalJSON = async <T>(file: string) => ((await exists(file)) ? fetchJSON<T>(fileUrl(ref, file, endpoint), true) : undefined);
  const optionalText = async (file: string) => ((await exists(file)) ? fetchText(fileUrl(ref, file, endpoint), true) : undefined);
  const [config, generation, tokJson, tokConfig, templateFile, names] = await Promise.all([
    fetchJSON<Record<string, unknown>>(fileUrl(ref, "config.json", endpoint)),
    optionalJSON<Record<string, unknown>>("generation_config.json"),
    fetchJSON<TokenizerJSON>(fileUrl(ref, "tokenizer.json", endpoint)),
    optionalJSON<TokenizerConfig>("tokenizer_config.json"),
    optionalText("chat_template.jinja"),
    listing.then((files) => safetensorsFiles(ref, endpoint, files)),
  ]);
  const cfg = configFromHF(config!, generation);
  // Gemma's RMSNorm scales by (1 + w); folding the 1 in at load keeps one norm kernel for every family.
  const transform = cfg.family === "gemma3" ? (name: string, v: Float32Array) => (name.endsWith("norm.weight") ? addOne(v) : undefined) : undefined;
  // Multimodal Gemma 3 (4B and up): load the language model's weights under their text names, skip the rest.
  const rename = config!.model_type === "gemma3" ? multimodalTextName : config!.model_type === "granitemoehybrid" ? graniteName : undefined;
  const scale = cfg.residualScale !== undefined || cfg.logitScale !== undefined ? (name: string) => weightScale(cfg, name) : undefined;
  const files: WeightFile[] = [];
  try {
    for (const name of names) {
      const { source, header } = await openParsed(ref, name, endpoint, readSafetensorsHeader);
      files.push({ name, source, tensors: header.tensors, transform, rename, scale });
    }
  } catch (err) {
    // Otherwise the shards already open keep their OPFS locks (and downloads) until the worker dies.
    for (const f of files) f.source.close?.();
    throw err;
  }
  const tokenizer = Tokenizer.fromJSON(tokJson!);
  tokenizer.chatTemplate = chatTemplate(tokConfig, templateFile);
  return { cfg, tokenizer, files };
}

function addOne(v: Float32Array): void {
  for (let i = 0; i < v.length; i++) v[i]! += 1;
}

interface TokenizerConfig {
  chat_template?: string | { name: string; template: string }[];
  bos_token?: string | { content: string } | null;
  eos_token?: string | { content: string } | null;
}

/** The default chat template: `chat_template.jinja` (newer repos) or the one in tokenizer_config.json. */
function chatTemplate(cfg: TokenizerConfig | undefined, file: string | undefined): ChatTemplate | undefined {
  const inline = cfg?.chat_template;
  const source = file ?? (Array.isArray(inline) ? inline.find((t) => t.name === "default")?.template : inline);
  if (!source) return undefined;
  const token = (t: TokenizerConfig["bos_token"]) => (typeof t === "string" ? t : t?.content);
  return { source, bosToken: token(cfg?.bos_token), eosToken: token(cfg?.eos_token) };
}

async function openGGUF(ref: ModelRef, endpoint?: string): Promise<{ cfg: LlamaConfig; tokenizer: Tokenizer; files: WeightFile[] }> {
  const file = ref.gguf!;
  const { source, header: gguf } = await openParsed(ref, file, endpoint, readGGUFHeader);
  const cfg = configFromGGUF(gguf.metadata, gguf.tensors.map((t) => t.name));
  const scale = cfg.residualScale !== undefined || cfg.logitScale !== undefined ? (name: string) => weightScale(cfg, name) : undefined;
  return {
    cfg,
    tokenizer: Tokenizer.fromGGUF(gguf.metadata),
    // Norms in gemma3 files already hold 1 + w, so unlike safetensors there is nothing to transform.
    files: [{ name: file, source, tensors: gguf.tensors, rename: (n) => ggufToHF(n, cfg.family), scale }],
  };
}
