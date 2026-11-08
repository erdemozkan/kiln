import type { Backend } from "@kiln/core";
import { createWebGPUBackend } from "@kiln/backend-webgpu";
import {
  CachedHttpSource,
  MemorySource,
  cacheKey,
  fetchJSON,
  fetchText,
  fileUrl,
  listRepoFiles,
  parseModelId,
  readGGUFHeader,
  readSafetensorsHeader,
  toF32,
  safetensorsFiles,
  type ByteSource,
  type ModelRef,
  type TensorEntry,
} from "@kiln/loader";
import { CpuBackend, ops, type KvDtype } from "@kiln/kernels";
import { LlamaModel, VisionTower, configFromGGUF, configFromHF, ggufToHF, graniteName, multimodalTextName, patchK, visionConfigFromGGUF, weightScale, type LlamaConfig } from "@kiln/models";
import { Tokenizer, type ChatTemplate, type TokenizerJSON } from "@kiln/tokenizers";
import { Engine } from "./engine";
import { llamaMatmulShapes, tuneMatmuls } from "./tune";
import { loadWeights, type LoadProgress, type Quant, type WeightFile } from "./weights";

export interface LoadOptions {
  /** Weight storage on the GPU: "auto" (default: q4, with GGUF tensors kept at their own precision), "q4", "q8" or "f32". See `Quant`. */
  quant?: Quant;
  /**
   * KV cache length in tokens. Default: the model's trained context, at most 8192 tokens and 512 MiB
   * of cache (see `defaultMaxSeq`).
   */
  maxSeq?: number;
  /**
   * GPU bytes for K/V of earlier prompts (other conversations, other system prompts), so returning to
   * one does not prefill it again. Default 256 MiB; 0 keeps only the live cache.
   */
  prefixCacheBytes?: number;
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
   * Where the model runs: "gpu" (WebGPU, fails without it), "cpu" (TypeScript kernels on the worker
   * thread, slow, for small models), or "auto" (default): WebGPU when the browser has a usable
   * adapter, else the CPU. Ignored when `backend` is given.
   */
  device?: "auto" | "gpu" | "cpu";
  /**
   * Threads for the CPU backend's matmuls (default: the browser's hardwareConcurrency, at most 4).
   * More than one needs a cross-origin isolated page (COOP/COEP headers); otherwise one is used.
   */
  cpuThreads?: number;
  /**
   * Time matmul variants on this GPU and keep the fastest per shape (default true). Runs while the
   * weights download and is cached per device, so later loads only read the cached choice.
   */
  tune?: boolean;
  /** Longest GPU work in one submit, in estimated ms (default 50; see `Engine.maxSubmitMs`). */
  maxSubmitMs?: number;
  /**
   * Gemma 3's vision tower, as llama.cpp's mmproj GGUF: a file name in the model's repo
   * ("mmproj-F16.gguf") or a full id ("org/repo/mmproj.gguf"). Chat messages may then hold images.
   */
  mmproj?: string;
  /** Images whose features stay on the GPU at once, with `mmproj` (default 4, 256 tokens each). */
  imageSlots?: number;
  onProgress?: (p: LoadProgress) => void;
}

/**
 * Loads a model from the Hub: "org/repo" for safetensors or "org/repo/file.gguf" for GGUF.
 * Weights stream into the GPU while downloading; in a worker they are cached in OPFS as they arrive.
 */
export async function createEngine(modelId: string, opts: LoadOptions = {}): Promise<Engine> {
  const ref = parseModelId(modelId, opts.revision);
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const quant = opts.quant ?? "auto";
  ops.setMatmulF16(!!opts.f16Math && backend.limits.features.has("shader-f16"));
  ops.setSubgroups(!!opts.subgroups);
  const { cfg, tokenizer, files } = ref.gguf ? await openGGUF(ref, opts.endpoint) : await openSafetensors(ref, opts.endpoint);
  try {
    const tuning =
      opts.tune === false || backend.name === "cpu"
        ? Promise.resolve()
        : tuneMatmuls(backend, llamaMatmulShapes(cfg, quant === "auto" ? "q4" : quant)).then(
            () => undefined,
            (err: unknown) => console.warn("[kiln] matmul tuning failed; using defaults:", err),
          );
    const weights = await loadWeights(backend, files, quant, opts.onProgress);
    const vision = opts.mmproj ? await openVision(backend, opts.mmproj.split("/").length >= 3 ? parseModelId(opts.mmproj, opts.revision) : { ...ref, gguf: opts.mmproj }, opts) : undefined;
    if (vision && vision.cfg.projectionDim !== cfg.hiddenSize) throw new Error(`mmproj projects to ${vision.cfg.projectionDim}, but the model is ${cfg.hiddenSize} wide`);
    await tuning;
    const imageTokens = vision ? (opts.imageSlots ?? 4) * vision.tokens : 0;
    const model = new LlamaModel(backend, cfg, weights, { maxSeq: opts.maxSeq, kvDtype: opts.kvDtype, fuse: opts.fuse, imageTokens });
    const engine = new Engine(backend, model, tokenizer, vision);
    if (opts.maxSubmitMs !== undefined) engine.maxSubmitMs = opts.maxSubmitMs;
    if (opts.prefixCacheBytes !== undefined) engine.prefixCacheBytes = opts.prefixCacheBytes;
    return engine;
  } finally {
    for (const f of files) f.source.close?.();
  }
}

/**
 * Gemma 3's vision tower from an mmproj GGUF. Matmul weights load as q8; position embeddings, norms
 * and biases as f32; the patch convolution is read as a [hidden, 3·14·14] matrix padded to `patchK`
 * columns, and the projection transposed to [text width, hidden], as the matmul kernels take them.
 */
async function openVision(backend: Backend, ref: ModelRef, opts: LoadOptions): Promise<VisionTower> {
  const { source, header } = await openParsed(ref, ref.gguf!, opts.endpoint, readGGUFHeader);
  try {
    const cfg = visionConfigFromGGUF(header.metadata);
    const linear = /^v\.blk\.\d+\.(attn_[qkv]|attn_out|ffn_up|ffn_down)\.weight$/;
    const ffn = /^v\.blk\.\d+\.ffn_(up|down)\.(weight|bias)$/;
    const manual = new Set(["v.patch_embd.weight", "mm.input_projection.weight"]);
    // The MLP width (4304) is padded with zeros to a multiple of 32 for the matmul kernels: zero rows
    // of ffn_up and its bias give gelu(0) = 0 in the extra columns, which zero ffn_down columns ignore.
    const I = cfg.intermediate;
    const Ip = Math.ceil(I / 32) * 32;
    const padded: TensorEntry[] = [];
    const parts: Uint8Array[] = [];
    let at = 0;
    for (const t of header.tensors.filter((t) => ffn.test(t.name))) {
      const raw = await source.read(t.offset, t.byteLength);
      const size = t.type === "F32" ? 4 : t.type === "F16" ? 2 : 0;
      if (!size) throw new Error(`mmproj: ${t.name} is ${t.type}; F16 or F32 expected`);
      let bytes: Uint8Array;
      let shape: number[];
      if (t.name.endsWith(".bias") || t.name.includes("ffn_up")) {
        // [I] or [I, hidden]: zero rows appended.
        const row = t.shape.length === 2 ? t.shape[1]! * size : size;
        bytes = new Uint8Array(Ip * row);
        bytes.set(raw);
        shape = t.shape.length === 2 ? [Ip, t.shape[1]!] : [Ip];
      } else {
        // ffn_down [hidden, I]: zero columns appended to every row.
        const rows = t.shape[0]!;
        bytes = new Uint8Array(rows * Ip * size);
        for (let r = 0; r < rows; r++) bytes.set(raw.subarray(r * I * size, (r + 1) * I * size), r * Ip * size);
        shape = [rows, Ip];
      }
      padded.push({ ...t, shape, offset: at, byteLength: bytes.length });
      parts.push(bytes);
      at += bytes.length;
    }
    const blob = new Uint8Array(at);
    parts.reduce((o, p) => (blob.set(p, o), o + p.length), 0);
    const padSource = new MemorySource(blob, "mmproj-ffn");
    const keep = header.tensors.filter((t) => !manual.has(t.name) && !ffn.test(t.name));
    const weights = new Map([
      ...(await loadWeights(backend, [{ name: ref.gguf!, source, tensors: keep.filter((t) => linear.test(t.name)) }], "q8", opts.onProgress)),
      ...(await loadWeights(backend, [{ name: ref.gguf!, source, tensors: keep.filter((t) => !linear.test(t.name)) }], "f32")),
      ...(await loadWeights(backend, [{ name: "mmproj-ffn", source: padSource, tensors: padded.filter((t) => t.name.endsWith(".weight")) }], "q8")),
      ...(await loadWeights(backend, [{ name: "mmproj-ffn", source: padSource, tensors: padded.filter((t) => t.name.endsWith(".bias")) }], "f32")),
    ]);
    const read = async (name: string) => {
      const t = header.tensors.find((x) => x.name === name);
      if (!t) throw new Error(`mmproj: missing ${name}`);
      const n = t.shape.reduce((a, b) => a * b, 1);
      return { values: toF32(t.type as Parameters<typeof toF32>[0], await source.read(t.offset, t.byteLength), n), shape: t.shape };
    };
    const upload = (values: Float32Array, shape: number[]) => {
      const buf = backend.alloc(values.byteLength, "storage", "vision/w");
      backend.write(buf, 0, values);
      return { shape, dtype: "f32" as const, buf };
    };
    const K = 3 * cfg.patchSize * cfg.patchSize;
    const patch = (await read("v.patch_embd.weight")).values;
    const patchW = new Float32Array(cfg.hidden * patchK(cfg));
    for (let r = 0; r < cfg.hidden; r++) patchW.set(patch.subarray(r * K, (r + 1) * K), r * patchK(cfg));
    weights.set("v.patch_embd.weight", upload(patchW, [cfg.hidden, patchK(cfg)]));
    // Stored [hidden, projectionDim] (x @ W in transformers); the kernels take [N, K].
    const proj = (await read("mm.input_projection.weight")).values;
    const t = new Float32Array(proj.length);
    for (let i = 0; i < cfg.hidden; i++) for (let j = 0; j < cfg.projectionDim; j++) t[j * cfg.hidden + i] = proj[i * cfg.projectionDim + j]!;
    weights.set("mm.input_projection.weight", upload(t, [cfg.projectionDim, cfg.hidden]));
    return new VisionTower(backend, cfg, weights);
  } finally {
    source.close?.();
  }
}

/** WebGPU, or the CPU backend when asked for or (for "auto") when WebGPU is missing or has no adapter. */
export async function openBackend(device: "auto" | "gpu" | "cpu", cpuThreads?: number): Promise<Backend> {
  const threads = cpuThreads ?? Math.min(4, globalThis.navigator?.hardwareConcurrency ?? 1);
  if (device === "cpu") return new CpuBackend({ threads });
  try {
    return await createWebGPUBackend();
  } catch (err) {
    if (device === "gpu") throw err;
    console.warn(`[kiln] ${err instanceof Error ? err.message : String(err)}; running on the CPU, which is much slower`);
    return new CpuBackend({ threads });
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
export async function openParsed<H>(ref: ModelRef, file: string, endpoint: string | undefined, parse: (s: ByteSource) => Promise<H>): Promise<{ source: ByteSource; header: H }> {
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
