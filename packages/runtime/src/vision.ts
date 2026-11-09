// Image models from transformers checkpoints (ViT, DeiT): image classification and image features,
// with transformers' image preprocessing (PIL-exact resizing).
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fileUrl, listRepoFiles, parseModelId, preprocessImage, readPytorchHeader, readSafetensorsHeader, toF32, weightFiles, type ByteSource, type ImageProcessorConfig, type RgbImage, type TensorEntry } from "@kiln/loader";
import { VitModel, vitConfigFromHF, vitName, vitPatchK, type VitConfig } from "@kiln/models";
import { openBackend, openParsed } from "./load";
import { loadWeights, type LoadProgress, type WeightFile } from "./weights";

export interface VisionLoadOptions {
  /** Matrix storage: "f32" (default), "q8" or "q4". */
  quant?: "f32" | "q8" | "q4";
  device?: "auto" | "gpu" | "cpu";
  cpuThreads?: number;
  revision?: string;
  endpoint?: string;
  backend?: Backend;
  onProgress?: (p: LoadProgress) => void;
}

export interface ImageClassificationOptions {
  /** Labels returned, best first (default 5; 0 or negative: all). */
  top_k?: number;
}

export interface ImageFeatureOptions {
  /** The pooled output ([1, hidden], ViT's pooler) instead of the last hidden state [tokens, hidden]. */
  pool?: boolean;
}

/** A linear head applied on the CPU: [out, in] weight and bias. */
interface Head {
  w: Float32Array;
  b: Float32Array;
  out: number;
}

function applyHead(h: Head, x: Float32Array): Float32Array {
  const n = x.length;
  const y = new Float32Array(h.out);
  for (let o = 0; o < h.out; o++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += h.w[o * n + i]! * x[i]!;
    y[o] = s + h.b[o]!;
  }
  return y;
}

/** Softmax of logits, then the `topK` best as { label, score } (all when topK ≤ 0). */
export function topLabels(logits: Float32Array, labels: readonly string[], topK = 5): { label: string; score: number }[] {
  const max = Math.max(...logits);
  const e = Array.from(logits, (v) => Math.exp(v - max));
  const s = e.reduce((a, b) => a + b, 0);
  const all = e.map((v, i) => ({ label: labels[i] ?? `LABEL_${i}`, score: v / s })).sort((a, b) => b.score - a.score);
  return topK > 0 ? all.slice(0, topK) : all;
}

export class VisionEngine {
  constructor(
    readonly backend: Backend,
    readonly cfg: VitConfig,
    readonly model: VitModel,
    readonly processor: ImageProcessorConfig,
    private readonly heads: { cls?: Head; distill?: Head; pooler?: Head },
  ) {}

  /** Pixel values [3, size, size] of an image, as the model's image processor makes them. */
  pixels(img: RgbImage): Float32Array {
    const p = preprocessImage(img, this.processor);
    const S = this.cfg.imageSize;
    if (p.width !== S || p.height !== S) throw new Error(`the image processor made a ${p.width}×${p.height} image; the model takes ${S}×${S}`);
    return p.data;
  }

  /** Class logits of an image (DeiT with a teacher: the mean of the CLS and distillation heads). */
  async logits(img: RgbImage): Promise<Float32Array> {
    if (!this.heads.cls) throw new Error("this model has no classification head");
    const h = await this.model.forward(this.pixels(img));
    const H = this.cfg.hidden;
    const y = applyHead(this.heads.cls, h.subarray(0, H));
    if (this.heads.distill) {
      const d = applyHead(this.heads.distill, h.subarray(H, 2 * H));
      for (let i = 0; i < y.length; i++) y[i] = (y[i]! + d[i]!) / 2;
    }
    return y;
  }

  async classify(img: RgbImage, options: ImageClassificationOptions = {}): Promise<{ label: string; score: number }[]> {
    return topLabels(await this.logits(img), this.cfg.labels, options.top_k ?? 5);
  }

  /** The last hidden state [tokens, hidden], or with `pool` the pooled output [1, hidden]. */
  async features(img: RgbImage, options: ImageFeatureOptions = {}): Promise<{ data: Float32Array; dims: number[] }> {
    const h = await this.model.forward(this.pixels(img));
    const H = this.cfg.hidden;
    if (!options.pool) return { data: h, dims: [1, this.model.tokens, H] };
    if (!this.heads.pooler) throw new Error("this model has no pooler; call without `pool`");
    return { data: applyHead(this.heads.pooler, h.subarray(0, H)).map(Math.tanh), dims: [1, H] };
  }

  destroy(): void {
    this.model.destroy();
  }
}

/** Loads a ViT or DeiT checkpoint (safetensors or pytorch_model.bin) from the Hugging Face Hub. */
export async function createVision(modelId: string, opts: VisionLoadOptions = {}): Promise<VisionEngine> {
  const ref = parseModelId(modelId, opts.revision);
  if (ref.gguf) throw new Error("vision models load from safetensors or PyTorch repos, not GGUF files");
  const ep = opts.endpoint;
  const listing = listRepoFiles(ref, ep);
  const has = async (f: string) => {
    const files = await listing;
    return !files || files.has(f);
  };
  const [config, prep, names] = await Promise.all([
    fetchJSON<Record<string, unknown>>(fileUrl(ref, "config.json", ep)),
    has("preprocessor_config.json").then((y) => (y ? fetchJSON<ImageProcessorConfig>(fileUrl(ref, "preprocessor_config.json", ep), true) : undefined)),
    listing.then((files) => weightFiles(ref, ep, files)),
  ]);
  const cfg = vitConfigFromHF(config!);
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const files: WeightFile[] = [];
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name.replace(/^(vit|deit)\./, ""), { source, entry: t });
      files.push({ name, source, tensors: header.tensors.filter((t) => vitName(t.name) !== undefined), rename: vitName });
    }
    const read = async (name: string) => {
      const t = located.get(name);
      if (!t) return undefined;
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
    };
    const need = async (name: string) => {
      const t = await read(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      return t;
    };
    const weights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `vit/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    const H = cfg.hidden;
    // Patch convolution [hidden, 3, p, p] as a matrix [hidden, patchK], zero-padded.
    const pw = await need("embeddings.patch_embeddings.projection.weight");
    const K = vitPatchK(cfg);
    const row = 3 * cfg.patchSize * cfg.patchSize;
    const patch = new Float32Array(H * K);
    for (let o = 0; o < H; o++) patch.set(pw.data.subarray(o * row, (o + 1) * row), o * K);
    upload("patch.weight", [H, K], patch);
    // Constant addend: position embeddings, plus the CLS / distillation tokens on the prefix rows and
    // the patch bias on the rest.
    const pos = await need("embeddings.position_embeddings");
    const T = cfg.prefix + (cfg.imageSize / cfg.patchSize) ** 2;
    if (pos.data.length !== T * H) throw new Error(`${modelId}: position embeddings hold ${pos.data.length / H} tokens, the model has ${T}`);
    const add = Float32Array.from(pos.data);
    const prefixTokens = [await need("embeddings.cls_token")];
    if (cfg.prefix === 2) prefixTokens.push(await need("embeddings.distillation_token"));
    prefixTokens.forEach((t, r) => {
      for (let i = 0; i < H; i++) add[r * H + i] += t.data[i]!;
    });
    const pb = (await read("embeddings.patch_embeddings.projection.bias"))?.data;
    if (pb) for (let r = cfg.prefix; r < T; r++) for (let i = 0; i < H; i++) add[r * H + i] += pb[i]!;
    upload("embed.add", [T, H], add);

    const head = async (p: string): Promise<Head | undefined> => {
      const w = await read(`${p}.weight`);
      if (!w) return undefined;
      const b = (await read(`${p}.bias`))?.data ?? new Float32Array(w.shape[0]!);
      return { w: w.data, b, out: w.shape[0]! };
    };
    const heads = {
      cls: (await head("classifier")) ?? (await head("cls_classifier")),
      distill: await head("distillation_classifier"),
      pooler: (await read("pooler.dense.weight")) ? await head("pooler.dense") : undefined,
    };
    const model = new VitModel(backend, cfg, weights);
    const processor: ImageProcessorConfig = prep ?? { size: { height: cfg.imageSize, width: cfg.imageSize } };
    return new VisionEngine(backend, cfg, model, processor, heads);
  } finally {
    for (const s of sources) s.close?.();
  }
}
