// Image models from transformers checkpoints (ViT, DeiT, CLIP, YOLOS): image classification, image
// features, zero-shot classification and object detection, with transformers' image preprocessing
// (PIL-exact resizing).
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fileUrl, listRepoFiles, parseModelId, preprocessImage, readPytorchHeader, readSafetensorsHeader, toF32, weightFiles, type ByteSource, type ImageProcessorConfig, type RgbImage, type TensorEntry } from "@kiln/loader";
import { ClipTextModel, VitModel, bicubicResize, clipPoolIndex, clipTextConfigFromHF, clipTextName, vitConfigFromHF, vitGrid, vitName, vitPatchK, type VitConfig, type VitEmbeddings } from "@kiln/models";
import { ClipTokenizer, type ClipTokenizerJSON } from "@kiln/tokenizers";
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

export interface ObjectDetectionOptions {
  /** Least score a detection needs (default 0.9). */
  threshold?: number;
  /** Boxes as fractions of the image instead of whole pixels (default false). */
  percentage?: boolean;
}

export interface DetectedObject {
  score: number;
  label: string;
  box: { xmin: number; ymin: number; xmax: number; ymax: number };
}

export interface ZeroShotImageOptions {
  /** Turns a label into the text the model reads (default "This is a photo of {}"). */
  hypothesis_template?: string;
}

/** CLIP's text side and the pieces that join the two towers. */
interface ClipParts {
  text: ClipTextModel;
  tokenizer: ClipTokenizer;
  eosId: number;
  /** post_layernorm of the pooled image token. */
  post: { w: Float32Array; b: Float32Array; eps: number };
  visual: Head;
  textProj: Head;
  /** exp(logit_scale). */
  scale: number;
  /** Text embeddings by text, already normalized. */
  cache: Map<string, Float32Array>;
}

function layerNorm(x: Float32Array, w: Float32Array, b: Float32Array, eps: number): Float32Array {
  let m = 0;
  for (const v of x) m += v;
  m /= x.length;
  let v2 = 0;
  for (const v of x) v2 += (v - m) ** 2;
  const inv = 1 / Math.sqrt(v2 / x.length + eps);
  return x.map((v, i) => (v - m) * inv * w[i]! + b[i]!);
}

function normalize(x: Float32Array): Float32Array {
  let n = 0;
  for (const v of x) n += v * v;
  n = Math.sqrt(n);
  return x.map((v) => v / n);
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
    private readonly clip?: ClipParts,
  ) {}

  /** CLIP's image embedding (post-LayerNorm of the CLS token, visual projection), unnormalized. */
  private async imageEmbed(img: RgbImage): Promise<Float32Array> {
    const c = this.clip!;
    const h = await this.model.forward(this.pixels(img));
    return applyHead(c.visual, layerNorm(h.subarray(0, this.cfg.hidden), c.post.w, c.post.b, c.post.eps));
  }

  /** CLIP's normalized text embedding of a text. */
  async textEmbed(text: string): Promise<Float32Array> {
    return (await this.textEmbeds([text]))[0]!;
  }

  /** CLIP's normalized text embeddings of several texts; the ones not cached run as one batch. */
  async textEmbeds(texts: readonly string[]): Promise<Float32Array[]> {
    const c = this.clip;
    if (!c) throw new Error("this model has no text tower");
    const missing = [...new Set(texts.filter((t) => !c.cache.has(t)))];
    if (missing.length) {
      const ids = missing.map((t) => c.tokenizer.encode(t));
      const hidden = await c.text.forwardBatch(ids, ids.map((s) => clipPoolIndex(s, c.eosId)));
      if (c.cache.size + missing.length > 256) c.cache.clear();
      missing.forEach((t, i) => c.cache.set(t, normalize(applyHead(c.textProj, hidden[i]!))));
    }
    return texts.map((t) => c.cache.get(t)!);
  }

  /**
   * Zero-shot classification (CLIP): every label put into the template, scored by the scaled cosine
   * similarity of its text embedding with the image's, softmaxed over the labels; best first. The
   * labels run as one batch, queued on the GPU alongside the image.
   */
  async zeroShot(img: RgbImage, labels: readonly string[], options: ZeroShotImageOptions = {}): Promise<{ score: number; label: string }[]> {
    if (!this.clip) throw new Error("zero-shot image classification needs a CLIP model");
    if (!labels.length) throw new Error("zero-shot image classification needs candidate labels");
    const template = options.hypothesis_template ?? "This is a photo of {}";
    const [image, texts] = await Promise.all([this.imageEmbed(img).then(normalize), this.textEmbeds(labels.map((l) => template.replace("{}", l)))]);
    const logits = new Float32Array(labels.length);
    texts.forEach((t, i) => {
      let d = 0;
      for (let j = 0; j < t.length; j++) d += t[j]! * image[j]!;
      logits[i] = this.clip!.scale * d;
    });
    return topLabels(logits, labels, 0).map(({ label, score }) => ({ score, label }));
  }

  /** Pixel values [3, size, size] of an image, as the model's image processor makes them. */
  pixels(img: RgbImage): Float32Array {
    const p = preprocessImage(img, this.processor);
    const S = this.cfg.imageSize;
    if (p.width !== S || p.height !== S) throw new Error(`the image processor made a ${p.width}×${p.height} image; the model takes ${S}×${S}`);
    return p.data;
  }

  /**
   * Object detection (YOLOS), post-processed as transformers.js does it: per detection token the
   * most likely class (skipped when it is "no object"), its softmax score against `threshold`, and
   * the box (center, size; sigmoid) turned into corners, in pixels of the original image (truncated
   * to integers) unless `percentage`.
   */
  async detect(img: RgbImage, options: ObjectDetectionOptions = {}): Promise<DetectedObject[]> {
    const threshold = options.threshold ?? 0.9;
    const p = preprocessImage(img, this.processor);
    const { logits, boxes, classes } = await this.model.detect(p.data, p.width, p.height);
    const res: DetectedObject[] = [];
    const sig = (x: number) => Math.fround(1 / (1 + Math.exp(-x)));
    for (let j = 0; j < this.cfg.detTokens; j++) {
      const l = logits.subarray(j * classes, (j + 1) * classes);
      let best = 0;
      for (let k = 1; k < classes; k++) if (l[k]! > l[best]!) best = k;
      if (best === classes - 1) continue;
      let z = 0;
      for (const v of l) z += Math.exp(v - l[best]!);
      const score = 1 / z;
      if (score < threshold) continue;
      const [cx, cy, w, h] = [0, 1, 2, 3].map((i) => sig(boxes[j * 4 + i]!)) as [number, number, number, number];
      let box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
      if (!options.percentage) box = box.map((v, i) => (v * (i % 2 ? img.height : img.width)) | 0);
      res.push({ score, label: this.cfg.labels[best] ?? `LABEL_${best}`, box: { xmin: box[0]!, ymin: box[1]!, xmax: box[2]!, ymax: box[3]! } });
    }
    return res;
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
    if (this.clip) {
      const e = await this.imageEmbed(img);
      return { data: e, dims: [1, e.length] };
    }
    const h = await this.model.forward(this.pixels(img));
    const H = this.cfg.hidden;
    if (!options.pool) return { data: h, dims: [1, this.model.tokens, H] };
    if (!this.heads.pooler) throw new Error("this model has no pooler; call without `pool`");
    return { data: applyHead(this.heads.pooler, h.subarray(0, H)).map(Math.tanh), dims: [1, H] };
  }

  destroy(): void {
    this.model.destroy();
    this.clip?.text.destroy();
  }
}

/** Loads a ViT, DeiT or CLIP checkpoint (safetensors or pytorch_model.bin) from the Hugging Face Hub. */
export async function createVision(modelId: string, opts: VisionLoadOptions = {}): Promise<VisionEngine> {
  const ref = parseModelId(modelId, opts.revision);
  if (ref.gguf) throw new Error("vision models load from safetensors or PyTorch repos, not GGUF files");
  const ep = opts.endpoint;
  const listing = listRepoFiles(ref, ep);
  const has = async (f: string) => {
    const files = await listing;
    return !files || files.has(f);
  };
  const [config, prep, names, tokJson] = await Promise.all([
    fetchJSON<Record<string, unknown>>(fileUrl(ref, "config.json", ep)),
    has("preprocessor_config.json").then((y) => (y ? fetchJSON<ImageProcessorConfig>(fileUrl(ref, "preprocessor_config.json", ep), true) : undefined)),
    listing.then((files) => weightFiles(ref, ep, files)),
    has("tokenizer.json").then((y) => (y ? fetchJSON<ClipTokenizerJSON>(fileUrl(ref, "tokenizer.json", ep), true) : undefined)),
  ]);
  const cfg = vitConfigFromHF(config!);
  const clip = cfg.modelType === "clip";
  if (clip && config!.model_type !== "clip") throw new Error(`${modelId}: a CLIP vision tower alone; zero-shot classification needs the full CLIP model`);
  if (clip && !tokJson) throw new Error(`${modelId}: no tokenizer.json`);
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
      const data = toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count);
      if (!t.entry.transposed) return { shape: t.entry.shape, data };
      // Stored [cols, rows]: transpose into the logical [rows, cols].
      const [rows, cols] = t.entry.shape as [number, number];
      const out = new Float32Array(count);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = data[c * rows + r]!;
      return { shape: t.entry.shape, data: out };
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
    const v = clip ? "vision_model.embeddings." : "embeddings.";
    const pw = await need(clip ? v + "patch_embedding.weight" : v + "patch_embeddings.projection.weight");
    const K = vitPatchK(cfg);
    const row = 3 * cfg.patchSize * cfg.patchSize;
    const patch = new Float32Array(H * K);
    for (let o = 0; o < H; o++) patch.set(pw.data.subarray(o * row, (o + 1) * row), o * K);
    upload("patch.weight", [H, K], patch);
    let embeddings: ((gh: number, gw: number) => VitEmbeddings) | undefined;
    if (cfg.detTokens) {
      embeddings = await yolosEmbeddings(cfg, need, read, v);
    } else {
      // Constant addend: position embeddings, plus the CLS / distillation tokens on the prefix rows and
      // the patch bias on the rest.
      const pos = await need(clip ? v + "position_embedding.weight" : v + "position_embeddings");
      const T = cfg.prefix + (cfg.imageSize / cfg.patchSize) ** 2;
      if (pos.data.length !== T * H) throw new Error(`${modelId}: position embeddings hold ${pos.data.length / H} tokens, the model has ${T}`);
      const add = Float32Array.from(pos.data);
      const prefixTokens = [await need(clip ? v + "class_embedding" : v + "cls_token")];
      if (cfg.prefix === 2) prefixTokens.push(await need(v + "distillation_token"));
      prefixTokens.forEach((t, r) => {
        for (let i = 0; i < H; i++) add[r * H + i] += t.data[i]!;
      });
      const pb = clip ? undefined : (await read(v + "patch_embeddings.projection.bias"))?.data;
      if (pb) for (let r = cfg.prefix; r < T; r++) for (let i = 0; i < H; i++) add[r * H + i] += pb[i]!;
      upload("embed.add", [T, H], add);
    }

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
    let parts: ClipParts | undefined;
    if (clip) {
      const tcfg = clipTextConfigFromHF(config!);
      const textWeights = await loadWeights(
        backend,
        files.map((f) => ({ ...f, tensors: located.size ? [...located.values()].filter((t) => t.source === f.source && clipTextName(t.entry.name) !== undefined).map((t) => t.entry) : [], rename: clipTextName })),
        opts.quant ?? "f32",
        opts.onProgress,
      );
      const [pw2, pb2, vp, tp, ls] = await Promise.all(["vision_model.post_layernorm.weight", "vision_model.post_layernorm.bias", "visual_projection.weight", "text_projection.weight", "logit_scale"].map(need));
      const proj = (t: { shape: number[]; data: Float32Array }): Head => ({ w: t.data, b: new Float32Array(t.shape[0]!), out: t.shape[0]! });
      parts = {
        text: new ClipTextModel(backend, tcfg, textWeights),
        tokenizer: new ClipTokenizer(tokJson!, tcfg.maxPositions),
        eosId: tcfg.eosId,
        post: { w: pw2!.data, b: pb2!.data, eps: cfg.eps },
        visual: proj(vp!),
        textProj: proj(tp!),
        scale: Math.exp(ls!.data[0]!),
        cache: new Map(),
      };
    }
    const model = new VitModel(backend, cfg, weights, embeddings);
    const processor: ImageProcessorConfig = prep ?? { size: { height: cfg.imageSize, width: cfg.imageSize } };
    return new VisionEngine(backend, cfg, model, processor, heads, parts);
  } finally {
    for (const s of sources) s.close?.();
  }
}

type ReadTensor = (name: string) => Promise<{ shape: number[]; data: Float32Array } | undefined>;

/**
 * YOLOS's addends for any patch grid: the patch part of the position embeddings (and of the mid
 * position embeddings, one set per layer but the last) fitted to the grid with torch's bicubic
 * interpolation, as transformers' InterpolateInitialPositionEmbeddings does; rows reordered to
 * Kiln's token order (detection tokens, CLS, patches) with the tokens and the patch bias added.
 */
async function yolosEmbeddings(cfg: VitConfig, need: (n: string) => Promise<{ shape: number[]; data: Float32Array }>, read: ReadTensor, v: string) {
  const H = cfg.hidden;
  const N = cfg.detTokens;
  const [gh0, gw0] = vitGrid(cfg, cfg.imageHW[1], cfg.imageHW[0]);
  const T0 = 1 + gh0 * gw0 + N;
  const pos = await need(v + "position_embeddings");
  if (pos.data.length !== T0 * H) throw new Error(`YOLOS: position embeddings hold ${pos.data.length / H} tokens, a ${cfg.imageHW.join(" × ")} image has ${T0}`);
  const cls = (await need(v + "cls_token")).data;
  const det = (await need(v + "detection_tokens")).data;
  const bias = (await read(v + "patch_embeddings.projection.bias"))?.data;
  const midAll = cfg.midPos ? (await need("encoder.mid_position_embeddings")).data : undefined;
  // One [T0, H] table → Kiln's rows for a grid: detection, CLS, then the patch grid resized.
  const fit = (table: Float32Array, gh: number, gw: number): Float32Array => {
    const out = new Float32Array((N + 1 + gh * gw) * H);
    out.set(table.subarray((T0 - N) * H, T0 * H), 0);
    out.set(table.subarray(0, H), N * H);
    const grid = new Float32Array(H * gh0 * gw0);
    for (let t = 0; t < gh0 * gw0; t++) for (let i = 0; i < H; i++) grid[i * gh0 * gw0 + t] = table[(1 + t) * H + i]!;
    const r = gh === gh0 && gw === gw0 ? grid : bicubicResize(grid, H, gh0, gw0, gh, gw);
    for (let t = 0; t < gh * gw; t++) for (let i = 0; i < H; i++) out[(N + 1 + t) * H + i] = r[i * gh * gw + t]!;
    return out;
  };
  return (gh: number, gw: number): VitEmbeddings => {
    const add = fit(pos.data, gh, gw);
    for (let i = 0; i < N * H; i++) add[i] += det[i]!;
    for (let i = 0; i < H; i++) add[N * H + i] += cls[i]!;
    if (bias) for (let t = N + 1; t < N + 1 + gh * gw; t++) for (let i = 0; i < H; i++) add[t * H + i] += bias[i]!;
    const mid = midAll ? Array.from({ length: cfg.layers - 1 }, (_, l) => fit(midAll.subarray(l * T0 * H, (l + 1) * T0 * H), gh, gw)) : undefined;
    return { add, mid };
  };
}
