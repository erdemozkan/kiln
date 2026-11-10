// Image models from transformers checkpoints (ViT, DeiT, DINOv2, CLIP, ResNet, ConvNeXt, YOLOS, DETR, Depth
// Anything, RMBG, SegFormer, Swin2SR):
// image classification, image features, zero-shot classification, object detection, panoptic
// segmentation, depth estimation, background removal, super-resolution and captioning, with transformers' image preprocessing
// (PIL-exact resizing).
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fileUrl, listRepoFiles, parseModelId, preprocessImage, resizeImage, readPytorchHeader, readSafetensorsHeader, toF32, weightFiles, type ByteSource, type ImageProcessorConfig, type RgbImage, type TensorEntry } from "@kiln/loader";
import { ClipTextModel, Seq2SeqModel, visionDecoderConfigFromHF, ConvNetModel, convNetConfigFromHF, convNetConvs, convNextName, type ConvNetConfig, SWIN2SR_MEAN, Swin2srModel, swin2srBias, swin2srConfigFromHF, DETR_IN_CHANNELS, DepthAnythingModel, ISNET_IN_CHANNELS, IsNetModel, SEGFORMER_IN_CHANNELS, SegformerModel, isnetConvs, segformerConfigFromHF, segformerName, type SegformerConfig, DetrModel, VitModel, bicubicResize, convMatrix, convTransposeMatrix, depthConfigFromHF, depthName, detrConfigFromHF, detrName, dinov2Positions, foldConvBn, clipPoolIndex, clipTextConfigFromHF, clipTextName, vitConfigFromHF, vitGrid, vitName, vitPatchK, type DepthAnythingConfig, type DetrConfig, type VitConfig, type VitEmbeddings } from "@kiln/models";
import { ClipTokenizer, encoderTokenizerFromJSON, type ClipTokenizerJSON, type EncoderTokenizer, type EncoderTokenizerJSON } from "@kiln/tokenizers";
import { openBackend, openParsed } from "./load";
import { loadWeights, type LoadProgress, type WeightFile } from "./weights";
import { createFlorence, type Florence2, type FlorenceOptions } from "./florence";
import { createOwl, type OwlDetector, type ZeroShotDetectionOptions } from "./owl";
import { createDonut, type DocumentQuestionAnsweringOptions, type DonutDocQA } from "./donut";

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

/** A depth map, as transformers.js's depth-estimation pipeline returns it. */
export interface DepthEstimate {
  /** Relative depth (larger is nearer) [height, width] at the original image size. */
  predicted_depth: { data: Float32Array; dims: number[] };
  /** The same map scaled to 0…255, one byte per pixel. */
  depth: { width: number; height: number; channels: 1; data: Uint8Array };
}

export interface ImageSegmentationOptions {
  /** Least class score a query needs to become a segment (default 0.5). */
  threshold?: number;
  /** Mask probability a pixel needs to count toward a segment's own area (default 0.5). */
  mask_threshold?: number;
  /** Share of that area a segment must keep after overlaps are resolved (default 0.8). */
  overlap_mask_area_threshold?: number;
}

/** A segment: its class, score, and mask (255 inside, 0 outside) at the image's size. A matting model's one segment has neither class nor score (null), as in transformers.js. */
export interface ImageSegment {
  score: number | null;
  label: string | null;
  mask: { width: number; height: number; channels: 1; data: Uint8Array };
}

export interface ImageToTextOptions extends Omit<FlorenceOptions, "max_new_tokens"> {
  /** Most tokens to generate (default: up to the model's max_length, 20, counting the start token; Florence-2: 1024; Donut: its decoder's positions). */
  max_new_tokens?: number;
  /** Donut: the decoder prompt (default its task token, such as "<s_cord-v2>"). */
  prompt?: string;
  /** Longest output counting the decoder start token (transformers' max_length; default 20). */
  max_length?: number;
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
    /** The vision transformer; DETR has none. */
    private readonly vit: { cfg: VitConfig; model: VitModel } | undefined,
    readonly processor: ImageProcessorConfig,
    private readonly heads: { cls?: Head; distill?: Head; pooler?: Head },
    private readonly clip?: ClipParts,
    private readonly detr?: { cfg: DetrConfig; model: DetrModel },
    private readonly depthModel?: { cfg: DepthAnythingConfig; model: DepthAnythingModel },
    private readonly matte?: IsNetModel,
    private readonly semantic?: { cfg: SegformerConfig; model: SegformerModel },
    private readonly sr?: Swin2srModel,
    private readonly convnet?: { cfg: ConvNetConfig; model: ConvNetModel; head: Head; norm?: { w: Float32Array; b: Float32Array; eps: number }; labels: string[] },
    private readonly captioner?: { decoder: Seq2SeqModel; tokenizer: EncoderTokenizer; maxLength: number },
    private readonly florence?: Florence2,
    private readonly owl?: OwlDetector,
    private readonly donut?: DonutDocQA,
  ) {}

  get cfg(): VitConfig {
    if (!this.vit) throw new Error("this model only detects objects");
    return this.vit.cfg;
  }

  get model(): VitModel {
    if (!this.vit) throw new Error("this model only detects objects");
    return this.vit.model;
  }

  /** Family, depth, width, input size and labels, for the model info. */
  summary(): { family: string; layers: number; hiddenSize: number; imageSize: number; labels: Record<number, string> } {
    const labels = (l: string[]) => Object.fromEntries(l.map((v, i) => [i, v]));
    if (this.matte) return { family: "rmbg", layers: 0, hiddenSize: 0, imageSize: 1024, labels: {} };
    if (this.donut) {
      const c = this.donut.cfg;
      return { family: "donut", layers: c.depths.reduce((a, b) => a + b, 0), hiddenSize: c.embedDim * 2 ** (c.depths.length - 1), imageSize: Math.max(...c.imageHW), labels: {} };
    }
    if (this.owl) {
      const c = this.owl.cfg;
      return { family: c.kind, layers: c.vision.layers + c.text.layers, hiddenSize: c.vision.hidden, imageSize: c.vision.imageSize, labels: {} };
    }
    if (this.florence) {
      const c = this.florence.cfg;
      return { family: "florence2", layers: c.vision.depths.reduce((a, b) => a + b, 0) * 2 + c.text.encLayers + c.text.decLayers, hiddenSize: c.text.dModel, imageSize: c.imageSize, labels: {} };
    }
    if (this.convnet) {
      const c = this.convnet.cfg;
      return { family: c.kind, layers: c.depths.reduce((a, b) => a + b, 0), hiddenSize: c.hidden[c.hidden.length - 1]!, imageSize: 224, labels: Object.fromEntries(this.convnet.labels.map((v, i) => [i, v])) };
    }
    if (this.sr) return { family: "swin2sr", layers: this.sr.cfg.depths.reduce((a, b) => a + b, 0), hiddenSize: this.sr.cfg.dim, imageSize: this.sr.cfg.gridSize, labels: {} };
    if (this.semantic) {
      const c = this.semantic.cfg;
      return { family: "segformer", layers: c.depths.reduce((a, b) => a + b, 0), hiddenSize: c.hidden[c.hidden.length - 1]!, imageSize: 512, labels: Object.fromEntries(c.labels.map((v, i) => [i, v])) };
    }
    if (this.depthModel) {
      const c = this.depthModel.cfg;
      return { family: "depth_anything", layers: c.layers, hiddenSize: c.hidden, imageSize: c.imageSize, labels: {} };
    }
    if (this.detr) {
      const c = this.detr.cfg;
      const size = this.processor.size as { shortest_edge?: number } | undefined;
      return { family: "detr", layers: c.encLayers + c.decLayers, hiddenSize: c.dModel, imageSize: size?.shortest_edge ?? 800, labels: labels(c.labels) };
    }
    const c = this.cfg;
    return { family: c.modelType, layers: c.layers, hiddenSize: c.hidden, imageSize: c.imageSize, labels: labels(c.labels) };
  }

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

  /** The vision transformer's last hidden state for an image; DINOv2 takes any size (its embeddings follow the patch grid). */
  private async hidden(img: RgbImage): Promise<Float32Array> {
    if (this.cfg.modelType !== "dinov2") return this.model.forward(this.pixels(img));
    const p = preprocessImage(img, this.processor);
    return this.model.forward(p.data, p.width, p.height);
  }

  /**
   * Object detection (YOLOS, DETR), post-processed as transformers.js does it: per detection token
   * or query the most likely class (skipped when it is "no object"), its softmax score against
   * `threshold`, and the box (center, size; sigmoid) turned into corners, in pixels of the original
   * image (truncated to integers) unless `percentage`.
   */
  async detect(img: RgbImage, options: ObjectDetectionOptions = {}): Promise<DetectedObject[]> {
    const threshold = options.threshold ?? 0.9;
    const p = preprocessImage(img, this.processor);
    const model = this.detr?.model ?? this.model;
    const labels = this.detr?.cfg.labels ?? this.cfg.labels;
    const { logits, boxes, classes } = await model.detect(p.data, p.width, p.height);
    const res: DetectedObject[] = [];
    const sig = (x: number) => Math.fround(1 / (1 + Math.exp(-x)));
    for (let j = 0; j < boxes.length / 4; j++) {
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
      res.push({ score, label: labels[best] ?? `LABEL_${best}`, box: { xmin: box[0]!, ymin: box[1]!, xmax: box[2]!, ymax: box[3]! } });
    }
    return res;
  }

  /**
   * Zero-shot object detection (OWL-ViT, OWLv2): boxes for any text labels (see OwlDetector.detect).
   * The image side is kept for the last few images, so new labels on the same image only run the
   * text tower.
   */
  async detectZeroShot(img: RgbImage, labels: readonly string[], options: ZeroShotDetectionOptions = {}): Promise<DetectedObject[]> {
    if (!this.owl) throw new Error("zero-shot object detection needs an OWL-ViT or OWLv2 model");
    return this.owl.detect(img, labels, options);
  }

  /**
   * Document question answering (Donut): the answer to a question about a page image, as
   * transformers' pipeline gives it ([{ answer }], null when the model gives none). The page's
   * encoding is kept, so more questions on it only run the decoder.
   */
  async answerDocument(img: RgbImage, question: string, options: DocumentQuestionAnsweringOptions = {}): Promise<{ answer: string | null }[]> {
    if (!this.donut) throw new Error("document question answering needs a Donut model (naver-clova-ix/donut-base-finetuned-docvqa)");
    return this.donut.answer(img, question, options);
  }

  /** The Donut model, for its parts. */
  get docQA(): DonutDocQA | undefined {
    return this.donut;
  }

  /** The OWL detector, for raw predictions. */
  get detector(): OwlDetector | undefined {
    return this.owl;
  }

  /**
   * Depth estimation (Depth Anything), post-processed as transformers.js does it: the predicted depth
   * resized bilinearly (pixel centers aligned) to the original image size, and that map scaled
   * from its minimum and maximum to 0…255 (truncated to bytes).
   */
  async estimateDepth(img: RgbImage): Promise<DepthEstimate> {
    if (!this.depthModel) throw new Error("this model does not estimate depth");
    const p = preprocessImage(img, this.processor);
    const d = await this.depthModel.model.predict(p.data, p.width, p.height);
    const [W, H] = [img.width, img.height];
    const out = new Float32Array(W * H);
    const f = Math.fround;
    const src = (i: number, inSize: number, outSize: number) => Math.max(f(f((i + 0.5) * f(inSize / outSize)) - 0.5), 0);
    for (let y = 0; y < H; y++) {
      const sy = src(y, d.height, H);
      const y0 = Math.min(Math.floor(sy), d.height - 1);
      const y1 = Math.min(y0 + 1, d.height - 1);
      const ly = sy - y0;
      for (let x = 0; x < W; x++) {
        const sx = src(x, d.width, W);
        const x0 = Math.min(Math.floor(sx), d.width - 1);
        const x1 = Math.min(x0 + 1, d.width - 1);
        const lx = sx - x0;
        const top = (1 - lx) * d.depth[y0 * d.width + x0]! + lx * d.depth[y0 * d.width + x1]!;
        const bot = (1 - lx) * d.depth[y1 * d.width + x0]! + lx * d.depth[y1 * d.width + x1]!;
        out[y * W + x] = (1 - ly) * top + ly * bot;
      }
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of out) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const range = f(hi - lo);
    const bytes = new Uint8Array(W * H);
    for (let i = 0; i < out.length; i++) bytes[i] = f(f(f(out[i]! - lo) / range) * 255);
    return { predicted_depth: { data: out, dims: [H, W] }, depth: { width: W, height: H, channels: 1, data: bytes } };
  }

  /**
   * Panoptic segmentation (DETR), post-processed as transformers.js does it: queries whose best class
   * is not "no object" and scores above `threshold`; their mask probabilities (sigmoid) resized
   * bilinearly to the image and weighted by the score; each pixel to the strongest query; a segment
   * kept when it holds more than `overlap_mask_area_threshold` of the pixels its own mask passes
   * `mask_threshold` on. Masks run only for the kept queries.
   */
  async segment(img: RgbImage, options: ImageSegmentationOptions = {}): Promise<ImageSegment[]> {
    if (this.matte) return [{ score: null, label: null, mask: await this.matteMask(img) }];
    if (this.semantic) {
      // Semantic (SegFormer), as transformers.js: scores resized to the image, each pixel's best
      // class, one mask per class present (in class order), no score.
      const p = preprocessImage(img, this.processor);
      const seg = await this.semantic.model.segment(p.data, p.width, p.height, img.width, img.height);
      const present = new Set(seg);
      return [...present].sort((a, b) => a - b).map((l) => {
        const data = new Uint8Array(seg.length);
        for (let i = 0; i < seg.length; i++) if (seg[i] === l) data[i] = 255;
        return { score: null, label: this.semantic!.cfg.labels[l] ?? `LABEL_${l}`, mask: { width: img.width, height: img.height, channels: 1 as const, data } };
      });
    }
    if (!this.detr) throw new Error("this model does not segment images");
    const threshold = options.threshold ?? 0.5;
    const maskThreshold = options.mask_threshold ?? 0.5;
    const overlap = options.overlap_mask_area_threshold ?? 0.8;
    const p = preprocessImage(img, this.processor);
    const scores = new Map<number, { score: number; label: number }>();
    const pick = (logits: Float32Array, classes: number) => {
      const picked: number[] = [];
      for (let q = 0; q < logits.length / classes; q++) {
        const l = logits.subarray(q * classes, (q + 1) * classes);
        let best = 0;
        for (let k = 1; k < classes; k++) if (l[k]! > l[best]!) best = k;
        if (best === classes - 1) continue;
        let z = 0;
        for (const v of l) z += Math.exp(v - l[best]!);
        const score = 1 / z;
        if (score > threshold) {
          picked.push(q);
          scores.set(q, { score, label: best });
        }
      }
      return picked;
    };
    const r = await this.detr.model.segment(p.data, p.width, p.height, pick);
    const [W, H] = [img.width, img.height];
    if (!r.picked.length) return [];
    // Each mask: sigmoid, bilinear to the image (transformers.js's interpolate), times its score.
    const probs = r.masks.map((m, i) => {
      const score = scores.get(r.picked[i]!)!.score;
      const sig = m.map((v) => 1 / (1 + Math.exp(-v)));
      const out = new Float32Array(W * H);
      const sx = W / r.maskW;
      const sy = H / r.maskH;
      for (let y = 0; y < H; y++) {
        const fy = (y + 0.5) / sy - 0.5;
        const y1 = Math.max(Math.floor(fy), 0);
        const y2 = Math.min(Math.floor(fy) + 1, r.maskH - 1);
        const t = fy - y1;
        for (let x = 0; x < W; x++) {
          const fx = (x + 0.5) / sx - 0.5;
          const x1 = Math.max(Math.floor(fx), 0);
          const x2 = Math.min(Math.floor(fx) + 1, r.maskW - 1);
          const s = fx - x1;
          const v = (1 - s) * (1 - t) * sig[y1 * r.maskW + x1]! + s * (1 - t) * sig[y1 * r.maskW + x2]! + (1 - s) * t * sig[y2 * r.maskW + x1]! + s * t * sig[y2 * r.maskW + x2]!;
          out[y * W + x] = Math.fround(Math.fround(v) * score);
        }
      }
      return out;
    });
    const owner = new Int32Array(W * H);
    const best = new Float32Array(W * H);
    probs.forEach((pr, i) => {
      for (let j = 0; j < pr.length; j++)
        if (pr[j]! > best[j]!) {
          owner[j] = i;
          best[j] = pr[j]!;
        }
    });
    const out: ImageSegment[] = [];
    probs.forEach((pr, i) => {
      let area = 0;
      let own = 0;
      for (let j = 0; j < pr.length; j++) {
        if (owner[j] === i) own++;
        if (pr[j]! >= maskThreshold) area++;
      }
      if (!own || !area || own / area <= overlap) return;
      const data = new Uint8Array(W * H);
      for (let j = 0; j < data.length; j++) if (owner[j] === i) data[j] = 255;
      const { score, label } = scores.get(r.picked[i]!)!;
      out.push({ score, label: this.detr!.cfg.labels[label] ?? `LABEL_${label}`, mask: { width: W, height: H, channels: 1, data } });
    });
    return out;
  }

  /**
   * The foreground matte (RMBG) as transformers.js makes it: sigmoid of the model's output at its input
   * size, times 255 truncated to bytes, then resized to the image (bilinear, PIL's filter; transformers.js
   * resizes on a canvas).
   */
  private async matteMask(img: RgbImage): Promise<{ width: number; height: number; channels: 1; data: Uint8Array }> {
    const p = preprocessImage(img, this.processor);
    const logits = await this.matte!.predict(p.data, p.width, p.height);
    const f = Math.fround;
    const grey = new Uint8Array(logits.length);
    for (let i = 0; i < logits.length; i++) grey[i] = Math.trunc(f(f(1 / (1 + Math.exp(-logits[i]!))) * 255));
    const data = p.width === img.width && p.height === img.height ? grey : Uint8Array.from(resizeImage({ width: p.width, height: p.height, data: grey }, img.width, img.height, 2).data);
    return { width: img.width, height: img.height, channels: 1, data };
  }

  /** Background removal (RMBG): the image as RGBA with the matte as its alpha, as transformers.js returns it. */
  async removeBackground(img: RgbImage): Promise<{ width: number; height: number; channels: 4; data: Uint8Array }> {
    if (!this.matte) throw new Error("this model does not remove backgrounds (use a matting model such as briaai/RMBG-1.4)");
    const mask = await this.matteMask(img);
    const n = img.width * img.height;
    const C = img.data.length / n;
    const data = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) {
      data[i * 4] = img.data[i * C]!;
      data[i * 4 + 1] = img.data[i * C + 1]!;
      data[i * 4 + 2] = img.data[i * C + 2]!;
      data[i * 4 + 3] = mask.data[i]!;
    }
    return { width: img.width, height: img.height, channels: 4, data };
  }

  /**
   * Super-resolution (Swin2SR), as transformers.js's image-to-image pipeline does it: pixels rescaled
   * to 0…1 and, only when a side is not a multiple of pad_size, padded at the bottom and right by
   * reflection (transformers.js's "symmetric", which does not repeat the edge pixel; Python pads
   * every image), then upscaled; the output (padding included) clamped to 0…1 and rounded to bytes.
   */
  async upscale(img: RgbImage): Promise<{ width: number; height: number; channels: 3; data: Uint8Array }> {
    const sr = this.sr;
    if (!sr) throw new Error("this model does not upscale images (use a Swin2SR model)");
    const p = preprocessImage(img, { ...this.processor, do_resize: false, do_center_crop: false, do_normalize: false });
    const size = (n: number, m: number) => n + ((m - (n % m)) % m);
    const pad = (this.processor as { pad_size?: number }).pad_size ?? 8;
    const [W, H] = [size(p.width, pad), size(p.height, pad)];
    // The model reflect-pads further to whole windows, then crops its output back to W × H.
    const [Wm, Hm] = [size(W, sr.cfg.window), size(H, sr.cfg.window)];
    // Index i of a side of length n extended by reflection (the edge pixel not repeated).
    const at = (i: number, n: number) => (i < n ? i : n === 1 ? 0 : Math.abs(((i + n - 1) % (2 * n - 2)) - (n - 1)));
    const f = Math.fround;
    const x = new Float32Array(3 * Hm * Wm);
    for (let c = 0; c < 3; c++)
      for (let y = 0; y < Hm; y++)
        for (let xx = 0; xx < Wm; xx++) {
          // Model padding over the processor's, which is over the image.
          const v = p.data[(c * p.height + at(at(y, H), p.height)) * p.width + at(at(xx, W), p.width)]!;
          x[(c * Hm + y) * Wm + xx] = f(f(v - f(SWIN2SR_MEAN[c]!)) * sr.cfg.imgRange);
        }
    const up = sr.cfg.upscale;
    const out = await sr.upscale(x, Wm, Hm);
    const [ow, oh] = [W * up, H * up];
    const data = new Uint8Array(ow * oh * 3);
    for (let c = 0; c < 3; c++)
      for (let y = 0; y < oh; y++)
        for (let xx = 0; xx < ow; xx++) {
          const v = f(f(out[(c * Hm * up + y) * Wm * up + xx]! / sr.cfg.imgRange) + f(SWIN2SR_MEAN[c]!));
          data[(y * ow + xx) * 3 + c] = Math.round(f(Math.min(Math.max(v, 0), 1) * 255));
        }
    return { width: ow, height: oh, channels: 3, data };
  }

  /**
   * Image captioning (VisionEncoderDecoder: ViT + GPT-2), as transformers.js's image-to-text pipeline
   * does it: the image's last hidden state as the encoder output, greedy decoding from the decoder
   * start token until end-of-sequence or max_length (20 by default, the start token included),
   * special tokens dropped and the text trimmed.
   */
  async caption(img: RgbImage, options: ImageToTextOptions = {}): Promise<{ generated_text: string }[]> {
    if (this.florence) return this.florence.run(img, options);
    if (this.donut) return this.donut.parse(img, { prompt: options.prompt, max_new_tokens: options.max_new_tokens });
    const cap = this.captioner;
    if (!cap) throw new Error("this model does not caption images");
    const h = await this.model.forward(this.pixels(img));
    const dec = cap.decoder;
    await dec.encodeHidden(h, h.length / dec.cfg.dModel);
    const limit = Math.min(options.max_new_tokens ?? (options.max_length ?? cap.maxLength) - 1, dec.maxTarget - 1);
    const ids: number[] = [];
    let tok = dec.cfg.decoderStart;
    for (let t = 0; t < limit; t++) {
      tok = await dec.next(tok, t);
      if (tok === dec.cfg.eos) break;
      ids.push(tok);
    }
    return [{ generated_text: cap.tokenizer.decode(ids, true).trim() }];
  }

  /**
   * Class logits of an image (DeiT with a teacher: the mean of the CLS and distillation heads; DINOv2:
   * the head over [CLS, mean of the patch tokens]; ResNet: over the mean of the last feature map;
   * ConvNeXt: over its LayerNorm).
   */
  async logits(img: RgbImage): Promise<Float32Array> {
    if (this.convnet) {
      const c = this.convnet;
      const p = preprocessImage(img, this.processor);
      const f = await c.model.features(p.data, p.width, p.height);
      const mean = new Float32Array(f.channels);
      for (let ch = 0; ch < f.channels; ch++) {
        let s = 0;
        for (let r = 0; r < f.rows; r++) s += f.data[r * f.channels + ch]!;
        mean[ch] = s / f.rows;
      }
      return applyHead(c.head, c.norm ? layerNorm(mean, c.norm.w, c.norm.b, c.norm.eps) : mean);
    }
    if (!this.heads.cls) throw new Error("this model has no classification head");
    const h = await this.hidden(img);
    const H = this.cfg.hidden;
    if (this.cfg.modelType === "dinov2") {
      const x = new Float32Array(2 * H);
      x.set(h.subarray(0, H));
      const T = h.length / H - 1;
      for (let i = 0; i < H; i++) {
        let s = 0;
        for (let t = 1; t <= T; t++) s += h[t * H + i]!;
        x[H + i] = s / T;
      }
      return applyHead(this.heads.cls, x);
    }
    const y = applyHead(this.heads.cls, h.subarray(0, H));
    if (this.heads.distill) {
      const d = applyHead(this.heads.distill, h.subarray(H, 2 * H));
      for (let i = 0; i < y.length; i++) y[i] = (y[i]! + d[i]!) / 2;
    }
    return y;
  }

  async classify(img: RgbImage, options: ImageClassificationOptions = {}): Promise<{ label: string; score: number }[]> {
    return topLabels(await this.logits(img), this.convnet?.labels ?? this.cfg.labels, options.top_k ?? 5);
  }


  /** The last hidden state [tokens, hidden], or with `pool` the pooled output [1, hidden]. */
  async features(img: RgbImage, options: ImageFeatureOptions = {}): Promise<{ data: Float32Array; dims: number[] }> {
    if (this.clip) {
      const e = await this.imageEmbed(img);
      return { data: e, dims: [1, e.length] };
    }
    const h = await this.hidden(img);
    const H = this.cfg.hidden;
    if (!options.pool) return { data: h, dims: [1, h.length / H, H] };
    if (!this.heads.pooler) throw new Error("this model has no pooler; call without `pool`");
    return { data: applyHead(this.heads.pooler, h.subarray(0, H)).map(Math.tanh), dims: [1, H] };
  }

  destroy(): void {
    this.donut?.destroy();
    this.owl?.destroy();
    this.florence?.destroy();
    this.vit?.model.destroy();
    this.detr?.model.destroy();
    this.depthModel?.model.destroy();
    this.matte?.destroy();
    this.semantic?.model.destroy();
    this.sr?.destroy();
    this.convnet?.model.destroy();
    this.captioner?.decoder.destroy();
    this.clip?.text.destroy();
  }
}

/** Loads a ViT, DeiT, CLIP, YOLOS, DETR, OWL-ViT or OWLv2 checkpoint (safetensors or pytorch_model.bin) from the Hugging Face Hub. */
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
  if (config!.model_type === "detr") return createDetr(modelId, ref, config!, prep, names, opts);
  if (config!.model_type === "depth_anything") return createDepth(modelId, ref, config!, prep, names, opts);
  if (((config!.architectures as string[] | undefined) ?? []).includes("BriaRMBG")) return createRmbg(modelId, ref, prep, names, opts);
  if (config!.model_type === "segformer") return createSegformer(modelId, ref, config!, prep, names, opts);
  if (config!.model_type === "swin2sr") return createSwin2sr(modelId, ref, config!, prep, names, opts);
  if (config!.model_type === "resnet" || config!.model_type === "convnext") return createConvNet(modelId, ref, config!, prep, names, opts);
  if (config!.model_type === "vision-encoder-decoder" && (config!.encoder as { model_type?: string } | undefined)?.model_type === "donut-swin") {
    return createDonut(modelId, ref, config!, prep, names, opts, listing, (d, backend) => new VisionEngine(backend, undefined, d.processor, {}, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, d));
  }
  if (config!.model_type === "vision-encoder-decoder") return createCaptioner(modelId, ref, config!, prep, names, opts, listing);
  if (config!.model_type === "owlv2" || config!.model_type === "owlvit") {
    return createOwl(modelId, ref, config!, prep, names, opts, listing, (o, backend) => new VisionEngine(backend, undefined, o.processor, {}, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, o));
  }
  if (config!.model_type === "florence2") {
    return createFlorence(modelId, ref, config!, prep, names, opts, (f, backend) => new VisionEngine(backend, undefined, f.processor, {}, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, f));
  }
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
      for (const t of header.tensors) located.set(t.name.replace(/^(vit|deit|dinov2)\./, ""), { source, entry: t });
      // DINOv2's output projections are loaded below, scaled by the layer scales.
      const rename = cfg.modelType === "dinov2" ? (n: string) => (/attention\.output\.dense|mlp\.fc2/.test(n) ? undefined : vitName(n)) : vitName;
      files.push({ name, source, tensors: header.tensors.filter((t) => vitName(t.name) !== undefined), rename });
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
    } else if (cfg.modelType === "dinov2") {
      // Layer scale γ folded into the projection before it: γ ⊙ (W x + b).
      for (let l = 0; l < cfg.layers; l++) {
        for (const [from, gamma, to] of [["attention.output.dense", "layer_scale1.lambda1", "o"], ["mlp.fc2", "layer_scale2.lambda1", "down"]] as const) {
          const [w, b, g] = await Promise.all([need(`encoder.layer.${l}.${from}.weight`), need(`encoder.layer.${l}.${from}.bias`), need(`encoder.layer.${l}.${gamma}`)]);
          const [rows, cols] = w.shape as [number, number];
          for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) w.data[r * cols + c] = w.data[r * cols + c]! * g.data[r]!;
            b.data[r] = b.data[r]! * g.data[r]!;
          }
          upload(`l${l}.${to}.weight`, [rows, cols], w.data);
          upload(`l${l}.${to}.bias`, [rows], b.data);
        }
      }
      // CLS first, then the patch grid's position embeddings fitted to the image (bicubic) plus the patch bias.
      const pos = await need(v + "position_embeddings");
      const cls = (await need(v + "cls_token")).data;
      const pb = (await need(v + "patch_embeddings.projection.bias")).data;
      const n0 = Math.round(Math.sqrt(pos.data.length / H - 1));
      const grid = pos.data.subarray(H);
      embeddings = (gh, gw) => {
        const add = new Float32Array((1 + gh * gw) * H);
        for (let i = 0; i < H; i++) add[i] = cls[i]! + pos.data[i]!;
        add.set(gh === n0 && gw === n0 ? grid : dinov2Positions(grid, H, n0, gh, gw), H);
        for (let t = 1; t <= gh * gw; t++) for (let i = 0; i < H; i++) add[t * H + i] += pb[i]!;
        return { add };
      };
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
    return new VisionEngine(backend, { cfg, model }, processor, heads, parts);
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

/**
 * DETR: the transformer and heads load as weights; the backbone's convolutions are read, their
 * frozen batch norms folded in (eps 1e-5, as DetrFrozenBatchNorm2d) and uploaded as the conv kernel
 * takes them; the 1×1 input projection becomes a matrix.
 */
async function createDetr(
  modelId: string,
  ref: ReturnType<typeof parseModelId>,
  config: Record<string, unknown>,
  prep: ImageProcessorConfig | undefined,
  names: Awaited<ReturnType<typeof weightFiles>>,
  opts: VisionLoadOptions,
): Promise<VisionEngine> {
  const cfg = detrConfigFromHF(config);
  if (!prep) throw new Error(`${modelId}: no preprocessor_config.json`);
  const ep = opts.endpoint;
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const files: WeightFile[] = [];
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      // Panoptic checkpoints wrap the detector as "detr.".
      for (const t of header.tensors) located.set(t.name.replace(/^detr\./, ""), { source, entry: t });
      files.push({ name, source, tensors: header.tensors.filter((t) => detrName(t.name) !== undefined), rename: detrName });
    }
    const need = async (name: string) => {
      const t = located.get(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
    };
    const weights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `detr/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    const bb = "model.backbone.conv_encoder.model.";
    const convBn = async (conv: string, bn: string, to: string, cp?: number) => {
      const w = await need(`${bb}${conv}.weight`);
      const [weight, bias, mean, variance] = await Promise.all(["weight", "bias", "running_mean", "running_var"].map((k) => need(`${bb}${bn}.${k}`)));
      const f = foldConvBn(w.data, w.shape, { weight: weight.data, bias: bias.data, mean: mean.data, var: variance.data }, cp);
      upload(`${to}.weight`, [w.shape[0]!, f.K], f.w);
      upload(`${to}.bias`, [w.shape[0]!], f.b);
    };
    await convBn("conv1", "bn1", "bb.conv1", DETR_IN_CHANNELS);
    for (let s = 0; s < cfg.stages.length; s++) {
      for (let b = 0; b < cfg.stages[s]!; b++) {
        const p = `layer${s + 1}.${b}`;
        for (const i of [1, 2, 3]) await convBn(`${p}.conv${i}`, `${p}.bn${i}`, `bb.${p}.conv${i}`);
        if (b === 0) await convBn(`${p}.downsample.0`, `${p}.downsample.1`, `bb.${p}.down`);
      }
    }
    const proj = await need("model.input_projection.weight");
    upload("proj.weight", [proj.shape[0]!, proj.shape[1]!], proj.data);
    upload("proj.bias", [proj.shape[0]!], (await need("model.input_projection.bias")).data);
    const qpos = await need("model.query_position_embeddings.weight");
    upload("query.pos", qpos.shape, qpos.data);
    if (located.has("mask_head.lay1.weight")) {
      const mh = "mask_head.";
      const conv = async (from: string, to: string) => {
        const w = await need(`${mh}${from}.weight`);
        const m = convMatrix(w.data, w.shape);
        upload(`${to}.weight`, [w.shape[0]!, m.K], m.w);
        upload(`${to}.bias`, [w.shape[0]!], (await need(`${mh}${from}.bias`)).data);
      };
      // lay1 reads [projected features (dModel) | box attention (heads)]: split along its input channels.
      const l1 = await need(mh + "lay1.weight");
      const [O, Cin, k] = l1.shape as [number, number, number];
      const D = cfg.dModel;
      const part = (c0: number, c1: number) => {
        const out = new Float32Array(O * (c1 - c0) * k * k);
        for (let o = 0; o < O; o++) out.set(l1.data.subarray((o * Cin + c0) * k * k, (o * Cin + c1) * k * k), o * (c1 - c0) * k * k);
        return convMatrix(out, [O, c1 - c0, k, k]);
      };
      const a = part(0, D);
      const b = part(D, Cin);
      upload("mask.lay1a.weight", [O, a.K], a.w);
      upload("mask.lay1a.bias", [O], (await need(mh + "lay1.bias")).data);
      upload("mask.lay1b.weight", [O, b.K], b.w);
      upload("mask.lay1b.bias", [O], new Float32Array(O));
      for (const i of [2, 3, 4, 5]) await conv(`lay${i}`, `mask.lay${i}`);
      for (const i of [1, 2, 3]) await conv(`adapter${i}`, `mask.adapter${i}`);
      await conv("out_lay", "mask.out");
      for (const i of [1, 2, 3, 4, 5]) for (const f of ["weight", "bias"]) {
        const t = await need(`${mh}gn${i}.${f}`);
        upload(`mask.gn${i}.${f}`, t.shape, t.data);
      }
    }
    return new VisionEngine(backend, undefined, prep, {}, undefined, { cfg, model: new DetrModel(backend, cfg, weights) });
  } finally {
    for (const s of sources) s.close?.();
  }
}

/**
 * Depth Anything: DINOv2's attention, MLP-in and norms load as weights; the patch convolution
 * becomes a matrix, the layer scales are folded into the attention output and MLP-out projections,
 * the position embeddings are fitted to each grid, and the neck and head convolutions are reshaped
 * for the conv kernels.
 */
async function createDepth(
  modelId: string,
  ref: ReturnType<typeof parseModelId>,
  config: Record<string, unknown>,
  prep: ImageProcessorConfig | undefined,
  names: Awaited<ReturnType<typeof weightFiles>>,
  opts: VisionLoadOptions,
): Promise<VisionEngine> {
  const cfg = depthConfigFromHF(config);
  if (!prep) throw new Error(`${modelId}: no preprocessor_config.json`);
  const ep = opts.endpoint;
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const files: WeightFile[] = [];
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
      files.push({ name, source, tensors: header.tensors.filter((t) => depthName(t.name) !== undefined), rename: depthName });
    }
    const need = async (name: string) => {
      const t = located.get(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
    };
    const weights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `depth/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    const H = cfg.hidden;
    const e = "backbone.embeddings.";
    const pw = await need(e + "patch_embeddings.projection.weight");
    const K = Math.ceil((3 * cfg.patchSize ** 2) / 32) * 32;
    const row = 3 * cfg.patchSize ** 2;
    const patch = new Float32Array(H * K);
    for (let o = 0; o < H; o++) patch.set(pw.data.subarray(o * row, (o + 1) * row), o * K);
    upload("patch.weight", [H, K], patch);
    // Layer scale γ folded into the projection before it: γ ⊙ (W x + b).
    const scaled = async (from: string, gamma: string, to: string) => {
      const [w, b, g] = await Promise.all([need(`${from}.weight`), need(`${from}.bias`), need(gamma)]);
      const [rows, cols] = w.shape as [number, number];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) w.data[r * cols + c] = w.data[r * cols + c]! * g.data[r]!;
        b.data[r] = b.data[r]! * g.data[r]!;
      }
      upload(`${to}.weight`, [rows, cols], w.data);
      upload(`${to}.bias`, [rows], b.data);
    };
    for (let l = 0; l < cfg.layers; l++) {
      const p = `backbone.encoder.layer.${l}.`;
      await scaled(p + "attention.output.dense", p + "layer_scale1.lambda1", `l${l}.o`);
      await scaled(p + "mlp.fc2", p + "layer_scale2.lambda1", `l${l}.down`);
    }
    const conv = async (from: string, to: string, bias = true) => {
      const w = await need(`${from}.weight`);
      const m = convMatrix(w.data, w.shape);
      upload(`${to}.weight`, [w.shape[0]!, m.K], m.w);
      upload(`${to}.bias`, [w.shape[0]!], bias ? (await need(`${from}.bias`)).data : new Float32Array(w.shape[0]!));
    };
    const n = "neck.";
    for (let i = 0; i < cfg.neck.length; i++) {
      const r = `${n}reassemble_stage.layers.${i}.`;
      await conv(r + "projection", `neck.re${i}.proj`);
      const f = cfg.factors[i]!;
      if (f > 1) {
        const w = await need(r + "resize.weight");
        upload(`neck.re${i}.resize.weight`, [f * f, w.shape[1]!, w.shape[0]!], convTransposeMatrix(w.data, w.shape));
        upload(`neck.re${i}.resize.bias`, [w.shape[1]!], (await need(r + "resize.bias")).data);
      } else if (f < 1) await conv(r + "resize", `neck.re${i}.resize`);
      await conv(`${n}convs.${i}`, `neck.conv${i}`, false);
      const u = `${n}fusion_stage.layers.${i}.`;
      for (const k of [1, 2]) for (const c of [1, 2]) await conv(`${u}residual_layer${k}.convolution${c}`, `neck.fu${i}.res${k}.c${c}`);
      await conv(u + "projection", `neck.fu${i}.proj`);
    }
    for (const k of [1, 2, 3]) await conv(`head.conv${k}`, `head.conv${k}`);

    const pos = await need(e + "position_embeddings");
    const cls = (await need(e + "cls_token")).data;
    const pb = (await need(e + "patch_embeddings.projection.bias")).data;
    const n0 = Math.round(Math.sqrt(pos.data.length / H - 1));
    const grid = pos.data.subarray(H);
    const addend = (gh: number, gw: number) => {
      const P = gh * gw;
      const out = new Float32Array((P + 1) * H);
      out.set(gh === n0 && gw === n0 ? grid : dinov2Positions(grid, H, n0, gh, gw), 0);
      for (let t = 0; t < P; t++) for (let i = 0; i < H; i++) out[t * H + i] += pb[i]!;
      for (let i = 0; i < H; i++) out[P * H + i] = cls[i]! + pos.data[i]!;
      return out;
    };
    const model = new DepthAnythingModel(backend, cfg, weights, addend);
    return new VisionEngine(backend, undefined, prep, {}, undefined, undefined, { cfg, model });
  } finally {
    for (const s of sources) s.close?.();
  }
}

/**
 * BRIA RMBG-1.4: every convolution read, its batch norm folded in (eps 1e-5) and its weight split
 * along the input channels where briarmbg.py concatenates (isnetConvs lists the splits).
 */
async function createRmbg(
  modelId: string,
  ref: ReturnType<typeof parseModelId>,
  prep: ImageProcessorConfig | undefined,
  names: Awaited<ReturnType<typeof weightFiles>>,
  opts: VisionLoadOptions,
): Promise<VisionEngine> {
  if (!prep) throw new Error(`${modelId}: no preprocessor_config.json`);
  const ep = opts.endpoint;
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
    }
    const need = async (name: string) => {
      const t = located.get(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
    };
    const weights = new Map<string, Tensor>();
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `rmbg/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    const convs = [...isnetConvs()];
    let done = 0;
    for (const [path, { splits, bn }] of convs) {
      const base = bn ? `${path}.conv_s1` : path;
      const [w, b] = await Promise.all([need(`${base}.weight`), need(`${base}.bias`)]);
      const [O, Cin, k] = w.shape as [number, number, number];
      if (Cin !== splits.reduce((a, c) => a + c, 0)) throw new Error(`${modelId}: ${base}.weight has ${Cin} input channels, expected ${splits.join(" + ")}`);
      const bias = Float32Array.from(b.data);
      if (bn) {
        const [g, beta, mean, variance] = await Promise.all(["weight", "bias", "running_mean", "running_var"].map((f) => need(`${path}.bn_s1.${f}`)));
        for (let o = 0; o < O; o++) {
          const s = Math.fround(g.data[o]! / Math.sqrt(variance.data[o]! + 1e-5));
          bias[o] = (b.data[o]! - mean.data[o]!) * s + beta.data[o]!;
          for (let i = o * Cin * k * k; i < (o + 1) * Cin * k * k; i++) w.data[i] = w.data[i]! * s;
        }
      }
      let c0 = 0;
      splits.forEach((ci, i) => {
        const part = new Float32Array(O * ci * k * k);
        for (let o = 0; o < O; o++) part.set(w.data.subarray((o * Cin + c0) * k * k, (o * Cin + c0 + ci) * k * k), o * ci * k * k);
        const m = convMatrix(part, [O, ci, k, k], path === "conv_in" ? ISNET_IN_CHANNELS : ci);
        upload(`${path}.p${i}.weight`, [O, m.K], m.w);
        upload(`${path}.p${i}.bias`, [O], i === 0 ? bias : new Float32Array(O));
        c0 += ci;
      });
      opts.onProgress?.({ file: names.names[0]!, loaded: ++done, total: convs.length });
    }
    return new VisionEngine(backend, undefined, prep, {}, undefined, undefined, undefined, new IsNetModel(backend, weights));
  } finally {
    for (const s of sources) s.close?.();
  }
}

/**
 * SegFormer: matrices, biases and norms load as weights; the patch, sequence-reduction and depthwise
 * convolutions are reshaped for their kernels; the decode head's fuse convolution is split by stage
 * with its batch norm (eps 1e-5) folded in, and the classifier becomes a matrix.
 */
async function createSegformer(
  modelId: string,
  ref: ReturnType<typeof parseModelId>,
  config: Record<string, unknown>,
  prep: ImageProcessorConfig | undefined,
  names: Awaited<ReturnType<typeof weightFiles>>,
  opts: VisionLoadOptions,
): Promise<VisionEngine> {
  const cfg = segformerConfigFromHF(config);
  if (!prep) throw new Error(`${modelId}: no preprocessor_config.json`);
  const ep = opts.endpoint;
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const files: WeightFile[] = [];
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
      files.push({ name, source, tensors: header.tensors.filter((t) => segformerName(t.name) !== undefined), rename: segformerName });
    }
    const need = async (name: string) => {
      const t = located.get(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
    };
    const weights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `segformer/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    const conv = async (from: string, to: string, cp?: number) => {
      const w = await need(`${from}.weight`);
      const m = convMatrix(w.data, w.shape, cp);
      upload(`${to}.weight`, [w.shape[0]!, m.K], m.w);
      upload(`${to}.bias`, [w.shape[0]!], (await need(`${from}.bias`)).data);
    };
    const e = "segformer.encoder.";
    for (let s = 0; s < cfg.hidden.length; s++) {
      await conv(`${e}patch_embeddings.${s}.proj`, `s${s}.patch`, s === 0 ? SEGFORMER_IN_CHANNELS : undefined);
      for (let j = 0; j < cfg.depths[s]!; j++) {
        const b = `${e}block.${s}.${j}.`;
        if (cfg.srRatios[s]! > 1) await conv(b + "attention.self.sr", `b${s}.${j}.sr`);
        const dw = await need(b + "mlp.dwconv.dwconv.weight");
        const C = dw.shape[0]!;
        const w9 = new Float32Array(9 * C);
        for (let c = 0; c < C; c++) for (let t = 0; t < 9; t++) w9[t * C + c] = dw.data[c * 9 + t]!;
        upload(`b${s}.${j}.dw.weight`, [9, C], w9);
        upload(`b${s}.${j}.dw.bias`, [C], (await need(b + "mlp.dwconv.dwconv.bias")).data);
      }
    }
    // linear_fuse over the concatenation [stage n-1 … stage 0] of decoder-width maps, BN folded.
    const d = "decode_head.";
    const fuse = await need(d + "linear_fuse.weight");
    const [O, Cin] = fuse.shape as [number, number];
    const [g, beta, mean, variance] = await Promise.all(["weight", "bias", "running_mean", "running_var"].map((f) => need(`${d}batch_norm.${f}`)));
    const parts = cfg.hidden.length;
    const Cp = Cin / parts;
    const bias = new Float32Array(O);
    const scale = new Float32Array(O);
    for (let o = 0; o < O; o++) {
      scale[o] = Math.fround(g.data[o]! / Math.sqrt(variance.data[o]! + 1e-5));
      bias[o] = beta.data[o]! - mean.data[o]! * scale[o]!;
    }
    for (let k = 0; k < parts; k++) {
      const w = new Float32Array(O * Cp);
      for (let o = 0; o < O; o++) for (let c = 0; c < Cp; c++) w[o * Cp + c] = fuse.data[o * Cin + k * Cp + c]! * scale[o]!;
      upload(`dec.fuse.p${k}.weight`, [O, Cp], w);
      upload(`dec.fuse.p${k}.bias`, [O], k === 0 ? bias : new Float32Array(O));
    }
    const cls = await need(d + "classifier.weight");
    upload("dec.cls.weight", [cls.shape[0]!, cls.shape[1]!], cls.data);
    upload("dec.cls.bias", [cls.shape[0]!], (await need(d + "classifier.bias")).data);
    const model = new SegformerModel(backend, cfg, weights);
    return new VisionEngine(backend, undefined, prep, {}, undefined, undefined, undefined, undefined, { cfg, model });
  } finally {
    for (const s of sources) s.close?.();
  }
}

/**
 * Swin2SR: every weight read and padded from the model's width to the next multiple of 32 (zero rows
 * and columns, LayerNorm weights and biases zero past the width); each layer's attention bias table
 * computed from its continuous-position MLP, and its logit scale clamped and exponentiated.
 */
async function createSwin2sr(
  modelId: string,
  ref: ReturnType<typeof parseModelId>,
  config: Record<string, unknown>,
  prep: ImageProcessorConfig | undefined,
  names: Awaited<ReturnType<typeof weightFiles>>,
  opts: VisionLoadOptions,
): Promise<VisionEngine> {
  const cfg = swin2srConfigFromHF(config);
  const ep = opts.endpoint;
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
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
    const weights = new Map<string, Tensor>();
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `swin2sr/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    const pad32 = (n: number) => Math.ceil(n / 32) * 32;
    const vec = (v: Float32Array, n: number) => {
      const out = new Float32Array(n);
      out.set(v);
      return out;
    };
    const linear = async (from: string, to: string) => {
      const w = await need(`${from}.weight`);
      const [N, K] = w.shape as [number, number];
      const [Np, Kp] = [pad32(N), pad32(K)];
      const m = new Float32Array(Np * Kp);
      for (let r = 0; r < N; r++) m.set(w.data.subarray(r * K, (r + 1) * K), r * Kp);
      upload(`${to}.weight`, [Np, Kp], m);
      const b = await read(`${from}.bias`);
      if (b) upload(`${to}.bias`, [Np], vec(b.data, Np));
    };
    const norm = async (from: string, to: string) => {
      for (const f of ["weight", "bias"]) upload(`${to}.${f}`, [pad32(cfg.dim)], vec((await need(`${from}.${f}`)).data, pad32(cfg.dim)));
    };
    const conv = async (from: string, to: string, cin?: number) => {
      const w = await need(`${from}.weight`);
      const O = w.shape[0]!;
      const Op = O > 4 ? pad32(O) : O;
      const m = convMatrix(w.data, w.shape, cin ?? pad32(w.shape[1]!));
      upload(`${to}.weight`, [Op, m.K], vec(m.w, Op * m.K));
      upload(`${to}.bias`, [Op], vec((await need(`${from}.bias`)).data, Op));
    };
    const m = "swin2sr.";
    await conv(m + "first_convolution", "first", 4);
    await conv(m + "embeddings.patch_embeddings.projection", "patch");
    await norm(m + "embeddings.patch_embeddings.layernorm", "patchLn");
    let done = 0;
    const total = cfg.depths.reduce((a, b) => a + b, 0);
    for (let s = 0; s < cfg.depths.length; s++) {
      const st = `${m}encoder.stages.${s}.`;
      for (let l = 0; l < cfg.depths[s]!; l++) {
        const L = `${st}layers.${l}.`;
        const p = `s${s}.${l}`;
        await linear(L + "attention.self.query", `${p}.q`);
        await linear(L + "attention.self.key", `${p}.k`);
        await linear(L + "attention.self.value", `${p}.v`);
        await linear(L + "attention.output.dense", `${p}.o`);
        await linear(L + "intermediate.dense", `${p}.up`);
        await linear(L + "output.dense", `${p}.down`);
        await norm(L + "layernorm_before", `${p}.ln1`);
        await norm(L + "layernorm_after", `${p}.ln2`);
        const heads = cfg.heads[s]!;
        const ls = (await need(L + "attention.self.logit_scale")).data;
        upload(`${p}.scale`, [heads], Float32Array.from(ls, (v) => Math.exp(Math.min(v, Math.log(100)))));
        const [w1, b1, w2] = await Promise.all(["0.weight", "0.bias", "2.weight"].map((f) => need(`${L}attention.self.continuous_position_bias_mlp.${f}`)));
        const N = cfg.window ** 2;
        upload(`${p}.bias`, [heads * N * N], swin2srBias(w1.data, b1.data, w2.data, heads, cfg.window));
        opts.onProgress?.({ file: names.names[0]!, loaded: ++done, total });
      }
      await conv(st + "conv", `s${s}.conv`);
      await conv(st + "patch_embed.projection", `s${s}.proj`);
    }
    await norm(m + "layernorm", "lnF");
    await conv(m + "conv_after_body", "afterBody");
    await conv("upsample.conv_before_upsample", "beforeUp");
    for (let i = 0; 2 ** i < cfg.upscale; i++) await conv(`upsample.upsample.convolution_${i}`, `up${i}`);
    await conv("upsample.final_convolution", "final");
    const processor: ImageProcessorConfig = prep ?? {};
    return new VisionEngine(backend, undefined, processor, {}, undefined, undefined, undefined, undefined, undefined, new Swin2srModel(backend, cfg, weights));
  } finally {
    for (const s of sources) s.close?.();
  }
}

/**
 * ResNet / ConvNeXt classifiers: ResNet's convolutions with their batch norms folded in (eps 1e-5),
 * ConvNeXt's patchify, downsampling and depthwise convolutions reshaped for their kernels, its MLP
 * matrices loaded as weights with the layer scale folded into the second; the classifier (and
 * ConvNeXt's final LayerNorm) applied on the CPU to the pooled features.
 */
async function createConvNet(
  modelId: string,
  ref: ReturnType<typeof parseModelId>,
  config: Record<string, unknown>,
  prep: ImageProcessorConfig | undefined,
  names: Awaited<ReturnType<typeof weightFiles>>,
  opts: VisionLoadOptions,
): Promise<VisionEngine> {
  const cfg = convNetConfigFromHF(config);
  if (!prep) throw new Error(`${modelId}: no preprocessor_config.json`);
  const ep = opts.endpoint;
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const files: WeightFile[] = [];
    const rename = (n: string) => (cfg.kind === "convnext" && !/pwconv2/.test(n) ? convNextName(n) : undefined);
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
      files.push({ name, source, tensors: header.tensors.filter((t) => rename(t.name) !== undefined), rename });
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
      const buf = backend.alloc(data.byteLength, "storage", `${cfg.kind}/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    for (const c of convNetConvs(cfg)) {
      if (c.kind === "dw") {
        const w = await need(`${c.path}.weight`);
        const [C, , k] = w.shape as [number, number, number];
        const m = new Float32Array(k * k * C);
        for (let ch = 0; ch < C; ch++) for (let t = 0; t < k * k; t++) m[t * C + ch] = w.data[ch * k * k + t]!;
        upload(`${c.to}.weight`, [k * k, C], m);
        upload(`${c.to}.bias`, [C], (await need(`${c.path}.bias`)).data);
        continue;
      }
      if (c.kind === "conv") {
        const w = await need(`${c.path}.weight`);
        const m = convMatrix(w.data, w.shape, c.cp);
        upload(`${c.to}.weight`, [w.shape[0]!, m.K], m.w);
        upload(`${c.to}.bias`, [w.shape[0]!], (await need(`${c.path}.bias`)).data);
        continue;
      }
      const w = await need(`${c.path}.convolution.weight`);
      const [weight, bias, mean, variance] = await Promise.all(["weight", "bias", "running_mean", "running_var"].map((f) => need(`${c.path}.normalization.${f}`)));
      const f = foldConvBn(w.data, w.shape, { weight: weight.data, bias: bias.data, mean: mean.data, var: variance.data }, c.cp);
      upload(`${c.to}.weight`, [w.shape[0]!, f.K], f.w);
      upload(`${c.to}.bias`, [w.shape[0]!], f.b);
    }
    let norm: { w: Float32Array; b: Float32Array; eps: number } | undefined;
    if (cfg.kind === "convnext") {
      for (let s = 0; s < cfg.hidden.length; s++)
        for (let l = 0; l < cfg.depths[s]!; l++) {
          const p = `convnext.encoder.stages.${s}.layers.${l}.`;
          const [w, b] = await Promise.all([need(p + "pwconv2.weight"), need(p + "pwconv2.bias")]);
          const g = await read(p + "layer_scale_parameter");
          const [rows, cols] = w.shape as [number, number];
          if (g)
            for (let r = 0; r < rows; r++) {
              for (let c = 0; c < cols; c++) w.data[r * cols + c] = w.data[r * cols + c]! * g.data[r]!;
              b.data[r] = b.data[r]! * g.data[r]!;
            }
          upload(`s${s}.${l}.down.weight`, [rows, cols], w.data);
          upload(`s${s}.${l}.down.bias`, [rows], b.data);
        }
      norm = { w: (await need("convnext.layernorm.weight")).data, b: (await need("convnext.layernorm.bias")).data, eps: cfg.eps };
    }
    const hw = (await read("classifier.1.weight")) ?? (await need("classifier.weight"));
    const hb = (await read("classifier.1.bias")) ?? (await need("classifier.bias"));
    const labels: string[] = [];
    for (const [k, v] of Object.entries((config.id2label as Record<string, string> | undefined) ?? {})) labels[Number(k)] = v;
    const model = new ConvNetModel(backend, cfg, weights);
    return new VisionEngine(backend, undefined, prep, {}, undefined, undefined, undefined, undefined, undefined, undefined, { cfg, model, head: { w: hw.data, b: hb.data, out: hw.shape[0]! }, norm, labels });
  } finally {
    for (const s of sources) s.close?.();
  }
}

/**
 * A VisionEncoderDecoder captioner (ViT encoder, GPT-2 decoder with cross-attention): the encoder on
 * the ViT code (its names under "encoder."), the decoder on the seq2seq code, GPT-2's Conv1D
 * matrices (stored [in, out]) transposed and the fused attention projections split into q, k, v.
 */
async function createCaptioner(
  modelId: string,
  ref: ReturnType<typeof parseModelId>,
  config: Record<string, unknown>,
  prep: ImageProcessorConfig | undefined,
  names: Awaited<ReturnType<typeof weightFiles>>,
  opts: VisionLoadOptions,
  listing: Promise<Set<string> | undefined>,
): Promise<VisionEngine> {
  const enc = (config.encoder as Record<string, unknown> | undefined) ?? {};
  if (enc.model_type !== "vit") throw new Error(`${modelId}: VisionEncoderDecoder encoder "${String(enc.model_type)}" is not supported (vit)`);
  const cfg = vitConfigFromHF(enc);
  const dcfg = visionDecoderConfigFromHF(config);
  if (dcfg.dModel !== cfg.hidden) throw new Error(`${modelId}: encoder width ${cfg.hidden} and decoder width ${dcfg.dModel} differ (a projection is not supported)`);
  const ep = opts.endpoint;
  const files0 = await listing;
  if (files0 && !files0.has("tokenizer.json")) throw new Error(`${modelId} has no tokenizer.json`);
  const tokenizer = encoderTokenizerFromJSON((await fetchJSON<EncoderTokenizerJSON>(fileUrl(ref, "tokenizer.json", ep)))!);
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const vitFiles: WeightFile[] = [];
    const decFiles: WeightFile[] = [];
    const vitRename = (n: string) => (n.startsWith("encoder.") ? vitName(n.slice(8)) : undefined);
    const dt = "decoder.transformer.";
    const decRename = (n: string): string | undefined => {
      if (n === dt + "wte.weight") return "shared.weight";
      if (n === dt + "wpe.weight") return "dec.pos";
      if (n === dt + "ln_f.weight" || n === dt + "ln_f.bias") return `dec.lnF.${n.split(".").pop()}`;
      const m = /^decoder\.transformer\.h\.(\d+)\.(ln_1|ln_2|ln_cross_attn)\.(weight|bias)$/.exec(n);
      return m ? `dec.l${m[1]}.${({ ln_1: "ln1", ln_2: "ln2", ln_cross_attn: "lnc" } as Record<string, string>)[m[2]!]}.${m[3]}` : undefined;
    };
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
      vitFiles.push({ name, source, tensors: header.tensors.filter((t) => vitRename(t.name) !== undefined), rename: vitRename });
      decFiles.push({ name, source, tensors: header.tensors.filter((t) => decRename(t.name) !== undefined), rename: decRename });
    }
    const need = async (name: string) => {
      const t = located.get(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
    };
    const quant = opts.quant ?? "f32";
    const vitWeights = await loadWeights(backend, vitFiles, quant, opts.onProgress);
    const decWeights = await loadWeights(backend, decFiles, quant, opts.onProgress);
    const upload = (into: Map<string, Tensor>, name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `caption/${name}`);
      backend.write(buf, 0, data);
      into.set(name, { shape, dtype: "f32", buf } as Tensor);
    };

    // ViT: the patch matrix and the constant addend (CLS + position embeddings + patch bias), as createVision does.
    const H = cfg.hidden;
    const v = "encoder.embeddings.";
    const pw = await need(v + "patch_embeddings.projection.weight");
    const K = vitPatchK(cfg);
    const row = 3 * cfg.patchSize * cfg.patchSize;
    const patch = new Float32Array(H * K);
    for (let o = 0; o < H; o++) patch.set(pw.data.subarray(o * row, (o + 1) * row), o * K);
    upload(vitWeights, "patch.weight", [H, K], patch);
    const pos = await need(v + "position_embeddings");
    const T = cfg.prefix + (cfg.imageSize / cfg.patchSize) ** 2;
    if (pos.data.length !== T * H) throw new Error(`${modelId}: position embeddings hold ${pos.data.length / H} tokens, the model has ${T}`);
    const add = Float32Array.from(pos.data);
    const cls = (await need(v + "cls_token")).data;
    for (let i = 0; i < H; i++) add[i] += cls[i]!;
    const pb = (await need(v + "patch_embeddings.projection.bias")).data;
    for (let r = cfg.prefix; r < T; r++) for (let i = 0; i < H; i++) add[r * H + i] += pb[i]!;
    upload(vitWeights, "embed.add", [T, H], add);

    // GPT-2's Conv1D layers: stored [in, out], transposed to [out, in]; fused projections split.
    const conv1d = async (from: string, to: string[]) => {
      const w = await need(`${from}.weight`);
      const b = await need(`${from}.bias`);
      const [In, Out] = w.shape as [number, number];
      const part = Out / to.length;
      to.forEach((name, j) => {
        const m = new Float32Array(part * In);
        for (let o = 0; o < part; o++) for (let i = 0; i < In; i++) m[o * In + i] = w.data[i * Out + j * part + o]!;
        upload(decWeights, `${name}.weight`, [part, In], m);
        upload(decWeights, `${name}.bias`, [part], b.data.slice(j * part, (j + 1) * part));
      });
    };
    for (let l = 0; l < dcfg.decLayers; l++) {
      const h = `${dt}h.${l}.`;
      const p = `dec.l${l}`;
      await conv1d(h + "attn.c_attn", [`${p}.q`, `${p}.k`, `${p}.v`]);
      await conv1d(h + "attn.c_proj", [`${p}.o`]);
      await conv1d(h + "crossattention.q_attn", [`${p}.cq`]);
      await conv1d(h + "crossattention.c_attn", [`${p}.ck`, `${p}.cv`]);
      await conv1d(h + "crossattention.c_proj", [`${p}.co`]);
      await conv1d(h + "mlp.c_fc", [`${p}.fc1`]);
      await conv1d(h + "mlp.c_proj", [`${p}.fc2`]);
    }
    const vit = new VitModel(backend, cfg, vitWeights);
    const decoder = new Seq2SeqModel(backend, dcfg, decWeights, { maxTarget: 256 });
    // transformers' generation default when the config sets no max_length.
    const maxLength = Number(config.max_length ?? 20);
    return new VisionEngine(backend, { cfg, model: vit }, prep ?? { size: cfg.imageSize }, {}, undefined, undefined, undefined, undefined, undefined, undefined, undefined, { decoder, tokenizer, maxLength });
  } finally {
    for (const s of sources) s.close?.();
  }
}
