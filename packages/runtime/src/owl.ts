// OWL-ViT and OWLv2: open-vocabulary object detection. The image tower and its heads run as one GPU
// plan and give, per patch, a box and a class embedding; text queries are embedded by the CLIP text
// tower and matched to the patches on the CPU, so new queries on the same image skip the image.
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fetchText, fileUrl, preprocessImage, readPytorchHeader, readSafetensorsHeader, toF32, toRgb, type ByteSource, type ImageProcessorConfig, type ModelRef, type RgbImage, type TensorEntry } from "@kiln/loader";
import { ClipTextModel, OWL_CLASS_EXTRA, VitModel, clipPoolIndex, owlBoxBias, owlConfigFromHF, owlTail, owlTextName, owlVisionName, vitGrid, vitPatchK, type OwlConfig } from "@kiln/models";
import { ClipTokenizer, type ClipTokenizerJSON } from "@kiln/tokenizers";
import { imageKey } from "./engine";
import { openBackend, openParsed } from "./load";
import { loadWeights, type WeightFile } from "./weights";
import type { DetectedObject, VisionEngine, VisionLoadOptions } from "./vision";

export interface ZeroShotDetectionOptions {
  /** Least score a detection needs, exclusive (default 0.1). */
  threshold?: number;
  /** Most detections returned, best first (default: all). */
  top_k?: number;
  /** Boxes as fractions of the image instead of whole pixels (default false). */
  percentage?: boolean;
}

/** 1-D Gaussian weights of scipy.ndimage.gaussian_filter1d (truncate 4): radius int(4σ + 0.5). */
function gaussianWeights(sigma: number): Float64Array {
  const r = Math.trunc(4 * sigma + 0.5);
  const w = new Float64Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) s += w[i + r] = Math.exp((-0.5 / (sigma * sigma)) * i * i);
  for (let i = 0; i < w.length; i++) w[i]! /= s;
  return w;
}

/** Index i of a line of n extended by scipy's "mirror" (d c b | a b c d | c b a: the edge not repeated). */
function mirror(i: number, n: number): number {
  if (n === 1) return 0;
  const p = 2 * n - 2;
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
}

/**
 * OWLv2's image processor (transformers v4's Owlv2ImageProcessor, after the original
 * implementation): pixels rescaled to 0…1 and padded at the bottom and right to a square with 0.5;
 * when shrinking, a Gaussian blur of σ = (factor − 1) / 2 (scipy's gaussian_filter, mirrored edges);
 * resized to size × size by linear interpolation (scipy's zoom with grid_mode, mirrored edges);
 * clipped to the padded image's range; normalized. Float32 between steps, as transformers' numpy
 * arrays are. The blur runs only for the rows and columns the resize reads. Returns [3, size, size].
 */
export function owlv2Pixels(img: RgbImage, size: number, mean: readonly number[], std: readonly number[], rescale = 0.00392156862745098): Float32Array {
  const f = Math.fround;
  const rgb = toRgb(img);
  const { width: W, height: H } = rgb;
  const data = rgb.data;
  const S = Math.max(W, H);
  const lut = new Float32Array(256);
  for (let v = 0; v < 256; v++) lut[v] = v * rescale;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = lut[data[i]!]!;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (W !== H) [lo, hi] = [Math.min(lo, 0.5), Math.max(hi, 0.5)];
  const factor = S / size;
  // Linear interpolation at input coordinate (o + 0.5) · S / size − 0.5 on both axes, mirrored past the edges.
  const i0s = new Int32Array(size);
  const i1s = new Int32Array(size);
  // scipy's order-1 weights: w0 = 1 − t, w1 = 1 − w0.
  const w0s = new Float64Array(size);
  const w1s = new Float64Array(size);
  for (let o = 0; o < size; o++) {
    let c = (o + 0.5) * factor - 0.5;
    if (S > 1) {
      if (c < 0) c = -c;
      if (c > S - 1) c = 2 * (S - 1) - c;
    } else c = 0;
    const i0 = Math.floor(c);
    w0s[o] = 1 - (c - i0);
    w1s[o] = 1 - w0s[o]!;
    i0s[o] = mirror(i0, S);
    i1s[o] = mirror(i0 + 1, S);
  }
  // The input indices the interpolation reads (the same on both axes), and each one's place among them.
  const needed = [...new Set([...i0s, ...i1s])].sort((a, b) => a - b);
  const place = new Int32Array(S).fill(-1);
  needed.forEach((v, j) => (place[v] = j));
  const N = needed.length;
  const sigma = Math.max(0, (factor - 1) / 2);
  const w = sigma > 0 ? gaussianWeights(sigma) : undefined;
  const R = w ? (w.length - 1) / 2 : 0;
  const acc = new Float64Array(S * 3);
  const line = new Float32Array(S * 3);
  // Source row y as 0…1 values, padded with 0.5 to S (the last few kept: the blur windows of
  // consecutive rows overlap).
  const srcRows = new Map<number, Float32Array>();
  const srcRow = (y: number): Float32Array => {
    let v = srcRows.get(y);
    if (v) return v;
    if (srcRows.size >= 2 * R + 4) {
      v = srcRows.get(srcRows.keys().next().value!)!;
      srcRows.delete(srcRows.keys().next().value!);
    } else v = new Float32Array(S * 3);
    if (y < H) {
      const off = y * W * 3;
      for (let x = 0; x < W * 3; x++) v[x] = lut[data[off + x]!]!;
      v.fill(0.5, W * 3);
    } else v.fill(0.5);
    srcRows.set(y, v);
    return v;
  };
  const taps = 2 * R + 1;
  // Per needed column, the line offsets the column pass reads.
  const colTaps = new Int32Array(N * taps);
  for (let j = 0; j < N; j++) for (let k = 0; k < taps; k++) colTaps[j * taps + k] = mirror(needed[j]! - R + k, S) * 3;
  // Row r of the blurred, padded image at the needed columns: [N · 3].
  const rowAt = (r: number): Float32Array => {
    const out = new Float32Array(N * 3);
    if (!w) {
      for (let j = 0; j < N; j++) {
        const x = needed[j]!;
        for (let c = 0; c < 3; c++) out[j * 3 + c] = r < H && x < W ? lut[data[(r * W + x) * 3 + c]!]! : 0.5;
      }
      return out;
    }
    // Along the rows axis, every column. As scipy's correlate1d sums a symmetric kernel: the center
    // tap, then each pair from the outermost in.
    const mid = srcRow(mirror(r, S));
    for (let x = 0; x < S * 3; x++) acc[x] = mid[x]! * w[R]!;
    for (let k = 0; k < R; k++) {
      const a = srcRow(mirror(r - R + k, S));
      const b = srcRow(mirror(r + R - k, S));
      const wk = w[k]!;
      for (let x = 0; x < S * 3; x++) acc[x] += (a[x]! + b[x]!) * wk;
    }
    for (let x = 0; x < S * 3; x++) line[x] = acc[x]!;
    // Along the columns axis, at the needed columns.
    for (let j = 0; j < N; j++) {
      const t = j * taps;
      const o = colTaps[t + R]!;
      let s0 = line[o]! * w[R]!;
      let s1 = line[o + 1]! * w[R]!;
      let s2 = line[o + 2]! * w[R]!;
      for (let k = 0; k < R; k++) {
        const a = colTaps[t + k]!;
        const b = colTaps[t + taps - 1 - k]!;
        const wk = w[k]!;
        s0 += (line[a]! + line[b]!) * wk;
        s1 += (line[a + 1]! + line[b + 1]!) * wk;
        s2 += (line[a + 2]! + line[b + 2]!) * wk;
      }
      out[j * 3] = s0;
      out[j * 3 + 1] = s1;
      out[j * 3 + 2] = s2;
    }
    return out;
  };
  const rows = new Map<number, Float32Array>();
  const row = (r: number): Float32Array => {
    let v = rows.get(r);
    if (!v) {
      if (rows.size >= 4) rows.delete(rows.keys().next().value!);
      rows.set(r, (v = rowAt(r)));
    }
    return v;
  };
  const n = size * size;
  const out = new Float32Array(3 * n);
  const m = mean.map(f);
  const sd = std.map(f);
  const as = Int32Array.from(i0s, (i) => place[i]! * 3);
  const bs = Int32Array.from(i1s, (i) => place[i]! * 3);
  for (let y = 0; y < size; y++) {
    const top = row(i0s[y]!);
    const bot = row(i1s[y]!);
    const [uy, ty] = [w0s[y]!, w1s[y]!];
    for (let c = 0; c < 3; c++) {
      const [mc, sc, base] = [m[c]!, sd[c]!, c * n + y * size];
      for (let x = 0; x < size; x++) {
        const a = as[x]! + c;
        const b = bs[x]! + c;
        const [ux, tx] = [w0s[x]!, w1s[x]!];
        // As scipy's NI_ZoomShift sums: each corner's value times its weights, in raster order.
        const sum = top[a]! * uy * ux + top[b]! * uy * tx + bot[a]! * ty * ux + bot[b]! * ty * tx;
        const v = Math.min(Math.max(f(sum), lo), hi);
        out[base + x] = f(f(v - mc) / sc);
      }
    }
  }
  return out;
}

/** What the image side gives per patch, kept per image. */
interface OwlImage {
  /** Class embeddings [patches, queryDim], each divided by (its norm + 1e-6). */
  embeds: Float32Array;
  /** Logit shift and scale (ELU + 1 applied) per patch. */
  shift: Float32Array;
  scale: Float32Array;
  /** Boxes [patches, 4]: corners (x0, y0, x1, y1) as fractions of the model's input. */
  boxes: Float32Array;
}

export class OwlDetector {
  private readonly images = new Map<string, OwlImage>();
  private readonly queries = new Map<string, Float32Array>();

  constructor(
    readonly cfg: OwlConfig,
    readonly processor: ImageProcessorConfig,
    private readonly vision: VitModel,
    private readonly text: ClipTextModel,
    private readonly tokenizer: ClipTokenizer,
    /** text_projection [projection, hidden]. */
    private readonly textProj: { w: Float32Array; out: number },
  ) {}

  /** Pixel values [3, size, size] as the model's image processor makes them. */
  pixels(img: RgbImage): Float32Array {
    const S = this.cfg.vision.imageSize;
    if (this.cfg.kind === "owlv2") {
      const p = this.processor;
      const arr = (v: number | number[] | undefined, d: number[]) => (Array.isArray(v) ? v : v === undefined ? d : [v, v, v]);
      return owlv2Pixels(img, S, arr(p.image_mean, [0.48145466, 0.4578275, 0.40821073]), arr(p.image_std, [0.26862954, 0.26130258, 0.27577711]), p.do_rescale === false ? 1 : (p.rescale_factor ?? 1 / 255));
    }
    const p = preprocessImage(img, this.processor);
    if (p.width !== S || p.height !== S) throw new Error(`the image processor made a ${p.width}×${p.height} image; the model takes ${S}×${S}`);
    return p.data;
  }

  /** The image side's outputs for pixel values [3, size, size] (see owlTail), before any CPU work. */
  raw(pixels: Float32Array): Promise<Record<string, Float32Array>> {
    return this.vision.tailOutputs(pixels);
  }

  private async image(img: RgbImage): Promise<OwlImage> {
    const key = imageKey(img);
    const hit = this.images.get(key);
    if (hit) {
      this.images.delete(key);
      this.images.set(key, hit);
      return hit;
    }
    const out = await this.raw(this.pixels(img));
    const D = this.cfg.queryDim;
    const row = D + OWL_CLASS_EXTRA;
    const P = out.box!.length / 4;
    const f = Math.fround;
    const embeds = new Float32Array(P * D);
    const shift = new Float32Array(P);
    const scale = new Float32Array(P);
    const boxes = new Float32Array(P * 4);
    const cls = out.cls!;
    for (let p = 0; p < P; p++) {
      let s = 0;
      for (let i = 0; i < D; i++) s += cls[p * row + i]! ** 2;
      const norm = f(f(Math.sqrt(s)) + 1e-6);
      for (let i = 0; i < D; i++) embeds[p * D + i] = f(cls[p * row + i]! / norm);
      shift[p] = cls[p * row + D]!;
      const z = cls[p * row + D + 1]!;
      scale[p] = f((z > 0 ? z : f(Math.expm1(z))) + 1);
      const [cx, cy, w, h] = [0, 1, 2, 3].map((k) => f(1 / (1 + Math.exp(-out.box![p * 4 + k]!)))) as [number, number, number, number];
      boxes.set([f(cx - f(0.5 * w)), f(cy - f(0.5 * h)), f(cx + f(0.5 * w)), f(cy + f(0.5 * h))], p * 4);
    }
    const res = { embeds, shift, scale, boxes };
    if (this.images.size >= 4) this.images.delete(this.images.keys().next().value!);
    this.images.set(key, res);
    return res;
  }

  /** Query embeddings of texts as the class head uses them: projected, normalized, divided by (norm + 1e-6). */
  async queryEmbeds(texts: readonly string[]): Promise<Float32Array[]> {
    const missing = [...new Set(texts.filter((t) => !this.queries.has(t)))];
    if (missing.length) {
      const ids = missing.map((t) => this.tokenizer.encode(t));
      const hidden = await this.text.forwardBatch(ids, ids.map((s) => clipPoolIndex(s, this.cfg.text.eosId)));
      if (this.queries.size + missing.length > 256) this.queries.clear();
      const f = Math.fround;
      const { w, out } = this.textProj;
      missing.forEach((t, i) => {
        const h = hidden[i]!;
        const n = h.length;
        const e = new Float32Array(out);
        for (let o = 0; o < out; o++) {
          let s = 0;
          for (let k = 0; k < n; k++) s += w[o * n + k]! * h[k]!;
          e[o] = s;
        }
        // Owl{ViT,v2}Model normalizes the text embeddings; the class head divides by (norm + 1e-6) again.
        for (const eps of [0, 1e-6]) {
          let s = 0;
          for (const v of e) s += v * v;
          const norm = f(f(Math.sqrt(s)) + eps);
          for (let k = 0; k < out; k++) e[k] = f(e[k]! / norm);
        }
        this.queries.set(t, e);
      });
    }
    return texts.map((t) => this.queries.get(t)!);
  }

  /**
   * Class logits [patches, queries] and boxes [patches, 4] (corners as fractions of the model's input)
   * of an image for text queries, as Owl{ViT,v2}ForObjectDetection gives them.
   */
  async predict(img: RgbImage, texts: readonly string[]): Promise<{ logits: Float32Array; boxes: Float32Array }> {
    const [im, q] = await Promise.all([this.image(img), this.queryEmbeds(texts)]);
    const D = this.cfg.queryDim;
    const P = im.shift.length;
    const Q = q.length;
    const f = Math.fround;
    const logits = new Float32Array(P * Q);
    for (let p = 0; p < P; p++)
      for (let j = 0; j < Q; j++) {
        const t = q[j]!;
        let s = 0;
        for (let i = 0; i < D; i++) s += im.embeds[p * D + i]! * t[i]!;
        logits[p * Q + j] = f(f(f(s) + im.shift[p]!) * im.scale[p]!);
      }
    return { logits, boxes: im.boxes };
  }

  /**
   * Zero-shot object detection, post-processed as transformers' Owl{ViT,v2}ImageProcessor does: per
   * patch the best query and its sigmoid score, kept above `threshold`; boxes scaled to the image
   * (OWLv2: to the padded square, the longest side, so they land on the image) and truncated to whole
   * pixels as transformers.js's pipeline does, unless `percentage`; best first.
   */
  async detect(img: RgbImage, labels: readonly string[], options: ZeroShotDetectionOptions = {}): Promise<DetectedObject[]> {
    if (!labels.length) throw new Error("zero-shot object detection needs candidate labels");
    const threshold = options.threshold ?? 0.1;
    const { logits, boxes } = await this.predict(img, labels);
    const Q = labels.length;
    const [W, H] = [img.width, img.height];
    const S = Math.max(W, H);
    const [sx, sy] = this.cfg.kind === "owlv2" ? [S, S] : [W, H];
    const res: DetectedObject[] = [];
    for (let p = 0; p < logits.length / Q; p++) {
      let best = 0;
      for (let j = 1; j < Q; j++) if (logits[p * Q + j]! > logits[p * Q + best]!) best = j;
      const score = Math.fround(1 / (1 + Math.exp(-logits[p * Q + best]!)));
      if (!(score > threshold)) continue;
      let b = Array.from(boxes.subarray(p * 4, p * 4 + 4), (v, i) => v * (i % 2 ? sy : sx));
      b = options.percentage ? b.map((v, i) => v / (i % 2 ? H : W)) : b.map((v) => v | 0);
      res.push({ score, label: labels[best]!, box: { xmin: b[0]!, ymin: b[1]!, xmax: b[2]!, ymax: b[3]! } });
    }
    res.sort((a, b) => b.score - a.score);
    return options.top_k !== undefined && options.top_k > 0 ? res.slice(0, options.top_k) : res;
  }

  destroy(): void {
    this.vision.destroy();
    this.text.destroy();
  }
}

/**
 * Loads OWL-ViT or OWLv2 (safetensors or pytorch_model.bin). The tokenizer comes from tokenizer.json,
 * or from vocab.json and merges.txt (Google's repos have only those).
 */
export async function createOwl(
  modelId: string,
  ref: ModelRef,
  config: Record<string, unknown>,
  prep: ImageProcessorConfig | undefined,
  names: { names: string[]; format: "safetensors" | "pytorch" },
  opts: VisionLoadOptions,
  files: Promise<Set<string> | undefined>,
  make: (o: OwlDetector, backend: Backend) => VisionEngine,
): Promise<VisionEngine> {
  const cfg = owlConfigFromHF(config);
  const ep = opts.endpoint;
  const listing = await files;
  let tokJson: ClipTokenizerJSON;
  if (!listing || listing.has("tokenizer.json")) tokJson = (await fetchJSON<ClipTokenizerJSON>(fileUrl(ref, "tokenizer.json", ep)))!;
  else {
    const [vocab, merges] = await Promise.all([
      fetchJSON<Record<string, number>>(fileUrl(ref, "vocab.json", ep)),
      fetchText(fileUrl(ref, "merges.txt", ep)).then((t) => t!),
    ]);
    const lines = merges.split("\n").filter((l) => l && !l.startsWith("#version"));
    tokJson = { model: { type: "BPE", vocab: vocab!, merges: lines, end_of_word_suffix: "</w>", unk_token: "<|endoftext|>" } };
  }
  const tokenizer = new ClipTokenizer(tokJson, cfg.text.maxPositions);
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const wfiles: WeightFile[] = [];
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name.replace(/^owl(?:v2|vit)\./, ""), { source, entry: t });
      wfiles.push({ name, source, tensors: header.tensors.filter((t) => owlVisionName(t.name) !== undefined), rename: owlVisionName });
    }
    const need = async (name: string) => {
      const t = located.get(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      const data = toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count);
      if (!t.entry.transposed) return { shape: t.entry.shape, data };
      const [rows, cols] = t.entry.shape as [number, number];
      const out = new Float32Array(count);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = data[c * rows + r]!;
      return { shape: t.entry.shape, data: out };
    };
    const quant = opts.quant ?? "f32";
    const weights = await loadWeights(backend, wfiles, quant, opts.onProgress);
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `owl/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    const v = cfg.vision;
    const Hd = v.hidden;
    // Patch convolution [hidden, 3, p, p] as [hidden, patchK]; CLIP's patch embedding has no bias.
    const pw = await need("vision_model.embeddings.patch_embedding.weight");
    const K = vitPatchK(v);
    const row = 3 * v.patchSize * v.patchSize;
    const patch = new Float32Array(Hd * K);
    for (let o = 0; o < Hd; o++) patch.set(pw.data.subarray(o * row, (o + 1) * row), o * K);
    upload("patch.weight", [Hd, K], patch);
    const [gh, gw] = vitGrid(v, v.imageSize, v.imageSize);
    const T = 1 + gh * gw;
    const pos = await need("vision_model.embeddings.position_embedding.weight");
    if (pos.data.length !== T * Hd) throw new Error(`${modelId}: position embeddings hold ${pos.data.length / Hd} tokens, the model has ${T}`);
    const add = Float32Array.from(pos.data);
    const cls = (await need("vision_model.embeddings.class_embedding")).data;
    for (let i = 0; i < Hd; i++) add[i] += cls[i]!;
    upload("embed.add", [T, Hd], add);
    upload("boxBias", [gh * gw, 4], owlBoxBias(gh, gw));
    // The class head's dense0, logit_shift and logit_scale as one [queryDim + 32, hidden] linear.
    const D = cfg.queryDim;
    const [d0w, d0b, shw, shb, scw, scb] = await Promise.all(
      ["dense0.weight", "dense0.bias", "logit_shift.weight", "logit_shift.bias", "logit_scale.weight", "logit_scale.bias"].map((n) => need(`class_head.${n}`)),
    );
    const cw = new Float32Array((D + OWL_CLASS_EXTRA) * Hd);
    const cb = new Float32Array(D + OWL_CLASS_EXTRA);
    cw.set(d0w!.data);
    cw.set(shw!.data, D * Hd);
    cw.set(scw!.data, (D + 1) * Hd);
    cb.set(d0b!.data);
    cb[D] = shb!.data[0]!;
    cb[D + 1] = scb!.data[0]!;
    upload("cls.weight", [D + OWL_CLASS_EXTRA, Hd], cw);
    upload("cls.bias", [D + OWL_CLASS_EXTRA], cb);

    const textWeights = await loadWeights(
      backend,
      wfiles.map((f) => ({ ...f, tensors: [...located.values()].filter((t) => t.source === f.source && owlTextName(t.entry.name) !== undefined).map((t) => t.entry), rename: owlTextName })),
      quant,
      opts.onProgress,
    );
    const tp = await need("text_projection.weight");
    const vision = new VitModel(backend, v, weights, undefined, owlTail(cfg));
    const text = new ClipTextModel(backend, cfg.text, textWeights);
    // OWL-ViT's processor config gives size as [height, width].
    let processor: ImageProcessorConfig = prep ?? { size: { height: v.imageSize, width: v.imageSize }, resample: 3 };
    const size = processor.size as unknown;
    if (Array.isArray(size)) processor = { ...processor, size: { height: Number(size[0]), width: Number(size[1]) } };
    return make(new OwlDetector(cfg, processor, vision, text, tokenizer, { w: tp.data, out: tp.shape[0]! }), backend);
  } finally {
    for (const s of sources) s.close?.();
  }
}
