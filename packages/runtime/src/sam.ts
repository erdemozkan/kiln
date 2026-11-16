// Segment Anything (SAM, SlimSAM) from transformers checkpoints: masks for the points and boxes a
// user gives, with transformers' SamProcessor around the model (resize, pad, coordinates, masks
// back at the image's size). The image embedding of the last image is kept, so further prompts on
// the same image run only the mask decoder.
import type { Backend, Tensor } from "@kiln/core";
import { preprocessImage, readPytorchHeader, readSafetensorsHeader, toF32, type ByteSource, type ImageProcessorConfig, type ModelRef, type RgbImage, type TensorEntry } from "@kiln/loader";
import { SamModel, convMatrix, convTransposeMatrix, samConfigFromHF, samGrid, samImagePositional, samName, samSparsePrompt, type SamConfig, type SamPromptWeights } from "@kiln/models";
import { imageKey } from "./engine";
import { openBackend, openParsed } from "./load";
import { loadWeights, type WeightFile } from "./weights";
import type { VisionLoadOptions } from "./vision";

/** Prompts for `pipeline("mask-generation")`, in the image's own pixel coordinates (as transformers' SamProcessor takes them). */
export interface MaskGenerationOptions {
  /** Points [x, y]. */
  points?: [number, number][];
  /** A label per point: 1 the object (default), 0 not the object. */
  labels?: number[];
  /** Boxes [x1, y1, x2, y2] around the object. */
  boxes?: [number, number, number, number][];
  /** Three masks of different extent (default), or the single best guess. */
  multimask_output?: boolean;
  /** Logits above this are inside the mask (default 0). */
  mask_threshold?: number;
}

/** A mask at the image's size (255 inside, 0 outside) and the model's estimate of its IoU. */
export interface GeneratedMask {
  score: number;
  mask: { width: number; height: number; channels: 1; data: Uint8Array };
}

/** The resized size SamProcessor uses: the longest side scaled to `edge`, both sides rounded. */
export function samResizeSize(width: number, height: number, edge: number): [number, number] {
  const scale = edge / Math.max(width, height);
  return [Math.floor(width * scale + 0.5), Math.floor(height * scale + 0.5)];
}

/** A loaded SAM: the model, its processor settings and the prompt encoder's weights. */
export class SamSegmenter {
  private key: string | undefined;

  constructor(
    readonly model: SamModel,
    readonly processor: ImageProcessorConfig,
    private readonly prompts: SamPromptWeights,
  ) {}

  get cfg(): SamConfig {
    return this.model.cfg;
  }

  /** Runs the image encoder unless `img` is the image encoded last; returns the resized size. */
  private encode(img: RgbImage): [number, number] {
    const n = this.cfg.imageSize;
    const resized = samResizeSize(img.width, img.height, n);
    const key = imageKey(img);
    if (key === this.key) return resized;
    const p = preprocessImage(img, { ...this.processor, size: { width: resized[0], height: resized[1] } });
    // Padded at the bottom and right with zeros (after normalizing), to the model's square input.
    const px = new Float32Array(3 * n * n);
    for (let c = 0; c < 3; c++) for (let y = 0; y < p.height; y++) px.set(p.data.subarray((c * p.height + y) * p.width, (c * p.height + y + 1) * p.width), (c * n + y) * n);
    this.key = undefined;
    this.model.encode(px);
    this.key = key;
    return resized;
  }

  /** Masks for the prompts on `img`, at its size, with their IoU scores (in the model's order). */
  async segment(img: RgbImage, o: MaskGenerationOptions): Promise<GeneratedMask[]> {
    const n = this.cfg.imageSize;
    const resized = this.encode(img);
    const sx = resized[0] / img.width;
    const sy = resized[1] / img.height;
    const points = (o.points ?? []).map(([x, y], i) => {
      const label = o.labels?.[i] ?? 1;
      if (label !== 0 && label !== 1) throw new Error(`mask-generation: a point's label is 1 (the object) or 0 (not the object), got ${label}`);
      return { x: x * sx, y: y * sy, label };
    });
    const boxes = (o.boxes ?? []).map(([x1, y1, x2, y2]) => [x1 * sx, y1 * sy, x2 * sx, y2 * sy] as [number, number, number, number]);
    if (!points.length && !boxes.length) throw new Error("mask-generation: give points or boxes to segment");
    const sparse = samSparsePrompt(this.prompts, { points, boxes }, n);
    const single = o.multimask_output === false;
    const count = single ? 1 : this.cfg.dec.masks - 1;
    const first = single ? 0 : 1;
    const r = await this.model.segment(sparse, { first, count, cw: resized[0], ch: resized[1], width: img.width, height: img.height, threshold: o.mask_threshold ?? 0 });
    return r.masks.map((data, k) => ({ score: r.iou[first + k]!, mask: { width: img.width, height: img.height, channels: 1 as const, data } }));
  }

  destroy(): void {
    this.model.destroy();
  }
}

/** transformers' get_rel_pos interpolation of a relative position table [n, D] to `size` rows (linear, align_corners=False). */
function resizeRelPos(t: Float32Array, n: number, D: number, size: number): Float32Array {
  if (n === size) return t;
  const out = new Float32Array(size * D);
  const s = n / size;
  for (let i = 0; i < size; i++) {
    const f = Math.max((i + 0.5) * s - 0.5, 0);
    const lo = Math.min(Math.floor(f), n - 1);
    const hi = Math.min(lo + 1, n - 1);
    const w = f - lo;
    for (let d = 0; d < D; d++) out[i * D + d] = t[lo * D + d]! * (1 - w) + t[hi * D + d]! * w;
  }
  return out;
}

/**
 * Loads SAM: the vision encoder's matrices and norms as weights (quantized as asked); the patch
 * convolution as a matrix with the position embeddings and its bias as one addend; each layer's
 * relative position tables at its window's size; the neck's convolutions as matrices; the mask
 * decoder's tensors as they are (f32), its transposed convolutions reshaped, and the image's
 * positional embedding computed once.
 */
export async function createSam(modelId: string, ref: ModelRef, config: Record<string, unknown>, prep: ImageProcessorConfig | undefined, names: { names: string[]; format: string }, opts: VisionLoadOptions): Promise<{ backend: Backend; sam: SamSegmenter }> {
  const cfg = samConfigFromHF(config);
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
      for (const t of header.tensors) located.set(t.name.replace(/^sam\./, ""), { source, entry: t });
      files.push({ name, source, tensors: header.tensors.filter((t) => samName(t.name.replace(/^sam\./, "")) !== undefined), rename: (n) => samName(n.replace(/^sam\./, "")) });
    }
    const need = async (name: string) => {
      const t = located.get(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
    };
    const weights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);
    const upload = (name: string, shape: number[], data: Float32Array | Uint32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `sam/${name}`);
      backend.write(buf, 0, data);
      weights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    const v = "vision_encoder.";
    const H = cfg.hidden;
    const g = samGrid(cfg);
    const D = H / cfg.heads;
    const pw = await need(v + "patch_embed.projection.weight");
    upload("patch.weight", [H, 3 * cfg.patchSize ** 2], pw.data);
    const add = Float32Array.from((await need(v + "pos_embed")).data);
    if (add.length !== g * g * H) throw new Error(`${modelId}: pos_embed holds ${add.length / H} positions, the model has ${g * g}`);
    const pb = (await need(v + "patch_embed.projection.bias")).data;
    for (let t = 0; t < g * g; t++) for (let i = 0; i < H; i++) add[t * H + i] += pb[i]!;
    upload("embed.add", [g * g, H], add);
    const wins = new Set<number>();
    for (let l = 0; l < cfg.layers; l++) {
      const win = cfg.globalLayers.includes(l) ? g : cfg.window;
      wins.add(win);
      for (const axis of ["h", "w"] as const) {
        const t = await need(`${v}layers.${l}.attn.rel_pos_${axis}`);
        if (t.shape[1] !== D) throw new Error(`${modelId}: rel_pos_${axis} of layer ${l} has heads of ${t.shape[1]}, the model ${D}`);
        upload(`l${l}.r${axis}`, [2 * win - 1, D], resizeRelPos(t.data, t.shape[0]!, D, 2 * win - 1));
      }
      if (!cfg.qkvBias) upload(`l${l}.qkv.bias`, [3 * H], new Float32Array(3 * H));
    }
    for (const win of wins) {
      const B = Math.ceil(g / win) ** 2;
      upload(`lens.${win}`, [B], new Uint32Array(B).fill(win * win));
    }
    const C = cfg.channels;
    upload("neck.conv1.weight", [C, H], (await need(v + "neck.conv1.weight")).data);
    const c2 = await need(v + "neck.conv2.weight");
    const m2 = convMatrix(c2.data, c2.shape);
    upload("neck.conv2.weight", [C, m2.K], m2.w);
    upload("neck.zero", [C], new Float32Array(C));
    for (const n of ["ln1", "ln2"]) for (const f of ["weight", "bias"]) upload(`neck.${n}.${f}`, [C], (await need(`${v}neck.layer_norm${n.slice(2)}.${f}`)).data);
    // The mask decoder, f32 as it is; its transposed convolutions as [f·f, O, C].
    for (const name of located.keys()) {
      if (!name.startsWith("mask_decoder.") || /^mask_decoder\.(upscale_conv\d|iou_token|mask_tokens)\./.test(name)) continue;
      const data = await need(name);
      upload(`dec.${name.slice("mask_decoder.".length)}`, data.shape, data.data);
    }
    for (const i of [1, 2]) {
      const w = await need(`mask_decoder.upscale_conv${i}.weight`);
      upload(`dec.upscale${i}.weight`, [4, w.shape[1]!, w.shape[0]!], convTransposeMatrix(w.data, w.shape));
      upload(`dec.upscale${i}.bias`, [w.shape[1]!], (await need(`mask_decoder.upscale_conv${i}.bias`)).data);
    }
    const gaussian = (await need("shared_image_embedding.positional_embedding")).data;
    upload("dec.image_pe", [g * g, C], samImagePositional(gaussian, g));
    upload("dec.no_mask", [C], (await need("prompt_encoder.no_mask_embed.weight")).data);
    const iouToken = (await need("mask_decoder.iou_token.weight")).data;
    const maskTokens = (await need("mask_decoder.mask_tokens.weight")).data;
    const outputTokens = new Float32Array(iouToken.length + maskTokens.length);
    outputTokens.set(iouToken, 0);
    outputTokens.set(maskTokens, iouToken.length);
    const prompts: SamPromptWeights = {
      gaussian,
      pointEmbed: await Promise.all([0, 1, 2, 3].map(async (i) => (await need(`prompt_encoder.point_embed.${i}.weight`)).data)),
      notAPoint: (await need("prompt_encoder.not_a_point_embed.weight")).data,
    };
    const model = new SamModel(backend, cfg, weights, outputTokens);
    return { backend, sam: new SamSegmenter(model, prep, prompts) };
  } finally {
    for (const s of sources) s.close?.();
  }
}
