import { ActivationPool, Ctx, STEP_BYTES, packStep, physical, type Backend, type Buf, type Dispatch, type Tensor } from "@kiln/core";
import { ops } from "@kiln/kernels";
import { UnsupportedModelError } from "./config";
import { seq2seqConfigFromHF, type Seq2SeqConfig } from "./seq2seq";

/** Florence-2's DaViT vision tower (Florence2VisionConfig, model_type "davit"). */
export interface DavitConfig {
  dims: number[];
  heads: number[];
  groups: number[];
  depths: number[];
  patchSize: number[];
  patchStride: number[];
  patchPadding: number[];
  patchPrenorm: boolean[];
  window: number;
  mlpRatio: number;
  /** Width of the image tokens handed to the language model. */
  projection: number;
  /** Rows of the 2-D position embedding tables. */
  maxPos: number;
}

export interface Florence2Config {
  vision: DavitConfig;
  text: Seq2SeqConfig;
  imageSize: number;
}

export function florence2ConfigFromHF(c: Record<string, unknown>, imageSize = 768): Florence2Config {
  if (c.model_type !== "florence2") throw new UnsupportedModelError(`model type "${String(c.model_type)}" is not Florence-2`);
  const v = (c.vision_config as Record<string, unknown> | undefined) ?? {};
  const t = (c.text_config as Record<string, unknown> | undefined) ?? {};
  if ((v.model_type ?? "davit") !== "davit") throw new UnsupportedModelError(`Florence-2 vision tower "${String(v.model_type)}" is not supported (davit)`);
  const pos = (v.image_pos_embed as { type?: string; max_pos_embeddings?: number } | undefined) ?? {};
  if ((pos.type ?? "learned_abs_2d") !== "learned_abs_2d") throw new UnsupportedModelError(`Florence-2 image position embedding "${pos.type}" is not supported`);
  const temporal = (v.visual_temporal_embedding as { type?: string } | undefined) ?? {};
  if ((temporal.type ?? "COSINE") !== "COSINE") throw new UnsupportedModelError(`Florence-2 temporal embedding "${temporal.type}" is not supported`);
  const src = ((v.image_feature_source as string[] | undefined) ?? ["spatial_avg_pool", "temporal_avg_pool"]).join(",");
  if (src !== "spatial_avg_pool,temporal_avg_pool") throw new UnsupportedModelError(`Florence-2 image feature source "${src}" is not supported`);
  const arr = (k: string, d: number[]) => ((v[k] as number[] | undefined) ?? d).map(Number);
  const vision: DavitConfig = {
    dims: arr("dim_embed", [128, 256, 512, 1024]),
    heads: arr("num_heads", [4, 8, 16, 32]),
    groups: arr("num_groups", [4, 8, 16, 32]),
    depths: arr("depths", [1, 1, 9, 1]),
    patchSize: arr("patch_size", [7, 3, 3, 3]),
    patchStride: arr("patch_stride", [4, 2, 2, 2]),
    patchPadding: arr("patch_padding", [3, 1, 1, 1]),
    patchPrenorm: ((v.patch_prenorm as boolean[] | undefined) ?? [false, true, true, true]).map(Boolean),
    window: Number(v.window_size ?? 12),
    mlpRatio: 4,
    projection: Number(v.projection_dim ?? c.projection_dim ?? 768),
    maxPos: Number(pos.max_pos_embeddings ?? 50),
  };
  const text = seq2seqConfigFromHF({ ...t, model_type: "bart" });
  if (text.dModel !== vision.projection) throw new UnsupportedModelError(`Florence-2: image tokens of width ${vision.projection} for a ${text.dModel}-wide language model`);
  return { vision, text: { ...text, modelType: "florence2" }, imageSize };
}

/**
 * Kiln's name for a Florence-2 vision tensor (microsoft/Florence-2-* checkpoints), or undefined:
 * convs.{i} → conv{i}, blocks.{i}.{j}.{spatial,channel}_block → b{i}.{j}.{s,c} with conv1 / conv2 →
 * dw1 / dw2, the attention's norm → ln1, qkv, proj, the FFN's norm → ln2, fc1, fc2.
 */
export function davitName(name: string): string | undefined {
  let m = /^vision_tower\.convs\.(\d+)\.(proj|norm)\.(weight|bias)$/.exec(name);
  if (m) return m[2] === "proj" ? `conv${m[1]}.${m[3]}` : `conv${m[1]}.norm.${m[3]}`;
  m = /^vision_tower\.blocks\.(\d+)\.(\d+)\.(spatial|channel)_block\.(.+)\.(weight|bias)$/.exec(name);
  if (!m) return undefined;
  const part: Record<string, string> = {
    "conv1.fn.dw": "dw1",
    "conv2.fn.dw": "dw2",
    "window_attn.norm": "ln1",
    "channel_attn.norm": "ln1",
    "window_attn.fn.qkv": "qkv",
    "channel_attn.fn.qkv": "qkv",
    "window_attn.fn.proj": "proj",
    "channel_attn.fn.proj": "proj",
    "ffn.norm": "ln2",
    "ffn.fn.net.fc1": "fc1",
    "ffn.fn.net.fc2": "fc2",
  };
  const p = part[m[4]!];
  return p ? `b${m[1]}.${m[2]}.${m[3] === "spatial" ? "s" : "c"}.${p}.${m[5]}` : undefined;
}

/**
 * The addend of the last feature map [side², C]: LearnedAbsolutePositionEmbedding2D (column
 * embedding in the first half of each row, row embedding in the second) plus the cosine temporal
 * embedding of frame 0 (sin 0 = 0 at even indices, cos 0 = 1 at odd ones).
 */
export function davitPositions(rowEmb: Float32Array, colEmb: Float32Array, side: number, C: number): Float32Array {
  const half = C / 2;
  const out = new Float32Array(side * side * C);
  for (let y = 0; y < side; y++)
    for (let x = 0; x < side; x++) {
      const o = (y * side + x) * C;
      for (let i = 0; i < half; i++) out[o + i] = colEmb[x * half + i]!;
      for (let i = 0; i < C - half; i++) out[o + half + i] = rowEmb[y * (C - half) + i]!;
      for (let i = 1; i < C; i += 2) out[o + i] = Math.fround(out[o + i]! + 1);
    }
  return out;
}

/**
 * The DaViT tower on the GPU, one plan for the fixed input size: per stage a patch convolution (the
 * first with LayerNorm after it, the others before), then pairs of blocks — a spatial block (3×3
 * depthwise conv plus input, window attention, depthwise conv, MLP) and a channel block (the same
 * with channel attention) — then the 2-D position and temporal embeddings and the projection to the
 * language model's width. Returns the last feature map (with positions added) [side², C] and its
 * projection [side², projection]; the caller adds the mean token and the LayerNorm.
 */
export class DavitModel {
  private plan: { dispatches: readonly Dispatch[]; input: Buf; feats: Tensor; proj: Tensor; ctx: Ctx } | undefined;
  private readonly pool: ActivationPool;
  private readonly step: Buf;
  /** Side of the last feature map. */
  readonly side: number;

  constructor(
    readonly backend: Backend,
    readonly cfg: Florence2Config,
    private readonly weights: Map<string, Tensor>,
  ) {
    this.pool = new ActivationPool(backend, "davit");
    this.step = backend.alloc(STEP_BYTES, "storage", "davit/step");
    backend.write(this.step, 0, packStep({ pos0: 0, lastRow: 0, seed: 0, temperature: 0 }));
    let s = cfg.imageSize;
    const v = cfg.vision;
    for (let i = 0; i < v.dims.length; i++) s = Math.floor((s + 2 * v.patchPadding[i]! - v.patchSize[i]!) / v.patchStride[i]!) + 1;
    this.side = s;
  }

  private w(name: string): Tensor {
    const t = this.weights.get(name);
    if (!t) throw new Error(`DaViT weight "${name}" was not loaded`);
    return t;
  }

  private build() {
    const c = this.cfg.vision;
    const ctx = new Ctx(this.backend, this.step, "davit", this.pool);
    const S = this.cfg.imageSize;
    const input = ctx.buffer(S * S * 4 * 4, "davit/pixels");
    const lin = (x: Tensor, p: string, residual?: Tensor) => ops.linear(ctx, x, this.w(`${p}.weight`), { bias: this.w(`${p}.bias`), residual });
    const ln = (x: Tensor, p: string) => ops.layernorm(ctx, x, this.w(`${p}.weight`), this.w(`${p}.bias`), 1e-5);
    let x: Tensor = { shape: [S * S, 4], dtype: "f32", buf: input };
    let side = S;
    for (let i = 0; i < c.dims.length; i++) {
      if (c.patchPrenorm[i]) x = ln(x, `conv${i}.norm`);
      const k = c.patchSize[i]!;
      const r = ops.conv2d(ctx, x, this.w(`conv${i}.weight`), this.w(`conv${i}.bias`), { H: side, W: side, k, stride: c.patchStride[i]!, pad: c.patchPadding[i]! });
      x = r.y;
      side = r.H;
      if (!c.patchPrenorm[i]) x = ln(x, `conv${i}.norm`);
      for (let j = 0; j < c.depths[i]!; j++) {
        for (const kind of ["s", "c"] as const) {
          const p = `b${i}.${j}.${kind}`;
          ops.addInPlace(ctx, x, ops.dwConv(ctx, x, this.w(`${p}.dw1.weight`), this.w(`${p}.dw1.bias`), { H: side, W: side }));
          const qkv = lin(ln(x, `${p}.ln1`), `${p}.qkv`);
          const a = kind === "s" ? ops.windowAttention(ctx, qkv, { H: side, W: side, heads: c.heads[i]!, win: c.window }) : ops.channelAttention(ctx, qkv, { groups: c.groups[i]! });
          lin(a, `${p}.proj`, x);
          ops.addInPlace(ctx, x, ops.dwConv(ctx, x, this.w(`${p}.dw2.weight`), this.w(`${p}.dw2.bias`), { H: side, W: side }));
          const up = lin(ln(x, `${p}.ln2`), `${p}.fc1`);
          ops.act(ctx, up, "gelu");
          lin(up, `${p}.fc2`, x);
        }
      }
    }
    ops.addInPlace(ctx, x, this.w("pos"));
    const proj = ops.linear(ctx, x, this.w("projection"));
    return { dispatches: ctx.dispatches, input, feats: x, proj, ctx };
  }

  /**
   * The image tokens [1 + side², projection] for normalized pixels [3, size, size]: the mean of the
   * feature map, then every position, projected (image_projection, no bias) and LayerNormed.
   */
  async imageTokens(pixels: Float32Array, projection: Float32Array, norm: { w: Float32Array; b: Float32Array }): Promise<Float32Array> {
    const S = this.cfg.imageSize;
    const n = S * S;
    if (pixels.length !== 3 * n) throw new Error(`Florence-2: expected 3 × ${S} × ${S} pixel values, got ${pixels.length}`);
    this.plan ??= this.build();
    const p = this.plan;
    const nhwc = new Float32Array(n * 4);
    for (let ch = 0; ch < 3; ch++) for (let i = 0; i < n; i++) nhwc[i * 4 + ch] = pixels[ch * n + i]!;
    const be = this.backend;
    be.write(p.input, 0, nhwc);
    be.submit(p.dispatches);
    const T = this.side * this.side;
    const C = this.cfg.vision.dims[this.cfg.vision.dims.length - 1]!;
    const P = this.cfg.vision.projection;
    const [feats, proj] = await Promise.all([be.readback(physical(p.feats.buf), 0, T * C * 4), be.readback(physical(p.proj.buf), 0, T * P * 4)]);
    const f = new Float32Array(feats);
    // spatial_avg_pool: the mean token, projected here (image_projection [C, P]).
    const mean = new Float64Array(C);
    for (let t = 0; t < T; t++) for (let i = 0; i < C; i++) mean[i] += f[t * C + i]!;
    const out = new Float32Array((T + 1) * P);
    for (let o = 0; o < P; o++) {
      let s = 0;
      for (let i = 0; i < C; i++) s += Math.fround(mean[i]! / T) * projection[i * P + o]!;
      out[o] = s;
    }
    out.set(new Float32Array(proj), P);
    // image_proj_norm.
    for (let r = 0; r <= T; r++) {
      let m = 0;
      for (let i = 0; i < P; i++) m += out[r * P + i]!;
      m /= P;
      let v = 0;
      for (let i = 0; i < P; i++) v += (out[r * P + i]! - m) ** 2;
      const inv = 1 / Math.sqrt(v / P + 1e-5);
      for (let i = 0; i < P; i++) out[r * P + i] = (out[r * P + i]! - m) * inv * norm.w[i]! + norm.b[i]!;
    }
    return out;
  }

  destroy(): void {
    for (const t of this.weights.values()) this.backend.free(t.buf);
    this.plan?.ctx.destroy();
    this.pool.destroy();
    this.backend.free(this.step);
  }
}

/**
 * The tokens Florence2Processor adds to the BART tokenizer at load, in its order (ids from 50265 on
 * for the base vocabulary): <od> </od> <ocr> </ocr>, <loc_0> … <loc_999>, then the region, caption,
 * grounding, segmentation and polygon markers. Repos converted for transformers.js carry them in
 * tokenizer.json already; Microsoft's do not.
 */
export const FLORENCE2_ADDED_TOKENS: readonly string[] = [
  "<od>", "</od>", "<ocr>", "</ocr>",
  ...Array.from({ length: 1000 }, (_, i) => `<loc_${i}>`),
  "<cap>", "</cap>", "<ncap>", "</ncap>", "<dcap>", "</dcap>", "<grounding>", "</grounding>", "<seg>", "</seg>", "<sep>",
  "<region_cap>", "</region_cap>", "<region_to_desciption>", "</region_to_desciption>", "<proposal>", "</proposal>", "<poly>", "</poly>", "<and>",
];

/** Florence-2's task tokens: the prompt each stands for, and how its answer is parsed. */
export const FLORENCE2_TASKS: Record<string, { prompt: string; parse: "pure_text" | "description_with_bboxes" | "bboxes" | "phrase_grounding" | "ocr" | "polygons" | "description_with_bboxes_or_polygons"; input?: true }> = {
  "<OCR>": { prompt: "What is the text in the image?", parse: "pure_text" },
  "<OCR_WITH_REGION>": { prompt: "What is the text in the image, with regions?", parse: "ocr" },
  "<CAPTION>": { prompt: "What does the image describe?", parse: "pure_text" },
  "<DETAILED_CAPTION>": { prompt: "Describe in detail what is shown in the image.", parse: "pure_text" },
  "<MORE_DETAILED_CAPTION>": { prompt: "Describe with a paragraph what is shown in the image.", parse: "pure_text" },
  "<OD>": { prompt: "Locate the objects with category name in the image.", parse: "description_with_bboxes" },
  "<DENSE_REGION_CAPTION>": { prompt: "Locate the objects in the image, with their descriptions.", parse: "description_with_bboxes" },
  "<REGION_PROPOSAL>": { prompt: "Locate the region proposals in the image.", parse: "bboxes" },
  "<CAPTION_TO_PHRASE_GROUNDING>": { prompt: "Locate the phrases in the caption: {input}", parse: "phrase_grounding", input: true },
  "<REFERRING_EXPRESSION_SEGMENTATION>": { prompt: "Locate {input} in the image with mask", parse: "polygons", input: true },
  "<REGION_TO_SEGMENTATION>": { prompt: "What is the polygon mask of region {input}", parse: "polygons", input: true },
  "<OPEN_VOCABULARY_DETECTION>": { prompt: "Locate {input} in the image.", parse: "description_with_bboxes_or_polygons", input: true },
  "<REGION_TO_CATEGORY>": { prompt: "What is the region {input}?", parse: "pure_text", input: true },
  "<REGION_TO_DESCRIPTION>": { prompt: "What does the region {input} describe?", parse: "pure_text", input: true },
  "<REGION_TO_OCR>": { prompt: "What text is in the region {input}?", parse: "pure_text", input: true },
};

/** The prompt for a task token (with its text input appended to it, as the processors take it), or the text itself. */
export function florence2Prompt(text: string): string {
  const t = FLORENCE2_TASKS[text];
  if (t && !t.input) return t.prompt;
  for (const [task, spec] of Object.entries(FLORENCE2_TASKS)) {
    if (spec.input && text.includes(task)) return spec.prompt.replaceAll("{input}", text.replaceAll(task, ""));
  }
  return text;
}

/** A location bin (0…999) as a coordinate: the bin's center, scaled to the image side. */
const dequant = (bin: number, size: number) => ((bin + 0.5) / 1000) * size;

function parsePolygons(text: string, width: number, height: number, allowEmpty: boolean): { label: string; polygons: number[][] }[] {
  const loc = "(?:<loc_\\d+>|<sep>|<poly>|</poly>)";
  const phrases = text.match(new RegExp(allowEmpty ? `(?:${loc}){4,}` : `[^<]+(?:${loc}){4,}`, "g")) ?? [];
  const out: { label: string; polygons: number[][] }[] = [];
  for (const phraseText of phrases) {
    const stripped = phraseText.replace(/^loc_\d+>/, "");
    if (stripped === "" && !allowEmpty) continue;
    const label = /^\s*(.*?)(?=<od>|<\/od>|<box>|<\/box>|<bbox>|<\/bbox>|<loc_|<poly>)/.exec(stripped);
    if (!label) continue;
    const instances = phraseText.includes("<poly>") && phraseText.includes("</poly>") ? [...phraseText.matchAll(/<poly>(.*?)<\/poly>/g)].map((m) => m[1]!) : [phraseText];
    for (const inst of instances) {
      const polygons: number[][] = [];
      for (const m of inst.matchAll(/((?:<loc_\d+>)+)(?:<sep>|$)/g)) {
        let bins = [...m[1]!.matchAll(/<loc_(\d+)>/g)].map((x) => Number(x[1]));
        if (bins.length % 2) bins = bins.slice(0, -1);
        polygons.push(bins.map((b, i) => dequant(b, i % 2 ? height : width)));
      }
      if (polygons.length) out.push({ label: label[0].trim(), polygons });
    }
  }
  return out;
}

/**
 * A generated answer parsed for its task, keyed by the task token, as the processors'
 * post_process_generation returns it: text, { labels, bboxes } ([x1, y1, x2, y2] in pixels),
 * { labels, quad_boxes } (OCR with regions), { labels, polygons } (segmentation), or both boxes and
 * polygons (open-vocabulary detection).
 */
export function florence2Parse(text: string, task: string, width: number, height: number): Record<string, unknown> {
  const kind = FLORENCE2_TASKS[task]?.parse ?? "pure_text";
  const clean = text.replaceAll("<s>", "").replaceAll("</s>", "").replaceAll("<pad>", "");
  const size = [width, height];
  const boxes = (re: RegExp) => {
    const labels: string[] = [];
    const items: number[][] = [];
    for (const [, label, ...locs] of clean.matchAll(re)) {
      labels.push(label ? label.trim() : labels.at(-1) ?? "");
      items.push(locs.map((x, i) => dequant(Number(x), size[i % 2]!)));
    }
    return { labels, items };
  };
  let answer: unknown;
  switch (kind) {
    case "pure_text":
      answer = clean;
      break;
    case "ocr": {
      const r = boxes(/(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/gm);
      answer = { labels: r.labels, quad_boxes: r.items };
      break;
    }
    case "polygons": {
      const p = parsePolygons(clean, width, height, false);
      answer = { labels: p.map((x) => x.label), polygons: p.map((x) => x.polygons) };
      break;
    }
    case "description_with_bboxes_or_polygons": {
      // Either polygons or boxes, never both (Python's rule).
      const poly = clean.includes("<poly>");
      const r = poly ? { labels: [], items: [] } : boxes(/([^<]+)?<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/gm);
      const p = poly ? parsePolygons(clean, width, height, false) : [];
      answer = { bboxes: r.items, bboxes_labels: r.labels, polygons: p.map((x) => x.polygons), polygons_labels: p.map((x) => x.label) };
      break;
    }
    default: {
      const r = boxes(/([^<]+)?<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/gm);
      answer = { labels: r.labels, bboxes: r.items };
    }
  }
  return { [task]: answer };
}
