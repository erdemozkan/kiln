// Donut (VisionEncoderDecoder: a Swin encoder and an mBART decoder) for document question answering:
// the page through Donut's processor and encoder once (the last two pages are kept), then greedy
// decoding from the DocVQA prompt.
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fileUrl, readPytorchHeader, readSafetensorsHeader, resizeImage, resizeImageReducing, toF32, toRgb, type ByteSource, type ImageProcessorConfig, type ModelRef, type RgbImage, type TensorEntry } from "@kiln/loader";
import { DonutSwinModel, Seq2SeqModel, donutHidden, donutPatchK, donutSwinConfigFromHF, donutSwinName, seq2seqName, swinRelativeBias, visionDecoderConfigFromHF, type DonutSwinConfig } from "@kiln/models";
import { encoderTokenizerFromJSON, type EncoderTokenizer, type EncoderTokenizerJSON } from "@kiln/tokenizers";
import { imageKey } from "./engine";
import { openBackend, openParsed } from "./load";
import { loadWeights, type WeightFile } from "./weights";
import type { VisionEngine, VisionLoadOptions } from "./vision";

export interface DocumentQuestionAnsweringOptions {
  /** Most tokens to generate (default 256; the decoder's 128 positions, prompt included, cap it). */
  max_new_tokens?: number;
}

/** What Donut's processor needs from preprocessor_config.json. */
export interface DonutProcessing {
  /** Output size [height, width]. */
  size: [number, number];
  resample: number;
  alignLongAxis: boolean;
  resize: boolean;
  thumbnail: boolean;
  pad: boolean;
  rescale: number;
  mean: number[];
  std: number[];
}

/** Donut's processor settings; the old list form of `size` is [width, height], as transformers reads it. */
export function donutProcessing(p: Record<string, unknown> | undefined, cfg: DonutSwinConfig): DonutProcessing {
  const c = p ?? {};
  const s = c.size;
  const size: [number, number] = Array.isArray(s)
    ? [Number(s[1]), Number(s[0])]
    : s && typeof s === "object"
      ? [Number((s as { height: number }).height), Number((s as { width: number }).width)]
      : cfg.imageHW;
  if (size[0] !== cfg.imageHW[0] || size[1] !== cfg.imageHW[1]) throw new Error(`Donut: the processor makes ${size[0]} × ${size[1]} images, the encoder takes ${cfg.imageHW.join(" × ")}`);
  const triple = (v: unknown, d: number[]) => (Array.isArray(v) ? v.map(Number) : typeof v === "number" ? [v, v, v] : d);
  return {
    size,
    resample: Number(c.resample ?? 2),
    alignLongAxis: c.do_align_long_axis === true,
    resize: c.do_resize !== false,
    thumbnail: c.do_thumbnail !== false,
    pad: c.do_pad !== false,
    rescale: c.do_rescale === false ? 1 : Number(c.rescale_factor ?? 1 / 255),
    mean: triple(c.image_mean, [0.5, 0.5, 0.5]),
    std: triple(c.image_std, [0.5, 0.5, 0.5]),
  };
}

/** The image as Donut's processor sizes it, before padding (PIL's resampling throughout). */
export function donutResize(img: RgbImage, p: DonutProcessing): RgbImage {
  let im = toRgb(img);
  const [SH, SW] = p.size;
  if (p.alignLongAxis && ((SW < SH && im.width > im.height) || (SW > SH && im.width < im.height))) {
    // np.rot90(image, 3): a quarter turn clockwise.
    const { width: W, height: H } = im;
    const out = new Uint8Array(W * H * 3);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out.set(im.data.subarray((y * W + x) * 3, (y * W + x) * 3 + 3), (x * H + (H - 1 - y)) * 3);
    im = { width: H, height: W, data: out };
  }
  if (p.resize) {
    // The shortest edge to the smaller output side, the other side by the aspect ratio (truncated).
    const edge = Math.min(SH, SW);
    const [w, h] = im.width <= im.height ? [edge, Math.trunc((edge * im.height) / im.width)] : [Math.trunc((edge * im.width) / im.height), edge];
    im = resizeImage(im, w, h, p.resample);
  }
  if (p.thumbnail) {
    // Fit inside the output size (bicubic, reducing_gap 2), never enlarging.
    let h = Math.min(im.height, SH);
    let w = Math.min(im.width, SW);
    if (h !== im.height || w !== im.width) {
      if (im.height > im.width) w = Math.trunc((im.width * h) / im.height);
      else if (im.width > im.height) h = Math.trunc((im.height * w) / im.width);
      im = resizeImageReducing(im, w, h, 3, 2);
    }
  }
  return im;
}

/**
 * Pixel values [3, height, width] as DonutImageProcessor makes them: resized and thumbnailed (see
 * donutResize), padded to the output size with black around the center (left and top get the
 * smaller half), rescaled and normalized in float32.
 */
export function donutPixels(img: RgbImage, p: DonutProcessing): Float32Array {
  const im = donutResize(img, p);
  const [SH, SW] = p.size;
  if (!p.pad && (im.width !== SW || im.height !== SH)) throw new Error(`Donut: a ${im.width} × ${im.height} image without padding does not fit the encoder`);
  const top = Math.floor((SH - im.height) / 2);
  const left = Math.floor((SW - im.width) / 2);
  const f = Math.fround;
  const n = SH * SW;
  const out = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) {
    const m = f(p.mean[c]!);
    const s = f(p.std[c]!);
    const bg = f(f(0 - m) / s);
    out.fill(bg, c * n, (c + 1) * n);
    for (let y = 0; y < im.height; y++)
      for (let x = 0; x < im.width; x++) out[c * n + (y + top) * SW + x + left] = f(f(f(im.data[(y * im.width + x) * 3 + c]! * p.rescale) - m) / s);
  }
  return out;
}

/** What Donut's image-to-text gives: the cleaned sequence and its parse (transformers' `token2json`). */
export interface DonutParse {
  generated_text: string;
  json: unknown;
}

export interface DonutParseOptions {
  /** The decoder prompt (default: the model's task token, e.g. "<s_cord-v2>" or "<s_rvlcdip>"). */
  prompt?: string;
  /** Most tokens to generate (default: up to the decoder's last position, as Donut's examples set max_length). */
  max_new_tokens?: number;
}

/**
 * DonutProcessor.token2json, ported: `<s_key>…</s_key>` pairs become keys (nested when the content
 * holds pairs, a list when items are separated by `<sep/>`), categorical tokens such as
 * `<invoice/>` become their name; text with no pair becomes `{ text_sequence }`.
 */
export function donutToken2Json(tokens: string, addedVocab: ReadonlySet<string>, isInner = false): unknown {
  const output: Record<string, unknown> = {};
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  while (tokens) {
    const start = /<s_(.*?)>/i.exec(tokens);
    if (!start) break;
    const key = start[1]!;
    const end = new RegExp(`</s_${esc(key)}>`, "i").exec(tokens);
    const startToken = start[0];
    if (!end) {
      tokens = tokens.replace(startToken, "");
      continue;
    }
    const endToken = end[0];
    const content = new RegExp(`${esc(startToken)}([\\s\\S]*?)${esc(endToken)}`, "i").exec(tokens);
    if (content) {
      const c = content[1]!.trim();
      if (c.includes("<s_") && c.includes("</s_")) {
        const value = donutToken2Json(c, addedVocab, true) as unknown[];
        if (value.length) output[key] = value.length === 1 ? value[0] : value;
      } else {
        const leaves = c.split("<sep/>").map((l) => {
          const leaf = l.trim();
          return addedVocab.has(leaf) && leaf[0] === "<" && leaf.endsWith("/>") ? leaf.slice(1, -2) : leaf;
        });
        output[key] = leaves.length === 1 ? leaves[0] : leaves;
      }
    }
    tokens = tokens.slice(tokens.indexOf(endToken) + endToken.length).trim();
    if (tokens.startsWith("<sep/>")) return [output, ...(donutToken2Json(tokens.slice(6), addedVocab, true) as unknown[])];
  }
  if (Object.keys(output).length) return isInner ? [output] : output;
  return isInner ? [] : { text_sequence: tokens };
}

/**
 * The model's task token: the one `<s_…>` added token without a closing `</s_…>`, leaving out the
 * pre-training tasks (`<s_iitcdip>`, `<s_synthdog>`); undefined when there is not exactly one.
 */
export function donutTaskToken(added: readonly string[]): string | undefined {
  const set = new Set(added);
  const tasks = added.filter((t) => /^<s_[^>]+>$/.test(t) && !set.has(`</${t.slice(1)}`) && t !== "<s_iitcdip>" && t !== "<s_synthdog>");
  return tasks.length === 1 ? tasks[0] : undefined;
}

export class DonutDocQA {
  private readonly pages = new Map<string, Float32Array>();

  constructor(
    readonly cfg: DonutSwinConfig,
    readonly processing: DonutProcessing,
    readonly processor: ImageProcessorConfig,
    private readonly encoder: DonutSwinModel,
    private readonly decoder: Seq2SeqModel,
    readonly tokenizer: EncoderTokenizer,
    /** The tokenizer's added tokens (the task, field and categorical tokens). */
    readonly added: readonly string[] = [],
  ) {}

  /** The model's task token ("<s_cord-v2>", "<s_rvlcdip>", "<s_docvqa>"…), if it has one. */
  get taskToken(): string | undefined {
    return donutTaskToken(this.added);
  }

  /**
   * Donut's own task on an image, as its model cards run it: the task token as the decoder prompt,
   * greedy decoding with `<unk>` banned, up to the decoder's last position; the sequence decoded with
   * its special tokens, end and padding tokens dropped, the first tag (the task token) removed, and
   * parsed with token2json.
   */
  async parse(img: RgbImage, options: DonutParseOptions = {}): Promise<DonutParse[]> {
    const task = options.prompt ?? this.taskToken;
    if (!task) throw new Error("Donut: this model has no single task token; pass `prompt`");
    const prompt = this.textIds(task);
    const unk = this.tokenizer.tokenId("<unk>");
    const ids = await this.generate(img, prompt, options.max_new_tokens ?? this.decoder.maxTarget, unk === undefined ? [] : [unk]);
    let seq = this.tokenizer.decode([...prompt, ...ids], false).replaceAll("</s>", "").replaceAll("<pad>", "");
    seq = seq.replace(/<.*?>/, "").trim();
    return [{ generated_text: seq, json: donutToken2Json(seq, new Set(this.added)) }];
  }

  /** The encoder's output [tokens, hidden] for an image (kept for the last two images). */
  async encode(img: RgbImage): Promise<Float32Array> {
    const key = imageKey(img);
    let h = this.pages.get(key);
    if (!h) {
      h = await this.encoder.forward(donutPixels(img, this.processing));
      if (this.pages.size >= 2) this.pages.delete(this.pages.keys().next().value!);
      this.pages.set(key, h);
    }
    return h;
  }

  /** Ids of a text without special tokens added (the task tokens inside it are matched as tokens). */
  textIds(text: string): number[] {
    return (this.tokenizer as unknown as { textIds(t: string): number[] }).textIds(text);
  }

  /**
   * Greedy generation after `prompt` (ids) on an image: the encoder output given to the decoder, the
   * prompt fed a token at a time, then tokens until end-of-sequence, `maxNew` of them, or the
   * decoder's last position. Returns the generated ids (end-of-sequence not included).
   */
  async generate(img: RgbImage, prompt: readonly number[], maxNew = 256, banned: readonly number[] = []): Promise<number[]> {
    const h = await this.encode(img);
    const dec = this.decoder;
    await dec.encodeHidden(h, h.length / dec.cfg.dModel);
    if (!prompt.length || prompt.length >= dec.maxTarget) throw new Error(`Donut: a prompt of ${prompt.length} tokens does not fit the decoder's ${dec.maxTarget} positions`);
    const step = banned.length ? { banned } : undefined;
    let tok = 0;
    for (let t = 0; t < prompt.length; t++) tok = await dec.next(prompt[t]!, t, step);
    const out: number[] = [];
    for (let t = prompt.length; tok !== dec.cfg.eos && out.length < maxNew; t++) {
      out.push(tok);
      if (t >= dec.maxTarget) break;
      tok = await dec.next(tok, t, step);
    }
    return out;
  }

  /**
   * Document question answering, as transformers' pipeline does it for Donut: the prompt
   * `<s_docvqa><s_question>{question}</s_question><s_answer>`, greedy decoding, and the text up to the
   * last `</s_answer>` (Python's greedy match) as the answer, null when the model never closes it.
   */
  async answer(img: RgbImage, question: string, options: DocumentQuestionAnsweringOptions = {}): Promise<{ answer: string | null }[]> {
    const prompt = this.textIds(`<s_docvqa><s_question>${question}</s_question><s_answer>`);
    const ids = await this.generate(img, prompt, options.max_new_tokens ?? 256);
    const close = this.tokenizer.tokenId("</s_answer>");
    const end = close === undefined ? -1 : ids.lastIndexOf(close);
    if (end < 0) return [{ answer: null }];
    return [{ answer: this.tokenizer.decode(ids.slice(0, end), false).trim() }];
  }

  destroy(): void {
    this.encoder.destroy();
    this.decoder.destroy();
  }
}

/** Loads Donut (naver-clova-ix/donut-*: safetensors or pytorch_model.bin) with its tokenizer.json. */
export async function createDonut(
  modelId: string,
  ref: ModelRef,
  config: Record<string, unknown>,
  prep: ImageProcessorConfig | undefined,
  names: { names: string[]; format: "safetensors" | "pytorch" },
  opts: VisionLoadOptions,
  files: Promise<Set<string> | undefined>,
  make: (d: DonutDocQA, backend: Backend) => VisionEngine,
): Promise<VisionEngine> {
  const cfg = donutSwinConfigFromHF((config.encoder as Record<string, unknown> | undefined) ?? {});
  const dcfg = visionDecoderConfigFromHF(config);
  if (dcfg.dModel !== donutHidden(cfg)) throw new Error(`${modelId}: encoder width ${donutHidden(cfg)} and decoder width ${dcfg.dModel} differ (a projection is not supported)`);
  const processing = donutProcessing(prep as Record<string, unknown> | undefined, cfg);
  const ep = opts.endpoint;
  const listing = await files;
  if (listing && !listing.has("tokenizer.json")) throw new Error(`${modelId} has no tokenizer.json`);
  const tokJson = (await fetchJSON<EncoderTokenizerJSON>(fileUrl(ref, "tokenizer.json", ep)))!;
  const tokenizer = encoderTokenizerFromJSON(tokJson);
  const added = (tokJson.added_tokens ?? []).map((t) => t.content);
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const encFiles: WeightFile[] = [];
    const decFiles: WeightFile[] = [];
    const encRename = (n: string) => (n.startsWith("encoder.") ? donutSwinName(n.slice(8)) : undefined);
    // The decoder's lm_head is its token embedding (tied), so only the embedding loads.
    const decRename = (n: string) => (n.startsWith("decoder.model.") ? seq2seqName(n.slice(14)) : undefined);
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
      encFiles.push({ name, source, tensors: header.tensors.filter((t) => encRename(t.name) !== undefined), rename: encRename });
      decFiles.push({ name, source, tensors: header.tensors.filter((t) => decRename(t.name) !== undefined), rename: decRename });
    }
    const need = async (name: string) => {
      const t = located.get(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
    };
    const quant = opts.quant ?? "f32";
    const encWeights = await loadWeights(backend, encFiles, quant, opts.onProgress);
    const decWeights = await loadWeights(backend, decFiles, quant, opts.onProgress);
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `donut/${name}`);
      backend.write(buf, 0, data);
      encWeights.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    // Patch convolution [embed, 3, p, p] as [embed, patchK], zero-padded.
    const pw = await need("encoder.embeddings.patch_embeddings.projection.weight");
    const E = cfg.embedDim;
    const K = donutPatchK(cfg);
    const row = 3 * cfg.patchSize * cfg.patchSize;
    const patch = new Float32Array(E * K);
    for (let o = 0; o < E; o++) patch.set(pw.data.subarray(o * row, (o + 1) * row), o * K);
    upload("patch.weight", [E, K], patch);
    upload("patch.bias", [E], (await need("encoder.embeddings.patch_embeddings.projection.bias")).data);
    // Each block's relative-position table expanded to [heads, w², w²].
    const N = cfg.window * cfg.window;
    for (let s = 0; s < cfg.depths.length; s++)
      for (let b = 0; b < cfg.depths[s]!; b++) {
        const t = await need(`encoder.encoder.layers.${s}.blocks.${b}.attention.self.relative_position_bias_table`);
        upload(`s${s}.b${b}.bias`, [cfg.heads[s]!, N, N], swinRelativeBias(t.data, cfg.heads[s]!, cfg.window));
      }
    const encoder = new DonutSwinModel(backend, cfg, encWeights);
    const decoder = new Seq2SeqModel(backend, dcfg, decWeights);
    const processor: ImageProcessorConfig = { ...(prep ?? {}), size: { height: processing.size[0], width: processing.size[1] } };
    return make(new DonutDocQA(cfg, processing, processor, encoder, decoder, tokenizer, added), backend);
  } finally {
    for (const s of sources) s.close?.();
  }
}
