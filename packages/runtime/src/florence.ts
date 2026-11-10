// Florence-2 (microsoft/Florence-2-*): the DaViT vision tower, the image tokens ahead of the task
// prompt's in the BART encoder, greedy generation with the model's processors, and the
// processor's post_process_generation.
import type { Tensor } from "@kiln/core";
import { fetchJSON, fileUrl, preprocessImage, readPytorchHeader, readSafetensorsHeader, toF32, type ImageProcessorConfig, type ModelRef, type RgbImage, type ByteSource, type TensorEntry } from "@kiln/loader";
import { DavitModel, FLORENCE2_ADDED_TOKENS, Seq2SeqModel, convMatrix, davitName, davitPositions, florence2ConfigFromHF, florence2Parse, florence2Prompt, seq2seqName, type Florence2Config } from "@kiln/models";
import { encoderTokenizerFromJSON, type EncoderTokenizerJSON } from "@kiln/tokenizers";
import { openBackend, openParsed } from "./load";
import { loadWeights, type WeightFile } from "./weights";
import { Seq2SeqEngine } from "./seq2seq";
import type { VisionEngine, VisionLoadOptions } from "./vision";

export interface FlorenceOptions {
  /** The task token ("<CAPTION>", "<OD>", "<OCR>", …; default "<CAPTION>"), or a prompt of your own. */
  task?: string;
  /** Text input for tasks that take one ("<CAPTION_TO_PHRASE_GROUNDING>", "<OPEN_VOCABULARY_DETECTION>", …). */
  text_input?: string;
  /** Most tokens to generate (default 1024, as Microsoft's examples). */
  max_new_tokens?: number;
}

/** Florence-2's parts, held by its VisionEngine. */
export class Florence2 {
  constructor(
    readonly cfg: Florence2Config,
    readonly processor: ImageProcessorConfig,
    readonly vision: DavitModel,
    readonly engine: Seq2SeqEngine,
    /** image_projection [C, P] and image_proj_norm, for the mean token and the LayerNorm. */
    private readonly projection: Float32Array,
    private readonly norm: { w: Float32Array; b: Float32Array },
    /** The padding id's embedding, subtracted from the image rows (they are added onto it). */
    private readonly padRow: Float32Array,
  ) {}

  /**
   * Runs a task on an image: [{ generated_text, [task]: parsed answer }] — the raw text with special
   * tokens, and post_process_generation's result (boxes in the image's pixels).
   */
  async run(img: RgbImage, o: FlorenceOptions = {}): Promise<{ generated_text: string; [task: string]: unknown }[]> {
    const task = o.task ?? "<CAPTION>";
    const p = preprocessImage(img, this.processor);
    const S = this.cfg.imageSize;
    if (p.width !== S || p.height !== S) throw new Error(`Florence-2: the image processor made a ${p.width}×${p.height} image; the model takes ${S}×${S}`);
    const tokens = await this.vision.imageTokens(p.data, this.projection, this.norm);
    const d = this.cfg.text.dModel;
    const P = tokens.length / d;
    const prompt = this.engine.tokenizer.encode(florence2Prompt(task + (o.text_input ?? "")), undefined, this.cfg.text.maxPositions - P).ids;
    const ids = [...new Array<number>(P).fill(this.cfg.text.pad), ...prompt];
    const extra = new Float32Array(ids.length * d);
    for (let r = 0; r < P; r++) for (let i = 0; i < d; i++) extra[r * d + i] = tokens[r * d + i]! - this.padRow[i]!;
    const model = this.engine.model as Seq2SeqModel;
    const out = await this.engine.generate([], { max_new_tokens: o.max_new_tokens ?? 1024 }, () => model.encodePrefixed(ids, extra));
    const text = this.engine.tokenizer.decode(out, false);
    return [{ generated_text: text, ...florence2Parse(text, task, img.width, img.height) }];
  }

  destroy(): void {
    this.vision.destroy();
    this.engine.destroy();
  }
}

/**
 * Loads a Florence-2 checkpoint (microsoft/Florence-2-base, -large and their -ft versions): the DaViT
 * tower's convolutions as conv matrices (the first over 4 input channels), depthwise kernels as
 * [9, C], the position table precomputed, image_projection transposed for the GPU; the language
 * model's weights under the seq2seq names.
 */
export async function createFlorence(
  modelId: string,
  ref: ModelRef,
  config: Record<string, unknown>,
  prep: ImageProcessorConfig | undefined,
  names: { names: string[]; format: "safetensors" | "pytorch" },
  opts: VisionLoadOptions,
  make: (f: Florence2, backend: import("@kiln/core").Backend) => VisionEngine,
): Promise<VisionEngine> {
  const cfg = florence2ConfigFromHF(config);
  const ep = opts.endpoint;
  const tokJson = (await fetchJSON<EncoderTokenizerJSON>(fileUrl(ref, "tokenizer.json", ep)))!;
  // Microsoft's tokenizer.json lacks the task and location tokens the processor adds at load.
  const added = (tokJson.added_tokens ??= []);
  if (!added.some((t) => t.content === "<loc_0>")) {
    const vocab = (tokJson.model as { vocab?: Record<string, number> }).vocab ?? {};
    let next = Math.max(...Object.values(vocab), ...added.map((t) => t.id)) + 1;
    for (const content of FLORENCE2_ADDED_TOKENS) added.push({ id: next++, content, special: true });
  }
  const tokenizer = encoderTokenizerFromJSON(tokJson);
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const located = new Map<string, { source: ByteSource; entry: TensorEntry }>();
  try {
    const files: WeightFile[] = [];
    const rename = (n: string) => (n.startsWith("language_model.") ? seq2seqName(n.slice("language_model.".length)) : undefined);
    let shared = false;
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) located.set(t.name, { source, entry: t });
      const tensors = header.tensors.filter((t) => {
        const n = rename(t.name);
        if (n === "lm_head.weight") return false;
        if (n !== "shared.weight") return n !== undefined;
        if (shared) return false;
        return (shared = true);
      });
      files.push({ name, source, tensors, rename });
    }
    const need = async (name: string) => {
      const t = located.get(name);
      if (!t) throw new Error(`${modelId}: "${name}" is missing`);
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      return { shape: t.entry.shape, data: toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count) };
    };
    const textWeights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);

    const vw = new Map<string, Tensor>();
    const upload = (name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `davit/${name}`);
      backend.write(buf, 0, data);
      vw.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    for (const name of located.keys()) {
      const to = davitName(name);
      if (!to) continue;
      const t = await need(name);
      if (/^conv\d+\.weight$/.test(to)) {
        const m = convMatrix(t.data, t.shape, to === "conv0.weight" ? 4 : t.shape[1]);
        upload(to, [t.shape[0]!, m.K], m.w);
      } else if (/\.dw[12]\.weight$/.test(to)) {
        // [C, 1, 3, 3] → [9, C].
        const C = t.shape[0]!;
        const taps = t.data.length / C;
        const m = new Float32Array(taps * C);
        for (let c = 0; c < C; c++) for (let k = 0; k < taps; k++) m[k * C + c] = t.data[c * taps + k]!;
        upload(to, [taps, C], m);
      } else upload(to, t.shape, t.data);
    }
    const vision = new DavitModel(backend, cfg, vw);
    const C = cfg.vision.dims[cfg.vision.dims.length - 1]!;
    const Pd = cfg.vision.projection;
    const [row, col, proj, nw, nb] = await Promise.all(["image_pos_embed.row_embeddings.weight", "image_pos_embed.column_embeddings.weight", "image_projection", "image_proj_norm.weight", "image_proj_norm.bias"].map(need));
    upload("pos", [vision.side * vision.side, C], davitPositions(row!.data, col!.data, vision.side, C));
    const pt = new Float32Array(Pd * C);
    for (let i = 0; i < C; i++) for (let o = 0; o < Pd; o++) pt[o * C + i] = proj!.data[i * Pd + o]!;
    upload("projection", [Pd, C], pt);
    const embed = await need("language_model.model.shared.weight");
    const d = cfg.text.dModel;
    const padRow = embed.data.slice(cfg.text.pad * d, (cfg.text.pad + 1) * d);

    const text = (config.text_config as Record<string, unknown> | undefined) ?? {};
    const gen = Object.fromEntries(Object.entries(text).filter(([k]) => /length|ngram|token_id|penalty|bad_words|num_beams/.test(k)));
    const seq = new Seq2SeqModel(backend, cfg.text, textWeights, { prefixed: true });
    const engine = new Seq2SeqEngine(backend, cfg.text, tokenizer, seq, gen);
    const processor: ImageProcessorConfig = prep ?? { size: { height: 768, width: 768 }, resample: 3, do_center_crop: false, image_mean: [0.485, 0.456, 0.406], image_std: [0.229, 0.224, 0.225] };
    return make(new Florence2(cfg, processor, vision, engine, proj!.data, { w: nw!.data, b: nb!.data }, padRow), backend);
  } finally {
    for (const s of sources) s.close?.();
  }
}
