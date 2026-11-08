// Encoder models (BERT, DistilBERT, RoBERTa and their fine-tunes) and the text tasks transformers.js
// runs on them, with its output shapes: feature-extraction, text-classification,
// token-classification, question-answering, fill-mask and zero-shot-classification. The body runs on
// the backend (GPU or CPU); the small task heads run here in TypeScript on the final hidden states.
import type { Backend } from "@kiln/core";
import { fetchJSON, fetchText, fileUrl, listRepoFiles, parseModelId, readPytorchHeader, readSafetensorsHeader, toF32, weightFiles, type ByteSource, type TensorEntry } from "@kiln/loader";
import { EncoderModel, encoderConfigFromHF, encoderName, type EncoderConfig } from "@kiln/models";
import { encoderTokenizerFromJSON, wordPieceFromVocab, type EncoderTokenizer, type EncoderTokenizerJSON, type Encoding } from "@kiln/tokenizers";
import { openBackend, openParsed } from "./load";
import { loadWeights, type LoadProgress, type WeightFile } from "./weights";

export interface EncoderLoadOptions {
  /** Matrix storage: "f32" (default; encoders are small), "q8" or "q4". Embeddings, norms and biases stay f32. */
  quant?: "f32" | "q8" | "q4";
  device?: "auto" | "gpu" | "cpu";
  cpuThreads?: number;
  revision?: string;
  endpoint?: string;
  backend?: Backend;
  onProgress?: (p: LoadProgress) => void;
}

/** A float tensor as transformers.js returns one (row-major `data`, shape `dims`). */
export interface FloatTensor {
  type: "float32";
  data: Float32Array;
  dims: number[];
}

export interface Label {
  label: string;
  score: number;
}

export interface TokenEntity {
  entity: string;
  score: number;
  index: number;
  word: string;
}

export interface EntityGroup {
  entity_group: string;
  score: number;
  word: string;
}

export interface Answer {
  answer: string;
  score: number;
}

export interface MaskFill {
  score: number;
  token: number;
  token_str: string;
  sequence: string;
}

export interface ZeroShotResult {
  sequence: string;
  labels: string[];
  scores: number[];
}

type Batch = string | string[];
const many = (x: Batch): string[] => (Array.isArray(x) ? x : [x]);

function softmax(x: ArrayLike<number>): number[] {
  let m = -Infinity;
  for (let i = 0; i < x.length; i++) m = Math.max(m, x[i]!);
  const e = Array.from(x, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Indices of the k largest values, largest first (k = -1 or undefined for all). */
function topk(x: readonly number[], k?: number | null): number[] {
  const idx = x.map((_, i) => i).sort((a, b) => x[b]! - x[a]!);
  return k === undefined || k === null || k < 0 ? idx : idx.slice(0, k);
}

/** y = W x + b for W [N, K] row-major. */
function matvec(w: Float32Array, b: Float32Array | undefined, x: Float32Array, N: number, K: number): Float32Array {
  const y = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    let s = b ? b[n]! : 0;
    const o = n * K;
    for (let k = 0; k < K; k++) s += w[o + k]! * x[k]!;
    y[n] = s;
  }
  return y;
}


const tanhInPlace = (x: Float32Array) => x.forEach((v, i) => (x[i] = Math.tanh(v)));
const reluInPlace = (x: Float32Array) => x.forEach((v, i) => (x[i] = Math.max(0, v)));

interface Located {
  source: ByteSource;
  entry: TensorEntry;
}

/** A loaded encoder: tokenizer, body on the backend, heads read from the weight files on first use. */
export class EncoderEngine {
  private readonly heads = new Map<string, Float32Array | undefined>();

  constructor(
    readonly backend: Backend,
    readonly cfg: EncoderConfig,
    readonly tokenizer: EncoderTokenizer,
    readonly model: EncoderModel,
    private readonly tensors: Map<string, Located>,
    private readonly sources: ByteSource[],
    /** Longest input in tokens, special tokens included. */
    readonly maxLength: number,
  ) {}

  /** A head or embedding tensor as f32, trying the family's prefix and old gamma / beta names. */
  private async read(name: string): Promise<Float32Array | undefined> {
    if (this.heads.has(name)) return this.heads.get(name);
    const alts = [name, `${this.cfg.family}.${name}`];
    for (const n of [...alts]) {
      if (n.endsWith(".weight")) alts.push(n.replace(/\.weight$/, ".gamma"));
      if (n.endsWith(".bias")) alts.push(n.replace(/\.bias$/, ".beta"));
    }
    let out: Float32Array | undefined;
    for (const n of alts) {
      const t = this.tensors.get(n);
      if (!t) continue;
      const count = t.entry.shape.reduce((a, b) => a * b, 1);
      out = toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), count);
      break;
    }
    this.heads.set(name, out);
    return out;
  }

  private async need(name: string): Promise<Float32Array> {
    const t = await this.read(name);
    if (!t) throw new Error(`this model has no "${name}" (is it fine-tuned for this task?)`);
    return t;
  }

  /** Token ids and final hidden states of a text or a pair of texts. */
  async encode(a: string, b?: string): Promise<{ enc: Encoding; h: Float32Array }> {
    return (await this.encodeMany([[a, b]]))[0]!;
  }

  /** The same for several inputs, their forward passes queued back to back (see forwardMany). */
  async encodeMany(inputs: readonly (readonly [string, string?])[]): Promise<{ enc: Encoding; h: Float32Array }[]> {
    const encs = inputs.map(([a, b]) => this.tokenizer.encode(a, b, this.maxLength));
    const hs = await this.model.forwardMany(encs.map((e) => ({ ids: e.ids, typeIds: e.typeIds })));
    return encs.map((enc, i) => ({ enc, h: hs[i]! }));
  }

  private row(h: Float32Array, t: number): Float32Array {
    const H = this.cfg.hidden;
    return h.subarray(t * H, (t + 1) * H);
  }

  /** Sequence-classification logits from the first token, through the family's head. */
  private async classify(h: Float32Array): Promise<Float32Array> {
    const H = this.cfg.hidden;
    const cls = this.row(h, 0);
    const L = (await this.need("classifier.bias")).length;
    if (this.cfg.family === "roberta") {
      const x = matvec(await this.need("classifier.dense.weight"), await this.read("classifier.dense.bias"), cls, H, H);
      tanhInPlace(x);
      return matvec(await this.need("classifier.out_proj.weight"), await this.read("classifier.out_proj.bias"), x, (await this.need("classifier.out_proj.bias")).length, H);
    }
    if (this.cfg.family === "distilbert") {
      const x = matvec(await this.need("pre_classifier.weight"), await this.read("pre_classifier.bias"), cls, H, H);
      reluInPlace(x);
      return matvec(await this.need("classifier.weight"), await this.read("classifier.bias"), x, L, H);
    }
    const pooled = matvec(await this.need("pooler.dense.weight"), await this.read("pooler.dense.bias"), cls, H, H);
    tanhInPlace(pooled);
    return matvec(await this.need("classifier.weight"), await this.read("classifier.bias"), pooled, L, H);
  }

  private label(i: number): string {
    return this.cfg.id2label?.[i] ?? `LABEL_${i}`;
  }

  async featureExtraction(texts: Batch, opts: { pooling?: "none" | "mean" | "cls" | "first_token"; normalize?: boolean } = {}): Promise<FloatTensor> {
    const H = this.cfg.hidden;
    const pooling = opts.pooling ?? "none";
    const outs: Float32Array[] = [];
    let longest = 0;
    for (const { enc, h } of await this.encodeMany(many(texts).map((t) => [t] as const))) {
      longest = Math.max(longest, enc.ids.length);
      if (pooling === "none") outs.push(h);
      else if (pooling === "mean") {
        const m = new Float32Array(H);
        const T = enc.ids.length;
        for (let i = 0; i < T; i++) for (let d = 0; d < H; d++) m[d]! += h[i * H + d]!;
        outs.push(m.map((v) => v / T));
      } else if (pooling === "cls" || pooling === "first_token") outs.push(h.slice(0, H));
      else throw new Error(`pooling "${String(pooling)}" is not supported (none, mean, cls)`);
    }
    if (opts.normalize) {
      for (const o of outs) {
        // Per vector of H values (per token without pooling).
        for (let r = 0; r < o.length / H; r++) {
          const v = o.subarray(r * H, (r + 1) * H);
          const n = Math.max(Math.hypot(...v), 1e-12);
          for (let d = 0; d < H; d++) v[d]! /= n;
        }
      }
    }
    if (pooling !== "none") {
      const data = new Float32Array(outs.length * H);
      outs.forEach((o, i) => data.set(o, i * H));
      return { type: "float32", data, dims: [outs.length, H] };
    }
    // Rows past a shorter text's length stay zero (transformers.js would hold padding there).
    const data = new Float32Array(outs.length * longest * H);
    outs.forEach((o, i) => data.set(o, i * longest * H));
    return { type: "float32", data, dims: [outs.length, longest, H] };
  }

  async textClassification(texts: Batch, opts: { top_k?: number | null } = {}): Promise<Label[] | Label[][]> {
    const k = opts.top_k === undefined ? 1 : opts.top_k;
    const out: (Label | Label[])[] = [];
    for (const { h } of await this.encodeMany(many(texts).map((t) => [t] as const))) {
      const logits = await this.classify(h);
      const probs = this.cfg.multiLabel ? Array.from(logits, (v) => 1 / (1 + Math.exp(-v))) : softmax(logits);
      const vals = topk(probs, k).map((i) => ({ label: this.label(i), score: probs[i]! }));
      if (k === 1) out.push(...vals);
      else out.push(vals);
    }
    return (Array.isArray(texts) || k === 1 ? out : out[0]) as Label[] | Label[][];
  }

  async tokenClassification(texts: Batch, opts: { ignore_labels?: string[]; aggregation_strategy?: "none" | "simple" } = {}): Promise<(TokenEntity | EntityGroup)[] | (TokenEntity | EntityGroup)[][]> {
    const ignore = opts.ignore_labels ?? ["O"];
    const agg = opts.aggregation_strategy ?? "none";
    if (agg !== "none" && agg !== "simple") throw new Error(`aggregation_strategy "${String(agg)}" must be "none" or "simple"`);
    const H = this.cfg.hidden;
    const w = await this.need("classifier.weight");
    const b = await this.read("classifier.bias");
    const L = w.length / H;
    const out: (TokenEntity | EntityGroup)[][] = [];
    for (const { enc, h } of await this.encodeMany(many(texts).map((t) => [t] as const))) {
      const tokens: TokenEntity[] = [];
      for (let j = 0; j < enc.ids.length; j++) {
        const probs = softmax(matvec(w, b, this.row(h, j), L, H));
        const top = topk(probs, 1)[0]!;
        const entity = this.label(top);
        if (ignore.includes(entity)) continue;
        const word = this.tokenizer.decode([enc.ids[j]!], true);
        if (word === "") continue;
        tokens.push({ entity, score: probs[top]!, index: j, word });
      }
      out.push(agg === "simple" ? this.group(tokens, enc.ids) : tokens);
    }
    return Array.isArray(texts) ? out : out[0]!;
  }

  /** transformers.js's "simple" aggregation: consecutive tokens of one tag (B-/I-/E-/S- prefixes) as one entity. */
  private group(tokens: TokenEntity[], ids: number[]): EntityGroup[] {
    const tag = (e: string): [string, string] => (e[1] === "-" && "BIES".includes(e[0]!) ? [e[0]!, e.slice(2)] : ["I", e]);
    const groups: { tag: string; start: number; end: number }[] = [];
    let open: string | null = null;
    tokens.forEach((t, i) => {
      const [p, g] = tag(t.entity);
      if (open === g && p !== "B" && p !== "S") {
        groups.at(-1)!.end = i + 1;
        if (p === "E") open = null;
      } else {
        groups.push({ tag: g, start: i, end: i + 1 });
        open = p === "S" ? null : g;
      }
    });
    return groups.map(({ tag: g, start, end }) => {
      const part = tokens.slice(start, end);
      return {
        entity_group: g,
        score: part.reduce((a, t) => a + t.score, 0) / part.length,
        word: this.tokenizer.decode(part.map((t) => ids[t.index]!), true),
      };
    });
  }

  async questionAnswering(question: Batch, context: Batch, opts: { top_k?: number } = {}): Promise<Answer | Answer[] | (Answer | Answer[])[]> {
    const k = opts.top_k ?? 1;
    const H = this.cfg.hidden;
    const w = await this.need("qa_outputs.weight");
    const b = await this.read("qa_outputs.bias");
    const qs = many(question);
    const cs = Array.isArray(context) ? context : qs.map(() => context);
    const out: (Answer | Answer[])[] = [];
    const encoded = await this.encodeMany(qs.map((q, i) => [q, cs[i]] as const));
    for (let q = 0; q < qs.length; q++) {
      const { enc, h } = encoded[q]!;
      const T = enc.ids.length;
      const start = new Array<number>(T);
      const end = new Array<number>(T);
      for (let t = 0; t < T; t++) {
        const se = matvec(w, b, this.row(h, t), 2, H);
        // Only context tokens can start or end an answer (the first token's score is zeroed below, as transformers.js does).
        const ok = t === 0 || (enc.seq[t] === 1 && !this.tokenizer.specialIds.has(enc.ids[t]!));
        start[t] = ok ? se[0]! : -Infinity;
        end[t] = ok ? se[1]! : -Infinity;
      }
      const ps = softmax(start);
      const pe = softmax(end);
      ps[0] = 0;
      pe[0] = 0;
      const options: [number, number, number][] = [];
      for (let i = 0; i < T; i++) for (let j = i; j < T; j++) if (ps[i]! * pe[j]! > 0) options.push([i, j, ps[i]! * pe[j]!]);
      options.sort((x, y) => y[2] - x[2]);
      const res = options.slice(0, k).map(([i, j, s]) => ({ answer: this.tokenizer.decode(enc.ids.slice(i, j + 1), true), score: s }));
      if (k === 1) out.push(...res);
      else out.push(res);
    }
    return Array.isArray(question) ? out : (out[0] as Answer | Answer[]);
  }

  /** Vocabulary logits at one position through the family's masked-LM head (output layer tied to the embeddings unless stored). */
  private async maskLogits(x: Float32Array): Promise<Float32Array> {
    const f = this.cfg.family;
    const [dense, ln, bias, decoder] =
      f === "distilbert"
        ? ["vocab_transform", "vocab_layer_norm", "vocab_projector.bias", "vocab_projector.weight"]
        : f === "roberta"
          ? ["lm_head.dense", "lm_head.layer_norm", "lm_head.bias", "lm_head.decoder.weight"]
          : ["cls.predictions.transform.dense", "cls.predictions.transform.LayerNorm", "cls.predictions.bias", "cls.predictions.decoder.weight"];
    // On the backend: the vocabulary matmul (30k × 768 for BERT) is most of the work. A decoder that
    // is not stored is the word embedding table, already there.
    return this.model.maskLogits(x, {
      dense: await this.need(`${dense}.weight`),
      denseBias: await this.read(`${dense}.bias`),
      lnW: await this.need(`${ln}.weight`),
      lnB: await this.need(`${ln}.bias`),
      eps: f === "distilbert" ? 1e-12 : this.cfg.eps,
      exact: this.cfg.act === "gelu",
      decoder: await this.read(decoder),
      bias: await this.read(bias),
    });
  }

  async fillMask(texts: Batch, opts: { top_k?: number } = {}): Promise<MaskFill[] | MaskFill[][]> {
    const k = opts.top_k ?? 5;
    const mask = this.tokenizer.maskToken;
    const maskId = mask === undefined ? undefined : this.tokenizer.tokenId(mask);
    if (maskId === undefined) throw new Error("this tokenizer has no mask token");
    const out: MaskFill[][] = [];
    for (const { enc, h } of await this.encodeMany(many(texts).map((t) => [t] as const))) {
      const at = enc.ids.indexOf(maskId);
      if (at < 0) throw new Error(`Mask token (${mask}) not found in text.`);
      const probs = softmax(await this.maskLogits(this.row(h, at)));
      out.push(
        topk(probs, k).map((id) => {
          const seq = enc.ids.slice();
          seq[at] = id;
          return { score: probs[id]!, token: id, token_str: this.tokenizer.decode([id], false), sequence: this.tokenizer.decode(seq, true) };
        }),
      );
    }
    return Array.isArray(texts) ? out : out[0]!;
  }

  async zeroShotClassification(
    texts: Batch,
    labels: string | string[],
    opts: { hypothesis_template?: string; multi_label?: boolean } = {},
  ): Promise<ZeroShotResult | ZeroShotResult[]> {
    const ids = Object.fromEntries(Object.entries(this.cfg.id2label ?? {}).map(([i, l]) => [l.toLowerCase(), Number(i)]));
    const entail = ids["entailment"] ?? 2;
    const contra = ids["contradiction"] ?? ids["not_entailment"] ?? 0;
    const cands = Array.isArray(labels) ? labels : [labels];
    const template = opts.hypothesis_template ?? "This example is {}.";
    const each = !!opts.multi_label || cands.length === 1;
    const out: ZeroShotResult[] = [];
    for (const premise of many(texts)) {
      const logits: Float32Array[] = [];
      for (const { h } of await this.encodeMany(cands.map((c) => [premise, template.replace("{}", c)] as const))) logits.push(await this.classify(h));
      const scores = each ? logits.map((l) => softmax([l[contra]!, l[entail]!])[1]!) : softmax(logits.map((l) => l[entail]!));
      const order = topk(scores);
      out.push({ sequence: premise, labels: order.map((i) => cands[i]!), scores: order.map((i) => scores[i]!) });
    }
    return Array.isArray(texts) ? out : out[0]!;
  }

  destroy(): void {
    this.model.destroy();
    for (const s of this.sources) s.close?.();
    this.heads.clear();
  }
}

/** Loads an encoder repo ("org/repo", safetensors) for the text tasks. */
export async function createEncoder(modelId: string, opts: EncoderLoadOptions = {}): Promise<EncoderEngine> {
  const ref = parseModelId(modelId, opts.revision);
  if (ref.gguf) throw new Error("encoder models load from safetensors repos, not GGUF files");
  const ep = opts.endpoint;
  const listing = listRepoFiles(ref, ep);
  const has = async (f: string) => {
    const files = await listing;
    return !files || files.has(f);
  };
  const [config, tokJson, tokCfg, vocabTxt, names] = await Promise.all([
    fetchJSON<Record<string, unknown>>(fileUrl(ref, "config.json", ep)),
    has("tokenizer.json").then((y) => (y ? fetchJSON<EncoderTokenizerJSON>(fileUrl(ref, "tokenizer.json", ep), true) : undefined)),
    has("tokenizer_config.json").then((y) => (y ? fetchJSON<Record<string, unknown>>(fileUrl(ref, "tokenizer_config.json", ep), true) : undefined)),
    has("vocab.txt").then((y) => (y ? fetchText(fileUrl(ref, "vocab.txt", ep), true) : undefined)),
    listing.then((files) => weightFiles(ref, ep, files)),
  ]);
  const cfg = encoderConfigFromHF(config!);
  let tokenizer: EncoderTokenizer;
  if (tokJson) tokenizer = encoderTokenizerFromJSON(tokJson);
  else if (vocabTxt) tokenizer = wordPieceFromVocab(vocabTxt, { lowercase: (tokCfg?.do_lower_case as boolean | undefined) ?? true, stripAccents: tokCfg?.strip_accents as boolean | null | undefined });
  else throw new Error(`${modelId} has neither tokenizer.json nor vocab.txt`);
  const declared = Number(tokCfg?.model_max_length);
  const maxLength = Math.min(cfg.maxPositions, Number.isFinite(declared) && declared > 0 && declared < 1e6 ? declared : Infinity);

  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  const tensors = new Map<string, Located>();
  try {
    const files: WeightFile[] = [];
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) tensors.set(t.name, { source, entry: t });
      files.push({ name, source, tensors: header.tensors.filter((t) => encoderName(t.name) !== undefined), rename: encoderName });
    }
    const weights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);
    return new EncoderEngine(backend, cfg, tokenizer, new EncoderModel(backend, cfg, weights), tensors, sources, maxLength);
  } catch (err) {
    for (const s of sources) s.close?.();
    throw err;
  }
}
