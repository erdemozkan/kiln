// transformers.js's lower-level API on Kiln's own pieces, so code written for it moves over by
// changing the import: `AutoTokenizer` (with `PreTrainedTokenizer`), `RawImage`, `read_audio` and
// `env`. Tokenizers run on the calling thread (they are plain TypeScript); models stay in the worker.
import { fetchJSON, fetchText, fileUrl, listRepoFiles, parseModelId, readAudio, readImage, resizeImage, type AudioInput, type ImageInput, type RgbImage } from "@kiln/loader";
import {
  ClipTokenizer,
  byteToUnicode,
  RobertaTokenizer,
  TemplatedTokenizer,
  Template,
  Tokenizer,
  cleanup,
  byteLevelJSONFromVocabMerges,
  encoderTokenizerFromJSON,
  wordPieceJSONFromVocab,
  type ClipTokenizerJSON,
  type EncoderTokenizer,
  type EncoderTokenizerJSON,
  type TokenizerJSON,
} from "@kiln/tokenizers";
import { Tensor } from "./tensor";

/**
 * transformers.js's `env`. Kiln reads `remoteHost` (a mirror of the Hub, used by `AutoTokenizer` and
 * `pipeline` unless a call passes `endpoint`); the other settings are accepted and have no effect:
 * Kiln always loads from the Hub (or `remoteHost`) and caches weights in OPFS itself.
 */
export const env: {
  remoteHost: string;
  remotePathTemplate: string;
  allowRemoteModels: boolean;
  allowLocalModels: boolean;
  localModelPath: string;
  useBrowserCache: boolean;
  useFSCache: boolean;
  cacheDir: string | null;
  backends: Record<string, unknown>;
  [key: string]: unknown;
} = {
  remoteHost: "https://huggingface.co/",
  remotePathTemplate: "{model}/resolve/{revision}/",
  allowRemoteModels: true,
  allowLocalModels: false,
  localModelPath: "/models/",
  useBrowserCache: true,
  useFSCache: false,
  cacheDir: null,
  backends: { onnx: { wasm: {}, webgpu: {} } },
};

/** The Hub endpoint `env.remoteHost` names, or undefined for the Hub itself. */
export function envEndpoint(): string | undefined {
  const h = env.remoteHost.replace(/\/+$/, "");
  return h === "https://huggingface.co" ? undefined : h;
}

type TokenConfig = string | { content: string; __type?: string } | null | undefined;

/** tokenizer_config.json, as far as the tokenizer reads it. */
export interface TokenizerConfig {
  tokenizer_class?: string;
  chat_template?: string | { name: string; template: string }[];
  model_max_length?: number;
  padding_side?: "left" | "right";
  clean_up_tokenization_spaces?: boolean;
  add_bos_token?: boolean;
  add_eos_token?: boolean;
  bos_token?: TokenConfig;
  eos_token?: TokenConfig;
  unk_token?: TokenConfig;
  sep_token?: TokenConfig;
  pad_token?: TokenConfig;
  cls_token?: TokenConfig;
  mask_token?: TokenConfig;
  [key: string]: unknown;
}

export interface PretrainedOptions {
  revision?: string;
  /** Hub endpoint (a mirror or a local server); `env.remoteHost` by default. */
  endpoint?: string;
  /** Accepted for transformers.js compatibility; the tokenizer's two small files load without progress events. */
  progress_callback?: unknown;
  [key: string]: unknown;
}

export interface TokenizerCallOptions {
  text_pair?: string | string[] | null;
  /** true pads to the longest input, "max_length" to `max_length`. */
  padding?: boolean | "max_length";
  add_special_tokens?: boolean;
  truncation?: boolean | null;
  max_length?: number | null;
  /** Tensors (default) or plain arrays. */
  return_tensor?: boolean;
  return_token_type_ids?: boolean | null;
}

export interface BatchEncoding {
  input_ids: Tensor | number[] | number[][];
  attention_mask: Tensor | number[] | number[][];
  token_type_ids?: Tensor | number[] | number[][];
}

export interface ChatTemplateOptions {
  chat_template?: string | null;
  tools?: unknown[] | null;
  documents?: Record<string, string>[] | null;
  add_generation_prompt?: boolean;
  /** Token ids (default) or the rendered prompt. */
  tokenize?: boolean;
  padding?: boolean;
  truncation?: boolean;
  max_length?: number | null;
  return_tensor?: boolean;
  /** `{ input_ids, attention_mask }` (default) or `input_ids` alone. */
  return_dict?: boolean;
  tokenizer_kwargs?: TokenizerCallOptions;
  [key: string]: unknown;
}

const SPECIAL_TOKEN_ATTRIBUTES = ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"] as const;
/** transformers.js's tokenizer classes that return token type ids by default. */
const TOKEN_TYPE_CLASSES = new Set(["BertTokenizer", "AlbertTokenizer", "ElectraTokenizer", "DebertaTokenizer", "DebertaV2Tokenizer", "MobileBertTokenizer", "SqueezeBertTokenizer", "ConvBertTokenizer", "RoFormerTokenizer", "HerbertTokenizer", "XLMTokenizer"]);

function tokenFromConfig(cfg: TokenizerConfig, ...keys: string[]): string | null {
  for (const k of keys) {
    const v = cfg[k] as TokenConfig;
    if (!v) continue;
    return typeof v === "string" ? v : v.content;
  }
  return null;
}

/**
 * The post-processor the encoder tokenizers understand: a Sequence (Llama 3's ByteLevel then
 * TemplateProcessing) reduces to the step that adds special tokens.
 */
function flattenPostProcessor(pp: EncoderTokenizerJSON["post_processor"]): EncoderTokenizerJSON["post_processor"] {
  if (pp?.type !== "Sequence") return pp;
  const steps = ((pp as { processors?: NonNullable<EncoderTokenizerJSON["post_processor"]>[] }).processors ?? []).filter((p) => p.type !== "ByteLevel");
  return steps.length ? flattenPostProcessor(steps[steps.length - 1]) : null;
}

function isLlmBPE(json: EncoderTokenizerJSON): boolean {
  const type = json.model.type ?? (json.model.merges ? "BPE" : Array.isArray(json.model.vocab) ? "Unigram" : "WordPiece");
  // SentencePiece-style BPE with byte fallback (Mistral v0.3) is an LLM tokenizer too.
  return type === "BPE" && (!JSON.stringify(json.pre_tokenizer ?? {}).includes("Metaspace") || !!(json.model as { byte_fallback?: boolean }).byte_fallback);
}

const BYTE_OF = new Map(Array.from({ length: 256 }, (_, b) => [byteToUnicode(b), b] as const));

/** CLIP's BPE ("</w>" closing each word) behind the encoder-tokenizer interface. */
class ClipCompat extends TemplatedTokenizer {
  readonly maskToken = undefined;
  readonly specialIds: ReadonlySet<number>;
  private readonly clip: ClipTokenizer;
  private readonly idTo: string[] = [];
  private readonly ids = new Map<string, number>();
  private readonly added = new Set<string>();

  constructor(json: EncoderTokenizerJSON) {
    super(json.post_processor);
    this.clip = new ClipTokenizer(json as unknown as ClipTokenizerJSON, Infinity);
    for (const [t, i] of Object.entries(json.model.vocab as Record<string, number>)) {
      this.idTo[i] = t;
      this.ids.set(t, i);
    }
    for (const t of json.added_tokens ?? []) {
      this.idTo[t.id] = t.content;
      this.ids.set(t.content, t.id);
      this.added.add(t.content);
    }
    this.specialIds = new Set((json.added_tokens ?? []).filter((t) => t.special).map((t) => t.id));
  }

  get vocabSize(): number {
    return this.idTo.length;
  }

  token(id: number): string {
    return this.idTo[id] ?? "";
  }

  tokenId(token: string): number | undefined {
    return this.ids.get(token);
  }

  protected tokenize(text: string): number[] {
    return this.clip.encode(text).slice(1, -1);
  }

  /** Byte-level decoding of runs between added tokens, then "</w>" as a space (transformers.js's BPEDecoder). */
  decode(ids: readonly number[], skipSpecial = true): string {
    let out = "";
    let run: number[] = [];
    const flush = () => {
      out += new TextDecoder().decode(Uint8Array.from(run));
      run = [];
    };
    for (const id of ids) {
      if (skipSpecial && this.specialIds.has(id)) continue;
      const t = this.token(id);
      if (this.added.has(t)) {
        flush();
        out += t;
      } else for (const ch of t) run.push(BYTE_OF.get(ch) ?? 0);
    }
    flush();
    return out.replaceAll("</w>", " ");
  }
}

/** An encoding of one text or pair before padding. */
interface Encoded {
  input_ids: number[];
  attention_mask: number[];
  token_type_ids?: number[];
}

/**
 * transformers.js's PreTrainedTokenizer over Kiln's tokenizers: call it on a text or a batch for
 * `{ input_ids, attention_mask }` tensors (int64, as transformers.js returns them), `encode` /
 * `decode` / `batch_decode` / `tokenize`, the special tokens and their ids, and `apply_chat_template`
 * with the model's own Jinja template.
 */
export class PreTrainedTokenizer {
  readonly config: TokenizerConfig;
  return_token_type_ids: boolean;
  padding_side: "left" | "right";
  readonly mask_token: string | null;
  readonly mask_token_id: number | undefined;
  readonly pad_token: string | null;
  pad_token_id: number | undefined;
  readonly sep_token: string | null;
  readonly sep_token_id: number | undefined;
  readonly unk_token: string | null;
  readonly unk_token_id: number | undefined;
  readonly bos_token: string | null;
  readonly bos_token_id: number | undefined;
  readonly eos_token: string | null;
  readonly eos_token_id: number | undefined;
  readonly cls_token: string | null;
  readonly all_special_ids: number[];
  readonly all_special_tokens: string[];
  chat_template: string | Record<string, string> | null;
  private readonly enc: EncoderTokenizer;
  /** The byte-level BPE behind `enc`, for decoding without transformers' cleanup. */
  private readonly bpe: Tokenizer | undefined;
  private readonly special: Set<number>;
  private readonly robertaPairs: boolean;
  private readonly templates = new Map<string, Template>();

  constructor(json: TokenizerJSON | EncoderTokenizerJSON, config: TokenizerConfig = {}, chatTemplate?: string) {
    const j = { ...(json as EncoderTokenizerJSON) };
    j.post_processor = flattenPostProcessor(j.post_processor);
    if ((j.model as { end_of_word_suffix?: string | null }).end_of_word_suffix) this.enc = new ClipCompat(j);
    else if (isLlmBPE(j)) {
      this.bpe = Tokenizer.fromJSON({ ...(j as unknown as TokenizerJSON), model: { ...(j as unknown as TokenizerJSON).model, type: "BPE" } } as TokenizerJSON);
      this.enc = new RobertaTokenizer(j, this.bpe);
    } else this.enc = encoderTokenizerFromJSON(j);
    this.robertaPairs = j.post_processor?.type === "RobertaProcessing";
    this.config = config;
    this.return_token_type_ids = TOKEN_TYPE_CLASSES.has(config.tokenizer_class?.replace(/Fast$/, "") ?? "");
    this.padding_side = config.padding_side ?? "right";
    const id = (t: string | null) => (t === null ? undefined : this.enc.tokenId(t));
    this.mask_token = tokenFromConfig(config, "mask_token");
    this.mask_token_id = id(this.mask_token);
    // A repo without tokenizer_config.json still pads with its own pad token (transformers.js pads with undefined).
    const padding = (json as { padding?: { pad_token?: string } | null }).padding;
    this.pad_token = tokenFromConfig(config, "pad_token", "eos_token") ?? padding?.pad_token ?? ["<pad>", "[PAD]"].find((t) => this.enc.tokenId(t) !== undefined) ?? null;
    this.pad_token_id = id(this.pad_token);
    this.sep_token = tokenFromConfig(config, "sep_token");
    this.sep_token_id = id(this.sep_token);
    this.unk_token = tokenFromConfig(config, "unk_token");
    this.unk_token_id = id(this.unk_token);
    this.bos_token = tokenFromConfig(config, "bos_token");
    this.bos_token_id = id(this.bos_token);
    this.eos_token = tokenFromConfig(config, "eos_token");
    this.eos_token_id = id(this.eos_token);
    this.cls_token = tokenFromConfig(config, "cls_token");
    const specials = (json.added_tokens ?? []).filter((t) => t.special);
    this.all_special_ids = specials.map((t) => t.id);
    this.all_special_tokens = specials.map((t) => t.content);
    this.special = new Set(this.all_special_ids);
    const extra = (config.additional_special_tokens as (string | { content: string })[] | undefined) ?? [];
    for (const t of extra) {
      const i = id(typeof t === "string" ? t : t.content);
      if (i !== undefined) this.special.add(i);
    }
    const inline = config.chat_template;
    this.chat_template = chatTemplate ?? (Array.isArray(inline) ? Object.fromEntries(inline.map((t) => [t.name, t.template])) : inline ?? null);
    // Callable, as transformers.js's tokenizers are.
    const call = (text: string | string[], options?: TokenizerCallOptions) => this._call(text, options);
    Object.setPrototypeOf(call, Object.getPrototypeOf(this));
    return Object.assign(call, this) as unknown as PreTrainedTokenizer;
  }

  /**
   * tokenizer.json, tokenizer_config.json and chat_template.jinja (when the repo has one) from the Hub;
   * BERT repos with only vocab.txt, and RoBERTa / BART repos with vocab.json and merges.txt, as
   * transformers builds their slow tokenizers.
   */
  static async from_pretrained(id: string, options: PretrainedOptions = {}): Promise<PreTrainedTokenizer & ((text: string | string[], options?: TokenizerCallOptions) => BatchEncoding)> {
    const ref = parseModelId(id, options.revision);
    const ep = options.endpoint ?? envEndpoint();
    const listing = listRepoFiles(ref, ep);
    const has = (f: string) => listing.then((files) => !files || files.has(f));
    const optional = async <T>(f: string, text = false): Promise<T | undefined> => {
      if (!(await has(f))) return undefined;
      return (text ? fetchText(fileUrl(ref, f, ep), true) : fetchJSON<T>(fileUrl(ref, f, ep), true)) as Promise<T | undefined>;
    };
    const [json, config, jinja] = await Promise.all([optional<TokenizerJSON>("tokenizer.json"), optional<TokenizerConfig>("tokenizer_config.json"), optional<string>("chat_template.jinja", true)]);
    const cfg = config ?? {};
    let tok: TokenizerJSON | EncoderTokenizerJSON | undefined = json;
    if (!tok) {
      const [vocabTxt, vocabJson, merges] = await Promise.all([optional<string>("vocab.txt", true), optional<Record<string, number>>("vocab.json"), optional<string>("merges.txt", true)]);
      if (vocabTxt) tok = wordPieceJSONFromVocab(vocabTxt, { lowercase: (cfg.do_lower_case as boolean | undefined) ?? true, stripAccents: cfg.strip_accents as boolean | null | undefined });
      else if (vocabJson && merges) tok = byteLevelJSONFromVocabMerges(vocabJson, merges);
      else throw new Error(`${id} has no tokenizer.json, vocab.txt, or vocab.json and merges.txt`);
    }
    return new this(tok, cfg, jinja) as PreTrainedTokenizer & ((text: string | string[], options?: TokenizerCallOptions) => BatchEncoding);
  }

  get model_max_length(): number {
    return this.config.model_max_length ?? Infinity;
  }

  get add_bos_token(): boolean | undefined {
    return this.config.add_bos_token;
  }

  get add_eos_token(): boolean | undefined {
    return this.config.add_eos_token;
  }

  get_vocab(): Map<string, number> {
    const v = new Map<string, number>();
    for (let i = 0; i < this.enc.vocabSize; i++) {
      const t = this.enc.token(i);
      if (t !== undefined) v.set(t, i);
    }
    return v;
  }

  convert_tokens_to_ids(tokens: string): number | undefined;
  convert_tokens_to_ids(tokens: string[]): (number | undefined)[];
  convert_tokens_to_ids(tokens: string | string[]): number | undefined | (number | undefined)[] {
    return typeof tokens === "string" ? this.enc.tokenId(tokens) : tokens.map((t) => this.enc.tokenId(t));
  }

  /** Ids of `text` (and `text_pair`), with the model's special tokens unless `add_special_tokens` is false. */
  encode(text: string, { text_pair = null, add_special_tokens = true }: { text_pair?: string | null; add_special_tokens?: boolean; return_token_type_ids?: boolean | null } = {}): number[] {
    return this.encodePlus(text, text_pair, add_special_tokens, false).input_ids;
  }

  /** The token strings of `text` (byte-level ones as stored, "Ġthe"). */
  tokenize(text: string, { pair = null, add_special_tokens = false }: { pair?: string | null; add_special_tokens?: boolean } = {}): string[] {
    return this.encode(text, { text_pair: pair, add_special_tokens }).map((i) => this.enc.token(i));
  }

  decode(ids: number[] | bigint[] | Tensor, { skip_special_tokens = false, clean_up_tokenization_spaces = null }: { skip_special_tokens?: boolean; clean_up_tokenization_spaces?: boolean | null } = {}): string {
    let list: number[];
    if (ids instanceof Tensor) {
      if (ids.dims.length > 2 || (ids.dims.length === 2 && ids.dims[0] !== 1)) throw new Error("decode takes a 1-D tensor (or [1, n]); use batch_decode for a batch");
      list = Array.from(ids.data as ArrayLike<number | bigint>, Number);
    } else list = (ids as (number | bigint)[]).map(Number);
    if (!list.length || !Number.isInteger(list[0])) throw new Error("token_ids must be a non-empty array of integers.");
    if (skip_special_tokens) list = list.filter((i) => !this.special.has(i));
    // Kiln's decoders skip their own special ids when asked; the filtering above is transformers.js's.
    let text = this.bpe ? this.bpe.decode(list, false) : this.enc.decode(list, false);
    // Decoders with an end-of-word suffix trim when skipping special tokens (transformers.js's CLIP).
    if (skip_special_tokens && this.enc instanceof ClipCompat) text = text.trim();
    return clean_up_tokenization_spaces ?? this.config.clean_up_tokenization_spaces ?? true ? cleanup(text) : text;
  }

  batch_decode(batch: number[][] | Tensor, args: { skip_special_tokens?: boolean; clean_up_tokenization_spaces?: boolean | null } = {}): string[] {
    const rows = batch instanceof Tensor ? (batch.tolist() as (number | bigint)[][]).map((r) => r.map(Number)) : batch;
    return rows.map((r) => this.decode(r, args));
  }

  private encodePlus(text: string, pair: string | null, special: boolean, typeIds: boolean): Encoded {
    let ids: number[];
    let types: number[];
    if (special) {
      const e = this.enc.encode(text, pair ?? undefined);
      ids = e.ids;
      types = e.typeIds;
      // RobertaProcessing gives the pair's part (from its leading </s>) segment 1, as tokenizers does;
      // Kiln's own templates keep 0 for models with one token type.
      if (pair !== null && this.robertaPairs) {
        const b = e.seq.indexOf(1);
        if (b > 0) types = types.map((t, i) => (i >= b - 1 ? 1 : t));
      }
    } else {
      const a = this.textIds(text);
      const b = pair === null ? [] : this.textIds(pair);
      ids = [...a, ...b];
      types = [...a.map(() => 0), ...b.map(() => 1)];
    }
    return { input_ids: ids, attention_mask: ids.map(() => 1), ...(typeIds ? { token_type_ids: types } : {}) };
  }

  private textIds(text: string): number[] {
    return (this.enc as TemplatedTokenizer).textIds(text);
  }

  /** transformers.js's tokenizer call: one text or a batch, padded and truncated as asked. */
  _call(text: string | string[], options: TokenizerCallOptions = {}): BatchEncoding {
    const { text_pair = null, add_special_tokens = true, padding = false, return_token_type_ids = null } = options;
    let { truncation = null, max_length = null } = options;
    const returnTensor = options.return_tensor ?? true;
    const types = return_token_type_ids ?? this.return_token_type_ids;
    const batched = Array.isArray(text);
    let rows: Encoded[];
    if (batched) {
      if (!text.length) throw new Error("text array must be non-empty");
      if (text_pair !== null && (!Array.isArray(text_pair) || text_pair.length !== text.length)) throw new Error("text_pair must be an array of the same length as text");
      rows = text.map((t, i) => this.encodePlus(t, text_pair === null ? null : (text_pair as string[])[i]!, add_special_tokens, types));
    } else {
      if (text === null || text === undefined) throw new Error("text may not be null or undefined");
      if (Array.isArray(text_pair)) throw new Error("When `text` is a string, `text_pair` must be a string too");
      rows = [this.encodePlus(text, text_pair, add_special_tokens, types)];
    }
    if (max_length === null) max_length = this.model_max_length;
    else if (truncation === null) {
      if (padding === true) max_length = this.model_max_length;
      else if (padding === false) truncation = true;
    }
    if (padding === true) max_length = Math.min(Math.max(...rows.map((r) => r.input_ids.length)), max_length ?? Infinity);
    max_length = Math.min(max_length, this.model_max_length);
    if (padding && this.pad_token_id === undefined && rows.some((r) => r.input_ids.length < max_length!)) {
      throw new Error("Asking to pad but the tokenizer does not have a padding token; set tokenizer.pad_token_id (the eos token's id is common)");
    }
    if (padding || truncation) {
      for (const r of rows) {
        const n = r.input_ids.length;
        if (n > max_length && truncation) for (const k of Object.keys(r) as (keyof Encoded)[]) r[k] = r[k]!.slice(0, max_length);
        else if (n < max_length && padding) {
          for (const k of Object.keys(r) as (keyof Encoded)[]) {
            const fill = new Array<number>(max_length - n).fill(k === "input_ids" ? this.pad_token_id! : 0);
            r[k] = this.padding_side === "right" ? [...r[k]!, ...fill] : [...fill, ...r[k]!];
          }
        }
      }
    }
    const keys = Object.keys(rows[0]!) as (keyof Encoded)[];
    const out: Record<string, unknown> = {};
    if (returnTensor) {
      if (rows.some((r) => keys.some((k) => r[k]!.length !== rows[0]![k]!.length))) {
        throw new Error("Unable to create tensor, you should probably activate truncation and/or padding with 'padding=true' and 'truncation=true' to have batched tensors with the same length.");
      }
      const dims = [rows.length, rows[0]!.input_ids.length];
      for (const k of keys) out[k] = new Tensor("int64", BigInt64Array.from(rows.flatMap((r) => r[k]!).map(BigInt)), dims);
    } else for (const k of keys) out[k] = batched ? rows.map((r) => r[k]) : rows[0]![k];
    return out as unknown as BatchEncoding;
  }

  /** The template `apply_chat_template` uses: by name from a dict of templates, the tool-use one when tools are given, else the default. */
  get_chat_template({ chat_template = null, tools = null }: { chat_template?: string | null; tools?: unknown[] | null } = {}): string {
    const own = this.chat_template;
    if (own && typeof own === "object") {
      if (chat_template !== null && Object.hasOwn(own, chat_template)) return own[chat_template]!;
      if (chat_template !== null) return chat_template;
      if (tools !== null && "tool_use" in own) return own.tool_use!;
      if ("default" in own) return own.default!;
      throw new Error(`This model has several chat templates and none is the default; pass one by name (${Object.keys(own).sort().join(", ")})`);
    }
    if (chat_template !== null) return chat_template;
    if (own) return own;
    throw new Error("Cannot use apply_chat_template() because tokenizer.chat_template is not set and no template argument was passed!");
  }

  /** The conversation through the model's chat template: token ids (as `_call` returns them), or the text with `tokenize: false`. */
  apply_chat_template(conversation: { role: string; content: unknown }[], options: ChatTemplateOptions = {}): string | Tensor | BatchEncoding | number[] {
    const { tools = null, documents = null, chat_template = null, add_generation_prompt = false, tokenize = true, padding = false, truncation = false, max_length = null, return_tensor = true, return_dict = true, tokenizer_kwargs = {}, ...kwargs } = options;
    const source = this.get_chat_template({ chat_template, tools });
    let tpl = this.templates.get(source);
    if (!tpl) this.templates.set(source, (tpl = new Template(source)));
    const specials: Record<string, string> = {};
    for (const k of SPECIAL_TOKEN_ATTRIBUTES) {
      const v = tokenFromConfig(this.config, k);
      if (v) specials[k] = v;
    }
    const rendered = tpl.render({ messages: conversation, add_generation_prompt, tools, documents, ...specials, ...kwargs });
    if (!tokenize) return rendered;
    const out = this._call(rendered, { add_special_tokens: false, padding, truncation, max_length, return_tensor, ...tokenizer_kwargs });
    return return_dict ? out : (out.input_ids as Tensor | number[]);
  }
}

/** transformers.js's AutoTokenizer: `AutoTokenizer.from_pretrained(id)` for any repo with a tokenizer.json. */
export class AutoTokenizer {
  static from_pretrained(id: string, options: PretrainedOptions = {}): ReturnType<typeof PreTrainedTokenizer.from_pretrained> {
    return PreTrainedTokenizer.from_pretrained(id, options);
  }
}

/**
 * transformers.js's read_audio: mono samples at `sampling_rate`, from a URL, Blob or bytes, decoded
 * as it does (the browser's AudioContext at that rate; stereo mixed as √2 · (left + right) / 2, after
 * ffmpeg). Without an AudioContext (a worker, Node), Kiln's own reader: WAV parsed and resampled,
 * channels averaged.
 */
export async function read_audio(input: AudioInput, sampling_rate: number): Promise<Float32Array> {
  const Ctx = (globalThis as { AudioContext?: new (o: { sampleRate: number }) => { decodeAudioData(b: ArrayBuffer): Promise<{ numberOfChannels: number; length: number; getChannelData(c: number): Float32Array }>; close?(): Promise<void> } }).AudioContext;
  if (input instanceof Float32Array || !Ctx) return readAudio(input, sampling_rate);
  let bytes: ArrayBuffer;
  if (typeof input === "string" || input instanceof URL) {
    const r = await fetch(input);
    if (!r.ok) throw new Error(`${String(input)}: HTTP ${r.status}`);
    bytes = await r.arrayBuffer();
  } else if (input instanceof ArrayBuffer) bytes = input.slice(0);
  else if (input instanceof Uint8Array) bytes = input.slice().buffer;
  else bytes = await input.arrayBuffer();
  const ctx = new Ctx({ sampleRate: sampling_rate });
  try {
    const d = await ctx.decodeAudioData(bytes);
    if (d.numberOfChannels !== 2) return d.getChannelData(0);
    const [l, r] = [d.getChannelData(0), d.getChannelData(1)];
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = (Math.SQRT2 * (l[i]! + r[i]!)) / 2;
    return out;
  } finally {
    await ctx.close?.();
  }
}

/**
 * transformers.js's RawImage: 8-bit pixels with 1 to 4 channels. Pipelines take one wherever they
 * take an image.
 */
export class RawImage {
  constructor(
    public data: Uint8ClampedArray | Uint8Array,
    public width: number,
    public height: number,
    public channels: 1 | 2 | 3 | 4,
  ) {
    if (data.length !== width * height * channels) throw new Error(`RawImage: ${data.length} bytes is not ${width} × ${height} × ${channels}`);
  }

  get size(): [number, number] {
    return [this.width, this.height];
  }

  /** Decodes a URL, Blob, bytes, ImageData, ImageBitmap or canvas (RGBA, as the browser decodes it). */
  static async read(input: RawImage | ImageInput | HTMLCanvasElement | OffscreenCanvas): Promise<RawImage> {
    if (input instanceof RawImage) return input;
    if ((typeof HTMLCanvasElement !== "undefined" && input instanceof HTMLCanvasElement) || (typeof OffscreenCanvas !== "undefined" && input instanceof OffscreenCanvas)) return RawImage.fromCanvas(input);
    const img = await readImage(input as ImageInput);
    const ch = img.data.length / (img.width * img.height);
    return new RawImage(img.data, img.width, img.height, ch as 3 | 4);
  }

  static fromURL(url: string | URL): Promise<RawImage> {
    return RawImage.read(url);
  }

  static fromBlob(blob: Blob): Promise<RawImage> {
    return RawImage.read(blob);
  }

  static fromCanvas(canvas: HTMLCanvasElement | OffscreenCanvas): RawImage {
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
    if (!ctx) throw new Error("RawImage.fromCanvas needs a 2D canvas");
    const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return new RawImage(d.data, d.width, d.height, 4);
  }

  /** From a tensor [channels, height, width] (CHW, default) or [height, width, channels], values 0–255. */
  static fromTensor(t: Tensor, channel_format: "CHW" | "HWC" = "CHW"): RawImage {
    if (t.dims.length !== 3) throw new Error("RawImage.fromTensor takes a 3-D tensor");
    const [a, b, c] = t.dims as [number, number, number];
    const [ch, h, w] = channel_format === "CHW" ? [a, b, c] : [c, a, b];
    const src = t.data as ArrayLike<number>;
    const out = new Uint8ClampedArray(ch * h * w);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        for (let k = 0; k < ch; k++) out[(y * w + x) * ch + k] = Number(channel_format === "CHW" ? src[(k * h + y) * w + x]! : src[(y * w + x) * ch + k]!);
    return new RawImage(out, w, h, ch as 1 | 2 | 3 | 4);
  }

  private convert(to: 1 | 3 | 4): RawImage {
    if (this.channels === to) return this;
    const n = this.width * this.height;
    const out = new Uint8ClampedArray(n * to);
    const c = this.channels;
    for (let i = 0; i < n; i++) {
      const p = i * c;
      const [r, g, b] = c <= 2 ? [this.data[p]!, this.data[p]!, this.data[p]!] : [this.data[p]!, this.data[p + 1]!, this.data[p + 2]!];
      const alpha = c === 4 ? this.data[p + 3]! : c === 2 ? this.data[p + 1]! : 255;
      if (to === 1) out[i] = Math.round(0.2989 * r + 0.587 * g + 0.114 * b);
      else {
        out[i * to] = r;
        out[i * to + 1] = g;
        out[i * to + 2] = b;
        if (to === 4) out[i * to + 3] = alpha;
      }
    }
    return new RawImage(out, this.width, this.height, to);
  }

  rgb(): RawImage {
    return this.convert(3);
  }

  rgba(): RawImage {
    return this.convert(4);
  }

  grayscale(): RawImage {
    return this.convert(1);
  }

  /**
   * Resized with PIL's filters ("nearest", "bilinear" (default), "bicubic", "lanczos", or their
   * numbers), as Python's PIL would; a null side keeps the aspect ratio.
   */
  async resize(width: number | null, height: number | null, { resample = 2 }: { resample?: number | "nearest" | "bilinear" | "bicubic" | "lanczos" } = {}): Promise<RawImage> {
    const w = width ?? Math.round((this.width * height!) / this.height);
    const h = height ?? Math.round((this.height * width!) / this.width);
    const r = typeof resample === "number" ? resample : { nearest: 0, lanczos: 1, bilinear: 2, bicubic: 3 }[resample];
    const ch = this.channels;
    const src: RgbImage = ch === 3 || ch === 4 ? this : this.rgb();
    const out = resizeImage(src, w, h, r);
    const img = new RawImage(out.data, out.width, out.height, src === this ? ch : 3);
    return ch === 1 ? img.grayscale() : img;
  }

  /** The box from (x_min, y_min) to (x_max, y_max), inclusive. */
  async crop([x0, y0, x1, y1]: [number, number, number, number]): Promise<RawImage> {
    x0 = Math.max(0, x0);
    y0 = Math.max(0, y0);
    x1 = Math.min(this.width - 1, x1);
    y1 = Math.min(this.height - 1, y1);
    const w = x1 - x0 + 1;
    const h = y1 - y0 + 1;
    const c = this.channels;
    const out = new Uint8ClampedArray(w * h * c);
    for (let y = 0; y < h; y++) out.set(this.data.subarray(((y0 + y) * this.width + x0) * c, ((y0 + y) * this.width + x0 + w) * c), y * w * c);
    return new RawImage(out, w, h, c);
  }

  async center_crop(width: number, height: number): Promise<RawImage> {
    const x0 = Math.round((this.width - width) / 2);
    const y0 = Math.round((this.height - height) / 2);
    return this.crop([x0, y0, x0 + width - 1, y0 + height - 1]);
  }

  /** A uint8 tensor [channels, height, width] (CHW, default) or [height, width, channels]. */
  toTensor(channel_format: "CHW" | "HWC" = "CHW"): Tensor {
    const { width: w, height: h, channels: c } = this;
    if (channel_format === "HWC") return new Tensor("uint8", new Uint8Array(this.data), [h, w, c]);
    const out = new Uint8Array(c * h * w);
    for (let i = 0; i < w * h; i++) for (let k = 0; k < c; k++) out[k * w * h + i] = this.data[i * c + k]!;
    return new Tensor("uint8", out, [c, h, w]);
  }

  toCanvas(): OffscreenCanvas {
    const rgba = this.rgba();
    const canvas = new OffscreenCanvas(this.width, this.height);
    canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba.data), this.width, this.height), 0, 0);
    return canvas;
  }

  /** Encoded as PNG (default) or another type the browser's canvas encodes. */
  toBlob(type = "image/png", quality = 1): Promise<Blob> {
    return this.toCanvas().convertToBlob({ type, quality });
  }
}
