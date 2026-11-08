// Encoder-decoder models (BART, Marian, M2M100 / NLLB) and the text2text tasks transformers.js runs
// on them: summarization, translation and text2text-generation, with its generation settings.
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fetchText, fileUrl, listRepoFiles, parseModelId, readPytorchHeader, readSafetensorsHeader, toF32, weightFiles, whisperFeatures, type ByteSource, type TensorEntry } from "@kiln/loader";
import { Seq2SeqModel, T5Model, seq2seqConfigFromHF, seq2seqName, sinusoidalTable, t5ConfigFromHF, t5Name, t5Transform, type Seq2SeqConfig, type T5Config } from "@kiln/models";
import { byteLevelFromVocabMerges, encoderTokenizerFromJSON, marianTokenizer, type EncoderTokenizer, type EncoderTokenizerJSON } from "@kiln/tokenizers";
import { openBackend, openParsed } from "./load";
import { loadWeights, type LoadProgress, type WeightFile } from "./weights";

export interface Seq2SeqLoadOptions {
  /** Matrix storage: "f32" (default), "q8" or "q4". */
  quant?: "f32" | "q8" | "q4";
  device?: "auto" | "gpu" | "cpu";
  cpuThreads?: number;
  revision?: string;
  endpoint?: string;
  backend?: Backend;
  /** Longest output in tokens (default the model's position limit). */
  maxTarget?: number;
  onProgress?: (p: LoadProgress) => void;
}

/** transformers' generation settings Kiln applies (greedy search; `num_beams` is ignored, as transformers.js does). */
export interface Seq2SeqGenerateOptions {
  max_new_tokens?: number;
  /** Total decoder length including the start token, when max_new_tokens is not given. */
  max_length?: number;
  min_length?: number;
  no_repeat_ngram_size?: number;
  repetition_penalty?: number;
  forced_bos_token_id?: number;
  forced_eos_token_id?: number | number[];
  /** Single-token entries are never generated. */
  bad_words_ids?: number[][];
  /** Source language code for multilingual translation models (NLLB: "eng_Latn"). */
  src_lang?: string;
  /** Target language code; its token is forced first (NLLB: "fra_Latn"). */
  tgt_lang?: string;
}

const GEN_KEYS = ["max_new_tokens", "max_length", "min_length", "no_repeat_ngram_size", "repetition_penalty", "forced_bos_token_id", "forced_eos_token_id", "bad_words_ids"] as const;

/** Tokens that would repeat an n-gram already in `ids`. */
export function bannedNgramTokens(ids: readonly number[], n: number): Set<number> {
  const out = new Set<number>();
  if (n <= 0 || ids.length + 1 < n) return out;
  const prefix = ids.slice(ids.length - (n - 1));
  for (let i = 0; i + n <= ids.length; i++) {
    let same = true;
    for (let j = 0; j < n - 1 && same; j++) same = ids[i + j] === prefix[j];
    if (same) out.add(ids[i + n - 1]!);
  }
  return out;
}

export class Seq2SeqEngine {
  constructor(
    readonly backend: Backend,
    readonly cfg: Seq2SeqConfig,
    readonly tokenizer: EncoderTokenizer,
    readonly model: Seq2SeqModel | T5Model,
    /** The model's generation_config.json (or the generation fields of config.json). */
    readonly generation: Record<string, unknown>,
  ) {}

  /**
   * Token ids of a source. NLLB / M2M100 put the source language code first and `</s>` last
   * (`src_lang`, default eng_Latn); other models use their tokenizer's template.
   */
  encodeSource(text: string, srcLang?: string): number[] {
    const max = this.cfg.maxPositions;
    if (this.cfg.family !== "m2m100") return this.tokenizer.encode(text, undefined, max).ids;
    const tok = this.tokenizer as EncoderTokenizer & { textIds?: (t: string) => number[] };
    if (!tok.textIds) throw new Error("this tokenizer cannot build NLLB inputs");
    const lang = this.langId(srcLang ?? "eng_Latn");
    return [lang, ...tok.textIds(text).slice(0, max - 2), this.cfg.eos];
  }

  /** The token of a language code: NLLB's "fra_Latn", or M2M100's "__fr__" given as "fr". */
  langId(code: string): number {
    const id = this.tokenizer.tokenId(code) ?? this.tokenizer.tokenId(`__${code}__`);
    if (id === undefined) throw new Error(`unknown language code "${code}"`);
    return id;
  }

  /** Greedy generation from one source; the generated ids (start and forced tokens included). */
  async generate(source: readonly number[], options: Seq2SeqGenerateOptions = {}): Promise<number[]> {
    const g: Record<string, unknown> = { ...this.generation };
    for (const k of GEN_KEYS) if (options[k] !== undefined) g[k] = options[k];
    if (options.max_new_tokens !== undefined) delete g.max_length;
    if (options.tgt_lang !== undefined) g.forced_bos_token_id = this.langId(options.tgt_lang);
    else if (this.cfg.family === "m2m100" && g.forced_bos_token_id === undefined) throw new Error("translation with this model needs tgt_lang (e.g. \"fra_Latn\")");
    const start = Number(g.decoder_start_token_id ?? this.cfg.decoderStart);
    const eos = new Set<number>([g.eos_token_id ?? this.cfg.eos].flat() as number[]);
    const maxLen = Math.min(
      this.model.maxTarget,
      g.max_new_tokens !== undefined ? 1 + Number(g.max_new_tokens) : g.max_length !== undefined ? Number(g.max_length) : 20,
    );
    const minLen = Number(g.min_length ?? 0);
    const ngram = Number(g.no_repeat_ngram_size ?? 0);
    const rep = Number(g.repetition_penalty ?? 1);
    const forcedBos = g.forced_bos_token_id as number | undefined | null;
    const forcedEos = g.forced_eos_token_id === undefined || g.forced_eos_token_id === null ? undefined : [g.forced_eos_token_id].flat()[0] as number;
    const bad = ((g.bad_words_ids as number[][] | undefined) ?? []).filter((b) => b.length === 1).map((b) => b[0]!);

    await this.model.encode(source);
    const ids = [start];
    while (ids.length < maxLen) {
      // transformers' logits processors as banned tokens for the backend's greedy choice.
      const banned = new Set<number>(bad);
      if (ids.length < minLen) for (const e of eos) banned.add(e);
      for (const b of bannedNgramTokens(ids, ngram)) banned.add(b);
      let next = await this.model.next(ids.at(-1)!, ids.length - 1, { banned, repetitionPenalty: rep, seen: ids });
      if (ids.length === 1 && forcedBos !== undefined && forcedBos !== null) next = forcedBos;
      else if (ids.length === maxLen - 1 && forcedEos !== undefined) next = forcedEos;
      ids.push(next);
      if (eos.has(next)) break;
    }
    return ids;
  }

  /** Source text to output text: prefix, tokenize, generate, decode without special tokens. */
  async run(text: string, options: Seq2SeqGenerateOptions = {}, prefix = ""): Promise<string> {
    const ids = await this.generate(this.encodeSource(prefix + text, options.src_lang), options);
    return this.tokenizer.decode(ids, true);
  }

  /**
   * Whisper: text of 16 kHz mono audio. The decoder starts from transformers.js's prompt (start,
   * language and task for multilingual models, no-timestamps), with the model's suppressed tokens
   * and its begin-suppressed ones at the first step. Audio longer than a window (30 s) is split
   * into consecutive windows whose texts are joined (transformers.js keeps only the first window
   * unless asked to chunk).
   */
  async transcribe(samples: Float32Array, options: WhisperOptions = {}): Promise<{ text: string }> {
    const a = this.cfg.audio;
    if (!a || !(this.model instanceof Seq2SeqModel)) throw new Error("this model does not take audio");
    const g = this.generation;
    const window = 16000 * (options.chunk_length_s ?? 30);
    const texts: string[] = [];
    for (let off = 0; off < Math.max(samples.length, 1); off += window) {
      const feats = whisperFeatures(samples.subarray(off, off + window), { mels: a.mels });
      await this.model.encodeAudio(feats);
      const init = this.whisperPrompt(options);
      const suppress = new Set<number>((g.suppress_tokens as number[] | undefined) ?? []);
      const begin = new Set<number>((g.begin_suppress_tokens as number[] | undefined) ?? []);
      const eos = Number(g.eos_token_id ?? this.cfg.eos);
      const maxLen = Math.min(this.model.maxTarget, options.max_new_tokens !== undefined ? init.length + options.max_new_tokens : Number(g.max_length ?? this.model.maxTarget));
      const ids = [...init];
      // The prompt goes in a token at a time; only the last step's choice is kept.
      for (let t = 0; t + 1 < init.length; t++) await this.model.next(init[t]!, t);
      while (ids.length < maxLen) {
        const banned = ids.length === init.length ? new Set([...suppress, ...begin]) : suppress;
        const next = await this.model.next(ids.at(-1)!, ids.length - 1, { banned });
        ids.push(next);
        if (next === eos) break;
      }
      texts.push(this.tokenizer.decode(ids.slice(init.length), true));
    }
    return { text: texts.join("") };
  }

  /** transformers.js's Whisper prompt: [start, language, task, no-timestamps] (English-only models: [start, no-timestamps]). */
  whisperPrompt(options: WhisperOptions = {}): number[] {
    const g = this.generation;
    const out = [Number(g.decoder_start_token_id ?? this.cfg.decoderStart)];
    if (g.is_multilingual) {
      const lang = whisperLanguage(options.language ?? "en");
      const langId = (g.lang_to_id as Record<string, number> | undefined)?.[`<|${lang}|>`];
      if (langId === undefined) throw new Error(`Whisper: unknown language "${options.language}"`);
      const taskId = (g.task_to_id as Record<string, number> | undefined)?.[options.task ?? "transcribe"];
      if (taskId === undefined) throw new Error(`Whisper: unknown task "${options.task}"`);
      out.push(langId, taskId);
    } else if (options.language || options.task) {
      throw new Error("Whisper: language and task cannot be set for an English-only model");
    }
    if (g.no_timestamps_token_id !== undefined) out.push(Number(g.no_timestamps_token_id));
    return out;
  }

  destroy(): void {
    this.model.destroy();
  }
}

/** Options of automatic-speech-recognition with Whisper (transformers.js's names). */
export interface WhisperOptions {
  /** Spoken language: a code ("fr") or English name ("french"); multilingual models only (default English). */
  language?: string;
  /** "transcribe" (default) or "translate" (to English); multilingual models only. */
  task?: "transcribe" | "translate";
  max_new_tokens?: number;
  /** Window length in seconds (default 30, Whisper's). */
  chunk_length_s?: number;
}

const LANGUAGE_NAMES: Record<string, string> = {
  english: "en", french: "fr", german: "de", spanish: "es", italian: "it", portuguese: "pt", dutch: "nl", russian: "ru",
  chinese: "zh", japanese: "ja", korean: "ko", arabic: "ar", turkish: "tr", hindi: "hi", polish: "pl", ukrainian: "uk",
  swedish: "sv", greek: "el", czech: "cs", romanian: "ro", hungarian: "hu", danish: "da", finnish: "fi", norwegian: "no",
  hebrew: "he", indonesian: "id", vietnamese: "vi", thai: "th", persian: "fa", catalan: "ca",
};

/** A Whisper language code from a code, an English name or a <|code|> token. */
export function whisperLanguage(l: string): string {
  const s = l.toLowerCase().replace(/^<\|/, "").replace(/\|>$/, "");
  return LANGUAGE_NAMES[s] ?? s;
}

/** Loads an encoder-decoder repo ("org/repo", safetensors) for summarization, translation and text2text. */
export async function createSeq2Seq(modelId: string, opts: Seq2SeqLoadOptions = {}): Promise<Seq2SeqEngine> {
  const ref = parseModelId(modelId, opts.revision);
  if (ref.gguf) throw new Error("encoder-decoder models load from safetensors repos, not GGUF files");
  const ep = opts.endpoint;
  const listing = listRepoFiles(ref, ep);
  const has = async (f: string) => {
    const files = await listing;
    return !files || files.has(f);
  };
  const [config, generation, tokJson, names] = await Promise.all([
    fetchJSON<Record<string, unknown>>(fileUrl(ref, "config.json", ep)),
    has("generation_config.json").then((y) => (y ? fetchJSON<Record<string, unknown>>(fileUrl(ref, "generation_config.json", ep), true) : undefined)),
    has("tokenizer.json").then((y) => (y ? fetchJSON<EncoderTokenizerJSON>(fileUrl(ref, "tokenizer.json", ep), true) : undefined)),
    listing.then((files) => weightFiles(ref, ep, files)),
  ]);
  const t5 = config!.model_type === "t5" || config!.model_type === "mt5";
  const cfg: Seq2SeqConfig = t5 ? t5ConfigFromHF(config!) : seq2seqConfigFromHF(config!);
  const rename = t5 ? t5Name : seq2seqName;
  let tokenizer: EncoderTokenizer;
  if (tokJson) tokenizer = encoderTokenizerFromJSON(tokJson);
  else if (cfg.family === "bart" && (await has("vocab.json")) && (await has("merges.txt"))) {
    const [vocab, merges] = await Promise.all([fetchJSON<Record<string, number>>(fileUrl(ref, "vocab.json", ep)), fetchText(fileUrl(ref, "merges.txt", ep))]);
    tokenizer = byteLevelFromVocabMerges(vocab!, merges!);
  } else if (cfg.family === "marian" && (await has("source.spm")) && (await has("vocab.json"))) {
    const [spm, vocab] = await Promise.all([
      fetch(fileUrl(ref, "source.spm", ep)).then(async (r) => {
        if (!r.ok) throw new Error(`source.spm: HTTP ${r.status}`);
        return new Uint8Array(await r.arrayBuffer());
      }),
      fetchJSON<Record<string, number>>(fileUrl(ref, "vocab.json", ep)),
    ]);
    tokenizer = marianTokenizer(spm, vocab!);
  } else throw new Error(`${modelId} has no tokenizer.json`);
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  try {
    const files: WeightFile[] = [];
    const tensorsByName = new Map<string, { source: ByteSource; entry: TensorEntry }>();
    let shared = false;
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      for (const t of header.tensors) tensorsByName.set(t.name, { source, entry: t });
      // The shared embedding is often stored three times (shared, encoder and decoder embed_tokens): load one.
      const tensors = header.tensors.filter((t) => {
        const n = rename(t.name);
        // A tied output layer is the shared embedding again.
        if (n === "lm_head.weight" && config!.tie_word_embeddings !== false) return false;
        if (n !== "shared.weight") return n !== undefined && !(cfg.positions !== "learned" && n.endsWith(".pos"));
        if (shared) return false;
        shared = true;
        return true;
      });
      files.push({ name, source, tensors, rename, transform: t5 ? t5Transform(cfg as T5Config) : undefined });
    }
    const weights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);
    if (cfg.audio) {
      // Conv weights [out, in, k] → rows [out, k·in] tap-major, padded to a multiple of 32 (im2col1d's rows).
      for (const n of ["conv1", "conv2"]) {
        const t = tensorsByName.get(`model.encoder.${n}.weight`);
        if (!t) throw new Error(`Whisper: model.encoder.${n}.weight is missing`);
        const [O, C, k] = t.entry.shape as [number, number, number];
        const raw = toF32(t.entry.type as Parameters<typeof toF32>[0], await t.source.read(t.entry.offset, t.entry.byteLength), O * C * k);
        const row = Math.ceil((k * C) / 32) * 32;
        const w = new Float32Array(O * row);
        for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) for (let j = 0; j < k; j++) w[o * row + j * C + c] = raw[(o * C + c) * k + j]!;
        const buf = backend.alloc(w.byteLength, "storage", `whisper/${n}`);
        backend.write(buf, 0, w);
        weights.set(`enc.${n}.weight`, { shape: [O, row], dtype: "f32", buf } as Tensor);
      }
    }
    if (!t5 && cfg.positions !== "learned") {
      const rows = cfg.maxPositions + cfg.posOffset;
      const table = sinusoidalTable(cfg.positions, rows, cfg.dModel, cfg.positions === "sinusoidal-m2m100" ? cfg.pad : undefined);
      for (const side of ["enc", "dec"]) {
        const buf = backend.alloc(table.byteLength, "storage", `seq2seq/${side}-pos`);
        backend.write(buf, 0, table);
        weights.set(`${side}.pos`, { shape: [rows, cfg.dModel], dtype: "f32", buf } as Tensor);
      }
    }
    const gen = generation ?? Object.fromEntries(Object.entries(config!).filter(([k]) => /length|ngram|token_id|penalty|bad_words|num_beams/.test(k)));
    const model = t5 ? new T5Model(backend, cfg as T5Config, weights, { maxTarget: opts.maxTarget }) : new Seq2SeqModel(backend, cfg, weights, { maxTarget: opts.maxTarget });
    return new Seq2SeqEngine(backend, cfg, tokenizer, model, gen);
  } finally {
    for (const s of sources) s.close?.();
  }
}
