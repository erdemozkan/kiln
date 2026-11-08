// Encoder-decoder models (BART, Marian, M2M100 / NLLB) and the text2text tasks transformers.js runs
// on them: summarization, translation and text2text-generation, with its generation settings.
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fetchText, fileUrl, listRepoFiles, parseModelId, readPytorchHeader, readSafetensorsHeader, weightFiles, type ByteSource } from "@kiln/loader";
import { Seq2SeqModel, seq2seqConfigFromHF, seq2seqName, sinusoidalTable, type Seq2SeqConfig } from "@kiln/models";
import { byteLevelFromVocabMerges, encoderTokenizerFromJSON, type EncoderTokenizer, type EncoderTokenizerJSON } from "@kiln/tokenizers";
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
    readonly model: Seq2SeqModel,
    /** The model's generation_config.json (or the generation fields of config.json). */
    readonly generation: Record<string, unknown>,
  ) {}

  /** Token ids of the source, with the language code first for NLLB / M2M100 when `src_lang` is given. */
  encodeSource(text: string, srcLang?: string): number[] {
    const enc = this.tokenizer.encode(text, undefined, this.cfg.maxPositions).ids;
    if (srcLang === undefined) return enc;
    const lang = this.tokenizer.tokenId(srcLang) ?? this.tokenizer.tokenId(`__${srcLang}__`);
    if (lang === undefined) throw new Error(`unknown source language "${srcLang}"`);
    // NLLB's tokenizer.json template has no language slot: the code goes first, before the text.
    return [lang, ...enc.filter((id) => id !== lang)];
  }

  /** Greedy generation from one source; the generated ids (start and forced tokens included). */
  async generate(source: readonly number[], options: Seq2SeqGenerateOptions = {}): Promise<number[]> {
    const g: Record<string, unknown> = { ...this.generation };
    for (const k of GEN_KEYS) if (options[k] !== undefined) g[k] = options[k];
    if (options.max_new_tokens !== undefined) delete g.max_length;
    if (options.tgt_lang !== undefined) {
      const lang = this.tokenizer.tokenId(options.tgt_lang) ?? this.tokenizer.tokenId(`__${options.tgt_lang}__`);
      if (lang === undefined) throw new Error(`unknown target language "${options.tgt_lang}"`);
      g.forced_bos_token_id = lang;
    }
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
      const logits = await this.model.step(ids.at(-1)!, ids.length - 1);
      if (rep !== 1) for (const id of new Set(ids)) logits[id] = logits[id]! < 0 ? logits[id]! * rep : logits[id]! / rep;
      for (const b of bad) logits[b] = -Infinity;
      if (ids.length < minLen) for (const e of eos) logits[e] = -Infinity;
      for (const b of bannedNgramTokens(ids, ngram)) logits[b] = -Infinity;
      let next: number;
      if (ids.length === 1 && forcedBos !== undefined && forcedBos !== null) next = forcedBos;
      else if (ids.length === maxLen - 1 && forcedEos !== undefined) next = forcedEos;
      else {
        next = 0;
        for (let i = 1; i < logits.length; i++) if (logits[i]! > logits[next]!) next = i;
      }
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

  destroy(): void {
    this.model.destroy();
  }
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
  const cfg = seq2seqConfigFromHF(config!);
  let tokenizer: EncoderTokenizer;
  if (tokJson) tokenizer = encoderTokenizerFromJSON(tokJson);
  else if (cfg.family === "bart" && (await has("vocab.json")) && (await has("merges.txt"))) {
    const [vocab, merges] = await Promise.all([fetchJSON<Record<string, number>>(fileUrl(ref, "vocab.json", ep)), fetchText(fileUrl(ref, "merges.txt", ep))]);
    tokenizer = byteLevelFromVocabMerges(vocab!, merges!);
  } else throw new Error(`${modelId} has no tokenizer.json${cfg.family === "marian" ? " (Marian's SentencePiece models are not supported yet)" : ""}`);
  const backend = opts.backend ?? (await openBackend(opts.device ?? "auto", opts.cpuThreads));
  const sources: ByteSource[] = [];
  try {
    const files: WeightFile[] = [];
    let shared = false;
    for (const name of names.names) {
      const { source, header } = await openParsed(ref, name, ep, names.format === "safetensors" ? readSafetensorsHeader : readPytorchHeader);
      sources.push(source);
      // The shared embedding is often stored three times (shared, encoder and decoder embed_tokens): load one.
      const tensors = header.tensors.filter((t) => {
        const n = seq2seqName(t.name);
        if (n !== "shared.weight") return n !== undefined && !(cfg.positions !== "learned" && n.endsWith(".pos"));
        if (shared) return false;
        shared = true;
        return true;
      });
      files.push({ name, source, tensors, rename: seq2seqName });
    }
    const weights = await loadWeights(backend, files, opts.quant ?? "f32", opts.onProgress);
    if (cfg.positions !== "learned") {
      const rows = cfg.maxPositions + cfg.posOffset;
      const table = sinusoidalTable(cfg.positions, rows, cfg.dModel, cfg.positions === "sinusoidal-m2m100" ? cfg.pad : undefined);
      for (const side of ["enc", "dec"]) {
        const buf = backend.alloc(table.byteLength, "storage", `seq2seq/${side}-pos`);
        backend.write(buf, 0, table);
        weights.set(`${side}.pos`, { shape: [rows, cfg.dModel], dtype: "f32", buf } as Tensor);
      }
    }
    const gen = generation ?? Object.fromEntries(Object.entries(config!).filter(([k]) => /length|ngram|token_id|penalty|bad_words|num_beams/.test(k)));
    return new Seq2SeqEngine(backend, cfg, tokenizer, new Seq2SeqModel(backend, cfg, weights, { maxTarget: opts.maxTarget }), gen);
  } finally {
    for (const s of sources) s.close?.();
  }
}
