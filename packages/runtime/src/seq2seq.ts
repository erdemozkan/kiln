// Encoder-decoder models (BART, Marian, M2M100 / NLLB) and the text2text tasks transformers.js runs
// on them: summarization, translation and text2text-generation, with its generation settings.
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fetchText, fileUrl, listRepoFiles, parseModelId, readPytorchHeader, readSafetensorsHeader, toF32, weightFiles, whisperFeatures, type ByteSource, type TensorEntry } from "@kiln/loader";
import { Seq2SeqModel, T5Model, seq2seqConfigFromHF, seq2seqName, sinusoidalTable, t5ConfigFromHF, t5Name, t5Transform, type Seq2SeqConfig, type T5Config } from "@kiln/models";
import { byteLevelFromVocabMerges, encoderTokenizerFromJSON, marianTokenizer, type EncoderTokenizer, type EncoderTokenizerJSON } from "@kiln/tokenizers";
import { openBackend, openParsed } from "./load";
import { loadWeights, type LoadProgress, type WeightFile } from "./weights";
import { PUNCTUATION_ONLY, collateWords, whisperTokenTimestamps, type WhisperWord } from "./whisper-words";

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
  async generate(source: readonly number[], options: Seq2SeqGenerateOptions = {}, encode?: () => Promise<void>): Promise<number[]> {
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

    await (encode ? encode() : this.model.encode(source));
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
   * Whisper: text of 16 kHz mono audio, as transformers.js's pipeline computes it. The decoder starts
   * from its prompt (start, language and task for multilingual models, then no-timestamps unless
   * timestamps are asked for), with the model's suppressed tokens and its begin-suppressed ones at
   * the first step. With `chunk_length_s`, the audio is cut into overlapping windows (`stride_length_s`
   * each side, default a sixth of the window) whose token sequences are merged where they agree;
   * audio longer than 30 s is chunked that way by default (transformers.js keeps only the first 30 s).
   * `return_timestamps: true` adds `chunks`: [{ timestamp: [start, end], text }]; `"word"` makes
   * them words, timed from the cross-attention of the model's alignment heads (see whisper-words).
   */
  async transcribe(samples: Float32Array, options: WhisperOptions = {}): Promise<{ text: string; chunks?: WhisperChunk[] | WhisperWord[] }> {
    if (!this.cfg.audio || !(this.model instanceof Seq2SeqModel)) throw new Error("this model does not take audio");
    const word = options.return_timestamps === "word";
    const timestamps = options.return_timestamps === true || word;
    const model = this.model;
    if (word) {
      const heads = this.generation.alignment_heads as [number, number][] | undefined;
      if (!heads?.length) throw new Error("Whisper: this model's generation config has no alignment_heads, so word-level timestamps are not available");
      model.enableAlignment(heads, this.cfg.audio.frames / 2);
    }
    model.recordAlignment = word;
    try {
      return await this.transcribeWindows(samples, options, timestamps, word);
    } finally {
      model.recordAlignment = false;
    }
  }

  private async transcribeWindows(samples: Float32Array, options: WhisperOptions, timestamps: boolean, word: boolean): Promise<{ text: string; chunks?: WhisperChunk[] | WhisperWord[] }> {
    const audio = this.cfg.audio!;
    const rate = 16000;
    let chunkLen = options.chunk_length_s ?? 0;
    if (!chunkLen && samples.length > 30 * rate) chunkLen = 30;
    const windows: { start: number; end: number; stride: [number, number, number] }[] = [];
    if (chunkLen > 0) {
      const strideS = options.stride_length_s ?? chunkLen / 6;
      if (chunkLen <= strideS) throw new Error("chunk_length_s must be larger than stride_length_s");
      const window = Math.round(rate * chunkLen);
      const stride = Math.round(rate * strideS);
      const jump = window - 2 * stride;
      for (let off = 0; ; off += jump) {
        const end = Math.min(off + window, samples.length);
        const first = off === 0;
        const last = off + window >= samples.length;
        windows.push({ start: off, end, stride: [end - off, first ? 0 : stride, last ? 0 : stride] });
        if (last) break;
      }
    } else windows.push({ start: 0, end: samples.length, stride: [samples.length, 0, 0] });
    const g = this.generation;
    const noTs = Number(g.no_timestamps_token_id);
    const eos = Number(g.eos_token_id ?? this.cfg.eos);
    const seqs: AsrSequence[] = [];
    for (const w of windows) {
      const out = await this.whisperGenerate(samples.subarray(w.start, w.end), options, timestamps, word);
      const stride = w.stride.map((x) => x / rate) as [number, number, number];
      if (!out.times) seqs.push({ tokens: out.tokens, stride });
      else {
        // As transformers.js: the prompt is cut at the first timestamp, the times rounded to 0.01 s.
        const prefix = Math.max(out.tokens.findIndex((t) => t > noTs), 0);
        seqs.push({ tokens: out.tokens.slice(prefix), stride, tokenTimestamps: out.times.slice(prefix).map(round2) });
      }
    }
    return decodeAsr(seqs, {
      timestampBegin: noTs + 1,
      firstSpecial: Math.min(eos, Number(g.decoder_start_token_id ?? this.cfg.decoderStart)),
      timePrecision: 30 / (audio.frames / 2),
      timestamps,
      decode: (ids) => this.tokenizer.decode(ids, true),
      words: word ? { eos, language: options.language ? whisperLanguage(options.language) : undefined } : undefined,
    });
  }

  /**
   * One window's generated ids, prompt included. With timestamps, transformers' timestamp rule
   * applies: the first token is a timestamp, timestamps come in pairs, and a timestamp is chosen
   * whenever the timestamp tokens' total probability beats the best text token's. Each step still
   * picks on the backend; steps where both kinds are allowed also read back the 1,501 timestamp
   * logits and the chosen text token's logit. Timestamp mode then seeks as transformers does: a
   * pass that stops on a segment boundary before the window's end is followed by another from that
   * boundary (the features shifted left and zero-padded), its timestamps offset to match.
   */
  private async whisperGenerate(samples: Float32Array, options: WhisperOptions, timestamps: boolean, word = false): Promise<{ tokens: number[]; times?: number[] }> {
    const a = this.cfg.audio!;
    const g = this.generation;
    const feats = whisperFeatures(samples, { mels: a.mels });
    const init = this.whisperPrompt(options, timestamps);
    // transformers.js crops the attention to this many frames here (mel frames, not encoder ones).
    if (!timestamps || options.max_new_tokens !== undefined) return this.whisperPass(feats, init, options, timestamps, word ? Math.floor(samples.length / 160) : undefined);
    const eos = Number(g.eos_token_id ?? this.cfg.eos);
    const tsBegin = Number(g.no_timestamps_token_id) + 1;
    const total = a.frames;
    const all: number[] = [];
    const allTimes: number[] = [];
    for (let seek = 0; seek < total; ) {
      const shifted = new Float32Array(feats.length);
      shifted.set(feats.subarray(seek * a.mels));
      const pass = await this.whisperPass(shifted, init, options, true, word ? Math.floor((total - seek) / 2) : undefined);
      const gen = pass.tokens.slice(init.length);
      const times = pass.times?.slice(init.length).map((t) => t + (seek / 2) * 0.02);
      if (gen.at(-1) === eos) gen.pop();
      if (!gen.length) break;
      const isTs = gen.map((t) => t >= tsBegin);
      const single = gen.length >= 2 && isTs[gen.length - 1]! && !isTs[gen.length - 2]!;
      const pairs: number[] = [];
      for (let i = 0; i + 1 < gen.length; i++) if (isTs[i] && isTs[i + 1]) pairs.push(i + 1);
      let advance = total - seek;
      let keep = gen.length;
      if (pairs.length && !single) {
        const last = pairs.at(-1)!;
        advance = (gen[last - 1]! - tsBegin) * 2;
        keep = last;
      }
      const offset = Math.floor(seek / 2);
      for (let i = 0; i < keep; i++) if (gen[i]! >= tsBegin) gen[i] = Math.min(gen[i]! + offset, tsBegin + 1500);
      all.push(...gen.slice(0, keep));
      if (times) allTimes.push(...times.slice(0, keep));
      seek += advance;
    }
    const tokens = [...init, ...all, eos];
    return word ? { tokens, times: [...new Array<number>(init.length).fill(0), ...allTimes, 0] } : { tokens };
  }

  /**
   * One decoding pass over log-mel features [frames, mels] from a prompt. With `frames`, also every
   * token's start time from the alignment heads' cross-attention over the first `frames` encoder frames.
   */
  private async whisperPass(feats: Float32Array, init: number[], options: WhisperOptions, timestamps: boolean, frames?: number): Promise<{ tokens: number[]; times?: number[] }> {
    const model = this.model as Seq2SeqModel;
    const g = this.generation;
    await model.encodeAudio(feats);
    const suppress = (g.suppress_tokens as number[] | undefined) ?? [];
    const begin = (g.begin_suppress_tokens as number[] | undefined) ?? [];
    const eos = Number(g.eos_token_id ?? this.cfg.eos);
    const noTs = Number(g.no_timestamps_token_id);
    const tsBegin = noTs + 1;
    const V = this.cfg.vocab;
    const range = (lo: number, hi: number) => Array.from({ length: Math.max(0, hi - lo) }, (_, i) => lo + i);
    const tsRange = timestamps ? range(tsBegin, V) : [];
    const maxLen = Math.min(model.maxTarget, options.max_new_tokens !== undefined ? init.length + options.max_new_tokens : Number(g.max_length ?? model.maxTarget));
    const ids = [...init];
    // The prompt goes in a token at a time; only the last step's choice is kept.
    for (let t = 0; t + 1 < init.length; t++) await model.next(init[t]!, t);
    while (ids.length < maxLen) {
      const first = ids.length === init.length;
      const banned: number[] = first ? [...suppress, ...begin] : [...suppress];
      const pos = ids.length - 1;
      const prev = ids.at(-1)!;
      let next: number;
      if (!timestamps) next = await model.next(prev, pos, { banned });
      else {
        banned.push(noTs);
        const seq = ids.slice(init.length);
        const lastTs = seq.length >= 1 && seq.at(-1)! >= tsBegin;
        const penultTs = seq.length < 2 || seq.at(-2)! >= tsBegin;
        if (first) next = await model.next(prev, pos, { banned: [...banned, ...range(0, tsBegin)] });
        else if (lastTs && penultTs) next = await model.next(prev, pos, { banned: [...banned, ...tsRange] });
        else {
          if (lastTs) banned.push(...range(0, eos));
          const d = await model.nextDetail(prev, pos, { banned: [...banned, ...tsRange], tail: tsBegin });
          next = whisperTimestampChoice(d.id, d.logit, d.tail!, tsBegin);
        }
      }
      ids.push(next);
      if (next === eos) break;
    }
    if (frames === undefined) return { tokens: ids };
    const rows = ids.length - 1;
    const heads = (g.alignment_heads as unknown[]).length;
    const S = this.cfg.audio!.frames / 2;
    const weights = await model.alignment(rows);
    const medianWidth = Number(g.median_filter_width ?? 7);
    return { tokens: ids, times: whisperTokenTimestamps(weights, { rows, heads, S, frames, nInit: init.length, medianWidth }) };
  }

  /** transformers.js's Whisper prompt: [start, language, task, no-timestamps] (English-only models: [start, no-timestamps]); no-timestamps is left out when timestamps are wanted. */
  whisperPrompt(options: WhisperOptions = {}, timestamps = false): number[] {
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
    if (!timestamps && g.no_timestamps_token_id !== undefined) out.push(Number(g.no_timestamps_token_id));
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
  /** Window length in seconds for chunked transcription (default: none, or 30 for audio past 30 s). */
  chunk_length_s?: number;
  /** Overlap on each side of a window, in seconds (default chunk_length_s / 6). */
  stride_length_s?: number;
  /** true: segments with start and end times; "word": words with start and end times. */
  return_timestamps?: boolean | "word";
}

export interface WhisperChunk {
  timestamp: [number | null, number | null];
  text: string;
}

/**
 * Whisper's timestamp rule for one step: the best text token (`textId`, logit `textLogit`) unless the
 * timestamp tokens' logits (`tail`, from `tsBegin` up) have a larger log-sum-exp, then the best timestamp.
 */
export function whisperTimestampChoice(textId: number, textLogit: number, tail: Float32Array, tsBegin: number): number {
  let max = -Infinity;
  let arg = 0;
  for (let i = 0; i < tail.length; i++) if (tail[i]! > max) [max, arg] = [tail[i]!, i];
  let sum = 0;
  for (let i = 0; i < tail.length; i++) sum += Math.exp(tail[i]! - max);
  const lse = max + Math.log(sum);
  return lse > textLogit ? tsBegin + arg : textId;
}

/** One window's ids and its stride [length, left overlap, right overlap] in seconds. */
export interface AsrSequence {
  tokens: number[];
  stride: [number, number, number];
  /** Word timestamps: each token's start time within the window, in seconds. */
  tokenTimestamps?: number[];
}

const round2 = (x: number) => Math.round(x * 100) / 100;

/**
 * transformers' longest-common-sequence merge of overlapping windows' token lists: each neighbour
 * pair is aligned where most tokens agree, and cut in the middle of the agreeing stretch.
 */
export function findLongestCommonSequence(sequences: number[][]): number[] {
  return mergeSequences(sequences).tokens;
}

/** As findLongestCommonSequence, also cutting per-token [start, end] times the same way (agreeing tokens must not start more than 0.1 s earlier on the right). */
export function mergeSequences(sequences: number[][], times?: [number, number | null][][]): { tokens: number[]; times: [number, number | null][] } {
  const timed = !!times && times.length > 0;
  let left = sequences[0] ?? [];
  let leftT = timed ? times![0]! : [];
  const total: number[] = [];
  const totalT: [number, number | null][] = [];
  for (let i = 1; i < sequences.length; i++) {
    const right = sequences[i]!;
    const L = left.length;
    const R = right.length;
    let best = 0;
    let idx = [L, L, 0, 0];
    for (let j = 1; j < L + R; j++) {
      const ls = Math.max(0, L - j);
      const le = Math.min(L, L + R - j);
      const rs = Math.max(0, j - L);
      const re = Math.min(R, j);
      let matches = 0;
      for (let k = 0; k < le - ls; k++) {
        if (left[ls + k] === right[rs + k] && (!timed || leftT[ls + k]![0] - 0.1 <= times![i]![rs + k]![0])) matches++;
      }
      const score = matches / j + j / 10000;
      if (matches > 1 && score > best) {
        best = score;
        idx = [ls, le, rs, re];
      }
    }
    const [ls, le, rs, re] = idx as [number, number, number, number];
    const leftMid = Math.floor((le + ls) / 2);
    let rightMid = Math.floor((re + rs) / 2);
    if (timed && best === 0 && L > 0) {
      const lastLeft = leftT[L - 1]![0];
      const k = times![i]!.findIndex((t) => t[0] >= lastLeft);
      rightMid = k === -1 ? R : k;
    }
    total.push(...left.slice(0, leftMid));
    left = right.slice(rightMid);
    if (timed) {
      totalT.push(...leftT.slice(0, leftMid));
      leftT = times![i]!.slice(rightMid);
    }
  }
  total.push(...left);
  if (timed) totalT.push(...leftT);
  return { tokens: total, times: totalT };
}

/**
 * transformers.js's WhisperTokenizer._decode_asr (segment or word timestamps, no languages): walks
 * every window's tokens, opens and closes segments on timestamp pairs (offset by the windows before,
 * skipping timestamps inside the overlaps), and merges the text of overlapping windows. With
 * `words`, the chunks are the words of every segment instead.
 */
export function decodeAsr(
  sequences: readonly AsrSequence[],
  o: {
    timestampBegin: number;
    firstSpecial: number;
    timePrecision: number;
    timestamps: boolean;
    decode: (ids: number[]) => string;
    words?: { eos: number; language?: string };
  },
): { text: string; chunks?: WhisperChunk[] | WhisperWord[] } {
  const tsBegin = o.timestampBegin;
  const tsEnd = tsBegin + 1500;
  const words = o.words;
  const chunks: (WhisperChunk & { words?: WhisperWord[] })[] = [];
  const newChunk = (): WhisperChunk => ({ timestamp: [null, null], text: "" });
  let chunk: WhisperChunk & { words?: WhisperWord[] } = newChunk();
  let timeOffset = 0;
  let previous: number[][] = [];
  let previousT: [number, number | null][][] = [];
  let skip = false;
  const collate = (r: { tokens: number[]; times: [number, number | null][] }) => collateWords(r.tokens, r.times, { decode: o.decode, ...words! });
  for (const out of sequences) {
    const ids = out.tokens;
    const tt = words ? out.tokenTimestamps ?? [] : [];
    let lastTimestamp: number | null = null;
    let firstTimestamp = tsBegin;
    const [chunkLen, strideLeft, strideRight] = out.stride;
    timeOffset -= strideLeft;
    const rightStrideStart = chunkLen - strideRight;
    if (strideLeft) firstTimestamp = strideLeft / o.timePrecision + tsBegin;
    if (strideRight) {
      for (let i = ids.length - 1; i >= 0; i--) {
        const t = ids[i]!;
        if (t >= tsBegin) {
          if (lastTimestamp !== null && (t - tsBegin) * o.timePrecision < rightStrideStart) break;
          lastTimestamp = t;
        }
      }
    }
    let current: number[] = [];
    let currentT: [number, number | null][] = [];
    for (let i = 0; i < ids.length; i++) {
      const t = ids[i]!;
      if (t >= o.firstSpecial && t < tsBegin) continue; // special tokens
      if (t >= tsBegin && t <= tsEnd) {
        const time = round2((t - tsBegin) * o.timePrecision + timeOffset);
        if (lastTimestamp !== null && t >= lastTimestamp) skip = true;
        else if (skip || (previous.length > 0 && t < firstTimestamp)) skip = false;
        else if (chunk.timestamp[0] === null) chunk.timestamp[0] = time;
        else if (time !== chunk.timestamp[0]) {
          chunk.timestamp[1] = time;
          previous.push(current);
          if (words) previousT.push(currentT);
          const merged = mergeSequences(previous, previousT);
          chunk.text = o.decode(merged.tokens);
          if (words) {
            chunk.words = collate(merged);
            for (const w of chunk.words) if (w.timestamp[1]! > time && time >= w.timestamp[0]) w.timestamp[1] = time;
          }
          chunks.push(chunk);
          previous = [];
          current = [];
          previousT = [];
          currentT = [];
          chunk = newChunk();
        }
      } else {
        current.push(t);
        if (words) {
          const start = round2(tt[i]! + timeOffset);
          let end: number | null = null;
          if (i + 1 < tt.length) {
            end = round2(tt[i + 1]! + timeOffset);
            // Punctuation gets no duration, so long pauses do not stretch it.
            if (PUNCTUATION_ONLY.test(o.decode([t]))) end = round2(Math.min(start + o.timePrecision, end));
          }
          currentT.push([start, end]);
        }
      }
    }
    timeOffset += chunkLen - strideRight;
    if (current.length) {
      previous.push(current);
      if (words) previousT.push(currentT);
    } else if (previous.every((p) => p.length === 0)) {
      chunk = newChunk();
      previous = [];
      previousT = [];
    }
  }
  if (previous.length) {
    const merged = mergeSequences(previous, previousT);
    chunk.text = o.decode(merged.tokens);
    if (words) chunk.words = collate(merged);
    chunks.push(chunk);
  }
  const text = chunks.map((c) => c.text).join("");
  if (words) return { text, chunks: chunks.flatMap((c) => c.words ?? []) };
  return o.timestamps ? { text, chunks } : { text };
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
    // Whisper's word timestamps take the median filter width from config.json (transformers' default 7).
    if (config!.median_filter_width !== undefined && gen.median_filter_width === undefined) gen.median_filter_width = config!.median_filter_width;
    return new Seq2SeqEngine(backend, cfg, tokenizer, model, gen);
  } finally {
    for (const s of sources) s.close?.();
  }
}
