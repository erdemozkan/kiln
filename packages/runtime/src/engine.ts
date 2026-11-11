import { physical, type Backend, type Buf, type Dispatch, type Profile } from "@kiln/core";
import { GrammarMatcher, TokenTrie, compileFormat, type ResponseFormat } from "@kiln/grammar";
import { LlamaModel, MAX_VERIFY_ROWS, type ForwardPlan, type Idefics3Processing, type ImagePixels, type VisionTower } from "@kiln/models";
import { idefics3PromptString, idefics3Tiles } from "./idefics3";
import { detectChatFormat, type ChatDocument, type ChatFormat, type ChatMessage, type Tokenizer, type ToolSpec } from "@kiln/tokenizers";
import { SubmitPacer } from "./pace";
import { PrefixStore } from "./prefix-store";
import { DraftGate, acceptedLength, lookupDraft } from "./speculate";
import { loadLoraAdapter, type LoraAdapter } from "./lora";
import type { LoraBuffers } from "@kiln/models";

export interface GenerateOptions {
  /** The LoRA adapter to generate with (a name given to `loadAdapter`), or null for the base model; default the current one. */
  adapter?: string | null;
  maxTokens?: number;
  /** 0 = greedy (default 0.7). */
  temperature?: number;
  seed?: number;
  /** Sample only among the k most likely tokens (0 = off, default). */
  topK?: number;
  /** Nucleus sampling: the smallest set of tokens reaching this probability (1 = off, default). */
  topP?: number;
  /** HF-style repetition penalty over prompt and generated tokens (1 = off, default). */
  repetitionPenalty?: number;
  signal?: AbortSignal;
  /** Extra token ids that end generation. */
  stopTokenIds?: number[];
  /**
   * Decode steps chained into one GPU submit (default 4). Higher amortises submit and readback latency;
   * tokens then arrive in bursts of this size, and up to two bursts are computed past a stop token.
   */
  stepsPerSubmit?: number;
  /**
   * Reuse the KV cache for the longest prefix this prompt shares with the previous call's tokens
   * (default true), so multi-turn chats only prefill the new turn.
   */
  reuseCache?: boolean;
  /** Qwen3 reasoning toggle; ignored by other families. */
  enableThinking?: boolean;
  /**
   * Functions the model may call (`chat` only), passed to the model's chat template as transformers
   * does. Calls come back as text in the template's format; `parseToolCalls` reads the
   * `<tool_call>`, Mistral and Granite ones. Built-in fallback formats reject tools.
   */
  tools?: ToolSpec[];
  /** Documents to ground the answer in (`chat` only), for templates that read `documents`. */
  documents?: ChatDocument[];
  /**
   * "auto" (default): the model decides whether to call a tool, unconstrained. "required", or one
   * function as `{ type: "function", function: { name } }`: the reply is exactly one call to one of
   * `tools` (or that one), with arguments that follow its `parameters` schema (see the `tool_call`
   * response format). Needs a template in the `<tool_call>` format; not combined with `format`.
   */
  toolChoice?: "auto" | "required" | { type: "function"; function: { name: string } };
  /**
   * Constrain the output: `{ type: "json" }` for any JSON value, `{ type: "json", schema }` for a JSON
   * Schema (properties in declared order), `{ type: "choice", choices }` for one of a set of strings,
   * `{ type: "regex", pattern }` for a regular expression, `{ type: "grammar", grammar }` for a GBNF
   * grammar. Each step then waits for the CPU to compute the next token mask, so steps are not chained.
   */
  format?: ResponseFormat;
  /**
   * Speculative decoding by prompt lookup: when the last tokens appeared earlier in the prompt or the
   * output, up to this many of the tokens that followed are guessed and checked in one forward pass
   * (default 7, which fills one 8-row matmul tile, and 0 on the CPU backend, where a checked row costs as
   * much as a decoded one; 0 = off; at most 31). Faster when the output repeats its
   * input, and a check is only spent when it is likely to pay off on this GPU. Works with `format`:
   * drafts are cut at the first token the grammar rejects and each row samples under its own mask.
   * Sampling draws the same random numbers either way, but the multi-row kernels round differently
   * from single-token decode, so a near-tied token can come out differently, as with f16 math. Not
   * combined with `repetitionPenalty` (that decodes normally).
   */
  draftTokens?: number;
}

export interface GenerateStats {
  promptTokens: number;
  /** Prompt tokens whose KV entries were reused from the previous call instead of prefilled. */
  cachedTokens: number;
  generatedTokens: number;
  prefillMs: number;
  decodeMs: number;
  /** Time to first token. */
  ttftMs: number;
  /** CPU time spent computing grammar masks (format only). */
  maskMs?: number;
  /** GPU submits this call made (work over `maxSubmitMs` is cut into several). */
  submits?: number;
  /** Speculative decoding (draftTokens): forward passes, draft tokens proposed and draft tokens kept. */
  speculation?: { passes: number; drafted: number; accepted: number };
  /**
   * Why generation ended: "stop" (an end-of-turn or stop token), "maxTokens", "context" (the reply
   * filled the context before maxTokens), "format" (the structured value is complete) or "abort"
   * (the signal fired, or the caller stopped reading).
   */
  stopReason?: StopReason;
}

export type StopReason = "stop" | "maxTokens" | "context" | "format" | "abort";

/** K/V cache contents for a token prefix, from `Engine.exportCache` (see `LlamaModel.readKv`). */
export interface CacheSnapshot {
  tokens: number[];
  kvDtype: string;
  layers: number;
  /** Bytes per position per layer, for K and for V each. */
  rowBytes: number;
  data: Uint8Array;
}

export interface TextDelta {
  text: string;
  tokenId: number;
}

/** Runs of image token ids (from `vocab` on) in a prompt, as [start, end) positions. */
export function imageSpans(ids: readonly number[], vocab: number): [number, number][] {
  const spans: [number, number][] = [];
  for (let i = 0; i < ids.length; i++) {
    if (ids[i]! < vocab) continue;
    const s = i;
    while (i < ids.length && ids[i]! >= vocab) i++;
    spans.push([s, i]);
  }
  return spans;
}

/** FNV-1a over an image's bytes and size: the same picture gets the same slot (and so the same ids). */
export function imageKey(img: ImagePixels): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193 ^ img.width;
  const d = img.data;
  for (let i = 0; i < d.length; i++) {
    h1 = Math.imul(h1 ^ d[i]!, 16777619);
    h2 = Math.imul(h2 ^ d[i]!, 2246822519);
  }
  return `${img.width}x${img.height}:${(h1 >>> 0).toString(16)}${(h2 >>> 0).toString(16)}`;
}

/** Shortest cached prefix worth saving before a call overwrites it. */
const MIN_SAVED_TOKENS = 32;

/** Prefill buckets. Prompts longer than the largest bucket are prefilled in chunks. */
const BUCKETS = [16, 64, 128];

export class Engine {
  readonly chatFormat: ChatFormat | undefined;
  lastStats: GenerateStats | undefined;
  /** The stop token that ended the last generation (undefined when it ended otherwise). */
  lastStopId: number | undefined;
  private readonly plans = new Map<number, ForwardPlan>();
  private readonly verifyPlans = new Map<number, ForwardPlan>();
  private readonly advances = new Map<Buf, { dispatches: readonly Dispatch[]; destroy(): void }>();
  private readonly chains = new Map<number, Dispatch[]>();
  /** LoRA adapters loaded, by name, and the one in the slots (undefined: the base model). */
  private readonly adapters = new Map<string, LoraAdapter>();
  private activeAdapter: string | undefined;
  /** Each loaded adapter laid out for the slot rank in GPU buffers, so a switch is GPU copies only. */
  private readonly prepared = new Map<string, LoraBuffers>();
  private readonly stopIds: Set<number>;
  private busy = false;
  /** Tokens whose K/V currently sit in the cache at positions 0…n-1 (empty while generating). */
  private cached: number[] = [];
  /**
   * Longest GPU work in one submit, in estimated milliseconds (default 50). Longer work is cut into
   * several submits so the display and other apps get the GPU in between (see `SubmitPacer`).
   */
  maxSubmitMs = 50;
  /** GPU cost per dispatch of single-row plans and per dispatch-row of multi-row ones, as measured. */
  private readonly pace = { rows1: new SubmitPacer(0.12), rows: new SubmitPacer(0.025) };
  /** Submits since the current call started (GenerateStats.submits). */
  private submits = 0;
  /** Token bytes in a prefix tree, built on the first constrained call. */
  private trie: { trie: TokenTrie; tokens: (Uint8Array | undefined)[] } | undefined;
  /** Upper bound on the cache positions written so far (for ring caches, see `minReuseStart`). */
  private written = 0;
  /**
   * K/V of earlier prefixes kept on the GPU (other conversations, other system prompts), up to
   * `prefixCacheBytes` (default 256 MiB, 0 to keep none). A call starts from whichever prefix, the
   * live cache or a saved one, shares the most tokens with its prompt; the live cache is saved first
   * when a call would overwrite it. Continuing one conversation saves nothing.
   */
  private readonly prefixes = new PrefixStore<Buf[]>(256 * 2 ** 20, (bufs) => bufs.forEach((b) => this.backend.free(b)));

  get prefixCacheBytes(): number {
    return this.prefixes.budget;
  }

  set prefixCacheBytes(bytes: number) {
    this.prefixes.budget = bytes;
    if (bytes <= 0) this.prefixes.clear();
  }

  /** Saved prefixes and the GPU bytes they hold. */
  get savedPrefixes(): { count: number; bytes: number } {
    return { count: this.prefixes.size, bytes: this.prefixes.bytes };
  }

  /** Images whose features sit in `model.imageFeatures`, by slot (see `imageSlot`). */
  private readonly imageSlots: ({ key: string; used: number } | undefined)[] = [];
  private imageClock = 0;

  constructor(
    readonly backend: Backend,
    readonly model: LlamaModel,
    readonly tokenizer: Tokenizer,
    /** A vision tower (Gemma 3 with its mmproj file, or SmolVLM): chat messages may then hold images. */
    readonly vision?: VisionTower,
  ) {
    try {
      this.chatFormat = detectChatFormat(tokenizer);
    } catch {
      this.chatFormat = undefined;
    }
    const stopStrings = this.chatFormat?.stop ?? [];
    this.stopIds = new Set([
      ...model.cfg.eosTokenIds,
      ...stopStrings.map((s) => tokenizer.tokenId(s)).filter((id): id is number => id !== undefined),
    ]);
  }

  async *chat(messages: readonly ChatMessage[], opts: GenerateOptions = {}): AsyncGenerator<TextDelta> {
    if (!this.chatFormat) throw new Error("this model has no supported chat format; use complete() instead");
    // Only reasoning models know <think>; injecting it elsewhere derails them.
    const thinks = this.tokenizer.tokenId("<think>") !== undefined;
    const images = messages.flatMap((m) => (typeof m.content === "string" ? [] : m.content.flatMap((p) => (p.type === "image" ? [p.image as ImagePixels] : []))));
    if (images.length && !this.vision) throw new Error("this model takes no images (load Gemma 3 with its mmproj file, or a SmolVLM model, to give it images)");
    const idefics3 = this.vision?.processing;
    // Templates see image parts without their pixels; Gemma's writes <start_of_image> for each,
    // SmolVLM's <image>. SmolVLM's template reads every message's content as a list of parts.
    const shown = messages.map((m) =>
      typeof m.content === "string"
        ? idefics3
          ? { ...m, content: [{ type: "text" as const, text: m.content }] }
          : m
        : { ...m, content: m.content.map((p) => (p.type === "image" ? { type: "image" as const, image: null } : p)) },
    );
    const prompt = this.chatFormat.render(shown, {
      enableThinking: thinks ? opts.enableThinking : undefined,
      tools: opts.tools,
      documents: opts.documents,
    });
    if (!images.length) {
      yield* this.complete(prompt, this.forceToolCall(opts));
      return;
    }
    if (idefics3) {
      yield* this.decodeIds(await this.withTiles(prompt, images, idefics3), this.forceToolCall(opts));
      return;
    }
    const slots: number[] = [];
    for (const img of images) slots.push(await this.imageSlot(img));
    yield* this.decodeIds(this.withImages(prompt, slots), this.forceToolCall(opts));
  }

  /**
   * Prompt ids as Idefics3's processor expands them (SmolVLM): each <image> the template wrote becomes
   * the image's tiles and global view, each as its row/column tag and `tokens` image tokens (see
   * `idefics3PromptString`). Each tile has its own feature slot, and its run of image tokens gets
   * ids vocabSize + slot · tokens + i.
   */
  private async withTiles(prompt: string, images: readonly ImagePixels[], p: Idefics3Processing): Promise<number[]> {
    const vision = this.vision!;
    const parts = prompt.split("<image>");
    if (parts.length - 1 !== images.length) throw new Error(`the chat template wrote ${parts.length - 1} image markers for ${images.length} images`);
    let text = parts[0]!;
    const tiles: ImagePixels[] = [];
    for (let i = 0; i < images.length; i++) {
      const t = idefics3Tiles(images[i]!, p);
      tiles.push(...t.tiles);
      text += idefics3PromptString(t.rows, t.cols, vision.tokens) + parts[i + 1]!;
    }
    const cap = (this.model.imageFeatures!.size / 4 / this.model.cfg.hiddenSize / vision.tokens) | 0;
    if (tiles.length > cap) throw new Error(`these images make ${tiles.length} tiles, but ${cap} fit; load with a larger imageSlots, or imageSplitting: false`);
    const slots: number[] = [];
    for (const tile of tiles) slots.push(await this.imageSlot(tile));
    const ids = this.tokenizer.encode(text);
    const per = vision.tokens;
    let k = 0;
    for (let i = 0; i < ids.length; i++) {
      if (ids[i] !== p.imageTokenId) continue;
      const base = this.model.cfg.vocabSize + slots[k++]! * per;
      for (let j = 0; j < per; j++, i++) {
        if (ids[i] !== p.imageTokenId) throw new Error("Idefics3: a run of image tokens is shorter than the image's tokens");
        ids[i] = base + j;
      }
      i--;
    }
    if (k !== tiles.length) throw new Error(`Idefics3: the prompt holds ${k} tiles' image tokens for ${tiles.length} tiles`);
    return ids;
  }

  /**
   * Prompt ids with each image's tokens in place, as Gemma 3's processor expands them: every
   * <start_of_image> the template wrote becomes "\n\n<start_of_image>", the image's 256 soft tokens
   * and "<end_of_image>\n\n". The soft tokens get ids vocabSize + slot · 256 + i, which the embedding
   * reads from the image features (and which differ between images, so cached prefixes never mix them).
   */
  private withImages(prompt: string, slots: readonly number[]): number[] {
    const boi = "<start_of_image>";
    const parts = prompt.split(boi);
    if (parts.length - 1 !== slots.length) throw new Error(`the chat template wrote ${parts.length - 1} image markers for ${slots.length} images`);
    const boiId = this.tokenizer.tokenId(boi)!;
    const ids = this.tokenizer.encode(parts.join(`\n\n${boi}<end_of_image>\n\n`));
    const out: number[] = [];
    const per = this.vision!.tokens;
    let k = 0;
    for (const id of ids) {
      out.push(id);
      if (id !== boiId) continue;
      const base = this.model.cfg.vocabSize + slots[k++]! * per;
      for (let i = 0; i < per; i++) out.push(base + i);
    }
    return out;
  }

  /**
   * The feature slot holding this image, running the vision tower if it is not there yet (the least
   * recently used slot is reused; cached prefixes that saw its old image are forgotten).
   */
  private async imageSlot(img: ImagePixels): Promise<number> {
    const vision = this.vision!;
    const key = imageKey(img);
    const cap = (this.model.imageFeatures!.size / 4 / this.model.cfg.hiddenSize / vision.tokens) | 0;
    let slot = this.imageSlots.findIndex((s) => s?.key === key);
    if (slot < 0) {
      slot = this.imageSlots.length < cap ? this.imageSlots.length : this.imageSlots.reduce((lru, s, i) => (s!.used < this.imageSlots[lru]!.used ? i : lru), 0);
      if (this.imageSlots[slot]) this.forgetImage(slot);
      this.imageSlots[slot] = { key, used: 0 };
      await vision.encode(img, this.model.imageFeatures!, slot * vision.tokens * this.model.cfg.hiddenSize * 4);
    }
    this.imageSlots[slot]!.used = ++this.imageClock;
    return slot;
  }

  /** Drops every cached prefix that holds slot's image tokens, before the slot takes another image. */
  private forgetImage(slot: number): void {
    const per = this.vision!.tokens;
    const lo = this.model.cfg.vocabSize + slot * per;
    const holds = (ids: readonly number[]) => ids.findIndex((id) => id >= lo && id < lo + per);
    const at = holds(this.cached);
    if (at >= 0) this.cached = this.cached.slice(0, at);
    this.prefixes.dropIf((tokens) => holds(tokens) >= 0);
  }

  /** `opts` with `toolChoice` turned into a `tool_call` format. */
  private forceToolCall(opts: GenerateOptions): GenerateOptions {
    const choice = opts.toolChoice ?? "auto";
    if (choice === "auto") return opts;
    if (!opts.tools?.length) throw new Error("toolChoice needs tools");
    if (opts.format) throw new Error("toolChoice and format cannot be combined");
    if (!this.chatFormat?.toolCallTags) throw new Error("toolChoice needs a chat template that writes calls as <tool_call>{…}</tool_call>");
    const tools = choice === "required" ? opts.tools : opts.tools.filter((t) => t.function.name === choice.function.name);
    if (!tools.length) throw new Error(`toolChoice names ${JSON.stringify((choice as { function: { name: string } }).function.name)}, which is not among the tools`);
    return { ...opts, format: { type: "tool_call", tools: tools.map((t) => ({ name: t.function.name, parameters: t.function.parameters })) } };
  }

  /**
   * Log-probabilities of each choice as the assistant's whole reply to `messages` (the choice's
   * tokens then the end-of-turn token, so a label is not favoured for being a prefix of another):
   * for scoring labels with the model's own probabilities. Text messages only.
   */
  async scoreChoices(messages: readonly ChatMessage[], choices: readonly string[], opts: { adapter?: string | null; enableThinking?: boolean } = {}): Promise<number[]> {
    if (!this.chatFormat) throw new Error("this model has no supported chat format");
    if (messages.some((m) => typeof m.content !== "string")) throw new Error("scoreChoices takes text messages only");
    const thinks = this.tokenizer.tokenId("<think>") !== undefined;
    const prompt = this.chatFormat.render(messages, { enableThinking: thinks ? (opts.enableThinking ?? false) : undefined });
    const promptIds = this.tokenizer.encode(prompt);
    const end = this.chatFormat.stop.map((s) => this.tokenizer.tokenId(s)).find((id) => id !== undefined) ?? this.model.cfg.eosTokenIds[0];
    if (end === undefined) throw new Error("this model's chat format has no end-of-turn token to score choices with");
    const conts = choices.map((c) => {
      // Encoded with the prompt where the boundary allows, as the model would have produced it.
      const full = this.tokenizer.encode(prompt + c);
      const joined = full.length > promptIds.length && promptIds.every((id, i) => full[i] === id);
      return [...(joined ? full.slice(promptIds.length) : this.tokenizer.encode(c)), end];
    });
    if (opts.adapter !== undefined) this.setAdapter(opts.adapter);
    return this.scoreIds(promptIds, conts);
  }

  /**
   * Sum of log-probabilities of each continuation after `promptIds` (up to MAX_VERIFY_ROWS tokens
   * each): the prompt is prefilled once (reusing the cached prefix), then each continuation runs as
   * one verify pass whose rows' logits are read back.
   */
  async scoreIds(promptIds: readonly number[], conts: readonly (readonly number[])[]): Promise<number[]> {
    if (this.busy) throw new Error("Engine is already generating; await or abort the previous call");
    const n = promptIds.length;
    if (n === 0) throw new Error("prompt is empty");
    if (promptIds.some((id) => id >= this.model.cfg.vocabSize)) throw new Error("scoring prompts with images is not supported");
    for (const c of conts) {
      if (!c.length) throw new Error("a continuation is empty");
      if (c.length > MAX_VERIFY_ROWS) throw new Error(`a continuation has ${c.length} tokens; at most ${MAX_VERIFY_ROWS} are scored`);
    }
    if (n + Math.max(...conts.map((c) => c.length)) > this.model.maxSeq) throw new Error("prompt and choices do not fit the context");
    const prefix = promptIds.slice(0, n - 1);
    let start = 0;
    while (start < this.cached.length && start < prefix.length && this.cached[start] === prefix[start]) start++;
    if (start < this.model.minReuseStart(this.written)) start = 0;
    if (start < this.cached.length) this.saveLive();
    this.cached = [];
    this.busy = true;
    let failed = false;
    let longest = 0;
    try {
      const step = { seed: 0, temperature: 0 };
      let pos = start;
      while (pos < prefix.length) {
        const remaining = prefix.length - pos;
        const top = BUCKETS[BUCKETS.length - 1]!;
        const T = this.backend.name === "cpu" ? Math.min(remaining, top) : (BUCKETS.find((b) => b >= remaining) ?? top);
        const k = Math.min(T, remaining);
        const plan = this.plan(T);
        const tokens = new Uint32Array(T);
        tokens.set(prefix.slice(pos, pos + k));
        this.backend.write(plan.tokens, 0, tokens);
        this.model.setStep({ pos0: pos, lastRow: k - 1, ...step });
        this.submit(plan.body, T);
        pos += k;
      }
      const out: number[] = [];
      for (const c of conts) {
        const T = c.length;
        longest = Math.max(longest, T);
        const plan = this.verifyPlan(T);
        this.backend.write(plan.tokens, 0, Uint32Array.from([promptIds[n - 1]!, ...c.slice(0, -1)]));
        this.model.setStep({ pos0: n - 1, lastRow: 0, ...step });
        this.submit(plan.dispatches, T);
        const V = plan.logits.shape[1]!;
        const rows = new Float32Array(await this.backend.readback(physical(plan.logits.buf), 0, T * V * 4));
        let lp = 0;
        for (let t = 0; t < T; t++) {
          const row = rows.subarray(t * V, (t + 1) * V);
          let max = -Infinity;
          for (let i = 0; i < V; i++) if (row[i]! > max) max = row[i]!;
          let sum = 0;
          for (let i = 0; i < V; i++) sum += Math.exp(row[i]! - max);
          lp += row[c[t]!]! - max - Math.log(sum);
        }
        out.push(lp);
      }
      return out;
    } catch (err) {
      failed = true;
      throw err;
    } finally {
      // The prefix's K/V stand; positions from n - 1 on hold the last continuation's.
      this.cached = failed ? [] : [...prefix];
      this.written = failed ? this.model.maxSeq : Math.max(this.written, n - 1 + longest);
      this.busy = false;
    }
  }

  async *complete(prompt: string, opts: GenerateOptions = {}): AsyncGenerator<TextDelta> {
    yield* this.decodeIds(this.tokenizer.encode(prompt), opts);
  }

  /** Generates from prompt ids, yielding text as it decodes. */
  private async *decodeIds(ids: readonly number[], opts: GenerateOptions): AsyncGenerator<TextDelta> {
    const decoder = this.tokenizer.decoder();
    for await (const id of this.generateTokens(ids, opts)) {
      const text = decoder.push(id);
      if (text) yield { text, tokenId: id };
    }
    const tail = decoder.flush();
    if (tail) yield { text: tail, tokenId: -1 };
  }

  /**
   * Token-level generation. Each decode step is a replay of the same recorded plan. The GPU feeds each
   * sampled id back as the next input and advances the position itself, so `stepsPerSubmit` steps go
   * out in one submit and come back in one 4·k-byte readback; two batches are kept in flight so the
   * GPU never waits on the CPU.
   */
  async *generateTokens(promptIds: readonly number[], opts: GenerateOptions = {}): AsyncGenerator<number> {
    if (this.busy) throw new Error("Engine is already generating; await or abort the previous call");
    if (promptIds.length === 0) throw new Error("prompt is empty");
    if (opts.adapter !== undefined) this.setAdapter(opts.adapter);
    const maxSeq = this.model.maxSeq;
    if (promptIds.length >= maxSeq) {
      const trained = this.model.cfg.contextLength;
      const hint = trained && trained > maxSeq ? `; load with a larger maxSeq (this model was trained for ${trained})` : "";
      throw new Error(`prompt has ${promptIds.length} tokens but the context is ${maxSeq}${hint}`);
    }
    const requested = opts.maxTokens ?? 512;
    const maxTokens = Math.min(requested, maxSeq - promptIds.length);
    /** Set when a stop token ends the reply (as opposed to the caller ending it). */
    const ended: { stop: boolean; id?: number } = { stop: false };
    const temperature = opts.temperature ?? 0.7;
    const seed = opts.seed ?? (Math.random() * 2 ** 32) >>> 0;
    const sampling = { topK: opts.topK ?? 0, topP: opts.topP ?? 1, repetitionPenalty: opts.repetitionPenalty ?? 1 };
    if (sampling.topK < 0 || !(sampling.topP > 0 && sampling.topP <= 1) || !(sampling.repetitionPenalty > 0)) {
      throw new Error(`invalid sampling options ${JSON.stringify(sampling)}`);
    }
    const stop = new Set([...this.stopIds, ...(opts.stopTokenIds ?? [])]);
    const t0 = performance.now();
    const grammar = opts.format ? this.matcher(opts.format, [...stop]) : undefined;
    const draftTokens = sampling.repetitionPenalty !== 1 ? 0 : Math.min(MAX_VERIFY_ROWS - 1, Math.max(0, Math.floor(opts.draftTokens ?? (this.backend.name === "cpu" ? 0 : 7))));
    this.busy = true;
    this.lastStopId = undefined;
    // Reuse the shared prefix, but always prefill at least the last prompt token to get its logits.
    let start = 0;
    const reuse = opts.reuseCache ?? true;
    if (reuse) {
      const prev = this.cached;
      while (start < prev.length && start < promptIds.length - 1 && prev[start] === promptIds[start]) start++;
      // Ring caches (sliding layers) keep only the last window + slack positions written; resuming
      // further back would read overwritten keys, so the prompt is prefilled from the start instead.
      if (start < this.model.minReuseStart(this.written)) start = 0;
    }
    const saved = reuse ? this.prefixes.best(promptIds, start, (w) => this.model.minReuseStart(w)) : undefined;
    // The live cache is about to be overwritten past `start` (or replaced): keep it for later calls.
    if (saved || start < this.cached.length) this.saveLive();
    if (saved) {
      this.model.restoreKv(saved.entry.tokens.length, saved.entry.data);
      this.written = saved.entry.written;
      start = saved.length;
    }
    // An image's tokens attend to each other both ways (Gemma 3), so each image is prefilled whole, in
    // one chunk. SmolVLM's are causal, like text.
    const spans = this.vision?.bidirectional ? imageSpans(promptIds, this.model.cfg.vocabSize) : [];
    for (const [lo, hi] of spans) if (start > lo && start < hi) start = lo;
    this.cached = [];
    const stats: GenerateStats = { promptTokens: promptIds.length, cachedTokens: start, generatedTokens: 0, prefillMs: 0, decodeMs: 0, ttftMs: 0 };
    this.submits = 0;
    // Ids read back so far (including any past a stop token) and decode steps submitted.
    const readIds: number[] = [];
    let queued = 0;
    let failed = false;
    /** Speculative decoding: prompt plus yielded ids; all but the last have K/V in the cache. */
    let spoken: number[] | undefined;

    try {
      this.model.setSeen(promptIds);
      if (grammar) {
        const t = performance.now();
        this.model.setAllowed(grammar.mask());
        stats.maskMs = performance.now() - t;
      } else this.model.setAllowed(undefined);
      let pos = start;
      let out = this.plan(1).out;
      let prefillUnits = 0;
      while (pos < promptIds.length) {
        const span = spans.find(([, hi]) => hi > pos);
        let T: number;
        let n: number;
        let bidi: { spanLo?: number; spanHi?: number } = {};
        if (span && pos >= span[0]) {
          // One image: exactly its tokens, attending to each other in both directions.
          T = n = span[1] - pos;
          bidi = { spanLo: span[0], spanHi: span[1] };
        } else {
          const remaining = (span ? span[0] : promptIds.length) - pos;
          const top = BUCKETS[BUCKETS.length - 1]!;
          // Padding rows are nearly free on a GPU, which reuses a few plan sizes, but cost their full
          // work on the CPU, which builds a plan of the exact size instead.
          T = this.backend.name === "cpu" ? Math.min(remaining, top) : (BUCKETS.find((b) => b >= remaining) ?? top);
          n = Math.min(T, remaining);
        }
        const plan = this.plan(T);
        const tokens = new Uint32Array(T);
        tokens.set(promptIds.slice(pos, pos + n));
        this.backend.write(plan.tokens, 0, tokens);
        this.model.setStep({ pos0: pos, lastRow: n - 1, seed, temperature, ...sampling, ...bidi });
        // Only the last chunk's logits are read; earlier chunks skip the output head.
        prefillUnits += this.submit(pos + n < promptIds.length ? plan.body : plan.dispatches, T);
        pos += n;
        out = plan.out;
      }

      if (draftTokens > 0) {
        spoken = [...promptIds];
        yield* this.decodeSpeculative(out, spoken, { draftTokens, maxTokens, stop, signal: opts.signal, seed, temperature, sampling, stats, t0, grammar, prefillUnits, ended });
        return;
      }

      const decode = this.plan(1);
      // A grammar needs each sampled id on the CPU before the next mask exists.
      const K = grammar ? 1 : Math.max(1, Math.floor(opts.stepsPerSubmit ?? 4));
      const reads: Promise<ArrayBuffer>[] = [];
      /** Single-row dispatches behind each read, for timing (0: not decode steps). */
      const readUnits: number[] = [];
      const submitBatch = () => {
        // The first batch only records the id sampled by prefill; later ones run k decode steps.
        const k = queued === 0 ? 1 : Math.min(K, maxTokens - queued);
        if (queued === 0) {
          this.submit(this.advance(out, decode.tokens));
          readUnits.push(0);
        } else readUnits.push(this.submit(this.chain(k)));
        reads.push(this.backend.readback(this.model.history, queued * 4, k * 4));
        queued += k;
      };
      submitBatch();
      if (!grammar && queued < maxTokens) submitBatch();
      let tDecode = 0;
      let tRead = 0;
      let done = false;
      while (!done && reads.length > 0) {
        const ids = new Uint32Array(await reads.shift()!);
        const units = readUnits.shift() ?? 0;
        const now = performance.now();
        readIds.push(...ids);
        if (!tDecode) {
          stats.ttftMs = stats.prefillMs = now - t0;
          tDecode = now;
          this.pace.rows.observe(stats.prefillMs, prefillUnits);
        } else if (units) {
          // With the next batch already queued, the gap between reads is this batch's GPU time.
          this.pace.rows1.observe(now - tRead, units);
        }
        tRead = now;
        if (grammar) {
          const id = ids[0]!;
          if (!stop.has(id)) {
            if (!grammar.accept(id)) throw new Error(`grammar: sampled token ${id} is not allowed (mask out of sync)`);
            // Once the value is complete and nothing can extend it, end without another step.
            if (grammar.finished) done = true;
            else if (queued < maxTokens) {
              // The forward pass only needs the id just sampled, so the GPU runs it while the CPU
              // computes the mask; the mask write is queued before the sampling dispatch that reads it.
              this.submit(decode.dispatches.slice(0, -1));
              const t = performance.now();
              this.model.setAllowed(grammar.mask());
              stats.maskMs! += performance.now() - t;
              this.submit(this.grammarTail());
              readUnits.push(0);
              reads.push(this.backend.readback(this.model.history, queued * 4, 4));
              queued++;
            }
          }
        } else if (queued < maxTokens && !opts.signal?.aborted && !ids.some((id) => stop.has(id))) {
          // Once this batch ends the reply, nothing more is queued: steps past the end would only delay
          // the next call, which waits for them on the queue. One batch may still be in flight.
          submitBatch();
        }
        for (const id of ids) {
          if (stop.has(id) || opts.signal?.aborted) {
            ended.stop = stop.has(id);
            if (ended.stop) ended.id = id;
            done = true;
            break;
          }
          stats.generatedTokens++;
          yield id;
        }
      }
      stats.decodeMs = tDecode ? performance.now() - tDecode : 0;
    } catch (err) {
      failed = true;
      throw err;
    } finally {
      // Decode step j fed id j at position promptLength + j; every submitted step writes its K/V,
      // including steps still in flight, which run before anything submitted later. This also holds
      // when the consumer stops iterating early.
      if (!failed) this.cached = spoken ? spoken.slice(0, -1) : [...promptIds, ...readIds.slice(0, Math.max(0, queued - 1))];
      // Verify passes write up to MAX_VERIFY_ROWS positions past the kept ones; plain decode none.
      this.written = failed ? this.model.maxSeq : this.cached.length + (spoken ? MAX_VERIFY_ROWS : 0);
      stats.submits = this.submits;
      stats.stopReason = opts.signal?.aborted
        ? "abort"
        : grammar?.finished
          ? "format"
          : stats.generatedTokens >= maxTokens
            ? maxTokens < requested ? "context" : "maxTokens"
            : ended.stop
              ? "stop"
              : "abort";
      this.lastStats = stats;
      this.lastStopId = ended.stop ? ended.id : undefined;
      this.busy = false;
    }
  }

  /**
   * Decoding with prompt-lookup drafts. A verify pass feeds the last token plus a draft, samples every
   * row, and keeps the draft up to the first token the model chose differently, plus the model's own
   * token there. K/V written for rejected draft rows lie past the next pass's start, which overwrites
   * them before anything reads them, so nothing has to be rolled back.
   *
   * Between passes, decoding runs as in plain mode: chained steps, two submits in flight, the first run
   * starting straight from the id prefill sampled. Drafts are then looked up on the tokens one submit
   * behind; once one looks worth a pass (`DraftGate`), no more steps are queued, and the pass drafts
   * again from the up-to-date tokens.
   *
   * With a grammar, a draft is cut at the first token the grammar rejects, and row r samples under the
   * mask after the first r draft tokens; the masks are computed while the GPU runs the forward pass.
   * Between passes it decodes one constrained step at a time, as plain constrained decoding does.
   */
  private async *decodeSpeculative(
    first: Buf,
    spoken: number[],
    o: {
      draftTokens: number;
      maxTokens: number;
      stop: Set<number>;
      signal: AbortSignal | undefined;
      seed: number;
      temperature: number;
      sampling: { topK: number; topP: number; repetitionPenalty: number };
      stats: GenerateStats;
      t0: number;
      grammar: GrammarMatcher | undefined;
      prefillUnits: number;
      ended: { stop: boolean; id?: number };
    },
  ): AsyncGenerator<number> {
    const { stats, grammar } = o;
    const spec = (stats.speculation = { passes: 0, drafted: 0, accepted: 0 });
    const gate = new DraftGate();
    const step = { lastRow: 0, seed: o.seed, temperature: o.temperature, ...o.sampling };
    const decode = this.plan(1);
    const maxSeq = this.model.maxSeq;
    let emit: number[] = [];
    let fromPrefill = true;
    let tDecode = 0;
    const firstRead = (now: number) => {
      if (tDecode) return false;
      stats.ttftMs = stats.prefillMs = now - o.t0;
      tDecode = now;
      this.pace.rows.observe(stats.prefillMs, o.prefillUnits);
      return true;
    };
    /** CPU time for masks goes to stats.maskMs. */
    const mask = () => {
      const t = performance.now();
      const m = grammar!.mask();
      stats.maskMs! += performance.now() - t;
      return m;
    };
    /** The draft for the current tokens, cut where the grammar (if any) rejects it or at a stop token. */
    const draftFor = () => {
      const d = lookupDraft(spoken, Math.min(o.draftTokens, maxSeq - spoken.length));
      if (!grammar) return d;
      const at = grammar.save();
      let k = 0;
      while (k < d.length && !o.stop.has(d[k]!) && grammar.accept(d[k]!)) k++;
      grammar.restore(at);
      return d.slice(0, k);
    };
    /** Records and yields ids; false once generation is over. */
    const speak = function* (ids: Iterable<number>): Generator<number, boolean> {
      for (const id of ids) {
        if (o.stop.has(id) || o.signal?.aborted) {
          o.ended.stop = o.stop.has(id);
          if (o.ended.stop) o.ended.id = id;
          return false;
        }
        if (grammar && !grammar.accept(id)) throw new Error(`grammar: sampled token ${id} is not allowed (mask out of sync)`);
        spoken.push(id);
        stats.generatedTokens++;
        yield id;
        // A complete value that nothing can extend ends without another step.
        if (stats.generatedTokens >= o.maxTokens || grammar?.finished) return false;
      }
      return true;
    };
    try {
      for (;;) {
        if (!(yield* speak(emit))) return;
        emit = [];
        const pos = spoken.length - 1;
        const draft = fromPrefill ? [] : draftFor();
        if (draft.length && gate.worth(draft.length)) {
          const t = performance.now();
          const plan = this.verifyPlan(1 + draft.length);
          this.backend.write(plan.tokens, 0, Uint32Array.from([spoken[pos]!, ...draft]));
          this.model.setStep({ pos0: pos, ...step });
          // The forward pass runs while the CPU computes the row masks; sampling is submitted after.
          const units = this.submit(plan.dispatches.slice(0, -1), plan.T);
          const masks: (Uint32Array | undefined)[] = [];
          if (grammar) {
            const at = grammar.save();
            for (let r = 0; r <= draft.length; r++) {
              masks.push(mask().slice());
              if (r < draft.length) grammar.accept(draft[r]!);
            }
            grammar.restore(at);
          } else masks.length = 1 + draft.length;
          this.model.setAllowedRows(masks);
          this.submit(plan.dispatches.slice(-1), plan.T);
          const sampled = new Uint32Array(await this.backend.readback(plan.out, 0, (1 + draft.length) * 4));
          firstRead(performance.now());
          const n = acceptedLength(draft, sampled);
          emit = Array.from(sampled.subarray(0, n + 1));
          gate.verified(performance.now() - t, draft.length, n);
          if (!grammar) this.pace.rows.observe(performance.now() - t, units);
          spec.passes++;
          spec.drafted += draft.length;
          spec.accepted += n;
          continue;
        }
        if (grammar) {
          // One constrained step: forward, mask while it runs, then sampling and advance.
          const t = performance.now();
          if (fromPrefill) {
            this.submit(this.advance(first, decode.tokens));
            fromPrefill = false;
          } else {
            this.backend.write(decode.tokens, 0, Uint32Array.of(spoken[pos]!));
            this.model.setStep({ pos0: pos, n: 0, ...step });
            this.submit(decode.dispatches.slice(0, -1));
            this.model.setAllowed(mask());
            this.submit(this.grammarTail());
          }
          emit = Array.from(new Uint32Array(await this.backend.readback(this.model.history, 0, 4)));
          const now = performance.now();
          if (!firstRead(now)) gate.plain(now - t, 1, draft.length > 0);
          continue;
        }
        // A plain run. Step j of the run feeds the id sampled by step j - 1 (the GPU advances itself).
        // The first run starts from prefill's step state and only records the id it sampled.
        const budget = Math.min(maxSeq - pos, o.maxTokens - stats.generatedTokens);
        if (budget <= 0) return;
        const reads: { ids: Promise<ArrayBuffer>; k: number; units: number }[] = [];
        let queued = 0;
        if (fromPrefill) {
          this.submit(this.advance(first, decode.tokens));
          reads.push({ ids: this.backend.readback(this.model.history, 0, 4), k: 1, units: 0 });
          queued = 1;
          fromPrefill = false;
        } else {
          this.backend.write(decode.tokens, 0, Uint32Array.of(spoken[pos]!));
          this.model.setStep({ pos0: pos, n: 0, ...step });
        }
        const submit = () => {
          const k = Math.min(4, budget - queued);
          if (k <= 0) return;
          const units = this.submit(this.chain(k));
          reads.push({ ids: this.backend.readback(this.model.history, queued * 4, k * 4), k, units });
          queued += k;
        };
        // Two submits in flight, as in plain decoding; the first run's prefill read is one of them.
        if (!reads.length) submit();
        submit();
        let draining = false;
        let t = performance.now();
        while (reads.length) {
          const r = reads.shift()!;
          const ids = new Uint32Array(await r.ids);
          const now = performance.now();
          // The first read is prefill's own id: it says nothing about decode speed.
          const isFirst = firstRead(now);
          if (!(yield* speak(ids))) return;
          if (draining) continue;
          const d = draftFor();
          const worth = d.length > 0 && gate.worth(d.length);
          if (!isFirst) {
            gate.plain(now - t, r.k, d.length > 0 && !worth);
            this.pace.rows1.observe(now - t, r.units);
          }
          t = now;
          if (worth) draining = true;
          else submit();
        }
      }
    } finally {
      stats.decodeMs = tDecode ? performance.now() - tDecode : 0;
    }
  }

  /**
   * Submits dispatches, cut into several submits when their estimated GPU time exceeds
   * `maxSubmitMs`. A plan of T > 1 rows costs as if it had at least 8 (the matmul row tile). Returns
   * the work's cost units, for `SubmitPacer.observe`.
   */
  private submit(dispatches: readonly Dispatch[], rows = 1): number {
    const pacer = rows > 1 ? this.pace.rows : this.pace.rows1;
    const units = dispatches.length * (rows > 1 ? Math.max(rows, 8) : 1);
    for (const part of SubmitPacer.split(dispatches, pacer.pieces(units, dispatches.length, this.maxSubmitMs))) {
      this.backend.submit(part);
      this.submits++;
    }
    return units;
  }

  private verifyPlan(T: number): ForwardPlan {
    let p = this.verifyPlans.get(T);
    if (!p) {
      p = this.model.buildPlan(T, true);
      this.verifyPlans.set(T, p);
    }
    return p;
  }

  private matcher(format: ResponseFormat, stopIds: number[]): GrammarMatcher {
    const grammar = compileFormat(format);
    if (!this.trie) {
      const tokens = Array.from({ length: this.tokenizer.vocabSize }, (_, id) => this.tokenizer.tokenBytes(id));
      this.trie = { trie: new TokenTrie(tokens), tokens };
    }
    return new GrammarMatcher(grammar, this.trie.trie, this.trie.tokens, this.model.cfg.vocabSize, stopIds);
  }

  /**
   * The K/V of the tokens currently cached (the last call's prompt and reply), to restore with
   * `importCache` in a later session so a shared prefix is not prefilled again.
   */
  async exportCache(): Promise<CacheSnapshot> {
    if (this.busy) throw new Error("exportCache: the engine is generating");
    const tokens = [...this.cached];
    const m = this.model;
    return { tokens, kvDtype: m.kvDtype, layers: m.cfg.numLayers, rowBytes: m.kvRowBytes, data: await m.readKv(tokens.length) };
  }

  /** Loads a snapshot from `exportCache`; returns false (and changes nothing) if it does not fit this model. */
  importCache(s: CacheSnapshot): boolean {
    const m = this.model;
    if (this.busy || s.kvDtype !== m.kvDtype || s.layers !== m.cfg.numLayers || s.rowBytes !== m.kvRowBytes || s.tokens.length >= m.maxSeq) return false;
    if (s.data.length !== m.kvSnapshotBytes(s.tokens.length)) return false;
    m.writeKv(s.tokens.length, s.data);
    // Image feature slots start empty after a load, so ids of an earlier image would match a new one's.
    const image = s.tokens.findIndex((id) => id >= m.cfg.vocabSize);
    this.cached = image < 0 ? [...s.tokens] : s.tokens.slice(0, image);
    this.written = s.tokens.length;
    return true;
  }

  /**
   * Loads a PEFT LoRA adapter repo under `name` (it is used once `setAdapter(name)` or a call's
   * `adapter` picks it). Its projections must match this model's shapes.
   */
  async loadAdapter(name: string, id: string, opts: { revision?: string; endpoint?: string } = {}): Promise<{ name: string; id: string; rank: number; modules: string[]; base?: string }> {
    const a = await loadLoraAdapter(id, opts);
    const c = this.model.cfg;
    for (const key of a.weights.keys()) if (Number(key.split(".")[0]) >= c.numLayers) throw new Error(`${id} adapts layer ${key.split(".")[0]}, but this model has ${c.numLayers} layers (another base model?)`);
    const old = this.prepared.get(name);
    if (old) this.model.freeLoraBuffers(old);
    this.prepared.delete(name);
    this.adapters.set(name, a);
    // Laid out on the GPU now, so switching to it later costs copies only.
    this.prepareAll();
    if (this.activeAdapter === name) this.setAdapter(name, true);
    return { name, id, rank: a.rank, modules: a.modules, base: a.base };
  }

  /** The slot rank every loaded adapter fits. */
  private loraRankNeeded(): number {
    return LlamaModel.loraSlotRank(Math.max(0, ...[...this.adapters.values()].map((x) => x.rank)));
  }

  /** Lays out every loaded adapter for the slot rank they need, re-laying any prepared for another rank. */
  private prepareAll(): void {
    const R = this.loraRankNeeded();
    for (const [name, a] of this.adapters) {
      const p = this.prepared.get(name);
      if (p && p.rank === R) continue;
      if (p) this.model.freeLoraBuffers(p);
      this.prepared.set(name, this.model.prepareLora(a.weights, R));
    }
  }

  /** Names of the adapters loaded, and the one in use (undefined: the base model). */
  get adapterInfo(): { loaded: string[]; active: string | undefined } {
    return { loaded: [...this.adapters.keys()], active: this.activeAdapter };
  }

  /**
   * Generates with adapter `name` from now on, or with the base model (null). Switching between
   * adapters writes their weights into the model's LoRA slots (no plan is rebuilt unless a larger
   * rank needs bigger slots); going back to the base model frees the slots. Cached prefixes are
   * dropped either way: their K/V belong to the weights that made them.
   */
  setAdapter(name: string | null, force = false): void {
    if (this.busy) throw new Error("setAdapter: the engine is generating");
    if (!force && (name ?? undefined) === this.activeAdapter) return;
    if (name === null) {
      if (this.model.loraRank) {
        this.model.enableLora(0);
        this.dropPlans();
      }
    } else {
      if (!this.adapters.has(name)) throw new Error(`no adapter named "${name}" is loaded (loaded: ${[...this.adapters.keys()].join(", ") || "none"})`);
      const R = this.loraRankNeeded();
      if (this.model.loraRank !== R) {
        // Slots for the largest adapter loaded, so switching among them needs no further rebuild.
        this.model.enableLora(R);
        this.dropPlans();
      }
      this.prepareAll();
      this.model.useLora(this.prepared.get(name)!);
    }
    this.activeAdapter = name ?? undefined;
    this.resetCache();
  }

  /** Forgets adapter `name` (switching to the base model if it was in use). */
  unloadAdapter(name: string): void {
    if (this.activeAdapter === name) this.setAdapter(null);
    this.adapters.delete(name);
    const p = this.prepared.get(name);
    if (p) this.model.freeLoraBuffers(p);
    this.prepared.delete(name);
  }

  /** Destroys every recorded plan, for a model whose graph changed (LoRA slots added or removed). */
  private dropPlans(): void {
    for (const p of this.plans.values()) p.destroy();
    for (const p of this.verifyPlans.values()) p.destroy();
    for (const a of this.advances.values()) a.destroy();
    this.plans.clear();
    this.verifyPlans.clear();
    this.advances.clear();
    this.chains.clear();
  }

  /** Forgets the cached prefix and the saved ones; the next call prefills its whole prompt. */
  resetCache(): void {
    this.cached = [];
    this.prefixes.clear();
  }

  /** Saves the live cache's K/V (GPU to GPU) unless it is short or already saved. */
  private saveLive(): void {
    const n = this.cached.length;
    if (n < MIN_SAVED_TOKENS || this.prefixes.budget <= 0) return;
    this.prefixes.add(this.cached, this.written, this.model.kvSnapshotBytes(n), () => this.model.saveKv(n));
  }

  /** Start timing every kernel the engine runs (GPU timestamps on WebGPU, wall clock on the CPU). */
  startProfile(): void {
    if (!this.backend.startProfile) throw new Error(`[kiln] the ${this.backend.name} backend cannot profile`);
    this.backend.startProfile();
  }

  /** Stop timing; time per kernel and shape since `startProfile`, most expensive first. */
  async stopProfile(): Promise<Profile> {
    if (!this.backend.stopProfile) throw new Error(`[kiln] the ${this.backend.name} backend cannot profile`);
    return this.backend.stopProfile();
  }

  destroy(): void {
    for (const p of this.prepared.values()) this.model.freeLoraBuffers(p);
    this.prepared.clear();
    this.prefixes.clear();
    for (const p of this.plans.values()) p.destroy();
    for (const p of this.verifyPlans.values()) p.destroy();
    for (const a of this.advances.values()) a.destroy();
    this.plans.clear();
    this.verifyPlans.clear();
    this.advances.clear();
    this.chains.clear();
    this.model.destroy();
  }

  /** Feeds the id sampled into `sampled` to the decode plan and advances the step. */
  private advance(sampled: Buf, tokens: Buf): readonly Dispatch[] {
    let a = this.advances.get(sampled);
    if (!a) {
      a = this.model.buildAdvance(sampled, tokens);
      this.advances.set(sampled, a);
    }
    return a.dispatches;
  }

  /** The end of a constrained decode step: sampling (with the mask written just before), then advance. */
  private grammarTail(): readonly Dispatch[] {
    let c = this.chains.get(-1);
    if (!c) {
      const decode = this.plan(1);
      c = [decode.dispatches[decode.dispatches.length - 1]!, ...this.advance(decode.out, decode.tokens)];
      this.chains.set(-1, c);
    }
    return c;
  }

  /** k chained decode steps: forward, then advance, k times. */
  private chain(k: number): readonly Dispatch[] {
    let c = this.chains.get(k);
    if (!c) {
      const decode = this.plan(1);
      const step = [...decode.dispatches, ...this.advance(decode.out, decode.tokens)];
      c = Array.from({ length: k }, () => step).flat();
      this.chains.set(k, c);
    }
    return c;
  }

  private plan(T: number): ForwardPlan {
    let p = this.plans.get(T);
    if (!p) {
      p = this.model.buildPlan(T);
      this.plans.set(T, p);
    }
    return p;
  }
}
