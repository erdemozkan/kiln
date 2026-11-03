import type { Backend, Buf, Dispatch } from "@kiln/core";
import { GrammarMatcher, TokenTrie, compileFormat, type ResponseFormat } from "@kiln/grammar";
import { LlamaModel, type ForwardPlan } from "@kiln/models";
import { detectChatFormat, type ChatFormat, type ChatMessage, type Tokenizer } from "@kiln/tokenizers";

export interface GenerateOptions {
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
   * Constrain the output: `{ type: "json" }` for any JSON value, `{ type: "json", schema }` for a JSON
   * Schema (properties in declared order), `{ type: "choice", choices }` for one of a set of strings.
   * Each step then waits for the CPU to compute the next token mask, so steps are not chained.
   */
  format?: ResponseFormat;
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
}

export interface TextDelta {
  text: string;
  tokenId: number;
}

/** Prefill buckets. Prompts longer than the largest bucket are prefilled in chunks. */
const BUCKETS = [16, 64, 128];

export class Engine {
  readonly chatFormat: ChatFormat | undefined;
  lastStats: GenerateStats | undefined;
  private readonly plans = new Map<number, ForwardPlan>();
  private readonly advances = new Map<Buf, { dispatches: readonly Dispatch[]; destroy(): void }>();
  private readonly chains = new Map<number, Dispatch[]>();
  private readonly stopIds: Set<number>;
  private busy = false;
  /** Tokens whose K/V currently sit in the cache at positions 0…n-1 (empty while generating). */
  private cached: number[] = [];
  /** Token bytes in a prefix tree, built on the first constrained call. */
  private trie: { trie: TokenTrie; tokens: (Uint8Array | undefined)[] } | undefined;

  constructor(
    readonly backend: Backend,
    readonly model: LlamaModel,
    readonly tokenizer: Tokenizer,
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
    const prompt = this.chatFormat.render(messages, { enableThinking: thinks ? opts.enableThinking : undefined });
    yield* this.complete(prompt, opts);
  }

  async *complete(prompt: string, opts: GenerateOptions = {}): AsyncGenerator<TextDelta> {
    const ids = this.tokenizer.encode(prompt);
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
    const maxSeq = this.model.maxSeq;
    if (promptIds.length >= maxSeq) throw new Error(`prompt has ${promptIds.length} tokens but the context is ${maxSeq}`);
    const maxTokens = Math.min(opts.maxTokens ?? 512, maxSeq - promptIds.length);
    const temperature = opts.temperature ?? 0.7;
    const seed = opts.seed ?? (Math.random() * 2 ** 32) >>> 0;
    const sampling = { topK: opts.topK ?? 0, topP: opts.topP ?? 1, repetitionPenalty: opts.repetitionPenalty ?? 1 };
    if (sampling.topK < 0 || !(sampling.topP > 0 && sampling.topP <= 1) || !(sampling.repetitionPenalty > 0)) {
      throw new Error(`invalid sampling options ${JSON.stringify(sampling)}`);
    }
    const stop = new Set([...this.stopIds, ...(opts.stopTokenIds ?? [])]);
    const t0 = performance.now();
    const grammar = opts.format ? this.matcher(opts.format, [...stop]) : undefined;
    this.busy = true;
    // Reuse the shared prefix, but always prefill at least the last prompt token to get its logits.
    let start = 0;
    if (opts.reuseCache ?? true) {
      const prev = this.cached;
      while (start < prev.length && start < promptIds.length - 1 && prev[start] === promptIds[start]) start++;
    }
    this.cached = [];
    const stats: GenerateStats = { promptTokens: promptIds.length, cachedTokens: start, generatedTokens: 0, prefillMs: 0, decodeMs: 0, ttftMs: 0 };
    // Ids read back so far (including any past a stop token) and decode steps submitted.
    const readIds: number[] = [];
    let queued = 0;
    let failed = false;

    try {
      this.model.setSeen(promptIds);
      if (grammar) {
        const t = performance.now();
        this.model.setAllowed(grammar.mask());
        stats.maskMs = performance.now() - t;
      } else this.model.setAllowed(undefined);
      let pos = start;
      let out = this.plan(1).out;
      while (pos < promptIds.length) {
        const remaining = promptIds.length - pos;
        const T = BUCKETS.find((b) => b >= remaining) ?? BUCKETS[BUCKETS.length - 1]!;
        const n = Math.min(T, remaining);
        const plan = this.plan(T);
        const tokens = new Uint32Array(T);
        tokens.set(promptIds.slice(pos, pos + n));
        this.backend.write(plan.tokens, 0, tokens);
        this.model.setStep({ pos0: pos, lastRow: n - 1, seed, temperature, ...sampling });
        this.backend.submit(plan.dispatches);
        pos += n;
        out = plan.out;
      }

      const decode = this.plan(1);
      // A grammar needs each sampled id on the CPU before the next mask exists.
      const K = grammar ? 1 : Math.max(1, Math.floor(opts.stepsPerSubmit ?? 4));
      const reads: Promise<ArrayBuffer>[] = [];
      const submitBatch = () => {
        // The first batch only records the id sampled by prefill; later ones run k decode steps.
        const k = queued === 0 ? 1 : Math.min(K, maxTokens - queued);
        this.backend.submit(queued === 0 ? this.advance(out, decode.tokens) : this.chain(k));
        reads.push(this.backend.readback(this.model.history, queued * 4, k * 4));
        queued += k;
      };
      submitBatch();
      if (!grammar && queued < maxTokens) submitBatch();
      let tDecode = 0;
      let done = false;
      while (!done && reads.length > 0) {
        const ids = new Uint32Array(await reads.shift()!);
        readIds.push(...ids);
        if (!tDecode) {
          stats.ttftMs = stats.prefillMs = performance.now() - t0;
          tDecode = performance.now();
        }
        if (grammar) {
          const id = ids[0]!;
          if (!stop.has(id)) {
            if (!grammar.accept(id)) throw new Error(`grammar: sampled token ${id} is not allowed (mask out of sync)`);
            // Once the value is complete and nothing can extend it, end without another step.
            if (grammar.finished) done = true;
            else if (queued < maxTokens) {
              // The forward pass only needs the id just sampled, so the GPU runs it while the CPU
              // computes the mask; the mask write is queued before the sampling dispatch that reads it.
              this.backend.submit(decode.dispatches.slice(0, -1));
              const t = performance.now();
              this.model.setAllowed(grammar.mask());
              stats.maskMs! += performance.now() - t;
              this.backend.submit(this.grammarTail());
              reads.push(this.backend.readback(this.model.history, queued * 4, 4));
              queued++;
            }
          }
        } else if (queued < maxTokens) submitBatch();
        for (const id of ids) {
          if (stop.has(id) || opts.signal?.aborted) {
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
      if (!failed) this.cached = [...promptIds, ...readIds.slice(0, Math.max(0, queued - 1))];
      this.lastStats = stats;
      this.busy = false;
    }
  }

  private matcher(format: ResponseFormat, stopIds: number[]): GrammarMatcher {
    const grammar = compileFormat(format);
    if (!this.trie) {
      const tokens = Array.from({ length: this.tokenizer.vocabSize }, (_, id) => this.tokenizer.tokenBytes(id));
      this.trie = { trie: new TokenTrie(tokens), tokens };
    }
    return new GrammarMatcher(grammar, this.trie.trie, this.trie.tokens, this.model.cfg.vocabSize, stopIds);
  }

  /** Forgets the cached prefix; the next call prefills its whole prompt. */
  resetCache(): void {
    this.cached = [];
  }

  destroy(): void {
    for (const p of this.plans.values()) p.destroy();
    for (const a of this.advances.values()) a.destroy();
    this.plans.clear();
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
