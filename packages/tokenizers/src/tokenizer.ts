import { TOOL_CALL_MARKERS, type ChatTemplate } from "./chat";
import { GPT2_PATTERN, LLAMA3_PATTERN, QWEN2_PATTERN, compilePreTokenizer, splitIsolated, toJsRegex, type PreStep } from "./pretokenize";

/** GPT-2 byte ↔ printable-unicode table used by byte-level BPE. */
function byteTables(): { enc: string[]; dec: Map<string, number> } {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n++);
    }
  }
  const enc = new Array<string>(256);
  const dec = new Map<string, number>();
  bs.forEach((b, i) => {
    enc[b] = String.fromCharCode(cs[i]!);
    dec.set(enc[b]!, b);
  });
  return { enc, dec };
}
const BYTES = byteTables();

/** The printable stand-in byte-level BPE uses for a raw byte. */
export const byteToUnicode = (b: number): string => BYTES.enc[b]!;

export interface AddedToken {
  id: number;
  content: string;
  special: boolean;
}

export interface TokenizerJSON {
  model: {
    type: string;
    vocab: Record<string, number>;
    merges: (string | [string, string])[];
    ignore_merges?: boolean;
    byte_fallback?: boolean;
  };
  added_tokens?: AddedToken[];
  normalizer?: { type: string; normalizers?: { type: string; prepend?: string; content?: string }[] } | null;
  pre_tokenizer?: Parameters<typeof compilePreTokenizer>[0];
}

/**
 * How text maps to the symbols BPE merges.
 * - byte-level (GPT-2, Llama 3, Qwen): UTF-8 bytes as printable stand-ins, split by a pre-tokenizer regex.
 * - metaspace (SentencePiece BPE: Llama 2, Mistral, Phi-3, Gemma): spaces become "▁", a "▁" is
 *   prepended, the whole text is one word of Unicode characters, and characters outside the vocabulary
 *   fall back to `<0xXX>` byte tokens.
 */
type Scheme = { kind: "byte-level" } | { kind: "metaspace"; prepend: "always" | "first" | "never"; split: boolean };

/** U+2581, SentencePiece's visible space. */
const SPACE = "\u2581";
const TOOL_MARKERS = new Set(TOOL_CALL_MARKERS);
const BYTE_TOKEN = /^<0x([0-9A-Fa-f]{2})>$/;

export interface EncodeOptions {
  /** Parse special-token strings such as `<|im_start|>` as single tokens. Default true. */
  allowSpecial?: boolean;
}

export class Tokenizer {
  private readonly ranks = new Map<string, number>();
  private readonly idToToken: string[] = [];
  private readonly cache = new Map<string, number[]>();
  private readonly addedRe: RegExp | undefined;
  private readonly added = new Map<string, AddedToken>();
  readonly specialIds = new Set<number>();
  private readonly scheme: Scheme;
  /** The model's chat template, when the loader found one. */
  chatTemplate: ChatTemplate | undefined;

  constructor(
    private readonly vocab: Map<string, number>,
    merges: readonly string[],
    addedTokens: readonly AddedToken[],
    private readonly pre: PreStep[],
    private readonly opts: {
      ignoreMerges?: boolean;
      nfc?: boolean;
      scheme?: Scheme;
      /** SentencePiece scores by id, for GGUF "llama" vocabularies, which merge by score instead of a merge list. */
      scores?: readonly number[];
    } = {},
  ) {
    merges.forEach((m, i) => this.ranks.set(m, i));
    this.scheme = opts.scheme ?? { kind: "byte-level" };
    for (const [tok, id] of vocab) this.idToToken[id] = tok;
    for (const t of addedTokens) {
      this.added.set(t.content, t);
      this.idToToken[t.id] = t.content;
      if (t.special) this.specialIds.add(t.id);
    }
    if (addedTokens.length) {
      const alts = [...addedTokens]
        .map((t) => t.content)
        .sort((a, b) => b.length - a.length)
        .map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      this.addedRe = new RegExp(alts.join("|"), "gu");
    }
  }

  static fromJSON(json: TokenizerJSON): Tokenizer {
    if (json.model.type !== "BPE") throw new Error(`tokenizer model "${json.model.type}" is not supported yet (BPE only; Unigram models such as T5 are not)`);
    const merges = json.model.merges.map((m) => (Array.isArray(m) ? m.join(" ") : m));
    const vocab = new Map(Object.entries(json.model.vocab));
    const added = json.added_tokens ?? [];
    if (JSON.stringify(json.pre_tokenizer ?? {}).includes("ByteLevel")) {
      return new Tokenizer(vocab, merges, added, compilePreTokenizer(json.pre_tokenizer), {
        ignoreMerges: json.model.ignore_merges,
        nfc: json.normalizer?.type === "NFC",
      });
    }
    const scheme = metaspaceScheme(json);
    if (!scheme) throw new Error("unsupported tokenizer: only byte-level BPE (GPT-2, Llama 3, Qwen) and SentencePiece-style BPE (Llama 2, Mistral, Phi-3) are supported");
    return new Tokenizer(vocab, merges, added, [], { ignoreMerges: json.model.ignore_merges, scheme });
  }

  /** Builds a tokenizer from GGUF `tokenizer.ggml.*` metadata: "gpt2" (byte-level BPE) or "llama" (SentencePiece). */
  static fromGGUF(meta: Map<string, unknown>): Tokenizer {
    const model = meta.get("tokenizer.ggml.model");
    if (model !== "gpt2" && model !== "llama") throw new Error(`GGUF tokenizer model "${String(model)}" is not supported yet (gpt2 and llama only)`);
    const tokens = meta.get("tokenizer.ggml.tokens") as string[];
    if (model === "llama") return withTemplate(Tokenizer.fromSentencePieceGGUF(meta, tokens), meta, tokens);
    const types = (meta.get("tokenizer.ggml.token_type") as number[] | undefined) ?? [];
    const merges = (meta.get("tokenizer.ggml.merges") as string[] | undefined) ?? [];
    const pre = String(meta.get("tokenizer.ggml.pre") ?? "default");
    const vocab = new Map<string, number>();
    const added: AddedToken[] = [];
    tokens.forEach((t, id) => {
      // 3 = control, 4 = user-defined: matched before pre-tokenization.
      if (types[id] === 3 || types[id] === 4) added.push({ id, content: t, special: types[id] === 3 });
      else vocab.set(t, id);
    });
    // llama.cpp gives dbrx (Granite 4) Llama 3's split, and refact (Granite 3) SmolLM's isolated digits.
    const pattern = pre === "llama-bpe" || pre === "llama3" || pre === "dbrx" ? LLAMA3_PATTERN : pre.startsWith("qwen") ? QWEN2_PATTERN : GPT2_PATTERN;
    const re = toJsRegex(pattern);
    const steps: PreStep[] = [(ps) => ps.flatMap((p) => splitIsolated(p, re))];
    if (pre === "smollm" || pre === "refact") steps.unshift((ps) => ps.flatMap((p) => splitIsolated(p, /\p{N}/gu)));
    return withTemplate(new Tokenizer(vocab, merges, added, steps, { ignoreMerges: pre === "llama-bpe" || pre === "llama3" }), meta, tokens);
  }

  /** GGUF "llama" vocabularies: SentencePiece pieces with scores; merges pick the highest-scoring piece. */
  private static fromSentencePieceGGUF(meta: Map<string, unknown>, tokens: string[]): Tokenizer {
    const types = (meta.get("tokenizer.ggml.token_type") as number[] | undefined) ?? [];
    const scores = (meta.get("tokenizer.ggml.scores") as number[] | undefined) ?? [];
    const vocab = new Map<string, number>();
    const added: AddedToken[] = [];
    tokens.forEach((t, id) => {
      // 3 = control (<s>, </s>), 4 = user-defined; 6 = byte tokens, kept for the byte fallback.
      if (types[id] === 3 || types[id] === 4) added.push({ id, content: t, special: types[id] === 3 });
      else vocab.set(t, id);
    });
    const prefix = meta.get("tokenizer.ggml.add_space_prefix") !== false;
    return new Tokenizer(vocab, [], added, [], { scheme: { kind: "metaspace", prepend: prefix ? "first" : "never", split: false }, scores });
  }

  get vocabSize(): number {
    return this.idToToken.length;
  }

  tokenId(token: string): number | undefined {
    return this.added.get(token)?.id ?? this.vocab.get(token);
  }

  /**
   * The bytes a token decodes to, or undefined for added tokens (`<|im_end|>` and the like), which
   * text never produces.
   */
  tokenBytes(id: number): Uint8Array | undefined {
    const tok = this.idToToken[id];
    if (tok === undefined || this.added.has(tok)) return undefined;
    if (this.scheme.kind === "byte-level") return Uint8Array.from(Array.from(tok, (c) => BYTES.dec.get(c) ?? 63));
    const byte = BYTE_TOKEN.exec(tok);
    if (byte) return Uint8Array.of(parseInt(byte[1]!, 16));
    return UTF8.encode(tok.replaceAll(SPACE, " "));
  }

  encode(text: string, opts: EncodeOptions = {}): number[] {
    if (this.opts.nfc) text = text.normalize("NFC");
    const ids: number[] = [];
    const segments = opts.allowSpecial !== false && this.addedRe ? splitIsolated(text, this.addedRe) : [text];
    for (const seg of segments) {
      const added = opts.allowSpecial !== false ? this.added.get(seg) : undefined;
      if (added) {
        ids.push(added.id);
        continue;
      }
      if (this.scheme.kind === "metaspace") {
        if (!seg) continue;
        const { prepend, split } = this.scheme;
        let s = seg.replaceAll(" ", SPACE);
        if (prepend === "always" || (prepend === "first" && ids.length === 0 && text.startsWith(seg))) s = SPACE + s;
        for (const p of split ? s.split(new RegExp(`(?=${SPACE})`)) : [s]) ids.push(...this.bpe(p));
        continue;
      }
      let pieces = [seg];
      for (const step of this.pre) pieces = step(pieces);
      for (const p of pieces) {
        const mapped = Array.from(UTF8.encode(p), (b) => BYTES.enc[b]).join("");
        ids.push(...this.bpe(mapped));
      }
    }
    return ids;
  }

  decode(ids: readonly number[], skipSpecial = true): string {
    const d = this.decoder(skipSpecial);
    return ids.map((id) => d.push(id)).join("") + d.flush();
  }

  /** Incremental decoder that holds back incomplete UTF-8 sequences between tokens. */
  decoder(skipSpecial = true): { push(id: number): string; flush(): string } {
    const utf8 = new TextDecoder("utf-8");
    // SentencePiece decoders drop the space the encoder prepended: the first one of the output.
    let strip = this.scheme.kind === "metaspace" && this.scheme.prepend !== "never";
    const out = (text: string) => {
      if (strip && text) {
        strip = false;
        if (text.startsWith(" ")) return text.slice(1);
      }
      return text;
    };
    return {
      push: (id) => {
        const tok = this.idToToken[id];
        if (tok === undefined) return "";
        if (this.added.has(tok)) {
          // Tool-call markers stay: they are how `parseToolCalls` finds Mistral's and Granite's calls.
          if (skipSpecial && this.specialIds.has(id) && !TOOL_MARKERS.has(tok)) return "";
          // Added tokens go through the decoder like any other in HF: Gemma's whitespace tokens ("▁▁")
          // come out as spaces. Special ones are kept as written.
          const text = this.scheme.kind === "metaspace" && !this.specialIds.has(id) ? tok.replaceAll(SPACE, " ") : tok;
          return out(utf8.decode() + text);
        }
        return out(utf8.decode(this.tokenBytes(id), { stream: true }));
      },
      flush: () => out(utf8.decode()),
    };
  }

  private bpe(word: string): number[] {
    const hit = this.cache.get(word);
    if (hit) return hit;
    const direct = this.vocab.get(word);
    if (this.opts.ignoreMerges && direct !== undefined) return [direct];
    const scores = this.opts.scores;
    const rank = scores
      ? (a: string, b: string) => {
          const id = this.vocab.get(a + b);
          return id === undefined ? undefined : -(scores[id] ?? 0);
        }
      : (a: string, b: string) => this.ranks.get(`${a} ${b}`);
    const ids: number[] = [];
    for (const p of mergeSymbols(Array.from(word), rank)) {
      const id = this.vocab.get(p);
      if (id !== undefined) ids.push(id);
      else if (this.scheme.kind === "metaspace") {
        for (const b of UTF8.encode(p)) {
          const bid = this.vocab.get(`<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`);
          if (bid === undefined) throw new Error(`tokenizer: no byte token for 0x${b.toString(16)}`);
          ids.push(bid);
        }
      } else throw new Error(`tokenizer: "${p}" is not in the vocabulary`);
    }
    // Byte-level words are short and repeat; SentencePiece words are whole texts.
    if (word.length <= 64 && this.cache.size < 50_000) this.cache.set(word, ids);
    return ids;
  }
}

const UTF8 = new TextEncoder();

function withTemplate(tok: Tokenizer, meta: Map<string, unknown>, tokens: string[]): Tokenizer {
  const source = meta.get("tokenizer.chat_template");
  if (typeof source === "string") {
    const name = (key: string) => (meta.has(key) ? tokens[Number(meta.get(key))] : undefined);
    tok.chatTemplate = { source, bosToken: name("tokenizer.ggml.bos_token_id"), eosToken: name("tokenizer.ggml.eos_token_id") };
  }
  return tok;
}

/** The SentencePiece-style scheme of a tokenizer.json, or undefined if it is not one. */
function metaspaceScheme(json: TokenizerJSON): Scheme | undefined {
  if (!json.model.byte_fallback) return undefined;
  const norms = json.normalizer?.type === "Sequence" ? json.normalizer.normalizers ?? [] : json.normalizer ? [json.normalizer as { type: string; prepend?: string; content?: string }] : [];
  const replaces = norms.some((n) => n.type === "Replace" && n.content === SPACE);
  const prepends = norms.some((n) => n.type === "Prepend" && n.prepend === SPACE);
  if (replaces) return { kind: "metaspace", prepend: prepends ? "always" : "never", split: false };
  const pre = json.pre_tokenizer as { type: string; pretokenizers?: unknown[] } | null | undefined;
  const steps = (pre?.type === "Sequence" ? pre.pretokenizers ?? [] : pre ? [pre] : []) as { type: string; prepend_scheme?: string; add_prefix_space?: boolean; split?: boolean; replacement?: string }[];
  const ms = steps.find((p) => p.type === "Metaspace");
  if (!ms || (ms.replacement && ms.replacement !== SPACE)) return undefined;
  const prepend = ms.prepend_scheme === "first" || ms.prepend_scheme === "never" ? ms.prepend_scheme : ms.add_prefix_space === false ? "never" : "always";
  return { kind: "metaspace", prepend, split: ms.split ?? true };
}

/**
 * Byte-pair merging: repeatedly joins the adjacent pair with the lowest rank, leftmost first, until no
 * pair has one. A heap over a linked list keeps this O(n log n), so a whole SentencePiece text (one
 * word) costs little more than many short words.
 */
function mergeSymbols(symbols: string[], rank: (a: string, b: string) => number | undefined): string[] {
  const n = symbols.length;
  if (n < 2) return symbols;
  const sym = [...symbols];
  const prev = Int32Array.from({ length: n }, (_, i) => i - 1);
  const next = Int32Array.from({ length: n }, (_, i) => (i + 1 < n ? i + 1 : -1));
  const alive = new Uint8Array(n).fill(1);
  // Heap of [rank, left index, left symbol, right symbol]; stale entries are skipped when popped.
  type Cand = [number, number, string, string];
  const heap: Cand[] = [];
  const less = (a: Cand, b: Cand) => a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
  const pushCand = (c: Cand) => {
    heap.push(c);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const popCand = (): Cand => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && less(heap[l]!, heap[m]!)) m = l;
        if (r < heap.length && less(heap[r]!, heap[m]!)) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m]!, heap[i]!];
        i = m;
      }
    }
    return top;
  };
  const consider = (i: number) => {
    const j = next[i]!;
    if (i < 0 || j < 0) return;
    const r = rank(sym[i]!, sym[j]!);
    if (r !== undefined) pushCand([r, i, sym[i]!, sym[j]!]);
  };
  for (let i = 0; i + 1 < n; i++) consider(i);
  while (heap.length) {
    const [, i, a, b] = popCand();
    const j = next[i]!;
    // Skip pairs a merge has since changed.
    if (!alive[i] || j < 0 || sym[i] !== a || sym[j] !== b) continue;
    sym[i] = a + b;
    alive[j] = 0;
    next[i] = next[j]!;
    if (next[j]! >= 0) prev[next[j]!] = i;
    consider(prev[i]!);
    consider(i);
  }
  const out: string[] = [];
  for (let i = 0; i >= 0; i = next[i]!) out.push(sym[i]!);
  return out;
}
