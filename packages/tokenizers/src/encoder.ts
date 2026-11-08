// Tokenizers for encoder models (BERT, DistilBERT, RoBERTa): text, or a pair of texts, to ids with
// the model's special tokens and segment ids, as tokenizers' post-processors build them.
import { sentencePieceFromJSON } from "./spm";
import { TemplatedTokenizer, type EncoderTokenizer, type EncoderTokenizerJSON } from "./template";
import { Tokenizer, type TokenizerJSON } from "./tokenizer";

export type { EncoderTokenizer, EncoderTokenizerJSON, Encoding } from "./template";
export { TemplatedTokenizer } from "./template";

const isControl = (c: string) => /\p{Cc}|\p{Cf}/u.test(c) && c !== "\t" && c !== "\n" && c !== "\r";
const isWhitespace = (c: string) => /\s/u.test(c);
const isPunct = (c: string) => {
  const cp = c.codePointAt(0)!;
  return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126) || /\p{P}/u.test(c);
};
const isCJK = (cp: number) =>
  (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b73f) ||
  (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) || (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);

/**
 * BERT's WordPiece: BertNormalizer (clean text, spaces around CJK ideographs, lowercase and strip
 * accents when asked), BertPreTokenizer (split on whitespace and punctuation), then greedy longest
 * match with "##" continuations; a word that cannot be covered becomes [UNK]. Added tokens such as
 * [MASK] are matched in the raw text first.
 */
export class WordPieceTokenizer extends TemplatedTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly idTo: string[] = [];
  private readonly addedRe: RegExp | undefined;
  private readonly added = new Map<string, number>();
  readonly specialIds = new Set<number>();
  readonly maskToken: string | undefined;
  private readonly unk: number;
  private readonly prefix: string;
  private readonly maxChars: number;
  private readonly lower: boolean;
  private readonly strip: boolean;
  private readonly clean: boolean;
  private readonly cjk: boolean;

  constructor(json: EncoderTokenizerJSON) {
    super(json.post_processor);
    const v = json.model.vocab;
    this.vocab = new Map(Array.isArray(v) ? v : Object.entries(v));
    for (const [t, id] of this.vocab) this.idTo[id] = t;
    for (const t of json.added_tokens ?? []) {
      this.added.set(t.content, t.id);
      this.idTo[t.id] = t.content;
      if (t.special) this.specialIds.add(t.id);
    }
    if (this.added.size) {
      const alts = [...this.added.keys()].sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      this.addedRe = new RegExp(alts.join("|"), "g");
    }
    this.maskToken = [...this.added.keys()].find((t) => /mask/i.test(t));
    this.unk = this.vocab.get(json.model.unk_token ?? "[UNK]") ?? 0;
    this.prefix = json.model.continuing_subword_prefix ?? "##";
    this.maxChars = json.model.max_input_chars_per_word ?? 100;
    const n = json.normalizer;
    this.lower = n?.lowercase ?? false;
    this.strip = n?.strip_accents ?? this.lower;
    this.clean = n?.clean_text ?? true;
    this.cjk = n?.handle_chinese_chars ?? true;
  }

  get vocabSize(): number {
    return this.idTo.length;
  }

  token(id: number): string {
    return this.idTo[id] ?? "";
  }

  tokenId(token: string): number | undefined {
    return this.added.get(token) ?? this.vocab.get(token);
  }

  protected tokenize(text: string): number[] {
    const out: number[] = [];
    let last = 0;
    if (this.addedRe) {
      for (const m of text.matchAll(this.addedRe)) {
        this.words(text.slice(last, m.index), out);
        out.push(this.added.get(m[0])!);
        last = m.index! + m[0].length;
      }
    }
    this.words(text.slice(last), out);
    return out;
  }

  private normalize(text: string): string {
    let s = "";
    for (const c of text) {
      const cp = c.codePointAt(0)!;
      if (this.clean && (cp === 0 || cp === 0xfffd || isControl(c))) continue;
      if (this.clean && isWhitespace(c)) s += " ";
      else if (this.cjk && isCJK(cp)) s += ` ${c} `;
      else s += c;
    }
    if (this.strip) s = s.normalize("NFD").replace(/\p{Mn}/gu, "");
    if (this.lower) s = s.toLowerCase();
    return s;
  }

  private words(text: string, out: number[]): void {
    if (!text) return;
    for (const w of this.normalize(text).split(/\s+/u)) {
      if (!w) continue;
      // Each punctuation character is a word of its own.
      let cur = "";
      for (const c of w) {
        if (isPunct(c)) {
          if (cur) this.piece(cur, out);
          this.piece(c, out);
          cur = "";
        } else cur += c;
      }
      if (cur) this.piece(cur, out);
    }
  }

  private piece(word: string, out: number[]): void {
    const chars = [...word];
    if (chars.length > this.maxChars) {
      out.push(this.unk);
      return;
    }
    const ids: number[] = [];
    for (let start = 0; start < chars.length; ) {
      let end = chars.length;
      let id: number | undefined;
      for (; end > start; end--) {
        const sub = (start > 0 ? this.prefix : "") + chars.slice(start, end).join("");
        id = this.vocab.get(sub);
        if (id !== undefined) break;
      }
      if (id === undefined) {
        out.push(this.unk);
        return;
      }
      ids.push(id);
      start = end;
    }
    out.push(...ids);
  }

  decode(ids: readonly number[], skipSpecial = true): string {
    const toks = ids.filter((id) => !(skipSpecial && this.specialIds.has(id))).map((id) => this.token(id));
    return cleanup(toks.join(" ").replaceAll(` ${this.prefix}`, "").replaceAll(this.prefix, ""));
  }
}

/** transformers' clean_up_tokenization: no space before punctuation and English contractions. */
export function cleanup(s: string): string {
  return s
    .replaceAll(" .", ".")
    .replaceAll(" ?", "?")
    .replaceAll(" !", "!")
    .replaceAll(" ,", ",")
    .replaceAll(" ' ", "'")
    .replaceAll(" n't", "n't")
    .replaceAll(" 'm", "'m")
    .replaceAll(" 's", "'s")
    .replaceAll(" 've", "'ve")
    .replaceAll(" 're", "'re");
}

/** RoBERTa's byte-level BPE with `<s> A </s></s> B </s>`. Added tokens marked lstrip (`<mask>`) take the space before them. */
export class RobertaTokenizer extends TemplatedTokenizer {
  readonly maskToken: string | undefined;
  private readonly lstrip: RegExp | undefined;

  constructor(
    json: EncoderTokenizerJSON,
    private readonly bpe: Tokenizer,
  ) {
    super(json.post_processor);
    const added = json.added_tokens ?? [];
    this.maskToken = added.find((t) => /mask/i.test(t.content))?.content;
    const l = added.filter((t) => t.lstrip).map((t) => t.content.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.lstrip = l.length ? new RegExp(`\\s+(${l.join("|")})`, "g") : undefined;
  }

  get vocabSize(): number {
    return this.bpe.vocabSize;
  }

  get specialIds(): ReadonlySet<number> {
    return this.bpe.specialIds;
  }

  token(id: number): string {
    return this.bpe.tokenString(id);
  }

  tokenId(token: string): number | undefined {
    return this.bpe.tokenId(token);
  }

  protected tokenize(text: string): number[] {
    return this.bpe.encode(this.lstrip ? text.replace(this.lstrip, "$1") : text);
  }

  /** Decoded with transformers' clean_up_tokenization_spaces (BART and RoBERTa repos turn it on). */
  decode(ids: readonly number[], skipSpecial = true): string {
    return cleanup(this.bpe.decode(ids, skipSpecial));
  }
}

/**
 * The tokenizer of an encoder repo, from its tokenizer.json: WordPiece (BERT, DistilBERT, MiniLM,
 * BGE) or byte-level BPE (RoBERTa). Unigram (XLM-R, T5) is not supported yet.
 */
export function encoderTokenizerFromJSON(json: EncoderTokenizerJSON): EncoderTokenizer {
  // Older files leave the model type out: merges mean BPE, [piece, score] pairs Unigram.
  const type = json.model.type ?? (json.model.merges ? "BPE" : Array.isArray(json.model.vocab) ? "Unigram" : "WordPiece");
  if (type === "WordPiece") return new WordPieceTokenizer(json);
  // SentencePiece-style (Unigram, or BPE under a Metaspace pre-tokenizer: T5, NLLB, XLM-R).
  if (type === "Unigram" || (type === "BPE" && JSON.stringify(json.pre_tokenizer ?? {}).includes("Metaspace"))) return sentencePieceFromJSON(json);
  if (type === "BPE") return new RobertaTokenizer(json, Tokenizer.fromJSON({ ...json, model: { ...json.model, type } } as unknown as TokenizerJSON));
  throw new Error(`encoder tokenizer model "${type}" is not supported`);
}

/**
 * BERT's tokenizer from vocab.txt (one token per line, for repos without a tokenizer.json), as
 * transformers' BertTokenizer builds it: `[CLS] A [SEP] B [SEP]`, lowercased when `lowercase`.
 */
export function wordPieceFromVocab(vocabTxt: string, opts: { lowercase?: boolean; stripAccents?: boolean | null } = {}): EncoderTokenizer {
  const lines = vocabTxt.split(/\r?\n/);
  if (lines.at(-1) === "") lines.pop();
  const vocab: Record<string, number> = {};
  lines.forEach((t, i) => (vocab[t] = i));
  const specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].filter((t) => vocab[t] !== undefined);
  const id = (t: string) => {
    if (vocab[t] === undefined) throw new Error(`vocab.txt has no ${t}`);
    return vocab[t]!;
  };
  return new WordPieceTokenizer({
    model: { type: "WordPiece", vocab, unk_token: "[UNK]", continuing_subword_prefix: "##", max_input_chars_per_word: 100 },
    normalizer: { type: "BertNormalizer", lowercase: opts.lowercase ?? true, strip_accents: opts.stripAccents ?? null, clean_text: true, handle_chinese_chars: true },
    post_processor: { type: "BertProcessing", cls: ["[CLS]", id("[CLS]")], sep: ["[SEP]", id("[SEP]")] },
    added_tokens: specials.map((t) => ({ id: vocab[t]!, content: t, special: true })),
  });
}

/**
 * RoBERTa / BART's byte-level BPE from vocab.json and merges.txt (repos without a tokenizer.json),
 * with `<s> A </s>` and `<s> A </s></s> B </s>` as transformers' RobertaTokenizer builds them.
 */
export function byteLevelFromVocabMerges(vocab: Record<string, number>, mergesTxt: string): EncoderTokenizer {
  const merges = mergesTxt.split(/\r?\n/).filter((l) => l && !l.startsWith("#version"));
  const specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"].filter((t) => vocab[t] !== undefined);
  const added = specials.map((t) => ({ id: vocab[t]!, content: t, special: true, lstrip: t === "<mask>" }));
  const json: EncoderTokenizerJSON = {
    model: { type: "BPE", vocab, merges },
    pre_tokenizer: { type: "ByteLevel", add_prefix_space: false, trim_offsets: true } as never,
    post_processor: { type: "RobertaProcessing", cls: ["<s>", vocab["<s>"]!], sep: ["</s>", vocab["</s>"]!] },
    added_tokens: added,
  };
  return encoderTokenizerFromJSON(json);
}
