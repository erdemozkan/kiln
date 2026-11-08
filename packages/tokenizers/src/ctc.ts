import { cleanup } from "./encoder";

/**
 * A CTC tokenizer (wav2vec2, HuBERT): one token per audio frame, decoded as transformers.js's CTC
 * decoder does: repeats merged, the blank (pad) token dropped, clean_up_tokenization, the word
 * delimiter turned into a space, trimmed.
 */
export class CtcTokenizer {
  private readonly tokens: string[] = [];
  readonly blank: number;

  constructor(
    vocab: Record<string, number>,
    readonly pad = "<pad>",
    readonly delimiter = "|",
    readonly clean = true,
  ) {
    for (const [t, id] of Object.entries(vocab)) this.tokens[id] = t;
    const b = vocab[pad];
    if (b === undefined) throw new Error(`CTC vocabulary has no pad token "${pad}"`);
    this.blank = b;
  }

  get vocabSize(): number {
    return this.tokens.length;
  }

  /** The token string of each frame's id, merged and decoded. */
  decode(ids: ArrayLike<number>): string {
    const merged: string[] = [];
    let prev = -1;
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i]!;
      if (id !== prev) merged.push(this.tokens[id] ?? "");
      prev = id;
    }
    const text = merged.filter((t) => t !== this.pad).join("");
    return this.clean ? cleanup(text).replaceAll(this.delimiter, " ").trim() : text;
  }
}

/** A CTC tokenizer from tokenizer.json (its CTC decoder) or vocab.json + tokenizer_config.json. */
export function ctcTokenizer(o: { tokenizerJson?: { model?: { vocab?: Record<string, number> }; decoder?: Record<string, unknown> | null }; vocab?: Record<string, number>; config?: Record<string, unknown> }): CtcTokenizer {
  const dec = o.tokenizerJson?.decoder;
  if (dec?.type === "CTC" && o.tokenizerJson?.model?.vocab) {
    return new CtcTokenizer(o.tokenizerJson.model.vocab, String(dec.pad_token ?? "<pad>"), String(dec.word_delimiter_token ?? "|"), dec.cleanup !== false);
  }
  const vocab = o.vocab ?? o.tokenizerJson?.model?.vocab;
  if (!vocab) throw new Error("no CTC vocabulary (vocab.json)");
  const c = o.config ?? {};
  const name = (v: unknown, d: string) => (typeof v === "string" ? v : typeof v === "object" && v && typeof (v as { content?: unknown }).content === "string" ? (v as { content: string }).content : d);
  return new CtcTokenizer(vocab, name(c.pad_token, "<pad>"), name(c.word_delimiter_token, "|"));
}
