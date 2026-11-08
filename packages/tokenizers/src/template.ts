// Shared by the encoder tokenizers: the Encoding / EncoderTokenizer types and the post-processor
// templates (special tokens, segment ids, longest-first truncation).
import type { AddedToken, TokenizerJSON } from "./tokenizer";

/** One encoded input. */
export interface Encoding {
  ids: number[];
  /** Segment (token type) per token: 0 for the first text, 1 for the second (BERT pairs). */
  typeIds: number[];
  /** Which text each token came from: 0 or 1, or -1 for a special token. */
  seq: number[];
}

export interface EncoderTokenizer {
  /** `a`, or the pair (a, b), with special tokens; the longer text is cut first past `maxLength`. */
  encode(a: string, b?: string, maxLength?: number): Encoding;
  decode(ids: readonly number[], skipSpecial?: boolean): string;
  /** The vocabulary string of an id (WordPiece "##ing", byte-level "Ġthe"). */
  token(id: number): string;
  tokenId(token: string): number | undefined;
  readonly maskToken: string | undefined;
  readonly vocabSize: number;
  readonly specialIds: ReadonlySet<number>;
}

interface TemplatePiece {
  SpecialToken?: { id: string; type_id: number };
  Sequence?: { id: "A" | "B"; type_id: number };
}

export interface EncoderTokenizerJSON extends Omit<TokenizerJSON, "model" | "normalizer"> {
  model: { type?: string; vocab: Record<string, number> | [string, number][]; merges?: (string | [string, string])[]; unk_token?: string; continuing_subword_prefix?: string; max_input_chars_per_word?: number };
  normalizer?: { type: string; lowercase?: boolean; strip_accents?: boolean | null; clean_text?: boolean; handle_chinese_chars?: boolean } | null;
  post_processor?: {
    type: string;
    single?: TemplatePiece[];
    pair?: TemplatePiece[];
    special_tokens?: Record<string, { ids: number[] }>;
    cls?: [string, number];
    sep?: [string, number];
  } | null;
  added_tokens?: (AddedToken & { lstrip?: boolean; rstrip?: boolean })[];
}

/** One piece of a post-processor template: special token ids, or a text's tokens, with a segment id. */
type Piece = { special: number[]; type: number } | { seq: 0 | 1; type: number };

function templateOf(pp: EncoderTokenizerJSON["post_processor"], which: "single" | "pair"): Piece[] {
  if (pp?.type === "TemplateProcessing") {
    return (pp[which] ?? []).map((p) =>
      p.SpecialToken
        ? { special: pp.special_tokens?.[p.SpecialToken.id]?.ids ?? [], type: p.SpecialToken.type_id }
        : { seq: p.Sequence!.id === "A" ? 0 : 1, type: p.Sequence!.type_id },
    );
  }
  if (pp?.type === "RobertaProcessing" || pp?.type === "BertProcessing") {
    const cls = pp.cls![1];
    const sep = pp.sep![1];
    if (pp.type === "BertProcessing") {
      return which === "single" ? [{ special: [cls], type: 0 }, { seq: 0, type: 0 }, { special: [sep], type: 0 }] : [{ special: [cls], type: 0 }, { seq: 0, type: 0 }, { special: [sep], type: 0 }, { seq: 1, type: 1 }, { special: [sep], type: 1 }];
    }
    // <s> A </s></s> B </s>, all segment 0.
    return which === "single" ? [{ special: [cls], type: 0 }, { seq: 0, type: 0 }, { special: [sep], type: 0 }] : [{ special: [cls], type: 0 }, { seq: 0, type: 0 }, { special: [sep, sep], type: 0 }, { seq: 1, type: 0 }, { special: [sep], type: 0 }];
  }
  return which === "single" ? [{ seq: 0, type: 0 }] : [{ seq: 0, type: 0 }, { seq: 1, type: 1 }];
}

/** Shared by both tokenizer kinds: templates, truncation (longest first) and segment ids. */
export abstract class TemplatedTokenizer implements EncoderTokenizer {
  private readonly single: Piece[];
  private readonly pair: Piece[];
  abstract readonly maskToken: string | undefined;
  abstract readonly vocabSize: number;
  abstract readonly specialIds: ReadonlySet<number>;

  constructor(pp: EncoderTokenizerJSON["post_processor"]) {
    this.single = templateOf(pp, "single");
    this.pair = templateOf(pp, "pair");
  }

  /** Ids of one text, no special tokens added. */
  protected abstract tokenize(text: string): number[];

  /** Ids of one text without the template's special tokens. */
  textIds(text: string): number[] {
    return this.tokenize(text);
  }
  abstract decode(ids: readonly number[], skipSpecial?: boolean): string;
  abstract token(id: number): string;
  abstract tokenId(token: string): number | undefined;

  encode(a: string, b?: string, maxLength = Infinity): Encoding {
    const template = b === undefined ? this.single : this.pair;
    const texts = [this.tokenize(a), b === undefined ? [] : this.tokenize(b)];
    const added = template.reduce((n, p) => n + ("special" in p ? p.special.length : 0), 0);
    // Longest first: drop the last token of the longer text until everything fits.
    while (texts[0]!.length + texts[1]!.length + added > maxLength && texts[0]!.length + texts[1]!.length > 0) {
      const longer = texts[0]!.length >= texts[1]!.length ? 0 : 1;
      texts[longer]!.pop();
    }
    const out: Encoding = { ids: [], typeIds: [], seq: [] };
    for (const p of template) {
      const ids = "special" in p ? p.special : texts[p.seq]!;
      for (const id of ids) {
        out.ids.push(id);
        out.typeIds.push(p.type);
        out.seq.push("special" in p ? -1 : p.seq);
      }
    }
    return out;
  }
}

