import { byteToUnicode } from "./tokenizer";

/** The parts of a CLIP tokenizer.json this tokenizer reads. */
export interface ClipTokenizerJSON {
  model: { type: string; vocab: Record<string, number>; merges: (string | [string, string])[]; end_of_word_suffix?: string | null; unk_token?: string | null };
  added_tokens?: { id: number; content: string }[];
  post_processor?: { cls?: [string, number]; sep?: [string, number] } | null;
}

const CLIP_SPLIT = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/gu;

/**
 * CLIP's tokenizer: NFC, whitespace runs to one space, lowercase; the pieces CLIP's split keeps
 * (whitespace dropped), byte-level mapped; BPE with "</w>" closing each word; then
 * <|startoftext|> … <|endoftext|>, at most `maxLength` tokens.
 */
export class ClipTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly ranks = new Map<string, number>();
  private readonly suffix: string;
  private readonly cache = new Map<string, number[]>();
  readonly bos: number;
  readonly eos: number;
  private readonly unk: number | undefined;

  constructor(
    json: ClipTokenizerJSON,
    readonly maxLength = 77,
  ) {
    if (json.model.type !== "BPE") throw new Error(`CLIP tokenizer: model type ${json.model.type} is not BPE`);
    this.vocab = new Map(Object.entries(json.model.vocab));
    for (const t of json.added_tokens ?? []) this.vocab.set(t.content, t.id);
    json.model.merges.forEach((m, i) => this.ranks.set(Array.isArray(m) ? m.join(" ") : m, i));
    this.suffix = json.model.end_of_word_suffix ?? "</w>";
    this.bos = json.post_processor?.cls?.[1] ?? this.vocab.get("<|startoftext|>")!;
    this.eos = json.post_processor?.sep?.[1] ?? this.vocab.get("<|endoftext|>")!;
    this.unk = json.model.unk_token ? this.vocab.get(json.model.unk_token) : undefined;
  }

  private bpe(word: string): number[] {
    const hit = this.cache.get(word);
    if (hit) return hit;
    const special = this.vocab.get(word);
    if (word === "<|startoftext|>" || word === "<|endoftext|>") return [special!];
    let parts = Array.from(word);
    parts[parts.length - 1] += this.suffix;
    while (parts.length > 1) {
      let best = -1;
      let rank = Infinity;
      for (let i = 0; i + 1 < parts.length; i++) {
        const r = this.ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (r !== undefined && r < rank) [rank, best] = [r, i];
      }
      if (best < 0) break;
      const a = parts[best]!;
      const b = parts[best + 1]!;
      // Merge every occurrence of the pair, left to right, as the BPE models do.
      const next: string[] = [];
      for (let i = 0; i < parts.length; i++) {
        if (i + 1 < parts.length && parts[i] === a && parts[i + 1] === b) {
          next.push(a + b);
          i++;
        } else next.push(parts[i]!);
      }
      parts = next;
    }
    const ids: number[] = [];
    for (const p of parts) {
      const id = this.vocab.get(p);
      if (id !== undefined) ids.push(id);
      else if (this.unk !== undefined) ids.push(this.unk);
    }
    // Bounded: a long stream of distinct words (or hostile input) must not grow it without end.
    if (this.cache.size >= 50_000) this.cache.clear();
    this.cache.set(word, ids);
    return ids;
  }

  /** Token ids of a text, with the start and end tokens. */
  encode(text: string): number[] {
    const norm = text.normalize("NFC").replace(/\s+/gu, " ").toLowerCase();
    const ids: number[] = [];
    for (const m of norm.matchAll(CLIP_SPLIT)) {
      const mapped = Array.from(new TextEncoder().encode(m[0]), byteToUnicode).join("");
      ids.push(...this.bpe(m[0].startsWith("<|") ? m[0] : mapped));
    }
    return [this.bos, ...ids.slice(0, this.maxLength - 2), this.eos];
  }
}
