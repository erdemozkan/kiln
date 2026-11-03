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
  normalizer?: { type: string } | null;
  pre_tokenizer?: Parameters<typeof compilePreTokenizer>[0];
}

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

  constructor(
    private readonly vocab: Map<string, number>,
    merges: readonly string[],
    addedTokens: readonly AddedToken[],
    private readonly pre: PreStep[],
    private readonly opts: { ignoreMerges?: boolean; nfc?: boolean } = {},
  ) {
    merges.forEach((m, i) => this.ranks.set(m, i));
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
    if (json.model.type !== "BPE") throw new Error(`tokenizer model "${json.model.type}" is not supported yet (BPE only)`);
    const usesByteLevel = JSON.stringify(json.pre_tokenizer ?? {}).includes("ByteLevel");
    if (!usesByteLevel) throw new Error("only byte-level BPE tokenizers are supported in the MVP (Llama 3, Qwen, SmolLM)");
    const merges = json.model.merges.map((m) => (Array.isArray(m) ? m.join(" ") : m));
    return new Tokenizer(new Map(Object.entries(json.model.vocab)), merges, json.added_tokens ?? [], compilePreTokenizer(json.pre_tokenizer), {
      ignoreMerges: json.model.ignore_merges,
      nfc: json.normalizer?.type === "NFC",
    });
  }

  /** Builds a tokenizer from GGUF `tokenizer.ggml.*` metadata ("gpt2" models only). */
  static fromGGUF(meta: Map<string, unknown>): Tokenizer {
    const model = meta.get("tokenizer.ggml.model");
    if (model !== "gpt2") throw new Error(`GGUF tokenizer model "${String(model)}" is not supported yet (gpt2 only)`);
    const tokens = meta.get("tokenizer.ggml.tokens") as string[];
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
    const pattern = pre === "llama-bpe" || pre === "llama3" ? LLAMA3_PATTERN : pre.startsWith("qwen") ? QWEN2_PATTERN : GPT2_PATTERN;
    const re = toJsRegex(pattern);
    const steps: PreStep[] = [(ps) => ps.flatMap((p) => splitIsolated(p, re))];
    if (pre === "smollm") steps.unshift((ps) => ps.flatMap((p) => splitIsolated(p, /\p{N}/gu)));
    return new Tokenizer(vocab, merges, added, steps, { ignoreMerges: pre === "llama-bpe" || pre === "llama3" });
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
    return Uint8Array.from(Array.from(tok, (c) => BYTES.dec.get(c) ?? 63));
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
      let pieces = [seg];
      for (const step of this.pre) pieces = step(pieces);
      for (const p of pieces) {
        const mapped = Array.from(new TextEncoder().encode(p), (b) => BYTES.enc[b]).join("");
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
    return {
      push: (id) => {
        const tok = this.idToToken[id];
        if (tok === undefined) return "";
        if (this.added.has(tok)) {
          if (skipSpecial && this.specialIds.has(id)) return "";
          return utf8.decode() + tok;
        }
        return utf8.decode(this.tokenBytes(id), { stream: true });
      },
      flush: () => utf8.decode(),
    };
  }

  private bpe(word: string): number[] {
    const hit = this.cache.get(word);
    if (hit) return hit;
    const direct = this.vocab.get(word);
    if (this.opts.ignoreMerges && direct !== undefined) return [direct];
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = Infinity;
      let pair = "";
      for (let i = 0; i < parts.length - 1; i++) {
        const key = `${parts[i]} ${parts[i + 1]}`;
        const r = this.ranks.get(key);
        if (r !== undefined && r < best) {
          best = r;
          pair = key;
        }
      }
      if (best === Infinity) break;
      const merged: string[] = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && `${parts[i]} ${parts[i + 1]}` === pair) {
          merged.push(parts[i]! + parts[i + 1]!);
          i++;
        } else merged.push(parts[i]!);
      }
      parts = merged;
    }
    const ids = parts.map((p) => {
      const id = this.vocab.get(p);
      if (id === undefined) throw new Error(`tokenizer: "${p}" is not in the vocabulary`);
      return id;
    });
    if (this.cache.size < 50_000) this.cache.set(word, ids);
    return ids;
  }
}
