// SentencePiece-style tokenizers for encoder-decoder models: Unigram (T5, Marian) and character BPE
// (NLLB, M2M100), with SentencePiece's Precompiled normalizer and Metaspace pre-tokenization, from a
// tokenizer.json or from a SentencePiece .model / .spm file.
import { TemplatedTokenizer, type EncoderTokenizerJSON } from "./template";

const SPACE = "▁";
const UTF8 = new TextEncoder();
const UTF8D = new TextDecoder();

/**
 * SentencePiece's precompiled character map: a Darts double-array trie over UTF-8 byte strings
 * whose values index a pool of NUL-terminated replacements. At each position the longest match is
 * replaced; anything unmatched is copied a character at a time (as normalizer.cc does).
 */
export class PrecompiledCharsMap {
  private readonly units: Uint32Array;
  private readonly pool: Uint8Array;

  constructor(blob: Uint8Array) {
    const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
    const trieBytes = dv.getUint32(0, true);
    // Copied unit by unit: `blob` may be a view (a Node Buffer's slice) at any alignment.
    this.units = new Uint32Array(trieBytes >>> 2);
    for (let k = 0; k < this.units.length; k++) this.units[k] = dv.getUint32(4 + 4 * k, true);
    this.pool = blob.subarray(4 + trieBytes);
  }

  static fromBase64(b64: string): PrecompiledCharsMap {
    return new PrecompiledCharsMap(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)));
  }

  /** Length and pool offset of the longest match at `key[from…]`, or undefined. */
  private longest(key: Uint8Array, from: number): { len: number; value: number } | undefined {
    const u = this.units;
    const offset = (x: number) => (x >>> 10) << ((x & (1 << 9)) >>> 6);
    let pos = offset(u[0]!);
    let best: { len: number; value: number } | undefined;
    for (let i = from; i < key.length; i++) {
      const c = key[i]!;
      pos ^= c;
      const unit = u[pos];
      if (unit === undefined || (unit & 0x800000ff) >>> 0 !== c) break;
      pos ^= offset(unit);
      if ((unit >>> 8) & 1) best = { len: i - from + 1, value: u[pos]! & 0x7fffffff };
    }
    return best;
  }

  normalize(text: string): string {
    const bytes = UTF8.encode(text);
    const out: number[] = [];
    for (let i = 0; i < bytes.length; ) {
      const m = this.longest(bytes, i);
      if (m) {
        for (let j = m.value; j < this.pool.length && this.pool[j] !== 0; j++) out.push(this.pool[j]!);
        i += m.len;
        continue;
      }
      // One UTF-8 character, as is.
      const b = bytes[i]!;
      const n = b < 0x80 ? 1 : b < 0xe0 ? 2 : b < 0xf0 ? 3 : 4;
      for (let j = 0; j < n && i < bytes.length; j++) out.push(bytes[i++]!);
    }
    return UTF8D.decode(Uint8Array.from(out));
  }
}

type Model =
  | { kind: "unigram"; pieces: Map<string, { id: number; score: number }>; maxLen: number; unk: number; unkScore: number }
  | { kind: "bpe"; vocab: Map<string, number>; ranks: Map<string, number>; unk: number | undefined };

export interface SentencePieceOptions {
  charsmap?: PrecompiledCharsMap;
  /** Collapse runs of spaces (tokenizers' Replace " {2,}", SentencePiece's remove_extra_whitespaces). */
  collapseSpaces?: boolean;
  /** Prepend ▁ to the text (Metaspace add_prefix_space / prepend_scheme, SentencePiece add_dummy_prefix). */
  prefixSpace?: boolean;
}

/**
 * Text → normalized (charsmap, collapsed spaces) → spaces as ▁, a ▁ prepended → words split before
 * each ▁ → pieces (Unigram Viterbi or BPE merges), unknown runs fused into one unk. Added tokens
 * (special tokens, language codes, <extra_id_N>) are matched in the raw text first.
 */
export class SentencePieceTokenizer extends TemplatedTokenizer {
  private readonly idTo: string[] = [];
  private readonly added = new Map<string, number>();
  private readonly addedRe: RegExp | undefined;
  readonly specialIds = new Set<number>();
  readonly maskToken: string | undefined = undefined;
  private readonly cache = new Map<string, number[]>();

  constructor(
    private readonly model: Model,
    added: { id: number; content: string; special: boolean }[],
    post: EncoderTokenizerJSON["post_processor"],
    private readonly opts: SentencePieceOptions,
  ) {
    super(post);
    if (model.kind === "unigram") for (const [p, v] of model.pieces) this.idTo[v.id] = p;
    else for (const [p, id] of model.vocab) this.idTo[id] = p;
    for (const t of added) {
      this.added.set(t.content, t.id);
      this.idTo[t.id] = t.content;
      if (t.special) this.specialIds.add(t.id);
    }
    if (this.added.size) {
      const alts = [...this.added.keys()].sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      this.addedRe = new RegExp(alts.join("|"), "g");
    }
  }

  get vocabSize(): number {
    return this.idTo.length;
  }

  token(id: number): string {
    return this.idTo[id] ?? "";
  }

  tokenId(token: string): number | undefined {
    if (this.added.has(token)) return this.added.get(token);
    return this.model.kind === "unigram" ? this.model.pieces.get(token)?.id : this.model.vocab.get(token);
  }

  /** Pieces of a text (no special tokens added), as strings. */
  pieces(text: string): string[] {
    return this.tokenize(text).map((id) => this.token(id));
  }

  protected tokenize(text: string): number[] {
    const out: number[] = [];
    let last = 0;
    let first = true;
    const span = (s: string) => {
      if (!s) return;
      this.span(s, first, out);
      first = false;
    };
    if (this.addedRe) {
      for (const m of text.matchAll(this.addedRe)) {
        span(text.slice(last, m.index));
        out.push(this.added.get(m[0])!);
        last = m.index! + m[0].length;
      }
    }
    span(text.slice(last));
    return out;
  }

  private span(text: string, first: boolean, out: number[]): void {
    let s = this.opts.charsmap ? this.opts.charsmap.normalize(text) : text;
    // SentencePiece's remove_extra_whitespaces: runs collapsed, both ends trimmed.
    if (this.opts.collapseSpaces) s = s.replace(/ {2,}/g, " ").trim();
    s = s.replaceAll(" ", SPACE);
    if (this.opts.prefixSpace && first && !s.startsWith(SPACE)) s = SPACE + s;
    // Words start at each ▁ (the ▁ stays with the word that follows it).
    for (const w of s.split(new RegExp(`(?=${SPACE})`, "u"))) {
      if (!w) continue;
      let ids = this.cache.get(w);
      if (!ids) {
        ids = this.model.kind === "unigram" ? this.viterbi(w) : this.bpe(w);
        if (this.cache.size < 50000) this.cache.set(w, ids);
      }
      out.push(...ids);
    }
  }

  /** Best-scoring segmentation; characters no piece covers become one unk each, then runs of unk fuse. */
  private viterbi(word: string): number[] {
    const m = this.model as Extract<Model, { kind: "unigram" }>;
    const chars = [...word];
    const n = chars.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const from = new Int32Array(n + 1).fill(-1);
    const piece = new Int32Array(n + 1).fill(-1);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let s = "";
      let covered = false;
      for (let j = i; j < Math.min(n, i + m.maxLen); j++) {
        s += chars[j];
        const p = m.pieces.get(s);
        if (!p) continue;
        if (j === i) covered = true;
        const sc = best[i]! + p.score;
        if (sc > best[j + 1]!) {
          best[j + 1] = sc;
          from[j + 1] = i;
          piece[j + 1] = p.id;
        }
      }
      if (!covered) {
        const sc = best[i]! + m.unkScore;
        if (sc > best[i + 1]!) {
          best[i + 1] = sc;
          from[i + 1] = i;
          piece[i + 1] = m.unk;
        }
      }
    }
    const ids: number[] = [];
    for (let k = n; k > 0; k = from[k]!) ids.push(piece[k]!);
    ids.reverse();
    return ids.filter((id, i) => !(id === m.unk && ids[i - 1] === m.unk));
  }

  /** Merges by rank over the word's characters; symbols missing from the vocabulary become unk (runs fused). */
  private bpe(word: string): number[] {
    const m = this.model as Extract<Model, { kind: "bpe" }>;
    const sym = [...word];
    for (;;) {
      let bestRank = Infinity;
      let at = -1;
      for (let i = 0; i + 1 < sym.length; i++) {
        const r = m.ranks.get(`${sym[i]} ${sym[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          at = i;
        }
      }
      if (at < 0) break;
      const a = sym[at]!;
      const b = sym[at + 1]!;
      // Every occurrence of the pair, left to right.
      const next: string[] = [];
      for (let i = 0; i < sym.length; i++) {
        if (i + 1 < sym.length && sym[i] === a && sym[i + 1] === b) {
          next.push(a + b);
          i++;
        } else next.push(sym[i]!);
      }
      sym.splice(0, sym.length, ...next);
    }
    const ids: number[] = [];
    for (const s of sym) {
      const id = m.vocab.get(s) ?? m.unk;
      if (id === undefined) continue;
      if (id === m.unk && ids.at(-1) === m.unk) continue;
      ids.push(id);
    }
    return ids;
  }

  decode(ids: readonly number[], skipSpecial = true): string {
    const s = ids
      .filter((id) => !(skipSpecial && this.specialIds.has(id)))
      .map((id) => this.token(id))
      .join("")
      .replaceAll(SPACE, " ");
    return s.startsWith(" ") ? s.slice(1) : s;
  }
}

/** A tokenizer.json with a Unigram or (non-byte-level) BPE model and SentencePiece normalization. */
export function sentencePieceFromJSON(json: EncoderTokenizerJSON): SentencePieceTokenizer {
  const norms = json.normalizer?.type === "Sequence" ? ((json.normalizer as { normalizers?: Record<string, unknown>[] }).normalizers ?? []) : json.normalizer ? [json.normalizer as Record<string, unknown>] : [];
  const pre = json.pre_tokenizer as { type?: string; add_prefix_space?: boolean; prepend_scheme?: string; pretokenizers?: { type: string; add_prefix_space?: boolean; prepend_scheme?: string }[] } | null | undefined;
  const meta = pre?.type === "Metaspace" ? pre : pre?.pretokenizers?.find((p) => p.type === "Metaspace");
  const opts: SentencePieceOptions = {
    charsmap: (() => {
      const p = norms.find((n) => n.type === "Precompiled") as { precompiled_charsmap?: string } | undefined;
      return p?.precompiled_charsmap ? PrecompiledCharsMap.fromBase64(p.precompiled_charsmap) : undefined;
    })(),
    collapseSpaces: norms.some((n) => n.type === "Replace" && JSON.stringify(n.pattern).includes(" {2,}")),
    prefixSpace: meta ? meta.add_prefix_space ?? meta.prepend_scheme !== "never" : false,
  };
  const added = json.added_tokens ?? [];
  const m = json.model as unknown as { type?: string; vocab: [string, number][] | Record<string, number>; unk_id?: number; unk_token?: string; merges?: (string | [string, string])[] };
  let model: Model;
  if (Array.isArray(m.vocab)) {
    const pieces = new Map<string, { id: number; score: number }>();
    let minScore = Infinity;
    let maxLen = 1;
    m.vocab.forEach(([p, score], id) => {
      pieces.set(p, { id, score });
      minScore = Math.min(minScore, score);
      maxLen = Math.max(maxLen, [...p].length);
    });
    model = { kind: "unigram", pieces, maxLen, unk: m.unk_id ?? 0, unkScore: minScore - 10 };
  } else {
    const vocab = new Map(Object.entries(m.vocab));
    const ranks = new Map<string, number>();
    (m.merges ?? []).forEach((x, i) => ranks.set(Array.isArray(x) ? x.join(" ") : x, i));
    model = { kind: "bpe", vocab, ranks, unk: m.unk_token !== undefined ? vocab.get(m.unk_token) : undefined };
  }
  return new SentencePieceTokenizer(model, added, json.post_processor, opts);
}

/** Fields of a SentencePiece ModelProto (sentencepiece_model.proto). */
interface SpmModel {
  pieces: { piece: string; score: number; type: number }[];
  modelType: number;
  charsmap?: Uint8Array;
  addDummyPrefix: boolean;
  removeExtraWhitespaces: boolean;
}

/** Minimal protobuf decoding of a .model / .spm file. */
export function parseSentencePieceModel(bytes: Uint8Array): SpmModel {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const varint = (p: { i: number }) => {
    let v = 0;
    let shift = 0;
    for (;;) {
      const b = bytes[p.i++]!;
      v += (b & 0x7f) * 2 ** shift;
      if (!(b & 0x80)) return v;
      shift += 7;
    }
  };
  /** Calls `f(field, wire, start, end)` for each field of the message in [from, to). */
  const fields = (from: number, to: number, f: (field: number, wire: number, p: { i: number }, len: number) => void) => {
    const p = { i: from };
    while (p.i < to) {
      const key = varint(p);
      const field = Math.floor(key / 8);
      const wire = key & 7;
      if (wire === 0) f(field, wire, p, varint(p));
      else if (wire === 2) {
        const len = varint(p);
        f(field, wire, p, len);
        p.i += len;
      } else if (wire === 5) {
        f(field, wire, p, 4);
        p.i += 4;
      } else if (wire === 1) {
        f(field, wire, p, 8);
        p.i += 8;
      } else throw new Error(`sentencepiece: unsupported wire type ${wire}`);
    }
  };
  const model: SpmModel = { pieces: [], modelType: 1, addDummyPrefix: true, removeExtraWhitespaces: true };
  fields(0, bytes.length, (field, wire, p, len) => {
    if (field === 1 && wire === 2) {
      const piece = { piece: "", score: 0, type: 1 };
      fields(p.i, p.i + len, (f2, w2, q, l2) => {
        if (f2 === 1 && w2 === 2) piece.piece = UTF8D.decode(bytes.subarray(q.i, q.i + l2));
        else if (f2 === 2 && w2 === 5) piece.score = dv.getFloat32(q.i, true);
        else if (f2 === 3 && w2 === 0) piece.type = l2;
      });
      model.pieces.push(piece);
    } else if (field === 2 && wire === 2) {
      fields(p.i, p.i + len, (f2, w2, _q, v) => {
        if (f2 === 3 && w2 === 0) model.modelType = v;
      });
    } else if (field === 3 && wire === 2) {
      fields(p.i, p.i + len, (f2, w2, q, l2) => {
        if (f2 === 2 && w2 === 2) model.charsmap = Uint8Array.prototype.slice.call(bytes, q.i, q.i + l2);
        else if (f2 === 3 && w2 === 0) model.addDummyPrefix = l2 !== 0;
        else if (f2 === 4 && w2 === 0) model.removeExtraWhitespaces = l2 !== 0;
      });
    }
  });
  return model;
}

/**
 * Marian's tokenizer: pieces from the source (or target) SentencePiece model, ids through the
 * shared vocab.json, `</s>` appended; decoding maps ids back to pieces.
 */
export function marianTokenizer(spm: Uint8Array, vocab: Record<string, number>, eosToken = "</s>", unkToken = "<unk>", padToken = "<pad>"): SentencePieceTokenizer {
  const m = parseSentencePieceModel(spm);
  if (m.modelType !== 1) throw new Error("Marian: only Unigram SentencePiece models are supported");
  // Segment with the spm's own pieces and scores, then renumber through vocab.json.
  const pieces = new Map<string, { id: number; score: number }>();
  let minScore = Infinity;
  let maxLen = 1;
  const unk = vocab[unkToken] ?? 1;
  for (const p of m.pieces) {
    if (p.type !== 1 && p.type !== 4) continue; // normal and user-defined pieces only
    pieces.set(p.piece, { id: vocab[p.piece] ?? unk, score: p.score });
    minScore = Math.min(minScore, p.score);
    maxLen = Math.max(maxLen, [...p.piece].length);
  }
  const tok = new SentencePieceTokenizer(
    { kind: "unigram", pieces, maxLen, unk, unkScore: minScore - 10 },
    [eosToken, unkToken, padToken].filter((t) => vocab[t] !== undefined).map((t) => ({ id: vocab[t]!, content: t, special: true })),
    { type: "TemplateProcessing", single: [{ Sequence: { id: "A", type_id: 0 } }, { SpecialToken: { id: eosToken, type_id: 0 } }], special_tokens: { [eosToken]: { ids: [vocab[eosToken]!] } } },
    { charsmap: m.charsmap ? new PrecompiledCharsMap(m.charsmap) : undefined, collapseSpaces: m.removeExtraWhitespaces, prefixSpace: m.addDummyPrefix },
  );
  // Ids outside the spm (other-language pieces of the shared vocabulary) still decode.
  for (const [p, id] of Object.entries(vocab)) if (!tok.token(id)) (tok as unknown as { idTo: string[] }).idTo[id] = p;
  return tok;
}
