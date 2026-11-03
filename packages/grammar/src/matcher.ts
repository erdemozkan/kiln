/**
 * A byte-level pushdown matcher. A grammar is a tree of `Node`s; a match in progress is a stack of
 * immutable `Frame`s (top first). Frames never change, so a state can be stepped down every branch of
 * the token trie without copying, and identical states are recognised by their `key`.
 *
 * The matcher is nondeterministic: stepping a byte returns every state it can lead to, so alternatives
 * (`anyOf`, enum values sharing a prefix, optional keys) are tracked side by side until they diverge.
 */

export type Node =
  | { kind: "lit"; bytes: Uint8Array }
  | { kind: "string"; minLength: number; maxLength: number }
  | { kind: "number"; integer: boolean }
  | { kind: "object"; props: readonly Prop[]; additional: Node | undefined; ws?: Whitespace }
  | { kind: "array"; items: Node; minItems: number; maxItems: number; ws?: Whitespace }
  | { kind: "alt"; nodes: readonly Node[] };

/** Whitespace allowed between the tokens of an object or array. */
export interface Whitespace {
  /** Longest run. */
  max: number;
  /** Newlines and tabs as well as spaces. */
  breaks: boolean;
}

/** Enough for pretty-printing, too short to loop on. */
export const PRETTY: Whitespace = { max: 32, breaks: true };
/** One space at most, as in `{"a": 1, "b": [2, 3]}`. */
export const COMPACT: Whitespace = { max: 1, breaks: false };

export interface Prop {
  /** The key as JSON, without its opening quote: `name"`. */
  key: Uint8Array;
  value: Node;
  required: boolean;
}

/** One level of a match in progress. */
export interface Frame {
  /** Identifies the frame's grammar position; equal keys behave identically. */
  readonly key: string;
  /** Replacements of this frame after consuming `b` (an empty replacement: the frame finished). */
  step(b: number): Frame[][];
  /** Replacements of this frame that consume nothing (an empty one: it may finish here). */
  eps(): Frame[][];
}

/** A match state: frames top first. `null` is the empty stack, the end of a complete match. */
export class StackNode {
  private k: string | undefined;
  constructor(
    readonly f: Frame,
    readonly up: Stack,
  ) {}
  /** Identifies the state; computed on demand, since most states in a trie walk are never compared. */
  get key(): string {
    return (this.k ??= this.up ? `${this.f.key}|${this.up.key}` : this.f.key);
  }
}
export type Stack = StackNode | null;

export const push = (up: Stack, f: Frame): Stack => new StackNode(f, up);

/** Pushes `frames` (bottom first) onto `up`. */
function pushAll(up: Stack, frames: readonly Frame[]): Stack {
  let s = up;
  for (const f of frames) s = push(s, f);
  return s;
}

/** Every state reachable from `s` by consuming byte `b`. */
export function stepStack(s: Stack, b: number, out: Stack[] = []): Stack[] {
  if (!s) return out;
  for (const r of s.f.step(b)) out.push(r.length === 1 && r[0] === s.f ? s : pushAll(s.up, r));
  for (const r of s.f.eps()) stepStack(pushAll(s.up, r), b, out);
  return out;
}

/** Whether the input so far is a complete match. */
export function accepts(s: Stack): boolean {
  if (!s) return true;
  return s.f.eps().some((r) => accepts(pushAll(s.up, r)));
}

/** States after consuming `b` from any of `states`, without duplicates. */
export function stepAll(states: readonly Stack[], b: number): Stack[] {
  const out: Stack[] = [];
  for (const s of states) stepStack(s, b, out);
  if (out.length < 2) return out;
  const seen = new Set<string>();
  return out.filter((s) => {
    const k = s?.key ?? "";
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

const isWs = (b: number) => b === 0x20 || b === 0x0a || b === 0x0d || b === 0x09;
const isDigit = (b: number) => b >= 0x30 && b <= 0x39;
const isHex = (b: number) => isDigit(b) || (b >= 0x41 && b <= 0x46) || (b >= 0x61 && b <= 0x66);
const ch = (c: string) => c.charCodeAt(0);
const NONE: Frame[][] = [];
const DONE: Frame[][] = [[]];

/** Longest run of digits in each part of a number. */
const MAX_DIGITS = 16;

/** Optional whitespace. */
class Ws implements Frame {
  readonly key: string;
  constructor(
    readonly rule: Whitespace = PRETTY,
    readonly n = 0,
  ) {
    this.key = `w${rule.max}${rule.breaks ? "b" : ""}:${n}`;
  }
  step(b: number): Frame[][] {
    const ok = this.rule.breaks ? isWs(b) : b === 0x20;
    return ok && this.n < this.rule.max ? [[new Ws(this.rule, this.n + 1)]] : NONE;
  }
  eps(): Frame[][] {
    return DONE;
  }
}

/** The rest of a fixed byte string. */
class Lit implements Frame {
  readonly key: string;
  constructor(
    readonly bytes: Uint8Array,
    readonly id: string,
    readonly i = 0,
  ) {
    this.key = `l${id}@${i}`;
  }
  step(b: number): Frame[][] {
    if (this.bytes[this.i] !== b) return NONE;
    return this.i + 1 === this.bytes.length ? DONE : [[new Lit(this.bytes, this.id, this.i + 1)]];
  }
  eps(): Frame[][] {
    return this.bytes.length === 0 ? DONE : NONE;
  }
}

/** Phases of a JSON string: before the quote, in the body, after a backslash, in a \u escape (4 hex left = 4). */
class Str implements Frame {
  readonly key: string;
  constructor(
    readonly node: { minLength: number; maxLength: number },
    readonly phase: "open" | "body" | "esc" | 1 | 2 | 3 | 4 = "open",
    /** Characters so far; only tracked when a length bound needs it. */
    readonly len = 0,
  ) {
    this.key = `s${node.minLength},${node.maxLength}:${phase}:${len}`;
  }
  private next(phase: Str["phase"], len = this.len): Frame[][] {
    const bounded = this.node.minLength > 0 || this.node.maxLength < Infinity;
    const l = bounded ? Math.min(len, this.node.maxLength + 1) : 0;
    // Unbounded string bodies stay the same frame, so walking a long run of text allocates nothing.
    return [[phase === this.phase && l === this.len ? this : new Str(this.node, phase, l)]];
  }
  step(b: number): Frame[][] {
    const { phase, len, node } = this;
    if (phase === "open") return b === ch('"') ? this.next("body") : NONE;
    if (phase === "body") {
      if (b === ch('"')) return len >= node.minLength ? DONE : NONE;
      // A UTF-8 continuation byte does not start a character.
      const add = b >= 0x80 && b < 0xc0 ? 0 : 1;
      if (len + add > node.maxLength) return NONE;
      if (b === ch("\\")) return this.next("esc");
      if (b < 0x20) return NONE;
      return this.next("body", len + add);
    }
    if (phase === "esc") {
      if (b === ch("u")) return this.next(4, len + 1);
      return '"\\/bfnrt'.includes(String.fromCharCode(b)) ? this.next("body", len + 1) : NONE;
    }
    if (!isHex(b)) return NONE;
    return phase === 1 ? this.next("body") : this.next((phase - 1) as 1 | 2 | 3);
  }
  eps(): Frame[][] {
    return NONE;
  }
}

type NumPhase = "start" | "minus" | "zero" | "int" | "dot" | "frac" | "e" | "esign" | "exp";

/** -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?, with each digit run capped. */
class Num implements Frame {
  readonly key: string;
  constructor(
    readonly integer: boolean,
    readonly phase: NumPhase = "start",
    readonly digits = 0,
  ) {
    this.key = `n${integer ? "i" : "f"}:${phase}:${digits}`;
  }
  private to(phase: NumPhase, digits = 0) {
    return [[new Num(this.integer, phase, digits)]];
  }
  step(b: number): Frame[][] {
    const { phase, digits, integer } = this;
    const d = isDigit(b);
    const exp = b === ch("e") || b === ch("E");
    switch (phase) {
      case "start":
      case "minus":
        if (phase === "start" && b === ch("-")) return this.to("minus");
        return b === ch("0") ? this.to("zero") : d ? this.to("int", 1) : NONE;
      case "int":
      case "zero":
        if (d) return phase === "int" && digits < MAX_DIGITS ? this.to("int", digits + 1) : NONE;
        if (integer) return NONE;
        return b === ch(".") ? this.to("dot") : exp ? this.to("e") : NONE;
      case "dot":
        return d ? this.to("frac", 1) : NONE;
      case "frac":
        if (d) return digits < MAX_DIGITS ? this.to("frac", digits + 1) : NONE;
        return exp && !integer ? this.to("e") : NONE;
      case "e":
      case "esign":
        if (phase === "e" && (b === ch("+") || b === ch("-"))) return this.to("esign");
        return d ? this.to("exp", 1) : NONE;
      case "exp":
        return d && digits < 3 ? this.to("exp", digits + 1) : NONE;
    }
  }
  eps(): Frame[][] {
    return this.phase === "zero" || this.phase === "int" || this.phase === "frac" || this.phase === "exp" ? DONE : NONE;
  }
}

/**
 * An object. Declared properties come in schema order; optional ones may be skipped, and keys outside
 * the declared ones are allowed only when `additional` is set.
 */
class Obj implements Frame {
  readonly key: string;
  constructor(
    readonly node: Extract<Node, { kind: "object" }>,
    readonly id: string,
    readonly phase: "open" | "first" | "key" | "inKey" | "colon" | "after" = "open",
    /** First declared property that may still come. */
    readonly idx = 0,
    /** inKey: declared properties whose key matches so far, and bytes matched. colon: chosen property (-1: additional). */
    readonly cands: readonly number[] = [],
    readonly pos = 0,
  ) {
    this.key = `o${id}:${phase}:${idx}:${cands.join(",")}:${pos}`;
  }
  private with(phase: Obj["phase"], idx = this.idx, cands: readonly number[] = [], pos = 0) {
    return new Obj(this.node, this.id, phase, idx, cands, pos);
  }
  /** Declared properties that may come next: from idx up to and including the first required one. */
  private open(): number[] {
    const out: number[] = [];
    for (let j = this.idx; j < this.node.props.length; j++) {
      out.push(j);
      if (this.node.props[j]!.required) break;
    }
    return out;
  }
  private canClose() {
    return this.node.props.slice(this.idx).every((p) => !p.required);
  }
  step(b: number): Frame[][] {
    const { node, phase } = this;
    switch (phase) {
      case "open":
        return b === ch("{") ? [[this.with("first"), new Ws(this.node.ws)]] : NONE;
      case "first":
      case "key": {
        if (b === ch("}")) return phase === "first" && this.canClose() ? DONE : NONE;
        if (b !== ch('"')) return NONE;
        const out: Frame[][] = [];
        const cands = this.open();
        if (cands.length) out.push([this.with("inKey", this.idx, cands, 0)]);
        if (node.additional) out.push([this.with("colon", this.idx, [-1]), new Ws(this.node.ws), new Str({ minLength: 0, maxLength: Infinity }, "body")]);
        return out;
      }
      case "inKey": {
        const next = this.cands.filter((j) => node.props[j]!.key[this.pos] === b);
        const out: Frame[][] = [];
        const done = next.filter((j) => node.props[j]!.key.length === this.pos + 1);
        for (const j of done) out.push([this.with("colon", this.idx, [j]), new Ws(this.node.ws)]);
        const going = next.filter((j) => node.props[j]!.key.length > this.pos + 1);
        if (going.length) out.push([this.with("inKey", this.idx, going, this.pos + 1)]);
        return out;
      }
      case "colon": {
        if (b !== ch(":")) return NONE;
        const j = this.cands[0]!;
        const value = j < 0 ? node.additional! : node.props[j]!.value;
        return [[this.with("after", j < 0 ? this.idx : j + 1), new Ws(this.node.ws), start(value), new Ws(this.node.ws)]];
      }
      case "after":
        if (b === ch("}")) return this.canClose() ? DONE : NONE;
        if (b === ch(",") && (this.open().length > 0 || node.additional)) return [[this.with("key"), new Ws(this.node.ws)]];
        return NONE;
    }
  }
  eps(): Frame[][] {
    return NONE;
  }
}

class Arr implements Frame {
  readonly key: string;
  constructor(
    readonly node: Extract<Node, { kind: "array" }>,
    readonly id: string,
    readonly phase: "open" | "first" | "after" = "open",
    readonly count = 0,
  ) {
    this.key = `a${id}:${phase}:${node.maxItems < Infinity || node.minItems > 0 ? count : 0}`;
  }
  private item(count: number): Frame[] {
    return [new Arr(this.node, this.id, "after", count), new Ws(this.node.ws), start(this.node.items)];
  }
  step(b: number): Frame[][] {
    const { node, phase, count } = this;
    if (phase === "open") return b === ch("[") ? [[new Arr(node, this.id, "first"), new Ws(this.node.ws)]] : NONE;
    if (b === ch("]")) return (phase === "first" ? 0 : count) >= node.minItems ? DONE : NONE;
    if (phase === "after" && b === ch(",") && count < node.maxItems) return [[...this.item(count + 1), new Ws(this.node.ws)]];
    return NONE;
  }
  eps(): Frame[][] {
    return this.phase === "first" && this.node.maxItems > 0 ? [this.item(1)] : NONE;
  }
}

/** A choice between grammars, expanded without consuming anything. */
class Alt implements Frame {
  readonly key: string;
  constructor(
    readonly node: Extract<Node, { kind: "alt" }>,
    id: string,
  ) {
    this.key = `x${id}`;
  }
  step(): Frame[][] {
    return NONE;
  }
  eps(): Frame[][] {
    return this.node.nodes.map((n) => [start(n)]);
  }
}

/** Distinct ids for grammar nodes, so frame keys tell positions in different nodes apart. */
const ids = new WeakMap<object, string>();
let nextId = 0;
function idOf(node: object): string {
  let id = ids.get(node);
  if (!id) ids.set(node, (id = (nextId++).toString(36)));
  return id;
}

/** The frame that matches `node` from its first byte. */
export function start(node: Node): Frame {
  switch (node.kind) {
    case "lit":
      return new Lit(node.bytes, idOf(node));
    case "string":
      return new Str(node);
    case "number":
      return new Num(node.integer);
    case "object":
      return new Obj(node, idOf(node));
    case "array":
      return new Arr(node, idOf(node));
    case "alt":
      return new Alt(node, idOf(node));
  }
}
