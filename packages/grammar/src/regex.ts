import type { Node } from "./matcher";

/**
 * Regular expressions (the JavaScript syntax, without backreferences, lookaround or word boundaries)
 * compiled to grammar nodes, matched against the whole output: `^` and `$` at the ends are optional.
 * Supported: literals and escapes, `.`, classes `[a-z\d]` and `[^…]`, `\d \w \s` and their negations,
 * groups `(…)` `(?:…)` `(?<name>…)`, alternation, and the quantifiers `* + ? {n} {n,} {n,m}` (a trailing
 * `?` for laziness is accepted and means nothing to a grammar).
 */

const utf8 = new TextEncoder();
type Ranges = [number, number][];
type CharClass = { ranges: Ranges; negate: boolean };

export const DIGIT: Ranges = [[0x30, 0x39]];
export const WORD: Ranges = [[0x30, 0x39], [0x41, 0x5a], [0x5f, 0x5f], [0x61, 0x7a]];
export const SPACE: Ranges = [[0x09, 0x0d], [0x20, 0x20], [0xa0, 0xa0], [0x1680, 0x1680], [0x2000, 0x200a], [0x2028, 0x2029], [0x202f, 0x202f], [0x205f, 0x205f], [0x3000, 0x3000], [0xfeff, 0xfeff]];

/** Disjoint, sorted ranges. */
function normalize(r: Ranges): Ranges {
  const s = [...r].sort((a, b) => a[0] - b[0]);
  const out: Ranges = [];
  for (const [a, z] of s) {
    const last = out[out.length - 1];
    if (last && a <= last[1] + 1) last[1] = Math.max(last[1], z);
    else out.push([a, z]);
  }
  return out;
}

/** The code points of 0…0x10ffff outside `r`. */
function complement(r: Ranges): Ranges {
  const out: Ranges = [];
  let at = 0;
  for (const [a, z] of normalize(r)) {
    if (a > at) out.push([at, a - 1]);
    at = z + 1;
  }
  if (at <= 0x10ffff) out.push([at, 0x10ffff]);
  return out;
}

export const lit = (s: string): Node => ({ kind: "lit", bytes: utf8.encode(s) });
const seq = (nodes: Node[]): Node => (nodes.length === 1 ? nodes[0]! : { kind: "seq", nodes });
const alt = (nodes: Node[]): Node => (nodes.length === 1 ? nodes[0]! : { kind: "alt", nodes });

/** Characters a JSON string body cannot hold as themselves: control characters, `"` and `\`. */
const JSON_SPECIAL: Ranges = [[0, 0x1f], [0x22, 0x22], [0x5c, 0x5c]];

/** JSON's escape of a character that cannot appear raw in a string. */
function jsonEscape(cp: number): string {
  const short: Record<number, string> = { 0x22: '\\"', 0x5c: "\\\\", 0x0a: "\\n", 0x0d: "\\r", 0x09: "\\t", 0x08: "\\b", 0x0c: "\\f" };
  return short[cp] ?? `\\u${cp.toString(16).padStart(4, "0")}`;
}

export interface RegexOptions {
  /**
   * Match the text as the body of a JSON string (JSON Schema `pattern`): a literal `"`, `\` or control
   * character becomes its JSON escape, and classes leave those characters out.
   */
  json?: boolean;
}

/** Parses a character class body after `[` (or `[^`): ranges, escapes. Shared with GBNF. */
export function parseClass(src: string, at: number, end = "]"): { cls: CharClass; at: number } {
  let negate = false;
  if (src[at] === "^") {
    negate = true;
    at++;
  }
  const ranges: Ranges = [];
  let first = true;
  while (at < src.length && (src[at] !== end || first)) {
    first = false;
    let lo: number;
    if (src[at] === "\\") {
      const e = parseEscape(src, at + 1, true);
      at = e.at;
      if (e.ranges) {
        ranges.push(...e.ranges);
        continue;
      }
      lo = e.cp!;
    } else {
      lo = src.codePointAt(at)!;
      at += lo > 0xffff ? 2 : 1;
    }
    if (src[at] === "-" && src[at + 1] !== end && at + 1 < src.length) {
      at++;
      let hi: number;
      if (src[at] === "\\") {
        const e = parseEscape(src, at + 1, true);
        if (e.ranges) throw new Error(`regex: a class escape cannot end a range in "${src}"`);
        at = e.at;
        hi = e.cp!;
      } else {
        hi = src.codePointAt(at)!;
        at += hi > 0xffff ? 2 : 1;
      }
      if (hi < lo) throw new Error(`regex: range out of order in "${src}"`);
      ranges.push([lo, hi]);
    } else ranges.push([lo, lo]);
  }
  if (src[at] !== end) throw new Error(`regex: unterminated class in "${src}"`);
  return { cls: { ranges: normalize(ranges), negate }, at: at + 1 };
}

/** Parses an escape after the backslash: a code point, or a class (\d, \w, \s and negations). */
export function parseEscape(src: string, at: number, inClass = false): { cp?: number; ranges?: Ranges; at: number } {
  const c = src[at];
  if (c === undefined) throw new Error("regex: trailing backslash");
  const simple: Record<string, number> = { n: 10, r: 13, t: 9, f: 12, v: 11, "0": 0 };
  if (c in simple && !(c === "0" && /[0-9]/.test(src[at + 1] ?? ""))) return { cp: simple[c]!, at: at + 1 };
  if (inClass && c === "b") return { cp: 8, at: at + 1 };
  const classes: Record<string, Ranges> = { d: DIGIT, D: complement(DIGIT), w: WORD, W: complement(WORD), s: SPACE, S: complement(SPACE) };
  if (c in classes) return { ranges: classes[c]!, at: at + 1 };
  if (c === "x") {
    const h = src.slice(at + 1, at + 3);
    if (!/^[0-9a-fA-F]{2}$/.test(h)) throw new Error(`regex: bad \\x escape in "${src}"`);
    return { cp: parseInt(h, 16), at: at + 3 };
  }
  if (c === "u") {
    if (src[at + 1] === "{") {
      const close = src.indexOf("}", at);
      return { cp: parseInt(src.slice(at + 2, close), 16), at: close + 1 };
    }
    const h = src.slice(at + 1, at + 5);
    if (!/^[0-9a-fA-F]{4}$/.test(h)) throw new Error(`regex: bad \\u escape in "${src}"`);
    return { cp: parseInt(h, 16), at: at + 5 };
  }
  if (/[1-9]/.test(c) || c === "k") throw new Error("regex: backreferences are not supported");
  if (c === "b" || c === "B") throw new Error("regex: word boundaries are not supported");
  if (c === "p" || c === "P") throw new Error("regex: Unicode property escapes are not supported");
  return { cp: src.codePointAt(at)!, at: at + (src.codePointAt(at)! > 0xffff ? 2 : 1) };
}

/** A character-class node, adjusted for JSON string bodies if asked. */
export function classNode(cls: CharClass, json = false): Node {
  if (!json) return { kind: "chars", ranges: cls.ranges, negate: cls.negate };
  // Leave out what a JSON string cannot hold raw (negated: add to the excluded set).
  if (cls.negate) return { kind: "chars", ranges: normalize([...cls.ranges, ...JSON_SPECIAL]), negate: true };
  const allowed = complement(JSON_SPECIAL);
  const out: Ranges = [];
  for (const [a, z] of cls.ranges) for (const [b, y] of allowed) if (a <= y && z >= b) out.push([Math.max(a, b), Math.min(z, y)]);
  return { kind: "chars", ranges: out, negate: false };
}

/** A single character as a literal (its JSON escape inside a JSON string, if it needs one). */
export function charNode(cp: number, json = false): Node {
  if (json && (cp < 0x20 || cp === 0x22 || cp === 0x5c)) return lit(jsonEscape(cp));
  return lit(String.fromCodePoint(cp));
}

/** Parses a quantifier at `at`, if any: {min, max, at}. */
export function parseQuantifier(src: string, at: number): { min: number; max: number; at: number } | undefined {
  const c = src[at];
  let q: { min: number; max: number; at: number } | undefined;
  if (c === "*") q = { min: 0, max: Infinity, at: at + 1 };
  else if (c === "+") q = { min: 1, max: Infinity, at: at + 1 };
  else if (c === "?") q = { min: 0, max: 1, at: at + 1 };
  else if (c === "{") {
    const m = /^\{\s*(\d+)\s*(?:(,)\s*(\d*)\s*)?\}/.exec(src.slice(at));
    if (m) q = { min: +m[1]!, max: m[2] ? (m[3] ? +m[3] : Infinity) : +m[1]!, at: at + m[0].length };
  }
  if (q && q.max < q.min) throw new Error(`regex: quantifier {${q.min},${q.max}} is out of order`);
  return q;
}

/** Compiles a regular expression (see the file comment) to a grammar node matching the whole text. */
export function regexNode(pattern: string, opts: RegexOptions = {}): Node {
  const json = !!opts.json;
  let at = 0;
  const src = pattern;

  const alternation = (): Node => {
    const branches = [concat()];
    while (src[at] === "|") {
      at++;
      branches.push(concat());
    }
    return alt(branches);
  };

  const concat = (): Node => {
    const parts: Node[] = [];
    while (at < src.length && src[at] !== "|" && src[at] !== ")") {
      // Anchors: the whole text is matched anyway; only at the ends.
      if (src[at] === "^") {
        if (at !== 0) throw new Error(`regex: "^" is only supported at the start of "${src}"`);
        at++;
        continue;
      }
      if (src[at] === "$") {
        if (at !== src.length - 1) throw new Error(`regex: "$" is only supported at the end of "${src}"`);
        at++;
        continue;
      }
      let node = atom();
      const q = parseQuantifier(src, at);
      if (q) {
        at = q.at;
        if (src[at] === "?" || src[at] === "+") at++; // lazy or possessive: the same set of matches
        node = { kind: "repeat", node, min: q.min, max: q.max };
      }
      parts.push(node);
    }
    return parts.length ? seq(parts) : lit("");
  };

  const atom = (): Node => {
    const c = src[at]!;
    if (c === "(") {
      at++;
      if (src.startsWith("?:", at)) at += 2;
      else if (src.startsWith("?<", at) && src[at + 2] !== "=" && src[at + 2] !== "!") at = src.indexOf(">", at) + 1;
      else if (src[at] === "?") throw new Error("regex: lookaround is not supported");
      const node = alternation();
      if (src[at] !== ")") throw new Error(`regex: unbalanced parenthesis in "${src}"`);
      at++;
      return node;
    }
    if (c === "[") {
      const r = parseClass(src, at + 1);
      at = r.at;
      return classNode(r.cls, json);
    }
    if (c === ".") {
      at++;
      return classNode({ ranges: [[0x0a, 0x0a], [0x0d, 0x0d], [0x2028, 0x2029]], negate: true }, json);
    }
    if (c === "\\") {
      const e = parseEscape(src, at + 1);
      at = e.at;
      return e.ranges ? classNode({ ranges: e.ranges, negate: false }, json) : charNode(e.cp!, json);
    }
    if ("*+?{".includes(c) && parseQuantifier(src, at)) throw new Error(`regex: nothing to repeat in "${src}"`);
    const cp = src.codePointAt(at)!;
    at += cp > 0xffff ? 2 : 1;
    return charNode(cp, json);
  };

  const node = alternation();
  if (at !== src.length) throw new Error(`regex: unexpected "${src[at]}" in "${src}"`);
  return node;
}

/** Regexes for the JSON Schema string formats Kiln enforces. */
export const FORMATS: Record<string, string> = {
  date: "[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])",
  time: "([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\\.[0-9]{1,9})?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])",
  "date-time": "[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\\.[0-9]{1,9})?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])",
  email: "[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(\\.[A-Za-z0-9-]{1,63})*\\.[A-Za-z]{2,24}",
  uuid: "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
  ipv4: "(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(\\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}",
};
