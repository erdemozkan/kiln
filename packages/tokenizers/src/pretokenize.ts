/** Pre-tokenizer steps from tokenizer.json, compiled to JS regexes. */

export const GPT2_PATTERN = String.raw`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`;
export const LLAMA3_PATTERN = String.raw`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;
export const QWEN2_PATTERN = String.raw`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;

/** Qwen 3.5: combining marks stay with their letters. */
export const QWEN35_PATTERN = String.raw`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;

/** Largest token id a tokenizer file may use (16M): ids index arrays sized by the largest one. */
export const MAX_TOKEN_ID = 2 ** 24;

/** Throws unless `id` is a usable token id (a hostile tokenizer.json could otherwise size arrays by an id of billions). */
export function checkTokenId(id: unknown): number {
  if (typeof id !== "number" || !Number.isInteger(id) || id < 0 || id >= MAX_TOKEN_ID) throw new Error(`tokenizer: token id ${String(id)} is out of range`);
  return id;
}

/** Most ways a pattern from a model repo may match one stretch of text (see assertLinearRegex). */
const MAX_REGEX_PATHS = 4096;

/**
 * Throws if a regex from a model repo has a shape that can take exponential or high-polynomial time:
 *  - a repeated group that itself holds a repetition or a choice ("(a+)+", "(a|aa)*", "(a?b?)*");
 *  - a backreference;
 *  - more than `maxLoops` open-ended repetitions one after another ("a*a*a*a*b" tries every way of
 *    sharing a run of a's between them);
 *  - so many choices in a row that their combinations pass MAX_REGEX_PATHS ("(a|a)(a|a)…", "a?a?…").
 * The patterns tokenizers and adapters really use are far inside these limits. What remains possible
 * is a low polynomial, the same as for the stock patterns.
 */
export function assertLinearRegex(source: string, what: string, maxLoops = 3): void {
  const bad = (why: string) => new Error(`${what}: the pattern ${JSON.stringify(source.length > 80 ? `${source.slice(0, 80)}…` : source)} is not supported (${why})`);
  if (source.length > 4096) throw bad("too long");
  /** paths: ways to match, an upper bound; loops: open-ended repetitions in a row. */
  type Cost = { paths: number; loops: number };
  let i = 0;

  /** The quantifier at `i`, consumed with its lazy mark; `max` is Infinity for an open-ended one. */
  const quantifier = (): { min: number; max: number } | null => {
    const c = source[i];
    let q: { min: number; max: number } | null = null;
    if (c === "*") q = { min: 0, max: Infinity };
    else if (c === "+") q = { min: 1, max: Infinity };
    else if (c === "?") q = { min: 0, max: 1 };
    else if (c === "{") {
      const m = /^\{(\d*)(,(\d*))?\}/.exec(source.slice(i, i + 24));
      if (!m || (m[1] === "" && !m[3])) return null;
      const min = Number(m[1] || 0);
      q = { min, max: m[2] === undefined ? min : m[3] === "" ? Infinity : Number(m[3]) };
      i += m[0].length - 1;
    }
    if (!q) return null;
    i++;
    if (source[i] === "?") i++;
    return q;
  };

  const sequence = (depth: number): Cost => {
    let paths = 1;
    let loops = 0;
    while (i < source.length && source[i] !== "|" && source[i] !== ")") {
      const c = source[i]!;
      let item: Cost = { paths: 1, loops: 0 };
      let group = false;
      if (c === "\\") {
        const next = source[i + 1] ?? "";
        if (/[1-9k]/.test(next)) throw bad("backreferences");
        i += 2;
        // \p{L}, \u{1F600}, \x{41}: the braces belong to the escape, they are not a quantifier.
        if (/[pPux]/.test(next) && source[i] === "{") {
          const end = source.indexOf("}", i);
          i = end < 0 ? source.length : end + 1;
        }
      } else if (c === "[") {
        for (i++; i < source.length && source[i] !== "]"; i++) if (source[i] === "\\") i++;
        i++;
      } else if (c === "(") {
        if (depth >= 32) throw bad("groups nested too deep");
        i++;
        // (?:…), (?i:…), (?=…), (?!…), (?<=…), (?<!…), (?<name>…)
        if (source[i] === "?") {
          const m = /^\?(?:<[=!]|[=!]|<[A-Za-z_]\w*>|[a-zA-Z-]*:)/.exec(source.slice(i, i + 64));
          if (!m) throw bad("an unknown group");
          i += m[0].length;
        }
        item = alternatives(depth + 1);
        if (source[i] !== ")") throw bad("unbalanced parentheses");
        i++;
        group = true;
      } else i++;
      const q = quantifier();
      if (q) {
        if (q.max === Infinity) {
          if (group && (item.paths > 1 || item.loops > 0)) throw bad("a repeated group holding a repetition or alternatives");
          item = { paths: item.paths, loops: item.loops + 1 };
        } else if (q.max > 1 || q.min !== q.max) {
          item = { paths: item.paths ** Math.max(1, q.max) * (q.max - q.min + 1), loops: item.loops * Math.max(1, q.max) };
        }
      }
      paths *= item.paths;
      loops += item.loops;
      if (!(paths <= MAX_REGEX_PATHS)) throw bad("too many alternatives in a row");
      if (loops > maxLoops) throw bad("too many open-ended repetitions in a row");
    }
    return { paths, loops };
  };

  const alternatives = (depth: number): Cost => {
    let paths = 0;
    let loops = 0;
    for (;;) {
      const s = sequence(depth);
      paths += s.paths;
      loops = Math.max(loops, s.loops);
      if (source[i] !== "|") break;
      i++;
    }
    if (paths > MAX_REGEX_PATHS) throw bad("too many alternatives in a row");
    return { paths, loops };
  };

  alternatives(0);
  if (i < source.length) throw bad("unbalanced parentheses");
}

const KNOWN_PATTERNS = new Set([GPT2_PATTERN, LLAMA3_PATTERN, QWEN2_PATTERN, QWEN35_PATTERN]);

/**
 * Rust-regex → JS: expands scoped case-insensitive groups `(?i:...)`, which JS engines before
 * ES2025 reject, into explicit `[aA]` classes.
 */
export function toJsRegex(pattern: string): RegExp {
  if (!KNOWN_PATTERNS.has(pattern)) assertLinearRegex(pattern, "tokenizer.json pre-tokenizer");
  let out = "";
  for (let i = 0; i < pattern.length; ) {
    if (pattern.startsWith("(?i:", i)) {
      let depth = 1;
      let j = i + 4;
      let inner = "";
      while (j < pattern.length && depth > 0) {
        const c = pattern[j]!;
        if (c === "\\") {
          inner += c + pattern[j + 1];
          j += 2;
          continue;
        }
        if (c === "(") depth++;
        if (c === ")" && --depth === 0) break;
        inner += /[a-z]/i.test(c) ? `[${c.toLowerCase()}${c.toUpperCase()}]` : c;
        j++;
      }
      out += `(?:${inner})`;
      i = j + 1;
    } else {
      out += pattern[i];
      i++;
    }
  }
  return new RegExp(out, "gu");
}

/** Splits so that matches and the gaps between them become separate pieces ("Isolated"). */
export function splitIsolated(text: string, re: RegExp): string[] {
  const out: string[] = [];
  let last = 0;
  re.lastIndex = 0;
  for (const m of text.matchAll(re)) {
    const at = m.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    if (m[0]) out.push(m[0]);
    last = at + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export type PreStep = (pieces: string[]) => string[];

interface RawPre {
  type: string;
  pretokenizers?: RawPre[];
  pattern?: { Regex?: string; String?: string };
  individual_digits?: boolean;
  use_regex?: boolean;
  add_prefix_space?: boolean;
}

/** Returns the split steps; byte-level mapping itself is applied by the tokenizer afterwards. */
export function compilePreTokenizer(raw: RawPre | null | undefined): PreStep[] {
  if (!raw) return [];
  switch (raw.type) {
    case "Sequence":
      return (raw.pretokenizers ?? []).flatMap(compilePreTokenizer);
    case "Split": {
      const re = raw.pattern?.Regex
        ? toJsRegex(raw.pattern.Regex)
        : new RegExp((raw.pattern?.String ?? "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gu");
      return [(ps) => ps.flatMap((p) => splitIsolated(p, re))];
    }
    case "Digits": {
      const re = raw.individual_digits ? /\p{N}/gu : /\p{N}+/gu;
      return [(ps) => ps.flatMap((p) => splitIsolated(p, re))];
    }
    case "ByteLevel": {
      const steps: PreStep[] = [];
      if (raw.add_prefix_space) steps.push((ps) => (ps.length && !ps[0]!.startsWith(" ") ? [" " + ps[0], ...ps.slice(1)] : ps));
      if (raw.use_regex !== false) {
        const re = toJsRegex(GPT2_PATTERN);
        steps.push((ps) => ps.flatMap((p) => splitIsolated(p, re)));
      }
      return steps;
    }
    default:
      throw new Error(`pre-tokenizer "${raw.type}" is not supported yet`);
  }
}
