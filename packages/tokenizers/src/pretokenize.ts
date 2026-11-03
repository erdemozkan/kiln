/** Pre-tokenizer steps from tokenizer.json, compiled to JS regexes. */

export const GPT2_PATTERN = String.raw`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`;
export const LLAMA3_PATTERN = String.raw`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;
export const QWEN2_PATTERN = String.raw`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;

/**
 * Rust-regex → JS: expands scoped case-insensitive groups `(?i:...)`, which JS engines before
 * ES2025 reject, into explicit `[aA]` classes.
 */
export function toJsRegex(pattern: string): RegExp {
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
