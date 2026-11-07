import type { Node } from "./matcher";
import { charNode, classNode, lit, parseClass, parseEscape, parseQuantifier } from "./regex";

/**
 * GBNF, llama.cpp's grammar format, compiled to a grammar node:
 *
 *     root   ::= answer ("," ws answer)*
 *     answer ::= "yes" | "no" | [0-9]+
 *     ws     ::= [ \t]*
 *
 * Rules are `name ::= alternatives`; elements are string literals ("…", with \n \t \" \\ \xHH \uHHHH
 * escapes), classes `[…]` and `[^…]`, `.`, rule names, groups `(…)`, and the quantifiers `* + ?` and
 * `{m}` `{m,}` `{m,n}`. `#` starts a comment. As in llama.cpp, a rule ends at a newline unless inside
 * parentheses. Rules may refer to each other recursively, but not left-recursively (a rule that can
 * reach itself without matching a character first would never finish expanding).
 */
export function gbnfNode(grammar: string, root = "root"): Node {
  const src = grammar;
  let at = 0;
  const rules = new Map<string, Node & { nodes: Node[] }>();
  const defined = new Set<string>();
  /** A rule's node: a shell, filled when its definition is parsed, so references may come first. */
  const rule = (name: string) => {
    let n = rules.get(name);
    if (!n) rules.set(name, (n = { kind: "alt", nodes: [] }));
    return n;
  };

  const space = (newlines: boolean) => {
    for (;;) {
      const c = src[at];
      if (c === "#") while (at < src.length && src[at] !== "\n") at++;
      else if (c === " " || c === "\t" || c === "\r" || (newlines && c === "\n")) at++;
      else return;
    }
  };
  const name = () => {
    const m = /^[A-Za-z0-9_-]+/.exec(src.slice(at));
    if (!m) throw new Error(`grammar: expected a rule name at ${at}: "${src.slice(at, at + 20)}"`);
    at += m[0].length;
    return m[0];
  };

  const alternatives = (nested: boolean): Node => {
    const branches = [sequence(nested)];
    while (src[at] === "|") {
      at++;
      space(true);
      branches.push(sequence(nested));
    }
    return branches.length === 1 ? branches[0]! : { kind: "alt", nodes: branches };
  };

  const sequence = (nested: boolean): Node => {
    const parts: Node[] = [];
    for (;;) {
      space(nested);
      const c = src[at];
      if (c === undefined || c === "|" || c === ")" || c === "\n") break;
      let node: Node;
      if (c === '"') {
        at++;
        const chars: Node[] = [];
        while (src[at] !== '"') {
          if (at >= src.length) throw new Error("grammar: unterminated string");
          if (src[at] === "\\") {
            const e = parseEscape(src, at + 1);
            if (e.ranges) throw new Error("grammar: class escapes are not allowed in strings");
            at = e.at;
            chars.push(charNode(e.cp!));
          } else {
            const cp = src.codePointAt(at)!;
            at += cp > 0xffff ? 2 : 1;
            chars.push(charNode(cp));
          }
        }
        at++;
        const bytes = chars.flatMap((n) => [...(n as { bytes: Uint8Array }).bytes]);
        node = { kind: "lit", bytes: Uint8Array.from(bytes) };
      } else if (c === "[") {
        const r = parseClass(src, at + 1);
        at = r.at;
        node = classNode(r.cls);
      } else if (c === ".") {
        at++;
        node = { kind: "chars", ranges: [], negate: true };
      } else if (c === "(") {
        at++;
        space(true);
        node = alternatives(true);
        space(true);
        if (src[at] !== ")") throw new Error(`grammar: expected ")" at ${at}`);
        at++;
      } else node = rule(name());
      const q = parseQuantifier(src, at);
      if (q) {
        at = q.at;
        node = { kind: "repeat", node, min: q.min, max: q.max };
      }
      parts.push(node);
    }
    return parts.length === 0 ? lit("") : parts.length === 1 ? parts[0]! : { kind: "seq", nodes: parts };
  };

  for (;;) {
    space(true);
    if (at >= src.length) break;
    const n = name();
    space(false);
    if (!src.startsWith("::=", at)) throw new Error(`grammar: expected "::=" after rule "${n}"`);
    at += 3;
    space(true);
    if (defined.has(n)) throw new Error(`grammar: rule "${n}" is defined twice`);
    defined.add(n);
    rule(n).nodes.push(alternatives(false));
  }
  for (const n of rules.keys()) if (!defined.has(n)) throw new Error(`grammar: rule "${n}" is used but not defined`);
  if (!defined.has(root)) throw new Error(`grammar: no "${root}" rule`);
  return rule(root);
}
