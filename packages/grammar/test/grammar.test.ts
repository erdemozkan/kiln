import { describe, expect, it } from "vitest";
import { GrammarMatcher, TokenTrie, accepts, compileFormat, push, start, stepAll, type JsonSchema, type ResponseFormat, type Stack } from "../src";

const utf8 = new TextEncoder();

/** Whether `text` fully matches, and whether it is a prefix of some match. */
function match(f: ResponseFormat, text: string): { prefix: boolean; full: boolean } {
  let s: Stack[] = [push(null, start(compileFormat(f)))];
  for (const b of utf8.encode(text)) {
    s = stepAll(s, b);
    if (!s.length) return { prefix: false, full: false };
  }
  return { prefix: true, full: s.some(accepts) };
}
const json = (schema?: JsonSchema): ResponseFormat => ({ type: "json", schema, whitespace: "pretty" });
const full = (f: ResponseFormat, t: string) => match(f, t).full;

describe("JSON grammar", () => {
  it("accepts any JSON without a schema", () => {
    for (const t of ['{"a": [1, -2.5e3, true, null, "x\\n\\u00e9"], "b": {}}', "[]", '"hi"', "0", "-0.5", "false", '{\n  "k": [\n    1\n  ]\n}', '"日本"']) {
      expect(full(json(), t), t).toBe(true);
    }
    for (const t of ["{", "[1,]", "01", "1.", '{"a" 1}', "tru", '{"a":1,}', '"\u0001"', "nul", "+1", '"\\x"']) {
      expect(full(json(), t), t).toBe(false);
    }
    expect(match(json(), '{"a": [1, ').prefix).toBe(true);
    expect(match(json(), "[1]]").prefix).toBe(false);
  });

  it("caps whitespace runs and digits", () => {
    expect(match(json(), "[" + " ".repeat(40)).prefix).toBe(false);
    expect(match(json(), "1".repeat(17)).prefix).toBe(false);
  });

  it("allows only single spaces with compact whitespace", () => {
    // The default.
    const compact = (schema?: JsonSchema): ResponseFormat => ({ type: "json", schema });
    expect(full(compact(), '{"a": [1, 2], "b": {"c": null}}')).toBe(true);
    expect(full(compact(), '{"a":[1,2]}')).toBe(true);
    expect(match(compact(), '{\n').prefix).toBe(false);
    expect(match(compact(), '{"a":  1').prefix).toBe(false);
    const s: JsonSchema = { type: "object", properties: { xs: { type: "array", items: { type: "object", properties: { n: { type: "integer" } } } } } };
    expect(full(compact(s), '{"xs": [{"n": 1}, {"n": 2}]}')).toBe(true);
    expect(match(compact(s), '{"xs": [{"n": 1},\n').prefix).toBe(false);
    // Strings keep their content: whitespace rules apply between tokens only.
    expect(full(compact(), '"a\\n  b  c"')).toBe(true);
  });

  const person: JsonSchema = {
    type: "object",
    properties: {
      name: { type: "string", minLength: 1, maxLength: 5 },
      age: { type: "integer" },
      tags: { type: "array", items: { enum: ["a", "b"] }, maxItems: 2 },
      role: { anyOf: [{ const: "admin" }, { type: "null" }] },
    },
    required: ["name", "role"],
  };

  it("follows properties in order, with optional ones skippable", () => {
    const f = json(person);
    expect(full(f, '{"name": "Ann", "age": 30, "tags": ["a", "b"], "role": "admin"}')).toBe(true);
    expect(full(f, '{"name":"Ann","role":null}')).toBe(true);
    expect(full(f, '{"name":"Ann","tags":[],"role":null}')).toBe(true);
    // Required missing, wrong order, extra key, out-of-range values.
    expect(full(f, '{"name":"Ann"}')).toBe(false);
    expect(match(f, '{"role"').prefix).toBe(false);
    expect(match(f, '{"name":"A","x"').prefix).toBe(false);
    expect(match(f, '{"name":""').prefix).toBe(false);
    expect(match(f, '{"name":"Annabel').prefix).toBe(false);
    expect(match(f, '{"name":"A","age":1.5').prefix).toBe(false);
    expect(match(f, '{"name":"A","tags":["c"').prefix).toBe(false);
    expect(match(f, '{"name":"A","tags":["a","b",').prefix).toBe(false);
  });

  it("allows additional properties only when asked", () => {
    const s: JsonSchema = { type: "object", properties: { a: { type: "number" } }, additionalProperties: { type: "boolean" } };
    expect(full(json(s), '{"a":1,"zz":true}')).toBe(true);
    expect(full(json(s), '{"zz":true,"a":1}')).toBe(true);
    expect(full(json(s), '{"zz":1}')).toBe(false);
  });

  it("resolves recursive $refs", () => {
    const tree: JsonSchema = {
      $ref: "#/$defs/node",
      $defs: { node: { type: "object", properties: { v: { type: "integer" }, kids: { type: "array", items: { $ref: "#/$defs/node" } } }, required: ["v"] } },
    };
    expect(full(json(tree), '{"v":1,"kids":[{"v":2},{"v":3,"kids":[]}]}')).toBe(true);
    expect(full(json(tree), '{"v":1,"kids":[{}]}')).toBe(false);
  });

  it("rejects keywords it cannot enforce", () => {
    expect(() => compileFormat(json({ type: "array", uniqueItems: true }))).toThrow(/uniqueItems/);
    expect(() => compileFormat(json({ type: "string", pattern: "a", maxLength: 3 }))).toThrow(/pattern/);
  });

  it("enforces string patterns and formats, escaping what JSON must", () => {
    const zip = json({ type: "string", pattern: "^[0-9]{5}(-[0-9]{4})?$" });
    expect(full(zip, '"12345"')).toBe(true);
    expect(full(zip, '"12345-6789"')).toBe(true);
    expect(full(zip, '"1234"')).toBe(false);
    expect(match(zip, '"123a').prefix).toBe(false);
    // A literal quote or backslash in the pattern is written as JSON's escape; classes leave them out.
    const q = json({ type: "string", pattern: 'say "[a-z]+\\\\' });
    expect(full(q, '"say \\"hi\\\\"')).toBe(true);
    expect(full(json({ type: "string", pattern: "[^x]{2}" }), '"a\""')).toBe(false);
    const date = json({ type: "string", format: "date" });
    expect(full(date, '"2026-11-07"')).toBe(true);
    expect(full(date, '"2026-13-07"')).toBe(false);
    expect(full(json({ type: "string", format: "email" }), '"ada@example.org"')).toBe(true);
    expect(full(json({ type: "string", format: "email" }), '"ada@"')).toBe(false);
    expect(full(json({ type: "string", format: "uuid" }), '"123e4567-e89b-12d3-a456-426614174000"')).toBe(true);
    expect(full(json({ type: "string", format: "ipv4" }), '"192.168.0.256"')).toBe(false);
    // Unknown formats stay annotations.
    expect(full(json({ type: "string", format: "uri" }), '"anything"')).toBe(true);
  });

  it("keeps numbers within minimum and maximum, refusing digits that cannot lead there", () => {
    const age = json({ type: "integer", minimum: 18, maximum: 120 });
    for (const [t, ok] of [["18", true], ["120", true], ["65", true], ["17", false], ["121", false], ["-5", false], ["1", false]] as const) expect(full(age, t), t).toBe(ok);
    expect(match(age, "1").prefix).toBe(true); // 18 or 1xx still possible
    expect(match(age, "13").prefix).toBe(false); // 13 too small, 13x too large
    expect(match(age, "-").prefix).toBe(false);
    expect(match(age, "2").prefix).toBe(true);
    expect(match(age, "20").prefix).toBe(true);
    expect(match(age, "200").prefix).toBe(false);
    const p = json({ type: "number", exclusiveMinimum: 0, maximum: 1 });
    for (const [t, ok] of [["0.5", true], ["1", true], ["0", false], ["1.5", false], ["0.0001", true], ["-0.1", false], ["1.0", true], ["1.01", false]] as const) expect(full(p, t), t).toBe(ok);
    expect(match(p, "0.").prefix).toBe(true);
    expect(match(p, "2").prefix).toBe(false);
    expect(match(p, "1e").prefix).toBe(false);
    // Draft 4 booleans.
    expect(full(json({ type: "integer", minimum: 0, exclusiveMinimum: true }), "0")).toBe(false);
    const neg = json({ type: "integer", maximum: -10 });
    expect(full(neg, "-10")).toBe(true);
    expect(full(neg, "-9")).toBe(false);
    expect(full(neg, "-1000")).toBe(true);
    expect(match(neg, "5").prefix).toBe(false);
  });

  it("enforces multipleOf, with or without bounds", () => {
    const even = json({ type: "integer", multipleOf: 2, minimum: 0, maximum: 9 });
    for (const [t, ok] of [["0", true], ["4", true], ["8", true], ["3", false], ["10", false]] as const) expect(full(even, t), t).toBe(ok);
    expect(match(even, "9").prefix).toBe(false); // no even number starts with 9 below 10
    const tens = json({ type: "integer", multipleOf: 10 });
    expect(full(tens, "120")).toBe(true);
    expect(full(tens, "125")).toBe(false);
    expect(match(tens, "12").prefix).toBe(true);
    expect(full(tens, "1e2")).toBe(false); // bounded numbers have no exponent
    const step = json({ type: "number", multipleOf: 0.1, minimum: 0, maximum: 1 });
    for (const [t, ok] of [["0.3", true], ["0.7", true], ["1", true], ["0.30", true], ["0.35", false], ["1.1", false]] as const) expect(full(step, t), t).toBe(ok);
    expect(match(step, "0.35").prefix).toBe(false);
    const half = json({ type: "number", multipleOf: 0.5, exclusiveMinimum: 1 });
    expect(full(half, "1")).toBe(false);
    expect(full(half, "1.5")).toBe(true);
    expect(full(half, "2.25")).toBe(false);
    expect(() => match(json({ type: "number", multipleOf: 0 }), "1")).toThrow(/positive/);
  });

  it("matches regular expressions", () => {
    const r = (pattern: string): ResponseFormat => ({ type: "regex", pattern });
    expect(full(r("(yes|no)!?"), "yes!")).toBe(true);
    expect(full(r("(yes|no)!?"), "maybe")).toBe(false);
    expect(full(r("\\d{3}-\\d{4}"), "555-1234")).toBe(true);
    expect(full(r("\\d{3}-\\d{4}"), "55-1234")).toBe(false);
    expect(full(r("[A-Z][a-z]*( [A-Z][a-z]*)*"), "Ada Lovelace")).toBe(true);
    expect(full(r("[A-Z][a-z]*( [A-Z][a-z]*)*"), "ada")).toBe(false);
    expect(full(r("[^\\s]+"), "日本語")).toBe(true);
    expect(full(r("[a-c]+"), "日")).toBe(false);
    expect(match(r("[a-c]+"), "\u00e9").prefix).toBe(false); // refused at the lead byte
    expect(full(r("a{2,3}"), "aaaa")).toBe(false);
    expect(full(r("(a*)*b"), "aab")).toBe(true);
    expect(full(r("x.y"), "x\ny")).toBe(false);
    expect(full(r("\\u00e9+"), "\u00e9\u00e9")).toBe(true);
    for (const bad of ["(a", "a)", "\\1", "a(?=b)", "\\bword", "*a"]) expect(() => compileFormat(r(bad)), bad).toThrow();
  });

  it("matches GBNF grammars, recursive rules included", () => {
    const g = (grammar: string): ResponseFormat => ({ type: "grammar", grammar });
    const list = g(`
      # A list of answers.
      root   ::= answer ("," ws answer)*
      answer ::= "yes" | "no" | [0-9]+
      ws     ::= [ \\t]*
    `);
    expect(full(list, "yes, 42,no")).toBe(true);
    expect(full(list, "yes,maybe")).toBe(false);
    const arith = g(`root ::= expr
expr ::= term (("+" | "-") term)*
term ::= num | "(" expr ")"
num  ::= [0-9]{1,3}`);
    expect(full(arith, "(1+2)-(30+(4))")).toBe(true);
    expect(full(arith, "(1+2")).toBe(false);
    expect(full(arith, "1234")).toBe(false);
    expect(match(arith, "((((1").prefix).toBe(true);
    const nested = g(`root ::= "{" (
        "a" | "b"
      )+ "}"`);
    expect(full(nested, "{abba}")).toBe(true);
    expect(full(g(`root ::= "\\x41" . "\\n"`), "A\u00e9\n")).toBe(true);
    expect(() => compileFormat(g("root ::= missing"))).toThrow(/not defined/);
    expect(() => compileFormat(g("other ::= \"a\""))).toThrow(/root/);
  });

  it("matches one of a set of strings", () => {
    const f: ResponseFormat = { type: "choice", choices: ["yes", "yes please", "no"] };
    expect(full(f, "yes")).toBe(true);
    expect(full(f, "yes please")).toBe(true);
    expect(full(f, "ye")).toBe(false);
    expect(match(f, "yes p").prefix).toBe(true);
    expect(match(f, "maybe").prefix).toBe(false);
  });
});

describe("tool call grammar", () => {
  const weather = { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" }, unit: { enum: ["celsius", "fahrenheit"] } }, required: ["city"] } } as const;
  const time = { name: "get_time" };
  const f: ResponseFormat = { type: "tool_call", tools: [weather as never, time] };

  it("accepts one call to a listed tool with schema-valid arguments", () => {
    for (const t of [
      '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
      '<tool_call>\n{"name":"get_weather","arguments":{"city":"Rome","unit":"celsius"}}\n</tool_call>',
      '<tool_call>\n{"name": "get_time", "arguments": {"anything": [1]}}\n</tool_call>',
    ]) {
      expect(full(f, t), t).toBe(true);
    }
  });

  it("rejects other names, bad arguments, and anything around the call", () => {
    for (const t of [
      '<tool_call>\n{"name": "get_wether", "arguments": {"city": "Paris"}}\n</tool_call>',
      '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>',
      '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "unit": "kelvin"}}\n</tool_call>',
      '<tool_call>\n{"name": "get_weather", "arguments": {"city": 3}}\n</tool_call>',
      '<tool_call>\n{"arguments": {"city": "Paris"}, "name": "get_weather"}\n</tool_call>',
      'Sure! <tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>',
      '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call> done',
    ]) {
      expect(match(f, t).full, t).toBe(false);
    }
    expect(match(f, '<tool_call>\n{"name": "get_').prefix).toBe(true);
    expect(match(f, '<tool_call>\n{"name": "get_t').prefix).toBe(true);
    expect(match(f, '<tool_call>\n{"name": "set_').prefix).toBe(false);
    expect(() => compileFormat({ type: "tool_call", tools: [] })).toThrow(/at least one tool/);
  });
});

describe("token masks", () => {
  // Single bytes plus some multi-byte tokens, a stop token (no bytes) and a padded vocabulary.
  const vocab = [...Array.from({ length: 256 }, (_, b) => String.fromCharCode(b)), '{"', '":', "name", 'name":', ' "', '",', "}", "true", "tr", "Ann", '"}'];
  const tokens: (Uint8Array | undefined)[] = vocab.map((t) => Uint8Array.from(t, (c) => c.charCodeAt(0)));
  const STOP = tokens.length;
  tokens.push(undefined);
  const V = tokens.length + 5;
  const trie = new TokenTrie(tokens);
  const id = (t: string) => vocab.indexOf(t);
  const allowed = (m: Uint32Array) => [...Array(V).keys()].filter((i) => (m[i >>> 5]! >>> (i & 31)) & 1);

  it("allows exactly the tokens that keep the text a prefix of a match", () => {
    const schema: JsonSchema = { type: "object", properties: { name: { type: "string" }, ok: { type: "boolean" } }, required: ["name", "ok"] };
    const g = new GrammarMatcher(compileFormat(json(schema)), trie, tokens, V, [STOP]);
    expect(allowed(g.mask()).map((i) => vocab[i])).toEqual(["{", '{"']);
    expect(g.accept(id('{"'))).toBe(true);
    const next = allowed(g.mask()).map((i) => vocab[i]);
    expect(next).toEqual(expect.arrayContaining(["n", "name", 'name":']));
    expect(next).not.toContain("o");
    for (const t of ['name":', ' "', "Ann", '",']) expect(g.accept(id(t)), t).toBe(true);
    expect(g.accept(id("x"))).toBe(false);
    for (const t of ['"', "o", "k", '":', "tr"]) expect(g.accept(id(t)), t).toBe(true);
    expect(allowed(g.mask()).map((i) => vocab[i])).toEqual(["u"]);
    for (const t of ["u", "e", "}"]) expect(g.accept(id(t))).toBe(true);
    expect(g.finished).toBe(true);
    expect(allowed(g.mask())).toEqual([STOP]);
  });

  it("brute force agrees with the trie walk", () => {
    const g = new GrammarMatcher(compileFormat(json()), trie, tokens, V, [STOP]);
    for (const t of ["[", "1", ","]) g.accept(id(t));
    const want: number[] = [];
    tokens.forEach((bytes, i) => {
      if (!bytes) return;
      const probe = new GrammarMatcher(compileFormat(json()), trie, tokens, V, [STOP]);
      for (const t of ["[", "1", ","]) probe.accept(id(t));
      if (probe.accept(i)) want.push(i);
    });
    expect(allowed(g.mask())).toEqual(want);
  });

  it("the string-body shortcut gives the same masks as the full walk", () => {
    // Tokens that end, escape or break out of a string, alongside plain text and UTF-8 bytes.
    const extra = ['"', '",', '", "', '"}', '"]', "\\", '\\"', "\\n", "\\u00", "a\"b", 'x"', "\n", "\t", "é", "日本", "hello world", ' "', ":", '":"'];
    // Ids 0-255 stay single raw bytes, so feeding a text byte by byte is exact.
    const toks: (Uint8Array | undefined)[] = [...tokens.slice(0, -1), ...extra.map((t) => utf8.encode(t))];
    toks.push(undefined);
    const W = toks.length + 3;
    const tr = new TokenTrie(toks);
    const schema: JsonSchema = {
      type: "object",
      properties: {
        name: { type: "string" },
        short: { type: "string", maxLength: 3 },
        kind: { enum: ["a", "ab"] },
        tags: { type: "array", items: { type: "string" } },
        any: {},
      },
      required: ["name", "short", "kind", "tags", "any"],
    };
    const texts: [ResponseFormat, string][] = [
      [json(schema), '{"name": "Ann \\"Lee\\" é 日本", "short": "ab", "kind": "ab", "tags": ["x", "y z"], "any": {"q": ["r", "s\\n"]}}'],
      [{ type: "json" }, '["a", {"b": "c\\u00e9"}, "d"]'],
      [{ type: "json", schema: { anyOf: [{ type: "string" }, { type: "string", maxLength: 2 }] } }, '"abc"'],
    ];
    let checked = 0;
    for (const [f, text] of texts) {
      checked = 0;
      const fast = new GrammarMatcher(compileFormat(f), tr, toks, W, [toks.length - 1]);
      const slow = new GrammarMatcher(compileFormat(f), tr, toks, W, [toks.length - 1]);
      // Byte by byte, so every string phase and every alternative is visited.
      for (const b of utf8.encode(text)) {
        GrammarMatcher.fastStrings = false;
        const want = slow.mask().slice();
        GrammarMatcher.fastStrings = true;
        expect(fast.mask(), `${JSON.stringify(text)} after ${checked} bytes`).toEqual(want);
        expect(fast.accept(b)).toBe(true);
        expect(slow.accept(b)).toBe(true);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(4);
  });
});
