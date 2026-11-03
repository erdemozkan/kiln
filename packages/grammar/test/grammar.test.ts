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
const json = (schema?: JsonSchema): ResponseFormat => ({ type: "json", schema });
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
    expect(() => compileFormat(json({ type: "string", pattern: "^a" }))).toThrow(/pattern/);
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
});
