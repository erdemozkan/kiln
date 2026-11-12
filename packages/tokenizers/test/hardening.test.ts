import { describe, expect, it } from "vitest";
import { GPT2_PATTERN, LLAMA3_PATTERN, QWEN2_PATTERN, PrecompiledCharsMap, Template, Tokenizer, assertLinearRegex, checkTokenId, parseSentencePieceModel, toJsRegex } from "../src";

describe("hostile chat templates fail instead of hanging", () => {
  const render = (src: string) => new Template(src).render({ messages: [] });
  it("rejects zero steps and oversized ranges", () => {
    expect(() => render("{% for i in range(1, 0, 0) %}x{% endfor %}")).toThrow(/non-zero step/);
    expect(() => render("{{ 'abc'[::0] }}")).toThrow(/slice step/);
    expect(() => render("{% for i in range(1000000000) %}x{% endfor %}")).toThrow(/range\(\) of more than/);
    expect(render("{% for i in range(3, 0, -1) %}{{ i }}{% endfor %}")).toBe("321");
    expect(render("{{ 'abcdef'[::2] }}{{ 'abc'[::-1] }}")).toBe("acecba");
  });

  it("bounds nested loops, output and string growth", () => {
    expect(() => render("{% for i in range(100000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}")).toThrow(/runs too long/);
    expect(() => render("{% set ns = namespace(s='xxxxxxxxxxxxxxxx') %}{% for i in range(40) %}{% set ns.s = ns.s ~ ns.s %}{% endfor %}")).toThrow(/too long/);
    expect(() => render("{{ 'x' * 100000000 }}")).toThrow(/too long/);
    expect(() => render("{% set ns = namespace(l=[1, 2]) %}{% for i in range(40) %}{% set ns.l = ns.l + ns.l %}{% endfor %}")).toThrow(/too long/);
  });

  it("does not read through a namespace's prototype, and trims long whitespace quickly", () => {
    expect(render("{% set ns = namespace(a=1) %}{{ ns.constructor is defined }}{{ ns.__proto__ is defined }}{{ ns.a }}")).toBe("FalseFalse1");
    const t0 = performance.now();
    expect(new Template(`a${" ".repeat(400000)}b{%- if true %}c{% endif %}`).render()).toMatch(/^a +bc$/);
    expect(performance.now() - t0).toBeLessThan(2000);
  });
});

describe("regexes from model repos", () => {
  it("accepts the patterns tokenizers use", () => {
    for (const p of [GPT2_PATTERN, LLAMA3_PATTERN, QWEN2_PATTERN, String.raw`[\r\n]`, String.raw`\s?\p{L}+|\p{N}{1,3}`, String.raw`(?:\d{3})|[^\s]+`, "q_proj", String.raw`layers\.\d+\.mlp\.down_proj`, "(q|v)_proj"]) {
      expect(() => assertLinearRegex(p, "test")).not.toThrow();
    }
    expect("a1 b".match(toJsRegex(String.raw`\p{N}| ?\p{L}+`))).toEqual(["a", "1", " b"]);
  });

  it("rejects shapes that backtrack exponentially", () => {
    for (const p of ["(a+)+$", "(a|aa)*b", "([a-z_.0-9]*)*!", "(x{2,})+", String.raw`(a)\1`, "((a+)b)*", "a".repeat(5000)]) {
      expect(() => assertLinearRegex(p, "test"), p.slice(0, 20)).toThrow(/not supported/);
    }
    expect(() => toJsRegex("(a+)+$")).toThrow(/not supported/);
  });
});

describe("tokenizer files with absurd sizes", () => {
  it("rejects token ids that would size arrays in the billions", () => {
    expect(checkTokenId(5)).toBe(5);
    for (const id of [-1, 1.5, 4e9, "7", NaN]) expect(() => checkTokenId(id)).toThrow(/out of range/);
    const json = { model: { type: "BPE", vocab: { a: 0 }, merges: [] }, pre_tokenizer: { type: "ByteLevel" } as never, added_tokens: [{ id: 4_000_000_000, content: "<x>", special: true }] };
    expect(() => Tokenizer.fromJSON(json)).toThrow(/out of range/);
  });

  it("rejects a SentencePiece model whose field runs past the file", () => {
    // Field 1 (pieces), length-delimited, claiming 2^35 bytes.
    expect(() => parseSentencePieceModel(Uint8Array.of(0x0a, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0x00))).toThrow(/past the end|truncated/);
    // A varint that never ends.
    expect(() => parseSentencePieceModel(Uint8Array.of(0x0a, 0xff, 0xff))).toThrow(/truncated/);
  });

  it("rejects a charsmap whose trie is larger than its data", () => {
    const blob = new Uint8Array(12);
    new DataView(blob.buffer).setUint32(0, 0xfffffffc, true);
    expect(() => new PrecompiledCharsMap(blob)).toThrow(/does not fit/);
  });
});
