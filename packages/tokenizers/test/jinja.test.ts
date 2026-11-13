import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { Template, TemplateError } from "../src/jinja";

interface Fixture {
  conversations: Record<string, unknown[]>;
  templates: { name: string; template: string; cases: { messages: string; vars: Record<string, unknown>; text?: string; error?: string }[] }[];
}

/**
 * Chat templates of permissively licensed models, rendered by Python jinja2 the way transformers does
 * (trim_blocks, lstrip_blocks, loopcontrols, a json.dumps `tojson`, strftime_now pinned to one date).
 * KILN_TEMPLATE_FIXTURES may name another fixture file of the same shape to check as well.
 */
const files = [fileURLToPath(new URL("./fixtures/chat-templates.json", import.meta.url)), ...(process.env.KILN_TEMPLATE_FIXTURES ? [process.env.KILN_TEMPLATE_FIXTURES] : [])];
const globals = { strftime_now: () => "26 Jul 2024" };

for (const file of files) {
  const fx = JSON.parse(readFileSync(file, "utf8")) as Fixture;
  describe(`chat templates in ${file.split("/").pop()}`, () => {
    for (const t of fx.templates) {
      it(`${t.name} renders like jinja2`, () => {
        const tpl = new Template(t.template, { globals });
        for (const c of t.cases) {
          const vars = { ...c.vars, messages: fx.conversations[c.messages] };
          const label = `${c.messages} ${JSON.stringify(c.vars)}`;
          if (c.error !== undefined) expect(() => tpl.render(vars), label).toThrow(new TemplateError(c.error));
          else expect(tpl.render(vars), label).toBe(c.text);
        }
      });
    }
  });
}

describe("jinja subset", () => {
  const r = (src: string, vars: Record<string, unknown> = {}) => new Template(src).render(vars);

  it("trims blocks and honours whitespace control", () => {
    expect(r("{% if true %}\n  a\n{% endif %}\nb")).toBe("  a\nb");
    expect(r("x  {%- if true -%}  y  {%- endif %}")).toBe("xy");
    expect(r("  {% if true %}a{% endif %}")).toBe("a");
    expect(r("x {% if true %}a{% endif %}")).toBe("x a");
    expect(r("{# note #}\nz")).toBe("z");
  });

  it("evaluates expressions with Python semantics", () => {
    expect(r("{{ 7 // 2 }} {{ -7 % 3 }} {{ 'a' ~ 1 ~ none }} {{ [1, 2] + [3] }}")).toBe("3 2 a1None [1, 2, 3]");
    expect(r("{{ 'hello world'.split(' ')[-1].upper() }}|{{ ' x '.strip() }}|{{ 'abc'[::-1] }}|{{ 'abcd'[1:3] }}")).toBe("WORLD|x|cba|bc");
    expect(r("{{ x.y is defined }} {{ x is none }} {{ 'b' in 'abc' }} {{ 3 not in [1, 2] }}", { x: null })).toBe("False True True True");
    expect(r("{{ d | tojson }} {{ d | tojson(indent=2) }}", { d: { a: [1, "é"], b: {} } })).toBe('{"a": [1, "é"], "b": {}} {\n  "a": [\n    1,\n    "é"\n  ],\n  "b": {}\n}');
    expect(r("{{ 'yes' if n > 1 else 'no' }}{{ ms | selectattr('role', 'equalto', 'user') | list | length }}", { n: 2, ms: [{ role: "user" }, { role: "x" }] })).toBe("yes1");
  });

  it("scopes loops, namespaces and macros like Jinja", () => {
    expect(r("{% set x = 1 %}{% for i in [1, 2] %}{% set x = i %}{% endfor %}{{ x }}")).toBe("1");
    expect(r("{% set ns = namespace(x=1) %}{% for i in [1, 2] %}{% set ns.x = i %}{% endfor %}{{ ns.x }}")).toBe("2");
    expect(r("{% for k, v in d.items() if v %}{{ loop.index }}{{ k }}{{ '' if loop.last else ',' }}{% endfor %}", { d: { a: 1, b: 0, c: 2 } })).toBe("1a,2c");
    expect(r("{% for i in [1, 2, 3] %}{% if i == 2 %}{% continue %}{% endif %}{{ i }}{% endfor %}{% for i in [] %}{% else %}empty{% endfor %}")).toBe("13empty");
    expect(r("{% macro tag(name, v='?') %}<{{ name }}={{ v }}>{% endmacro %}{{ tag('a') }}{{ tag('b', v=2) }}")).toBe("<a=?><b=2>");
  });

  it("reports raise_exception and unknown syntax as TemplateError", () => {
    expect(() => r("{{ raise_exception('nope') }}")).toThrow(new TemplateError("nope"));
    expect(() => new Template("{% if x %}")).toThrow(TemplateError);
    expect(() => r("{{ x | frobnicate }}")).toThrow(/unknown filter/);
  });
});

describe("chained comparisons and argument unpacking", () => {
  const render = (src: string, vars: Record<string, unknown> = {}) => new Template(src).render(vars);
  it("chains comparisons as Python does", () => {
    expect(render("{{ 1 < 2 < 3 }}{{ 1 < 3 < 2 }}{{ 3 > 2 > 1 }}{{ 3 > 2 > 5 }}")).toBe("TrueFalseTrueFalse");
    expect(render("{{ 0 <= n < 10 }}{{ 0 <= m < 10 }}", { n: 5, m: 10 })).toBe("TrueFalse");
    expect(render("{{ 1 == 1 != 2 }}{{ 1 < 2 == 2 }}{{ 2 in [1, 2] == true }}")).toBe("TrueTrueFalse");
    expect(render("{{ 1 < 2 not in [3] }}{{ 'a' in 'abc' in ['abc'] }}")).toBe("TrueTrue");
    // The middle operand is evaluated once, and nothing after a comparison that fails.
    expect(render("{% set ns = namespace(n=0) %}{% macro bump() %}{% set ns.n = ns.n + 1 %}5{% endmacro %}{{ 1 < bump() | int < 9 }}{{ ns.n }}")).toBe("True1");
    expect(render("{{ 2 < 1 < raise_exception('not reached') }}")).toBe("False");
    // One comparison is what it was.
    expect(render("{{ 1 < 2 }}{{ 'b' not in 'abc' }}{{ (1 < 2) == true }}")).toBe("TrueFalseTrue");
  });

  it("unpacks * and ** in calls and filters", () => {
    expect(render("{{ range(*bounds) | list }}", { bounds: [1, 4] })).toBe("[1, 2, 3]");
    expect(render("{% macro f(a, b, c='-') %}{{ a }}{{ c }}{{ b }}{% endmacro %}{{ f(*[1, 2]) }} {{ f(0, *[9]) }} {{ f(*[1, 2], **{'c': '+'}) }}")).toBe("1-2 0-9 1+2");
    expect(render("{{ ['a', 'b'] | join(*[', ']) }}{{ ns | tojson }}", { ns: null })).toBe("a, bnull");
    expect(render("{% set opts = {'indent': 2} %}{{ [1] | tojson(**opts) }}")).toBe("[\n  1\n]");
    expect(render("{{ dict(**{'a': 1}, b=2) | tojson }}")).toBe('{"a": 1, "b": 2}');
    expect(() => render("{{ range(*5) }}")).toThrow(/needs a list/);
    expect(() => render("{{ range(**[1]) }}")).toThrow(/needs a mapping/);
    expect(() => render("{{ [*a] }}", { a: [1] })).toThrow();
  });
});
