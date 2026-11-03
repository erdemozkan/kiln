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
