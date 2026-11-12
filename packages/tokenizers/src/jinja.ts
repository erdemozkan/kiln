/**
 * A Jinja2 subset for Hugging Face chat templates, with the environment transformers uses:
 * `trim_blocks` and `lstrip_blocks` on, `loopcontrols`, no autoescape.
 *
 * Supported: `{{ }}` output, `{% if/elif/else %}`, `{% for %}` (with `loop.*`, an `if` filter, `else`,
 * `break`/`continue`), `{% set %}` (names, tuples, namespace attributes and blocks), `{% macro %}`,
 * `{% generation %}`, comments and `-` whitespace control. Expressions cover literals, lists, dicts,
 * attribute and item access, slices, calls, filters, tests, arithmetic, comparisons, `in`, `~`,
 * boolean logic and `a if b else c`, with Python semantics where templates depend on them.
 */

export class TemplateError extends Error {}

// Chat templates come with the model repo, so a render is bounded: a hostile (or broken) template
// fails with a TemplateError instead of hanging the page or exhausting memory. Real templates stay
// orders of magnitude below these.
/** Longest list `range()` builds (Jinja's sandbox uses the same limit). */
const MAX_RANGE = 100_000;
/** Longest string an expression may build, and the most a render may output, in UTF-16 units. */
const MAX_STRING = 32 * 2 ** 20;
/** Statements and loop iterations one render may execute. */
const MAX_STEPS = 2_000_000;

function bounded(s: string): string {
  if (s.length > MAX_STRING) throw new TemplateError("the template builds a string that is too long");
  return s;
}

/** A missing variable or attribute. Renders as "", is falsy, and stays undefined when accessed. */
class Undef {
  constructor(readonly name: string) {}
  toString() {
    return "";
  }
}
const isUndef = (v: unknown): v is Undef => v instanceof Undef;

/** A `namespace()` object: the one kind of value `{% set ns.attr = … %}` may change. */
class Namespace {
  constructor(readonly vars: Record<string, unknown>) {}
}

type Fn = (args: unknown[], kwargs: Record<string, unknown>) => unknown;
class Callable {
  constructor(readonly fn: Fn) {}
}

// ── Lexing ───────────────────────────────────────────────────────────────────────────────────────

type Seg = { kind: "text"; text: string } | { kind: "out" | "stmt"; src: string; pos: number };

/** Splits the source into text and tags, applying `-` trimming, trim_blocks and lstrip_blocks. */
function segment(src: string): Seg[] {
  type Raw = { kind: "text"; text: string } | { kind: "out" | "stmt" | "comment"; src: string; pos: number; lt: string; rt: string };
  const raw: Raw[] = [];
  let i = 0;
  while (i < src.length) {
    const open = src.slice(i).search(/\{[{%#]/);
    if (open < 0) {
      raw.push({ kind: "text", text: src.slice(i) });
      break;
    }
    if (open > 0) raw.push({ kind: "text", text: src.slice(i, i + open) });
    const start = i + open;
    const type = src[start + 1]!;
    const close = type === "{" ? "}}" : type === "%" ? "%}" : "#}";
    let j = start + 2;
    if (type === "#") {
      j = src.indexOf("#}", j);
    } else {
      // Skip string literals, which may contain the closing delimiter.
      while (j < src.length && !src.startsWith(close, j)) {
        const c = src[j]!;
        if (c === "'" || c === '"') {
          j++;
          while (j < src.length && src[j] !== c) j += src[j] === "\\" ? 2 : 1;
        }
        j++;
      }
    }
    if (j < 0 || j >= src.length) throw new TemplateError(`unclosed tag at offset ${start}`);
    let inner = src.slice(start + 2, j);
    let lt = "";
    let rt = "";
    if (inner.startsWith("-") || inner.startsWith("+")) {
      lt = inner[0]!;
      inner = inner.slice(1);
    }
    if (inner.endsWith("-") || inner.endsWith("+")) {
      rt = inner[inner.length - 1]!;
      inner = inner.slice(0, -1);
    }
    raw.push({ kind: type === "{" ? "out" : type === "%" ? "stmt" : "comment", src: inner, pos: start, lt, rt });
    i = j + 2;
  }
  // Whitespace control.
  for (let k = 0; k < raw.length; k++) {
    const r = raw[k]!;
    if (r.kind === "text") continue;
    const block = r.kind !== "out";
    const prev = raw[k - 1];
    if (prev?.kind === "text") {
      if (r.lt === "-") prev.text = prev.text.trimEnd();
      else if (block && r.lt !== "+") {
        // lstrip_blocks: drop spaces and tabs between the start of the line and the tag. Text without
        // a newline starts a line only at the very start of the template.
        const m = /(^|\n)[ \t]*$/.exec(prev.text);
        if (m && (m[1] || k === 1)) prev.text = prev.text.slice(0, m.index + m[1]!.length);
      }
    }
    const next = raw[k + 1];
    if (next?.kind === "text") {
      if (r.rt === "-") next.text = next.text.replace(/^\s+/, "");
      else if (block && r.rt !== "+") next.text = next.text.replace(/^\r?\n/, "");
    }
  }
  return raw.flatMap((r): Seg[] => (r.kind === "comment" || (r.kind === "text" && r.text === "") ? [] : [r.kind === "text" ? r : { kind: r.kind, src: r.src, pos: r.pos }]));
}

type Tok = { t: "name" | "str" | "num" | "op"; v: string; n?: number };

const OPS = ["**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">", "=", "(", ")", "[", "]", "{", "}", ".", ":", ",", "|"];

function lex(src: string, pos: number): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i]!;
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    if (c === "'" || c === '"') {
      let s = "";
      i++;
      while (i < src.length && src[i] !== c) {
        if (src[i] === "\\") {
          const e = src[++i]!;
          s += e === "n" ? "\n" : e === "t" ? "\t" : e === "r" ? "\r" : e === "0" ? "\0" : e;
        } else s += src[i];
        i++;
      }
      if (i >= src.length) throw new TemplateError(`unterminated string near offset ${pos}`);
      i++;
      out.push({ t: "str", v: s });
      continue;
    }
    const num = /^\d+(\.\d+)?([eE][+-]?\d+)?/.exec(src.slice(i));
    if (num) {
      out.push({ t: "num", v: num[0], n: Number(num[0]) });
      i += num[0].length;
      continue;
    }
    const name = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
    if (name) {
      out.push({ t: "name", v: name[0] });
      i += name[0].length;
      continue;
    }
    const op = OPS.find((o) => src.startsWith(o, i));
    if (!op) throw new TemplateError(`unexpected "${c}" near offset ${pos}`);
    out.push({ t: "op", v: op });
    i += op.length;
  }
  return out;
}

// ── Parsing ──────────────────────────────────────────────────────────────────────────────────────

type Expr =
  | { k: "lit"; v: unknown }
  | { k: "var"; name: string }
  | { k: "list"; items: Expr[] }
  | { k: "dict"; items: [Expr, Expr][] }
  | { k: "attr"; obj: Expr; name: string }
  | { k: "item"; obj: Expr; key: Expr }
  | { k: "slice"; obj: Expr; start?: Expr; stop?: Expr; step?: Expr }
  | { k: "call"; fn: Expr; args: Expr[]; kwargs: [string, Expr][] }
  | { k: "filter"; value: Expr; name: string; args: Expr[]; kwargs: [string, Expr][] }
  | { k: "test"; value: Expr; name: string; args: Expr[]; negate: boolean }
  | { k: "un"; op: string; a: Expr }
  | { k: "bin"; op: string; a: Expr; b: Expr }
  | { k: "cond"; test: Expr; then: Expr; else?: Expr };

class ExprParser {
  private i = 0;
  constructor(private readonly toks: Tok[]) {}

  get done() {
    return this.i >= this.toks.length;
  }
  peek(o = 0) {
    return this.toks[this.i + o];
  }
  isOp(v: string, o = 0) {
    const t = this.peek(o);
    return t?.t === "op" && t.v === v;
  }
  isName(v: string, o = 0) {
    const t = this.peek(o);
    return t?.t === "name" && t.v === v;
  }
  next() {
    const t = this.toks[this.i++];
    if (!t) throw new TemplateError("unexpected end of expression");
    return t;
  }
  expectOp(v: string) {
    if (!this.isOp(v)) throw new TemplateError(`expected "${v}", got "${this.peek()?.v ?? "end"}"`);
    this.i++;
  }
  expectName(v?: string): string {
    const t = this.next();
    if (t.t !== "name" || (v && t.v !== v)) throw new TemplateError(`expected ${v ?? "a name"}, got "${t.v}"`);
    return t.v;
  }

  /** Assignment targets: `a`, `a, b`, `(a, b)` or `ns.attr`. */
  targets(): Expr {
    const one = (): Expr => {
      if (this.isOp("(")) {
        this.i++;
        const t = this.targets();
        this.expectOp(")");
        return t;
      }
      let e: Expr = { k: "var", name: this.expectName() };
      while (this.isOp(".")) {
        this.i++;
        e = { k: "attr", obj: e, name: this.expectName() };
      }
      return e;
    };
    const first = one();
    if (!this.isOp(",")) return first;
    const items = [first];
    while (this.isOp(",")) {
      this.i++;
      items.push(one());
    }
    return { k: "list", items };
  }

  /** An expression, or a bare tuple `a, b`. */
  tuple(): Expr {
    const first = this.expr();
    if (!this.isOp(",")) return first;
    const items = [first];
    while (this.isOp(",")) {
      this.i++;
      if (this.done || this.isOp(")") || this.isName("in") || this.isOp("=")) break;
      items.push(this.expr());
    }
    return { k: "list", items };
  }

  expr(): Expr {
    const e = this.or();
    if (this.isName("if")) {
      this.i++;
      const test = this.or();
      let other: Expr | undefined;
      if (this.isName("else")) {
        this.i++;
        other = this.expr();
      }
      return { k: "cond", test, then: e, else: other };
    }
    return e;
  }
  /** Without the trailing `if … else …`, so `for x in y if cond` keeps its filter. */
  or(): Expr {
    let a = this.and();
    while (this.isName("or")) {
      this.i++;
      a = { k: "bin", op: "or", a, b: this.and() };
    }
    return a;
  }
  private and(): Expr {
    let a = this.not();
    while (this.isName("and")) {
      this.i++;
      a = { k: "bin", op: "and", a, b: this.not() };
    }
    return a;
  }
  private not(): Expr {
    if (this.isName("not")) {
      this.i++;
      return { k: "un", op: "not", a: this.not() };
    }
    return this.compare();
  }
  private compare(): Expr {
    let a = this.add();
    for (;;) {
      const t = this.peek();
      if (t?.t === "op" && ["==", "!=", "<", ">", "<=", ">="].includes(t.v)) {
        this.i++;
        a = { k: "bin", op: t.v, a, b: this.add() };
      } else if (this.isName("in")) {
        this.i++;
        a = { k: "bin", op: "in", a, b: this.add() };
      } else if (this.isName("not") && this.isName("in", 1)) {
        this.i += 2;
        a = { k: "un", op: "not", a: { k: "bin", op: "in", a, b: this.add() } };
      } else return a;
    }
  }
  private add(): Expr {
    let a = this.concat();
    while (this.isOp("+") || this.isOp("-")) {
      const op = this.next().v;
      a = { k: "bin", op, a, b: this.concat() };
    }
    return a;
  }
  private concat(): Expr {
    let a = this.mul();
    while (this.isOp("~")) {
      this.i++;
      a = { k: "bin", op: "~", a, b: this.mul() };
    }
    return a;
  }
  private mul(): Expr {
    let a = this.pow();
    while (this.isOp("*") || this.isOp("/") || this.isOp("//") || this.isOp("%")) {
      const op = this.next().v;
      a = { k: "bin", op, a, b: this.pow() };
    }
    return a;
  }
  private pow(): Expr {
    let a = this.unary();
    while (this.isOp("**")) {
      this.i++;
      a = { k: "bin", op: "**", a, b: this.unary() };
    }
    return a;
  }
  private unary(): Expr {
    if (this.isOp("-") || this.isOp("+")) {
      const op = this.next().v;
      return this.filters({ k: "un", op, a: this.unary() });
    }
    return this.filters(this.postfix(this.primary()));
  }
  /** `|filter` and `is test`, which bind tighter than any operator. */
  private filters(e: Expr): Expr {
    for (;;) {
      if (this.isOp("|")) {
        this.i++;
        const name = this.expectName();
        const { args, kwargs } = this.isOp("(") ? this.callArgs() : { args: [], kwargs: [] };
        e = { k: "filter", value: e, name, args, kwargs };
      } else if (this.isName("is")) {
        this.i++;
        let negate = false;
        if (this.isName("not")) {
          this.i++;
          negate = true;
        }
        const name = this.expectName();
        let args: Expr[] = [];
        if (this.isOp("(")) args = this.callArgs().args;
        else if (!this.done && !this.atBoundary()) args = [this.postfix(this.primary())];
        e = { k: "test", value: e, name, args, negate };
      } else return e;
    }
  }
  /** Tokens that end a test without an argument (`x is defined and …`). */
  private atBoundary(): boolean {
    const t = this.peek()!;
    if (t.t === "op") return [")", "]", "}", ",", ":", "|", "==", "!=", "+", "-", "*", "/", "%", "~", "<", ">", "<=", ">=", "="].includes(t.v);
    return t.t === "name" && ["and", "or", "if", "else", "in", "not", "is", "recursive"].includes(t.v);
  }
  private primary(): Expr {
    const t = this.next();
    if (t.t === "str") {
      let v = t.v;
      while (this.peek()?.t === "str") v += this.next().v;
      return { k: "lit", v };
    }
    if (t.t === "num") return { k: "lit", v: t.n };
    if (t.t === "name") {
      switch (t.v) {
        case "true":
        case "True":
          return { k: "lit", v: true };
        case "false":
        case "False":
          return { k: "lit", v: false };
        case "none":
        case "None":
          return { k: "lit", v: null };
      }
      return { k: "var", name: t.v };
    }
    if (t.v === "(") {
      if (this.isOp(")")) {
        this.i++;
        return { k: "list", items: [] };
      }
      const e = this.tuple();
      this.expectOp(")");
      return e;
    }
    if (t.v === "[") {
      const items: Expr[] = [];
      while (!this.isOp("]")) {
        items.push(this.expr());
        if (!this.isOp("]")) this.expectOp(",");
      }
      this.i++;
      return { k: "list", items };
    }
    if (t.v === "{") {
      const items: [Expr, Expr][] = [];
      while (!this.isOp("}")) {
        const key = this.expr();
        this.expectOp(":");
        items.push([key, this.expr()]);
        if (!this.isOp("}")) this.expectOp(",");
      }
      this.i++;
      return { k: "dict", items };
    }
    throw new TemplateError(`unexpected "${t.v}"`);
  }
  private postfix(e: Expr): Expr {
    for (;;) {
      if (this.isOp(".")) {
        this.i++;
        const t = this.next();
        e = t.t === "num" ? { k: "item", obj: e, key: { k: "lit", v: t.n } } : { k: "attr", obj: e, name: t.v };
      } else if (this.isOp("[")) {
        this.i++;
        const part = () => (this.isOp(":") || this.isOp("]") ? undefined : this.expr());
        const start = part();
        if (this.isOp(":")) {
          this.i++;
          const stop = part();
          let step: Expr | undefined;
          if (this.isOp(":")) {
            this.i++;
            step = part();
          }
          e = { k: "slice", obj: e, start, stop, step };
        } else e = { k: "item", obj: e, key: start! };
        this.expectOp("]");
      } else if (this.isOp("(")) {
        const { args, kwargs } = this.callArgs();
        e = { k: "call", fn: e, args, kwargs };
      } else return e;
    }
  }
  private callArgs(): { args: Expr[]; kwargs: [string, Expr][] } {
    this.expectOp("(");
    const args: Expr[] = [];
    const kwargs: [string, Expr][] = [];
    while (!this.isOp(")")) {
      if (this.peek()?.t === "name" && this.isOp("=", 1)) {
        const name = this.next().v;
        this.i++;
        kwargs.push([name, this.expr()]);
      } else args.push(this.expr());
      if (!this.isOp(")")) this.expectOp(",");
    }
    this.i++;
    return { args, kwargs };
  }
}

type Node =
  | { k: "text"; text: string }
  | { k: "out"; e: Expr }
  | { k: "if"; branches: { test: Expr; body: Node[] }[]; else: Node[] }
  | { k: "for"; target: Expr; iter: Expr; filter?: Expr; body: Node[]; else: Node[] }
  | { k: "set"; target: Expr; value: Expr }
  | { k: "setBlock"; name: string; body: Node[] }
  | { k: "macro"; name: string; params: [string, Expr | undefined][]; body: Node[] }
  | { k: "break" }
  | { k: "continue" }
  | { k: "block"; body: Node[] };

/** Builds the statement tree from segments. */
class Parser {
  private i = 0;
  constructor(private readonly segs: Seg[]) {}

  parse(): Node[] {
    const body = this.body([]);
    if (this.i < this.segs.length) throw new TemplateError(`unexpected {% ${(this.segs[this.i] as { src: string }).src.trim()} %}`);
    return body;
  }

  /** Nodes up to (not including) a statement whose keyword is in `ends`. */
  private body(ends: string[]): Node[] {
    const out: Node[] = [];
    while (this.i < this.segs.length) {
      const s = this.segs[this.i]!;
      if (s.kind === "text") {
        out.push({ k: "text", text: s.text });
        this.i++;
        continue;
      }
      const toks = lex(s.src, s.pos);
      if (s.kind === "out") {
        const p = new ExprParser(toks);
        out.push({ k: "out", e: p.expr() });
        if (!p.done) throw new TemplateError(`unexpected "${p.peek()!.v}" in {{ ${s.src.trim()} }}`);
        this.i++;
        continue;
      }
      const kw = toks[0]?.v ?? "";
      if (ends.includes(kw)) return out;
      this.i++;
      out.push(this.statement(kw, new ExprParser(toks.slice(1)), s));
    }
    if (ends.length) throw new TemplateError(`missing {% ${ends[ends.length - 1]} %}`);
    return out;
  }

  /** The keyword of the statement that ended the last `body` call, consuming it. */
  private take(): { kw: string; p: ExprParser } {
    const s = this.segs[this.i++] as { src: string; pos: number };
    const toks = lex(s.src, s.pos);
    return { kw: toks[0]!.v, p: new ExprParser(toks.slice(1)) };
  }

  private statement(kw: string, p: ExprParser, s: Seg): Node {
    switch (kw) {
      case "if": {
        const branches = [{ test: p.expr(), body: this.body(["elif", "else", "endif"]) }];
        let other: Node[] = [];
        for (;;) {
          const end = this.take();
          if (end.kw === "elif") branches.push({ test: end.p.expr(), body: this.body(["elif", "else", "endif"]) });
          else if (end.kw === "else") other = this.body(["endif"]);
          else return { k: "if", branches, else: other };
        }
      }
      case "for": {
        const target = p.targets();
        p.expectName("in");
        const iter = p.or();
        let filter: Expr | undefined;
        if (p.isName("if")) {
          p.next();
          filter = p.expr();
        }
        const body = this.body(["else", "endfor"]);
        let other: Node[] = [];
        if (this.take().kw === "else") {
          other = this.body(["endfor"]);
          this.take();
        }
        return { k: "for", target, iter, filter, body, else: other };
      }
      case "set": {
        const target = p.targets();
        if (p.done) {
          if (target.k !== "var") throw new TemplateError("block set needs a plain name");
          const body = this.body(["endset"]);
          this.take();
          return { k: "setBlock", name: target.name, body };
        }
        p.expectOp("=");
        return { k: "set", target, value: p.tuple() };
      }
      case "macro": {
        const name = p.expectName();
        p.expectOp("(");
        const params: [string, Expr | undefined][] = [];
        while (!p.isOp(")")) {
          const n = p.expectName();
          let d: Expr | undefined;
          if (p.isOp("=")) {
            p.next();
            d = p.expr();
          }
          params.push([n, d]);
          if (!p.isOp(")")) p.expectOp(",");
        }
        const body = this.body(["endmacro"]);
        this.take();
        return { k: "macro", name, params, body };
      }
      case "generation": {
        const body = this.body(["endgeneration"]);
        this.take();
        return { k: "block", body };
      }
      case "break":
      case "continue":
        return { k: kw };
      default:
        throw new TemplateError(`unsupported statement {% ${(s as { src: string }).src.trim()} %}`);
    }
  }
}

// ── Evaluation ───────────────────────────────────────────────────────────────────────────────────

class Scope {
  private readonly vars = new Map<string, unknown>();
  constructor(readonly parent?: Scope) {}
  get(name: string): unknown {
    if (this.vars.has(name)) return this.vars.get(name);
    return this.parent ? this.parent.get(name) : new Undef(name);
  }
  set(name: string, v: unknown) {
    this.vars.set(name, v);
  }
}

class LoopSignal {
  constructor(readonly kind: "break" | "continue") {}
}

const truthy = (v: unknown): boolean => {
  if (isUndef(v) || v === null || v === undefined || v === false || v === 0 || v === "") return false;
  if (Array.isArray(v)) return v.length > 0;
  if (v instanceof Namespace) return true;
  if (typeof v === "object") return Object.keys(v as object).length > 0;
  return true;
};

/** Python's str(). */
function str(v: unknown): string {
  if (isUndef(v) || v === undefined) return "";
  if (v === null) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "string") return v;
  if (typeof v === "number") return String(v);
  return pyRepr(v);
}

function pyRepr(v: unknown): string {
  if (typeof v === "string") return `'${v.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n")}'`;
  if (Array.isArray(v)) return `[${v.map(pyRepr).join(", ")}]`;
  if (v && typeof v === "object") return `{${Object.entries(v).map(([k, x]) => `${pyRepr(k)}: ${pyRepr(x)}`).join(", ")}}`;
  return str(v);
}

/** `json.dumps(v, ensure_ascii=False, indent=indent)`, as transformers' `tojson` filter. */
export function pyJson(v: unknown, indent?: number, level = 0): string {
  if (v === null || v === undefined || isUndef(v)) return "null";
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "number" || typeof v === "boolean") return JSON.stringify(v);
  const nl = indent === undefined ? "" : "\n" + " ".repeat(indent * (level + 1));
  const end = indent === undefined ? "" : "\n" + " ".repeat(indent * level);
  const sep = indent === undefined ? ", " : ",";
  if (Array.isArray(v)) return v.length ? `[${nl}${v.map((x) => pyJson(x, indent, level + 1)).join(sep + nl)}${end}]` : "[]";
  const obj = v instanceof Namespace ? v.vars : (v as Record<string, unknown>);
  const entries = Object.entries(obj);
  return entries.length ? `{${nl}${entries.map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x, indent, level + 1)}`).join(sep + nl)}${end}}` : "{}";
}

function equal(a: unknown, b: unknown): boolean {
  if (isUndef(a) || isUndef(b)) return isUndef(a) && isUndef(b);
  if (a === b) return true;
  if (typeof a === "object" && typeof b === "object" && a && b) return pyJson(a) === pyJson(b);
  return false;
}

function iterate(v: unknown): unknown[] {
  if (Array.isArray(v)) return v;
  if (typeof v === "string") return Array.from(v);
  if (isUndef(v) || v === null) return [];
  if (v instanceof Namespace) return Object.keys(v.vars);
  if (typeof v === "object") return Object.keys(v as object);
  throw new TemplateError(`${str(v)} is not iterable`);
}

function getAttr(obj: unknown, name: string): unknown {
  if (isUndef(obj) || obj === null || obj === undefined) return new Undef(name);
  if (obj instanceof Namespace) return Object.prototype.hasOwnProperty.call(obj.vars, name) ? obj.vars[name] : new Undef(name);
  if (typeof obj === "string") return strMethod(obj, name) ?? new Undef(name);
  if (Array.isArray(obj)) return listMethod(obj, name) ?? new Undef(name);
  if (typeof obj === "object") {
    const rec = obj as Record<string, unknown>;
    if (Object.prototype.hasOwnProperty.call(rec, name)) return rec[name];
    return dictMethod(rec, name) ?? new Undef(name);
  }
  return new Undef(name);
}

function getItem(obj: unknown, key: unknown): unknown {
  if (Array.isArray(obj) || typeof obj === "string") {
    const n = Number(key);
    if (!Number.isInteger(n)) return getAttr(obj, String(key));
    const i = n < 0 ? obj.length + n : n;
    return i >= 0 && i < obj.length ? obj[i] : new Undef(String(key));
  }
  if (obj && typeof obj === "object" && !(obj instanceof Undef)) {
    const rec = obj instanceof Namespace ? obj.vars : (obj as Record<string, unknown>);
    const k = String(key);
    return Object.prototype.hasOwnProperty.call(rec, k) ? rec[k] : new Undef(k);
  }
  return new Undef(String(key));
}

function slice(v: unknown, start?: number, stop?: number, step = 1): unknown {
  const seq = typeof v === "string" ? Array.from(v) : iterate(v);
  const n = seq.length;
  const norm = (x: number | undefined, d: number) => (x === undefined ? d : x < 0 ? Math.max(0, n + x) : Math.min(n, x));
  const out: unknown[] = [];
  if (!Number.isInteger(step) || step === 0) throw new TemplateError("slice step must be a non-zero integer");
  if (step > 0) for (let i = norm(start, 0); i < norm(stop, n); i += step) out.push(seq[i]);
  else {
    const s = start === undefined ? n - 1 : start < 0 ? n + start : Math.min(n - 1, start);
    const e = stop === undefined ? -1 : stop < 0 ? n + stop : stop;
    for (let i = s; i > e; i += step) out.push(seq[i]);
  }
  return typeof v === "string" ? out.join("") : out;
}

const fn = (f: Fn) => new Callable(f);

function stripChars(s: string, chars: unknown, side: "both" | "left" | "right"): string {
  const set = typeof chars === "string" ? chars : undefined;
  const isStrip = (c: string) => (set === undefined ? /\s/.test(c) : set.includes(c));
  let a = 0;
  let b = s.length;
  if (side !== "right") while (a < b && isStrip(s[a]!)) a++;
  if (side !== "left") while (b > a && isStrip(s[b - 1]!)) b--;
  return s.slice(a, b);
}

function strMethod(s: string, name: string): Callable | undefined {
  switch (name) {
    case "strip":
      return fn(([c]) => stripChars(s, c, "both"));
    case "lstrip":
      return fn(([c]) => stripChars(s, c, "left"));
    case "rstrip":
      return fn(([c]) => stripChars(s, c, "right"));
    case "upper":
      return fn(() => s.toUpperCase());
    case "lower":
      return fn(() => s.toLowerCase());
    case "title":
      return fn(() => title(s));
    case "capitalize":
      return fn(() => capitalize(s));
    case "split":
      return fn(([sep, max]) => {
        const limit = typeof max === "number" && max >= 0 ? max : Infinity;
        if (sep === undefined || sep === null) {
          const parts = s.trim().split(/\s+/).filter(Boolean);
          return limit === Infinity ? parts : [...parts.slice(0, limit), ...(parts.length > limit ? [parts.slice(limit).join(" ")] : [])];
        }
        const parts = s.split(String(sep));
        return limit === Infinity || parts.length <= limit + 1 ? parts : [...parts.slice(0, limit), parts.slice(limit).join(String(sep))];
      });
    case "startswith":
      return fn(([p]) => (Array.isArray(p) ? p.some((x) => s.startsWith(String(x))) : s.startsWith(String(p))));
    case "endswith":
      return fn(([p]) => (Array.isArray(p) ? p.some((x) => s.endsWith(String(x))) : s.endsWith(String(p))));
    case "replace":
      return fn(([a, b, count]) => {
        let n = typeof count === "number" ? count : Infinity;
        let out = "";
        let rest = s;
        const from = String(a);
        while (n-- > 0) {
          const i = rest.indexOf(from);
          if (i < 0 || from === "") break;
          out += rest.slice(0, i) + String(b);
          rest = rest.slice(i + from.length);
        }
        return out + rest;
      });
    case "find":
      return fn(([x]) => s.indexOf(String(x)));
    case "count":
      return fn(([x]) => (String(x) ? s.split(String(x)).length - 1 : s.length + 1));
    case "join":
      return fn(([xs]) => iterate(xs).map(str).join(s));
    case "format":
      return fn((args) => {
        let i = 0;
        return s.replace(/\{(\d*)\}/g, (_, d: string) => str(args[d === "" ? i++ : Number(d)]));
      });
  }
  return undefined;
}

function listMethod(xs: unknown[], name: string): Callable | undefined {
  if (name === "index") return fn(([x]) => xs.findIndex((y) => equal(x, y)));
  if (name === "count") return fn(([x]) => xs.filter((y) => equal(x, y)).length);
  return undefined;
}

function dictMethod(d: Record<string, unknown>, name: string): Callable | undefined {
  switch (name) {
    case "items":
      return fn(() => Object.entries(d));
    case "keys":
      return fn(() => Object.keys(d));
    case "values":
      return fn(() => Object.values(d));
    case "get":
      return fn(([k, def]) => (Object.prototype.hasOwnProperty.call(d, String(k)) ? d[String(k)] : def ?? null));
  }
  return undefined;
}

const title = (s: string) => s.replace(/[A-Za-z]+/g, (w) => w[0]!.toUpperCase() + w.slice(1).toLowerCase());
const capitalize = (s: string) => (s ? s[0]!.toUpperCase() + s.slice(1).toLowerCase() : s);

function test(name: string, v: unknown, args: unknown[]): boolean {
  switch (name) {
    case "defined":
      return !isUndef(v);
    case "undefined":
      return isUndef(v);
    case "none":
      return v === null;
    case "string":
      return typeof v === "string";
    case "number":
      return typeof v === "number";
    case "integer":
      return Number.isInteger(v);
    case "float":
      return typeof v === "number" && !Number.isInteger(v);
    case "boolean":
      return typeof v === "boolean";
    case "true":
      return v === true;
    case "false":
      return v === false;
    case "mapping":
      return !!v && typeof v === "object" && !Array.isArray(v) && !isUndef(v);
    case "sequence":
    case "iterable":
      return typeof v === "string" || Array.isArray(v) || (!!v && typeof v === "object" && !isUndef(v));
    case "callable":
      return v instanceof Callable;
    case "odd":
      return Number(v) % 2 === 1;
    case "even":
      return Number(v) % 2 === 0;
    case "divisibleby":
      return Number(v) % Number(args[0]) === 0;
    case "eq":
    case "equalto":
    case "sameas":
    case "==":
      return equal(v, args[0]);
    case "ne":
      return !equal(v, args[0]);
    case "in":
      return contains(args[0], v);
    case "lower":
      return typeof v === "string" && v === v.toLowerCase();
    case "upper":
      return typeof v === "string" && v === v.toUpperCase();
  }
  throw new TemplateError(`unknown test "${name}"`);
}

function contains(container: unknown, x: unknown): boolean {
  if (typeof container === "string") return container.includes(str(x));
  if (Array.isArray(container)) return container.some((y) => equal(x, y));
  if (container instanceof Namespace) return str(x) in container.vars;
  if (container && typeof container === "object" && !isUndef(container)) return Object.prototype.hasOwnProperty.call(container, str(x));
  return false;
}

export interface TemplateOptions {
  /**
   * Values and functions every render sees, on top of the built-ins (raise_exception, strftime_now,
   * namespace, range, dict). Functions get positional arguments followed by a kwargs object.
   */
  globals?: Record<string, unknown>;
}

/** JS functions become template callables; everything else passes through. */
const wrap = (v: unknown): unknown => (typeof v === "function" ? fn((args, kw) => (v as (...a: unknown[]) => unknown)(...args, kw)) : v);

export class Template {
  private readonly nodes: Node[];
  private readonly globals: Record<string, unknown>;

  constructor(source: string, opts: TemplateOptions = {}) {
    this.nodes = new Parser(segment(source)).parse();
    this.globals = {
      raise_exception: fn(([msg]) => {
        throw new TemplateError(str(msg));
      }),
      strftime_now: fn(([f]) => strftime(new Date(), str(f))),
      namespace: fn((_, kw) => new Namespace({ ...kw })),
      range: fn((args) => {
        const [a, b, s] = args.length === 1 ? [0, Number(args[0]), 1] : [Number(args[0]), Number(args[1]), Number(args[2] ?? 1)];
        if (![a, b, s].every(Number.isFinite) || s === 0) throw new TemplateError("range() needs finite numbers and a non-zero step");
        if (Math.max(0, Math.ceil((b - a) / s)) > MAX_RANGE) throw new TemplateError(`range() of more than ${MAX_RANGE} items`);
        const out: number[] = [];
        for (let i = a; s > 0 ? i < b : i > b; i += s) out.push(i);
        return out;
      }),
      dict: fn((_, kw) => ({ ...kw })),
      ...Object.fromEntries(Object.entries(opts.globals ?? {}).map(([k, v]) => [k, wrap(v)])),
    };
  }

  render(vars: Record<string, unknown> = {}): string {
    const root = new Scope();
    for (const [k, v] of Object.entries(this.globals)) root.set(k, v);
    const scope = new Scope(root);
    for (const [k, v] of Object.entries(vars)) if (v !== undefined) scope.set(k, wrap(v));
    const out: string[] = [];
    this.steps = 0;
    this.written = 0;
    this.exec(this.nodes, scope, out);
    return out.join("");
  }

  /** Statements executed and characters written by the render in progress (see MAX_STEPS, MAX_STRING). */
  private steps = 0;
  private written = 0;

  private step(): void {
    if (++this.steps > MAX_STEPS) throw new TemplateError("the template runs too long (a loop without end?)");
  }

  private write(out: string[], text: string): void {
    this.written += text.length;
    if (this.written > MAX_STRING) throw new TemplateError("the template's output is too long");
    out.push(text);
  }

  private exec(nodes: Node[], scope: Scope, out: string[]): void {
    for (const n of nodes) {
      this.step();
      switch (n.k) {
        case "text":
          this.write(out, n.text);
          break;
        case "out":
          this.write(out, str(this.eval(n.e, scope)));
          break;
        case "if": {
          const b = n.branches.find((br) => truthy(this.eval(br.test, scope)));
          this.exec(b ? b.body : n.else, scope, out);
          break;
        }
        case "for":
          this.execFor(n, scope, out);
          break;
        case "set":
          this.assign(n.target, this.eval(n.value, scope), scope);
          break;
        case "setBlock": {
          const buf: string[] = [];
          this.exec(n.body, scope, buf);
          scope.set(n.name, buf.join(""));
          break;
        }
        case "macro":
          scope.set(
            n.name,
            fn((args, kw) => {
              const inner = new Scope(scope);
              n.params.forEach(([p, d], i) => inner.set(p, i < args.length ? args[i] : p in kw ? kw[p] : d ? this.eval(d, scope) : new Undef(p)));
              const buf: string[] = [];
              this.exec(n.body, inner, buf);
              return buf.join("");
            }),
          );
          break;
        case "block":
          this.exec(n.body, scope, out);
          break;
        case "break":
        case "continue":
          throw new LoopSignal(n.k);
      }
    }
  }

  private execFor(n: Extract<Node, { k: "for" }>, scope: Scope, out: string[]): void {
    const iterable = this.eval(n.iter, scope);
    let items = iterate(iterable);
    // Iterating a dict yields keys, as in Python.
    if (n.filter) {
      items = items.filter((it) => {
        const s = new Scope(scope);
        this.assign(n.target, it, s);
        return truthy(this.eval(n.filter!, s));
      });
    }
    if (!items.length) {
      this.exec(n.else, scope, out);
      return;
    }
    const len = items.length;
    for (let i = 0; i < len; i++) {
      this.step();
      const s = new Scope(scope);
      this.assign(n.target, items[i], s);
      s.set("loop", {
        index: i + 1,
        index0: i,
        revindex: len - i,
        revindex0: len - i - 1,
        first: i === 0,
        last: i === len - 1,
        length: len,
        previtem: i > 0 ? items[i - 1] : new Undef("previtem"),
        nextitem: i < len - 1 ? items[i + 1] : new Undef("nextitem"),
        cycle: fn((args) => args[i % args.length]),
      });
      try {
        this.exec(n.body, s, out);
      } catch (e) {
        if (!(e instanceof LoopSignal)) throw e;
        if (e.kind === "break") break;
      }
    }
  }

  private assign(target: Expr, value: unknown, scope: Scope): void {
    if (target.k === "var") scope.set(target.name, value);
    else if (target.k === "list") {
      const vs = iterate(value);
      if (vs.length !== target.items.length) throw new TemplateError(`cannot unpack ${vs.length} values into ${target.items.length}`);
      target.items.forEach((t, i) => this.assign(t, vs[i], scope));
    } else if (target.k === "attr") {
      const obj = this.eval(target.obj, scope);
      if (!(obj instanceof Namespace)) throw new TemplateError("can only set attributes of a namespace()");
      obj.vars[target.name] = value;
    } else throw new TemplateError("invalid assignment target");
  }

  private eval(e: Expr, scope: Scope): unknown {
    switch (e.k) {
      case "lit":
        return e.v;
      case "var":
        return scope.get(e.name);
      case "list":
        return e.items.map((x) => this.eval(x, scope));
      case "dict":
        return Object.fromEntries(e.items.map(([k, v]) => [str(this.eval(k, scope)), this.eval(v, scope)]));
      case "attr":
        return getAttr(this.eval(e.obj, scope), e.name);
      case "item":
        return getItem(this.eval(e.obj, scope), this.eval(e.key, scope));
      case "slice": {
        const num = (x?: Expr) => {
          const v = x && this.eval(x, scope);
          return v === undefined || v === null || isUndef(v) ? undefined : Number(v);
        };
        return slice(this.eval(e.obj, scope), num(e.start), num(e.stop), num(e.step) ?? 1);
      }
      case "call": {
        const f = this.eval(e.fn, scope);
        if (!(f instanceof Callable)) throw new TemplateError(`${exprName(e.fn)} is not callable`);
        return f.fn(
          e.args.map((a) => this.eval(a, scope)),
          Object.fromEntries(e.kwargs.map(([k, v]) => [k, this.eval(v, scope)])),
        );
      }
      case "filter":
        return this.filter(e.name, this.eval(e.value, scope), e.args.map((a) => this.eval(a, scope)), Object.fromEntries(e.kwargs.map(([k, v]) => [k, this.eval(v, scope)])), scope);
      case "test": {
        const r = test(e.name, this.eval(e.value, scope), e.args.map((a) => this.eval(a, scope)));
        return e.negate ? !r : r;
      }
      case "un": {
        const a = this.eval(e.a, scope);
        if (e.op === "not") return !truthy(a);
        return e.op === "-" ? -Number(a) : Number(a);
      }
      case "cond":
        return truthy(this.eval(e.test, scope)) ? this.eval(e.then, scope) : e.else ? this.eval(e.else, scope) : new Undef("else");
      case "bin":
        return this.binary(e, scope);
    }
  }

  private binary(e: Extract<Expr, { k: "bin" }>, scope: Scope): unknown {
    if (e.op === "and") {
      const a = this.eval(e.a, scope);
      return truthy(a) ? this.eval(e.b, scope) : a;
    }
    if (e.op === "or") {
      const a = this.eval(e.a, scope);
      return truthy(a) ? a : this.eval(e.b, scope);
    }
    const a = this.eval(e.a, scope);
    const b = this.eval(e.b, scope);
    switch (e.op) {
      case "==":
        return equal(a, b);
      case "!=":
        return !equal(a, b);
      case "<":
        return (a as number) < (b as number);
      case ">":
        return (a as number) > (b as number);
      case "<=":
        return (a as number) <= (b as number);
      case ">=":
        return (a as number) >= (b as number);
      case "in":
        return contains(b, a);
      case "~":
        return bounded(str(a) + str(b));
      case "+":
        if (Array.isArray(a) && Array.isArray(b)) {
          if (a.length + b.length > 10 * MAX_RANGE) throw new TemplateError("the template builds a list that is too long");
          return [...a, ...b];
        }
        if (typeof a === "string" || typeof b === "string") {
          if (typeof a !== "string" || typeof b !== "string") throw new TemplateError(`cannot add ${pyRepr(a)} and ${pyRepr(b)}`);
          return bounded(a + b);
        }
        return Number(a) + Number(b);
      case "-":
        return Number(a) - Number(b);
      case "*":
        if (typeof a === "string") {
          const times = Math.max(0, Math.floor(Number(b)) || 0);
          if (a.length * times > MAX_STRING) throw new TemplateError("the template builds a string that is too long");
          return a.repeat(times);
        }
        return Number(a) * Number(b);
      case "/":
        return Number(a) / Number(b);
      case "//":
        return Math.floor(Number(a) / Number(b));
      case "%": {
        const m = Number(a) % Number(b);
        return m !== 0 && Math.sign(m) !== Math.sign(Number(b)) ? m + Number(b) : m;
      }
      case "**":
        return Number(a) ** Number(b);
    }
    throw new TemplateError(`unknown operator ${e.op}`);
  }

  private filter(name: string, v: unknown, args: unknown[], kw: Record<string, unknown>, scope: Scope): unknown {
    switch (name) {
      case "length":
      case "count":
        return typeof v === "string" ? Array.from(v).length : iterate(v).length;
      case "tojson":
        return pyJson(v, (kw.indent ?? args[0]) as number | undefined);
      case "trim":
        return stripChars(str(v), args[0], "both");
      case "upper":
        return str(v).toUpperCase();
      case "lower":
        return str(v).toLowerCase();
      case "title":
        return title(str(v));
      case "capitalize":
        return capitalize(str(v));
      case "string":
        return str(v);
      case "safe":
      case "e":
      case "escape":
        return name === "safe" ? v : str(v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&#34;").replace(/'/g, "&#39;");
      case "int":
        return Number.isNaN(parseInt(str(v), 10)) ? (args[0] ?? 0) : Math.trunc(Number(v));
      case "float":
        return Number.isNaN(Number(v)) ? (args[0] ?? 0) : Number(v);
      case "abs":
        return Math.abs(Number(v));
      case "default":
      case "d":
        return isUndef(v) || ((args[1] ?? kw.boolean) && !truthy(v)) ? (args[0] ?? "") : v;
      case "list":
        return iterate(v);
      case "first":
        return iterate(v)[0] ?? new Undef("first");
      case "last": {
        const xs = iterate(v);
        return xs.length ? xs[xs.length - 1] : new Undef("last");
      }
      case "reverse":
        return typeof v === "string" ? Array.from(v).reverse().join("") : [...iterate(v)].reverse();
      case "join": {
        const attr = kw.attribute as string | undefined;
        return iterate(v)
          .map((x) => str(attr ? getAttr(x, attr) : x))
          .join(str(args[0] ?? kw.d ?? ""));
      }
      case "replace":
        return (strMethod(str(v), "replace")!.fn(args, {}) as string);
      case "items":
        return v && typeof v === "object" && !isUndef(v) ? Object.entries(v instanceof Namespace ? v.vars : v) : [];
      case "unique": {
        const seen: unknown[] = [];
        for (const x of iterate(v)) if (!seen.some((y) => equal(x, y))) seen.push(x);
        return seen;
      }
      case "sum":
        return iterate(v).reduce((n: number, x) => n + Number(kw.attribute ? getAttr(x, String(kw.attribute)) : x), Number(kw.start ?? 0));
      case "indent": {
        const width = (kw.width ?? args[0] ?? 4) as number | string;
        const pad = typeof width === "number" ? " ".repeat(width) : width;
        const first = !!(kw.first ?? args[1]);
        const blank = !!(kw.blank ?? args[2]);
        return str(v)
          .split("\n")
          .map((l, i) => ((i > 0 || first) && (blank || l !== "") ? pad + l : l))
          .join("\n");
      }
      case "selectattr":
      case "rejectattr": {
        const [attr, t, ...targs] = args as [string, string | undefined, ...unknown[]];
        const keep = (x: unknown) => {
          const a = getAttr(x, attr);
          return t ? test(t, a, targs) : truthy(a);
        };
        return iterate(v).filter((x) => keep(x) === (name === "selectattr"));
      }
      case "select":
      case "reject": {
        const [t, ...targs] = args as [string | undefined, ...unknown[]];
        return iterate(v).filter((x) => (t ? test(t, x, targs) : truthy(x)) === (name === "select"));
      }
      case "map": {
        if (kw.attribute !== undefined) return iterate(v).map((x) => {
          const a = getAttr(x, String(kw.attribute));
          return isUndef(a) && "default" in kw ? kw.default : a;
        });
        const [f, ...fargs] = args as [string, ...unknown[]];
        return iterate(v).map((x) => this.filter(f, x, fargs, {}, scope));
      }
      case "dictsort":
        return Object.entries((v ?? {}) as object).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
      case "round": {
        const p = Number(args[0] ?? 0);
        return Math.round(Number(v) * 10 ** p) / 10 ** p;
      }
    }
    const custom = this.globals[`filter:${name}`];
    if (custom instanceof Callable) return custom.fn([v, ...args], kw);
    throw new TemplateError(`unknown filter "${name}"`);
  }
}

function exprName(e: Expr): string {
  return e.k === "var" ? e.name : e.k === "attr" ? `${exprName(e.obj)}.${e.name}` : "expression";
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

/** The strftime directives chat templates use (dates in system prompts). */
export function strftime(d: Date, f: string): string {
  const p2 = (n: number) => String(n).padStart(2, "0");
  return f.replace(/%([a-zA-Z%])/g, (_, c: string) => {
    switch (c) {
      case "d":
        return p2(d.getDate());
      case "m":
        return p2(d.getMonth() + 1);
      case "Y":
        return String(d.getFullYear());
      case "y":
        return p2(d.getFullYear() % 100);
      case "B":
        return MONTHS[d.getMonth()]!;
      case "b":
        return MONTHS[d.getMonth()]!.slice(0, 3);
      case "A":
        return DAYS[d.getDay()]!;
      case "a":
        return DAYS[d.getDay()]!.slice(0, 3);
      case "H":
        return p2(d.getHours());
      case "M":
        return p2(d.getMinutes());
      case "S":
        return p2(d.getSeconds());
      case "j":
        return String(Math.floor((d.getTime() - new Date(d.getFullYear(), 0, 1).getTime()) / 86400000) + 1).padStart(3, "0");
      case "%":
        return "%";
    }
    return `%${c}`;
  });
}
