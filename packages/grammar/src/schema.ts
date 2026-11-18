import { COMPACT, PRETTY, type Node, type NumberBounds, type Prop, type Whitespace } from "./matcher";
import { gbnfNode } from "./gbnf";
import { FORMATS, regexNode } from "./regex";

/** The JSON Schema subset `compileSchema` understands. */
export interface JsonSchema {
  type?: JsonType | JsonType[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  /** Default false when `properties` is given: only declared keys, in declared order. */
  additionalProperties?: boolean | JsonSchema;
  items?: JsonSchema;
  minItems?: number;
  maxItems?: number;
  minLength?: number;
  maxLength?: number;
  /** A regular expression the string's value must match in full (see `regexNode`). */
  pattern?: string;
  /** Enforced: date, time, date-time, email, uuid, ipv4. Other formats are annotations. */
  format?: string;
  minimum?: number;
  maximum?: number;
  /** A number (draft 6 and later), or true to make `minimum` exclusive (draft 4). */
  exclusiveMinimum?: number | boolean;
  exclusiveMaximum?: number | boolean;
  multipleOf?: number;
  enum?: unknown[];
  const?: unknown;
  anyOf?: JsonSchema[];
  oneOf?: JsonSchema[];
  allOf?: JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  definitions?: Record<string, JsonSchema>;
  [annotation: string]: unknown;
}

type JsonType = "string" | "number" | "integer" | "boolean" | "null" | "object" | "array";

/** What generated text must look like. */
export type ResponseFormat =
  /**
   * A JSON value; any JSON without a schema. Whitespace between tokens is "compact" by default: one
   * space at most, no line breaks. Small models given more pad the output with newlines and repeat
   * themselves (SmolLM2-135M: 57 tokens instead of 47 for the same object). "pretty" allows runs of
   * up to 32 spaces, tabs and newlines.
   */
  | { type: "json"; schema?: JsonSchema; whitespace?: "pretty" | "compact" }
  /** Exactly one of these strings. */
  | { type: "choice"; choices: string[] }
  /** Text matching a regular expression in full (JavaScript syntax; see `regexNode` for the subset). */
  | { type: "regex"; pattern: string }
  /** Text matching a GBNF grammar, llama.cpp's format, from rule `root` (default "root"). */
  | { type: "grammar"; grammar: string; root?: string }
  /**
   * One call to one of these functions, as `<tool_call>\n{"name": …, "arguments": …}\n</tool_call>`
   * (the format of Qwen, SmolLM3 and Hermes-style templates), with the arguments following the
   * function's `parameters` schema. Whitespace inside the JSON is compact.
   */
  | { type: "tool_call"; tools: { name: string; parameters?: JsonSchema }[]; syntax?: ToolCallSyntax };

const utf8 = new TextEncoder();
const lit = (s: string): Node => ({ kind: "lit", bytes: utf8.encode(s) });

/** Grammar of any JSON value. */
export function anyJson(ws: Whitespace = PRETTY): Node {
  const any: Node & { nodes: Node[] } = { kind: "alt", nodes: [] };
  any.nodes.push(
    { kind: "object", props: [], additional: any, ws },
    { kind: "array", items: any, minItems: 0, maxItems: Infinity, ws },
    { kind: "string", minLength: 0, maxLength: Infinity },
    { kind: "number", integer: false },
    lit("true"),
    lit("false"),
    lit("null"),
  );
  return any;
}

/** Keywords that would change what matches but are not implemented; ignoring them would accept wrong output. */
const UNSUPPORTED = ["patternProperties", "not", "if", "then", "else", "dependentRequired", "dependentSchemas", "propertyNames", "prefixItems", "contains", "uniqueItems"];

export function compileSchema(root: JsonSchema, ws: Whitespace = PRETTY): Node {
  const refs = new Map<string, Node & { nodes: Node[] }>();
  let any: Node | undefined;

  const target = (ref: string): JsonSchema => {
    const m = /^#\/(\$defs|definitions)\/(.+)$/.exec(ref);
    const t = m ? (root[m[1] as "$defs" | "definitions"] ?? {})[m[2]!.replace(/~1/g, "/").replace(/~0/g, "~")] : ref === "#" ? root : undefined;
    if (!t) throw new Error(`schema: cannot resolve $ref "${ref}" (only #, #/$defs/… and #/definitions/…)`);
    return t;
  };

  const resolve = (ref: string): Node => {
    let node = refs.get(ref);
    if (node) return node;
    const t = target(ref);
    // A shell first, so recursive references point back at it.
    node = { kind: "alt", nodes: [] };
    refs.set(ref, node);
    node.nodes.push(compile(t));
    return node;
  };

  const compile = (s: JsonSchema | boolean): Node => {
    if (s === true) return (any ??= anyJson(ws));
    if (s === false) throw new Error("schema: `false` matches nothing");
    for (const k of UNSUPPORTED) if (k in s) throw new Error(`schema: "${k}" is not supported`);
    // allOf first: its members (and the keywords beside it) are folded into one schema.
    if (s.allOf) return compile(mergeAllOf(s, target));
    if (s.$ref) return resolve(s.$ref);
    if (s.const !== undefined) return lit(JSON.stringify(s.const));
    if (s.enum) return { kind: "alt", nodes: s.enum.map((v) => lit(JSON.stringify(v))) };
    const alts = s.anyOf ?? s.oneOf;
    if (alts) return { kind: "alt", nodes: alts.map(compile) };
    const types = s.type === undefined ? undefined : Array.isArray(s.type) ? s.type : [s.type];
    if (!types) {
      if (s.properties || s.additionalProperties !== undefined) return compileType(s, "object");
      if (s.items) return compileType(s, "array");
      return (any ??= anyJson(ws));
    }
    const nodes = types.filter((t) => !(t === "integer" && types.includes("number"))).map((t) => compileType(s, t));
    return nodes.length === 1 ? nodes[0]! : { kind: "alt", nodes };
  };

  const compileType = (s: JsonSchema, t: JsonType): Node => {
    switch (t) {
      case "string": {
        const pattern = s.pattern ?? (s.format ? FORMATS[s.format] : undefined);
        if (pattern === undefined) return { kind: "string", minLength: s.minLength ?? 0, maxLength: s.maxLength ?? Infinity };
        if (s.minLength !== undefined || s.maxLength !== undefined) throw new Error("schema: minLength / maxLength together with pattern or format are not supported");
        if (s.pattern !== undefined && s.format !== undefined && FORMATS[s.format]) throw new Error("schema: pattern together with an enforced format is not supported");
        return { kind: "seq", nodes: [lit('"'), regexNode(pattern, { json: true }), lit('"')] };
      }
      case "number":
      case "integer": {
        const b = bounds(s);
        return b ? { kind: "number", integer: t === "integer", bounds: b } : { kind: "number", integer: t === "integer" };
      }
      case "boolean":
        return { kind: "alt", nodes: [lit("true"), lit("false")] };
      case "null":
        return lit("null");
      case "array":
        return { kind: "array", items: s.items ? compile(s.items) : (any ??= anyJson(ws)), minItems: s.minItems ?? 0, maxItems: s.maxItems ?? Infinity, ws };
      case "object": {
        const required = new Set(s.required ?? []);
        const props: Prop[] = Object.entries(s.properties ?? {}).map(([k, v]) => ({
          key: utf8.encode(JSON.stringify(k).slice(1)),
          value: compile(v),
          required: required.has(k),
        }));
        const extra = s.additionalProperties ?? !s.properties;
        return { kind: "object", props, additional: extra === false ? undefined : compile(extra), ws };
      }
    }
  };

  return compile(root);
}

/** Keywords that describe a schema without changing what matches it: the first one met is kept. */
const ANNOTATIONS = new Set(["title", "description", "default", "examples", "$comment", "$id", "$schema", "deprecated", "readOnly", "writeOnly", "$defs", "definitions"]);
/** Most alternatives an allOf over anyOf / oneOf members may expand to. */
const MAX_ALLOF_ALTERNATIVES = 64;

/** Two schemas that nothing matches together: inside alternatives that branch is dropped, elsewhere it is an error. */
class NoMatch extends Error {}

const sameJson = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const typesOf = (s: JsonSchema): JsonType[] | undefined => (s.type === undefined ? undefined : Array.isArray(s.type) ? s.type : [s.type]);

/** The lower bound of a schema as [value, exclusive], from either draft's keywords. */
function lower(s: JsonSchema): [number, boolean] | undefined {
  const ex = typeof s.exclusiveMinimum === "number" ? s.exclusiveMinimum : undefined;
  if (ex !== undefined && (s.minimum === undefined || ex >= s.minimum)) return [ex, true];
  return s.minimum === undefined ? undefined : [s.minimum, s.exclusiveMinimum === true];
}
function upper(s: JsonSchema): [number, boolean] | undefined {
  const ex = typeof s.exclusiveMaximum === "number" ? s.exclusiveMaximum : undefined;
  if (ex !== undefined && (s.maximum === undefined || ex <= s.maximum)) return [ex, true];
  return s.maximum === undefined ? undefined : [s.maximum, s.exclusiveMaximum === true];
}

/**
 * One schema that matches what both `a` and `b` match (neither holds allOf, anyOf, oneOf or $ref:
 * `mergeAllOf` has dealt with those). Keywords it cannot combine exactly throw, so an allOf is
 * either enforced in full or refused; none of it is dropped silently.
 */
function intersect(a: JsonSchema, b: JsonSchema): JsonSchema {
  const out: JsonSchema = { ...a };
  for (const [k, v] of Object.entries(b)) {
    if (v === undefined) continue;
    const have = a[k];
    if (have === undefined) {
      out[k] = v;
      continue;
    }
    if (ANNOTATIONS.has(k) || sameJson(have, v)) continue;
    switch (k) {
      case "type": {
        const [x, y] = [typesOf(a)!, typesOf(b)!];
        // An integer is a number, so "number" on one side keeps "integer" from the other.
        const both = x.filter((t) => y.includes(t) || (t === "integer" && y.includes("number")));
        for (const t of y) if (t === "integer" && x.includes("number") && !both.includes("integer")) both.push("integer");
        if (!both.length) throw new NoMatch(`schema: allOf asks for types that exclude each other (${x.join(", ")} and ${y.join(", ")})`);
        out.type = both.length === 1 ? both[0]! : both;
        break;
      }
      case "properties": {
        const props: Record<string, JsonSchema> = { ...a.properties };
        for (const [name, p] of Object.entries(b.properties!)) props[name] = name in props ? { allOf: [props[name]!, p] } : p;
        out.properties = props;
        break;
      }
      case "required":
        out.required = [...new Set([...a.required!, ...b.required!])];
        break;
      case "additionalProperties": {
        const [x, y] = [a.additionalProperties!, b.additionalProperties!];
        out.additionalProperties = x === false || y === false ? false : x === true ? y : y === true ? x : { allOf: [x, y] };
        break;
      }
      case "items":
        out.items = { allOf: [a.items!, b.items!] };
        break;
      case "enum": {
        const keep = a.enum!.filter((x) => b.enum!.some((y) => sameJson(x, y)));
        if (!keep.length) throw new NoMatch("schema: allOf combines enums with no value in common");
        out.enum = keep;
        break;
      }
      case "minItems":
      case "minLength":
        out[k] = Math.max(have as number, v as number);
        break;
      case "maxItems":
      case "maxLength":
        out[k] = Math.min(have as number, v as number);
        break;
      case "minimum":
      case "maximum":
      case "exclusiveMinimum":
      case "exclusiveMaximum":
        break; // Set together below.
      case "const":
        throw new NoMatch("schema: allOf combines two different const values");
      default:
        throw new Error(`schema: allOf cannot combine two different "${k}"`);
    }
  }
  // Bounds: the tighter of each side, written back in the draft 6 form.
  const pick = (x: [number, boolean] | undefined, y: [number, boolean] | undefined, sign: 1 | -1) => (!x ? y : !y ? x : x[0] === y[0] ? ([x[0], x[1] || y[1]] as [number, boolean]) : sign * x[0] > sign * y[0] ? x : y);
  const [lo, hi] = [pick(lower(a), lower(b), 1), pick(upper(a), upper(b), -1)];
  for (const k of ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]) delete out[k];
  if (lo) out[lo[1] ? "exclusiveMinimum" : "minimum"] = lo[0];
  if (hi) out[hi[1] ? "exclusiveMaximum" : "maximum"] = hi[0];
  // A const or an enum beside the other side's: the const must be one of the enum's values.
  if (out.const !== undefined && out.enum) {
    if (!out.enum.some((x) => sameJson(x, out.const))) throw new NoMatch("schema: allOf combines a const with an enum that does not hold it");
    delete out.enum;
  }
  return out;
}

/**
 * `s` with its allOf folded in: the members, the members' own allOf and $ref (looked up with
 * `target`), and the keywords beside the allOf become one schema. A member with anyOf or oneOf
 * turns the result into alternatives, each the rest combined with one of its branches.
 */
function mergeAllOf(s: JsonSchema, target: (ref: string) => JsonSchema, depth = 0): JsonSchema {
  if (depth > 32) throw new Error("schema: allOf refers back to itself");
  const { allOf, ...beside } = s;
  // Every part as a list of alternatives; plain parts have one.
  const parts: JsonSchema[][] = [];
  const add = (m: JsonSchema | boolean, d: number): void => {
    if (m === true) return;
    if (m === false) throw new Error("schema: `false` matches nothing");
    if (d > 32) throw new Error("schema: allOf refers back to itself");
    for (const k of UNSUPPORTED) if (k in m) throw new Error(`schema: "${k}" is not supported`);
    const { allOf: inner, $ref, anyOf, oneOf, ...own } = m;
    if (Object.keys(own).some((k) => !ANNOTATIONS.has(k))) parts.push([own]);
    if ($ref) add(target($ref), d + 1);
    for (const x of inner ?? []) add(x, d + 1);
    const alts = anyOf ?? oneOf;
    if (alts) parts.push(alts.map((x) => (x.allOf || x.$ref || x.anyOf || x.oneOf ? mergeAllOf({ allOf: [x] }, target, d + 1) : x)));
  };
  add(beside, depth);
  for (const m of allOf ?? []) add(m, depth + 1);
  let combined: JsonSchema[] = [{}];
  for (const alts of parts) {
    const next: JsonSchema[] = [];
    let empty: NoMatch | undefined;
    for (const have of combined) {
      for (const alt of alts) {
        // A branch that is itself alternatives (from a nested merge) multiplies out the same way.
        for (const leaf of alt.anyOf ?? [alt]) {
          try {
            next.push(intersect(have, leaf));
          } catch (err) {
            if (!(err instanceof NoMatch)) throw err;
            empty = err;
          }
        }
      }
    }
    if (!next.length) throw empty ?? new Error("schema: allOf matches nothing");
    if (next.length > MAX_ALLOF_ALTERNATIVES) throw new Error(`schema: allOf expands to more than ${MAX_ALLOF_ALTERNATIVES} alternatives`);
    combined = next;
  }
  return combined.length === 1 ? combined[0]! : { anyOf: combined };
}

/** A schema's numeric bounds (either draft's exclusive keywords) and multipleOf; undefined when there are none. */
function bounds(s: JsonSchema): NumberBounds | undefined {
  const b: NumberBounds = {};
  if (s.minimum !== undefined) b.min = s.minimum;
  if (s.maximum !== undefined) b.max = s.maximum;
  if (typeof s.exclusiveMinimum === "number") {
    if (b.min === undefined || s.exclusiveMinimum >= b.min) [b.min, b.exclusiveMin] = [s.exclusiveMinimum, true];
  } else if (s.exclusiveMinimum === true) b.exclusiveMin = true;
  if (typeof s.exclusiveMaximum === "number") {
    if (b.max === undefined || s.exclusiveMaximum <= b.max) [b.max, b.exclusiveMax] = [s.exclusiveMaximum, true];
  } else if (s.exclusiveMaximum === true) b.exclusiveMax = true;
  if (s.multipleOf !== undefined) {
    if (!(s.multipleOf > 0)) throw new Error("schema: multipleOf must be a positive number");
    b.multipleOf = s.multipleOf;
  }
  return b.min === undefined && b.max === undefined && b.multipleOf === undefined ? undefined : b;
}

export function compileFormat(f: ResponseFormat): Node {
  if (f.type === "json") {
    const ws = f.whitespace === "pretty" ? PRETTY : COMPACT;
    return f.schema ? compileSchema(f.schema, ws) : anyJson(ws);
  }
  if (f.type === "choice") {
    if (!f.choices.length) throw new Error("choice format needs at least one choice");
    return { kind: "alt", nodes: f.choices.map(lit) };
  }
  if (f.type === "tool_call") return toolCallGrammar(f.tools, f.syntax);
  if (f.type === "regex") return regexNode(f.pattern);
  if (f.type === "grammar") return gbnfNode(f.grammar, f.root);
  throw new Error(`unknown response format ${JSON.stringify(f)}`);
}

/**
 * How a `tool_call` format writes its calls. "tags" (default): one call as
 * `<tool_call>\n{…}\n</tool_call>`. "tags_body": the same after an opening tag the model already
 * wrote, and "tags_inner" without the closing tag as well (the caller writes it). "list_body":
 * Granite's JSON list of one or more calls, after its `<|tool_call|>` marker.
 */
export type ToolCallSyntax = "tags" | "tags_body" | "tags_inner" | "list_body";

/** See the `tool_call` response format. */
export function toolCallGrammar(tools: readonly { name: string; parameters?: JsonSchema }[], syntax: ToolCallSyntax = "tags"): Node {
  if (!tools.length) throw new Error("tool_call format needs at least one tool");
  const one: JsonSchema = {
    anyOf: tools.map((t) => ({
      type: "object",
      properties: { name: { const: t.name }, arguments: t.parameters ?? { type: "object" } },
      required: ["name", "arguments"],
    })),
  };
  if (syntax === "list_body") return compileSchema({ type: "array", items: one, minItems: 1 }, COMPACT);
  const call = compileSchema(one, COMPACT);
  return { kind: "seq", nodes: [lit(syntax === "tags" ? "<tool_call>\n" : "\n"), call, lit(syntax === "tags_inner" ? "\n" : "\n</tool_call>")] };
}
