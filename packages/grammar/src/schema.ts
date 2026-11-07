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
  | { type: "tool_call"; tools: { name: string; parameters?: JsonSchema }[] };

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

  const resolve = (ref: string): Node => {
    let node = refs.get(ref);
    if (node) return node;
    const m = /^#\/(\$defs|definitions)\/(.+)$/.exec(ref);
    const target = m ? (root[m[1] as "$defs" | "definitions"] ?? {})[m[2]!.replace(/~1/g, "/").replace(/~0/g, "~")] : ref === "#" ? root : undefined;
    if (!target) throw new Error(`schema: cannot resolve $ref "${ref}" (only #, #/$defs/… and #/definitions/…)`);
    // A shell first, so recursive references point back at it.
    node = { kind: "alt", nodes: [] };
    refs.set(ref, node);
    node.nodes.push(compile(target));
    return node;
  };

  const compile = (s: JsonSchema | boolean): Node => {
    if (s === true) return (any ??= anyJson(ws));
    if (s === false) throw new Error("schema: `false` matches nothing");
    for (const k of UNSUPPORTED) if (k in s) throw new Error(`schema: "${k}" is not supported`);
    if (s.$ref) return resolve(s.$ref);
    if (s.const !== undefined) return lit(JSON.stringify(s.const));
    if (s.enum) return { kind: "alt", nodes: s.enum.map((v) => lit(JSON.stringify(v))) };
    const alts = s.anyOf ?? s.oneOf;
    if (alts) return { kind: "alt", nodes: alts.map(compile) };
    if (s.allOf) {
      if (s.allOf.length !== 1) throw new Error("schema: allOf is only supported with one entry");
      return compile(s.allOf[0]!);
    }
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

/** A schema's numeric bounds, from either draft's exclusive keywords; undefined when there are none. */
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
  return b.min === undefined && b.max === undefined ? undefined : b;
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
  if (f.type === "tool_call") return toolCallGrammar(f.tools);
  if (f.type === "regex") return regexNode(f.pattern);
  if (f.type === "grammar") return gbnfNode(f.grammar, f.root);
  throw new Error(`unknown response format ${JSON.stringify(f)}`);
}

/** See the `tool_call` response format. */
export function toolCallGrammar(tools: readonly { name: string; parameters?: JsonSchema }[]): Node {
  if (!tools.length) throw new Error("tool_call format needs at least one tool");
  const call = compileSchema(
    {
      anyOf: tools.map((t) => ({
        type: "object",
        properties: { name: { const: t.name }, arguments: t.parameters ?? { type: "object" } },
        required: ["name", "arguments"],
      })),
    },
    COMPACT,
  );
  return { kind: "seq", nodes: [lit("<tool_call>\n"), call, lit("\n</tool_call>")] };
}
