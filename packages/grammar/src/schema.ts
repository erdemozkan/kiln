import type { Node, Prop } from "./matcher";

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
  /** A JSON value; any JSON without a schema. */
  | { type: "json"; schema?: JsonSchema }
  /** Exactly one of these strings. */
  | { type: "choice"; choices: string[] };

const utf8 = new TextEncoder();
const lit = (s: string): Node => ({ kind: "lit", bytes: utf8.encode(s) });

/** Grammar of any JSON value. */
export function anyJson(): Node {
  const any: Node & { nodes: Node[] } = { kind: "alt", nodes: [] };
  any.nodes.push(
    { kind: "object", props: [], additional: any },
    { kind: "array", items: any, minItems: 0, maxItems: Infinity },
    { kind: "string", minLength: 0, maxLength: Infinity },
    { kind: "number", integer: false },
    lit("true"),
    lit("false"),
    lit("null"),
  );
  return any;
}

/** Keywords that would change what matches but are not implemented; ignoring them would accept wrong output. */
const UNSUPPORTED = ["pattern", "patternProperties", "not", "if", "then", "else", "dependentRequired", "dependentSchemas", "propertyNames", "prefixItems", "contains", "uniqueItems"];

export function compileSchema(root: JsonSchema): Node {
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
    if (s === true) return (any ??= anyJson());
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
      return (any ??= anyJson());
    }
    const nodes = types.filter((t) => !(t === "integer" && types.includes("number"))).map((t) => compileType(s, t));
    return nodes.length === 1 ? nodes[0]! : { kind: "alt", nodes };
  };

  const compileType = (s: JsonSchema, t: JsonType): Node => {
    switch (t) {
      case "string":
        return { kind: "string", minLength: s.minLength ?? 0, maxLength: s.maxLength ?? Infinity };
      case "number":
      case "integer":
        return { kind: "number", integer: t === "integer" };
      case "boolean":
        return { kind: "alt", nodes: [lit("true"), lit("false")] };
      case "null":
        return lit("null");
      case "array":
        return { kind: "array", items: s.items ? compile(s.items) : (any ??= anyJson()), minItems: s.minItems ?? 0, maxItems: s.maxItems ?? Infinity };
      case "object": {
        const required = new Set(s.required ?? []);
        const props: Prop[] = Object.entries(s.properties ?? {}).map(([k, v]) => ({
          key: utf8.encode(JSON.stringify(k).slice(1)),
          value: compile(v),
          required: required.has(k),
        }));
        const extra = s.additionalProperties ?? !s.properties;
        return { kind: "object", props, additional: extra === false ? undefined : compile(extra) };
      }
    }
  };

  return compile(root);
}

export function compileFormat(f: ResponseFormat): Node {
  if (f.type === "json") return f.schema ? compileSchema(f.schema) : anyJson();
  if (f.type === "choice") {
    if (!f.choices.length) throw new Error("choice format needs at least one choice");
    return { kind: "alt", nodes: f.choices.map(lit) };
  }
  throw new Error(`unknown response format ${JSON.stringify(f)}`);
}
