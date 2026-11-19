// The Node.js side of the Python package: reads requests from stdin, runs them on Kiln, writes the
// replies to stdout. A message is a frame: a 4-byte little-endian length, that many bytes of JSON
// (the header), then the header's `blobs` (byte lengths) as raw bytes one after another. Arrays of
// numbers travel as blobs, not as JSON.
//
// Started as: node bridge.mjs <path or URL of Kiln's index.js>
import { pathToFileURL } from "node:url";

// stdout carries frames only: anything else printed in this process goes to stderr.
const out = process.stdout.write.bind(process.stdout);
process.stdout.write = (...args) => process.stderr.write(...args);

const entry = process.argv[2];
if (!entry) throw new Error("bridge.mjs: give the path of Kiln's index.js");
const kiln = await import(/^[a-z]+:\/\//i.test(entry) ? entry : pathToFileURL(entry).href);

const ARRAYS = {
  float32: Float32Array,
  float64: Float64Array,
  int8: Int8Array,
  uint8: Uint8Array,
  uint8c: Uint8ClampedArray,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
};
const NAMES = new Map(Object.entries(ARRAYS).map(([name, ctor]) => [ctor, name]));

function send(header, blobs = []) {
  header.blobs = blobs.map((b) => b.byteLength);
  const json = Buffer.from(JSON.stringify(header), "utf8");
  const len = Buffer.alloc(4);
  len.writeUInt32LE(json.length);
  out(len);
  out(json);
  for (const b of blobs) out(b);
}

/** A result as JSON, its typed arrays moved to `blobs`. Functions are left out. */
function encode(v, blobs, seen = new Set()) {
  if (v === null || v === undefined) return null;
  switch (typeof v) {
    case "number":
      return Number.isFinite(v) ? v : { $float: String(v) };
    case "string":
    case "boolean":
      return v;
    case "bigint":
      return { $bigint: v.toString() };
    case "function":
    case "symbol":
      return undefined;
  }
  if (ArrayBuffer.isView(v) && NAMES.has(v.constructor)) {
    blobs.push(Buffer.from(v.buffer, v.byteOffset, v.byteLength));
    return { $blob: blobs.length - 1, dtype: NAMES.get(v.constructor) === "uint8c" ? "uint8" : NAMES.get(v.constructor) };
  }
  if (v instanceof ArrayBuffer) {
    blobs.push(Buffer.from(v));
    return { $blob: blobs.length - 1, dtype: "uint8" };
  }
  if (seen.has(v)) return null;
  seen.add(v);
  try {
    if (Array.isArray(v)) return v.map((x) => encode(x, blobs, seen) ?? null);
    if (v instanceof Map) return Object.fromEntries([...v].map(([k, x]) => [String(k), encode(x, blobs, seen) ?? null]));
    if (v instanceof Set) return [...v].map((x) => encode(x, blobs, seen) ?? null);
    if (v instanceof Date) return v.toISOString();
    if (v instanceof Error) return { message: v.message };
    const o = {};
    for (const k of Object.keys(v)) {
      const x = encode(v[k], blobs, seen);
      if (x !== undefined) o[k] = x;
    }
    return o;
  } finally {
    seen.delete(v);
  }
}

/** A request's value back from JSON: blobs as typed arrays, callbacks as functions that message Python. */
function decode(v, blobs, key) {
  if (v === null || typeof v !== "object") return v ?? undefined;
  if (Array.isArray(v)) return v.map((x) => decode(x, blobs));
  if ("$blob" in v) {
    const b = blobs[v.$blob];
    const Ctor = ARRAYS[v.dtype];
    if (!Ctor) throw new Error(`unknown array type "${v.dtype}"`);
    // A copy at offset 0: the frame's bytes may sit at an offset the element size does not divide.
    const copy = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    return new Ctor(copy, 0, copy.byteLength / Ctor.BYTES_PER_ELEMENT);
  }
  if ("$callback" in v) {
    const id = v.$callback;
    const call = (...args) => {
      const bl = [];
      send({ event: "callback", callback: id, args: encode(args, bl) }, bl);
    };
    // Kiln's streamer is an object with put and end (transformers.js's TextStreamer).
    return key === "streamer" ? { put: (text) => text && call(text), end() {} } : call;
  }
  if ("$bigint" in v) return BigInt(v.$bigint);
  if ("$float" in v) return Number(v.$float);
  const o = {};
  for (const k of Object.keys(v)) o[k] = decode(v[k], blobs, k);
  return o;
}

const handles = new Map();
let nextHandle = 1;

const ops = {
  hello: () => ({ node: process.versions.node, platform: process.platform, arch: process.arch, tasks: Object.keys(kiln.DEFAULT_MODELS ?? {}) }),
  async pipeline({ task, model, options }) {
    const p = await kiln.pipeline(task, model ?? undefined, options ?? {});
    const handle = nextHandle++;
    handles.set(handle, p);
    return { handle, task: p.task, info: p.info };
  },
  async call({ handle, args }) {
    const p = handles.get(handle);
    if (!p) throw new Error("this pipeline was disposed");
    return p(...args);
  },
  async dispose({ handle }) {
    const p = handles.get(handle);
    handles.delete(handle);
    await p?.dispose?.();
    return null;
  },
  /** One of Kiln's exported functions by name (cos_sim, setAccessToken, chunkText and the like). */
  async invoke({ name, args }) {
    const f = kiln[name];
    if (typeof f !== "function") throw new Error(`Kiln exports no function "${name}"`);
    return f(...args);
  },
  /** One of Kiln's exported strings by name (SKILL_INSTRUCTIONS). */
  constant({ name }) {
    if (typeof kiln[name] !== "string") throw new Error(`Kiln exports no text "${name}"`);
    return kiln[name];
  },
  /** Kiln's built-in skills, kept here to run: their specs go to Python, which calls `skill` for each call. */
  skills({ only, options }) {
    const { allowPrivate, ...opts } = options ?? {};
    if (allowPrivate) opts.allowUrl = () => true;
    const make = { calculate: kiln.calculatorSkill, get_date_time: () => kiln.dateTimeSkill(), search_wikipedia: kiln.wikipediaSkill, read_web_page: kiln.webPageSkill };
    const names = only ?? Object.keys(make);
    const tools = names.map((n) => {
      if (!Object.hasOwn(make, n)) throw new Error(`Kiln has no skill "${n}"`);
      return make[n](opts);
    });
    const handle = nextHandle++;
    handles.set(handle, new Map(tools.map((t) => [t.function.name, t])));
    return { handle, specs: tools.map((t) => ({ type: t.type, function: t.function })) };
  },
  async skill({ handle, name, args }) {
    const tool = handles.get(handle)?.get?.(name);
    if (!tool) throw new Error(`no skill "${name}" here`);
    return tool.execute(args ?? {}, {});
  },
  /** Sets fields of Kiln's `env`. */
  env({ values }) {
    Object.assign(kiln.env, values);
    return null;
  },
  /** For the package's own tests: the arguments back, after calling every function among them with them. */
  echo({ args }) {
    for (const a of args) if (typeof a === "function") a(...args.filter((x) => typeof x !== "function"));
    return args;
  },
};

let queue = Promise.resolve();
function handle(header, blobs) {
  queue = queue.then(async () => {
    const id = header.id;
    try {
      const op = ops[header.op];
      if (!op) throw new Error(`unknown request "${header.op}"`);
      const value = await op(decode(header.body ?? {}, blobs));
      const bl = [];
      send({ id, ok: true, value: encode(value, bl) ?? null }, bl);
    } catch (err) {
      send({ id, ok: false, error: { message: err instanceof Error ? err.message : String(err), stack: err instanceof Error ? err.stack : undefined } });
    }
  });
}

// Chunks are joined only once a whole frame may be there, so a large frame is not copied per chunk.
let chunks = [];
let have = 0;
let need = 4;
process.stdin.on("data", (chunk) => {
  chunks.push(chunk);
  have += chunk.length;
  while (have >= need) {
    let buffered = chunks.length === 1 ? chunks[0] : Buffer.concat(chunks);
    const n = buffered.readUInt32LE(0);
    let total = 4 + n;
    let header;
    if (buffered.length >= total) {
      header = JSON.parse(buffered.toString("utf8", 4, 4 + n));
      total += header.blobs.reduce((a, b) => a + b, 0);
    }
    if (buffered.length < total) {
      chunks = [buffered];
      need = total;
      return;
    }
    const blobs = [];
    let at = 4 + n;
    for (const size of header.blobs) {
      blobs.push(buffered.subarray(at, at + size));
      at += size;
    }
    buffered = buffered.subarray(total);
    chunks = buffered.length ? [buffered] : [];
    have = buffered.length;
    need = 4;
    handle(header, blobs);
  }
});
// Python closes stdin when it is done (or has died): finish the request in hand, then leave.
process.stdin.on("end", () => void queue.then(() => process.exit(0)));
