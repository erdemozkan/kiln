// PyTorch checkpoints (pytorch_model.bin), read as tensor byte ranges like a safetensors header: the
// zip format torch.save has written since 1.6, and the legacy format before it. Only the pickle that
// describes the tensors is parsed (a small subset of pickle, enough for state dicts); weights are read
// later straight from their offsets.
import type { ByteSource } from "./source";
import { UnsupportedTensorError, type TensorEntry, MAX_HEADER_BYTES, checkEntries } from "./entry";

type Float = "F32" | "F16" | "BF16";

/** Storage classes and their element sizes; float ones load, the rest are buffers and are skipped. */
const STORAGES: Record<string, { size: number; type?: Float }> = {
  FloatStorage: { size: 4, type: "F32" },
  HalfStorage: { size: 2, type: "F16" },
  BFloat16Storage: { size: 2, type: "BF16" },
  DoubleStorage: { size: 8 },
  LongStorage: { size: 8 },
  IntStorage: { size: 4 },
  ShortStorage: { size: 2 },
  CharStorage: { size: 1 },
  ByteStorage: { size: 1 },
  BoolStorage: { size: 1 },
};

/** A reference to a storage, as the pickle's persistent ids name them. */
interface StorageRef {
  kind: "storage";
  cls: string;
  key: string;
  numel: number;
}

interface TensorRef {
  kind: "tensor";
  storage: StorageRef;
  offset: number;
  shape: number[];
  stride: number[];
}

class Global {
  constructor(
    readonly module: string,
    readonly name: string,
  ) {}
}

const MARK = Symbol("mark");

/**
 * A pickle machine for torch state dicts: dicts, lists, tuples, numbers, strings, globals, and the
 * calls torch uses to rebuild tensors (anything else rebuilds to its arguments). Returns the value
 * and the offset just past STOP.
 */
export function unpickle(bytes: Uint8Array, start = 0, persistent: (pid: unknown) => unknown = (p) => p): { value: unknown; end: number } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const stack: unknown[] = [];
  const memo = new Map<number, unknown>();
  const td = new TextDecoder();
  let i = start;
  const need = (n: number) => {
    if (i + n > bytes.length) throw new RangeError("pickle: truncated");
  };
  const u8 = () => (need(1), bytes[i++]!);
  const u16 = () => (need(2), (i += 2), dv.getUint16(i - 2, true));
  const u32 = () => (need(4), (i += 4), dv.getUint32(i - 4, true));
  const take = (n: number) => (need(n), bytes.subarray((i += n) - n, i));
  const popMark = (): unknown[] => {
    let k = stack.length - 1;
    while (k >= 0 && stack[k] !== MARK) k--;
    if (k < 0) throw new Error("pickle: no mark");
    const items = stack.splice(k + 1);
    stack.pop();
    return items;
  };
  const call = (f: unknown, args: unknown[]): unknown => {
    if (f instanceof Global) {
      const n = `${f.module}.${f.name}`;
      if (n === "collections.OrderedDict" || n === "builtins.dict") return new Map();
      if (n === "torch._utils._rebuild_tensor_v2" || n === "torch._utils._rebuild_tensor") {
        const [storage, offset, shape, stride] = args as [StorageRef, number, number[], number[]];
        return { kind: "tensor", storage, offset, shape, stride } satisfies TensorRef;
      }
      if (n === "torch._utils._rebuild_parameter" || n === "torch._utils._rebuild_parameter_with_state") return args[0];
    }
    return args;
  };
  const setItems = (target: unknown, kv: unknown[]) => {
    if (target instanceof Map) for (let k = 0; k + 1 < kv.length; k += 2) target.set(kv[k], kv[k + 1]);
  };
  for (;;) {
    const op = u8();
    switch (op) {
      case 0x80: // PROTO
        u8();
        break;
      case 0x95: // FRAME
        take(8);
        break;
      case 0x7d: // EMPTY_DICT
        stack.push(new Map());
        break;
      case 0x5d: // EMPTY_LIST
        stack.push([]);
        break;
      case 0x29: // EMPTY_TUPLE
        stack.push([]);
        break;
      case 0x8f: // EMPTY_SET
        stack.push([]);
        break;
      case 0x28: // MARK
        stack.push(MARK);
        break;
      case 0x74: // TUPLE
      case 0x6c: // LIST
        stack.push(popMark());
        break;
      case 0x64: {
        // DICT
        const m = new Map();
        setItems(m, popMark());
        stack.push(m);
        break;
      }
      case 0x85: // TUPLE1
        stack.push([stack.pop()]);
        break;
      case 0x86: {
        const b = stack.pop();
        stack.push([stack.pop(), b]);
        break;
      }
      case 0x87: {
        const c = stack.pop();
        const b = stack.pop();
        stack.push([stack.pop(), b, c]);
        break;
      }
      case 0x4e: // NONE
        stack.push(null);
        break;
      case 0x88: // NEWTRUE
        stack.push(true);
        break;
      case 0x89: // NEWFALSE
        stack.push(false);
        break;
      case 0x4a: // BININT
        need(4);
        stack.push(dv.getInt32(i, true));
        i += 4;
        break;
      case 0x4b: // BININT1
        stack.push(u8());
        break;
      case 0x4d: // BININT2
        stack.push(u16());
        break;
      case 0x8a: {
        // LONG1
        const n = u8();
        const b = take(n);
        let v = 0n;
        for (let k = n - 1; k >= 0; k--) v = (v << 8n) | BigInt(b[k]!);
        if (n && b[n - 1]! & 0x80) v -= 1n << BigInt(8 * n);
        stack.push(Number(v));
        break;
      }
      case 0x47: // BINFLOAT (big-endian)
        need(8);
        stack.push(dv.getFloat64(i, false));
        i += 8;
        break;
      case 0x58: // BINUNICODE
        stack.push(td.decode(take(u32())));
        break;
      case 0x8c: // SHORT_BINUNICODE
        stack.push(td.decode(take(u8())));
        break;
      case 0x8d: // BINUNICODE8
        need(8);
        i += 8;
        stack.push(td.decode(take(Number(dv.getBigUint64(i - 8, true)))));
        break;
      case 0x54: // BINSTRING
        stack.push(td.decode(take(u32())));
        break;
      case 0x55: // SHORT_BINSTRING
        stack.push(td.decode(take(u8())));
        break;
      case 0x42: // BINBYTES
        stack.push(take(u32()));
        break;
      case 0x43: // SHORT_BINBYTES
        stack.push(take(u8()));
        break;
      case 0x63: {
        // GLOBAL "module\nname\n"
        const line = () => {
          let j = i;
          while (j < bytes.length && bytes[j] !== 0x0a) j++;
          if (j >= bytes.length) throw new RangeError("pickle: truncated");
          const s = td.decode(bytes.subarray(i, j));
          i = j + 1;
          return s;
        };
        const mod = line();
        stack.push(new Global(mod, line()));
        break;
      }
      case 0x93: {
        // STACK_GLOBAL
        const name = stack.pop() as string;
        stack.push(new Global(stack.pop() as string, name));
        break;
      }
      case 0x71: // BINPUT
        memo.set(u8(), stack.at(-1));
        break;
      case 0x72: // LONG_BINPUT
        memo.set(u32(), stack.at(-1));
        break;
      case 0x94: // MEMOIZE
        memo.set(memo.size, stack.at(-1));
        break;
      case 0x68: // BINGET
        stack.push(memo.get(u8()));
        break;
      case 0x6a: // LONG_BINGET
        stack.push(memo.get(u32()));
        break;
      case 0x52: {
        // REDUCE
        const args = stack.pop() as unknown[];
        stack.push(call(stack.pop(), args));
        break;
      }
      case 0x81: {
        // NEWOBJ
        const args = stack.pop() as unknown[];
        stack.push(call(stack.pop(), args));
        break;
      }
      case 0x62: // BUILD (state on an object: kept as is)
        stack.pop();
        break;
      case 0x73: {
        // SETITEM
        const v = stack.pop();
        const k = stack.pop();
        setItems(stack.at(-1), [k, v]);
        break;
      }
      case 0x75: {
        // SETITEMS
        const kv = popMark();
        setItems(stack.at(-1), kv);
        break;
      }
      case 0x61: {
        // APPEND
        const v = stack.pop();
        (stack.at(-1) as unknown[]).push(v);
        break;
      }
      case 0x65: {
        // APPENDS
        const items = popMark();
        (stack.at(-1) as unknown[]).push(...items);
        break;
      }
      case 0x90: {
        // ADDITEMS
        const items = popMark();
        (stack.at(-1) as unknown[]).push(...items);
        break;
      }
      case 0x51: // BINPERSID
        stack.push(persistent(stack.pop()));
        break;
      case 0x2e: // STOP
        return { value: stack.pop(), end: i };
      default:
        throw new Error(`pickle: opcode 0x${op.toString(16)} at ${i - 1} is not supported`);
    }
  }
}

/** ('storage', <class>, key, location, numel) → a StorageRef. */
function storageRef(pid: unknown): StorageRef {
  const t = pid as [string, Global | string, string, string, number];
  if (!Array.isArray(t) || t[0] !== "storage") throw new Error("pytorch: unexpected persistent id");
  const cls = t[1] instanceof Global ? t[1].name : String(t[1]);
  return { kind: "storage", cls, key: String(t[2]), numel: Number(t[4]) };
}

/**
 * Name → tensor from a state dict ("state_dict" / "model" wrappers unwrapped). A dict of state dicts
 * (Kokoro's: one per module) is flattened with its keys as prefixes: "bert.module.embeddings…".
 */
function tensorsOf(value: unknown): Map<string, TensorRef> {
  let v = value;
  for (const k of ["state_dict", "model"]) if (v instanceof Map && v.has(k) && v.get(k) instanceof Map) v = v.get(k);
  if (!(v instanceof Map)) throw new Error("pytorch: the checkpoint is not a state dict");
  const out = new Map<string, TensorRef>();
  const walk = (m: Map<unknown, unknown>, prefix: string) => {
    for (const [k, t] of m) {
      if (t instanceof Map) walk(t, `${prefix}${String(k)}.`);
      else if (t && (t as TensorRef).kind === "tensor") out.set(prefix + String(k), t as TensorRef);
    }
  };
  walk(v, "");
  return out;
}

function entries(tensors: Map<string, TensorRef>, storageOffset: (s: StorageRef) => number): TensorEntry[] {
  const out: TensorEntry[] = [];
  for (const [name, t] of tensors) {
    const info = Object.prototype.hasOwnProperty.call(STORAGES, t.storage.cls) ? STORAGES[t.storage.cls] : undefined;
    if (!info) throw new UnsupportedTensorError(`${name}: storage ${t.storage.cls} is not supported`);
    if (!info.type) continue;
    const numel = t.shape.reduce((a, b) => a * b, 1);
    // A transposed 2-D view saved as is (CLIP's projections): strides [1, rows].
    const transposed = t.shape.length === 2 && t.shape[0]! > 1 && t.shape[1]! > 1 && t.stride[0] === 1 && t.stride[1] === t.shape[0];
    let expect = 1;
    for (let d = t.shape.length - 1; d >= 0 && !transposed; d--) {
      if (t.shape[d]! > 1 && t.stride[d] !== expect) throw new UnsupportedTensorError(`${name}: non-contiguous tensors are not supported`);
      expect *= t.shape[d]!;
    }
    out.push({ name, shape: t.shape, type: info.type, offset: storageOffset(t.storage) + t.offset * info.size, byteLength: numel * info.size, ...(transposed ? { transposed } : {}) });
  }
  return out.sort((a, b) => a.offset - b.offset);
}

/** Reads `[0, n)` of a source, growing the window until `parse` stops running out of bytes. */
async function parseGrowing<T>(src: ByteSource, from: number, parse: (b: Uint8Array) => T, first = 1 << 20): Promise<T> {
  const total = await src.size();
  for (let n = first; ; n *= 4) {
    const len = Math.min(n, total - from, MAX_HEADER_BYTES);
    try {
      return parse(await src.read(from, len));
    } catch (e) {
      // Only a pickle that ran out of bytes grows the window (not a stack overflow, also a RangeError).
      if (!(e instanceof RangeError) || !/truncated/.test(e.message) || from + len >= total) throw e;
      if (len >= MAX_HEADER_BYTES) throw new Error(`${src.name}: pickle header larger than ${MAX_HEADER_BYTES / 2 ** 20} MB`);
    }
  }
}

/** Zip entries (name → data offset, size) from the central directory; stored entries only. */
async function zipEntries(src: ByteSource): Promise<Map<string, { at: number; size: number }>> {
  const total = await src.size();
  const tailLen = Math.min(total, 65536 + 22);
  const tail = await src.read(total - tailLen, tailLen);
  const tv = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
  let e = tail.length - 22;
  while (e >= 0 && tv.getUint32(e, true) !== 0x06054b50) e--;
  if (e < 0) throw new Error("pytorch: zip end record not found");
  let count = tv.getUint16(e + 10, true);
  let cdSize = tv.getUint32(e + 12, true);
  let cdAt = tv.getUint32(e + 16, true);
  if (cdAt === 0xffffffff || count === 0xffff) {
    // ZIP64: the locator sits just before the end record.
    const loc = e - 20;
    if (loc < 0 || tv.getUint32(loc, true) !== 0x07064b50) throw new Error("pytorch: zip64 locator not found");
    const rec = await src.read(Number(tv.getBigUint64(loc + 8, true)), 56);
    const rv = new DataView(rec.buffer, rec.byteOffset, rec.byteLength);
    count = Number(rv.getBigUint64(32, true));
    cdSize = Number(rv.getBigUint64(40, true));
    cdAt = Number(rv.getBigUint64(48, true));
  }
  // The directory must lie inside the file, with no more entries than its bytes can hold (46 each).
  if (![count, cdSize, cdAt].every(Number.isSafeInteger) || cdAt + cdSize > total || count * 46 > cdSize) throw new Error("pytorch: bad zip directory");
  const cd = await src.read(cdAt, cdSize);
  const cv = new DataView(cd.buffer, cd.byteOffset, cd.byteLength);
  const td = new TextDecoder();
  const out = new Map<string, { at: number; size: number }>();
  let p = 0;
  for (let k = 0; k < count; k++) {
    if (cv.getUint32(p, true) !== 0x02014b50) throw new Error("pytorch: bad zip directory");
    const method = cv.getUint16(p + 10, true);
    let size = cv.getUint32(p + 24, true);
    const nameLen = cv.getUint16(p + 28, true);
    const extraLen = cv.getUint16(p + 30, true);
    const commentLen = cv.getUint16(p + 32, true);
    let local = cv.getUint32(p + 42, true);
    const name = td.decode(cd.subarray(p + 46, p + 46 + nameLen));
    // ZIP64 extra field: the 0xffffffff values follow in order (sizes, then the local offset).
    for (let x = p + 46 + nameLen; x < p + 46 + nameLen + extraLen; ) {
      const id = cv.getUint16(x, true);
      const len = cv.getUint16(x + 2, true);
      if (id === 1) {
        let f = x + 4;
        if (cv.getUint32(p + 24, true) === 0xffffffff) (size = Number(cv.getBigUint64(f, true))), (f += 8);
        if (cv.getUint32(p + 20, true) === 0xffffffff) f += 8;
        if (local === 0xffffffff) local = Number(cv.getBigUint64(f, true));
      }
      x += 4 + len;
    }
    if (method !== 0) throw new Error(`pytorch: ${name} is compressed; only stored zip entries are supported`);
    if (!Number.isSafeInteger(local) || !Number.isSafeInteger(size) || local + 30 + size > total) throw new Error(`pytorch: zip entry ${name} lies outside the file`);
    out.set(name, { at: local, size });
    p += 46 + nameLen + extraLen + commentLen;
  }
  // Data starts after each local header (its name and extra lengths can differ from the directory's).
  // A few at a time: each is a small read, possibly its own range request.
  const all = [...out.values()];
  for (let k = 0; k < all.length; k += 16) {
    await Promise.all(
      all.slice(k, k + 16).map(async (v) => {
        const h = await src.read(v.at, 30);
        const hv = new DataView(h.buffer, h.byteOffset, h.byteLength);
        v.at += 30 + hv.getUint16(26, true) + hv.getUint16(28, true);
        if (v.at + v.size > total) throw new Error("pytorch: a zip entry runs past the end of the file");
      }),
    );
  }
  return out;
}

const LEGACY_MAGIC = 0x1950a86a20f9469cfc6c;

/** Tensor byte ranges of a pytorch_model.bin (zip or legacy format). */
export async function readPytorchHeader(src: ByteSource): Promise<{ tensors: TensorEntry[] }> {
  const header = await readPytorchTensors(src);
  checkEntries(src.name, header.tensors, await src.size());
  return header;
}

async function readPytorchTensors(src: ByteSource): Promise<{ tensors: TensorEntry[] }> {
  const head = await src.read(0, Math.min(4, await src.size()));
  if (head[0] === 0x50 && head[1] === 0x4b) {
    const files = await zipEntries(src);
    const pkl = [...files.keys()].find((n) => n.endsWith("/data.pkl") || n === "data.pkl");
    if (!pkl) throw new Error("pytorch: data.pkl not found in the archive");
    const root = pkl.slice(0, pkl.length - "data.pkl".length);
    const { at, size } = files.get(pkl)!;
    const { value } = unpickle(await src.read(at, size), 0, storageRef);
    return {
      tensors: entries(tensorsOf(value), (s) => {
        const f = files.get(`${root}data/${s.key}`);
        if (!f) throw new Error(`pytorch: storage ${s.key} not found in the archive`);
        return f.at;
      }),
    };
  }
  if (head[0] !== 0x80) throw new Error("pytorch: not a zip or pickle checkpoint");
  // Legacy: magic, protocol, sys info, the state dict, the storage keys, then each storage as an
  // 8-byte element count and its data, in key order.
  return parseGrowing(src, 0, (b) => {
    const magic = unpickle(b, 0);
    if (magic.value !== LEGACY_MAGIC) throw new Error("pytorch: bad legacy magic number");
    const proto = unpickle(b, magic.end);
    const sys = unpickle(b, proto.end);
    const storages = new Map<string, StorageRef>();
    const dict = unpickle(b, sys.end, (pid) => {
      const s = storageRef(pid);
      storages.set(s.key, s);
      return s;
    });
    const keys = unpickle(b, dict.end);
    const at = new Map<string, number>();
    let off = keys.end;
    for (const k of keys.value as string[]) {
      const s = storages.get(String(k));
      if (!s) throw new Error(`pytorch: storage ${String(k)} is not referenced`);
      at.set(s.key, off + 8);
      off += 8 + s.numel * (STORAGES[s.cls]?.size ?? 1);
    }
    return { tensors: entries(tensorsOf(dict.value), (s) => at.get(s.key)!) };
  });
}
