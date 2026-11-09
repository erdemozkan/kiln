import { roundUp, type Backend, type Buf, type BindingKind, type Dispatch, type Grid, type KernelSpec, type Pipeline } from "./backend";
import { byteSize, type DType, type Tensor } from "./tensor";
import { packUniform, type Field } from "./uniform";

/**
 * An intermediate recorded by `Ctx.alloc`. It has no storage until the plan is compiled, when the
 * memory planner points it at a buffer of the shared `ActivationPool`.
 */
export class PlannedBuf implements Buf {
  /** The pool buffer this intermediate lives in, set when its plan compiles. */
  target: Buf | undefined;
  constructor(
    readonly size: number,
    readonly label: string,
    /** Working memory of the dispatches that bind it; never a result, even when written last. */
    readonly scratch = false,
  ) {}
}

/** The buffer that backs `buf`: itself, or a compiled intermediate's pool buffer. */
export function physical(buf: Buf): Buf {
  if (!(buf instanceof PlannedBuf)) return buf;
  if (!buf.target) throw new Error(`${buf.label}: intermediate used before its plan was compiled`);
  return buf.target;
}

/**
 * Storage for plan intermediates, shared by every plan that is given it. All plans run on one queue
 * and no intermediate outlives its plan, so the buffers are free again whenever a plan finishes.
 */
export class ActivationPool {
  readonly buffers: Buf[] = [];

  constructor(
    readonly backend: Backend,
    readonly label = "act",
  ) {}

  get bytes(): number {
    return this.buffers.reduce((n, b) => n + b.size, 0);
  }

  add(bytes: number): Buf {
    const buf = this.backend.alloc(bytes, "storage", `${this.label}/act`);
    this.buffers.push(buf);
    return buf;
  }

  destroy(): void {
    for (const b of this.buffers) this.backend.free(b);
    this.buffers.length = 0;
  }
}

interface Pending {
  pipeline: Pipeline;
  buffers: readonly Buf[];
  kinds: readonly BindingKind[];
  grid: Grid;
  label: string;
}

/**
 * Records a forward pass into a fixed dispatch list ("compile once, replay many").
 *
 * Intermediates from `alloc` are placeholders. Reading `dispatches` compiles the plan: each
 * intermediate's lifetime is the span of dispatches that bind it, and the planner packs them into as
 * few pool buffers as possible, largest first, so intermediates whose lifetimes do not overlap share a
 * buffer. Two intermediates bound by the same dispatch are both live there, so a dispatch never sees
 * one buffer twice. One written by its last dispatch is a result the caller may read back, and stays
 * live to the end of the plan (except `scratch` memory). WebGPU orders storage access between dispatches, which makes the reuse
 * safe within a pass and, since the pool is only reused by later plans on the same queue, across plans.
 */
export class Ctx {
  private readonly pending: Pending[] = [];
  private compiled: Dispatch[] | undefined;
  private readonly owned: Buf[] = [];
  private readonly uniforms = new Map<string, Buf>();
  private readonly pool: ActivationPool;
  private readonly ownsPool: boolean;

  constructor(
    readonly backend: Backend,
    /** Step state (see `packStep`), written before every replay. */
    readonly step: Buf,
    readonly label = "plan",
    /** Pool to share with other plans; by default the plan gets its own. */
    pool?: ActivationPool,
  ) {
    this.pool = pool ?? new ActivationPool(backend, label);
    this.ownsPool = !pool;
  }

  /** The recorded dispatches. The first read compiles the plan; nothing can be recorded after it. */
  get dispatches(): readonly Dispatch[] {
    this.compiled ??= this.compile();
    return this.compiled;
  }

  alloc(shape: readonly number[], dtype: DType = "f32"): Tensor {
    if (dtype === "q4" || dtype === "q8") throw new Error(`Ctx.alloc: ${dtype} intermediates are not supported`);
    this.assertOpen();
    return { shape, dtype, buf: new PlannedBuf(roundUp(byteSize(shape, dtype), 16), `${this.label}/act`) };
  }

  /** Working memory a kernel writes and reads within its own dispatch, free again right after it. */
  scratch(shape: readonly number[], dtype: DType = "f32"): Tensor {
    this.assertOpen();
    return { shape, dtype, buf: new PlannedBuf(roundUp(byteSize(shape, dtype), 16), `${this.label}/scratch`, true) };
  }

  /** A buffer that is never shared (plan inputs and outputs). */
  buffer(bytes: number, label: string): Buf {
    return this.newBuf(roundUp(bytes, 16), `${this.label}/${label}`);
  }

  /** Static per-dispatch parameters. Identical contents share one buffer. */
  uniform(fields: readonly Field[]): Buf {
    const key = fields.map((f) => f.join(":")).join(",");
    let buf = this.uniforms.get(key);
    if (!buf) {
      const data = packUniform(fields);
      buf = this.backend.alloc(data.byteLength, "uniform", `${this.label}/uniform`);
      this.backend.write(buf, 0, data);
      this.uniforms.set(key, buf);
      this.owned.push(buf);
    }
    return buf;
  }

  run(spec: KernelSpec, buffers: readonly Buf[], grid: Grid, label = spec.key): void {
    this.assertOpen();
    if (buffers.length !== spec.bindings.length) {
      throw new Error(`${spec.key}: expected ${spec.bindings.length} buffers, got ${buffers.length}`);
    }
    if (grid.some((g) => g === 0)) return;
    const max = this.backend.limits.maxComputeWorkgroupsPerDimension;
    if (grid.some((g) => g > max)) throw new Error(`${spec.key}: grid [${grid}] exceeds ${max} per dimension`);
    this.pending.push({ pipeline: this.backend.pipeline(spec), buffers, kinds: spec.bindings, grid, label });
  }

  /** Dispatches recorded so far (the index the next one will have). */
  get size(): number {
    return this.pending.length;
  }

  /** Bytes of pool buffers this plan binds. */
  get activationBytes(): number {
    const used = new Set<Buf>();
    for (const p of this.pending) for (const b of p.buffers) if (b instanceof PlannedBuf && b.target) used.add(b.target);
    return [...used].reduce((n, b) => n + b.size, 0);
  }

  destroy(): void {
    for (const b of this.owned) this.backend.free(b);
    this.owned.length = 0;
    this.uniforms.clear();
    if (this.ownsPool) this.pool.destroy();
  }

  private compile(): Dispatch[] {
    this.place();
    return this.pending.map((p) => ({ binding: this.backend.bind(p.pipeline, p.buffers.map(physical)), grid: p.grid, label: p.label }));
  }

  /** Assigns every intermediate a pool buffer (see the class comment). */
  private place(): void {
    const life = new Map<PlannedBuf, { first: number; last: number; written: boolean }>();
    this.pending.forEach((p, i) =>
      p.buffers.forEach((b, j) => {
        if (!(b instanceof PlannedBuf)) return;
        const l = life.get(b);
        const written = p.kinds[j] === "rw";
        if (!l) life.set(b, { first: i, last: i, written });
        else if (i > l.last) Object.assign(l, { last: i, written });
        else l.written ||= written;
      }),
    );
    const end = this.pending.length;
    for (const [b, l] of life) if (l.written && !b.scratch) l.last = end;
    // Intervals already placed in each pool buffer by this plan.
    const taken = new Map<Buf, { first: number; last: number }[]>();
    // Dispatches between l and the nearest interval in buf (Infinity if buf is unused here), or -1 on overlap.
    const gap = (buf: Buf, l: { first: number; last: number }) => {
      let g = Infinity;
      for (const t of taken.get(buf) ?? []) {
        if (t.last >= l.first && l.last >= t.first) return -1;
        g = Math.min(g, t.last < l.first ? l.first - t.last : t.first - l.last);
      }
      return g;
    };
    // Largest first; each goes to the smallest buffer that fits and is free for its lifetime, and among
    // those to the one whose other tenants are nearest in time, leaving long free spans whole.
    const order = [...life].sort(([a, la], [b, lb]) => b.size - a.size || la.first - lb.first);
    for (const [vb, l] of order) {
      let best: { buf: Buf; gap: number } | undefined;
      for (const buf of this.pool.buffers) {
        if (buf.size < vb.size) continue;
        const g = gap(buf, l);
        if (g >= 0 && (!best || buf.size < best.buf.size || (buf.size === best.buf.size && g < best.gap))) best = { buf, gap: g };
      }
      const buf = best?.buf ?? this.pool.add(vb.size);
      taken.set(buf, [...(taken.get(buf) ?? []), l]);
      vb.target = buf;
    }
  }

  private assertOpen(): void {
    if (this.compiled) throw new Error(`${this.label}: plan is already compiled`);
  }

  private newBuf(bytes: number, label: string): Buf {
    const buf = this.backend.alloc(bytes, "storage", label);
    this.owned.push(buf);
    return buf;
  }
}
