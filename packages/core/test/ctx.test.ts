import { describe, expect, it } from "vitest";
import { ActivationPool, Ctx, f32, packStep, packUniform, u32, type Backend, type Buf, type KernelSpec, physical } from "../src";

function fakeBackend() {
  const allocs: Buf[] = [];
  const writes: [Buf, Uint8Array][] = [];
  const backend: Backend = {
    name: "fake",
    limits: { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 30, maxComputeWorkgroupsPerDimension: 65535, features: new Set() },
    alloc: (size, _u, label = "buf") => {
      const b = { size, label };
      allocs.push(b);
      return b;
    },
    free: () => {},
    write: (buf, _o, data) => void writes.push([buf, new Uint8Array(data.buffer, data.byteOffset, data.byteLength)]),
    copy: () => {},
    pipeline: (spec) => ({ key: spec.key }),
    bind: (pipeline) => ({ pipeline }),
    submit: () => {},
    readback: async () => new ArrayBuffer(4),
    destroy: () => {},
  };
  return { backend, allocs, writes };
}

const SPEC: KernelSpec = { key: "k", code: "", bindings: ["uniform", "rw"] };

describe("uniform packing", () => {
  it("pads to 16 bytes and encodes types", () => {
    const b = packUniform([u32(7), f32(0.5)]);
    expect(b.byteLength).toBe(16);
    const dv = new DataView(b.buffer);
    expect(dv.getUint32(0, true)).toBe(7);
    expect(dv.getFloat32(4, true)).toBe(0.5);
    expect(packStep({ pos0: 1, lastRow: 2, seed: 3, temperature: 0.7 }).byteLength).toBe(32);
  });
});

describe("Ctx", () => {
  // One "rw" output from one "read" input per dispatch, like an elementwise op.
  const OP: KernelSpec = { key: "op", code: "", bindings: ["read", "rw"] };
  const phys = (t: { buf: Buf }) => physical(t.buf);

  it("shares a buffer between intermediates whose lifetimes do not overlap", () => {
    const { backend } = fakeBackend();
    const ctx = new Ctx(backend, backend.alloc(16, "uniform"));
    const input = ctx.buffer(128, "in");
    const a = ctx.alloc([4, 8]);
    const b = ctx.alloc([8, 4]);
    const c = ctx.alloc([2, 8]);
    const d = ctx.alloc([4, 8]);
    ctx.run(OP, [input, a.buf], [1, 1, 1]);
    ctx.run(OP, [a.buf, b.buf], [1, 1, 1]);
    ctx.run(OP, [b.buf, c.buf], [1, 1, 1]);
    ctx.run(OP, [c.buf, d.buf], [1, 1, 1]);
    ctx.run(OP, [d.buf, ctx.buffer(128, "out")], [1, 1, 1]);
    expect(ctx.dispatches).toHaveLength(5);
    // a and c never overlap, nor b and d; neighbours (bound together) never share.
    expect(phys(c)).toBe(phys(a));
    expect(phys(d)).toBe(phys(b));
    expect(phys(a)).not.toBe(phys(b));
    expect(ctx.activationBytes).toBe(2 * 128);
  });

  it("keeps results written by their last dispatch, but not scratch", () => {
    const { backend } = fakeBackend();
    const ctx = new Ctx(backend, backend.alloc(16, "uniform"));
    const input = ctx.buffer(64, "in");
    const result = ctx.alloc([16]);
    const s = ctx.scratch([16]);
    const later = ctx.alloc([16]);
    const s2 = ctx.scratch([16]);
    ctx.run(OP, [input, result.buf], [1, 1, 1]);
    ctx.run(OP, [input, s.buf], [1, 1, 1]);
    ctx.run(OP, [input, later.buf], [1, 1, 1]);
    ctx.run(OP, [input, s2.buf], [1, 1, 1]);
    void ctx.dispatches;
    // Results stay live to the end, so nothing after them reuses their buffers; scratch is free after its dispatch.
    expect(phys(later)).toBe(phys(s));
    expect(new Set([result, later, s2].map(phys)).size).toBe(3);
  });

  it("shares one pool across plans and freezes compiled plans", () => {
    const { backend } = fakeBackend();
    const step = backend.alloc(16, "uniform");
    const pool = new ActivationPool(backend);
    const build = (n: number) => {
      const ctx = new Ctx(backend, step, `T${n}`, pool);
      const x = ctx.alloc([n]);
      const y = ctx.alloc([n]);
      ctx.run(OP, [ctx.buffer(4 * n, "in"), x.buf], [1, 1, 1]);
      ctx.run(OP, [x.buf, y.buf], [1, 1, 1]);
      ctx.run(OP, [y.buf, ctx.buffer(4 * n, "out")], [1, 1, 1]);
      void ctx.dispatches;
      return ctx;
    };
    const big = build(64);
    const small = build(16);
    expect(pool.buffers).toHaveLength(2);
    expect(pool.bytes).toBe(2 * 256);
    expect(() => big.alloc([4])).toThrow(/compiled/);
    expect(() => physical(new Ctx(backend, step).alloc([4]).buf)).toThrow(/before its plan was compiled/);
    small.destroy();
    big.destroy();
    expect(pool.buffers).toHaveLength(2);
    pool.destroy();
    expect(pool.buffers).toHaveLength(0);
  });

  it("dedupes identical uniforms and records dispatches", () => {
    const { backend, writes } = fakeBackend();
    const ctx = new Ctx(backend, backend.alloc(16, "uniform"));
    const out = ctx.alloc([4]);
    const u1 = ctx.uniform([u32(1), u32(2)]);
    expect(ctx.uniform([u32(1), u32(2)])).toBe(u1);
    expect(writes).toHaveLength(1);
    ctx.run(SPEC, [u1, out.buf], [2, 1, 1]);
    ctx.run(SPEC, [u1, out.buf], [0, 1, 1]);
    expect(ctx.dispatches).toHaveLength(1);
  });

  it("validates binding counts and grid limits", () => {
    const { backend } = fakeBackend();
    const ctx = new Ctx(backend, backend.alloc(16, "uniform"));
    const out = ctx.alloc([4]);
    expect(() => ctx.run(SPEC, [out.buf], [1, 1, 1])).toThrow(/expected 2 buffers/);
    expect(() => ctx.run(SPEC, [out.buf, out.buf], [70000, 1, 1])).toThrow(/exceeds/);
  });
});
