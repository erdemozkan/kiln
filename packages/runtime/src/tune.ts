import { Ctx, byteSize, type Backend, type Tensor, type WeightDType } from "@kiln/core";
import { ops, type MatmulLanes } from "@kiln/kernels";
import { readCacheJSON, writeCacheJSON } from "@kiln/loader";
import type { LlamaConfig } from "@kiln/models";

/** Bump when matmul kernels change so stale choices are re-measured. */
const VERSION = 1;
const CACHE_FILE = "matmul-tuning.json";
const LANES: readonly MatmulLanes[] = [8, 16, 32];
/** Scratch weights above this size are not tuned (keeps tuning memory small); they keep the default. */
const MAX_SCRATCH = 64 * 2 ** 20;

export interface MatmulShape {
  wdtype: WeightDType;
  mt: 1 | 8;
  N: number;
  K: number;
  norm?: boolean;
  silu?: boolean;
}

/** The matmuls a Llama forward pass runs, for decode (mt 1) and prefill (mt 8). */
export function llamaMatmulShapes(cfg: LlamaConfig, wdtype: WeightDType): MatmulShape[] {
  const { hiddenSize: Hd, intermediateSize: I, numHeads: H, numKvHeads: Hkv, headDim: D } = cfg;
  const out: MatmulShape[] = [];
  for (const mt of [1, 8] as const) {
    if (cfg.qkNorm) out.push({ wdtype, mt, N: H * D, K: Hd }, { wdtype, mt, N: Hkv * D, K: Hd });
    else out.push({ wdtype, mt, N: (H + 2 * Hkv) * D, K: Hd, norm: true });
    out.push({ wdtype, mt, N: Hd, K: H * D }, { wdtype, mt, N: 2 * I, K: Hd, norm: true, silu: true }, { wdtype, mt, N: Hd, K: I });
  }
  out.push({ wdtype, mt: 1, N: cfg.vocabSize, K: Hd, norm: true });
  return out;
}

/**
 * Picks lanes per weight row for every shape by timing each variant on this GPU, on scratch buffers
 * (weights' values do not matter for speed, so this can run while the real weights download).
 * Results are cached per device in OPFS; a cache hit applies them without touching the GPU.
 */
export async function tuneMatmuls(be: Backend, shapes: readonly MatmulShape[]): Promise<Record<string, MatmulLanes>> {
  const cacheKey = `v${VERSION}:${be.deviceId ?? "unknown"}`;
  const cache = (await readCacheJSON<Record<string, Record<string, MatmulLanes>>>(CACHE_FILE)) ?? {};
  const known = cache[cacheKey] ?? {};
  const todo = shapes.filter((s) => !(ops.lanesKey(s.wdtype, s.mt, s.N, s.K) in known));
  for (const s of todo) {
    const lanes = await fastest(be, s);
    if (lanes) known[ops.lanesKey(s.wdtype, s.mt, s.N, s.K)] = lanes;
  }
  for (const [k, v] of Object.entries(known)) ops.setMatmulLanes(k, v);
  if (todo.length) await writeCacheJSON(CACHE_FILE, { ...cache, [cacheKey]: known });
  return known;
}

async function fastest(be: Backend, s: MatmulShape): Promise<MatmulLanes | undefined> {
  const wBytes = byteSize([s.N, s.K], s.wdtype);
  const quant = s.wdtype !== "f32";
  if (wBytes > MAX_SCRATCH) return undefined;
  const M = s.mt === 1 ? 1 : 64;
  const bufs = [be.alloc(M * s.K * 4), be.alloc(wBytes), be.alloc(quant ? (s.N * s.K) / 8 : 4), be.alloc(s.K * 4), be.alloc(32)];
  const [xb, wb, sb, nb, step] = bufs as [typeof bufs[0], typeof bufs[0], typeof bufs[0], typeof bufs[0], typeof bufs[0]];
  const x: Tensor = { shape: [M, s.K], dtype: "f32", buf: xb };
  const w: Tensor = quant ? { shape: [s.N, s.K], dtype: s.wdtype, buf: wb, scales: sb } : { shape: [s.N, s.K], dtype: "f32", buf: wb };
  const norm = s.norm ? { w: { shape: [s.K], dtype: "f32" as const, buf: nb }, eps: 1e-5 } : undefined;
  // Enough repeats that the timed work dwarfs submit overhead, scaled down for big shapes.
  const reps = Math.max(4, Math.min(30, Math.round((8 * 2 ** 20) / wBytes)));
  let best: { lanes: MatmulLanes; ms: number } | undefined;
  try {
    for (const lanes of LANES) {
      const ctx = new Ctx(be, step, "tune");
      let y: Tensor | undefined;
      for (let i = 0; i < reps; i++) y = ops.linear(ctx, x, w, { lanes, norm, silu: s.silu });
      const fence = () => be.readback(y!.buf, 0, 4);
      be.submit(ctx.dispatches);
      await fence();
      let ms = Infinity;
      for (let t = 0; t < 3; t++) {
        const t0 = performance.now();
        be.submit(ctx.dispatches);
        await fence();
        ms = Math.min(ms, performance.now() - t0);
      }
      ctx.destroy();
      if (!best || ms < best.ms * 0.97) best = { lanes, ms };
    }
  } finally {
    for (const b of bufs) be.free(b);
  }
  return best?.lanes;
}
