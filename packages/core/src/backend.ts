/**
 * Hardware abstraction layer. Everything above this file is backend-agnostic;
 * a backend only has to allocate buffers, compile kernels and run dispatch lists.
 */

export type BindingKind = "uniform" | "read" | "rw";

export interface KernelSpec {
  /** Unique cache key. Must change whenever `code` changes (include every specialization). */
  readonly key: string;
  readonly code: string;
  /** One entry per `@binding(i)` in `@group(0)`, in order. */
  readonly bindings: readonly BindingKind[];
  readonly entry?: string;
}

/** Opaque buffer handle. Backends extend it with their native object. */
export interface Buf {
  readonly size: number;
  readonly label: string;
}

export interface Pipeline {
  readonly key: string;
}

/** A pipeline plus a fixed set of buffers, created once and replayed every step. */
export interface Binding {
  readonly pipeline: Pipeline;
}

export type Grid = readonly [number, number, number];

export interface Dispatch {
  readonly binding: Binding;
  readonly grid: Grid;
  readonly label: string;
}

export interface DeviceLimits {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly features: ReadonlySet<string>;
}

export type BufUsage = "storage" | "uniform";

export interface Backend {
  readonly name: string;
  /** Identifies the physical device (vendor, architecture, driver description) for per-device caches. */
  readonly deviceId?: string;
  readonly limits: DeviceLimits;
  alloc(bytes: number, usage?: BufUsage, label?: string): Buf;
  free(buf: Buf): void;
  /** Queue-ordered write: lands after everything submitted before it. */
  write(buf: Buf, offset: number, data: ArrayBufferView): void;
  copy(src: Buf, srcOffset: number, dst: Buf, dstOffset: number, bytes: number): void;
  pipeline(spec: KernelSpec): Pipeline;
  bind(pipeline: Pipeline, buffers: readonly Buf[]): Binding;
  submit(dispatches: readonly Dispatch[]): void;
  /** Enqueues the copy immediately (queue order) and resolves once the bytes are on the CPU. */
  readback(buf: Buf, offset: number, bytes: number): Promise<ArrayBuffer>;
  destroy(): void;
}

export const roundUp = (n: number, multiple: number): number => Math.ceil(n / multiple) * multiple;
