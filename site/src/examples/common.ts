// Shared by the example pages: the layout, element lookup, status lines, load progress and errors.
import "../layout.ts";
import type { LoadProgress, ProgressInfo } from "kiln-webgpu";

export const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

/** Shows `text` in the page's status line. */
export function status(text: string): void {
  $("status").textContent = text;
}

/** A load progress handler that fills the page's progress bar and status line. */
export function progress(label: string): (p: LoadProgress) => void {
  const bar = $<HTMLProgressElement>("progress");
  bar.hidden = false;
  return (p) => {
    if (p.total) bar.value = p.loaded / p.total;
    status(`${label}: ${p.file} ${p.total ? `${Math.round((100 * p.loaded) / p.total)}%` : ""}`);
    if (p.total && p.loaded >= p.total) bar.hidden = true;
  };
}

/** The same for a pipeline's `progress_callback` (transformers.js's progress events). */
export function pipelineProgress(label: string): (p: ProgressInfo) => void {
  const bar = $<HTMLProgressElement>("progress");
  return (p) => {
    if (p.status === "progress") {
      bar.hidden = false;
      bar.value = (p.progress ?? 0) / 100;
      status(`${label}: ${p.file} ${Math.round(p.progress ?? 0)}%`);
    } else if (p.status === "done" || p.status === "ready") bar.hidden = true;
  };
}

/** Runs `fn` with `button` disabled, reporting errors in the status line. */
export async function busy(button: HTMLButtonElement, fn: () => Promise<void>): Promise<void> {
  button.disabled = true;
  try {
    await fn();
  } catch (err) {
    status(`Error: ${err instanceof Error ? err.message : String(err)}`);
    console.error(err);
  } finally {
    button.disabled = false;
  }
}

/** Notes where the example will run (WebGPU or the slower CPU fallback). */
export function backendNote(): void {
  const gpu = "gpu" in navigator;
  const note = document.createElement("p");
  note.className = "muted";
  note.textContent = gpu
    ? "Runs on your GPU with WebGPU. The first run downloads the model and caches it in the browser."
    : "This browser has no WebGPU, so the model runs on the CPU: expect it to be slow.";
  $("status").before(note);
}
