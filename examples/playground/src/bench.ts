import type { BenchResult } from "./bench.worker";
import { dashboard } from "./charts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const out = $("out");

async function bench(model: string, quant: "q4" | "q8" | "f32"): Promise<BenchResult> {
  const worker = new Worker(new URL("./bench.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise<BenchResult>((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ ok: boolean; result?: BenchResult; error?: string }>) =>
        e.data.ok ? resolve(e.data.result!) : reject(new Error(e.data.error));
      worker.postMessage({ model, quant });
    });
  } finally {
    worker.terminate();
  }
}

const LAST = "kiln-bench-last";
/** Rendered results, redrawn at the new width when the page is resized. */
const drawn = new Map<HTMLElement, BenchResult>();
let lastWidth = out.clientWidth;
let resizeTimer: ReturnType<typeof setTimeout> | undefined;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (out.clientWidth === lastWidth) return;
    lastWidth = out.clientWidth;
    for (const [section, r] of drawn) section.querySelector(".dash-host")!.innerHTML = dashboard(r, lastWidth);
  }, 150);
});

function render(r: BenchResult, at = new Date()): void {
  const section = document.createElement("section");
  const plans = r.plans
    .map(
      (p) => `<h2 style="font-size:16px">T=${p.T} at position ${p.pos0}: ${p.stepMs.toFixed(2)} ms per step${p.T > 1 ? ` (${((p.T / p.stepMs) * 1000).toFixed(0)} tok/s prefill)` : ` (${(1000 / p.stepMs).toFixed(1)} tok/s GPU-bound)`}</h2>
      <table><thead><tr><th>Kernel</th><th>Dispatches</th><th>ms</th><th>Share</th></tr></thead><tbody>
      ${p.kernels.map((k) => `<tr><td>${k.kernel}</td><td class="num">${k.dispatches}</td><td class="num">${k.ms.toFixed(3)}</td><td class="num">${((k.ms / p.stepMs) * 100).toFixed(0)}%</td></tr>`).join("")}
      </tbody></table>`,
    )
    .join("");
  section.innerHTML = `<p class="run-meta">${r.model} · ${r.quant} · ${at.toLocaleString()}</p><div class="dash-host">${dashboard(r, out.clientWidth)}</div>
    <details class="raw"><summary>Per-kernel tables and raw numbers</summary>${plans}
    ${(r.chained ?? []).map((c) => `<p class="status">${c.stepsPerSubmit} step(s) per submit: ${c.decodeTokS.toFixed(1)} tok/s · TTFT ${c.ttftMs.toFixed(0)} ms · ${c.sameIds ? "same ids" : "IDS DIFFER"}</p>`).join("")}
    ${r.lanes ? `<p class="status">Load ${(r.loadMs ?? 0).toFixed(0)} ms · tuned lanes: ${Object.entries(r.lanes).map(([k, v]) => `${k}→${v}`).join(", ")}</p>` : ""}
    ${r.prefixReuse ? `<p class="status">Prefix reuse: ${r.prefixReuse.cachedTokens} of ${r.prefixReuse.promptTokens} prompt tokens cached · TTFT ${r.prefixReuse.ttftMs.toFixed(0)} ms (cold ${r.prefixReuse.coldTtftMs.toFixed(0)} ms) · ${r.prefixReuse.matches ? "continuation matches" : "CONTINUATION DIFFERS"}</p>` : ""}
    ${(r.prefillAttention ?? []).map((a) => `<p class="status">Prefill T=${a.T} at ${a.pos0}: tiled attention ${a.tiledMs.toFixed(1)} ms (attention ${a.tiledAttnMs.toFixed(1)}) · per-row ${a.rowsMs.toFixed(1)} ms (attention ${a.rowsAttnMs.toFixed(1)})</p>`).join("")}
    ${([["JSON schema", r.grammar], ["JSON schema, pretty whitespace", r.grammarPretty]] as const).map(([name, g]) => (g ? `<p class="status">${name}, greedy: ${g.tokens} tokens in ${g.ms.toFixed(0)} ms, ${g.decodeTokS.toFixed(1)} tok/s · masks ${g.maskMs.toFixed(0)} ms · ${g.valid ? "valid JSON" : "INVALID JSON"}: ${g.text.replace(/</g, "&lt;")}</p>` : "")).join("")}
    ${r.subgroups ? `<p class="status">Subgroups (size ≥ ${r.subgroups.minSize}): decode step at 256 ${r.subgroups.stepMsOn.toFixed(2)} ms with, ${r.subgroups.stepMsOff.toFixed(2)} ms without · ${r.subgroups.decodeTokS.toFixed(1)} tok/s (8 per submit) · first ${r.subgroups.matching} of ${r.subgroups.of} greedy tokens match</p>` : ""}
    ${r.f16Math ? `<p class="status">f16 matmul math: ${r.f16Math.decodeTokS.toFixed(1)} tok/s (8 per submit) · step at 256: ${r.f16Math.stepMs256.toFixed(2)} ms · first ${r.f16Math.matching} of ${r.f16Math.of} greedy tokens match f32</p>` : ""}
    ${r.kvF32Match ? `<p class="status">Default vs unfused + f32 KV cache: first ${r.kvF32Match.matching} of ${r.kvF32Match.of} greedy tokens identical · unfused f32-cache step at position 1536: ${r.kvF32Match.stepMs1536.toFixed(2)} ms</p>` : ""}
    ${r.profile ? `<h2 style="font-size:16px">GPU time per kernel: ${r.profile.promptTokens}-token prompt + ${r.profile.generated} tokens, ${r.profile.totalMs.toFixed(0)} ms in total</h2>
      <table><thead><tr><th>Dispatch</th><th>Calls</th><th>ms</th><th>Share</th></tr></thead><tbody>
      ${r.profile.entries.map((e) => `<tr><td>${e.label}</td><td class="num">${e.calls}</td><td class="num">${e.ms.toFixed(1)}</td><td class="num">${((e.ms / r.profile!.totalMs) * 100).toFixed(1)}%</td></tr>`).join("")}
      </tbody></table><p class="status">${r.profile.note}</p>` : ""}
    <p class="status">End to end: ${r.e2e.promptTokens} prompt tokens · TTFT ${r.e2e.ttftMs.toFixed(0)} ms · ${r.e2e.generated} tokens at ${r.e2e.decodeTokS.toFixed(1)} tok/s</p>
    ${r.probe ? `<p class="status">${Object.entries(r.probe).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ")}</p>` : ""}
    <p class="msg">${r.e2e.text.replace(/</g, "&lt;")}</p></details>`;
  out.prepend(section);
  drawn.set(section, r);
}

$<HTMLButtonElement>("run").onclick = async () => {
  const btn = $<HTMLButtonElement>("run");
  btn.disabled = true;
  try {
    const r = await bench($<HTMLInputElement>("model").value.trim(), $<HTMLSelectElement>("quant").value as "q4" | "q8" | "f32");
    render(r);
    try {
      localStorage.setItem(LAST, JSON.stringify({ at: Date.now(), r }));
    } catch {
      // Storage may be unavailable (private window); the run is still shown.
    }
    (window as unknown as { __bench: unknown }).__bench = r;
  } catch (err) {
    out.insertAdjacentHTML("beforeend", `<p class="error">${String(err)}</p>`);
    (window as unknown as { __bench: unknown }).__bench = { error: String(err) };
  } finally {
    btn.disabled = false;
  }
};

// Show the previous run's charts until the next run finishes.
try {
  const saved = JSON.parse(localStorage.getItem(LAST) ?? "null") as { at: number; r: BenchResult } | null;
  if (saved?.r?.plans) render(saved.r, new Date(saved.at));
} catch {
  // Missing or unreadable: start empty.
}
