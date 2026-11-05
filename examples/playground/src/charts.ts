import type { BenchResult, KernelTime } from "./bench.worker";

/**
 * Dependency-free SVG charts for the bench page. Colours come from CSS variables in style.css
 * (--c1…--c7, --accent, --line, --muted), so light and dark mode follow the page; bars grow in on
 * render (`.grow`) and every mark has a <title> for an exact-value tooltip.
 */

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const fmt = (v: number, digits = 1) => v.toFixed(digits);

/** A round axis maximum at or above `v` (1, 2, 2.5 or 5 × 10^n). */
function niceMax(v: number): number {
  const p = 10 ** Math.floor(Math.log10(v || 1));
  return ([1, 2, 2.5, 5, 10].find((m) => m * p >= v) ?? 10) * p;
}

function tile(label: string, value: string, unit: string, note = ""): string {
  return `<div class="tile"><div class="tile-label">${esc(label)}</div><div class="tile-value">${esc(value)}<span>${esc(unit)}</span></div>${note ? `<div class="tile-note">${esc(note)}</div>` : ""}</div>`;
}

function card(title: string, sub: string, body: string): string {
  return `<figure class="chart"><figcaption><strong>${esc(title)}</strong><span>${esc(sub)}</span></figcaption>${body}</figure>`;
}

/** Vertical bars with value labels on top. */
function columns(items: { label: string; value: number; unit: string; highlight?: boolean }[]): string {
  const W = 360, H = 200, top = 22, bottom = 30, left = 8;
  const max = niceMax(Math.max(...items.map((i) => i.value)) * 1.08);
  const slot = (W - left * 2) / items.length;
  const bw = Math.min(56, slot * 0.56);
  const y = (v: number) => top + (H - top - bottom) * (1 - v / max);
  const grid = [0.25, 0.5, 0.75, 1].map((f) => `<line class="grid" x1="0" x2="${W}" y1="${y(max * f)}" y2="${y(max * f)}"/>`).join("");
  const bars = items
    .map((it, i) => {
      const x = left + slot * i + (slot - bw) / 2;
      const yy = y(it.value);
      return `<g class="mark"><title>${esc(it.label)}: ${fmt(it.value)} ${esc(it.unit)}</title>
        <rect class="grow-y ${it.highlight ? "fill-accent" : "fill-soft"}" style="animation-delay:${i * 90}ms" x="${x}" y="${yy}" width="${bw}" height="${H - bottom - yy}" rx="5"/>
        <text class="val" x="${x + bw / 2}" y="${yy - 7}" text-anchor="middle">${fmt(it.value)}</text>
        <text class="tick" x="${x + bw / 2}" y="${H - bottom + 18}" text-anchor="middle">${esc(it.label)}</text></g>`;
    })
    .join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(items.map((i) => `${i.label} ${fmt(i.value)} ${i.unit}`).join(", "))}">${grid}<line class="base" x1="0" x2="${W}" y1="${H - bottom}" y2="${H - bottom}"/>${bars}</svg>`;
}

/** Paired horizontal bars (new vs old) per row, with the speedup at the end. */
function pairs(rows: { label: string; a: number; b: number }[], aName: string, bName: string, unit: string, W: number): string {
  const narrow = W < 500;
  const rowH = 46, labelW = narrow ? 96 : 128, right = narrow ? 72 : 96, top = 6;
  const H = top + rows.length * rowH + 4;
  const max = niceMax(Math.max(...rows.flatMap((r) => [r.a, r.b])));
  const x = (v: number) => labelW + (W - labelW - right) * (v / max);
  const body = rows
    .map((r, i) => {
      const y0 = top + i * rowH;
      const speed = r.b / r.a;
      return `<g class="mark">
        <text class="tick" x="${labelW - 12}" y="${y0 + 22}" text-anchor="end">${esc(r.label)}</text>
        <g><title>${esc(aName)}: ${fmt(r.a)} ${unit}</title><rect class="grow-x fill-accent" style="animation-delay:${i * 80}ms" x="${labelW}" y="${y0 + 4}" width="${x(r.a) - labelW}" height="15" rx="4"/>
        <text class="val" x="${x(r.a) + 6}" y="${y0 + 16}">${fmt(r.a)}</text></g>
        <g><title>${esc(bName)}: ${fmt(r.b)} ${unit}</title><rect class="grow-x fill-hatch" style="animation-delay:${i * 80 + 40}ms" x="${labelW}" y="${y0 + 22}" width="${x(r.b) - labelW}" height="15" rx="4"/>
        <text class="val muted" x="${x(r.b) + 6}" y="${y0 + 34}">${fmt(r.b)}</text></g>
        <text class="speed ${speed >= 1.05 ? "up" : speed <= 0.95 ? "down" : ""}" x="${W - 4}" y="${y0 + 26}" text-anchor="end">${speed >= 1.05 ? fmt(speed, 2) + (narrow ? "×" : "× faster") : speed <= 0.95 ? fmt(1 / speed, 2) + "× slower" : narrow ? "same" : "about the same"}</text>
      </g>`;
    })
    .join("");
  const legend = `<div class="legend"><span><i class="sw fill-accent"></i>${esc(aName)}</span><span><i class="sw fill-hatch"></i>${esc(bName)}</span><span class="muted">${esc(unit)} of GPU time per step</span></div>`;
  const defs = `<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="hatch-bg"/><line x1="0" y1="0" x2="0" y2="6" class="hatch-line"/></pattern></defs>`;
  return `${legend}<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(rows.map((r) => `${r.label}: ${aName} ${fmt(r.a)}, ${bName} ${fmt(r.b)} ${unit}`).join("; "))}">${defs}${body}</svg>`;
}

/** "matmul/q4/mt1 3072x576+norm+silu" → "matmul 3072×576 · norm+silu". */
function kernelName(k: string): string {
  const m = /^matmul\/\w+\/mt\d+ (\d+)x(\d+)\+?(.*)$/.exec(k);
  if (m) return `matmul ${m[1]}×${m[2]}${m[3] ? ` · ${m[3]}` : ""}`;
  return k.replace(/\/(f16|f32|q4|q8)$/, "").replace(/\//g, " ");
}

/** One 100% stacked bar per plan: each kernel's share of the summed kernel time. */
function stacks(plans: { label: string; total: number; kernels: KernelTime[] }[], W: number, top = 6): string {
  // Shared colour per kernel name across bars: the slowest kernels of the first plan come first.
  const names: string[] = [];
  for (const p of plans) for (const k of [...p.kernels].sort((a, b) => b.ms - a.ms).slice(0, top)) if (!names.includes(k.kernel)) names.push(k.kernel);
  const keep = names.slice(0, 7);
  const color = (k: string) => (keep.includes(k) ? `c${keep.indexOf(k) + 1}` : "c-other");
  const barH = 26, gap = 34, labelW = W < 500 ? 92 : 128;
  const H = plans.length * (barH + gap);
  const bars = plans
    .map((p, i) => {
      const sum = p.kernels.reduce((n, k) => n + k.ms, 0);
      const other = p.kernels.filter((k) => !keep.includes(k.kernel)).reduce((n, k) => n + k.ms, 0);
      const segs = [...p.kernels.filter((k) => keep.includes(k.kernel)).sort((a, b) => keep.indexOf(a.kernel) - keep.indexOf(b.kernel)), ...(other > 0 ? [{ kernel: "other", dispatches: 0, ms: other }] : [])];
      let x = labelW;
      const y = i * (barH + gap) + 18;
      const rects = segs
        .map((k) => {
          const w = ((W - labelW) * k.ms) / sum;
          const r = `<g class="mark"><title>${esc(k.kernel === "other" ? "other kernels" : kernelName(k.kernel))}: ${fmt(k.ms, 2)} ms (${fmt((k.ms / sum) * 100, 0)}%)</title><rect class="${k.kernel === "other" ? "c-other" : color(k.kernel)}" x="${x + 1}" y="${y}" width="${Math.max(0, w - 2)}" height="${barH}" rx="3"/>${w > 34 ? `<text class="seg" x="${x + w / 2}" y="${y + 17}" text-anchor="middle">${fmt((k.ms / sum) * 100, 0)}%</text>` : ""}</g>`;
          x += w;
          return r;
        })
        .join("");
      return `<text class="tick" x="${labelW - 12}" y="${y + 17}" text-anchor="end">${esc(p.label)}</text><text class="tick muted" x="${labelW}" y="${y - 6}">${fmt(p.total, 1)} ms per step</text><g class="grow-x" style="animation-delay:${i * 120}ms">${rects}</g>`;
    })
    .join("");
  const legend = `<div class="legend">${keep.map((k) => `<span><i class="sw ${color(k)}"></i>${esc(kernelName(k))}</span>`).join("")}<span><i class="sw c-other"></i>other</span></div>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of decode kernel time per plan">${bars}</svg>${legend}`;
}

/** Single stacked bar of GPU memory. */
function memory(parts: { label: string; mib: number; cls: string }[]): string {
  const total = parts.reduce((n, p) => n + p.mib, 0);
  const segs = parts
    .map((p) => `<div class="mem-seg ${p.cls}" style="flex:${p.mib} 1 0" title="${esc(p.label)}: ${fmt(p.mib)} MiB"></div>`)
    .join("");
  const legend = parts.map((p) => `<span><i class="sw ${p.cls}"></i>${esc(p.label)} <b>${fmt(p.mib)} MiB</b></span>`).join("");
  return `<div class="mem grow-x">${segs}</div><div class="legend">${legend}<span class="muted">${fmt(total)} MiB total</span></div>`;
}

/** The chart dashboard for one bench result, laid out for a container `width` px wide. */
export function dashboard(r: BenchResult, width = 788): string {
  // Wide charts are drawn at the card's inner width so their text stays at its CSS size.
  const W = Math.max(280, Math.min(760, Math.round(width - 34)));
  const by = (k: number) => r.chained?.find((c) => c.stepsPerSubmit === k);
  const tiles = [
    tile("Decode", fmt(by(4)?.decodeTokS ?? r.e2e.decodeTokS), "tok/s", "4 steps per submit"),
    tile("Decode, 8 per submit", fmt(by(8)?.decodeTokS ?? 0), "tok/s", by(8)?.sameIds === false ? "ids differ!" : "same greedy ids"),
    tile("Time to first token", fmt(r.prefixReuse?.ttftMs ?? r.e2e.ttftMs, 0), "ms", r.prefixReuse ? `${r.prefixReuse.cachedTokens} of ${r.prefixReuse.promptTokens} tokens cached` : `${r.e2e.promptTokens}-token prompt`),
    tile("GPU memory", fmt(((r.probe?.weightMB ?? 0) + (r.probe?.kvMB ?? 0) + (r.probe?.activationMB ?? 0)), 0), "MiB", "weights + KV + intermediates"),
  ].join("");
  const charts: string[] = [];
  if (r.chained?.length) {
    charts.push(card("Decode speed", "end to end, tokens per second", columns([
      ...r.chained.map((c) => ({ label: `${c.stepsPerSubmit} per submit`, value: c.decodeTokS, unit: "tok/s", highlight: c.stepsPerSubmit === 4 })),
      ...(r.grammar ? [{ label: "JSON schema", value: r.grammar.decodeTokS, unit: "tok/s" }] : []),
      ...(r.grammarPretty ? [{ label: "JSON, pretty", value: r.grammarPretty.decodeTokS, unit: "tok/s" }] : []),
    ])));
  }
  if (r.prefixReuse) {
    charts.push(card("Prefix reuse", "time to first token, ms", columns([
      { label: "cold", value: r.prefixReuse.coldTtftMs, unit: "ms" },
      { label: `${r.prefixReuse.cachedTokens} cached`, value: r.prefixReuse.ttftMs, unit: "ms", highlight: true },
    ])));
  }
  const wide: string[] = [];
  if (r.prefillAttention?.length) {
    wide.push(card("Prefill: tiled vs per-row attention", "whole prefill step", pairs(r.prefillAttention.map((a) => ({ label: `${a.T} tok @ ${a.pos0}`, a: a.tiledMs, b: a.rowsMs })), "tiled", "per-row", "ms", W)));
  }
  const decode = r.plans.filter((p) => p.T === 1);
  if (decode.length) {
    wide.push(card("Where a decode step goes", "share of kernel time, each kernel replayed alone", stacks(decode.map((p) => ({ label: `position ${p.pos0}`, total: p.stepMs, kernels: p.kernels })), W)));
  }
  if (r.probe?.weightMB) {
    wide.push(card("GPU memory", "what the model holds on the device", memory([
      { label: "weights", mib: r.probe.weightMB, cls: "c1" },
      { label: "KV cache", mib: r.probe.kvMB ?? 0, cls: "c3" },
      { label: "intermediates", mib: r.probe.activationMB ?? 0, cls: "c2" },
    ])));
  }
  return `<div class="dash"><div class="tiles">${tiles}</div><div class="pair">${charts.join("")}</div>${wide.join("")}</div>`;
}
