// The benchmark page: measured end-to-end timings, Kiln against transformers.js, same machine, same
// inputs. Every number here comes from the comparisons in docs/technical-notes.md.
import "./layout.ts";

interface Row {
  area: "Language" | "Text encoders" | "Seq2seq" | "Speech" | "Vision" | "Multimodal";
  task: string;
  model: string;
  kiln: number;
  tjs: number;
  /** "ms" (lower is better) or "tok/s" (higher is better). */
  unit: "ms" | "tok/s";
  note?: string;
}

const ROWS: Row[] = [
  { area: "Language", task: "Decoding", model: "SmolLM2-135M, q4", kiln: 64, tjs: 13.4, unit: "tok/s", note: "74 tok/s with draft tokens" },
  { area: "Language", task: "Decoding", model: "Qwen2.5-0.5B, q4", kiln: 31, tjs: 4.9, unit: "tok/s", note: "transformers.js q4f16 gave repeated dashes" },
  { area: "Language", task: "Decoding on the CPU", model: "SmolLM2-135M, q8", kiln: 57, tjs: 24, unit: "tok/s", note: "transformers.js on WASM, 2 threads" },
  { area: "Text encoders", task: "Feature extraction, 16 sentences", model: "all-MiniLM-L6-v2", kiln: 209, tjs: 298, unit: "ms" },
  { area: "Text encoders", task: "Sentiment analysis", model: "DistilBERT SST-2", kiln: 51, tjs: 101, unit: "ms" },
  { area: "Text encoders", task: "NER", model: "bert-base-NER", kiln: 100, tjs: 184, unit: "ms" },
  { area: "Text encoders", task: "Question answering", model: "DistilBERT SQuAD", kiln: 63, tjs: 99, unit: "ms" },
  { area: "Text encoders", task: "Fill-mask", model: "bert-base-uncased", kiln: 149, tjs: 300, unit: "ms" },
  { area: "Text encoders", task: "Zero-shot classification, 5 labels", model: "DistilBERT MNLI", kiln: 306, tjs: 711, unit: "ms", note: "Kiln 260–306 ms" },
  { area: "Seq2seq", task: "Summarization, 64 tokens", model: "DistilBART-CNN-6-6", kiln: 3400, tjs: 9950, unit: "ms" },
  { area: "Seq2seq", task: "Translation", model: "opus-mt-en-de", kiln: 1020, tjs: 2850, unit: "ms" },
  { area: "Seq2seq", task: "Text2text", model: "flan-t5-small", kiln: 331, tjs: 1198, unit: "ms" },
  { area: "Speech", task: "Text-to-speech", model: "MMS-TTS (VITS)", kiln: 617, tjs: 1330, unit: "ms" },
  { area: "Speech", task: "Text-to-speech", model: "SpeechT5 + HiFi-GAN", kiln: 1900, tjs: 6400, unit: "ms" },
  { area: "Speech", task: "Text-to-speech, 82 characters", model: "Kokoro-82M", kiln: 3800, tjs: 6300, unit: "ms" },
  { area: "Vision", task: "Image classification", model: "ResNet-50", kiln: 337, tjs: 370, unit: "ms" },
  { area: "Vision", task: "Image classification", model: "ConvNeXt-tiny", kiln: 374, tjs: 434, unit: "ms" },
  { area: "Vision", task: "Image classification", model: "DINOv2-small", kiln: 383, tjs: 683, unit: "ms" },
  { area: "Vision", task: "Object detection", model: "DETR-ResNet-50", kiln: 1570, tjs: 4020, unit: "ms" },
  { area: "Vision", task: "Panoptic segmentation", model: "DETR-ResNet-50 panoptic", kiln: 2400, tjs: 28200, unit: "ms" },
  { area: "Vision", task: "Semantic segmentation", model: "SegFormer-b0", kiln: 750, tjs: 3900, unit: "ms" },
  { area: "Vision", task: "Depth estimation", model: "Depth Anything V2 small", kiln: 3100, tjs: 7400, unit: "ms" },
  { area: "Vision", task: "Background removal", model: "RMBG-1.4", kiln: 2900, tjs: 4000, unit: "ms" },
  { area: "Vision", task: "2× super-resolution", model: "Swin2SR", kiln: 3400, tjs: 10400, unit: "ms" },
  { area: "Vision", task: "Zero-shot detection", model: "OWL-ViT B/32", kiln: 639, tjs: 1674, unit: "ms", note: "transformers.js 3.8.1 (4.3.2 fails to load OWL models)" },
  { area: "Vision", task: "Zero-shot detection", model: "OWLv2 B/16", kiln: 5400, tjs: 32700, unit: "ms", note: "transformers.js 3.8.1" },
  { area: "Multimodal", task: "Document QA", model: "Donut DocVQA", kiln: 17600, tjs: 33500, unit: "ms", note: "Kiln 16.9–18.3 s, transformers.js 33–34 s" },
  { area: "Multimodal", task: "Receipt to JSON", model: "Donut CORD", kiln: 5900, tjs: 12800, unit: "ms" },
  { area: "Multimodal", task: "Image question, 13 tiles", model: "SmolVLM-256M", kiln: 17400, tjs: 25000, unit: "ms" },
  { area: "Multimodal", task: "Image question, one view", model: "SmolVLM-256M", kiln: 2300, tjs: 4300, unit: "ms" },
  { area: "Multimodal", task: "Describe a photo", model: "Gemma 3 4B", kiln: 39000, tjs: 55000, unit: "ms", note: "Kiln 36–42 s (vision tower q8), transformers.js 49–61 s (decoder q4)" },
];

const speedup = (r: Row) => (r.unit === "ms" ? r.tjs / r.kiln : r.kiln / r.tjs);
const fmt = (v: number, unit: Row["unit"]) => (unit === "tok/s" ? `${v} tok/s` : v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 1 : 2)} s` : `${v} ms`);
const label = (r: Row) => `${r.model}: ${r.task}`;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string, cls?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

/** Horizontal bars of speed-up (one series), sorted, with a parity line at 1× and a hover tooltip per bar. */
function chart(host: HTMLElement): void {
  const rows = [...ROWS].sort((a, b) => speedup(b) - speedup(a));
  const W = 760;
  const labelW = 300;
  const barH = 16;
  const gap = 8;
  const top = 8;
  const H = top + rows.length * (barH + gap) + 28;
  const max = Math.ceil(Math.max(...rows.map(speedup)));
  const x = (v: number) => labelW + (v / max) * (W - labelW - 56);
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Speed-up of Kiln over transformers.js per task; the table below lists every value.");
  const add = (name: string, attrs: Record<string, string | number>, text?: string) => {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text !== undefined) n.textContent = text;
    svg.append(n);
    return n;
  };
  // Grid ticks.
  const step = max > 10 ? 2 : 1;
  for (let v = 0; v <= max; v += step) {
    add("line", { x1: x(v), x2: x(v), y1: top - 4, y2: H - 22, class: "axis", "stroke-width": 0.5 });
    add("text", { x: x(v), y: H - 6, "text-anchor": "middle" }, `${v}×`);
  }
  // Parity: as fast as transformers.js.
  add("line", { x1: x(1), x2: x(1), y1: top - 4, y2: H - 22, class: "axis", "stroke-width": 1.5, "stroke-dasharray": "4 3" });
  if (step > 1) add("text", { x: x(1), y: H - 6, "text-anchor": "middle" }, "1×");
  const tip = el("div", undefined, "tooltip");
  tip.hidden = true;
  host.append(tip);
  rows.forEach((r, i) => {
    const y = top + i * (barH + gap);
    const s = speedup(r);
    add("text", { x: labelW - 10, y: y + barH - 4, "text-anchor": "end" }, label(r).length > 46 ? `${label(r).slice(0, 45)}…` : label(r));
    const w = Math.max(2, x(s) - x(0));
    // Square at the baseline, 4px rounded at the data end.
    const r4 = Math.min(4, w / 2);
    const d = `M${x(0)},${y} h${w - r4} a${r4},${r4} 0 0 1 ${r4},${r4} v${barH - 2 * r4} a${r4},${r4} 0 0 1 -${r4},${r4} h-${w - r4} z`;
    const hit = add("rect", { x: 0, y: y - gap / 2, width: W, height: barH + gap, fill: "transparent", class: "bar-hit", tabindex: 0 });
    add("path", { d, class: "bar" });
    add("text", { x: x(s) + 6, y: y + barH - 4, class: "value" }, `${s.toFixed(1)}×`);
    const show = (ev: Event) => {
      tip.replaceChildren();
      tip.append(el("strong", `${s.toFixed(1)}× faster`), el("div", label(r)), el("div", `Kiln ${fmt(r.kiln, r.unit)} · transformers.js ${fmt(r.tjs, r.unit)}`));
      if (r.note) tip.append(el("div", r.note, "muted"));
      tip.hidden = false;
      const box = host.getBoundingClientRect();
      const p = ev instanceof PointerEvent ? { x: ev.clientX - box.left, y: ev.clientY - box.top } : { x: (x(s) / W) * box.width, y: ((y + barH) / H) * box.height };
      tip.style.left = `${Math.min(p.x + 12, box.width - 330)}px`;
      tip.style.top = `${p.y + 12}px`;
    };
    hit.addEventListener("pointermove", show);
    hit.addEventListener("focus", show);
    hit.addEventListener("pointerleave", () => (tip.hidden = true));
    hit.addEventListener("blur", () => (tip.hidden = true));
  });
  host.prepend(svg);
}

function table(host: HTMLElement): void {
  const t = el("table");
  const head = el("tr");
  for (const [h, num] of [["Task", false], ["Model", false], ["Kiln", true], ["transformers.js", true], ["Speed-up", true], ["Note", false]] as const) {
    const th = el("th", h);
    if (num) th.className = "num";
    head.append(th);
  }
  t.append(el("thead"));
  t.tHead!.append(head);
  const body = el("tbody");
  let area = "";
  for (const r of ROWS) {
    if (r.area !== area) {
      area = r.area;
      const tr = el("tr");
      const th = el("th", area);
      th.colSpan = 6;
      tr.append(th);
      body.append(tr);
    }
    const tr = el("tr");
    tr.append(el("td", r.task), el("td", r.model), el("td", fmt(r.kiln, r.unit), "num"), el("td", fmt(r.tjs, r.unit), "num"), el("td", `${speedup(r).toFixed(1)}×`, "num"), el("td", r.note ?? "", "muted"));
    body.append(tr);
  }
  t.append(body);
  host.append(t);
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2;
};
document.getElementById("median")!.textContent = `${median(ROWS.map(speedup)).toFixed(1)}×`;
document.getElementById("count")!.textContent = String(ROWS.length);
chart(document.getElementById("chart")!);
table(document.getElementById("table")!);
