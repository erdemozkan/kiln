import type { MaskGenerationPipeline, MaskGenerationResult } from "kiln-webgpu";
import { $, backendNote, busy, pipelineProgress, status } from "./common";
// The library loads when a demo first runs, not with the page.
const kiln = () => import("kiln-webgpu");

const MODEL = "nielsr/slimsam-50-uniform";
const SAMPLE = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg";
let segmenter: Promise<MaskGenerationPipeline> | undefined;
let pixels: ImageData | undefined;
let photo: ImageBitmap | undefined;
let points: { x: number; y: number; label: number }[] = [];
let mask: MaskGenerationResult | undefined;
let running = false;
backendNote();

const canvas = $<HTMLCanvasElement>("canvas");
const ctx = canvas.getContext("2d")!;

/** The photo, the selected area tinted, and the points (green on the object, red left out). */
function draw() {
  if (!photo) return;
  ctx.drawImage(photo, 0, 0);
  if (mask) {
    const tint = new ImageData(photo.width, photo.height);
    for (let i = 0; i < mask.mask.data.length; i++) {
      if (!mask.mask.data[i]) continue;
      tint.data.set([249, 115, 22, 110], i * 4);
    }
    const layer = new OffscreenCanvas(photo.width, photo.height);
    layer.getContext("2d")!.putImageData(tint, 0, 0);
    ctx.drawImage(layer, 0, 0);
  }
  const r = Math.max(5, photo.width / 120);
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = p.label ? "#16a34a" : "#dc2626";
    ctx.fill();
    ctx.lineWidth = r / 3;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }
}

async function show(blob: Blob) {
  photo = await createImageBitmap(blob);
  canvas.width = photo.width;
  canvas.height = photo.height;
  ctx.drawImage(photo, 0, 0);
  pixels = ctx.getImageData(0, 0, photo.width, photo.height);
  points = [];
  mask = undefined;
  $<HTMLButtonElement>("save").disabled = true;
  status("Click the object you want to cut out.");
}
$<HTMLInputElement>("file").addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) void show(f);
});
void fetch(SAMPLE).then((r) => r.blob()).then(show);

async function segment() {
  if (!pixels || !points.length) return;
  running = true;
  try {
    if (!segmenter) {
      status("Loading SlimSAM…");
      segmenter = kiln().then((k) => k.pipeline("mask-generation", MODEL, { progress_callback: pipelineProgress("Loading SlimSAM") }));
    }
    const seg = await segmenter;
    status("Segmenting…");
    const t0 = performance.now();
    const masks = await seg(pixels, { points: points.map((p) => [p.x, p.y]), labels: points.map((p) => p.label) });
    mask = masks.reduce((a, b) => (b.score > a.score ? b : a));
    draw();
    $<HTMLButtonElement>("save").disabled = false;
    status(`Selected in ${((performance.now() - t0) / 1000).toFixed(2)} s (the model's quality estimate: ${mask.score.toFixed(2)}). Click to add points, Shift-click to leave parts out.`);
  } catch (err) {
    status(`Error: ${err instanceof Error ? err.message : String(err)}`);
    console.error(err);
  } finally {
    running = false;
  }
}

canvas.addEventListener("click", (e) => {
  if (!photo || running) return;
  const box = canvas.getBoundingClientRect();
  const x = ((e.clientX - box.left) / box.width) * canvas.width;
  const y = ((e.clientY - box.top) / box.height) * canvas.height;
  points.push({ x, y, label: e.shiftKey ? 0 : 1 });
  draw();
  void segment();
});

$("clear").addEventListener("click", () => {
  points = [];
  mask = undefined;
  $<HTMLButtonElement>("save").disabled = true;
  draw();
  status("Click the object you want to cut out.");
});

// The selected area as a PNG with a transparent background, cropped to the selection.
$("save").addEventListener("click", () =>
  busy($<HTMLButtonElement>("save"), async () => {
    if (!photo || !pixels || !mask) return;
    const { width, height, data } = mask.mask;
    let [x0, y0, x1, y1] = [width, height, -1, -1];
    const out = new ImageData(width, height);
    for (let i = 0; i < data.length; i++) {
      if (!data[i]) continue;
      out.data.set(pixels.data.subarray(i * 4, i * 4 + 3), i * 4);
      out.data[i * 4 + 3] = 255;
      const [x, y] = [i % width, Math.floor(i / width)];
      [x0, y0, x1, y1] = [Math.min(x0, x), Math.min(y0, y), Math.max(x1, x), Math.max(y1, y)];
    }
    if (x1 < 0) return;
    const full = new OffscreenCanvas(width, height);
    full.getContext("2d")!.putImageData(out, 0, 0);
    const cut = new OffscreenCanvas(x1 - x0 + 1, y1 - y0 + 1);
    cut.getContext("2d")!.drawImage(full, -x0, -y0);
    const url = URL.createObjectURL(await cut.convertToBlob({ type: "image/png" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "cutout.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }),
);
