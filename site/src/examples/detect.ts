import { pipeline, type ZeroShotObjectDetectionPipeline } from "kiln-webgpu";
import { $, backendNote, busy, pipelineProgress, status } from "./common";

const SAMPLE = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg";
let detector: ZeroShotObjectDetectionPipeline | undefined;
let image: Blob | undefined;
backendNote();

async function show(blob: Blob) {
  image = blob;
  const bmp = await createImageBitmap(blob);
  const canvas = $<HTMLCanvasElement>("canvas");
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  canvas.getContext("2d")!.drawImage(bmp, 0, 0);
}
$<HTMLInputElement>("file").addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) void show(f);
});
void fetch(SAMPLE).then((r) => r.blob()).then(show);

$("run").addEventListener("click", () =>
  busy($<HTMLButtonElement>("run"), async () => {
    if (!image) return;
    status("Loading OWL-ViT…");
    detector ??= await pipeline("zero-shot-object-detection", "google/owlvit-base-patch32", { progress_callback: pipelineProgress("Loading OWL-ViT") });
    const labels = $<HTMLInputElement>("labels").value.split(",").map((s) => s.trim()).filter(Boolean);
    status("Detecting…");
    const t0 = performance.now();
    const found = (await detector(image, labels, { threshold: 0.15 })) as { score: number; label: string; box: { xmin: number; ymin: number; xmax: number; ymax: number } }[];
    await show(image);
    const ctx = $<HTMLCanvasElement>("canvas").getContext("2d")!;
    ctx.lineWidth = Math.max(2, ctx.canvas.width / 300);
    ctx.font = `${Math.max(14, ctx.canvas.width / 40)}px system-ui`;
    for (const d of found) {
      ctx.strokeStyle = "#f97316";
      ctx.strokeRect(d.box.xmin, d.box.ymin, d.box.xmax - d.box.xmin, d.box.ymax - d.box.ymin);
      const tag = `${d.label} ${d.score.toFixed(2)}`;
      ctx.fillStyle = "#f97316";
      ctx.fillRect(d.box.xmin, d.box.ymin, ctx.measureText(tag).width + 8, parseInt(ctx.font) + 6);
      ctx.fillStyle = "#fff";
      ctx.fillText(tag, d.box.xmin + 4, d.box.ymin + parseInt(ctx.font));
    }
    status(`${found.length} boxes in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  }),
);
