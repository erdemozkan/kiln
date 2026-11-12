import { load, type JsonSchema, type KilnModel } from "kiln-webgpu";
import { $, backendNote, busy, progress, status } from "./common";

const SAMPLE = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/invoice.png";
const schema: JsonSchema = {
  type: "object",
  properties: {
    invoice_number: { type: "string" },
    invoice_date: { type: "string" },
    due_date: { type: "string" },
    bill_to: { type: "string" },
    total: { type: "number" },
  },
  required: ["invoice_number", "invoice_date", "due_date", "bill_to", "total"],
  additionalProperties: false,
};
let vlm: KilnModel | undefined;
let image: Blob | undefined;
backendNote();
$("schema").textContent = JSON.stringify(schema, null, 2);

async function show(blob: Blob) {
  image = blob;
  $<HTMLImageElement>("preview").src = URL.createObjectURL(blob);
}
$<HTMLInputElement>("file").addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) void show(f);
});
$("sample").addEventListener("click", async () => show(await (await fetch(SAMPLE)).blob()));

$("run").addEventListener("click", () =>
  busy($<HTMLButtonElement>("run"), async () => {
    if (!image) await show(await (await fetch(SAMPLE)).blob());
    vlm ??= await load("HuggingFaceTB/SmolVLM-500M-Instruct", { onProgress: progress("Loading SmolVLM-500M") });
    status("Reading the invoice (the first image takes a while: it is cut into tiles)…");
    const t0 = performance.now();
    let json = "";
    const prompt = "Read this invoice. Reply with JSON only, with the keys invoice_number, invoice_date, due_date, bill_to and total (a number).";
    for await (const t of vlm.chat([{ role: "user", content: [{ type: "image", image: image! }, { type: "text", text: prompt }] }], { format: { type: "json", schema }, temperature: 0 })) {
      json += t;
      $("output").textContent = json;
    }
    $("output").textContent = JSON.stringify(JSON.parse(json), null, 2);
    status(`${((performance.now() - t0) / 1000).toFixed(1)} s · the shape is guaranteed; check the values`);
  }),
);
