import { load, tasks, type KilnModel } from "kiln-webgpu";
import { $, backendNote, busy, progress, status } from "./common";

let llm: KilnModel | undefined;
backendNote();
const ready = async () => (llm ??= await load("Qwen/Qwen2.5-0.5B-Instruct", { quant: "q8", onProgress: progress("Loading Qwen2.5-0.5B") }));

$("classify").addEventListener("click", () =>
  busy($<HTMLButtonElement>("classify"), async () => {
    const t = tasks(await ready());
    const labels = $<HTMLInputElement>("labels").value.split(",").map((s) => s.trim()).filter(Boolean);
    status("Scoring the labels…");
    const r = await t.classify($<HTMLTextAreaElement>("text").value, labels);
    $("output").textContent = r.labels.map((l, i) => `${(r.scores[i]! * 100).toFixed(1).padStart(5)}%  ${l}`).join("\n");
    status("Probabilities from the model's own scores for each label");
  }),
);
$("ner").addEventListener("click", () =>
  busy($<HTMLButtonElement>("ner"), async () => {
    const t = tasks(await ready());
    const types = $<HTMLInputElement>("types").value.split(",").map((s) => s.trim()).filter(Boolean);
    status("Finding entities…");
    const text = $<HTMLTextAreaElement>("nertext").value;
    const ents = await t.ner(text, types);
    $("output").textContent = ents.length ? ents.map((e) => `${e.entity_group.padEnd(10)} ${e.word.padEnd(20)} ${(e.score * 100).toFixed(0).padStart(3)}%  (${e.start}–${e.end})`).join("\n") : "(none found)";
    status(`${ents.length} mentions, each found verbatim in the text`);
  }),
);
