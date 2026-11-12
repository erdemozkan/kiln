import { load, type JsonSchema, type KilnModel } from "kiln-webgpu";
import { $, backendNote, busy, progress, status } from "./common";

let llm: KilnModel | undefined;
backendNote();

$("run").addEventListener("click", () =>
  busy($<HTMLButtonElement>("run"), async () => {
    let schema: JsonSchema;
    try {
      schema = JSON.parse($<HTMLTextAreaElement>("schema").value);
    } catch {
      status("The schema is not valid JSON.");
      return;
    }
    llm ??= await load("Qwen/Qwen2.5-0.5B-Instruct", { quant: "q8", onProgress: progress("Loading Qwen2.5-0.5B") });
    status("Extracting…");
    const t0 = performance.now();
    let json = "";
    const messages = [
      { role: "system" as const, content: "Extract the requested information from the text as JSON. Use only what the text says." },
      { role: "user" as const, content: $<HTMLTextAreaElement>("text").value },
    ];
    for await (const t of llm.chat(messages, { format: { type: "json", schema }, temperature: 0 })) {
      json += t;
      $("output").textContent = json;
    }
    $("output").textContent = JSON.stringify(JSON.parse(json), null, 2);
    status(`Valid against the schema by construction · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  }),
);
