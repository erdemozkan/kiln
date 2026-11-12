import { load, type KilnModel } from "kiln-webgpu";
import { $, backendNote, busy, progress, status } from "./common";

let llm: KilnModel | undefined;
backendNote();

$("run").addEventListener("click", () =>
  busy($<HTMLButtonElement>("run"), async () => {
    if (!llm) {
      llm = await load("HuggingFaceTB/SmolLM2-135M-Instruct", { onProgress: progress("Loading SmolLM2-135M") });
      status("Loading the two adapters…");
      await llm.loadAdapter("science", "Miladsaeedi70/smollm2-135m-scientific-sft-lora");
      await llm.loadAdapter("linkedin", "ArtemMozharov/smollm2-135m-linkedin-translator");
    }
    const prompt = $<HTMLInputElement>("prompt").value;
    for (const [name, cell] of [[null, "base"], ["science", "science"], ["linkedin", "linkedin"]] as const) {
      const t0 = performance.now();
      await llm.setAdapter(name);
      const switchMs = performance.now() - t0;
      let text = "";
      for await (const t of llm.chat([{ role: "user", content: prompt }], { maxTokens: 80, temperature: 0 })) {
        text += t;
        $(cell).textContent = text;
      }
      $(`${cell}-ms`).textContent = `switched in ${switchMs.toFixed(0)} ms`;
    }
    status("One base model, three behaviours");
  }),
);
