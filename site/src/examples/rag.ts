import { Retriever, load, type KilnModel } from "kiln-webgpu";
import { $, backendNote, busy, pipelineProgress, progress, status } from "./common";

let rag: Retriever | undefined;
let llm: KilnModel | undefined;
let indexed = "";
backendNote();

$("ask").addEventListener("click", () =>
  busy($<HTMLButtonElement>("ask"), async () => {
    const docs = $<HTMLTextAreaElement>("docs").value;
    if (!rag) {
      status("Loading the embedding model…");
      rag = await Retriever.create({ progress_callback: pipelineProgress("Loading all-MiniLM-L6-v2") });
    }
    if (docs !== indexed) {
      for (const c of [...rag.chunks]) rag.remove(c.docId);
      const parts = docs.split(/\n\s*\n/).map((t) => t.trim()).filter(Boolean);
      await rag.add(parts, { onProgress: (d, n) => status(`Embedding ${d} / ${n} chunks…`) });
      indexed = docs;
    }
    llm ??= await load("Qwen/Qwen2.5-0.5B-Instruct", { quant: "q8", onProgress: progress("Loading Qwen2.5-0.5B") });
    const q = $<HTMLInputElement>("question").value;
    const { stream, sources } = await rag.ask(llm, q, { k: 2, minScore: 0.2, temperature: 0, maxTokens: 200 });
    const src = $("sources");
    src.replaceChildren(...sources.map((s, i) => Object.assign(document.createElement("li"), { textContent: `[${i + 1}] (${s.score.toFixed(2)}) ${s.text.slice(0, 160)}${s.text.length > 160 ? "…" : ""}` })));
    let answer = "";
    status("Answering…");
    for await (const t of stream) {
      answer += t;
      $("output").textContent = answer;
    }
    const s = await stream.stats;
    status(`${rag.size} chunks indexed · ${s?.cachedTokens ?? 0} of ${s?.promptTokens ?? 0} prompt tokens from the cache`);
  }),
);
