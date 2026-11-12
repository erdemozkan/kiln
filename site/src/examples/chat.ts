import type { KilnModel, ChatMessage } from "kiln-webgpu";
import { $, backendNote, busy, progress, status } from "./common";
// The library loads when a demo first runs, not with the page.
const kiln = () => import("kiln-webgpu");

const MODEL = "Qwen/Qwen2.5-0.5B-Instruct";
let llm: KilnModel | undefined;
const history: ChatMessage[] = [{ role: "system", content: "You are a helpful, concise assistant." }];
backendNote();

$("send").addEventListener("click", () =>
  busy($<HTMLButtonElement>("send"), async () => {
    const input = $<HTMLTextAreaElement>("input");
    const text = input.value.trim();
    if (!text) return;
    llm ??= await (await kiln()).load(MODEL, { quant: "q8", onProgress: progress("Loading Qwen2.5-0.5B") });
    history.push({ role: "user", content: text });
    input.value = "";
    const log = $("output");
    log.append(Object.assign(document.createElement("div"), { textContent: `You: ${text}` }));
    const reply = document.createElement("div");
    reply.textContent = "Assistant: ";
    log.append(reply);
    status("Generating…");
    const stream = llm.chat(history, { maxTokens: 400, temperature: 0.3 });
    let answer = "";
    for await (const t of stream) {
      answer += t;
      reply.textContent = `Assistant: ${answer}`;
    }
    history.push({ role: "assistant", content: answer });
    const s = await stream.stats;
    if (s) status(`${s.generatedTokens} tokens at ${Math.round(s.generatedTokens / (s.decodeMs / 1000))} tok/s · ${s.cachedTokens} of ${s.promptTokens} prompt tokens reused from the cache · ${llm.info.backend}`);
  }),
);
