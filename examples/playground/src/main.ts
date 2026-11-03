import { load, type ChatMessage, type KilnModel, type ResponseFormat } from "@kiln/api";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const modelInput = $<HTMLInputElement>("model");
const quant = $<HTMLSelectElement>("quant");
const loadBtn = $<HTMLButtonElement>("load");
const progress = $<HTMLProgressElement>("progress");
const status = $("status");
const log = $("log");
const prompt = $<HTMLTextAreaElement>("prompt");
const send = $<HTMLButtonElement>("send");
const stop = $<HTMLButtonElement>("stop");
const temp = $<HTMLInputElement>("temp");
const stats = $("stats");
const formatSel = $<HTMLSelectElement>("format");
const schemaBox = $<HTMLTextAreaElement>("schema");
formatSel.onchange = () => (schemaBox.hidden = formatSel.value !== "schema");

/** The selected output constraint; throws on a schema that is not valid JSON. */
function format(): ResponseFormat | undefined {
  if (formatSel.value === "json") return { type: "json" };
  if (formatSel.value === "schema") return { type: "json", schema: JSON.parse(schemaBox.value) };
  return undefined;
}

let model: KilnModel | undefined;
let abort: AbortController | undefined;
const history: ChatMessage[] = [];

function bubble(role: string): HTMLElement {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<span class="role">${role}</span>`;
  const body = document.createElement("span");
  el.append(body);
  log.append(el);
  return body;
}

loadBtn.onclick = async () => {
  loadBtn.disabled = true;
  status.classList.remove("error");
  status.textContent = "Starting… (first load downloads the weights; later loads come from the browser cache)";
  progress.value = 0;
  const t0 = performance.now();
  try {
    await model?.unload();
    model = await load(modelInput.value.trim(), {
      quant: quant.value as "q4" | "f32",
      onProgress: (p) => {
        progress.value = p.total ? p.loaded / p.total : 0;
        status.textContent = `${p.file}: ${(p.loaded / 2 ** 20).toFixed(0)} / ${(p.total / 2 ** 20).toFixed(0)} MiB`;
      },
    });
    progress.value = 1;
    const i = model.info;
    status.textContent = `Loaded ${i.model} (${i.family}, ${i.layers} layers, ctx ${i.maxSeq}, ${i.chatFormat ?? "no chat format"}) in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
    prompt.disabled = send.disabled = false;
    history.length = 0;
    log.textContent = "";
  } catch (err) {
    status.classList.add("error");
    status.textContent = String(err instanceof Error ? err.message : err);
  } finally {
    loadBtn.disabled = false;
  }
};

send.onclick = async () => {
  if (!model || !prompt.value.trim()) return;
  const text = prompt.value.trim();
  history.push({ role: "user", content: text });
  bubble("user").textContent = text;
  const out = bubble("assistant");
  prompt.value = "";
  send.disabled = true;
  stop.disabled = false;
  abort = new AbortController();
  let reply = "";
  try {
    const fmt = format();
    // The repetition penalty would push structured output away from the quotes and braces it must repeat.
    const stream = model.chat(history, { temperature: Number(temp.value), topP: 0.9, repetitionPenalty: fmt ? 1 : 1.1, maxTokens: 256, signal: abort.signal, enableThinking: false, format: fmt });
    for await (const delta of stream) {
      reply += delta;
      out.textContent = reply;
    }
    const s = await stream.stats;
    if (s) {
      const tps = s.decodeMs ? ((s.generatedTokens - 1) / s.decodeMs) * 1000 : 0;
      stats.textContent = `${s.promptTokens} prompt tokens (${s.cachedTokens} cached) · TTFT ${s.ttftMs.toFixed(0)} ms · ${s.generatedTokens} tokens · ${tps.toFixed(1)} tok/s decode${s.maskMs !== undefined ? ` · masks ${s.maskMs.toFixed(0)} ms` : ""}`;
    }
    history.push({ role: "assistant", content: reply });
  } catch (err) {
    out.classList.add("error");
    out.textContent = String(err instanceof Error ? err.message : err);
  } finally {
    send.disabled = false;
    stop.disabled = true;
  }
};

stop.onclick = () => abort?.abort();
