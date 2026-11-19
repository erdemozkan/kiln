// A page to try Kiln's built-in skills: ready scenarios, each run by a model through tool calls.
import { load, runTools, skills, SKILL_INSTRUCTIONS, type KilnModel, type RunnableTool } from "@kiln/api";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const modelSelect = $<HTMLSelectElement>("model");
const device = $<HTMLSelectElement>("device");
const loadBtn = $<HTMLButtonElement>("load");
const progress = $<HTMLProgressElement>("progress");
const status = $("status");
const scenarioBox = $("scenarios");
const prompt = $<HTMLTextAreaElement>("prompt");
const runBtn = $<HTMLButtonElement>("run");
const stopBtn = $<HTMLButtonElement>("stop");
const choice = $<HTMLSelectElement>("choice");
const check = $<HTMLInputElement>("check");
const specs = $("specs");
const log = $("log");
const summary = $("summary");

// ?model=…&device=cpu preselect the controls (for testing on a machine whose GPU should stay idle).
const query = new URLSearchParams(location.search);
if (query.get("device")) device.value = query.get("device")!;
if (query.get("model")) {
  const id = query.get("model")!;
  if (![...modelSelect.options].some((o) => o.value === id)) modelSelect.add(new Option(id, id));
  modelSelect.value = id;
}

/**
 * ?endpoint=… loads from a mirror of the Hub (same paths), for testing. Only this page's own server
 * or this machine: a link must not be able to point the page at somebody else's models.
 */
const endpoint = ((): string | undefined => {
  const given = query.get("endpoint");
  if (!given) return undefined;
  const url = new URL(given, location.href);
  const here = url.origin === location.origin || ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (!here) console.warn(`[skills] ignoring ?endpoint=${given}: only this site or this machine`);
  return here ? url.href.replace(/\/$/, "") : undefined;
})();

// Kiln's built-in skills: real arithmetic, the real clock, Wikipedia, and a web page's text.
const ALL = skills({ maxChars: 1500 });
const [calculate, dateTime, wikipedia, webPage] = ALL as [RunnableTool, RunnableTool, RunnableTool, RunnableTool];

interface Scenario {
  title: string;
  about: string;
  prompt: string;
  tools: RunnableTool[];
  system?: string;
  choice?: "auto" | "required";
}
const SCENARIOS: Scenario[] = [
  { title: "Arithmetic", about: "The calculator works it out exactly.", prompt: "What is 1234 * 5678 + 99?", tools: ALL },
  { title: "Time in a city", about: "The real clock, in that city's time zone.", prompt: "What time is it in Tokyo right now?", tools: ALL },
  { title: "Today's date", about: "Models do not know what day it is.", prompt: "What day of the week is it today?", tools: ALL },
  { title: "Look it up", about: "A live Wikipedia search.", prompt: "Who was Ada Lovelace? Look it up on Wikipedia.", tools: ALL },
  { title: "Another language", about: "The Turkish Wikipedia.", prompt: "Kapadokya nedir? Türkçe Vikipedi'ye bak.", tools: ALL },
  { title: "Read a page", about: "Fetches a page and answers from its text.", prompt: "Read https://raw.githubusercontent.com/erdemozkan/kiln/main/README.md and tell me in two sentences what Kiln is.", tools: ALL },
  { title: "Look up, then calculate", about: "Two skills for one question.", prompt: "How tall is Mount Everest in metres, and what is half of that?", tools: ALL },
  { title: "A site that says no", about: "Most sites forbid being read from another page; the model is told.", prompt: "Read https://example.com and tell me what it says.", tools: [webPage] },
  { title: "No skill needed", about: "The model should just answer.", prompt: "Say hello in French and in Turkish.", tools: [calculate, dateTime, wikipedia] },
];

let model: KilnModel | undefined;
/** The id of the model that is loaded (the list may show another by the time Run is pressed). */
let loaded = "";
let current = SCENARIOS[0]!;
let abort: AbortController | undefined;

/** The system message for this run. Only the small Qwen models gain from Kiln's; the others do better with the prompt they were trained on (docs/skill-models.md). */
const instructions = (): string | undefined => current.system ?? (/hermes|granite|functiongemma/i.test(loaded) ? undefined : SKILL_INSTRUCTIONS);

function pick(s: Scenario): void {
  current = s;
  prompt.value = s.prompt;
  choice.value = s.choice ?? "auto";
  specs.textContent = JSON.stringify(s.tools.map((t) => t.function), null, 2) + `\n\nSystem message: ${s.system ?? SKILL_INSTRUCTIONS} (sent to the Qwen models only)`;
  for (const b of scenarioBox.children) b.setAttribute("aria-pressed", String((b as HTMLElement).dataset.title === s.title));
}
for (const s of SCENARIOS) {
  const b = document.createElement("button");
  b.className = "scenario";
  b.dataset.title = s.title;
  const title = document.createElement("strong");
  title.textContent = s.title;
  const about = document.createElement("span");
  about.textContent = s.about;
  b.append(title, about);
  b.onclick = () => pick(s);
  scenarioBox.append(b);
}
pick(current);

function bubble(kind: string, role: string, text = ""): HTMLElement {
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  const r = document.createElement("span");
  r.className = "role";
  r.textContent = role;
  const body = document.createElement("span");
  body.textContent = text;
  el.append(r, body);
  log.append(el);
  el.scrollIntoView({ block: "nearest" });
  return body;
}

loadBtn.onclick = async () => {
  loadBtn.disabled = runBtn.disabled = true;
  status.classList.remove("error");
  progress.value = 0;
  const t0 = performance.now();
  try {
    await model?.unload();
    model = undefined;
    const id = modelSelect.value;
    model = await load(id, {
      // The smallest models pick skills far better at 8 bits than at 4.
      quant: /0\.5B|350m/i.test(modelSelect.value) ? "q8" : /functiongemma/i.test(modelSelect.value) ? "f32" : "auto",
      ...(endpoint ? { endpoint } : {}),
      device: device.value as "auto" | "gpu" | "cpu",
      onProgress: (p) => {
        progress.value = p.total ? p.loaded / p.total : 0;
        status.textContent = `${p.file}: ${(p.loaded / 2 ** 20).toFixed(0)} / ${(p.total / 2 ** 20).toFixed(0)} MiB`;
      },
    });
    progress.value = 1;
    loaded = id;
    const i = model.info;
    status.textContent = `Loaded ${i.model} on the ${i.backend === "cpu" ? "CPU" : "GPU"} in ${((performance.now() - t0) / 1000).toFixed(1)} s. Pick a scenario and press Run.`;
    runBtn.disabled = false;
  } catch (err) {
    status.classList.add("error");
    status.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    loadBtn.disabled = false;
  }
};

runBtn.onclick = async () => {
  if (!model || !prompt.value.trim()) return;
  runBtn.disabled = loadBtn.disabled = true;
  stopBtn.disabled = false;
  summary.classList.remove("error");
  summary.textContent = "Running…";
  log.textContent = "";
  abort = new AbortController();
  bubble("user", "You", prompt.value.trim());
  // Each reply of the model gets a bubble of its own; a call's markup is shown as the model wrote it.
  let reply: HTMLElement | undefined;
  const t0 = performance.now();
  try {
    const r = await runTools(model, [...(instructions() ? [{ role: "system" as const, content: instructions()! }] : []), { role: "user", content: prompt.value.trim() }], {
      tools: current.tools,
      toolChoice: choice.value as "auto" | "required",
      checkToolCalls: check.checked,
      temperature: 0,
      maxTokens: 300,
      enableThinking: false,
      signal: abort.signal,
      onText: (delta) => {
        reply ??= bubble("assistant", "Model");
        reply.textContent += delta;
      },
      onToolCall: (step) => {
        bubble("call", "The page ran", `${step.call.function.name}(${JSON.stringify(step.call.function.arguments)})`);
        bubble(step.error !== undefined ? "result failed" : "result", step.error !== undefined ? "It failed" : "It returned", step.error ?? JSON.stringify(step.result));
        reply = undefined;
      },
    });
    const calls = r.steps.length;
    summary.textContent = `${calls} tool call${calls === 1 ? "" : "s"}${r.steps.some((s) => s.error !== undefined) ? " (some failed)" : ""}, ${((performance.now() - t0) / 1000).toFixed(1)} s${r.unfinished ? " · stopped while the model was still calling tools" : ""}`;
  } catch (err) {
    summary.classList.add("error");
    summary.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    runBtn.disabled = loadBtn.disabled = false;
    stopBtn.disabled = true;
  }
};
stopBtn.onclick = () => abort?.abort();
