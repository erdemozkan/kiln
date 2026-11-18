// A page to try tool calls: ready scenarios, tools that run in the page on made-up data.
import { load, runTools, type KilnModel, type RunnableTool } from "@kiln/api";

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

const tool = (name: string, description: string, properties: Record<string, unknown>, required: string[], execute: RunnableTool["execute"]): RunnableTool => ({
  type: "function",
  function: { name, description, parameters: { type: "object", properties, required } },
  execute,
});

const WEATHER: Record<string, { celsius: number; sky: string }> = {
  paris: { celsius: 18, sky: "light rain" },
  tokyo: { celsius: 24, sky: "clear" },
  istanbul: { celsius: 21, sky: "partly cloudy" },
  "new york": { celsius: 12, sky: "windy" },
};
const getWeather = tool(
  "get_weather",
  "Get the current weather in a city.",
  { city: { type: "string", description: "The city's name" }, unit: { enum: ["celsius", "fahrenheit"] } },
  ["city", "unit"],
  ({ city, unit }) => {
    const w = WEATHER[String(city).toLowerCase()];
    if (!w) throw new Error(`no weather station in ${String(city)}`);
    return { city, temperature: unit === "fahrenheit" ? Math.round(w.celsius * 1.8 + 32) : w.celsius, unit, sky: w.sky };
  },
);

const OFFSETS: Record<string, number> = { paris: 1, tokyo: 9, istanbul: 3, "new york": -5, london: 0 };
const getTime = tool("get_time", "Get the current local time in a city.", { city: { type: "string" } }, ["city"], ({ city }) => {
  const offset = OFFSETS[String(city).toLowerCase()];
  if (offset === undefined) throw new Error(`unknown city ${String(city)}`);
  const d = new Date(Date.now() + offset * 3600e3);
  return { city, time: d.toISOString().slice(11, 16), utc_offset: offset };
});

const calculate = tool(
  "calculate",
  "Do exact arithmetic on two numbers.",
  { a: { type: "number" }, operation: { enum: ["add", "subtract", "multiply", "divide"] }, b: { type: "number" } },
  ["a", "operation", "b"],
  ({ a, operation, b }) => {
    const x = Number(a);
    const y = Number(b);
    if (operation === "divide" && y === 0) throw new Error("division by zero");
    return { result: operation === "add" ? x + y : operation === "subtract" ? x - y : operation === "multiply" ? x * y : x / y };
  },
);

const ORDERS: Record<string, unknown> = {
  "A-1042": { status: "shipped", carrier: "DHL", expected: "Friday", items: ["blue kettle"] },
  "A-1043": { status: "processing", expected: "next week", items: ["tea set", "spoons"] },
};
const getOrder = tool("get_order", "Look up an order by its id.", { order_id: { type: "string", pattern: "A-[0-9]{4}" } }, ["order_id"], ({ order_id }) => {
  const o = ORDERS[String(order_id)];
  if (!o) throw new Error(`no order ${String(order_id)}`);
  return o;
});
const cancelOrder = tool("cancel_order", "Cancel an order that has not shipped yet.", { order_id: { type: "string", pattern: "A-[0-9]{4}" } }, ["order_id"], ({ order_id }) => {
  const o = ORDERS[String(order_id)] as { status: string } | undefined;
  if (!o) throw new Error(`no order ${String(order_id)}`);
  if (o.status === "shipped") throw new Error("the order has already shipped and cannot be cancelled");
  return { order_id, cancelled: true };
});

const lights: Record<string, { on: boolean; brightness: number }> = { kitchen: { on: false, brightness: 0 }, bedroom: { on: true, brightness: 40 } };
const setLight = tool(
  "set_light",
  "Turn a room's light on or off and set its brightness.",
  { room: { enum: ["kitchen", "bedroom"] }, on: { type: "boolean" }, brightness: { type: "integer", minimum: 0, maximum: 100 } },
  ["room", "on"],
  ({ room, on, brightness }) => {
    const l = (lights[String(room)] = { on: Boolean(on), brightness: on ? Number(brightness ?? 100) : 0 });
    return { room, ...l };
  },
);

interface Scenario {
  title: string;
  about: string;
  prompt: string;
  tools: RunnableTool[];
  system?: string;
  choice?: "auto" | "required";
}
const SCENARIOS: Scenario[] = [
  { title: "Weather", about: "One call, then an answer in words.", prompt: "What's the weather in Paris right now, in celsius?", tools: [getWeather] },
  { title: "Calculator", about: "The model hands the arithmetic to a function.", prompt: "What is 1234 multiplied by 5678?", tools: [calculate] },
  { title: "Pick the right tool", about: "Three tools offered; one fits.", prompt: "What time is it in Tokyo?", tools: [getWeather, getTime, calculate] },
  { title: "Two calls", about: "Time and weather for one city.", prompt: "What time is it in Istanbul, and what's the weather there in celsius?", tools: [getWeather, getTime] },
  {
    title: "Order support",
    about: "Look an order up, with a system message.",
    system: "You are the support assistant of a small shop. Use the tools to look things up; never guess an order's status.",
    prompt: "Where is my order A-1042?",
    tools: [getOrder, cancelOrder],
  },
  {
    title: "A tool that fails",
    about: "The function throws; the model reads the error.",
    system: "You are the support assistant of a small shop. Use the tools; tell the customer plainly when something cannot be done.",
    prompt: "Please cancel order A-1042.",
    tools: [getOrder, cancelOrder],
  },
  { title: "Smart home", about: "Arguments with an enum, a boolean and a bounded integer.", prompt: "Turn the kitchen light on at half brightness.", tools: [setLight] },
  { title: "Forced call", about: "Tool choice \"required\": the reply must be a call.", prompt: "I'm heading to New York tomorrow.", tools: [getWeather, getTime], choice: "required" },
  { title: "No tool needed", about: "The model should just answer.", prompt: "Say hello in French and in Turkish.", tools: [getWeather, calculate] },
];

let model: KilnModel | undefined;
let current = SCENARIOS[0]!;
let abort: AbortController | undefined;

function pick(s: Scenario): void {
  current = s;
  prompt.value = s.prompt;
  choice.value = s.choice ?? "auto";
  specs.textContent = JSON.stringify(s.tools.map((t) => t.function), null, 2) + (s.system ? `\n\nSystem message: ${s.system}` : "");
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
    model = await load(modelSelect.value, {
      quant: "auto",
      device: device.value as "auto" | "gpu" | "cpu",
      onProgress: (p) => {
        progress.value = p.total ? p.loaded / p.total : 0;
        status.textContent = `${p.file}: ${(p.loaded / 2 ** 20).toFixed(0)} / ${(p.total / 2 ** 20).toFixed(0)} MiB`;
      },
    });
    progress.value = 1;
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
    const r = await runTools(model, [...(current.system ? [{ role: "system" as const, content: current.system }] : []), { role: "user", content: prompt.value.trim() }], {
      tools: current.tools,
      toolChoice: choice.value as "auto" | "required",
      checkToolCalls: check.checked,
      temperature: 0,
      maxTokens: 256,
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
