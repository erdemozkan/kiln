# Tools and skills

A chat model on its own can only write text. With tools it can ask your code to do something (work
out a sum, read the clock, look a fact up) and then answer from the result. Kiln runs that loop for
you, and ships four tools that do real work, called skills.

- [Try it](#try-it)
- [Built-in skills](#built-in-skills)
- [Your own tools](#your-own-tools)
- [What `runTools` returns](#what-runtools-returns)
- [Options](#options)
- [Checked calls](#checked-calls)
- [Which model](#which-model)
- [Privacy and safety](#privacy-and-safety)
- [From Python](#from-python)
- [Running the loop yourself](#running-the-loop-yourself)
- [Limits](#limits)

## Try it

The playground has a Skills page with nine ready scenarios (a sum, the time in Tokyo, a Wikipedia
lookup, a question that needs two skills, one that needs none …). Pick a model, load it, press a
scenario, and watch each call and its result.

- Online: <https://erdemozkan.github.io/kiln/playground/tools.html>
- From this repository: `npm install && npm run dev`, then open `http://localhost:5173/tools.html`

Its source is [examples/playground/src/tools.ts](../examples/playground/src/tools.ts).

## Built-in skills

`skills()` returns the four skills, ready for `runTools`:

```js
import { load, runTools, skills } from "kiln-webgpu";

const llm = await load("ibm-granite/granite-4.0-1b-GGUF/granite-4.0-1b-Q8_0.gguf");
const { content, steps } = await runTools(
  llm,
  [{ role: "user", content: "How tall is Mount Everest, and what is half of that?" }],
  { tools: skills(), maxTokens: 200 },
);
console.log(content);                                   // the answer in words
for (const s of steps) console.log(s.call.function.name, s.call.function.arguments, s.result ?? s.error);
```

| Skill | What it does | Network |
| --- | --- | --- |
| `calculate` | Arithmetic on an expression: exact for whole numbers up to 2^53, 12 significant digits otherwise. It has its own parser; the model's text is never run as code. | none |
| `get_date_time` | The date, weekday and time now, here or in a city or IANA time zone. | none |
| `search_wikipedia` | The start of the best matching Wikipedia article, in any language edition. | the query goes to Wikipedia |
| `read_web_page` | A page's readable text. | the address goes to that site |

Each is also exported alone, to mix with your own tools: `calculatorSkill()`, `dateTimeSkill()`,
`wikipediaSkill(options)`, `webPageSkill(options)`.

`skills(options)` takes:

| Option | Default | Meaning |
| --- | --- | --- |
| `maxChars` | 2000 | Most characters of text a network skill hands the model. It all goes into the context, so keep it small for small models. |
| `language` | `"en"` | The Wikipedia edition used when the model names none. |
| `allowUrl(url)` | refuses this machine, private networks and the page's own site | Decides whether `read_web_page` may fetch an address. |
| `fetch` | the global `fetch` | The function the network skills use, for a proxy or a test. |
| `userAgent` | a built-in name | Your application's name, sent to Wikipedia only. Wikimedia asks clients to identify themselves. |

## Your own tools

A tool is an OpenAI-style function description plus the function that runs it:

```js
import { load, runTools, calculatorSkill } from "kiln-webgpu";

const orders = { "A-1001": { status: "shipped", eta: "2026-11-24" } };

const orderStatus = {
  type: "function",
  function: {
    name: "get_order_status",
    description: "Look up an order by its number.",
    parameters: {
      type: "object",
      properties: { order: { type: "string", description: "The order number, like A-1001" } },
      required: ["order"],
    },
  },
  execute: async ({ order }) => orders[order] ?? { error: "no such order" },
};

const llm = await load("ibm-granite/granite-4.0-1b-GGUF/granite-4.0-1b-Q8_0.gguf");
const { content } = await runTools(llm, [{ role: "user", content: "Where is my order A-1001?" }], {
  tools: [orderStatus, calculatorSkill()],
});
```

- `execute` gets the call's arguments as one object, and may be async. A string it returns goes to
  the model as it is; anything else goes as JSON.
- Its second argument is `{ signal }`, the run's stop signal. Pass it to `fetch` so a stopped run
  does not wait for the network.
- If `execute` throws, or the model calls a tool that does not exist, the model gets
  `{"error": "…"}` as the answer and can try again.
- Write the `description` for the model: say what the tool is for in one plain sentence. Avoid
  examples in it; small models copy them into their calls.
- Fewer tools work better. Offer only the ones that fit the conversation.

## What `runTools` returns

| Field | Meaning |
| --- | --- |
| `content` | The model's final answer. |
| `steps` | Every call the model made: `call` (name and arguments), and `result` or `error`. |
| `messages` | The whole conversation with the calls, the tools' answers and the final reply. Pass it back with the next user message to continue the chat. |
| `unfinished` | True when the model was still calling tools after `maxRounds` replies, or the run was stopped (`signal`). `content` is then its last text. |

## Options

`runTools(llm, messages, options)` takes `tools` plus:

| Option | Default | Meaning |
| --- | --- | --- |
| `maxRounds` | 5 | Most replies the model may write, the final answer included. |
| `onText(delta)` | | The model's text as it is written, call markup included. |
| `onToolCall(step)` | | Called after each call has run. |
| `toolChoice` | `"auto"` | `"required"` forces a call in the first reply (models with `<tool_call>` tags only). Later replies are always free. |
| `signal` | | An `AbortSignal` that stops the run: the model, and a skill's network request. |
| `checkToolCalls` | `true` | See below. |

Every generation option works too (`maxTokens`, `temperature`, `enableThinking` …). Use
`temperature: 0` for repeatable calls.

## Checked calls

Small models often write a call that is almost right: a name that does not exist, a missing
argument, broken JSON. Kiln prevents that. The model writes freely until it starts a call; from
there the call is generated under a grammar built from your tools, so the name is always one of
them and the arguments always follow that tool's `parameters` schema. Then the model writes freely
again.

- It covers models that write calls in `<tool_call>` tags (Qwen2.5, Qwen3, Hermes 3, Granite 4.0)
  and Granite 3's call list.
- FunctionGemma and Mistral write calls in formats of their own. Kiln reads them but does not check
  them.
- `checkToolCalls: false` turns the check off.
- A tool whose schema uses a keyword the grammar does not support is offered unchecked.
- A call the model drafts inside a reasoning block (`<think>`) is not run.
- The check guarantees the form of a call, not its sense: a model can still call the wrong tool, or
  pass a wrong but valid value.

## Which model

The model's chat template must support tools. [skill-models.md](skill-models.md) has the scores of
every model tried on the same eight questions. In short:

| Model | Download | Passed | Notes |
| --- | --- | --- | --- |
| Granite 4.0 1B (Q8_0 GGUF) | 1.7 GB | 8 of 8 | The first choice. 1 to 12 s per question on a GPU; slow on a CPU. |
| Qwen3 4B (Q4_K_M GGUF) | 2.5 GB | 7 of 8 | Needs a GPU and a small `maxSeq`. |
| Hermes 3, Llama 3.2 3B (Q4_K_M GGUF) | 2.0 GB | 6 of 8 | Never calls the calculator. |
| Qwen2.5-1.5B-Instruct (q4) | 3.1 GB | 6 of 8 | |
| FunctionGemma 270M | 0.5 GB | 5 of 8 | One-step questions in English. |
| Qwen2.5-0.5B-Instruct (q8) | 1.0 GB | 5 of 8 | One-step questions. At q4 it stops calling skills. |

`SKILL_INSTRUCTIONS` is a system message that tells a model to use the skills instead of asking the
user or answering from memory:

```js
import { SKILL_INSTRUCTIONS } from "kiln-webgpu";
const messages = [{ role: "system", content: SKILL_INSTRUCTIONS }, { role: "user", content: "…" }];
```

Use it with the small Qwen2.5 models. Leave it out for Granite 4.0, Hermes 3 and FunctionGemma:
each does better with the prompt it was trained on, which Kiln writes for you.

## Privacy and safety

- The model runs on the user's device. `calculate` and `get_date_time` use no network.
- `search_wikipedia` sends the model's query to Wikipedia. `read_web_page` sends a request to the
  address the model chose. Nothing else is sent: no cookies, no referrer. But the model writes the
  query and the address, so words from the conversation can be in them.
- The model picks the address, and a page it has read can suggest another. So `read_web_page`
  refuses `localhost`, private networks (10.x, 192.168.x …) and the page's own site, on every
  redirect too, unless your `allowUrl` allows them. In Node.js it also looks the name up and refuses
  one that points at a private address. A name can still change between that check and the
  request, so do not run it next to services that trust the local network.
- A network skill gives up after 15 seconds and reads at most 2 MB of a page.
- In a browser, `read_web_page` can only read sites that let other pages read them, and most do
  not. The model is told when a site refuses. Wikipedia articles are read through Wikipedia's API,
  which does allow it. In Node.js there is no such limit.
- Text that comes back from the network goes into the model's context, and a page can contain
  instructions aimed at the model. Do not give a model that reads the web tools that change
  anything (send, buy, delete) without asking the user first.
- Your own tools run with your page's rights. Check their arguments as you would check any input.

## From Python

`run_tools` does the same with Python functions:

```python
from kiln_webgpu import pipeline, run_tools

def get_order_status(order):
    return {"order": order, "status": "shipped", "eta": "2026-11-24"}

tools = [{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Look up an order by its number.",
        "parameters": {"type": "object", "properties": {"order": {"type": "string"}}, "required": ["order"]},
    },
    "execute": get_order_status,
}]

chat = pipeline("text-generation", "Qwen/Qwen2.5-1.5B-Instruct")
result = run_tools(chat, [{"role": "user", "content": "Where is my order A-1001?"}], tools, max_new_tokens=128)
print(result["content"])
```

`execute` is called with the call's arguments as keyword arguments. The built-in skills are not
available from Python yet.

## Running the loop yourself

`runTools` is a short loop over two pieces you can use directly: pass `tools` to `llm.chat()`, and
read the reply with `parseToolCalls`.

```js
import { parseToolCalls } from "kiln-webgpu";

let text = "";
for await (const delta of llm.chat(messages, { tools: specs })) text += delta;
const { content, toolCalls } = parseToolCalls(text);   // toolCalls: [{ function: { name, arguments } }]
```

Run each call. Then append the assistant's reply as
`{ role: "assistant", content, tool_calls: toolCalls }`, one
`{ role: "tool", name, tool_call_id: call.id, content }` message per call (`content` a string: the
result as JSON; `tool_call_id` only when the call has an `id`, as Mistral's do), and chat again.

## Limits

- Models this small manage one or two steps. Longer chains (three steps or more) have not been
  measured.
- A skill's result takes room in the context. With several lookups in one chat, lower `maxChars` or
  raise the model's `maxSeq`.
- The skills are four. A code runner, document search and skills that use Kiln's other models are
  planned; see the [future features plan](future-features-plan.md).
