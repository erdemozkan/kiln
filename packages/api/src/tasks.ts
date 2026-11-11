// One LLM, many tasks: classification with your own labels (scored with the model's probabilities),
// extraction to your own JSON schema, named entities of your own types, extractive question
// answering and summaries. Structured results come through Kiln's grammar, so they always parse.
import type { ChatMessage } from "@kiln/tokenizers";
import type { JsonSchema } from "@kiln/grammar";
import type { GenerateOptions, KilnModel } from "./client";

/** What the tasks need of a loaded model (a `KilnModel`). */
export type TaskModel = Pick<KilnModel, "chat" | "scoreChoices">;

/** Options every task takes: generation options for the call, and instructions to replace the default ones. */
export interface TaskOptions extends Omit<GenerateOptions, "format" | "tools" | "toolChoice"> {
  /** The system message (each task has a default). */
  instructions?: string;
}

export interface ClassifyOptions extends TaskOptions {
  /** Score each label on its own (yes / no), as zero-shot classification's multi_label. */
  multi_label?: boolean;
  /** Labelled examples shown to the model first, as earlier turns. */
  examples?: [text: string, label: string][];
}

/** transformers.js's zero-shot output: labels best first with their probabilities, plus the best label. */
export interface ClassifyResult {
  sequence: string;
  labels: string[];
  scores: number[];
  label: string;
}

const softmax = (xs: number[]) => {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

async function collect(llm: TaskModel, messages: ChatMessage[], opts: GenerateOptions): Promise<string> {
  let text = "";
  for await (const t of llm.chat(messages, opts)) text += t;
  return text;
}

function genOptions(opts: TaskOptions): GenerateOptions {
  const { instructions: _i, ...rest } = opts;
  return { temperature: 0, ...rest };
}

/**
 * Classifies a text into one of `labels` (any words: "billing", "bug report", …). Each label is
 * scored as the model's complete reply, so the scores are the model's own probabilities over the
 * labels, not a sampled guess. With `multi_label`, each label is a separate yes / no question.
 */
export async function classify(llm: TaskModel, text: string, labels: string[], opts: ClassifyOptions = {}): Promise<ClassifyResult> {
  if (!labels.length) throw new Error("classify needs at least one label");
  const { examples = [], multi_label, adapter, enableThinking } = opts;
  const score = { adapter, enableThinking };
  let scores: number[];
  if (multi_label) {
    scores = [];
    for (const label of labels) {
      const system = opts.instructions ?? `Decide whether the text belongs to the category "${label}". Reply with yes or no only.`;
      const lp = await llm.scoreChoices([{ role: "system", content: system }, { role: "user", content: text }], ["yes", "no"], score);
      scores.push(softmax(lp)[0]!);
    }
  } else {
    const system = opts.instructions ?? `Classify the text into exactly one of these categories: ${labels.map((l) => `"${l}"`).join(", ")}. Reply with the category only.`;
    const shots: ChatMessage[] = examples.flatMap(([t, l]) => [
      { role: "user" as const, content: t },
      { role: "assistant" as const, content: l },
    ]);
    scores = softmax(await llm.scoreChoices([{ role: "system", content: system }, ...shots, { role: "user", content: text }], labels, score));
  }
  const order = labels.map((_, i) => i).sort((a, b) => scores[b]! - scores[a]!);
  return { sequence: text, labels: order.map((i) => labels[i]!), scores: order.map((i) => scores[i]!), label: labels[order[0]!]! };
}

/**
 * Extracts what `schema` describes from a text, as a value that matches the schema (Kiln's
 * grammar enforces it token by token).
 */
export async function extract<T = unknown>(llm: TaskModel, text: string, schema: JsonSchema, opts: TaskOptions = {}): Promise<T> {
  const system = opts.instructions ?? "Extract the requested information from the text as JSON. Use only what the text says; copy names, numbers and dates as written.";
  const out = await collect(llm, [{ role: "system", content: system }, { role: "user", content: text }], { maxTokens: 1024, ...genOptions(opts), format: { type: "json", schema } });
  return JSON.parse(out) as T;
}

/** An entity found in the text, as transformers.js's aggregated token classification gives it. */
export interface Entity {
  entity_group: string;
  word: string;
  /** The model's probability of this type among the types given. */
  score: number;
  /** Character offsets in the text. */
  start: number;
  end: number;
}

export interface NerOptions extends TaskOptions {
  /**
   * One listing pass over all types (faster) instead of the default: candidates from that pass and
   * from one pass per type, each then typed by scoring the types.
   */
  fast?: boolean;
}

const MENTIONS: JsonSchema = { type: "object", properties: { mentions: { type: "array", items: { type: "string" } } }, required: ["mentions"], additionalProperties: false };

/**
 * Named entities of your own types ("person", "drug", "invoice number", …). Mentions are gathered
 * by listing (all types at once, then each type on its own; small models stop lists early), kept
 * only where they occur verbatim in the text, and each is given the type the model scores highest
 * for it there ("other" drops it). `dropped` counts listed mentions not in the text or typed "other".
 */
export async function ner(llm: TaskModel, text: string, types: string[], opts: NerOptions = {}): Promise<Entity[] & { dropped: number }> {
  if (!types.length) throw new Error("ner needs at least one entity type");
  const quoted = types.map((t) => `"${t}"`).join(", ");
  const joint: JsonSchema = {
    type: "object",
    properties: {
      entities: {
        type: "array",
        items: { type: "object", properties: { text: { type: "string" }, type: { enum: types } }, required: ["text", "type"], additionalProperties: false },
      },
    },
    required: ["entities"],
    additionalProperties: false,
  };
  const listed: { text: string; type?: string }[] = (
    await extract<{ entities: { text: string; type: string }[] }>(llm, text, joint, {
      ...opts,
      instructions: opts.instructions ?? `List every mention in the text of these entity types: ${quoted}. Copy each mention exactly as it is written in the text, in the order they appear. Reply with JSON.`,
    })
  ).entities;
  if (!opts.fast) {
    for (const type of types) {
      const r = await extract<{ mentions: string[] }>(llm, text, MENTIONS, { ...opts, instructions: `List every ${type} named in the text, copied exactly as written. Reply with JSON; an empty list if there is none.` });
      for (const m of r.mentions) listed.push({ text: m });
    }
  }
  // Distinct mentions that occur in the text, in order of their first occurrence.
  let dropped = 0;
  const words = new Map<string, number>();
  for (const e of listed) {
    const w = e.text.trim();
    if (!w || words.has(w)) continue;
    const at = text.indexOf(w);
    if (at < 0) dropped++;
    else words.set(w, at);
  }
  // A mention inside a longer one ("Merkel" in "Angela Merkel") is the longer one's.
  const keep = [...words.keys()].filter((w) => ![...words.keys()].some((o) => o !== w && o.includes(w)));
  const out: Entity[] = [];
  for (const word of keep) {
    let type = listed.find((e) => e.text.trim() === word && e.type)?.type;
    let score = 1;
    if (!opts.fast || !type) {
      // Asked as a completion ("… "Paris" is a"), which small models type far better than an
      // instruction; "other" lets a listed word that is no entity of these types drop out.
      const choices = [...types, "other"];
      const lp = await llm.scoreChoices([{ role: "system", content: "You label named entities." }, { role: "user", content: `${text}\n\n"${word}" is a` }], choices, {
        adapter: opts.adapter,
        enableThinking: opts.enableThinking,
      });
      const p = softmax(lp);
      const best = p.indexOf(Math.max(...p));
      if (best === types.length) {
        dropped++;
        continue;
      }
      type = types[best]!;
      score = p[best]!;
    }
    // Every occurrence of the mention.
    for (let at = text.indexOf(word); at >= 0; at = text.indexOf(word, at + word.length)) out.push({ entity_group: type!, word, score, start: at, end: at + word.length });
  }
  out.sort((a, b) => a.start - b.start);
  return Object.assign(out, { dropped });
}

/** An answer copied from the context, or null when the context does not hold one. */
export interface AnswerResult {
  answer: string | null;
  /** The model's probability that the context answers the question. */
  score: number;
  /** Character offsets of the answer in the context. */
  start?: number;
  end?: number;
}

/**
 * Extractive question answering: the model copies the shortest span of `context` that answers
 * `question`; a separate scored yes / no decides whether the context answers it at all (below
 * `threshold`, default 0.5, the answer is null), and an answer not found verbatim in the context
 * is null too.
 */
export async function answer(llm: TaskModel, question: string, context: string, opts: TaskOptions & { threshold?: number } = {}): Promise<AnswerResult> {
  const { threshold = 0.5, ...rest } = opts;
  const input = `Context:\n${context}\n\nQuestion: ${question}`;
  const decide = await classify(llm, input, ["yes", "no"], { adapter: rest.adapter, enableThinking: rest.enableThinking, instructions: "Does the context contain the answer to the question? Reply with yes or no." });
  const score = decide.labels[0] === "yes" ? decide.scores[0]! : decide.scores[1]!;
  if (score < threshold) return { answer: null, score };
  const schema: JsonSchema = { type: "object", properties: { answer: { type: "string" } }, required: ["answer"], additionalProperties: false };
  const system =
    rest.instructions ?? "Answer the question with the shortest exact phrase copied from the context. If the context does not contain the answer, reply with an empty string. Reply with JSON.";
  const a = (await extract<{ answer: string }>(llm, input, schema, { ...rest, instructions: system })).answer.trim();
  const at = a ? context.indexOf(a) : -1;
  return at < 0 ? { answer: null, score } : { answer: a, score, start: at, end: at + a.length };
}

/** A summary of `text` in about `sentences` sentences (default 3). */
export async function summarize(llm: TaskModel, text: string, opts: TaskOptions & { sentences?: number } = {}): Promise<string> {
  const { sentences = 3, ...rest } = opts;
  const system = rest.instructions ?? `Summarize the text in at most ${sentences} sentence${sentences === 1 ? "" : "s"}. Reply with the summary only.`;
  return (await collect(llm, [{ role: "system", content: system }, { role: "user", content: text }], { maxTokens: 64 * sentences + 64, ...genOptions(rest) })).trim();
}

/**
 * The tasks bound to one model:
 *
 *   const t = tasks(llm);
 *   await t.classify("My card was charged twice", ["billing", "bug", "feature request"]);
 *   await t.extract(invoiceText, { type: "object", properties: { total: { type: "number" } }, required: ["total"] });
 *   await t.ner("Aspirin was given to Ms. Kaur at St. Mary's.", ["drug", "person", "hospital"]);
 */
export function tasks(llm: TaskModel) {
  return {
    classify: (text: string, labels: string[], opts?: ClassifyOptions) => classify(llm, text, labels, opts),
    extract: <T = unknown>(text: string, schema: JsonSchema, opts?: TaskOptions) => extract<T>(llm, text, schema, opts),
    ner: (text: string, types: string[], opts?: NerOptions) => ner(llm, text, types, opts),
    answer: (question: string, context: string, opts?: TaskOptions & { threshold?: number }) => answer(llm, question, context, opts),
    summarize: (text: string, opts?: TaskOptions & { sentences?: number }) => summarize(llm, text, opts),
  };
}
