// The tool loop: the model asks for functions, Kiln runs them and passes the answers back, until
// the model answers in words.
import { parseToolCalls, type ChatMessage, type ToolCall, type ToolSpec } from "@kiln/tokenizers";
import type { GenerateOptions, KilnModel } from "./client";

/** What a tool's function gets besides the arguments. */
export interface ToolContext {
  /** The run's `signal`: a tool that waits (on the network, say) should stop when it fires. */
  signal?: AbortSignal;
}

/** A tool with the function that runs it. `execute` gets the call's arguments; what it returns goes back to the model as JSON. */
export interface RunnableTool extends ToolSpec {
  execute(args: Record<string, unknown>, context?: ToolContext): unknown;
}

export interface RunToolsOptions extends Omit<GenerateOptions, "tools"> {
  tools: RunnableTool[];
  /** Most replies the model may write, the final answer included (default 5). */
  maxRounds?: number;
  /** Called with the model's text as it is generated (call markup included). */
  onText?(delta: string): void;
  /** Called after each call has run. */
  onToolCall?(step: ToolStep): void;
}

/** One call the model made, and what its function returned (or the error it threw). */
export interface ToolStep {
  call: ToolCall;
  result?: unknown;
  error?: string;
}

export interface RunToolsResult {
  /** The model's final answer. */
  content: string;
  /** The conversation with the assistant's calls, the tools' answers and the final reply appended. */
  messages: ChatMessage[];
  steps: ToolStep[];
  /**
   * True when the model was still calling tools after `maxRounds` replies (`content` is its last
   * text), or when the run was stopped through `signal` (`content` is the reply as far as it got,
   * call markup included; calls in it are not run).
   */
  unfinished: boolean;
}

/**
 * Chats with tools until the model answers without calling one. A function that throws, a call to
 * an unknown tool: the model gets `{"error": …}` as that call's answer and can try again.
 * `toolChoice` applies to the first reply only (forcing a call every round would never end).
 */
export async function runTools(llm: Pick<KilnModel, "chat">, messages: readonly ChatMessage[], opts: RunToolsOptions): Promise<RunToolsResult> {
  const { tools, maxRounds = 5, onText, onToolCall, toolChoice, ...gen } = opts;
  if (!tools.length) throw new Error("runTools needs at least one tool");
  const names = new Set<string>();
  for (const t of tools) {
    if (names.has(t.function.name)) throw new Error(`runTools: two tools are named ${JSON.stringify(t.function.name)}`);
    names.add(t.function.name);
  }
  const specs: ToolSpec[] = tools.map(({ type, function: f }) => ({ type, function: f }));
  const byName = new Map(tools.map((t) => [t.function.name, t]));
  const history = [...messages];
  const steps: ToolStep[] = [];
  let content = "";
  for (let round = 0; round < maxRounds; round++) {
    let text = "";
    for await (const delta of llm.chat(history, { ...gen, tools: specs, toolChoice: round === 0 ? toolChoice : undefined })) {
      text += delta;
      onText?.(delta);
    }
    // Stopped: the reply was cut off, and a call in it may be cut too. Nothing in it is run.
    if (gen.signal?.aborted) {
      history.push({ role: "assistant", content: text });
      return { content: text, messages: history, steps, unfinished: true };
    }
    const reply = parseToolCalls(text);
    content = reply.content;
    if (!reply.toolCalls.length) {
      history.push({ role: "assistant", content: text });
      return { content, messages: history, steps, unfinished: false };
    }
    history.push({ role: "assistant", content: reply.content, tool_calls: reply.toolCalls });
    for (const call of reply.toolCalls) {
      const step: ToolStep = { call };
      const tool = byName.get(call.function.name);
      try {
        if (!tool) throw new Error(`no tool named ${JSON.stringify(call.function.name)}`);
        step.result = await tool.execute(call.function.arguments, { signal: gen.signal });
      } catch (err) {
        step.error = err instanceof Error ? err.message : String(err);
      }
      steps.push(step);
      onToolCall?.(step);
      const answer = step.error !== undefined ? { error: step.error } : step.result;
      history.push({
        role: "tool",
        name: call.function.name,
        ...(call.id ? { tool_call_id: call.id } : {}),
        content: typeof answer === "string" ? answer : JSON.stringify(answer ?? null),
      });
    }
    if (gen.signal?.aborted) break;
  }
  return { content, messages: history, steps, unfinished: true };
}
