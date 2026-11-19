"""pipeline(): Kiln's transformers.js-style pipelines, called from Python."""
import json
import os

from ._bridge import BRIDGE
from ._types import KilnError

IMAGE_TASKS = {
    "image-classification",
    "image-feature-extraction",
    "zero-shot-image-classification",
    "object-detection",
    "zero-shot-object-detection",
    "depth-estimation",
    "image-segmentation",
    "background-removal",
    "mask-generation",
    "image-to-image",
    "image-to-text",
    "document-question-answering",
}
AUDIO_TASKS = {"automatic-speech-recognition", "audio-classification"}


class _File(os.PathLike):
    """A file to send by content: images as pixels, anything else as its bytes."""

    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        return self.path


def _file(v):
    """`v` as a file to read if it is a string naming one (a URL or a sentence stays a string)."""
    return _File(v) if isinstance(v, str) and len(v) < 4096 and os.path.isfile(os.path.expanduser(v)) and not v.startswith(("http://", "https://")) else v


def _media(v):
    """Chat messages with the image, audio and video-frame paths in their content parts marked as files."""
    if isinstance(v, (list, tuple)):
        return [_media(x) for x in v]
    if isinstance(v, dict):
        out = {k: _media(x) for k, x in v.items()}
        kind = out.get("type")
        if kind in ("image", "audio") and kind in out:
            out[kind] = _file(os.path.expanduser(out[kind]) if isinstance(out[kind], str) and out[kind].startswith("~") else out[kind])
        elif kind == "video" and isinstance(out.get("video"), dict) and isinstance(out["video"].get("frames"), list):
            out["video"] = dict(out["video"], frames=[_file(f) for f in out["video"]["frames"]])
        return out
    return v


class Pipeline:
    """A loaded pipeline. Call it with the inputs Kiln's pipeline of the same task takes; keyword
    arguments are its options (`pipe(text, max_new_tokens=64)`).

    Images: a file path, a Pillow image, a NumPy array [height, width, 3 or 4] of bytes, or a RawImage.
    Audio: a .wav path, its bytes, or 16 kHz samples as a NumPy float array.
    """

    def __init__(self, handle, task, info):
        self._handle = handle
        self.task = task
        self.info = info

    def _args(self, args, options):
        args = list(args)
        if args and self.task in IMAGE_TASKS | AUDIO_TASKS:
            first = args[0]
            if isinstance(first, str) and first.startswith("~"):
                first = os.path.expanduser(first)
            args[0] = [_file(x) for x in first] if isinstance(first, (list, tuple)) else _file(first)
        else:
            args = [_media(a) for a in args]
        if options:
            args.append(options)
        return args

    def _body(self, args, options):
        if self._handle is None:
            raise KilnError("this pipeline was disposed")
        return {"handle": self._handle, "args": self._args(args, options)}

    def __call__(self, *args, **options):
        return BRIDGE.request("call", self._body(args, options))

    def stream(self, *args, **options):
        """Text generation piece by piece: yields the new text as the model writes it.

            for piece in chat.stream(messages, max_new_tokens=128):
                print(piece, end="", flush=True)
        """
        options = dict(options)
        marker = options["streamer"] = lambda text: None
        for kind, a, b in BRIDGE.events("call", self._body(args, options)):
            if kind != "callback":
                return
            if a is marker:
                yield b[0]
            else:
                a(*b)

    def dispose(self):
        """Unloads the model."""
        handle, self._handle = self._handle, None
        if handle is not None and BRIDGE.proc is not None:
            BRIDGE.request("dispose", {"handle": handle})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.dispose()

    def __repr__(self):
        return "Pipeline(task=%r)" % (self.task,)


def pipeline(task, model=None, **options):
    """Loads a model for `task` and returns the pipeline to call.

    `model` is a Hugging Face repo id ("org/name", or "org/name/file.gguf"), or the path of a model
    folder or a .gguf file on disk; left out, the task's default model. Options are Kiln's:
    `device` ("auto", "gpu" or "cpu"), `dtype` ("q4", "q8", "fp32"), `revision`,
    `progress_callback` (called with download progress) and the rest.
    """
    if isinstance(model, os.PathLike):
        model = os.fspath(model)
    if isinstance(model, str) and model.startswith("~"):
        model = os.path.expanduser(model)
    r = BRIDGE.request("pipeline", {"task": task, "model": model, "options": options})
    return Pipeline(r["handle"], r["task"], r.get("info"))


def run_tools(pipe, messages, tools, max_rounds=5, on_text=None, on_tool_call=None, **options):
    """Chats with tools until the model answers without calling one.

    `pipe` is a text-generation pipeline of a model whose chat template supports tools (Qwen2.5,
    Qwen3, Granite). `tools` are OpenAI-style function specs, each with the function that runs it
    under "execute":

        def get_weather(city, unit="celsius"):
            return {"city": city, "temperature": 18, "unit": unit}

        tools = [{"type": "function",
                  "function": {"name": "get_weather", "description": "Get the weather in a city.",
                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
                  "execute": get_weather}]
        result = run_tools(chat, [{"role": "user", "content": "Weather in Paris?"}], tools, max_new_tokens=128)
        print(result["content"])

    `execute` is called with the call's arguments as keyword arguments; what it returns goes back to
    the model as JSON, and an exception as {"error": message}, which the model reads. Other keyword
    arguments are the pipeline's options (`max_new_tokens`, `temperature`, `tool_choice` for the first
    reply only ...). `on_text(piece)` receives the model's text as it is written, `on_tool_call(step)`
    each finished call.

    Returns a dict: "content" (the final answer), "messages" (the conversation with the calls, the
    tools' answers and the reply appended), "steps" (each {"call", "result"} or {"call", "error"})
    and "unfinished" (True when the model was still calling tools after `max_rounds` replies).
    """
    if not tools:
        raise KilnError("run_tools needs at least one tool")
    specs = [{k: v for k, v in t.items() if k != "execute"} for t in tools]
    by_name = {t["function"]["name"]: t for t in tools}
    if len(by_name) < len(tools):
        raise KilnError("two tools share a name; each tool needs its own")
    for t in tools:
        if not callable(t.get("execute")):
            raise KilnError("tool %r has no function under \"execute\"" % t["function"]["name"])
    options = dict(options)
    choice = options.pop("tool_choice", None)
    choice = options.pop("toolChoice", choice)
    history = list(messages)
    steps = []
    content = ""
    for round_ in range(max_rounds):
        extra = {"toolChoice": choice} if choice is not None and round_ == 0 else {}
        text = ""
        for piece in pipe.stream(history, tools=specs, **extra, **options):
            text += piece
            if on_text:
                on_text(piece)
        reply = invoke("parseToolCalls", text)
        content = reply["content"]
        calls = reply["toolCalls"]
        if not calls:
            history.append({"role": "assistant", "content": text})
            return {"content": content, "messages": history, "steps": steps, "unfinished": False}
        history.append({"role": "assistant", "content": content, "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            step = {"call": call}
            try:
                if name not in by_name:
                    raise KilnError("no tool named %r" % name)
                result = by_name[name]["execute"](**call["function"]["arguments"])
                # Characters as they are (not \uXXXX): the model reads this text. A result JSON
                # cannot hold (a datetime, a set) is the tool's error, like one it raised.
                answer = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                step["result"] = result
            except Exception as err:  # the model gets to read what went wrong
                step["error"] = str(err)
                answer = json.dumps({"error": step["error"]}, ensure_ascii=False)
            steps.append(step)
            if on_tool_call:
                on_tool_call(step)
            message = {"role": "tool", "name": name, "content": answer}
            if call.get("id"):
                message["tool_call_id"] = call["id"]
            history.append(message)
    return {"content": content, "messages": history, "steps": steps, "unfinished": True}


def invoke(name, *args):
    """Calls one of Kiln's exported functions by name, for example invoke("cos_sim", a, b)."""
    return BRIDGE.request("invoke", {"name": name, "args": list(args)})


def set_access_token(token):
    """The Hugging Face token for gated or private repos (None to clear it)."""
    BRIDGE.request("invoke", {"name": "setAccessToken", "args": [token]})


def set_env(**values):
    """Sets fields of Kiln's `env` (transformers.js's names: remoteHost, allowLocalModels, localModelPath ...)."""
    BRIDGE.request("env", {"values": values})


def runtime():
    """Node's version and platform, and the tasks with a default model."""
    return BRIDGE.request("hello")


def shutdown():
    """Stops the Node.js process (it also stops when Python exits); the next request starts a new one."""
    BRIDGE.stop()
