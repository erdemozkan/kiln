"""pipeline(): Kiln's transformers.js-style pipelines, called from Python."""
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
