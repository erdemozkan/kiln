# Kiln for Python

Run AI models on any GPU (through WebGPU) or on the CPU, straight from their original safetensors or
GGUF files, with the same pipelines as [Kiln.js](https://github.com/erdemozkan/kiln). No CUDA and no
PyTorch: the package has no Python dependencies.

```bash
pip install kiln-webgpu
```

```python
from kiln_webgpu import pipeline

embed = pipeline("feature-extraction", "sentence-transformers/all-MiniLM-L6-v2")
vectors = embed(["Hello world", "Hi"], pooling="mean", normalize=True)
print(vectors.dims)        # [2, 384]
rows = vectors.tolist()    # or vectors.numpy()

chat = pipeline("text-generation", "HuggingFaceTB/SmolLM2-360M-Instruct")
for piece in chat.stream([{"role": "user", "content": "Name three colors."}], max_new_tokens=64):
    print(piece, end="", flush=True)
```

## What it needs

- **Python 3.8 or newer.**
- **Node.js 22.12 or newer** on your PATH (or its path in `KILN_NODE`). The models run in a Node.js
  process that this package starts and talks to; Python sends the inputs and gets the results.
- **For the GPU:** Dawn's WebGPU package for Node.js, installed once with

  ```bash
  python -m kiln_webgpu install-gpu
  ```

  It goes to a cache folder in your home directory (about 140 MB). Without it everything runs on the
  CPU. On a Mac the GPU needs macOS 15 or newer: older releases of that package, the only ones that
  load on an older macOS, can hang the GPU, so there Kiln runs on the CPU and the command says so.

## Using it

`pipeline(task, model, **options)` loads a model; calling the result runs it. Tasks, inputs, options
and results are those of Kiln's JavaScript pipelines (transformers.js's names), with options as
keyword arguments:

```python
classify = pipeline("text-classification")
classify("I love this.")                          # [{"label": "POSITIVE", "score": 0.99...}]

asr = pipeline("automatic-speech-recognition")
asr("speech.wav")                                 # {"text": "..."}

sam = pipeline("mask-generation", "nielsr/slimsam-50-uniform")
masks = sam("photo.jpg", points=[[450, 600]])     # needs Pillow to read the file
masks[0]["mask"].save("mask.png")
```

- **Models:** a Hugging Face repo id (`"org/name"`, or `"org/name/file.gguf"`), or the path of a
  model folder or a `.gguf` file on disk. `set_access_token(token)` for gated repos.
- **Options at load:** `device` (`"auto"`, `"gpu"`, `"cpu"`), `dtype` (`"q4"`, `"q8"`, `"fp32"`),
  `revision`, `progress_callback` (called with download progress).
- **Images:** a file path or a Pillow image (both need Pillow), a NumPy array `[height, width, 3 or 4]`
  of bytes, or a `RawImage(width, height, channels, data)`.
- **Audio:** a `.wav` path, its bytes, or 16 kHz samples as a NumPy float array.
- **Results:** arrays of numbers come back as a `Tensor` (`dims`, `tolist()`, `numpy()`), images as
  a `RawImage` (`save("x.png")`, `numpy()`, `to_pil()`); everything else as lists and dicts.
  `write_wav(path, samples, sampling_rate)` saves generated speech.
- **Streaming:** `pipe.stream(...)` yields text as a text-generation model writes it.
- **Tool calls:** `run_tools(chat, messages, tools, max_new_tokens=128)` chats until the model
  answers in words; each tool is an OpenAI-style function spec with the Python function that runs
  it under `"execute"` (called with the call's arguments as keyword arguments). It returns the
  answer, every call with its result or error, and the conversation.
- **Built-in skills:** `skills()` returns four ready tools for `run_tools`: a calculator, the date
  and time, a Wikipedia lookup and a web page reader.

  ```python
  from kiln_webgpu import pipeline, run_tools, skills

  chat = pipeline("text-generation", "ibm-granite/granite-4.0-1b-GGUF/granite-4.0-1b-Q8_0.gguf")
  result = run_tools(chat, [{"role": "user", "content": "What time is it in Tokyo right now?"}], skills(), max_new_tokens=200)
  print(result["content"])
  ```

  Options: `max_chars`, `language`, `user_agent`, `allow_private`. Each skill also comes alone
  (`calculator_skill()`, `date_time_skill()`, `wikipedia_skill()`, `web_page_skill()`), to mix with
  your own tools, and `skill_instructions()` is the system message for the small Qwen2.5 models.
  The [tools guide](https://github.com/erdemozkan/kiln/blob/main/docs/tools.md) has the details,
  privacy included.
- `pipe.dispose()` unloads a model (or use `with pipeline(...) as pipe:`); `shutdown()` stops Node.

Requests run one at a time. Downloads are not cached in Node.js yet, so a model from the Hub is
fetched again by each new process; a model folder on disk avoids that.

## Building from the repository

```bash
npm install && npm run build:python   # builds Kiln and copies it into python/kiln_webgpu/_js
pip install ./python
npm run test:python
```
