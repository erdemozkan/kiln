// Pictures from text with a diffusers Stable Diffusion checkpoint whose decoder is TAESD (SDXS and
// the like): CLIP's text tower reads the prompt, the UNet takes noise to an image latent in a few
// Euler steps, TAESD turns the latent into pixels.
import type { Backend, Tensor } from "@kiln/core";
import { fetchJSON, fetchText, fileUrl, listRepoFiles, parseModelId, readSafetensorsHeader, toF32, type ByteSource, type TensorEntry } from "@kiln/loader";
import { ClipTextModel, TaesdDecoder, UNetModel, clipTextConfigFromHF, clipTextName, convMatrix, matVec, silu, taesdConfigFromDiffusers, taesdLayers, timestepFeatures, unetConfigFromDiffusers, unetResnets, type UNetConfig } from "@kiln/models";
import { ClipTokenizer } from "@kiln/tokenizers";
import { openParsed, withBackend } from "./load";
import { loadWeights, type LoadProgress } from "./weights";

export interface TextToImageLoadOptions {
  /** The text tower's matrices: "f32" (default), "q8" or "q4". */
  quant?: "f32" | "q8" | "q4";
  device?: "auto" | "gpu" | "cpu";
  cpuThreads?: number;
  revision?: string;
  endpoint?: string;
  backend?: Backend;
  onProgress?: (p: LoadProgress) => void;
}

export interface TextToImageOptions {
  /** Same seed, same prompt, same picture (default: a random seed). */
  seed?: number;
  /** Image size in pixels, a multiple of 32 (default 512 × 512, what SDXS-512 was trained at). */
  width?: number;
  height?: number;
  /** Denoising steps (default 1: SDXS makes a picture in one). */
  num_inference_steps?: number;
  /** Classifier-free guidance; above 1 runs the UNet twice per step (default 0: none, as SDXS wants). */
  guidance_scale?: number;
  /** What to steer away from, with guidance above 1 (default ""). */
  negative_prompt?: string;
  /** Starting noise [4, height / 8, width / 8] (channels first, as diffusers holds it) instead of a seeded one. */
  latents?: Float32Array;
}

/** A generated picture: RGB bytes, row by row. */
export interface GeneratedImage {
  width: number;
  height: number;
  channels: 3;
  data: Uint8Array;
}

/**
 * The scheduler settings this runs. "euler" is EulerDiscreteScheduler; "ddim" a first-order DDIM
 * step, which is what DEISMultistepScheduler and DPMSolverMultistepScheduler do for one step (and
 * DDIMScheduler always); for several steps those two would use higher orders, which this does not.
 */
export interface SchedulerConfig {
  kind: "euler" | "ddim";
  betaStart: number;
  betaEnd: number;
  schedule: "scaled_linear" | "linear";
  trainSteps: number;
  spacing: "trailing" | "leading" | "linspace";
  stepsOffset: number;
  /** ᾱ after the last step: 1, or the first training step's (DDIM's set_alpha_to_one). */
  alphaToOne: boolean;
}

/** The scheduler from its class (model_index.json's, which diffusers goes by) and scheduler_config.json. */
export function schedulerFromDiffusers(className: string, c: Record<string, unknown>): SchedulerConfig {
  const kinds: Record<string, "euler" | "ddim"> = {
    EulerDiscreteScheduler: "euler",
    DEISMultistepScheduler: "ddim",
    DPMSolverMultistepScheduler: "ddim",
    DDIMScheduler: "ddim",
  };
  const kind = kinds[className];
  if (!kind) throw new Error(`scheduler "${className}" is not supported`);
  if ((c.prediction_type ?? "epsilon") !== "epsilon") throw new Error(`prediction type "${String(c.prediction_type)}" is not supported (epsilon)`);
  const schedule = String(c.beta_schedule ?? "linear");
  if (schedule !== "scaled_linear" && schedule !== "linear") throw new Error(`beta schedule "${schedule}" is not supported`);
  const spacing = String(c.timestep_spacing ?? "leading");
  if (spacing !== "trailing" && spacing !== "leading" && spacing !== "linspace") throw new Error(`timestep spacing "${spacing}" is not supported`);
  return {
    kind,
    betaStart: Number(c.beta_start ?? 0.0001),
    betaEnd: Number(c.beta_end ?? 0.02),
    schedule,
    trainSteps: Number(c.num_train_timesteps ?? 1000),
    spacing,
    stepsOffset: Number(c.steps_offset ?? 0),
    alphaToOne: c.set_alpha_to_one === true,
  };
}

/** ᾱ at every training timestep (in float32, as torch computes it). */
export function alphasCumprod(s: SchedulerConfig): Float32Array {
  const N = s.trainSteps;
  const ac = new Float32Array(N);
  let p = 1;
  for (let i = 0; i < N; i++) {
    const f = N === 1 ? 0 : i / (N - 1);
    const beta = s.schedule === "scaled_linear" ? (Math.sqrt(s.betaStart) + f * (Math.sqrt(s.betaEnd) - Math.sqrt(s.betaStart))) ** 2 : s.betaStart + f * (s.betaEnd - s.betaStart);
    ac[i] = p = Math.fround(p * Math.fround(1 - Math.fround(beta)));
  }
  return ac;
}

/** The timesteps of `n` steps, the noisiest first. */
export function scheduleTimesteps(s: SchedulerConfig, n: number): number[] {
  const N = s.trainSteps;
  if (s.spacing === "trailing") return Array.from({ length: n }, (_, i) => Math.round(N - (i * N) / n) - 1);
  if (s.spacing === "leading") {
    const r = Math.floor(N / n);
    return Array.from({ length: n }, (_, i) => (n - 1 - i) * r + s.stepsOffset);
  }
  return Array.from({ length: n }, (_, i) => Math.round((N - 1) * (1 - (n === 1 ? 0 : i / (n - 1)))));
}

/** A Gaussian noise generator from a seed (mulberry32 and Box–Muller). */
export function gaussianNoise(seed: number, n: number): Float32Array {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(next(), 1e-12);
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/** CHW → HWC (and back with `back`). */
function transpose(x: Float32Array, C: number, HW: number, back = false): Float32Array {
  const y = new Float32Array(x.length);
  for (let c = 0; c < C; c++) for (let i = 0; i < HW; i++) back ? (y[c * HW + i] = x[i * C + c]!) : (y[i * C + c] = x[c * HW + i]!);
  return y;
}

/** A loaded text-to-image model. */
export class TextToImage {
  private biasesAt?: number;

  constructor(
    readonly backend: Backend,
    private readonly tokenizer: ClipTokenizer,
    private readonly padId: number,
    private readonly text: ClipTextModel,
    readonly unet: UNetModel,
    private readonly decoder: TaesdDecoder,
    private readonly scheduler: SchedulerConfig,
    /** Each ResNet block's conv1 bias plus its time projection, for a timestep. */
    private readonly timeBiases: (t: number) => Map<string, Float32Array>,
    private readonly latentScale: number,
    /** Latent pixels per image pixel along a side. */
    readonly factor: number,
  ) {}

  get cfg(): UNetConfig {
    return this.unet.cfg;
  }

  /** The text context [77, hidden] of a prompt: its tokens padded to 77 with the pad token. */
  async encodePrompt(prompt: string): Promise<{ ids: number[]; hidden: Float32Array }> {
    const L = this.unet.contextLength;
    const ids = this.tokenizer.encode(prompt).slice(0, L);
    if (ids.length === L) ids[L - 1] = this.tokenizer.eos;
    while (ids.length < L) ids.push(this.padId);
    return { ids, hidden: await this.text.hiddenStates(ids) };
  }

  private async eps(latent: Float32Array, H: number, W: number, t: number, context: Float32Array): Promise<Float32Array> {
    if (this.biasesAt !== t) {
      this.unet.setTimeBiases(this.timeBiases(t));
      this.biasesAt = t;
    }
    return this.unet.predict(latent, H, W, context);
  }

  /** A picture for `prompt`. */
  async generate(prompt: string, o: TextToImageOptions = {}): Promise<GeneratedImage> {
    const width = o.width ?? 512;
    const height = o.height ?? 512;
    const f = this.factor * 2 ** (this.cfg.channels.length - 1);
    if (width % f || height % f || width < f || height < f) throw new Error(`text-to-image: width and height must be multiples of ${f}, got ${width} × ${height}`);
    const steps = o.num_inference_steps ?? 1;
    if (!Number.isInteger(steps) || steps < 1 || steps > 100) throw new Error(`text-to-image: num_inference_steps must be 1 to 100, got ${steps}`);
    const g = o.guidance_scale ?? 0;
    const [H, W] = [height / this.factor, width / this.factor];
    const C = this.cfg.inChannels;
    const n = H * W * C;
    let z: Float32Array;
    if (o.latents) {
      if (o.latents.length !== n) throw new Error(`text-to-image: latents must hold ${C} × ${H} × ${W} values, got ${o.latents.length}`);
      z = transpose(o.latents, C, H * W);
    } else z = gaussianNoise(o.seed ?? Math.floor(Math.random() * 2 ** 32), n);
    const cond = (await this.encodePrompt(prompt)).hidden;
    const uncond = g > 1 ? (await this.encodePrompt(o.negative_prompt ?? "")).hidden : undefined;

    const x = await this.denoise(z, H, W, steps, cond, uncond, g);
    return this.decode(x, H, W, true);
  }

  /** The latent [H·W, C] (channels last) the scheduler reaches from noise z in `steps` steps. */
  async denoise(z: Float32Array, H: number, W: number, steps: number, cond: Float32Array, uncond?: Float32Array, g = 0): Promise<Float32Array> {
    const n = z.length;
    const ts = scheduleTimesteps(this.scheduler, steps);
    const ac = alphasCumprod(this.scheduler);
    const euler = this.scheduler.kind === "euler";
    // σ = √((1 − ᾱ) / ᾱ). Euler starts at the noise times √(σ₀² + 1) and feeds x / √(σ² + 1); DDIM starts at the noise and feeds x.
    const sigma = (t: number) => Math.sqrt((1 - ac[t]!) / ac[t]!);
    const x = euler ? z.map((v) => v * Math.sqrt(sigma(ts[0]!) ** 2 + 1)) : Float32Array.from(z);
    for (let i = 0; i < steps; i++) {
      const t = ts[i]!;
      const k = euler ? 1 / Math.sqrt(sigma(t) ** 2 + 1) : 1;
      const input = k === 1 ? x : x.map((v) => v * k);
      let e = await this.eps(input, H, W, t, cond);
      if (uncond) {
        const u = await this.eps(input, H, W, t, uncond);
        e = e.map((v, j) => u[j]! + g * (v - u[j]!));
      }
      const next = ts[i + 1];
      if (euler) {
        const dt = (next === undefined ? 0 : sigma(next)) - sigma(t);
        for (let j = 0; j < n; j++) x[j] = x[j]! + e[j]! * dt;
      } else {
        // x₀ = (x − √(1 − ᾱₜ) ε) / √ᾱₜ, then x = √ᾱₚ x₀ + √(1 − ᾱₚ) ε.
        const at = ac[t]!;
        const ap = next !== undefined ? ac[next]! : this.scheduler.alphaToOne ? 1 : ac[0]!;
        for (let j = 0; j < n; j++) {
          const x0 = (x[j]! - Math.sqrt(1 - at) * e[j]!) / Math.sqrt(at);
          x[j] = Math.sqrt(ap) * x0 + Math.sqrt(1 - ap) * e[j]!;
        }
      }
    }
    return x;
  }

  /** Pixels of a latent [H·W, C] (channels last; `channelsLast` false: [C, H, W]). */
  async decode(latent: Float32Array, H: number, W: number, channelsLast = false): Promise<GeneratedImage> {
    const C = this.cfg.inChannels;
    const x = channelsLast ? latent : transpose(latent, C, H * W);
    // DecoderTiny clamps its input softly: tanh(x / 3) · 3.
    const clamped = x.map((v) => Math.tanh(v / this.latentScale / 3) * 3);
    const { pixels, width, height } = await this.decoder.decode(clamped, H, W);
    // DecoderTiny returns 2p − 1 and the pipeline maps that back to p: the pixels are p, clamped to 0…1.
    const data = new Uint8Array(width * height * 3);
    for (let i = 0; i < data.length; i++) data[i] = Math.round(Math.min(Math.max(pixels[i]!, 0), 1) * 255);
    return { width, height, channels: 3, data };
  }

  destroy(): void {
    this.text.destroy();
    this.unet.destroy();
    this.decoder.destroy();
  }
}

/** Loads a diffusers Stable Diffusion repo with a TAESD decoder (SDXS-512-DreamShaper, SDXS-512-0.9) from the Hugging Face Hub. */
export function createTextToImage(modelId: string, opts: TextToImageLoadOptions = {}): Promise<TextToImage> {
  return withBackend(opts, (o) => loadTextToImage(modelId, o));
}

async function loadTextToImage(modelId: string, opts: TextToImageLoadOptions): Promise<TextToImage> {
  const ref = parseModelId(modelId, opts.revision);
  const ep = opts.endpoint;
  const json = <T>(f: string, optional = false) => fetchJSON<T>(fileUrl(ref, f, ep), optional);
  const index = (await json<Record<string, unknown>>("model_index.json"))!;
  if (index._class_name !== "StableDiffusionPipeline") throw new Error(`${modelId}: a "${String(index._class_name)}" is not supported (StableDiffusionPipeline)`);
  const [unetJson, vaeJson, schedJson, textJson, vocab, merges, special] = await Promise.all([
    json<Record<string, unknown>>("unet/config.json"),
    json<Record<string, unknown>>("vae/config.json"),
    json<Record<string, unknown>>("scheduler/scheduler_config.json"),
    json<Record<string, unknown>>("text_encoder/config.json"),
    json<Record<string, number>>("tokenizer/vocab.json"),
    fetchText(fileUrl(ref, "tokenizer/merges.txt", ep)),
    json<Record<string, unknown>>("tokenizer/special_tokens_map.json", true),
  ]);
  const ucfg = unetConfigFromDiffusers(unetJson!);
  if (vaeJson!._class_name !== "AutoencoderTiny") throw new Error(`${modelId}: its decoder is a "${String(vaeJson!._class_name)}"; only TAESD (AutoencoderTiny) is supported so far`);
  const tcfg = taesdConfigFromDiffusers(vaeJson!);
  const scheduler = schedulerFromDiffusers(String((index.scheduler as string[] | undefined)?.[1] ?? schedJson!._class_name), schedJson!);
  const textCfg = clipTextConfigFromHF({ text_config: textJson });
  if (textCfg.hidden !== ucfg.crossDim) throw new Error(`${modelId}: the text tower is ${textCfg.hidden} wide, the UNet's cross-attention ${ucfg.crossDim}`);
  const tokenizer = new ClipTokenizer({ model: { type: "BPE", vocab: vocab!, merges: merges!.split("\n").filter((l) => l && !l.startsWith("#version")) } }, textCfg.maxPositions);
  const padTok = special?.pad_token;
  const padName = typeof padTok === "string" ? padTok : ((padTok as { content?: string } | undefined)?.content ?? "<|endoftext|>");
  const padId = vocab![padName] ?? tokenizer.eos;

  const backend = opts.backend!;
  const sources: ByteSource[] = [];
  const open = async (file: string) => {
    const { source, header } = await openParsed(ref, file, ep, readSafetensorsHeader);
    sources.push(source);
    return { source, tensors: header.tensors };
  };
  const files = await listRepoFiles(ref, ep);
  const pick = (dir: string, base: string) => {
    const names = [`${dir}/${base}.safetensors`, `${dir}/${base}.fp16.safetensors`];
    const found = names.find((n) => !files || files.has(n));
    if (!found) throw new Error(`${modelId}: no ${dir}/${base}.safetensors`);
    return found;
  };
  try {
    // Text tower.
    const tf = await open(pick("text_encoder", "model"));
    const textWeights = await loadWeights(backend, [{ name: "text_encoder", source: tf.source, tensors: tf.tensors.filter((t) => clipTextName(t.name) !== undefined), rename: clipTextName }], opts.quant ?? "f32", opts.onProgress);
    const text = new ClipTextModel(backend, textCfg, textWeights);

    // UNet: every tensor read as f32 and reshaped for its kernel.
    const uf = await open(pick("unet", "diffusion_pytorch_model"));
    const read = async (src: ByteSource, t: TensorEntry) => toF32(t.type as Parameters<typeof toF32>[0], await src.read(t.offset, t.byteLength), t.shape.reduce((a, b) => a * b, 1));
    const unetWeights = new Map<string, Tensor>();
    const upload = (into: Map<string, Tensor>, name: string, shape: number[], data: Float32Array) => {
      const buf = backend.alloc(data.byteLength, "storage", `diffusion/${name}`);
      backend.write(buf, 0, data);
      into.set(name, { shape, dtype: "f32", buf } as Tensor);
    };
    const cpu = new Map<string, Float32Array>();
    const total = uf.tensors.reduce((a, t) => a + t.byteLength, 0);
    let loaded = 0;
    for (const t of uf.tensors) {
      const data = await read(uf.source, t);
      loaded += t.byteLength;
      opts.onProgress?.({ file: "unet", loaded, total });
      const name = t.name.replace(/\.weight$|\.bias$/, "");
      const kind = t.name.endsWith(".weight") ? "weight" : "bias";
      if (name.startsWith("time_embedding.") || name.endsWith(".time_emb_proj")) {
        cpu.set(t.name, data);
        continue;
      }
      if (name.endsWith(".conv1") && kind === "bias") cpu.set(t.name, data);
      const isProj = /\.(proj_in|proj_out)$/.test(name);
      if (kind === "weight" && t.shape.length === 4 && !isProj) {
        const m = convMatrix(data, t.shape);
        upload(unetWeights, t.name, [t.shape[0]!, m.K], m.w);
      } else if (kind === "weight" && t.shape.length === 4) {
        upload(unetWeights, t.name, [t.shape[0]!, t.shape[1]!], data);
      } else if (/\.ff\.net\.0\.proj$/.test(name)) {
        // GEGLU: diffusers' [hidden | gate] rows → [gate | hidden], the order gatedAct reads.
        const rows = t.shape[0]! / 2;
        const per = data.length / t.shape[0]!;
        const swapped = new Float32Array(data.length);
        swapped.set(data.subarray(rows * per), 0);
        swapped.set(data.subarray(0, rows * per), rows * per);
        upload(unetWeights, t.name, [...t.shape], swapped);
      } else upload(unetWeights, t.name, [...t.shape], data);
    }
    const unet = new UNetModel(backend, ucfg, unetWeights);
    const resnets = unetResnets(ucfg);
    const need = (k: string) => {
      const v = cpu.get(k);
      if (!v) throw new Error(`${modelId}: "unet/${k}" is missing`);
      return v;
    };
    const timeBiases = (t: number) => {
      const feats = timestepFeatures(t, ucfg.timeDim, ucfg.flipSinToCos, ucfg.freqShift);
      const h = silu(matVec(need("time_embedding.linear_1.weight"), need("time_embedding.linear_1.bias"), feats));
      const emb = silu(matVec(need("time_embedding.linear_2.weight"), need("time_embedding.linear_2.bias"), h));
      const out = new Map<string, Float32Array>();
      for (const r of resnets) {
        const add = matVec(need(`${r.prefix}.time_emb_proj.weight`), need(`${r.prefix}.time_emb_proj.bias`), emb);
        const b = need(`${r.prefix}.conv1.bias`);
        out.set(r.prefix, b.map((v, i) => v + add[i]!));
      }
      return out;
    };

    // TAESD's decoder.
    const vf = await open(pick("vae", "diffusion_pytorch_model"));
    const located = new Map(vf.tensors.map((t) => [t.name, t]));
    const vaeWeights = new Map<string, Tensor>();
    const conv = async (p: string, bias: boolean) => {
      const w = located.get(`${p}.weight`);
      if (!w) throw new Error(`${modelId}: "vae/${p}.weight" is missing`);
      const m = convMatrix(await read(vf.source, w), w.shape);
      upload(vaeWeights, `${p}.weight`, [w.shape[0]!, m.K], m.w);
      const b = bias ? located.get(`${p}.bias`) : undefined;
      upload(vaeWeights, `${p}.bias`, [w.shape[0]!], b ? await read(vf.source, b) : new Float32Array(w.shape[0]!));
    };
    for (const l of taesdLayers(tcfg)) {
      if (l.kind === "conv") await conv(`decoder.layers.${l.index}`, l.bias);
      else if (l.kind === "block") for (const k of [0, 2, 4]) await conv(`decoder.layers.${l.index}.conv.${k}`, true);
    }
    const decoder = new TaesdDecoder(backend, tcfg, vaeWeights);
    const latentScale = Number(vaeJson!.scaling_factor ?? 1);
    return new TextToImage(backend, tokenizer, padId, text, unet, decoder, scheduler, timeBiases, latentScale, 2 ** (tcfg.blocks.length - 1));
  } finally {
    for (const s of sources) s.close?.();
  }
}

