import { ops } from "@kiln/kernels";
import type { Ctx, Tensor } from "@kiln/core";
import { clipTextConfigFromHF, clipTextName, type ClipTextConfig } from "./clip";
import { UnsupportedModelError } from "./config";
import { vitConfigFromHF, vitName, type VitConfig, type VitTail } from "./vit";

/** OWL-ViT and OWLv2: a CLIP pair whose image tower predicts a box and class embedding per patch. */
export interface OwlConfig {
  kind: "owlvit" | "owlv2";
  vision: VitConfig;
  text: ClipTextConfig;
  /** Width of the class embeddings (the text tower's hidden size). */
  queryDim: number;
}

/** Columns past the class embedding in the class head's output: logit shift, logit scale, zero padding. */
export const OWL_CLASS_EXTRA = 32;

export function owlConfigFromHF(config: Record<string, unknown>): OwlConfig {
  const kind = String(config.model_type ?? "");
  if (kind !== "owlvit" && kind !== "owlv2") throw new UnsupportedModelError(`model type "${kind}" is not OWL-ViT or OWLv2`);
  // Owl{ViT,v2}VisionConfig's defaults where they differ from CLIP's.
  const vc = { image_size: 768, patch_size: kind === "owlv2" ? 16 : 32, ...((config.vision_config as Record<string, unknown> | undefined) ?? {}) };
  const vision = vitConfigFromHF({ ...vc, model_type: "clip_vision_model" });
  const tc = { max_position_embeddings: 16, ...((config.text_config as Record<string, unknown> | undefined) ?? {}) };
  // OWL pools the text at its highest token id (the end token), whatever eos_token_id says.
  const text = { ...clipTextConfigFromHF({ ...config, text_config: tc }), eosId: 2 };
  return { kind, vision, text, queryDim: text.hidden };
}

const strip = (name: string) => name.replace(/^owl(?:v2|vit)\./, "");

/**
 * Kiln's name for an OWL image-side tensor loaded as is: the vision encoder (as a CLIP tower), the
 * post-LayerNorm ("lnPost"), the patch LayerNorm ("lnOwl") and the box MLP ("box.N"). Embeddings and
 * the class head are loaded otherwise; OWLv2's objectness head, which text queries do not use, is not.
 */
export function owlVisionName(name: string): string | undefined {
  const n = strip(name).replace(/^vision_model\.pre_layernorm\./, "vision_model.pre_layrnorm.");
  let m = /^vision_model\.post_layernorm\.(weight|bias)$/.exec(n);
  if (m) return `lnPost.${m[1]}`;
  m = /^layer_norm\.(weight|bias)$/.exec(n);
  if (m) return `lnOwl.${m[1]}`;
  m = /^box_head\.dense(\d)\.(weight|bias)$/.exec(n);
  if (m) return `box.${m[1]}.${m[2]}`;
  return n.startsWith("vision_model.") ? vitName(n) : undefined;
}

/** Kiln's name for an OWL text tensor (as CLIP's text tower), or undefined. */
export function owlTextName(name: string): string | undefined {
  return clipTextName(strip(name));
}

/**
 * The box bias of a gh × gw patch grid [gh · gw, 4] (center x, y, width, height before the sigmoid),
 * in float32 as transformers computes it: each patch's lower-right corner as the center, the patch
 * size as the size, both as logits (log(p + 1e-4) − log1p(−p + 1e-4)).
 */
export function owlBoxBias(gh: number, gw: number): Float32Array {
  const f = Math.fround;
  const e = f(1e-4);
  const logit = (p: number) => f(f(Math.log(f(p + e))) - f(Math.log1p(f(-p + e))));
  const out = new Float32Array(gh * gw * 4);
  const sw = logit(f(1 / gw));
  const sh = logit(f(1 / gh));
  for (let y = 0; y < gh; y++)
    for (let x = 0; x < gw; x++) {
      const i = (y * gw + x) * 4;
      out[i] = logit(Math.min(f((x + 1) / gw), 1));
      out[i + 1] = logit(Math.min(f((y + 1) / gh), 1));
      out[i + 2] = sw;
      out[i + 3] = sh;
    }
  return out;
}

/**
 * OWL's heads after the image encoder, run in the encoder's plan: post-LayerNorm of every token, the
 * patches times the class token, LayerNorm → per patch
 * - "cls" [patches, queryDim + 32]: the class embedding, then the logit shift and scale (the class
 *   head's three linears stacked as one weight "cls.weight");
 * - "box" [patches, 4]: the box MLP (exact GELU) plus the box bias ("boxBias"), before the sigmoid.
 */
export function owlTail(cfg: OwlConfig): VitTail {
  return (ctx: Ctx, hidden: Tensor, w: (name: string) => Tensor) => {
    const eps = cfg.vision.eps;
    const post = ops.layernorm(ctx, hidden, w("lnPost.weight"), w("lnPost.bias"), eps);
    const feats = ops.layernorm(ctx, ops.mulFirstRow(ctx, post), w("lnOwl.weight"), w("lnOwl.bias"), eps);
    const lin = (x: Tensor, p: string) => ops.linear(ctx, x, w(`${p}.weight`), { bias: w(`${p}.bias`) });
    const h = lin(feats, "box.0");
    ops.act(ctx, h, "gelu");
    const h2 = lin(h, "box.1");
    ops.act(ctx, h2, "gelu");
    const box = lin(h2, "box.2");
    ops.addInPlace(ctx, box, w("boxBias"));
    return { cls: lin(feats, "cls"), box };
  };
}
