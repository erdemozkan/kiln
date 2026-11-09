// Image preprocessing as transformers' image processors do it (PIL resampling, center crop, rescale,
// normalize), so pixel values match Python transformers exactly, not just a canvas resize.

/** An image as 8-bit RGB or RGBA rows. */
export interface RgbImage {
  width: number;
  height: number;
  /** width · height · channels bytes, channels 3 or 4. */
  data: Uint8Array | Uint8ClampedArray;
}

type Filter = { support: number; f: (x: number) => number };

const BICUBIC_A = -0.5;
/** PIL's resampling filters, by transformers' `resample` numbers (PIL enum values). */
const FILTERS: Record<number, Filter> = {
  // 0 NEAREST is handled separately.
  1: {
    // LANCZOS
    support: 3,
    f: (x) => {
      const sinc = (v: number) => (v === 0 ? 1 : Math.sin(Math.PI * v) / (Math.PI * v));
      return x > -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0;
    },
  },
  2: { support: 1, f: (x) => (x < 0 ? (x > -1 ? 1 + x : 0) : x < 1 ? 1 - x : 0) }, // BILINEAR
  3: {
    // BICUBIC
    support: 2,
    f: (x) => {
      const a = BICUBIC_A;
      x = Math.abs(x);
      if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
      if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
      return 0;
    },
  },
  4: { support: 0.5, f: (x) => (x >= -0.5 && x < 0.5 ? 1 : 0) }, // BOX
  5: {
    // HAMMING
    support: 1,
    f: (x) => {
      if (x < 0) x = -x;
      if (x === 0) return 1;
      if (x >= 1) return 0;
      x *= Math.PI;
      return (Math.sin(x) / x) * (0.54 + 0.46 * Math.cos(x));
    },
  },
};

const PRECISION_BITS = 32 - 8 - 2;

/** PIL's precompute_coeffs + normalize_coeffs_8bpc: per output pixel, [first input, count] and fixed-point weights. */
function coeffs(inSize: number, outSize: number, filter: Filter): { bounds: Int32Array; k: Int32Array; ksize: number } {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Float64Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const w = filter.f((x + xmin - center + 0.5) * ss);
      kk[xx * ksize + x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) if (ww !== 0) kk[xx * ksize + x] /= ww;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  const k = new Int32Array(outSize * ksize);
  for (let i = 0; i < kk.length; i++) k[i] = Math.trunc(kk[i]! < 0 ? -0.5 + kk[i]! * (1 << PRECISION_BITS) : 0.5 + kk[i]! * (1 << PRECISION_BITS));
  return { bounds, k, ksize };
}

const clip8 = (v: number) => {
  const r = v >> PRECISION_BITS;
  return r < 0 ? 0 : r > 255 ? 255 : r;
};

/**
 * PIL's Image.resize for 8-bit images (ImagingResample): a horizontal then a vertical pass, each
 * rounding to bytes, with PIL's fixed-point weights. `resample` uses PIL's numbering (0 nearest,
 * 1 lanczos, 2 bilinear, 3 bicubic, 4 box, 5 hamming), as transformers' configs do.
 */
export function resizeImage(img: RgbImage, width: number, height: number, resample = 2): RgbImage {
  const C = img.data.length / (img.width * img.height);
  // One channel too: greyscale (PIL's "L") resamples the same way, e.g. a mask.
  if (C !== 1 && C !== 3 && C !== 4) throw new Error("image data must be grey, RGB or RGBA bytes");
  if (width === img.width && height === img.height) return { width, height, data: Uint8Array.from(img.data) };
  if (resample === 0) {
    // PIL's nearest: source pixel floor((x + 0.5) · scale).
    const out = new Uint8Array(width * height * C);
    for (let y = 0; y < height; y++) {
      const sy = Math.min(img.height - 1, Math.floor((y + 0.5) * (img.height / height)));
      for (let x = 0; x < width; x++) {
        const sx = Math.min(img.width - 1, Math.floor((x + 0.5) * (img.width / width)));
        for (let c = 0; c < C; c++) out[(y * width + x) * C + c] = img.data[(sy * img.width + sx) * C + c]!;
      }
    }
    return { width, height, data: out };
  }
  const filter = FILTERS[resample];
  if (!filter) throw new Error(`unknown resample filter ${resample}`);
  let src: RgbImage = img;
  if (width !== img.width) {
    const { bounds, k, ksize } = coeffs(img.width, width, filter);
    const out = new Uint8Array(width * img.height * C);
    for (let y = 0; y < img.height; y++)
      for (let x = 0; x < width; x++) {
        const xmin = bounds[x * 2]!;
        const xmax = bounds[x * 2 + 1]!;
        for (let c = 0; c < C; c++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let i = 0; i < xmax; i++) ss += img.data[(y * img.width + xmin + i) * C + c]! * k[x * ksize + i]!;
          out[(y * width + x) * C + c] = clip8(ss);
        }
      }
    src = { width, height: img.height, data: out };
  }
  if (height !== src.height) {
    const { bounds, k, ksize } = coeffs(src.height, height, filter);
    const out = new Uint8Array(width * height * C);
    for (let y = 0; y < height; y++) {
      const ymin = bounds[y * 2]!;
      const ymax = bounds[y * 2 + 1]!;
      for (let x = 0; x < width; x++)
        for (let c = 0; c < C; c++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let i = 0; i < ymax; i++) ss += src.data[((ymin + i) * width + x) * C + c]! * k[y * ksize + i]!;
          out[(y * width + x) * C + c] = clip8(ss);
        }
    }
    src = { width, height, data: out };
  }
  return src;
}

/** The middle width × height of an image (numpy-style: offsets floor((in - out) / 2)), zero-padded if smaller. */
export function centerCrop(img: RgbImage, width: number, height: number): RgbImage {
  const C = img.data.length / (img.width * img.height);
  const top = Math.floor((img.height - height) / 2);
  const left = Math.floor((img.width - width) / 2);
  const out = new Uint8Array(width * height * C);
  for (let y = 0; y < height; y++) {
    const sy = y + top;
    if (sy < 0 || sy >= img.height) continue;
    for (let x = 0; x < width; x++) {
      const sx = x + left;
      if (sx < 0 || sx >= img.width) continue;
      for (let c = 0; c < C; c++) out[(y * width + x) * C + c] = img.data[(sy * img.width + sx) * C + c]!;
    }
  }
  return { width, height, data: out };
}

/** RGB bytes of an RGB or RGBA image (alpha dropped). */
export function toRgb(img: RgbImage): RgbImage {
  const n = img.width * img.height;
  if (img.data.length === n * 3) return img;
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[i * 3 + c] = img.data[i * 4 + c]!;
  return { width: img.width, height: img.height, data: out };
}

/** The fields of a preprocessor_config.json that Kiln's image processors read. */
export interface ImageProcessorConfig {
  image_processor_type?: string;
  feature_extractor_type?: string;
  do_resize?: boolean;
  size?: number | { height?: number; width?: number; shortest_edge?: number; longest_edge?: number };
  resample?: number;
  crop_pct?: number;
  do_center_crop?: boolean;
  crop_size?: number | { height?: number; width?: number };
  do_rescale?: boolean;
  rescale_factor?: number;
  do_normalize?: boolean;
  image_mean?: number | number[];
  image_std?: number | number[];
  do_convert_rgb?: boolean;
  /** DPT: scale both sides by the same factor (the one nearer 1) instead of squashing to `size`. */
  keep_aspect_ratio?: boolean;
  /** DPT: round each output side to a multiple of this. */
  ensure_multiple_of?: number;
}

/** Processors whose integer `size` means the shortest edge (transformers' default_to_square=False). */
const SHORTEST_EDGE = /CLIP|ConvNext|Bit|MobileNet|MobileViT|LevitImage|EfficientNet|Chinese/;

/** Output size of transformers' shortest-edge resize: the short side becomes `size`, the long side int(size · long / short). */
function shortestEdge(w: number, h: number, size: number, longest?: number): [number, number] {
  const short = Math.min(w, h);
  const long = Math.max(w, h);
  let ns = size;
  let nl = Math.trunc((size * long) / short);
  if (longest !== undefined && nl > longest) {
    ns = Math.trunc((longest * ns) / nl);
    nl = longest;
  }
  return w <= h ? [ns, nl] : [nl, ns];
}

const triple = (v: number | number[] | undefined, d: number[]): number[] => (v === undefined ? d : typeof v === "number" ? [v, v, v] : v);

/** Python's round(): halves go to the even neighbour. */
function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  return d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f;
}

/**
 * Output [width, height] of DPT's resize (transformers' DPTImageProcessor): to `width` × `height`,
 * or with `keep` both sides scaled by whichever factor is nearer 1; each side then rounded
 * (half to even) to a multiple of `multiple`.
 */
export function dptResizeSize(w: number, h: number, width: number, height: number, keep: boolean, multiple: number): [number, number] {
  let sh = height / h;
  let sw = width / w;
  if (keep) {
    if (Math.abs(1 - sw) < Math.abs(1 - sh)) sh = sw;
    else sw = sh;
  }
  const fit = (v: number) => {
    const x = roundHalfEven(v / multiple) * multiple;
    return x < 0 ? Math.ceil(v / multiple) * multiple : x;
  };
  return [fit(sw * w), fit(sh * h)];
}

/**
 * Output [width, height] of the DETR-family resize (transformers' get_size_with_aspect_ratio): the
 * short side becomes `size` unless the long side would pass `max`, then the long side becomes about
 * `max`. YOLOS checks the orientation first and rounds both sides down to multiples of 16.
 */
export function detrResizeSize(w: number, h: number, size: number, max: number | undefined, yolos: boolean): [number, number] {
  let raw: number | undefined;
  if (max !== undefined) {
    const lo = Math.min(w, h);
    const hi = Math.max(w, h);
    if ((hi / lo) * size > max) {
      raw = (max * lo) / hi;
      size = roundHalfEven(raw);
    }
  }
  const s = raw ?? size;
  const keep = (h <= w && h === size) || (w <= h && w === size);
  let ow: number;
  let oh: number;
  if (!yolos && keep) [ow, oh] = [w, h];
  else if (w < h) [ow, oh] = [size, Math.trunc((s * h) / w)];
  else if (yolos && keep) [ow, oh] = [w, h];
  else [ow, oh] = [Math.trunc((s * w) / h), size];
  if (yolos) [ow, oh] = [ow - (ow % 16), oh - (oh % 16)];
  return [ow, oh];
}

/**
 * Pixel values [3, height, width] of an image as the model's image processor makes them (Python
 * transformers' slow processors: PIL resize, center crop, rescale and normalize in float32).
 */
export function preprocessImage(img: RgbImage, cfg: ImageProcessorConfig): { data: Float32Array; width: number; height: number } {
  const kind = cfg.image_processor_type ?? cfg.feature_extractor_type ?? "";
  // transformers' per-processor default when the config leaves it out: bicubic for these, else bilinear.
  const resample = cfg.resample ?? (/DeiT|CLIP|Beit|BEiT|Bit|DPT|Siglip|Chinese|Owl/.test(kind) ? 3 : 2);
  // Images go in as RGB (PIL's convert("RGB") drops alpha).
  let im = toRgb(img);
  if (cfg.do_resize !== false && cfg.size !== undefined) {
    // An integer size on a shortest-edge processor is its shortest edge (get_size_dict with
    // default_to_square=False), so ConvNeXt's crop_pct applies to it too.
    const s = typeof cfg.size === "number" && SHORTEST_EDGE.test(kind) ? { shortest_edge: cfg.size } : cfg.size;
    let w: number;
    let h: number;
    if (typeof s === "number") {
      if (SHORTEST_EDGE.test(kind)) [w, h] = shortestEdge(im.width, im.height, s);
      else [w, h] = [s, s];
    } else if (s.shortest_edge !== undefined) {
      let edge = s.shortest_edge;
      // ConvNeXt: below 384, resize to shortest_edge / crop_pct, then crop shortest_edge.
      if (cfg.crop_pct !== undefined && /ConvNext/.test(kind) && edge < 384) edge = Math.trunc(edge / cfg.crop_pct);
      if (/Yolos|Detr/.test(kind)) [w, h] = detrResizeSize(im.width, im.height, edge, s.longest_edge, /Yolos/.test(kind));
      // ConvNeXt at 384 and above: straight to a square, no crop.
      else if (/ConvNext/.test(kind) && edge >= 384) [w, h] = [edge, edge];
      else [w, h] = shortestEdge(im.width, im.height, edge, s.longest_edge);
    } else if (/DPT/.test(kind)) {
      [w, h] = dptResizeSize(im.width, im.height, s.width ?? s.height!, s.height ?? s.width!, cfg.keep_aspect_ratio === true, cfg.ensure_multiple_of ?? 1);
    } else [w, h] = [s.width ?? s.height!, s.height ?? s.width!];
    im = resizeImage(im, w, h, resample);
    if (/ConvNext/.test(kind) && typeof s === "object" && s.shortest_edge !== undefined && s.shortest_edge < 384 && cfg.crop_pct !== undefined) {
      im = centerCrop(im, s.shortest_edge, s.shortest_edge);
    }
  }
  if (cfg.do_center_crop && cfg.crop_size !== undefined) {
    const c = cfg.crop_size;
    const [w, h] = typeof c === "number" ? [c, c] : [c.width ?? c.height!, c.height ?? c.width!];
    im = centerCrop(im, w, h);
  }
  const C = im.data.length / (im.width * im.height);
  const n = im.width * im.height;
  const scale = cfg.do_rescale === false ? 1 : (cfg.rescale_factor ?? 1 / 255);
  const norm = cfg.do_normalize !== false;
  const mean = triple(cfg.image_mean, [0.5, 0.5, 0.5]).map(Math.fround);
  const std = triple(cfg.image_std, [0.5, 0.5, 0.5]).map(Math.fround);
  const out = new Float32Array(3 * n);
  const fr = Math.fround;
  for (let c = 0; c < 3; c++)
    for (let i = 0; i < n; i++) {
      let v = fr(im.data[i * C + c]! * scale);
      if (norm) v = fr(fr(v - mean[c]!) / std[c]!);
      out[c * n + i] = v;
    }
  return { data: out, width: im.width, height: im.height };
}

/** Anything an image can come from: pixels, a URL, a Blob / File, encoded bytes, or a browser image. */
export type ImageInput = RgbImage | string | URL | Blob | ArrayBuffer | Uint8Array | ImageBitmap | ImageData;

/**
 * Decodes an image to RGBA bytes with the browser's decoders (createImageBitmap, no color space
 * conversion or premultiplied alpha), in a page or a worker. Already-decoded pixels pass through.
 */
export async function readImage(input: ImageInput): Promise<RgbImage> {
  if (typeof input === "object" && "data" in input && "width" in input && !(typeof ImageData !== "undefined" && input instanceof ImageData)) return input as RgbImage;
  if (typeof ImageData !== "undefined" && input instanceof ImageData) return { width: input.width, height: input.height, data: input.data };
  let source: ImageBitmap;
  if (typeof ImageBitmap !== "undefined" && input instanceof ImageBitmap) source = input;
  else {
    let blob: Blob;
    if (typeof input === "string" || input instanceof URL) {
      const r = await fetch(input);
      if (!r.ok) throw new Error(`image: HTTP ${r.status} for ${String(input)}`);
      blob = await r.blob();
    } else if (input instanceof Blob) blob = input;
    else blob = new Blob([input instanceof ArrayBuffer ? input : (input as Uint8Array)]);
    if (typeof createImageBitmap === "undefined") throw new Error("decoding images needs createImageBitmap (a browser); pass RGB(A) pixels instead");
    source = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  }
  const canvas = new OffscreenCanvas(source.width, source.height);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(source, 0, 0);
  const d = ctx.getImageData(0, 0, source.width, source.height);
  return { width: d.width, height: d.height, data: d.data };
}
