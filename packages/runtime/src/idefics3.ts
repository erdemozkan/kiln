import { resizeImage, toRgb, type RgbImage } from "@kiln/loader";
import type { Idefics3Processing } from "@kiln/models";

/** An image as Idefics3's processor cuts it: `rows` × `cols` tiles (none without splitting), then the global view. */
export interface Idefics3Tiles {
  rows: number;
  cols: number;
  /** RGB tiles of tileSize², row by row, then the global view. */
  tiles: RgbImage[];
}

/**
 * The size Idefics3's processor first resizes to: the longest edge set to `max` (up or down, the
 * other side by the aspect ratio, truncated and then made even), then scaled under 4096.
 */
export function idefics3ResizeSize(width: number, height: number, max: number): [number, number] {
  const aspect = width / height;
  let [w, h] = [width, height];
  if (w >= h) {
    w = max;
    h = Math.trunc(w / aspect);
    if (h % 2) h++;
  } else {
    h = max;
    w = Math.trunc(h * aspect);
    if (w % 2) w++;
  }
  [w, h] = [Math.max(w, 1), Math.max(h, 1)];
  const bound = 4096;
  const a2 = w / h;
  if (w >= h && w > bound) [w, h] = [bound, Math.trunc(bound / a2)];
  else if (h > w && h > bound) [w, h] = [Math.trunc(bound * a2), bound];
  return [Math.max(w, 1), Math.max(h, 1)];
}

/** The size Idefics3's `resize_for_vision_encoder` takes an image to: both sides multiples of the tile. */
export function idefics3TileGrid(width: number, height: number, tile: number): [number, number] {
  const aspect = width / height;
  let [w, h] = [width, height];
  if (w >= h) {
    w = Math.ceil(w / tile) * tile;
    h = Math.ceil(Math.trunc(w / aspect) / tile) * tile;
  } else {
    h = Math.ceil(h / tile) * tile;
    w = Math.ceil(Math.trunc(h * aspect) / tile) * tile;
  }
  return [w, h];
}

/**
 * Idefics3ImageProcessor on one image, with PIL's resampling: the longest edge to `longestEdge`; then,
 * with splitting, both sides up to multiples of the tile, cut into tiles, plus the whole image squashed
 * to one tile as the global view; without, the image squashed to one tile.
 */
export function idefics3Tiles(img: RgbImage, p: Idefics3Processing): Idefics3Tiles {
  const T = p.tileSize;
  const rgb = toRgb(img);
  const [w0, h0] = idefics3ResizeSize(rgb.width, rgb.height, p.longestEdge);
  let im = resizeImage(rgb, w0, h0, p.resample);
  if (!p.split) return { rows: 0, cols: 0, tiles: [resizeImage(im, T, T, p.resample)] };
  const [w, h] = idefics3TileGrid(im.width, im.height, T);
  im = resizeImage(im, w, h, p.resample);
  if (w <= T && h <= T) return { rows: 0, cols: 0, tiles: [im] };
  const rows = Math.ceil(h / T);
  const cols = Math.ceil(w / T);
  const tiles: RgbImage[] = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const data = new Uint8Array(T * T * 3);
      for (let y = 0; y < T; y++) data.set(im.data.subarray(((r * T + y) * w + c * T) * 3, ((r * T + y) * w + c * T + T) * 3), y * T * 3);
      tiles.push({ width: T, height: T, data });
    }
  tiles.push(resizeImage(im, T, T, p.resample));
  return { rows, cols, tiles };
}

/**
 * The text Idefics3's processor puts in place of one `<image>`: per tile `<fake_token_around_image>`,
 * `<row_r_col_c>` and the image tokens, a newline after each row, then the global view.
 */
export function idefics3PromptString(rows: number, cols: number, tokens: number): string {
  const fake = "<fake_token_around_image>";
  const image = "<image>".repeat(tokens);
  const global = `${fake}<global-img>${image}${fake}`;
  if (!rows && !cols) return global;
  let s = "";
  for (let r = 1; r <= rows; r++) {
    for (let c = 1; c <= cols; c++) s += `${fake}<row_${r}_col_${c}>${image}`;
    s += "\n";
  }
  return `${s}\n${global}`;
}
