import { describe, expect, it } from "vitest";
import { idefics3PromptString, idefics3ResizeSize, idefics3TileGrid, idefics3Tiles } from "../src/idefics3";

const P = { longestEdge: 2048, tileSize: 512, split: true, resample: 1, imageTokenId: 49190 };

describe("Idefics3 image processing", () => {
  it("sizes images as transformers' processor does", () => {
    expect(idefics3ResizeSize(640, 480, 2048)).toEqual([2048, 1536]);
    expect(idefics3ResizeSize(480, 640, 2048)).toEqual([1536, 2048]);
    // Upscaled too; an odd short side is made even.
    expect(idefics3ResizeSize(300, 200, 2048)).toEqual([2048, 1366]);
    expect(idefics3ResizeSize(1000, 333, 2048)).toEqual([2048, 682]);
    expect(idefics3TileGrid(2048, 1366, 512)).toEqual([2048, 1536]);
    expect(idefics3TileGrid(2048, 682, 512)).toEqual([2048, 1024]);
    expect(idefics3TileGrid(682, 2048, 512)).toEqual([1024, 2048]);
  });

  it("cuts tiles row by row, then the global view", () => {
    const img = { width: 8, height: 4, data: new Uint8Array(8 * 4 * 4).fill(200) };
    const t = idefics3Tiles(img, { ...P, longestEdge: 16, tileSize: 4 });
    expect([t.rows, t.cols, t.tiles.length]).toEqual([2, 4, 9]);
    expect(t.tiles.every((x) => x.width === 4 && x.height === 4 && x.data.length === 48)).toBe(true);
    const one = idefics3Tiles(img, { ...P, longestEdge: 16, tileSize: 4, split: false });
    expect([one.rows, one.cols, one.tiles.length]).toEqual([0, 0, 1]);
  });

  it("puts the tiles' tokens in the prompt", () => {
    expect(idefics3PromptString(0, 0, 2)).toBe("<fake_token_around_image><global-img><image><image><fake_token_around_image>");
    expect(idefics3PromptString(1, 2, 1)).toBe(
      "<fake_token_around_image><row_1_col_1><image><fake_token_around_image><row_1_col_2><image>\n\n<fake_token_around_image><global-img><image><fake_token_around_image>",
    );
  });
});
