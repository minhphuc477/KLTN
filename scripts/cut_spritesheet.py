#!/usr/bin/env python3
"""Cut a spritesheet into tiles and optionally build a compact tileset atlas.

Usage:
  python scripts/cut_spritesheet.py --input "Data/Assets/NES - The Legend of Zelda - Tilesets - Dungeon Tileset.png" \
        --tile-width 16 --tile-height 16 --out-dir Data/Assets/tiles --make-atlas Data/Assets/tileset_atlas.png

Default tile size is 16x16 (NES standard). The script writes tiles as PNGs named tile_r_c.png and, if requested, creates a compact atlas packing tiles in rows.
"""
import argparse
from pathlib import Path

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Pillow is required: pip install pillow")


def slice_spritesheet(path: Path, tile_w: int, tile_h: int, out_dir: Path, margin: int = 0, spacing: int = 0):
    img = Image.open(path).convert('RGBA')
    w, h = img.size
    cols = (w - 2 * margin + spacing) // (tile_w + spacing)
    rows = (h - 2 * margin + spacing) // (tile_h + spacing)

    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x = margin + c * (tile_w + spacing)
            y = margin + r * (tile_h + spacing)
            if x + tile_w <= w and y + tile_h <= h:
                tile = img.crop((x, y, x + tile_w, y + tile_h))
                name = f"tile_{r}_{c}.png"
                tile.save(out_dir / name)
                tiles.append((r, c, out_dir / name))
    return tiles


def make_atlas(tiles, tile_w: int, tile_h: int, out_path: Path, cols: int = 16, padding: int = 0):
    # tiles: list of (r,c,path)
    import math
    n = len(tiles)
    if n == 0:
        raise ValueError('No tiles provided')
    cols = cols or min(16, n)
    rows = math.ceil(n / cols)
    atlas_w = cols * (tile_w + padding) - padding
    atlas_h = rows * (tile_h + padding) - padding
    atlas = Image.new('RGBA', (atlas_w, atlas_h), (0, 0, 0, 0))
    for i, (_, _, p) in enumerate(tiles):
        tile = Image.open(p).convert('RGBA')
        row = i // cols
        col = i % cols
        x = col * (tile_w + padding)
        y = row * (tile_h + padding)
        atlas.paste(tile, (x, y), tile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(out_path)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input spritesheet image path')
    parser.add_argument('--tile-width', type=int, default=16)
    parser.add_argument('--tile-height', type=int, default=16)
    parser.add_argument('--margin', type=int, default=0)
    parser.add_argument('--spacing', type=int, default=0)
    parser.add_argument('--out-dir', '-o', default='Data/Assets/tiles')
    parser.add_argument('--make-atlas', default='', help='Optional atlas output path')
    parser.add_argument('--atlas-cols', type=int, default=16)
    parser.add_argument('--atlas-padding', type=int, default=0)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not in_path.exists():
        raise SystemExit(f'Input file not found: {in_path}')

    tiles = slice_spritesheet(in_path, args.tile_width, args.tile_height, out_dir, margin=args.margin, spacing=args.spacing)
    print(f'Wrote {len(tiles)} tiles to {out_dir}')

    if args.make_atlas:
        atlas_path = Path(args.make_atlas)
        make_atlas(tiles, args.tile_width, args.tile_height, atlas_path, cols=args.atlas_cols, padding=args.atlas_padding)
        print(f'Atlas created: {atlas_path}')
