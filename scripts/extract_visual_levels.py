"""CLI: extract visual levels (tiles -> per-room grids + overlays)

Example:
  python scripts/extract_visual_levels.py \
    --templates Data/tileset/Dungeon\ Tileset.png \
    --input Data/The\ Legend\ of\ Zelda/Original/tloz1_1.png \
    --out-dir artifacts/visual_extracts --visualize

Outputs (per input image):
  - <name>.npy       : (H,W,2) array, [:,:,0]=template-index, [:,:,1]=confidence
  - <name>.txt       : VGLC-style character grid (best-effort)
  - <name>_overlay.png : colored overlay for quick inspection
"""
from __future__ import annotations
import argparse
import os
import numpy as np
from pathlib import Path

from src.data_processing.visual_extractor import extract_grid, write_vis_overlay


def _to_vglc_char(idx: int) -> str:
    # best-effort mapping: unknown -> '-', template indices -> letters/digits
    if idx < 0:
        return '-'
    if idx < 10:
        return str(idx)
    return chr(ord('A') + (idx - 10) % 26)


def _write_txt(grid_arr: np.ndarray, out_txt: str):
    ids = grid_arr[:, :, 0].astype(int)
    lines = []
    for r in range(ids.shape[0]):
        lines.append(''.join(_to_vglc_char(int(x)) for x in ids[r]))
    with open(out_txt, 'w', encoding='utf8') as f:
        f.write('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--templates', '-t', required=True, help='tileset image or templates folder')
    p.add_argument('--input', '-i', required=True, help='input image (PNG) or folder')
    p.add_argument('--out-dir', '-o', required=True)
    p.add_argument('--tile-size', type=int, default=16)
    p.add_argument('--visualize', action='store_true')
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--npz', action='store_true', help='Save compressed .npz with metadata')
    args = p.parse_args()

    inp = Path(args.input)
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    inputs = [inp] if inp.is_file() else sorted([p for p in inp.iterdir() if p.suffix.lower() in ('.png', '.jpg')])
    for f in inputs:
        name = f.stem
        print(f"Processing {f} -> {outd/name}")
        thresholds = {'match': args.threshold} if args.threshold else None
        arr = extract_grid(str(f), args.templates, tile_px=args.tile_size, thresholds=thresholds)
        np.save(outd / f"{name}.npy", arr)
        _write_txt(arr, str(outd / f"{name}.txt"))
        if args.visualize:
            write_vis_overlay(arr, args.templates, str(outd / f"{name}_overlay.png"))
        
        # Write NPZ with metadata (template hash + extraction config)
        if args.npz:
            import hashlib
            import json
            template_path = Path(args.templates)
            if template_path.is_file():
                template_hash = hashlib.md5(template_path.read_bytes()).hexdigest()[:8]
            else:
                template_hash = "folder"
            metadata = {
                'template_hash': template_hash,
                'tile_size': args.tile_size,
                'threshold': args.threshold,
                'source_image': str(f),
                'extraction_date': str(np.datetime64('now'))
            }
            np.savez_compressed(
                outd / f"{name}.npz",
                grid=arr,
                metadata=json.dumps(metadata)
            )
            print(f"  saved: {name}.npz with metadata (hash={template_hash})")
        
        print(f"  saved: {name}.npy  {name}.txt {name}_overlay.png (if --visualize)")


if __name__ == '__main__':
    main()
