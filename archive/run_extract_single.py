import os, json
from pathlib import Path
import numpy as np

proj = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(proj))

from src.data_processing.visual_extractor import extract_grid, write_vis_overlay

DATA_ROOT = proj / 'Data' / 'The Legend of Zelda'
OUT = proj / 'artifacts' / 'visual_extracts_demo'
OUT.mkdir(parents=True, exist_ok=True)

img = DATA_ROOT / 'Original' / 'tloz1_1.png'
tileset = proj / 'assets' / 'Dungeon Tileset.png'
if not tileset.exists():
    # fallback to demo tileset created by demo
    tileset = proj / 'artifacts' / 'demo' / 'demo_tileset.png'

print('Using tileset:', tileset)
print('Input image:', img)

arr = extract_grid(str(img), str(tileset), tile_px=16)
name = img.stem
np.save(OUT / f"{name}.npy", arr)

# write txt
ids = arr[:, :, 0].astype(int)
with open(OUT / f"{name}.txt", 'w', encoding='utf8') as f:
    for r in range(ids.shape[0]):
        f.write(''.join((str(x) if x>=0 and x<10 else chr(ord('A')+(x-10)%26)) for x in ids[r]))
        f.write('\n')

# overlay
write_vis_overlay(arr, str(tileset), str(OUT / f"{name}_overlay.png"))

# npz metadata
import hashlib
template_hash = hashlib.md5(tileset.read_bytes()).hexdigest()[:8]
meta = {
    'template_hash': template_hash,
    'tile_size': 16,
    'source_image': str(img)
}
np.savez_compressed(OUT / f"{name}.npz", grid=arr, metadata=json.dumps(meta))

print('Wrote:', list(OUT.glob(f"{name}.*")))
print('Done')
