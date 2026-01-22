"""Curate a tileset spritesheet into unique tiles, clusters, and atlas plan.

Produces JSON outputs and preview images under an output directory.

Heuristics (brief):
- blank/empty: very low pixel variance or mostly transparent.
- floor: low edge strength and relatively uniform color.
- wall: strong vertical or horizontal edges.
- corner: strong edges on two adjacent sides.
- door: a prominent vertical strip with contrast against surrounding pixels.
- stairs: repeated step-like diagonal pattern (detected via angled edge responses).
- deco: higher detail/entropy and multiple colors present.
- item/sprite: small high-contrast blob(s) on otherwise uniform background.
- water: blue-dominant tiles.
- rope: thin vertical/braided-like thin dark lines on light BG.
- edge: border-like pixels on one side.

This script is intentionally conservative: if unsure, it tags with multiple possible labels and lower confidence.

Usage: python scripts/curate_tileset.py --source "Data/Assets/NES - The Legend of Zelda - Tilesets - Dungeon Tileset.png"

"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageDraw

# --- Utilities ---

def sha1_of_array(arr: np.ndarray) -> str:
    m = hashlib.sha1()
    m.update(arr.tobytes())
    return m.hexdigest()


def phash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    """Compute a simple perceptual hash (DCT-based) returning a flattened normalized vector."""
    # Convert to grayscale and resize to (hash_size*4, hash_size*4) for DCT
    import numpy as _np
    im = img.convert("L").resize((hash_size * 4, hash_size * 4), Image.BICUBIC)
    arr = _np.array(im, dtype=float)
    # dct
    from scipy.fftpack import dct

    d = dct(dct(arr.T, norm="ortho").T, norm="ortho")
    dlow = d[:hash_size, :hash_size]
    med = _np.median(dlow)
    return (dlow > med).astype(int).ravel()


# Fallback phash if scipy missing (use average hash)
def ahash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    im = img.convert("L").resize((hash_size, hash_size), Image.BICUBIC)
    arr = np.array(im, dtype=float)
    med = np.mean(arr)
    return (arr > med).astype(int).ravel()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def compute_mean_color(arr: np.ndarray) -> Tuple[int, int, int]:
    # arr is HxWxC
    m = arr.reshape(-1, arr.shape[2]).mean(axis=0)
    return tuple(int(x) for x in m[:3])


# --- Tag heuristics ---

def tag_tile(arr: np.ndarray) -> List[Tuple[str, float]]:
    """Return list of (tag, confidence).

    Heuristics (implemented):
    - blank/empty: std dev of pixels small or alpha mostly 0.
    - water: blue channel dominating and some wave-like variance.
    - floor: low edge magnitude (sobel) and relatively uniform.
    - wall: prominent vertical or horizontal edges.
    - corner: two strong edge directions on adjacent sides.
    - door: vertical high-contrast strip near center.
    - stairs: detect diagonal/step patterns via convolution.
    - deco: high color variance and many colors.
    - item/sprite: small connected components of non-background color.
    - edge: border-like pixels on one side.
    - rope: thin vertical dark lines and brownish color.
    """
    tags: List[Tuple[str, float]] = []
    h, w, c = arr.shape
    rgb = arr[:, :, :3].astype(float)
    # alpha handling
    alpha = arr[:, :, 3] if c == 4 else None
    flat = rgb.reshape(-1, 3)

    std = flat.std()
    mean = flat.mean(axis=0)

    # blank/transparent
    if alpha is not None:
        alpha_mean = alpha.mean()
        if alpha_mean < 10:
            return [("blank", 1.0)]
    if std < 8 and np.max(flat) - np.min(flat) < 16:
        tags.append(("blank", 0.9))

    # water (blue-dominant)
    blue_dom = (mean[2] > mean[0] + 12) and (mean[2] > mean[1] + 8)
    if blue_dom:
        tags.append(("water", 0.8))

    # edges via sobel
    gray = np.dot(rgb, [0.2989, 0.5870, 0.1140]).astype(float)
    # Try to use scipy.ndimage for convolution; fall back to numpy gradients if not available
    try:
        from scipy import ndimage

        sx = ndimage.convolve(gray, [[1, 0, -1], [2, 0, -2], [1, 0, -1]], mode="reflect")
        sy = ndimage.convolve(gray, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], mode="reflect")
    except Exception:
        # simple sobel-like via numpy gradient
        gy, gx = np.gradient(gray)
        sx = gx
        sy = gy
    mag = np.hypot(sx, sy)
    mag_mean = mag.mean()
    mag_max = mag.max()

    strong = []
    if mag_mean > 10 or mag_max > 50:
        # possibly wall or deco
        # directional strength
        vert = np.abs(sx).mean()
        horiz = np.abs(sy).mean()
        if vert > horiz * 1.2:
            tags.append(("wall", 0.7))
        if horiz > vert * 1.2:
            tags.append(("wall", 0.7))
        # corner: edges on two adjacent sides -> check sums on edges
        side_thresh = 10
        left = mag[:, 0:2].mean()
        right = mag[:, -2:].mean()
        top = mag[0:2, :].mean()
        bottom = mag[-2:, :].mean()
        sides = [("left", left), ("top", top), ("right", right), ("bottom", bottom)]
        strong = [s for s, v in sides if v > side_thresh]
        if len(strong) >= 2:
            tags.append(("corner", 0.75))

    # door: vertical strip in center with contrast
    center_strip = rgb[:, w // 2 - 1 : w // 2 + 1, :]
    cs_std = center_strip.reshape(-1, 3).std()
    if cs_std > 20 and mag_mean > 8:
        tags.append(("door", 0.6))

    # stairs heuristic: repeated horizontal edges at offsets
    horiz_profile = mag.mean(axis=1)
    peaks = (horiz_profile > horiz_profile.mean() + horiz_profile.std()).sum()
    if peaks >= 2 and peaks <= 5:
        tags.append(("stairs", 0.6))

    # deco vs item/sprite
    unique_colors = len({tuple(c) for c in flat.astype(int)})
    if unique_colors > (h * w) * 0.15:
        tags.append(("deco", 0.7))
    elif unique_colors < 8 and std > 6:
        tags.append(("item", 0.75))

    # rope: thin verticals and brownish mean
    brownish = (mean[0] > mean[2]) and (mean[0] > mean[1]) and mean[0] > 80
    thin_vert = (np.logical_and(mag > 40, sx != 0)).sum() > (h * 2)
    if brownish and thin_vert:
        tags.append(("rope", 0.5))

    # edge: if one side strong more than others
    if len(strong) == 1:
        tags.append(("edge", 0.6))

    # fallback floor if few tags
    if not tags:
        tags.append(("floor", 0.6))

    # consolidate and normalize confidences
    total = sum([t[1] for t in tags]) or 1.0
    tags_norm = [(t, min(1.0, round(c / total, 3))) for t, c in tags]
    return tags_norm


# --- Core processing ---

def slice_tiles(img: Image.Image, tile_size: int = 8) -> List[Tuple[int, int, Image.Image]]:
    w, h = img.size
    tiles = []
    rows = h // tile_size
    cols = w // tile_size
    for r in range(rows):
        for c in range(cols):
            box = (c * tile_size, r * tile_size, (c + 1) * tile_size, (r + 1) * tile_size)
            tile = img.crop(box)
            tiles.append((r, c, tile))
    return tiles


@dataclass
class TileRecord:
    uid: str
    hash: str
    freq: int
    sample_rc: Tuple[int, int]
    group: int
    tags: List[Tuple[str, float]]
    bbox: Tuple[int, int, int, int]
    mean_color: Tuple[int, int, int]


def curate(
    source_file: str,
    out_dir: str,
    tile_size: int = 8,
    top_n: int | None = None,
    min_freq: int = 1,
    k: int = 16,
):
    ensure_dir(out_dir)
    img = Image.open(source_file).convert("RGBA")

    # collect tiles
    tiles = slice_tiles(img, tile_size)
    tile_hashes = []
    tile_map = []  # list of (r,c,image)
    for r, c, t in tiles:
        arr = np.array(t)
        hsh = sha1_of_array(arr)
        tile_hashes.append(hsh)
        tile_map.append(((r, c), t, arr))

    freq = Counter(tile_hashes)

    # unique tiles
    unique_hashes = list(freq.keys())

    # map hash -> representative tile arr and sample rc
    rep = {}
    for (r, c), t, arr in tile_map:
        h = sha1_of_array(arr)
        if h not in rep:
            rep[h] = {
                "arr": arr,
                "sample_rc": (r, c),
                "tile": t,
            }

    # Build feature vectors (phash fallback to ahash)
    feats = []
    keys = []
    use_scipy = True
    try:
        import scipy  # type: ignore
        from scipy import fftpack  # ensure available
    except Exception:
        use_scipy = False

    for h in unique_hashes:
        timg = Image.fromarray(rep[h]["arr"].astype(np.uint8))
        if use_scipy:
            vec = phash(timg, hash_size=8)
        else:
            vec = ahash(timg, hash_size=8)
        feats.append(vec.astype(float))
        keys.append(h)
    feats = np.vstack(feats)

    # k-means clustering (simple numpy implementation)
    def kmeans(X: np.ndarray, k: int, iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        # initialize with kmeans++
        n = X.shape[0]
        rng = np.random.default_rng(12345)
        centers = np.empty((k, X.shape[1]), dtype=float)
        centers[0] = X[rng.integers(0, n)]
        distances = np.full(n, np.inf)
        for i in range(1, k):
            d = np.linalg.norm(X - centers[i - 1], axis=1) ** 2
            distances = np.minimum(distances, d)
            probs = distances / distances.sum()
            idx = rng.choice(n, p=probs)
            centers[i] = X[idx]
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            # assign
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                if (labels == j).any():
                    centers[j] = X[labels == j].mean(axis=0)
        return labels, centers

    k = max(1, min(k, feats.shape[0]))
    labels, centers = kmeans(feats, k)

    # Build groupings
    groups: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"uids": [], "count": 0})
    tile_records: Dict[str, TileRecord] = {}
    for uid, label in zip(keys, labels):
        GROUP = int(label)
        freq_count = int(freq[uid])
        sample = rep[uid]
        arr = sample["arr"]
        mean_color = compute_mean_color(arr)
        tags = tag_tile(arr)
        tr = TileRecord(
            uid=uid,
            hash=uid,
            freq=freq_count,
            sample_rc=sample["sample_rc"],
            group=GROUP,
            tags=tags,
            bbox=(0, 0, tile_size, tile_size),
            mean_color=mean_color,
        )
        tile_records[uid] = tr
        groups[GROUP]["uids"].append(uid)
        groups[GROUP]["count"] += freq_count

    # sort unique tiles by frequency
    sorted_uids = sorted(list(keys), key=lambda u: tile_records[u].freq, reverse=True)

    # apply min_freq and top_n
    selected = [u for u in sorted_uids if tile_records[u].freq >= min_freq]
    if top_n is not None:
        selected = selected[:top_n]

    # compute atlas plan: try to compute atlas dims power-of-two optimizing packing as simple grid
    n_tiles = len(selected)
    # choose tile_size same as base (can recommend meta-tile later)
    atlas_tile_size = tile_size
    widths = [2 ** i for i in range(0, 12)]
    best = None
    for w in widths:
        h_needed = math.ceil(n_tiles / w)
        if h_needed <= w:  # square or taller
            best = (w, max(1, h_needed))
            break
    if best is None:
        # fallback to wide
        w = int(2 ** math.ceil(math.log2(int(math.ceil(math.sqrt(n_tiles))))))
        h_needed = math.ceil(n_tiles / w)
        best = (w, h_needed)

    atlas_dims = (best[0] * atlas_tile_size, best[1] * atlas_tile_size)

    # assemble and write outputs
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "source_file": source_file,
        "generated_by": "scripts/curate_tileset.py",
        "generated_at": now,
        "credit": "Sprites ripped by Mister Mike",
    }

    # tiles_curation.json
    tiles_out = {
        "_meta": metadata,
        "tiles": {},
    }
    for uid, tr in tile_records.items():
        tiles_out["tiles"][uid] = {
            "hash": tr.hash,
            "freq": tr.freq,
            "sample_rc": tr.sample_rc,
            "group": tr.group,
            "tags": tr.tags,
            "bbox": tr.bbox,
            "mean_color": tr.mean_color,
        }

    with open(os.path.join(out_dir, "tiles_curation.json"), "w", encoding="utf-8") as f:
        json.dump(tiles_out, f, indent=2)

    # groups.json
    groups_out = {
        "_meta": metadata,
        "groups": {},
    }
    for gid, info in groups.items():
        groups_out["groups"][str(gid)] = {"uids": info["uids"], "count": info["count"]}
    with open(os.path.join(out_dir, "groups.json"), "w", encoding="utf-8") as f:
        json.dump(groups_out, f, indent=2)

    # atlas_plan.json
    atlas_plan = {
        "_meta": metadata,
        "atlas_tile_size": atlas_tile_size,
        "atlas_pixels": atlas_dims,
        "tile_count": n_tiles,
        "ordered_uids": selected,
    }
    with open(os.path.join(out_dir, "atlas_plan.json"), "w", encoding="utf-8") as f:
        json.dump(atlas_plan, f, indent=2)

    # images: unique_tiles.png
    # build grid based on sqrt
    samples = [rep[uid]["tile"] for uid in sorted_uids]
    # grid dims
    cols = int(math.ceil(math.sqrt(len(samples))))
    rows = int(math.ceil(len(samples) / cols))
    uimg = Image.new("RGBA", (cols * tile_size, rows * tile_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(uimg)
    for i, t in enumerate(samples):
        r = i // cols
        c = i % cols
        uimg.paste(t, (c * tile_size, r * tile_size))
    uimg.save(os.path.join(out_dir, "unique_tiles.png"))

    # groups_preview.png: one row per group with up to 8 representatives
    max_per_row = 8
    grows = len(groups)
    gimg = Image.new(
        "RGBA", (max_per_row * tile_size, grows * tile_size), (0, 0, 0, 0)
    )
    gj = 0
    for gid, info in sorted(groups.items()):
        uids = info["uids"][:max_per_row]
        for i, uid in enumerate(uids):
            t = Image.fromarray(rep[uid]["arr"].astype(np.uint8))
            gimg.paste(t, (i * tile_size, gj * tile_size))
        gj += 1
    gimg.save(os.path.join(out_dir, "groups_preview.png"))

    # recommended_atlas_mockup.png
    at_w, at_h = atlas_dims
    atlas_img = Image.new("RGBA", (at_w, at_h), (0, 0, 0, 0))
    for i, uid in enumerate(selected):
        tr = tile_records[uid]
        t = Image.fromarray(rep[uid]["arr"].astype(np.uint8))
        col = i % (best[0])
        row = i // (best[0])
        atlas_img.paste(t, (col * atlas_tile_size, row * atlas_tile_size))
    atlas_img.save(os.path.join(out_dir, "recommended_atlas_mockup.png"))

    print(f"Wrote outputs to {out_dir}")
    return {
        "metadata": metadata,
        "n_base_tiles": len(tiles),
        "n_unique": len(unique_hashes),
        "n_selected": n_tiles,
        "atlas_dims": atlas_dims,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="Data/Assets/tileset_curation")
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args(argv)
    ensure_dir(args.out)
    res = curate(
        args.source, args.out, tile_size=args.tile_size, top_n=args.top_n, min_freq=args.min_freq, k=args.k
    )
    print("Summary:")
    print(res)


if __name__ == "__main__":
    main()
