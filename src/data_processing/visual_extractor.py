"""Visual extractor — tile/template matcher for Zelda screenshots.

Provides:
- extract_grid(image_path, templates_dir) -> np.ndarray (semantic-like numeric grid)
- helper functions to auto-slice a tileset, run robust template-matching with
  normalized cross-correlation and an MSE fingerprint fallback, and produce
  a confidence mask.

Design goals: deterministic CV (no ML), robust to small offsets/HUD, marks
uncertain tiles and falls back to nearest-neighbour fingerprinting.

Minimal deps: numpy, opencv-python, Pillow
"""
from __future__ import annotations
import os
from typing import Tuple, Dict, Optional
import numpy as np
import cv2
from PIL import Image

# Defaults (matches VGLC Zelda): room is 16×11 tiles, tile size 16px in source
ROOM_H, ROOM_W = 16, 11
DEFAULT_TILE_PX = 16


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGBA')
    return np.asarray(img)


def _auto_slice_tileset(tileset_path: str, tile_px: int = DEFAULT_TILE_PX) -> Dict[str, np.ndarray]:
    """Slice a regular tileset into tile_px×tile_px templates.

    Returns dict: name -> RGB template (H,W,4)
    """
    img = _load_image(tileset_path)
    h, w = img.shape[:2]
    cols = w // tile_px
    rows = h // tile_px
    templates = {}
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * tile_px, c * tile_px
            patch = img[y0:y0 + tile_px, x0:x0 + tile_px].copy()
            # Skip fully-transparent tiles
            if patch[..., 3].mean() < 4:
                continue
            name = f"t_{r}_{c}"
            templates[name] = patch
    return templates


def _estimate_hud_crop(img: np.ndarray, tile_px: int = DEFAULT_TILE_PX) -> int:
    """Estimate number of pixels to crop from the top (HUD). Returns crop_y.

    Heuristic: scan in tile-sized steps from top and find first horizontal band
    whose variance (in grayscale) increases significantly compared to above.
    """
    gray = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    h = gray.shape[0]
    max_scan = min(h // 3, tile_px * 6)
    scores = []
    for y in range(0, max_scan, max(1, tile_px // 2)):
        band = gray[y:y + tile_px]
        scores.append(band.var())
    if len(scores) < 2:
        return 0
    scores = np.array(scores)
    # find first index where variance rises > median + 0.8*std
    thresh = np.median(scores) + 0.8 * scores.std()
    idx = np.argmax(scores > thresh)
    if scores[idx] <= thresh:
        return 0
    crop_y = max(0, idx * max(1, tile_px // 2) - tile_px // 2)
    return int(crop_y)


def _match_templates(img: np.ndarray, templates: Dict[str, np.ndarray],
                     tile_px: int = DEFAULT_TILE_PX,
                     match_threshold: float = 0.65) -> Tuple[np.ndarray, np.ndarray]:
    """Match templates on image and return (grid_ids, confidence_mask).

    grid_ids: (ROOM_H, ROOM_W) numeric indices into template list (or -1)
    confidence_mask: same shape with float in [0,1]
    """
    h, w = img.shape[:2]
    # crop to nearest room multiple
    h_crop = (h // tile_px) * tile_px
    w_crop = (w // tile_px) * tile_px
    imgc = img[:h_crop, :w_crop, :3]
    gray = cv2.cvtColor(imgc, cv2.COLOR_RGB2GRAY)

    tpl_names = list(templates.keys())
    tpl_arrs = [cv2.cvtColor(templates[n][..., :3], cv2.COLOR_RGBA2RGB) if templates[n].shape[2] == 4 else templates[n][..., :3]
                for n in tpl_names]
    tpl_gray = [cv2.cvtColor(t, cv2.COLOR_RGB2GRAY) for t in tpl_arrs]

    rows = h_crop // tile_px
    cols = w_crop // tile_px
    grid_ids = -1 * np.ones((rows, cols), dtype=int)
    conf = np.zeros((rows, cols), dtype=float)

    # Precompute fingerprints for NN fallback (downscale + mean color)
    def fingerprint(a):
        small = cv2.resize(a, (4, 4), interpolation=cv2.INTER_AREA)
        return small.mean(axis=(0, 1)).ravel()

    tpl_fp = np.vstack([fingerprint(t) for t in tpl_arrs]) if tpl_arrs else np.zeros((0, 3))

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * tile_px, c * tile_px
            patch = imgc[y0:y0 + tile_px, x0:x0 + tile_px]
            pg = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

            best_score = -1.0
            best_idx = -1

            # Template matching (normalized cross-corr)
            for i, tg in enumerate(tpl_gray):
                # allow small offsets by matching patch vs tpl with matchTemplate
                try:
                    res = cv2.matchTemplate(pg, tg, cv2.TM_CCOEFF_NORMED)
                except Exception:
                    continue
                # take max over small neighborhood to tolerate shifts
                score = float(res.max()) if res.size else -1.0
                if score > best_score:
                    best_score = score
                    best_idx = i

            # Auto thresholding: accept if score above match_threshold
            if best_score >= match_threshold:
                grid_ids[r, c] = best_idx
                conf[r, c] = float(np.clip((best_score - match_threshold) / (1.0 - match_threshold), 0.0, 1.0))
                continue

            # Fallback: fingerprint NN (grayscale + color)
            if tpl_fp.shape[0] > 0:
                fp = fingerprint(patch)
                dists = np.linalg.norm(tpl_fp - fp[None, :], axis=1)
                nn = int(dists.argmin())
                # compute a pseudo-confidence from distance (heuristic)
                pseudo_conf = float(np.clip(1.0 - (dists[nn] / 80.0), 0.0, 0.9))
                grid_ids[r, c] = nn
                conf[r, c] = pseudo_conf * 0.5  # mark as uncertain
            else:
                grid_ids[r, c] = -1
                conf[r, c] = 0.0

    return grid_ids, conf


def extract_grid(image_path: str, templates_dir: str,
                 tile_px: int = DEFAULT_TILE_PX,
                 thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Public API: returns a (ROOM_H, ROOM_W) semantic-like numeric grid.

    - If the input is a full screenshot, function will auto-detect and crop
      to the top-left room grid based on tile_px multiples and HUD estimation.
    - templates_dir can be either a tileset image or a folder with template PNGs.
    """
    img = _load_image(image_path)
    # load templates
    if os.path.isdir(templates_dir):
        templates = {}
        for fn in sorted(os.listdir(templates_dir)):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                templates[fn] = np.asarray(Image.open(os.path.join(templates_dir, fn)).convert('RGBA'))
    else:
        templates = _auto_slice_tileset(templates_dir, tile_px)

    crop_y = _estimate_hud_crop(img, tile_px)
    if crop_y > 0:
        img = img[crop_y:]

    # If image larger than one room, take the top-left room-sized crop
    h, w = img.shape[:2]
    need_h = ROOM_H * tile_px
    need_w = ROOM_W * tile_px
    if h < need_h or w < need_w:
        raise ValueError(f"Image too small for one room ({h}x{w} < {need_h}x{need_w})")

    room_img = img[:need_h, :need_w, :3]

    # Calibrate threshold from template / tile variance if not provided
    match_threshold = 0.65
    if thresholds and 'match' in thresholds:
        match_threshold = float(thresholds['match'])
    else:
        # estimate from per-tile variance
        tile_vars = []
        for r in range(ROOM_H):
            for c in range(ROOM_W):
                y0, x0 = r * tile_px, c * tile_px
                tile = cv2.cvtColor(room_img[y0:y0 + tile_px, x0:x0 + tile_px], cv2.COLOR_RGB2GRAY)
                tile_vars.append(tile.var())
        mv = float(np.median(tile_vars))
        sv = float(np.std(tile_vars))
        match_threshold = float(np.clip(0.45 + (mv / (mv + sv + 1e-6)) * 0.25, 0.45, 0.82))

    grid_ids, conf = _match_templates(room_img, templates, tile_px=tile_px, match_threshold=match_threshold)

    # Convert template indices -> semantic IDs (best-effort): map unique templates to semantic symbols
    # Simple mapping: most frequent template -> FLOOR (1), templates with large border->WALL (2), etc.
    # For now, return template-index grid and an accompanying confidence mask via structured ndarray
    semantic = np.int16(grid_ids)
    # mark unknown as VOID (-1 -> 0)
    semantic[semantic < 0] = -1

    # Attach confidence as float32 separately (caller can use it)
    # For backward compatibility return a view: stacked array (H, W, 2) where [:,:,0]=ids, [:,:,1]=conf
    out = np.zeros((ROOM_H, ROOM_W, 2), dtype=np.float32)
    out[:, :, 0] = semantic
    out[:, :, 1] = conf
    return out


def write_vis_overlay(out_arr: np.ndarray, templates_dir: str, out_path: str):
    """Create an RGB overlay visualization and save it to out_path.

    out_arr is the stacked array from extract_grid.
    """
    ids = out_arr[:, :, 0].astype(int)
    conf = out_arr[:, :, 1]
    h, w = ids.shape
    tile_px = DEFAULT_TILE_PX
    canvas = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8) + 30

    # color map for visualization (stable hash by id)
    def color_for(i):
        if i < 0:
            return (20, 20, 30)
        r = (97 * (i + 3)) % 255
        g = (61 * (i + 7)) % 255
        b = (151 * (i + 11)) % 255
        return (r, g, b)

    for r in range(h):
        for c in range(w):
            i = ids[r, c]
            col = color_for(i)
            y0, x0 = r * tile_px, c * tile_px
            canvas[y0:y0 + tile_px, x0:x0 + tile_px] = col
            if conf[r, c] < 0.5:
                # mark uncertain with a red tint
                canvas[y0:y0 + tile_px, x0:x0 + tile_px] = (np.array(col) * np.array([1.0, 0.4, 0.4])).astype(np.uint8)

    Image.fromarray(canvas).save(out_path)
