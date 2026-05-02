"""
Generate a fixed-topology visual audit for the diffusion / fast-sampler branch.

This script keeps Block I constant by generating one mission graph up front and
then reusing it across multiple runtime variants:

- full diffusion with the generation defaults
- full diffusion in the teacher CFG regime
- fast sampler with the generation defaults
- fast sampler in the teacher CFG regime

The resulting artifacts are written side-by-side so visual regressions and
runtime guidance mismatches are easy to inspect.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
import yaml
from PIL import Image, ImageDraw
from networkx.readwrite import json_graph

try:
    import torch
except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.definitions import semantic_to_vglc_char
from src.pipeline.dungeon_pipeline import (
    NeuralSymbolicDungeonPipeline,
    pipeline_kwargs_from_resolved_config,
)
from src.pipeline.room_stitching import (
    StitchedRoomLayout,
    build_stitched_room_layout,
    compute_layout_quality_metrics,
)
from src.evaluation.end_to_end_level_metrics import (
    DEFAULT_REFERENCE_ROOM_LIMIT,
    compute_end_to_end_structural_metrics,
    load_reference_room_texts,
)
from src.simulation.search_factory import (
    VALIDATION_EXCLUDED_ALGORITHMS,
    iter_game_state_algorithm_specs,
)


VALIDATION_SEARCH_SUITE_VERSION = "2026-04-15.validation_search_suite_v2"


def _json_sanitize(value: Any) -> Any:
    """Recursively replace non-finite numerics with JSON-safe nulls."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    return value


def _load_resolved_config(run_dir: Path) -> Dict[str, Any]:
    json_path = run_dir / "resolved_config.json"
    if json_path.exists():
        raw_json = json_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw_json)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    yaml_path = run_dir / "resolved_config.yaml"
    if yaml_path.exists():
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload

    raise FileNotFoundError(
        f"Could not load a valid resolved config from {json_path} or {yaml_path}."
    )


def _resolve_dataset_data_root(run_dir: Path) -> Path:
    resolved = _load_resolved_config(run_dir)
    dataset_cfg = dict(resolved.get("dataset", {}))
    data_root = Path(str(dataset_cfg.get("data_dir", "Data/The Legend of Zelda")))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    return data_root


def _resolve_export_device(resolved: Mapping[str, Any]) -> str:
    override = str(os.environ.get("KLTN_EXPORT_DEVICE", "")).strip().lower()
    if override in {"auto", "cuda", "cpu"}:
        return override
    runtime = resolved.get("runtime", {}) if isinstance(resolved, Mapping) else {}
    configured = str(runtime.get("device", "auto")).strip().lower() if isinstance(runtime, Mapping) else "auto"
    if configured in {"auto", "cuda", "cpu"}:
        return configured
    return "auto"


def _resolve_export_execution_kwargs() -> Dict[str, Any]:
    sequential_flag = str(os.environ.get("KLTN_EXPORT_SEQUENTIAL", "")).strip().lower()
    sequential = sequential_flag in {"1", "true", "yes", "on"}
    raw_batch_size = str(os.environ.get("KLTN_EXPORT_MAX_BATCH_SIZE", "")).strip()
    try:
        max_batch_size = max(1, int(raw_batch_size)) if raw_batch_size else (1 if sequential else 8)
    except ValueError:
        max_batch_size = 1 if sequential else 8
    return {
        "batch_independent_rooms": not sequential,
        "max_batch_size": int(max_batch_size),
    }


def _tile_color(tile: int) -> Tuple[int, int, int]:
    palette = {
        0: (16, 18, 24),
        1: (214, 198, 155),
        2: (82, 98, 122),
        3: (125, 93, 66),
        10: (246, 214, 52),
        11: (246, 214, 52),
        12: (246, 214, 52),
        13: (181, 93, 212),
        14: (200, 44, 44),
        15: (68, 175, 98),
        21: (68, 175, 98),
        22: (246, 214, 52),
        23: (200, 44, 44),
        31: (255, 166, 71),
        32: (110, 203, 220),
        41: (76, 73, 164),
        42: (193, 180, 227),
        43: (168, 92, 198),
    }
    if tile in palette:
        return palette[tile]
    return ((97 * (tile + 3)) % 255, (61 * (tile + 7)) % 255, (151 * (tile + 11)) % 255)


_ROOM_SHEET_BACKGROUND: Tuple[int, int, int] = (234, 239, 247)


def _release_torch_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if torch is not None:
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    message = str(exc).strip().lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message or "cudnn" in message)


def _normalized_execution_kwargs(execution_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "batch_independent_rooms": bool(execution_kwargs.get("batch_independent_rooms", True)),
        "max_batch_size": max(1, int(execution_kwargs.get("max_batch_size", 8) or 1)),
    }


def _build_generation_retry_plan(execution_kwargs: Mapping[str, Any]) -> List[Dict[str, Any]]:
    base = _normalized_execution_kwargs(execution_kwargs)
    attempts = [
        {
            "name": "configured",
            "device_override": None,
            "execution_kwargs": dict(base),
        }
    ]
    safe_cuda = {
        "batch_independent_rooms": False,
        "max_batch_size": 1,
    }
    if safe_cuda != base:
        attempts.append(
            {
                "name": "sequential_cuda_batch1",
                "device_override": None,
                "execution_kwargs": safe_cuda,
            }
        )
    attempts.append(
        {
            "name": "sequential_cpu_batch1",
            "device_override": "cpu",
            "execution_kwargs": safe_cuda,
        }
    )
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for attempt in attempts:
        key = (
            str(attempt.get("device_override") or "configured"),
            bool(attempt["execution_kwargs"]["batch_independent_rooms"]),
            int(attempt["execution_kwargs"]["max_batch_size"]),
        )
        if key in seen:
            continue
        deduped.append(attempt)
        seen.add(key)
    return deduped


def _crop_grid_to_non_void(grid: np.ndarray, *, void_tile: int = 0, margin: int = 1) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int32)
    r0, c0, r1, c1 = _non_void_bounds(grid, void_tile=void_tile, margin=margin)
    return grid[r0:r1 + 1, c0:c1 + 1]


def _non_void_bounds(
    grid: np.ndarray,
    *,
    void_tile: int = 0,
    margin: int = 1,
) -> Tuple[int, int, int, int]:
    grid = np.asarray(grid, dtype=np.int32)
    occupied = np.argwhere(grid != int(void_tile))
    if occupied.size == 0:
        return (0, 0, int(grid.shape[0] - 1), int(grid.shape[1] - 1))
    r0, c0 = occupied.min(axis=0)
    r1, c1 = occupied.max(axis=0)
    r0 = max(0, int(r0) - int(margin))
    c0 = max(0, int(c0) - int(margin))
    r1 = min(grid.shape[0] - 1, int(r1) + int(margin))
    c1 = min(grid.shape[1] - 1, int(c1) + int(margin))
    return (r0, c0, r1, c1)


def save_grid_png(
    grid: np.ndarray,
    out_path: Path,
    tile_px: int = 16,
    *,
    crop_void: bool = False,
) -> None:
    grid = np.asarray(grid, dtype=np.int32)
    if crop_void:
        grid = _crop_grid_to_non_void(grid)
    h, w = grid.shape
    canvas = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            y0 = r * tile_px
            x0 = c * tile_px
            canvas[y0:y0 + tile_px, x0:x0 + tile_px] = _tile_color(int(grid[r, c]))
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    accent_tiles = {3, 20, 21, 22, 23, 30, 31, 32, 33, 43}
    for r in range(h):
        for c in range(w):
            tile = int(grid[r, c])
            if tile not in accent_tiles:
                continue
            _draw_semantic_overlay_tile(
                draw,
                tile=tile,
                x0=c * tile_px,
                y0=r * tile_px,
                tile_px=tile_px,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _draw_semantic_overlay_tile(
    draw: ImageDraw.ImageDraw,
    *,
    tile: int,
    x0: int,
    y0: int,
    tile_px: int,
) -> None:
    """
    Add lightweight semantic glyphs to the regular export PNGs.

    These exports are intentionally flatter than the stylized sheets, but they
    still need to make interactive tiles readable in report figures.
    """
    x1 = x0 + tile_px - 1
    y1 = y0 + tile_px - 1
    inset = max(1, tile_px // 6)
    cx = x0 + tile_px // 2
    cy = y0 + tile_px // 2

    if int(tile) == 3:  # pushable block
        draw.rectangle(
            [x0 + inset, y0 + inset, x1 - inset, y1 - inset],
            outline=(78, 52, 31),
            width=max(1, tile_px // 10),
        )
        stud = max(1, tile_px // 8)
        draw.rectangle([cx - stud, cy - stud, cx + stud, cy + stud], fill=(236, 215, 173))
    elif int(tile) == 20:  # enemy
        eye_r = max(1, tile_px // 10)
        draw.ellipse([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=(86, 0, 0), width=max(1, tile_px // 10))
        draw.ellipse([cx - tile_px // 5 - eye_r, cy - eye_r, cx - tile_px // 5 + eye_r, cy + eye_r], fill=(18, 18, 18))
        draw.ellipse([cx + tile_px // 5 - eye_r, cy - eye_r, cx + tile_px // 5 + eye_r, cy + eye_r], fill=(18, 18, 18))
    elif int(tile) in {30, 31}:  # key / boss key
        ring_r = max(2, tile_px // 6)
        ring_x = x0 + inset + ring_r + 1
        ring_y = y0 + inset + ring_r + 1
        color = (255, 228, 120) if int(tile) == 30 else (255, 205, 140)
        draw.ellipse([ring_x - ring_r, ring_y - ring_r, ring_x + ring_r, ring_y + ring_r], outline=color, width=max(1, tile_px // 12))
        draw.line([ring_x + ring_r, ring_y, x1 - inset, ring_y], fill=color, width=max(1, tile_px // 12))
    elif int(tile) in {32, 33}:  # item markers
        color = (180, 245, 255) if int(tile) == 32 else (232, 232, 232)
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=color, width=max(1, tile_px // 12))
        draw.line([x0 + inset, y1 - inset, x1 - inset, y0 + inset], fill=color, width=max(1, tile_px // 12))
    elif int(tile) == 21:  # start
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=(214, 255, 214), width=max(1, tile_px // 10))
    elif int(tile) == 22:  # triforce
        draw.polygon([(cx, y0 + inset), (x0 + inset, y1 - inset), (x1 - inset, y1 - inset)], outline=(160, 122, 18), fill=(255, 236, 126))
    elif int(tile) == 23:  # boss
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=(110, 0, 0), width=max(1, tile_px // 10))
    elif int(tile) == 43:  # puzzle marker
        draw.line([x0 + inset, cy, x1 - inset, cy], fill=(245, 195, 255), width=max(1, tile_px // 10))
        draw.line([cx, y0 + inset, cx, y1 - inset], fill=(245, 195, 255), width=max(1, tile_px // 10))


def _tile_to_vglc_char(tile: int) -> str:
    try:
        return str(semantic_to_vglc_char(int(tile)))
    except Exception:
        return "F"


def save_grid_txt(grid: np.ndarray, out_path: Path) -> str:
    grid = np.asarray(grid, dtype=np.int32)
    lines = ["".join(_tile_to_vglc_char(int(v)) for v in row) for row in grid]
    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def save_grid_json(grid: np.ndarray, out_path: Path) -> None:
    grid = np.asarray(grid, dtype=np.int32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(grid.tolist(), indent=2), encoding="utf-8")


def _char_color(ch: str) -> Tuple[int, int, int]:
    palette = {
        "-": (16, 18, 24),
        "F": (214, 198, 155),
        "W": (82, 98, 122),
        "B": (125, 93, 66),
        "D": (246, 214, 52),
        "M": (200, 44, 44),
        "S": (68, 175, 98),
        "P": (168, 92, 198),
        "O": (255, 166, 71),
        "I": (110, 203, 220),
        "K": (255, 166, 71),
    }
    return palette.get(ch, (130, 130, 130))


def _crop_char_lines(lines: List[str], *, void_char: str = "-", margin: int = 1) -> List[str]:
    if not lines:
        return lines
    occupied: List[Tuple[int, int]] = []
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch != void_char:
                occupied.append((r, c))
    if not occupied:
        return lines
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    r0 = max(0, min(rows) - int(margin))
    c0 = max(0, min(cols) - int(margin))
    r1 = min(len(lines) - 1, max(rows) + int(margin))
    c1 = min(len(lines[0]) - 1, max(cols) + int(margin))
    return [line[c0:c1 + 1] for line in lines[r0:r1 + 1]]


def save_char_grid_png_from_txt(txt_path: Path, out_path: Path, tile_px: int = 16, *, crop_void: bool = True) -> None:
    lines = [line.rstrip("\n") for line in txt_path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line != ""]
    if crop_void:
        lines = _crop_char_lines(lines)
    h = len(lines)
    w = len(lines[0]) if h else 0
    canvas = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            canvas[r * tile_px:(r + 1) * tile_px, c * tile_px:(c + 1) * tile_px] = _char_color(ch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def save_rooms_sheet(room_grids: Dict[int, np.ndarray], out_path: Path, tile_px: int = 16, columns: int = 4) -> None:
    room_ids = sorted(int(rid) for rid in room_grids.keys())
    if not room_ids:
        raise ValueError("room_grids is empty")

    rows = int(math.ceil(len(room_ids) / max(1, columns)))
    sheet_h = rows * ROOM_HEIGHT * tile_px
    sheet_w = max(1, columns) * ROOM_WIDTH * tile_px
    canvas = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, room_id in enumerate(room_ids):
        grid = np.asarray(room_grids[room_id], dtype=np.int32)
        room_canvas = np.zeros((ROOM_HEIGHT * tile_px, ROOM_WIDTH * tile_px, 3), dtype=np.uint8)
        for r in range(ROOM_HEIGHT):
            for c in range(ROOM_WIDTH):
                room_canvas[r * tile_px:(r + 1) * tile_px, c * tile_px:(c + 1) * tile_px] = _tile_color(int(grid[r, c]))
        row = idx // columns
        col = idx % columns
        y0 = row * ROOM_HEIGHT * tile_px
        x0 = col * ROOM_WIDTH * tile_px
        canvas[y0:y0 + room_canvas.shape[0], x0:x0 + room_canvas.shape[1]] = room_canvas

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def _draw_stylized_tile(draw: ImageDraw.ImageDraw, *, tile: int, x0: int, y0: int, tile_px: int) -> None:
    base = _tile_color(int(tile))
    x1 = x0 + tile_px - 1
    y1 = y0 + tile_px - 1
    draw.rectangle([x0, y0, x1, y1], fill=base)

    def clamp_rgb(rgb: Tuple[int, int, int], delta: int) -> Tuple[int, int, int]:
        return tuple(max(0, min(255, int(v) + int(delta))) for v in rgb)

    hi = clamp_rgb(base, 22)
    lo = clamp_rgb(base, -22)
    draw.line([x0, y0, x1, y0], fill=hi, width=max(1, tile_px // 12))
    draw.line([x0, y0, x0, y1], fill=hi, width=max(1, tile_px // 12))
    draw.line([x0, y1, x1, y1], fill=lo, width=max(1, tile_px // 12))
    draw.line([x1, y0, x1, y1], fill=lo, width=max(1, tile_px // 12))

    inset = max(1, tile_px // 6)
    cx = x0 + tile_px // 2
    cy = y0 + tile_px // 2

    if int(tile) == 2:  # wall
        brick = max(2, tile_px // 4)
        for y in range(y0 + brick, y1, brick):
            draw.line([x0 + 1, y, x1 - 1, y], fill=clamp_rgb(base, 34), width=1)
        for x in range(x0 + brick, x1, brick):
            draw.line([x, y0 + 1, x, y1 - 1], fill=clamp_rgb(base, -8), width=1)
    elif int(tile) == 3:  # block
        draw.rectangle(
            [x0 + inset, y0 + inset, x1 - inset, y1 - inset],
            fill=clamp_rgb(base, -10),
            outline=clamp_rgb(base, -34),
            width=max(1, tile_px // 10),
        )
        stud = max(1, tile_px // 8)
        draw.rectangle([cx - stud, cy - stud, cx + stud, cy + stud], fill=clamp_rgb(base, 26))
        draw.line([x0 + inset, cy, x0 + inset + stud + 1, cy], fill=clamp_rgb(base, 34), width=max(1, tile_px // 14))
        draw.line([x1 - inset - stud - 1, cy, x1 - inset, cy], fill=clamp_rgb(base, 34), width=max(1, tile_px // 14))
    elif int(tile) in {10, 11, 12, 13, 14, 15}:  # doors
        door_w = max(4, tile_px // 2)
        draw.rectangle([cx - door_w // 2, y0 + 1, cx + door_w // 2, y1 - 1], fill=clamp_rgb(base, -25))
        if int(tile) == 11:
            draw.ellipse([cx - 2, cy - 4, cx + 2, cy], fill=(255, 220, 120))
            draw.rectangle([cx - 1, cy, cx + 1, cy + 4], fill=(255, 220, 120))
        elif int(tile) == 14:
            draw.rectangle([cx - door_w // 2, y0 + 1, cx + door_w // 2, y0 + inset], fill=(215, 70, 70))
        elif int(tile) == 13:
            draw.line([x0 + inset, cy, x1 - inset, cy], fill=(220, 170, 255), width=max(1, tile_px // 10))
    elif int(tile) == 20:  # enemy
        eye_r = max(1, tile_px // 10)
        body_top = y0 + inset
        body_bottom = y1 - inset
        draw.ellipse([x0 + inset, body_top, x1 - inset, body_bottom], fill=clamp_rgb(base, 6), outline=clamp_rgb(base, -24))
        draw.ellipse([cx - tile_px // 5 - eye_r, cy - eye_r, cx - tile_px // 5 + eye_r, cy + eye_r], fill=(18, 18, 18))
        draw.ellipse([cx + tile_px // 5 - eye_r, cy - eye_r, cx + tile_px // 5 + eye_r, cy + eye_r], fill=(18, 18, 18))
        draw.line([cx - tile_px // 6, y1 - inset - 2, cx + tile_px // 6, y1 - inset - 2], fill=(255, 210, 210), width=max(1, tile_px // 14))
    elif int(tile) == 21:  # start
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=(210, 255, 210), width=max(1, tile_px // 10))
    elif int(tile) == 22:  # triforce
        draw.polygon([(cx, y0 + inset), (x0 + inset, y1 - inset), (x1 - inset, y1 - inset)], fill=(255, 235, 130), outline=(180, 140, 20))
    elif int(tile) == 23:  # boss
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], fill=(160, 30, 30))
    elif int(tile) == 30:  # small key
        ring_r = max(2, tile_px // 6)
        ring_x = x0 + inset + ring_r + 1
        ring_y = y0 + inset + ring_r + 1
        draw.ellipse([ring_x - ring_r, ring_y - ring_r, ring_x + ring_r, ring_y + ring_r], outline=(255, 232, 120), width=max(1, tile_px // 12))
        draw.line([ring_x + ring_r, ring_y, x1 - inset, ring_y], fill=(255, 220, 110), width=max(1, tile_px // 12))
        draw.line([x1 - inset - 2, ring_y, x1 - inset - 2, ring_y + ring_r + 1], fill=(255, 220, 110), width=max(1, tile_px // 12))
        draw.line([x1 - inset - 5, ring_y, x1 - inset - 5, ring_y + ring_r], fill=(255, 220, 110), width=max(1, tile_px // 12))
    elif int(tile) == 31:  # boss key
        draw.ellipse([x0 + inset, y0 + inset, x0 + inset + tile_px // 3, y0 + inset + tile_px // 3], outline=(255, 225, 140), width=max(1, tile_px // 12))
        draw.rectangle([cx - 1, cy - 1, x1 - inset, cy + 1], fill=(255, 190, 100))
    elif int(tile) == 32:  # key item
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=(180, 245, 255), width=max(1, tile_px // 10))
        draw.line([x0 + inset, y1 - inset, x1 - inset, y0 + inset], fill=(180, 245, 255), width=max(1, tile_px // 12))
    elif int(tile) == 33:  # minor item
        draw.polygon(
            [(cx, y0 + inset), (x0 + inset, cy), (cx, y1 - inset), (x1 - inset, cy)],
            fill=(225, 225, 225),
            outline=(160, 160, 160),
        )
    elif int(tile) == 42:  # stair
        step_h = max(2, tile_px // 6)
        for i in range(4):
            draw.rectangle([x0 + inset + i, y1 - inset - (i + 1) * step_h, x1 - inset - i, y1 - inset - i * step_h], fill=clamp_rgb(base, -8))
    elif int(tile) == 43:  # puzzle
        draw.line([x0 + inset, cy, x1 - inset, cy], fill=(245, 195, 255), width=max(1, tile_px // 10))
        draw.line([cx, y0 + inset, cx, y1 - inset], fill=(245, 195, 255), width=max(1, tile_px // 10))


def save_stylized_grid_png(
    grid: np.ndarray,
    out_path: Path,
    tile_px: int = 24,
    *,
    crop_void: bool = True,
) -> None:
    grid = np.asarray(grid, dtype=np.int32)
    if crop_void:
        grid = _crop_grid_to_non_void(grid)
    h, w = grid.shape
    canvas = Image.new("RGB", (w * tile_px, h * tile_px), _tile_color(0))
    draw = ImageDraw.Draw(canvas)
    for r in range(h):
        for c in range(w):
            _draw_stylized_tile(
                draw,
                tile=int(grid[r, c]),
                x0=c * tile_px,
                y0=r * tile_px,
                tile_px=tile_px,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_stylized_rooms_sheet(room_grids: Dict[int, np.ndarray], out_path: Path, tile_px: int = 20, columns: int = 4) -> None:
    room_ids = sorted(int(rid) for rid in room_grids.keys())
    if not room_ids:
        raise ValueError("room_grids is empty")

    rows = int(math.ceil(len(room_ids) / max(1, columns)))
    margin = max(4, tile_px // 2)
    sheet_h = rows * (ROOM_HEIGHT * tile_px + margin) + margin
    sheet_w = max(1, columns) * (ROOM_WIDTH * tile_px + margin) + margin
    # Use a neutral sheet background instead of VOID-black so the canonical
    # two-tile Zelda wall shell does not read as an extra outer wall layer.
    canvas = Image.new("RGB", (sheet_w, sheet_h), _ROOM_SHEET_BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    for idx, room_id in enumerate(room_ids):
        grid = np.asarray(room_grids[room_id], dtype=np.int32)
        row = idx // columns
        col = idx % columns
        x0 = margin + col * (ROOM_WIDTH * tile_px + margin)
        y0 = margin + row * (ROOM_HEIGHT * tile_px + margin)
        for r in range(ROOM_HEIGHT):
            for c in range(ROOM_WIDTH):
                _draw_stylized_tile(
                    draw,
                    tile=int(grid[r, c]),
                    x0=x0 + c * tile_px,
                    y0=y0 + r * tile_px,
                    tile_px=tile_px,
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _compute_generation_validation(
    *,
    dungeon_grid: np.ndarray,
    mission_graph: nx.Graph,
    room_puzzle_metadata: Optional[Mapping[str, Any]] = None,
    room_positions: Optional[Mapping[Any, Tuple[int, int]]] = None,
    room_to_node: Optional[Mapping[Any, Any]] = None,
    node_to_room: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute explicit playability diagnostics for export summaries.

    Placement and repair metrics are not enough once post-overlay correction can
    hide visually bad auxiliary outputs. This runs the repo's graph/grid
    validators plus the CBS behavioral probe so audits carry actual playability
    evidence.
    """
    grid = np.asarray(dungeon_grid, dtype=np.int32)
    payload: Dict[str, Any] = {}
    env_kwargs = {
        "graph": mission_graph,
        "room_to_node": room_to_node,
        "room_positions": room_positions,
        "node_to_room": node_to_room,
        "room_puzzle_metadata": room_puzzle_metadata,
    }

    try:
        from src.evaluation.validator import ExternalValidator
        from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
        from src.simulation.search_base import GameStateSearchConfig, SearchRepresentation
        from src.simulation.search_factory import run_game_state_solver
        from src.simulation.validator import GraphGuidedValidator, ZeldaLogicEnv, ZeldaValidator
        from src.utils.graph_utils import filter_virtual_nodes, validate_goal_subgraph
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "error": f"validation_import_error: {exc}",
        }

    def _finite_or_none(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
        raw = str(os.environ.get(name, "")).strip()
        if not raw:
            return int(default)
        try:
            return max(int(min_value), int(raw))
        except ValueError:
            return int(default)

    validation_mode = str(os.environ.get("KLTN_VALIDATION_MODE", "full")).strip().lower()
    if validation_mode not in {"full", "core", "oracle_only"}:
        validation_mode = "full"

    def _run_search_suite() -> Dict[str, Any]:
        if validation_mode == "full":
            oracle_timeout_default = int(max(200000, grid.size * 64))
            comparison_timeout_default = int(max(50000, grid.size * 16))
            advanced_timeout_default = int(max(100000, grid.size * 24))
        else:
            oracle_timeout_default = int(max(50000, grid.size * 24))
            comparison_timeout_default = int(max(12000, grid.size * 6))
            advanced_timeout_default = int(max(20000, grid.size * 10))
        oracle_timeout = _env_int("KLTN_VALIDATION_ASTAR_TIMEOUT", oracle_timeout_default)
        comparison_timeout = _env_int("KLTN_VALIDATION_COMPARISON_TIMEOUT", comparison_timeout_default)
        advanced_timeout = _env_int("KLTN_VALIDATION_ADVANCED_TIMEOUT", advanced_timeout_default)
        all_specs = list(iter_game_state_algorithm_specs())
        if validation_mode == "oracle_only":
            specs = [spec for spec in all_specs if str(spec.key) == "astar"]
        elif validation_mode == "core":
            specs = [
                spec
                for spec in all_specs
                if str(spec.key) in {"astar", "bfs", "dijkstra", "greedy"}
            ]
        else:
            specs = all_specs
        suite: Dict[str, Any] = {
            "search_suite_version": VALIDATION_SEARCH_SUITE_VERSION,
            "mode": str(validation_mode),
            "tile_state_space": {},
            "agreement": {},
            "excluded_algorithms": dict(VALIDATION_EXCLUDED_ALGORITHMS),
            "omitted_by_mode": [
                str(spec.key) for spec in all_specs if spec not in specs
            ],
            "notes": {
                "oracle": "A* remains the hard grid-level oracle in this suite.",
                "replanning": "D* Lite is reported as an incremental replanning probe, not the primary static correctness oracle.",
                "behavioral_probe": "CBS is reported separately because it is a bounded-rational behavior probe, not the hard correctness oracle.",
            },
        }

        astar_path_length: Optional[int] = None
        astar_states: Optional[int] = None
        astar_success = False

        for spec in specs:
            algorithm_name = str(spec.key)
            algorithm_idx = int(spec.index)
            env = ZeldaLogicEnv(semantic_grid=grid, render_mode=False, **env_kwargs)
            started = time.perf_counter()
            try:
                if algorithm_name == "astar":
                    timeout = oracle_timeout
                elif algorithm_name in {"dstar_lite", "bidirectional_astar"}:
                    timeout = advanced_timeout
                elif algorithm_name == "dfs_iddfs":
                    timeout = advanced_timeout
                else:
                    timeout = comparison_timeout
                result = run_game_state_solver(
                    env,
                    algorithm_idx=algorithm_idx,
                    config=GameStateSearchConfig(
                        timeout=timeout,
                        allow_diagonals=False,
                        rules_profile="vglc_strict",
                        representation=SearchRepresentation.TILE,
                        max_depth=max(500, int(grid.size)),
                        use_iddfs=True,
                    ),
                )
                elapsed = float(time.perf_counter() - started)
                entry = {
                    "success": bool(result.success),
                    "path_length": int(len(result.path or [])),
                    "states_explored": int(result.states_explored or 0),
                    "time_sec": elapsed,
                    "algorithm": str(result.algorithm),
                    "validation_role": str(spec.validation_role),
                    "canonical_use": str(spec.canonical_use),
                    "rules_profile": "vglc_strict",
                    "allow_diagonals": False,
                    "timeout_limit_states": int(timeout),
                    **dict(result.metadata or {}),
                }
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                elapsed = float(time.perf_counter() - started)
                if algorithm_name == "astar":
                    timeout = oracle_timeout
                elif algorithm_name in {"dstar_lite", "bidirectional_astar", "dfs_iddfs"}:
                    timeout = advanced_timeout
                else:
                    timeout = comparison_timeout
                entry = {
                    "success": False,
                    "path_length": 0,
                    "states_explored": 0,
                    "time_sec": elapsed,
                    "algorithm": str(algorithm_name).upper(),
                    "validation_role": str(spec.validation_role),
                    "canonical_use": str(spec.canonical_use),
                    "rules_profile": "vglc_strict",
                    "allow_diagonals": False,
                    "timeout_limit_states": int(timeout),
                    "error_message": f"{algorithm_name}_validation_error: {exc}",
                }
            finally:
                try:
                    env.close()
                except Exception:
                    pass

            suite["tile_state_space"][algorithm_name] = entry
            if algorithm_name == "astar":
                astar_success = bool(entry["success"])
                astar_path_length = int(entry["path_length"])
                astar_states = int(entry["states_explored"])

        for algorithm_name, entry in suite["tile_state_space"].items():
            if astar_success and bool(entry.get("success", False)):
                entry["path_ratio_vs_astar"] = _finite_or_none(
                    float(entry["path_length"]) / float(max(1, int(astar_path_length or 0)))
                )
                entry["states_ratio_vs_astar"] = _finite_or_none(
                    float(entry["states_explored"]) / float(max(1, int(astar_states or 0)))
                )
            else:
                entry["path_ratio_vs_astar"] = None
                entry["states_ratio_vs_astar"] = None

        bfs_entry = suite["tile_state_space"].get("bfs", {})
        dijkstra_entry = suite["tile_state_space"].get("dijkstra", {})
        greedy_entry = suite["tile_state_space"].get("greedy", {})
        suite["agreement"] = {
            "astar_success": bool(astar_success),
            "bfs_matches_astar_path_length": bool(
                astar_success
                and bool(bfs_entry.get("success", False))
                and int(bfs_entry.get("path_length", 0)) == int(astar_path_length or 0)
            ),
            "dijkstra_matches_astar_path_length": bool(
                astar_success
                and bool(dijkstra_entry.get("success", False))
                and int(dijkstra_entry.get("path_length", 0)) == int(astar_path_length or 0)
            ),
            "greedy_suboptimality_vs_astar": _finite_or_none(greedy_entry.get("path_ratio_vs_astar")),
            "all_algorithms_solved": all(
                bool(entry.get("success", False))
                for entry in suite["tile_state_space"].values()
            ),
        }
        return suite

    try:
        validator = ZeldaValidator()
        if validation_mode == "full":
            validator_timeout_default = int(max(120000, grid.size * 32))
            softlock_timeout_default = int(max(100000, grid.size * 28))
        else:
            validator_timeout_default = int(max(40000, grid.size * 12))
            softlock_timeout_default = int(max(30000, grid.size * 10))
        validator_timeout = _env_int("KLTN_VALIDATION_VALIDATOR_TIMEOUT", validator_timeout_default)
        softlock_timeout = _env_int("KLTN_VALIDATION_SOFTLOCK_TIMEOUT", softlock_timeout_default)
        grid_result = validator.validate_single(
            grid,
            render=False,
            persona_mode="balanced",
            solver_timeout=validator_timeout,
            **env_kwargs,
        )
        softlock_safe, softlock_issues = validator.check_soft_locks_deterministic(
            grid,
            solver_timeout=softlock_timeout,
            **env_kwargs,
        )
        payload["astar_grid"] = {
            "solvable": bool(grid_result.is_solvable),
            "is_valid_syntax": bool(grid_result.is_valid_syntax),
            "path_length": int(getattr(grid_result, "path_length", 0) or 0),
            "reachability": float(getattr(grid_result, "reachability", 0.0) or 0.0),
            "backtracking_score": float(getattr(grid_result, "backtracking_score", 0.0) or 0.0),
            "error_message": str(getattr(grid_result, "error_message", "") or ""),
            "solver_used": str(getattr(grid_result, "solver_used", "astar") or "astar"),
            "primary_solver_solved": bool(getattr(grid_result, "primary_solver_solved", False)),
            "primary_solver_error": str(getattr(grid_result, "primary_solver_error", "") or ""),
            "states_explored": int(getattr(grid_result, "states_explored", 0) or 0),
        }
        payload["softlock_check"] = {
            "is_safe": bool(softlock_safe),
            "issues": [str(item) for item in list(softlock_issues or [])],
        }
        try:
            from types import SimpleNamespace

            graph_rooms: Dict[str, Any] = {}
            for room_slot, offset in dict(room_positions or {}).items():
                if not isinstance(offset, (list, tuple)) or len(offset) < 2:
                    continue
                room_id = dict(room_to_node or {}).get(room_slot)
                if room_id is None:
                    continue
                row_off, col_off = int(offset[0]), int(offset[1])
                room_grid = grid[row_off : row_off + ROOM_HEIGHT, col_off : col_off + ROOM_WIDTH]
                graph_rooms[str(room_id)] = SimpleNamespace(grid=np.asarray(room_grid, dtype=np.int32))

            graph_guided = GraphGuidedValidator().validate_dungeon_with_graph(
                SimpleNamespace(graph=mission_graph, rooms=graph_rooms)
            )
            payload["graph_guided_oracle"] = {
                "solvable": bool(graph_guided.is_solvable),
                "graph_path": [int(node) for node in list(graph_guided.graph_path or [])],
                "subgraph_path": [int(node) for node in list(graph_guided.subgraph_path or [])],
                "missing_rooms": [int(node) for node in list(graph_guided.missing_rooms or [])],
                "connectivity_score": float(graph_guided.connectivity_score or 0.0),
                "error_message": str(graph_guided.error_message or ""),
                "room_traversable_count": int(
                    sum(
                        1
                        for item in dict(graph_guided.room_validations or {}).values()
                        if bool(item.get("is_traversable", False))
                    )
                ),
                "room_validation_count": int(len(dict(graph_guided.room_validations or {}))),
            }
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            payload["graph_guided_oracle"] = {
                "solvable": False,
                "graph_path": [],
                "subgraph_path": [],
                "missing_rooms": [],
                "connectivity_score": 0.0,
                "error_message": f"graph_guided_validation_error: {exc}",
                "room_traversable_count": 0,
                "room_validation_count": 0,
            }
        payload["search_algorithms"] = _run_search_suite()

        if validation_mode == "full":
            cbs_timeout_default = int(max(5000, grid.size * 6))
        else:
            cbs_timeout_default = int(max(2500, grid.size * 3))
        cbs_timeout = _env_int("KLTN_VALIDATION_CBS_TIMEOUT", cbs_timeout_default)
        env_cbs = ZeldaLogicEnv(semantic_grid=grid, render_mode=False, **env_kwargs)
        try:
            cbs = CognitiveBoundedSearch(env_cbs, persona="balanced", timeout=cbs_timeout, seed=123)
            cbs_success, cbs_path, cbs_states, cbs_metrics = cbs.solve()
        finally:
            try:
                env_cbs.close()
            except Exception:
                pass

        optimal_path_length = int(getattr(grid_result, "path_length", 0) or 0)
        cbs_path_length = int(len(cbs_path or []))
        confusion_ratio = (
            float(cbs_path_length) / float(max(1, optimal_path_length))
            if optimal_path_length > 0 and cbs_success
            else float("inf")
        )
        payload["cbs_balanced"] = {
            "seed": 123,
            "success": bool(cbs_success),
            "path_length": cbs_path_length,
            "states_explored": int(cbs_states or 0),
            "confusion_ratio_vs_astar": float(confusion_ratio),
            "confusion_index": float(getattr(cbs_metrics, "confusion_index", 0.0) or 0.0),
            "navigation_entropy": float(getattr(cbs_metrics, "navigation_entropy", 0.0) or 0.0),
            "room_entropy": float(getattr(cbs_metrics, "room_entropy", 0.0) or 0.0),
            "cognitive_load": float(getattr(cbs_metrics, "cognitive_load", 0.0) or 0.0),
            "aha_latency": int(getattr(cbs_metrics, "aha_latency", 0) or 0),
            "deliberation_events": int(getattr(cbs_metrics, "deliberation_events", 0) or 0),
            "budget_exhaustion_events": int(getattr(cbs_metrics, "budget_exhaustion_events", 0) or 0),
            "peak_frustration": float(getattr(cbs_metrics, "peak_frustration", 0.0) or 0.0),
            "affordance_reactivations": int(getattr(cbs_metrics, "affordance_reactivations", 0) or 0),
            "affordance_guided_steps": int(getattr(cbs_metrics, "affordance_guided_steps", 0) or 0),
            "inventory_change_events": int(getattr(cbs_metrics, "inventory_change_events", 0) or 0),
            "focus_switches": int(getattr(cbs_metrics, "focus_switches", 0) or 0),
            "focus_guided_steps": int(getattr(cbs_metrics, "focus_guided_steps", 0) or 0),
            "status": "success" if bool(cbs_success) else ("budget_exhausted" if int(cbs_states or 0) >= int(cbs_timeout) else "failed"),
        }
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        payload["astar_grid"] = {
            "solvable": False,
            "is_valid_syntax": False,
            "path_length": 0,
            "reachability": 0.0,
            "backtracking_score": 0.0,
            "error_message": f"grid_validation_error: {exc}",
            "solver_used": "astar",
            "primary_solver_solved": False,
            "primary_solver_error": f"grid_validation_error: {exc}",
            "states_explored": 0,
        }
        payload["softlock_check"] = {
            "is_safe": False,
            "issues": [f"grid_validation_error: {exc}"],
        }
        payload["graph_guided_oracle"] = {
            "solvable": False,
            "graph_path": [],
            "subgraph_path": [],
            "missing_rooms": [],
            "connectivity_score": 0.0,
            "error_message": f"graph_guided_validation_error: {exc}",
            "room_traversable_count": 0,
            "room_validation_count": 0,
        }
        payload["search_algorithms"] = {
            "tile_state_space": {},
            "agreement": {
                "astar_success": False,
                "bfs_matches_astar_path_length": False,
                "dijkstra_matches_astar_path_length": False,
                "greedy_suboptimality_vs_astar": None,
                "all_algorithms_solved": False,
            },
        }
        payload["cbs_balanced"] = {
            "success": False,
            "path_length": 0,
            "states_explored": 0,
            "confusion_ratio_vs_astar": float("inf"),
            "confusion_index": 0.0,
            "navigation_entropy": 0.0,
            "room_entropy": 0.0,
            "cognitive_load": 0.0,
            "aha_latency": 0,
            "deliberation_events": 0,
            "budget_exhaustion_events": 0,
            "peak_frustration": 0.0,
            "affordance_reactivations": 0,
            "affordance_guided_steps": 0,
            "inventory_change_events": 0,
            "focus_switches": 0,
            "focus_guided_steps": 0,
            "status": "failed",
            "error_message": f"cbs_validation_error: {exc}",
        }

    try:
        mission_graph_physical, virtual_info = filter_virtual_nodes(mission_graph)
        goal_gauntlet_valid, goal_gauntlet_errors = validate_goal_subgraph(mission_graph_physical)
        graph_validator = ExternalValidator()
        graph_result = graph_validator.validate(mission_graph_physical)
        payload["graph_progression"] = {
            "solvable": bool(graph_result.is_solvable),
            "path_length": int(getattr(graph_result, "path_length", 0) or 0),
            "states_explored": int(getattr(graph_result, "states_explored", 0) or 0),
            "failure_reason": str(getattr(graph_result, "failure_reason", "") or ""),
            "goal_gauntlet_valid": bool(goal_gauntlet_valid),
            "goal_gauntlet_errors": [str(item) for item in list(goal_gauntlet_errors or [])],
            "virtual_nodes_removed": int(len(virtual_info.get("removed_nodes", []) or [])),
        }
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        payload["graph_progression"] = {
            "solvable": False,
            "path_length": 0,
            "states_explored": 0,
            "failure_reason": f"graph_validation_error: {exc}",
            "goal_gauntlet_valid": False,
            "goal_gauntlet_errors": [f"graph_validation_error: {exc}"],
            "virtual_nodes_removed": 0,
        }

    payload["mechanical_contract"] = {
        "tile_oracle_solved": bool(payload.get("astar_grid", {}).get("solvable", False)),
        "graph_guided_solved": bool(payload.get("graph_guided_oracle", {}).get("solvable", False)),
        "graph_progression_solved": bool(payload.get("graph_progression", {}).get("solvable", False)),
        "goal_gauntlet_valid": bool(payload.get("graph_progression", {}).get("goal_gauntlet_valid", False)),
        "softlock_safe": bool(payload.get("softlock_check", {}).get("is_safe", False)),
    }
    payload["mechanical_contract"]["hybrid_oracle_pass"] = bool(
        payload["mechanical_contract"]["graph_guided_solved"]
        and payload["mechanical_contract"]["graph_progression_solved"]
        and payload["mechanical_contract"]["goal_gauntlet_valid"]
        and payload["mechanical_contract"]["softlock_safe"]
    )

    payload["available"] = True
    return payload


def build_validation_context_from_generation_result(result: Any) -> Dict[str, Any]:
    """
    Extract stitched room metadata required by the hard oracle and P-CBS.

    Validation works in stitched room-slot space. Using raw room ids here causes
    subtle mismatches once puzzle plans and room-local progression metadata are
    attached after stitching.
    """
    stitched_layout = getattr(result, "stitched_layout", None)
    slot_positions = dict(getattr(stitched_layout, "slot_positions", {}) or {})
    room_offsets = dict(getattr(stitched_layout, "room_offsets", {}) or {})

    room_positions: Dict[Any, Tuple[int, int]] = {}
    room_to_node: Dict[Any, Any] = {}
    node_to_room: Dict[Any, Any] = {}

    for room_id, slot_pos in slot_positions.items():
        room_to_node[slot_pos] = room_id
        node_to_room[room_id] = slot_pos
        offset = room_offsets.get(room_id)
        if isinstance(offset, (list, tuple)) and len(offset) >= 2:
            room_positions[slot_pos] = (int(offset[0]), int(offset[1]))

    return {
        "room_puzzle_metadata": dict(getattr(result, "puzzle_metadata", {}) or {}),
        "room_positions": room_positions,
        "room_to_node": room_to_node,
        "node_to_room": node_to_room,
    }


def build_validation_search_stats_payload(validation: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the dedicated validation-search JSON sidecar from full validation output."""
    validation_dict = dict(validation or {})
    return {
        "search_suite_version": VALIDATION_SEARCH_SUITE_VERSION,
        "search_algorithms": dict(validation_dict.get("search_algorithms", {})),
        "hard_oracle": {
            "astar_grid": dict(validation_dict.get("astar_grid", {})),
            "graph_guided_oracle": dict(validation_dict.get("graph_guided_oracle", {})),
            "graph_progression": dict(validation_dict.get("graph_progression", {})),
            "softlock_check": dict(validation_dict.get("softlock_check", {})),
            "mechanical_contract": dict(validation_dict.get("mechanical_contract", {})),
        },
        "behavioral_probe": {
            "cbs_balanced": dict(validation_dict.get("cbs_balanced", {})),
        },
    }


_ROOM_OVERLAY_COLORS: List[Tuple[int, int, int]] = [
    (86, 180, 233),
    (230, 159, 0),
    (0, 158, 115),
    (204, 121, 167),
    (213, 94, 0),
    (240, 228, 66),
    (0, 114, 178),
    (196, 78, 82),
    (127, 127, 127),
]


def _room_sort_key(room_id: Any) -> Tuple[int, Any]:
    try:
        return (0, int(room_id))
    except Exception:
        return (1, str(room_id))


def _room_overlay_label(room_id: Any, graph: nx.Graph) -> str:
    attrs = graph.nodes.get(room_id, {}) if room_id in graph else {}
    raw = str(attrs.get("type", attrs.get("label", "")) or "").strip().upper()
    if raw:
        raw = raw.replace("RESOURCE_FARM", "BOMB").replace("BOSS_DOOR", "BD")
    return f"{room_id}:{raw or 'ROOM'}"


def build_room_layout_payload(
    graph: nx.Graph,
    stitched_layout: StitchedRoomLayout,
) -> Dict[str, Any]:
    graph_positions: Dict[Any, Tuple[int, int]] = {}
    for node_id in graph.nodes():
        attrs = graph.nodes[node_id]
        pos = attrs.get("pos", attrs.get("position"))
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            graph_positions[node_id] = (int(pos[0]), int(pos[1]))

    normalized_graph_positions: Dict[Any, Tuple[int, int]] = {}
    if graph_positions:
        min_r = min(r for r, _ in graph_positions.values())
        min_c = min(c for _, c in graph_positions.values())
        normalized_graph_positions = {
            node_id: (int(r - min_r), int(c - min_c))
            for node_id, (r, c) in graph_positions.items()
        }

    layout_metrics = compute_layout_quality_metrics(
        graph,
        stitched_layout.slot_positions,
    )

    entries: List[Dict[str, Any]] = []
    for room_id in sorted(stitched_layout.layout_map.keys(), key=_room_sort_key):
        bbox = stitched_layout.layout_map[room_id]
        x_min, y_min, x_max, y_max = bbox
        attrs = graph.nodes.get(room_id, {}) if room_id in graph else {}
        graph_pos = normalized_graph_positions.get(room_id)
        slot_pos = stitched_layout.slot_positions.get(room_id)
        slot_matches_graph_pos = bool(graph_pos is not None and slot_pos == graph_pos)
        entries.append(
            {
                "room_id": int(room_id) if str(room_id).isdigit() else str(room_id),
                "label": str(attrs.get("label", "")),
                "type": str(attrs.get("type", "")),
                "graph_pos": list(graph_pos) if graph_pos is not None else None,
                "slot_position": list(slot_pos) if slot_pos is not None else None,
                "room_offset": list(stitched_layout.room_offsets.get(room_id, ())),
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "center": [int((y_min + y_max) // 2), int((x_min + x_max) // 2)],
                "slot_matches_graph_pos": slot_matches_graph_pos,
            }
        )

    return {
        "room_count": int(len(entries)),
        "layout_quality": {
            key: (float(value) if isinstance(value, (int, float)) and value is not None else value)
            for key, value in layout_metrics.items()
        },
        "primary_quality_metric_name": "graph_edge_slot_adjacency_rate",
        "primary_quality_metric_value": layout_metrics.get("graph_edge_slot_adjacency_rate"),
        "rooms": entries,
    }


def save_room_layout_json(
    graph: nx.Graph,
    stitched_layout: StitchedRoomLayout,
    out_path: Path,
) -> Dict[str, Any]:
    payload = build_room_layout_payload(graph, stitched_layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_room_alignment_overlay(
    dungeon_grid: np.ndarray,
    graph: nx.Graph,
    stitched_layout: StitchedRoomLayout,
    out_path: Path,
    *,
    tile_px: int = 20,
    crop_void: bool = True,
) -> None:
    dungeon_grid = np.asarray(dungeon_grid, dtype=np.int32)
    if crop_void:
        r0, c0, r1, c1 = _non_void_bounds(dungeon_grid, void_tile=0, margin=1)
    else:
        r0, c0, r1, c1 = 0, 0, int(dungeon_grid.shape[0] - 1), int(dungeon_grid.shape[1] - 1)
    cropped = dungeon_grid[r0:r1 + 1, c0:c1 + 1]

    canvas = Image.new("RGB", (cropped.shape[1] * tile_px, cropped.shape[0] * tile_px), _tile_color(0))
    draw = ImageDraw.Draw(canvas)
    for row in range(cropped.shape[0]):
        for col in range(cropped.shape[1]):
            _draw_stylized_tile(
                draw,
                tile=int(cropped[row, col]),
                x0=col * tile_px,
                y0=row * tile_px,
                tile_px=tile_px,
            )

    for idx, room_id in enumerate(sorted(stitched_layout.layout_map.keys(), key=_room_sort_key)):
        x_min, y_min, x_max, y_max = stitched_layout.layout_map[room_id]
        if x_max < c0 or x_min > c1 or y_max < r0 or y_min > r1:
            continue
        color = _ROOM_OVERLAY_COLORS[idx % len(_ROOM_OVERLAY_COLORS)]
        lx0 = max(0, x_min - c0) * tile_px
        ly0 = max(0, y_min - r0) * tile_px
        lx1 = (min(c1, x_max) - c0 + 1) * tile_px - 1
        ly1 = (min(r1, y_max) - r0 + 1) * tile_px - 1
        border_w = max(2, tile_px // 10)
        draw.rectangle([lx0, ly0, lx1, ly1], outline=color, width=border_w)

        label = _room_overlay_label(room_id, graph)
        label_x = lx0 + max(2, tile_px // 6)
        label_y = ly0 + max(2, tile_px // 6)
        try:
            text_bbox = draw.textbbox((label_x, label_y), label)
            draw.rectangle(text_bbox, fill=(245, 245, 245), outline=color)
        except Exception:
            pass
        draw.text((label_x, label_y), label, fill=(12, 16, 24))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_room_layout_artifacts(
    *,
    dungeon_grid: np.ndarray,
    rooms: Mapping[Any, Any],
    mission_graph: nx.Graph,
    variant_dir: Path,
    tile_px: int = 20,
) -> Dict[str, Any]:
    stitched_layout = build_stitched_room_layout(
        rooms,
        mission_graph,
        fill_tile=0,
    )
    layout_payload = save_room_layout_json(
        mission_graph,
        stitched_layout,
        variant_dir / "room_layout.json",
    )
    base_grid = np.asarray(dungeon_grid, dtype=np.int32)
    if tuple(stitched_layout.dungeon_grid.shape) != tuple(base_grid.shape):
        base_grid = np.asarray(stitched_layout.dungeon_grid, dtype=np.int32)
    save_room_alignment_overlay(
        base_grid,
        mission_graph,
        stitched_layout,
        variant_dir / "dungeon_grid_alignment.png",
        tile_px=tile_px,
        crop_void=True,
    )
    return layout_payload


ROOM_HEIGHT = 16
ROOM_WIDTH = 11


def _resolve_vqvae_checkpoint(run_dir: Path) -> Path:
    direct = run_dir / "checkpoints" / "vqvae" / "vqvae_pretrained.pth"
    if direct.exists():
        return direct

    diffusion_meta = run_dir / "checkpoints" / "diffusion" / "best_model.pth.meta.json"
    if diffusion_meta.exists():
        payload = json.loads(diffusion_meta.read_text(encoding="utf-8"))
        extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
        candidate = extra.get("vqvae_checkpoint") if isinstance(extra, dict) else None
        if candidate:
            path = Path(str(candidate))
            if path.exists():
                return path

    raise FileNotFoundError(
        "Could not resolve a trained VQ-VAE checkpoint for visual audit. "
        f"Checked {direct} and diffusion metadata {diffusion_meta}."
    )


def add_generation_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--semantic-role-prior-strength",
        type=float,
        default=None,
        help="Override generation.semantic_role_prior_strength for this export only.",
    )
    parser.add_argument(
        "--symbolic-max-repair-attempts",
        type=int,
        default=None,
        help="Override generation.symbolic_max_repair_attempts for this export only.",
    )
    parser.add_argument(
        "--symbolic-repair-margin",
        type=int,
        default=None,
        help="Override generation.symbolic_repair_margin for this export only.",
    )
    parser.add_argument(
        "--symbolic-adjacency-threshold",
        type=float,
        default=None,
        help="Override generation.symbolic_adjacency_threshold for this export only.",
    )
    parser.add_argument(
        "--semantic-puzzle-offset",
        type=int,
        default=None,
        help="Override generation.semantic_puzzle_offset for this export only.",
    )
    parser.add_argument(
        "--semantic-constrained-decoding-enabled",
        dest="semantic_constrained_decoding_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override generation.semantic_constrained_decoding_enabled for this export only.",
    )
    parser.add_argument(
        "--semantic-marker-logit-bias",
        type=float,
        default=None,
        help="Override generation.semantic_marker_logit_bias for this export only.",
    )
    parser.add_argument(
        "--semantic-marker-suppression-bias",
        type=float,
        default=None,
        help="Override generation.semantic_marker_suppression_bias for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-scaffold-enabled",
        dest="puzzle_room_scaffold_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override generation.puzzle_room_scaffold_enabled for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-structure-enabled",
        dest="puzzle_room_structure_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override generation.puzzle_room_structure_enabled for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-scaffold-min-structure-tiles",
        type=int,
        default=None,
        help="Override generation.puzzle_room_scaffold_min_structure_tiles for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-archetype-mode",
        type=str,
        choices=("auto", "gate", "serpentine", "hub", "island", "combat"),
        default=None,
        help="Override generation.puzzle_room_archetype_mode for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-branch-density",
        type=float,
        default=None,
        help="Override generation.puzzle_room_branch_density for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-block-budget",
        type=int,
        default=None,
        help="Override generation.puzzle_room_block_budget for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-preserve-route-margin",
        type=int,
        default=None,
        help="Override generation.puzzle_room_preserve_route_margin for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-switch-pocket-depth",
        type=int,
        default=None,
        help="Override generation.puzzle_room_switch_pocket_depth for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-resource-bypass-offset",
        type=int,
        default=None,
        help="Override generation.puzzle_room_resource_bypass_offset for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-key-pocket-depth",
        type=int,
        default=None,
        help="Override generation.puzzle_room_key_pocket_depth for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-item-slot-depth",
        type=int,
        default=None,
        help="Override generation.puzzle_room_item_slot_depth for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-toggle-corridor-offset",
        type=int,
        default=None,
        help="Override generation.puzzle_room_toggle_corridor_offset for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-novelty-enabled",
        dest="puzzle_room_novelty_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override generation.puzzle_room_novelty_enabled for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-candidate-count",
        type=int,
        default=None,
        help="Override generation.puzzle_room_candidate_count for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-novelty-weight",
        type=float,
        default=None,
        help="Override generation.puzzle_room_novelty_weight for this export only.",
    )
    parser.add_argument(
        "--puzzle-room-min-quality-gain",
        type=float,
        default=None,
        help="Override generation.puzzle_room_min_quality_gain for this export only.",
    )
    parser.add_argument(
        "--validator-plan-max-states",
        type=int,
        default=None,
        help="Override generation.validator_plan_max_states for this export only.",
    )
    parser.add_argument(
        "--deterministic-graph-marker-overlay-enabled",
        dest="deterministic_graph_marker_overlay_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override generation.deterministic_graph_marker_overlay_enabled for this export only.",
    )
    parser.add_argument(
        "--fast-sampler-teacher-fallback-enabled",
        dest="fast_sampler_teacher_fallback_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override generation.fast_sampler_teacher_fallback_enabled for this export only.",
    )
    parser.add_argument(
        "--masked-room-teacher-fallback-enabled",
        dest="masked_room_teacher_fallback_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override generation.masked_room_teacher_fallback_enabled for this export only.",
    )


def generation_overrides_from_namespace(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if getattr(args, "semantic_role_prior_strength", None) is not None:
        overrides["semantic_role_prior_strength"] = float(args.semantic_role_prior_strength)
    if getattr(args, "symbolic_max_repair_attempts", None) is not None:
        overrides["symbolic_max_repair_attempts"] = int(args.symbolic_max_repair_attempts)
    if getattr(args, "symbolic_repair_margin", None) is not None:
        overrides["symbolic_repair_margin"] = int(args.symbolic_repair_margin)
    if getattr(args, "symbolic_adjacency_threshold", None) is not None:
        overrides["symbolic_adjacency_threshold"] = float(args.symbolic_adjacency_threshold)
    if getattr(args, "semantic_puzzle_offset", None) is not None:
        overrides["semantic_puzzle_offset"] = int(args.semantic_puzzle_offset)
    if getattr(args, "semantic_constrained_decoding_enabled", None) is not None:
        overrides["semantic_constrained_decoding_enabled"] = bool(args.semantic_constrained_decoding_enabled)
    if getattr(args, "semantic_marker_logit_bias", None) is not None:
        overrides["semantic_marker_logit_bias"] = float(args.semantic_marker_logit_bias)
    if getattr(args, "semantic_marker_suppression_bias", None) is not None:
        overrides["semantic_marker_suppression_bias"] = float(args.semantic_marker_suppression_bias)
    if getattr(args, "puzzle_room_scaffold_enabled", None) is not None:
        overrides["puzzle_room_scaffold_enabled"] = bool(args.puzzle_room_scaffold_enabled)
    if getattr(args, "puzzle_room_structure_enabled", None) is not None:
        overrides["puzzle_room_structure_enabled"] = bool(args.puzzle_room_structure_enabled)
    if getattr(args, "puzzle_room_scaffold_min_structure_tiles", None) is not None:
        overrides["puzzle_room_scaffold_min_structure_tiles"] = int(args.puzzle_room_scaffold_min_structure_tiles)
    if getattr(args, "puzzle_room_archetype_mode", None) is not None:
        overrides["puzzle_room_archetype_mode"] = str(args.puzzle_room_archetype_mode)
    if getattr(args, "puzzle_room_branch_density", None) is not None:
        overrides["puzzle_room_branch_density"] = float(args.puzzle_room_branch_density)
    if getattr(args, "puzzle_room_block_budget", None) is not None:
        overrides["puzzle_room_block_budget"] = int(args.puzzle_room_block_budget)
    if getattr(args, "puzzle_room_preserve_route_margin", None) is not None:
        overrides["puzzle_room_preserve_route_margin"] = int(args.puzzle_room_preserve_route_margin)
    if getattr(args, "puzzle_room_switch_pocket_depth", None) is not None:
        overrides["puzzle_room_switch_pocket_depth"] = int(args.puzzle_room_switch_pocket_depth)
    if getattr(args, "puzzle_room_resource_bypass_offset", None) is not None:
        overrides["puzzle_room_resource_bypass_offset"] = int(args.puzzle_room_resource_bypass_offset)
    if getattr(args, "puzzle_room_key_pocket_depth", None) is not None:
        overrides["puzzle_room_key_pocket_depth"] = int(args.puzzle_room_key_pocket_depth)
    if getattr(args, "puzzle_room_item_slot_depth", None) is not None:
        overrides["puzzle_room_item_slot_depth"] = int(args.puzzle_room_item_slot_depth)
    if getattr(args, "puzzle_room_toggle_corridor_offset", None) is not None:
        overrides["puzzle_room_toggle_corridor_offset"] = int(args.puzzle_room_toggle_corridor_offset)
    if getattr(args, "puzzle_room_novelty_enabled", None) is not None:
        overrides["puzzle_room_novelty_enabled"] = bool(args.puzzle_room_novelty_enabled)
    if getattr(args, "puzzle_room_candidate_count", None) is not None:
        overrides["puzzle_room_candidate_count"] = int(args.puzzle_room_candidate_count)
    if getattr(args, "puzzle_room_novelty_weight", None) is not None:
        overrides["puzzle_room_novelty_weight"] = float(args.puzzle_room_novelty_weight)
    if getattr(args, "puzzle_room_min_quality_gain", None) is not None:
        overrides["puzzle_room_min_quality_gain"] = float(args.puzzle_room_min_quality_gain)
    if getattr(args, "validator_plan_max_states", None) is not None:
        overrides["validator_plan_max_states"] = int(args.validator_plan_max_states)
    if getattr(args, "deterministic_graph_marker_overlay_enabled", None) is not None:
        overrides["deterministic_graph_marker_overlay_enabled"] = bool(
            args.deterministic_graph_marker_overlay_enabled
        )
    if getattr(args, "fast_sampler_teacher_fallback_enabled", None) is not None:
        overrides["fast_sampler_teacher_fallback_enabled"] = bool(args.fast_sampler_teacher_fallback_enabled)
    if getattr(args, "masked_room_teacher_fallback_enabled", None) is not None:
        overrides["masked_room_teacher_fallback_enabled"] = bool(args.masked_room_teacher_fallback_enabled)
    return overrides


def _apply_generation_overrides(
    resolved: Mapping[str, Any],
    generation_overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    updated = copy.deepcopy(dict(resolved))
    if not generation_overrides:
        return updated
    generation = updated.setdefault("generation", {})
    for key, value in generation_overrides.items():
        generation[str(key)] = value
    return updated


def _generation_policy_summary(pipeline: NeuralSymbolicDungeonPipeline) -> Dict[str, Any]:
    return {
        "topology_anchor_policy_version": str(
            getattr(pipeline, "topology_anchor_policy_version", "unknown")
        ),
        "semantic_role_prior_strength": float(
            getattr(pipeline, "default_semantic_role_prior_strength", 0.15)
        ),
        "symbolic_max_repair_attempts": int(
            getattr(pipeline, "symbolic_max_repair_attempts", 5)
        ),
        "symbolic_repair_margin": int(
            getattr(pipeline, "symbolic_repair_margin", 2)
        ),
        "symbolic_adjacency_threshold": float(
            getattr(pipeline, "symbolic_adjacency_threshold", 0.01)
        ),
        "semantic_anchor_threshold": float(
            getattr(pipeline, "default_semantic_anchor_threshold", 0.5)
        ),
        "semantic_puzzle_offset": int(
            getattr(pipeline, "default_semantic_puzzle_offset", 2)
        ),
        "semantic_constrained_decoding_enabled": bool(
            getattr(pipeline, "default_semantic_constrained_decoding_enabled", True)
        ),
        "semantic_marker_logit_bias": float(
            getattr(pipeline, "default_semantic_marker_logit_bias", 10000.0)
        ),
        "semantic_marker_suppression_bias": float(
            getattr(pipeline, "default_semantic_marker_suppression_bias", 100.0)
        ),
        "puzzle_room_scaffold_enabled": bool(
            getattr(pipeline, "default_puzzle_room_scaffold_enabled", True)
        ),
        "puzzle_room_structure_enabled": bool(
            getattr(pipeline, "default_puzzle_room_structure_enabled", True)
        ),
        "puzzle_room_scaffold_min_structure_tiles": int(
            getattr(pipeline, "default_puzzle_room_scaffold_min_structure_tiles", 10)
        ),
        "puzzle_room_archetype_mode": str(
            getattr(pipeline, "default_puzzle_room_archetype_mode", "auto")
        ),
        "puzzle_room_branch_density": float(
            getattr(pipeline, "default_puzzle_room_branch_density", 0.75)
        ),
        "puzzle_room_block_budget": int(
            getattr(pipeline, "default_puzzle_room_block_budget", 28)
        ),
        "puzzle_room_preserve_route_margin": int(
            getattr(pipeline, "default_puzzle_room_preserve_route_margin", 0)
        ),
        "puzzle_room_switch_pocket_depth": int(
            getattr(pipeline, "default_puzzle_room_switch_pocket_depth", 3)
        ),
        "puzzle_room_resource_bypass_offset": int(
            getattr(pipeline, "default_puzzle_room_resource_bypass_offset", 2)
        ),
        "puzzle_room_key_pocket_depth": int(
            getattr(pipeline, "default_puzzle_room_key_pocket_depth", 3)
        ),
        "puzzle_room_item_slot_depth": int(
            getattr(pipeline, "default_puzzle_room_item_slot_depth", 3)
        ),
        "puzzle_room_toggle_corridor_offset": int(
            getattr(pipeline, "default_puzzle_room_toggle_corridor_offset", 2)
        ),
        "puzzle_room_novelty_enabled": bool(
            getattr(pipeline, "default_puzzle_room_novelty_enabled", True)
        ),
        "puzzle_room_candidate_count": int(
            getattr(pipeline, "default_puzzle_room_candidate_count", 4)
        ),
        "puzzle_room_novelty_weight": float(
            getattr(pipeline, "default_puzzle_room_novelty_weight", 0.45)
        ),
        "puzzle_room_min_quality_gain": float(
            getattr(pipeline, "default_puzzle_room_min_quality_gain", 0.5)
        ),
        "validator_plan_max_states": int(
            getattr(pipeline, "default_validator_plan_max_states", 512)
        ),
        "deterministic_graph_marker_overlay_enabled": bool(
            getattr(pipeline, "default_deterministic_graph_marker_overlay_enabled", True)
        ),
        "fast_sampler_teacher_fallback_enabled": bool(
            getattr(pipeline, "default_fast_sampler_teacher_fallback_enabled", True)
        ),
        "masked_room_teacher_fallback_enabled": bool(
            getattr(pipeline, "default_masked_room_teacher_fallback_enabled", True)
        ),
        "masked_room_sampling_temperature": float(
            getattr(pipeline, "default_masked_room_sampling_temperature", 1.0)
        ),
        "masked_room_sampling_schedule": str(
            getattr(pipeline, "default_masked_room_sampling_schedule", "cosine")
        ),
        "masked_room_sampling_stochastic": bool(
            getattr(pipeline, "default_masked_room_sampling_stochastic", True)
        ),
        "masked_room_corrector_steps": int(
            getattr(pipeline, "default_masked_room_corrector_steps", 1)
        ),
        "masked_room_corrector_mask_ratio": float(
            getattr(pipeline, "default_masked_room_corrector_mask_ratio", 0.1)
        ),
    }


def build_pipeline(
    run_dir: Path,
    *,
    generation_overrides: Optional[Mapping[str, Any]] = None,
    device_override: Optional[str] = None,
) -> NeuralSymbolicDungeonPipeline:
    resolved = _load_resolved_config(run_dir)
    resolved = _apply_generation_overrides(resolved, generation_overrides)
    export_device = str(device_override).strip().lower() if device_override else _resolve_export_device(resolved)
    pipeline_kwargs = pipeline_kwargs_from_resolved_config(resolved)
    vqvae_checkpoint = _resolve_vqvae_checkpoint(run_dir)

    fast_reselected = run_dir / "checkpoints" / "fast_sampler" / "fast_sampler_best_reselected.pth"
    fast_best = run_dir / "checkpoints" / "fast_sampler" / "fast_sampler_best.pth"
    fast_final = run_dir / "checkpoints" / "fast_sampler" / "fast_sampler_final.pth"
    fast_checkpoint = (
        fast_reselected
        if fast_reselected.exists()
        else (fast_best if fast_best.exists() else fast_final)
    )

    pipeline_kwargs.update(
        {
            "room_generator_mode": "latent_diffusion",
            "fast_sampling_checkpoint": str(fast_checkpoint) if fast_checkpoint.exists() else None,
            "fast_sampling_steps": int(resolved["fast_sampler"]["num_inference_steps"]),
        }
    )

    return NeuralSymbolicDungeonPipeline(
        vqvae_checkpoint=str(vqvae_checkpoint),
        diffusion_checkpoint=str(run_dir / "checkpoints" / "diffusion" / "best_model.pth"),
        condition_encoder_checkpoint=str(run_dir / "checkpoints" / "diffusion" / "best_model.pth"),
        logic_net_checkpoint=str(run_dir / "checkpoints" / "diffusion" / "best_model.pth"),
        device=export_device,
        enable_logging=False,
        **pipeline_kwargs,
    )


def _generate_dungeon_with_oom_backoff(
    *,
    pipeline_builder: Any,
    run_dir: Path,
    mission_graph: nx.Graph,
    generation_overrides: Optional[Mapping[str, Any]],
    execution_kwargs: Mapping[str, Any],
    status_writer: Any,
    generation_kwargs: Mapping[str, Any],
) -> Tuple[NeuralSymbolicDungeonPipeline, Any, Dict[str, Any]]:
    last_exc: Optional[BaseException] = None
    retry_plan = _build_generation_retry_plan(execution_kwargs)
    for attempt_index, attempt in enumerate(retry_plan, start=1):
        device_override = attempt.get("device_override")
        attempt_name = str(attempt.get("name", f"attempt_{attempt_index}"))
        attempt_kwargs = _normalized_execution_kwargs(attempt.get("execution_kwargs", {}))
        pipeline = None
        try:
            status_writer(
                "building_pipeline",
                attempt=int(attempt_index),
                attempt_name=attempt_name,
                device=str(device_override or "configured"),
                execution_kwargs=attempt_kwargs,
            )
            pipeline = pipeline_builder(
                run_dir,
                generation_overrides=generation_overrides,
                device_override=device_override,
            )
            pipeline.runtime_diagnostics = {}
            status_writer(
                "generating_dungeon",
                attempt=int(attempt_index),
                attempt_name=attempt_name,
                device=str(device_override or "configured"),
                execution_kwargs=attempt_kwargs,
            )
            result = pipeline.generate_dungeon(
                mission_graph=copy.deepcopy(mission_graph),
                **dict(generation_kwargs),
                **attempt_kwargs,
            )
            return pipeline, result, {
                "attempt": int(attempt_index),
                "attempt_name": attempt_name,
                "device": str(device_override or "configured"),
                "execution_kwargs": attempt_kwargs,
                "oom_retry_count": int(attempt_index - 1),
            }
        except (RuntimeError, ValueError) as exc:
            if not _is_cuda_oom_error(exc):
                raise
            last_exc = exc
            status_writer(
                "generation_retry_after_oom",
                attempt=int(attempt_index),
                attempt_name=attempt_name,
                device=str(device_override or "configured"),
                execution_kwargs=attempt_kwargs,
                error=str(exc),
            )
            if pipeline is not None:
                try:
                    del pipeline
                except Exception:
                    pass
            _release_torch_memory()
            continue
    if last_exc is not None:
        raise RuntimeError(
            "Dungeon generation failed after exhausting the CUDA/CPU OOM retry ladder."
        ) from last_exc
    raise RuntimeError("Dungeon generation failed before any retry attempt was executed.")


def export_variant(
    *,
    run_dir: Path,
    mission_graph: nx.Graph,
    variant_name: str,
    out_dir: Path,
    guidance_scale: float,
    logic_guidance_scale: float,
    num_diffusion_steps: int,
    use_fast_sampling: bool,
    seed: int,
    generation_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    variant_dir = out_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    status_path = variant_dir / "export_status.json"

    def _write_status(stage: str, **extra: Any) -> None:
        payload = {
            "variant_name": str(variant_name),
            "stage": str(stage),
            "seed": int(seed),
            "guidance_scale": float(guidance_scale),
            "logic_guidance_scale": float(logic_guidance_scale),
            "num_diffusion_steps": int(num_diffusion_steps),
            "use_fast_sampling": bool(use_fast_sampling),
        }
        payload.update({str(k): v for k, v in extra.items()})
        status_path.write_text(json.dumps(_json_sanitize(payload), indent=2), encoding="utf-8")

    execution_kwargs = _resolve_export_execution_kwargs()
    pipeline, result, generation_execution = _generate_dungeon_with_oom_backoff(
        pipeline_builder=build_pipeline,
        run_dir=run_dir,
        mission_graph=mission_graph,
        generation_overrides=generation_overrides,
        execution_kwargs=execution_kwargs,
        status_writer=_write_status,
        generation_kwargs={
            "generate_topology": False,
            "guidance_scale": float(guidance_scale),
            "logic_guidance_scale": float(logic_guidance_scale),
            "num_diffusion_steps": int(num_diffusion_steps),
            "use_fast_sampling": bool(use_fast_sampling),
            "apply_repair": True,
            "enable_map_elites": False,
            "seed": int(seed),
        },
    )
    _write_status(
        "generation_complete",
        generation_time_sec=float(result.generation_time),
        room_count=int(len(result.rooms)),
    )
    rooms_dir = variant_dir / "rooms"
    room_texts: Dict[int, str] = {}
    room_hashes: Dict[str, str] = {}
    room_grids: Dict[int, np.ndarray] = {}

    for room_id, room in sorted(result.rooms.items(), key=lambda kv: int(kv[0])):
        grid = np.asarray(room.room_grid, dtype=np.int32)
        room_grids[int(room_id)] = grid
        save_grid_png(grid, rooms_dir / f"room_{room_id}.png", tile_px=20)
        save_stylized_grid_png(grid, rooms_dir / f"room_{room_id}_stylized.png", tile_px=20, crop_void=False)
        room_text = save_grid_txt(grid, rooms_dir / f"room_{room_id}.txt")
        room_texts[int(room_id)] = room_text
        room_hashes[str(room_id)] = hashlib.sha256(room_text.encode("utf-8")).hexdigest()[:16]

    save_grid_png(
        np.asarray(result.dungeon_grid, dtype=np.int32),
        variant_dir / "dungeon_grid.png",
        tile_px=16,
        crop_void=False,
    )
    save_grid_json(
        np.asarray(result.dungeon_grid, dtype=np.int32),
        variant_dir / "dungeon_grid_ids.json",
    )
    dungeon_preview = save_grid_txt(np.asarray(result.dungeon_grid, dtype=np.int32), variant_dir / "dungeon_grid.txt")
    (variant_dir / "preview.txt").write_text(dungeon_preview, encoding="utf-8")
    save_char_grid_png_from_txt(
        variant_dir / "dungeon_grid.txt",
        variant_dir / "dungeon_grid_readable.png",
        tile_px=16,
        crop_void=True,
    )
    save_rooms_sheet(room_grids, variant_dir / "rooms_sheet.png", tile_px=16, columns=4)
    save_stylized_grid_png(
        np.asarray(result.dungeon_grid, dtype=np.int32),
        variant_dir / "dungeon_grid_stylized.png",
        tile_px=20,
        crop_void=True,
    )
    save_stylized_rooms_sheet(room_grids, variant_dir / "rooms_sheet_stylized.png", tile_px=18, columns=4)
    layout_payload = write_room_layout_artifacts(
        dungeon_grid=np.asarray(result.dungeon_grid, dtype=np.int32),
        rooms=result.rooms,
        mission_graph=mission_graph,
        variant_dir=variant_dir,
        tile_px=20,
    )

    cleanup_totals = {
        key: int(
            sum(
                int(room.metrics.get(key, 0))
                for room in result.rooms.values()
            )
        )
        for key in (
            "neural_invalid_door_tiles_removed",
            "neural_interior_obstacle_tiles_removed",
            "neural_interior_obstacle_components_removed",
            "repair_invalid_door_tiles_removed",
            "repair_interior_obstacle_tiles_removed",
            "repair_interior_obstacle_components_removed",
        )
    }
    dungeon_grid = np.asarray(result.dungeon_grid, dtype=np.int32).copy()
    result_metrics = dict(result.metrics)
    generation_time_sec = float(result.generation_time)
    room_count = int(len(result.rooms))
    validation_context = build_validation_context_from_generation_result(result)
    runtime_diagnostics = dict(pipeline.runtime_diagnostics)
    topology_anchor_policy = _generation_policy_summary(pipeline)
    diffusion_inference_checkpoint_state_key = str(
        getattr(pipeline.diffusion, "inference_checkpoint_state_key", "unknown")
    )
    diffusion_training_cfg_scale = float(
        getattr(pipeline.diffusion, "training_cfg_scale", float("nan"))
    )
    tile_hist = {str(int(k)): int(v) for k, v in Counter(int(v) for v in dungeon_grid.ravel()).items()}
    reference_room_texts = load_reference_room_texts(
        str(_resolve_dataset_data_root(run_dir)),
        max_rooms=DEFAULT_REFERENCE_ROOM_LIMIT,
    )
    end_to_end_evaluation = compute_end_to_end_structural_metrics(
        room_texts=room_texts,
        dungeon_text=dungeon_preview,
        reference_room_texts=reference_room_texts,
    )

    _write_status(
        "preparing_validation",
        generation_time_sec=generation_time_sec,
        room_count=room_count,
    )
    del room_texts
    del room_grids
    del pipeline
    del result
    _release_torch_memory()
    validation = _compute_generation_validation(
        dungeon_grid=dungeon_grid,
        mission_graph=mission_graph,
        **validation_context,
    )
    _write_status(
        "validation_complete",
        generation_time_sec=generation_time_sec,
        room_count=room_count,
    )

    summary = {
        "name": variant_name,
        "guidance_scale": float(guidance_scale),
        "logic_guidance_scale": float(logic_guidance_scale),
        "num_diffusion_steps": int(num_diffusion_steps),
        "use_fast_sampling": bool(use_fast_sampling),
        "generation_overrides_applied": dict(generation_overrides or {}),
        "diffusion_inference_checkpoint_state_key": diffusion_inference_checkpoint_state_key,
        "diffusion_training_cfg_scale": diffusion_training_cfg_scale,
        "generation_execution": generation_execution,
        "metrics": {
            **result_metrics,
            "generation_time_sec": generation_time_sec,
        },
        "runtime_diagnostics": runtime_diagnostics,
        "topology_anchor_policy": topology_anchor_policy,
        "semantic_metrics": {
            key: result_metrics.get(key)
            for key in (
                "total_graph_marker_expected",
                "total_graph_marker_overwrites",
                "avg_neural_graph_marker_exact_match_rate",
                "avg_final_pre_overlay_graph_marker_exact_match_rate",
                "avg_final_post_overlay_graph_marker_exact_match_rate",
                "avg_final_graph_marker_overwrite_rate",
                "avg_neural_semantic_anchor_error",
                "avg_final_pre_overlay_semantic_anchor_error",
                "avg_final_post_overlay_semantic_anchor_error",
            )
        },
        "cleanup_totals": cleanup_totals,
        "tile_hist": tile_hist,
        "room_hashes": room_hashes,
        "end_to_end_evaluation": end_to_end_evaluation,
        "layout": {
            "room_count": int(layout_payload.get("room_count", 0)),
            "primary_quality_metric_name": layout_payload.get("primary_quality_metric_name"),
            "primary_quality_metric_value": layout_payload.get("primary_quality_metric_value"),
            **dict(layout_payload.get("layout_quality", {})),
        },
        "validation": validation,
    }
    summary = _json_sanitize(summary)
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (variant_dir / "validation_search_stats.json").write_text(
        json.dumps(_json_sanitize(build_validation_search_stats_payload(summary.get("validation", {}))), indent=2),
        encoding="utf-8",
    )
    _write_status(
        "complete",
        generation_time_sec=generation_time_sec,
        room_count=room_count,
        summary_path=str(variant_dir / "summary.json"),
    )
    _release_torch_memory()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-topology visual audit for diffusion / fast-sampler.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/zelda_hmolqd_fulltrain_rerun"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/zelda_hmolqd_fulltrain_rerun/fast_sampler_audit_post_fix"),
    )
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--num-rooms", type=int, default=8)
    parser.add_argument("--topology-population", type=int, default=50)
    parser.add_argument("--topology-generations", type=int, default=100)
    add_generation_override_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_overrides = generation_overrides_from_namespace(args)
    pipeline = build_pipeline(args.run_dir, generation_overrides=generation_overrides)

    prepared = pipeline.prepare_dungeon_generation(
        mission_graph=None,
        generate_topology=True,
        num_rooms=int(args.num_rooms),
        population_size=int(args.topology_population),
        generations=int(args.topology_generations),
        seed=int(args.seed),
    )
    mission_graph = copy.deepcopy(prepared.mission_graph)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mission_graph.json").write_text(
        json.dumps(_json_sanitize(json_graph.node_link_data(mission_graph, edges="links")), indent=2),
        encoding="utf-8",
    )

    variants = [
        {
            "name": "diffusion_cfg75_logic1_steps50",
            "guidance_scale": 7.5,
            "logic_guidance_scale": 1.0,
            "num_diffusion_steps": 50,
            "use_fast_sampling": False,
        },
        {
            "name": "diffusion_cfg3_logic0_steps50",
            "guidance_scale": 3.0,
            "logic_guidance_scale": 0.0,
            "num_diffusion_steps": 50,
            "use_fast_sampling": False,
        },
        {
            "name": "fast_cfg75_logic1_steps4",
            "guidance_scale": 7.5,
            "logic_guidance_scale": 1.0,
            "num_diffusion_steps": 4,
            "use_fast_sampling": True,
        },
        {
            "name": "fast_cfg3_logic0_steps4",
            "guidance_scale": 3.0,
            "logic_guidance_scale": 0.0,
            "num_diffusion_steps": 4,
            "use_fast_sampling": True,
        },
    ]

    summaries: List[Dict[str, Any]] = []
    for variant in variants:
        summaries.append(
            export_variant(
                run_dir=args.run_dir,
                mission_graph=mission_graph,
                variant_name=str(variant["name"]),
                out_dir=args.output_dir,
                guidance_scale=float(variant["guidance_scale"]),
                logic_guidance_scale=float(variant["logic_guidance_scale"]),
                num_diffusion_steps=int(variant["num_diffusion_steps"]),
                use_fast_sampling=bool(variant["use_fast_sampling"]),
                seed=int(args.seed),
                generation_overrides=generation_overrides,
            )
        )

    by_name = {entry["name"]: entry for entry in summaries}
    post = {
        "summaries": summaries,
        "fast_variants_identical": (
            by_name["fast_cfg75_logic1_steps4"]["room_hashes"]
            == by_name["fast_cfg3_logic0_steps4"]["room_hashes"]
        ),
        "diffusion_variants_identical": (
            by_name["diffusion_cfg75_logic1_steps50"]["room_hashes"]
            == by_name["diffusion_cfg3_logic0_steps50"]["room_hashes"]
        ),
    }
    post = _json_sanitize(post)
    (args.output_dir / "summary.json").write_text(json.dumps(post, indent=2), encoding="utf-8")
    print(json.dumps(_json_sanitize({"output": str(args.output_dir / "summary.json")}), indent=2))


if __name__ == "__main__":
    main()
