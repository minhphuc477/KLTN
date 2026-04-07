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
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from networkx.readwrite import json_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.definitions import semantic_to_vglc_char
from src.pipeline.dungeon_pipeline import (
    NeuralSymbolicDungeonPipeline,
    pipeline_kwargs_from_resolved_config,
)
from src.pipeline.room_stitching import StitchedRoomLayout, build_stitched_room_layout


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


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
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=clamp_rgb(base, -30), width=max(1, tile_px // 10))
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
    elif int(tile) == 21:  # start
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=(210, 255, 210), width=max(1, tile_px // 10))
    elif int(tile) == 22:  # triforce
        draw.polygon([(cx, y0 + inset), (x0 + inset, y1 - inset), (x1 - inset, y1 - inset)], fill=(255, 235, 130), outline=(180, 140, 20))
    elif int(tile) == 23:  # boss
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], fill=(160, 30, 30))
    elif int(tile) == 31:  # boss key
        draw.ellipse([x0 + inset, y0 + inset, x0 + inset + tile_px // 3, y0 + inset + tile_px // 3], outline=(255, 225, 140), width=max(1, tile_px // 12))
        draw.rectangle([cx - 1, cy - 1, x1 - inset, cy + 1], fill=(255, 190, 100))
    elif int(tile) == 32:  # key item
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=(180, 245, 255), width=max(1, tile_px // 10))
        draw.line([x0 + inset, y1 - inset, x1 - inset, y0 + inset], fill=(180, 245, 255), width=max(1, tile_px // 12))
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
        pos = graph.nodes[node_id].get("pos")
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

    entries: List[Dict[str, Any]] = []
    slot_match_count = 0
    slot_comparable_count = 0
    for room_id in sorted(stitched_layout.layout_map.keys(), key=_room_sort_key):
        bbox = stitched_layout.layout_map[room_id]
        x_min, y_min, x_max, y_max = bbox
        attrs = graph.nodes.get(room_id, {}) if room_id in graph else {}
        graph_pos = normalized_graph_positions.get(room_id)
        slot_pos = stitched_layout.slot_positions.get(room_id)
        slot_matches_graph_pos = bool(graph_pos is not None and slot_pos == graph_pos)
        if graph_pos is not None:
            slot_comparable_count += 1
            if slot_matches_graph_pos:
                slot_match_count += 1
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
        "graph_slot_match_rate": (
            float(slot_match_count / max(1, slot_comparable_count))
            if slot_comparable_count > 0
            else None
        ),
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
        "deterministic_graph_marker_overlay_enabled": bool(
            getattr(pipeline, "default_deterministic_graph_marker_overlay_enabled", True)
        ),
        "fast_sampler_teacher_fallback_enabled": bool(
            getattr(pipeline, "default_fast_sampler_teacher_fallback_enabled", True)
        ),
        "masked_room_teacher_fallback_enabled": bool(
            getattr(pipeline, "default_masked_room_teacher_fallback_enabled", True)
        ),
    }


def build_pipeline(
    run_dir: Path,
    *,
    generation_overrides: Optional[Mapping[str, Any]] = None,
) -> NeuralSymbolicDungeonPipeline:
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    resolved = _apply_generation_overrides(resolved, generation_overrides)
    pipeline_kwargs = pipeline_kwargs_from_resolved_config(resolved)
    vqvae_checkpoint = _resolve_vqvae_checkpoint(run_dir)

    fast_best = run_dir / "checkpoints" / "fast_sampler" / "fast_sampler_best.pth"
    fast_final = run_dir / "checkpoints" / "fast_sampler" / "fast_sampler_final.pth"
    fast_checkpoint = fast_best if fast_best.exists() else fast_final

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
        device="auto",
        enable_logging=False,
        **pipeline_kwargs,
    )


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
    pipeline = build_pipeline(run_dir, generation_overrides=generation_overrides)
    pipeline.runtime_diagnostics = {}
    result = pipeline.generate_dungeon(
        mission_graph=copy.deepcopy(mission_graph),
        generate_topology=False,
        guidance_scale=float(guidance_scale),
        logic_guidance_scale=float(logic_guidance_scale),
        num_diffusion_steps=int(num_diffusion_steps),
        use_fast_sampling=bool(use_fast_sampling),
        apply_repair=True,
        enable_map_elites=False,
        seed=int(seed),
    )

    variant_dir = out_dir / variant_name
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
    tile_hist = {str(int(k)): int(v) for k, v in Counter(int(v) for v in np.asarray(result.dungeon_grid).ravel()).items()}

    summary = {
        "name": variant_name,
        "guidance_scale": float(guidance_scale),
        "logic_guidance_scale": float(logic_guidance_scale),
        "num_diffusion_steps": int(num_diffusion_steps),
        "use_fast_sampling": bool(use_fast_sampling),
        "diffusion_inference_checkpoint_state_key": str(
            getattr(pipeline.diffusion, "inference_checkpoint_state_key", "unknown")
        ),
        "diffusion_training_cfg_scale": float(
            getattr(pipeline.diffusion, "training_cfg_scale", float("nan"))
        ),
        "metrics": {
            **dict(result.metrics),
            "generation_time_sec": float(result.generation_time),
        },
        "runtime_diagnostics": dict(pipeline.runtime_diagnostics),
        "topology_anchor_policy": _generation_policy_summary(pipeline),
        "semantic_metrics": {
            key: dict(result.metrics).get(key)
            for key in (
                "total_graph_marker_expected",
                "total_graph_marker_overwrites",
                "avg_neural_graph_marker_exact_match_rate",
                "avg_final_pre_overlay_graph_marker_exact_match_rate",
                "avg_final_post_overlay_graph_marker_exact_match_rate",
                "avg_final_graph_marker_overwrite_rate",
                "avg_neural_semantic_anchor_error",
                "avg_final_pre_overlay_semantic_anchor_error",
            )
        },
        "cleanup_totals": cleanup_totals,
        "tile_hist": tile_hist,
        "room_hashes": room_hashes,
        "layout": {
            "room_count": int(layout_payload.get("room_count", 0)),
            "graph_slot_match_rate": layout_payload.get("graph_slot_match_rate"),
        },
    }
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
        json.dumps(json_graph.node_link_data(mission_graph, edges="links"), indent=2),
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
    (args.output_dir / "summary.json").write_text(json.dumps(post, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
