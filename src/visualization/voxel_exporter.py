"""
Semantic voxel exporter for engine-agnostic 3D level interchange.

Supports:
- ASCII tile maps (W, ., ~)
- Numeric semantic grids
- Export to Wavefront OBJ
- Export to JSON for Unity/Godot import pipelines
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Voxel:
    x: float
    y: float
    z: float
    kind: str


def load_ascii_grid(path: Path) -> List[str]:
    rows = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.strip()]
    if not rows:
        raise ValueError("Input ASCII grid is empty")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("All rows in ASCII grid must have same width")
    return rows


def load_json_grid(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "grid" in payload:
        payload = payload["grid"]
    arr = np.asarray(payload)
    if arr.ndim != 2:
        raise ValueError("JSON grid must be 2D")
    return arr


def semantic_to_voxels(
    grid: Sequence[Sequence[object]],
    *,
    tile_size: float = 1.0,
) -> List[Voxel]:
    voxels: List[Voxel] = []

    # Character + numeric semantic mapping.
    wall_tokens = {"W", "#", "2"}
    floor_tokens = {".", "F", "1"}
    water_tokens = {"~", "20"}

    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    for r in range(h):
        for c in range(w):
            raw = grid[r][c]
            token = str(raw).strip().upper()
            x = float(c) * tile_size
            z = float(r) * tile_size

            if token in wall_tokens:
                for y_idx in (1, 2, 3):
                    voxels.append(Voxel(x=x, y=float(y_idx) * tile_size, z=z, kind="wall"))
            elif token in floor_tokens:
                voxels.append(Voxel(x=x, y=0.0, z=z, kind="floor"))
            elif token in water_tokens:
                voxels.append(Voxel(x=x, y=0.5 * tile_size, z=z, kind="water"))
            else:
                # Unknown token falls back to floor for robust interchange.
                voxels.append(Voxel(x=x, y=0.0, z=z, kind="floor"))

    return voxels


def _cube_vertices(x: float, y: float, z: float, size: float) -> List[Tuple[float, float, float]]:
    return [
        (x, y, z),
        (x + size, y, z),
        (x + size, y + size, z),
        (x, y + size, z),
        (x, y, z + size),
        (x + size, y, z + size),
        (x + size, y + size, z + size),
        (x, y + size, z + size),
    ]


def export_obj(voxels: Sequence[Voxel], output_path: Path, cube_size: float = 1.0) -> None:
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int, int]] = []

    for v in voxels:
        base_idx = len(vertices) + 1
        vertices.extend(_cube_vertices(v.x, v.y, v.z, cube_size))
        faces.extend(
            [
                (base_idx + 0, base_idx + 1, base_idx + 2, base_idx + 3),
                (base_idx + 4, base_idx + 5, base_idx + 6, base_idx + 7),
                (base_idx + 0, base_idx + 1, base_idx + 5, base_idx + 4),
                (base_idx + 1, base_idx + 2, base_idx + 6, base_idx + 5),
                (base_idx + 2, base_idx + 3, base_idx + 7, base_idx + 6),
                (base_idx + 3, base_idx + 0, base_idx + 4, base_idx + 7),
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Semantic voxel OBJ export\n")
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.5f} {vy:.5f} {vz:.5f}\n")
        for a, b, c, d in faces:
            f.write(f"f {a} {b} {c} {d}\n")


def export_engine_json(voxels: Sequence[Voxel], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "kltn-semantic-voxel-v1",
        "voxels": [
            {"x": v.x, "y": v.y, "z": v.z, "kind": v.kind}
            for v in voxels
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _coerce_grid_to_python_2d(arr: np.ndarray) -> List[List[object]]:
    return [[arr[r, c].item() if hasattr(arr[r, c], "item") else arr[r, c] for c in range(arr.shape[1])] for r in range(arr.shape[0])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export semantic dungeon grids to 3D voxel assets")
    parser.add_argument("--input-txt", type=str, default=None, help="Path to ASCII grid (.txt)")
    parser.add_argument("--input-json", type=str, default=None, help="Path to JSON grid (2D array or {grid: ...})")
    parser.add_argument("--obj-out", type=str, default=None, help="Output OBJ path")
    parser.add_argument("--json-out", type=str, default=None, help="Output engine JSON path")
    parser.add_argument("--tile-size", type=float, default=1.0, help="Tile world size")
    args = parser.parse_args()

    if not args.input_txt and not args.input_json:
        raise ValueError("Provide --input-txt or --input-json")
    if not args.obj_out and not args.json_out:
        raise ValueError("Provide --obj-out or --json-out (or both)")

    if args.input_txt:
        grid_2d = [list(row) for row in load_ascii_grid(Path(args.input_txt))]
    else:
        arr = load_json_grid(Path(args.input_json))
        grid_2d = _coerce_grid_to_python_2d(arr)

    voxels = semantic_to_voxels(grid_2d, tile_size=float(args.tile_size))

    if args.obj_out:
        export_obj(voxels, Path(args.obj_out), cube_size=float(args.tile_size))
    if args.json_out:
        export_engine_json(voxels, Path(args.json_out))


if __name__ == "__main__":
    main()
