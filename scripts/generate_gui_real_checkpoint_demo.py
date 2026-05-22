"""Generate a GUI-loadable demo dungeon from the real AI checkpoint.

This exporter intentionally avoids the older all-entity route overlay. It uses
the GUI checkpoint discovery path, runs the canonical neural-symbolic pipeline,
normalizes the result only for solver validation, and writes a numeric TXT level
that `gui_runner.py --load-txt` can import directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.definitions import ID_TO_NAME, ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE as SP
from src.evaluation.pcbs_validation import prepare_dungeon_grid_for_validation
from src.gui.ai.generation_pipeline import (
    discover_best_output_checkpoint,
    generate_comprehensive_demo_graph,
    generate_dungeon_with_pipeline,
    generate_mission_graph,
    load_canonical_generation_pipeline,
    resolve_checkpoint_path,
)
from src.gui.components.constants import GUI_ALGORITHM_NAMES
from src.gui.solver.process_worker import solve_in_subprocess


LOGGER = logging.getLogger("gui_real_checkpoint_demo")


def _parse_algorithms(raw: str) -> Tuple[int, ...]:
    values: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return tuple(dict.fromkeys(values))


def _entity_counts(grid: np.ndarray) -> Dict[str, int]:
    counts = Counter(int(v) for v in np.asarray(grid, dtype=np.int32).ravel())
    return {
        str(ID_TO_NAME.get(tile_id, tile_id)): int(count)
        for tile_id, count in sorted(counts.items(), key=lambda kv: kv[0])
    }


def _room_count_from_path(path: Sequence[Sequence[int]]) -> int:
    rooms = {
        (int(point[0]) // ROOM_HEIGHT, int(point[1]) // ROOM_WIDTH)
        for point in path
        if len(point) >= 2
    }
    return int(len(rooms))


def _validate_algorithms(
    grid: np.ndarray,
    *,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    algorithms: Iterable[int],
    timeout_states: int,
) -> Dict[str, Any]:
    per_algorithm: Dict[str, Any] = {}
    best_route: Optional[Dict[str, Any]] = None
    all_ok = True
    any_ok = False
    for alg in algorithms:
        alg = int(alg)
        alg_name = GUI_ALGORITHM_NAMES[alg] if 0 <= alg < len(GUI_ALGORITHM_NAMES) else f"Algorithm {alg}"
        started = time.perf_counter()
        result = solve_in_subprocess(
            grid,
            start,
            goal,
            alg,
            {"strict_original_mode": False, "diagonal_movement": False},
            {
                "timeout": int(timeout_states),
                "max_depth": int(timeout_states),
                "representation": "tile",
                "allow_diagonals": False,
                "rules_profile": "vglc_strict",
            },
        )
        elapsed = time.perf_counter() - started
        path = [list(map(int, point)) for point in (result.get("path") or [])]
        success = bool(result.get("success")) and bool(path)
        all_ok = all_ok and success
        any_ok = any_ok or success
        solver_result = dict(result.get("solver_result") or {})
        payload = {
            "name": alg_name,
            "success": bool(success),
            "path_length": int(len(path)),
            "visited_rooms": _room_count_from_path(path),
            "time_sec": round(float(elapsed), 4),
            "states_explored": int(
                solver_result.get("states_explored", solver_result.get("nodes", 0)) or 0
            ),
            "solver_algorithm": solver_result.get("algorithm"),
            "message": result.get("message"),
        }
        per_algorithm[str(alg)] = payload
        if success and (best_route is None or len(path) > int(best_route["path_length"])):
            best_route = {
                "algorithm_idx": alg,
                "algorithm": alg_name,
                "path": path,
                **payload,
            }
        LOGGER.info(
            "%s: success=%s path=%d visited_rooms=%d time=%.2fs",
            alg_name,
            success,
            len(path),
            payload["visited_rooms"],
            elapsed,
        )

    return {
        "success": bool(any_ok),
        "all_success": bool(all_ok),
        "start": list(map(int, start)),
        "goal": list(map(int, goal)),
        "algorithms": per_algorithm,
        "best_route": best_route,
    }


def _build_mission_data(mode: str, seed: int, rooms: int) -> Dict[str, Any]:
    if mode == "random":
        return generate_mission_graph(random, seed=int(seed), num_rooms=int(rooms))
    return generate_comprehensive_demo_graph(seed=int(seed))


def _write_png(grid: np.ndarray, out_path: Path, tile_px: int) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    colors = {
        int(SP["VOID"]): (12, 12, 16),
        int(SP["FLOOR"]): (170, 150, 105),
        int(SP["WALL"]): (62, 76, 93),
        int(SP["BLOCK"]): (110, 83, 52),
        int(SP["DOOR_OPEN"]): (108, 180, 120),
        int(SP["DOOR_LOCKED"]): (230, 184, 60),
        int(SP["DOOR_BOMB"]): (210, 88, 72),
        int(SP["DOOR_PUZZLE"]): (122, 116, 210),
        int(SP["DOOR_BOSS"]): (200, 70, 170),
        int(SP["DOOR_SOFT"]): (120, 145, 155),
        int(SP["ENEMY"]): (190, 65, 65),
        int(SP["START"]): (55, 180, 95),
        int(SP["TRIFORCE"]): (250, 220, 70),
        int(SP["BOSS"]): (115, 38, 65),
        int(SP["KEY_SMALL"]): (242, 207, 90),
        int(SP["KEY_BOSS"]): (250, 130, 220),
        int(SP["KEY_ITEM"]): (82, 185, 225),
        int(SP["ITEM_MINOR"]): (235, 185, 105),
        int(SP["ELEMENT"]): (45, 105, 200),
        int(SP["ELEMENT_FLOOR"]): (70, 150, 210),
        int(SP["STAIR"]): (120, 90, 175),
        int(SP["PUZZLE"]): (150, 95, 220),
    }
    grid = np.asarray(grid, dtype=np.int32)
    image = Image.new("RGB", (grid.shape[1] * tile_px, grid.shape[0] * tile_px), colors[int(SP["VOID"])])
    draw = ImageDraw.Draw(image)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            tile_id = int(grid[row, col])
            x0 = col * tile_px
            y0 = row * tile_px
            draw.rectangle(
                [x0, y0, x0 + tile_px - 1, y0 + tile_px - 1],
                fill=colors.get(tile_id, (240, 0, 240)),
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return True


def _json_ready_graph(mission_graph: Any) -> Dict[str, Any]:
    nodes = {
        str(node_id): {
            "type": str(node.node_type.name),
            "position": list(map(int, node.position)),
            "difficulty": float(node.difficulty),
        }
        for node_id, node in getattr(mission_graph, "nodes", {}).items()
    }
    edges = [
        {
            "source": int(edge.source),
            "target": int(edge.target),
            "type": str(edge.edge_type.name),
            "item_required": edge.item_required,
            "key_required": edge.key_required,
            "switch_id": edge.switch_id,
        }
        for edge in getattr(mission_graph, "edges", [])
    ]
    return {"nodes": nodes, "edges": edges}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Generation device.")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--graph-mode", choices=("comprehensive", "random"), default="comprehensive")
    parser.add_argument("--rooms", type=int, default=10, help="Room count for --graph-mode random.")
    parser.add_argument("--algorithms", default="0-6", help="Comma/range list of GUI solver algorithms to validate.")
    parser.add_argument("--timeout-states", type=int, default=200000)
    parser.add_argument("--require-all", action="store_true", help="Fail unless every requested solver succeeds.")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--strict-checkpoints", action="store_true")
    parser.add_argument("--out-txt", default="examples/gui_demo_real_checkpoint_new_quest_level.txt")
    parser.add_argument("--out-json", default="examples/gui_demo_real_checkpoint_new_quest_metadata.json")
    parser.add_argument("--out-route-json", default="examples/gui_demo_real_checkpoint_new_quest_route.json")
    parser.add_argument("--out-png", default="outputs/gui_demo_real_checkpoint_new_quest_level.png")
    parser.add_argument("--tile-px", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    import torch

    discovered = discover_best_output_checkpoint()
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    mission_data = _build_mission_data(args.graph_mode, int(args.seed), int(args.rooms))
    mission_graph = mission_data["mission_graph"]

    LOGGER.info("Best discovered checkpoint: %s", discovered)
    LOGGER.info("Using checkpoint: %s", checkpoint)
    pipeline = load_canonical_generation_pipeline(
        checkpoint,
        torch.device(str(args.device)),
        LOGGER,
        strict_checkpoint_mode=bool(args.strict_checkpoints),
    )
    result = generate_dungeon_with_pipeline(
        pipeline,
        mission_graph,
        seed=int(args.seed),
        logger=LOGGER,
    )

    prepared = prepare_dungeon_grid_for_validation(result.dungeon_grid)
    grid = np.asarray(prepared.grid, dtype=np.int32)
    validation: Dict[str, Any]
    if args.skip_validation:
        validation = {
            "success": None,
            "all_success": None,
            "start": list(map(int, prepared.start)),
            "goal": list(map(int, prepared.goal)),
            "algorithms": {},
            "best_route": None,
        }
    else:
        validation = _validate_algorithms(
            grid,
            start=prepared.start,
            goal=prepared.goal,
            algorithms=_parse_algorithms(args.algorithms),
            timeout_states=int(args.timeout_states),
        )
        if validation["success"] is False or (bool(args.require_all) and validation.get("all_success") is False):
            raise RuntimeError(f"Generated dungeon failed validation: {validation}")

    out_txt = Path(args.out_txt).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_route = Path(args.out_route_json).expanduser().resolve()
    out_png = Path(args.out_png).expanduser().resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_txt, grid.astype(np.int32), fmt="%d")

    png_written = _write_png(grid, out_png, int(args.tile_px))
    route_payload = validation.get("best_route")
    if route_payload:
        out_route.parent.mkdir(parents=True, exist_ok=True)
        out_route.write_text(
            json.dumps(
                {
                    "name": "gui_demo_real_checkpoint_new_quest",
                    "source_level": str(out_txt),
                    "checkpoint": str(checkpoint),
                    "seed": int(args.seed),
                    "start": validation.get("start"),
                    "goal": validation.get("goal"),
                    **route_payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    layout_map = {}
    stitched_layout = getattr(result, "stitched_layout", None)
    if stitched_layout is not None:
        layout_map = {
            str(room_id): list(map(int, bbox))
            for room_id, bbox in dict(getattr(stitched_layout, "layout_map", {}) or {}).items()
        }

    metadata = {
        "name": "gui_demo_real_checkpoint_new_quest",
        "created_by": "scripts/generate_gui_real_checkpoint_demo.py",
        "created_mode": "real_checkpoint_direct_no_showcase_overlay",
        "seed": int(args.seed),
        "graph_mode": str(args.graph_mode),
        "best_discovered_checkpoint": str(discovered) if discovered is not None else None,
        "checkpoint": str(checkpoint),
        "shape": [int(grid.shape[0]), int(grid.shape[1])],
        "rooms": int(len(getattr(result, "rooms", {}) or {})),
        "entity_counts": _entity_counts(grid),
        "generation_metrics": dict(getattr(result, "metrics", {}) or {}),
        "generation_time_sec": float(getattr(result, "generation_time", 0.0) or 0.0),
        "validation_preparation": prepared.diagnostics,
        "validation": validation,
        "mission_graph": _json_ready_graph(mission_graph),
        "layout_map": layout_map,
        "artifacts": {
            "level_txt": str(out_txt),
            "metadata_json": str(out_json),
            "route_json": str(out_route) if route_payload else None,
            "preview_png": str(out_png) if png_written else None,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    LOGGER.info("Wrote level: %s", out_txt)
    LOGGER.info("Wrote metadata: %s", out_json)
    if route_payload:
        LOGGER.info("Wrote route: %s", out_route)
    if png_written:
        LOGGER.info("Wrote preview: %s", out_png)
    LOGGER.info(
        "GUI command: python gui_runner.py --advanced --load-txt %s --solver-algorithm %s --solver-timeout 200",
        out_txt,
        int(route_payload.get("algorithm_idx", 0)) if route_payload else 0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
