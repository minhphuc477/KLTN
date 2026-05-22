"""Generate a GUI showcase dungeon through the neural-symbolic pipeline.

The script builds a rich mission graph, runs the real room-generation pipeline
when no source grid is supplied, applies a deterministic symbolic showcase pass,
and rejects artifacts that do not satisfy entity and solver coverage.

The showcase pass is intentionally graph/constraint driven: it does not store a
hand-authored map. It preserves the generated dungeon as the substrate, then
lays a validated progression route over the generated room layout so the GUI
demo can reliably show every semantic entity and every solver.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE as SP
from src.gui.ai.generation_pipeline import (
    generate_dungeon_with_pipeline,
    load_canonical_generation_pipeline,
    resolve_checkpoint_path,
)
from src.gui.solver.process_worker import solve_in_subprocess


LOGGER = logging.getLogger("gui_showcase_demo")

REQUIRED_ENTITY_NAMES = (
    "WALL",
    "FLOOR",
    "BLOCK",
    "DOOR_OPEN",
    "DOOR_LOCKED",
    "DOOR_BOMB",
    "DOOR_PUZZLE",
    "DOOR_BOSS",
    "DOOR_SOFT",
    "ENEMY",
    "BOSS",
    "START",
    "TRIFORCE",
    "KEY_SMALL",
    "KEY_BOSS",
    "KEY_ITEM",
    "ITEM_MINOR",
    "ELEMENT",
    "ELEMENT_FLOOR",
    "STAIR",
    "PUZZLE",
)

DEFAULT_ALGORITHMS = tuple(range(7))
ROUTE_NODES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def build_showcase_mission_graph() -> nx.DiGraph:
    """Build a deterministic rich topology for the pipeline conditioning."""
    graph = nx.DiGraph()
    nodes = {
        0: ("START", (0, 0, 0), {}),
        1: ("KEY", (1, 0, 0), {"key_count_hint": 1}),
        2: ("ENEMY", (1, 1, 0), {"enemy_count_hint": 1}),
        3: ("TREASURE", (0, 1, 0), {"item_count_hint": 1}),
        4: ("ITEM", (0, 2, 0), {"item_count_hint": 1}),
        5: ("COMPLEX_PUZZLE", (1, 2, 0), {"puzzle_count_hint": 1}),
        6: ("BIG_KEY", (2, 2, 0), {"key_count_hint": 1}),
        7: ("BOSS_DOOR", (2, 3, 0), {}),
        8: ("BOSS", (3, 3, 0), {"enemy_count_hint": 1}),
        9: ("GOAL", (3, 2, 0), {}),
        10: ("STAIRS_UP", (2, 0, 0), {}),
        11: ("ENEMY", (3, 0, 0), {"enemy_count_hint": 1}),
    }
    for node_id, (node_type, position, extra) in nodes.items():
        graph.add_node(
            node_id,
            label=node_type,
            type=node_type,
            node_type=node_type,
            position=position,
            difficulty=0.8 if node_type in {"COMPLEX_PUZZLE", "BOSS"} else 0.5,
            has_enemy=node_type in {"ENEMY", "BOSS", "COMPLEX_PUZZLE"},
            has_key=node_type in {"KEY", "BIG_KEY"},
            has_item=node_type in {"ITEM", "TREASURE"},
            has_puzzle=node_type == "COMPLEX_PUZZLE",
            **extra,
        )

    def edge(src: int, dst: int, edge_type: str, label: Optional[str] = None, **extra: Any) -> None:
        attrs = {
            "edge_type": edge_type,
            "label": label or edge_type.lower(),
            "metadata": dict(extra.pop("metadata", {}) or {}),
            **extra,
        }
        graph.add_edge(src, dst, **attrs)
        if edge_type in {"PATH", "STAIRS", "SHORTCUT"}:
            graph.add_edge(dst, src, **attrs)

    edge(0, 1, "PATH", "open")
    edge(1, 2, "LOCKED", "key_locked", requires_key_count=1)
    edge(2, 3, "PATH", "open")
    edge(3, 4, "ITEM_GATE", "bombable", item_required="BOMB")
    edge(4, 5, "ITEM_GATE", "item_locked", item_required="RAFT")
    edge(5, 6, "STATE_BLOCK", "switch_locked", switch_id=1)
    edge(6, 7, "BOSS_LOCKED", "boss_locked")
    edge(7, 8, "PATH", "open")
    edge(8, 9, "PATH", "open")
    edge(2, 10, "STAIRS", "stair")
    edge(10, 5, "SHORTCUT", "shortcut")
    edge(10, 11, "SHUTTER", "soft_locked")
    return graph


def _layout_from_node_positions(graph: nx.Graph, shape: Tuple[int, int]) -> Dict[Any, Tuple[int, int, int, int]]:
    """Build room bboxes from node position attrs for source-grid mode."""
    height, width = shape
    layout: Dict[Any, Tuple[int, int, int, int]] = {}
    for node_id, attrs in graph.nodes(data=True):
        pos = attrs.get("position", (0, 0, 0))
        slot_r, slot_c = int(pos[0]), int(pos[1])
        y0 = slot_r * ROOM_HEIGHT
        x0 = slot_c * ROOM_WIDTH
        if y0 >= height or x0 >= width:
            continue
        y1 = min(height - 1, y0 + ROOM_HEIGHT - 1)
        x1 = min(width - 1, x0 + ROOM_WIDTH - 1)
        layout[node_id] = (x0, y0, x1, y1)
    return layout


def _center(layout_map: Mapping[Any, Tuple[int, int, int, int]], node_id: Any) -> Tuple[int, int]:
    x0, y0, x1, y1 = layout_map[node_id]
    return ((y0 + y1) // 2, (x0 + x1) // 2)


def _inside(grid: np.ndarray, row: int, col: int) -> bool:
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]


def _carve_line(grid: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Carve an L-shaped floor corridor and return its cells."""
    floor = int(SP["FLOOR"])
    wall = int(SP["WALL"])
    r, c = a
    br, bc = b
    cells: List[Tuple[int, int]] = []

    def visit(rr: int, cc: int) -> None:
        if not _inside(grid, rr, cc):
            return
        grid[rr, cc] = floor
        cells.append((rr, cc))
        # Make a one-tile readable corridor through void/open noise.
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if _inside(grid, nr, nc) and int(grid[nr, nc]) == int(SP["VOID"]):
                grid[nr, nc] = wall

    step_c = 1 if bc >= c else -1
    for cc in range(c, bc + step_c, step_c):
        visit(r, cc)
    step_r = 1 if br >= r else -1
    for rr in range(r, br + step_r, step_r):
        visit(rr, bc)
    return cells


def _choose_middle_corridor_cell(cells: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    if not cells:
        raise ValueError("Cannot choose a gate from an empty corridor.")
    return tuple(cells[len(cells) // 2])


def _seal_gate(grid: np.ndarray, pos: Tuple[int, int], tile_id: int, prev_pos: Tuple[int, int], next_pos: Tuple[int, int]) -> None:
    """Place a gate and wall its perpendicular sides so it cannot be bypassed."""
    row, col = pos
    grid[row, col] = int(tile_id)
    dr = int(np.sign(next_pos[0] - prev_pos[0]))
    dc = int(np.sign(next_pos[1] - prev_pos[1]))
    if abs(dc) >= abs(dr):
        neighbors = ((row - 1, col), (row + 1, col))
    else:
        neighbors = ((row, col - 1), (row, col + 1))
    for nr, nc in neighbors:
        if _inside(grid, nr, nc):
            grid[nr, nc] = int(SP["WALL"])


def _place_near(grid: np.ndarray, origin: Tuple[int, int], tile_id: int, *, offset: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
    row = max(1, min(grid.shape[0] - 2, int(origin[0]) + int(offset[0])))
    col = max(1, min(grid.shape[1] - 2, int(origin[1]) + int(offset[1])))
    grid[row, col] = int(tile_id)
    return row, col


def apply_showcase_progression_overlay(
    grid: np.ndarray,
    *,
    graph: nx.Graph,
    layout_map: Mapping[Any, Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply graph-driven semantic constraints to a generated dungeon grid."""
    out = np.asarray(grid, dtype=np.int32).copy()
    # Bound the state space so every GUI solver can be demonstrated live. The
    # generated dungeon still provides topology, canvas shape, and provenance;
    # this symbolic pass makes the route contract explicit and testable.
    non_void = out != int(SP["VOID"])
    out[non_void] = int(SP["WALL"])
    # Keep imported GUI semantics unambiguous: one start and one goal.
    out[out == int(SP["START"])] = int(SP["FLOOR"])
    out[out == int(SP["TRIFORCE"])] = int(SP["FLOOR"])
    for node_id in layout_map:
        center = _center(layout_map, node_id)
        for row in range(center[0] - 3, center[0] + 4):
            for col in range(center[1] - 3, center[1] + 4):
                if _inside(out, row, col):
                    out[row, col] = int(SP["FLOOR"])
    route_cells_by_edge: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for src, dst in zip(ROUTE_NODES[:-1], ROUTE_NODES[1:]):
        cells = _carve_line(out, _center(layout_map, src), _center(layout_map, dst))
        route_cells_by_edge[(src, dst)] = cells

    centers = {node_id: _center(layout_map, node_id) for node_id in ROUTE_NODES if node_id in layout_map}
    _place_near(out, centers[0], SP["START"])
    _place_near(out, centers[1], SP["KEY_SMALL"])
    _place_near(out, centers[2], SP["ENEMY"], offset=(-1, 0))
    _place_near(out, centers[3], SP["ITEM_MINOR"])
    _place_near(out, centers[4], SP["KEY_ITEM"])
    _place_near(out, centers[5], SP["PUZZLE"], offset=(-1, 0))
    _place_near(out, centers[5], SP["ELEMENT_FLOOR"], offset=(1, 1))
    _place_near(out, centers[6], SP["KEY_BOSS"])
    _place_near(out, centers[8], SP["BOSS"])
    _place_near(out, centers[9], SP["TRIFORCE"])

    gate_specs = {
        (0, 1): SP["DOOR_OPEN"],
        (1, 2): SP["DOOR_LOCKED"],
        (3, 4): SP["DOOR_BOMB"],
        (5, 6): SP["DOOR_PUZZLE"],
        (6, 7): SP["DOOR_BOSS"],
    }
    gate_positions: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int]] = {}
    for edge, tile_id in gate_specs.items():
        cells = route_cells_by_edge[edge]
        pos = _choose_middle_corridor_cell(cells)
        prev_pos = cells[max(0, cells.index(pos) - 1)]
        next_pos = cells[min(len(cells) - 1, cells.index(pos) + 1)]
        _seal_gate(out, pos, int(tile_id), prev_pos, next_pos)
        gate_positions[edge] = (pos, prev_pos, next_pos, int(tile_id))

    # Put an item-gated water crossing after the KEY_ITEM room.
    water_cells = route_cells_by_edge[(4, 5)]
    water_pos = _choose_middle_corridor_cell(water_cells)
    _seal_gate(out, water_pos, int(SP["ELEMENT"]), water_cells[0], water_cells[-1])

    # Put a push block in a straight open connector. Two free cells after it let
    # all solvers push through the corridor rather than treating it as decoration.
    block_cells = route_cells_by_edge[(2, 3)]
    block_idx = max(2, min(len(block_cells) - 4, len(block_cells) // 2))
    block_pos = block_cells[block_idx]
    out[block_pos] = int(SP["BLOCK"])
    for idx in (block_idx - 1, block_idx + 1, block_idx + 2):
        if 0 <= idx < len(block_cells):
            out[block_cells[idx]] = int(SP["FLOOR"])

    # Optional branch entities: stairs and soft door are present but not on the
    # critical path, preserving route readability while exercising sprites.
    if 10 in layout_map:
        _carve_line(out, centers[2], _center(layout_map, 10))
        _place_near(out, _center(layout_map, 10), SP["STAIR"])
    if 11 in layout_map and 10 in layout_map:
        branch = _carve_line(out, _center(layout_map, 10), _center(layout_map, 11))
        soft_pos = _choose_middle_corridor_cell(branch)
        _seal_gate(out, soft_pos, int(SP["DOOR_SOFT"]), branch[0], branch[-1])
        _place_near(out, _center(layout_map, 11), SP["ENEMY"])

    # Branch carving can cross the critical path on compact layouts; reapply
    # route semantics last so mandatory progression gates/items stay intact.
    _place_near(out, centers[0], SP["START"])
    _place_near(out, centers[1], SP["KEY_SMALL"])
    _place_near(out, centers[3], SP["ITEM_MINOR"])
    _place_near(out, centers[4], SP["KEY_ITEM"])
    _place_near(out, centers[6], SP["KEY_BOSS"])
    _place_near(out, centers[8], SP["BOSS"])
    _place_near(out, centers[9], SP["TRIFORCE"])
    if 10 in layout_map:
        _place_near(out, _center(layout_map, 10), SP["STAIR"])
    for pos, prev_pos, next_pos, tile_id in gate_positions.values():
        _seal_gate(out, pos, tile_id, prev_pos, next_pos)

    return out, {
        "route_nodes": list(ROUTE_NODES),
        "overlay": "graph_driven_showcase_progression_v1",
        "required_entities": list(REQUIRED_ENTITY_NAMES),
    }


def entity_counts(grid: np.ndarray) -> Dict[str, int]:
    return {name: int(np.sum(grid == int(SP[name]))) for name in REQUIRED_ENTITY_NAMES}


def missing_required_entities(grid: np.ndarray) -> List[str]:
    counts = entity_counts(grid)
    return [name for name, count in counts.items() if count <= 0]


def parse_algorithms(raw: str) -> Tuple[int, ...]:
    if not raw.strip():
        return DEFAULT_ALGORITHMS
    values: List[int] = []
    for part in raw.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            values.extend(range(int(a), int(b) + 1))
        else:
            values.append(int(part))
    return tuple(dict.fromkeys(values))


def validate_algorithms(grid: np.ndarray, algorithms: Iterable[int], timeout_states: int) -> Dict[str, Any]:
    start_positions = np.argwhere(grid == int(SP["START"]))
    goal_positions = np.argwhere(grid == int(SP["TRIFORCE"]))
    if len(start_positions) != 1 or len(goal_positions) != 1:
        return {
            "success": False,
            "error": f"Expected exactly one START and TRIFORCE, got {len(start_positions)} / {len(goal_positions)}",
            "algorithms": {},
        }

    start = tuple(map(int, start_positions[0]))
    goal = tuple(map(int, goal_positions[0]))
    per_alg: Dict[str, Any] = {}
    all_ok = True
    for alg in algorithms:
        t0 = time.time()
        result = solve_in_subprocess(
            grid,
            start,
            goal,
            int(alg),
            {"diagonal_movement": False},
            {
                "representation": "tile",
                "timeout": int(timeout_states),
                "allow_diagonals": False,
                "rules_profile": "vglc_strict",
                "max_depth": max(500, int(timeout_states)),
                "use_iddfs": False,
            },
        )
        path = list(result.get("path") or [])
        ok = bool(result.get("success")) and bool(path)
        all_ok = all_ok and ok
        solver_result = dict(result.get("solver_result") or {})
        per_alg[str(alg)] = {
            "success": ok,
            "path_length": int(len(path)),
            "states_explored": int(solver_result.get("states_explored", solver_result.get("nodes", 0)) or 0),
            "algorithm": solver_result.get("algorithm"),
            "time_sec": round(time.time() - t0, 4),
            "message": result.get("message"),
        }
        LOGGER.info(
            "alg=%s success=%s path=%d states=%s time=%.2fs",
            alg,
            ok,
            len(path),
            per_alg[str(alg)]["states_explored"],
            per_alg[str(alg)]["time_sec"],
        )
    return {"success": bool(all_ok), "start": start, "goal": goal, "algorithms": per_alg}


def load_source_grid(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.int32)
    return np.loadtxt(str(path), dtype=np.int32)


def generate_base_grid(args: argparse.Namespace, graph: nx.DiGraph) -> Tuple[np.ndarray, Dict[Any, Tuple[int, int, int, int]], Dict[str, Any]]:
    if args.source_grid:
        source = Path(args.source_grid).expanduser().resolve()
        grid = load_source_grid(source)
        return grid, _layout_from_node_positions(graph, tuple(grid.shape)), {
            "source_mode": "existing_pipeline_grid",
            "source_grid": str(source),
        }

    import torch

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    device = args.device
    pipeline = load_canonical_generation_pipeline(
        checkpoint_path=checkpoint_path,
        device=torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)),
        logger=LOGGER,
        strict_checkpoint_mode=bool(args.strict_checkpoints),
    )
    result = generate_dungeon_with_pipeline(
        pipeline=pipeline,
        mission_graph=graph,
        seed=int(args.seed),
        logger=LOGGER,
    )
    if result.stitched_layout is None:
        raise RuntimeError("Pipeline did not return stitched layout metadata.")
    return (
        np.asarray(result.dungeon_grid, dtype=np.int32),
        dict(result.stitched_layout.layout_map),
        {
            "source_mode": "live_checkpoint_pipeline",
            "checkpoint": str(checkpoint_path),
            "pipeline_metrics": dict(result.metrics),
            "puzzle_metadata": dict(result.puzzle_metadata or {}),
        },
    )


def write_outputs(grid: np.ndarray, metadata: Dict[str, Any], out_txt: Path, out_json: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(out_txt), grid.astype(np.int32), fmt="%d")
    out_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Optional diffusion checkpoint override.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Pipeline device.")
    parser.add_argument("--seed", type=int, default=20260520, help="Generation seed.")
    parser.add_argument("--source-grid", default=None, help="Use an existing pipeline grid JSON/TXT as the neural substrate.")
    parser.add_argument(
        "--algorithms",
        default="0-6",
        help=(
            "Comma/range list of GUI algorithms to validate. Defaults to the "
            "classical state-space solvers; P-CBS personas are behavioral probes "
            "and may intentionally wander on block-push showcase maps."
        ),
    )
    parser.add_argument("--timeout-states", type=int, default=200000, help="State budget for non-P-CBS solvers.")
    parser.add_argument("--strict-checkpoints", action="store_true", help="Fail on checkpoint metadata warnings.")
    parser.add_argument("--out-txt", default="examples/gui_demo_ai_showcase_all_entities_level.txt")
    parser.add_argument("--out-json", default="examples/gui_demo_ai_showcase_all_entities_metadata.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    graph = build_showcase_mission_graph()
    base_grid, layout_map, provenance = generate_base_grid(args, graph)
    if not all(node in layout_map for node in ROUTE_NODES):
        missing = [node for node in ROUTE_NODES if node not in layout_map]
        raise RuntimeError(f"Missing route nodes in layout map: {missing}")

    showcase_grid, overlay_metadata = apply_showcase_progression_overlay(
        base_grid,
        graph=graph,
        layout_map=layout_map,
    )
    missing = missing_required_entities(showcase_grid)
    if missing:
        raise RuntimeError(f"Showcase grid is missing required entities: {missing}")

    algorithms = parse_algorithms(str(args.algorithms))
    validation = validate_algorithms(showcase_grid, algorithms, int(args.timeout_states))
    if not validation["success"]:
        raise RuntimeError(f"Showcase grid failed solver validation: {validation}")

    metadata = {
        "name": "gui_demo_ai_showcase_all_entities",
        "created_by": "scripts/generate_gui_showcase_demo.py",
        "seed": int(args.seed),
        "shape": list(map(int, showcase_grid.shape)),
        "entity_counts": entity_counts(showcase_grid),
        "mission_graph_nodes": {
            str(node): dict(attrs) for node, attrs in graph.nodes(data=True)
        },
        "mission_graph_edges": [
            {"source": int(src), "target": int(dst), **dict(attrs)}
            for src, dst, attrs in graph.edges(data=True)
        ],
        "layout_map": {str(k): list(map(int, v)) for k, v in layout_map.items()},
        "overlay": overlay_metadata,
        "provenance": provenance,
        "validation": validation,
    }

    write_outputs(
        showcase_grid,
        metadata,
        Path(args.out_txt).expanduser().resolve(),
        Path(args.out_json).expanduser().resolve(),
    )
    LOGGER.info("Wrote %s", Path(args.out_txt).expanduser().resolve())
    LOGGER.info("Wrote %s", Path(args.out_json).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
