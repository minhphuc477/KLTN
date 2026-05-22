"""
Validation handoff and paper-facing A* vs P-CBS comparison utilities.

This module makes the stitched-dungeon -> validator boundary explicit.
The generator can emit room-local VOID or invalid semantic IDs during
intermediate stages; the hard oracle and P-CBS validator should only see a
canonical semantic grid with a single start/goal pair and no enclosed VOID
leaks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE, TileID
from src.evaluation.search_benchmark_utils import (
    confusion_ratio_vs_oracle,
    path_efficiency_ratio,
    run_astar_oracle,
)
from src.simulation.cognitive_bounded_search import PersonaDrivenCognitiveBoundedSearch
from src.simulation.validator import CONDITIONAL_IDS, ZeldaLogicEnv


GridPos = Tuple[int, int]


@dataclass
class PreparedValidationDungeon:
    """Canonical stitched dungeon ready for hard-oracle and P-CBS validation."""

    grid: np.ndarray
    start: GridPos
    goal: GridPos
    diagnostics: Dict[str, Any]


def _valid_tile_ids() -> np.ndarray:
    return np.asarray(sorted({int(v) for v in SEMANTIC_PALETTE.values()}), dtype=np.int32)


def _json_ready(value: Any) -> Any:
    """Recursively convert tuple-key mappings into JSON-safe structures."""
    if isinstance(value, dict):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _coerce_grid(source: Any) -> np.ndarray:
    if hasattr(source, "global_grid"):
        source = getattr(source, "global_grid")
    array = np.asarray(source)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D semantic grid, got shape={tuple(array.shape)!r}")

    if np.issubdtype(array.dtype, np.floating):
        out = np.nan_to_num(array, nan=float(TileID.FLOOR), posinf=float(TileID.WALL), neginf=float(TileID.WALL))
        return np.rint(out).astype(np.int32, copy=False)

    try:
        return array.astype(np.int32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Semantic grid is not coercible to int32") from exc


def _resolve_terminals(source: Any, grid: np.ndarray) -> Tuple[Optional[GridPos], Optional[GridPos]]:
    start = getattr(source, "start_global", None) if hasattr(source, "start_global") else None
    goal = getattr(source, "triforce_global", None) if hasattr(source, "triforce_global") else None
    if start is None:
        starts = np.argwhere(grid == int(TileID.START))
        if starts.size:
            start = tuple(int(v) for v in starts[0])
    if goal is None:
        goals = np.argwhere(grid == int(TileID.TRIFORCE))
        if goals.size:
            goal = tuple(int(v) for v in goals[0])
    return start, goal


def _normalize_terminal(
    grid: np.ndarray,
    *,
    tile_id: int,
    chosen: Optional[GridPos],
    replacement_tile: int,
) -> Tuple[GridPos, int]:
    matches = np.argwhere(grid == int(tile_id))
    removed = max(0, int(matches.shape[0]) - 1)
    if chosen is None:
        if matches.size <= 0:
            raise ValueError(f"Missing required terminal tile_id={int(tile_id)}")
        chosen = tuple(int(v) for v in matches[0])
    if not (0 <= int(chosen[0]) < grid.shape[0] and 0 <= int(chosen[1]) < grid.shape[1]):
        raise ValueError(f"Terminal {chosen!r} is outside grid bounds {tuple(grid.shape)!r}")

    if matches.size > 0:
        grid[grid == int(tile_id)] = int(replacement_tile)
    grid[int(chosen[0]), int(chosen[1])] = int(tile_id)
    return (int(chosen[0]), int(chosen[1])), removed


def _mark_exterior_void(void_mask: np.ndarray) -> np.ndarray:
    exterior = np.zeros_like(void_mask, dtype=bool)
    if not bool(np.any(void_mask)):
        return exterior

    queue: deque[GridPos] = deque()
    height, width = void_mask.shape

    def push_if_void(row: int, col: int) -> None:
        if 0 <= row < height and 0 <= col < width and bool(void_mask[row, col]) and not bool(exterior[row, col]):
            exterior[row, col] = True
            queue.append((row, col))

    for col in range(width):
        push_if_void(0, col)
        push_if_void(height - 1, col)
    for row in range(height):
        push_if_void(row, 0)
        push_if_void(row, width - 1)

    while queue:
        row, col = queue.popleft()
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            push_if_void(row + d_row, col + d_col)

    return exterior


def prepare_dungeon_grid_for_validation(
    source: Any,
    *,
    start: Optional[GridPos] = None,
    goal: Optional[GridPos] = None,
    invalid_fill_tile: int = int(TileID.FLOOR),
    enclosed_void_fill_tile: int = int(TileID.WALL),
) -> PreparedValidationDungeon:
    """
    Sanitize a stitched dungeon before hard-oracle/P-CBS validation.

    Guarantees:
    - grid is int32 and 2D
    - semantic IDs belong to the canonical palette
    - enclosed VOID leaks are removed
    - exactly one START and one TRIFORCE exist
    """
    grid = _coerce_grid(source)
    inferred_start, inferred_goal = _resolve_terminals(source, grid)
    if start is None:
        start = inferred_start
    if goal is None:
        goal = inferred_goal

    valid_ids = _valid_tile_ids()
    invalid_mask = ~np.isin(grid, valid_ids)
    invalid_ids = [int(v) for v in np.unique(grid[invalid_mask])] if bool(np.any(invalid_mask)) else []
    invalid_count = int(np.sum(invalid_mask))
    if invalid_count > 0:
        grid[invalid_mask] = int(invalid_fill_tile)

    void_mask = grid == int(TileID.VOID)
    exterior_void = _mark_exterior_void(void_mask)
    enclosed_void = void_mask & ~exterior_void
    enclosed_void_count = int(np.sum(enclosed_void))
    if enclosed_void_count > 0:
        grid[enclosed_void] = int(enclosed_void_fill_tile)

    floor_id = int(TileID.FLOOR)
    start, removed_start = _normalize_terminal(
        grid,
        tile_id=int(TileID.START),
        chosen=start,
        replacement_tile=floor_id,
    )
    goal, removed_goal = _normalize_terminal(
        grid,
        tile_id=int(TileID.TRIFORCE),
        chosen=goal,
        replacement_tile=floor_id,
    )
    if start == goal:
        raise ValueError("Start and goal collapse to the same grid position after sanitization")

    diagnostics = {
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "invalid_tile_count": invalid_count,
        "invalid_tile_ids": invalid_ids,
        "enclosed_void_tiles_filled": enclosed_void_count,
        "exterior_void_tiles_retained": int(np.sum(grid == int(TileID.VOID))),
        "duplicate_start_tiles_removed": removed_start,
        "duplicate_goal_tiles_removed": removed_goal,
        "start": [int(start[0]), int(start[1])],
        "goal": [int(goal[0]), int(goal[1])],
    }
    return PreparedValidationDungeon(grid=grid, start=start, goal=goal, diagnostics=diagnostics)


def _count_puzzle_stall_steps(grid: np.ndarray, path: Iterable[GridPos], goal: GridPos) -> int:
    """
    Estimate hesitation inside conditional/puzzle regions.

    A step is counted as a stall when the agent remains on conditional content
    without reducing the Manhattan distance to the goal.
    """
    ordered = list(path)
    if len(ordered) < 2:
        return 0

    stall_steps = 0
    puzzle_like = {int(TileID.PUZZLE), *[int(v) for v in CONDITIONAL_IDS]}
    for current, nxt in zip(ordered[:-1], ordered[1:]):
        row, col = int(current[0]), int(current[1])
        tile_id = int(grid[row, col])
        if tile_id not in puzzle_like:
            continue
        current_dist = abs(row - int(goal[0])) + abs(col - int(goal[1]))
        next_dist = abs(int(nxt[0]) - int(goal[0])) + abs(int(nxt[1]) - int(goal[1]))
        if next_dist >= current_dist:
            stall_steps += 1
    return stall_steps


def compute_pcbs_readability_metrics(
    *,
    oracle: Mapping[str, Any],
    pcbs_success: bool,
    pcbs_solution_length: int,
    pcbs_trajectory_length: int,
    pcbs_states: int,
    timeout_pcbs: int,
    pcbs_metrics: Any,
    puzzle_stall_steps: int,
) -> Dict[str, Any]:
    """
    Derive report-facing P-CBS calibration metrics.

    The raw P-CBS fields remain available; these derived fields make tables
    easier to compare across maps/personas by separating optimal solvability
    from bounded-rationality effort.
    """
    oracle_success = bool(oracle.get("success", False))
    oracle_path_length = int(oracle.get("path_length", 0) or 0)
    path_delta: Optional[int] = None
    if oracle_success and bool(pcbs_success) and oracle_path_length > 0:
        path_delta = int(pcbs_solution_length) - int(oracle_path_length)

    total_steps = max(1, int(getattr(pcbs_metrics, "total_steps", pcbs_trajectory_length) or pcbs_trajectory_length))
    unique_tiles = max(1, int(getattr(pcbs_metrics, "unique_tiles_visited", 1) or 1))
    revisit_rate = max(0.0, float(total_steps - unique_tiles) / float(max(1, total_steps)))
    normalized_confusion = float(np.clip(float(getattr(pcbs_metrics, "confusion_index", 0.0) or 0.0) / 3.0, 0.0, 1.0))
    normalized_entropy = float(np.clip(float(getattr(pcbs_metrics, "navigation_entropy", 0.0) or 0.0) / 2.0, 0.0, 1.0))
    normalized_load = float(np.clip(float(getattr(pcbs_metrics, "cognitive_load", 0.0) or 0.0) / 2.5, 0.0, 1.0))
    budget_fraction = float(np.clip(float(pcbs_states) / float(max(1, int(timeout_pcbs))), 0.0, 1.0))
    stall_fraction = float(np.clip(float(puzzle_stall_steps) / float(max(1, pcbs_trajectory_length)), 0.0, 1.0))

    bounded_rationality_index = float(
        np.clip(
            (0.24 * normalized_confusion)
            + (0.18 * normalized_entropy)
            + (0.22 * normalized_load)
            + (0.20 * budget_fraction)
            + (0.16 * stall_fraction),
            0.0,
            1.0,
        )
    )
    if oracle_success and not bool(pcbs_success):
        bounded_rationality_index = float(np.clip(bounded_rationality_index + 0.20, 0.0, 1.0))
    cognitive_effort_index = float(
        np.clip(
            (0.30 * normalized_load)
            + (0.24 * normalized_confusion)
            + (0.20 * revisit_rate)
            + (0.16 * budget_fraction)
            + (0.10 * stall_fraction),
            0.0,
            1.0,
        )
    )
    readability_score = float(np.clip(1.0 - bounded_rationality_index, 0.0, 1.0))

    return {
        "oracle_pcbs_path_delta": path_delta,
        "bounded_rationality_index": bounded_rationality_index,
        "readability_score": readability_score,
        "cognitive_effort_index": cognitive_effort_index,
        "state_budget_fraction": budget_fraction,
        "revisit_rate": revisit_rate,
        "puzzle_stall_fraction": stall_fraction,
        "oracle_solved_but_pcbs_failed": bool(oracle_success and not bool(pcbs_success)),
    }


def _extract_validation_env_kwargs(source: Any) -> Dict[str, Any]:
    """Best-effort extraction of stitched puzzle metadata from pipeline outputs."""
    graph = getattr(source, "mission_graph", None)
    stitched_layout = getattr(source, "stitched_layout", None)
    slot_positions = dict(getattr(stitched_layout, "slot_positions", {}) or {})
    room_offsets = dict(getattr(stitched_layout, "room_offsets", {}) or {})
    room_positions = {
        slot_pos: tuple(room_offsets.get(room_id, ()))
        for room_id, slot_pos in slot_positions.items()
        if room_id in room_offsets
    }
    if room_positions:
        room_to_node = {slot_pos: room_id for room_id, slot_pos in slot_positions.items()}
        node_to_room = {room_id: slot_pos for room_id, slot_pos in slot_positions.items()}
    else:
        room_to_node = None
        node_to_room = None
    return {
        "graph": graph,
        "room_positions": room_positions or None,
        "room_to_node": room_to_node,
        "node_to_room": node_to_room,
        "room_puzzle_metadata": dict(getattr(source, "puzzle_metadata", {}) or {}) or None,
    }


def evaluate_astar_vs_pcbs(
    source: Any,
    *,
    persona: str = "novice",
    timeout_astar: int = 200000,
    timeout_pcbs: int = 50000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the hard oracle and P-CBS on the same sanitized dungeon grid."""
    prepared = prepare_dungeon_grid_for_validation(source)
    grid = prepared.grid
    start = prepared.start
    goal = prepared.goal
    manhattan = abs(int(start[0]) - int(goal[0])) + abs(int(start[1]) - int(goal[1]))
    env_kwargs = _extract_validation_env_kwargs(source)

    env_astar = ZeldaLogicEnv(semantic_grid=grid, **env_kwargs)
    oracle = run_astar_oracle(env_astar, timeout=int(timeout_astar))

    env_pcbs = ZeldaLogicEnv(semantic_grid=grid, **env_kwargs)
    pcbs = PersonaDrivenCognitiveBoundedSearch(
        env_pcbs,
        persona=str(persona),
        timeout=int(timeout_pcbs),
        seed=seed,
    )
    pcbs_success, pcbs_path, pcbs_states, pcbs_metrics = pcbs.solve()
    pcbs_status = "success"
    if not pcbs_success:
        pcbs_status = "budget_exhausted" if int(pcbs_states) >= int(timeout_pcbs) else "failed"

    pcbs_trajectory_length = int(len(pcbs_path))
    pcbs_solution_length = pcbs_trajectory_length if bool(pcbs_success) else 0
    total_revisits = max(0, int(pcbs_metrics.total_steps) - int(pcbs_metrics.unique_tiles_visited))
    puzzle_stall_steps = _count_puzzle_stall_steps(grid, pcbs_path, goal)
    confusion_ratio = confusion_ratio_vs_oracle(
        int(oracle["path_length"]),
        pcbs_solution_length,
        oracle_status=str(oracle["status"]),
        candidate_success=bool(pcbs_success),
    )
    readability_metrics = compute_pcbs_readability_metrics(
        oracle=oracle,
        pcbs_success=bool(pcbs_success),
        pcbs_solution_length=pcbs_solution_length,
        pcbs_trajectory_length=pcbs_trajectory_length,
        pcbs_states=int(pcbs_states),
        timeout_pcbs=int(timeout_pcbs),
        pcbs_metrics=pcbs_metrics,
        puzzle_stall_steps=puzzle_stall_steps,
    )

    return {
        "validation_handoff": prepared.diagnostics,
        "oracle": {
            **oracle,
            "path_efficiency_ratio": path_efficiency_ratio(int(oracle["path_length"]), manhattan),
        },
        "pcbs": {
            "persona": str(persona),
            "seed": int(seed),
            "timeout": int(timeout_pcbs),
            "success": bool(pcbs_success),
            "path_length": int(pcbs_solution_length),
            "trajectory_length": int(pcbs_trajectory_length),
            "states_explored": int(pcbs_states),
            "path_efficiency_ratio": path_efficiency_ratio(pcbs_solution_length, manhattan),
            "navigation_entropy": float(pcbs_metrics.navigation_entropy),
            "room_entropy": float(pcbs_metrics.room_entropy),
            "unique_rooms_visited": int(getattr(pcbs_metrics, "unique_rooms_visited", 0) or 0),
            "total_revisits": int(total_revisits),
            "aha_latency": int(pcbs_metrics.aha_latency),
            "goal_sighting_latency": int(pcbs_metrics.aha_latency),
            "puzzle_stall_steps": int(puzzle_stall_steps),
            "confusion_index": float(pcbs_metrics.confusion_index),
            "cognitive_load": float(pcbs_metrics.cognitive_load),
            "deliberation_events": int(getattr(pcbs_metrics, "deliberation_events", 0) or 0),
            "budget_exhaustion_events": int(getattr(pcbs_metrics, "budget_exhaustion_events", 0) or 0),
            "peak_frustration": float(getattr(pcbs_metrics, "peak_frustration", 0.0) or 0.0),
            "status": pcbs_status,
            "affordance_reactivations": int(getattr(pcbs_metrics, "affordance_reactivations", 0) or 0),
            "affordance_guided_steps": int(getattr(pcbs_metrics, "affordance_guided_steps", 0) or 0),
            "inventory_change_events": int(getattr(pcbs_metrics, "inventory_change_events", 0) or 0),
            "focus_switches": int(getattr(pcbs_metrics, "focus_switches", 0) or 0),
            "focus_guided_steps": int(getattr(pcbs_metrics, "focus_guided_steps", 0) or 0),
            **readability_metrics,
            "metrics": _json_ready(pcbs_metrics.to_dict()),
        },
        "comparison": {
            "oracle_status": str(oracle["status"]),
            "oracle_solved": bool(oracle["success"]),
            "pcbs_solved": bool(pcbs_success),
            "pcbs_status": pcbs_status,
            "cognitive_gap": bool(oracle["success"] and not pcbs_success),
            "confusion_ratio_vs_oracle": float(confusion_ratio) if np.isfinite(confusion_ratio) else None,
            "oracle_pcbs_path_delta": readability_metrics["oracle_pcbs_path_delta"],
            "bounded_rationality_index": float(readability_metrics["bounded_rationality_index"]),
            "readability_score": float(readability_metrics["readability_score"]),
            "cognitive_effort_index": float(readability_metrics["cognitive_effort_index"]),
        },
    }


def build_ieee_markdown_table(result: Mapping[str, Any], *, map_name: str = "Generated Dungeon") -> str:
    """Render a compact markdown table for report drafting."""
    oracle = dict(result.get("oracle", {}))
    pcbs = dict(result.get("pcbs", {}))
    comparison = dict(result.get("comparison", {}))
    rows = [
        (
            "A* Oracle",
            bool(oracle.get("success", False)),
            int(oracle.get("path_length", 0)),
            round(float(oracle.get("path_efficiency_ratio", 0.0)), 3),
            0,
            0.0,
            0,
            str(oracle.get("status", "")),
        ),
        (
            f"P-CBS ({pcbs.get('persona', 'novice')})",
            bool(pcbs.get("success", False)),
            int(pcbs.get("path_length", 0)),
            round(float(pcbs.get("path_efficiency_ratio", 0.0)), 3),
            int(pcbs.get("total_revisits", 0)),
            round(float(pcbs.get("navigation_entropy", 0.0)), 3),
            int(pcbs.get("goal_sighting_latency", pcbs.get("aha_latency", 0))),
            str(comparison.get("pcbs_status") or pcbs.get("status") or ""),
        ),
    ]

    lines = [
        f"| Map | Solver | Solved | Path Length | PER | Total Revisits | Navigation Entropy | Goal-Sighting Latency | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for solver, solved, path_len, per, revisits, nav_entropy, aha_latency, status in rows:
        lines.append(
            f"| {map_name} | {solver} | {int(bool(solved))} | {int(path_len)} | "
            f"{float(per):.3f} | {int(revisits)} | {float(nav_entropy):.3f} | "
            f"{int(aha_latency)} | {status} |"
        )
    return "\n".join(lines)


__all__ = [
    "PreparedValidationDungeon",
    "prepare_dungeon_grid_for_validation",
    "compute_pcbs_readability_metrics",
    "evaluate_astar_vs_pcbs",
    "build_ieee_markdown_table",
]
