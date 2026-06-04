"""
Validation handoff and paper-facing A* vs P-CBS comparison utilities.

This module makes the stitched-dungeon -> validator boundary explicit.
The generator can emit room-local VOID or invalid semantic IDs during
intermediate stages; the hard oracle and P-CBS validator should only see a
canonical semantic grid with a single start/goal pair and no enclosed VOID
leaks.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE, TileID
from src.evaluation.search_benchmark_utils import (
    confusion_ratio_vs_oracle,
    path_efficiency_ratio,
    run_astar_oracle,
)
from src.simulation.cognitive_bounded_search import (
    AgentPersona,
    PersonaConfig,
    PersonaDrivenCognitiveBoundedSearch,
)
from src.simulation.validator import CONDITIONAL_IDS, ZeldaLogicEnv


GridPos = Tuple[int, int]

PCBS_STRUCTURAL_METRIC_SOURCE = "structural_bounded_rationality_metric_v2"
PCBS_READABILITY_WEIGHT_SOURCE = PCBS_STRUCTURAL_METRIC_SOURCE
PCBS_BOUNDED_RATIONALITY_WEIGHTS: Dict[str, float] = {
    "confusion": 0.24,
    "navigation_entropy": 0.18,
    "cognitive_load": 0.22,
    "state_budget": 0.20,
    "puzzle_stall": 0.16,
}
PCBS_COGNITIVE_EFFORT_WEIGHTS: Dict[str, float] = {
    "cognitive_load": 0.30,
    "confusion": 0.24,
    "revisit_rate": 0.20,
    "state_budget": 0.16,
    "puzzle_stall": 0.10,
}


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


def _normalize_weight_map(
    weights: Optional[Mapping[str, Any]],
    defaults: Mapping[str, float],
) -> Dict[str, float]:
    """Return a finite normalized weight map over the default metric keys."""
    cleaned: Dict[str, float] = {}
    for key, default in defaults.items():
        try:
            value = float((weights or {}).get(key, default))
        except (TypeError, ValueError):
            value = float(default)
        if not np.isfinite(value) or value < 0.0:
            value = float(default)
        cleaned[str(key)] = float(value)
    total = float(sum(cleaned.values()))
    if total <= 0.0:
        return dict(defaults)
    return {key: float(value / total) for key, value in cleaned.items()}


def _load_pcbs_calibration_payload(path: Optional[str | Path]) -> Dict[str, Any]:
    """Load the JSON artifact produced by calibrate_pcbs_personas_from_telemetry.py."""
    if path is None:
        return {}
    calibration_path = Path(path)
    if not calibration_path.exists():
        raise FileNotFoundError(calibration_path)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P-CBS calibration artifact must be a JSON object: {calibration_path}")
    return payload


def _persona_config_from_payload(
    *,
    persona: str,
    payload: Mapping[str, Any],
) -> Optional[PersonaConfig]:
    """Materialize a PersonaConfig from a calibration payload when available."""
    if not payload:
        return None
    persona_key = str(persona or "balanced").strip().lower()
    entry = payload.get(persona_key) or payload.get("observed")
    if not isinstance(entry, Mapping):
        return None
    raw_config = entry.get("calibrated_config")
    if not isinstance(raw_config, Mapping):
        raw_config = entry.get("overrides")
    if not isinstance(raw_config, Mapping):
        return None

    valid_fields = {field.name for field in fields(PersonaConfig)}
    try:
        base_persona = AgentPersona(persona_key)
    except ValueError:
        base_persona = AgentPersona.BALANCED
    base = asdict(PersonaConfig.get_persona(base_persona))
    for key, value in raw_config.items():
        if key in valid_fields:
            base[key] = value
    return PersonaConfig(**base)


def _calibration_weight_payload(
    payload: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], str]:
    """Extract optional calibrated readability weights from a payload."""
    if not payload:
        return None, None, PCBS_STRUCTURAL_METRIC_SOURCE
    source = str(payload.get("weight_source") or payload.get("source") or "telemetry_calibration_artifact")
    bounded = payload.get("bounded_rationality_weights")
    effort = payload.get("cognitive_effort_weights")
    if isinstance(bounded, Mapping) or isinstance(effort, Mapping):
        return (
            _normalize_weight_map(bounded if isinstance(bounded, Mapping) else None, PCBS_BOUNDED_RATIONALITY_WEIGHTS),
            _normalize_weight_map(effort if isinstance(effort, Mapping) else None, PCBS_COGNITIVE_EFFORT_WEIGHTS),
            source,
        )
    return None, None, source


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


def count_pcbs_puzzle_stall_steps(grid: np.ndarray, path: Iterable[GridPos], goal: GridPos) -> int:
    """Public wrapper for P-CBS puzzle-stall reporting."""
    return _count_puzzle_stall_steps(grid, path, goal)


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
    bounded_rationality_weights: Optional[Mapping[str, float]] = None,
    cognitive_effort_weights: Optional[Mapping[str, float]] = None,
    weight_source: str = PCBS_STRUCTURAL_METRIC_SOURCE,
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
    bounded_weights = _normalize_weight_map(
        bounded_rationality_weights,
        PCBS_BOUNDED_RATIONALITY_WEIGHTS,
    )
    effort_weights = _normalize_weight_map(
        cognitive_effort_weights,
        PCBS_COGNITIVE_EFFORT_WEIGHTS,
    )

    bounded_rationality_index = float(
        np.clip(
            (bounded_weights["confusion"] * normalized_confusion)
            + (bounded_weights["navigation_entropy"] * normalized_entropy)
            + (bounded_weights["cognitive_load"] * normalized_load)
            + (bounded_weights["state_budget"] * budget_fraction)
            + (bounded_weights["puzzle_stall"] * stall_fraction),
            0.0,
            1.0,
        )
    )
    if oracle_success and not bool(pcbs_success):
        bounded_rationality_index = float(np.clip(bounded_rationality_index + 0.20, 0.0, 1.0))
    cognitive_effort_index = float(
        np.clip(
            (effort_weights["cognitive_load"] * normalized_load)
            + (effort_weights["confusion"] * normalized_confusion)
            + (effort_weights["revisit_rate"] * revisit_rate)
            + (effort_weights["state_budget"] * budget_fraction)
            + (effort_weights["puzzle_stall"] * stall_fraction),
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
        "weight_source": str(weight_source or PCBS_STRUCTURAL_METRIC_SOURCE),
        "bounded_rationality_weights": dict(bounded_weights),
        "cognitive_effort_weights": dict(effort_weights),
        "metric_interpretation": (
            "structural_bounded_rationality"
            if str(weight_source or "").startswith("structural_")
            else "telemetry_calibrated"
        ),
        "oracle_solved_but_pcbs_failed": bool(oracle_success and not bool(pcbs_success)),
    }


def classify_pcbs_outcome(
    *,
    oracle: Mapping[str, Any],
    pcbs_success: bool,
    pcbs_solution_length: int,
    pcbs_trajectory_length: int,
    pcbs_states: int,
    timeout_pcbs: int,
    pcbs_metrics: Any,
    readability_metrics: Mapping[str, Any],
    puzzle_stall_steps: int,
) -> Dict[str, Any]:
    """
    Classify the P-CBS result for paper tables.

    The class is intentionally descriptive rather than causal: it separates
    hard invalidity from bounded-rational failure and names the dominant
    pressure that made the run costly.
    """
    oracle_success = bool(oracle.get("success", False))
    oracle_path_length = int(oracle.get("path_length", 0) or 0)
    timeout = max(1, int(timeout_pcbs))
    states = max(0, int(pcbs_states))
    trajectory_length = max(1, int(pcbs_trajectory_length))

    confusion = float(getattr(pcbs_metrics, "confusion_index", 0.0) or 0.0)
    entropy = float(getattr(pcbs_metrics, "navigation_entropy", 0.0) or 0.0)
    load = float(getattr(pcbs_metrics, "cognitive_load", 0.0) or 0.0)
    budget_exhaustion_events = int(getattr(pcbs_metrics, "budget_exhaustion_events", 0) or 0)
    affordance_reactivations = int(getattr(pcbs_metrics, "affordance_reactivations", 0) or 0)
    focus_switches = int(getattr(pcbs_metrics, "focus_switches", 0) or 0)

    pressures = {
        "confusion": float(np.clip(confusion / 3.0, 0.0, 1.0)),
        "navigation_entropy": float(np.clip(entropy / 2.0, 0.0, 1.0)),
        "cognitive_load": float(np.clip(load / 2.5, 0.0, 1.0)),
        "state_budget": float(np.clip(states / float(timeout), 0.0, 1.0)),
        "puzzle_stall": float(np.clip(float(puzzle_stall_steps) / float(trajectory_length), 0.0, 1.0)),
        "affordance_reactivation": float(np.clip(affordance_reactivations / 3.0, 0.0, 1.0)),
        "focus_switching": float(np.clip(focus_switches / 6.0, 0.0, 1.0)),
    }
    dominant_pressure, dominant_value = max(pressures.items(), key=lambda item: item[1])

    bounded_index = float(readability_metrics.get("bounded_rationality_index", 0.0) or 0.0)
    path_delta = readability_metrics.get("oracle_pcbs_path_delta")
    path_delta_value = None if path_delta is None else int(path_delta)

    if not oracle_success:
        outcome_class = "oracle_unsolved"
        calibration_bucket = "hard_invalid"
        failure_driver = "hard_oracle_failure"
    elif bool(pcbs_success):
        if oracle_path_length > 0 and path_delta_value is not None:
            relative_delta = path_delta_value / float(max(1, oracle_path_length))
        else:
            relative_delta = 0.0
        if relative_delta <= 0.10 and bounded_index < 0.35:
            outcome_class = "bounded_success_near_oracle"
            calibration_bucket = "readable"
            failure_driver = "none"
        elif relative_delta <= 0.50 and bounded_index < 0.65:
            outcome_class = "bounded_success_costly"
            calibration_bucket = "readable_but_costly"
            failure_driver = dominant_pressure
        else:
            outcome_class = "bounded_success_high_cost"
            calibration_bucket = "readable_but_costly"
            failure_driver = dominant_pressure
    else:
        calibration_bucket = "bounded_gap"
        if states >= timeout or budget_exhaustion_events > 0:
            outcome_class = "bounded_budget_exhausted"
            failure_driver = "state_budget"
        elif int(puzzle_stall_steps) > 0 or pressures["affordance_reactivation"] >= 0.34:
            outcome_class = "puzzle_readability_failure"
            failure_driver = "puzzle_stall" if int(puzzle_stall_steps) > 0 else "affordance_reactivation"
        elif pressures["confusion"] >= 0.50 or pressures["navigation_entropy"] >= 0.75:
            outcome_class = "navigation_confusion_failure"
            failure_driver = "confusion" if pressures["confusion"] >= pressures["navigation_entropy"] else "navigation_entropy"
        elif pressures["cognitive_load"] >= 0.50:
            outcome_class = "cognitive_load_failure"
            failure_driver = "cognitive_load"
        else:
            outcome_class = "bounded_failure_unclassified"
            failure_driver = dominant_pressure

    return {
        "pcbs_outcome_class": outcome_class,
        "pcbs_calibration_bucket": calibration_bucket,
        "pcbs_failure_driver": failure_driver,
        "pcbs_dominant_pressure": dominant_pressure,
        "pcbs_dominant_pressure_value": float(dominant_value),
        "pcbs_pressure_confusion": pressures["confusion"],
        "pcbs_pressure_navigation_entropy": pressures["navigation_entropy"],
        "pcbs_pressure_cognitive_load": pressures["cognitive_load"],
        "pcbs_pressure_state_budget": pressures["state_budget"],
        "pcbs_pressure_puzzle_stall": pressures["puzzle_stall"],
        "pcbs_pressure_affordance_reactivation": pressures["affordance_reactivation"],
        "pcbs_pressure_focus_switching": pressures["focus_switching"],
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
    calibration_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run the hard oracle and P-CBS on the same sanitized dungeon grid."""
    prepared = prepare_dungeon_grid_for_validation(source)
    grid = prepared.grid
    start = prepared.start
    goal = prepared.goal
    manhattan = abs(int(start[0]) - int(goal[0])) + abs(int(start[1]) - int(goal[1]))
    env_kwargs = _extract_validation_env_kwargs(source)
    calibration_payload = _load_pcbs_calibration_payload(calibration_path)
    calibrated_config = _persona_config_from_payload(
        persona=str(persona),
        payload=calibration_payload,
    )
    bounded_weights, effort_weights, metric_weight_source = _calibration_weight_payload(calibration_payload)
    persona_source = (
        "telemetry_calibrated_persona"
        if calibrated_config is not None
        else "built_in_persona"
    )

    env_astar = ZeldaLogicEnv(semantic_grid=grid.copy(), **copy.deepcopy(env_kwargs))
    oracle = run_astar_oracle(env_astar, timeout=int(timeout_astar))

    env_pcbs = ZeldaLogicEnv(semantic_grid=grid.copy(), **copy.deepcopy(env_kwargs))
    pcbs = PersonaDrivenCognitiveBoundedSearch(
        env_pcbs,
        persona=str(persona),
        timeout=int(timeout_pcbs),
        seed=seed,
        custom_config=calibrated_config,
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
        bounded_rationality_weights=bounded_weights,
        cognitive_effort_weights=effort_weights,
        weight_source=metric_weight_source,
    )
    outcome_metrics = classify_pcbs_outcome(
        oracle=oracle,
        pcbs_success=bool(pcbs_success),
        pcbs_solution_length=pcbs_solution_length,
        pcbs_trajectory_length=pcbs_trajectory_length,
        pcbs_states=int(pcbs_states),
        timeout_pcbs=int(timeout_pcbs),
        pcbs_metrics=pcbs_metrics,
        readability_metrics=readability_metrics,
        puzzle_stall_steps=puzzle_stall_steps,
    )
    pcbs_report_metrics = {**readability_metrics, **outcome_metrics}

    return {
        "validation_handoff": prepared.diagnostics,
        "oracle": {
            **oracle,
            "path_efficiency_ratio": path_efficiency_ratio(int(oracle["path_length"]), manhattan),
        },
        "pcbs": {
            "persona": str(persona),
            "persona_source": persona_source,
            "calibration_path": str(calibration_path) if calibration_path is not None else None,
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
            "loop_escape_events": int(getattr(pcbs_metrics, "loop_escape_events", 0) or 0),
            **pcbs_report_metrics,
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
            "pcbs_outcome_class": str(outcome_metrics["pcbs_outcome_class"]),
            "pcbs_calibration_bucket": str(outcome_metrics["pcbs_calibration_bucket"]),
            "pcbs_failure_driver": str(outcome_metrics["pcbs_failure_driver"]),
            "pcbs_dominant_pressure": str(outcome_metrics["pcbs_dominant_pressure"]),
            "pcbs_dominant_pressure_value": float(outcome_metrics["pcbs_dominant_pressure_value"]),
            "pcbs_persona_source": persona_source,
            "pcbs_metric_interpretation": str(readability_metrics["metric_interpretation"]),
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
            "hard_oracle",
            "none",
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
            str(comparison.get("pcbs_outcome_class") or pcbs.get("pcbs_outcome_class") or ""),
            str(comparison.get("pcbs_failure_driver") or pcbs.get("pcbs_failure_driver") or ""),
        ),
    ]

    lines = [
        f"| Map | Solver | Solved | Path Length | PER | Total Revisits | Navigation Entropy | Goal-Sighting Latency | Status | Outcome | Driver |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for solver, solved, path_len, per, revisits, nav_entropy, aha_latency, status, outcome, driver in rows:
        lines.append(
            f"| {map_name} | {solver} | {int(bool(solved))} | {int(path_len)} | "
            f"{float(per):.3f} | {int(revisits)} | {float(nav_entropy):.3f} | "
            f"{int(aha_latency)} | {status} | {outcome} | {driver} |"
        )
    return "\n".join(lines)


__all__ = [
    "PreparedValidationDungeon",
    "prepare_dungeon_grid_for_validation",
    "count_pcbs_puzzle_stall_steps",
    "PCBS_READABILITY_WEIGHT_SOURCE",
    "PCBS_BOUNDED_RATIONALITY_WEIGHTS",
    "PCBS_COGNITIVE_EFFORT_WEIGHTS",
    "compute_pcbs_readability_metrics",
    "classify_pcbs_outcome",
    "evaluate_astar_vs_pcbs",
    "build_ieee_markdown_table",
]
