# ruff: noqa: E402
"""Run role-separated search benchmarks for Zelda validation.

This script implements the search work from
``docs/SEARCH_ALGORITHM_AUDIT_AND_RECOMMENDATION_2026_04_18.md``:

* full game-state A* and Dijkstra are reported as validation/exact solvers;
* P-CBS personas are reported as bounded-player simulations;
* Bidirectional A*, D* Lite, BFS, DFS, and Greedy are marked as diagnostics;
* checkpoint-backed learned guidance is evaluated only as an equal-f A* tie-break ablation;
* JPS is only benchmarked on static binary grids where its assumptions hold.

The output is intentionally role-tagged so downstream tables cannot silently
treat a diagnostic speed comparison as a Zelda playability oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.grid_solvers import astar as grid_astar
from bench.grid_solvers import jps as grid_jps
from bench.suite import corridor_grid, maze_grid, synthetic_open_grid
from src.core.definitions import CHAR_TO_SEMANTIC, SEMANTIC_PALETTE
from src.simulation.cognitive_bounded_search import solve_with_pcbs
from src.simulation.search_base import GameStateSearchConfig
from src.simulation.search_factory import (
    VALIDATION_EXCLUDED_ALGORITHMS,
    environment_requires_full_state_oracle,
    iter_game_state_algorithm_specs,
    recommended_game_state_algorithm_specs,
    run_game_state_solver,
)
from src.simulation.validator import (
    BLOCKING_IDS,
    CONDITIONAL_IDS,
    PUSHABLE_IDS,
    WATER_IDS,
    ZeldaLogicEnv,
)


ROLE_FIELDS = [
    "map_id",
    "solver_key",
    "solver_label",
    "family",
    "validation_role",
    "canonical_use",
    "success",
    "path_length",
    "states_explored",
    "time_sec",
    "fallback_used",
    "status",
    "failure_reason",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    return value


def _semantic_int(value: str) -> int:
    return int(CHAR_TO_SEMANTIC.get(value, SEMANTIC_PALETTE["VOID"]))


def load_semantic_grid(path: Path) -> np.ndarray:
    """Load a semantic grid from npy, json, csv, or VGLC-like text."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        grid_payload = payload.get("grid", payload) if isinstance(payload, dict) else payload
        return np.asarray(grid_payload, dtype=np.int64)
    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if row:
                    rows.append([int(cell) for cell in row])
        return np.asarray(rows, dtype=np.int64)

    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise ValueError(f"No grid rows found in {path}")
    if all(all(ch.lstrip("-").isdigit() or ch in ", \t" for ch in line) for line in lines):
        rows = []
        for line in lines:
            sep = "," if "," in line else None
            rows.append([int(cell) for cell in line.split(sep) if str(cell).strip()])
        return np.asarray(rows, dtype=np.int64)
    width = max(len(line) for line in lines)
    return np.asarray(
        [[_semantic_int(line[col]) if col < len(line) else SEMANTIC_PALETTE["VOID"] for col in range(width)] for line in lines],
        dtype=np.int64,
    )


def _empty_room(height: int = 11, width: int = 16) -> np.ndarray:
    grid = np.full((height, width), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int64)
    grid[0, :] = int(SEMANTIC_PALETTE["WALL"])
    grid[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    grid[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    grid[:, -1] = int(SEMANTIC_PALETTE["WALL"])
    return grid


def synthetic_semantic_maps() -> Dict[str, np.ndarray]:
    """Small maps for smoke checks; not a substitute for final experiments."""
    static = _empty_room()
    static[1, 1] = int(SEMANTIC_PALETTE["START"])
    static[-2, -2] = int(SEMANTIC_PALETTE["TRIFORCE"])

    locked = _empty_room()
    locked[1, 1] = int(SEMANTIC_PALETTE["START"])
    locked[1, 3] = int(SEMANTIC_PALETTE["KEY_SMALL"])
    locked[1, 7] = int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    locked[1, 12] = int(SEMANTIC_PALETTE["TRIFORCE"])
    locked[2:8, 6] = int(SEMANTIC_PALETTE["WALL"])
    locked[1, 6] = int(SEMANTIC_PALETTE["FLOOR"])

    detour = _empty_room()
    detour[1, 1] = int(SEMANTIC_PALETTE["START"])
    detour[9, 14] = int(SEMANTIC_PALETTE["TRIFORCE"])
    detour[4, 2:14] = int(SEMANTIC_PALETTE["WALL"])
    detour[4, 13] = int(SEMANTIC_PALETTE["FLOOR"])
    return {
        "synthetic_static_open": static,
        "synthetic_locked_key": locked,
        "synthetic_static_detour": detour,
    }


def load_input_maps(paths: Sequence[Path], synthetic_smoke: bool) -> Dict[str, np.ndarray]:
    maps: Dict[str, np.ndarray] = {}
    if synthetic_smoke or not paths:
        maps.update(synthetic_semantic_maps())
    for path in paths:
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in {".npy", ".json", ".csv", ".txt", ".vglc"}:
                    maps[child.stem] = load_semantic_grid(child)
        else:
            maps[path.stem] = load_semantic_grid(path)
    return maps


def _path_length(path: Sequence[Tuple[int, int]]) -> int:
    return max(0, len(path) - 1)


def _result_status(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("oracle_status") or metadata.get("solver_status") or metadata.get("status") or "")


def run_game_state_rows(
    map_id: str,
    grid: np.ndarray,
    config: GameStateSearchConfig,
    include_diagnostics: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    env_for_selection = ZeldaLogicEnv(grid.copy())
    specs = recommended_game_state_algorithm_specs(env_for_selection, include_diagnostics=include_diagnostics)
    if config.learned_heuristic_model_path:
        learned_spec = next(
            spec
            for spec in iter_game_state_algorithm_specs()
            if spec.key == "learned_tiebreak_astar"
        )
        specs = list(specs) + [learned_spec]
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        env = ZeldaLogicEnv(grid.copy())
        start = time.perf_counter()
        result = run_game_state_solver(env, int(spec.index), config)
        elapsed = time.perf_counter() - start
        metadata = dict(result.metadata or {})
        rows.append(
            {
                "map_id": map_id,
                "solver_key": spec.key,
                "solver_label": spec.label,
                "family": "game_state_search",
                "validation_role": spec.validation_role,
                "canonical_use": spec.canonical_use,
                "success": bool(result.success),
                "path_length": _path_length(result.path),
                "states_explored": int(result.states_explored),
                "time_sec": float(elapsed),
                "fallback_used": bool(metadata.get("fallback_used", False)),
                "status": _result_status(metadata),
                "failure_reason": str(metadata.get("failure_reason", "") or ""),
                "metadata": metadata,
            }
        )
    selection = {
        "requires_full_state_oracle": environment_requires_full_state_oracle(env_for_selection),
        "selected_solver_keys": [spec.key for spec in specs],
        "excluded_algorithms": dict(VALIDATION_EXCLUDED_ALGORITHMS),
    }
    return rows, selection


def run_pcbs_rows(
    map_id: str,
    grid: np.ndarray,
    personas: Iterable[str],
    timeout: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for persona in personas:
        persona = str(persona).strip().lower()
        if not persona:
            continue
        start = time.perf_counter()
        try:
            success, path, states, metrics = solve_with_pcbs(
                grid.copy(),
                persona=persona,
                timeout=timeout,
                seed=seed,
            )
            failure_reason = ""
            metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else {}
        except Exception as exc:  # keep the benchmark table complete without hiding the failure
            success, path, states = False, [], 0
            failure_reason = f"{type(exc).__name__}: {exc}"
            metrics_dict = {}
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "map_id": map_id,
                "solver_key": f"pcbs_{persona}",
                "solver_label": f"P-CBS {persona.title()}",
                "family": "bounded_player_simulation",
                "validation_role": "human_like_diagnostic",
                "canonical_use": "persona_matched_budget_ablation",
                "success": bool(success),
                "path_length": _path_length(path),
                "states_explored": int(states),
                "time_sec": float(elapsed),
                "fallback_used": False,
                "status": "success" if success else "failed",
                "failure_reason": failure_reason,
                "metadata": {"persona": persona, "metrics": metrics_dict},
            }
        )
    return rows


def _semantic_to_static_binary(grid: np.ndarray) -> List[List[int]]:
    blocked = set(int(value) for value in BLOCKING_IDS | CONDITIONAL_IDS | PUSHABLE_IDS | WATER_IDS)
    return [[1 if int(value) in blocked else 0 for value in row] for row in np.asarray(grid)]


def _find_tile(grid: np.ndarray, tile_id: int, default: Tuple[int, int]) -> Tuple[int, int]:
    found = np.argwhere(np.asarray(grid) == int(tile_id))
    if found.size:
        return int(found[0][0]), int(found[0][1])
    return default


def run_static_grid_rows(map_id: str, grid: np.ndarray) -> List[Dict[str, Any]]:
    """Run A*/JPS only when Zelda-state mechanics are absent."""
    env = ZeldaLogicEnv(grid.copy())
    if environment_requires_full_state_oracle(env):
        return [
            {
                "map_id": map_id,
                "solver_key": "static_grid_jps",
                "solver_label": "JPS",
                "family": "static_grid_speed_ablation",
                "validation_role": "skipped_stateful_map",
                "canonical_use": "not_valid_for_zeldalike_oracle",
                "success": False,
                "path_length": 0,
                "states_explored": 0,
                "time_sec": 0.0,
                "fallback_used": False,
                "status": "skipped",
                "failure_reason": "Map contains inventory, conditional, graph, puzzle, or push-block mechanics.",
                "metadata": {"assumption": "static_uniform_grid_only"},
            }
        ]

    binary = _semantic_to_static_binary(grid)
    h, w = grid.shape
    start_pos = _find_tile(grid, int(SEMANTIC_PALETTE["START"]), (1, 1))
    goal_pos = _find_tile(grid, int(SEMANTIC_PALETTE["TRIFORCE"]), (h - 2, w - 2))
    rows = []
    for allow_diagonal in (False, True):
        for solver_key, solver_func in (("static_astar", grid_astar), ("static_jps", grid_jps)):
            start_time = time.perf_counter()
            if solver_key == "static_astar":
                path, nodes = solver_func(binary, start_pos, goal_pos, allow_diagonal=allow_diagonal)
            else:
                result = solver_func(binary, start_pos, goal_pos, allow_diagonal=allow_diagonal)
                path, nodes = result[:2]
            elapsed = time.perf_counter() - start_time
            rows.append(
                {
                    "map_id": map_id,
                    "solver_key": f"{solver_key}_{'diag' if allow_diagonal else 'cardinal'}",
                    "solver_label": "Static A*" if solver_key == "static_astar" else "JPS",
                    "family": "static_grid_speed_ablation",
                    "validation_role": "static_grid_diagnostic",
                    "canonical_use": "not_valid_for_zeldalike_oracle",
                    "success": bool(path),
                    "path_length": _path_length(path or []),
                    "states_explored": int(nodes),
                    "time_sec": float(elapsed),
                    "fallback_used": False,
                    "status": "success" if path else "failed",
                    "failure_reason": "",
                    "metadata": {
                        "allow_diagonal": bool(allow_diagonal),
                        "assumption": "static_uniform_grid_only",
                    },
                }
            )
    return rows


def run_synthetic_static_grid_ablation() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    maps = {
        "bench_open_64_low_obstacle": synthetic_open_grid(64, 0.05, seed=11),
        "bench_open_64_medium_obstacle": synthetic_open_grid(64, 0.15, seed=12),
        "bench_corridor_64_spacing_4": corridor_grid(64, spacing=4),
        "bench_maze_64_seed_13": maze_grid(64, seed=13),
    }
    for map_id, binary in maps.items():
        binary[0][0] = 0
        binary[-1][-1] = 0
        for allow_diagonal in (False, True):
            for solver_key, solver_func in (("static_astar", grid_astar), ("static_jps", grid_jps)):
                start = (0, 0)
                goal = (len(binary) - 1, len(binary[0]) - 1)
                start_time = time.perf_counter()
                if solver_key == "static_astar":
                    path, nodes = solver_func(binary, start, goal, allow_diagonal=allow_diagonal)
                else:
                    result = solver_func(binary, start, goal, allow_diagonal=allow_diagonal)
                    path, nodes = result[:2]
                elapsed = time.perf_counter() - start_time
                rows.append(
                    {
                        "map_id": map_id,
                        "solver_key": f"{solver_key}_{'diag' if allow_diagonal else 'cardinal'}",
                        "solver_label": "Static A*" if solver_key == "static_astar" else "JPS",
                        "family": "static_grid_speed_ablation",
                        "validation_role": "static_grid_diagnostic",
                        "canonical_use": "not_valid_for_zeldalike_oracle",
                        "success": bool(path),
                        "path_length": _path_length(path or []),
                        "states_explored": int(nodes),
                        "time_sec": float(elapsed),
                        "fallback_used": False,
                        "status": "success" if path else "failed",
                        "failure_reason": "",
                        "metadata": {
                            "allow_diagonal": bool(allow_diagonal),
                            "assumption": "static_uniform_grid_only",
                        },
                    }
                )
    return rows


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ROLE_FIELDS + ["metadata_json"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row.get(field, "") for field in ROLE_FIELDS}
            flat["metadata_json"] = json.dumps(_json_sanitize(row.get("metadata", {})), sort_keys=True, default=_json_default)
            writer.writerow(flat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="*", type=Path, default=[], help="Grid files or directories to benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/search_role_benchmark"))
    parser.add_argument("--timeout", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Add tiny synthetic maps for code-path verification.")
    parser.add_argument("--include-static-grid-ablation", action="store_true")
    parser.add_argument(
        "--pcbs-personas",
        default="novice,balanced,expert",
        help="Comma-separated P-CBS personas. Use an empty string to skip P-CBS.",
    )
    parser.add_argument("--allow-diagonals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rules-profile", default="vglc_strict")
    parser.add_argument("--max-depth", type=int, default=500)
    parser.add_argument(
        "--learned-heuristic-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional HeuristicTrainer checkpoint. Adds A* with neural equal-f "
            "tie-breaking; it never replaces the canonical oracle row."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    maps = load_input_maps(args.input, synthetic_smoke=bool(args.synthetic_smoke))
    if not maps:
        raise SystemExit("No input maps found. Pass --input or --synthetic-smoke.")

    config = GameStateSearchConfig(
        timeout=int(args.timeout),
        allow_diagonals=bool(args.allow_diagonals),
        rules_profile=str(args.rules_profile),
        max_depth=int(args.max_depth),
        learned_heuristic_model_path=(
            str(args.learned_heuristic_checkpoint)
            if args.learned_heuristic_checkpoint is not None
            else None
        ),
    )
    personas = [part.strip() for part in str(args.pcbs_personas).split(",") if part.strip()]

    all_rows: List[Dict[str, Any]] = []
    selections: Dict[str, Dict[str, Any]] = {}
    for map_id, grid in maps.items():
        game_rows, selection = run_game_state_rows(
            map_id,
            np.asarray(grid, dtype=np.int64),
            config,
            include_diagnostics=bool(args.include_diagnostics),
        )
        all_rows.extend(game_rows)
        selections[map_id] = selection
        if personas:
            all_rows.extend(run_pcbs_rows(map_id, np.asarray(grid, dtype=np.int64), personas, int(args.timeout), int(args.seed)))
        if args.include_static_grid_ablation:
            all_rows.extend(run_static_grid_rows(map_id, np.asarray(grid, dtype=np.int64)))

    if args.include_static_grid_ablation:
        all_rows.extend(run_synthetic_static_grid_ablation())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "search_role_benchmark.csv"
    json_path = args.output_dir / "search_role_benchmark.json"
    _write_rows_csv(csv_path, all_rows)
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "timeout": int(args.timeout),
                    "seed": int(args.seed),
                    "include_diagnostics": bool(args.include_diagnostics),
                    "include_static_grid_ablation": bool(args.include_static_grid_ablation),
                    "pcbs_personas": personas,
                    "rules_profile": str(args.rules_profile),
                    "allow_diagonals": bool(args.allow_diagonals),
                    "max_depth": int(args.max_depth),
                    "learned_heuristic_checkpoint": (
                        str(args.learned_heuristic_checkpoint)
                        if args.learned_heuristic_checkpoint is not None
                        else None
                    ),
                },
                "selection": selections,
                "rows": _json_sanitize(all_rows),
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_rows)} rows to {csv_path}")
    print(f"Wrote JSON artifact to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
