"""Shared utilities for external PCG baseline experiments.

The baselines in this package intentionally stay outside the core
neuro-symbolic pipeline. They consume the same Zelda room tensors, emit the same
JSON schema, and make every evaluation fallback explicit.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.definitions import SEMANTIC_PALETTE, TileID
from src.evaluation.pcbs_validation import evaluate_astar_vs_pcbs, prepare_dungeon_grid_for_validation

Grid = np.ndarray
GridPos = Tuple[int, int]


@dataclass(frozen=True)
class BaselineEvalConfig:
    """Evaluation options shared by all external baselines."""

    name: str
    seed: int = 42
    timeout_astar: int = 5000
    timeout_pcbs: int = 1000
    run_pcbs: bool = False
    persona: str = "novice"
    terminal_policy: str = "preserve_or_inject_corners"


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch if available."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def finite_float(value: Any, default: float = 0.0) -> float:
    """Convert a metric to a finite float for JSON tables."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def json_ready(value: Any) -> Any:
    """Recursively convert NumPy/scalar values into JSON-safe Python values."""
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_room_grids(
    data_dir: str | Path,
    *,
    max_samples: Optional[int] = None,
    dungeon_ids: Optional[Iterable[int]] = None,
    variants: Optional[Iterable[int]] = None,
) -> List[Grid]:
    """Load canonical Zelda room grids as integer `[H, W]` arrays."""
    from src.zelda_data.zelda_loader import ZeldaRoomDataset

    dataset = ZeldaRoomDataset(
        str(data_dir),
        normalize=False,
        load_graphs=False,
        dungeon_ids=dungeon_ids,
        variants=variants,
    )
    limit = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    grids: List[Grid] = []
    for idx in range(limit):
        item = dataset[idx]
        if hasattr(item, "detach"):
            array = item.detach().cpu().numpy()
        else:
            array = np.asarray(item)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected room grid shape [H,W], got {tuple(array.shape)!r}")
        grids.append(np.rint(array).astype(np.int32, copy=False))
    return grids


def flatten_grids_to_tokens(grids: Sequence[Grid], *, num_tile_ids: int = 44) -> np.ndarray:
    """Convert `[N,H,W]` integer grids into clipped `[N,H*W]` token IDs."""
    if not grids:
        raise ValueError("Cannot tokenize an empty grid collection.")
    arrays = [np.asarray(grid) for grid in grids]
    shape = arrays[0].shape
    if any(array.ndim != 2 or array.shape != shape for array in arrays):
        shapes = [tuple(array.shape) for array in arrays]
        raise ValueError(f"All grids must share one [H,W] shape, got {shapes!r}")
    stacked = np.stack(arrays, axis=0)
    return np.clip(np.rint(stacked), 0, int(num_tile_ids) - 1).astype(np.int64).reshape(len(arrays), -1)


def _valid_tile_ids() -> np.ndarray:
    return np.asarray(sorted({int(v) for v in SEMANTIC_PALETTE.values()}), dtype=np.int32)


def _walkable_mask(grid: Grid) -> np.ndarray:
    walkable = {
        int(TileID.FLOOR),
        int(TileID.DOOR_OPEN),
        int(TileID.START),
        int(TileID.TRIFORCE),
        int(TileID.KEY_SMALL),
        int(TileID.KEY_BOSS),
        int(TileID.KEY_ITEM),
        int(TileID.ITEM_MINOR),
        int(TileID.ELEMENT_FLOOR),
        int(TileID.STAIR),
        int(TileID.PUZZLE),
    }
    return np.isin(grid, list(walkable))


def _nearest_walkable(mask: np.ndarray, target: GridPos, forbidden: Optional[GridPos] = None) -> GridPos:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot place terminals in a grid with no walkable cells.")
    target_arr = np.asarray(target, dtype=np.int64)
    order = np.argsort(np.sum(np.abs(coords - target_arr[None, :]), axis=1))
    for idx in order:
        pos = (int(coords[idx, 0]), int(coords[idx, 1]))
        if forbidden is None or pos != forbidden:
            return pos
    return (int(coords[order[0], 0]), int(coords[order[0], 1]))


def canonicalize_generated_grid(
    grid: Any,
    *,
    terminal_policy: str = "preserve_or_inject_corners",
) -> Tuple[Grid, Dict[str, Any]]:
    """
    Make a generated room valid for the shared hard-oracle/P-CBS validator.

    Generated baselines often omit START/TRIFORCE or emit multiple copies. This
    function preserves one existing terminal when possible and otherwise injects
    terminals at distant walkable cells. The diagnostics must be reported with
    baseline results because terminal injection changes the standalone task.
    """
    array = np.asarray(grid)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected generated grid shape [H,W], got {tuple(array.shape)!r}")

    out = np.nan_to_num(array.astype(np.float32), nan=float(TileID.FLOOR), posinf=float(TileID.WALL), neginf=float(TileID.WALL))
    out = np.rint(out).astype(np.int32, copy=False)
    valid_ids = _valid_tile_ids()
    invalid_mask = ~np.isin(out, valid_ids)
    invalid_count = int(np.sum(invalid_mask))
    if invalid_count:
        out[invalid_mask] = int(TileID.FLOOR)

    if not np.any(_walkable_mask(out)):
        out[:, :] = int(TileID.WALL)
        if out.shape[0] < 2 or out.shape[1] < 2:
            raise ValueError("Generated grid is too small for terminal injection.")
        out[0, 0] = int(TileID.FLOOR)
        out[-1, -1] = int(TileID.FLOOR)

    starts = np.argwhere(out == int(TileID.START))
    goals = np.argwhere(out == int(TileID.TRIFORCE))
    diagnostics: Dict[str, Any] = {
        "invalid_tile_count": invalid_count,
        "terminal_policy": str(terminal_policy),
        "start_injected": False,
        "goal_injected": False,
        "duplicate_start_tiles_removed": max(0, int(starts.shape[0]) - 1) if starts.size else 0,
        "duplicate_goal_tiles_removed": max(0, int(goals.shape[0]) - 1) if goals.size else 0,
    }
    floor_id = int(TileID.FLOOR)

    if starts.size:
        start = (int(starts[0, 0]), int(starts[0, 1]))
    else:
        start = _nearest_walkable(_walkable_mask(out), (0, 0))
        diagnostics["start_injected"] = True

    if goals.size:
        goal = (int(goals[0, 0]), int(goals[0, 1]))
    else:
        goal = _nearest_walkable(_walkable_mask(out), (out.shape[0] - 1, out.shape[1] - 1), forbidden=start)
        diagnostics["goal_injected"] = True

    if start == goal:
        mask = _walkable_mask(out)
        goal = _nearest_walkable(mask, (out.shape[0] - 1, out.shape[1] - 1), forbidden=start)
        if start == goal:
            # Last-resort two-cell carving for degenerate generated samples.
            out[0, 0] = floor_id
            out[-1, -1] = floor_id
            start = (0, 0)
            goal = (out.shape[0] - 1, out.shape[1] - 1)
            diagnostics["terminal_carving_applied"] = True

    out[out == int(TileID.START)] = floor_id
    out[out == int(TileID.TRIFORCE)] = floor_id
    out[start] = int(TileID.START)
    out[goal] = int(TileID.TRIFORCE)
    diagnostics["start"] = [int(start[0]), int(start[1])]
    diagnostics["goal"] = [int(goal[0]), int(goal[1])]
    return out.astype(np.int32, copy=False), diagnostics


def tile_histogram(grid: Grid, *, num_tile_ids: int = 44) -> List[float]:
    counts = np.bincount(np.clip(np.rint(grid).astype(np.int64), 0, num_tile_ids - 1).reshape(-1), minlength=num_tile_ids)
    total = float(max(1, counts.sum()))
    return [float(v / total) for v in counts.tolist()]


def summarize_diversity(generated: Sequence[Grid], reference: Sequence[Grid]) -> Dict[str, Any]:
    """Compute cheap, deterministic diversity metrics for smoke and paper tables."""
    if not generated:
        return {
            "num_samples": 0,
            "unique_ratio": 0.0,
            "mean_pairwise_hamming": 0.0,
            "nearest_reference_hamming": 0.0,
            "tile_histogram_l1_to_reference": 0.0,
        }
    gen_arrays = [np.asarray(grid, dtype=np.int32) for grid in generated]
    ref_arrays = [np.asarray(grid, dtype=np.int32) for grid in reference if np.asarray(grid).shape == gen_arrays[0].shape]
    unique_ratio = len({array.tobytes() for array in gen_arrays}) / float(max(1, len(gen_arrays)))

    pairs: List[float] = []
    pair_limit = min(len(gen_arrays), 64)
    for i in range(pair_limit):
        for j in range(i + 1, pair_limit):
            pairs.append(float(np.mean(gen_arrays[i] != gen_arrays[j])))

    nearest: List[float] = []
    for grid in gen_arrays[:64]:
        if ref_arrays:
            nearest.append(min(float(np.mean(grid != ref)) for ref in ref_arrays[:128]))
    gen_hist = np.asarray([tile_histogram(grid) for grid in gen_arrays], dtype=np.float64).mean(axis=0)
    ref_hist = np.asarray([tile_histogram(grid) for grid in (ref_arrays or gen_arrays)], dtype=np.float64).mean(axis=0)
    return {
        "num_samples": int(len(gen_arrays)),
        "unique_ratio": finite_float(unique_ratio),
        "mean_pairwise_hamming": finite_float(np.mean(pairs) if pairs else 0.0),
        "nearest_reference_hamming": finite_float(np.mean(nearest) if nearest else 0.0),
        "tile_histogram_l1_to_reference": finite_float(np.abs(gen_hist - ref_hist).sum()),
    }


def evaluate_generated_grids(
    generated: Sequence[Grid],
    reference: Sequence[Grid],
    config: BaselineEvalConfig,
) -> Dict[str, Any]:
    """Evaluate generated grids with the shared hard oracle and optional P-CBS."""
    rows: List[Dict[str, Any]] = []
    astar_solved = 0
    pcbs_solved = 0
    terminal_injected = 0
    terminal_normalized = 0
    for idx, raw in enumerate(generated):
        canonical, canonical_diag = canonicalize_generated_grid(
            raw,
            terminal_policy=config.terminal_policy,
        )
        injected = bool(canonical_diag.get("start_injected")) or bool(canonical_diag.get("goal_injected"))
        duplicate_removed = (
            int(canonical_diag.get("duplicate_start_tiles_removed", 0) or 0)
            + int(canonical_diag.get("duplicate_goal_tiles_removed", 0) or 0)
        ) > 0
        terminal_injected += int(injected)
        terminal_normalized += int(injected or duplicate_removed)
        try:
            if config.run_pcbs:
                result = evaluate_astar_vs_pcbs(
                    canonical,
                    persona=config.persona,
                    timeout_astar=config.timeout_astar,
                    timeout_pcbs=config.timeout_pcbs,
                    seed=config.seed + idx,
                )
                astar = result["oracle"]
                pcbs = result["pcbs"]
                astar_solved += int(bool(astar.get("success", False)))
                pcbs_solved += int(bool(pcbs.get("success", False)))
                rows.append(
                    {
                        "index": idx,
                        "canonicalization": canonical_diag,
                        "astar": {
                            "success": bool(astar.get("success", False)),
                            "status": str(astar.get("status", "")),
                            "path_length": int(astar.get("path_length", 0) or 0),
                            "states_explored": int(astar.get("states_explored", 0) or 0),
                        },
                        "pcbs": {
                            "success": bool(pcbs.get("success", False)),
                            "status": str(pcbs.get("status", "")),
                            "path_length": int(pcbs.get("path_length", 0) or 0),
                            "confusion_index": finite_float(pcbs.get("confusion_index", 0.0)),
                            "bounded_rationality_index": finite_float(pcbs.get("bounded_rationality_index", 0.0)),
                        },
                    }
                )
            else:
                prepared = prepare_dungeon_grid_for_validation(canonical)
                from src.evaluation.search_benchmark_utils import run_astar_oracle
                from src.simulation.validator import ZeldaLogicEnv

                astar = run_astar_oracle(
                    ZeldaLogicEnv(semantic_grid=prepared.grid.copy()),
                    timeout=config.timeout_astar,
                )
                astar_solved += int(bool(astar.get("success", False)))
                rows.append(
                    {
                        "index": idx,
                        "canonicalization": canonical_diag,
                        "validation_handoff": prepared.diagnostics,
                        "astar": {
                            "success": bool(astar.get("success", False)),
                            "status": str(astar.get("status", "")),
                            "path_length": int(astar.get("path_length", 0) or 0),
                            "states_explored": int(astar.get("states_explored", 0) or 0),
                        },
                        "pcbs": None,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "index": idx,
                    "canonicalization": canonical_diag,
                    "error": f"{type(exc).__name__}: {exc}",
                    "astar": {"success": False, "status": "error", "path_length": 0, "states_explored": 0},
                    "pcbs": None,
                }
            )

    n = max(1, len(generated))
    return {
        "baseline": config.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(config),
        "aggregate": {
            "num_samples": int(len(generated)),
            "astar_solvable_rate": finite_float(astar_solved / n),
            "pcbs_solvable_rate": finite_float(pcbs_solved / n) if config.run_pcbs else None,
            "pcbs_evaluated": bool(config.run_pcbs),
            "terminal_injection_rate": finite_float(terminal_injected / n),
            "terminal_normalization_rate": finite_float(terminal_normalized / n),
            **summarize_diversity(generated, reference),
        },
        "samples": rows,
    }


def write_json_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    return output
