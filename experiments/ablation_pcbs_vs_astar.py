"""Compare hard A* solvability against P-CBS cognitive playability metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.common import json_ready, load_room_grids, set_reproducible_seed
from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, TileID
from src.evaluation.pcbs_validation import evaluate_astar_vs_pcbs


def _synthetic_grid() -> np.ndarray:
    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(TileID.WALL), dtype=np.int32)
    grid[1:-1, 1:-1] = int(TileID.FLOOR)
    grid[1, 1] = int(TileID.START)
    grid[-2, -2] = int(TileID.TRIFORCE)
    return grid


def _load_json_grids(path: str | None) -> List[np.ndarray]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    grids = payload.get("grids", payload)
    if not isinstance(grids, list):
        raise ValueError("JSON input must be a list of grids or an object with a 'grids' list.")
    return [np.asarray(grid, dtype=np.int32) for grid in grids]


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(1, len(rows))
    oracle_solved = sum(1 for row in rows if row["oracle"].get("success"))
    pcbs_solved = sum(1 for row in rows if row["pcbs"].get("success"))
    cognitive_gaps = sum(1 for row in rows if row["comparison"].get("cognitive_gap"))
    return {
        "count": len(rows),
        "oracle_solvable_rate": oracle_solved / n,
        "pcbs_solvable_rate": pcbs_solved / n,
        "cognitive_gap_rate": cognitive_gaps / n,
        "mean_pcbs_states_explored": float(np.mean([row["pcbs"].get("states_explored", 0) for row in rows])) if rows else 0.0,
        "mean_pcbs_confusion_index": float(np.mean([row["pcbs"].get("confusion_index", 0.0) for row in rows])) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--timeout-astar", type=int, default=5000)
    parser.add_argument("--timeout-pcbs", type=int, default=1000)
    parser.add_argument("--persona", type=str, default="novice")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/outputs/ablation_pcbs_vs_astar.json")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    if args.dry_run:
        grids = [_synthetic_grid()]
    else:
        grids = _load_json_grids(args.input_json)
        if not grids:
            grids = load_room_grids(args.data_dir, max_samples=args.max_samples)

    rows = []
    for idx, grid in enumerate(grids[: int(args.max_samples)]):
        result = evaluate_astar_vs_pcbs(
            grid,
            persona=args.persona,
            timeout_astar=args.timeout_astar,
            timeout_pcbs=args.timeout_pcbs,
            seed=args.seed + idx,
        )
        rows.append({"index": idx, **result})
    payload = {"config": vars(args), "summary": _summarize(rows), "results": rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
