"""Overlapping-pattern Wave Function Collapse baseline for Zelda rooms."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.common import (
    BaselineEvalConfig,
    evaluate_generated_grids,
    load_room_grids,
    set_reproducible_seed,
    write_json_report,
)
from src.core.definitions import TileID

Direction = Tuple[int, int]
DIRECTIONS: Tuple[Direction, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class OverlappingPatternModel:
    pattern_size: int
    patterns: np.ndarray
    weights: np.ndarray
    compatible: Dict[Tuple[int, int], List[Set[int]]]

    @property
    def num_patterns(self) -> int:
        return int(self.patterns.shape[0])


def _overlap_compatible(a: np.ndarray, b: np.ndarray, direction: Direction) -> bool:
    dr, dc = direction
    if dr == -1:
        return bool(np.array_equal(a[:-1, :], b[1:, :]))
    if dr == 1:
        return bool(np.array_equal(a[1:, :], b[:-1, :]))
    if dc == -1:
        return bool(np.array_equal(a[:, :-1], b[:, 1:]))
    if dc == 1:
        return bool(np.array_equal(a[:, 1:], b[:, :-1]))
    raise ValueError(direction)


def build_overlapping_pattern_model(grids: Sequence[np.ndarray], *, pattern_size: int = 3) -> OverlappingPatternModel:
    """Extract overlapping `N x N` tile patterns and legal overlaps."""
    n = int(pattern_size)
    if n < 2:
        raise ValueError("pattern_size must be at least 2 for overlapping WFC.")
    counts: Dict[bytes, int] = {}
    pattern_by_key: Dict[bytes, np.ndarray] = {}
    for grid in grids:
        arr = np.asarray(grid, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[0] < n or arr.shape[1] < n:
            continue
        for row in range(arr.shape[0] - n + 1):
            for col in range(arr.shape[1] - n + 1):
                pattern = arr[row : row + n, col : col + n].copy()
                key = pattern.tobytes()
                counts[key] = counts.get(key, 0) + 1
                pattern_by_key[key] = pattern
    if not counts:
        raise ValueError("No WFC patterns could be extracted from the reference grids.")

    keys = sorted(counts, key=lambda key: (-counts[key], key))
    patterns = np.stack([pattern_by_key[key] for key in keys], axis=0).astype(np.int32, copy=False)
    weights = np.asarray([counts[key] for key in keys], dtype=np.float64)
    weights = weights / max(1.0, float(weights.sum()))
    compatible: Dict[Tuple[int, int], List[Set[int]]] = {}
    for direction in DIRECTIONS:
        compatible[direction] = []
        for i in range(patterns.shape[0]):
            allowed = {
                j
                for j in range(patterns.shape[0])
                if _overlap_compatible(patterns[i], patterns[j], direction)
            }
            compatible[direction].append(allowed)
    return OverlappingPatternModel(pattern_size=n, patterns=patterns, weights=weights, compatible=compatible)


def _propagate(
    domains: List[List[Set[int]]],
    *,
    row: int,
    col: int,
    model: OverlappingPatternModel,
) -> bool:
    height = len(domains)
    width = len(domains[0])
    queue = [(row, col)]
    while queue:
        cur_r, cur_c = queue.pop(0)
        source_domain = domains[cur_r][cur_c]
        for dr, dc in DIRECTIONS:
            nr, nc = cur_r + dr, cur_c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            allowed: Set[int] = set()
            for pattern_idx in source_domain:
                allowed.update(model.compatible[(dr, dc)][pattern_idx])
            new_domain = domains[nr][nc] & allowed
            if not new_domain:
                return False
            if len(new_domain) < len(domains[nr][nc]):
                domains[nr][nc] = new_domain
                queue.append((nr, nc))
    return True


def _sample_from_domain(domain: Iterable[int], weights: np.ndarray, rng: random.Random) -> int:
    choices = sorted(int(v) for v in domain)
    probs = np.asarray([float(weights[idx]) for idx in choices], dtype=np.float64)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return int(rng.choice(choices))
    probs = probs / total
    return int(rng.choices(choices, weights=probs.tolist(), k=1)[0])


def collapse_patterns(
    model: OverlappingPatternModel,
    *,
    output_shape: Tuple[int, int],
    seed: int,
    max_steps: int = 10000,
) -> np.ndarray:
    """Run local WFC collapse and reconstruct a tile grid."""
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    n = int(model.pattern_size)
    grid_h = max(1, out_h - n + 1)
    grid_w = max(1, out_w - n + 1)
    rng = random.Random(int(seed))
    all_patterns = set(range(model.num_patterns))
    domains: List[List[Set[int]]] = [[set(all_patterns) for _ in range(grid_w)] for _ in range(grid_h)]

    for _step in range(int(max_steps)):
        unresolved = [
            (len(domains[r][c]), r, c)
            for r in range(grid_h)
            for c in range(grid_w)
            if len(domains[r][c]) > 1
        ]
        if not unresolved:
            break
        _, row, col = min(unresolved, key=lambda item: (item[0], rng.random()))
        chosen = _sample_from_domain(domains[row][col], model.weights, rng)
        domains[row][col] = {chosen}
        if not _propagate(domains, row=row, col=col, model=model):
            raise RuntimeError("WFC contradiction during propagation.")

    if any(len(domains[r][c]) != 1 for r in range(grid_h) for c in range(grid_w)):
        raise RuntimeError("WFC failed to fully collapse within max_steps.")

    out = np.full((out_h, out_w), int(TileID.FLOOR), dtype=np.int32)
    votes: List[List[List[int]]] = [[[] for _ in range(out_w)] for _ in range(out_h)]
    for r in range(grid_h):
        for c in range(grid_w):
            pattern_idx = next(iter(domains[r][c]))
            pattern = model.patterns[pattern_idx]
            for pr in range(n):
                for pc in range(n):
                    gr, gc = r + pr, c + pc
                    if gr < out_h and gc < out_w:
                        votes[gr][gc].append(int(pattern[pr, pc]))
    for r in range(out_h):
        for c in range(out_w):
            if votes[r][c]:
                values, counts = np.unique(np.asarray(votes[r][c], dtype=np.int32), return_counts=True)
                out[r, c] = int(values[int(np.argmax(counts))])
    return out


def generate_wfc_samples(
    model: OverlappingPatternModel,
    *,
    output_shape: Tuple[int, int],
    num_samples: int,
    seed: int,
    retries: int = 20,
) -> List[np.ndarray]:
    samples: List[np.ndarray] = []
    for sample_idx in range(int(num_samples)):
        last_error: Exception | None = None
        for attempt in range(int(retries)):
            try:
                samples.append(
                    collapse_patterns(
                        model,
                        output_shape=output_shape,
                        seed=int(seed) + sample_idx * 1009 + attempt,
                    )
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"WFC failed after {retries} retries for sample {sample_idx}: {last_error}")
    return samples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the overlapping-pattern WFC Zelda baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("Data/The Legend of Zelda"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines/wfc"))
    parser.add_argument("--num-generate", type=int, default=1000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--pattern-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-pcbs", action="store_true")
    args = parser.parse_args(argv)

    set_reproducible_seed(args.seed)
    reference = load_room_grids(args.data_dir, max_samples=args.max_train_samples)
    if not reference:
        raise RuntimeError("No Zelda room grids were loaded.")
    model = build_overlapping_pattern_model(reference, pattern_size=args.pattern_size)
    num_generate = 4 if args.dry_run else int(args.num_generate)
    generated = generate_wfc_samples(
        model,
        output_shape=tuple(reference[0].shape),
        num_samples=num_generate,
        seed=args.seed,
    )
    report = evaluate_generated_grids(
        generated,
        reference,
        BaselineEvalConfig(
            name="wfc_overlapping_patterns",
            seed=args.seed,
            run_pcbs=bool(args.run_pcbs and not args.dry_run),
        ),
    )
    report["wfc"] = {
        "pattern_size": int(args.pattern_size),
        "num_patterns": int(model.num_patterns),
        "dry_run": bool(args.dry_run),
    }
    output_path = write_json_report(args.output_dir / "wfc_baseline_report.json", report)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
