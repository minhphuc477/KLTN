#!/usr/bin/env python
"""Benchmark deterministic headless Zelda RL transitions without PyGame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.definitions import SEMANTIC_PALETTE
from src.evaluation.rl_playtester import HeadlessZeldaPersonaEnv, benchmark_headless_steps


def _default_grid() -> np.ndarray:
    grid = np.full((16, 11), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int64)
    grid[1:-1, 1:-1] = int(SEMANTIC_PALETTE["FLOOR"])
    grid[1, 1] = int(SEMANTIC_PALETTE["START"])
    grid[-2, -2] = int(SEMANTIC_PALETTE["TRIFORCE"])
    grid[5, 5] = int(SEMANTIC_PALETTE["KEY_SMALL"])
    grid[8, 5] = int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    return grid


def run(args: argparse.Namespace) -> dict[str, float]:
    grid = np.load(args.grid) if args.grid is not None else _default_grid()
    env = HeadlessZeldaPersonaEnv(
        grid,
        persona=args.persona,
        observation_mode=args.observation_mode,
        max_steps=args.max_episode_steps,
    )
    try:
        first, _ = env.reset(seed=args.seed)
        second, _ = env.reset(seed=args.seed)
        if not np.array_equal(first, second):
            raise RuntimeError("Environment reset is not deterministic for the same seed.")
        metrics = benchmark_headless_steps(env, steps=args.steps, seed=args.seed)
    finally:
        env.close()
    metrics["meets_target"] = float(metrics["elapsed_seconds"] <= float(args.target_seconds))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        tmp.replace(args.output)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=None, help="Optional .npy semantic grid.")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--target-seconds", type=float, default=2.0)
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
    parser.add_argument("--persona", choices=["speedrunner", "explorer", "combatant", "cautious"], default="speedrunner")
    parser.add_argument("--observation-mode", choices=["vector", "grid"], default="vector")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/headless_rl_benchmark.json"))
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
