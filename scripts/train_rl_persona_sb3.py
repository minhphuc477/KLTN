#!/usr/bin/env python
"""Train a reproducible SB3 PPO persona against the canonical headless logic env."""

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
from src.evaluation.rl_playtester import HeadlessZeldaPersonaEnv


def _default_grid() -> np.ndarray:
    grid = np.full((16, 11), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int64)
    grid[1:-1, 1:-1] = int(SEMANTIC_PALETTE["FLOOR"])
    grid[1, 1] = int(SEMANTIC_PALETTE["START"])
    grid[-2, -2] = int(SEMANTIC_PALETTE["TRIFORCE"])
    grid[5, 5] = int(SEMANTIC_PALETTE["KEY_SMALL"])
    grid[8, 5] = int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    return grid


def train(args: argparse.Namespace) -> dict[str, object]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is required for this optional RL ablation. "
            "Install stable-baselines3 and gymnasium in the experiment environment."
        ) from exc

    grid = np.load(args.grid) if args.grid else _default_grid()

    def _make_env():
        return Monitor(
            HeadlessZeldaPersonaEnv(
                grid,
                persona=args.persona,
                observation_mode=args.observation_mode,
                max_steps=args.max_episode_steps,
            )
        )

    environment = DummyVecEnv([_make_env])
    model = PPO(
        # Canonical rooms are only 16x11, smaller than NatureCNN's receptive
        # field assumptions. MlpPolicy safely flattens either observation mode.
        "MlpPolicy",
        environment,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.rollout_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=1,
        device=args.device,
    )
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar)
        mean_reward, reward_std = evaluate_policy(
            model,
            environment,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output))
    finally:
        environment.close()
    report = {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "persona": args.persona,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "mean_eval_reward": float(mean_reward),
        "std_eval_reward": float(reward_std),
        "model": str(args.output.with_suffix(".zip")),
    }
    report_path = args.output.with_suffix(".json")
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--persona", choices=("speedrunner", "explorer", "combatant", "cautious"), default="speedrunner")
    parser.add_argument("--observation-mode", choices=("vector", "grid"), default="vector")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, sort_keys=True))
