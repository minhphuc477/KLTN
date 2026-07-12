"""Deterministic headless RL adapter over the canonical Zelda logic engine."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv

try:  # Optional training dependency.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - depends on optional package
    gym = None
    spaces = None


@dataclass(frozen=True)
class PersonaReward:
    step_penalty: float
    goal_reward: float
    new_tile_reward: float = 0.0
    enemy_reward: float = 0.0
    blocked_penalty: float = 0.0

    @classmethod
    def from_name(cls, name: str) -> "PersonaReward":
        normalized = str(name).strip().lower()
        if normalized in {"explorer", "explore"}:
            return cls(step_penalty=-0.1, goal_reward=100.0, new_tile_reward=10.0, blocked_penalty=-0.25)
        if normalized in {"combatant", "combat"}:
            return cls(step_penalty=-0.1, goal_reward=100.0, enemy_reward=5.0, blocked_penalty=-0.25)
        if normalized in {"cautious", "safe"}:
            return cls(step_penalty=-0.2, goal_reward=100.0, new_tile_reward=0.2, blocked_penalty=-1.0)
        return cls(step_penalty=-1.0, goal_reward=100.0, blocked_penalty=-0.5)


_GymBase = gym.Env if gym is not None else object


class HeadlessZeldaPersonaEnv(_GymBase):
    """Gymnasium-compatible wrapper with four moves plus one transition action."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        semantic_grid: np.ndarray,
        *,
        persona: str = "speedrunner",
        observation_mode: str = "vector",
        max_steps: int = 10_000,
        graph: Optional[Any] = None,
        room_to_node: Optional[Mapping[Any, Any]] = None,
        room_positions: Optional[Mapping[Any, Tuple[int, int]]] = None,
        node_to_room: Optional[Mapping[Any, Any]] = None,
        room_puzzle_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if observation_mode not in {"vector", "grid"}:
            raise ValueError("observation_mode must be 'vector' or 'grid'.")
        self.semantic_grid = np.asarray(semantic_grid, dtype=np.int64).copy()
        self.observation_mode = observation_mode
        self.reward_config = PersonaReward.from_name(persona)
        self.max_steps = int(max(1, max_steps))
        self._env_kwargs = {
            "graph": graph,
            "room_to_node": dict(room_to_node or {}),
            "room_positions": dict(room_positions or {}),
            "node_to_room": dict(node_to_room or {}),
            "room_puzzle_metadata": dict(room_puzzle_metadata or {}),
        }
        self.logic_env = ZeldaLogicEnv(self.semantic_grid, **self._env_kwargs)
        self.logic_env.max_steps = self.max_steps
        self._transition_helper = StateSpaceAStar(self.logic_env, timeout=1)
        self._visited: set[Tuple[int, int]] = set()
        self._rng = np.random.default_rng(0)
        if spaces is not None:
            self.action_space = spaces.Discrete(5)
            if observation_mode == "grid":
                self.observation_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3, self.logic_env.height, self.logic_env.width),
                    dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    def _observation(self) -> np.ndarray:
        state = self.logic_env.state
        row, col = state.position
        goal = self.logic_env.goal_pos or state.position
        if self.observation_mode == "grid":
            tile_scale = float(max(1, max(int(value) for value in SEMANTIC_PALETTE.values())))
            tiles = self.logic_env.grid.astype(np.float32) / tile_scale
            agent = np.zeros_like(tiles, dtype=np.float32)
            target = np.zeros_like(tiles, dtype=np.float32)
            agent[int(row), int(col)] = 1.0
            target[int(goal[0]), int(goal[1])] = 1.0
            return np.stack([tiles, agent, target], axis=0)
        height = float(max(1, self.logic_env.height - 1))
        width = float(max(1, self.logic_env.width - 1))
        return np.asarray(
            [
                (2.0 * float(row) / height) - 1.0,
                (2.0 * float(col) / width) - 1.0,
                float(goal[0] - row) / height,
                float(goal[1] - col) / width,
                min(1.0, float(state.keys) / 5.0),
                min(1.0, float(state.bomb_count) / 5.0),
                float(bool(state.has_boss_key)),
                float(bool(state.has_item)),
                max(-1.0, min(1.0, float(state.current_floor) / 5.0)),
                min(1.0, float(len(state.opened_doors)) / 10.0),
                min(1.0, float(len(state.collected_items)) / 10.0),
                min(1.0, float(self.logic_env.step_count) / float(self.max_steps)),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        del options
        if gym is not None:
            super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        state = self.logic_env.reset()
        self._visited = {tuple(state.position)}
        return self._observation(), {"position": tuple(state.position)}

    def _transition(self) -> tuple[float, bool, Dict[str, Any]]:
        state = self.logic_env.state
        candidates: list[tuple[Tuple[int, int], Optional[str]]] = []
        if int(self.logic_env.grid[state.position]) == int(SEMANTIC_PALETTE["STAIR"]):
            candidates.extend((position, "stair") for position in self._transition_helper.get_stair_destinations(state.position))
        candidates.extend(
            (position, edge_type)
            for position, _cost, edge_type in self._transition_helper.get_controlled_virtual_destinations(state.position, state)
        )
        candidates.extend(
            (position, edge_type)
            for position, _cost, edge_type in self._transition_helper.get_graph_warp_destinations(state.position, state)
        )
        if not candidates:
            return -1.0, False, {"msg": "No transition available"}
        destination, edge_type = sorted(set(candidates), key=lambda item: (item[0], str(item[1])))[0]
        transition_state = state
        if edge_type not in {None, "stair"}:
            allowed, transition_state = self._transition_helper.apply_graph_edge_transition(
                state,
                state.position,
                destination,
                str(edge_type),
            )
            if not allowed:
                return -1.0, False, {"msg": "Transition requirements not met"}
        self.logic_env.state = transition_state
        tile = int(self.logic_env.grid[destination])
        allowed, new_state, reward, info = self.logic_env.try_move(destination, tile)
        if not allowed:
            self.logic_env.state = state
            return -1.0, False, dict(info)
        new_state.current_floor = self.logic_env.floor_for_position(destination, default=state.current_floor)
        self.logic_env.state = new_state
        won = destination == self.logic_env.goal_pos
        self.logic_env.done = bool(won)
        self.logic_env.won = bool(won)
        return float(reward), bool(won), {**dict(info), "transition": str(edge_type)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        if action < 0 or action >= 5:
            raise ValueError(f"Action must be in [0, 4], got {action}.")
        previous = tuple(self.logic_env.state.position)
        target_tile = None
        if action == 4:
            if self.logic_env.done:
                return self._observation(), 0.0, bool(self.logic_env.won), not bool(self.logic_env.won), {
                    "msg": "Episode already done",
                    "position": previous,
                    "unique_tiles": len(self._visited),
                }
            self.logic_env.step_count += 1
            base_reward, terminated, info = self._transition()
        else:
            state, base_reward, done, info = self.logic_env.step(action)
            terminated = bool(done and self.logic_env.won)
            target_tile = int(self.logic_env.original_grid[state.position])
        current = tuple(self.logic_env.state.position)
        first_visit = current not in self._visited
        self._visited.add(current)
        moved = current != previous
        reward = float(self.reward_config.step_penalty)
        if terminated:
            reward += float(self.reward_config.goal_reward)
        if first_visit:
            reward += float(self.reward_config.new_tile_reward)
        if not moved:
            reward += float(self.reward_config.blocked_penalty)
        if moved and target_tile in {int(SEMANTIC_PALETTE["ENEMY"]), int(SEMANTIC_PALETTE["BOSS"])}:
            reward += float(self.reward_config.enemy_reward)
        reward += 0.01 * float(base_reward)
        truncated = bool(not terminated and self.logic_env.step_count >= self.max_steps)
        return self._observation(), reward, terminated, truncated, {
            **dict(info),
            "position": current,
            "unique_tiles": len(self._visited),
        }

    def close(self) -> None:
        self.logic_env.close()


def benchmark_headless_steps(
    environment: HeadlessZeldaPersonaEnv,
    *,
    steps: int = 100_000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    environment.reset(seed=seed)
    started = time.perf_counter()
    resets = 0
    for _ in range(int(max(1, steps))):
        _observation, _reward, terminated, truncated, _info = environment.step(int(rng.integers(0, 5)))
        if terminated or truncated:
            resets += 1
            environment.reset(seed=seed + resets)
    elapsed = time.perf_counter() - started
    return {
        "steps": float(steps),
        "elapsed_seconds": float(elapsed),
        "steps_per_second": float(steps) / max(elapsed, 1e-12),
        "resets": float(resets),
    }


__all__ = ["HeadlessZeldaPersonaEnv", "PersonaReward", "benchmark_headless_steps"]
