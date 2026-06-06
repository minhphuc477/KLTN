"""CPU-only RL ablations for validating P-CBS persona behavior.

This module is intentionally evaluation-only: tabular Q-learning gives a
small, interpretable comparison arm without replacing the hand-authored P-CBS
solver or adding a neural dependency to validation.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.cognitive_bounded_search import AgentPersona, solve_with_pcbs
from src.simulation.validator import ACTION_DELTAS, BLOCKING_IDS, ZeldaLogicEnv

GridPos = Tuple[int, int]


@dataclass
class RLAblationMetrics:
    """Behavior summary for one trained tabular RL agent on one level."""

    reward_variant: str
    memory_capacity: int
    episodes: int
    success_rate: float
    final_success: bool
    total_steps: int
    unique_tiles_visited: int
    confusion_index: float
    navigation_entropy: float
    cognitive_load: float
    linearity_ratio: float
    path: List[GridPos] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_variant": self.reward_variant,
            "memory_capacity": int(self.memory_capacity),
            "episodes": int(self.episodes),
            "success_rate": round(float(self.success_rate), 4),
            "final_success": bool(self.final_success),
            "total_steps": int(self.total_steps),
            "unique_tiles_visited": int(self.unique_tiles_visited),
            "confusion_index": round(float(self.confusion_index), 4),
            "navigation_entropy": round(float(self.navigation_entropy), 4),
            "cognitive_load": round(float(self.cognitive_load), 4),
            "linearity_ratio": round(float(self.linearity_ratio), 4),
            "path": list(self.path),
        }


class BeliefStateQAgent:
    """Compact tabular Q-learning agent for per-level online ablations."""

    def __init__(
        self,
        memory_capacity: int,
        *,
        epsilon: float = 0.30,
        alpha: float = 0.35,
        gamma: float = 0.92,
        seed: Optional[int] = None,
    ):
        self.memory_capacity = int(max(1, memory_capacity))
        self.epsilon = float(max(0.0, min(1.0, epsilon)))
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.gamma = float(max(0.0, min(1.0, gamma)))
        self.rng = random.Random(seed)
        self.q_values: DefaultDict[Tuple[int, int, int], np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.float32))
        self.memory: deque[GridPos] = deque(maxlen=self.memory_capacity)
        self.visited_counts: DefaultDict[GridPos, int] = defaultdict(int)

    def reset_episode(self) -> None:
        self.memory.clear()
        self.visited_counts.clear()

    def state(self, pos: GridPos, goal: Optional[GridPos]) -> Tuple[int, int, int]:
        sector = self._compass_sector(pos, goal)
        memory_bin = min(2, int(len(self.memory) * 3 // max(1, self.memory_capacity)))
        explored_bin = min(3, int(len(self.visited_counts) // 10))
        return (sector, memory_bin, explored_bin)

    def observe(self, pos: GridPos) -> None:
        normalized = (int(pos[0]), int(pos[1]))
        self.memory.append(normalized)
        self.visited_counts[normalized] += 1

    def choose_action(self, state: Tuple[int, int, int], valid_actions: Sequence[int], *, greedy: bool = False) -> int:
        valid = [int(action) for action in valid_actions if 0 <= int(action) < 4]
        if not valid:
            valid = [0, 1, 2, 3]
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))
        q = self.q_values[state]
        return int(max(valid, key=lambda action: float(q[int(action)])))

    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        next_valid_actions: Sequence[int],
        done: bool,
    ) -> None:
        action_i = int(action)
        current = float(self.q_values[state][action_i])
        if done:
            target = float(reward)
        else:
            valid = [int(a) for a in next_valid_actions if 0 <= int(a) < 4] or [0, 1, 2, 3]
            target = float(reward) + self.gamma * max(float(self.q_values[next_state][a]) for a in valid)
        self.q_values[state][action_i] = current + self.alpha * (target - current)

    def _compass_sector(self, pos: GridPos, goal: Optional[GridPos]) -> int:
        if goal is None:
            return 8
        dr = int(goal[0]) - int(pos[0])
        dc = int(goal[1]) - int(pos[1])
        if dr == 0 and dc == 0:
            return 8
        angle = math.atan2(-dr, dc)
        return int(round(((angle % (2.0 * math.pi)) / (2.0 * math.pi)) * 8.0)) % 8


def train_belief_state_q_agent(
    grid: np.ndarray,
    *,
    reward_variant: str = "goal",
    memory_capacity: int = 7,
    episodes: int = 50,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    epsilon: float = 0.30,
) -> RLAblationMetrics:
    """Train a tabular RL agent online on one level and return final metrics."""
    semantic_grid = np.asarray(grid, dtype=np.int64)
    env = ZeldaLogicEnv(semantic_grid=semantic_grid.copy())
    agent = BeliefStateQAgent(memory_capacity=memory_capacity, epsilon=epsilon, seed=seed)
    step_cap = int(max_steps or max(32, semantic_grid.size * 4))
    successes = 0

    for _episode in range(int(max(1, episodes))):
        success, _path = _run_q_episode(
            env,
            agent,
            reward_variant=reward_variant,
            max_steps=step_cap,
            train=True,
            greedy=False,
        )
        successes += int(success)

    final_success, final_path = _run_q_episode(
        env,
        agent,
        reward_variant=reward_variant,
        max_steps=step_cap,
        train=False,
        greedy=True,
    )
    return _metrics_from_path(
        final_path,
        start=env.start_pos,
        goal=env.goal_pos,
        reward_variant=reward_variant,
        memory_capacity=int(memory_capacity),
        episodes=int(max(1, episodes)),
        success_rate=float(successes) / float(max(1, episodes)),
        final_success=bool(final_success),
    )


def run_pcbs_rl_alignment_ablation(
    grid: np.ndarray,
    *,
    personas: Sequence[str] = ("speedrunner", "explorer", "cautious", "forgetful", "novice", "greedy"),
    reward_variants: Sequence[str] = ("goal", "explore", "safe", "memory"),
    memory_capacities: Sequence[int] = (4, 7, 10),
    episodes: int = 50,
    timeout_pcbs: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the release-paper RL-vs-P-CBS behavioral alignment protocol."""
    persona_metrics: Dict[str, Dict[str, Any]] = {}
    for persona in personas:
        success, path, _states, metrics = solve_with_pcbs(
            np.asarray(grid, dtype=np.int64),
            persona=str(persona),
            timeout=int(timeout_pcbs),
            seed=seed,
        )
        persona_metrics[str(persona)] = {
            "success": bool(success),
            "path": list(path),
            "confusion_index": float(metrics.confusion_index),
            "navigation_entropy": float(metrics.navigation_entropy),
            "cognitive_load": float(metrics.cognitive_load),
            "total_steps": int(metrics.total_steps),
            "unique_tiles_visited": int(metrics.unique_tiles_visited),
        }

    rl_results: List[Dict[str, Any]] = []
    for reward_variant in reward_variants:
        for capacity in memory_capacities:
            metrics = train_belief_state_q_agent(
                grid,
                reward_variant=str(reward_variant),
                memory_capacity=int(capacity),
                episodes=int(episodes),
                seed=seed,
            )
            best_persona, distance, alignment_score = _closest_persona(metrics.to_dict(), persona_metrics)
            rl_results.append({
                **metrics.to_dict(),
                "closest_persona": best_persona,
                "persona_distance": round(float(distance), 4),
                "alignment_score": round(float(alignment_score), 4),
            })

    return {
        "persona_metrics": persona_metrics,
        "rl_results": rl_results,
        "cross_persona_agreement_rate": compute_cross_persona_agreement(persona_metrics),
        "persona_divergence": compute_persona_divergence_from_paths(
            {persona: payload.get("path", []) for persona, payload in persona_metrics.items()}
        ),
    }


def compute_cross_persona_agreement(persona_metrics: Mapping[str, Mapping[str, Any]]) -> float:
    """Fraction of personas that agree on the majority success/failure outcome."""
    outcomes = [bool(payload.get("success", False)) for payload in persona_metrics.values()]
    if not outcomes:
        return 0.0
    success_count = sum(int(value) for value in outcomes)
    majority = max(success_count, len(outcomes) - success_count)
    return float(majority) / float(len(outcomes))


def compute_persona_divergence_from_paths(paths_by_persona: Mapping[str, Iterable[GridPos]], smoothing: float = 1e-10) -> float:
    """Average symmetric KL divergence between persona visit distributions."""
    distributions: Dict[str, Dict[GridPos, float]] = {}
    all_positions: set[GridPos] = set()
    eps = float(max(float(smoothing), 1e-12))
    for persona, path_iter in paths_by_persona.items():
        counts: DefaultDict[GridPos, float] = defaultdict(float)
        for raw_pos in path_iter:
            pos = (int(raw_pos[0]), int(raw_pos[1]))
            counts[pos] += 1.0
            all_positions.add(pos)
        total = float(sum(counts.values()))
        if total <= 0:
            distributions[str(persona)] = {}
        else:
            distributions[str(persona)] = {pos: count / total for pos, count in counts.items()}
    personas = sorted(distributions)
    if len(personas) < 2 or not all_positions:
        return 0.0
    divergences: List[float] = []
    for i, left in enumerate(personas):
        for right in personas[i + 1:]:
            p = distributions[left]
            q = distributions[right]
            kl_pq = 0.0
            kl_qp = 0.0
            for pos in all_positions:
                pv = max(float(p.get(pos, eps)), eps)
                qv = max(float(q.get(pos, eps)), eps)
                kl_pq += pv * math.log(pv / qv)
                kl_qp += qv * math.log(qv / pv)
            divergences.append(0.5 * (kl_pq + kl_qp))
    return float(np.mean(divergences)) if divergences else 0.0


def _run_q_episode(
    env: ZeldaLogicEnv,
    agent: BeliefStateQAgent,
    *,
    reward_variant: str,
    max_steps: int,
    train: bool,
    greedy: bool,
) -> Tuple[bool, List[GridPos]]:
    state = env.reset()
    agent.reset_episode()
    start = (int(state.position[0]), int(state.position[1]))
    agent.observe(start)
    path: List[GridPos] = [start]
    done = False

    for _step in range(int(max(1, max_steps))):
        current_pos = (int(env.state.position[0]), int(env.state.position[1]))
        state_key = agent.state(current_pos, env.goal_pos)
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state_key, valid_actions, greedy=greedy)
        prev_dist = _manhattan(current_pos, env.goal_pos)
        visited_before = agent.visited_counts[current_pos] > 1
        next_state, _env_reward, done, _info = env.step(int(action))
        next_pos = (int(next_state.position[0]), int(next_state.position[1]))
        agent.observe(next_pos)
        path.append(next_pos)
        reward = _reward(
            reward_variant,
            env=env,
            prev_pos=current_pos,
            new_pos=next_pos,
            prev_dist=prev_dist,
            solved=bool(done and env.won),
            visited_before=visited_before,
            memory_load=float(len(agent.memory)) / float(max(1, agent.memory_capacity)),
        )
        next_key = agent.state(next_pos, env.goal_pos)
        if train:
            agent.update(state_key, action, reward, next_key, env.get_valid_actions(), bool(done))
        if done:
            break
    return bool(done and env.won), path


def _reward(
    variant: str,
    *,
    env: ZeldaLogicEnv,
    prev_pos: GridPos,
    new_pos: GridPos,
    prev_dist: int,
    solved: bool,
    visited_before: bool,
    memory_load: float,
) -> float:
    if solved:
        return 10.0
    progress = float(prev_dist - _manhattan(new_pos, env.goal_pos)) / float(max(1, prev_dist))
    step_cost = -0.01
    key = str(variant or "goal").strip().lower()
    if key in {"explore", "curiosity", "r_explore"}:
        return step_cost + (0.5 if not visited_before else 0.0) + 0.2 * progress
    if key in {"safe", "cautious", "r_safe"}:
        return step_cost + progress - 2.0 * _enemy_proximity(env.original_grid, new_pos)
    if key in {"memory", "forgetful", "r_memory"}:
        return step_cost + progress - (0.3 if visited_before else 0.0) - 0.15 * float(memory_load)
    return step_cost + 0.3 * progress


def _metrics_from_path(
    path: List[GridPos],
    *,
    start: Optional[GridPos],
    goal: Optional[GridPos],
    reward_variant: str,
    memory_capacity: int,
    episodes: int,
    success_rate: float,
    final_success: bool,
) -> RLAblationMetrics:
    total_steps = len(path)
    unique_tiles = len(set(path))
    revisits = max(0, total_steps - unique_tiles)
    expected_revisits = float(unique_tiles) * math.log(float(max(2, unique_tiles)))
    confusion = float(revisits) / max(1.0, expected_revisits)
    direction_counts: DefaultDict[str, int] = defaultdict(int)
    for prev, cur in zip(path, path[1:]):
        dr = int(cur[0]) - int(prev[0])
        dc = int(cur[1]) - int(prev[1])
        direction_counts[_direction_name((dr, dc))] += 1
    entropy = _entropy(direction_counts.values())
    lower_bound = 1
    if start is not None and goal is not None:
        lower_bound = max(1, _manhattan(start, goal) + 1)
    cognitive_load = min(1.0, float(unique_tiles) / float(max(1, memory_capacity * 3)))
    return RLAblationMetrics(
        reward_variant=str(reward_variant),
        memory_capacity=int(memory_capacity),
        episodes=int(episodes),
        success_rate=float(success_rate),
        final_success=bool(final_success),
        total_steps=int(total_steps),
        unique_tiles_visited=int(unique_tiles),
        confusion_index=float(confusion),
        navigation_entropy=float(entropy),
        cognitive_load=float(cognitive_load),
        linearity_ratio=float(total_steps) / float(lower_bound),
        path=list(path),
    )


def _closest_persona(rl_metrics: Mapping[str, Any], persona_metrics: Mapping[str, Mapping[str, Any]]) -> Tuple[str, float, float]:
    best_persona = ""
    best_distance = float("inf")
    for persona, metrics in persona_metrics.items():
        distance = (
            abs(float(rl_metrics.get("confusion_index", 0.0)) - float(metrics.get("confusion_index", 0.0)))
            + abs(float(rl_metrics.get("navigation_entropy", 0.0)) - float(metrics.get("navigation_entropy", 0.0))) / 2.0
            + abs(float(rl_metrics.get("cognitive_load", 0.0)) - float(metrics.get("cognitive_load", 0.0))) / 2.5
        ) / 3.0
        if distance < best_distance:
            best_persona = str(persona)
            best_distance = float(distance)
    alignment_score = max(0.0, 1.0 - best_distance)
    return best_persona, best_distance, alignment_score


def _manhattan(left: GridPos, right: Optional[GridPos]) -> int:
    if right is None:
        return 0
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def _enemy_proximity(grid: np.ndarray, pos: GridPos) -> float:
    enemy_ids = {int(SEMANTIC_PALETTE["ENEMY"]), int(SEMANTIC_PALETTE["BOSS"])}
    rr, cc = int(pos[0]), int(pos[1])
    penalty = 0.0
    for er, ec in np.argwhere(np.isin(grid, list(enemy_ids))):
        dist = abs(rr - int(er)) + abs(cc - int(ec))
        if dist <= 2:
            penalty = max(penalty, (3.0 - float(dist)) / 3.0)
    return float(penalty)


def _direction_name(direction: GridPos) -> str:
    for action, delta in ACTION_DELTAS.items():
        if tuple(delta) == tuple(direction):
            return str(action.name)
    return "STAY"


def _entropy(counts: Iterable[int]) -> float:
    values = [float(count) for count in counts if float(count) > 0]
    total = float(sum(values))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in values:
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


__all__ = [
    "BeliefStateQAgent",
    "RLAblationMetrics",
    "train_belief_state_q_agent",
    "run_pcbs_rl_alignment_ablation",
    "compute_cross_persona_agreement",
    "compute_persona_divergence_from_paths",
]
