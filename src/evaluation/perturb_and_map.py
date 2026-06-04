"""Perturb-and-MAP reachability evaluation utilities.

The functions in this module are evaluation/ablation tools. They repeatedly
perturb a walkability cost field and solve a hard shortest-path problem, then
aggregate the resulting path support. They do not claim differentiability.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

GridCoord = Tuple[int, int]


@dataclass(frozen=True)
class PerturbAndMAPReachabilityResult:
    """Aggregated hard-solver evidence over perturbed cost maps."""

    reachability: float
    mean_cost: float
    num_successes: int
    num_samples: int
    path_frequency: np.ndarray


def _as_numpy_grid(walkability: np.ndarray | torch.Tensor | Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(walkability, torch.Tensor):
        grid = walkability.detach().cpu().float().numpy()
    else:
        grid = np.asarray(walkability, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError(f"walkability must be a 2D grid, got shape {tuple(grid.shape)}.")
    if grid.size == 0:
        raise ValueError("walkability must not be empty.")
    return np.clip(grid.astype(np.float32, copy=False), 0.0, 1.0)


def _validate_coord(coord: GridCoord, height: int, width: int, name: str) -> GridCoord:
    row, col = int(coord[0]), int(coord[1])
    if row < 0 or row >= height or col < 0 or col >= width:
        raise ValueError(f"{name}={coord!r} is outside grid shape {(height, width)}.")
    return row, col


def _neighbors(row: int, col: int, height: int, width: int) -> Iterable[GridCoord]:
    if row > 0:
        yield row - 1, col
    if row + 1 < height:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < width:
        yield row, col + 1


def _astar(costs: np.ndarray, traversable: np.ndarray, start: GridCoord, goal: GridCoord) -> Tuple[float, List[GridCoord]]:
    height, width = costs.shape
    if not traversable[start] or not traversable[goal]:
        return math.inf, []

    def heuristic(node: GridCoord) -> float:
        return float(abs(node[0] - goal[0]) + abs(node[1] - goal[1]))

    frontier: List[Tuple[float, float, GridCoord]] = [(heuristic(start), 0.0, start)]
    came_from: dict[GridCoord, Optional[GridCoord]] = {start: None}
    best_cost: dict[GridCoord, float] = {start: 0.0}

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal:
            path: List[GridCoord] = []
            node: Optional[GridCoord] = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return current_cost, path
        if current_cost > best_cost.get(current, math.inf):
            continue
        for next_node in _neighbors(current[0], current[1], height, width):
            if not traversable[next_node]:
                continue
            next_cost = current_cost + float(costs[next_node])
            if next_cost < best_cost.get(next_node, math.inf):
                best_cost[next_node] = next_cost
                came_from[next_node] = current
                heapq.heappush(frontier, (next_cost + heuristic(next_node), next_cost, next_node))

    return math.inf, []


def perturb_and_map_reachability(
    walkability: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    start: GridCoord,
    goal: GridCoord,
    *,
    num_samples: int = 16,
    noise_scale: float = 0.25,
    obstacle_penalty: float = 8.0,
    blocked_threshold: float = 0.05,
    seed: Optional[int] = None,
) -> PerturbAndMAPReachabilityResult:
    """
    Estimate hard reachability by solving A* over perturbed MAP cost fields.

    Args:
        walkability: 2D probabilities/scores in [0, 1].
        start: Start cell as ``(row, col)``.
        goal: Goal cell as ``(row, col)``.
        num_samples: Number of perturbed hard solves.
        noise_scale: Gumbel perturbation scale added to cell traversal costs.
        obstacle_penalty: Cost multiplier for low-walkability cells.
        blocked_threshold: Cells at or below this walkability are impassable.
        seed: Optional deterministic NumPy RNG seed.
    """
    grid = _as_numpy_grid(walkability)
    height, width = grid.shape
    start = _validate_coord(start, height, width, "start")
    goal = _validate_coord(goal, height, width, "goal")
    sample_count = int(num_samples)
    if sample_count <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}.")
    scale = float(max(0.0, noise_scale))

    traversable = grid > float(blocked_threshold)
    base_cost = 1.0 + float(max(0.0, obstacle_penalty)) * (1.0 - grid)
    rng = np.random.default_rng(seed)
    path_counts = np.zeros_like(grid, dtype=np.float32)
    costs: List[float] = []

    for _ in range(sample_count):
        if scale > 0.0:
            perturbation = rng.gumbel(loc=0.0, scale=scale, size=grid.shape).astype(np.float32)
            sample_cost = np.maximum(base_cost + perturbation, 1e-4)
        else:
            sample_cost = base_cost
        cost, path = _astar(sample_cost, traversable, start, goal)
        if math.isfinite(cost):
            costs.append(float(cost))
            for row, col in path:
                path_counts[row, col] += 1.0

    successes = len(costs)
    return PerturbAndMAPReachabilityResult(
        reachability=float(successes) / float(sample_count),
        mean_cost=float(np.mean(costs)) if costs else math.inf,
        num_successes=successes,
        num_samples=sample_count,
        path_frequency=path_counts / float(sample_count),
    )

