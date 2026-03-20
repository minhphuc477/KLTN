"""
Factory for game-state search solvers.
"""

from __future__ import annotations

from typing import Any

from src.simulation.astar_search import AStarGameStateSolver
from src.simulation.bfs_search import BFSGameStateSolver
from src.simulation.dijkstra_search import DijkstraGameStateSolver
from src.simulation.greedy_search import GreedyGameStateSolver
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult


def run_game_state_solver(
    env: Any,
    algorithm_idx: int,
    config: GameStateSearchConfig,
) -> GameStateSearchResult:
    """
    Dispatch and run a game-state solver by algorithm index.

    Mapping:
    - 0: A*
    - 1: BFS
    - 2: Dijkstra
    - 3: Greedy
    """
    if algorithm_idx == 0:
        return AStarGameStateSolver(env, config).solve()
    if algorithm_idx == 1:
        return BFSGameStateSolver(env, config).solve()
    if algorithm_idx == 2:
        return DijkstraGameStateSolver(env, config).solve()
    if algorithm_idx == 3:
        return GreedyGameStateSolver(env, config).solve()
    raise ValueError(f"Unsupported game-state algorithm index: {algorithm_idx}")

