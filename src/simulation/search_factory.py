"""
Factory for game-state search solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from src.simulation.astar_search import AStarGameStateSolver
from src.simulation.bfs_search import BFSGameStateSolver
from src.simulation.bidirectional_search import BidirectionalAStarGameStateSolver
from src.simulation.dfs_search import DFSGameStateSolver
from src.simulation.dijkstra_search import DijkstraGameStateSolver
from src.simulation.dstar_search import DStarLiteGameStateSolver
from src.simulation.greedy_search import GreedyGameStateSolver
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult


@dataclass(frozen=True)
class GameStateAlgorithmSpec:
    """Canonical metadata for a game-state solver exposed by the repo."""

    index: int
    key: str
    label: str
    validation_role: str = "comparison"
    canonical_use: str = "comparison"


GAME_STATE_ALGORITHM_SPECS: Tuple[GameStateAlgorithmSpec, ...] = (
    GameStateAlgorithmSpec(index=0, key="astar", label="A*", validation_role="oracle", canonical_use="hard_oracle"),
    GameStateAlgorithmSpec(index=1, key="bfs", label="BFS", canonical_use="exact_baseline"),
    GameStateAlgorithmSpec(index=2, key="dijkstra", label="Dijkstra", canonical_use="exact_fallback"),
    GameStateAlgorithmSpec(index=3, key="greedy", label="Greedy", canonical_use="heuristic_baseline"),
    GameStateAlgorithmSpec(index=4, key="dstar_lite", label="D* Lite", validation_role="replanning", canonical_use="incremental_replanning"),
    GameStateAlgorithmSpec(index=5, key="dfs_iddfs", label="DFS/IDDFS", canonical_use="exhaustive_probe"),
    GameStateAlgorithmSpec(index=6, key="bidirectional_astar", label="Bidirectional A*", canonical_use="comparison"),
)

SUPPORTED_GAME_STATE_ALGORITHMS: Dict[int, str] = {
    int(spec.index): str(spec.label) for spec in GAME_STATE_ALGORITHM_SPECS
}

VALIDATION_EXCLUDED_ALGORITHMS: Dict[str, str] = {
    "parallel_astar": (
        "Excluded from canonical export validation because multiprocessing adds "
        "high runtime overhead and non-trivial platform variance while targeting "
        "the same optimality contract as A*."
    ),
    "multi_goal": (
        "Excluded because end-to-end dungeon validation here is single-goal "
        "START->TRIFORCE, not multi-goal routing."
    ),
    "key_economy_validator": (
        "Excluded because it validates graph-level key economy, not tile/grid playability."
    ),
    "solver_comparison": (
        "Excluded because it is a benchmark harness, not an independent solver."
    ),
}


def iter_game_state_algorithm_specs() -> Iterable[GameStateAlgorithmSpec]:
    """Return canonical solver specs in the order used by GUI and validation."""
    return GAME_STATE_ALGORITHM_SPECS


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
    - 4: D* Lite
    - 5: DFS / IDDFS
    - 6: Bidirectional A*
    """
    if algorithm_idx == 0:
        return AStarGameStateSolver(env, config).solve()
    if algorithm_idx == 1:
        return BFSGameStateSolver(env, config).solve()
    if algorithm_idx == 2:
        return DijkstraGameStateSolver(env, config).solve()
    if algorithm_idx == 3:
        return GreedyGameStateSolver(env, config).solve()
    if algorithm_idx == 4:
        return DStarLiteGameStateSolver(env, config).solve()
    if algorithm_idx == 5:
        return DFSGameStateSolver(env, config).solve()
    if algorithm_idx == 6:
        return BidirectionalAStarGameStateSolver(env, config).solve()
    raise ValueError(f"Unsupported game-state algorithm index: {algorithm_idx}")

