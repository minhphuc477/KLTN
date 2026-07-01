"""
Factory for game-state search solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from src.simulation.astar_search import AStarGameStateSolver
from src.simulation.bfs_search import BFSGameStateSolver
from src.simulation.bidirectional_search import BidirectionalAStarGameStateSolver
from src.simulation.dfs_search import DFSGameStateSolver
from src.simulation.dijkstra_search import DijkstraGameStateSolver
from src.simulation.dstar_search import DStarLiteGameStateSolver
from src.simulation.greedy_search import GreedyGameStateSolver
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult
from src.simulation.validator import CONDITIONAL_IDS, PICKUP_IDS, PUSHABLE_IDS, WATER_IDS, SEMANTIC_PALETTE


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
    GameStateAlgorithmSpec(index=1, key="bfs", label="BFS", validation_role="small_exact_baseline", canonical_use="exact_baseline"),
    GameStateAlgorithmSpec(index=2, key="dijkstra", label="Dijkstra", validation_role="cost_exact_baseline", canonical_use="exact_fallback"),
    GameStateAlgorithmSpec(index=3, key="greedy", label="Greedy", validation_role="inadmissible_baseline", canonical_use="heuristic_baseline"),
    GameStateAlgorithmSpec(index=4, key="dstar_lite", label="D* Lite", validation_role="replanning_diagnostic", canonical_use="incremental_replanning"),
    GameStateAlgorithmSpec(index=5, key="dfs_iddfs", label="DFS/IDDFS", validation_role="bounded_exhaustive_probe", canonical_use="exhaustive_probe"),
    GameStateAlgorithmSpec(index=6, key="bidirectional_astar", label="Bidirectional A*", validation_role="reversible_grid_diagnostic", canonical_use="reversible_grid_comparison"),
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


def _present_tile_ids(env: Any) -> set[int]:
    grid = getattr(env, "original_grid", None)
    if grid is None:
        grid = getattr(env, "grid", None)
    if grid is None:
        return set()
    try:
        return {int(value) for value in grid.reshape(-1)}
    except Exception:
        return set()


def environment_requires_full_state_oracle(env: Any) -> bool:
    """
    Return whether validation needs full Zelda state-space search.

    A* over full game state remains the canonical oracle when inventory,
    consumables, movable blocks, water/item traversal, enemies, puzzle stages,
    or directed/graph transitions are present. Bidirectional grid search and
    JPS-style grid pruning are not valid independent oracles for those maps.
    """
    stateful_ids = (
        set(CONDITIONAL_IDS)
        | set(PICKUP_IDS)
        | set(PUSHABLE_IDS)
        | set(WATER_IDS)
        | {
            int(SEMANTIC_PALETTE["ENEMY"]),
            int(SEMANTIC_PALETTE["BOSS"]),
            int(SEMANTIC_PALETTE["PUZZLE"]),
        }
    )
    present_ids = _present_tile_ids(env)
    return bool(
        present_ids.intersection(stateful_ids)
        or getattr(env, "graph", None)
        or getattr(env, "_puzzle_stage_lookup", None)
        or getattr(env, "block_underlay_tiles", None)
    )


def recommended_game_state_algorithm_specs(
    env: Any,
    *,
    include_diagnostics: bool = False,
) -> List[GameStateAlgorithmSpec]:
    """
    Select algorithms by problem class instead of exposing all solvers as peers.

    - Full-state A* is the only canonical oracle for Zelda mechanics.
    - Dijkstra is an exact cost baseline/fallback, not the primary oracle.
    - Bidirectional A* is useful on reversible stateless grids only.
    - D* Lite is a replanning diagnostic and should not be reported as a
      one-shot validation oracle unless the experiment is explicitly about
      incremental replanning.
    """
    specs = {spec.key: spec for spec in GAME_STATE_ALGORITHM_SPECS}
    selected: List[GameStateAlgorithmSpec] = [specs["astar"]]
    if include_diagnostics:
        selected.append(specs["dijkstra"])

    if environment_requires_full_state_oracle(env):
        if include_diagnostics:
            selected.extend([specs["greedy"], specs["dfs_iddfs"], specs["dstar_lite"]])
        return selected

    selected.append(specs["bidirectional_astar"])
    if include_diagnostics:
        selected.extend([specs["bfs"], specs["dstar_lite"]])
    return selected


def run_recommended_game_state_solver(
    env: Any,
    config: GameStateSearchConfig,
) -> GameStateSearchResult:
    """Run the canonical publication-safe solver for the current environment."""
    spec = recommended_game_state_algorithm_specs(env, include_diagnostics=False)[0]
    result = run_game_state_solver(env, int(spec.index), config)
    result.metadata.setdefault("selection_reason", str(spec.canonical_use))
    result.metadata.setdefault("validation_role", str(spec.validation_role))
    return result


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

