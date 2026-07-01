"""
Legacy parallel-search compatibility API.

The former implementation started every process from the same root while
sharing a global closed set. The first worker claimed the root, leaving the
other workers with no frontier, so it did not perform parallel A*. It also
duplicated only part of Zelda's state transition rules.

Keep the public class for callers that still request ``parallel_astar``, but
delegate to the canonical inventory-aware A* implementation. This is an
explicit compatibility path, not a parallel speedup.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .validator import GameState, StateSpaceAStar, ZeldaLogicEnv, game_state_key


class ParallelAStarSolver:
    """Compatibility wrapper backed by canonical state-space A*."""

    def __init__(self, env: ZeldaLogicEnv, n_workers: Optional[int] = None):
        self.env = env
        self.n_workers = max(1, int(n_workers or 1))
        self.used_fallback = True

    def solve(
        self,
        start_state: Optional[GameState] = None,
    ) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Solve with the canonical state-space A* implementation.

        ``start_state`` remains accepted for API compatibility. The canonical
        solver starts from the environment's configured initial state, so a
        different custom state is rejected instead of being silently ignored.
        """
        if start_state is not None:
            canonical_start = self.env.reset()
            if game_state_key(start_state) != game_state_key(canonical_start):
                raise ValueError(
                    "ParallelAStarSolver compatibility mode only supports the "
                    "environment's configured initial state"
                )
        solver = StateSpaceAStar(self.env, search_mode="astar")
        return solver.solve()


def benchmark_parallel_vs_sequential(
    env: ZeldaLogicEnv,
    start_state: GameState,
) -> Dict[str, Any]:
    """
    Measure compatibility-wrapper overhead against direct canonical A*.

    This function is retained for old callers. ``speedup`` is intentionally
    omitted because both branches now execute the same algorithm.
    """
    sequential_solver = StateSpaceAStar(env, search_mode="astar")
    started = time.perf_counter()
    seq_success, seq_path, seq_states = sequential_solver.solve()
    sequential_time = time.perf_counter() - started

    compatibility_solver = ParallelAStarSolver(env)
    started = time.perf_counter()
    compat_success, compat_path, compat_states = compatibility_solver.solve(start_state)
    compatibility_time = time.perf_counter() - started

    return {
        "sequential_time": sequential_time,
        "compatibility_time": compatibility_time,
        "sequential_states": seq_states,
        "compatibility_states": compat_states,
        "sequential_path_length": len(seq_path) if seq_success else 0,
        "path_length": len(compat_path) if compat_success else 0,
        "success": compat_success,
        "backend": "canonical_astar",
    }
