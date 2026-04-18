"""
BFS game-state search wrapper.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.search_benchmark_utils import oracle_status_from_outcome
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult
from src.simulation.validator import StateSpaceAStar


class BFSGameStateSolver:
    """Run breadth-first search over full game state."""

    def __init__(self, env: Any, config: GameStateSearchConfig):
        self.env = env
        self.config = config

    def solve(self) -> GameStateSearchResult:
        solver = StateSpaceAStar(
            self.env,
            timeout=self.config.timeout,
            priority_options=self.config.to_priority_options(),
            search_mode="bfs",
        )
        success, path, diagnostics = solver.solve_with_diagnostics()
        return GameStateSearchResult(
            success=bool(success),
            path=list(path or []),
            states_explored=int(getattr(diagnostics, "states_explored", 0) or 0),
            algorithm="BFS",
            metadata={
                "failure_reason": str(getattr(diagnostics, "failure_reason", "") or ""),
                "solver_status": str(
                    oracle_status_from_outcome(
                        bool(success),
                        str(getattr(diagnostics, "failure_reason", "") or ""),
                    )
                ),
                "time_taken_ms": float(getattr(diagnostics, "time_taken_ms", 0.0) or 0.0),
                "states_pruned_dominated": int(
                    getattr(diagnostics, "states_pruned_dominated", 0) or 0
                ),
                "max_queue_size": int(getattr(diagnostics, "max_queue_size", 0) or 0),
            },
        )

