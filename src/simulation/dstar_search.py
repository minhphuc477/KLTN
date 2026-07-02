"""Backward D* Lite replanning wrapper with full-state A* fallback."""

from __future__ import annotations

from typing import Any

from src.simulation.dstar_lite import DStarLiteSolver
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult


class DStarLiteGameStateSolver:
    """Run backward D* Lite where valid and preserve fallback metadata."""

    def __init__(self, env: Any, config: GameStateSearchConfig):
        self.env = env
        self.config = config

    def solve(self) -> GameStateSearchResult:
        solver = DStarLiteSolver(
            self.env,
            heuristic_mode="balanced",
            timeout=int(self.config.timeout),
            allow_diagonals=bool(self.config.allow_diagonals),
        )
        success, path, states = solver.solve(self.env.state.copy())
        return GameStateSearchResult(
            success=bool(success),
            path=list(path or []),
            states_explored=int(states or 0),
            algorithm=(
                "D* Lite replanning (fallback: full-state A*)"
                if getattr(solver, "used_fallback", False)
                else "D* Lite replanning"
            ),
            metadata={
                "fallback_used": bool(getattr(solver, "used_fallback", False)),
                "replans": int(getattr(solver, "replans_count", 0) or 0),
                "allow_diagonals": bool(self.config.allow_diagonals),
                "intended_use": "incremental_replanning",
                "independent_oracle": False,
                "textbook_dstar_lite": not bool(getattr(solver, "used_fallback", False)),
                "problem_scope": (
                    "full_state_astar_fallback"
                    if getattr(solver, "used_fallback", False)
                    else "reversible_position_graph"
                ),
            },
        )
