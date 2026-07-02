"""Forward incremental replanning wrapper.

This adapter exposes the historical ``dstar_lite`` CLI key for compatibility,
but the implementation is a forward LPA*/D* Lite-style diagnostic over full
Zelda game state, not a textbook backward D* Lite oracle.
"""

from __future__ import annotations

from typing import Any

from src.simulation.dstar_lite import DStarLiteSolver
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult


class DStarLiteGameStateSolver:
    """Run the forward replanning diagnostic with A* fallback metadata preserved."""

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
                "Forward LPA* replanning (fallback: A*)"
                if getattr(solver, "used_fallback", False)
                else "Forward LPA* replanning"
            ),
            metadata={
                "fallback_used": bool(getattr(solver, "used_fallback", False)),
                "replans": int(getattr(solver, "replans_count", 0) or 0),
                "allow_diagonals": bool(self.config.allow_diagonals),
                "intended_use": "incremental_replanning",
                "independent_oracle": False,
                "textbook_dstar_lite": False,
            },
        )
