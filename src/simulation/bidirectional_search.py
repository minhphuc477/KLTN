"""
Bidirectional A* game-state search wrapper.
"""

from __future__ import annotations

from typing import Any

from src.simulation.bidirectional_astar import BidirectionalAStar
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult


class BidirectionalAStarGameStateSolver:
    """Run bidirectional A* over full game state."""

    def __init__(self, env: Any, config: GameStateSearchConfig):
        self.env = env
        self.config = config

    def solve(self) -> GameStateSearchResult:
        solver = BidirectionalAStar(
            self.env,
            timeout=int(self.config.timeout),
            allow_diagonals=bool(self.config.allow_diagonals),
            heuristic_mode="balanced",
        )
        success, path, states = solver.solve()
        return GameStateSearchResult(
            success=bool(success),
            path=list(path or []),
            states_explored=int(states or 0),
            algorithm=(
                "Bidirectional A* (fallback: A*)"
                if getattr(solver, "used_fallback", False)
                else "Bidirectional A*"
            ),
            metadata={
                "fallback_used": bool(getattr(solver, "used_fallback", False)),
                "meeting_point": getattr(solver, "meeting_point", None),
                "collision_checks": int(getattr(solver, "collision_checks", 0) or 0),
                "intended_use": "comparison_search",
                "independent_oracle": False,
            },
        )
