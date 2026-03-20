"""
A* game-state search wrapper.
"""

from __future__ import annotations

from typing import Any

from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult
from src.simulation.validator import StateSpaceAStar


class AStarGameStateSolver:
    """Run A* over full game state with configurable representation mode."""

    def __init__(self, env: Any, config: GameStateSearchConfig):
        self.env = env
        self.config = config

    def solve(self) -> GameStateSearchResult:
        solver = StateSpaceAStar(
            self.env,
            timeout=self.config.timeout,
            priority_options=self.config.to_priority_options(),
            search_mode="astar",
        )
        success, path, states = solver.solve()
        return GameStateSearchResult(
            success=bool(success),
            path=list(path or []),
            states_explored=int(states or 0),
            algorithm="A*",
        )

