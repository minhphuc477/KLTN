"""
DFS / IDDFS game-state search wrapper.
"""

from __future__ import annotations

from typing import Any

from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult
from src.simulation.state_space_dfs import StateSpaceDFS


class DFSGameStateSolver:
    """Run DFS / IDDFS over full game state."""

    def __init__(self, env: Any, config: GameStateSearchConfig):
        self.env = env
        self.config = config

    def solve(self) -> GameStateSearchResult:
        solver = StateSpaceDFS(
            self.env,
            timeout=int(self.config.timeout),
            max_depth=int(self.config.max_depth),
            allow_diagonals=bool(self.config.allow_diagonals),
            use_iddfs=bool(self.config.use_iddfs),
        )
        success, path, states = solver.solve()
        return GameStateSearchResult(
            success=bool(success),
            path=list(path or []),
            states_explored=int(states or 0),
            algorithm="DFS/IDDFS" if bool(self.config.use_iddfs) else "DFS",
            metadata={
                "max_depth_reached": int(getattr(solver.metrics, "max_depth_reached", 0) or 0),
                "backtrack_count": int(getattr(solver.metrics, "backtrack_count", 0) or 0),
                "cycle_detections": int(getattr(solver.metrics, "cycle_detections", 0) or 0),
                "use_iddfs": bool(self.config.use_iddfs),
            },
        )
