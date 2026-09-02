"""
Bidirectional A* game-state search wrapper.
"""

from __future__ import annotations

from typing import Any

from src.simulation.bidirectional_astar import BidirectionalAStar
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult
from src.simulation.search_status import oracle_status_from_outcome


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
        if success:
            termination_status = "solved"
            failure_reason = ""
        elif self.env.start_pos is None or self.env.goal_pos is None:
            termination_status = "invalid"
            failure_reason = "No start or goal found in map"
        elif int(states or 0) >= int(self.config.timeout):
            termination_status = "budget_exhausted"
            failure_reason = "Bidirectional/A* shared state budget reached"
        else:
            termination_status = "incomplete"
            failure_reason = (
                "Bidirectional comparison search stopped without a solution or a retained "
                "complete-frontier certificate"
            )
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
                "failure_reason": failure_reason,
                "termination_status": termination_status,
                "solver_status": oracle_status_from_outcome(
                    bool(success),
                    failure_reason,
                    termination_status,
                ),
            },
        )
