"""Backward D* Lite replanning wrapper with full-state A* fallback."""

from __future__ import annotations

from typing import Any

from src.simulation.dstar_lite import DStarLiteSolver
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult, apply_rules_profile
from src.simulation.search_status import oracle_status_from_outcome


class DStarLiteGameStateSolver:
    """Run backward D* Lite where valid and preserve fallback metadata."""

    def __init__(self, env: Any, config: GameStateSearchConfig):
        self.env = env
        self.config = config

    def solve(self) -> GameStateSearchResult:
        apply_rules_profile(self.env, self.config.rules_profile)
        solver = DStarLiteSolver(
            self.env,
            heuristic_mode="balanced",
            timeout=int(self.config.timeout),
            allow_diagonals=bool(self.config.allow_diagonals),
        )
        success, path, states = solver.solve(self.env.state.copy())
        retained_status = str(getattr(solver, "last_termination_status", "not_run"))
        if retained_status != "not_run":
            termination_status = retained_status
            failure_reason = str(getattr(solver, "last_failure_reason", ""))
        elif success:
            termination_status = "solved"
            failure_reason = ""
        elif self.env.start_pos is None or self.env.goal_pos is None:
            termination_status = "invalid"
            failure_reason = "No start or goal found in map"
        elif int(states or 0) >= int(self.config.timeout):
            termination_status = "budget_exhausted"
            failure_reason = "D* Lite/A* shared state budget reached"
        else:
            termination_status = "incomplete"
            failure_reason = (
                "D* Lite comparison search stopped without a solution or a retained "
                "complete-frontier certificate"
            )
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
                "vertex_updates": int(getattr(solver, "states_updated", 0) or 0),
                "allow_diagonals": bool(self.config.allow_diagonals),
                "intended_use": "incremental_replanning",
                "independent_oracle": False,
                "textbook_dstar_lite": not bool(getattr(solver, "used_fallback", False)),
                "problem_scope": (
                    "full_state_astar_fallback"
                    if getattr(solver, "used_fallback", False)
                    else "reversible_position_graph"
                ),
                "failure_reason": failure_reason,
                "termination_status": termination_status,
                "solver_status": oracle_status_from_outcome(
                    bool(success),
                    failure_reason,
                    termination_status,
                ),
            },
        )
