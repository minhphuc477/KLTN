"""A* with checkpoint-backed neural ordering inside equal-f plateaus."""

from __future__ import annotations

from typing import Any

from src.ml.heuristic_learning import MLHeuristicAStar
from src.simulation.search_base import GameStateSearchConfig, GameStateSearchResult
from src.simulation.search_status import oracle_status_from_outcome
from src.simulation.validator import StateSpaceAStar


class LearnedTieBreakAStarGameStateSolver:
    """Preserve canonical A* costs while using a neural secondary key."""

    def __init__(self, env: Any, config: GameStateSearchConfig):
        self.env = env
        self.config = config

    def solve(self) -> GameStateSearchResult:
        learned = MLHeuristicAStar(
            self.env,
            model_path=self.config.learned_heuristic_model_path,
            require_model=True,
            require_matching_shape=False,
        )
        priority_options = self.config.to_priority_options()
        priority_options["secondary_heuristic"] = learned.heuristic
        priority_options["secondary_heuristic_name"] = "learned_cost"
        solver = StateSpaceAStar(
            self.env,
            timeout=self.config.timeout,
            priority_options=priority_options,
            search_mode="astar",
        )
        success, path, diagnostics = solver.solve_with_diagnostics()
        failure_reason = str(getattr(diagnostics, "failure_reason", "") or "")
        trained_shape = learned.trained_map_shape
        environment_shape = (
            int(getattr(self.env, "height", 0) or 0),
            int(getattr(self.env, "width", 0) or 0),
        )
        return GameStateSearchResult(
            success=bool(success),
            path=list(path or []),
            states_explored=int(getattr(diagnostics, "states_explored", 0) or 0),
            algorithm="A* + learned tie-break",
            metadata={
                "failure_reason": failure_reason,
                "oracle_status": str(oracle_status_from_outcome(bool(success), failure_reason)),
                "time_taken_ms": float(getattr(diagnostics, "time_taken_ms", 0.0) or 0.0),
                "states_pruned_dominated": int(
                    getattr(diagnostics, "states_pruned_dominated", 0) or 0
                ),
                "max_queue_size": int(getattr(diagnostics, "max_queue_size", 0) or 0),
                "secondary_heuristic": "learned_cost",
                "checkpoint": str(self.config.learned_heuristic_model_path),
                "checkpoint_map_shape": list(trained_shape) if trained_shape is not None else None,
                "environment_map_shape": list(environment_shape),
                "cross_shape_ood": bool(trained_shape is not None and trained_shape != environment_shape),
                "optimality_contract": "canonical_f_primary_neural_tiebreak_only",
            },
        )
