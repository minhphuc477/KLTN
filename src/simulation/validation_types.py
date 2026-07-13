"""Dependency-light result and option contracts for Zelda validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ValidationResult:
    """Results from validating a single map."""

    is_solvable: bool
    is_valid_syntax: bool
    reachability: float
    path_length: int
    backtracking_score: float
    logical_errors: List[str]
    path_cost: Optional[float] = None
    path: List[Tuple[int, int]] = field(default_factory=list)
    error_message: str = ""
    solver_used: str = "astar"
    primary_solver_solved: Optional[bool] = None
    primary_solver_error: str = ""
    states_explored: int = 0
    termination_status: str = "unknown"
    proven_unsolvable: bool = False
    final_inventory: Optional[Dict[str, Any]] = None
    path_interactions: Dict[str, int] = field(default_factory=dict)
    route_replay_status: str = "not_run"
    route_replay_error: str = ""
    route_replay_path_cost: Optional[float] = None
    solver_consistency_status: str = "not_requested"
    solver_consistent: Optional[bool] = None
    solver_consistency_path_length: Optional[int] = None
    solver_consistency_path_cost: Optional[float] = None
    solver_consistency_states_explored: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_solvable": self.is_solvable,
            "is_valid_syntax": self.is_valid_syntax,
            "reachability": self.reachability,
            "path_length": self.path_length,
            "path_cost": self.path_cost,
            "backtracking_score": self.backtracking_score,
            "logical_errors": self.logical_errors,
            "error_message": self.error_message,
            "termination_status": self.termination_status,
            "proven_unsolvable": self.proven_unsolvable,
            "final_inventory": dict(self.final_inventory or {}),
            "path_interactions": dict(self.path_interactions or {}),
            "route_replay_status": self.route_replay_status,
            "route_replay_error": self.route_replay_error,
            "route_replay_path_cost": self.route_replay_path_cost,
            "solver_consistency_status": self.solver_consistency_status,
            "solver_consistent": self.solver_consistent,
            "solver_consistency_path_length": self.solver_consistency_path_length,
            "solver_consistency_path_cost": self.solver_consistency_path_cost,
            "solver_consistency_states_explored": self.solver_consistency_states_explored,
        }


@dataclass
class SolverOptions:
    """Starting inventory and search behavior configuration."""

    start_keys: int = 0
    start_bombs: int = 1
    start_boss_key: bool = False
    start_item: bool = False
    timeout: int = 200000
    allow_diagonals: bool = False
    heuristic_mode: str = "balanced"
    rules_profile: str = "vglc_strict"

    @classmethod
    def for_level(cls, level_type: str = "normal") -> "SolverOptions":
        if level_type == "bomb_heavy":
            return cls(start_bombs=3)
        if level_type == "key_heavy":
            return cls(start_keys=1, start_bombs=1)
        if level_type == "speedrun":
            return cls(start_bombs=1, allow_diagonals=True, heuristic_mode="speedrunner")
        return cls()


@dataclass
class SolverDiagnostics:
    """Detailed diagnostics from a solver run."""

    success: bool
    states_explored: int
    states_pruned_dominated: int = 0
    max_queue_size: int = 0
    time_taken_ms: float = 0.0
    failure_reason: str = ""
    path_length: int = 0
    path_cost: Optional[float] = None
    final_inventory: Optional[Dict[str, Any]] = None
    termination_status: str = "unknown"
    route_replay_status: str = "not_run"
    route_replay_error: str = ""
    route_replay_path_cost: Optional[float] = None

    def summary(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED: {self.failure_reason}"
        pruning_total = max(1, self.states_explored + self.states_pruned_dominated)
        return f"""
=== Solver Diagnostics ===
Status: {status}
States Explored: {self.states_explored:,}
States Pruned (dominated): {self.states_pruned_dominated:,}
Pruning Efficiency: {100.0 * self.states_pruned_dominated / pruning_total:.1f}%
Max Queue Size: {self.max_queue_size:,}
Time Taken: {self.time_taken_ms:.1f}ms
Path Length: {self.path_length}
=========================="""


@dataclass
class BatchValidationResult:
    """Results from validating a batch of maps."""

    total_maps: int
    valid_syntax_count: int
    solvable_count: int
    solvability_rate: float
    avg_reachability: float
    avg_path_length: float
    avg_backtracking: float
    diversity_score: float
    individual_results: List[ValidationResult] = field(default_factory=list)

    def summary(self) -> str:
        valid_rate = 100 * self.valid_syntax_count / max(1, self.total_maps)
        return f"""
=== Batch Validation Summary ===
Total Maps: {self.total_maps}
Valid Syntax: {self.valid_syntax_count} ({valid_rate:.1f}%)
Solvable: {self.solvable_count} ({100 * self.solvability_rate:.1f}%)
Avg Reachability: {100 * self.avg_reachability:.1f}%
Avg Path Length: {self.avg_path_length:.1f}
Avg Backtracking: {self.avg_backtracking:.2f}
Diversity Score: {self.diversity_score:.3f}
================================
"""


__all__ = [
    "BatchValidationResult",
    "SolverDiagnostics",
    "SolverOptions",
    "ValidationResult",
]
