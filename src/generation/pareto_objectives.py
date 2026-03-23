"""Pareto-style topology objective utilities for evolutionary search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ParetoObjectiveResult:
    required_loops: float
    required_branching: float
    loops_raw: float
    branching_raw: float
    loops_violation: float
    branching_violation: float
    pareto_feasible: bool
    pareto_score: float


def compute_pareto_objectives(
    descriptor_metrics: Dict[str, float],
    *,
    curve_alignment_score: float,
    required_loops: float = 2.0,
    required_branching: float = 1.5,
) -> ParetoObjectiveResult:
    """Compute loop/branching Pareto constraints and aggregate objective score."""
    loops_raw = float(descriptor_metrics.get("cyclomatic_complexity", 0.0))
    branching_raw = float(descriptor_metrics.get("branching_factor_raw", 0.0))

    loops_violation = max(0.0, required_loops - loops_raw) / max(1.0, required_loops)
    branching_violation = max(0.0, required_branching - branching_raw) / max(1.0, required_branching)
    pareto_feasible = (loops_raw >= required_loops) and (branching_raw >= required_branching)

    loop_objective = float(np.clip(loops_raw / max(1.0, required_loops), 0.0, 1.0))
    branch_objective = float(np.clip(branching_raw / max(1.0, required_branching), 0.0, 1.0))

    pareto_score = float(
        np.clip(
            0.40 * float(curve_alignment_score) + 0.30 * loop_objective + 0.30 * branch_objective,
            0.0,
            1.0,
        )
    )

    return ParetoObjectiveResult(
        required_loops=float(required_loops),
        required_branching=float(required_branching),
        loops_raw=float(loops_raw),
        branching_raw=float(branching_raw),
        loops_violation=float(loops_violation),
        branching_violation=float(branching_violation),
        pareto_feasible=bool(pareto_feasible),
        pareto_score=float(pareto_score),
    )


def apply_pareto_metrics(descriptor_metrics: Dict[str, float], result: ParetoObjectiveResult) -> None:
    """Write standardized Pareto diagnostics into descriptor metrics."""
    descriptor_metrics["pareto_loop_requirement"] = float(result.required_loops)
    descriptor_metrics["pareto_branch_requirement"] = float(result.required_branching)
    descriptor_metrics["pareto_loops_violation"] = float(result.loops_violation)
    descriptor_metrics["pareto_branching_violation"] = float(result.branching_violation)
    descriptor_metrics["pareto_feasible"] = float(1.0 if result.pareto_feasible else 0.0)
    descriptor_metrics["pareto_score"] = float(result.pareto_score)
