"""Evolutionary individual data model."""

from __future__ import annotations

from ._shared import *

@dataclass
class Individual:
    """Individual in the population (genome + fitness)."""
    genome: List[int]
    fitness: float = 0.0
    feasible: bool = False
    constraint_violation: float = float("inf")
    topology_realism_error: float = float("inf")
    generation_rejection_ratio: float = 1.0
    phenotype: Optional[MissionGraph] = None
    descriptor_metrics: Dict[str, float] = field(default_factory=dict)
    rule_fitness_deltas: Dict[int, float] = field(default_factory=dict)
    generation: int = 0
    evaluated: bool = False
