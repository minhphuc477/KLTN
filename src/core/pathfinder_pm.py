"""Perturb-and-MAP pathfinder autograd entry points.

This module keeps the audit-facing name `DifferentiablePerturbedAStar` while
reusing the core implementation in `src.core.perturb_and_map`.
"""

from __future__ import annotations

from src.core.perturb_and_map import (
    DifferentiablePerturbedAStar,
    PerturbAndMAPDistanceFunction,
    perturb_and_map_distance,
)

__all__ = [
    "DifferentiablePerturbedAStar",
    "PerturbAndMAPDistanceFunction",
    "perturb_and_map_distance",
]

