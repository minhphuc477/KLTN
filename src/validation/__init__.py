"""Headless validation helpers with lazy visualization compatibility exports."""

from importlib import import_module
from typing import Any

from src.validation.end_to_end import (
    EndToEndValidationReport,
    ValidationStageEvidence,
    build_end_to_end_validation_report,
    validate_grid_representation,
)
from src.validation.pacing import evaluate_solution_path_pacing
from src.validation.topology import evaluate_graph_topology_characteristics
from src.validation.global_state import (
    GlobalStateValidationResult,
    validate_attached_global_state_contract,
    validate_global_state_progression,
)

_VISUALIZATION_EXPORTS = {
    "CollisionAlignmentRepairer",
    "CollisionAlignmentValidator",
    "CollisionConfig",
    "CollisionMismatch",
    "CollisionType",
    "ValidationResult",
}


def __getattr__(name: str) -> Any:
    """Load legacy collision-visualization contracts only when requested."""
    if name not in _VISUALIZATION_EXPORTS:
        raise AttributeError(name)
    module = import_module(
        "src.visualization.validation.collision_alignment_validator"
    )
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "CollisionAlignmentRepairer",
    "CollisionAlignmentValidator",
    "CollisionConfig",
    "CollisionMismatch",
    "CollisionType",
    "ValidationResult",
    "EndToEndValidationReport",
    "ValidationStageEvidence",
    "build_end_to_end_validation_report",
    "validate_grid_representation",
    "evaluate_solution_path_pacing",
    "evaluate_graph_topology_characteristics",
    "GlobalStateValidationResult",
    "validate_attached_global_state_contract",
    "validate_global_state_progression",
]
