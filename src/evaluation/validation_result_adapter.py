"""Normalization helpers for validation results across modules.

Different validators in this repository expose similar solvability outcomes
with slightly different schemas. These helpers provide one canonical view for
evaluation/reporting code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class CanonicalValidationResult:
    """Canonical read model for solvability and path metrics."""

    is_solvable: bool
    path_length: int
    solution_path: List[Any]
    failure_reason: str


def normalize_validation_result(result: Any) -> CanonicalValidationResult:
    """Convert arbitrary validator output into a canonical shape.

    Supports object-based result dataclasses and dict-like payloads.
    """

    if result is None:
        return CanonicalValidationResult(False, 0, [], "Missing validation result")

    if isinstance(result, dict):
        is_solvable = bool(result.get("is_solvable", result.get("solvable", False)))
        solution_path_raw = result.get("solution_path", result.get("path", []))
        path_length_raw = result.get("path_length", 0)
        failure_reason = str(result.get("failure_reason", result.get("error_message", "")) or "")
    else:
        is_solvable = bool(getattr(result, "is_solvable", getattr(result, "solvable", False)))
        solution_path_raw = getattr(result, "solution_path", getattr(result, "path", []))
        path_length_raw = getattr(result, "path_length", 0)
        failure_reason = str(
            getattr(result, "failure_reason", getattr(result, "error_message", "")) or ""
        )

    if solution_path_raw is None:
        solution_path: List[Any] = []
    elif isinstance(solution_path_raw, list):
        solution_path = solution_path_raw
    else:
        solution_path = list(solution_path_raw)

    try:
        path_length = int(path_length_raw)
    except (TypeError, ValueError):
        path_length = 0

    if path_length <= 0 and solution_path:
        path_length = max(0, len(solution_path) - 1)

    return CanonicalValidationResult(
        is_solvable=is_solvable,
        path_length=max(0, path_length),
        solution_path=solution_path,
        failure_reason=failure_reason,
    )
