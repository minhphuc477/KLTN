"""Stable solver outcome status helpers shared by simulation and evaluation."""

from __future__ import annotations

from typing import Final


SOLUTION_TERMINATIONS: Final = frozenset({"solved"})
NO_PATH_PROOF_TERMINATIONS: Final = frozenset({"exhausted"})
INVALID_PROBLEM_TERMINATIONS: Final = frozenset({"invalid"})
RESOURCE_LIMIT_TERMINATIONS: Final = frozenset(
    {"budget_exhausted", "depth_limit", "incomplete"}
)
ERROR_TERMINATIONS: Final = frozenset(
    {"route_replay_failed", "validator_error"}
)


def search_proof_status(success: bool, termination_status: str | None) -> str:
    """Classify what a search run actually proves.

    A found route is evidence only when the solver reports a successful
    termination.  A failed run proves absence of a route only when a complete
    finite-state frontier was emptied.  Resource limits and implementation
    errors remain inconclusive.
    """
    termination = str(termination_status or "").strip().lower()
    if bool(success):
        if termination in SOLUTION_TERMINATIONS or not termination:
            return "solution_witness"
        return "inconsistent"
    if termination in NO_PATH_PROOF_TERMINATIONS:
        return "no_path_proof"
    if termination in INVALID_PROBLEM_TERMINATIONS:
        return "invalid_problem"
    if termination in RESOURCE_LIMIT_TERMINATIONS:
        return "resource_limited"
    if termination in ERROR_TERMINATIONS:
        return "execution_error"
    return "inconclusive"


def proven_unsolvable_from_outcome(success: bool, termination_status: str | None) -> bool:
    """Return True only for complete frontier exhaustion without a solution."""
    return search_proof_status(success, termination_status) == "no_path_proof"


def search_outcome_is_indeterminate(success: bool, termination_status: str | None) -> bool:
    """Return whether the run supplies neither a solution nor a no-path proof."""
    return search_proof_status(success, termination_status) in {
        "resource_limited",
        "execution_error",
        "inconclusive",
        "inconsistent",
    }


def oracle_status_from_outcome(
    success: bool,
    failure_reason: str,
    termination_status: str | None = None,
) -> str:
    """Normalize a structured outcome without parsing human-readable text."""
    termination = str(termination_status or "").strip().lower()
    proof = search_proof_status(success, termination)
    if proof == "solution_witness":
        return "solved"
    if proof == "no_path_proof":
        return "no_path"
    if proof == "invalid_problem":
        return "invalid_map"
    explicit_statuses = {
        "budget_exhausted": "timeout",
        "depth_limit": "inconclusive",
        "incomplete": "inconclusive",
        "route_replay_failed": "failed",
        "validator_error": "failed",
    }
    if termination in explicit_statuses:
        return explicit_statuses[termination]
    _ = failure_reason  # retained for API compatibility; prose is not evidence
    return "failed"
