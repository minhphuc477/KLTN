"""Stable solver outcome status helpers shared by simulation and evaluation."""

from __future__ import annotations


def oracle_status_from_outcome(success: bool, failure_reason: str) -> str:
    """Normalize solver outcomes into the repository status vocabulary."""
    if bool(success):
        return "solved"
    reason = str(failure_reason or "").strip().lower()
    if not reason:
        return "failed"
    if "timeout" in reason:
        return "timeout"
    if "no path" in reason:
        return "no_path"
    if "no goal" in reason or "no start" in reason:
        return "invalid_map"
    return "failed"
