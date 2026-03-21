"""Environment-driven runtime flags for gui_runner."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GuiRuntimeFlags:
    log_level: str
    debug_input_active: bool
    debug_sync_solver: bool
    debug_solver_flow: bool


def load_runtime_flags() -> GuiRuntimeFlags:
    """Load GUI runtime flags from environment variables."""
    log_level = os.environ.get("KLTN_LOG_LEVEL", "").upper()
    return GuiRuntimeFlags(
        log_level=log_level,
        debug_input_active=os.environ.get("KLTN_DEBUG_INPUT", "") == "1",
        debug_sync_solver=os.environ.get("KLTN_SYNC_SOLVER", "0") == "1",
        debug_solver_flow=os.environ.get("KLTN_DEBUG_SOLVER_FLOW", "0") == "1",
    )
