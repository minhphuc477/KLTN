"""Pure helpers for solver-start synchronization and recovery decisions."""

from typing import Iterable, Tuple


_VALID_REPRESENTATIONS = {"tile", "graph", "hybrid"}
_REPRESENTATION_BY_INDEX = {0: "hybrid", 1: "tile", 2: "graph"}
_BASELINE_CELLS = 16 * 11 * 8


def sync_solver_dropdown_settings(
    algorithm_idx: int,
    search_representation: str,
    ara_weight: float,
    widgets: Iterable,
) -> Tuple[int, str, float]:
    """Read dropdown-like widget values and return normalized solver settings."""
    alg_idx = int(algorithm_idx or 0)
    rep_mode = str(search_representation or "hybrid").lower()
    weight = float(ara_weight or 1.0)

    for widget in widgets or []:
        control_name = getattr(widget, "control_name", None)
        if control_name == "algorithm":
            try:
                alg_idx = int(widget.selected)
            except Exception:
                pass
        elif control_name == "representation":
            rep_mode = _REPRESENTATION_BY_INDEX.get(getattr(widget, "selected", 0), rep_mode)
        elif control_name == "ara_weight":
            try:
                weight = float(widget.options[widget.selected])
            except Exception:
                pass

    if rep_mode not in _VALID_REPRESENTATIONS:
        rep_mode = "hybrid"
    if weight <= 0:
        weight = 1.0

    return alg_idx, rep_mode, float(weight)


def default_solver_timeout_for_algorithm(algorithm_idx: int) -> float:
    """Return a baseline timeout in seconds for a selected algorithm."""
    if algorithm_idx == 0:
        return 60.0
    if algorithm_idx in (1, 2):
        return 180.0
    if algorithm_idx == 3:
        return 90.0
    if algorithm_idx == 4:
        return 120.0
    return 240.0


def scale_timeout_by_grid_size(default_timeout: float, grid_cells: int) -> float:
    """Scale timeout by grid size, clamped to avoid extreme values."""
    if grid_cells <= 0:
        return float(default_timeout)
    scale = max(1.0, min(3.0, float(grid_cells) / float(_BASELINE_CELLS)))
    return float(default_timeout) * scale


def evaluate_solver_recovery_state(
    has_process: bool,
    process_alive: bool,
    solver_done: bool,
    solver_age: float,
    solver_timeout: float,
) -> Tuple[bool, str]:
    """Decide whether solver state is stale and should be force-recovered."""
    if not has_process:
        return True, "No process handle (spawn failed or already cleaned)"
    if not process_alive and not solver_done:
        return True, "Process dead but not marked done"
    if process_alive and solver_age > solver_timeout:
        return True, f"Process alive but timed out ({solver_age:.1f}s > {solver_timeout}s)"
    return False, ""
