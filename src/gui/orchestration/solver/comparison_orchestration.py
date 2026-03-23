"""Solver comparison and MAP-Elites orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.gameplay.map_elites_controls import (
    map_elites_worker as _map_elites_worker_flow_helper,
    start_map_elites as _start_map_elites_flow_helper,
)
from src.gui.rendering.helpers import (
    render_solver_comparison_overlay as _render_solver_comparison_overlay_helper,
)
from src.gui.solver.comparison_runner import (
    run_solver_comparison as _run_solver_comparison_helper,
    set_last_solver_metrics as _set_last_solver_metrics_helper,
)


def set_last_solver_metrics(gui, name, nodes, time_ms, path_len):
    """Record the latest metrics for a solver."""
    return _set_last_solver_metrics_helper(
        gui=gui,
        name=name,
        nodes=nodes,
        time_ms=time_ms,
        path_len=path_len,
    )


def run_solver_comparison(
    gui,
    *,
    logger,
    time_module,
    game_state_cls,
    solve_in_subprocess,
    threading_module,
):
    """Start asynchronous solver comparison worker."""
    return _run_solver_comparison_helper(
        gui=gui,
        logger=logger,
        time_module=time_module,
        game_state_cls=game_state_cls,
        solve_in_subprocess=solve_in_subprocess,
        threading_module=threading_module,
    )


def start_map_elites(gui, n_samples, resolution, *, threading_module):
    """Start MAP-Elites evaluation in a background worker."""
    return _start_map_elites_flow_helper(
        gui=gui,
        n_samples=n_samples,
        resolution=resolution,
        threading_module=threading_module,
    )


def map_elites_worker(gui, maps, n_samples, resolution, *, logger, os_module):
    """Execute MAP-Elites worker flow."""
    return _map_elites_worker_flow_helper(
        gui=gui,
        maps=maps,
        n_samples=n_samples,
        resolution=resolution,
        logger=logger,
        os_module=os_module,
    )


def render_solver_comparison_overlay(gui, surface, *, pygame):
    """Render sidebar overlay of comparison results."""
    _render_solver_comparison_overlay_helper(
        surface=surface,
        results=getattr(gui, "solver_comparison_results", None),
        screen_w=gui.screen_w,
        sidebar_width=gui.SIDEBAR_WIDTH,
        pygame=pygame,
    )
