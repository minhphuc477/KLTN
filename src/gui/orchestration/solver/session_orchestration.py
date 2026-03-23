"""Solver session-state orchestration bridges for ZeldaGUI."""

from __future__ import annotations

import os
import time

import numpy as np

from src.gui.solver.core_state import (
    clear_solver_state as _clear_solver_state,
    sync_solver_dropdown_settings as _sync_solver_dropdown_settings,
)
from src.gui.solver.prestart_cleanup import (
    cleanup_preview_before_solver_start as _cleanup_preview_before_solver_start,
    reset_solver_visual_state_before_start as _reset_solver_visual_state_before_start,
)
from src.gui.solver.recovery import (
    compute_solver_timeout_seconds as _compute_solver_timeout_seconds,
    force_solver_recovery_state as _force_solver_recovery_state,
    log_active_solver_state as _log_active_solver_state,
    prepare_active_solver_for_new_start as _prepare_active_solver_for_new_start,
    terminate_hung_solver_process as _terminate_hung_solver_process,
)
from src.gui.solver.start_flow import start_auto_solve as _start_auto_solve
from src.gui.solver.start_logic import evaluate_solver_recovery_state
from src.gui.solver.sync_execution import run_solver_sync as _run_solver_sync
from src.gui.solver.launching import create_solver_temp_files as _create_solver_temp_files_helper


def clear_solver_state(*, gui, reason, logger):
    _clear_solver_state(gui=gui, reason=reason, logger=logger)


def sync_solver_dropdown_settings(*, gui, sync_fn):
    return _sync_solver_dropdown_settings(gui=gui, sync_fn=sync_fn)


def start_auto_solve(*, gui, logger, debug_sync_solver):
    _start_auto_solve(gui=gui, logger=logger, debug_sync_solver=debug_sync_solver)


def prepare_active_solver_for_new_start(
    *,
    gui,
    logger,
    compute_timeout_seconds,
    terminate_hung_process,
    force_recovery_state,
    log_active_state,
):
    return _prepare_active_solver_for_new_start(
        gui=gui,
        logger=logger,
        time_module=time,
        evaluate_solver_recovery_state=evaluate_solver_recovery_state,
        compute_timeout_seconds=compute_timeout_seconds,
        terminate_hung_process=terminate_hung_process,
        force_recovery_state=force_recovery_state,
        log_active_state=log_active_state,
    )


def log_active_solver_state(*, gui, logger):
    _log_active_solver_state(gui=gui, logger=logger, os_module=os, time_module=time)


def compute_solver_timeout_seconds(
    *,
    gui,
    active_alg,
    default_solver_timeout_for_algorithm,
    scale_timeout_by_grid_size,
):
    return _compute_solver_timeout_seconds(
        gui=gui,
        active_alg=active_alg,
        default_solver_timeout_for_algorithm=default_solver_timeout_for_algorithm,
        scale_timeout_by_grid_size=scale_timeout_by_grid_size,
        np_module=np,
        os_module=os,
    )


def terminate_hung_solver_process(*, proc, logger):
    _terminate_hung_solver_process(proc=proc, logger=logger)


def force_solver_recovery_state(*, gui, recovery_reason, logger):
    _force_solver_recovery_state(gui=gui, recovery_reason=recovery_reason, logger=logger)


def cleanup_preview_before_solver_start(*, gui, logger):
    _cleanup_preview_before_solver_start(gui=gui, logger=logger, os_module=os)


def reset_solver_visual_state_before_start(*, gui):
    _reset_solver_visual_state_before_start(gui=gui)


def run_solver_sync(*, gui, logger, solve_in_subprocess, algorithm_idx):
    _run_solver_sync(
        gui=gui,
        logger=logger,
        solve_in_subprocess=solve_in_subprocess,
        algorithm_idx=algorithm_idx,
    )


def create_solver_temp_files(*, grid_arr):
    return _create_solver_temp_files_helper(grid_arr)
