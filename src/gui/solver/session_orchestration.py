"""Solver session-state orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def clear_solver_state(*, gui, reason, logger, clear_solver_state_helper):
    clear_solver_state_helper(gui=gui, reason=reason, logger=logger)


def sync_solver_dropdown_settings(*, gui, sync_fn, sync_solver_dropdown_settings_helper):
    return sync_solver_dropdown_settings_helper(gui=gui, sync_fn=sync_fn)


def start_auto_solve(*, gui, logger, debug_sync_solver, start_auto_solve_helper):
    start_auto_solve_helper(gui=gui, logger=logger, debug_sync_solver=debug_sync_solver)


def prepare_active_solver_for_new_start(
    *,
    gui,
    logger,
    time_module,
    evaluate_solver_recovery_state,
    compute_timeout_seconds,
    terminate_hung_process,
    force_recovery_state,
    log_active_state,
    prepare_active_solver_for_new_start_helper,
):
    return prepare_active_solver_for_new_start_helper(
        gui=gui,
        logger=logger,
        time_module=time_module,
        evaluate_solver_recovery_state=evaluate_solver_recovery_state,
        compute_timeout_seconds=compute_timeout_seconds,
        terminate_hung_process=terminate_hung_process,
        force_recovery_state=force_recovery_state,
        log_active_state=log_active_state,
    )


def log_active_solver_state(*, gui, logger, os_module, time_module, log_active_solver_state_helper):
    log_active_solver_state_helper(gui=gui, logger=logger, os_module=os_module, time_module=time_module)


def compute_solver_timeout_seconds(
    *,
    gui,
    active_alg,
    default_solver_timeout_for_algorithm,
    scale_timeout_by_grid_size,
    np_module,
    os_module,
    compute_solver_timeout_seconds_helper,
):
    return compute_solver_timeout_seconds_helper(
        gui=gui,
        active_alg=active_alg,
        default_solver_timeout_for_algorithm=default_solver_timeout_for_algorithm,
        scale_timeout_by_grid_size=scale_timeout_by_grid_size,
        np_module=np_module,
        os_module=os_module,
    )


def terminate_hung_solver_process(*, proc, logger, terminate_hung_solver_process_helper):
    terminate_hung_solver_process_helper(proc=proc, logger=logger)


def force_solver_recovery_state(*, gui, recovery_reason, logger, force_solver_recovery_state_helper):
    force_solver_recovery_state_helper(gui=gui, recovery_reason=recovery_reason, logger=logger)


def cleanup_preview_before_solver_start(*, gui, logger, os_module, cleanup_preview_before_solver_start_helper):
    cleanup_preview_before_solver_start_helper(gui=gui, logger=logger, os_module=os_module)


def reset_solver_visual_state_before_start(*, gui, reset_solver_visual_state_before_start_helper):
    reset_solver_visual_state_before_start_helper(gui=gui)


def run_solver_sync(*, gui, logger, solve_in_subprocess, algorithm_idx, run_solver_sync_helper):
    run_solver_sync_helper(
        gui=gui,
        logger=logger,
        solve_in_subprocess=solve_in_subprocess,
        algorithm_idx=algorithm_idx,
    )


def create_solver_temp_files(*, grid_arr, create_solver_temp_files_helper):
    return create_solver_temp_files_helper(grid_arr)
