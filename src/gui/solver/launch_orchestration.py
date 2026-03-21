"""Solver launch/session orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def start_preview_for_current_map(
    *,
    gui,
    logger,
    pygame_module,
    multiprocessing_module,
    threading_module,
    time_module,
    run_preview_and_dump,
    start_preview_for_current_map_helper,
):
    start_preview_for_current_map_helper(
        gui=gui,
        logger=logger,
        pygame_module=pygame_module,
        multiprocessing_module=multiprocessing_module,
        threading_module=threading_module,
        time_module=time_module,
        run_preview_and_dump=run_preview_and_dump,
    )


def schedule_solver(*, gui, algorithm_idx, logger, time_module, threading_module, schedule_solver_helper):
    return schedule_solver_helper(
        gui=gui,
        algorithm_idx=algorithm_idx,
        logger=logger,
        time_module=time_module,
        threading_module=threading_module,
    )


def launch_solver_worker(
    *,
    gui,
    kwargs,
    logger,
    launch_solver_process,
    start_solver_thread_fallback,
    multiprocessing_module,
    launch_solver_worker_helper,
):
    launch_solver_worker_helper(
        gui=gui,
        kwargs=kwargs,
        logger=logger,
        launch_solver_process=launch_solver_process,
        start_solver_thread_fallback=start_solver_thread_fallback,
        multiprocessing_module=multiprocessing_module,
    )


def launch_solver_process(
    *,
    gui,
    launch_kwargs,
    run_solver_and_dump,
    multiprocessing_module,
    logger,
    launch_solver_process_helper,
):
    launch_solver_process_helper(
        gui=gui,
        launch_kwargs=launch_kwargs,
        run_solver_and_dump=run_solver_and_dump,
        multiprocessing_module=multiprocessing_module,
        logger=logger,
    )


def solver_thread_fallback_worker(
    *,
    gui,
    launch_kwargs,
    solve_in_subprocess,
    logger,
    solver_thread_fallback_worker_helper,
):
    solver_thread_fallback_worker_helper(
        gui=gui,
        launch_kwargs=launch_kwargs,
        solve_in_subprocess=solve_in_subprocess,
        logger=logger,
    )


def start_solver_thread_fallback(
    *,
    gui,
    launch_kwargs,
    threading_module,
    worker_target,
    logger,
    start_solver_thread_fallback_helper,
):
    start_solver_thread_fallback_helper(
        gui=gui,
        launch_kwargs=launch_kwargs,
        threading_module=threading_module,
        worker_target=worker_target,
        logger=logger,
    )
