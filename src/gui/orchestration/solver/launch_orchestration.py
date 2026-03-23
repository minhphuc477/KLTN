"""Solver launch/session orchestration bridges for ZeldaGUI."""

from __future__ import annotations

import multiprocessing
import threading
import time

try:
    import pygame
except ImportError:  # pragma: no cover - guarded for headless environments
    pygame = None

from src.gui.gameplay.preview_startup import start_preview_for_current_map as _start_preview_for_current_map
from src.gui.solver.launching import (
    launch_solver_process as _launch_solver_process,
    solver_thread_fallback_worker as _solver_thread_fallback_worker,
    start_solver_thread_fallback as _start_solver_thread_fallback,
)
from src.gui.solver.scheduling import schedule_solver as _schedule_solver
from src.gui.solver.worker_bootstrap import launch_solver_worker as _launch_solver_worker


def start_preview_for_current_map(
    *,
    gui,
    logger,
    run_preview_and_dump,
):
    _start_preview_for_current_map(
        gui=gui,
        logger=logger,
        pygame_module=pygame,
        multiprocessing_module=multiprocessing,
        threading_module=threading,
        time_module=time,
        run_preview_and_dump=run_preview_and_dump,
    )


def schedule_solver(*, gui, algorithm_idx, logger):
    return _schedule_solver(
        gui=gui,
        algorithm_idx=algorithm_idx,
        logger=logger,
        time_module=time,
        threading_module=threading,
    )


def launch_solver_worker(
    *,
    gui,
    kwargs,
    logger,
    launch_solver_process,
    start_solver_thread_fallback,
):
    _launch_solver_worker(
        gui=gui,
        kwargs=kwargs,
        logger=logger,
        launch_solver_process=launch_solver_process,
        start_solver_thread_fallback=start_solver_thread_fallback,
        multiprocessing_module=multiprocessing,
    )


def launch_solver_process(
    *,
    gui,
    launch_kwargs,
    run_solver_and_dump,
    logger,
):
    _launch_solver_process(
        gui=gui,
        launch_kwargs=launch_kwargs,
        run_solver_and_dump=run_solver_and_dump,
        multiprocessing_module=multiprocessing,
        logger=logger,
    )


def solver_thread_fallback_worker(
    *,
    gui,
    launch_kwargs,
    solve_in_subprocess,
    logger,
):
    _solver_thread_fallback_worker(
        gui=gui,
        launch_kwargs=launch_kwargs,
        solve_in_subprocess=solve_in_subprocess,
        logger=logger,
    )


def start_solver_thread_fallback(
    *,
    gui,
    launch_kwargs,
    worker_target,
    logger,
):
    _start_solver_thread_fallback(
        gui=gui,
        launch_kwargs=launch_kwargs,
        threading_module=threading,
        worker_target=worker_target,
        logger=logger,
    )
