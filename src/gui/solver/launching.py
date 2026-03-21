"""Helpers for solver worker launch and fallback execution."""

import os
import pickle
import tempfile
from typing import Any, Dict, Tuple


def create_solver_temp_files(grid_arr: Any) -> Tuple[str, str | None]:
    """Create output and optional grid temp files for solver worker launch."""
    fd_out, out_file = tempfile.mkstemp(prefix="zave_solver_out_", suffix=".pkl")
    os.close(fd_out)
    try:
        os.remove(out_file)
    except Exception:
        pass

    grid_file = None
    try:
        import numpy as _np

        fd, grid_file = tempfile.mkstemp(prefix="zave_grid_", suffix=".npy")
        os.close(fd)
        _np.save(grid_file, _np.array(grid_arr, dtype=_np.int64))
    except Exception:
        try:
            fd, grid_file = tempfile.mkstemp(prefix="zave_grid_", suffix=".pkl")
            os.close(fd)
            with open(grid_file, "wb") as grid_file_handle:
                pickle.dump(grid_arr, grid_file_handle)
        except Exception:
            grid_file = None

    return out_file, grid_file


def launch_solver_process(
    gui: Any,
    launch_kwargs: Dict[str, Any],
    run_solver_and_dump: Any,
    multiprocessing_module: Any,
    logger: Any,
) -> None:
    """Launch solver in a subprocess and update GUI solver state."""
    grid_arg = launch_kwargs["grid_file"] if launch_kwargs["grid_file"] else launch_kwargs["grid_arr"]
    logger.info(
        "SOLVER: Creating subprocess with gridfile=%s, outfile=%s",
        launch_kwargs["grid_file"],
        launch_kwargs["out_file"],
    )
    proc = multiprocessing_module.Process(
        target=run_solver_and_dump,
        args=(
            grid_arg,
            launch_kwargs["start"],
            launch_kwargs["goal"],
            launch_kwargs["alg_idx"],
            launch_kwargs["flags"],
            launch_kwargs["priority_options"],
            launch_kwargs["out_file"],
        ),
        kwargs={
            "graph": launch_kwargs["graph"],
            "room_to_node": launch_kwargs["room_to_node"],
            "room_positions": launch_kwargs["room_positions"],
            "node_to_room": launch_kwargs["node_to_room"],
        },
        daemon=True,
    )
    proc.start()
    logger.info("SOLVER: Subprocess started pid=%s, is_alive=%s", getattr(proc, "pid", None), proc.is_alive())
    gui.solver_proc = proc
    gui.solver_thread = None
    gui.solver_starting = False
    gui._set_message("Solver started in background")
    logger.info("SOLVER: Process handle stored, waiting for completion in run loop")


def solver_thread_fallback_worker(
    gui: Any,
    launch_kwargs: Dict[str, Any],
    solve_in_subprocess: Any,
    logger: Any,
) -> None:
    """Run solver in thread fallback and write result pickle for run-loop polling."""
    try:
        result = solve_in_subprocess(
            launch_kwargs["grid_arr"],
            launch_kwargs["start"],
            launch_kwargs["goal"],
            launch_kwargs["alg_idx"],
            launch_kwargs["flags"],
            launch_kwargs["priority_options"],
            graph=launch_kwargs["graph"],
            room_to_node=launch_kwargs["room_to_node"],
            room_positions=launch_kwargs["room_positions"],
            node_to_room=launch_kwargs["node_to_room"],
        )
        logger.info("SOLVER: Thread fallback completed, success=%s", result.get("success") if result else None)
        try:
            with open(launch_kwargs["out_file"], "wb") as out_file_handle:
                pickle.dump(result, out_file_handle)
            logger.info("SOLVER: Thread fallback wrote result to %s", launch_kwargs["out_file"])
        except Exception as write_err:
            logger.exception("SOLVER: Thread fallback failed to write output: %s", write_err)
    except Exception as solve_err:
        logger.exception("SOLVER: Thread fallback solver exception: %s", solve_err)
    finally:
        gui.solver_running = False
        gui.solver_start_time = None
        gui.solver_starting = False
        gui.solver_thread = None
        logger.info("SOLVER: Thread fallback finished, solver_running=False (main loop will poll results)")


def start_solver_thread_fallback(
    gui: Any,
    launch_kwargs: Dict[str, Any],
    threading_module: Any,
    worker_target: Any,
    logger: Any,
) -> None:
    """Start thread-based fallback worker, with robust state reset on failure."""
    try:
        thread = threading_module.Thread(target=worker_target, kwargs=launch_kwargs, daemon=True)
        gui.solver_thread = thread
        thread.start()
        gui._set_message("Solver started in background (thread fallback)")
    except Exception as thread_err:
        logger.exception("SOLVER: Failed to start thread fallback: %s", thread_err)
        gui.solver_running = False
        gui.solver_proc = None
        gui.solver_thread = None
        gui.solver_outfile = None
        if hasattr(gui, "solver_algorithm_idx"):
            delattr(gui, "solver_algorithm_idx")
        gui.solver_gridfile = None
        gui.solver_start_time = None
        gui.solver_starting = False
        gui._set_message("Failed to start solver")
        logger.error("SOLVER: Complete failure - all solver mechanisms exhausted")