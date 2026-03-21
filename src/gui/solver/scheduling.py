"""Helpers for orchestrating solver scheduling from GUI state."""

from typing import Any, Optional


def schedule_solver(gui: Any, algorithm_idx: Optional[int], logger: Any, time_module: Any, threading_module: Any) -> bool:
    """Schedule solver execution with lock-safe slot reservation and async launch."""
    # Make scheduling atomic to avoid races when multiple callers attempt to
    # start the solver concurrently.
    with gui._solver_lock:
        if gui.solver_running:
            gui._set_message("Solver already running...")
            logger.warning("SOLVER: _schedule_solver blocked - solver_running already True")
            return False
        # Reserve solver slot immediately (prevents TOCTOU).
        gui.solver_running = True
        gui.solver_done = False
        gui.solver_start_time = time_module.time()
        gui.solver_starting = True

    gui._sync_solver_dropdown_settings()

    # Use explicit algorithm_idx parameter when provided.
    if algorithm_idx is not None:
        current_alg_idx = algorithm_idx
        gui.algorithm_idx = algorithm_idx
        logger.info("SOLVER_FIX: Using explicit algorithm_idx=%d passed to _schedule_solver()", algorithm_idx)
    else:
        current_alg_idx = getattr(gui, "algorithm_idx", None)
        logger.info("SOLVER: Using self.algorithm_idx=%s (no explicit arg provided)", current_alg_idx)

    logger.info(
        "SOLVER: Acquired solver lock, solver_running=True, solver_done=False, start_time=%.3f, algorithm_idx=%s",
        gui.solver_start_time,
        current_alg_idx,
    )

    gui.solver_algorithm_idx = current_alg_idx if current_alg_idx is not None else 0
    gui._auto_recenter_done = False

    request = gui._build_solver_request(algorithm_idx=current_alg_idx)
    if request is None:
        gui._clear_solver_state(reason="missing start/goal")
        logger.warning("SOLVER: Missing start/goal - cleared solver state")
        return False

    out_file, grid_file = gui._create_solver_temp_files(request["grid_arr"])
    logger.info("SOLVER: Starting subprocess, pickle_path=%s", out_file)
    logger.info(
        "SOLVER: start=%s, goal=%s, algorithm_idx=%s",
        request["start"],
        request["goal"],
        request["alg_idx"],
    )

    # Set outfile/gridfile immediately so run loop can find them.
    gui.solver_outfile = out_file
    gui.solver_gridfile = grid_file
    logger.info("SOLVER: File handles stored: outfile=%s, gridfile=%s", out_file, grid_file)

    launch_kwargs = {
        "grid_arr": request["grid_arr"],
        "start": request["start"],
        "goal": request["goal"],
        "alg_idx": request["alg_idx"],
        "flags": request["flags"],
        "priority_options": request["priority_options"],
        "out_file": out_file,
        "grid_file": grid_file,
        "graph": request["graph"],
        "room_to_node": request["room_to_node"],
        "room_positions": request["room_positions"],
        "node_to_room": request["node_to_room"],
    }

    # Start the process on a worker thread to avoid blocking caller.
    try:
        worker = threading_module.Thread(target=gui._launch_solver_worker, kwargs=launch_kwargs, daemon=True)
        worker.start()
    except Exception:
        gui._launch_solver_worker(**launch_kwargs)

    return True