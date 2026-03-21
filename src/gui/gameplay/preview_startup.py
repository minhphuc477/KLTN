"""Helpers for non-blocking preview startup after map load."""

import os
import tempfile
from typing import Any


def start_preview_for_current_map(
    gui: Any,
    logger: Any,
    pygame_module: Any,
    multiprocessing_module: Any,
    threading_module: Any,
    time_module: Any,
    run_preview_and_dump: Any,
) -> None:
    """Run lightweight post-load preview workflow without blocking UI."""
    gui._sync_solver_dropdown_settings()

    if getattr(gui, "env", None) and getattr(gui.env, "done", False):
        try:
            gui._load_current_map()
        except Exception:
            logger.exception("Failed to reload current map during preview startup")

    try:
        scan_start = time_module.time()
        gui._scan_and_mark_items()
        scan_duration = (time_module.time() - scan_start)
        if scan_duration > 0.05:
            logger.debug("Item scan took %.3fs", scan_duration)
    except Exception:
        logger.exception("Item scanning failed")

    solver_name = gui._algorithm_name(gui.algorithm_idx) if hasattr(gui, "algorithm_idx") else "A*"
    gui.message = f"Solving ({solver_name})..."
    try:
        gui._render()
        pygame_module.display.flip()
    except Exception:
        logger.debug("Render/flip failed during preview startup (may be uninitialized yet)")

    logger.info(
        "Starting solver: algorithm=%s, solver_running=%s",
        solver_name,
        getattr(gui, "solver_running", False),
    )

    if os.environ.get("KLTN_DISABLE_PREVIEW") == "1":
        logger.info("Automatic preview startup disabled via KLTN_DISABLE_PREVIEW=1")
        return

    if getattr(gui, "preview_thread", None) and getattr(gui, "preview_thread").is_alive():
        gui._set_message("Preview already running...")
        return

    def _preview_worker() -> None:
        try:
            current_dungeon = gui.maps[gui.current_map_idx]
            if hasattr(current_dungeon, "graph") and current_dungeon.graph and not getattr(gui, "force_grid_algorithm", False):
                try:
                    from src.data.zelda_core import DungeonSolver, ValidationMode

                    solver = DungeonSolver()
                    result = solver.solve(current_dungeon, mode=ValidationMode.FULL)
                    if result.get("solvable", False):
                        gui.solver_result = result
                        success, path, teleports = gui._smart_grid_path()
                        if success:
                            gui.preview_result = {"path": path, "solver_result": result, "teleports": teleports}
                            return

                        success2, path2, teleports2 = gui._graph_guided_path()
                        if success2:
                            gui.preview_result = {"path": path2, "solver_result": result, "teleports": teleports2}
                            return
                except Exception:
                    logger.debug("Graph-based quick solve failed in worker", exc_info=True)

            try:
                success, path, teleports = gui._smart_grid_path()
                if success:
                    gui.preview_result = {"path": path, "solver_result": {}, "teleports": teleports}
                    return
            except Exception:
                logger.debug("Grid quick solve failed in worker", exc_info=True)
        finally:
            gui.preview_thread = None

        if os.environ.get("KLTN_ALLOW_HEAVY", "1") == "1":
            try:
                gui._schedule_solver()
            except Exception:
                logger.exception("Failed to schedule heavy solver from preview worker")
        else:
            gui._set_message("No preview found; heavy solver disabled", 3.0)

    def _spawn_preview_process_async() -> None:
        try:
            import numpy as np

            current = gui.maps[gui.current_map_idx]
            grid_data = current.global_grid if hasattr(current, "global_grid") else current
            graph = getattr(current, "graph", None)
            room_to_node = getattr(current, "room_to_node", None)
            room_positions = getattr(current, "room_positions", None)
            node_to_room = getattr(current, "node_to_room", None)

            fd, grid_file = tempfile.mkstemp(prefix="zave_preview_", suffix=".npy")
            os.close(fd)
            np.save(grid_file, np.array(grid_data, dtype=np.int64))

            fd_out, preview_out = tempfile.mkstemp(prefix="zave_preview_out_", suffix=".pkl")
            os.close(fd_out)
            try:
                os.remove(preview_out)
            except Exception:
                pass

            start_pos = getattr(getattr(gui, "env", None), "start_pos", None)
            goal_pos = getattr(getattr(gui, "env", None), "goal_pos", None)
            if start_pos is None or goal_pos is None:
                logger.warning("Preview process skipped: missing start/goal positions")
                gui.preview_result = None
                gui.preview_done = True
                gui._set_message("Preview unavailable (missing start/goal)", 2.0)
                return

            preview_priority_options = {
                "tie_break": gui.feature_flags.get("priority_tie_break", False),
                "key_boost": gui.feature_flags.get("priority_key_boost", False),
                "enable_ara": gui.feature_flags.get("enable_ara", False),
                "ara_weight": float(getattr(gui, "ara_weight", 1.0)),
                "representation": str(getattr(gui, "search_representation", "hybrid")),
                "allow_diagonals": True,
            }

            proc = multiprocessing_module.Process(
                target=run_preview_and_dump,
                args=(
                    grid_file,
                    tuple(start_pos),
                    tuple(goal_pos),
                    getattr(gui, "algorithm_idx", 0),
                    dict(gui.feature_flags),
                    preview_priority_options,
                    preview_out,
                ),
                kwargs={
                    "graph": graph,
                    "room_to_node": room_to_node,
                    "room_positions": room_positions,
                    "node_to_room": node_to_room,
                },
                daemon=True,
            )
            logger.debug(
                "Starting preview process for map %s -> outfile=%s gridfile=%s",
                gui.current_map_idx,
                preview_out,
                grid_file,
            )
            proc.start()
            logger.debug("Preview process started pid=%s alive=%s", getattr(proc, "pid", None), proc.is_alive())
            gui.preview_proc = proc
            gui.preview_outfile = preview_out
            gui.preview_gridfile = grid_file
            gui.preview_result = None
            gui.preview_done = False
            gui._set_message("Preview search started (background)")
        except Exception:
            logger.exception("Failed to spawn preview process; falling back to threaded worker")
            gui.preview_result = None
            gui.preview_thread = threading_module.Thread(target=_preview_worker, daemon=True)
            gui.preview_thread.start()
            gui._set_message("Preview search started (thread)")

    startup_thread = threading_module.Thread(target=_spawn_preview_process_async, daemon=True)
    startup_thread.start()
