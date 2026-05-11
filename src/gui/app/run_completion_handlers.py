"""Post-event-loop completion handlers for gui_runner run loop."""

from __future__ import annotations

from src.gui.components.constants import GUI_ALGORITHM_NAMES


def handle_parallel_search_completion(gui, logger, path_preview_dialog_cls):
    """Apply finished parallel-search result on the main thread."""
    if not (getattr(gui, "parallel_search_done", False) and getattr(gui, "parallel_search_result", None)):
        return

    best = gui.parallel_search_result
    name = (
        GUI_ALGORITHM_NAMES[best["alg"]]
        if 0 <= best["alg"] < min(4, len(GUI_ALGORITHM_NAMES))
        else f"Alg{best['alg']}"
    )
    gui._set_message(f"Parallel best: {name} ({best['nodes']} nodes, {best['time_ms']:.0f}ms)")
    gui.parallel_search_done = False
    gui.parallel_search_result = None

    try:
        gui.auto_path = best["path"]
        gui.preview_overlay_visible = True
        logger.debug("Parallel search: setting preview_overlay_visible=True (parallel best path)")
        gui.path_preview_dialog = path_preview_dialog_cls(
            path=gui.auto_path,
            env=gui.env,
            solver_result={},
            speed_multiplier=gui.speed_multiplier,
        )
        gui._set_message("Parallel result ready (sidebar preview)")
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning(f"Failed to display parallel search preview: {exc}")


def handle_preview_process_completion(gui, os_module, logger, safe_unpickle_fn, path_preview_dialog_cls):
    """Apply finished preview-process result on the main thread."""
    proc = getattr(gui, "preview_proc", None)
    if not (proc and not getattr(gui, "preview_done", False)):
        return

    proc_alive = False
    try:
        proc_alive = proc.is_alive()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        proc_alive = False

    if not proc_alive:
        out = getattr(gui, "preview_outfile", None)
        res = None
        try:
            if out:
                res = safe_unpickle_fn(out)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("Failed to read preview output: %s", exc)
        finally:
            try:
                proc.join(timeout=0.1)
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
            try:
                if out and os_module.path.exists(out):
                    os_module.remove(out)
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
            try:
                grid_file = getattr(gui, "preview_gridfile", None)
                if grid_file and os_module.path.exists(grid_file):
                    os_module.remove(grid_file)
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
            gui.preview_proc = None
            gui.preview_outfile = None
            gui.preview_gridfile = None
            gui.preview_done = True

        if res:
            try:
                if res.get("success") and res.get("path"):
                    gui.auto_path = res.get("path")
                    gui.preview_overlay_visible = True
                    logger.debug("Preview result: setting preview_overlay_visible=True (preview has path)")
                    try:
                        solver_result_preview = (res.get("solver_result") or {}) if res else {}
                        gui.path_preview_dialog = path_preview_dialog_cls(
                            path=gui.auto_path,
                            env=gui.env,
                            solver_result=solver_result_preview,
                            speed_multiplier=gui.speed_multiplier,
                        )
                    except (AttributeError, RuntimeError, ValueError, TypeError):
                        gui.path_preview_dialog = None
                    gui._set_message("Preview ready (sidebar)")
                else:
                    msg = res.get("message") or "Preview finished with no path"
                    gui._set_message(msg)
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.exception("Failed to apply preview output on main thread: %s", exc)
        else:
            gui._set_message("Preview finished (no output)")
        gui.preview_done = True

    if getattr(gui, "solver_running", False):
        gui.status_message = "Solving..."
    else:
        gui.status_message = "Ready"


def handle_solver_process_completion(
    gui,
    os_module,
    time_module,
    np_module,
    logger,
    compute_solver_timeout_seconds_fn,
    safe_unpickle_fn,
    find_path_tile_violations_fn,
    path_preview_dialog_cls,
):
    """Apply finished solver-process (or thread fallback) result on the main thread."""
    if getattr(gui, "solver_done", False):
        return

    proc = getattr(gui, "solver_proc", None)
    solver_thread = getattr(gui, "solver_thread", None)
    proc_alive = False
    thread_alive = False
    solver_starting = getattr(gui, "solver_starting", False)

    active_alg = int(getattr(gui, "solver_algorithm_idx", getattr(gui, "algorithm_idx", 0)))
    grid_cells = None
    try:
        current_map = gui.maps[gui.current_map_idx]
        grid_ref = current_map.global_grid if hasattr(current_map, "global_grid") else current_map
        grid_cells = int(np_module.asarray(grid_ref).size)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass

    solver_timeout = compute_solver_timeout_seconds_fn(
        active_alg,
        grid_cell_count=grid_cells,
        env_getter=os_module.environ.get,
    )
    solver_start_time = getattr(gui, "solver_start_time", None)
    timed_out = False

    if proc and solver_start_time and (time_module.time() - solver_start_time) > solver_timeout:
        timed_out = True
        logger.error("SOLVER: TIMEOUT after %.1fs - forcefully terminating", solver_timeout)
        if proc:
            try:
                proc.join(timeout=0.2)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.exception("SOLVER: Failed to terminate timed-out process: %s", exc)
        proc_alive = False

    if not timed_out:
        try:
            proc_alive = proc.is_alive() if proc else False
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("SOLVER: proc.is_alive() raised exception: %s", exc)
            proc_alive = False
        try:
            thread_alive = solver_thread.is_alive() if solver_thread else False
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("SOLVER: solver_thread.is_alive() raised exception: %s", exc)
            thread_alive = False

    try:
        startup_grace = float(os_module.environ.get("KLTN_SOLVER_STARTUP_GRACE", "1.5"))
    except (TypeError, ValueError):
        startup_grace = 1.5
    solver_age = (time_module.time() - solver_start_time) if solver_start_time else 0.0
    out = getattr(gui, "solver_outfile", None)
    out_exists = os_module.path.exists(out) if out else False

    if (
        solver_starting
        and proc is None
        and not thread_alive
        and not out_exists
        and not timed_out
        and solver_age < startup_grace
    ):
        logger.debug("SOLVER: Waiting for process start (age=%.2fs < %.2fs grace)", solver_age, startup_grace)
        return

    if thread_alive:
        logger.debug("SOLVER: Waiting for thread fallback completion (age=%.2fs)", solver_age)
        return

    if proc is not None and proc_alive:
        return

    try:
        proc_exitcode = None
        if proc is not None:
            proc_exitcode = getattr(proc, "exitcode", None)
            logger.info("SOLVER: Subprocess done, proc.is_alive()=False, exitcode=%s", proc.exitcode)
        else:
            logger.info("SOLVER: No subprocess handle (thread fallback or spawn failure)")

        out = getattr(gui, "solver_outfile", None)
        logger.info("SOLVER: Reading result from %s, exists=%s", out, os_module.path.exists(out) if out else "N/A")
        res = None
        try:
            if out:
                res = safe_unpickle_fn(out)
                path_len = len(res.get("path", []) or []) if res else 0
                solver_result_safe = (res.get("solver_result") or {}) if res else {}
                logger.info(
                    "SOLVER: Result loaded, path_len=%d, success=%s, keys=%s",
                    path_len,
                    res.get("success") if res else None,
                    solver_result_safe.get("keys_used", "N/A"),
                )
            else:
                logger.warning("SOLVER: Output file missing or path is None: %s", out)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("SOLVER: Failed to read solver output: %s", exc)

        if res:
            try:
                if res.get("success") and res.get("path"):
                    gui.auto_path = res.get("path")
                    solver_result = (res.get("solver_result") or {}) if res else {}

                    water_violations = find_path_tile_violations_fn(
                        gui.auto_path,
                        gui.env.grid,
                        blocked_tile_ids={40},
                    )

                    if water_violations:
                        print(f"\n{'='*60}")
                        print("ERROR: PATH GOES THROUGH WATER!")
                        print(f"Found {len(water_violations)} water tiles in path:")
                        for step, r, c, tid in water_violations[:5]:
                            print(f"  Step {step}: position ({r}, {c}) = tile ID {tid} (WATER)")
                        print(f"{'='*60}\n")
                        logger.error("PATH ERROR: %d water tiles in path!", len(water_violations))
                    else:
                        print(f"\n{'='*60}")
                        print("PATH VERIFIED: No water tiles")
                        print(f"Path length: {len(gui.auto_path)} steps")
                        print(f"{'='*60}\n")

                    print(f"\n{'='*60}")
                    print(f"PATH LOADED: {len(gui.auto_path)} steps")
                    if len(gui.auto_path) > 10:
                        print(f"First 10 steps: {gui.auto_path[:10]}")
                    print(f"{'='*60}\n")

                    logger.info(
                        "SOLVER: Path applied! auto_path len=%d, first=%s, last=%s",
                        len(gui.auto_path),
                        gui.auto_path[0] if gui.auto_path else None,
                        gui.auto_path[-1] if gui.auto_path else None,
                    )

                    force_preview = bool(getattr(gui, "preview_on_next_solver_result", False))
                    if getattr(gui, "auto_start_solver", False) and not force_preview:
                        logger.info("SOLVER: auto_start_solver=True, starting animation immediately")
                        gui._execute_auto_solve(gui.auto_path, solver_result, teleports=0)
                        gui._set_message(f"Auto-solve started! Path: {len(gui.auto_path)} steps")
                        logger.info(
                            "SOLVER: Animation started, auto_mode=%s, auto_step_idx=%s",
                            getattr(gui, "auto_mode", None),
                            getattr(gui, "auto_step_idx", None),
                        )
                    else:
                        logger.info(
                            "SOLVER: showing preview dialog (auto_start_solver=%s, force_preview=%s)",
                            getattr(gui, "auto_start_solver", False),
                            force_preview,
                        )
                        gui.path_preview_dialog = path_preview_dialog_cls(
                            path=gui.auto_path,
                            env=gui.env,
                            solver_result=solver_result,
                            speed_multiplier=gui.speed_multiplier,
                        )
                        if getattr(gui, "preview_modal_enabled", False):
                            gui.path_preview_mode = True
                            gui.preview_overlay_visible = False
                        else:
                            gui.path_preview_mode = False
                            gui.preview_overlay_visible = True
                        gui._set_message("Solver finished (press ENTER to start or ESC to dismiss)")
                    gui.preview_on_next_solver_result = False
                else:
                    msg = res.get("message") or "Solver finished with no path"
                    if timed_out:
                        msg = f"Solver timed out after {int(solver_timeout)}s"
                        logger.info("SOLVER: %s", msg)
                    elif msg == "output file missing" and proc_exitcode is not None and proc_exitcode < 0:
                        msg = f"Solver terminated (exitcode={proc_exitcode}) before writing output"
                        logger.info("SOLVER: %s", msg)
                    else:
                        logger.warning("SOLVER: No valid path in result: %s", msg)
                    gui._set_message(msg)
                    gui.preview_on_next_solver_result = False
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.exception("SOLVER: Failed to apply result on main thread: %s", exc)
                gui._set_message("Solver error (see logs)")
                gui.preview_on_next_solver_result = False
        else:
            if timed_out:
                logger.error("SOLVER: No result - subprocess timed out")
                gui._set_message(f"Solver timed out after {int(solver_timeout)}s")
            elif proc_exitcode is not None and proc_exitcode < 0:
                logger.info(
                    "SOLVER: Subprocess terminated before output (exitcode=%s)",
                    proc_exitcode,
                )
                gui._set_message(f"Solver terminated (exitcode={proc_exitcode})")
            else:
                logger.warning("SOLVER: No result loaded (res is None), subprocess may have crashed")
                gui._set_message("Solver finished (no output)")
            gui.preview_on_next_solver_result = False
    finally:
        logger.info("SOLVER: Entering cleanup finally block")
        try:
            if proc:
                proc.join(timeout=0.1)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("SOLVER: proc.join() failed: %s", exc)
        try:
            out = getattr(gui, "solver_outfile", None)
            if out and os_module.path.exists(out):
                os_module.remove(out)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("SOLVER: Failed to remove output file: %s", exc)
        try:
            grid_file = getattr(gui, "solver_gridfile", None)
            if grid_file and os_module.path.exists(grid_file):
                os_module.remove(grid_file)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("SOLVER: Failed to remove grid file: %s", exc)

        gui._clear_solver_state(reason="solver completed/failed")


def handle_ai_generation_completion(gui):
    """Apply finished AI-generation worker result on the main thread."""
    if not getattr(gui, "ai_gen_done", False):
        return

    res = getattr(gui, "ai_gen_result", None)
    gui.ai_gen_done = False
    gui.ai_gen_result = None
    if res and res.get("success") and res.get("grid") is not None:
        gui.maps.append(res["grid"])
        gui.map_names.append(res.get("name", "AI Generated"))
        gui.current_map_idx = len(gui.maps) - 1
        gui._load_current_map()
        gui._center_view()
        if getattr(gui, "effects", None):
            try:
                gui.effects.clear()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
        gui.step_count = 0
        gui.auto_path = []
        gui.auto_mode = False
        if res.get("clear_mixed_constraints"):
            gui.ai_constraint_boss_norm = None
            gui.ai_constraint_lock_norm = None
            gui.ai_constraint_key_norm = None
        if res.get("mission_graph_draft") is not None:
            gui.ai_mission_graph_draft = res["mission_graph_draft"]
        gui._set_message(res.get("message", "AI generation complete"), 3.0)
    else:
        message = "AI generation failed"
        if res and res.get("message"):
            message = res["message"]
        elif res and res.get("error"):
            message = f"AI generation failed: {res['error']}"
        gui._set_message(message, 3.0)
