"""Helpers for background solver-comparison execution in GUI."""

from typing import Any


def set_last_solver_metrics(gui: Any, name: str, nodes: int, time_ms: float, path_len: int) -> None:
    """Store last successful solver metrics for quick HUD/tooltip use."""
    gui.last_solver_metrics = {
        "name": name,
        "nodes": nodes,
        "time_ms": time_ms,
        "path_len": path_len,
    }


def run_solver_comparison(
    gui: Any,
    logger: Any,
    time_module: Any,
    game_state_cls: Any,
    solve_in_subprocess: Any,
    threading_module: Any,
) -> None:
    """Run asynchronous comparison across available solver families."""
    if getattr(gui, "solver_comparison_thread", None) and gui.solver_comparison_thread.is_alive():
        gui._show_toast("Solver comparison already running", 2.5, "warning")
        return
    gui._sync_solver_dropdown_settings()

    def _worker():
        results = []
        alg_names = [
            "A*",
            "BFS",
            "Dijkstra",
            "Greedy",
            "D* Lite",
            "StateSpace",
            "CBS (Balanced)",
            "CBS (Explorer)",
            "CBS (Cautious)",
        ]
        for _idx, name in enumerate(alg_names):
            start_t = time_module.time()

            if name == "D* Lite":
                try:
                    from src.simulation.dstar_lite import DStarLiteSolver

                    start_state = game_state_cls(
                        position=gui.env.start_pos,
                        opened_doors=gui.env.state.opened_doors.copy() if hasattr(gui.env, "state") else set(),
                    )
                    ds = DStarLiteSolver(gui.env)
                    success, path, nodes = ds.solve(start_state)
                    elapsed = (time_module.time() - start_t) * 1000
                    results.append(
                        {
                            "name": name,
                            "success": success,
                            "path_len": len(path),
                            "nodes": nodes,
                            "time_ms": elapsed,
                        }
                    )
                    if success:
                        gui._set_last_solver_metrics(name, nodes, elapsed, len(path))
                    continue
                except Exception as e:
                    results.append(
                        {
                            "name": name,
                            "success": False,
                            "path_len": 0,
                            "nodes": 0,
                            "time_ms": 0,
                            "error": str(e),
                        }
                    )
                    continue

            if name == "StateSpace":
                try:
                    _start_state = (
                        gui.env.get_state()
                        if hasattr(gui.env, "get_state")
                        else game_state_cls(position=gui.env.start_pos)
                    )
                    try:
                        from src.simulation.validator import StateSpaceAStar

                        temp_solver = gui.solver if gui.solver is not None else StateSpaceAStar(gui.env)
                    except Exception:
                        temp_solver = gui.solver
                    if temp_solver is None:
                        raise RuntimeError("State-space solver unavailable")
                    success, path, _states = temp_solver.solve()
                    elapsed = (time_module.time() - start_t) * 1000
                    nodes = getattr(temp_solver, "last_states_explored", getattr(temp_solver, "last_states", 0))
                    results.append(
                        {
                            "name": name,
                            "success": success,
                            "path_len": len(path),
                            "nodes": nodes,
                            "time_ms": elapsed,
                        }
                    )
                    if success and not gui.last_solver_metrics:
                        gui._set_last_solver_metrics(name, nodes, elapsed, len(path))
                    continue
                except Exception as e:
                    results.append(
                        {
                            "name": name,
                            "success": False,
                            "path_len": 0,
                            "nodes": 0,
                            "time_ms": 0,
                            "error": str(e),
                        }
                    )
                    continue

            if "CBS" in name:
                try:
                    from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch

                    persona_map = {
                        "CBS (Balanced)": "balanced",
                        "CBS (Explorer)": "explorer",
                        "CBS (Cautious)": "cautious",
                        "CBS (Forgetful)": "forgetful",
                        "CBS (Speedrunner)": "speedrunner",
                        "CBS (Greedy)": "greedy",
                    }
                    persona = persona_map.get(name, "balanced")
                    cbs = CognitiveBoundedSearch(gui.env, persona=persona, timeout=100000)
                    ok, path, states, metrics = cbs.solve()
                    elapsed = (time_module.time() - start_t) * 1000
                    if ok:
                        results.append(
                            {
                                "name": name,
                                "success": True,
                                "path_len": len(path),
                                "nodes": states,
                                "time_ms": elapsed,
                                "confusion": round(metrics.confusion_index, 3),
                                "cog_load": round(metrics.cognitive_load, 3),
                            }
                        )
                        if not gui.last_solver_metrics:
                            gui._set_last_solver_metrics(name, states, elapsed, len(path))
                    else:
                        results.append(
                            {
                                "name": name,
                                "success": False,
                                "path_len": 0,
                                "nodes": states,
                                "time_ms": elapsed,
                                "confusion": 0,
                                "cog_load": 0,
                            }
                        )
                    continue
                except Exception as e:
                    results.append(
                        {
                            "name": name,
                            "success": False,
                            "path_len": 0,
                            "nodes": 0,
                            "time_ms": 0,
                            "error": str(e),
                            "confusion": 0,
                            "cog_load": 0,
                        }
                    )
                    continue

            if name in {"A*", "BFS", "Dijkstra", "Greedy"}:
                try:
                    cur = gui.maps[gui.current_map_idx]
                    if hasattr(cur, "global_grid"):
                        grid_arr = cur.global_grid
                        graph = getattr(cur, "graph", None)
                        room_to_node = getattr(cur, "room_to_node", None)
                        room_positions = getattr(cur, "room_positions", None)
                        node_to_room = getattr(cur, "node_to_room", None)
                    else:
                        grid_arr = cur
                        graph = None
                        room_to_node = None
                        room_positions = None
                        node_to_room = None

                    if not gui.env or not getattr(gui.env, "start_pos", None) or not getattr(gui.env, "goal_pos", None):
                        raise RuntimeError("Start/goal not defined")

                    start = tuple(gui.env.start_pos)
                    goal = tuple(gui.env.goal_pos)
                    alg_dispatch = {"A*": 0, "BFS": 1, "Dijkstra": 2, "Greedy": 3}
                    alg_idx = alg_dispatch[name]
                    flags = dict(gui.feature_flags)
                    priority_options = {
                        "tie_break": gui.feature_flags.get("priority_tie_break", False),
                        "key_boost": gui.feature_flags.get("priority_key_boost", False),
                        "enable_ara": gui.feature_flags.get("enable_ara", False),
                        "ara_weight": float(getattr(gui, "ara_weight", 1.0)),
                        "representation": str(getattr(gui, "search_representation", "hybrid")),
                        "allow_diagonals": True,
                    }

                    res = solve_in_subprocess(
                        grid_arr,
                        start,
                        goal,
                        alg_idx,
                        flags,
                        priority_options,
                        graph=graph,
                        room_to_node=room_to_node,
                        room_positions=room_positions,
                        node_to_room=node_to_room,
                    )
                    elapsed = (time_module.time() - start_t) * 1000
                    success = bool(res and res.get("success"))
                    path = (res.get("path") or []) if res else []
                    solver_meta = (res.get("solver_result") or {}) if res else {}
                    nodes = int(solver_meta.get("nodes", 0) or solver_meta.get("states_explored", 0) or 0)

                    results.append(
                        {
                            "name": name,
                            "success": success,
                            "path_len": len(path),
                            "nodes": nodes,
                            "time_ms": elapsed,
                        }
                    )
                    if success and not gui.last_solver_metrics:
                        gui._set_last_solver_metrics(name, nodes, elapsed, len(path))
                except Exception as e:
                    results.append(
                        {
                            "name": name,
                            "success": False,
                            "path_len": 0,
                            "nodes": 0,
                            "time_ms": 0,
                            "error": str(e),
                        }
                    )
                continue

        gui.solver_comparison_results = results
        gui.show_solver_comparison_overlay = True
        gui._set_message("Solver comparison complete", 3.0)
        gui._show_toast("Solver comparison finished", 3.0, "success")

    gui.solver_comparison_thread = threading_module.Thread(target=_worker, daemon=True)
    gui.solver_comparison_thread.start()
    gui._show_toast("Solver comparison started (background)", 2.0, "info")