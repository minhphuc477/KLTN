import json
from pathlib import Path
from types import SimpleNamespace

from src.gui.services.route_io import export_route, load_route, load_route_file
from src.gui.services.route_payload import apply_loaded_route_data, build_route_export_payload
from src.gui.solver.start_flow import start_auto_solve


def _make_gui(tmp_path: Path):
    return SimpleNamespace(
        auto_path=[],
        solution_path=[],
        start_pos=(0, 0),
        goal_pos=(1, 1),
        last_algorithm="A*",
        last_solve_time=0.123,
        last_nodes_explored=7,
        route_export_dir=tmp_path / "exports" / "routes",
        repo_root=tmp_path,
        message="",
    )


def test_build_route_export_payload_shapes_metadata(tmp_path: Path):
    gui = _make_gui(tmp_path)
    payload = build_route_export_payload(gui, [(0, 0), (1, 0), (1, 1)])

    assert payload["start"] == (0, 0)
    assert payload["goal"] == (1, 1)
    assert payload["path_length"] == 3
    assert payload["algorithm"] == "A*"
    assert payload["solve_time_ms"] == 123.0
    assert payload["nodes_explored"] == 7


def test_apply_loaded_route_data_restores_state(tmp_path: Path):
    gui = _make_gui(tmp_path)
    data = {
        "start": [2, 3],
        "goal": [4, 5],
        "path": [[2, 3], [2, 4], [4, 5]],
        "algorithm": "Dijkstra",
        "solve_time_ms": 10.0,
        "nodes_explored": 99,
    }

    count = apply_loaded_route_data(gui, data)

    assert count == 3
    assert gui.start_pos == (2, 3)
    assert gui.goal_pos == (4, 5)
    assert gui.auto_path == [(2, 3), (2, 4), (4, 5)]
    assert gui.solution_path == [(2, 3), (2, 4), (4, 5)]
    assert gui.last_algorithm == "Dijkstra"
    assert gui.last_solve_time == 0.01
    assert gui.last_nodes_explored == 99


def test_export_route_writes_json(tmp_path: Path):
    gui = _make_gui(tmp_path)
    gui.auto_path = [(0, 0), (0, 1)]

    export_route(gui)

    files = sorted(gui.route_export_dir.glob("route_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["path"] == [[0, 0], [0, 1]]
    assert payload["path_length"] == 2
    assert gui.message.startswith("Route exported to")


def test_load_route_reads_latest_file(tmp_path: Path):
    gui = _make_gui(tmp_path)
    gui.route_export_dir.mkdir(parents=True, exist_ok=True)

    older = gui.route_export_dir / "route_20200101_000000.json"
    latest = gui.route_export_dir / "route_20990101_000000.json"
    older.write_text(json.dumps({"path": [[0, 0]], "start": [0, 0], "goal": [0, 0]}), encoding="utf-8")
    latest.write_text(
        json.dumps(
            {
                "path": [[1, 1], [1, 2]],
                "start": [1, 1],
                "goal": [1, 2],
                "algorithm": "A*",
                "solve_time_ms": 2.0,
                "nodes_explored": 3,
            }
        ),
        encoding="utf-8",
    )

    load_route(gui)

    assert gui.start_pos == (1, 1)
    assert gui.goal_pos == (1, 2)
    assert gui.auto_path == [(1, 1), (1, 2)]
    assert gui.solution_path == [(1, 1), (1, 2)]
    assert "2 steps" in gui.message


def test_load_route_file_reads_explicit_path_and_enables_preview(tmp_path: Path):
    gui = _make_gui(tmp_path)
    route_file = tmp_path / "demo_route.json"
    route_file.write_text(
        json.dumps(
            {
                "path": [[2, 2], [2, 3], [2, 4]],
                "start": [2, 2],
                "goal": [2, 4],
                "algorithm": "P-CBS balanced",
                "solver_result": {"persona": "balanced", "cbs_metrics": {"confusion_index": 0.1}},
            }
        ),
        encoding="utf-8",
    )

    assert load_route_file(gui, route_file) is True

    assert gui.start_pos == (2, 2)
    assert gui.goal_pos == (2, 4)
    assert gui.auto_path == [(2, 2), (2, 3), (2, 4)]
    assert gui.preview_overlay_visible is True
    assert gui.solver_result["persona"] == "balanced"
    assert gui.loaded_route_source == "route_json"
    assert "Enter to start" in gui.message


def test_start_auto_solve_uses_preloaded_route_instead_of_recomputing():
    calls = {"execute": 0, "prepare": 0}
    gui = SimpleNamespace(
        auto_path=[(0, 0), (0, 1)],
        auto_mode=False,
        loaded_route_source="demo_route.json",
        use_preloaded_route_on_solve=True,
        search_representation="tile",
        ara_weight=1.0,
        auto_start_solver=False,
    )
    gui._sync_solver_dropdown_settings = lambda: (0, "tile", 1.0)
    gui._algorithm_name = lambda _idx: "A*"

    def execute():
        calls["execute"] += 1

    def prepare():
        calls["prepare"] += 1
        return True

    gui._execute_auto_solve_from_preview = execute
    gui._prepare_active_solver_for_new_start = prepare

    start_auto_solve(gui, logger=SimpleNamespace(info=lambda *a, **k: None), debug_sync_solver=False)

    assert calls == {"execute": 1, "prepare": 0}


def test_start_auto_solve_recomputes_when_preloaded_route_flag_is_disabled():
    calls = {"execute": 0, "prepare": 0, "precheck": 0, "schedule": 0}
    gui = SimpleNamespace(
        auto_path=[(0, 0), (0, 1)],
        auto_mode=False,
        loaded_route_source="demo_route.json",
        use_preloaded_route_on_solve=False,
        search_representation="tile",
        ara_weight=1.0,
        auto_start_solver=False,
        preview_on_next_solver_result=False,
    )
    gui._sync_solver_dropdown_settings = lambda: (7, "tile", 1.0)
    gui._algorithm_name = lambda _idx: "P-CBS (Balanced)"
    gui._execute_auto_solve_from_preview = lambda: calls.__setitem__("execute", calls["execute"] + 1)
    gui._prepare_active_solver_for_new_start = lambda: calls.__setitem__("prepare", calls["prepare"] + 1) or True
    gui._run_prechecks_and_optional_prune = lambda: calls.__setitem__("precheck", calls["precheck"] + 1) or (True, "")
    gui._cleanup_preview_before_solver_start = lambda: None
    gui._reset_solver_visual_state_before_start = lambda: None
    gui._set_message = lambda *_args: None
    gui._schedule_solver = lambda algorithm_idx: calls.__setitem__("schedule", calls["schedule"] + 1)

    start_auto_solve(gui, logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None), debug_sync_solver=False)

    assert calls == {"execute": 0, "prepare": 1, "precheck": 1, "schedule": 1}

