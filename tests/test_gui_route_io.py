import json
from pathlib import Path
from types import SimpleNamespace

from src.gui.services.route_io import export_route, load_route
from src.gui.services.route_payload import apply_loaded_route_data, build_route_export_payload


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

