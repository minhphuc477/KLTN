import numpy as np
import gui_runner
import json
from pathlib import Path


def test_start_auto_solve_calls_schedule_once(monkeypatch):
    """_start_auto_solve should call _schedule_solver exactly once and pass algorithm index.

    Note: _start_auto_solve prefers the dropdown widget value when a widget manager
    is present; set widget_manager=None to force fallback to self.algorithm_idx.
    """
    gui = gui_runner.ZeldaGUI(maps=[gui_runner.create_test_map()])
    calls = []

    def spy(self, algorithm_idx=None):
        calls.append(algorithm_idx)

    monkeypatch.setattr(gui_runner.ZeldaGUI, '_schedule_solver', spy)
    # Force fallback to self.algorithm_idx by removing widget_manager
    gui.widget_manager = None
    gui.algorithm_idx = 2
    gui._start_auto_solve()

    assert len(calls) == 1
    assert calls[0] == 2


def test_start_auto_solve_does_not_schedule_when_alive_process(monkeypatch):
    """If solver_running is True but the process is alive, _start_auto_solve should return early and not reschedule."""
    gui = gui_runner.ZeldaGUI(maps=[gui_runner.create_test_map()])
    calls = []

    def spy(self, algorithm_idx=None):
        calls.append(algorithm_idx)

    monkeypatch.setattr(gui_runner.ZeldaGUI, '_schedule_solver', spy)

    # Simulate an alive solver process so recovery is NOT triggered
    class DummyProc:
        def is_alive(self):
            return True

    gui.solver_running = True
    gui.solver_proc = DummyProc()
    gui.solver_done = False

    gui._start_auto_solve()

    assert calls == []


def test_schedule_solver_is_thread_safe(monkeypatch):
    """Concurrent calls to _schedule_solver should result in exactly one actual schedule (thread-safe)."""
    gui = gui_runner.ZeldaGUI(maps=[gui_runner.create_test_map()])

    # Replace multiprocessing.Process with a dummy to avoid OS process spawn in test
    starts = []
    class DummyProcess:
        def __init__(self, *a, **k):
            starts.append(self)
            self._alive = False
        def start(self):
            self._alive = True
        def is_alive(self):
            return self._alive
        def join(self, timeout=None):
            self._alive = False
    monkeypatch.setattr(gui_runner.multiprocessing, 'Process', DummyProcess)

    results = []
    def caller():
        results.append(gui._schedule_solver())

    import threading, time
    t1 = threading.Thread(target=caller)
    t2 = threading.Thread(target=caller)
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Exactly one caller should have succeeded
    assert results.count(True) == 1
    assert results.count(False) == 1 or results.count(None) == 0

    # Wait briefly for the worker thread to run and create the DummyProcess
    deadline = time.time() + 1.0
    while time.time() < deadline and len(starts) < 1:
        time.sleep(0.01)
    assert len(starts) == 1


def test_export_route_writes_to_repo_local_route_dir(tmp_path: Path):
    """Route export should write under route_export_dir (repo-local by default)."""
    gui = gui_runner.ZeldaGUI(maps=[gui_runner.create_test_map()])
    gui.repo_root = tmp_path
    gui.route_export_dir = tmp_path / "exports" / "routes"

    gui.start_pos = (0, 0)
    gui.goal_pos = (0, 1)
    gui.auto_path = [(0, 0), (0, 1), (1, 1)]

    gui._export_route()

    route_files = sorted(gui.route_export_dir.glob("route_*.json"))
    assert len(route_files) == 1

    payload = json.loads(route_files[0].read_text(encoding="utf-8"))
    assert payload["path"] == [[0, 0], [0, 1], [1, 1]]
    assert payload["path_length"] == 3


def test_load_route_reads_from_repo_local_route_dir(tmp_path: Path):
    """Route load should read the latest route from route_export_dir."""
    gui = gui_runner.ZeldaGUI(maps=[gui_runner.create_test_map()])
    gui.repo_root = tmp_path
    gui.route_export_dir = tmp_path / "exports" / "routes"
    gui.route_export_dir.mkdir(parents=True, exist_ok=True)

    route_file = gui.route_export_dir / "route_99999999_235959.json"
    route_file.write_text(
        json.dumps(
            {
                "version": "1.0",
                "timestamp": "2099-01-01T00:00:00",
                "start": [2, 2],
                "goal": [3, 3],
                "path": [[2, 2], [2, 3], [3, 3]],
                "path_length": 3,
                "algorithm": "A*",
                "solve_time_ms": 12.0,
                "nodes_explored": 5,
            }
        ),
        encoding="utf-8",
    )

    gui._load_route()

    assert gui.start_pos == (2, 2)
    assert gui.goal_pos == (3, 3)
    assert gui.auto_path == [(2, 2), (2, 3), (3, 3)]
    assert gui.solution_path == [(2, 2), (2, 3), (3, 3)]

