import numpy as np
import gui_runner


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

