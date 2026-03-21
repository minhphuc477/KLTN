import time
from types import SimpleNamespace

from src.gui.services.solver_comparison_runner import run_solver_comparison, set_last_solver_metrics


class _DummyThread:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


def test_set_last_solver_metrics_updates_gui_state():
    gui = SimpleNamespace(last_solver_metrics=None)

    set_last_solver_metrics(gui=gui, name="A*", nodes=12, time_ms=8.5, path_len=6)

    assert gui.last_solver_metrics == {
        "name": "A*",
        "nodes": 12,
        "time_ms": 8.5,
        "path_len": 6,
    }


def test_run_solver_comparison_warns_when_already_running():
    toasts = []
    gui = SimpleNamespace(
        solver_comparison_thread=_DummyThread(alive=True),
        _show_toast=lambda m, d, k=None: toasts.append((m, d, k)),
        _sync_solver_dropdown_settings=lambda: None,
    )

    run_solver_comparison(
        gui=gui,
        logger=SimpleNamespace(),
        time_module=SimpleNamespace(time=lambda: 0.0),
        game_state_cls=SimpleNamespace,
        solve_in_subprocess=lambda *args, **kwargs: {},
        threading_module=__import__("threading"),
    )

    assert toasts[-1] == ("Solver comparison already running", 2.5, "warning")


def test_run_solver_comparison_starts_background_thread_and_populates_results():
    messages = []
    toasts = []

    class _Env:
        def __init__(self):
            self.start_pos = (0, 0)
            self.goal_pos = (0, 1)

    gui = SimpleNamespace(
        solver_comparison_thread=None,
        solver_comparison_results=None,
        show_solver_comparison_overlay=False,
        last_solver_metrics=None,
        env=_Env(),
        maps=[[[0, 0], [0, 0]]],
        current_map_idx=0,
        feature_flags={},
        ara_weight=1.0,
        search_representation="hybrid",
        solver=None,
        _sync_solver_dropdown_settings=lambda: None,
        _set_last_solver_metrics=lambda n, nodes, t, p: set_last_solver_metrics(
            gui=gui, name=n, nodes=nodes, time_ms=t, path_len=p
        ),
        _set_message=lambda m, d=0.0: messages.append((m, d)),
        _show_toast=lambda m, d, k=None: toasts.append((m, d, k)),
    )

    def _solve_stub(*_args, **_kwargs):
        return {
            "success": True,
            "path": [(0, 0), (0, 1)],
            "solver_result": {"nodes": 3},
        }

    run_solver_comparison(
        gui=gui,
        logger=SimpleNamespace(),
        time_module=__import__("time"),
        game_state_cls=SimpleNamespace,
        solve_in_subprocess=_solve_stub,
        threading_module=__import__("threading"),
    )

    assert gui.solver_comparison_thread is not None
    deadline = time.time() + 15.0
    while gui.solver_comparison_thread.is_alive() and time.time() < deadline:
        gui.solver_comparison_thread.join(timeout=0.25)
    assert not gui.solver_comparison_thread.is_alive()

    assert isinstance(gui.solver_comparison_results, list)
    assert len(gui.solver_comparison_results) == 9
    assert any(item["name"] == "A*" and item["success"] for item in gui.solver_comparison_results)
    assert gui.show_solver_comparison_overlay is True
    assert messages[-1] == ("Solver comparison complete", 3.0)
    assert ("Solver comparison started (background)", 2.0, "info") in toasts
    assert ("Solver comparison finished", 3.0, "success") in toasts

