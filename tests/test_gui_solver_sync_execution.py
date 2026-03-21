from src.gui.services.solver_sync_execution import run_solver_sync


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class DummyGui:
    def __init__(self):
        self.auto_path = []
        self.auto_mode = False
        self.auto_step_idx = 0
        self.messages = []

    def _set_message(self, msg, duration=0.0):
        self.messages.append((msg, duration))

    def _sync_solver_dropdown_settings(self):
        return None

    def _build_solver_request(self, algorithm_idx=None):
        return {
            "grid_arr": [[0]],
            "graph": None,
            "room_to_node": None,
            "room_positions": None,
            "node_to_room": None,
            "start": (0, 0),
            "goal": (0, 1),
            "alg_idx": algorithm_idx if algorithm_idx is not None else 0,
            "flags": {},
            "priority_options": {},
        }

    def _execute_auto_solve(self, path, solver_result, teleports=0):
        self.auto_mode = True
        self.auto_path = path


def test_run_solver_sync_success_executes_auto_solve():
    gui = DummyGui()

    def fake_solver(*args, **kwargs):
        return {"success": True, "path": [(0, 0), (0, 1)], "solver_result": {"ok": True}}

    run_solver_sync(gui=gui, logger=DummyLogger(), solve_in_subprocess=fake_solver, algorithm_idx=1)

    assert gui.auto_mode is True
    assert gui.auto_path == [(0, 0), (0, 1)]
    assert any("DEBUG: Solver done!" in msg for msg, _ in gui.messages)


def test_run_solver_sync_failure_sets_no_path_message():
    gui = DummyGui()

    def fake_solver(*args, **kwargs):
        return {"success": False, "message": "blocked"}

    run_solver_sync(gui=gui, logger=DummyLogger(), solve_in_subprocess=fake_solver, algorithm_idx=0)

    assert any("DEBUG: No path - blocked" == msg for msg, _ in gui.messages)

