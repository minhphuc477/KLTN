from src.gui.services.auto_solve_execution import execute_auto_solve, execute_auto_solve_from_preview


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class DummyEnv:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class DummyPreviewDialog:
    def __init__(self, solver_result=None):
        self.solver_result = solver_result or {"ok": True}


class DummyGui:
    def __init__(self):
        self.auto_path = []
        self.auto_mode = False
        self.auto_step_idx = 99
        self.auto_step_timer = 5.0
        self._auto_stuck_retries = 7
        self.preview_on_next_solver_result = True

        self.keys_used = 9
        self.bombs_used = 9
        self.boss_keys_used = 9
        self.used_items = [1]
        self.collected_items = [1]
        self.collected_positions = {(1, 1)}
        self.keys_collected = 9
        self.bombs_collected = 9
        self.boss_keys_collected = 9
        self.item_type_map = {1: "a"}
        self.item_pickup_times = {1: 1.0}

        self.env = DummyEnv()
        self.message = ""
        self.error_messages = []
        self.toasts = []
        self.solver_result = None
        self.last_solver_metrics = None

        self.path_preview_mode = True
        self.path_preview_dialog = DummyPreviewDialog(solver_result={"meta": 1})
        self.preview_overlay_visible = True

    def _show_error(self, msg):
        self.error_messages.append(msg)

    def _show_toast(self, msg, duration=0.0, toast_type="info"):
        self.toasts.append((msg, duration, toast_type))

    def _scan_items_along_path(self, path):
        return None

    def _get_path_items_display_text(self):
        return "Key,Bomb"


def test_execute_auto_solve_rejects_empty_path():
    gui = DummyGui()
    execute_auto_solve(gui=gui, path=[], solver_result={}, teleports=0, logger=DummyLogger())
    assert gui.error_messages[-1] == "No valid path to animate"


def test_execute_auto_solve_sets_state_and_message():
    gui = DummyGui()
    execute_auto_solve(
        gui=gui,
        path=[(0, 0), (0, 1)],
        solver_result={"keys_used": 0, "keys_available": 0},
        teleports=0,
        logger=DummyLogger(),
    )

    assert gui.auto_mode is True
    assert gui.auto_step_idx == 0
    assert gui.auto_path == [(0, 0), (0, 1)]
    assert gui.env.reset_calls == 1
    assert gui.message.startswith("Path: 2 steps")


def test_execute_auto_solve_from_preview_updates_dialog_and_message():
    gui = DummyGui()
    gui.auto_path = [(2, 2), (2, 3)]

    execute_auto_solve_from_preview(gui=gui, logger=DummyLogger())

    assert gui.auto_mode is True
    assert gui.env.reset_calls == 1
    assert gui.path_preview_mode is False
    assert gui.path_preview_dialog is None
    assert gui.preview_overlay_visible is False
    assert gui.solver_result == {"meta": 1}
    assert gui.message.startswith("Auto-solve started! Path: 2 steps")

