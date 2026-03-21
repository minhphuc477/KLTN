from src.gui.services.solver_start_flow import start_auto_solve


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class DummyGui:
    def __init__(self):
        self.search_representation = "hybrid"
        self.ara_weight = 1.0
        self.preview_on_next_solver_result = True
        self.schedule_calls = []
        self.run_sync_calls = []
        self.messages = []

    def _sync_solver_dropdown_settings(self):
        return 2, "graph", 1.5

    def _algorithm_name(self, idx):
        return f"Alg{idx}"

    def _prepare_active_solver_for_new_start(self):
        return True

    def _run_prechecks_and_optional_prune(self):
        return True, None

    def _cleanup_preview_before_solver_start(self):
        self.cleaned = True

    def _reset_solver_visual_state_before_start(self):
        self.visual_reset = True

    def _run_solver_sync(self, algorithm_idx=None):
        self.run_sync_calls.append(algorithm_idx)

    def _set_message(self, msg, duration=0.0):
        self.messages.append((msg, duration))

    def _schedule_solver(self, algorithm_idx=None):
        self.schedule_calls.append(algorithm_idx)


def test_start_auto_solve_schedules_with_synced_algorithm():
    gui = DummyGui()

    start_auto_solve(gui=gui, logger=DummyLogger(), debug_sync_solver=False)

    assert gui.schedule_calls == [2]
    assert gui.run_sync_calls == []
    assert gui.messages[0][0] == "Starting solver in background..."


def test_start_auto_solve_uses_sync_path_when_debug_enabled():
    gui = DummyGui()

    start_auto_solve(gui=gui, logger=DummyLogger(), debug_sync_solver=True)

    assert gui.run_sync_calls == [2]
    assert gui.schedule_calls == []

