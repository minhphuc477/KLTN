from src.gui.services.solver_prestart_cleanup import (
    cleanup_preview_before_solver_start,
    reset_solver_visual_state_before_start,
)


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class DummyProc:
    def __init__(self, alive=True):
        self._alive = alive
        self.terminated = False
        self.joined = False
        self.pid = 999

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False

    def join(self, timeout=None):
        self.joined = True


class DummyOs:
    class path:
        existing = set()

        @staticmethod
        def exists(path):
            return path in DummyOs.path.existing

    removed = []

    @staticmethod
    def remove(path):
        DummyOs.removed.append(path)
        DummyOs.path.existing.discard(path)


class DummyGui:
    def __init__(self):
        self.preview_proc = DummyProc(alive=True)
        self.preview_outfile = "out.pkl"
        self.preview_gridfile = "grid.npy"
        self.preview_done = True

        self.solver_done = True
        self.solver_thread = object()
        self.auto_path = [(1, 1)]
        self.auto_mode = True
        self.auto_step_idx = 9
        self.block_push_animations = [1]
        self.door_unlock_times = {1: 2}
        self.collected_items = [1]
        self.collected_positions = {(1, 1)}


def test_cleanup_preview_before_solver_start_terminates_and_clears():
    gui = DummyGui()
    DummyOs.path.existing = {"out.pkl", "grid.npy"}
    DummyOs.removed = []

    cleanup_preview_before_solver_start(gui=gui, logger=DummyLogger(), os_module=DummyOs)

    assert gui.preview_proc is None
    assert gui.preview_outfile is None
    assert gui.preview_gridfile is None
    assert gui.preview_done is False
    assert "out.pkl" in DummyOs.removed
    assert "grid.npy" in DummyOs.removed


def test_reset_solver_visual_state_before_start_clears_visual_fields():
    gui = DummyGui()

    reset_solver_visual_state_before_start(gui)

    assert gui.solver_done is False
    assert gui.solver_thread is None
    assert gui.auto_path == []
    assert gui.auto_mode is False
    assert gui.auto_step_idx == 0
    assert gui.block_push_animations == []
    assert gui.door_unlock_times == {}
    assert gui.collected_items == []
    assert gui.collected_positions == set()

