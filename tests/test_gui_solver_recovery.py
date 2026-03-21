from src.gui.services.solver_recovery import (
    compute_solver_timeout_seconds,
    force_solver_recovery_state,
    prepare_active_solver_for_new_start,
)


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class DummyTime:
    @staticmethod
    def time():
        return 200.0


class DummyNp:
    class Arr:
        size = 100

    @staticmethod
    def asarray(_):
        return DummyNp.Arr()


class DummyOs:
    environ = {}


class DummyGui:
    def __init__(self):
        self.maps = [object()]
        self.current_map_idx = 0
        self.messages = []
        self.solver_running = True
        self.solver_proc = None
        self.solver_thread = object()
        self.solver_outfile = "x"
        self.solver_gridfile = "y"
        self.solver_start_time = 150.0
        self.solver_starting = True
        self.solver_done = False
        self.algorithm_idx = 0

    def _set_message(self, msg, duration=0):
        self.messages.append((msg, duration))


def test_compute_solver_timeout_seconds_uses_env_override(monkeypatch):
    gui = DummyGui()
    os_mod = type("_O", (), {"environ": {"KLTN_SOLVER_TIMEOUT": "77"}})

    timeout = compute_solver_timeout_seconds(
        gui=gui,
        active_alg=0,
        default_solver_timeout_for_algorithm=lambda _: 60.0,
        scale_timeout_by_grid_size=lambda base, cells: base + cells,
        np_module=DummyNp,
        os_module=os_mod,
    )

    assert timeout == 77.0


def test_force_solver_recovery_state_resets_fields():
    gui = DummyGui()
    gui.solver_algorithm_idx = 2

    force_solver_recovery_state(gui=gui, recovery_reason="hung", logger=DummyLogger())

    assert gui.solver_running is False
    assert gui.solver_proc is None
    assert gui.solver_thread is None
    assert gui.solver_outfile is None
    assert gui.solver_gridfile is None
    assert gui.solver_start_time is None
    assert gui.solver_starting is False
    assert not hasattr(gui, "solver_algorithm_idx")
    assert gui.messages[-1][0].startswith("Recovered:")


def test_prepare_active_solver_for_new_start_allows_when_not_running():
    gui = DummyGui()
    gui.solver_running = False

    ok = prepare_active_solver_for_new_start(
        gui=gui,
        logger=DummyLogger(),
        time_module=DummyTime,
        evaluate_solver_recovery_state=lambda **_: (False, ""),
        compute_timeout_seconds=lambda *_: 60.0,
        terminate_hung_process=lambda *_: None,
        force_recovery_state=lambda *_: None,
        log_active_state=lambda: None,
    )

    assert ok is True

