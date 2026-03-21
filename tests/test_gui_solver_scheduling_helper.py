from src.gui.services.solver_scheduling import schedule_solver


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyThread:
    def __init__(self, target, kwargs, daemon):
        self.target = target
        self.kwargs = kwargs
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True


class DummyThreading:
    Thread = DummyThread


class DummyTime:
    @staticmethod
    def time():
        return 123.456


class DummyGui:
    def __init__(self):
        self._solver_lock = DummyLock()
        self.solver_running = False
        self.solver_done = True
        self.solver_start_time = None
        self.solver_starting = False
        self.algorithm_idx = 1
        self.solver_algorithm_idx = None
        self._auto_recenter_done = True
        self.solver_outfile = None
        self.solver_gridfile = None
        self.messages = []
        self.sync_called = False
        self.launch_calls = []
        self.clear_reason = None

    def _set_message(self, msg):
        self.messages.append(msg)

    def _sync_solver_dropdown_settings(self):
        self.sync_called = True

    def _build_solver_request(self, algorithm_idx=None):
        if algorithm_idx == -1:
            return None
        return {
            "grid_arr": [[0]],
            "graph": None,
            "room_to_node": None,
            "room_positions": None,
            "node_to_room": None,
            "start": (0, 0),
            "goal": (0, 1),
            "alg_idx": algorithm_idx if algorithm_idx is not None else self.algorithm_idx,
            "flags": {"a": 1},
            "priority_options": {"b": 2},
        }

    def _create_solver_temp_files(self, grid_arr):
        assert grid_arr == [[0]]
        return "out.pkl", "grid.npy"

    def _launch_solver_worker(self, **kwargs):
        self.launch_calls.append(kwargs)

    def _clear_solver_state(self, reason=""):
        self.clear_reason = reason
        self.solver_running = False
        self.solver_done = True


def test_schedule_solver_respects_running_guard():
    gui = DummyGui()
    gui.solver_running = True

    ok = schedule_solver(gui, algorithm_idx=0, logger=DummyLogger(), time_module=DummyTime(), threading_module=DummyThreading())

    assert ok is False
    assert gui.messages[-1] == "Solver already running..."


def test_schedule_solver_success_sets_state_and_starts_thread():
    gui = DummyGui()

    ok = schedule_solver(gui, algorithm_idx=4, logger=DummyLogger(), time_module=DummyTime(), threading_module=DummyThreading())

    assert ok is True
    assert gui.sync_called is True
    assert gui.solver_running is True
    assert gui.solver_done is False
    assert gui.solver_start_time == 123.456
    assert gui.solver_starting is True
    assert gui.algorithm_idx == 4
    assert gui.solver_algorithm_idx == 4
    assert gui._auto_recenter_done is False
    assert gui.solver_outfile == "out.pkl"
    assert gui.solver_gridfile == "grid.npy"
    assert len(gui.launch_calls) == 0


def test_schedule_solver_missing_request_clears_state():
    gui = DummyGui()

    ok = schedule_solver(gui, algorithm_idx=-1, logger=DummyLogger(), time_module=DummyTime(), threading_module=DummyThreading())

    assert ok is False
    assert gui.clear_reason == "missing start/goal"
    assert gui.solver_running is False

