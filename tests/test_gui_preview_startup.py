from src.gui.common.preview_startup import start_preview_for_current_map


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class DummyFlip:
    def flip(self):
        return None


class DummyPygame:
    display = DummyFlip()


class DummyThread:
    started = 0

    def __init__(self, target=None, kwargs=None, daemon=True):
        self.target = target
        self.kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        DummyThread.started += 1


class DummyThreading:
    Thread = DummyThread


class DummyTime:
    _t = 10.0

    @classmethod
    def time(cls):
        cls._t += 0.01
        return cls._t


class AliveThread:
    def is_alive(self):
        return True


class DummyGui:
    def __init__(self):
        self.algorithm_idx = 0
        self.solver_running = False
        self.preview_thread = None
        self.preview_messages = []
        self.message = ""
        self.env = None
        self.maps = [object()]
        self.current_map_idx = 0
        self.feature_flags = {}
        self.ara_weight = 1.0
        self.search_representation = "hybrid"

    def _sync_solver_dropdown_settings(self):
        return None

    def _scan_and_mark_items(self):
        return None

    def _algorithm_name(self, idx):
        return "A*"

    def _render(self):
        return None

    def _set_message(self, msg, duration=0.0):
        self.preview_messages.append((msg, duration))

    def _schedule_solver(self):
        self.preview_messages.append(("scheduled", 0.0))

    def _smart_grid_path(self):
        return False, [], 0

    def _graph_guided_path(self):
        return False, [], 0


def test_preview_startup_respects_disable_env(monkeypatch):
    monkeypatch.setenv("KLTN_DISABLE_PREVIEW", "1")
    DummyThread.started = 0
    gui = DummyGui()

    start_preview_for_current_map(
        gui=gui,
        logger=DummyLogger(),
        pygame_module=DummyPygame(),
        multiprocessing_module=None,
        threading_module=DummyThreading,
        time_module=DummyTime,
        run_preview_and_dump=None,
    )

    assert gui.message == "Solving (A*)..."
    assert DummyThread.started == 0


def test_preview_startup_skips_when_preview_already_running(monkeypatch):
    monkeypatch.delenv("KLTN_DISABLE_PREVIEW", raising=False)
    DummyThread.started = 0
    gui = DummyGui()
    gui.preview_thread = AliveThread()

    start_preview_for_current_map(
        gui=gui,
        logger=DummyLogger(),
        pygame_module=DummyPygame(),
        multiprocessing_module=None,
        threading_module=DummyThreading,
        time_module=DummyTime,
        run_preview_and_dump=None,
    )

    assert gui.preview_messages[-1][0] == "Preview already running..."
    assert DummyThread.started == 0

