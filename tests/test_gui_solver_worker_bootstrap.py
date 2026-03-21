from src.gui.services.solver_worker_bootstrap import launch_solver_worker


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class DummyProc:
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True


class DummyMultiprocessing:
    def __init__(self):
        self.processes = []

    def Process(self, *args, **kwargs):
        p = DummyProc()
        self.processes.append((p, args, kwargs))
        return p


class DummyGui:
    def __init__(self):
        self.solver_proc = None
        self.solver_thread = object()
        self.solver_starting = True
        self.messages = []

    def _set_message(self, msg):
        self.messages.append(msg)


def test_launch_solver_worker_uses_test_mode(monkeypatch):
    monkeypatch.setenv("KLTN_SOLVER_TEST", "1")

    gui = DummyGui()
    mp = DummyMultiprocessing()
    calls = {"launch": 0, "fallback": 0}

    launch_solver_worker(
        gui=gui,
        kwargs={"a": 1},
        logger=DummyLogger(),
        launch_solver_process=lambda **kwargs: calls.__setitem__("launch", calls["launch"] + 1),
        start_solver_thread_fallback=lambda **kwargs: calls.__setitem__("fallback", calls["fallback"] + 1),
        multiprocessing_module=mp,
    )

    assert len(mp.processes) == 1
    assert gui.solver_proc is not None
    assert gui.solver_thread is None
    assert gui.solver_starting is False
    assert gui.messages[-1] == "Test solver process started (sleep)"
    assert calls["launch"] == 0
    assert calls["fallback"] == 0


def test_launch_solver_worker_falls_back_on_exception(monkeypatch):
    monkeypatch.delenv("KLTN_SOLVER_TEST", raising=False)

    gui = DummyGui()
    mp = DummyMultiprocessing()
    calls = {"fallback": 0}

    def raising_launch(**kwargs):
        raise RuntimeError("boom")

    launch_solver_worker(
        gui=gui,
        kwargs={"x": 1},
        logger=DummyLogger(),
        launch_solver_process=raising_launch,
        start_solver_thread_fallback=lambda **kwargs: calls.__setitem__("fallback", calls["fallback"] + 1),
        multiprocessing_module=mp,
    )

    assert gui.solver_starting is False
    assert calls["fallback"] == 1

