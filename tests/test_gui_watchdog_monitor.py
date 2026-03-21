from src.gui.services.watchdog_monitor import watchdog_loop


class DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class DummyGui:
    def __init__(self):
        self._watchdog_enabled = False
        self._watchdog_dumps = 0
        self._watchdog_dump_limit = 1
        self._watchdog_threshold = 1.25
        self.last_frame_time = 0.0

    def _set_message(self, msg, duration=0.0):
        self.last_message = (msg, duration)


class DummyTime:
    @staticmethod
    def sleep(_):
        return None

    @staticmethod
    def time():
        return 0.0


class DummyTempfile:
    @staticmethod
    def gettempdir():
        return "."


def test_watchdog_loop_exits_when_disabled():
    gui = DummyGui()

    watchdog_loop(
        gui=gui,
        logger=DummyLogger(),
        os_module=__import__("os"),
        time_module=DummyTime,
        tempfile_module=DummyTempfile,
    )

    assert gui._watchdog_enabled is False

