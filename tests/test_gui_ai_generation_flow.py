import threading
from pathlib import Path

from src.gui.ai.ai_generation_controls import start_ai_dungeon_generation
from src.gui import ai_generation_worker


class _DummyGUI:
    def __init__(self):
        self.messages = []
        self.ai_gen_thread = None
        self.ai_gen_result = object()
        self.ai_gen_done = True

    def _set_message(self, message, duration=3.0):
        self.messages.append((message, duration))

    def _generate_ai_dungeon_worker(self):
        return None


def test_start_ai_generation_sets_thread_and_message():
    gui = _DummyGUI()
    start_ai_dungeon_generation(gui, threading)

    assert gui.ai_gen_thread is not None
    gui.ai_gen_thread.join(timeout=1.0)
    assert gui.ai_gen_result is None
    assert gui.ai_gen_done is False
    assert gui.messages[-1][0] == "AI generation started (background)"


def test_worker_reports_missing_checkpoint(monkeypatch):
    gui = _DummyGUI()

    def _missing_checkpoint():
        return Path("__definitely_missing_checkpoint__.pth")

    monkeypatch.setattr(ai_generation_worker, "resolve_checkpoint_path", _missing_checkpoint)

    class _Logger:
        def warning(self, *_args, **_kwargs):
            return None

        def exception(self, *_args, **_kwargs):
            return None

    ai_generation_worker.run_ai_generation_worker(gui, _Logger())

    assert gui.messages
    assert gui.messages[-1][0] == "No AI checkpoint found - train first!"

