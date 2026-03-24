import threading
from pathlib import Path

from src.gui.ai.ai_generation_controls import start_ai_dungeon_generation
from src.gui import ai_generation_worker
from src.gui.ai import generation_pipeline


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


def test_resolve_checkpoint_path_honors_env_override(monkeypatch):
    monkeypatch.setenv("KLTN_CHECKPOINT_PATH", "checkpoints/custom_model.pth")
    resolved = generation_pipeline.resolve_checkpoint_path()

    assert resolved.name == "custom_model.pth"


def test_generate_mission_graph_is_deterministic_with_seed():
    import random

    fixed_seed = 314159
    data_a = generation_pipeline.generate_mission_graph(random, seed=fixed_seed)
    data_b = generation_pipeline.generate_mission_graph(random, seed=fixed_seed)

    assert data_a["seed"] == fixed_seed
    assert data_b["seed"] == fixed_seed
    assert data_a["num_nodes"] == data_b["num_nodes"]
    assert data_a["num_edges"] == data_b["num_edges"]

