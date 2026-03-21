from src.gui.services.solver_core_state import clear_solver_state, sync_solver_dropdown_settings


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


class DummyGui:
    def __init__(self):
        self.solver_running = True
        self.solver_done = False
        self.solver_proc = object()
        self.solver_thread = object()
        self.solver_outfile = "a"
        self.solver_gridfile = "b"
        self.solver_start_time = 123.0
        self.solver_starting = True
        self.solver_algorithm_idx = 4

        self.algorithm_idx = 1
        self.search_representation = "tile"
        self.ara_weight = 2.0
        self.widget_manager = None


def test_clear_solver_state_resets_all_fields():
    gui = DummyGui()

    clear_solver_state(gui=gui, reason="test", logger=DummyLogger())

    assert gui.solver_running is False
    assert gui.solver_done is True
    assert gui.solver_proc is None
    assert gui.solver_thread is None
    assert gui.solver_outfile is None
    assert gui.solver_gridfile is None
    assert gui.solver_start_time is None
    assert gui.solver_starting is False
    assert not hasattr(gui, "solver_algorithm_idx")


def test_sync_solver_dropdown_settings_updates_gui_values():
    gui = DummyGui()

    def fake_sync(alg, rep, weight, widgets):
        assert alg == 1
        assert rep == "tile"
        assert weight == 2.0
        assert widgets == []
        return 3, "hybrid", 1.25

    out = sync_solver_dropdown_settings(gui=gui, sync_fn=fake_sync)

    assert out == (3, "hybrid", 1.25)
    assert gui.algorithm_idx == 3
    assert gui.search_representation == "hybrid"
    assert gui.ara_weight == 1.25

