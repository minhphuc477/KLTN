from src.gui.controls.path_controls import clear_path, reset_map, show_path_preview


class DummyEffects:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class DummyGui:
    def __init__(self):
        self.loaded = 0
        self.centered = 0
        self.effects = DummyEffects()
        self.step_count = 99
        self.path_preview_mode = True
        self.preview_overlay_visible = True
        self.path_preview_dialog = object()
        self.preview_on_next_solver_result = True
        self.message = ""

        self.auto_path = []
        self.solution_path = []
        self.auto_mode = True
        self.auto_step_idx = 9
        self.solver_result = {}
        self.speed_multiplier = 1.0
        self.env = object()
        self.preview_modal_enabled = False
        self.solver_running = False
        self.started = 0

    def _load_current_map(self):
        self.loaded += 1

    def _center_view(self):
        self.centered += 1

    def _start_auto_solve(self):
        self.started += 1


class DummyLogger:
    def exception(self, *args, **kwargs):
        return None


def test_reset_map_resets_preview_and_effects():
    gui = DummyGui()
    reset_map(gui)

    assert gui.loaded == 1
    assert gui.centered == 1
    assert gui.effects.cleared is True
    assert gui.step_count == 0
    assert gui.path_preview_mode is False
    assert gui.preview_overlay_visible is False
    assert gui.path_preview_dialog is None
    assert gui.preview_on_next_solver_result is False
    assert gui.message == "Map Reset"


def test_clear_path_resets_path_and_preview_state():
    gui = DummyGui()
    gui.auto_path = [(0, 0)]
    gui.solution_path = [(0, 1)]

    clear_path(gui)

    assert gui.auto_path == []
    assert gui.solution_path == []
    assert gui.auto_mode is False
    assert gui.auto_step_idx == 0
    assert gui.path_preview_dialog is None
    assert gui.message == "Path cleared"


def test_show_path_preview_uses_existing_path():
    gui = DummyGui()
    gui.solution_path = [(1, 1), (1, 2)]

    def factory(**kwargs):
        return {"preview": kwargs["path"]}

    show_path_preview(gui, factory, DummyLogger())

    assert gui.auto_path == [(1, 1), (1, 2)]
    assert gui.auto_mode is False
    assert gui.path_preview_dialog == {"preview": [(1, 1), (1, 2)]}
    assert gui.preview_overlay_visible is True


def test_show_path_preview_triggers_solver_when_no_path():
    gui = DummyGui()

    show_path_preview(gui, lambda **kwargs: None, DummyLogger())

    assert gui.preview_on_next_solver_result is True
    assert gui.started == 1
    assert gui.message == "No path yet, solving for preview..."

