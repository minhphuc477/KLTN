from src.gui.controls.control_panel_updates import (
    apply_algorithm_dropdown_update,
    apply_checkbox_widget_update,
    apply_control_panel_widget_updates,
    apply_dropdown_widget_update,
    apply_level_dropdown_update,
)


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class DummyRenderer:
    def __init__(self):
        self.show_heatmap = False


class DummyProc:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


class DummyWidgetManager:
    def __init__(self, widgets):
        self.widgets = widgets


class CheckboxWidget:
    def __init__(self, flag_name, checked):
        self.flag_name = flag_name
        self.checked = checked


class DropdownWidget:
    def __init__(self, control_name, selected=0, options=None):
        self.control_name = control_name
        self.selected = selected
        self.options = options or []


class DummyMap:
    def __init__(self, graph=None):
        self.graph = graph


class DummyGui:
    ZOOM_LEVELS = [16, 24, 32, 48, 64]

    def __init__(self):
        self.feature_flags = {}
        self.show_heatmap = False
        self.show_minimap = False
        self.show_topology = False
        self.show_topology_legend = False
        self.renderer = DummyRenderer()
        self.maps = [DummyMap(graph={"ok": True})]
        self.map_names = ["One"]
        self.current_map_idx = 0

        self.zoom_level_idx = 2
        self.zoom_idx = 1
        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
        self.assets_reloaded = False
        self.centered = False

        self.difficulty_idx = 0
        self.algorithm_idx = 0
        self.search_representation = "hybrid"
        self.ara_weight = 1.0

        self.current_preset_idx = 0
        self.presets = ["Balanced", "Debugging", "Fast Approx"]
        self.current_floor = 1
        self.match_apply_threshold = 0.85

        self.solver_running = False
        self.solver_proc = None
        self.preview_thread = None
        self.preview_proc = None
        self.auto_path = []
        self.auto_mode = True
        self._pending_solver_trigger = False

        self.clear_solver_reason = None
        self.message = ""
        self.messages = []
        self.effects = None
        self.step_count = 5
        self.loaded = 0

    def _set_message(self, message, duration=3.0):
        self.messages.append((message, duration))

    def _load_assets(self):
        self.assets_reloaded = True

    def _center_view(self):
        self.centered = True

    def _clear_solver_state(self, reason=""):
        self.clear_solver_reason = reason

    def _load_current_map(self):
        self.loaded += 1


def test_apply_checkbox_widget_update_heatmap_updates_renderer_and_message():
    gui = DummyGui()
    widget = CheckboxWidget("show_heatmap", True)

    apply_checkbox_widget_update(gui, widget, logger=DummyLogger())

    assert gui.feature_flags["show_heatmap"] is True
    assert gui.show_heatmap is True
    assert gui.renderer.show_heatmap is True
    assert gui.messages[-1][0] == "Heatmap: ON"


def test_apply_checkbox_widget_update_unchanged_minimap_does_not_emit_message():
    gui = DummyGui()
    gui.feature_flags["show_minimap"] = True
    gui.show_minimap = True
    widget = CheckboxWidget("show_minimap", True)

    apply_checkbox_widget_update(gui, widget, logger=DummyLogger())

    assert gui.messages == []


def test_apply_dropdown_widget_update_zoom_reloads_assets_and_centers():
    gui = DummyGui()
    widget = DropdownWidget("zoom", selected=5)

    apply_dropdown_widget_update(gui, widget, logger=DummyLogger())

    assert gui.zoom_level_idx == 5
    assert gui.zoom_idx == 4
    assert gui.TILE_SIZE == gui.ZOOM_LEVELS[4]
    assert gui.assets_reloaded is True
    assert gui.centered is True
    assert gui.message == "Zoom: 200%"


def test_apply_dropdown_widget_update_presets_applies_flags():
    gui = DummyGui()
    widget = DropdownWidget("presets", selected=1)

    apply_dropdown_widget_update(gui, widget, logger=DummyLogger())

    assert gui.current_preset_idx == 1
    assert gui.feature_flags["show_heatmap"] is True
    assert gui.feature_flags["solver_comparison"] is False
    assert gui.messages[-1][0] == "Preset applied: Debugging"


def test_apply_dropdown_widget_update_floor_changes_current_floor():
    gui = DummyGui()
    widget = DropdownWidget("floor", selected=2)

    apply_dropdown_widget_update(gui, widget, logger=DummyLogger())

    assert gui.current_floor == 3
    assert gui.messages[-1][0] == "Floor: 3"


def test_apply_dropdown_widget_update_match_threshold_changes_threshold():
    gui = DummyGui()
    widget = DropdownWidget("match_threshold", selected=1, options=["0.70", "0.80", "0.90"])

    apply_dropdown_widget_update(gui, widget, logger=DummyLogger())

    assert gui.match_apply_threshold == 0.80
    assert gui.messages[-1][0] == "Match threshold: 0.80"


def test_apply_level_dropdown_update_switches_loaded_level_and_stops_solver():
    gui = DummyGui()
    gui.maps = [DummyMap(), DummyMap()]
    gui.map_names = ["Dungeon 1", "Generated Level"]
    gui.solver_proc = DummyProc()
    gui.preview_proc = DummyProc()
    gui.preview_thread = object()
    gui.auto_path = [(1, 1)]
    gui.auto_mode = True
    widget = DropdownWidget("level", selected=1)

    apply_level_dropdown_update(gui, widget, logger=DummyLogger())

    assert gui.current_map_idx == 1
    assert gui.loaded == 1
    assert gui.centered is True
    assert gui.solver_proc.terminated is True
    assert gui.preview_proc.terminated is True
    assert gui.preview_thread is None
    assert gui.auto_path == []
    assert gui.auto_mode is False
    assert gui.clear_solver_reason == "level changed"
    assert "Generated Level" in gui.messages[-1][0]


def test_apply_algorithm_dropdown_update_stops_running_solver():
    gui = DummyGui()
    gui.algorithm_idx = 0
    gui.solver_running = True
    gui.solver_proc = DummyProc()
    gui.preview_thread = object()
    gui.preview_proc = DummyProc()
    widget = DropdownWidget("algorithm", selected=2)

    apply_algorithm_dropdown_update(gui, widget, logger=DummyLogger())

    assert gui.algorithm_idx == 2
    assert gui.solver_proc.terminated is True
    assert gui.preview_thread is None
    assert gui.preview_proc is None
    assert gui.clear_solver_reason == "algorithm changed to Dijkstra"
    assert "Switched to Dijkstra" in gui.messages[-1][0]


def test_apply_algorithm_dropdown_update_retriggers_when_path_exists():
    gui = DummyGui()
    gui.algorithm_idx = 0
    gui.auto_path = [(1, 1), (1, 2)]
    gui.auto_mode = True
    widget = DropdownWidget("algorithm", selected=4)

    apply_algorithm_dropdown_update(gui, widget, logger=DummyLogger())

    assert gui.algorithm_idx == 4
    assert gui.auto_path == []
    assert gui.auto_mode is False
    assert gui._pending_solver_trigger is True
    assert "Recomputing with D* Lite" in gui.messages[-1][0]


def test_apply_control_panel_widget_updates_runs_checkbox_and_dropdown():
    gui = DummyGui()
    widgets = [
        CheckboxWidget("show_minimap", True),
        DropdownWidget("difficulty", selected=2),
    ]
    wm = DummyWidgetManager(widgets)

    apply_control_panel_widget_updates(
        gui=gui,
        widget_manager=wm,
        checkbox_type=CheckboxWidget,
        logger=DummyLogger(),
    )

    assert gui.feature_flags["show_minimap"] is True
    assert gui.show_minimap is True
    assert gui.difficulty_idx == 2
    assert gui.message == "Difficulty: Hard"

