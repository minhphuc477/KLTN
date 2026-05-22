from types import SimpleNamespace

from src.gui.control_panel.view import update_control_panel_positions


class DummyRect:
    def __init__(self, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        self.top = self.y
        self.bottom = self.y + self.height

    def collidepoint(self, pos):
        px, py = pos
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height


class DummyPygame:
    Rect = DummyRect


class DummyLogger:
    def debug(self, *args, **kwargs):
        return None


class DummyFont:
    def get_height(self):
        return 14


class DummyWidgetManager:
    def __init__(self):
        self.widgets = []

    def add_widget(self, widget):
        self.widgets.append(widget)

    def snapshot_dropdown_state(self):
        return {}

    def apply_dropdown_state(self, state):
        return None


class DummyCheckbox:
    def __init__(self, pos, label, checked=False):
        self.pos = pos
        self.label = label
        self.checked = checked
        self.rect = DummyRect(pos[0], pos[1], 20, 20)
        self.full_rect = DummyRect(pos[0], pos[1], 260, 20)


class DummyDropdown:
    def __init__(self, pos, label, options, selected=0, keep_open_on_select=False):
        self.pos = pos
        self.label = label
        self.options = options
        self.selected = selected
        self.keep_open_on_select = keep_open_on_select
        self.is_open = False
        self.rect = DummyRect(pos[0], pos[1], 180, 28)
        self.full_rect = DummyRect(pos[0], pos[1] - 18, 180, 46)
        self.dropdown_rect = DummyRect(pos[0], pos[1] + 30, 180, len(options) * 24)


class DummyButton:
    def __init__(self, pos, label, callback, width=125, height=30):
        self.pos = pos
        self.label = label
        self.callback = callback
        self.rect = DummyRect(pos[0], pos[1], width, height)
        self.full_rect = self.rect


def _make_gui(*, advanced=False):
    gui = SimpleNamespace()
    gui.widget_manager = DummyWidgetManager()
    gui.control_panel_width_current = 360
    gui.max_panel_width = 500
    gui.SIDEBAR_WIDTH = 220
    gui.screen_w = 1000
    gui.screen_h = 800
    gui.control_panel_x = None
    gui.control_panel_y = None
    gui.minimap_size = 150
    gui.show_minimap = True
    gui.HUD_HEIGHT = 10
    gui.feature_flags = {"show_path": True, "show_minimap": True}
    gui.current_map_idx = 1
    gui.maps = [object(), object()]
    gui.map_names = ["Dungeon 1", "Generated Level"]
    gui.current_floor = 1
    gui.zoom_level_idx = 3
    gui.difficulty_idx = 1
    gui.algorithm_idx = 0
    gui.presets = ["Debugging", "Fast Approx"]
    gui.current_preset_idx = 0
    gui.search_representation = "hybrid"
    gui.ara_weight = 1.0
    gui.advanced_gui = advanced
    gui.font = DummyFont()
    gui.control_panel_scroll = 0
    gui._reposition_widgets = lambda x, y: None
    gui._start_auto_solve = lambda: None
    gui._stop_auto_solve = lambda: None
    gui._generate_dungeon = lambda: None
    gui._load_ai_model = lambda: None
    gui._generate_level = lambda: None
    gui._generate_ai_dungeon = lambda: None
    gui._reset_map = lambda: None
    gui._show_path_preview = lambda: None
    gui._clear_path = lambda: None
    gui._export_route = lambda: None
    gui._load_route = lambda: None
    gui._open_temp_folder = lambda: None
    gui._delete_temp_files = lambda: None
    gui._export_topology = lambda: None
    gui._run_solver_comparison = lambda: None
    gui._match_missing_nodes = lambda: None
    gui._apply_tentative_matches = lambda: None
    gui._undo_last_match = lambda: None
    gui._undo_prune = lambda: None
    gui._start_map_elites = lambda: None
    gui._import_level = lambda: None
    gui._export_map = lambda: None
    return gui


def _build(gui):
    update_control_panel_positions(
        gui,
        DummyPygame,
        DummyLogger(),
        widgets_available=True,
        checkbox_widget_cls=DummyCheckbox,
        dropdown_widget_cls=DummyDropdown,
        button_widget_cls=DummyButton,
        zoom_labels=["25%", "50%", "75%", "100%"],
        difficulty_names=["Easy", "Medium"],
        algorithm_names=["A*", "BFS"],
    )


def test_default_control_panel_is_demo_focused_with_level_selector():
    gui = _make_gui(advanced=False)
    _build(gui)

    checkbox_labels = [w.label for w in gui.widget_manager.widgets if isinstance(w, DummyCheckbox)]
    dropdown_names = [w.control_name for w in gui.widget_manager.widgets if isinstance(w, DummyDropdown)]
    button_labels = [w.label for w in gui.widget_manager.widgets if isinstance(w, DummyButton)]

    assert checkbox_labels == ["Show Solution Path", "Show Mini Map", "Show Room Graph"]
    assert dropdown_names == ["level", "zoom", "algorithm"]
    assert "Generated Level" in gui.widget_manager.widgets[3].options
    assert button_labels == ["Solve Level", "Stop", "Reset Level", "Load Model", "Generate Level", "Clear Path", "Import Level", "Export Map"]
    assert "AI Generate" not in button_labels


def test_advanced_control_panel_keeps_research_tools_available():
    gui = _make_gui(advanced=True)
    _build(gui)

    checkbox_labels = [w.label for w in gui.widget_manager.widgets if isinstance(w, DummyCheckbox)]
    dropdown_names = [w.control_name for w in gui.widget_manager.widgets if isinstance(w, DummyDropdown)]
    button_labels = [w.label for w in gui.widget_manager.widgets if isinstance(w, DummyButton)]

    assert checkbox_labels == [
        "Solver Comparison",
        "Parallel Search",
        "Multi-Goal Pathfinding",
        "ML Heuristic",
        "D* Lite Replanning",
        "Show Heatmap Overlay",
        "Show Path Overlay",
        "Show MAP-Elites Overlay",
        "Show Topology Overlay",
        "Topology Legend (details)",
        "Show Minimap",
        "Diagonal Movement",
        "Use Jump Point Search (JPS)",
        "Show JPS Overlay",
        "Speedrun Mode",
        "Strict Original LoZ Rules",
        "Dynamic Difficulty",
        "Force Grid Solver",
        "Enable Prechecks (fast checks before solve)",
        "Auto-Prune Dead-Ends on Precheck",
        "Priority: Tie-Break by Locks",
        "Priority: Key-Pickup Boost",
        "Enable ARA* (weighted A*)",
        "Allow Loaded Route Teleports",
        "Keep dropdown open after select",
    ]
    assert dropdown_names == [
        "level",
        "floor",
        "zoom",
        "ara_weight",
        "difficulty",
        "presets",
        "algorithm",
        "representation",
        "match_threshold",
    ]
    assert button_labels == [
        "Start Auto-Solve",
        "Stop",
        "Generate Dungeon",
        "AI Generate",
        "Load Model",
        "Reset",
        "Path Preview",
        "Import Level",
        "Export Map",
        "Clear Path",
        "Export Route",
        "Load Route",
        "Open Temp Folder",
        "Delete Temp Files",
        "Export Topology",
        "Compare Solvers",
        "Match Missing Nodes",
        "Apply Tentative Matches",
        "Undo Last Match",
        "Undo Prune",
        "Run MAP-Elites",
    ]
    assert gui.control_panel_can_scroll is True


def test_control_panel_rebuilds_when_feature_flags_change():
    gui = _make_gui(advanced=False)
    _build(gui)

    old_minimap_checkbox = next(
        w for w in gui.widget_manager.widgets if isinstance(w, DummyCheckbox) and w.label == "Show Mini Map"
    )
    gui.feature_flags["show_minimap"] = False
    _build(gui)

    minimap_checkbox = next(
        w for w in gui.widget_manager.widgets if isinstance(w, DummyCheckbox) and w.label == "Show Mini Map"
    )
    assert minimap_checkbox is not old_minimap_checkbox
    assert minimap_checkbox.checked is False
