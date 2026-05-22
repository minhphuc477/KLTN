from types import SimpleNamespace

import numpy as np

from src.gui.gameplay.import_export_controls import export_current_map, import_txt_level


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class DummyProcess:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


class DummyEffects:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class DummyGui:
    def __init__(self):
        self.maps = [np.array([[21, 1], [1, 22]], dtype=np.int32)]
        self.map_names = ["Base"]
        self.current_map_idx = 0
        self.messages = []
        self.loaded = 0
        self.centered = 0
        self.clear_reasons = []
        self.fail_import_load = False
        self.effects = DummyEffects()
        self.step_count = 8
        self.env = SimpleNamespace(grid=self.maps[0])

        self.auto_path = [(0, 0), (0, 1)]
        self.solution_path = [(0, 0), (0, 1)]
        self.auto_step_idx = 1
        self.auto_mode = True
        self.path_preview_mode = True
        self.preview_overlay_visible = True
        self.path_preview_dialog = object()
        self.preview_on_next_solver_result = True
        self.loaded_route_source = "route_json"
        self.solver_result = {"old": True}
        self.last_solver_metrics = {"nodes": 1}
        self.search_heatmap = {(0, 0): 1}
        self.path_items_summary = {"key": 1}
        self.path_item_positions = {"key": [(0, 0)]}
        self._pending_solver_trigger = True

        self.solver_proc = None
        self.preview_proc = None
        self.solver_thread = object()
        self.preview_thread = object()

    def _set_message(self, message, duration=3.0):
        self.message = message
        self.messages.append(message)

    def _clear_solver_state(self, reason=""):
        self.clear_reasons.append(reason)
        self.solver_running = False
        self.solver_done = True
        self.solver_proc = None
        self.solver_thread = None

    def _load_current_map(self):
        if self.fail_import_load and self.current_map_idx == len(self.maps) - 1:
            raise RuntimeError("load failed")
        self.loaded += 1
        self.env = SimpleNamespace(grid=self.maps[self.current_map_idx])

    def _center_view(self):
        self.centered += 1


def test_import_txt_level_can_import_multiple_levels_and_clears_stale_route_state(tmp_path):
    first = tmp_path / "first_level.txt"
    second = tmp_path / "second_level.txt"
    first_grid = np.array([[21, 1], [1, 22]], dtype=np.int32)
    second_grid = np.array([[21, 1, 1], [2, 1, 22]], dtype=np.int32)
    np.savetxt(first, first_grid, fmt="%d")
    np.savetxt(second, second_grid, fmt="%d")

    gui = DummyGui()
    assert import_txt_level(gui, first, logger_obj=DummyLogger()) is True

    solver_proc = DummyProcess()
    preview_proc = DummyProcess()
    gui.solver_proc = solver_proc
    gui.preview_proc = preview_proc
    gui.auto_path = [(9, 9)]
    gui.solution_path = [(9, 9)]
    gui.loaded_route_source = "route_json"
    gui.preview_overlay_visible = True
    gui.path_preview_dialog = object()
    gui.preview_on_next_solver_result = True

    assert import_txt_level(gui, second, logger_obj=DummyLogger()) is True

    assert solver_proc.terminated is True
    assert preview_proc.terminated is True
    assert gui.current_map_idx == 2
    assert gui.map_names[-1] == "Imported: second_level"
    assert np.array_equal(gui.maps[-1], second_grid)
    assert gui.auto_path == []
    assert gui.solution_path == []
    assert gui.auto_step_idx == 0
    assert gui.auto_mode is False
    assert gui.preview_overlay_visible is False
    assert gui.path_preview_dialog is None
    assert gui.preview_on_next_solver_result is False
    assert gui.loaded_route_source is None
    assert gui.solver_result is None
    assert gui.search_heatmap == {}
    assert gui.path_items_summary == {}
    assert gui.path_item_positions == {}
    assert gui.effects.cleared is True
    assert gui.step_count == 0
    assert gui.clear_reasons[-1] == "level import"
    assert "Imported level: second_level" in gui.messages[-1]


def test_export_current_map_writes_txt_and_png(tmp_path):
    gui = DummyGui()
    out_path = tmp_path / "exported_level.txt"

    assert export_current_map(gui, out_path, logger_obj=DummyLogger()) is True

    assert out_path.exists()
    assert out_path.with_suffix(".png").exists()
    assert np.array_equal(np.loadtxt(out_path, dtype=np.int32), gui.env.grid)
    assert "exported_level.png" in gui.messages[-1]


def test_import_txt_level_rolls_back_if_loaded_map_cannot_initialize(tmp_path):
    level_file = tmp_path / "bad_runtime_level.txt"
    np.savetxt(level_file, np.array([[21, 1], [1, 22]], dtype=np.int32), fmt="%d")

    gui = DummyGui()
    gui.fail_import_load = True

    assert import_txt_level(gui, level_file, logger_obj=DummyLogger()) is False

    assert len(gui.maps) == 1
    assert gui.map_names == ["Base"]
    assert gui.current_map_idx == 0
    assert np.array_equal(gui.env.grid, gui.maps[0])
    assert "Import failed while loading map" in gui.messages[-1]
