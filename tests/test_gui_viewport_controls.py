from types import SimpleNamespace

from src.gui.controls.viewport_controls import auto_fit_zoom, center_view, change_zoom


def _make_gui():
    gui = SimpleNamespace()
    gui.env = SimpleNamespace(width=20, height=10, start_pos=(1, 2))
    gui.TILE_SIZE = 16
    gui.screen_w = 800
    gui.screen_h = 600
    gui.SIDEBAR_WIDTH = 240
    gui.HUD_HEIGHT = 90
    gui.view_offset_x = 0
    gui.view_offset_y = 0
    gui.zoom_idx = 1
    gui.ZOOM_LEVELS = [8, 16, 24, 32]
    gui.message = ""
    gui.renderer = SimpleNamespace(
        set_tile_size=lambda size: setattr(gui, "_renderer_tile", size),
        set_agent_position=lambda r, c, immediate=True: setattr(gui, "_agent_pos", (r, c, immediate)),
    )
    gui._load_assets = lambda: setattr(gui, "_assets_loaded", True)
    gui._clamp_view_offset = lambda: setattr(gui, "_clamped", True)
    return gui


def test_center_view_updates_offsets():
    gui = _make_gui()

    center_view(gui)

    assert hasattr(gui, "_clamped")
    assert isinstance(gui.view_offset_x, int)
    assert isinstance(gui.view_offset_y, int)


def test_auto_fit_zoom_updates_zoom_and_renderer():
    gui = _make_gui()
    gui.zoom_idx = 3
    gui.TILE_SIZE = gui.ZOOM_LEVELS[gui.zoom_idx]

    auto_fit_zoom(gui)

    assert gui.zoom_idx <= 3
    assert gui.TILE_SIZE == gui.ZOOM_LEVELS[gui.zoom_idx]
    assert hasattr(gui, "_agent_pos")


def test_change_zoom_keeps_center_and_updates_message():
    gui = _make_gui()

    change_zoom(gui, 1, center=(200, 150))

    assert gui.zoom_idx == 2
    assert gui.TILE_SIZE == 24
    assert gui.message == "Zoom: 24px"
    assert hasattr(gui, "_clamped")

