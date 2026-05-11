from src.gui.common.fallbacks import get_visualization_fallbacks, get_widget_fallbacks


import importlib
import sys


class _FakeSurface:
    def __init__(self, size):
        self._size = size
        self._filled = None

    def fill(self, color):
        self._filled = color


class _FakePygame:
    def Surface(self, size):
        return _FakeSurface(size)


def test_visualization_fallbacks_keys_and_noop_interfaces():
    symbols = get_visualization_fallbacks(pygame_available=False, pygame_module=None)

    expected_keys = {
        "ZeldaRenderer",
        "ThemeConfig",
        "Vector2",
        "EffectManager",
        "PopEffect",
        "FlashEffect",
        "RippleEffect",
        "ItemCollectionEffect",
        "ItemUsageEffect",
        "ItemMarkerEffect",
        "ModernHUD",
        "HUDTheme",
        "PathPreviewDialog",
    }
    assert expected_keys.issubset(symbols.keys())

    renderer = symbols["ZeldaRenderer"](tile_size=16)
    assert renderer.tile_size == 16
    assert renderer.set_agent_position() is None
    assert renderer.render() is None

    hud = symbols["ModernHUD"]()
    assert hud.update_game_state() is None
    assert hasattr(hud, "inventory")


def test_visualization_fallback_renderer_tile_with_fake_pygame():
    fake_pygame = _FakePygame()
    symbols = get_visualization_fallbacks(
        pygame_available=True,
        pygame_module=fake_pygame,
    )

    renderer = symbols["ZeldaRenderer"](tile_size=20)
    tile = renderer.sprite_manager.get_tile(1, 20)
    assert isinstance(tile, _FakeSurface)
    assert tile._size == (20, 20)


def test_widget_fallbacks_manager_basic_flow():
    symbols = get_widget_fallbacks()

    widget_cls = symbols["CheckboxWidget"]
    manager_cls = symbols["WidgetManager"]

    manager = manager_cls()
    widget = widget_cls()
    manager.add_widget(widget)

    assert len(manager.widgets) == 1
    assert manager.handle_mouse_down((0, 0), 1) is False
    assert manager.handle_mouse_up((0, 0), 1) is False
    assert manager.snapshot_dropdown_state() == {}


def test_gui_runner_import_uses_visualization_fallback_when_optional_imports_fail():
    class BlockVisualizationImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("src.visualization"):
                raise ImportError("forced missing visualization dependency")
            return None

    for name in list(sys.modules):
        if name == "gui_runner" or name.startswith("src.visualization"):
            sys.modules.pop(name, None)

    blocker = BlockVisualizationImports()
    sys.meta_path.insert(0, blocker)
    try:
        module = importlib.import_module("gui_runner")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop("gui_runner", None)

    assert module.VISUALIZATION_AVAILABLE is False
    assert module.ZeldaRenderer(16).tile_size == 16

