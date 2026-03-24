from src.gui.common.fallbacks import get_visualization_fallbacks, get_widget_fallbacks


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

