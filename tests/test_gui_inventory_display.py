from types import SimpleNamespace

from src.gui.overlay.inventory_display import get_path_items_display_text, render_item_legend


class _Rendered:
    pass


class _Font:
    def render(self, text, antialias, color):
        _ = (text, antialias, color)
        return _Rendered()


class _Surface:
    def __init__(self):
        self.blits = []

    def fill(self, color):
        _ = color

    def blit(self, surf, pos):
        self.blits.append((surf, pos))


class _Pygame:
    SRCALPHA = object()

    @staticmethod
    def Surface(size, flags=None):
        _ = (size, flags)
        return _Surface()

    class draw:
        @staticmethod
        def rect(surface, color, rect, width=0):
            _ = (surface, color, rect, width)


def test_get_path_items_display_text_formats_compact_tokens():
    gui = SimpleNamespace(path_items_summary={"keys": 2, "doors_locked": 1, "triforce": 1})

    text = get_path_items_display_text(gui)

    assert "2[K]" in text
    assert "1[D]" in text
    assert "1[T]" in text


def test_render_item_legend_blits_content():
    env_state = SimpleNamespace(keys=1, bomb_count=2, has_boss_key=False)
    gui = SimpleNamespace(
        env=SimpleNamespace(state=env_state),
        screen_h=600,
        small_font=_Font(),
        path_items_summary={"keys": 1},
        keys_collected=1,
        total_keys=2,
        bombs_collected=0,
        total_bombs=0,
        boss_keys_collected=0,
        total_boss_keys=0,
        keys_used=0,
        bombs_used=0,
        _sync_inventory_counters=lambda: None,
    )
    surface = _Surface()

    render_item_legend(gui, surface, _Pygame)

    assert len(surface.blits) > 0

