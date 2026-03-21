from src.gui.overlay.help_overlay import render_help_overlay


class FakeSurface:
    def __init__(self):
        self.fills = []
        self.blits = []

    def fill(self, color):
        self.fills.append(color)

    def blit(self, obj, pos):
        self.blits.append((obj, pos))


class FakeFont:
    def render(self, text, aa, color):
        return (text, aa, color)


class FakePygame:
    SRCALPHA = 0

    def Surface(self, size, flags):
        _ = size, flags
        return FakeSurface()


class DummyGui:
    def __init__(self):
        self.screen_w = 800
        self.screen_h = 600
        self.screen = FakeSurface()
        self.big_font = FakeFont()
        self.font = FakeFont()
        self.small_font = FakeFont()


def test_render_help_overlay_blits_multiple_lines():
    gui = DummyGui()
    render_help_overlay(gui, FakePygame())
    # one overlay blit + many line blits
    assert len(gui.screen.blits) > 10

