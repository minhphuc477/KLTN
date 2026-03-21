from types import SimpleNamespace

from src.gui.rendering.render_helpers import (
    default_topology_semantics,
    render_solver_comparison_overlay,
    render_topology_overlay,
)


class _FakeText:
    def __init__(self, text: str):
        self._text = text

    def get_width(self):
        return max(1, len(self._text))

    def get_height(self):
        return 10


class _FakeFont:
    def render(self, text, _aa, _color):
        return _FakeText(str(text))


class _FakeSurface:
    def __init__(self, size=(320, 240), _flags=None):
        self._size = size
        self.blit_calls = []

    def get_size(self):
        return self._size

    def blit(self, obj, pos):
        self.blit_calls.append((obj, pos))


class _FakeRect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class _FakeDraw:
    def __init__(self):
        self.lines = []
        self.circles = []
        self.rects = []

    def line(self, _surface, _color, p1, p2, _width):
        self.lines.append((p1, p2))

    def circle(self, _surface, _color, center, _radius, _width=0):
        self.circles.append(center)

    def rect(self, _surface, _color, rect, _width=0):
        self.rects.append((rect.x, rect.y, rect.w, rect.h))


class _FakePygame:
    SRCALPHA = 1

    def __init__(self):
        self.draw = _FakeDraw()
        self.font = SimpleNamespace(SysFont=lambda *_a, **_k: _FakeFont())
        self.Rect = _FakeRect

    def Surface(self, size, flags=None):
        return _FakeSurface(size=size, _flags=flags)


class _FakeGraph:
    def __init__(self):
        self._nodes = [0, 1]
        self._edges = [(0, 1, {"type": "open"})]

    def nodes(self):
        return list(self._nodes)

    def edges(self, data=False):
        if data:
            return list(self._edges)
        return [(u, v) for u, v, _d in self._edges]


def test_default_topology_semantics_shape():
    semantics = default_topology_semantics()
    assert "nodes" in semantics
    assert "edges" in semantics
    assert "k" in semantics["nodes"]
    assert "k" in semantics["edges"]


def test_render_topology_overlay_draws_graph_without_exception():
    pygame = _FakePygame()
    surface = _FakeSurface((400, 300))
    current = SimpleNamespace(
        graph=_FakeGraph(),
        room_positions={(0, 0): (2, 2), (0, 1): (2, 14)},
        room_to_node={(0, 0): 0, (0, 1): 1},
    )

    render_topology_overlay(
        surface=surface,
        current=current,
        tile_size=16,
        view_offset_x=0,
        view_offset_y=0,
        pygame=pygame,
    )

    assert len(pygame.draw.lines) >= 1
    assert len(pygame.draw.circles) >= 2


def test_render_solver_comparison_overlay_renders_rows():
    pygame = _FakePygame()
    surface = _FakeSurface((500, 300))
    results = [
        {"name": "A*", "success": True, "path_len": 12, "nodes": 44, "time_ms": 9},
        {
            "name": "CBS (Balanced)",
            "success": True,
            "path_len": 14,
            "confusion": 0.3,
            "cog_load": 0.4,
            "time_ms": 11,
        },
    ]

    render_solver_comparison_overlay(
        surface=surface,
        results=results,
        screen_w=500,
        sidebar_width=180,
        pygame=pygame,
    )

    assert len(pygame.draw.rects) >= 2
    assert len(surface.blit_calls) >= 3

