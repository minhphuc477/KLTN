"""Route import/export orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.runtime.route_io import export_route as _export_route, load_route as _load_route


def export_route(*, gui):
    _export_route(gui)


def load_route(*, gui):
    _load_route(gui)
