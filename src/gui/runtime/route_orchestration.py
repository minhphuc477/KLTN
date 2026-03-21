"""Route import/export orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def export_route(*, gui, export_route_helper):
    export_route_helper(gui)


def load_route(*, gui, load_route_helper):
    load_route_helper(gui)
