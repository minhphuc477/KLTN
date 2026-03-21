"""Visualization and debug path initialization helpers for ZeldaGUI."""

from __future__ import annotations

from typing import Any


def initialize_visualization_components(*, gui: Any, visualization_available: bool, renderer_cls: Any, effects_cls: Any, hud_cls: Any) -> None:
    """Initialize renderer/effects/HUD with available implementation or fallbacks."""
    if visualization_available:
        gui.renderer = renderer_cls(gui.TILE_SIZE)
        gui.effects = effects_cls()
        gui.modern_hud = hud_cls()
        return

    # Fallback classes are already injected into gui_runner scope when imports fail.
    gui.renderer = renderer_cls(gui.TILE_SIZE)
    gui.effects = effects_cls()
    gui.modern_hud = hud_cls()


def initialize_debug_test_path(*, gui: Any, os_module: Any) -> None:
    """Optional debug path overlay setup driven by KLTN_DEBUG_TEST_PATH."""
    if os_module.environ.get("KLTN_DEBUG_TEST_PATH") == "1":
        gui._test_path = [(5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (8, 8), (8, 9), (8, 10)]
        print(f"[DEBUG_INIT] _test_path ENABLED with {len(gui._test_path)} points for visual testing")
        return

    gui._test_path = None
