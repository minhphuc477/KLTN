"""GUI startup orchestration helpers."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)


def _apply_startup_level_and_route_overrides(gui, *, os_module, print_fn=print):
    """Apply command-line/env startup imports after the GUI object exists."""
    load_txt = str(os_module.environ.get("KLTN_LOAD_TXT", "") or "").strip()
    if load_txt:
        try:
            from src.gui.gameplay.import_export_controls import import_txt_level

            ok = import_txt_level(gui, load_txt, logger_obj=logger)
            if ok:
                print_fn(f"Loaded startup TXT level: {load_txt}")
            else:
                print_fn(f"Failed to load startup TXT level: {load_txt}")
        except (AttributeError, RuntimeError, ValueError, TypeError, ImportError, OSError) as exc:
            print_fn(f"Failed to load startup TXT level: {exc}")
            logger.exception("Startup TXT import failed")

    load_route = str(os_module.environ.get("KLTN_LOAD_ROUTE_JSON", "") or "").strip()
    if load_route:
        try:
            from src.gui.runtime.route_io import load_route_file

            ok = load_route_file(gui, load_route)
            if ok:
                print_fn(f"Loaded startup route JSON: {load_route}")
            else:
                print_fn(f"Failed to load startup route JSON: {load_route}")
        except (AttributeError, RuntimeError, ValueError, TypeError, ImportError, OSError) as exc:
            print_fn(f"Failed to load startup route JSON: {exc}")
            logger.exception("Startup route import failed")


def run_gui_main(*, pygame_available, load_maps_fn, create_test_map_fn, gui_cls, print_fn=print):
    """Boot maps, create GUI instance, and run the interactive loop."""
    print_fn("=== ZAVE GUI Runner ===\n")

    if not pygame_available:
        print_fn("Pygame is not installed. Please run: pip install pygame")
        return

    maps, map_names = load_maps_fn()

    if maps:
        print_fn(f"Loaded {len(maps)} maps from data adapter")
    else:
        print_fn("Using test map")
        maps = [create_test_map_fn()]
        map_names = ["Test Map"]

    gui = gui_cls(maps, map_names)
    _apply_startup_level_and_route_overrides(gui, os_module=os, print_fn=print_fn)
    gui.run()
