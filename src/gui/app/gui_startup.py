"""GUI startup orchestration helpers."""

from __future__ import annotations


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
    gui.run()
