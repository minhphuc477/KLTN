"""Entrypoint orchestration bridges for gui_runner module-level functions."""

from __future__ import annotations

from src.gui.app.gui_startup import run_gui_main as _run_gui_main_helper
from src.gui.app.map_adapter_loader import load_maps_from_adapter as _load_maps_from_adapter_helper


def load_maps_from_adapter(*, os_module, file_path, print_fn):
    return _load_maps_from_adapter_helper(os_module=os_module, file_path=file_path, print_fn=print_fn)


def run_main_entry(*, pygame_available, load_maps_fn, create_test_map_fn, gui_cls, print_fn):
    _run_gui_main_helper(
        pygame_available=pygame_available,
        load_maps_fn=load_maps_fn,
        create_test_map_fn=create_test_map_fn,
        gui_cls=gui_cls,
        print_fn=print_fn,
    )
