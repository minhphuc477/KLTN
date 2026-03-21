"""Entrypoint orchestration bridges for gui_runner module-level functions."""

from __future__ import annotations


def load_maps_from_adapter(*, os_module, file_path, print_fn, load_maps_from_adapter_helper):
    return load_maps_from_adapter_helper(os_module=os_module, file_path=file_path, print_fn=print_fn)


def run_main_entry(*, pygame_available, load_maps_fn, create_test_map_fn, gui_cls, print_fn, run_gui_main_helper):
    run_gui_main_helper(
        pygame_available=pygame_available,
        load_maps_fn=load_maps_fn,
        create_test_map_fn=create_test_map_fn,
        gui_cls=gui_cls,
        print_fn=print_fn,
    )
