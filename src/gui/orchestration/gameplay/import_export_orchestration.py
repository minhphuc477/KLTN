"""Import/export orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.gameplay.import_export_controls import (
    import_txt_level as _import_txt_level,
    export_current_map as _export_current_map,
    open_import_dialog as _open_import_dialog,
    open_export_dialog as _open_export_dialog,
)


def import_txt_level(*, gui, filepath, logger):
    return _import_txt_level(gui, filepath, logger_obj=logger)


def export_current_map(*, gui, filepath, logger):
    return _export_current_map(gui, filepath, logger_obj=logger)


def open_import_dialog(*, gui, logger):
    return _open_import_dialog(gui, logger_obj=logger)


def open_export_dialog(*, gui, logger):
    return _open_export_dialog(gui, logger_obj=logger)
