"""Import and export controls for ZeldaGUI.

Provides:
- import_txt_level: Load a full-level stitch from a .txt file (space/comma/tab separated ints)
- export_current_map: Save the current semantic grid to a .txt file
- open_import_dialog: tkinter file-picker for importing
- open_export_dialog: tkinter file-picker for exporting
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_GUI_LOAD_EXCEPTIONS = (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError)


def _set_message(gui: Any, message: str) -> None:
    """Set a GUI status message even in lightweight test doubles."""
    if hasattr(gui, "_set_message"):
        gui._set_message(message)
    else:
        setattr(gui, "message", message)


def _terminate_process(gui: Any, attr_name: str, log: Any) -> None:
    proc = getattr(gui, attr_name, None)
    if not proc:
        return

    try:
        proc.terminate()
    except _GUI_LOAD_EXCEPTIONS as exc:
        log.warning("Failed to terminate %s during level import: %s", attr_name, exc)
    try:
        setattr(gui, attr_name, None)
    except _GUI_LOAD_EXCEPTIONS:
        pass


def _clear_import_runtime_state(gui: Any, log: Any, *, terminate_processes: bool) -> None:
    """Clear solver, route, and preview state that belongs to the previous level."""
    if terminate_processes:
        _terminate_process(gui, "solver_proc", log)
        _terminate_process(gui, "preview_proc", log)

    if hasattr(gui, "preview_thread"):
        gui.preview_thread = None
    if hasattr(gui, "solver_thread"):
        gui.solver_thread = None

    if hasattr(gui, "_clear_solver_state"):
        try:
            gui._clear_solver_state(reason="level import")
        except _GUI_LOAD_EXCEPTIONS as exc:
            log.warning("Failed to clear solver state during level import: %s", exc)
    else:
        for name, value in (
            ("solver_running", False),
            ("solver_done", True),
            ("solver_proc", None),
            ("solver_thread", None),
            ("solver_outfile", None),
            ("solver_gridfile", None),
            ("solver_start_time", None),
            ("solver_starting", False),
        ):
            try:
                setattr(gui, name, value)
            except _GUI_LOAD_EXCEPTIONS:
                pass

    for name, value in (
        ("auto_path", []),
        ("solution_path", []),
        ("auto_step_idx", 0),
        ("auto_mode", False),
        ("path_preview_mode", False),
        ("preview_overlay_visible", False),
        ("path_preview_dialog", None),
        ("preview_on_next_solver_result", False),
        ("preview_proc", None),
        ("preview_thread", None),
        ("preview_done", True),
        ("preview_result", None),
        ("preview_outfile", None),
        ("preview_gridfile", None),
        ("solver_result", None),
        ("last_solver_metrics", None),
        ("loaded_route_source", None),
        ("_pending_solver_trigger", False),
        ("search_heatmap", {}),
        ("path_items_summary", {}),
        ("path_item_positions", {}),
    ):
        try:
            setattr(gui, name, value)
        except _GUI_LOAD_EXCEPTIONS:
            pass


def import_txt_level(gui: Any, filepath: str, logger_obj: Any = None) -> bool:
    """Import a full stitched level from a text file of integer tile IDs.

    Supports:
    - Space-delimited (``np.savetxt`` default)
    - Comma-delimited CSV
    - Tab-delimited TSV

    The loaded grid is appended to ``gui.maps`` and immediately displayed.

    Args:
        gui: The ZeldaGUI instance.
        filepath: Absolute or relative path to the ``.txt`` file.
        logger_obj: Optional logger (falls back to module logger).

    Returns:
        True if the level was successfully loaded and displayed.
    """
    log = logger_obj or logger
    filepath = str(filepath).strip()
    if not filepath:
        _set_message(gui, "Import cancelled (no path)")
        return False

    resolved = Path(filepath).expanduser().resolve()
    if not resolved.exists():
        _set_message(gui, f"File not found: {resolved.name}")
        log.warning("Import failed — file not found: %s", resolved)
        return False

    try:
        # Detect delimiter from first line
        with open(resolved, "r", encoding="utf-8") as fh:
            first_line = fh.readline()

        if "," in first_line:
            delimiter = ","
        elif "\t" in first_line:
            delimiter = "\t"
        else:
            delimiter = None  # whitespace (numpy default)

        data = np.loadtxt(str(resolved), dtype=np.int32, delimiter=delimiter)

        if data.ndim != 2:
            _set_message(gui, "TXT file must contain a 2D grid")
            log.warning("Import failed — data is %dD, expected 2D", data.ndim)
            return False

        if data.shape[0] < 2 or data.shape[1] < 2:
            _set_message(gui, "Grid too small (min 2×2)")
            log.warning("Import failed — grid shape %s too small", data.shape)
            return False

        _clear_import_runtime_state(gui, log, terminate_processes=True)
        previous_map_idx = int(getattr(gui, "current_map_idx", 0))
        previous_maps_len = len(getattr(gui, "maps", []) or [])
        previous_names_len = len(getattr(gui, "map_names", []) or [])

        # Append to maps list
        gui.maps.append(data)
        map_name = resolved.stem
        gui.map_names.append(f"Imported: {map_name}")

        # Switch to the newly loaded map
        gui.current_map_idx = len(gui.maps) - 1
        try:
            gui._load_current_map()
            gui._center_view()
        except _GUI_LOAD_EXCEPTIONS as exc:
            del gui.maps[previous_maps_len:]
            del gui.map_names[previous_names_len:]
            if gui.maps:
                gui.current_map_idx = max(0, min(previous_map_idx, len(gui.maps) - 1))
                try:
                    gui._load_current_map()
                    gui._center_view()
                except _GUI_LOAD_EXCEPTIONS:
                    log.exception("Failed to restore previous map after import failure")
            else:
                gui.current_map_idx = 0
            _set_message(gui, f"Import failed while loading map: {exc}")
            log.exception("Import load error for %s", resolved)
            return False

        # Clear transient state
        if getattr(gui, "effects", None):
            gui.effects.clear()
        gui.step_count = 0
        _clear_import_runtime_state(gui, log, terminate_processes=False)

        _set_message(gui, f"Imported level: {map_name} ({data.shape[0]}×{data.shape[1]})")
        log.info(
            "Imported level from %s — shape %s, unique tiles %d",
            resolved,
            data.shape,
            len(np.unique(data)),
        )
        return True

    except (ValueError, TypeError) as exc:
        _set_message(gui, f"Failed to parse TXT: {exc}")
        log.exception("Import parse error for %s", resolved)
        return False
    except OSError as exc:
        _set_message(gui, f"File read error: {exc}")
        log.exception("Import IO error for %s", resolved)
        return False


def export_current_map(gui: Any, filepath: str, logger_obj: Any = None) -> bool:
    """Export the current semantic grid to a text file.

    Args:
        gui: The ZeldaGUI instance.
        filepath: Destination file path.
        logger_obj: Optional logger.

    Returns:
        True if the map was successfully saved.
    """
    log = logger_obj or logger
    filepath = str(filepath).strip()
    if not filepath:
        _set_message(gui, "Export cancelled (no path)")
        return False

    try:
        env = getattr(gui, "env", None)
        if env is None:
            _set_message(gui, "No active map to export")
            return False

        grid = getattr(env, "grid", None)
        if grid is None:
            grid = gui.maps[gui.current_map_idx]

        grid = np.asarray(grid, dtype=np.int32)
        resolved = Path(filepath).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)

        np.savetxt(str(resolved), grid, fmt="%d")
        png_path = resolved.with_suffix(".png")
        from src.gui.rendering.level_image_export import save_level_grid_png

        save_level_grid_png(grid, png_path, np_module=np)

        _set_message(gui, f"Exported: {resolved.name} + {png_path.name}")
        log.info("Exported map to %s and %s - shape %s", resolved, png_path, grid.shape)
        return True

    except (OSError, ValueError, TypeError) as exc:
        _set_message(gui, f"Export failed: {exc}")
        log.exception("Export error")
        return False


def open_import_dialog(gui: Any, logger_obj: Any = None) -> bool:
    """Open a native file picker to import a .txt level.

    Uses tkinter so it works even inside a pygame event loop.

    Returns:
        True if a file was successfully imported.
    """
    log = logger_obj or logger
    try:
        import tkinter as tk
        from tkinter import filedialog
    except (ImportError, RuntimeError) as exc:
        _set_message(gui, f"File dialog unavailable: {exc}")
        log.warning("tkinter not available for import dialog: %s", exc)
        return False

    tk_error = getattr(tk, "TclError", RuntimeError)
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        initial_dir = str(Path.cwd())
        # Demo imports usually live in examples/. Generated outputs remain easy to reach.
        examples_dir = Path.cwd() / "examples"
        outputs_dir = Path.cwd() / "outputs"
        if examples_dir.exists():
            initial_dir = str(examples_dir)
        elif outputs_dir.exists():
            initial_dir = str(outputs_dir)

        selected = filedialog.askopenfilename(
            title="Import Generated Level (.txt)",
            initialdir=initial_dir,
            filetypes=[
                ("Text Grid Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*"),
            ],
        )
    except (tk_error, AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
        _set_message(gui, f"File picker failed: {exc}")
        log.warning("Import file picker failed: %s", exc)
        return False
    finally:
        if root is not None:
            try:
                root.destroy()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass

    if not selected:
        _set_message(gui, "Import cancelled")
        return False

    return import_txt_level(gui, selected, logger_obj=log)


def open_export_dialog(gui: Any, logger_obj: Any = None) -> bool:
    """Open a native save-as dialog to export the current map.

    Returns:
        True if the map was successfully exported.
    """
    log = logger_obj or logger
    try:
        import tkinter as tk
        from tkinter import filedialog
    except (ImportError, RuntimeError) as exc:
        _set_message(gui, f"File dialog unavailable: {exc}")
        log.warning("tkinter not available for export dialog: %s", exc)
        return False

    tk_error = getattr(tk, "TclError", RuntimeError)
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        # Generate a sensible default filename
        map_name = "level"
        try:
            raw_name = gui.map_names[gui.current_map_idx]
            # Sanitize for filesystem
            safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(raw_name))
            map_name = safe_name.strip() or "level"
        except (IndexError, AttributeError, TypeError):
            pass

        exports_dir = Path.cwd() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        selected = filedialog.asksaveasfilename(
            title="Export Current Map",
            initialdir=str(exports_dir),
            initialfile=f"{map_name}.txt",
            defaultextension=".txt",
            filetypes=[
                ("Text Grid Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*"),
            ],
        )
    except (tk_error, AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
        _set_message(gui, f"File picker failed: {exc}")
        log.warning("Export file picker failed: %s", exc)
        return False
    finally:
        if root is not None:
            try:
                root.destroy()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass

    if not selected:
        _set_message(gui, "Export cancelled")
        return False

    return export_current_map(gui, selected, logger_obj=log)
