"""Helpers for GUI temp-folder operations and cleanup orchestration."""

import os
import tempfile
from typing import Any, Tuple

from src.gui.runtime.temp_file_tools import (
    delete_files as _delete_files_helper,
    find_temp_files as _find_temp_files_helper,
    list_existing_paths as _list_existing_paths_helper,
    open_folder as _open_folder_helper,
)


def open_temp_folder(gui: Any, tempfile_module: Any = None, open_folder_helper: Any = None) -> None:
    """Open OS temp folder and set user-facing message."""
    tempfile_module = tempfile_module or tempfile
    open_folder_helper = open_folder_helper or _open_folder_helper
    temp_dir = tempfile_module.gettempdir()
    ok, err = open_folder_helper(temp_dir)
    if ok:
        gui._set_message(f"Opened temp folder: {temp_dir}", 2.0)
    else:
        gui._set_message(f"Open temp folder failed: {err}", 3.0)


def collect_temp_file_candidates(
    gui: Any,
    tempfile_module: Any = None,
    list_existing_paths_helper: Any = None,
    find_temp_files_helper: Any = None,
) -> Tuple[list, list]:
    """Collect tracked and stale temp files used by solver/preview/watchdog flows."""
    tempfile_module = tempfile_module or tempfile
    list_existing_paths_helper = list_existing_paths_helper or _list_existing_paths_helper
    find_temp_files_helper = find_temp_files_helper or _find_temp_files_helper

    tracked = list_existing_paths_helper(
        [
            getattr(gui, "solver_outfile", None),
            getattr(gui, "solver_gridfile", None),
            getattr(gui, "preview_outfile", None),
            getattr(gui, "preview_gridfile", None),
        ]
    )
    temp_dir = tempfile_module.gettempdir()
    stale = find_temp_files_helper(
        temp_dir,
        [
            "zave_solver_out_*.pkl",
            "zave_grid_*.npy",
            "zave_grid_*.pkl",
            "zave_preview_*.npy",
            "zave_preview_out_*.pkl",
            "zave_watchdog_trace_*.txt",
            "zave_watchdog_shot_*.png",
        ],
    )
    return tracked, stale


def delete_temp_files(
    gui: Any,
    os_module: Any = None,
    logger: Any = None,
    collect_candidates_fn: Any = None,
    list_existing_paths_helper: Any = None,
    delete_files_helper: Any = None,
) -> None:
    """Delete stale temp files while preserving active solver/preview artifacts."""
    os_module = os_module or os
    list_existing_paths_helper = list_existing_paths_helper or _list_existing_paths_helper
    delete_files_helper = delete_files_helper or _delete_files_helper
    collect_candidates_fn = collect_candidates_fn or (lambda: collect_temp_file_candidates(gui))

    tracked, stale = collect_candidates_fn()
    active_tracked = set()
    if getattr(gui, "solver_running", False):
        active_tracked.update(
            list_existing_paths_helper(
                [
                    getattr(gui, "solver_outfile", None),
                    getattr(gui, "solver_gridfile", None),
                ]
            )
        )

    preview_proc = getattr(gui, "preview_proc", None)
    preview_alive = False
    try:
        preview_alive = bool(preview_proc and preview_proc.is_alive())
    except Exception:
        preview_alive = False
    if preview_alive:
        active_tracked.update(
            list_existing_paths_helper(
                [
                    getattr(gui, "preview_outfile", None),
                    getattr(gui, "preview_gridfile", None),
                ]
            )
        )

    candidates = list(dict.fromkeys(tracked + stale))
    to_delete = [p for p in candidates if p not in active_tracked]
    deleted, failures = delete_files_helper(to_delete)

    for attr in ("solver_outfile", "solver_gridfile", "preview_outfile", "preview_gridfile"):
        p = getattr(gui, attr, None)
        if p and not os_module.path.exists(str(p)):
            setattr(gui, attr, None)

    skipped = len([p for p in candidates if p in active_tracked])
    if deleted == 0 and not failures and skipped == 0:
        gui._set_message("No temp files found", 2.0)
        return

    msg = f"Deleted {deleted} temp files"
    if skipped:
        msg += f", skipped {skipped} active"
    if failures:
        msg += f", {len(failures)} failed"
        if logger is not None:
            logger.warning("Temp cleanup failures: %s", failures[:3])
    gui._set_message(msg, 3.0)
