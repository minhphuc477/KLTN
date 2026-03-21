"""Helpers for GUI temp-folder operations and cleanup orchestration."""

from typing import Any, Tuple


def open_temp_folder(gui: Any, tempfile_module: Any, open_folder_helper: Any) -> None:
    """Open OS temp folder and set user-facing message."""
    temp_dir = tempfile_module.gettempdir()
    ok, err = open_folder_helper(temp_dir)
    if ok:
        gui._set_message(f"Opened temp folder: {temp_dir}", 2.0)
    else:
        gui._set_message(f"Open temp folder failed: {err}", 3.0)


def collect_temp_file_candidates(
    gui: Any,
    tempfile_module: Any,
    list_existing_paths_helper: Any,
    find_temp_files_helper: Any,
) -> Tuple[list, list]:
    """Collect tracked and stale temp files used by solver/preview/watchdog flows."""
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
    os_module: Any,
    logger: Any,
    collect_candidates_fn: Any,
    list_existing_paths_helper: Any,
    delete_files_helper: Any,
) -> None:
    """Delete stale temp files while preserving active solver/preview artifacts."""
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
        logger.warning("Temp cleanup failures: %s", failures[:3])
    gui._set_message(msg, 3.0)
