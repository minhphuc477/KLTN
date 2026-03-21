"""MAP-Elites orchestration helpers extracted from gui_runner."""

from __future__ import annotations

from typing import Any


def start_map_elites(
    *,
    gui: Any,
    n_samples: int,
    resolution: int,
    threading_module: Any,
) -> None:
    """Start MAP-Elites in background if not already running."""
    if getattr(gui, "map_elites_thread", None) and gui.map_elites_thread.is_alive():
        gui._show_toast("MAP-Elites already running", 3.0, "warning")
        return

    maps_copy = list(getattr(gui, "maps", []) or [])
    if not maps_copy:
        gui._show_toast("No maps available for MAP-Elites", 3.0, "warning")
        return

    gui._show_toast(f"Starting MAP-Elites ({n_samples} samples, res={resolution})", 3.0)
    gui.map_elites_thread = threading_module.Thread(
        target=gui._map_elites_worker,
        args=(maps_copy, n_samples, resolution),
        daemon=True,
    )
    gui.map_elites_thread.start()


def map_elites_worker(
    *,
    gui: Any,
    maps: Any,
    n_samples: int,
    resolution: int,
    logger: Any,
    os_module: Any,
    run_map_elites_on_maps_fn: Any = None,
    plot_heatmap_fn: Any = None,
) -> None:
    """Run MAP-Elites evaluation and persist optional heatmap artifact."""
    try:
        if run_map_elites_on_maps_fn is None:
            from src.simulation.map_elites import run_map_elites_on_maps as run_map_elites_on_maps_fn
        if plot_heatmap_fn is None:
            try:
                from src.simulation.map_elites import plot_heatmap as plot_heatmap_fn
            except Exception:
                plot_heatmap_fn = None

        evaluator, occ = run_map_elites_on_maps_fn(maps, resolution=resolution)
        try:
            out_dir = getattr(gui, "artifacts_dir", ".") or "."
            os_module.makedirs(out_dir, exist_ok=True)
            out_path = os_module.path.join(out_dir, f"map_elites_heatmap_res{resolution}.png")
            if plot_heatmap_fn is not None:
                plot_heatmap_fn(occ, out_path)
                gui.map_elites_heatmap_path = out_path
            else:
                gui.map_elites_heatmap_path = None
        except Exception:
            gui.map_elites_heatmap_path = None

        gui.map_elites_result = evaluator
        gui._show_toast("MAP-Elites completed", 4.0, "success")
    except Exception as exc:
        logger.exception("MAP-Elites worker failed: %s", exc)
        gui._show_toast(f"MAP-Elites failed: {exc}", 4.0, "error")
