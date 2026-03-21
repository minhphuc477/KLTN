"""Route import/export operations for GUI runner."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.gui.runtime.route_payload import apply_loaded_route_data, build_route_export_payload


logger = logging.getLogger(__name__)


def export_route(gui: Any) -> None:
    """Export current route to a timestamped JSON file."""
    path = list(getattr(gui, "auto_path", []) or getattr(gui, "solution_path", []) or [])
    if not path:
        gui.message = "No route to export (solve first)"
        return

    try:
        export_dir = Path(getattr(gui, "route_export_dir", Path.cwd()))
        export_dir.mkdir(parents=True, exist_ok=True)

        export_data = build_route_export_payload(gui, path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = export_dir / f"route_{timestamp}.json"

        with export_file.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        try:
            display_path = export_file.relative_to(getattr(gui, "repo_root", Path.cwd()))
        except Exception:
            display_path = export_file
        gui.message = f"Route exported to {display_path}"
        logger.info("Route exported to %s", export_file)
    except Exception as e:
        gui.message = f"Route export failed: {e}"
        logger.error("Route export error: %s", e)


def load_route(gui: Any) -> None:
    """Load most recent route export from JSON and apply it to GUI state."""
    try:
        export_dir = Path(getattr(gui, "route_export_dir", Path.cwd()))
        route_files = sorted(export_dir.glob("route_*.json"), reverse=True)
        if not route_files:
            gui.message = "No saved routes found"
            return

        latest_file = route_files[0]
        with latest_file.open("r", encoding="utf-8") as f:
            route_data = json.load(f)

        required_keys = ["path", "start", "goal"]
        if not all(key in route_data for key in required_keys):
            gui.message = f"Invalid route file: {latest_file}"
            return

        path_len = apply_loaded_route_data(gui, route_data)
        try:
            display_path = latest_file.relative_to(getattr(gui, "repo_root", Path.cwd()))
        except Exception:
            display_path = latest_file
        gui.message = f"Route loaded from {display_path} ({path_len} steps)"
        logger.info("Route loaded from %s", latest_file)
    except Exception as e:
        gui.message = f"Route loading failed: {e}"
        logger.error("Route loading error: %s", e)
