"""Bootstrap helpers for ZeldaGUI initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_repo_export_dirs(*, gui: Any, path_cls: type[Path], logger: Any) -> None:
    """Initialize repository-local export directories."""
    gui.repo_root = path_cls(__file__).resolve().parents[3]
    gui.exports_root = gui.repo_root / "exports"
    gui.route_export_dir = gui.exports_root / "routes"
    gui.topology_export_dir = gui.exports_root / "topology"
    gui.artifacts_dir = str(gui.exports_root / "artifacts")

    try:
        gui.route_export_dir.mkdir(parents=True, exist_ok=True)
        gui.topology_export_dir.mkdir(parents=True, exist_ok=True)
        path_cls(gui.artifacts_dir).mkdir(parents=True, exist_ok=True)
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("Failed to create export directories under repo root")


def configure_windows_dpi_awareness(*, logger: Any) -> None:
    """Best-effort DPI-awareness setup on Windows before pygame init."""
    try:
        import ctypes

        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
            logger.debug("SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2) succeeded")
            return
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            pass

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            logger.debug("SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE) succeeded")
            return
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            pass

        try:
            ctypes.windll.user32.SetProcessDPIAware()
            logger.debug("SetProcessDPIAware() succeeded")
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Could not set process DPI awareness")
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("DPI awareness calls not supported on this platform")


def initialize_pygame_runtime(*, pygame: Any, logger: Any) -> None:
    """Initialize pygame and make cursor setting resilient across environments."""
    try:
        pygame.init()
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("Failed to initialize Pygame")
        raise

    try:
        orig_set_cursor = pygame.mouse.set_cursor

        def wrapped_set_cursor(cursor: Any) -> None:
            try:
                orig_set_cursor(cursor)
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.debug("set_cursor failed or unsupported in this environment", exc_info=True)

        pygame.mouse.set_cursor = wrapped_set_cursor
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Could not wrap pygame.mouse.set_cursor; continuing")

