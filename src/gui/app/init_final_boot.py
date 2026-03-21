"""Final constructor boot sequence helpers for ZeldaGUI."""

from __future__ import annotations

from typing import Any


def finalize_initial_map_boot(*, gui: Any, pygame: Any, logger: Any) -> None:
    """Load initial map, center view, initialize control panel, and paint first frame."""
    gui._load_current_map()
    gui._center_view()

    if gui.control_panel_enabled:
        gui._init_control_panel()

    try:
        gui._render()
        pygame.display.flip()
    except Exception:
        logger.debug("Initial frame paint failed during constructor bootstrap", exc_info=True)
