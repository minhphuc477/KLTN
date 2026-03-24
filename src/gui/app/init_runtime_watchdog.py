"""Runtime timing/watchdog initialization helpers for ZeldaGUI."""

from __future__ import annotations

from typing import Any


def initialize_runtime_timing_state(*, gui: Any, pygame: Any, os_module: Any, time_module: Any, threading_module: Any, logger: Any) -> None:
    """Initialize timing, display health, cursor/debug state, and watchdog."""
    def _safe_float_env(name: str, default: float) -> float:
        try:
            return float(os_module.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return float(default)

    def _safe_int_env(name: str, default: int) -> int:
        try:
            return int(os_module.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return int(default)

    gui._last_ungrab_attempt = 0.0

    gui._display_check_interval = _safe_float_env("KLTN_DISPLAY_CHECK_INTERVAL", 1.0)
    gui._display_check_last = time_module.time()
    gui._display_recovery_attempts = 0
    gui._display_recovery_attempts_limit = _safe_int_env("KLTN_DISPLAY_RECOVER_LIMIT", 3)

    gui.view_offset_x = 0
    gui.view_offset_y = 0
    gui.dragging = False
    gui.drag_start = (0, 0)
    gui.fullscreen = False

    gui.clock = pygame.time.Clock()
    gui.font = pygame.font.SysFont("Arial", 14, bold=True)
    gui.big_font = pygame.font.SysFont("Arial", 20, bold=True)
    gui.small_font = pygame.font.SysFont("Arial", 12)

    gui.debug_control_panel = os_module.environ.get("KLTN_DEBUG_CONTROL_PANEL", "0") == "1"
    gui.debug_panel_click_padding = _safe_int_env("KLTN_DEBUG_PANEL_PADDING", 40) if gui.debug_control_panel else 0

    gui.last_frame_time = time_module.time()
    gui.delta_time = 0.0
    gui._display_check_last = 0.0
    gui._display_check_interval = _safe_float_env("KLTN_DISPLAY_CHECK_INTERVAL", 1.0)

    try:
        import faulthandler
        faulthandler.enable(all_threads=True)

        gui._watchdog_enabled = os_module.environ.get("KLTN_ENABLE_WATCHDOG", "0") == "1"
        gui._watchdog_threshold = _safe_float_env("KLTN_WATCHDOG_THRESHOLD", 1.25)
        gui._watchdog_last_dump = 0.0
        gui._watchdog_dump_limit = _safe_int_env("KLTN_WATCHDOG_DUMP_LIMIT", 3)
        gui._watchdog_dumps = 0
        gui._watchdog_thread = None
        gui._watchdog_request_screenshot = None

        if gui._watchdog_enabled:
            def watchdog_start() -> None:
                try:
                    thread = threading_module.Thread(target=gui._watchdog_loop, daemon=True)
                    thread.start()
                    gui._watchdog_thread = thread
                    logger.debug("Watchdog thread started (threshold=%s s)", gui._watchdog_threshold)
                except (AttributeError, RuntimeError, ValueError, TypeError):
                    logger.exception("Failed to start watchdog thread")

            watchdog_start()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        gui._watchdog_enabled = False

    gui._consecutive_empty_frames = 0
    try:
        gui._empty_frame_recovery_threshold = _safe_int_env("KLTN_EMPTY_FRAME_RECOVERY", 8)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        gui._empty_frame_recovery_threshold = 8

