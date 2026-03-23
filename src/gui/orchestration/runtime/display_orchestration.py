"""Display and watchdog orchestration bridges for ZeldaGUI."""

from __future__ import annotations

import os
import tempfile
import time

from src.gui.runtime.display_diagnostics import (
    handle_watchdog_screenshot as _handle_watchdog_screenshot_helper,
    report_ui_state as _report_ui_state_helper,
)
from src.gui.runtime.display_lifecycle import (
    attempt_display_reinit as _attempt_display_reinit_helper,
    ensure_display_alive as _ensure_display_alive_helper,
    safe_set_mode as _safe_set_mode_helper,
)
from src.gui.runtime.watchdog_monitor import watchdog_loop as _watchdog_loop_helper
from src.gui.runtime.window_focus import (
    force_focus as _force_focus_helper,
    toggle_fullscreen as _toggle_fullscreen_helper,
)


def safe_set_mode(*, size, pygame, logger, flags=0, allow_fallback=True):
    return _safe_set_mode_helper(size, pygame, logger, flags=flags, allow_fallback=allow_fallback)


def attempt_display_reinit(*, gui, pygame, logger):
    return _attempt_display_reinit_helper(gui, pygame, logger)


def handle_watchdog_screenshot(*, gui, pygame, logger):
    return _handle_watchdog_screenshot_helper(gui, pygame, logger, os)


def report_ui_state(*, gui, logger):
    return _report_ui_state_helper(gui, logger)


def ensure_display_alive(*, gui, pygame, logger, force=False):
    return _ensure_display_alive_helper(gui, pygame, logger, force=force)


def force_focus(*, gui, pygame, logger):
    return _force_focus_helper(gui, pygame, logger, os)


def toggle_fullscreen(*, gui, pygame, logger, platform_module):
    return _toggle_fullscreen_helper(gui, pygame, logger, os, platform_module)


def watchdog_loop(*, gui, logger):
    _watchdog_loop_helper(
        gui=gui,
        logger=logger,
        os_module=os,
        time_module=time,
        tempfile_module=tempfile,
    )
    return
