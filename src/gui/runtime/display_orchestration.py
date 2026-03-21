"""Display and watchdog orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def safe_set_mode(*, size, pygame, logger, safe_set_mode_helper, flags=0, allow_fallback=True):
    return safe_set_mode_helper(size, pygame, logger, flags=flags, allow_fallback=allow_fallback)


def attempt_display_reinit(*, gui, pygame, logger, attempt_display_reinit_helper):
    return attempt_display_reinit_helper(gui, pygame, logger)


def handle_watchdog_screenshot(*, gui, pygame, logger, os_module, handle_watchdog_screenshot_helper):
    return handle_watchdog_screenshot_helper(gui, pygame, logger, os_module)


def report_ui_state(*, gui, logger, report_ui_state_helper):
    return report_ui_state_helper(gui, logger)


def ensure_display_alive(*, gui, pygame, logger, ensure_display_alive_helper, force=False):
    return ensure_display_alive_helper(gui, pygame, logger, force=force)


def force_focus(*, gui, force_focus_helper):
    return force_focus_helper(gui)


def toggle_fullscreen(*, gui, pygame, logger, os_module, platform_module, toggle_fullscreen_helper):
    return toggle_fullscreen_helper(gui, pygame, logger, os_module, platform_module)


def watchdog_loop(*, gui, logger, os_module, time_module, tempfile_module, watchdog_loop_helper):
    watchdog_loop_helper(
        gui=gui,
        logger=logger,
        os_module=os_module,
        time_module=time_module,
        tempfile_module=tempfile_module,
    )
    return
