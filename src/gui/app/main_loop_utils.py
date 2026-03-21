"""Utilities for the GUI main loop extracted from gui_runner."""

from __future__ import annotations


def resolve_test_mode_max_frames(max_frames, env):
    """Return bounded max_frames when running in test/CI mode."""
    if max_frames is not None:
        return max_frames
    if env.get("KLTN_TEST_MODE") or env.get("PYTEST_CURRENT_TEST") or env.get("CI"):
        try:
            return int(env.get("KLTN_RUN_MAX_FRAMES", "10"))
        except Exception:
            return 10
    return None


def should_attempt_focus_fallback(fullscreen, focused, now_ts, last_attempt_ts, cooldown_sec=2.0):
    """Determine whether input focus fallback should run this frame."""
    if fullscreen:
        return False
    if focused:
        return False
    if (now_ts - float(last_attempt_ts or 0.0)) <= float(cooldown_sec):
        return False
    return True
