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


def compute_solver_timeout_seconds(active_alg, grid_cell_count=None, env_getter=None):
    """Return timeout budget for solver worker based on algorithm and map scale."""
    if active_alg == 0:
        timeout_sec = 60.0
    elif active_alg in (1, 2):
        timeout_sec = 180.0
    elif active_alg == 3:
        timeout_sec = 90.0
    elif active_alg == 4:
        timeout_sec = 120.0
    else:
        timeout_sec = 240.0

    if grid_cell_count is not None:
        try:
            baseline_cells = 16 * 11 * 8  # ~8-room baseline
            ratio = float(grid_cell_count) / float(max(1, baseline_cells))
            scale = max(1.0, min(3.0, ratio))
            timeout_sec *= scale
        except Exception:
            pass

    if env_getter is not None:
        try:
            return float(env_getter("KLTN_SOLVER_TIMEOUT", str(timeout_sec)))
        except Exception:
            return float(timeout_sec)
    return float(timeout_sec)


def find_path_tile_violations(path, grid, blocked_tile_ids):
    """Return list of (index, row, col, tile_id) entries for invalid path cells."""
    violations = []
    if not path:
        return violations

    blocked = set(blocked_tile_ids or [])
    for idx, point in enumerate(path):
        try:
            r, c = point
            tile_id = int(grid[r, c])
        except Exception:
            continue
        if tile_id in blocked:
            violations.append((idx, int(r), int(c), tile_id))
    return violations
