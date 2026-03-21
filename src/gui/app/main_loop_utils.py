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


def run_auto_step_tick(gui, logger, frame_count):
    """Advance auto-solve timer and execute one step when the threshold is met."""
    if frame_count % 60 == 0:
        logger.debug(
            "DEBUG_ANIM: frame=%d auto_mode=%s env.done=%s auto_path_len=%d auto_step_idx=%s",
            frame_count,
            getattr(gui, "auto_mode", None),
            getattr(gui.env, "done", None) if gui.env else None,
            len(getattr(gui, "auto_path", []) or []),
            getattr(gui, "auto_step_idx", None),
        )

    if not (getattr(gui, "auto_mode", False) and not getattr(gui.env, "done", False)):
        return

    gui.auto_step_timer += gui.delta_time
    effective_interval = gui.auto_step_interval / max(0.1, gui.speed_multiplier)

    if frame_count % 30 == 0:
        logger.debug(
            "TIMER_DIAG: frame=%d timer=%.3f threshold=%.3f delta=%.3f step=%d/%d",
            frame_count,
            gui.auto_step_timer,
            effective_interval,
            gui.delta_time,
            gui.auto_step_idx,
            len(gui.auto_path) if gui.auto_path else 0,
        )

    if gui.auto_step_timer < effective_interval:
        return

    gui.auto_step_timer = 0.0
    logger.debug(
        "DEBUG_ANIM: Calling _auto_step() at frame=%d, step_idx=%d/%d",
        frame_count,
        gui.auto_step_idx,
        len(gui.auto_path) if gui.auto_path else 0,
    )
    gui._auto_step()
    gui._center_on_player()


def run_continuous_movement_tick(gui, pygame_module, action_enum):
    """Handle hold-to-move input including diagonal movement combinations."""
    if getattr(gui, "auto_mode", False):
        return
    if not any(gui.keys_held.values()):
        return

    gui.move_timer += gui.delta_time
    if gui.move_timer < gui.move_delay:
        return

    gui.move_timer = 0.0
    if gui.keys_held[pygame_module.K_UP] and gui.keys_held[pygame_module.K_LEFT]:
        gui._manual_step(action_enum.UP_LEFT)
    elif gui.keys_held[pygame_module.K_UP] and gui.keys_held[pygame_module.K_RIGHT]:
        gui._manual_step(action_enum.UP_RIGHT)
    elif gui.keys_held[pygame_module.K_DOWN] and gui.keys_held[pygame_module.K_LEFT]:
        gui._manual_step(action_enum.DOWN_LEFT)
    elif gui.keys_held[pygame_module.K_DOWN] and gui.keys_held[pygame_module.K_RIGHT]:
        gui._manual_step(action_enum.DOWN_RIGHT)
    elif gui.keys_held[pygame_module.K_UP]:
        gui._manual_step(action_enum.UP)
    elif gui.keys_held[pygame_module.K_DOWN]:
        gui._manual_step(action_enum.DOWN)
    elif gui.keys_held[pygame_module.K_LEFT]:
        gui._manual_step(action_enum.LEFT)
    elif gui.keys_held[pygame_module.K_RIGHT]:
        gui._manual_step(action_enum.RIGHT)
