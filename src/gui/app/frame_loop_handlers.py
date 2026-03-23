"""Frame-tail handlers for the gui_runner main loop."""

from __future__ import annotations


def update_heartbeat(gui, logger, time_module, heartbeat_last, heartbeat_interval):
    """Emit periodic heartbeat logs and return updated heartbeat timestamp."""
    try:
        now = time_module.time()
        if now - heartbeat_last > heartbeat_interval:
            heartbeat_last = now
            logger.debug(
                "GUI heartbeat - frame=%d auto_mode=%s solver_running=%s",
                getattr(gui, "frame_count", -1),
                getattr(gui, "auto_mode", False),
                getattr(gui, "solver_running", False),
            )
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        pass
    return heartbeat_last


def handle_pending_solver_trigger(gui, logger):
    """Start solver when deferred trigger flag is set."""
    should_start = False
    solver_lock = getattr(gui, "_solver_lock", None)
    if solver_lock is not None:
        with solver_lock:
            if getattr(gui, "_pending_solver_trigger", False):
                gui._pending_solver_trigger = False
                should_start = True
    elif getattr(gui, "_pending_solver_trigger", False):
        gui._pending_solver_trigger = False
        should_start = True

    if should_start:
        alg_name = gui._algorithm_name(gui.algorithm_idx)
        logger.info("Ã¢â€°Â¡Ã†â€™ÃƒÂ¶ÃƒÂ¤ Processing pending solver trigger: Starting %s solver...", alg_name)
        gui._start_auto_solve()


def render_and_present_frame(gui, pygame_module, logger):
    """Render then present frame, attempting fallback display recovery on failure."""
    gui._render()

    try:
        pygame_module.display.flip()
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("pygame.display.flip() failed; attempting pygame.display.update() and fallback")
        try:
            pygame_module.display.update()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("pygame.display.update() also failed")
            try:
                if not gui._ensure_display_alive():
                    logger.warning("Display not healthy after flip/update; attempted recovery")
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.exception("Attempted display recovery after flip/update failures")


def handle_watchdog_screenshot_request(gui, logger):
    """Process watchdog screenshot request safely on the main thread."""
    try:
        try:
            gui._handle_watchdog_screenshot()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("Error during watchdog screenshot handling")
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("Error handling watchdog screenshot request")


def run_periodic_display_health_check(gui, time_module, logger):
    """Perform throttled display health checks and notify on failed recovery."""
    try:
        now = time_module.time()
        if now - getattr(gui, "_display_check_last", 0.0) >= getattr(gui, "_display_check_interval", 1.0):
            gui._display_check_last = now
            ok = gui._ensure_display_alive()
            if not ok:
                gui._set_message("Display recovery attempted; see logs", 6.0)
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("Error during display health check")


def advance_frame_and_check_limit(frame_count, max_frames, logger):
    """Increment frame count and report whether run loop should continue."""
    frame_count += 1
    keep_running = True
    if max_frames is not None and frame_count >= max_frames:
        logger.debug("Exiting run loop due to max_frames=%r", max_frames)
        keep_running = False
    return frame_count, keep_running


def tick_frame_clock(gui):
    """Cap framerate with higher FPS while auto mode is active."""
    gui.clock.tick(60 if gui.auto_mode else 30)
