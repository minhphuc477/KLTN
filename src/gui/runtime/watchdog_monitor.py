"""Watchdog loop helper for UI stall diagnostics and optional recovery."""

from typing import Any


def watchdog_loop(gui: Any, logger: Any, os_module: Any, time_module: Any, tempfile_module: Any) -> None:
    """Background watchdog that writes stack traces and requests screenshots on stalls."""
    try:
        import faulthandler
    except Exception:
        logger.debug("faulthandler not available; watchdog disabled")
        return

    while getattr(gui, "_watchdog_enabled", False):
        try:
            time_module.sleep(0.5)
            last = getattr(gui, "last_frame_time", 0)
            now = time_module.time()
            if now - last <= getattr(gui, "_watchdog_threshold", 1.25):
                continue

            if gui._watchdog_dumps >= getattr(gui, "_watchdog_dump_limit", 3):
                continue

            gui._watchdog_dumps += 1
            gui._watchdog_last_dump = now

            ts = int(now)
            pid = os_module.getpid()
            tmpdir = tempfile_module.gettempdir()
            trace_path = os_module.path.join(tmpdir, f"zave_watchdog_trace_{pid}_{ts}.txt")

            try:
                with open(trace_path, "w") as trace_file:
                    trace_file.write(f"Watchdog dump: time={now} last_frame={last}\n")
                    faulthandler.dump_traceback(file=trace_file)
                logger.warning("Watchdog detected stall; stack dump written: %s", trace_path)
            except Exception:
                logger.exception("Failed writing watchdog stack dump")

            try:
                shot_path = os_module.path.join(tmpdir, f"zave_watchdog_shot_{pid}_{ts}.png")
                try:
                    gui._watchdog_request_screenshot = shot_path
                    logger.warning("Watchdog requested screenshot: %s", shot_path)
                except Exception:
                    logger.exception("Failed to set watchdog screenshot request")
            except Exception:
                logger.exception("Watchdog screenshot request failed")

            try:
                if os_module.environ.get("KLTN_WATCHDOG_TERMINATE_SOLVER") == "1":
                    proc = getattr(gui, "solver_proc", None)
                    if proc and proc.is_alive():
                        logger.warning("Watchdog terminating solver process pid=%s", getattr(proc, "pid", None))
                        try:
                            proc.terminate()
                        except Exception:
                            logger.exception("Failed to terminate solver process")
            except Exception:
                logger.exception("Watchdog failed to check/terminate solver process")

            try:
                gui._set_message(f"Watchdog: dumped trace ({os_module.path.basename(trace_path)})", 5.0)
            except Exception:
                pass
        except Exception:
            logger.exception("Uncaught exception in watchdog loop")

    logger.debug("Watchdog loop exiting")
