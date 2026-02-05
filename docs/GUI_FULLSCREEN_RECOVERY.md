# Fullscreen Recovery & Watchdog Improvements (QA Checklist)

Summary
-------
This change adds a robust display reinitialization path and self-healing checks to
prevent fullscreen blackouts and UI hangs. It also ensures watchdog screenshot
requests are handled on the main thread, and adds tests validating the behavior.

What changed
------------
- `gui_runner.ZeldaGUI` now includes:
  - `_safe_set_mode(size, flags)`: resilient wrapper for `pygame.display.set_mode`.
  - `_attempt_display_reinit()`: attempts full `pygame.display.quit()`/`init()` and re-creates the screen.
  - `_ensure_display_alive()`: throttled health check used by the main loop.
  - Improved `_toggle_fullscreen()` to use `_safe_set_mode()` and perform immediate redraws.
  - User-visible toast messages on fallback and recovery.
- Tests added/extended in `tests/test_gui_fullscreen_toggle.py`:
  - `test_toggle_fullscreen_roundtrip()` (existing) kept and validated
  - `test_toggle_fullscreen_while_solver_running()` verifies toggling while a background solver (test-mode) is running
  - `test_watchdog_screenshot_is_handled_and_saved()` simulates a watchdog screenshot request and verifies main thread saves it.

QA Checklist ✅
--------------
1. Run the new tests locally:
   - `pytest tests/test_gui_fullscreen_toggle.py -q`
2. Manual Windows verification:
   - Open the GUI (`python gui_runner.py`) on a Windows desktop.
   - Press `F11` to enter fullscreen. Verify the screen is not black and the UI is responsive.
   - Trigger a background solver test:
     - `set KLTN_SOLVER_TEST=1` and press `SPACE` to start the solver, then press `F11` while solver runs.
     - Verify UI remains responsive and solver completes/terminates cleanly.
   - Trigger watchdog screenshot:
     - `set KLTN_ENABLE_WATCHDOG=1` and `set KLTN_WATCHDOG_THRESHOLD=0.5` to shorten timeouts.
     - Cause main loop to stall artificially (e.g., start heavy operation), and confirm a screenshot file is created in temp dir and a toast message appears.
3. Failure modes:
   - If fullscreen cannot be initialized, a toast and message will notify the user and the app will remain in a windowed state.
   - Recoveries are logged under the application logger (INFO/WARNING/ERROR).

Notes & Alternatives
--------------------
- Safer alternative: detect platform and prefer `pygame.FULLSCREEN_DESKTOP` when available; this implementation prefers the conservative behavior compatible across SDL backends.
- The reinit sequence can be adjusted (longer waits, additional retries) if specific drivers require it.

Small cleanup manifest
----------------------
No code paths were removed. Minor refactorings were added as small helper methods.

If you'd like, I can open a draft PR with these changes and include this QA checklist as part of the PR description.
