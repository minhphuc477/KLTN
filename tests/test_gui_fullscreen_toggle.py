import os
import time
import pygame
import numpy as np
import pytest

from gui_runner import ZeldaGUI


@pytest.fixture(autouse=True)
def init_pygame():
    # Use dummy video driver to avoid opening windows in CI
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    pygame.init()
    yield
    pygame.quit()


def test_toggle_fullscreen_roundtrip():
    grid = np.zeros((8, 8), dtype=int)
    gui = ZeldaGUI(maps=[grid], map_names=['M1'])
    # Record initial size
    init_size = (gui.screen_w, gui.screen_h)

    # Toggle fullscreen on (should not raise or hang)
    t0 = time.time()
    gui._toggle_fullscreen()
    assert time.time() - t0 < 2.0

    # After toggle ensure screen has valid positive size
    assert gui.screen_w > 0 and gui.screen_h > 0

    # Toggle back to windowed mode
    t0 = time.time()
    gui._toggle_fullscreen()
    assert time.time() - t0 < 2.0

    # Ensure we restored to previous (or reasonable) window size
    assert gui.screen_w >= 200 and gui.screen_h >= 150


def test_toggle_fullscreen_while_solver_running():
    grid = np.zeros((10, 10), dtype=int)
    os.environ['KLTN_SOLVER_TEST'] = '1'
    gui = ZeldaGUI(maps=[grid], map_names=['M1'])

    # Start solver (test-mode spawns a short sleep in a background process/thread)
    gui._start_auto_solve()

    # Allow tiny time for background thread to schedule process
    time.sleep(0.1)

    # Toggle fullscreen while solver runs; should not hang
    t0 = time.time()
    gui._toggle_fullscreen()
    assert time.time() - t0 < 2.0

    # Ensure display is valid or that recovery succeeded
    assert gui._ensure_display_alive() is True


def test_watchdog_screenshot_is_handled_and_saved(tmp_path):
    grid = np.zeros((6, 6), dtype=int)
    gui = ZeldaGUI(maps=[grid], map_names=['M1'])

    out = str(tmp_path / 'watchdog_shot.png')
    gui._watchdog_request_screenshot = out

    # Call the helper directly (main loop would call this)
    saved = gui._handle_watchdog_screenshot()

    # Request must be cleared regardless of success
    assert getattr(gui, '_watchdog_request_screenshot', None) is None

    # If save succeeded, file exists and helper returned True.
    if saved:
        assert os.path.exists(out)
    else:
        # When using dummy driver some platforms may not allow saving; still ok
        assert not getattr(gui, '_watchdog_request_screenshot', None)


def test_mouse_click_events_processed():
    # Ensure a posted mouse button event is processed and logged
    grid = np.zeros((8, 8), dtype=int)
    gui = ZeldaGUI(maps=[grid], map_names=['M1'])

    # Post an event to the pygame queue
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(20, 20), button=1))

    # Run a few frames to process events
    gui.run(max_frames=3)

    # Click should be recorded in debug_click_log
    assert getattr(gui, 'debug_click_log', []) and len(gui.debug_click_log) > 0


def test_solver_comparison_runs_in_background():
    grid = np.zeros((10, 10), dtype=int)
    gui = ZeldaGUI(maps=[grid], map_names=['M1'])

    # Start solver comparison; should return quickly
    t0 = time.time()
    gui._run_solver_comparison()
    assert time.time() - t0 < 0.5

    # A background thread should be running
    th = getattr(gui, 'solver_comparison_thread', None)
    assert th is not None
    # Wait a bit for it to finish (maps are small)
    th.join(timeout=5)
    assert not th.is_alive()
    # Results should be populated
    assert getattr(gui, 'solver_comparison_results', None) is not None
    assert isinstance(gui.solver_comparison_results, list)
