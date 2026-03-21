"""Window focus and fullscreen toggle helpers."""

from typing import Any


def force_focus(gui: Any, pygame: Any, logger: Any, os_module: Any) -> bool:
    """Try to bring window to foreground on Windows; no-op on other platforms."""
    if os_module.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = pygame.display.get_wm_info().get("window")
        if not hwnd:
            logger.debug("No hwnd available for focus")
            return False

        fg = user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        fg_tid = user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
        cur_tid = kernel32.GetCurrentThreadId()

        attached = False
        try:
            attached = bool(user32.AttachThreadInput(fg_tid, cur_tid, True))
        except Exception:
            attached = False

        SW_SHOW = 5
        user32.ShowWindow(hwnd, SW_SHOW)
        try:
            user32.SetForegroundWindow(hwnd)
        except Exception:
            logger.debug("SetForegroundWindow failed; continuing")
        try:
            user32.BringWindowToTop(hwnd)
        except Exception:
            pass

        try:
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            HWND_TOPMOST = -1
            HWND_NOTOPMOST = -2
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
            user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
        except Exception:
            pass

        try:
            if attached:
                user32.AttachThreadInput(fg_tid, cur_tid, False)
        except Exception:
            pass

        pygame.event.pump()
        logger.debug("Attempted Win32 force-focus sequence")
        return True
    except Exception:
        logger.exception("Win32 focus helper failed")
        return False


def toggle_fullscreen(gui: Any, pygame: Any, logger: Any, os_module: Any, platform_module: Any) -> None:
    """Toggle fullscreen mode with robust fallback and redraw behavior."""
    gui.fullscreen = not gui.fullscreen
    try:
        if gui.fullscreen:
            try:
                gui._prev_window_size = (gui.screen_w, gui.screen_h)
            except Exception:
                gui._prev_window_size = getattr(gui, "_prev_window_size", (800, 600))

            try:
                disp = pygame.display.Info()
                new_size = (int(disp.current_w), int(disp.current_h))
            except Exception:
                new_size = (0, 0)

            flags = pygame.FULLSCREEN | getattr(pygame, "HWSURFACE", 0) | getattr(pygame, "DOUBLEBUF", 0)
            screen = gui._safe_set_mode(new_size, flags)
            if not screen:
                gui.fullscreen = False
                gui._set_message("Failed to enter fullscreen; reverted to windowed", 4.0)
                gui._show_toast("Fullscreen failed; using windowed mode", 4.0, "warning")
                return

            gui.screen = screen
            try:
                gui.screen_w, gui.screen_h = gui.screen.get_size()
            except Exception:
                try:
                    gui.screen_w, gui.screen_h = new_size
                except Exception:
                    gui.screen_w, gui.screen_h = (800, 600)

            try:
                default_grab = "0" if platform_module.system().lower().startswith("win") else "1"
                if os_module.environ.get("KLTN_FULLSCREEN_GRAB", default_grab) == "1":
                    pygame.event.set_grab(True)
                else:
                    logger.debug("KLTN_FULLSCREEN_GRAB=0 or platform indicates no grab; skipping event grab")
            except Exception:
                logger.debug("Could not set event grab for fullscreen")

            try:
                pygame.event.pump()
                gui._load_assets()
                gui._render()
                pygame.display.flip()
                gui._show_toast("Entered fullscreen", 2.5, "success")
            except Exception:
                logger.exception("Post-fullscreen redraw failed")
            return

        prev = getattr(gui, "_prev_window_size", (800, 600))
        screen = gui._safe_set_mode(prev, pygame.RESIZABLE)
        if not screen:
            logger.exception("Failed to restore windowed mode; falling back to 800x600")
            gui.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
            gui.screen_w, gui.screen_h = gui.screen.get_size()
            gui._show_toast("Restored to windowed (fallback)", 3.0, "warning")
            return

        gui.screen = screen
        gui.screen_w, gui.screen_h = gui.screen.get_size()
        try:
            pygame.event.set_grab(False)
        except Exception:
            logger.debug("Could not clear event grab on exiting fullscreen")
        try:
            pygame.event.pump()
            gui._load_assets()
            gui._render()
            pygame.display.flip()
            gui._show_toast("Exited fullscreen", 2.0, "info")
        except Exception:
            logger.exception("Post-windowed redraw failed")
    except Exception:
        logger.exception("Unhandled exception in _toggle_fullscreen")
        try:
            prev = getattr(gui, "_prev_window_size", (800, 600))
            gui.screen = pygame.display.set_mode(prev, pygame.RESIZABLE)
            gui.screen_w, gui.screen_h = gui.screen.get_size()
        except Exception:
            logger.exception("Failed to revert to previous window mode after fullscreen error")
            try:
                gui.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
                gui.screen_w, gui.screen_h = gui.screen.get_size()
            except Exception:
                gui.screen_w, gui.screen_h = (800, 600)

        try:
            pygame.event.pump()
        except Exception:
            pass

        try:
            gui._load_assets()
        except Exception:
            logger.exception("Failed to reload assets after fullscreen toggle")
        try:
            if gui.control_panel_enabled:
                gui._update_control_panel_positions()
        except Exception:
            pass

        try:
            gui._center_view()
            gui._render()
            pygame.display.flip()
        except Exception:
            logger.exception("Failed to render after fullscreen toggle")

        gui._set_message(f"Fullscreen: {'ON' if gui.fullscreen else 'OFF'}", 1.5)
