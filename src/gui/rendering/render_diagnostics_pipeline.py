"""Render diagnostics and recovery helpers extracted from gui_runner._render."""

from __future__ import annotations

from typing import Any

from src.gui.rendering.font_cache import get_sys_font


def handle_empty_frame_recovery(
    *,
    gui: Any,
    map_surface: Any,
    view_w: int,
    view_h: int,
    tiles_drawn: int,
    pygame: Any,
    logger: Any,
) -> None:
    """Render diagnostics and attempt recovery when no map tiles are visible."""
    if tiles_drawn == 0:
        try:
            if not getattr(gui, "_auto_recenter_done", False):
                logger.info("No tiles drawn - attempting auto-fit zoom + center")
                try:
                    gui._auto_fit_zoom()
                    gui._center_view()
                except (AttributeError, RuntimeError, ValueError, TypeError):
                    pass
                gui._auto_recenter_done = True

            diag_font = get_sys_font(pygame, "Arial", 18, bold=True)
            diag_text = diag_font.render("No map tiles visible - check zoom/offset", True, (255, 100, 100))
            tx = max(10, (view_w - diag_text.get_width()) // 2)
            ty = max(10, (view_h - diag_text.get_height()) // 2)

            box = pygame.Surface((diag_text.get_width() + 20, diag_text.get_height() + 18), pygame.SRCALPHA)
            box.fill((30, 10, 10, 200))
            map_surface.blit(box, (tx - 10, ty - 9))
            map_surface.blit(diag_text, (tx, ty))

            small = get_sys_font(pygame, "Arial", 12)
            try:
                map_w = gui.env.width if gui.env is not None else 0
                map_h = gui.env.height if gui.env is not None else 0
            except (AttributeError, RuntimeError, ValueError, TypeError):
                map_w = map_h = 0
            diag2 = small.render(
                f"Tile: {gui.TILE_SIZE}px  ViewOffset: ({gui.view_offset_x},{gui.view_offset_y})",
                True,
                (220, 220, 220),
            )
            diag3 = small.render(f"Map: {map_w}x{map_h}  View: {view_w}x{view_h}", True, (200, 200, 200))
            map_surface.blit(diag2, (10, ty + diag_text.get_height() + 8))
            map_surface.blit(diag3, (10, ty + diag_text.get_height() + 24))
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        try:
            gui._consecutive_empty_frames = getattr(gui, "_consecutive_empty_frames", 0) + 1
            if gui._consecutive_empty_frames >= getattr(gui, "_empty_frame_recovery_threshold", 8):
                logger.warning(
                    "Detected %d consecutive empty frames - attempting display reinit",
                    gui._consecutive_empty_frames,
                )
                try:
                    recovered = gui._attempt_display_reinit()
                    if recovered:
                        gui._show_toast("Recovered display after blank frames", 3.0, "success")
                        logger.info("Recovered display after empty-frame sequence")
                    else:
                        gui._show_toast("Display recovery failed", 4.0, "error")
                except (AttributeError, RuntimeError, ValueError, TypeError):
                    logger.exception("Error during forced display reinit")
                finally:
                    gui._consecutive_empty_frames = 0
        except (AttributeError, RuntimeError, ValueError, TypeError):
            logger.exception("Failed handling consecutive empty frames counter")
        return

    try:
        gui._consecutive_empty_frames = 0
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass

