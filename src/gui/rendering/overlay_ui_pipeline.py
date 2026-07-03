"""Overlay UI render helpers extracted from gui_runner._render."""

from __future__ import annotations

from typing import Any

from src.gui.rendering.font_cache import get_sys_font


def render_translucent_event_overlays(
    *,
    gui: Any,
    view_w: int,
    view_h: int,
    pygame: Any,
    logger: Any,
) -> None:
    """Render translucent overlays that communicate active click-capturing states."""
    try:
        if getattr(gui, "preview_overlay_visible", False):
            try:
                logger.debug("Rendering preview overlay (will capture clicks)")
                ov = pygame.Surface((view_w, view_h), pygame.SRCALPHA)
                ov.fill((40, 30, 40, 130))
                gui.screen.blit(ov, (0, 0))
                label = gui.big_font.render("PATH PREVIEW (overlay) - captures clicks", True, (255, 220, 120))
                gui.screen.blit(label, (20, view_h // 2 - 20))
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass

        if getattr(gui, "show_solver_comparison_overlay", False):
            try:
                logger.debug("Rendering solver comparison modal (captures clicks)")
                ov2 = pygame.Surface((view_w, view_h), pygame.SRCALPHA)
                ov2.fill((20, 20, 20, 180))
                gui.screen.blit(ov2, (0, 0))
                label2 = gui.big_font.render("SOLVER COMPARISON - modal", True, (200, 200, 255))
                gui.screen.blit(label2, (20, view_h // 2 - 20))
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass


def render_preview_layer(
    *,
    gui: Any,
    pygame: Any,
    logger: Any,
) -> None:
    """Render path preview dialog and non-modal sidebar preview panel."""
    if gui.path_preview_mode and gui.path_preview_dialog:
        try:
            gui.path_preview_dialog.render_path_overlay(
                gui.screen,
                gui.TILE_SIZE,
                gui.view_offset_x,
                gui.view_offset_y,
                gui.SIDEBAR_WIDTH,
                gui.HUD_HEIGHT,
            )
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Failed to render path overlay: %s", exc)

        try:
            gui.path_preview_dialog.render(gui.screen)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Failed to render path preview dialog: %s", exc)
        return

    if getattr(gui, "preview_overlay_visible", False) and getattr(gui, "path_preview_dialog", None):
        try:
            gui.path_preview_dialog.render_path_overlay(
                gui.screen,
                gui.TILE_SIZE,
                gui.view_offset_x,
                gui.view_offset_y,
                gui.SIDEBAR_WIDTH,
                gui.HUD_HEIGHT,
            )
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Failed to render path overlay (non-modal): %s", exc)

        try:
            sidebar_x = gui.screen_w - gui.SIDEBAR_WIDTH
            box_h = 80
            box_y = 120
            box_rect = pygame.Rect(sidebar_x + 10, box_y, gui.SIDEBAR_WIDTH - 20, box_h)
            pygame.draw.rect(gui.screen, (40, 40, 60), box_rect)
            pygame.draw.rect(gui.screen, (100, 150, 255), box_rect, 2)

            font = get_sys_font(pygame, "Arial", 14, bold=True)
            small = get_sys_font(pygame, "Arial", 12)
            path_len = len(gui.auto_path) if getattr(gui, "auto_path", None) else 0
            text1 = font.render(f"Preview: {path_len} steps", True, (200, 200, 255))
            gui.screen.blit(text1, (box_rect.x + 8, box_rect.y + 8))

            keys_used = getattr(gui, "solver_result", {}).get("keys_used", 0) if getattr(gui, "solver_result", None) else 0
            keys_avail = getattr(gui, "solver_result", {}).get("keys_available", 0) if getattr(gui, "solver_result", None) else 0
            keys_text = f"Keys: {keys_used} / {keys_avail}" if keys_avail > 0 else "Keys: None"
            gui.screen.blit(small.render(keys_text, True, (200, 200, 200)), (box_rect.x + 8, box_rect.y + 34))

            start_rect = pygame.Rect(box_rect.x + 8, box_rect.y + 48, 140, 24)
            dismiss_rect = pygame.Rect(box_rect.x + 156, box_rect.y + 48, 60, 24)
            pygame.draw.rect(gui.screen, (40, 140, 40), start_rect)
            pygame.draw.rect(gui.screen, (140, 40, 40), dismiss_rect)
            pygame.draw.rect(gui.screen, (100, 255, 100), start_rect, 1)
            pygame.draw.rect(gui.screen, (255, 100, 100), dismiss_rect, 1)
            gui.sidebar_start_button_rect = start_rect
            gui.sidebar_dismiss_button_rect = dismiss_rect

            start_text = small.render("Solve Level", True, (255, 255, 255))
            dismiss_text = small.render("Dismiss", True, (255, 255, 255))
            gui.screen.blit(start_text, (start_rect.x + 8, start_rect.y + 4))
            gui.screen.blit(dismiss_text, (dismiss_rect.x + 6, dismiss_rect.y + 4))
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Failed to render sidebar preview box: %s", exc)
        return

    gui.sidebar_start_button_rect = None
    gui.sidebar_dismiss_button_rect = None

