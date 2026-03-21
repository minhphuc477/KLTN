"""Preview overlay and preview dialog interaction handlers."""

from __future__ import annotations


def handle_preview_overlay_events(gui, event, pygame_module):
    """Handle preview overlay and path preview dialog interactions; returns True when consumed."""
    if getattr(gui, "preview_overlay_visible", False) and (
        gui.path_preview_dialog or getattr(gui, "auto_path", None)
    ):
        if event.type == pygame_module.KEYDOWN:
            if event.key == pygame_module.K_ESCAPE:
                gui.preview_overlay_visible = False
                gui.path_preview_dialog = None
                gui.message = "Path preview dismissed"
                return True
            if event.key == pygame_module.K_RETURN or event.key == pygame_module.K_SPACE:
                gui._execute_auto_solve_from_preview()
                return True

    if getattr(gui, "path_preview_mode", False) and getattr(gui, "path_preview_dialog", None):
        result = gui.path_preview_dialog.handle_input(event)
        if result == "start":
            gui._execute_auto_solve_from_preview()
            return True
        if result == "cancel":
            gui.path_preview_mode = False
            gui.preview_overlay_visible = True
            gui.message = "Path preview closed; overlay visible in sidebar/map (Enter to start or Esc to dismiss)"
            return True

    if (
        event.type == pygame_module.KEYDOWN
        and event.key == pygame_module.K_ESCAPE
        and getattr(gui, "show_solver_comparison_overlay", False)
    ):
        gui.show_solver_comparison_overlay = False
        gui._set_message("Solver comparison closed", 1.2)
        return True

    if (
        getattr(gui, "preview_overlay_visible", False)
        and event.type == pygame_module.MOUSEBUTTONDOWN
        and event.button == 1
    ):
        mouse_pos = event.pos
        if getattr(gui, "sidebar_start_button_rect", None) and gui.sidebar_start_button_rect.collidepoint(mouse_pos):
            gui._execute_auto_solve_from_preview()
            return True
        if getattr(gui, "sidebar_dismiss_button_rect", None) and gui.sidebar_dismiss_button_rect.collidepoint(mouse_pos):
            gui.preview_overlay_visible = False
            gui.path_preview_dialog = None
            gui.message = "Path preview dismissed"
            return True

    return False