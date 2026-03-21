"""Control-panel render and click orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def render_control_panel(*, gui, surface, pygame, logger, dropdown_widget_cls, render_control_panel_helper):
    render_control_panel_helper(
        gui,
        surface,
        pygame=pygame,
        logger=logger,
        dropdown_widget_cls=dropdown_widget_cls,
    )


def render_tooltips(*, gui, surface, mouse_pos, button_widget_cls, pygame, render_tooltips_helper):
    render_tooltips_helper(gui, surface, mouse_pos, button_widget_cls, pygame)


def draw_tooltip(*, gui, surface, pos, text, pygame, draw_tooltip_helper):
    draw_tooltip_helper(gui, surface, pos, text, pygame)


def handle_control_panel_click(*, gui, pos, button, event_type, logger, debug_input_active, dispatch_helper):
    return dispatch_helper(
        gui=gui,
        pos=pos,
        button=button,
        event_type=event_type,
        logger=logger,
        debug_input_active=debug_input_active,
    )


def control_panel_hit_rect(*, gui, pygame, control_panel_hit_rect_helper):
    return control_panel_hit_rect_helper(
        panel_rect=getattr(gui, "control_panel_rect", None),
        debug_control_panel=getattr(gui, "debug_control_panel", False),
        debug_panel_click_padding=getattr(gui, "debug_panel_click_padding", 0),
        rect_factory=pygame.Rect,
    )


def should_swallow_control_panel_click(*, gui, panel_hit_rect, pos, logger, should_swallow_control_panel_click_helper):
    return should_swallow_control_panel_click_helper(
        dragging=getattr(gui, "control_panel_scroll_dragging", False),
        ignore_click_until=getattr(gui, "control_panel_ignore_click_until", 0.0),
        panel_hit_rect=panel_hit_rect,
        pos=pos,
        logger=logger,
    )


def translate_control_panel_click(*, gui, pos, panel_hit_rect, translate_control_panel_click_helper):
    return translate_control_panel_click_helper(
        pos=pos,
        panel_hit_rect=panel_hit_rect,
        panel_rect=getattr(gui, "control_panel_rect", None),
        can_scroll=getattr(gui, "control_panel_can_scroll", False),
        control_panel_scroll=getattr(gui, "control_panel_scroll", 0),
    )


def handle_outside_control_panel_click(
    *,
    gui,
    panel_hit_rect,
    pos,
    button,
    dropdown_widget_cls,
    logger,
    handle_outside_control_panel_click_helper,
):
    return handle_outside_control_panel_click_helper(
        panel_hit_rect=panel_hit_rect,
        pos=pos,
        button=button,
        widget_manager=gui.widget_manager,
        dropdown_type=dropdown_widget_cls,
        logger=logger,
    )


def refresh_control_panel_layout_if_needed(
    *,
    gui,
    sc_pos,
    debug_input_active,
    logger,
    refresh_control_panel_layout_if_needed_helper,
):
    return refresh_control_panel_layout_if_needed_helper(
        widget_manager=gui.widget_manager,
        sc_pos=sc_pos,
        debug_input_active=debug_input_active,
        panel_rect=getattr(gui, "control_panel_rect", None),
        reposition_widgets=gui._reposition_widgets,
        logger=logger,
    )


def retry_control_panel_click_after_auto_scroll(
    *,
    gui,
    pos,
    sc_pos,
    button,
    handled,
    logger,
    retry_control_panel_click_after_auto_scroll_helper,
):
    handled, new_scroll, ignore_until = retry_control_panel_click_after_auto_scroll_helper(
        pos=pos,
        sc_pos=sc_pos,
        button=button,
        handled=handled,
        panel_rect=getattr(gui, "control_panel_rect", None),
        widget_manager=gui.widget_manager,
        can_scroll=getattr(gui, "control_panel_can_scroll", False),
        control_panel_scroll=getattr(gui, "control_panel_scroll", 0),
        control_panel_scroll_max=getattr(gui, "control_panel_scroll_max", 0),
        logger=logger,
    )
    gui.control_panel_scroll = new_scroll
    if ignore_until:
        gui.control_panel_ignore_click_until = ignore_until
    return handled


def apply_control_panel_widget_updates(*, gui, checkbox_widget_cls, logger, apply_control_panel_widget_updates_helper):
    apply_control_panel_widget_updates_helper(
        gui=gui,
        widget_manager=gui.widget_manager,
        checkbox_type=checkbox_widget_cls,
        logger=logger,
    )


def apply_checkbox_widget_update(*, gui, widget, logger, apply_checkbox_widget_update_helper):
    apply_checkbox_widget_update_helper(gui=gui, widget=widget, logger=logger)


def apply_dropdown_widget_update(*, gui, widget, logger, apply_dropdown_widget_update_helper):
    apply_dropdown_widget_update_helper(gui=gui, widget=widget, logger=logger)


def apply_algorithm_dropdown_update(*, gui, widget, logger, apply_algorithm_dropdown_update_helper):
    apply_algorithm_dropdown_update_helper(gui=gui, widget=widget, logger=logger)
