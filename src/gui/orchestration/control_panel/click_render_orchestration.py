"""Control-panel render and click orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.control_panel.click_dispatch import handle_control_panel_click as _dispatch_control_panel_click
from src.gui.control_panel.interactions import (
    control_panel_hit_rect as _control_panel_hit_rect,
    handle_outside_control_panel_click as _handle_outside_control_panel_click,
    refresh_control_panel_layout_if_needed as _refresh_control_panel_layout_if_needed,
    retry_control_panel_click_after_auto_scroll as _retry_control_panel_click_after_auto_scroll,
    should_swallow_control_panel_click as _should_swallow_control_panel_click,
    translate_control_panel_click as _translate_control_panel_click,
)
from src.gui.control_panel.updates import (
    apply_algorithm_dropdown_update as _apply_algorithm_dropdown_update,
    apply_checkbox_widget_update as _apply_checkbox_widget_update,
    apply_control_panel_widget_updates as _apply_control_panel_widget_updates,
    apply_dropdown_widget_update as _apply_dropdown_widget_update,
)
from src.gui.control_panel.view import render_control_panel as _render_control_panel
from src.gui.rendering.widget_tooltips import (
    draw_tooltip as _draw_tooltip,
    render_tooltips as _render_tooltips,
)


def render_control_panel(*, gui, surface, pygame, logger, dropdown_widget_cls):
    _render_control_panel(
        gui,
        surface,
        pygame=pygame,
        logger=logger,
        dropdown_widget_cls=dropdown_widget_cls,
    )


def render_tooltips(*, gui, surface, mouse_pos, button_widget_cls, pygame):
    _render_tooltips(gui, surface, mouse_pos, button_widget_cls, pygame)


def draw_tooltip(*, gui, surface, pos, text, pygame):
    _draw_tooltip(gui, surface, pos, text, pygame)


def handle_control_panel_click(*, gui, pos, button, event_type, logger, debug_input_active):
    return _dispatch_control_panel_click(
        gui=gui,
        pos=pos,
        button=button,
        event_type=event_type,
        logger=logger,
        debug_input_active=debug_input_active,
    )


def control_panel_hit_rect(*, gui, pygame):
    return _control_panel_hit_rect(
        panel_rect=getattr(gui, "control_panel_rect", None),
        debug_control_panel=getattr(gui, "debug_control_panel", False),
        debug_panel_click_padding=getattr(gui, "debug_panel_click_padding", 0),
        rect_factory=pygame.Rect,
    )


def should_swallow_control_panel_click(*, gui, panel_hit_rect, pos, logger):
    return _should_swallow_control_panel_click(
        dragging=getattr(gui, "control_panel_scroll_dragging", False),
        ignore_click_until=getattr(gui, "control_panel_ignore_click_until", 0.0),
        panel_hit_rect=panel_hit_rect,
        pos=pos,
        logger=logger,
    )


def translate_control_panel_click(*, gui, pos, panel_hit_rect):
    return _translate_control_panel_click(
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
):
    return _handle_outside_control_panel_click(
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
):
    return _refresh_control_panel_layout_if_needed(
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
):
    handled, new_scroll, ignore_until = _retry_control_panel_click_after_auto_scroll(
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


def apply_control_panel_widget_updates(*, gui, checkbox_widget_cls, logger):
    _apply_control_panel_widget_updates(
        gui=gui,
        widget_manager=gui.widget_manager,
        checkbox_type=checkbox_widget_cls,
        logger=logger,
    )


def apply_checkbox_widget_update(*, gui, widget, logger):
    _apply_checkbox_widget_update(gui=gui, widget=widget, logger=logger)


def apply_dropdown_widget_update(*, gui, widget, logger):
    _apply_dropdown_widget_update(gui=gui, widget=widget, logger=logger)


def apply_algorithm_dropdown_update(*, gui, widget, logger):
    _apply_algorithm_dropdown_update(gui=gui, widget=widget, logger=logger)
