"""Control-panel layout and widget state orchestration bridges."""

from __future__ import annotations

from src.gui.control_panel.view import (
    dump_control_panel_widget_state as _dump_control_panel_widget_state,
    reposition_widgets as _reposition_widgets,
    update_control_panel_positions as _update_control_panel_positions,
)


def update_control_panel_positions(
    *,
    gui,
    pygame,
    logger,
    widgets_available,
    checkbox_widget_cls,
    dropdown_widget_cls,
    button_widget_cls,
    zoom_labels,
    difficulty_names,
    algorithm_names,
):
    _update_control_panel_positions(
        gui,
        pygame,
        logger,
        widgets_available=widgets_available,
        checkbox_widget_cls=checkbox_widget_cls,
        dropdown_widget_cls=dropdown_widget_cls,
        button_widget_cls=button_widget_cls,
        zoom_labels=zoom_labels,
        difficulty_names=difficulty_names,
        algorithm_names=algorithm_names,
    )


def reposition_widgets(*, gui, panel_x, panel_y, checkbox_widget_cls, dropdown_widget_cls, button_widget_cls):
    _reposition_widgets(
        gui,
        panel_x,
        panel_y,
        checkbox_widget_cls=checkbox_widget_cls,
        dropdown_widget_cls=dropdown_widget_cls,
        button_widget_cls=button_widget_cls,
    )


def dump_control_panel_widget_state(*, gui, mouse_pos, logger, debug_input_active):
    _dump_control_panel_widget_state(
        gui,
        mouse_pos,
        logger=logger,
        debug_input_active=debug_input_active,
    )
