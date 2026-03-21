"""Control-panel layout and widget state orchestration bridges."""

from __future__ import annotations


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
    update_control_panel_positions_helper,
):
    update_control_panel_positions_helper(
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


def reposition_widgets(*, gui, panel_x, panel_y, checkbox_widget_cls, dropdown_widget_cls, button_widget_cls, reposition_widgets_helper):
    reposition_widgets_helper(
        gui,
        panel_x,
        panel_y,
        checkbox_widget_cls=checkbox_widget_cls,
        dropdown_widget_cls=dropdown_widget_cls,
        button_widget_cls=button_widget_cls,
    )


def dump_control_panel_widget_state(*, gui, mouse_pos, logger, debug_input_active, dump_control_panel_widget_state_helper):
    dump_control_panel_widget_state_helper(
        gui,
        mouse_pos,
        logger=logger,
        debug_input_active=debug_input_active,
    )
