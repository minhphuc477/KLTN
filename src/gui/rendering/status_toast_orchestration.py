"""Status/banner/toast orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def render_error_banner(*, gui, surface, pygame, time_module, render_error_banner_helper):
    render_error_banner_helper(gui, surface, pygame, time_module)


def render_solver_status_banner(*, gui, surface, pygame, math_module, time_module, logger, render_solver_status_banner_helper):
    render_solver_status_banner_helper(gui, surface, pygame, math_module, time_module, logger)


def render_status_bar(*, gui, surface, pygame, render_status_bar_helper):
    render_status_bar_helper(gui, surface, pygame)


def show_error(*, gui, message, logger, time_module, show_error_helper):
    return show_error_helper(gui, message, logger, time_module)


def show_message(*, gui, message, duration, logger, time_module, show_message_helper):
    return show_message_helper(gui, message, duration, logger, time_module)


def show_warning(*, gui, message, logger, show_warning_helper):
    show_warning_helper(gui, message, logger)


def set_message(*, gui, message, duration, time_module, set_message_helper):
    set_message_helper(gui, message, duration, time_module)


def show_toast(*, gui, message, duration, toast_type, toast_cls, show_toast_helper):
    show_toast_helper(gui, message, duration, toast_type, toast_cls)


def update_toasts(*, gui, update_toasts_helper):
    update_toasts_helper(gui)


def render_toasts(*, gui, surface, render_toasts_helper):
    render_toasts_helper(gui, surface)


def format_cbs_metrics_tooltip(*, cbs_metrics, format_cbs_metrics_tooltip_helper):
    return format_cbs_metrics_tooltip_helper(cbs_metrics)
