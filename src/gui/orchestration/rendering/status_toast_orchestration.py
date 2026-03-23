"""Status/banner/toast orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.rendering.status_display import (
    render_error_banner as _render_error_banner_helper,
    render_solver_status_banner as _render_solver_status_banner_helper,
    render_status_bar as _render_status_bar_helper,
    show_error as _show_error_helper,
    show_message as _show_message_helper,
    show_warning as _show_warning_helper,
)
from src.gui.runtime.toast_messages import (
    render_toasts as _render_toasts_helper,
    set_message as _set_message_helper,
    show_toast as _show_toast_helper,
    update_toasts as _update_toasts_helper,
)
from src.gui.solver.metrics_tooltips import (
    format_cbs_metrics_tooltip as _format_cbs_metrics_tooltip_helper,
)


def render_error_banner(*, gui, surface, pygame, time_module):
    _render_error_banner_helper(gui, surface, pygame, time_module)


def render_solver_status_banner(*, gui, surface, pygame, math_module, time_module, logger):
    _render_solver_status_banner_helper(gui, surface, pygame, math_module, time_module, logger)


def render_status_bar(*, gui, surface, pygame):
    _render_status_bar_helper(gui, surface, pygame)


def show_error(*, gui, message, logger, time_module):
    return _show_error_helper(gui, message, logger, time_module)


def show_message(*, gui, message, duration, logger, time_module):
    return _show_message_helper(gui, message, duration, logger, time_module)


def show_warning(*, gui, message, logger):
    _show_warning_helper(gui, message, logger)


def set_message(*, gui, message, duration, time_module):
    _set_message_helper(gui, message, duration, time_module)


def show_toast(*, gui, message, duration, toast_type, toast_cls):
    _show_toast_helper(gui, message, duration, toast_type, toast_cls)


def update_toasts(*, gui):
    _update_toasts_helper(gui)


def render_toasts(*, gui, surface):
    _render_toasts_helper(gui, surface)


def format_cbs_metrics_tooltip(*, cbs_metrics):
    return _format_cbs_metrics_tooltip_helper(cbs_metrics)
