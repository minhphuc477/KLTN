"""Gameplay control action orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def reset_map(*, gui, reset_map_helper):
    reset_map_helper(gui)


def show_path_preview(*, gui, path_preview_dialog_cls, logger, show_path_preview_helper):
    show_path_preview_helper(gui, path_preview_dialog_cls, logger)


def clear_path(*, gui, clear_path_helper):
    clear_path_helper(gui)


def start_ai_dungeon_generation(*, gui, threading_module, start_ai_dungeon_generation_helper):
    start_ai_dungeon_generation_helper(gui, threading_module)


def run_ai_dungeon_generation_worker(*, gui, logger, run_ai_generation_worker_helper):
    run_ai_generation_worker_helper(gui, logger)
