"""Gameplay control action orchestration bridges for ZeldaGUI."""

from __future__ import annotations

import threading

from src.gui.ai.generation_controls import start_ai_dungeon_generation as _start_ai_dungeon_generation
from src.gui.ai.generation_worker import run_ai_generation_worker as _run_ai_generation_worker
from src.gui.gameplay.path_controls import clear_path as _clear_path, reset_map as _reset_map, show_path_preview as _show_path_preview


def reset_map(*, gui):
    _reset_map(gui)


def show_path_preview(*, gui, path_preview_dialog_cls, logger):
    _show_path_preview(gui, path_preview_dialog_cls, logger)


def clear_path(*, gui):
    _clear_path(gui)


def start_ai_dungeon_generation(*, gui):
    _start_ai_dungeon_generation(gui, threading)


def run_ai_dungeon_generation_worker(*, gui, logger):
    _run_ai_generation_worker(gui, logger)
