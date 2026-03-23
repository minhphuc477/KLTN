"""Dungeon generation orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.gameplay.dungeon_generation_controls import (
    generate_dungeon as _generate_dungeon_flow_helper,
    stop_auto_solve as _stop_auto_solve_flow_helper,
)


def stop_auto_solve(*, gui):
    _stop_auto_solve_flow_helper(gui)


def generate_dungeon(*, gui, logger):
    _generate_dungeon_flow_helper(gui, logger)
