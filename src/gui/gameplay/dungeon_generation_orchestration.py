"""Dungeon generation orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def stop_auto_solve(*, gui, stop_auto_solve_flow_helper):
    stop_auto_solve_flow_helper(gui)


def generate_dungeon(*, gui, logger, generate_dungeon_flow_helper):
    generate_dungeon_flow_helper(gui, logger)
