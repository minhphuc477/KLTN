"""Topology match-control orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.topology.match_controls import (
    apply_tentative_matches as _apply_tentative_matches,
    match_missing_nodes as _match_missing_nodes,
    undo_last_match as _undo_last_match,
)


def match_missing_nodes(*, gui, matcher_cls, logger):
    return _match_missing_nodes(gui=gui, matcher_cls=matcher_cls, logger=logger)


def undo_last_match(*, gui, logger):
    return _undo_last_match(gui=gui, logger=logger)


def apply_tentative_matches(*, gui, logger):
    return _apply_tentative_matches(gui=gui, logger=logger)
