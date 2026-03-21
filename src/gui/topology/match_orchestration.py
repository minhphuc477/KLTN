"""Topology match-control orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def match_missing_nodes(*, gui, matcher_cls, logger, match_missing_nodes_helper):
    return match_missing_nodes_helper(gui=gui, matcher_cls=matcher_cls, logger=logger)


def undo_last_match(*, gui, logger, undo_last_match_helper):
    return undo_last_match_helper(gui=gui, logger=logger)


def apply_tentative_matches(*, gui, logger, apply_tentative_matches_helper):
    return apply_tentative_matches_helper(gui=gui, logger=logger)
