"""Module-level solver process API orchestration bridges for gui_runner."""

from __future__ import annotations

from src.gui.solver.process_worker import (
    run_preview_and_dump as _run_preview_and_dump_helper,
    run_solver_and_dump as _run_solver_and_dump_helper,
    solve_in_subprocess as _solve_in_subprocess_helper,
)
from src.gui.solver.utils import (
    convert_diagonal_to_4dir as _convert_diagonal_to_4dir_helper,
    safe_unpickle as _safe_unpickle_helper,
)


def safe_unpickle(*, path):
    return _safe_unpickle_helper(path)


def convert_diagonal_to_4dir(*, path, grid):
    return _convert_diagonal_to_4dir_helper(path, grid=grid)


def solve_in_subprocess(
    *,
    grid,
    start_pos,
    goal_pos,
    algorithm_idx,
    feature_flags,
    priority_options,
    graph,
    room_to_node,
    room_positions,
    node_to_room,
):
    return _solve_in_subprocess_helper(
        grid,
        start_pos,
        goal_pos,
        algorithm_idx,
        feature_flags,
        priority_options,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
    )


def run_solver_and_dump(
    *,
    grid_or_path,
    start_pos,
    goal_pos,
    algorithm_idx,
    feature_flags,
    priority_options,
    out_path,
    graph,
    room_to_node,
    room_positions,
    node_to_room,
):
    return _run_solver_and_dump_helper(
        grid_or_path,
        start_pos,
        goal_pos,
        algorithm_idx,
        feature_flags,
        priority_options,
        out_path,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
    )


def run_preview_and_dump(
    *,
    grid_or_path,
    start_pos,
    goal_pos,
    algorithm_idx,
    feature_flags,
    priority_options,
    out_path,
    graph,
    room_to_node,
    room_positions,
    node_to_room,
):
    return _run_preview_and_dump_helper(
        grid_or_path,
        start_pos,
        goal_pos,
        algorithm_idx,
        feature_flags,
        priority_options,
        out_path,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
    )
