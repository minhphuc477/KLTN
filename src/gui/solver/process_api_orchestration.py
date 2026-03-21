"""Module-level solver process API orchestration bridges for gui_runner."""

from __future__ import annotations


def safe_unpickle(*, path, safe_unpickle_helper):
    return safe_unpickle_helper(path)


def convert_diagonal_to_4dir(*, path, grid, convert_diagonal_to_4dir_helper):
    return convert_diagonal_to_4dir_helper(path, grid=grid)


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
    solve_in_subprocess_helper,
):
    return solve_in_subprocess_helper(
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
    run_solver_and_dump_helper,
):
    return run_solver_and_dump_helper(
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
    run_preview_and_dump_helper,
):
    return run_preview_and_dump_helper(
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
