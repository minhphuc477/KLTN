"""Dungeon solver helper functions extracted from zelda_core."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import networkx as nx


RoomPos = Tuple[int, int]
GlobalPos = Tuple[int, int]


def _find_special_rooms(
    room_positions: Dict[RoomPos, Tuple[int, int]],
    room_height: int,
    room_width: int,
    start_global: Optional[GlobalPos],
    triforce_global: Optional[GlobalPos],
) -> Tuple[Optional[RoomPos], Optional[RoomPos]]:
    """Locate room coordinates that contain start/triforce global positions."""
    start_room = None
    triforce_room = None

    for room_pos, (r_off, c_off) in room_positions.items():
        r_end = r_off + room_height
        c_end = c_off + room_width

        if start_global and r_off <= start_global[0] < r_end and c_off <= start_global[1] < c_end:
            start_room = room_pos

        if triforce_global and r_off <= triforce_global[0] < r_end and c_off <= triforce_global[1] < c_end:
            triforce_room = room_pos

    return start_room, triforce_room


def solve(
    stitched: Any,
    mode: str,
    solve_with_state_space_fn,
    solve_with_grid_fn,
) -> Dict[str, Any]:
    """Entrypoint logic for choosing graph state-space or grid fallback solver."""
    if stitched.start_global is None:
        return {"solvable": False, "reason": "No START position"}

    if stitched.triforce_global is None:
        return {"solvable": False, "reason": "No TRIFORCE position"}

    if stitched.graph and stitched.room_to_node:
        return solve_with_state_space_fn(stitched, mode)

    return solve_with_grid_fn(stitched)


def solve_with_state_space(
    stitched: Any,
    mode: str,
    room_height: int,
    room_width: int,
    state_space_solver_cls: Any,
) -> Dict[str, Any]:
    """Solve with inventory-aware graph state space."""
    start_room, triforce_room = _find_special_rooms(
        room_positions=stitched.room_positions,
        room_height=room_height,
        room_width=room_width,
        start_global=stitched.start_global,
        triforce_global=stitched.triforce_global,
    )

    if not start_room or not triforce_room:
        return {"solvable": False, "reason": "Could not locate start/triforce rooms"}

    start_node = stitched.room_to_node.get(start_room)
    triforce_node = stitched.room_to_node.get(triforce_room)

    if start_node is None:
        return {"solvable": False, "reason": f"Start room {start_room} not mapped to graph node"}

    if triforce_node is None:
        return {"solvable": False, "reason": f"Triforce room {triforce_room} not mapped to graph node"}

    solver = state_space_solver_cls(stitched.graph, mode=mode)
    result = solver.solve(start_node, triforce_node)

    if result.get("solvable"):
        result["start_room"] = start_room
        result["triforce_room"] = triforce_room
        result["mode"] = mode

    return result


def solve_with_graph(
    stitched: Any,
    room_height: int,
    room_width: int,
) -> Dict[str, Any]:
    """Legacy graph reachability check ignoring edge constraints."""
    start_room, triforce_room = _find_special_rooms(
        room_positions=stitched.room_positions,
        room_height=room_height,
        room_width=room_width,
        start_global=stitched.start_global,
        triforce_global=stitched.triforce_global,
    )

    if not start_room or not triforce_room:
        return {"solvable": False, "reason": "Could not locate start/triforce rooms"}

    start_node = stitched.room_to_node.get(start_room)
    triforce_node = stitched.room_to_node.get(triforce_room)

    if start_node is None:
        return {"solvable": False, "reason": f"Start room {start_room} not mapped to graph node"}

    if triforce_node is None:
        return {"solvable": False, "reason": f"Triforce room {triforce_room} not mapped to graph node"}

    try:
        path = nx.shortest_path(stitched.graph, start_node, triforce_node)
        return {
            "solvable": True,
            "path_length": len(path) - 1,
            "rooms_traversed": len(path),
        }
    except nx.NetworkXNoPath:
        return {
            "solvable": False,
            "reason": f"No graph path from node {start_node} to {triforce_node}",
        }


def solve_with_grid(
    stitched: Any,
    walkable_tiles,
    triforce_tile: int,
    room_height: int,
    room_width: int,
) -> Dict[str, Any]:
    """Grid-BFS fallback reachability check."""
    grid = stitched.global_grid
    start = stitched.start_global
    goal = stitched.triforce_global

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        pos, dist = queue.popleft()

        if pos == goal:
            rooms_hit = set()
            for room_pos, (r_off, c_off) in stitched.room_positions.items():
                r_end = r_off + room_height
                c_end = c_off + room_width
                if r_off <= pos[0] < r_end and c_off <= pos[1] < c_end:
                    rooms_hit.add(room_pos)

            return {
                "solvable": True,
                "path_length": dist,
                "rooms_traversed": len(rooms_hit) if rooms_hit else 1,
            }

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc

            if (nr, nc) in visited:
                continue

            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                tile = grid[nr, nc]
                if tile in walkable_tiles or tile == triforce_tile:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))

    return {
        "solvable": False,
        "reason": "No path found",
        "reachable_tiles": len(visited),
    }
