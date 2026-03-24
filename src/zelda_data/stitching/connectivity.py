"""Room connectivity helpers for stitched dungeon grids."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Set, Tuple

import numpy as np

RoomPos = Tuple[int, int]


def _boundary_carve_tiles(semantic_palette: Dict[str, int], reciprocal: bool) -> Tuple[int, int]:
    """Return (src_tile, dst_tile) for a stitched room boundary.

    If the reverse doorway is absent, preserve one-way intent by marking
    the source boundary as DOOR_SOFT while keeping destination as DOOR_OPEN.
    """
    door_open = int(semantic_palette["DOOR_OPEN"])
    if reciprocal:
        return door_open, door_open
    door_soft = int(semantic_palette.get("DOOR_SOFT", door_open))
    return door_soft, door_open


def ensure_room_connectivity(
    grid: np.ndarray,
    rooms: Dict[RoomPos, Any],
    semantic_palette: Dict[str, int],
    room_height: int,
    room_width: int,
) -> None:
    """Ensure each room has walkable paths from center to all present doors."""
    for pos, room in rooms.items():
        row, col = pos
        r_base = row * room_height
        c_base = col * room_width

        center_r = r_base + room_height // 2
        center_c = c_base + room_width // 2

        if grid[center_r, center_c] == semantic_palette["WALL"]:
            grid[center_r, center_c] = semantic_palette["FLOOR"]

        if room.doors.get("N"):
            for r in range(r_base + 1, center_r + 1):
                if grid[r, center_c] == semantic_palette["WALL"]:
                    grid[r, center_c] = semantic_palette["FLOOR"]

        if room.doors.get("S"):
            for r in range(center_r, r_base + room_height - 1):
                if grid[r, center_c] == semantic_palette["WALL"]:
                    grid[r, center_c] = semantic_palette["FLOOR"]

        if room.doors.get("W"):
            for c in range(c_base + 1, center_c + 1):
                if grid[center_r, c] == semantic_palette["WALL"]:
                    grid[center_r, c] = semantic_palette["FLOOR"]

        if room.doors.get("E"):
            for c in range(center_c, c_base + room_width - 1):
                if grid[center_r, c] == semantic_palette["WALL"]:
                    grid[center_r, c] = semantic_palette["FLOOR"]


def connect_doors(
    grid: np.ndarray,
    rooms: Dict[RoomPos, Any],
    semantic_palette: Dict[str, int],
    room_height: int,
    room_width: int,
) -> None:
    """Punch through adjacent room boundaries where reciprocal doors exist."""
    for pos, room in rooms.items():
        row, col = pos
        r_base = row * room_height
        c_base = col * room_width

        if room.doors.get("N"):
            north_pos = (row - 1, col)
            if north_pos in rooms:
                reciprocal = bool(rooms[north_pos].doors.get("S"))
                src_tile, dst_tile = _boundary_carve_tiles(semantic_palette, reciprocal)
                wall_row = r_base
                for c in range(c_base + 3, c_base + 8):
                    if 0 <= c < grid.shape[1]:
                        grid[wall_row, c] = src_tile
                        if wall_row > 0:
                            grid[wall_row - 1, c] = dst_tile
                for r in range(r_base + 1, r_base + 4):
                    for c in range(c_base + 4, c_base + 7):
                        if grid[r, c] == semantic_palette["WALL"]:
                            grid[r, c] = semantic_palette["FLOOR"]

        if room.doors.get("S"):
            south_pos = (row + 1, col)
            if south_pos in rooms:
                reciprocal = bool(rooms[south_pos].doors.get("N"))
                src_tile, dst_tile = _boundary_carve_tiles(semantic_palette, reciprocal)
                wall_row = r_base + room_height - 1
                for c in range(c_base + 3, c_base + 8):
                    if 0 <= c < grid.shape[1]:
                        grid[wall_row, c] = src_tile
                        if wall_row + 1 < grid.shape[0]:
                            grid[wall_row + 1, c] = dst_tile
                for r in range(r_base + room_height - 4, r_base + room_height - 1):
                    for c in range(c_base + 4, c_base + 7):
                        if grid[r, c] == semantic_palette["WALL"]:
                            grid[r, c] = semantic_palette["FLOOR"]

        if room.doors.get("W"):
            west_pos = (row, col - 1)
            if west_pos in rooms:
                reciprocal = bool(rooms[west_pos].doors.get("E"))
                src_tile, dst_tile = _boundary_carve_tiles(semantic_palette, reciprocal)
                wall_col = c_base
                for r in range(r_base + 5, r_base + 11):
                    if 0 <= r < grid.shape[0]:
                        grid[r, wall_col] = src_tile
                        if wall_col > 0:
                            grid[r, wall_col - 1] = dst_tile
                for r in range(r_base + 6, r_base + 10):
                    for c in range(c_base + 1, c_base + 4):
                        if grid[r, c] == semantic_palette["WALL"]:
                            grid[r, c] = semantic_palette["FLOOR"]

        if room.doors.get("E"):
            east_pos = (row, col + 1)
            if east_pos in rooms:
                reciprocal = bool(rooms[east_pos].doors.get("W"))
                src_tile, dst_tile = _boundary_carve_tiles(semantic_palette, reciprocal)
                wall_col = c_base + room_width - 1
                for r in range(r_base + 5, r_base + 11):
                    if 0 <= r < grid.shape[0]:
                        grid[r, wall_col] = src_tile
                        if wall_col + 1 < grid.shape[1]:
                            grid[r, wall_col + 1] = dst_tile
                for r in range(r_base + 6, r_base + 10):
                    for c in range(c_base + room_width - 4, c_base + room_width - 1):
                        if grid[r, c] == semantic_palette["WALL"]:
                            grid[r, c] = semantic_palette["FLOOR"]

    ensure_room_connectivity(
        grid=grid,
        rooms=rooms,
        semantic_palette=semantic_palette,
        room_height=room_height,
        room_width=room_width,
    )


def find_floor_near_door(
    grid: np.ndarray,
    r_off: int,
    c_off: int,
    semantic_palette: Dict[str, int],
    room_height: int,
    room_width: int,
) -> Tuple[int, int]:
    """Find a reachable walkable tile in a room, prioritizing door-connected areas."""
    room_slice = grid[r_off : r_off + room_height, c_off : c_off + room_width]

    walkable: Set[int] = {
        semantic_palette["FLOOR"],
        semantic_palette["DOOR_OPEN"],
        semantic_palette["STAIR"],
        semantic_palette["START"],
    }

    door_positions = []
    for r in range(room_height):
        for c in range(room_width):
            if room_slice[r, c] == semantic_palette["DOOR_OPEN"]:
                door_positions.append((r, c))

    reachable = set()
    for door_r, door_c in door_positions:
        if (door_r, door_c) in reachable:
            continue
        queue = deque([(door_r, door_c)])
        visited = {(door_r, door_c)}

        while queue:
            cr, cc = queue.popleft()
            if room_slice[cr, cc] in walkable:
                reachable.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < room_height and 0 <= nc < room_width:
                        if (nr, nc) not in visited and room_slice[nr, nc] in walkable:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

    start_positions = np.where(room_slice == semantic_palette["START"])
    if len(start_positions[0]) > 0:
        start_local = (int(start_positions[0][0]), int(start_positions[1][0]))
        if start_local in reachable:
            return (r_off + start_local[0], c_off + start_local[1])

    center_r = room_height // 2
    center_c = room_width // 2

    if reachable:
        best_pos = min(reachable, key=lambda p: abs(p[0] - center_r) + abs(p[1] - center_c))
        return (r_off + best_pos[0], c_off + best_pos[1])

    for dr in range(-5, 6):
        for dc in range(-4, 5):
            r, c = center_r + dr, center_c + dc
            if 0 <= r < room_height and 0 <= c < room_width:
                if room_slice[r, c] in walkable:
                    return (r_off + r, c_off + c)

    return (r_off + center_r, c_off + center_c)
