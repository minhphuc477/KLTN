"""Room connectivity helpers for stitched dungeon grids."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Set, Tuple

import numpy as np
from src.pipeline.room_stitching import carve_room_connection_between_bboxes

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
    """Punch through adjacent room boundaries using the shared bbox connector."""
    def _room_bbox(pos: RoomPos) -> Tuple[int, int, int, int]:
        row, col = pos
        room = rooms[pos]
        local_h = int(getattr(room, "height", 0) or getattr(room, "room_height", 0) or 0)
        local_w = int(getattr(room, "width", 0) or getattr(room, "room_width", 0) or 0)
        if local_h <= 0 or local_w <= 0:
            semantic_grid = getattr(room, "semantic_grid", None)
            char_grid = getattr(room, "char_grid", None)
            grid_ref = semantic_grid if isinstance(semantic_grid, np.ndarray) else char_grid
            if isinstance(grid_ref, np.ndarray) and grid_ref.ndim == 2:
                local_h, local_w = int(grid_ref.shape[0]), int(grid_ref.shape[1])
        if local_h <= 0:
            local_h = int(room_height)
        if local_w <= 0:
            local_w = int(room_width)
        y0 = row * int(room_height)
        x0 = col * int(room_width)
        return (x0, y0, x0 + local_w - 1, y0 + local_h - 1)

    def _dataset_connector_tiles(_edge_data: Dict[str, Any] | None, has_reverse_edge: bool) -> Tuple[int, int]:
        return _boundary_carve_tiles(semantic_palette, reciprocal=has_reverse_edge)

    directions = {
        "N": ((-1, 0), "S"),
        "S": ((1, 0), "N"),
        "W": ((0, -1), "E"),
        "E": ((0, 1), "W"),
    }

    for pos, room in rooms.items():
        for direction, (delta, reverse_direction) in directions.items():
            if not room.doors.get(direction):
                continue

            neighbor = (pos[0] + delta[0], pos[1] + delta[1])
            if neighbor not in rooms:
                continue

            reciprocal = bool(rooms[neighbor].doors.get(reverse_direction))
            if reciprocal and pos > neighbor:
                continue

    # Seal ALL room border tiles BEFORE carving connections.
    seal_stitched_dungeon_boundaries(
        grid=grid,
        rooms=rooms,
        semantic_palette=semantic_palette,
        room_height=room_height,
        room_width=room_width,
    )

    # NOW carve the real connections on top of the sealed borders.
    for pos, room in rooms.items():
        for direction, (delta, reverse_direction) in directions.items():
            if not room.doors.get(direction):
                continue

            neighbor = (pos[0] + delta[0], pos[1] + delta[1])
            if neighbor not in rooms:
                continue

            reciprocal = bool(rooms[neighbor].doors.get(reverse_direction))
            if reciprocal and pos > neighbor:
                continue

            carve_room_connection_between_bboxes(
                grid,
                _room_bbox(pos),
                _room_bbox(neighbor),
                has_reverse_edge=reciprocal,
                fill_tile=int(semantic_palette.get("VOID", 0)),
                connector_tile_resolver=_dataset_connector_tiles,
            )

    ensure_room_connectivity(
        grid=grid,
        rooms=rooms,
        semantic_palette=semantic_palette,
        room_height=room_height,
        room_width=room_width,
    )


def seal_stitched_dungeon_boundaries(
    grid: np.ndarray,
    rooms: Dict[RoomPos, Any],
    semantic_palette: Dict[str, int],
    room_height: int,
    room_width: int,
) -> None:
    """Seal ALL room border tiles with walls.

    This must run BEFORE carving connections so that every border tile
    from the original room data is replaced with WALL.  The subsequent
    carving step then writes the correct DOOR tiles at the precise
    connection points, overwriting these walls.
    """
    wall_id = int(semantic_palette["WALL"])
    H, W = grid.shape

    for pos, room in rooms.items():
        row, col = pos
        y0 = row * room_height
        x0 = col * room_width

        border_specs = [
            ("N", [y0], range(x0, x0 + room_width)),
            ("S", [y0 + room_height - 1], range(x0, x0 + room_width)),
            ("W", range(y0, y0 + room_height), [x0]),
            ("E", range(y0, y0 + room_height), [x0 + room_width - 1]),
        ]

        for _dir, b_rows, b_cols in border_specs:
            for r in b_rows:
                for c in b_cols:
                    if 0 <= r < H and 0 <= c < W:
                        grid[r, c] = wall_id


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
