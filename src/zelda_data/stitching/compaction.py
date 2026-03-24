"""Room compaction helpers for stitched dungeon layouts."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

RoomPos = Tuple[int, int]


def compact_rooms(
    rooms: Dict[RoomPos, Any],
    clone_room_with_position_fn: Callable[[Any, RoomPos], Any],
) -> Tuple[Dict[RoomPos, Any], Dict[RoomPos, RoomPos]]:
    """Remap room coordinates to remove empty rows/cols while preserving room data."""
    occupied_rows = sorted({pos[0] for pos in rooms.keys()})
    occupied_cols = sorted({pos[1] for pos in rooms.keys()})

    row_remap = {old_r: new_r for new_r, old_r in enumerate(occupied_rows)}
    col_remap = {old_c: new_c for new_c, old_c in enumerate(occupied_cols)}

    remapped_rooms: Dict[RoomPos, Any] = {}
    pos_map: Dict[RoomPos, RoomPos] = {}

    for old_pos, room in rooms.items():
        old_row, old_col = old_pos
        new_pos = (row_remap[old_row], col_remap[old_col])
        remapped_rooms[new_pos] = clone_room_with_position_fn(room, new_pos)
        pos_map[old_pos] = new_pos

    return remapped_rooms, pos_map
