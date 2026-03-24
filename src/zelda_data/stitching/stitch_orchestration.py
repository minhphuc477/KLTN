"""Stitching orchestration helpers extracted from zelda_core."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

RoomPos = Tuple[int, int]
Offset = Tuple[int, int]


def build_global_grid_from_rooms(
    rooms_remapped: Dict[RoomPos, Any],
    room_height: int,
    room_width: int,
) -> Tuple[np.ndarray, Dict[RoomPos, Offset]]:
    """Build a stitched global grid and per-room offsets from remapped rooms."""
    max_row = max(pos[0] for pos in rooms_remapped.keys())
    max_col = max(pos[1] for pos in rooms_remapped.keys())

    global_height = (max_row + 1) * room_height
    global_width = (max_col + 1) * room_width

    global_grid = np.zeros((global_height, global_width), dtype=np.int32)
    room_positions: Dict[RoomPos, Offset] = {}

    for pos, room in rooms_remapped.items():
        row, col = pos
        r_offset = row * room_height
        c_offset = col * room_width
        room_positions[pos] = (r_offset, c_offset)

        h, w = room.semantic_grid.shape
        global_grid[r_offset : r_offset + h, c_offset : c_offset + w] = room.semantic_grid

    return global_grid, room_positions


def build_room_node_mappings(
    dungeon_rooms: Dict[RoomPos, Any],
    pos_remap: Dict[RoomPos, RoomPos],
    graph: Any,
) -> Tuple[Dict[RoomPos, int], Dict[int, RoomPos]]:
    """Build room->node and node->room mappings including virtual graph nodes."""
    room_to_node = {
        pos_remap[old_pos]: room.graph_node_id
        for old_pos, room in dungeon_rooms.items()
        if old_pos in pos_remap and room.graph_node_id is not None
    }

    node_to_room: Dict[int, RoomPos] = {}
    for room_pos, node_id in room_to_node.items():
        node_to_room[node_id] = room_pos

    if graph is not None:
        for node_id in graph.nodes():
            if node_id in node_to_room:
                continue
            node_data = graph.nodes[node_id]
            if not node_data.get("is_virtual"):
                continue
            parent = node_data.get("virtual_parent")
            if parent is not None and parent in node_to_room:
                node_to_room[node_id] = node_to_room[parent]

    return room_to_node, node_to_room


def place_special_markers(
    global_grid: np.ndarray,
    room_positions: Dict[RoomPos, Offset],
    start_pos_remapped: Optional[RoomPos],
    triforce_pos_remapped: Optional[RoomPos],
    find_floor_near_door_fn: Callable[[np.ndarray, int, int], Tuple[int, int]],
    semantic_palette: Dict[str, int],
    room_height: int,
    room_width: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Place START/TRIFORCE markers in global grid based on remapped room positions."""
    start_global: Optional[Tuple[int, int]] = None
    triforce_global: Optional[Tuple[int, int]] = None

    if start_pos_remapped:
        r_off, c_off = room_positions[start_pos_remapped]
        start_global = find_floor_near_door_fn(global_grid, r_off, c_off)
        if start_global:
            global_grid[start_global[0], start_global[1]] = semantic_palette["START"]

    if triforce_pos_remapped:
        r_off, c_off = room_positions[triforce_pos_remapped]
        center_r = r_off + room_height // 2
        center_c = c_off + room_width // 2
        global_grid[center_r, center_c] = semantic_palette["TRIFORCE"]
        triforce_global = (center_r, center_c)

    return start_global, triforce_global


def project_output_metadata(
    pos_remap: Dict[RoomPos, RoomPos],
    room_positions: Dict[RoomPos, Offset],
    room_to_node: Dict[RoomPos, int],
    node_to_room: Dict[int, RoomPos],
    dungeon_start_pos: Optional[RoomPos],
    dungeon_triforce_pos: Optional[RoomPos],
    start_pos_remapped: Optional[RoomPos],
    triforce_pos_remapped: Optional[RoomPos],
) -> Tuple[
    Dict[RoomPos, Offset],
    Dict[RoomPos, int],
    Dict[int, RoomPos],
    Optional[RoomPos],
    Optional[RoomPos],
]:
    """Project stitched metadata back to original room coordinates for downstream users."""
    inv_pos_remap = {new_pos: old_pos for old_pos, new_pos in pos_remap.items()}

    room_positions_out = {
        inv_pos_remap.get(new_pos, new_pos): offset
        for new_pos, offset in room_positions.items()
    }
    room_to_node_out = {
        inv_pos_remap.get(new_pos, new_pos): node_id
        for new_pos, node_id in room_to_node.items()
    }
    node_to_room_out = {
        node_id: inv_pos_remap.get(new_pos, new_pos)
        for node_id, new_pos in node_to_room.items()
    }

    start_pos_out = dungeon_start_pos if dungeon_start_pos is not None else start_pos_remapped
    triforce_pos_out = (
        dungeon_triforce_pos if dungeon_triforce_pos is not None else triforce_pos_remapped
    )

    return (
        room_positions_out,
        room_to_node_out,
        node_to_room_out,
        start_pos_out,
        triforce_pos_out,
    )
