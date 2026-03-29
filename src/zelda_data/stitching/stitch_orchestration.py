"""Stitching orchestration helpers extracted from zelda_core."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from src.pipeline.room_stitching import StitchedRoomLayout, build_room_canvas_from_slots

RoomPos = Tuple[int, int]
Offset = Tuple[int, int]


def build_global_grid_from_rooms(
    rooms_remapped: Dict[RoomPos, Any],
    room_height: int,
    room_width: int,
) -> Tuple[np.ndarray, Dict[RoomPos, Offset]]:
    """Compatibility wrapper around the canonical stitched-room layout builder."""
    stitched = build_stitched_room_layout_from_rooms(
        rooms_remapped=rooms_remapped,
        room_height=room_height,
        room_width=room_width,
    )
    return stitched.dungeon_grid, stitched.room_offsets


def build_stitched_room_layout_from_rooms(
    rooms_remapped: Dict[RoomPos, Any],
    room_height: int,
    room_width: int,
) -> StitchedRoomLayout:
    """Build the canonical stitched-room layout object for remapped Zelda rooms."""
    room_grids: Dict[RoomPos, np.ndarray] = {}
    for pos, room in rooms_remapped.items():
        h, w = room.semantic_grid.shape
        if h != room_height or w != room_width:
            raise ValueError(
                "CRITICAL: Room dimension mismatch before stitching at "
                f"room {pos}: expected {room_height}x{room_width}, got {h}x{w}."
            )
        room_grids[pos] = np.asarray(room.semantic_grid, dtype=np.int32)

    return build_room_canvas_from_slots(
        room_grids=room_grids,
        slot_positions={pos: pos for pos in rooms_remapped.keys()},
        fill_tile=0,
    )


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
