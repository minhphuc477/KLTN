"""Conversion helpers for adapter compatibility extracted from zelda_core."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def convert_room_to_roomdata(room: Any, roomdata_cls: Any) -> Any:
    """Convert Room-like object into RoomData-like object."""
    doors_dict: Dict[str, Dict[str, str]] = {}
    direction_map = {"N": "north", "S": "south", "E": "east", "W": "west"}
    for direction, has_door in room.doors.items():
        if has_door:
            doors_dict[direction_map.get(direction, direction)] = {"type": "open"}

    contents = []
    if room.is_start:
        contents.append("start")
    if room.has_triforce:
        contents.append("triforce")
    if room.has_boss:
        contents.append("boss")

    return roomdata_cls(
        room_id=str(room.graph_node_id) if room.graph_node_id else f"{room.position[0]}_{room.position[1]}",
        grid=room.semantic_grid,
        contents=contents,
        doors=doors_dict,
        position=room.position,
    )


def convert_dungeon_to_dungeondata(
    dungeon: Any,
    convert_room_to_roomdata_fn,
    ml_feature_extractor_cls: Any,
    dungeondata_cls: Any,
) -> Any:
    """Convert Dungeon-like object into DungeonData-like object."""
    rooms_dict = {}
    for _pos, room in dungeon.rooms.items():
        room_data = convert_room_to_roomdata_fn(room)
        rooms_dict[room_data.room_id] = room_data

    ml_extractor = ml_feature_extractor_cls()
    tpe_vectors, node_order = ml_extractor.compute_laplacian_pe(dungeon.graph)
    node_features = ml_extractor.extract_node_features(dungeon.graph, node_order)
    p_matrix = ml_extractor.build_p_matrix(dungeon.graph, node_order)

    positions = [pos for pos in dungeon.rooms.keys()]
    if positions:
        max_r = max(p[0] for p in positions) + 1
        max_c = max(p[1] for p in positions) + 1
        layout = np.full((max_r, max_c), -1, dtype=int)
        for pos, room in dungeon.rooms.items():
            if room.graph_node_id is not None:
                layout[pos[0], pos[1]] = room.graph_node_id
    else:
        layout = np.zeros((0, 0), dtype=int)

    return dungeondata_cls(
        dungeon_id=dungeon.dungeon_id,
        rooms=rooms_dict,
        graph=dungeon.graph,
        layout=layout,
        tpe_vectors=tpe_vectors,
        p_matrix=p_matrix,
        node_features=node_features,
    )
