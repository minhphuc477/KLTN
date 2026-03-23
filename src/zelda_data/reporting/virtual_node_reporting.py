"""Helpers for reporting graph node-to-room mapping status."""

from __future__ import annotations

import logging
from typing import Any, List, Tuple


def log_virtual_node_report(dungeon: Any, context: str = "load", logger: logging.Logger | None = None) -> None:
    """Emit a per-dungeon mapping report for physical and virtual graph nodes."""
    log = logger or logging.getLogger(__name__)

    if dungeon is None or getattr(dungeon, "graph", None) is None:
        return

    room_to_node = {
        pos: room.graph_node_id
        for pos, room in getattr(dungeon, "rooms", {}).items()
        if getattr(room, "graph_node_id", None) is not None
    }
    node_to_room = {node_id: room_pos for room_pos, node_id in room_to_node.items()}

    physical_mapped_nodes: List[int] = []
    virtual_with_parent_room: List[Tuple[int, int, Tuple[int, int]]] = []
    virtual_unmapped_nodes: List[int] = []

    for node_id, attrs in dungeon.graph.nodes(data=True):
        is_pointer = bool(attrs.get("is_start_pointer", False))
        is_virtual = bool(attrs.get("is_virtual", False)) or is_pointer

        if not is_virtual:
            if node_id in node_to_room:
                physical_mapped_nodes.append(int(node_id))
            continue

        parent = attrs.get("virtual_parent")
        if parent is not None and parent in node_to_room:
            virtual_with_parent_room.append((int(node_id), int(parent), node_to_room[parent]))
        elif node_id not in node_to_room:
            virtual_unmapped_nodes.append(int(node_id))

    physical_mapped_nodes = sorted(set(physical_mapped_nodes))
    virtual_with_parent_room = sorted(set(virtual_with_parent_room), key=lambda x: x[0])
    virtual_unmapped_nodes = sorted(set(virtual_unmapped_nodes))

    log.info(
        "NODE_MAPPING_REPORT[%s:%s]: physical_mapped=%d, virtual_with_parent_room=%d, virtual_unmapped=%d",
        context,
        dungeon.dungeon_id,
        len(physical_mapped_nodes),
        len(virtual_with_parent_room),
        len(virtual_unmapped_nodes),
    )
    log.info(
        "NODE_MAPPING_REPORT[%s:%s]: physical_mapped_nodes=%s",
        context,
        dungeon.dungeon_id,
        physical_mapped_nodes,
    )
    log.info(
        "NODE_MAPPING_REPORT[%s:%s]: virtual_with_parent_room=%s",
        context,
        dungeon.dungeon_id,
        virtual_with_parent_room,
    )
    log.info(
        "NODE_MAPPING_REPORT[%s:%s]: virtual_unmapped_nodes=%s",
        context,
        dungeon.dungeon_id,
        virtual_unmapped_nodes,
    )
