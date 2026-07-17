"""Shared graph-to-grid validation context for stitched dungeons.

The tile-state oracle must receive the exact graph-to-room mapping used during
stitching.  Supplying only the final grid proves tile reachability, but cannot
enforce graph-owned gates while a path crosses from one room to another.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx


def build_stitched_validation_context(
    graph: Optional[nx.Graph],
    stitched_layout: Optional[Any],
) -> Dict[str, Any]:
    """Return the room mapping accepted by :class:`ZeldaLogicEnv`.

    The function intentionally returns only keyword arguments accepted by the
    validator.  Call :func:`has_complete_stitched_validation_context` when a
    production path must prove that graph semantics were enforced at tile
    level rather than merely reported.
    """
    if graph is None or stitched_layout is None:
        return {}

    slot_positions = dict(getattr(stitched_layout, "slot_positions", {}) or {})
    room_offsets = dict(getattr(stitched_layout, "room_offsets", {}) or {})
    if not slot_positions or not room_offsets:
        return {"graph": graph}

    room_to_node = {
        tuple(slot): node_id
        for node_id, slot in slot_positions.items()
        if node_id in graph and node_id in room_offsets
    }
    room_positions = {
        tuple(slot_positions[node_id]): tuple(room_offsets[node_id])
        for node_id in room_to_node.values()
    }
    node_to_room = {node_id: room for room, node_id in room_to_node.items()}
    return {
        "graph": graph,
        "room_to_node": room_to_node,
        "room_positions": room_positions,
        "node_to_room": node_to_room,
    }


def has_complete_stitched_validation_context(
    context: Dict[str, Any],
) -> bool:
    """Whether ``context`` can enforce graph transitions on the tile route."""
    return bool(
        context.get("graph") is not None
        and context.get("room_to_node")
        and context.get("room_positions")
        and context.get("node_to_room")
    )


__all__ = [
    "build_stitched_validation_context",
    "has_complete_stitched_validation_context",
]
