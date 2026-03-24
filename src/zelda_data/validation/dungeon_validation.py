"""Dungeon-level validation helpers extracted from zelda_core."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


def validate_dungeon(
    dungeon: Any,
    stitched: Optional[Any],
    parse_node_label_tokens_fn: Callable[[str], List[str]],
    semantic_palette: Dict[str, int],
    solver_cls: Any,
) -> Dict[str, Any]:
    """Validate dungeon integrity and produce detailed errors/warnings/stats."""
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {
        "num_rooms": len(dungeon.rooms),
        "num_graph_nodes": len(dungeon.graph.nodes()) if dungeon.graph else 0,
        "num_graph_edges": len(dungeon.graph.edges()) if dungeon.graph else 0,
        "rooms_with_node_assignment": 0,
        "graph_nodes_with_room": 0,
        "items_in_graph": 0,
        "items_placed_in_grid": 0,
        "locked_doors_in_graph": 0,
        "locked_doors_in_grid": 0,
    }

    if dungeon.graph:
        num_nodes = len(dungeon.graph.nodes())
        num_rooms = len(dungeon.rooms)

        if num_nodes > num_rooms:
            warnings.append(
                f"GRAPH_ROOM_MISMATCH: {num_nodes} graph nodes but only {num_rooms} physical rooms. "
                f"{num_nodes - num_rooms} nodes have no room."
            )
        elif num_rooms > num_nodes:
            warnings.append(
                f"GRAPH_ROOM_MISMATCH: {num_rooms} physical rooms but only {num_nodes} graph nodes. "
                f"{num_rooms - num_nodes} rooms have no graph data."
            )

    rooms_without_nodes = []
    for pos, room in dungeon.rooms.items():
        if room.graph_node_id is not None:
            stats["rooms_with_node_assignment"] += 1
        else:
            rooms_without_nodes.append(pos)

    if rooms_without_nodes:
        warnings.append(
            f"UNMAPPED_ROOMS: {len(rooms_without_nodes)} rooms have no graph node assignment: "
            f"{rooms_without_nodes[:5]}{'...' if len(rooms_without_nodes) > 5 else ''}"
        )

    if dungeon.graph:
        assigned_nodes = {
            room.graph_node_id for room in dungeon.rooms.values() if room.graph_node_id is not None
        }
        all_nodes = set(dungeon.graph.nodes())
        unassigned_nodes = all_nodes - assigned_nodes
        stats["graph_nodes_with_room"] = len(assigned_nodes)

        if unassigned_nodes:
            warnings.append(
                f"UNMAPPED_NODES: {len(unassigned_nodes)} graph nodes have no room: "
                f"{list(unassigned_nodes)[:5]}{'...' if len(unassigned_nodes) > 5 else ''}"
            )

    if dungeon.start_pos is None:
        errors.append("MISSING_START: No start position defined")
    elif dungeon.start_pos not in dungeon.rooms:
        errors.append(f"INVALID_START: Start position {dungeon.start_pos} is not a valid room")
    else:
        start_room = dungeon.rooms[dungeon.start_pos]
        if not start_room.has_stair and not start_room.is_start:
            warnings.append(f"UNUSUAL_START: Start room {dungeon.start_pos} has no stair marker")

    if dungeon.triforce_pos is None:
        errors.append("MISSING_TRIFORCE: No triforce position defined")
    elif dungeon.triforce_pos not in dungeon.rooms:
        errors.append(f"INVALID_TRIFORCE: Triforce position {dungeon.triforce_pos} is not a valid room")

    if dungeon.graph:
        for _node_id, attrs in dungeon.graph.nodes(data=True):
            label = attrs.get("label", "")
            parts = parse_node_label_tokens_fn(label)
            if any(p in parts for p in ["k", "K", "I", "i"]):
                stats["items_in_graph"] += 1

        for _, _, edata in dungeon.graph.edges(data=True):
            label = edata.get("label", "")
            edge_type = edata.get("edge_type", "")
            if label in ("k", "b", "K") or edge_type in ("key_locked", "bombable", "boss_locked"):
                stats["locked_doors_in_graph"] += 1

    if stitched is not None:
        item_tiles = {
            semantic_palette["KEY_SMALL"],
            semantic_palette["KEY_BOSS"],
            semantic_palette["KEY_ITEM"],
            semantic_palette["ITEM_MINOR"],
            semantic_palette["KEY"],
            semantic_palette["ITEM"],
        }
        for tile_id in item_tiles:
            stats["items_placed_in_grid"] += int(np.sum(stitched.global_grid == tile_id))

        locked_door_tiles = {
            semantic_palette["DOOR_LOCKED"],
            semantic_palette["DOOR_BOMB"],
            semantic_palette["DOOR_BOSS"],
        }
        for tile_id in locked_door_tiles:
            stats["locked_doors_in_grid"] += int(np.sum(stitched.global_grid == tile_id))

        if stats["items_in_graph"] > 0 and stats["items_placed_in_grid"] == 0:
            errors.append(
                f"ITEMS_NOT_MATERIALIZED: Graph specifies {stats['items_in_graph']} items "
                f"but 0 items found in grid"
            )
        elif stats["items_placed_in_grid"] < stats["items_in_graph"]:
            warnings.append(
                f"ITEMS_PARTIALLY_PLACED: Graph specifies {stats['items_in_graph']} items "
                f"but only {stats['items_placed_in_grid']} found in grid"
            )

        if stitched.start_global and stitched.triforce_global:
            solver = solver_cls()
            result = solver._solve_with_grid(stitched)
            if not result.get("solvable"):
                warnings.append(
                    "GRID_UNREACHABLE: No walkable path from start to triforce in grid. "
                    f"Reason: {result.get('reason', 'unknown')}"
                )
                stats["grid_reachable"] = False
            else:
                stats["grid_reachable"] = True
                stats["grid_path_length"] = result.get("path_length", 0)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }
