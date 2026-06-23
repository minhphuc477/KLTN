"""Graph-driven content placement helpers for stitched dungeons."""

from __future__ import annotations

from collections import deque
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.core.definitions import DOOR_POSITIONS

RoomPos = Tuple[int, int]
Offset = Tuple[int, int]


def place_items_from_graph(
    grid: np.ndarray,
    graph: nx.DiGraph,
    room_to_node: Dict[RoomPos, int],
    room_positions: Dict[RoomPos, Offset],
    parse_node_label_tokens_fn: Callable[[str], List[str]],
    semantic_palette: Dict[str, int],
    room_height: int,
    room_width: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Place item tiles from graph node labels into stitched grid rooms."""
    missing_items: List[Dict[str, Any]] = []
    fallback_placements: List[Tuple[int, str, str]] = []

    if graph is None:
        return missing_items

    node_to_room = {node_id: room_pos for room_pos, node_id in room_to_node.items()}

    walkable_for_item = {semantic_palette["FLOOR"]}
    convertible_for_item = {semantic_palette["VOID"]}

    item_type_names = {
        semantic_palette["KEY_SMALL"]: "KEY_SMALL",
        semantic_palette["KEY_BOSS"]: "KEY_BOSS",
        semantic_palette["KEY_ITEM"]: "KEY_ITEM",
        semantic_palette["ITEM_MINOR"]: "ITEM_MINOR",
    }

    def _find_fallback_room(unmapped_node: int) -> Optional[RoomPos]:
        visited = {unmapped_node}
        queue = deque([unmapped_node])

        while queue:
            current = queue.popleft()
            neighbors = list(graph.predecessors(current)) + list(graph.successors(current))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                if neighbor in node_to_room and node_to_room[neighbor] in room_positions:
                    return node_to_room[neighbor]
                queue.append(neighbor)
        return None

    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("label", "")
        parts = parse_node_label_tokens_fn(label)

        item_to_place = None
        if "k" in parts:
            item_to_place = semantic_palette["KEY_SMALL"]
        elif "K" in parts:
            item_to_place = semantic_palette["KEY_BOSS"]
        elif "I" in parts:
            item_to_place = semantic_palette["KEY_ITEM"]
        elif "i" in parts:
            item_to_place = semantic_palette["ITEM_MINOR"]

        if item_to_place is None:
            continue

        room_pos = node_to_room.get(node_id)
        used_fallback = False

        if room_pos is None:
            fallback_room = _find_fallback_room(node_id)
            if fallback_room is not None:
                room_pos = fallback_room
                used_fallback = True
                fallback_placements.append(
                    (node_id, label, item_type_names.get(item_to_place, str(item_to_place)))
                )
                logger.debug(
                    "ITEM_PLACEMENT_FALLBACK: Node %d (label='%s') with item '%s' has no direct room mapping. "
                    "Using fallback room %s from nearest mapped neighbor.",
                    node_id,
                    label,
                    item_type_names.get(item_to_place, str(item_to_place)),
                    room_pos,
                )
            else:
                logger.debug(
                    "ITEM_PLACEMENT_FAIL: Node %d (label='%s') with item '%s' has no room mapping "
                    "and no fallback room found. Item cannot be materialized in grid.",
                    node_id,
                    label,
                    item_type_names.get(item_to_place, str(item_to_place)),
                )
                missing_items.append(
                    {
                        "node_id": node_id,
                        "label": label,
                        "item_type": item_type_names.get(item_to_place, str(item_to_place)),
                        "reason": "no_room_mapping",
                    }
                )
                continue

        if room_pos not in room_positions:
            logger.debug(
                "ITEM_PLACEMENT_FAIL: Node %d mapped to room %s but room has no grid position. "
                "Item '%s' cannot be materialized.",
                node_id,
                room_pos,
                item_type_names.get(item_to_place, str(item_to_place)),
            )
            missing_items.append(
                {
                    "node_id": node_id,
                    "label": label,
                    "item_type": item_type_names.get(item_to_place, str(item_to_place)),
                    "reason": "no_grid_position",
                }
            )
            continue

        r_off, c_off = room_positions[room_pos]
        center_r = r_off + room_height // 2
        center_c = c_off + room_width // 2

        placed = False
        for radius in range(0, 6):
            if placed:
                break
            for dr in range(-radius, radius + 1):
                if placed:
                    break
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    r = center_r + dr
                    c = center_c + dc
                    if not (
                        r_off + 2 <= r < r_off + room_height - 2
                        and c_off + 2 <= c < c_off + room_width - 2
                    ):
                        continue
                    if grid[r, c] in walkable_for_item:
                        grid[r, c] = item_to_place
                        placed = True
                        logger.debug(
                            "Placed item %s at (%d, %d) for node %d in room %s",
                            item_type_names.get(item_to_place, str(item_to_place)),
                            r,
                            c,
                            node_id,
                            room_pos,
                        )
                        break

        if not placed:
            corners = [
                (r_off + 3, c_off + 3),
                (r_off + 3, c_off + room_width - 4),
                (r_off + room_height - 4, c_off + 3),
                (r_off + room_height - 4, c_off + room_width - 4),
            ]
            for r, c in corners:
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    if grid[r, c] in walkable_for_item:
                        grid[r, c] = item_to_place
                        placed = True
                        logger.debug(
                            "Placed item %s at corner (%d, %d) for node %d in room %s",
                            item_type_names.get(item_to_place, str(item_to_place)),
                            r,
                            c,
                            node_id,
                            room_pos,
                        )
                        break

        if not placed:
            for radius in range(0, 6):
                if placed:
                    break
                for dr in range(-radius, radius + 1):
                    if placed:
                        break
                    for dc in range(-radius, radius + 1):
                        if abs(dr) != radius and abs(dc) != radius:
                            continue
                        r = center_r + dr
                        c = center_c + dc
                        if not (
                            r_off + 2 <= r < r_off + room_height - 2
                            and c_off + 2 <= c < c_off + room_width - 2
                        ):
                            continue
                        if grid[r, c] in convertible_for_item:
                            grid[r, c] = item_to_place
                            placed = True
                            logger.debug(
                                "Placed item %s at converted void (%d, %d) for node %d in room %s",
                                item_type_names.get(item_to_place, str(item_to_place)),
                                r,
                                c,
                                node_id,
                                room_pos,
                            )
                            break

        if not placed:
            r, c = center_r, center_c
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                original_tile = grid[r, c]
                grid[r, c] = item_to_place
                placed = True
                logger.warning(
                    "ITEM_PLACEMENT_FORCED: Node %d item '%s' force-placed at center (%d, %d) "
                    "in room %s (overwrote tile %d). No valid floor found.",
                    node_id,
                    item_type_names.get(item_to_place, str(item_to_place)),
                    r,
                    c,
                    room_pos,
                    original_tile,
                )

        if not placed:
            logger.error(
                "ITEM_PLACEMENT_FAILED: Could not place item '%s' for node %d in room %s. "
                "All placement strategies exhausted.",
                item_type_names.get(item_to_place, str(item_to_place)),
                node_id,
                room_pos,
            )
            missing_items.append(
                {
                    "node_id": node_id,
                    "label": label,
                    "item_type": item_type_names.get(item_to_place, str(item_to_place)),
                    "reason": "placement_failed",
                    "used_fallback": used_fallback,
                }
            )

    if fallback_placements or missing_items:
        summary_parts = []
        if fallback_placements:
            summary_parts.append(f"{len(fallback_placements)} items placed via fallback")
        if missing_items:
            summary_parts.append(f"{len(missing_items)} items could not be placed")
        logger.info("ITEM_PLACEMENT_SUMMARY: %s", ", ".join(summary_parts))

    return missing_items


def place_entities_from_graph(
    grid: np.ndarray,
    graph: nx.DiGraph,
    room_to_node: Dict[RoomPos, int],
    room_positions: Dict[RoomPos, Offset],
    parse_node_label_tokens_fn: Callable[[str], List[str]],
    semantic_palette: Dict[str, int],
    room_height: int,
    room_width: int,
    logger: logging.Logger,
) -> None:
    """Place ENEMY/BOSS entities from graph node labels into stitched grid rooms."""
    if graph is None:
        return

    node_to_room = {v: k for k, v in room_to_node.items()}

    entity_map = {
        "e": semantic_palette["ENEMY"],
        "b": semantic_palette["BOSS"],
    }

    entity_names = {
        semantic_palette["ENEMY"]: "ENEMY",
        semantic_palette["BOSS"]: "BOSS",
    }

    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("label", "")
        parts = parse_node_label_tokens_fn(label)

        entity_to_place = None
        for part in parts:
            if part in entity_map:
                entity_to_place = entity_map[part]
                break

        if entity_to_place is None:
            continue

        room_pos = node_to_room.get(node_id)
        if room_pos is None or room_pos not in room_positions:
            continue

        r_off, c_off = room_positions[room_pos]
        room_slice = grid[r_off : r_off + room_height, c_off : c_off + room_width]
        if entity_to_place in room_slice:
            continue

        center_r = r_off + room_height // 2
        center_c = c_off + room_width // 2

        placed = False
        for radius in range(0, 4):
            if placed:
                break
            for dr in range(-radius, radius + 1):
                if placed:
                    break
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    r, c = center_r + dr, center_c + dc
                    if not (
                        r_off + 2 <= r < r_off + room_height - 2
                        and c_off + 2 <= c < c_off + room_width - 2
                    ):
                        continue
                    if grid[r, c] == semantic_palette["FLOOR"]:
                        grid[r, c] = entity_to_place
                        placed = True
                        logger.debug(
                            "Placed %s at (%d, %d) for node %d in room %s",
                            entity_names.get(entity_to_place, str(entity_to_place)),
                            r,
                            c,
                            node_id,
                            room_pos,
                        )
                        break

        if not placed:
            logger.debug(
                "ENTITY_PLACEMENT_SKIP: Could not place %s for node %d in room %s (no floor tile)",
                entity_names.get(entity_to_place, str(entity_to_place)),
                node_id,
                room_pos,
            )


def find_boundary_doors(
    grid: np.ndarray,
    from_offset: Offset,
    to_offset: Offset,
    from_room: RoomPos,
    to_room: RoomPos,
    room_height: int,
    room_width: int,
) -> List[Tuple[int, int]]:
    """Find boundary door cells between two adjacent rooms."""
    from_r, from_c = from_room
    to_r, to_c = to_room
    from_off_r, from_off_c = from_offset
    _to_off_r, _to_off_c = to_offset

    door_positions = []

    if to_r == from_r - 1:
        north = DOOR_POSITIONS["N"]
        boundary_r = from_off_r
        for c in range(from_off_c + int(north["col_start"]), from_off_c + int(north["col_end"])):
            if 0 <= c < grid.shape[1]:
                door_positions.append((boundary_r, c))
    elif to_r == from_r + 1:
        south = DOOR_POSITIONS["S"]
        boundary_r = from_off_r + room_height - 1
        for c in range(from_off_c + int(south["col_start"]), from_off_c + int(south["col_end"])):
            if 0 <= c < grid.shape[1]:
                door_positions.append((boundary_r, c))
    elif to_c == from_c - 1:
        west = DOOR_POSITIONS["W"]
        boundary_c = from_off_c
        for r in range(from_off_r + int(west["row_start"]), from_off_r + int(west["row_end"])):
            if 0 <= r < grid.shape[0]:
                door_positions.append((r, boundary_c))
    elif to_c == from_c + 1:
        east = DOOR_POSITIONS["E"]
        boundary_c = from_off_c + room_width - 1
        for r in range(from_off_r + int(east["row_start"]), from_off_r + int(east["row_end"])):
            if 0 <= r < grid.shape[0]:
                door_positions.append((r, boundary_c))

    return door_positions


def apply_door_types_from_graph(
    grid: np.ndarray,
    graph: nx.DiGraph,
    room_to_node: Dict[RoomPos, int],
    room_positions: Dict[RoomPos, Offset],
    semantic_palette: Dict[str, int],
    room_height: int,
    room_width: int,
    logger: logging.Logger,
) -> None:
    """Apply typed door semantics from graph edges onto stitched boundary doors."""
    if graph is None:
        return

    node_to_room = {v: k for k, v in room_to_node.items()}

    door_type_map = {
        "k": semantic_palette["DOOR_LOCKED"],
        "b": semantic_palette["DOOR_BOMB"],
        "l": semantic_palette["DOOR_SOFT"],
        "K": semantic_palette["DOOR_BOSS"],
        "s": semantic_palette["STAIR"],
    }

    door_type_names = {
        semantic_palette["DOOR_LOCKED"]: "DOOR_LOCKED",
        semantic_palette["DOOR_BOMB"]: "DOOR_BOMB",
        semantic_palette["DOOR_SOFT"]: "DOOR_SOFT",
        semantic_palette["DOOR_BOSS"]: "DOOR_BOSS",
        semantic_palette["STAIR"]: "STAIR",
    }

    for from_node, to_node, edge_data in graph.edges(data=True):
        label = edge_data.get("label", "")
        edge_type = edge_data.get("edge_type", "")

        door_type = door_type_map.get(label)
        if door_type is None and edge_type in ("key_locked", "locked"):
            door_type = semantic_palette["DOOR_LOCKED"]
        elif door_type is None and edge_type == "bombable":
            door_type = semantic_palette["DOOR_BOMB"]
        elif door_type is None and edge_type == "soft_locked":
            door_type = semantic_palette["DOOR_SOFT"]
        elif door_type is None and edge_type == "boss_locked":
            door_type = semantic_palette["DOOR_BOSS"]
        elif door_type is None and edge_type == "stair":
            door_type = semantic_palette["STAIR"]

        if door_type is None:
            continue

        from_room = node_to_room.get(from_node)
        to_room = node_to_room.get(to_node)

        if from_room is None:
            logger.debug(
                "DOOR_TYPE_SKIP: Edge %d->%d has no from_room mapping (node %d unmapped)",
                from_node,
                to_node,
                from_node,
            )
            continue
        if to_room is None:
            logger.debug(
                "DOOR_TYPE_SKIP: Edge %d->%d has no to_room mapping (node %d unmapped)",
                from_node,
                to_node,
                to_node,
            )
            continue
        if from_room not in room_positions:
            logger.debug(
                "DOOR_TYPE_SKIP: from_room %s not in room_positions for edge %d->%d",
                from_room,
                from_node,
                to_node,
            )
            continue
        if to_room not in room_positions:
            logger.debug(
                "DOOR_TYPE_SKIP: to_room %s not in room_positions for edge %d->%d",
                to_room,
                from_node,
                to_node,
            )
            continue

        from_r, from_c = from_room
        to_r, to_c = to_room
        row_diff = abs(to_r - from_r)
        col_diff = abs(to_c - from_c)

        if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
            logger.debug(
                "DOOR_TYPE_SKIP: Rooms %s and %s for edge %d->%d are not physically adjacent "
                "(may be connected via stair/warp)",
                from_room,
                to_room,
                from_node,
                to_node,
            )
            continue

        from_offset = room_positions[from_room]
        to_offset = room_positions[to_room]
        door_positions = find_boundary_doors(
            grid=grid,
            from_offset=from_offset,
            to_offset=to_offset,
            from_room=from_room,
            to_room=to_room,
            room_height=room_height,
            room_width=room_width,
        )

        if door_positions:
            center_idx = len(door_positions) // 2
            dr, dc = door_positions[center_idx]
            valid_to_convert: Set[int] = {
                semantic_palette["FLOOR"],
                semantic_palette["DOOR_OPEN"],
            }
            if grid[dr, dc] in valid_to_convert:
                grid[dr, dc] = door_type
                logger.debug(
                    "Set CENTER door at (%d, %d) to %s for edge %d->%d (rooms %s<->%s)",
                    dr,
                    dc,
                    door_type_names.get(door_type, str(door_type)),
                    from_node,
                    to_node,
                    from_room,
                    to_room,
                )
            else:
                logger.debug(
                    "DOOR_TYPE_SKIP: Center position (%d, %d) has tile %d, not convertible for edge %d->%d",
                    dr,
                    dc,
                    grid[dr, dc],
                    from_node,
                    to_node,
                )
        else:
            logger.debug(
                "DOOR_TYPE_SKIP: No boundary doors found between rooms %s and %s for edge %d->%d",
                from_room,
                to_room,
                from_node,
                to_node,
            )
