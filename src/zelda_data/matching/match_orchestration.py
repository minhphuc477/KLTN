"""Room-to-graph matching orchestration extracted from zelda_core."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import networkx as nx


def match_rooms_to_graph(
    rooms: Dict[Tuple[int, int], Any],
    graph: nx.DiGraph,
    dungeon_cls: Any,
    logger: Any,
    normalize_graph_fn,
    build_room_adjacency_fn,
    find_entrance_room_fn,
    match_rooms_to_nodes_bfs_fn,
    find_room_at_distance_fn,
    find_farthest_dead_end_fn,
):
    """Orchestrate room-to-graph matching while preserving legacy behavior."""
    num_graph_nodes = len(graph.nodes())
    num_rooms = len(rooms)

    if num_graph_nodes != num_rooms:
        mismatch_count = abs(num_graph_nodes - num_rooms)
        if num_graph_nodes > num_rooms:
            logger.info(
                "GRAPH-ROOM MISMATCH: Graph has %d nodes but %d physical rooms. "
                "%d virtual nodes will use fallback item placement.",
                num_graph_nodes,
                num_rooms,
                mismatch_count,
            )
        else:
            logger.info(
                "GRAPH-ROOM MISMATCH: Graph has %d nodes but %d physical rooms. "
                "%d rooms will lack graph-based annotations.",
                num_graph_nodes,
                num_rooms,
                mismatch_count,
            )

    dungeon = dungeon_cls(dungeon_id="", rooms=rooms, graph=graph)

    start_node = None
    start_pointer_node = None
    graph_triforce_node = None
    _boss_node = None

    for node, attrs in graph.nodes(data=True):
        if attrs.get("is_start"):
            start_node = node
            if attrs.get("is_start_pointer", False):
                start_pointer_node = node
        if attrs.get("is_triforce"):
            graph_triforce_node = node
        if attrs.get("is_boss"):
            _boss_node = node

    actual_start_node = start_node
    if start_pointer_node is not None:
        neighbors = list(graph.successors(start_pointer_node)) + list(graph.predecessors(start_pointer_node))
        if neighbors:
            actual_start_node = neighbors[0]
            logger.info(
                "START_POINTER: Node %d (label='s') is a pointer. Actual first room node: %d",
                start_pointer_node,
                actual_start_node,
            )
        graph.nodes[start_pointer_node]["is_virtual"] = True
        graph.nodes[start_pointer_node]["is_start_pointer"] = True

    normalize_graph_fn(graph)

    room_adjacency = build_room_adjacency_fn(rooms)
    seed_room_pos = find_entrance_room_fn(rooms)

    if seed_room_pos is None:
        for pos, room in rooms.items():
            if getattr(room, "has_stair", False):
                seed_room_pos = pos
                break

    if seed_room_pos is None:
        max_doors = 0
        for pos, room in rooms.items():
            door_count = sum(room.doors.values())
            if door_count > max_doors:
                max_doors = door_count
                seed_room_pos = pos

    room_to_node, node_to_room = match_rooms_to_nodes_bfs_fn(
        rooms, room_adjacency, graph, seed_room_pos, actual_start_node
    )

    if start_pointer_node is not None and start_pointer_node in node_to_room:
        pointer_room = node_to_room.pop(start_pointer_node, None)
        if pointer_room is not None and room_to_node.get(pointer_room) == start_pointer_node:
            room_to_node.pop(pointer_room, None)
            logger.info(
                "START_POINTER_FIX: Removed room %s assignment from pointer node %d",
                pointer_room,
                start_pointer_node,
            )

    for room_pos, node_id in room_to_node.items():
        rooms[room_pos].graph_node_id = node_id
        node_data = graph.nodes.get(node_id, {})
        rooms[room_pos].node_label = node_data.get("label", "")

    start_room_pos: Optional[Tuple[int, int]] = None
    effective_start = actual_start_node if actual_start_node is not None else start_node
    if effective_start is not None:
        start_room_pos = node_to_room.get(effective_start)
        if start_room_pos:
            logger.debug(
                "START_FROM_GRAPH: Graph node %d mapped to room %s (pointer=%s)",
                effective_start,
                start_room_pos,
                start_pointer_node is not None,
            )

    if start_room_pos is None:
        logger.warning(
            "START_FALLBACK: Could not find room for graph start_node=%s. "
            "Using seed room %s as fallback.",
            start_node,
            seed_room_pos,
        )
        start_room_pos = seed_room_pos

    if start_room_pos:
        rooms[start_room_pos].is_start = True
        dungeon.start_pos = start_room_pos

    triforce_room_pos = None
    graph_path_length = 0

    if graph_triforce_node is not None:
        triforce_room_pos = node_to_room.get(graph_triforce_node)
        if triforce_room_pos:
            logger.debug(
                "TRIFORCE_FROM_GRAPH: Graph node %d (is_triforce=True) mapped to room %s",
                graph_triforce_node,
                triforce_room_pos,
            )

    if triforce_room_pos is None and graph_triforce_node is not None and start_node is not None:
        try:
            graph_path = nx.shortest_path(graph.to_undirected(), start_node, graph_triforce_node)
            graph_path_length = len(graph_path) - 1
        except nx.NetworkXNoPath:
            graph_path_length = len(rooms) // 2

    if triforce_room_pos is None and graph_path_length > 0:
        triforce_room_pos = find_room_at_distance_fn(rooms, room_adjacency, start_room_pos, graph_path_length)

    if triforce_room_pos is None and start_room_pos:
        triforce_room_pos = find_farthest_dead_end_fn(rooms, start_room_pos)

    if triforce_room_pos:
        rooms[triforce_room_pos].has_triforce = True
        dungeon.triforce_pos = triforce_room_pos

    return dungeon
