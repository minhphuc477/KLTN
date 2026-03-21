"""Helpers for topology precheck, dead-end pruning, and prune undo flow."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple


def prune_dead_end_topology(
    *,
    gui: Any,
    current: Any,
    preserve_rooms: set,
    logger: Any,
    build_room_adjacency_fn: Callable[[Any, dict, dict], dict],
    node_has_critical_content_fn: Callable[[Any, Any], bool],
) -> List[Tuple[int, int]]:
    """Prune dead-end rooms when only topology mappings are available."""
    graph = getattr(current, "graph", None)
    room_to_node = dict(getattr(current, "room_to_node", {}) or {})
    if graph is None or not room_to_node:
        return []

    node_to_room = dict(getattr(current, "node_to_room", {}) or {v: k for k, v in room_to_node.items()})
    original_room_to_node = dict(room_to_node)
    removed_rooms: List[Tuple[int, int]] = []

    changed = True
    while changed:
        changed = False
        adjacency = build_room_adjacency_fn(graph, room_to_node, node_to_room)
        leaves = [
            room_pos
            for room_pos in list(room_to_node.keys())
            if len(adjacency.get(room_pos, set())) <= 1 and room_pos not in preserve_rooms
        ]
        if not leaves:
            break

        prunable = []
        for room_pos in leaves:
            node_id = room_to_node.get(room_pos)
            if node_id is None:
                prunable.append(room_pos)
                continue
            if not node_has_critical_content_fn(graph, node_id):
                prunable.append(room_pos)

        if not prunable:
            break

        for room_pos in prunable:
            room_to_node.pop(room_pos, None)
            removed_rooms.append(room_pos)
            changed = True

        kept_rooms = set(room_to_node.keys())
        node_to_room = {
            node_id: room_pos for node_id, room_pos in node_to_room.items() if room_pos in kept_rooms
        }

    if not removed_rooms:
        return []

    removed_nodes = [original_room_to_node[r] for r in removed_rooms if r in original_room_to_node]
    new_graph = graph
    try:
        if removed_nodes and hasattr(graph, "copy"):
            new_graph = graph.copy()
            new_graph.remove_nodes_from([n for n in set(removed_nodes) if n in new_graph])
    except Exception:
        logger.debug("Failed to remove pruned nodes from graph; keeping original graph", exc_info=True)
        new_graph = graph

    current.graph = new_graph
    current.room_to_node = room_to_node
    current.node_to_room = node_to_room
    return removed_rooms


def run_prechecks_and_optional_prune(
    *,
    gui: Any,
    current: Any,
    logger: Any,
    np_module: Any,
    semantic_palette: dict,
    action_deltas: dict,
    topology_has_path_fn: Callable[[Any, Any, Any], bool],
    min_locked_between_fn: Callable[[Any, Any, Any], float],
    walkable_grid_reachable_fn: Callable[[Any, tuple, tuple, set, dict], bool],
    node_has_small_key_fn: Callable[[dict], bool],
    room_for_global_position_fn: Callable[[Optional[Tuple[int, int]], dict], Optional[Tuple[int, int]]],
    zelda_dungeon_adapter: Any,
    capture_snapshot_fn: Callable[[Any, str], None],
    update_env_topology_view_fn: Callable[[Any], None],
    prune_dead_end_topology_fn: Callable[[Any, set], List[Tuple[int, int]]],
) -> Tuple[bool, Optional[str]]:
    """Run solve prechecks and optional dead-end topology pruning."""
    if not gui.feature_flags.get("enable_prechecks", False):
        return True, None

    if not getattr(gui, "env", None):
        return False, "PRECHECK_FAIL: Environment not initialized"

    start = getattr(gui.env, "start_pos", None)
    goal = getattr(gui.env, "goal_pos", None)
    if start is None or goal is None:
        return False, "PRECHECK_FAIL: Missing start/goal position"

    graph = getattr(current, "graph", None)
    room_positions = dict(getattr(current, "room_positions", {}) or {})
    room_to_node = dict(getattr(current, "room_to_node", {}) or {})
    start_room = room_for_global_position_fn(start, room_positions)
    goal_room = room_for_global_position_fn(goal, room_positions)

    if graph is not None and room_positions and room_to_node:
        start_node = room_to_node.get(start_room) if start_room is not None else None
        goal_node = room_to_node.get(goal_room) if goal_room is not None else None
        if start_node is not None and goal_node is not None:
            try:
                if not topology_has_path_fn(graph, start_node, goal_node):
                    return False, "PRECHECK_FAIL: Start and goal are disconnected in topology"
            except Exception:
                logger.debug("Topology connectivity precheck failed open", exc_info=True)

            try:
                min_locked = min_locked_between_fn(graph, start_node, goal_node)
                if min_locked != float("inf"):
                    tile_key_count = int((gui.env.grid == semantic_palette["KEY_SMALL"]).sum())
                    graph_key_count = 0
                    for _, attrs in graph.nodes(data=True):
                        if node_has_small_key_fn(attrs):
                            graph_key_count += 1
                    key_count = max(tile_key_count, graph_key_count)
                    if key_count < int(min_locked):
                        return (
                            False,
                            f"PRECHECK_FAIL: Insufficient small keys (need {int(min_locked)}, have {key_count})",
                        )
            except Exception:
                logger.debug("Locked-door key-count precheck failed open", exc_info=True)
    else:
        grid = getattr(gui.env, "grid", None)
        if isinstance(grid, np_module.ndarray):
            blocked = {semantic_palette["WALL"], semantic_palette["VOID"]}
            if not walkable_grid_reachable_fn(grid, start, goal, blocked, action_deltas):
                return False, "PRECHECK_FAIL: Start and goal disconnected on walkable grid"

    if not gui.feature_flags.get("auto_prune_on_precheck", False):
        return True, "Precheck passed"

    preserve_rooms = {rp for rp in (start_room, goal_room) if rp is not None}
    if not preserve_rooms:
        return True, "Precheck passed (prune skipped: start/goal room unknown)"

    removed_rooms: List[Tuple[int, int]] = []
    rooms = getattr(current, "rooms", None)
    if isinstance(rooms, dict) and rooms:
        pruned_rooms, removed_rooms = zelda_dungeon_adapter.prune_dead_ends(dict(rooms), preserve=preserve_rooms)
        if removed_rooms:
            capture_snapshot_fn(current, "auto_prune_on_precheck")
            original_room_to_node = dict(getattr(current, "room_to_node", {}) or {})
            original_node_to_room = dict(
                getattr(current, "node_to_room", {}) or {v: k for k, v in original_room_to_node.items()}
            )

            current.rooms = pruned_rooms
            current.room_to_node = {
                room_pos: node_id
                for room_pos, node_id in original_room_to_node.items()
                if room_pos in pruned_rooms
            }

            removed_nodes = [original_room_to_node[rp] for rp in removed_rooms if rp in original_room_to_node]
            try:
                if graph is not None and hasattr(graph, "copy"):
                    current.graph = graph.copy()
                    current.graph.remove_nodes_from([n for n in set(removed_nodes) if n in current.graph])
            except Exception:
                logger.debug("Failed to prune nodes from graph in room-based prune", exc_info=True)

            current.node_to_room = {
                node_id: room_pos
                for node_id, room_pos in original_node_to_room.items()
                if room_pos in current.room_to_node
            }
            for room_pos, room in current.rooms.items():
                room.graph_node_id = current.room_to_node.get(room_pos)
    else:
        capture_snapshot_fn(current, "auto_prune_on_precheck_topology")
        removed_rooms = prune_dead_end_topology_fn(current, preserve_rooms=preserve_rooms)
        if not removed_rooms:
            gui._precheck_snapshot = None

    if removed_rooms:
        update_env_topology_view_fn(current)
        return True, f"Precheck passed; pruned {len(removed_rooms)} dead-end room(s)"
    return True, "Precheck passed (no dead-end rooms pruned)"


def undo_prune(*, gui: Any, current: Any, logger: Any, update_env_topology_view_fn: Callable[[Any], None]) -> None:
    """Undo the latest prune snapshot if one exists."""
    if not hasattr(gui, "_precheck_snapshot") or not gui._precheck_snapshot:
        gui._set_message("No prune snapshot to undo", 2.0)
        return

    snap = gui._precheck_snapshot
    rooms_snapshot = snap.get("rooms")
    if isinstance(rooms_snapshot, dict):
        current.rooms = dict(rooms_snapshot)
    if "room_to_node" in snap:
        current.room_to_node = dict(snap.get("room_to_node", {}))
    if "node_to_room" in snap:
        current.node_to_room = dict(snap.get("node_to_room", {}))
    if "graph" in snap and snap.get("graph") is not None:
        current.graph = snap.get("graph")

    if isinstance(getattr(current, "rooms", None), dict):
        for pos, room in current.rooms.items():
            room.graph_node_id = (getattr(current, "room_to_node", {}) or {}).get(pos)

    gui._precheck_snapshot = None
    update_env_topology_view_fn(current)
    gui._set_message("Undo: restored topology before pruning", 3.0)
    logger.info("Undo prune: restored previous topology snapshot")
