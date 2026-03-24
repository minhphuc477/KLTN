"""Validation and pruning helpers extracted from zelda_core."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

from src.zelda_data.matching.topology_utils import build_room_adjacency

RoomPos = Tuple[int, int]


def precheck_dungeon(
    dungeon: Any,
    parse_edge_type_tokens_fn: Callable[[str, str], Set[str]],
    parse_node_label_tokens_fn: Callable[[str], List[str]],
    semantic_palette: Dict[str, int],
    logger: Any,
) -> Tuple[bool, Optional[str]]:
    """Run lightweight validity checks before expensive solving."""
    if dungeon is None:
        return False, "No dungeon data"

    if dungeon.start_pos is None:
        return False, "PRECHECK_FAIL: Missing start position"
    if dungeon.triforce_pos is None:
        return False, "PRECHECK_FAIL: Missing triforce position"

    graph = getattr(dungeon, "graph", None)
    if graph is None or len(graph) == 0:
        return False, "PRECHECK_FAIL: No topology graph available"

    start_node = None
    triforce_node = None
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("is_start"):
            start_node = node_id
        if attrs.get("is_triforce"):
            triforce_node = node_id

    room_to_node = getattr(dungeon, "room_to_node", {}) or {}
    if start_node is None:
        start_node = room_to_node.get(dungeon.start_pos)
    if triforce_node is None:
        triforce_node = room_to_node.get(dungeon.triforce_pos)

    if start_node is None or triforce_node is None:
        if len(graph.nodes()) < 2:
            return False, "PRECHECK_FAIL: Topology too small"
    else:
        try:
            if not nx.has_path(graph.to_undirected(), start_node, triforce_node):
                return False, "PRECHECK_FAIL: Start and triforce disconnected in topology"
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning(
                "precheck_dungeon: topology path check failed, continuing best-effort: %s",
                exc,
                exc_info=True,
            )

    def _locked_cost_from_payload(data: Any) -> int:
        if not isinstance(data, dict):
            return 0
        values = list(data.values())
        if values and all(isinstance(v, dict) for v in values):
            return min(_locked_cost_from_payload(v) for v in values)
        label = data.get("label", "")
        edge_types = parse_edge_type_tokens_fn(label=label, edge_type=data.get("edge_type", ""))
        return 1 if any(et in ("locked", "key_locked") for et in edge_types) else 0

    try:
        import heapq

        is_directed_graph = bool(getattr(graph, "is_directed", lambda: False)())

        def _neighbors(node_id: Any) -> Set[Any]:
            if is_directed_graph and hasattr(graph, "successors") and hasattr(graph, "predecessors"):
                return set(graph.successors(node_id)) | set(graph.predecessors(node_id))
            if hasattr(graph, "neighbors"):
                return set(graph.neighbors(node_id))
            return set()

        def _edge_payload(u: Any, v: Any) -> Dict[str, Any]:
            payload = graph.get_edge_data(u, v, {})
            if not payload and is_directed_graph:
                payload = graph.get_edge_data(v, u, {})
            return payload if isinstance(payload, dict) else {}

        def min_locked_between(source, target):
            dist = {source: 0}
            pq = [(0, source)]
            while pq:
                d, u = heapq.heappop(pq)
                if u == target:
                    return d
                if d != dist.get(u, 1e9):
                    continue
                for v in _neighbors(u):
                    c = _locked_cost_from_payload(_edge_payload(u, v))
                    nd = d + c
                    if nd < dist.get(v, 1e9):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            return 1e9

        if start_node is not None and triforce_node is not None:
            min_locked = min_locked_between(start_node, triforce_node)

            def _has_small_key_node(attrs: Dict[str, Any]) -> bool:
                if bool(attrs.get("is_boss_key") or attrs.get("has_boss_key")):
                    return False
                if bool(attrs.get("is_key") or attrs.get("has_key")):
                    return True

                label = str(attrs.get("label", ""))
                tokens = parse_node_label_tokens_fn(label)
                for tok in tokens:
                    if tok == "k" or tok.lower() == "key":
                        return True
                return False

            tile_key_count = 0
            for room in dungeon.rooms.values():
                semantic_grid = getattr(room, "semantic_grid", None)
                if semantic_grid is not None:
                    tile_key_count += int((semantic_grid == semantic_palette["KEY"]).sum())

            graph_key_count = 0
            for _, attrs in graph.nodes(data=True):
                if _has_small_key_node(attrs):
                    graph_key_count += 1

            key_count = max(tile_key_count, graph_key_count)
            if key_count < min_locked:
                return False, f"PRECHECK_FAIL: Insufficient small keys (need {min_locked}, have {key_count})"
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning(
            "precheck_dungeon: locked-door lower-bound check failed, continuing best-effort: %s",
            exc,
            exc_info=True,
        )

    return True, None


def prune_dead_ends(
    rooms: Dict[RoomPos, Any],
    parse_node_label_tokens_fn: Callable[[str], List[str]],
    semantic_palette: Dict[str, int],
    preserve: Optional[Set[RoomPos]] = None,
) -> Tuple[Dict[RoomPos, Any], List[RoomPos]]:
    """Iteratively remove leaf rooms that are not progression-critical."""

    def _room_has_critical_label(room: Any) -> bool:
        label = str(getattr(room, "node_label", "") or "")
        if not label:
            return False
        tokens = set(parse_node_label_tokens_fn(label))
        critical_tokens = {
            "s",
            "t",
            "b",
            "k",
            "K",
            "I",
            "start",
            "triforce",
            "boss",
            "key",
            "boss_key",
            "item",
        }
        return len(tokens.intersection(critical_tokens)) > 0

    preserve = set(preserve or [])
    pruned = dict(rooms)
    removed: List[RoomPos] = []
    changed = True
    while changed:
        changed = False
        adjacency = build_room_adjacency(pruned)
        leaves = [pos for pos, neighbors in adjacency.items() if len(neighbors) <= 1 and pos not in preserve]
        for pos in leaves:
            room = pruned.get(pos)
            if room is None:
                continue

            has_critical_item = False
            semantic_grid = getattr(room, "semantic_grid", None)
            if semantic_grid is not None:
                has_critical_item = bool(
                    (semantic_grid == semantic_palette["KEY"]).any()
                    or (semantic_grid == semantic_palette["KEY_BOSS"]).any()
                    or (semantic_grid == semantic_palette["KEY_ITEM"]).any()
                )

            if (
                room.has_triforce
                or room.has_boss
                or room.is_start
                or has_critical_item
                or _room_has_critical_label(room)
            ):
                continue

            pruned.pop(pos, None)
            removed.append(pos)
            changed = True

    return pruned, removed
