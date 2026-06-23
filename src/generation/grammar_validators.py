"""Wave-3 grammar validation helpers extracted from grammar monolith."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value or "")).upper()


def _node_type_name(node: Any) -> str:
    return _enum_name(getattr(node, "node_type", None))


def _progression_reachable_nodes(
    graph: Any,
    start_id: int,
    *,
    excluded_edges: Set[Tuple[int, int]],
) -> Set[int]:
    """Reachability that refuses to walk through unresolved resource gates."""
    if start_id not in getattr(graph, "nodes", {}):
        return set()

    reachable: Set[int] = {start_id}
    changed = True

    def _resources() -> Tuple[Set[int], Set[str], int, int]:
        key_ids: Set[int] = set()
        items: Set[str] = set()
        small_keys = 0
        tokens = 0
        for node_id in reachable:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            node_type = _node_type_name(node)
            key_id = getattr(node, "key_id", None)
            if node_type in {"KEY", "BIG_KEY"} and key_id is not None:
                key_ids.add(int(key_id))
            if node_type == "KEY":
                small_keys += 1
            item_type = getattr(node, "item_type", None)
            if node_type == "ITEM" and item_type:
                items.add(str(item_type))
            drops_resource = getattr(node, "drops_resource", None)
            if node_type == "RESOURCE_FARM" and drops_resource:
                items.add(str(drops_resource))
            if node_type == "TOKEN":
                tokens += 1
        return key_ids, items, small_keys, tokens

    def _node_open(node_id: int, key_ids: Set[int]) -> bool:
        node = graph.nodes.get(node_id)
        if node is None:
            return False
        node_type = _node_type_name(node)
        key_id = getattr(node, "key_id", None)
        if node_type in {"LOCK", "BOSS_DOOR"} and key_id is not None:
            return int(key_id) in key_ids
        return True

    def _edge_open(edge: Any, key_ids: Set[int], items: Set[str], small_keys: int, tokens: int) -> bool:
        edge_type = _enum_name(getattr(edge, "edge_type", None))
        key_required = getattr(edge, "key_required", None)
        requires_key_count = int(max(0, getattr(edge, "requires_key_count", 0) or 0))
        if edge_type in {"LOCKED", "BOSS_LOCKED"}:
            if key_required is not None and int(key_required) not in key_ids:
                return False
            if edge_type == "LOCKED" and requires_key_count > 0 and small_keys < requires_key_count:
                return False
            if key_required is None and requires_key_count <= 0:
                return False
        if requires_key_count > 0 and small_keys < requires_key_count:
            return False
        item_required = getattr(edge, "item_required", None)
        if edge_type == "ITEM_GATE" and item_required and str(item_required) not in items:
            return False
        token_count = int(max(0, getattr(edge, "token_count", 0) or 0))
        if edge_type == "MULTI_LOCK" and token_count > 0 and tokens < token_count:
            return False
        switches_required = set(getattr(edge, "switches_required", []) or [])
        if edge_type == "STATE_BLOCK" and not switches_required.issubset(reachable):
            return False
        return True

    while changed:
        changed = False
        key_ids, items, small_keys, tokens = _resources()
        for edge in getattr(graph, "edges", []):
            source = int(getattr(edge, "source"))
            target = int(getattr(edge, "target"))
            if (source, target) in excluded_edges:
                continue
            if source not in reachable or target in reachable:
                continue
            if not _node_open(target, key_ids):
                continue
            if not _edge_open(edge, key_ids, items, small_keys, tokens):
                continue
            reachable.add(target)
            changed = True
    return reachable


def _find_forward_path(
    adjacency: Dict[int, List[int]],
    start_id: int,
    goal_id: int,
) -> Optional[List[int]]:
    if start_id == goal_id:
        return [start_id]

    visited = {start_id}
    queue = deque([(start_id, [start_id])])
    while queue:
        current, path = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if neighbor == goal_id:
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))
    return None


def _find_forward_paths(
    adjacency: Dict[int, List[int]],
    start_id: int,
    goal_id: int,
    *,
    max_depth: int,
    max_paths: int = 128,
) -> List[List[int]]:
    """Enumerate bounded simple forward paths for validation-only graph checks."""
    if start_id == goal_id:
        return [[start_id]]

    paths: List[List[int]] = []
    queue = deque([(start_id, [start_id])])
    while queue and len(paths) < max_paths:
        current, path = queue.popleft()
        if len(path) > max_depth:
            continue
        for neighbor in adjacency.get(current, []):
            if neighbor in path:
                continue
            new_path = path + [neighbor]
            if neighbor == goal_id:
                paths.append(new_path)
                if len(paths) >= max_paths:
                    break
                continue
            queue.append((neighbor, new_path))
    return paths


def validate_skill_chains(graph: Any) -> bool:
    """
    Ensure tutorial sequences are properly ordered.

    Returns:
        True if all skill chains have proper difficulty progression.
    """
    graph.sanitize()
    tutorial_nodes = [n for n in graph.nodes.values() if getattr(n, "is_tutorial", False)]
    if not tutorial_nodes:
        return True

    forward_adj = graph.get_forward_adjacency_map()
    item_ids = [
        node.id
        for node in graph.nodes.values()
        if getattr(getattr(node, "node_type", None), "name", "") == "ITEM"
    ]
    climax_ids = [
        node.id
        for node in graph.nodes.values()
        if getattr(getattr(node, "node_type", None), "name", "") in {"BOSS_DOOR", "BOSS", "GOAL"}
    ]

    for tutorial in tutorial_nodes:
        if not any(_find_forward_path(forward_adj, item_id, tutorial.id) for item_id in item_ids):
            logger.warning(
                "Tutorial node %s is not reachable from any ITEM node via forward progression",
                tutorial.id,
            )
            return False

        candidate_paths: List[List[int]] = []
        for climax_id in climax_ids:
            paths = _find_forward_paths(
                forward_adj,
                tutorial.id,
                climax_id,
                max_depth=max(2, len(graph.nodes) + 1),
            )
            candidate_paths.extend(path for path in paths if len(path) >= 2)
        if not candidate_paths:
            logger.warning(
                "Tutorial node %s does not lead to any climax target via forward progression",
                tutorial.id,
            )
            return False

        valid_chain_found = False
        for path in sorted(candidate_paths, key=len):
            path_nodes = [graph.get_node(node_id) for node_id in path[1:]]
            combat_node = next(
                (node for node in path_nodes if getattr(getattr(node, "node_type", None), "name", "") == "COMBAT_PUZZLE"),
                None,
            )
            complex_node = next(
                (node for node in path_nodes if getattr(getattr(node, "node_type", None), "name", "") == "COMPLEX_PUZZLE"),
                None,
            )
            if combat_node is None or complex_node is None:
                continue
            if path.index(combat_node.id) >= path.index(complex_node.id):
                continue
            if combat_node.difficulty > complex_node.difficulty:
                continue
            valid_chain_found = True
            break
        if not valid_chain_found:
            logger.warning(
                "Tutorial node %s does not lead to a valid COMBAT_PUZZLE -> COMPLEX_PUZZLE skill chain",
                tutorial.id,
            )
            return False

    return True


def validate_battery_reachability(graph: Any) -> bool:
    """
    Ensure all switches in battery are reachable before locked door.

    Returns:
        True if all battery patterns are valid.
    """
    graph.sanitize()
    start = graph.get_start_node()
    if not start:
        return True

    battery_edges = [e for e in graph.edges if e.battery_id is not None]

    for edge in battery_edges:
        required_switches = edge.switches_required
        for switch_id in required_switches:
            reachable = graph.get_reachable_nodes(
                start.id,
                excluded_edges={(edge.source, edge.target)},
            )
            if switch_id not in reachable:
                logger.warning(
                    "Battery switch %s not reachable before lock %s->%s",
                    switch_id,
                    edge.source,
                    edge.target,
                )
                return False

    return True


def validate_resource_loops(graph: Any) -> bool:
    """
    Ensure resource farms are reachable before their gates.

    Returns:
        True if all resource farms are properly placed.
    """
    graph.sanitize()
    start = graph.get_start_node()
    if not start:
        return True

    farms_by_resource: Dict[str, List[Any]] = {}
    for farm in [
        n
        for n in graph.nodes.values()
        if getattr(getattr(n, "node_type", None), "name", "") == "RESOURCE_FARM"
    ]:
        resource = farm.drops_resource
        if not resource:
            continue
        farms_by_resource.setdefault(str(resource), []).append(farm)

    gates = [e for e in graph.edges if getattr(e, "item_required", None)]
    for gate in gates:
        resource = str(gate.item_required)
        resource_farms = farms_by_resource.get(resource, [])
        if not resource_farms:
            logger.warning(
                "Gate %s->%s requires resource %s but no matching farm exists",
                gate.source,
                gate.target,
                resource,
            )
            return False
        reachable = _progression_reachable_nodes(
            graph,
            start.id,
            excluded_edges={(gate.source, gate.target)},
        )
        if not any(farm.id in reachable for farm in resource_farms):
            logger.warning(
                "No %s resource farm is reachable before gate %s->%s",
                resource,
                gate.source,
                gate.target,
            )
            return False

    return True
