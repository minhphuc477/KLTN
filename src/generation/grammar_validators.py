"""Wave-3 grammar validation helpers extracted from grammar monolith."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
            path = _find_forward_path(forward_adj, tutorial.id, climax_id)
            if path and len(path) >= 2:
                candidate_paths.append(path)
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

    farms = [
        n
        for n in graph.nodes.values()
        if getattr(getattr(n, "node_type", None), "name", "") == "RESOURCE_FARM"
    ]

    for farm in farms:
        resource = farm.drops_resource
        if not resource:
            continue

        gates = [e for e in graph.edges if e.item_required == resource]
        for gate in gates:
            reachable = graph.get_reachable_nodes(
                start.id,
                excluded_edges={(gate.source, gate.target)},
            )
            if farm.id not in reachable:
                logger.warning(
                    "Resource farm %s (%s) not reachable before gate %s->%s",
                    farm.id,
                    resource,
                    gate.source,
                    gate.target,
                )
                return False

    return True
