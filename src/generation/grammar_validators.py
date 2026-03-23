"""Wave-3 grammar validation helpers extracted from grammar monolith."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_skill_chains(graph: Any) -> bool:
    """
    Ensure tutorial sequences are properly ordered.

    Returns:
        True if all skill chains have proper difficulty progression.
    """
    graph.sanitize()
    tutorial_nodes = [n for n in graph.nodes.values() if getattr(n, "is_tutorial", False)]
    pedagogical_types = {"COMBAT_PUZZLE", "COMPLEX_PUZZLE"}

    for tutorial in tutorial_nodes:
        successors = [
            n
            for n in graph.get_successors(tutorial.id, depth=3)
            if getattr(getattr(n, "node_type", None), "name", "") in pedagogical_types
        ]
        if len(successors) < 2:
            continue

        successors.sort(key=lambda n: graph.get_shortest_path_length(tutorial.id, n.id))
        first, second = successors[0], successors[1]
        if first.difficulty > second.difficulty:
            logger.warning(
                "Skill chain from %s has improper difficulty progression",
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
