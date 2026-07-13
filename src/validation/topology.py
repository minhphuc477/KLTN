"""Descriptive topology evidence for generated mission graphs.

These values are exact graph statistics, not proxies for enjoyment. They stay
separate from the hard progression oracle because trees, loops, and choke
points can all be valid design choices.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import networkx as nx


def _role(attrs: Mapping[str, Any]) -> str:
    value = attrs.get("node_type", attrs.get("type", attrs.get("label", "")))
    if hasattr(value, "name"):
        value = value.name
    return str(value).strip().upper().split(".")[-1]


def _unique_role_node(graph: nx.Graph, roles: set[str]) -> Optional[Any]:
    matches = []
    for node_id, raw_attrs in graph.nodes(data=True):
        attrs = dict(raw_attrs)
        semantic_match = _role(attrs) in roles
        if "START" in roles:
            semantic_match = semantic_match or bool(attrs.get("is_start", False))
        if roles & {"GOAL", "TRIFORCE"}:
            semantic_match = semantic_match or bool(
                attrs.get("is_goal", False)
                or attrs.get("has_goal", False)
                or attrs.get("is_triforce", False)
            )
        if semantic_match:
            matches.append(node_id)
    return matches[0] if len(matches) == 1 else None


def evaluate_graph_topology_characteristics(
    graph: nx.Graph,
    solution_path: Sequence[Any] = (),
) -> Dict[str, Any]:
    """Return connectivity, redundancy, and choke-point characteristics.

    Direction is removed only for this spatial-topology view. Resource order
    and one-way traversal remain the exact progression oracle's responsibility.
    """
    physical = nx.Graph(graph)
    node_count = int(physical.number_of_nodes())
    edge_count = int(physical.number_of_edges())
    if node_count == 0:
        return {
            "topology_characteristics_applicable": False,
            "topology_node_count": 0,
            "topology_edge_count": 0,
            "topology_failure_reason": "empty mission graph",
        }

    component_count = int(nx.number_connected_components(physical))
    cycle_rank = int(max(0, edge_count - node_count + component_count))
    articulation_nodes = (
        set(nx.articulation_points(physical)) if node_count >= 3 else set()
    )
    biconnected = list(nx.biconnected_components(physical))
    branch_nodes = {
        node_id for node_id, degree in physical.degree() if int(degree) >= 3
    }
    leaf_nodes = {
        node_id for node_id, degree in physical.degree() if int(degree) == 1
    }

    start = _unique_role_node(graph, {"START", "S"})
    goal = _unique_role_node(graph, {"GOAL", "TRIFORCE", "T"})
    start_goal_connected = bool(
        start is not None and goal is not None and nx.has_path(physical, start, goal)
    )
    node_disjoint_path_count: Optional[int] = None
    mandatory_checkpoints: list[Any] = []
    if start_goal_connected:
        node_disjoint_path_count = int(nx.node_connectivity(physical, start, goal))
        for node_id in articulation_nodes:
            if node_id in {start, goal}:
                continue
            reduced = physical.copy()
            reduced.remove_node(node_id)
            if not nx.has_path(reduced, start, goal):
                mandatory_checkpoints.append(node_id)

    checkpoint_set = set(mandatory_checkpoints)
    checkpoint_positions = [
        int(index)
        for index, node_id in enumerate(solution_path)
        if node_id in checkpoint_set
    ]
    return {
        "topology_characteristics_applicable": True,
        "topology_scope": "undirected_spatial_skeleton",
        "topology_node_count": node_count,
        "topology_edge_count": edge_count,
        "topology_component_count": component_count,
        "topology_connected": component_count == 1,
        "topology_cycle_rank": cycle_rank,
        "topology_cycle_rank_normalized": float(cycle_rank / max(1, node_count)),
        "topology_branch_node_count": int(len(branch_nodes)),
        "topology_leaf_node_count": int(len(leaf_nodes)),
        "topology_articulation_count": int(len(articulation_nodes)),
        "topology_articulation_ratio": float(len(articulation_nodes) / node_count),
        "topology_biconnected_component_count": int(len(biconnected)),
        "topology_start_goal_connected": start_goal_connected,
        "topology_start_goal_node_disjoint_path_count": node_disjoint_path_count,
        "topology_mandatory_checkpoint_count": int(len(mandatory_checkpoints)),
        "topology_mandatory_checkpoints": sorted(
            mandatory_checkpoints,
            key=lambda value: (type(value).__name__, repr(value)),
        ),
        "topology_solution_checkpoint_positions": checkpoint_positions,
    }


__all__ = ["evaluate_graph_topology_characteristics"]
