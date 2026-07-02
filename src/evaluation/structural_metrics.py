"""Structural graph metrics used by topology search and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass
class StructuralTopologyMetrics:
    cyclomatic_complexity: float
    branching_factor: float
    dead_end_ratio: float


def _node_role(mission_graph: nx.Graph, node: object) -> str:
    """Best-effort normalized role string for NetworkX or MissionGraph-derived nodes."""
    data = mission_graph.nodes[node] if node in mission_graph.nodes else {}
    raw_values = [
        data.get("node_type") if isinstance(data, dict) else None,
        data.get("type") if isinstance(data, dict) else None,
        data.get("label") if isinstance(data, dict) else None,
        node,
    ]
    tokens = []
    for value in raw_values:
        if value is None:
            continue
        name = getattr(value, "name", None)
        tokens.append(str(name if name is not None else value).strip().lower())
    joined = " ".join(tokens)
    if "start" in joined or "entry" in joined:
        return "start"
    if "goal" in joined or "triforce" in joined:
        return "goal"
    return ""


def compute_cyclomatic_complexity(mission_graph: nx.Graph) -> float:
    """Compute cyclomatic complexity M = E - N + P."""
    if mission_graph is None:
        return 0.0
    n_nodes = int(mission_graph.number_of_nodes())
    n_edges = int(mission_graph.number_of_edges())
    if n_nodes <= 0:
        return 0.0
    n_components = int(nx.number_connected_components(mission_graph.to_undirected()))
    complexity = float(n_edges - n_nodes + n_components)
    return max(0.0, complexity)


def compute_branching_factor(mission_graph: nx.Graph) -> float:
    """Compute average outgoing connectivity over non-terminal graph nodes."""
    if mission_graph is None or mission_graph.number_of_nodes() <= 0:
        return 0.0

    if mission_graph.is_directed():
        branch_degrees = [
            float(mission_graph.out_degree(n))
            for n in mission_graph.nodes()
            if _node_role(mission_graph, n) != "goal"
        ]
    else:
        branch_degrees = [
            float(mission_graph.degree(n))
            for n in mission_graph.nodes()
            if _node_role(mission_graph, n) not in {"start", "goal"}
        ]

    if not branch_degrees:
        return 0.0
    return float(sum(branch_degrees) / len(branch_degrees))


def compute_path_linearity(
    mission_graph: nx.Graph,
    path_nodes: list[object] | tuple[object, ...],
) -> float:
    """Measure critical-path corridor structure without rewarding dead-end padding."""
    if mission_graph is None or len(path_nodes) < 2:
        return 0.0
    physical = nx.Graph(mission_graph)
    ordered_path = [node for node in path_nodes if node in physical]
    path_set = set(ordered_path)
    path_edges = {
        frozenset((source, target))
        for source, target in zip(ordered_path[:-1], ordered_path[1:])
    }
    path_chords = sum(
        1
        for source, target in physical.subgraph(path_set).edges()
        if frozenset((source, target)) not in path_edges
    )
    reconnecting_branches = 0
    off_path = physical.subgraph(set(physical.nodes()) - path_set)
    for component in nx.connected_components(off_path):
        attachments = {
            neighbor
            for node in component
            for neighbor in physical.neighbors(node)
            if neighbor in path_set
        }
        reconnecting_branches += max(0, len(attachments) - 1)
    route_choice_pressure = min(
        1.0,
        float(path_chords + reconnecting_branches)
        / float(max(1, len(ordered_path) - 1)),
    )
    components = max(1, nx.number_connected_components(physical))
    cycle_rank = max(
        0,
        int(physical.number_of_edges())
        - int(physical.number_of_nodes())
        + int(components),
    )
    cycle_pressure = min(
        1.0,
        float(cycle_rank) / float(max(1, physical.number_of_nodes())),
    )
    return float(
        max(
            0.0,
            min(
                1.0,
                1.0 - (0.7 * route_choice_pressure) - (0.3 * cycle_pressure),
            ),
        )
    )


def analyze_structural_topology(mission_graph: nx.Graph) -> StructuralTopologyMetrics:
    """Return loop/branch/dead-end structural metrics."""
    if mission_graph is None or mission_graph.number_of_nodes() <= 0:
        return StructuralTopologyMetrics(
            cyclomatic_complexity=0.0,
            branching_factor=0.0,
            dead_end_ratio=0.0,
        )

    n_nodes = float(max(1, mission_graph.number_of_nodes()))
    if mission_graph.is_directed():
        dead_ends = sum(
            1
            for n in mission_graph.nodes()
            if (
                (int(mission_graph.in_degree(n)) == 0 and _node_role(mission_graph, n) != "start")
                or (int(mission_graph.out_degree(n)) == 0 and _node_role(mission_graph, n) != "goal")
            )
        )
    else:
        dead_ends = sum(
            1
            for n in mission_graph.nodes()
            if int(mission_graph.degree(n)) <= 1 and _node_role(mission_graph, n) not in {"start", "goal"}
        )

    return StructuralTopologyMetrics(
        cyclomatic_complexity=compute_cyclomatic_complexity(mission_graph),
        branching_factor=compute_branching_factor(mission_graph),
        dead_end_ratio=float(dead_ends / n_nodes),
    )
