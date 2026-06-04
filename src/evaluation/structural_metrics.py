"""Structural graph metrics used by topology search and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass
class StructuralTopologyMetrics:
    cyclomatic_complexity: float
    branching_factor: float
    dead_end_ratio: float


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
    """Compute average branching among non-terminal nodes."""
    if mission_graph is None or mission_graph.number_of_nodes() <= 0:
        return 0.0

    if mission_graph.is_directed():
        branch_degrees = [
            float(mission_graph.out_degree(n))
            for n in mission_graph.nodes()
            if int(mission_graph.out_degree(n)) > 1
        ]
    else:
        branch_degrees = [
            float(mission_graph.degree(n))
            for n in mission_graph.nodes()
            if int(mission_graph.degree(n)) > 1
        ]

    if not branch_degrees:
        return 0.0
    return float(sum(branch_degrees) / len(branch_degrees))


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
            if int(mission_graph.out_degree(n)) == 0 or int(mission_graph.in_degree(n)) == 0
        )
    else:
        dead_ends = sum(1 for n in mission_graph.nodes() if int(mission_graph.degree(n)) <= 1)

    return StructuralTopologyMetrics(
        cyclomatic_complexity=compute_cyclomatic_complexity(mission_graph),
        branching_factor=compute_branching_factor(mission_graph),
        dead_end_ratio=float(dead_ends / n_nodes),
    )
