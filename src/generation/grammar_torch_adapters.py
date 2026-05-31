"""PyTorch tensor adapters for mission grammar graphs.

The symbolic grammar can be imported and used without PyTorch.  This module
contains the optional tensor conversion helpers used by GNN conditioning.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from src.generation.grammar import MissionGraph, NodeType


def mission_graph_to_tensor(graph: MissionGraph) -> Tuple[Tensor, Tensor]:
    """Convert a mission graph into edge-index and node-feature tensors."""
    node_ids, id_to_idx = graph._node_index_map()

    sources = []
    targets = []
    for edge in graph.edges:
        if edge.source not in id_to_idx or edge.target not in id_to_idx:
            continue
        sources.append(id_to_idx[edge.source])
        targets.append(id_to_idx[edge.target])

    if sources:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    features = [graph.nodes[nid].to_feature_vector() for nid in node_ids]
    if features:
        node_features = torch.tensor(features, dtype=torch.float32)
    else:
        feature_dim = len(NodeType) + 14
        node_features = torch.zeros((0, feature_dim), dtype=torch.float32)

    return edge_index, node_features


def mission_graph_to_adjacency_matrix(graph: MissionGraph) -> Tensor:
    """Convert a mission graph into a dense adjacency tensor."""
    node_ids, id_to_idx = graph._node_index_map()
    adj = torch.zeros(len(node_ids), len(node_ids))

    for edge in graph.edges:
        if edge.source not in id_to_idx or edge.target not in id_to_idx:
            continue
        src_idx = id_to_idx[edge.source]
        tgt_idx = id_to_idx[edge.target]
        adj[src_idx, tgt_idx] = 1.0
        if edge.edge_type in graph.BIDIRECTIONAL_EDGE_TYPES:
            adj[tgt_idx, src_idx] = 1.0

    return adj


def mission_graph_compute_tpe(graph: MissionGraph) -> Tensor:
    """Compute topological positional encoding for graph nodes."""
    node_ids = sorted(graph.nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    tpe = torch.zeros(len(node_ids), 8)

    start = graph.get_start_node()
    goal = graph.get_goal_node()

    start_id = start.id if start else (node_ids[0] if node_ids else 0)
    goal_id = goal.id if goal else (node_ids[-1] if node_ids else 0)

    dist_from_start = graph._bfs_distances(start_id)
    dist_to_goal = graph._bfs_distances(
        goal_id,
        adjacency=graph._build_reverse_adjacency(),
    )
    max_dist = max([*dist_from_start.values(), 1])

    for nid in node_ids:
        idx = id_to_idx[nid]
        tpe[idx, 0] = dist_from_start.get(nid, max_dist) / max_dist
        tpe[idx, 1] = dist_to_goal.get(nid, max_dist) / max_dist

        degree = len(graph._adjacency.get(nid, []))
        tpe[idx, 2] = min(degree / 4.0, 1.0)

        if start and goal:
            on_path = (
                dist_from_start.get(nid, float("inf"))
                + dist_to_goal.get(nid, float("inf"))
                == dist_from_start.get(goal_id, float("inf"))
            )
            tpe[idx, 3] = 1.0 if on_path else 0.0

        node = graph.nodes[nid]
        tpe[idx, 4] = 1.0 if node.node_type == NodeType.KEY else 0.0
        tpe[idx, 5] = 1.0 if node.node_type == NodeType.LOCK else 0.0
        tpe[idx, 6] = node.difficulty
        tpe[idx, 7] = 1.0 if node.key_id is not None else 0.0

    return tpe


def graph_to_gnn_input(
    graph: MissionGraph,
    current_node_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert a mission graph to the tensor dictionary consumed by GNNs."""
    edge_index, node_features = mission_graph_to_tensor(graph)
    return {
        "edge_index": edge_index,
        "node_features": node_features,
        "tpe": mission_graph_compute_tpe(graph),
        "current_node": current_node_idx or 0,
        "adjacency": mission_graph_to_adjacency_matrix(graph),
    }
