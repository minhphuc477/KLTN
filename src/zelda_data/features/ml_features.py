"""ML feature extraction utilities extracted from zelda_core."""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np

from src.core.definitions import parse_edge_type_tokens, parse_node_label_tokens


class MLFeatureExtractor:
    """Extract ML-ready features from dungeon topology graphs."""

    @staticmethod
    def compute_laplacian_pe(G: nx.Graph, k_dim: int = 8) -> Tuple[np.ndarray, Dict[int, int]]:
        """Compute topology-aware positional encodings from Laplacian eigenvectors."""
        G_undirected = G.to_undirected() if G.is_directed() else G

        nodes = sorted(G_undirected.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        if n == 0:
            return np.zeros((0, k_dim)), {}

        adj = np.zeros((n, n))

        for u, v, data in G_undirected.edges(data=True):
            idx_u, idx_v = node_to_idx[u], node_to_idx[v]

            edge_type = data.get("edge_type", "open")
            weight = 0.5 if edge_type in ["locked", "bombable", "boss_locked", "soft_locked"] else 1.0

            adj[idx_u, idx_v] = weight
            adj[idx_v, idx_u] = weight

        degrees = np.sum(adj, axis=1)
        degree_matrix = np.diag(degrees)
        laplacian = degree_matrix - adj

        try:
            _eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

            start_idx = 1
            end_idx = min(start_idx + k_dim, n)
            tpe = eigenvectors[:, start_idx:end_idx]

            if tpe.shape[1] < k_dim:
                padding = np.zeros((n, k_dim - tpe.shape[1]))
                tpe = np.hstack([tpe, padding])

        except np.linalg.LinAlgError:
            tpe = np.zeros((n, k_dim))

        return tpe.astype(np.float32), node_to_idx

    @staticmethod
    def extract_node_features(G: nx.DiGraph, node_order: Dict[int, int]) -> np.ndarray:
        """Build 6D multi-hot node features compatible with training loaders."""
        n = len(node_order)
        features = np.zeros((n, 6), dtype=np.float32)

        for node_id, idx in node_order.items():
            if node_id not in G.nodes:
                continue

            attrs = G.nodes[node_id]
            label = attrs.get("label", "")
            parts = parse_node_label_tokens(label)

            features[idx, 0] = 1.0 if (attrs.get("has_enemy", False) or "e" in parts) else 0.0
            features[idx, 1] = 1.0 if (attrs.get("has_key", False) or "k" in parts) else 0.0
            features[idx, 2] = (
                1.0
                if (attrs.get("has_item", False) or "i" in parts or "I" in parts or "K" in parts)
                else 0.0
            )
            features[idx, 3] = (
                1.0
                if (attrs.get("has_triforce", False) or attrs.get("is_triforce", False) or "t" in parts)
                else 0.0
            )
            features[idx, 4] = (
                1.0 if (attrs.get("has_boss", False) or attrs.get("is_boss", False) or "b" in parts) else 0.0
            )
            features[idx, 5] = 1.0 if (attrs.get("has_puzzle", False) or "p" in parts) else 0.0

        return features

    @staticmethod
    def build_p_matrix(G: nx.DiGraph, node_order: Dict[int, int]) -> np.ndarray:
        """Build dependency matrix channels for key/bomb/boss key constraints."""
        n = len(node_order)
        p_matrix = np.zeros((n, n, 3), dtype=np.float32)

        for u, v, data in G.edges(data=True):
            if u not in node_order or v not in node_order:
                continue

            i, j = node_order[u], node_order[v]
            edge_types = parse_edge_type_tokens(
                label=data.get("label", ""),
                edge_type=data.get("edge_type", "open"),
            )

            if any(et in ("locked", "key_locked") for et in edge_types):
                p_matrix[i, j, 0] = 1.0
                p_matrix[j, i, 0] = 1.0
            elif "bombable" in edge_types:
                p_matrix[i, j, 1] = 1.0
                p_matrix[j, i, 1] = 1.0
            elif "boss_locked" in edge_types:
                p_matrix[i, j, 2] = 1.0
                p_matrix[j, i, 2] = 1.0

        return p_matrix
