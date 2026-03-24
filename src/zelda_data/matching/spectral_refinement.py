"""Spectral matching and local refinement helpers for room-node alignment."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

RoomPos = Tuple[int, int]


def edge_consistency_score(
    n2r: Dict[int, RoomPos],
    graph: nx.DiGraph,
    room_adj: Dict[RoomPos, List[RoomPos]],
) -> int:
    """Count directed graph edges whose mapped room endpoints are adjacent."""
    score = 0
    for u, v in graph.edges():
        ru = n2r.get(u)
        rv = n2r.get(v)
        if ru is None or rv is None:
            continue
        if rv in room_adj.get(ru, []):
            score += 1
    return score


def local_refine_assignments(
    n2r: Dict[int, RoomPos],
    graph: nx.DiGraph,
    room_adj: Dict[RoomPos, List[RoomPos]],
    score_matrix: Dict[Tuple[int, RoomPos], float],
    iterations: int = 100,
) -> Dict[int, RoomPos]:
    """Perform deterministic pairwise-swap local refinement for mapping quality."""
    n2r = dict(n2r)

    def objective(mapping: Dict[int, RoomPos]) -> float:
        assign_score = sum(float(score_matrix.get((n, r), 0.0)) for n, r in mapping.items())
        edge_score = edge_consistency_score(mapping, graph, room_adj)
        return assign_score + 0.5 * edge_score

    best_obj = objective(n2r)
    improved = True
    it = 0
    nodes = list(n2r.keys())
    while improved and it < iterations:
        improved = False
        it += 1
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a = nodes[i]
                b = nodes[j]
                ra = n2r[a]
                rb = n2r[b]
                if ra == rb:
                    continue
                n2r[a], n2r[b] = rb, ra
                new_obj = objective(n2r)
                if new_obj > best_obj + 1e-6:
                    best_obj = new_obj
                    improved = True
                else:
                    n2r[a], n2r[b] = ra, rb
        if not improved:
            break

    return n2r


def seeded_spectral_match(
    rooms: Dict[RoomPos, Any],
    graph: nx.DiGraph,
    build_room_adjacency_fn: Callable[[Dict[RoomPos, Any]], Dict[RoomPos, List[RoomPos]]],
    logger: Any,
    room_positions: Optional[Dict[RoomPos, Tuple[int, int]]] = None,
    seeds: Optional[Dict[RoomPos, int]] = None,
    k_dim: int = 8,
) -> Tuple[Dict[int, RoomPos], Dict[int, float]]:
    """Perform seeded spectral graph-room matching and return proposals/confidence."""
    del room_positions  # Reserved for future spatial refinement.

    try:
        from scipy.linalg import orthogonal_procrustes
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("scipy.linalg.orthogonal_procrustes not available: %s", exc)
        orthogonal_procrustes = None

    room_adj = build_room_adjacency_fn(rooms)
    room_graph = nx.Graph()
    room_graph.add_nodes_from(rooms.keys())
    for room_pos, neighbors in room_adj.items():
        for nb in neighbors:
            room_graph.add_edge(room_pos, nb)

    if len(graph) == 0 or len(room_graph) == 0:
        return {}, {}

    def laplacian_embedding(G: nx.Graph, dim: int):
        G_u = G.to_undirected() if G.is_directed() else G
        nodes = sorted(G_u.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        adj = np.zeros((n, n), dtype=float)
        for u, v in G_u.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        deg = np.sum(adj, axis=1)
        D = np.diag(deg)
        L = D - adj
        try:
            _eigvals, eigvecs = np.linalg.eigh(L)
            start = 1
            end = min(start + dim, n)
            emb = eigvecs[:, start:end]
            if emb.shape[1] < dim:
                pad = np.zeros((n, dim - emb.shape[1]))
                emb = np.hstack([emb, pad])
        except (FloatingPointError, TypeError, ValueError, np.linalg.LinAlgError):
            emb = np.zeros((n, dim))
        return nodes, emb

    graph_nodes, g_emb = laplacian_embedding(graph, k_dim)
    room_nodes, r_emb = laplacian_embedding(room_graph, k_dim)

    seed_pairs = []
    if seeds:
        for room_pos, node_id in seeds.items():
            if node_id in graph_nodes and room_pos in room_nodes:
                seed_pairs.append((node_id, room_pos))

    if not seed_pairs:
        return {}, {}

    g_idx = {n: i for i, n in enumerate(graph_nodes)}
    r_idx = {r: i for i, r in enumerate(room_nodes)}
    X = np.array([g_emb[g_idx[node_id]] for node_id, _ in seed_pairs])
    Y = np.array([r_emb[r_idx[room_pos]] for _, room_pos in seed_pairs])

    if orthogonal_procrustes is not None:
        try:
            R, _scale = orthogonal_procrustes(X, Y)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("orthogonal_procrustes failed in seeded_spectral_match: %s", exc)
            R = None
    else:
        try:
            U, _s, Vt = np.linalg.svd(X.T.dot(Y))
            R = U.dot(Vt)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("SVD-based alignment failed in seeded_spectral_match: %s", exc)
            R = None

    if R is None:
        return {}, {}

    g_emb_aligned = g_emb.dot(R)

    from scipy.optimize import linear_sum_assignment

    try:
        cost = np.zeros((len(graph_nodes), len(room_nodes)), dtype=float)
        for i, node_id in enumerate(graph_nodes):
            for j, room_pos in enumerate(room_nodes):
                cost[i, j] = np.linalg.norm(g_emb_aligned[i] - r_emb[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        proposals: Dict[int, RoomPos] = {}
        confidences: Dict[int, float] = {}
        for i, j in zip(row_ind, col_ind):
            node_id = graph_nodes[i]
            room_pos = room_nodes[j]
            proposals[node_id] = room_pos
            maxd = cost.max() if cost.size else 1.0
            dist = cost[i, j]
            confidences[node_id] = float(max(0.01, 1.0 - (dist / (maxd + 1e-6))))
        return proposals, confidences
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("Spectral matching assignment failed: %s", exc)
        return {}, {}
