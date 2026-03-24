"""Helpers for room<->graph matching extracted from zelda_core monolith."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Any

import networkx as nx

RoomPos = Tuple[int, int]


def deterministic_greedy_assignment(
    cost_matrix: List[List[float]],
    rooms_order: List[RoomPos],
    nodes_order: List[int],
    node_signature_fn: Callable[[int], Any],
) -> List[Tuple[RoomPos, int]]:
    """Build a deterministic greedy bipartite assignment from a cost matrix."""
    pairs = []
    for i in range(len(rooms_order)):
        for j, n in enumerate(nodes_order):
            pairs.append((cost_matrix[i][j], i, n))
    pairs.sort(key=lambda x: (x[0], node_signature_fn(x[2]), x[2]))

    used_r = set()
    used_n = set()
    assigned_pairs: List[Tuple[RoomPos, int]] = []
    for _cost, i, n in pairs:
        if i in used_r or n in used_n:
            continue
        used_r.add(i)
        used_n.add(n)
        assigned_pairs.append((rooms_order[i], n))
    return assigned_pairs


def solve_assignment_with_fallback(
    cost_matrix: List[List[float]],
    rooms_order: List[RoomPos],
    nodes_order: List[int],
    node_signature_fn: Callable[[int], Any],
    logger: logging.Logger,
    failure_log_prefix: str,
    max_hungarian_size: Optional[int] = None,
) -> List[Tuple[RoomPos, int]]:
    """Use Hungarian assignment when applicable, otherwise deterministic greedy."""
    assigned_pairs: List[Tuple[RoomPos, int]] = []
    try:
        use_hungarian = max_hungarian_size is None or (
            len(rooms_order) <= max_hungarian_size and len(nodes_order) <= max_hungarian_size
        )
        if use_hungarian:
            from scipy.optimize import linear_sum_assignment
            import numpy as np

            matrix = np.array(cost_matrix, dtype=float)
            row_ind, col_ind = linear_sum_assignment(matrix)
            for i, j in zip(row_ind, col_ind):
                if i < len(rooms_order) and j < len(nodes_order):
                    assigned_pairs.append((rooms_order[i], nodes_order[j]))
        else:
            raise RuntimeError("skip hungarian")
    except (ImportError, RuntimeError):
        assigned_pairs = deterministic_greedy_assignment(
            cost_matrix,
            rooms_order,
            nodes_order,
            node_signature_fn,
        )
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("%s: %s", failure_log_prefix, exc)
        assigned_pairs = deterministic_greedy_assignment(
            cost_matrix,
            rooms_order,
            nodes_order,
            node_signature_fn,
        )

    return assigned_pairs


def refine_mapping_with_swaps(
    rooms: Dict[RoomPos, Any],
    room_adjacency: Dict[RoomPos, List[RoomPos]],
    graph: nx.DiGraph,
    room_to_node: Dict[RoomPos, int],
    node_to_room: Dict[int, RoomPos],
    validate_mapping_fn: Callable[[Dict[RoomPos, Any], Dict[RoomPos, List[RoomPos]], nx.DiGraph, Dict[RoomPos, int]], float],
    max_iters: int = 100,
) -> float:
    """Try pairwise swaps to improve room-node adjacency consistency."""
    cur_cons = validate_mapping_fn(rooms, room_adjacency, graph, room_to_node)
    rooms_list = sorted(list(room_to_node.keys()))
    it = 0
    improved = True
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(len(rooms_list)):
            for j in range(i + 1, len(rooms_list)):
                r1 = rooms_list[i]
                r2 = rooms_list[j]
                n1 = room_to_node[r1]
                n2 = room_to_node[r2]
                # Swap and keep only if consistency improves.
                room_to_node[r1], room_to_node[r2] = n2, n1
                node_to_room[n1], node_to_room[n2] = r2, r1
                new_cons = validate_mapping_fn(rooms, room_adjacency, graph, room_to_node)
                if new_cons > cur_cons + 1e-9:
                    cur_cons = new_cons
                    improved = True
                    break
                room_to_node[r1], room_to_node[r2] = n1, n2
                node_to_room[n1], node_to_room[n2] = r1, r2
            if improved:
                break

    return cur_cons
