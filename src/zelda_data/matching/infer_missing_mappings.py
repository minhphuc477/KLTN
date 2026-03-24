"""Inference helpers for missing room-node mappings."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

RoomPos = Tuple[int, int]


def seed_from_special_nodes(
    rooms: Dict[RoomPos, Any],
    graph: nx.DiGraph,
    existing_room_to_node: Dict[RoomPos, int],
    existing_node_to_room: Dict[int, RoomPos],
    proposed_room_to_node: Dict[RoomPos, int],
    proposed_node_to_room: Dict[int, RoomPos],
    confidences: Dict[int, float],
) -> None:
    """Seed mapping using high-confidence start-node + stair-room anchor."""
    if existing_room_to_node:
        return

    start_node = None
    for node, attrs in graph.nodes(data=True):
        if attrs.get("is_start"):
            start_node = node
            break

    start_room_pos = None
    stair_rooms_with_doors = []
    for pos, room in rooms.items():
        if getattr(room, "has_stair", False):
            door_count = sum(room.doors.values())
            if door_count > 0:
                stair_rooms_with_doors.append((pos, door_count))

    if stair_rooms_with_doors:
        stair_rooms_with_doors.sort(key=lambda x: x[1], reverse=True)
        start_room_pos = stair_rooms_with_doors[0][0]

    if start_node is not None and start_room_pos is not None:
        existing_room_to_node[start_room_pos] = start_node
        existing_node_to_room[start_node] = start_room_pos
        proposed_room_to_node[start_room_pos] = start_node
        proposed_node_to_room[start_node] = start_room_pos
        confidences[start_node] = 0.98


def propagate_from_anchors(
    rooms: Dict[RoomPos, Any],
    graph: nx.DiGraph,
    existing_room_to_node: Dict[RoomPos, int],
    existing_node_to_room: Dict[int, RoomPos],
    proposed_room_to_node: Dict[RoomPos, int],
    proposed_node_to_room: Dict[int, RoomPos],
    confidences: Dict[int, float],
    match_rooms_to_nodes_bfs_fn: Callable[[Dict[RoomPos, Any], Dict[RoomPos, List[RoomPos]], nx.DiGraph, Optional[RoomPos], Optional[int]], Tuple[Dict[RoomPos, int], Dict[int, RoomPos]]],
    build_room_adjacency_fn: Callable[[Dict[RoomPos, Any]], Dict[RoomPos, List[RoomPos]]],
) -> None:
    """Expand mapping proposals from anchor pairs using BFS matcher."""
    anchors = list(existing_room_to_node.items())
    used_nodes = set(existing_node_to_room.keys())
    room_adj = build_room_adjacency_fn(rooms)

    for room_anchor, node_anchor in anchors:
        r2n, _n2r = match_rooms_to_nodes_bfs_fn(rooms, room_adj, graph, room_anchor, node_anchor)
        for rpos, nid in r2n.items():
            if rpos in existing_room_to_node or rpos in proposed_room_to_node:
                continue
            if nid in used_nodes or nid in proposed_node_to_room:
                continue
            proposed_room_to_node[rpos] = nid
            proposed_node_to_room[nid] = rpos
            confidences[nid] = 0.9
            used_nodes.add(nid)


def apply_label_hints(
    unmatched_nodes: List[int],
    unmatched_rooms: List[RoomPos],
    graph: nx.DiGraph,
    proposed_room_to_node: Dict[RoomPos, int],
    proposed_node_to_room: Dict[int, RoomPos],
    confidences: Dict[int, float],
) -> Tuple[List[int], List[RoomPos]]:
    """Apply robust coordinate-label hints (e.g., '(3,4)', '3_4', 'r:3,c:4')."""
    coord_re = re.compile(r"\(?\s*(\d+)\s*[,_x\\/\s:-]\s*(\d+)\s*\)?")

    remaining_nodes = list(unmatched_nodes)
    remaining_rooms = list(unmatched_rooms)
    for node in list(remaining_nodes):
        attrs = graph.nodes[node]
        label = attrs.get("label") or attrs.get("name") or ""
        m = coord_re.search(str(label))
        if not m:
            continue

        r = int(m.group(1))
        c = int(m.group(2))
        candidate = (r, c)
        if candidate in remaining_rooms and node not in proposed_node_to_room:
            proposed_node_to_room[node] = candidate
            proposed_room_to_node[candidate] = node
            confidences[node] = 0.98
            remaining_nodes.remove(node)
            remaining_rooms.remove(candidate)

    return remaining_nodes, remaining_rooms


def build_component_context(
    graph: nx.DiGraph,
    rooms: Dict[RoomPos, Any],
    room_adjacency: Dict[RoomPos, List[RoomPos]],
    existing_room_to_node: Dict[RoomPos, int],
) -> Tuple[Dict[int, int], Dict[RoomPos, int], Dict[int, Set[int]]]:
    """Build graph/room component IDs and anchored component candidate mapping."""
    graph_comp_of: Dict[int, int] = {}
    for comp in nx.weakly_connected_components(graph):
        comp_id = id(comp)
        for n in comp:
            graph_comp_of[n] = comp_id

    room_graph = nx.Graph()
    room_graph.add_nodes_from(rooms.keys())
    for room_pos, neighbors in room_adjacency.items():
        for nb in neighbors:
            room_graph.add_edge(room_pos, nb)

    room_comp_of: Dict[RoomPos, int] = {}
    for comp in nx.connected_components(room_graph):
        comp_id = id(comp)
        for r in comp:
            room_comp_of[r] = comp_id

    comp_room_candidates: Dict[int, Set[int]] = {}
    for room_pos, node_id in dict(existing_room_to_node).items():
        gc = graph_comp_of.get(node_id)
        rc = room_comp_of.get(room_pos)
        if gc is not None and rc is not None:
            comp_room_candidates.setdefault(gc, set()).add(rc)

    return graph_comp_of, room_comp_of, comp_room_candidates


def compute_normalized_room_centers(
    unmatched_rooms: List[RoomPos],
    room_positions: Optional[Dict[RoomPos, Tuple[int, int]]],
) -> Dict[RoomPos, Tuple[float, float]]:
    """Compute normalized room centers from global room offsets when available."""
    centers: Dict[RoomPos, Tuple[float, float]] = {}
    if not room_positions:
        return centers

    xs = [off[1] for off in room_positions.values() if off]
    ys = [off[0] for off in room_positions.values() if off]
    if not xs or not ys:
        return centers

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1)

    for room_pos in unmatched_rooms:
        off = room_positions.get(room_pos)
        if off:
            centers[room_pos] = ((off[1] - minx) / span, (off[0] - miny) / span)

    return centers


def build_score_matrix(
    unmatched_nodes: List[int],
    unmatched_rooms: List[RoomPos],
    graph: nx.DiGraph,
    rooms: Dict[RoomPos, Any],
    centers: Dict[RoomPos, Tuple[float, float]],
    graph_comp_of: Dict[int, int],
    room_comp_of: Dict[RoomPos, int],
    comp_room_candidates: Dict[int, Set[int]],
) -> Dict[Tuple[int, RoomPos], float]:
    """Build confidence-like score matrix for node-room matching candidates."""
    deg = {n: (graph.in_degree(n) + graph.out_degree(n)) for n in unmatched_nodes}
    room_degs = {r: sum(rooms[r].doors.values()) for r in unmatched_rooms}
    max_deg = max(list(deg.values()) + [1])

    score_matrix: Dict[Tuple[int, RoomPos], float] = {}
    for n in unmatched_nodes:
        for r in unmatched_rooms:
            node_deg = deg.get(n, 0)
            room_deg = room_degs.get(r, 0)

            deg_score = 1.0 - (abs(node_deg - (room_deg * 2)) / float(max_deg + room_deg + 1.0))
            spat_score = 0.5 if r in centers else 0.0

            comp_bonus = 0.0
            gc = graph_comp_of.get(n)
            rc = room_comp_of.get(r)
            if gc is not None and gc in comp_room_candidates and rc in comp_room_candidates[gc]:
                comp_bonus = 0.2

            score = 0.7 * deg_score + 0.25 * spat_score + comp_bonus
            score_matrix[(n, r)] = max(0.0, min(1.0, score))

    return score_matrix


def assign_pairs_from_scores(
    unmatched_nodes: List[int],
    unmatched_rooms: List[RoomPos],
    score_matrix: Dict[Tuple[int, RoomPos], float],
    logger: logging.Logger,
) -> List[Tuple[int, RoomPos]]:
    """Compute global assignment from score matrix using Hungarian, fallback to greedy."""
    assigned_pairs: List[Tuple[int, RoomPos]] = []
    try:
        from scipy.optimize import linear_sum_assignment

        nodes_idx = {n: i for i, n in enumerate(unmatched_nodes)}
        rooms_idx = {r: j for j, r in enumerate(unmatched_rooms)}
        cost = np.zeros((len(unmatched_nodes), len(unmatched_rooms)), dtype=np.float32)
        for (n, r), score in score_matrix.items():
            cost[nodes_idx[n], rooms_idx[r]] = -float(score)

        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            n = unmatched_nodes[i]
            r = unmatched_rooms[j]
            s = score_matrix.get((n, r), 0.0)
            if s > 0:
                assigned_pairs.append((n, r))
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("Hungarian assignment failed; falling back to greedy assignment: %s", exc)
        local_scores = dict(score_matrix)
        remaining_nodes = list(unmatched_nodes)
        remaining_rooms = list(unmatched_rooms)
        while local_scores and remaining_nodes and remaining_rooms:
            best = max(local_scores.items(), key=lambda kv: kv[1])[0]
            best_score = local_scores[best]
            n, r = best
            if best_score <= 0:
                break
            assigned_pairs.append((n, r))
            remaining_nodes = [x for x in remaining_nodes if x != n]
            remaining_rooms = [x for x in remaining_rooms if x != r]
            for key in list(local_scores.keys()):
                if key[0] == n or key[1] == r:
                    del local_scores[key]

    return assigned_pairs
