"""Shared graph-aware room stitching helpers for generation pipelines."""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np

from src.core.definitions import DOOR_POSITIONS, SEMANTIC_PALETTE, TileID, parse_edge_type_tokens
from src.pipeline.spatial_utils import (
    first_free_position,
    get_node_grid_position,
    stable_node_sort_key,
)

logger = logging.getLogger(__name__)

NodeSortKey = Callable[[Any], Tuple[int, Any]]
NodePositionGetter = Callable[[nx.Graph, Any], Optional[Tuple[int, int]]]
DiagnosticCallback = Callable[[str], None]
ConnectorTileResolver = Callable[[Optional[Dict[str, Any]], bool], Tuple[int, int]]

NON_SPATIAL_EDGE_TOKENS = frozenset(
    {
        "stairs",
        "stair",
        "warp",
        "teleport",
        "teleporter",
        "visual_link",
        "visual",
        "window",
        "balcony",
        "basement",
        "floor_transition",
        "floor",
        "layer",
    }
)


@dataclass
class StitchedRoomLayout:
    """Shared stitched-room output for grid consumers and bbox-based systems."""

    dungeon_grid: np.ndarray
    slot_positions: Dict[Any, Tuple[int, int]]
    room_offsets: Dict[Any, Tuple[int, int]]
    layout_map: Dict[Any, Tuple[int, int, int, int]]


def _edge_tokens(edge_data: Optional[Mapping[str, Any]]) -> set:
    data = edge_data or {}
    label = str(data.get("label", "") or "")
    edge_type = data.get("edge_type", data.get("type", ""))
    normalized_type = getattr(edge_type, "name", edge_type)
    tokens = set(parse_edge_type_tokens(label=label, edge_type=str(normalized_type or "")))
    for key in ("edge_type", "type", "label", "semantic"):
        value = data.get(key)
        if value is None:
            continue
        name = getattr(value, "name", value)
        tokens.update(
            part.strip().lower()
            for part in str(name).replace(",", " ").replace("_", " ").split()
            if part.strip()
        )
    return tokens


def _node_floor_key(graph: nx.Graph, node_id: Any) -> int:
    """Extract a coarse floor/layer identifier without changing 2D slot coordinates."""
    attrs = graph.nodes.get(node_id, {}) if node_id in graph else {}
    for key in ("floor", "z", "level", "layer", "virtual_layer"):
        if key not in attrs:
            continue
        value = attrs.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    for key in ("position", "pos", "grid_pos", "coord", "coords"):
        value = attrs.get(key)
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 3:
            try:
                return int(value[2])
            except (TypeError, ValueError):
                continue
        if isinstance(value, str):
            parts = value.replace("(", "").replace(")", "").split(",")
            if len(parts) >= 3:
                try:
                    return int(float(parts[2].strip()))
                except ValueError:
                    continue
    return 0


def _is_spatial_room_edge(
    graph: nx.Graph,
    source: Any,
    target: Any,
    edge_data: Optional[Mapping[str, Any]] = None,
) -> bool:
    """
    Return True only for edges the flat stitcher should realize as adjacent doors.

    Mission graphs may contain abstract, vertical, warp, and visual links. Forcing
    those links into strict 2D adjacency makes valid multi-floor/non-planar logic
    fail placement and then corrupts it by carving ordinary doors.
    """
    if source not in graph or target not in graph:
        return False
    tokens = _edge_tokens(edge_data)
    if tokens.intersection(NON_SPATIAL_EDGE_TOKENS):
        return False
    if _node_floor_key(graph, source) != _node_floor_key(graph, target):
        return False
    return True

def extract_room_grid(room_like: Any) -> np.ndarray:
    """Coerce a room-like object or raw array into a 2D int32 grid."""
    grid = getattr(room_like, "room_grid", room_like)
    arr = np.asarray(grid, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"Room grid must be a non-empty 2D array, got shape {arr.shape}.")
    return arr


def solve_component_strict_adjacency(
    comp_nodes: List[Any],
    adjacency: Dict[Any, set],
    explicit_pos: Dict[Any, Tuple[int, int]],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
    search_budget: Optional[int] = None,
) -> Dict[Any, Tuple[int, int]]:
    """Backtracking solver for a single connected component strict embedding."""
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    placement: Dict[Any, Tuple[int, int]] = {}
    occupied: set = set()

    root = comp_nodes[0]
    root_pos = explicit_pos.get(root, (0, 0))
    placement[root] = root_pos
    occupied.add(root_pos)
    edge_count = sum(len(adjacency.get(node, ())) for node in comp_nodes) // 2
    cycle_pressure = max(0, int(edge_count) - int(len(comp_nodes)) + 1)
    if search_budget is None:
        env_budget = os.environ.get("HMOLQD_STRICT_STITCH_BUDGET", "").strip()
        if env_budget:
            try:
                search_budget = int(env_budget)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HMOLQD_STRICT_STITCH_BUDGET=%r",
                    env_budget,
                )
                search_budget = None
    if search_budget is None:
        # Strict orthogonal graph embedding is exponential in the worst case.
        # Scale the default budget with both component size and loop pressure
        # rather than hiding a fixed global magic number in the solver.
        search_budget = int(
            max(
                50_000,
                min(
                    1_000_000,
                    256 * max(1, len(comp_nodes)) * max(1, len(comp_nodes))
                    * max(1, int(math.sqrt(cycle_pressure + 1))),
                ),
            )
        )
    initial_search_budget = int(max(1, search_budget))
    search_budget = initial_search_budget

    def _neighbors_of(pos: Tuple[int, int]) -> set:
        r, c = pos
        return {(r + dr, c + dc) for dr, dc in offsets}

    def _is_adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def _candidate_positions(node: Any) -> List[Tuple[int, int]]:
        placed_neighbors = [n for n in adjacency[node] if n in placement]
        if placed_neighbors:
            common = None
            for pn in placed_neighbors:
                neigh_cells = _neighbors_of(placement[pn])
                common = neigh_cells if common is None else (common & neigh_cells)
            candidates = sorted(common or set())
        else:
            frontier = set()
            for pos in placement.values():
                frontier |= _neighbors_of(pos)
            candidates = sorted(frontier)

        pref = explicit_pos.get(node)
        if pref is not None:
            candidates.append(pref)
            ring_budget = max(4, len(comp_nodes) + len(adjacency[node]))
            radius_limit = max(2, min(8, len(comp_nodes)))
            for radius in range(1, radius_limit + 1):
                ring_cells: List[Tuple[int, int]] = []
                for d_row in range(-radius, radius + 1):
                    d_col = radius - abs(d_row)
                    ring_cells.append((pref[0] + d_row, pref[1] + d_col))
                    if d_col != 0:
                        ring_cells.append((pref[0] + d_row, pref[1] - d_col))
                candidates.extend(ring_cells)
                if len(candidates) >= ring_budget:
                    break

        deduped_candidates: List[Tuple[int, int]] = []
        seen_candidates = set()
        for cand in candidates:
            if cand in seen_candidates:
                continue
            seen_candidates.add(cand)
            deduped_candidates.append(cand)
        candidates = deduped_candidates

        filtered = [cand for cand in candidates if cand not in occupied]
        if not filtered:
            return []

        def _score(cell: Tuple[int, int]) -> Tuple[float, int, int]:
            neigh_score = 0.0
            for pn in adjacency[node]:
                if pn in placement:
                    neigh_score += abs(cell[0] - placement[pn][0]) + abs(cell[1] - placement[pn][1])
            pref_penalty = 0.0
            if pref is not None:
                pref_penalty = abs(cell[0] - pref[0]) + abs(cell[1] - pref[1])
            return (neigh_score + (0.25 * pref_penalty), cell[0], cell[1])

        filtered.sort(key=_score)
        return filtered

    def _node_priority(node: Any) -> Tuple[int, int, Any]:
        placed_neighbors = sum(1 for n in adjacency[node] if n in placement)
        return (-placed_neighbors, -len(adjacency[node]), sort_key(node))

    def _check_partial(node: Any, pos: Tuple[int, int]) -> bool:
        if pos in occupied:
            return False
        for pn in adjacency[node]:
            if pn in placement and not _is_adjacent(pos, placement[pn]):
                return False
        return True

    def _dfs() -> bool:
        nonlocal search_budget
        if len(placement) == len(comp_nodes):
            return True
        if search_budget <= 0:
            return False
        search_budget -= 1

        unplaced = [n for n in comp_nodes if n not in placement]
        unplaced.sort(key=_node_priority)
        node = unplaced[0]

        for cand in _candidate_positions(node):
            if not _check_partial(node, cand):
                continue
            placement[node] = cand
            occupied.add(cand)
            if _dfs():
                return True
            occupied.remove(cand)
            placement.pop(node, None)
        return False

    if not _dfs():
        raise ValueError(
            "Failed strict adjacency placement for component with nodes "
            f"{comp_nodes}; exhausted budget={initial_search_budget}, "
            f"nodes={len(comp_nodes)}, edges={edge_count}, cycle_pressure={cycle_pressure}. "
            "Consider planarizing topology, providing explicit positions, or increasing "
            "HMOLQD_STRICT_STITCH_BUDGET for this ablation."
        )

    seen_edges = set()
    for u in comp_nodes:
        for v in adjacency[u]:
            edge_key = tuple(sorted((u, v), key=sort_key))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            if abs(placement[u][0] - placement[v][0]) + abs(placement[u][1] - placement[v][1]) != 1:
                raise ValueError(f"Strict adjacency invariant failed for edge {u!r}<->{v!r}")

    return placement


def _component_root_node(
    graph: nx.Graph,
    comp_nodes: List[Any],
    adjacency: Dict[Any, set],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
) -> Any:
    """Pick a stable semantic root for tree-style fallback placement."""
    component = set(comp_nodes)

    def _is_start_like(node_id: Any) -> bool:
        attrs = graph.nodes.get(node_id, {}) if node_id in graph else {}
        node_type = str(attrs.get("type", "") or "").strip().upper()
        label = str(attrs.get("label", "") or "").strip().upper()
        return node_type == "START" or label == "START" or bool(attrs.get("is_start"))

    start_like = [node_id for node_id in comp_nodes if node_id in component and _is_start_like(node_id)]
    if start_like:
        return min(start_like, key=sort_key)

    return min(
        comp_nodes,
        key=lambda node_id: (-len(adjacency.get(node_id, ())), sort_key(node_id)),
    )


def _component_tree_adjacency(
    graph: nx.Graph,
    comp_nodes: List[Any],
    adjacency: Dict[Any, set],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
) -> Tuple[List[Any], Dict[Any, set]]:
    """Build a deterministic BFS tree for fallback room placement."""
    root = _component_root_node(graph, comp_nodes, adjacency, sort_key=sort_key)
    component = set(comp_nodes)
    queue: deque[Any] = deque([root])
    visited = {root}
    bfs_order: List[Any] = []
    tree_adjacency: Dict[Any, set] = {node_id: set() for node_id in comp_nodes}

    while queue:
        node_id = queue.popleft()
        bfs_order.append(node_id)
        neighbors = sorted(adjacency.get(node_id, ()) & component, key=sort_key)
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
            tree_adjacency[node_id].add(neighbor)
            tree_adjacency[neighbor].add(node_id)

    remaining = [node_id for node_id in sorted(comp_nodes, key=sort_key) if node_id not in visited]
    bfs_order.extend(remaining)
    return bfs_order, tree_adjacency


def compute_relaxed_room_placement(
    graph: nx.Graph,
    room_ids: List[Any],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
    node_position_getter: NodePositionGetter = get_node_grid_position,
    first_free_position_fn: Callable[[Tuple[int, int], set], Tuple[int, int]] = first_free_position,
) -> Dict[Any, Tuple[int, int]]:
    """Deterministic non-overlapping fallback placement when strict embedding fails."""
    nodes = [n for n in room_ids if n in graph]
    if not nodes:
        return {}

    ordered_nodes = sorted(nodes, key=sort_key)
    placement: Dict[Any, Tuple[int, int]] = {}
    occupied: set = set()
    row_cursor = 0

    for node_id in ordered_nodes:
        pos = node_position_getter(graph, node_id)
        start = (int(pos[0]), int(pos[1])) if pos is not None else (row_cursor, 0)
        resolved = first_free_position_fn(start, occupied)
        placement[node_id] = resolved
        occupied.add(resolved)
        row_cursor = max(row_cursor, resolved[0] + 1)

    return placement


def compute_strict_room_placement(
    graph: nx.Graph,
    room_ids: List[Any],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
    node_position_getter: NodePositionGetter = get_node_grid_position,
    first_free_position_fn: Callable[[Tuple[int, int], set], Tuple[int, int]] = first_free_position,
    strict_search_budget: Optional[int] = None,
) -> Dict[Any, Tuple[int, int]]:
    """Compute a strict room placement where every graph edge is Manhattan-adjacent."""
    nodes = [n for n in room_ids if n in graph]
    if not nodes:
        return {}

    node_set = set(nodes)
    spatial_graph = nx.Graph()
    spatial_graph.add_nodes_from(nodes)
    for src, dst, edge_data in graph.edges(data=True):
        if src not in node_set or dst not in node_set:
            continue
        if not _is_spatial_room_edge(graph, src, dst, edge_data):
            continue
        spatial_graph.add_edge(src, dst)

    adjacency: Dict[Any, set] = {n: set(spatial_graph.neighbors(n)) & node_set for n in nodes}
    explicit_pos: Dict[Any, Tuple[int, int]] = {}
    for node_id in nodes:
        pos = node_position_getter(graph, node_id)
        if pos is not None:
            explicit_pos[node_id] = (int(pos[0]), int(pos[1]))

    max_degree = max((len(adjacency[node_id]) for node_id in nodes), default=0)
    if max_degree > 4:
        logger.warning(
            "Strict adjacency placement impossible: node degree exceeds 4 (max_degree=%s). Using relaxed placement.",
            max_degree,
        )
        return compute_relaxed_room_placement(
            graph,
            room_ids,
            sort_key=sort_key,
            node_position_getter=node_position_getter,
            first_free_position_fn=first_free_position_fn,
        )

    components = [sorted(comp, key=sort_key) for comp in nx.connected_components(spatial_graph.subgraph(nodes))]
    for node_id in nodes:
        if all(node_id not in comp for comp in components):
            components.append([node_id])
    components.sort(key=lambda comp: sort_key(comp[0]) if comp else (99, ""))

    placement: Dict[Any, Tuple[int, int]] = {}
    row_cursor = 0
    for comp in components:
        try:
            comp_positions = solve_component_strict_adjacency(
                comp,
                adjacency,
                explicit_pos,
                sort_key=sort_key,
                search_budget=strict_search_budget,
            )
        except ValueError as exc:
            logger.warning(
                "Strict adjacency solver failed for component %s (%s). Trying tree-preserving fallback.",
                comp,
                exc,
            )
            try:
                bfs_order, tree_adjacency = _component_tree_adjacency(
                    graph,
                    comp,
                    adjacency,
                    sort_key=sort_key,
                )
                comp_positions = solve_component_strict_adjacency(
                    bfs_order,
                    tree_adjacency,
                    {},
                    sort_key=sort_key,
                    search_budget=strict_search_budget,
                )
            except ValueError as tree_exc:
                logger.warning(
                    "Tree-preserving fallback also failed for component %s (%s). Using relaxed placement.",
                    comp,
                    tree_exc,
                )
                return compute_relaxed_room_placement(
                    graph,
                    room_ids,
                    sort_key=sort_key,
                    node_position_getter=node_position_getter,
                    first_free_position_fn=first_free_position_fn,
                )

        min_r = min(r for r, _ in comp_positions.values())
        max_r = max(r for r, _ in comp_positions.values())
        min_c = min(c for _, c in comp_positions.values())
        translated = {
            node_id: (r - min_r + row_cursor, c - min_c)
            for node_id, (r, c) in comp_positions.items()
        }
        placement.update(translated)
        row_cursor += (max_r - min_r + 2)

    return placement


def compute_graph_aware_room_slots(
    graph: nx.Graph,
    room_ids: List[Any],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
    node_position_getter: NodePositionGetter = get_node_grid_position,
    first_free_position_fn: Callable[[Tuple[int, int], set], Tuple[int, int]] = first_free_position,
    strict_search_budget: Optional[int] = None,
) -> Dict[Any, Tuple[int, int]]:
    """Compute one normalized slot position per room id using a shared policy."""
    placement = compute_strict_room_placement(
        graph,
        room_ids,
        sort_key=sort_key,
        node_position_getter=node_position_getter,
        first_free_position_fn=first_free_position_fn,
        strict_search_budget=strict_search_budget,
    )

    occupied = set(placement.values())
    next_row = max((r for r, _ in occupied), default=-1) + 1
    for room_id in sorted(room_ids, key=sort_key):
        if room_id in placement:
            continue
        pos = node_position_getter(graph, room_id) if room_id in graph else None
        start = (int(pos[0]), int(pos[1])) if pos is not None else (next_row, 0)
        resolved = first_free_position_fn(start, occupied)
        placement[room_id] = resolved
        occupied.add(resolved)
        next_row = max(next_row, resolved[0] + 1)

    if not placement:
        return {}

    min_r = min(r for r, _ in placement.values())
    min_c = min(c for _, c in placement.values())
    return {
        room_id: (r - min_r, c - min_c)
        for room_id, (r, c) in placement.items()
    }


def compute_layout_quality_metrics(
    graph: nx.Graph,
    slot_positions: Mapping[Any, Tuple[int, int]],
    *,
    sort_key: NodeSortKey = stable_node_sort_key,
    node_position_getter: NodePositionGetter = get_node_grid_position,
) -> Dict[str, Optional[float]]:
    """
    Compute layout-quality metrics that remain meaningful when graph coordinates are noisy.

    `graph_slot_match_rate` is retained for debugging, but the primary quality
    readout is edge-based:
    - how often graph-linked rooms are Manhattan-adjacent in stitched slots
    - how much extra slot distance graph edges incur on average
    """
    slots: Dict[Any, Tuple[int, int]] = {
        node_id: (int(pos[0]), int(pos[1]))
        for node_id, pos in slot_positions.items()
    }
    room_count = int(len(slots))

    graph_positions: Dict[Any, Tuple[int, int]] = {}
    for node_id in graph.nodes():
        if node_id not in slots:
            continue
        pos = node_position_getter(graph, node_id)
        if pos is None:
            continue
        graph_positions[node_id] = (int(pos[0]), int(pos[1]))

    normalized_graph_positions: Dict[Any, Tuple[int, int]] = {}
    if graph_positions:
        min_r = min(r for r, _ in graph_positions.values())
        min_c = min(c for _, c in graph_positions.values())
        normalized_graph_positions = {
            node_id: (int(r - min_r), int(c - min_c))
            for node_id, (r, c) in graph_positions.items()
        }

    comparable_nodes = sorted(normalized_graph_positions.keys(), key=sort_key)
    preference_distances: List[int] = []
    slot_match_count = 0
    for node_id in comparable_nodes:
        slot_pos = slots.get(node_id)
        graph_pos = normalized_graph_positions.get(node_id)
        if slot_pos is None or graph_pos is None:
            continue
        dist = abs(int(slot_pos[0]) - int(graph_pos[0])) + abs(int(slot_pos[1]) - int(graph_pos[1]))
        preference_distances.append(int(dist))
        if dist == 0:
            slot_match_count += 1

    comparable_count = int(len(preference_distances))
    duplicate_preferred_positions = max(0, int(len(graph_positions)) - int(len(set(graph_positions.values()))))

    edge_distances: List[int] = []
    seen_edges = set()
    for src, dst, edge_data in graph.edges(data=True):
        if not _is_spatial_room_edge(graph, src, dst, edge_data):
            continue
        if src not in slots or dst not in slots:
            continue
        edge_key = tuple(sorted((src, dst), key=sort_key))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        src_pos = slots[src]
        dst_pos = slots[dst]
        dist = abs(int(src_pos[0]) - int(dst_pos[0])) + abs(int(src_pos[1]) - int(dst_pos[1]))
        edge_distances.append(int(dist))

    comparable_edge_count = int(len(edge_distances))
    adjacent_edge_count = int(sum(1 for dist in edge_distances if dist == 1))
    mean_edge_distance = (
        float(sum(edge_distances) / comparable_edge_count)
        if comparable_edge_count > 0
        else None
    )
    mean_edge_excess = (
        float(sum(max(0, dist - 1) for dist in edge_distances) / comparable_edge_count)
        if comparable_edge_count > 0
        else None
    )

    return {
        "room_count": float(room_count),
        "graph_slot_match_rate": (
            float(slot_match_count / comparable_count)
            if comparable_count > 0
            else None
        ),
        "graph_position_coverage_rate": (
            float(comparable_count / max(1, room_count))
            if room_count > 0
            else None
        ),
        "graph_preferred_position_duplicate_rate": (
            float(duplicate_preferred_positions / max(1, len(graph_positions)))
            if graph_positions
            else None
        ),
        "graph_slot_preference_mean_distance": (
            float(sum(preference_distances) / comparable_count)
            if comparable_count > 0
            else None
        ),
        "graph_slot_preference_max_distance": (
            float(max(preference_distances))
            if preference_distances
            else None
        ),
        "graph_edge_slot_adjacency_rate": (
            float(adjacent_edge_count / comparable_edge_count)
            if comparable_edge_count > 0
            else None
        ),
        "graph_edge_slot_mean_distance": mean_edge_distance,
        "graph_edge_slot_mean_excess_distance": mean_edge_excess,
        "graph_edge_slot_max_distance": (
            float(max(edge_distances))
            if edge_distances
            else None
        ),
        "graph_edge_count_evaluated": float(comparable_edge_count),
    }


def _connector_tiles(
    edge_data: Optional[Dict[str, Any]],
    has_reverse_edge: bool,
) -> Tuple[int, int]:
    floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
    wall_id = int(SEMANTIC_PALETTE.get("WALL", 2))
    # Default connection tile is DOOR_OPEN so that the seal-then-carve
    # approach can always distinguish carved connections from stray floor.
    door_open_id = int(SEMANTIC_PALETTE.get("DOOR_OPEN", floor_id))

    data = edge_data or {}
    label = str(data.get("label", "") or "")
    edge_type = str(data.get("edge_type", data.get("type", "")) or "")
    edge_tokens = set(parse_edge_type_tokens(label=label, edge_type=edge_type))

    src_tile = door_open_id
    dst_tile = door_open_id

    if {"key_locked", "locked"}.intersection(edge_tokens):
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_LOCKED", floor_id))
        dst_tile = int(SEMANTIC_PALETTE.get("DOOR_LOCKED", floor_id))
    elif "boss_locked" in edge_tokens:
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_BOSS", floor_id))
        dst_tile = int(SEMANTIC_PALETTE.get("DOOR_BOSS", floor_id))
    elif "bombable" in edge_tokens:
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_BOMB", floor_id))
        dst_tile = int(SEMANTIC_PALETTE.get("DOOR_BOMB", floor_id))
    elif {"item_gate", "item_locked", "switch", "switch_locked", "on_off_gate", "state_block"}.intersection(edge_tokens):
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_PUZZLE", floor_id))
        dst_tile = int(SEMANTIC_PALETTE.get("DOOR_PUZZLE", floor_id))

    if (not has_reverse_edge) or {"soft_locked", "one_way", "shutter"}.intersection(edge_tokens):
        src_tile = int(SEMANTIC_PALETTE.get("DOOR_SOFT", src_tile))
        if dst_tile == wall_id:
            dst_tile = door_open_id

    return src_tile, dst_tile


def _bbox_center_row_col(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x_min, y_min, x_max, y_max = bbox
    return ((y_min + y_max) // 2, (x_min + x_max) // 2)


def _wall_off_corridor_path(
    global_grid: np.ndarray,
    corridor_cells: List[Tuple[int, int]],
    *,
    fill_tile: int,
) -> None:
    """Wrap relaxed-placement corridor floor with walls so it reads as a hallway."""
    if not corridor_cells:
        return

    wall_id = int(TileID.WALL)
    corridor_set = {(int(r), int(c)) for r, c in corridor_cells}
    h, w = global_grid.shape

    for row, col in corridor_set:
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row + d_row
            next_col = col + d_col
            if not (0 <= next_row < h and 0 <= next_col < w):
                continue
            if (next_row, next_col) in corridor_set:
                continue
            if int(global_grid[next_row, next_col]) != int(fill_tile):
                continue
            global_grid[next_row, next_col] = wall_id


def carve_room_connection_between_bboxes(
    global_grid: np.ndarray,
    src_bbox: Tuple[int, int, int, int],
    dst_bbox: Tuple[int, int, int, int],
    *,
    edge_data: Optional[Dict[str, Any]] = None,
    has_reverse_edge: bool = False,
    fill_tile: int = 0,
    diagnostic_callback: Optional[DiagnosticCallback] = None,
    connector_tile_resolver: Optional[ConnectorTileResolver] = None,
) -> None:
    """Carve a connection between two stitched room bounding boxes."""
    floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
    apron_replace_tiles = {
        int(fill_tile),
        int(TileID.WALL),
        int(TileID.BLOCK),
        int(TileID.ELEMENT),
        int(TileID.DOOR_OPEN),
        int(TileID.DOOR_LOCKED),
        int(TileID.DOOR_BOMB),
        int(TileID.DOOR_PUZZLE),
        int(TileID.DOOR_BOSS),
        int(TileID.DOOR_SOFT),
    }
    tile_resolver = connector_tile_resolver or _connector_tiles
    src_tile, dst_tile = tile_resolver(edge_data, has_reverse_edge)

    src_x_min, src_y_min, src_x_max, src_y_max = src_bbox
    dst_x_min, dst_y_min, dst_x_max, dst_y_max = dst_bbox

    def _door_rows(bbox: Tuple[int, int, int, int], direction: str) -> range:
        _x0, y0, _x1, y1 = bbox
        spec = DOOR_POSITIONS[direction]
        start = max(y0, y0 + int(spec["row_start"]))
        stop = min(y1, y0 + int(spec["row_end"]) - 1)
        return range(start, stop + 1)

    def _door_cols(bbox: Tuple[int, int, int, int], direction: str) -> range:
        x0, _y0, x1, _y1 = bbox
        spec = DOOR_POSITIONS[direction]
        start = max(x0, x0 + int(spec["col_start"]))
        stop = min(x1, x0 + int(spec["col_end"]) - 1)
        return range(start, stop + 1)

    def _canonical_overlap(first: range, second: range, low: int, high: int) -> List[int]:
        overlap = sorted(set(first).intersection(second))
        if overlap:
            return overlap
        center = max(low, min(high, (low + high) // 2))
        return [center]

    def _open_apron_cell(row: int, col: int) -> None:
        if not (0 <= row < global_grid.shape[0] and 0 <= col < global_grid.shape[1]):
            return
        if int(global_grid[row, col]) in apron_replace_tiles:
            global_grid[row, col] = floor_id

    if src_x_max + 1 == dst_x_min or dst_x_max + 1 == src_x_min:
        row_low = max(src_y_min + 1, dst_y_min + 1)
        row_high = min(src_y_max - 1, dst_y_max - 1)
        if row_low > row_high:
            row_low = max(src_y_min, dst_y_min)
            row_high = min(src_y_max, dst_y_max)
        if row_low <= row_high:
            if src_x_max < dst_x_min:
                src_boundary = src_x_max
                dst_boundary = dst_x_min
                src_apron = src_x_max - 1
                dst_apron = dst_x_min + 1
                rows = _canonical_overlap(_door_rows(src_bbox, "E"), _door_rows(dst_bbox, "W"), row_low, row_high)
            else:
                src_boundary = src_x_min
                dst_boundary = dst_x_max
                src_apron = src_x_min + 1
                dst_apron = dst_x_max - 1
                rows = _canonical_overlap(_door_rows(src_bbox, "W"), _door_rows(dst_bbox, "E"), row_low, row_high)
            for row in rows:
                global_grid[row, src_boundary] = src_tile
                global_grid[row, dst_boundary] = dst_tile
                _open_apron_cell(row, src_apron)
                _open_apron_cell(row, dst_apron)
            return

    if src_y_max + 1 == dst_y_min or dst_y_max + 1 == src_y_min:
        col_low = max(src_x_min + 1, dst_x_min + 1)
        col_high = min(src_x_max - 1, dst_x_max - 1)
        if col_low > col_high:
            col_low = max(src_x_min, dst_x_min)
            col_high = min(src_x_max, dst_x_max)
        if col_low <= col_high:
            if src_y_max < dst_y_min:
                src_boundary = src_y_max
                dst_boundary = dst_y_min
                src_apron = src_y_max - 1
                dst_apron = dst_y_min + 1
                cols = _canonical_overlap(_door_cols(src_bbox, "S"), _door_cols(dst_bbox, "N"), col_low, col_high)
            else:
                src_boundary = src_y_min
                dst_boundary = dst_y_max
                src_apron = src_y_min + 1
                dst_apron = dst_y_max - 1
                cols = _canonical_overlap(_door_cols(src_bbox, "N"), _door_cols(dst_bbox, "S"), col_low, col_high)
            for col in cols:
                global_grid[src_boundary, col] = src_tile
                global_grid[dst_boundary, col] = dst_tile
                _open_apron_cell(src_apron, col)
                _open_apron_cell(dst_apron, col)
            return

    src_r, src_c = _bbox_center_row_col(src_bbox)
    dst_r, dst_c = _bbox_center_row_col(dst_bbox)
    horizontal = abs(dst_c - src_c) >= abs(dst_r - src_r)

    if horizontal:
        src_anchor = (src_r, src_x_max if src_c <= dst_c else src_x_min)
        dst_anchor = (dst_r, dst_x_min if src_c <= dst_c else dst_x_max)
        start = (src_anchor[0], src_anchor[1] + (1 if src_c <= dst_c else -1))
        goal = (dst_anchor[0], dst_anchor[1] - (1 if src_c <= dst_c else -1))
    else:
        src_anchor = (src_y_max if src_r <= dst_r else src_y_min, src_c)
        dst_anchor = (dst_y_min if src_r <= dst_r else dst_y_max, dst_c)
        start = (src_anchor[0] + (1 if src_r <= dst_r else -1), src_anchor[1])
        goal = (dst_anchor[0] - (1 if src_r <= dst_r else -1), dst_anchor[1])

    H, W = global_grid.shape
    if not (0 <= start[0] < H and 0 <= start[1] < W and 0 <= goal[0] < H and 0 <= goal[1] < W):
        return

    global_grid[src_anchor[0], src_anchor[1]] = src_tile
    global_grid[dst_anchor[0], dst_anchor[1]] = dst_tile

    def neighbors(cell: Tuple[int, int]):
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                yield (nr, nc)

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    obstacle = global_grid != int(fill_tile)
    obstacle[start[0], start[1]] = False
    obstacle[goal[0], goal[1]] = False

    open_set: List[Tuple[int, int, Tuple[int, int]]] = []
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    gscore: Dict[Tuple[int, int], int] = {start: 0}
    closed = set()
    found = False
    expansions = 0
    max_expansions = int(max(1024, 4 * H * W))

    heappush(open_set, (heuristic(start, goal), 0, start))
    while open_set and expansions < max_expansions:
        _, cost, current = heappop(open_set)
        if cost > gscore.get(current, 10**12):
            continue
        if current in closed:
            continue
        closed.add(current)
        expansions += 1
        if current == goal:
            found = True
            break
        for nb in neighbors(current):
            if nb in closed or obstacle[nb[0], nb[1]]:
                continue
            tentative = cost + 1
            if nb not in gscore or tentative < gscore[nb]:
                gscore[nb] = tentative
                heappush(open_set, (tentative + heuristic(nb, goal), tentative, nb))
                came_from[nb] = current

    if found:
        current = goal
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        carved_cells: List[Tuple[int, int]] = []
        for row, col in path:
            if global_grid[row, col] == int(fill_tile):
                global_grid[row, col] = floor_id
                carved_cells.append((int(row), int(col)))
        _wall_off_corridor_path(
            global_grid,
            carved_cells,
            fill_tile=int(fill_tile),
        )
        return

    if expansions >= max_expansions and diagnostic_callback is not None:
        diagnostic_callback("corridor_astar_iteration_cap")
    if expansions >= max_expansions:
        logger.warning(
            "Corridor A* reached expansion cap (%d) between %s and %s; using fallback carve.",
            int(max_expansions),
            str(src_bbox),
            str(dst_bbox),
        )

    sr, sc = start
    tr, tc = goal
    step_c = 1 if tc >= sc else -1
    carved_cells: List[Tuple[int, int]] = []
    for col in range(sc, tc + step_c, step_c):
        if 0 <= sr < H and 0 <= col < W and global_grid[sr, col] == int(fill_tile):
            global_grid[sr, col] = floor_id
            carved_cells.append((int(sr), int(col)))
    step_r = 1 if tr >= sr else -1
    for row in range(sr, tr + step_r, step_r):
        if 0 <= row < H and 0 <= tc < W and global_grid[row, tc] == int(fill_tile):
            global_grid[row, tc] = floor_id
            carved_cells.append((int(row), int(tc)))
    _wall_off_corridor_path(
        global_grid,
        carved_cells,
        fill_tile=int(fill_tile),
    )


def seal_boundary_walls(
    dungeon_grid: np.ndarray,
    layout_map: Mapping[Any, Tuple[int, int, int, int]],
    graph: nx.Graph,
    slot_positions: Mapping[Any, Tuple[int, int]],
    *,
    fill_tile: int = 0,
) -> None:
    """Seal ALL room border tiles with walls.

    This must run BEFORE ``carve_room_connection_between_bboxes`` so that
    every border tile from the original AI-generated room grids is replaced
    with WALL.  The subsequent carving step then writes the correct DOOR
    tiles at the precise connection points, overwriting these walls.

    This two-pass approach (seal-then-carve) is simpler and more robust
    than trying to distinguish "real" carved doors from stray AI-generated
    door tiles after the fact.
    """
    wall_id = int(TileID.WALL)
    H, W = dungeon_grid.shape

    border_specs = [
        ("N", lambda b: ([b[1]], range(b[0], b[2] + 1))),
        ("S", lambda b: ([b[3]], range(b[0], b[2] + 1))),
        ("W", lambda b: (range(b[1], b[3] + 1), [b[0]])),
        ("E", lambda b: (range(b[1], b[3] + 1), [b[2]])),
    ]

    for room_id, bbox in layout_map.items():
        for _dir, range_fn in border_specs:
            rows, cols = range_fn(bbox)
            for row in rows:
                for col in cols:
                    if 0 <= row < H and 0 <= col < W:
                        dungeon_grid[row, col] = wall_id


def build_stitched_room_layout(
    rooms: Mapping[Any, Any],
    graph: nx.Graph,
    *,
    fill_tile: int = 0,
    sort_key: NodeSortKey = stable_node_sort_key,
    node_position_getter: NodePositionGetter = get_node_grid_position,
    first_free_position_fn: Callable[[Tuple[int, int], set], Tuple[int, int]] = first_free_position,
    enforce_room_dimensions: Optional[Tuple[int, int]] = None,
    carve_connections: bool = True,
    diagnostic_callback: Optional[DiagnosticCallback] = None,
    strict_search_budget: Optional[int] = None,
) -> StitchedRoomLayout:
    """Build a stitched dungeon grid and bbox layout map from room grids."""
    if not rooms:
        return StitchedRoomLayout(
            dungeon_grid=np.zeros((0, 0), dtype=np.int32),
            slot_positions={},
            room_offsets={},
            layout_map={},
        )

    room_grids: Dict[Any, np.ndarray] = {}
    for room_id, room_like in rooms.items():
        grid = extract_room_grid(room_like)
        if enforce_room_dimensions is not None and tuple(grid.shape) != tuple(enforce_room_dimensions):
            raise ValueError(
                "CRITICAL: Room dimension mismatch before stitching for "
                f"room {room_id}. Expected {enforce_room_dimensions[0]}x{enforce_room_dimensions[1]}, got {grid.shape[0]}x{grid.shape[1]}."
            )
        room_grids[room_id] = grid

    slot_positions = compute_graph_aware_room_slots(
        graph,
        list(room_grids.keys()),
        sort_key=sort_key,
        node_position_getter=node_position_getter,
        first_free_position_fn=first_free_position_fn,
        strict_search_budget=strict_search_budget,
    )

    stitched = build_room_canvas_from_slots(
        room_grids=room_grids,
        slot_positions=slot_positions,
        fill_tile=int(fill_tile),
    )

    # Seal ALL room border tiles with walls BEFORE carving connections.
    # This ensures stray floor/door tiles from AI-generated room grids
    # are replaced, then carving writes correct DOOR tiles on top.
    seal_boundary_walls(
        stitched.dungeon_grid,
        stitched.layout_map,
        graph,
        slot_positions,
        fill_tile=int(fill_tile),
    )

    if carve_connections:
        for u, v, edge_data in graph.edges(data=True):
            if not _is_spatial_room_edge(graph, u, v, edge_data):
                if diagnostic_callback is not None:
                    diagnostic_callback("non_spatial_graph_edge_not_carved")
                continue
            if u in stitched.layout_map and v in stitched.layout_map:
                carve_room_connection_between_bboxes(
                    stitched.dungeon_grid,
                    stitched.layout_map[u],
                    stitched.layout_map[v],
                    edge_data=edge_data or {},
                    has_reverse_edge=bool(graph.has_edge(v, u)),
                    fill_tile=int(fill_tile),
                    diagnostic_callback=diagnostic_callback,
                )

    return stitched


def build_room_canvas_from_slots(
    room_grids: Mapping[Any, np.ndarray],
    slot_positions: Mapping[Any, Tuple[int, int]],
    *,
    fill_tile: int = 0,
) -> StitchedRoomLayout:
    """Build a stitched canvas from precomputed slot positions."""
    if not room_grids or not slot_positions:
        return StitchedRoomLayout(
            dungeon_grid=np.zeros((0, 0), dtype=np.int32),
            slot_positions=dict(slot_positions),
            room_offsets={},
            layout_map={},
        )

    rows = sorted({row for row, _ in slot_positions.values()})
    cols = sorted({col for _, col in slot_positions.values()})
    row_to_idx = {row: idx for idx, row in enumerate(rows)}
    col_to_idx = {col: idx for idx, col in enumerate(cols)}

    row_heights = [0] * len(rows)
    col_widths = [0] * len(cols)
    for room_id, (slot_row, slot_col) in slot_positions.items():
        grid = room_grids[room_id]
        row_heights[row_to_idx[slot_row]] = max(row_heights[row_to_idx[slot_row]], int(grid.shape[0]))
        col_widths[col_to_idx[slot_col]] = max(col_widths[col_to_idx[slot_col]], int(grid.shape[1]))

    y_offsets = [0] * len(rows)
    x_offsets = [0] * len(cols)
    for idx in range(1, len(rows)):
        y_offsets[idx] = y_offsets[idx - 1] + row_heights[idx - 1]
    for idx in range(1, len(cols)):
        x_offsets[idx] = x_offsets[idx - 1] + col_widths[idx - 1]

    dungeon_grid = np.full((int(sum(row_heights)), int(sum(col_widths))), int(fill_tile), dtype=np.int32)
    room_offsets: Dict[Any, Tuple[int, int]] = {}
    layout_map: Dict[Any, Tuple[int, int, int, int]] = {}

    for room_id, grid in room_grids.items():
        slot_row, slot_col = slot_positions[room_id]
        y0 = y_offsets[row_to_idx[slot_row]]
        x0 = x_offsets[col_to_idx[slot_col]]
        h, w = int(grid.shape[0]), int(grid.shape[1])
        dungeon_grid[y0:y0 + h, x0:x0 + w] = grid
        room_offsets[room_id] = (y0, x0)
        layout_map[room_id] = (x0, y0, x0 + w - 1, y0 + h - 1)

    return StitchedRoomLayout(
        dungeon_grid=dungeon_grid,
        slot_positions=dict(slot_positions),
        room_offsets=room_offsets,
        layout_map=layout_map,
    )
