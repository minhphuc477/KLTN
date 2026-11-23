"""Graph extraction and topology conditioning builders for Zelda dungeons."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FLOOR_FEATURE_INDEX,
    GRAPH_NODE_FEATURE_DIM,
    GRAPH_TPE_DIM,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNELS,
    ROOM_TOPOLOGY_DIRECTIONAL_CHANNEL_GROUPS,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
    parse_edge_type_tokens,
    parse_node_label_tokens,
    select_primary_edge_type,
    semantic_grid_to_vglc_lines,
)
from src.core.condition_encoder import build_boundary_constraints
from src.pipeline.graph_features import (
    build_key_lock_pairs,
    compute_rrwp_edge_features,
    compute_tpe_features,
    encode_edge_feature_vector,
    extract_authoritative_floor_value,
    extract_node_feature_vector,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    build_puzzle_stage_condition_metadata,
    build_room_topology_condition_map,
    build_semantic_room_plan_trace,
    infer_puzzle_room_structure_enabled,
    nearest_walkable_point,
)
from src.pipeline.spatial_utils import (
    canonical_node_order,
    clamp_room_coord,
    earlier_generation_neighbors,
    parse_room_coord,
    stable_node_sort_key,
)
from src.utils.style_tokens import iter_style_metadata_candidates, resolve_style_token_id

logger = logging.getLogger(__name__)

__all__ = [
    "_extract_graph_from_dungeon",
    "_extract_graph_spatial_from_dungeon",
    "_build_room_graph_sample",
    "_room_role_flags",
    "_extract_explicit_style_id",
    "_EDGE_TYPE_ENCODING",
    "extract_start_goal",
    "validate_floor_conditioning_signal",
]


def extract_start_goal(
    grid: Any,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Extract start and goal positions from a dungeon or room grid."""
    if grid is None:
        return None, None
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    grid_arr = np.asarray(grid)
    if grid_arr.ndim == 3:
        grid_arr = grid_arr.squeeze(0)

    start_coords: Optional[Tuple[int, int]] = None
    goal_coords: Optional[Tuple[int, int]] = None

    start_id = SEMANTIC_PALETTE.get("START", 21)
    goal_id = SEMANTIC_PALETTE.get("TRIFORCE", 22)

    start_pos = np.where(grid_arr == start_id)
    if len(start_pos[0]) > 0:
        start_coords = (int(start_pos[0][0]), int(start_pos[1][0]))

    goal_pos = np.where(grid_arr == goal_id)
    if len(goal_pos[0]) > 0:
        goal_coords = (int(goal_pos[0][0]), int(goal_pos[1][0]))

    return start_coords, goal_coords


_EDGE_TYPE_ENCODING = {
    'open': 0, '': 0,
    'key_locked': 1, 'k': 1,
    'bombable': 2, 'b': 2,
    'soft_locked': 3, 'l': 3, 'one_way': 3, 'shutter': 3,
    'boss_locked': 4, 'K': 4,
    'item_locked': 5, 'I': 5,
    'stair': 6, 'stairs': 6, 'warp': 6,
    'switch': 7, 'switch_locked': 7, 'state_block': 7, 'on_off_gate': 7,
}


def _parse_label_tokens_local(label: Any) -> set:
    return {str(part).strip().lower() for part in parse_node_label_tokens(str(label or "")) if str(part).strip()}


def _coerce_bool_local(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


def _coerce_difficulty_local(value: Any) -> float:
    try:
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return 0.5


def _extract_explicit_style_id(
    room: Any,
    *,
    graph_node_attrs: Optional[Dict[str, Any]] = None,
    graph: Any = None,
) -> Optional[int]:
    """
    Extract an explicitly provided numeric style/theme token.

    We only forward numeric IDs that already exist in room/node/graph metadata.
    This avoids inventing a theme taxonomy from free-form labels or sector names.
    """
    graph_attrs = getattr(graph, "graph", None)
    candidate_values: List[Any] = []
    for key in ("style_id", "theme_id", "sector_theme_id", "sector_theme", "theme", "theme_name"):
        if room is not None and hasattr(room, key):
            candidate_values.append(getattr(room, key))
    candidate_values.extend(
        iter_style_metadata_candidates(graph_node_attrs, graph_attrs, keys=("style_id", "theme_id", "sector_theme_id"))
    )
    candidate_values.extend(
        iter_style_metadata_candidates(graph_node_attrs, graph_attrs, keys=("sector_theme", "theme", "theme_name"))
    )
    return resolve_style_token_id(*candidate_values)


def _extract_graph_spatial_from_dungeon(
    dungeon,
    *,
    node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
    edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
) -> dict:
    """Fallback graph extraction from room adjacency only."""
    nodes = []
    edges = []
    room_to_idx = {}

    for idx, (coord, _room) in enumerate(dungeon.rooms.items()):
        room_to_idx[coord] = idx
        nodes.append([0.0] * int(max(1, node_feature_dim)))

    for coord in dungeon.rooms:
        src_idx = room_to_idx[coord]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (coord[0] + dr, coord[1] + dc)
            if neighbor in room_to_idx:
                dst_idx = room_to_idx[neighbor]
                edges.append([src_idx, dst_idx])

    return {
        'node_features': np.array(nodes, dtype=np.float32) if nodes else np.zeros((0, int(max(1, node_feature_dim))), dtype=np.float32),
        'edge_index': np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64),
        'edge_attr': np.zeros((len(edges),), dtype=np.int64),
        'edge_features': np.zeros((len(edges), int(max(1, edge_feature_dim))), dtype=np.float32),
        'tpe': np.zeros((len(nodes), GRAPH_TPE_DIM), dtype=np.float32),
        'node_positions': np.stack(
            [
                np.arange(len(nodes), dtype=np.float32),
                np.zeros(len(nodes), dtype=np.float32),
            ],
            axis=1,
        ) if nodes else np.zeros((0, 2), dtype=np.float32),
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'start_node_id': -1,
        'node_to_idx': dict(room_to_idx),
        'floor_values': [0.0] * len(nodes),
        'floor_labels_present': [False] * len(nodes),
    }


def _extract_graph_from_dungeon(
    dungeon,
    *,
    node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
    edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
    allow_spatial_fallback: bool = False,
) -> dict:
    """Extract authoritative graph tensors from a stitched VGLC dungeon."""
    graph = getattr(dungeon, 'graph', None)
    if graph is None or len(graph.nodes()) == 0:
        if not bool(allow_spatial_fallback):
            raise ValueError(
                "Authoritative VGLC graph topology is missing; refusing to derive "
                "conditioning edges from the target stitched spatial layout. Pass "
                "allow_spatial_fallback=True only for explicit non-publication baselines."
            )
        return _extract_graph_spatial_from_dungeon(
            dungeon,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
        )

    nodes: List[List[float]] = []
    edges: List[List[int]] = []
    edge_attrs: List[int] = []
    edge_features_list: List[List[float]] = []
    node_id_to_idx: Dict[Any, int] = {}
    start_node_idx = -1
    node_positions: List[List[float]] = []
    floor_values: List[float] = []
    floor_labels_present: List[bool] = []

    room_position_by_graph_node = {}
    for room_pos, room in getattr(dungeon, "rooms", {}).items():
        graph_node_id = getattr(room, "graph_node_id", None)
        if graph_node_id is not None:
            room_position_by_graph_node[graph_node_id] = (int(room_pos[0]), int(room_pos[1]))

    canonical_ids = list(canonical_node_order(graph))
    canonical_rank = {node_id: rank for rank, node_id in enumerate(canonical_ids)}

    def _nearest_room_backed_node(node_id: Any) -> Optional[Any]:
        """Resolve semantic/virtual nodes to the nearest rendered room node."""
        if node_id in room_position_by_graph_node:
            return node_id
        # Preserve the adapter's explicit virtual-parent assignment first;
        # chained virtual nodes are resolved transitively.
        parent_visited = {node_id}
        parent = graph.nodes[node_id].get("virtual_parent")
        while parent is not None and parent not in parent_visited and parent in graph:
            if parent in room_position_by_graph_node:
                return parent
            parent_visited.add(parent)
            parent = graph.nodes[parent].get("virtual_parent")

        # Legacy corpus entries do not all carry the virtual flag/parent even
        # when no separate room was rendered.  Mirror the adapter's documented
        # topology-BFS policy and make equal-distance ties deterministic.
        queue = deque([node_id])
        visited = {node_id}
        while queue:
            current = queue.popleft()
            if graph.is_directed():
                neighbors = list(graph.successors(current)) + list(graph.predecessors(current))
            else:
                neighbors = list(graph.neighbors(current))
            neighbors = sorted(
                set(neighbors),
                key=lambda value: canonical_rank.get(value, len(canonical_rank)),
            )
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                if neighbor in room_position_by_graph_node:
                    return neighbor
                visited.add(neighbor)
                queue.append(neighbor)
        return None

    idx = 0
    ordered_node_rows = [(node_id, graph.nodes[node_id]) for node_id in canonical_ids]
    for node_id, data in ordered_node_rows:
        if data.get('is_start_pointer', False):
            continue

        node_id_to_idx[node_id] = idx
        node_features = extract_node_feature_vector(
            dict(data),
            node_dim=int(max(1, node_feature_dim)),
            device=torch.device("cpu"),
            parse_label_tokens=_parse_label_tokens_local,
            coerce_bool=_coerce_bool_local,
            coerce_difficulty=_coerce_difficulty_local,
        )
        nodes.append(node_features.cpu().numpy().astype(np.float32).tolist())
        floor_value = extract_authoritative_floor_value(data)
        floor_values.append(float(floor_value) if floor_value is not None else 0.0)
        floor_labels_present.append(floor_value is not None)
        pos = room_position_by_graph_node.get(_nearest_room_backed_node(node_id))
        if pos is None:
            pos = (idx, 0)
        node_positions.append([float(pos[0]), float(pos[1])])

        if data.get('is_start') and not data.get('is_start_pointer', False):
            start_node_idx = idx
        idx += 1

    if start_node_idx == -1:
        for node_id, data in graph.nodes(data=True):
            if data.get('is_start_pointer', False):
                for neighbor in list(graph.successors(node_id)) + list(graph.predecessors(node_id)):
                    if neighbor in node_id_to_idx:
                        start_node_idx = node_id_to_idx[neighbor]
                        break

    for u, v, data in graph.edges(data=True):
        if u not in node_id_to_idx or v not in node_id_to_idx:
            continue
        src_idx = node_id_to_idx[u]
        dst_idx = node_id_to_idx[v]
        edges.append([src_idx, dst_idx])
        constraints = parse_edge_type_tokens(
            label=data.get('label', ''),
            edge_type=data.get('edge_type', ''),
        )
        edge_type = select_primary_edge_type(constraints)
        edge_attrs.append(_EDGE_TYPE_ENCODING.get(edge_type, 0))
        edge_features_list.append(
            encode_edge_feature_vector(
                dict(data),
                edge_dim=int(max(1, edge_feature_dim)),
            )
        )

    node_features_arr = np.array(nodes, dtype=np.float32) if nodes else np.zeros((0, int(max(1, node_feature_dim))), dtype=np.float32)
    edge_index_arr = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_attr_arr = np.array(edge_attrs, dtype=np.int64) if edge_attrs else np.zeros((0,), dtype=np.int64)
    edge_features_arr = np.array(edge_features_list, dtype=np.float32) if edge_features_list else np.zeros((0, int(max(1, edge_feature_dim))), dtype=np.float32)
    node_positions_arr = np.array(node_positions, dtype=np.float32) if node_positions else np.zeros((0, 2), dtype=np.float32)
    edge_rrwp_tensor = compute_rrwp_edge_features(
        torch.tensor(edge_index_arr, dtype=torch.long),
        num_nodes=len(nodes),
        steps=int(GRAPH_TPE_DIM),
        device=torch.device("cpu"),
    )

    filtered_nodes = [
        node_id
        for node_id in canonical_node_order(graph)
        if node_id in node_id_to_idx
    ]
    # Every mission node needs an explicit differentiable room-passability
    # source. Semantic/virtual nodes without a separate rendered room reuse
    # their declared or topology-nearest physical room instead of silently
    # receiving the old constant-one fallback in LogicNet.
    node_passability_parent_idx: List[int] = []
    for node_id in filtered_nodes:
        data = graph.nodes[node_id]
        parent_id = _nearest_room_backed_node(node_id)
        node_passability_parent_idx.append(int(node_id_to_idx.get(parent_id, -1)))
    if len(filtered_nodes) <= 1:
        # Can't compute Laplacian eigenvectors on trivial graph - use zeros
        tpe_dim = int(GRAPH_TPE_DIM)
        tpe_tensor = torch.zeros((max(1, len(filtered_nodes)), tpe_dim), dtype=torch.float32)
    else:
        tpe_tensor = compute_tpe_features(
            graph=graph.subgraph(filtered_nodes).copy(),
            node_order=filtered_nodes,
            node_to_idx={node_id: i for i, node_id in enumerate(filtered_nodes)},
            node_features=torch.tensor(node_features_arr, dtype=torch.float32),
            device=torch.device("cpu"),
            parse_label_tokens=_parse_label_tokens_local,
            coerce_bool=_coerce_bool_local,
            coerce_difficulty=_coerce_difficulty_local,
        )
    start_node = next(
        (
            node_id
            for node_id, node_idx in node_id_to_idx.items()
            if int(node_idx) == int(start_node_idx)
        ),
        None,
    )
    target_node_idx = next(
        (
            int(node_id_to_idx[node_id])
            for node_id in filtered_nodes
            if int(node_id_to_idx[node_id]) < int(node_features_arr.shape[0])
            and int(node_features_arr.shape[1]) > 3
            and float(node_features_arr[int(node_id_to_idx[node_id]), 3]) > 0.5
        ),
        -1,
    )
    key_lock_pairs = build_key_lock_pairs(
        graph,
        node_id_to_idx,
        start_node=start_node,
    )
    return {
        'node_features': node_features_arr,
        'edge_index': edge_index_arr,
        'edge_attr': edge_attr_arr,
        'edge_features': edge_features_arr,
        'edge_rrwp': edge_rrwp_tensor.cpu().numpy().astype(np.float32),
        'tpe': tpe_tensor.cpu().numpy().astype(np.float32),
        'node_positions': node_positions_arr,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'start_node_id': start_node_idx,
        'target_idx': target_node_idx,
        'key_lock_pairs': key_lock_pairs,
        'node_to_idx': {node_id: node_id_to_idx[node_id] for node_id in filtered_nodes},
        'node_passability_parent_idx': np.asarray(node_passability_parent_idx, dtype=np.int64),
        'floor_values': floor_values,
        'floor_labels_present': floor_labels_present,
    }


def _incident_nodes_by_direction(
    dungeon,
    room_position: Tuple[int, int],
    graph_node_id: Any,
) -> Dict[str, List[Any]]:
    """Assign every non-self incident mission neighbor to a room direction."""
    assigned: Dict[str, List[Any]] = {
        direction: [] for direction in ("N", "S", "E", "W")
    }
    graph = getattr(dungeon, "graph", None)
    if graph is None or graph_node_id is None or graph_node_id not in graph:
        return assigned
    if graph.is_directed():
        incident = set(graph.predecessors(graph_node_id)) | set(graph.successors(graph_node_id))
    else:
        incident = set(graph.neighbors(graph_node_id))
    incident.discard(graph_node_id)
    room_position_by_node = {
        getattr(candidate_room, "graph_node_id", None): candidate_position
        for candidate_position, candidate_room in dungeon.rooms.items()
        if getattr(candidate_room, "graph_node_id", None) is not None
    }
    unresolved: List[Any] = []
    for neighbor_node in sorted(incident, key=stable_node_sort_key):
        neighbor_position = room_position_by_node.get(neighbor_node)
        direction = None
        if neighbor_position is not None:
            delta = (
                int(neighbor_position[0]) - int(room_position[0]),
                int(neighbor_position[1]) - int(room_position[1]),
            )
            direction = {
                (-1, 0): "N",
                (1, 0): "S",
                (0, 1): "E",
                (0, -1): "W",
            }.get(delta)
        if direction is not None:
            assigned[direction].append(neighbor_node)
        else:
            unresolved.append(neighbor_node)
    for neighbor_node in unresolved:
        free_direction = next(
            (direction for direction in ("N", "E", "S", "W") if not assigned[direction]),
            None,
        )
        if free_direction is None:
            free_direction = min(
                ("N", "E", "S", "W"),
                key=lambda direction: len(assigned[direction]),
            )
        assigned[free_direction].append(neighbor_node)
    return assigned


def _edge_constraint_tokens_by_direction(dungeon, room_position: Tuple[int, int], graph_node_id: Any) -> Dict[str, set]:
    """Collect boundary-edge constraints for one room from the mission graph."""
    edge_constraints: Dict[str, set] = {"N": set(), "S": set(), "E": set(), "W": set()}
    graph = getattr(dungeon, "graph", None)
    if graph is None or graph_node_id is None:
        return edge_constraints

    for direction, neighbor_nodes in _incident_nodes_by_direction(
        dungeon,
        room_position,
        graph_node_id,
    ).items():
        for neighbor_node in neighbor_nodes:
            for src, dst in ((graph_node_id, neighbor_node), (neighbor_node, graph_node_id)):
                if not graph.has_edge(src, dst):
                    continue
                edge_data = graph.get_edge_data(src, dst, default={}) or {}
                edge_constraints[direction].update(
                    parse_edge_type_tokens(
                        label=str(edge_data.get("label", "") or ""),
                        edge_type=str(edge_data.get("edge_type", edge_data.get("type", "")) or ""),
                    )
                )
    return edge_constraints


def _room_directional_flow(dungeon, room_position: Tuple[int, int], graph_node_id: Any) -> Tuple[Set[str], Set[str]]:
    """Infer incoming/outgoing doorway directions for one room from graph edges."""
    incoming: Set[str] = set()
    outgoing: Set[str] = set()
    graph = getattr(dungeon, "graph", None)
    if graph is None or graph_node_id is None:
        return incoming, outgoing

    for direction, neighbor_nodes in _incident_nodes_by_direction(
        dungeon,
        room_position,
        graph_node_id,
    ).items():
        for neighbor_node in neighbor_nodes:
            if not graph.is_directed():
                if graph.has_edge(graph_node_id, neighbor_node) or graph.has_edge(neighbor_node, graph_node_id):
                    incoming.add(direction)
                    outgoing.add(direction)
                continue
            if graph.has_edge(neighbor_node, graph_node_id):
                incoming.add(direction)
            if graph.has_edge(graph_node_id, neighbor_node):
                outgoing.add(direction)
    return incoming, outgoing


def _room_role_flags(room, graph_node_attrs: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    attrs = graph_node_attrs or {}
    tokens = {
        str(part).strip().lower()
        for part in parse_node_label_tokens(str(getattr(room, "node_label", "") or attrs.get("label", "")))
        if str(part).strip()
    }
    raw_type = str(attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or "").strip().lower()
    difficulty_rating = str(attrs.get("difficulty_rating", "") or "").strip().upper()
    return {
        "is_start": bool(getattr(room, "is_start", False) or attrs.get("is_start", False) or "s" in tokens or "start" in tokens),
        "has_enemy": bool(attrs.get("has_enemy", False) or "e" in tokens or "enemy" in tokens),
        "has_key": bool(attrs.get("has_key", False) or "k" in tokens or "key" in tokens),
        "has_item": bool(attrs.get("has_item", False) or "i" in tokens or "item" in tokens or "treasure" in tokens),
        "has_goal": bool(getattr(room, "has_triforce", False) or attrs.get("is_triforce", False) or "t" in tokens or "goal" in tokens or "triforce" in tokens),
        "has_boss": bool(getattr(room, "has_boss", False) or attrs.get("is_boss", False) or "b" in tokens or "boss" in tokens),
        "has_puzzle": bool(
            attrs.get("has_puzzle", False)
            or "p" in tokens
            or "puzzle" in tokens
            or raw_type in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"}
            or "puzzle" in raw_type
        ),
        "is_tutorial_puzzle": bool(attrs.get("is_tutorial", False) or raw_type == "tutorial_puzzle" or difficulty_rating == "SAFE"),
        "is_combat_puzzle": bool(raw_type == "combat_puzzle"),
        "is_complex_puzzle": bool(raw_type == "complex_puzzle" or difficulty_rating in {"HARD", "EXTREME"}),
        "is_switch_puzzle": bool(raw_type == "switch"),
    }


def _first_matching_tile(room_grid: np.ndarray, tile_ids: Set[int]) -> Optional[Tuple[int, int]]:
    for tile_id in tile_ids:
        hits = np.argwhere(room_grid == int(tile_id))
        if hits.size > 0:
            return (int(hits[0][0]), int(hits[0][1]))
    return None


def _content_anchor_points(
    room_grid: np.ndarray,
    room_role_flags: Dict[str, bool],
) -> Dict[str, Tuple[int, int]]:
    center = nearest_walkable_point(room_grid, (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
    anchors: Dict[str, Tuple[int, int]] = {}

    tile_anchor_specs = {
        "start": {int(SEMANTIC_PALETTE["START"])},
        "goal": {int(SEMANTIC_PALETTE["TRIFORCE"])},
        "key": {int(SEMANTIC_PALETTE["KEY_SMALL"]), int(SEMANTIC_PALETTE["KEY_BOSS"])},
        "item": {int(SEMANTIC_PALETTE["KEY_ITEM"]), int(SEMANTIC_PALETTE["ITEM_MINOR"])},
        "enemy": {int(SEMANTIC_PALETTE["ENEMY"])},
        "boss": {int(SEMANTIC_PALETTE["BOSS"])},
        "puzzle": {int(SEMANTIC_PALETTE["PUZZLE"])},
    }
    role_to_anchor = {
        "is_start": "start",
        "has_goal": "goal",
        "has_key": "key",
        "has_item": "item",
        "has_enemy": "enemy",
        "has_boss": "boss",
        "has_puzzle": "puzzle",
        "is_tutorial_puzzle": "puzzle",
        "is_combat_puzzle": "puzzle",
        "is_complex_puzzle": "puzzle",
        "is_switch_puzzle": "puzzle",
    }
    for role_key, anchor_name in role_to_anchor.items():
        if not room_role_flags.get(role_key, False):
            continue
        tile_point = _first_matching_tile(room_grid, tile_anchor_specs[anchor_name])
        snapped = nearest_walkable_point(room_grid, tile_point or (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
        if snapped is None:
            snapped = center
        if snapped is not None:
            anchors[anchor_name] = snapped
    return anchors


def _graph_room_start_goal(
    graph,
    graph_node_id: Any,
    *,
    incoming_dirs: Set[str],
    outgoing_dirs: Set[str],
    required_doors: Optional[Dict[str, bool]] = None,
    role_flags: Optional[Dict[str, Any]] = None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    attrs = {}
    if graph is not None and graph_node_id is not None and hasattr(graph, "nodes") and graph_node_id in graph.nodes:
        attrs = dict(graph.nodes[graph_node_id])

    start = (
        parse_room_coord(attrs.get("start_pos"))
        or parse_room_coord(attrs.get("entry_pos"))
        or parse_room_coord(attrs.get("entrance"))
    )
    goal = (
        parse_room_coord(attrs.get("goal_pos"))
        or parse_room_coord(attrs.get("exit_pos"))
        or parse_room_coord(attrs.get("exit"))
    )

    door_aprons = {
        "N": (2, ROOM_WIDTH // 2),
        "S": (ROOM_HEIGHT - 3, ROOM_WIDTH // 2),
        "W": (ROOM_HEIGHT // 2, 2),
        "E": (ROOM_HEIGHT // 2, ROOM_WIDTH - 3),
    }
    center = (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)
    dir_order = ("N", "S", "W", "E")
    req = dict(required_doors or {})
    active_dirs = [d for d in dir_order if bool(req.get(d, False))] if req else []
    incoming_active = [d for d in dir_order if d in incoming_dirs and (not req or bool(req.get(d, False)))]
    outgoing_active = [d for d in dir_order if d in outgoing_dirs and (not req or bool(req.get(d, False)))]
    flags = dict(role_flags or {})

    if start is None:
        if flags.get("is_start", False):
            start = center
        elif incoming_active:
            start = door_aprons[incoming_active[0]]
        elif active_dirs:
            start = door_aprons[active_dirs[0]]
        else:
            start = (ROOM_HEIGHT // 2, max(2, ROOM_WIDTH // 4))

    if goal is None:
        if flags.get("has_goal", False):
            goal = center
        elif outgoing_active:
            diff = [d for d in outgoing_active if door_aprons[d] != start]
            goal = door_aprons[diff[0] if diff else outgoing_active[0]]
        elif active_dirs:
            diff = [d for d in reversed(active_dirs) if door_aprons[d] != start]
            goal = door_aprons[diff[0]] if diff else center
        else:
            goal = (ROOM_HEIGHT // 2, min(ROOM_WIDTH - 3, (3 * ROOM_WIDTH) // 4))

    start = clamp_room_coord(start)
    goal = clamp_room_coord(goal)
    if start == goal:
        if start != center:
            goal = clamp_room_coord(center)
        else:
            alt_col = start[1] + 1 if start[1] + 1 <= ROOM_WIDTH - 3 else max(2, start[1] - 1)
            goal = clamp_room_coord((start[0], alt_col))
    return start, goal


def _build_room_graph_sample(
    dungeon,
    room_position: Tuple[int, int],
    room,
    base_graph: dict,
    *,
    topology_supervision_mode: str = "runtime_aligned",
    semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    puzzle_stage_topology_enabled: bool = False,
    puzzle_stage_trace_decay: float = DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
) -> dict:
    """Build one room-level graph-conditioning sample aligned with inference."""
    graph = getattr(dungeon, "graph", None)
    graph_node_id = getattr(room, "graph_node_id", None)
    node_to_idx = dict(base_graph.get("node_to_idx", {}) or {})
    current_node_idx = int(node_to_idx.get(graph_node_id, 0)) if node_to_idx else 0

    incident_nodes = _incident_nodes_by_direction(dungeon, room_position, graph_node_id)
    has_neighbor = {
        direction: bool(neighbor_nodes)
        for direction, neighbor_nodes in incident_nodes.items()
    }
    required_doors = dict(has_neighbor)
    incoming_dirs, outgoing_dirs = _room_directional_flow(dungeon, room_position, graph_node_id)

    graph_node_attrs = None
    if graph is not None and graph_node_id is not None and graph_node_id in graph.nodes:
        graph_node_attrs = dict(graph.nodes[graph_node_id])
    style_id = _extract_explicit_style_id(room, graph_node_attrs=graph_node_attrs, graph=graph)

    role_flags = _room_role_flags(room, graph_node_attrs)
    supervision_mode = str(topology_supervision_mode).strip().lower()
    if supervision_mode not in {"runtime_aligned", "oracle_room_grid"}:
        raise ValueError(
            f"topology_supervision_mode must be 'runtime_aligned' or 'oracle_room_grid', got {topology_supervision_mode!r}."
        )
    start, goal = _graph_room_start_goal(
        graph,
        graph_node_id,
        incoming_dirs=incoming_dirs,
        outgoing_dirs=outgoing_dirs,
        required_doors=required_doors,
        role_flags=role_flags,
    )
    traversability_trace = None
    if supervision_mode == "oracle_room_grid":
        room_grid = np.asarray(getattr(room, "semantic_grid", None), dtype=np.int32)
        oracle_start, oracle_goal = extract_start_goal(getattr(room, "semantic_grid", None))
        content_anchors = _content_anchor_points(room_grid, role_flags)
        if oracle_start is None:
            oracle_start = content_anchors.get("start")
        if oracle_goal is None:
            oracle_goal = content_anchors.get("goal")
        if oracle_start is None and bool(getattr(room, "is_start", False)):
            oracle_start = nearest_walkable_point(room_grid, (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
        if oracle_goal is None and bool(getattr(room, "has_triforce", False)):
            oracle_goal = nearest_walkable_point(room_grid, (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
        if oracle_start is not None:
            start = clamp_room_coord(oracle_start)
        if oracle_goal is not None:
            goal = clamp_room_coord(oracle_goal)
        traversability_trace = build_semantic_room_plan_trace(
            room_grid,
            start=start,
            goal=goal,
            required_doors=required_doors,
            incoming_dirs=incoming_dirs,
            outgoing_dirs=outgoing_dirs,
            edge_constraint_tokens=_edge_constraint_tokens_by_direction(dungeon, room_position, graph_node_id),
            room_role_flags=role_flags,
        )

    room_grid_for_structure = getattr(room, "semantic_grid", None)
    if room_grid_for_structure is None:
        room_grid_for_structure = getattr(room, "grid", np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32))
    room_grid_for_structure = np.asarray(room_grid_for_structure, dtype=np.int32)
    puzzle_room_structure_enabled = infer_puzzle_room_structure_enabled(
        room_grid_for_structure,
        role_flags,
    )
    puzzle_stage_condition = build_puzzle_stage_condition_metadata(
        room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
        start=start,
        goal=goal,
        required_doors=required_doors,
        incoming_dirs=incoming_dirs,
        outgoing_dirs=outgoing_dirs,
        edge_constraint_tokens=_edge_constraint_tokens_by_direction(dungeon, room_position, graph_node_id),
        room_role_flags=role_flags,
        room_grid=room_grid_for_structure,
        puzzle_structure_enabled=puzzle_room_structure_enabled,
        semantic_puzzle_offset=int(max(0, semantic_puzzle_offset)),
        stage_trace_decay=float(puzzle_stage_trace_decay),
    )
    if bool(puzzle_stage_topology_enabled):
        stage_trace = puzzle_stage_condition.get("stage_trace_mask")
        if isinstance(stage_trace, np.ndarray) and stage_trace.shape == (ROOM_HEIGHT, ROOM_WIDTH) and bool(np.any(stage_trace > 0)):
            traversability_trace = stage_trace.astype(np.float32, copy=False)
    serialized_stage_condition = dict(puzzle_stage_condition)
    serialized_stage_condition.pop("stage_trace_mask", None)

    room_topology_map = build_room_topology_condition_map(
        room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
        start=start,
        goal=goal,
        required_doors=required_doors,
        edge_constraint_tokens=_edge_constraint_tokens_by_direction(dungeon, room_position, graph_node_id),
        room_role_flags=role_flags,
        traversability_trace=traversability_trace,
        semantic_role_prior_strength=float(semantic_role_prior_strength),
        semantic_puzzle_offset=int(max(0, semantic_puzzle_offset)),
        puzzle_stage_topology_enabled=bool(puzzle_stage_topology_enabled),
        puzzle_stage_trace_decay=float(puzzle_stage_trace_decay),
        puzzle_structure_enabled=puzzle_room_structure_enabled,
    )

    def _direction_mask(directions: Set[str]) -> np.ndarray:
        mask = np.zeros((1, ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
        apron_slices = {
            "N": (slice(2, 3), slice(4, 7)),
            "S": (slice(ROOM_HEIGHT - 3, ROOM_HEIGHT - 2), slice(4, 7)),
            "W": (slice(7, 10), slice(2, 3)),
            "E": (slice(7, 10), slice(ROOM_WIDTH - 3, ROOM_WIDTH - 2)),
        }
        for direction in directions:
            rs, cs = apron_slices.get(str(direction).upper(), (None, None))
            if rs is not None and cs is not None:
                mask[0, rs, cs] = 1.0
        return mask

    logic_source_mask = _direction_mask(incoming_dirs)
    logic_target_mask = _direction_mask(outgoing_dirs)
    neighbor_maps: Dict[str, Optional[np.ndarray]] = {
        direction: None for direction in ("N", "S", "E", "W")
    }
    available_neighbor_nodes = earlier_generation_neighbors(graph, graph_node_id)
    room_by_node = {
        getattr(candidate_room, "graph_node_id", None): (candidate_pos, candidate_room)
        for candidate_pos, candidate_room in dungeon.rooms.items()
        if getattr(candidate_room, "graph_node_id", None) is not None
    }
    unresolved_neighbors: List[np.ndarray] = []
    for neighbor_node_id in sorted(available_neighbor_nodes, key=stable_node_sort_key):
        room_record = room_by_node.get(neighbor_node_id)
        if room_record is None:
            continue
        neighbor_pos, neighbor_room = room_record
        neighbor_grid = getattr(neighbor_room, "semantic_grid", None)
        if neighbor_grid is None:
            continue
        delta = (
            int(neighbor_pos[0]) - int(room_position[0]),
            int(neighbor_pos[1]) - int(room_position[1]),
        )
        direction = {
            (-1, 0): "N",
            (1, 0): "S",
            (0, 1): "E",
            (0, -1): "W",
        }.get(delta)
        neighbor_array = np.asarray(neighbor_grid, dtype=np.float32)
        if direction is not None and neighbor_maps[direction] is None:
            neighbor_maps[direction] = neighbor_array
        else:
            unresolved_neighbors.append(neighbor_array)

    for neighbor_array in unresolved_neighbors:
        free_direction = next(
            (direction for direction in ("N", "W", "E", "S") if neighbor_maps[direction] is None),
            None,
        )
        if free_direction is None:
            break
        neighbor_maps[free_direction] = neighbor_array

    return {
        'node_features': base_graph['node_features'],
        'edge_index': base_graph['edge_index'],
        'edge_attr': base_graph.get('edge_attr', np.zeros((0,), dtype=np.int64)),
        'edge_features': base_graph.get('edge_features', np.zeros((0, GRAPH_EDGE_FEATURE_DIM), dtype=np.float32)),
        'edge_rrwp': base_graph.get('edge_rrwp', np.zeros((base_graph.get('num_edges', 0), GRAPH_TPE_DIM), dtype=np.float32)),
        'tpe': base_graph.get('tpe', np.zeros((base_graph.get('num_nodes', 0), GRAPH_TPE_DIM), dtype=np.float32)),
        'node_positions': base_graph.get('node_positions', np.zeros((base_graph.get('num_nodes', 0), 2), dtype=np.float32)),
        'num_nodes': int(base_graph.get('num_nodes', 0)),
        'num_edges': int(base_graph.get('num_edges', 0)),
        'start_node_id': int(base_graph.get('start_node_id', -1)),
        'target_idx': int(base_graph.get('target_idx', -1)),
        'key_lock_pairs': list(base_graph.get('key_lock_pairs', []) or []),
        'floor_values': list(base_graph.get('floor_values', []) or []),
        'floor_labels_present': list(base_graph.get('floor_labels_present', []) or []),
        'node_to_idx': node_to_idx,
        'node_passability_parent_idx': base_graph.get(
            'node_passability_parent_idx',
            np.arange(int(base_graph.get('num_nodes', 0)), dtype=np.int64),
        ),
        'current_node_idx': current_node_idx,
        'room_position': np.array([float(room_position[0]), float(room_position[1])], dtype=np.float32),
        'boundary_constraints': build_boundary_constraints(has_neighbor=has_neighbor, required_door=required_doors).numpy().astype(np.float32),
        'room_topology_map': room_topology_map.astype(np.float32),
        'logic_source_mask': logic_source_mask,
        'logic_target_mask': logic_target_mask,
        'neighbor_maps': neighbor_maps,
        'topology_supervision_mode': supervision_mode,
        'has_puzzle': bool(role_flags.get("has_puzzle", False)),
        'puzzle_room_structure_enabled': bool(puzzle_room_structure_enabled),
        'puzzle_stage_condition': serialized_stage_condition,
        **({'style_id': int(style_id)} if style_id is not None else {}),
    }



def validate_floor_conditioning_signal(
    dataset: Dataset,
    *,
    node_feature_dim: int,
) -> None:
    """Reject a floor-conditioning ablation without observable floor labels.

    Feature widths up to the checkpoint-compatible baseline do not use floor
    conditioning. Wider schemas include floor/z at index 14 and therefore
    require at least one completely labelled dungeon containing two or more
    floors. Variation only between separate single-floor dungeons does not
    demonstrate within-dungeon floor conditioning.
    """
    floor_index = int(GRAPH_NODE_FLOOR_FEATURE_INDEX)
    if int(node_feature_dim) <= floor_index:
        return
    if int(node_feature_dim) != floor_index + 1:
        raise ValueError(
            "The supported floor-conditioning schema has exactly "
            f"{floor_index + 1} node features; got node_feature_dim={node_feature_dim}."
        )

    observed_floor_values: Set[float] = set()
    saw_authoritative_metadata = False
    saw_complete_floor_graph = False

    def _inspect_graph(graph: Any) -> Tuple[bool, bool]:
        nonlocal saw_authoritative_metadata, saw_complete_floor_graph
        if not isinstance(graph, dict):
            return False, False
        features = graph.get("node_features")
        if features is None:
            return False, False
        tensor = torch.as_tensor(features)
        if tensor.dim() != 2 or int(tensor.shape[1]) <= floor_index:
            raise ValueError(
                "Floor conditioning requires node_features with at least "
                f"{floor_index + 1} columns; got {tuple(tensor.shape)}."
            )
        floor_values = graph.get("floor_values")
        labels_present = graph.get("floor_labels_present")
        if floor_values is None or labels_present is None:
            return True, False
        saw_authoritative_metadata = True
        values = list(floor_values)
        present = [bool(value) for value in labels_present]
        node_count = int(tensor.shape[0])
        if len(values) != node_count or len(present) != node_count or not all(present):
            return True, False
        saw_complete_floor_graph = True
        graph_values = {
            round(float(value), 8)
            for value in values
            if np.isfinite(float(value))
        }
        observed_floor_values.update(graph_values)
        if len(graph_values) >= 2:
            return True, True
        return True, False

    observed_graph = False
    stored_graphs = getattr(dataset, "graphs", None)
    if isinstance(stored_graphs, list) and stored_graphs:
        for graph in stored_graphs:
            inspected, valid_multi_floor = _inspect_graph(graph)
            observed_graph = observed_graph or inspected
            if valid_multi_floor:
                return
    else:
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]
            graph = sample[1] if isinstance(sample, (tuple, list)) and len(sample) > 1 else None
            inspected, valid_multi_floor = _inspect_graph(graph)
            observed_graph = observed_graph or inspected
            if valid_multi_floor:
                return

    if not observed_graph:
        detail = "no graph features were observed"
    elif not saw_authoritative_metadata:
        detail = "authoritative raw floor metadata was not preserved"
    elif not saw_complete_floor_graph:
        detail = "no dungeon had floor labels for every conditioned node"
    else:
        detail = (
            "fewer than two distinct floor labels were observed within any one "
            f"dungeon ({sorted(observed_floor_values)})"
        )
    raise ValueError(
        "node_feature_dim enables neural floor conditioning, but "
        f"{detail}. Use the 14-feature baseline for single-floor data or provide "
        "authoritative multi-floor node positions before running this ablation."
    )


