"""
Zelda Dungeon Dataset Loader
============================

PyTorch Dataset and DataLoader for loading Zelda dungeon grids from text files
or from the existing VGLC format via zelda_core.

Supports:
1. Raw text files with ASCII dungeon grids
2. VGLC format via ZeldaDungeonAdapter
3. NumPy array conversion with proper semantic IDs

References:
- VGLC: Video Game Level Corpus (https://github.com/TheVGLC/TheVGLC)
- Local processed corpus uses row-major room arrays with shape `(16, 11)`
- In screen terms, the same room is often described as `16 columns x 11 rows`
"""

import os
import logging
import numpy as np
from typing import Optional, Callable, Tuple, Union, Any, Dict, List, Set, Iterable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

logger = logging.getLogger(__name__)

# =============================================================================
# TILE MAPPINGS
# =============================================================================

# Simple ASCII mapping for basic text files (legacy support)
TILE_MAPPING = {
    'F': 0,   # Floor
    'W': 1,   # Wall
    'D': 2,   # Door
    'K': 3,   # Key
    'L': 4,   # Locked door
    'E': 5,   # Enemy
    'S': 6,   # Start
    'G': 7,   # Goal/Triforce
    '.': 0,   # Floor (alternate)
    '-': -1,  # Void
}

# Import semantic palette from local zelda_core module
from .zelda_core import (
    ZeldaDungeonAdapter
)
from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    GRAPH_TPE_DIM,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
    semantic_grid_to_vglc_lines,
    parse_node_label_tokens,
    parse_edge_type_tokens,
    select_primary_edge_type,
)
from src.core.condition_encoder import build_boundary_constraints
from src.pipeline.graph_features import (
    compute_rrwp_edge_features,
    compute_tpe_features,
    encode_edge_feature_vector,
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
from src.pipeline.spatial_utils import clamp_room_coord, parse_room_coord
from src.utils.style_tokens import iter_style_metadata_candidates, resolve_style_token_id
from src.zelda_data.splits import normalize_dungeon_ids, normalize_variants
VGLC_AVAILABLE = True
logger.info("VGLC adapter available via zelda_core")


_EDGE_TYPE_ENCODING = {
    'open': 0, '': 0,
    'key_locked': 1, 'k': 1,
    'bombable': 2, 'b': 2,
    'soft_locked': 3, 'l': 3, 'one_way': 3, 'shutter': 3,
    'boss_locked': 4, 'K': 4,
    'item_locked': 5, 'I': 5,
    'stair': 6, 'stairs': 6, 'warp': 6, 's': 6,
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
    }


def _extract_graph_from_dungeon(
    dungeon,
    *,
    node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
    edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
) -> dict:
    """Extract authoritative graph tensors from a stitched VGLC dungeon."""
    graph = getattr(dungeon, 'graph', None)
    if graph is None or len(graph.nodes()) == 0:
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

    room_position_by_graph_node = {}
    for room_pos, room in getattr(dungeon, "rooms", {}).items():
        graph_node_id = getattr(room, "graph_node_id", None)
        if graph_node_id is not None:
            room_position_by_graph_node[graph_node_id] = (int(room_pos[0]), int(room_pos[1]))

    idx = 0
    for node_id, data in sorted(graph.nodes(data=True)):
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
        pos = room_position_by_graph_node.get(node_id)
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

    filtered_nodes = [node_id for node_id in sorted(graph.nodes()) if node_id in node_id_to_idx]
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
        'node_to_idx': {node_id: node_id_to_idx[node_id] for node_id in filtered_nodes},
    }


def _edge_constraint_tokens_by_direction(dungeon, room_position: Tuple[int, int], graph_node_id: Any) -> Dict[str, set]:
    """Collect boundary-edge constraints for one room from the mission graph."""
    edge_constraints: Dict[str, set] = {"N": set(), "S": set(), "E": set(), "W": set()}
    graph = getattr(dungeon, "graph", None)
    if graph is None or graph_node_id is None:
        return edge_constraints

    direction_to_neighbor = {
        "N": (room_position[0] - 1, room_position[1]),
        "S": (room_position[0] + 1, room_position[1]),
        "E": (room_position[0], room_position[1] + 1),
        "W": (room_position[0], room_position[1] - 1),
    }
    for direction, neighbor_pos in direction_to_neighbor.items():
        neighbor_room = dungeon.rooms.get(neighbor_pos)
        neighbor_node = getattr(neighbor_room, "graph_node_id", None) if neighbor_room is not None else None
        if neighbor_node is None:
            continue
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

    direction_to_neighbor = {
        "N": (room_position[0] - 1, room_position[1]),
        "S": (room_position[0] + 1, room_position[1]),
        "E": (room_position[0], room_position[1] + 1),
        "W": (room_position[0], room_position[1] - 1),
    }
    for direction, neighbor_pos in direction_to_neighbor.items():
        neighbor_room = dungeon.rooms.get(neighbor_pos)
        neighbor_node = getattr(neighbor_room, "graph_node_id", None) if neighbor_room is not None else None
        if neighbor_node is None:
            continue
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

    if start is None:
        start = (ROOM_HEIGHT // 2, 0) if incoming_dirs else (ROOM_HEIGHT // 2, ROOM_WIDTH // 4)
    if goal is None:
        goal = (ROOM_HEIGHT // 2, ROOM_WIDTH - 1) if outgoing_dirs else (ROOM_HEIGHT // 2, (3 * ROOM_WIDTH) // 4)

    start = clamp_room_coord(start)
    goal = clamp_room_coord(goal)
    if start == goal:
        goal = clamp_room_coord((goal[0], goal[1] + 1))
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

    has_neighbor = {
        "N": (room_position[0] - 1, room_position[1]) in dungeon.rooms,
        "S": (room_position[0] + 1, room_position[1]) in dungeon.rooms,
        "E": (room_position[0], room_position[1] + 1) in dungeon.rooms,
        "W": (room_position[0], room_position[1] - 1) in dungeon.rooms,
    }
    required_doors = {direction: bool(getattr(room, "doors", {}).get(direction, False)) for direction in ("N", "S", "E", "W")}
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
    )
    puzzle_room_structure_enabled = infer_puzzle_room_structure_enabled(
        room_grid_for_structure,
        role_flags,
    )

    neighbor_maps: Dict[str, Optional[np.ndarray]] = {}
    direction_to_neighbor = {
        "N": (room_position[0] - 1, room_position[1]),
        "S": (room_position[0] + 1, room_position[1]),
        "E": (room_position[0], room_position[1] + 1),
        "W": (room_position[0], room_position[1] - 1),
    }
    for direction, neighbor_pos in direction_to_neighbor.items():
        neighbor_room = dungeon.rooms.get(neighbor_pos)
        neighbor_grid = getattr(neighbor_room, "semantic_grid", None) if neighbor_room is not None else None
        if neighbor_grid is None:
            neighbor_maps[direction] = None
        else:
            neighbor_maps[direction] = np.asarray(neighbor_grid, dtype=np.float32)

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
        'node_to_idx': node_to_idx,
        'current_node_idx': current_node_idx,
        'room_position': np.array([float(room_position[0]), float(room_position[1])], dtype=np.float32),
        'boundary_constraints': build_boundary_constraints(has_neighbor=has_neighbor, required_door=required_doors).numpy().astype(np.float32),
        'room_topology_map': room_topology_map.astype(np.float32),
        'neighbor_maps': neighbor_maps,
        'topology_supervision_mode': supervision_mode,
        'has_puzzle': bool(role_flags.get("has_puzzle", False)),
        'puzzle_room_structure_enabled': bool(puzzle_room_structure_enabled),
        'puzzle_stage_condition': serialized_stage_condition,
        **({'style_id': int(style_id)} if style_id is not None else {}),
    }


# =============================================================================
# DATASET CLASS
# =============================================================================

class ZeldaDungeonDataset(Dataset):
    """
    PyTorch Dataset for Zelda dungeon grids.
    
    Supports loading from:
    1. Directory of .txt files (ASCII format)
    2. VGLC format via ZeldaDungeonAdapter
    3. Pre-loaded numpy arrays
    4. Paired NPZ format with (image, graph) pairs
    
    Args:
        data_dir: Directory containing dungeon files or VGLC data
        transform: Optional transform to apply to each sample
        use_vglc: Whether to use VGLC format via ZeldaDungeonAdapter
        normalize: Whether to normalize values to [0, 1]
        target_size: Target (height, width) for resizing, None for original
        load_graphs: Whether to load graph data for dual-stream training
        
    Returns:
        torch.Tensor of shape (1, H, W) representing the dungeon grid
        OR (image_tensor, graph_dict) if load_graphs=True
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        use_vglc: bool = False,
        normalize: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        pad_to_max: bool = True,  # Pad all samples to max size for batching
        load_graphs: bool = False,  # NEW: Load graph data for dual-stream
        node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
        dungeon_ids: Optional[Iterable[int]] = None,
        variants: Optional[Iterable[int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.normalize = normalize
        self.target_size = target_size
        self.use_vglc = use_vglc and VGLC_AVAILABLE
        self.pad_to_max = pad_to_max
        self.load_graphs = load_graphs
        self.node_feature_dim = int(max(1, node_feature_dim))
        self.edge_feature_dim = int(max(1, edge_feature_dim))
        self.dungeon_ids = normalize_dungeon_ids(dungeon_ids)
        self.variants = normalize_variants(variants)
        self.sample_metadata: List[Dict[str, Any]] = []
        
        # Track max dimensions for padding
        self.max_h = 0
        self.max_w = 0
        
        # Graph data storage
        self.graphs = [] if load_graphs else None
        
        if self.use_vglc:
            self._init_vglc()
        else:
            self._init_text_files()
        
        # If target_size not specified but pad_to_max is True, use max dims
        if self.target_size is None and self.pad_to_max and self.max_h > 0:
            self.target_size = (self.max_h, self.max_w)
            logger.info(f"Auto-set target_size to ({self.max_h}, {self.max_w}) for batching")
            
        logger.info(f"Loaded {len(self)} dungeon samples from {data_dir}")
    
    def _init_text_files(self) -> None:
        """Initialize dataset from text files."""
        self.files = [
            self.data_dir / f 
            for f in os.listdir(self.data_dir) 
            if f.endswith('.txt')
        ]
        self.samples = None  # Lazy loading
        
    def _init_vglc(self) -> None:
        """Initialize dataset from VGLC format."""
        self.files = []
        self.samples = []
        
        # Load all dungeons via adapter
        adapter = ZeldaDungeonAdapter(str(self.data_dir))
        
        dungeon_iter = self.dungeon_ids if self.dungeon_ids is not None else range(1, 10)
        for dungeon_num in dungeon_iter:  # Dungeons 1-9
            for variant in self.variants:  # Two quest variants
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant)
                    stitched = adapter.stitch_dungeon(dungeon)
                    grid = stitched.global_grid
                    self.samples.append(grid.astype(np.float32))
                    self.sample_metadata.append(
                        {
                            "dungeon_num": int(dungeon_num),
                            "variant": int(variant),
                            "dungeon_id": f"tloz{int(dungeon_num)}_{int(variant)}",
                        }
                    )
                    
                    # Extract graph if load_graphs is enabled
                    if self.load_graphs:
                        graph = self._extract_graph(dungeon)
                        self.graphs.append(graph)
                    
                    # Track max dimensions
                    h, w = grid.shape
                    self.max_h = max(self.max_h, h)
                    self.max_w = max(self.max_w, w)
                    
                    logger.debug(f"Loaded dungeon {dungeon_num} variant {variant}: {h}x{w}")
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to load dungeon {dungeon_num}v{variant}: {e}")
                    
        logger.info(f"Loaded {len(self.samples)} VGLC dungeons (max size: {self.max_h}x{self.max_w})")
    
    def _extract_graph(self, dungeon) -> dict:
        """Extract graph structure from dungeon for GNN training.
        
        Uses the DOT graph topology (dungeon.graph) as the authoritative
        source for node features and edge types. The 's' (start pointer)
        node is NOT included as a room node -- instead, its connected room
        is marked with start_node_id.
        """
        return _extract_graph_from_dungeon(
            dungeon,
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
        )
    
    def _extract_graph_spatial(self, dungeon) -> dict:
        """Fallback: extract graph from spatial adjacency when DOT graph unavailable."""
        return _extract_graph_spatial_from_dungeon(
            dungeon,
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
        )
    
    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Get a single dungeon grid as a tensor.
        
        Returns:
            If load_graphs=False: torch.Tensor of shape (1, H, W)
            If load_graphs=True: (image_tensor, graph_dict) tuple
        """
        if self.samples is not None:
            grid = self.samples[idx]
        else:
            grid = self._load_text_file(self.files[idx])
        
        # Convert to tensor
        tensor_map = torch.tensor(grid, dtype=torch.float32)
        
        # Add channel dimension if needed
        if tensor_map.dim() == 2:
            tensor_map = tensor_map.unsqueeze(0)
        
        # Normalize to [0, 1] using fixed num_classes divisor
        # IMPORTANT: Use fixed constant (43 = max tile ID) so that
        # grids_to_onehot / encode_to_latent can invert with *43 exactly.
        if self.normalize and tensor_map.max() > 1:
            NUM_TILE_IDS = 43  # TileID.PUZZLE = 43, the highest ID
            tensor_map = tensor_map / NUM_TILE_IDS
        
        # Resize if target size specified
        if self.target_size is not None:
            tensor_map = self._resize(tensor_map, self.target_size)
        
        # Apply custom transform
        if self.transform:
            tensor_map = self.transform(tensor_map)
        
        # Return with graph if requested
        if self.load_graphs and self.graphs is not None:
            graph = self.graphs[idx]
            edge_feature_dim = int(getattr(self, "edge_feature_dim", GRAPH_EDGE_FEATURE_DIM))
            return tensor_map, {
                'node_features': torch.tensor(graph['node_features'], dtype=torch.float32),
                'edge_index': torch.tensor(graph['edge_index'], dtype=torch.long),
                'edge_attr': torch.tensor(graph.get('edge_attr', np.zeros((0,), dtype=np.int64)), dtype=torch.long),
                'edge_features': torch.tensor(graph.get('edge_features', np.zeros((0, edge_feature_dim), dtype=np.float32)), dtype=torch.float32),
                'edge_rrwp': torch.tensor(graph.get('edge_rrwp', np.zeros((int(graph.get('num_edges', 0)), GRAPH_TPE_DIM), dtype=np.float32)), dtype=torch.float32),
                'tpe': torch.tensor(graph.get('tpe', np.zeros((0, GRAPH_TPE_DIM), dtype=np.float32)), dtype=torch.float32),
                'node_positions': torch.tensor(graph.get('node_positions', np.zeros((0, 2), dtype=np.float32)), dtype=torch.float32),
                'num_nodes': graph['num_nodes'],
                'num_edges': graph['num_edges'],
                'start_node_id': graph.get('start_node_id', -1),
                'node_to_idx': dict(graph.get('node_to_idx', {})),
            }
            
        return tensor_map
    
    def _load_text_file(self, filepath: Path) -> np.ndarray:
        """Load dungeon grid from text file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        dungeon_grid = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = [TILE_MAPPING.get(c, 0) for c in line]
            dungeon_grid.append(row)
        
        return np.array(dungeon_grid, dtype=np.float32)
    
    def _resize(self, tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Resize or pad tensor to target size for batching compatibility."""
        import torch.nn.functional as F
        
        target_h, target_w = size
        current_h, current_w = tensor.shape[-2], tensor.shape[-1]
        
        # If already correct size, return as-is
        if current_h == target_h and current_w == target_w:
            return tensor
        
        # Pad to target size (zero-padding for void areas)
        pad_h = target_h - current_h
        pad_w = target_w - current_w
        
        if pad_h >= 0 and pad_w >= 0:
            # Pad on right and bottom (left, right, top, bottom)
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            # Need to crop or interpolate if target is smaller
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            tensor = F.interpolate(tensor, size=size, mode='nearest')
            tensor = tensor.squeeze(0)  # (C, H, W)
        
        return tensor
    
    def get_raw_grid(self, idx: int) -> np.ndarray:
        """Get raw numpy array for a dungeon (before transforms)."""
        if self.samples is not None:
            return self.samples[idx]
        return self._load_text_file(self.files[idx])


# =============================================================================
# ROOM-LEVEL DATASET
# =============================================================================

class ZeldaRoomDataset(Dataset):
    """
    Dataset for individual rooms extracted from dungeons.
    
    Extracts canonical `(ROOM_HEIGHT, ROOM_WIDTH)` rooms from larger dungeon grids for training
    room-level generation models.
    
    Args:
        data_dir: Directory with VGLC data
        transform: Optional transform for each room
        normalize: Normalize to [0, 1]
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        load_graphs: bool = False,
        node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
        topology_supervision_mode: str = "runtime_aligned",
        semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
        semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
        puzzle_stage_topology_enabled: bool = False,
        puzzle_stage_trace_decay: float = DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
        dungeon_ids: Optional[Iterable[int]] = None,
        variants: Optional[Iterable[int]] = None,
    ):
        self.transform = transform
        self.normalize = normalize
        self.load_graphs = load_graphs
        self.node_feature_dim = int(max(1, node_feature_dim))
        self.edge_feature_dim = int(max(1, edge_feature_dim))
        self.topology_supervision_mode = str(topology_supervision_mode).strip().lower()
        self.semantic_role_prior_strength = float(max(0.0, min(1.0, semantic_role_prior_strength)))
        self.semantic_puzzle_offset = int(max(0, semantic_puzzle_offset))
        self.puzzle_stage_topology_enabled = bool(puzzle_stage_topology_enabled)
        self.puzzle_stage_trace_decay = float(max(0.05, min(1.0, puzzle_stage_trace_decay)))
        self.dungeon_ids = normalize_dungeon_ids(dungeon_ids)
        self.variants = normalize_variants(variants)
        self.rooms = []
        self.graphs = [] if load_graphs else None
        self.sample_metadata: List[Dict[str, Any]] = []
        if self.topology_supervision_mode not in {"runtime_aligned", "oracle_room_grid"}:
            raise ValueError(
                "topology_supervision_mode must be 'runtime_aligned' or 'oracle_room_grid'."
            )
        
        if not VGLC_AVAILABLE:
            raise ImportError("VGLC adapter required for room dataset")
        
        adapter = ZeldaDungeonAdapter(str(data_dir))
        
        dungeon_iter = self.dungeon_ids if self.dungeon_ids is not None else range(1, 10)
        for dungeon_num in dungeon_iter:
            for variant in self.variants:
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant)
                    dungeon_graph = _extract_graph_from_dungeon(
                        dungeon,
                        node_feature_dim=self.node_feature_dim,
                        edge_feature_dim=self.edge_feature_dim,
                    ) if load_graphs else None
                    for coord, room in dungeon.rooms.items():
                        grid = getattr(room, 'semantic_grid', None)
                        if grid is None:
                            grid = getattr(room, 'grid', None)
                        if grid is not None:
                            self.rooms.append(grid.astype(np.float32))
                            self.sample_metadata.append(
                                {
                                    "dungeon_num": int(dungeon_num),
                                    "variant": int(variant),
                                    "dungeon_id": f"tloz{int(dungeon_num)}_{int(variant)}",
                                    "room_coord": tuple(coord) if isinstance(coord, tuple) else coord,
                                }
                            )
                            if self.graphs is not None and dungeon_graph is not None:
                                self.graphs.append(
                                    _build_room_graph_sample(
                                        dungeon,
                                        coord,
                                        room,
                                        dungeon_graph,
                                        topology_supervision_mode=self.topology_supervision_mode,
                                        semantic_role_prior_strength=self.semantic_role_prior_strength,
                                        semantic_puzzle_offset=self.semantic_puzzle_offset,
                                        puzzle_stage_topology_enabled=self.puzzle_stage_topology_enabled,
                                        puzzle_stage_trace_decay=self.puzzle_stage_trace_decay,
                                    )
                                )
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    logger.debug(f"Skipping dungeon {dungeon_num}v{variant}: {e}")
        
        logger.info(f"Loaded {len(self.rooms)} individual rooms")
    
    def __len__(self) -> int:
        return len(self.rooms)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        grid = self.rooms[idx]
        tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        
        # Use fixed constant (43 = max tile ID) so grids_to_onehot / encode_to_latent
        # can invert exactly with *43
        if self.normalize and tensor.max() > 1:
            NUM_TILE_IDS = 43  # TileID.PUZZLE = 43
            tensor = tensor / NUM_TILE_IDS
        
        if self.transform:
            tensor = self.transform(tensor)

        if self.load_graphs and self.graphs is not None:
            graph = self.graphs[idx]
            num_tile_ids = 43
            edge_feature_dim = int(getattr(self, "edge_feature_dim", GRAPH_EDGE_FEATURE_DIM))
            neighbor_maps = {}
            for direction, room_map in dict(graph.get('neighbor_maps', {})).items():
                if room_map is None:
                    neighbor_maps[direction] = None
                    continue
                room_tensor = torch.tensor(room_map, dtype=torch.float32).unsqueeze(0)
                if self.normalize and room_tensor.max() > 1:
                    room_tensor = room_tensor / num_tile_ids
                neighbor_maps[direction] = room_tensor
            return tensor, {
                'node_features': torch.tensor(graph['node_features'], dtype=torch.float32),
                'edge_index': torch.tensor(graph['edge_index'], dtype=torch.long),
                'edge_attr': torch.tensor(graph.get('edge_attr', np.zeros((0,), dtype=np.int64)), dtype=torch.long),
                'edge_features': torch.tensor(graph.get('edge_features', np.zeros((0, edge_feature_dim), dtype=np.float32)), dtype=torch.float32),
                'edge_rrwp': torch.tensor(graph.get('edge_rrwp', np.zeros((int(graph.get('num_edges', 0)), GRAPH_TPE_DIM), dtype=np.float32)), dtype=torch.float32),
                'tpe': torch.tensor(graph.get('tpe', np.zeros((0, GRAPH_TPE_DIM), dtype=np.float32)), dtype=torch.float32),
                'node_positions': torch.tensor(graph.get('node_positions', np.zeros((0, 2), dtype=np.float32)), dtype=torch.float32),
                'room_topology_map': torch.tensor(graph.get('room_topology_map', np.zeros((ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)), dtype=torch.float32),
                'boundary_constraints': torch.tensor(graph.get('boundary_constraints', np.zeros((8,), dtype=np.float32)), dtype=torch.float32),
                'room_position': torch.tensor(graph.get('room_position', np.zeros((2,), dtype=np.float32)), dtype=torch.float32),
                'neighbor_maps': neighbor_maps,
                'num_nodes': graph['num_nodes'],
                'num_edges': graph['num_edges'],
                'start_node_id': graph.get('start_node_id', -1),
                'current_node_idx': int(graph.get('current_node_idx', 0)),
                'node_to_idx': dict(graph.get('node_to_idx', {})),
                'has_puzzle': bool(graph.get('has_puzzle', False)),
                'puzzle_room_structure_enabled': bool(
                    graph.get('puzzle_room_structure_enabled', graph.get('has_puzzle', False))
                ),
                'puzzle_stage_condition': dict(graph.get('puzzle_stage_condition', {})),
                **({'style_id': int(graph.get('style_id'))} if graph.get('style_id', None) is not None else {}),
            }

        return tensor


# =============================================================================
# GRAPH COLLATION FOR VARIABLE-SIZE GRAPHS
# =============================================================================

def graph_collate_fn(batch):
    """
    Custom collation function for batches containing (image, graph_dict) pairs.
    
    Handles variable-size graphs by:
    1. Stacking image tensors normally (they're already padded to same size)
    2. Storing graph dicts as a list (since node counts differ per dungeon)
    
    For per-sample graph processing during training, we iterate over
    individual graphs rather than trying to batch them into a single tensor.
    
    Args:
        batch: List of (image_tensor, graph_dict) tuples from __getitem__
        
    Returns:
        (images_batch, graph_list) where:
            - images_batch: [B, 1, H, W] stacked image tensors
            - graph_list: List of B graph dicts, each containing:
                - node_features: [N_i, 6] (variable N_i per graph)
                - edge_index: [2, E_i] (variable E_i per graph)
                - edge_attr: [E_i] edge type labels
                - num_nodes: int
                - num_edges: int
                - start_node_id: int
    """
    if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
        images = torch.stack([item[0] for item in batch])
        graphs = [item[1] for item in batch]
        return images, graphs
    else:
        # No graph data -- plain image batch
        return torch.stack(batch)


class DungeonBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that yields all room samples from one dungeon variant.

    Room-level diffusion needs this when training dungeon-scope graph losses:
    the batch dimension must correspond to the graph's node set, not to random
    rooms sampled from unrelated dungeons.
    """

    def __init__(
        self,
        groups: List[List[int]],
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self.groups = [list(group) for group in groups if group]
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> "DungeonBatchSampler":
        base_dataset = getattr(dataset, "dataset", dataset)
        subset_indices = list(getattr(dataset, "indices", range(len(base_dataset))))
        metadata = getattr(base_dataset, "sample_metadata", None)
        if metadata is None:
            raise ValueError("DungeonBatchSampler requires dataset.sample_metadata.")

        grouped: Dict[Any, List[Tuple[int, int]]] = {}
        for local_idx, base_idx in enumerate(subset_indices):
            if int(base_idx) >= len(metadata):
                continue
            meta = dict(metadata[int(base_idx)])
            key = meta.get("dungeon_id")
            if key is None:
                key = (meta.get("dungeon_num"), meta.get("variant"))
            order = int(meta.get("current_node_idx", local_idx)) if str(meta.get("current_node_idx", "")).strip() else local_idx
            grouped.setdefault(key, []).append((order, local_idx))

        groups = [
            [local_idx for _order, local_idx in sorted(items, key=lambda item: item[0])]
            for _key, items in sorted(grouped.items(), key=lambda item: str(item[0]))
        ]
        return cls(groups, shuffle=shuffle, drop_last=drop_last, seed=seed)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(order), generator=generator).tolist()
            order = [order[i] for i in perm]
        for group_idx in order:
            group = self.groups[group_idx]
            if self.drop_last and not group:
                continue
            yield list(group)

    def __len__(self) -> int:
        if self.drop_last:
            return sum(1 for group in self.groups if group)
        return len(self.groups)


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_dataloader(
    data_dir: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    drop_last: bool = True,
    use_vglc: bool = False,
    normalize: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
    transform: Optional[Callable] = None,
    room_level: bool = False,
    load_graphs: bool = False,
    node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
    edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
    topology_supervision_mode: str = "runtime_aligned",
    semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    puzzle_stage_topology_enabled: bool = False,
    puzzle_stage_trace_decay: float = DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    sampler: Optional[Sampler] = None,
    dungeon_batch_mode: bool = False,
    return_sampler: bool = False,
    dungeon_ids: Optional[Iterable[int]] = None,
    variants: Optional[Iterable[int]] = None,
) -> DataLoader:
    """
    Create a DataLoader for Zelda dungeon training.
    
    Args:
        data_dir: Directory containing dungeon data
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (0 for Windows compatibility)
        use_vglc: Use VGLC format via ZeldaDungeonAdapter
        normalize: Normalize values to [0, 1]
        target_size: Optional (H, W) to resize all dungeons
        transform: Optional transform to apply
        room_level: If True, use ZeldaRoomDataset for individual rooms
        
    Returns:
        PyTorch DataLoader
        
    Example:
        >>> loader = create_dataloader(
        ...     'Data/The Legend of Zelda',
        ...     batch_size=4,
        ...     use_vglc=True
        ... )
        >>> for batch in loader:
        ...     print(batch.shape)  # (4, 1, H, W)
    """
    if room_level:
        dataset = ZeldaRoomDataset(
            data_dir=data_dir,
            transform=transform,
            normalize=normalize,
            load_graphs=load_graphs,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            topology_supervision_mode=topology_supervision_mode,
            semantic_role_prior_strength=semantic_role_prior_strength,
            semantic_puzzle_offset=semantic_puzzle_offset,
            puzzle_stage_topology_enabled=puzzle_stage_topology_enabled,
            puzzle_stage_trace_decay=puzzle_stage_trace_decay,
            dungeon_ids=dungeon_ids,
            variants=variants,
        )
    else:
        dataset = ZeldaDungeonDataset(
            data_dir=data_dir,
            transform=transform,
            use_vglc=use_vglc,
            normalize=normalize,
            target_size=target_size,
            load_graphs=load_graphs,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            dungeon_ids=dungeon_ids,
            variants=variants,
        )
    
    batch_sampler = None
    if bool(dungeon_batch_mode):
        if sampler is not None:
            raise ValueError("dungeon_batch_mode cannot be combined with sampler.")
        if not room_level:
            raise ValueError("dungeon_batch_mode is only valid for room_level datasets.")
        batch_sampler = DungeonBatchSampler.from_dataset(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        "collate_fn": graph_collate_fn if load_graphs else None,
    }
    if batch_sampler is not None:
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
        sampler = batch_sampler
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(bool(shuffle) if sampler is None else False),
            sampler=sampler,
            drop_last=bool(drop_last),
            **dataloader_kwargs,
        )
    if return_sampler:
        return dataloader, sampler
    return dataloader


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def tensor_to_ascii(tensor: torch.Tensor, threshold: float = 0.5) -> str:
    """
    Convert tensor back to ASCII representation for visualization.
    
    Args:
        tensor: (1, H, W) or (H, W) tensor
        threshold: Threshold for binarization
        
    Returns:
        ASCII string representation
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    arr = tensor.detach().cpu()
    if arr.dim() == 3:
        if arr.shape[0] > 1:
            arr = arr.argmax(dim=0)
        else:
            arr = arr.squeeze(0)
    if arr.dim() != 2:
        raise ValueError(f"Expected 2D tensor/grid for ASCII conversion, got shape={tuple(arr.shape)}")

    grid = arr.numpy()
    # Backward-compatible path for normalized tensors in [0,1].
    if np.issubdtype(grid.dtype, np.floating):
        max_val = float(np.max(grid)) if grid.size > 0 else 0.0
        min_val = float(np.min(grid)) if grid.size > 0 else 0.0
        if min_val >= 0.0 and max_val <= 1.0:
            grid = np.rint(grid * 43.0)
    grid_int = np.rint(grid).astype(np.int32, copy=False)
    return '\n'.join(semantic_grid_to_vglc_lines(grid_int))


def extract_start_goal(
    grid: Union[torch.Tensor, np.ndarray]
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Extract start and goal positions from a dungeon grid.
    
    Args:
        grid: Dungeon grid tensor or array
        
    Returns:
        (start_coords, goal_coords) as (row, col) tuples, or None if not found
    """
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    
    if grid.ndim == 3:
        grid = grid.squeeze(0)
    
    start_coords = None
    goal_coords = None
    
    if SEMANTIC_PALETTE is not None:
        start_id = SEMANTIC_PALETTE.get('START', 21)
        goal_id = SEMANTIC_PALETTE.get('TRIFORCE', 22)
    else:
        start_id = TILE_MAPPING.get('S', 6)
        goal_id = TILE_MAPPING.get('G', 7)
    
    # Find start
    start_pos = np.where(grid == start_id)
    if len(start_pos[0]) > 0:
        start_coords = (int(start_pos[0][0]), int(start_pos[1][0]))
    
    # Find goal
    goal_pos = np.where(grid == goal_id)
    if len(goal_pos[0]) > 0:
        goal_coords = (int(goal_pos[0][0]), int(goal_pos[1][0]))
    
    return start_coords, goal_coords


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Zelda Dataset Loader')
    parser.add_argument('--data-dir', type=str, default='Data/The Legend of Zelda',
                        help='Path to dungeon data')
    parser.add_argument('--use-vglc', action='store_true',
                        help='Use VGLC format')
    parser.add_argument('--batch-size', type=int, default=4)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        use_vglc=args.use_vglc,
    )
    
    print(f"Dataset size: {len(loader.dataset)}")
    print(f"Number of batches: {len(loader)}")
    
    for batch in loader:
        print(f"Batch shape: {batch.shape}")
        print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")
        break
