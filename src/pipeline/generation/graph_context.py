"""Graph-context helpers for room and dungeon generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import torch

from src.core import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.definitions import DOOR_POSITIONS, GRAPH_TPE_DIM, TileID, parse_edge_type_tokens
from src.core.condition_encoder import build_boundary_constraints
from src.pipeline.graph_features import (
    compute_current_node_distance_features,
    compute_rrwp_edge_features,
    compute_tpe_features,
    encode_edge_feature_vector,
    extract_node_feature_vector,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    DEFAULT_VALIDATOR_PLAN_MAX_STATES,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    TOPOLOGY_ANCHOR_POLICY_VERSION,
    apply_puzzle_structure_control_to_conditioning,
    build_puzzle_stage_condition_metadata,
    build_room_semantic_anchor_points,
    build_semantic_room_plan_trace,
    build_room_topology_condition_map,
)
from src.pipeline.spatial_utils import stable_node_sort_key
from src.pipeline.types import RoomGenerationResult
from src.utils.style_tokens import iter_style_metadata_candidates, resolve_style_token_id
from src.zelda_data.vglc_utils import get_physical_start_node

logger = logging.getLogger(__name__)
_stable_node_sort_key = stable_node_sort_key


def _prepare_graph_context(pipeline, graph: nx.Graph, use_tpe: bool = True) -> Dict[str, Any]:
    """
    Prepare graph tensors for GNN conditioning with stable node indexing.

    Returns:
        Dict containing node_features, edge_index, edge_features, tpe,
        node_order, and node_to_idx.
    """
    node_dim, edge_dim = pipeline._condition_feature_dims()

    if graph is None or len(graph.nodes) == 0:
        empty_nodes = torch.zeros(0, node_dim, device=pipeline.device, dtype=torch.float32)
        empty_edges = torch.zeros(2, 0, dtype=torch.long, device=pipeline.device)
        empty_edge_feats = torch.zeros(0, edge_dim, device=pipeline.device, dtype=torch.float32)
        empty_edge_rrwp = torch.zeros(0, int(GRAPH_TPE_DIM), device=pipeline.device, dtype=torch.float32)
        empty_tpe = torch.zeros(0, 8, device=pipeline.device, dtype=torch.float32)
        empty_positions = torch.zeros(0, 2, device=pipeline.device, dtype=torch.float32)
        empty_mask = torch.zeros(0, device=pipeline.device, dtype=torch.float32)
        return {
            'node_features': empty_nodes,
            'edge_index': empty_edges,
            'edge_features': empty_edge_feats,
            'edge_rrwp': empty_edge_rrwp,
            'tpe': empty_tpe,
            'node_positions': empty_positions,
            'node_mask': empty_mask,
            'node_order': [],
            'node_to_idx': {},
            'start_node_id': -1,
            'target_idx': -1,
        }

    # Deterministic order is required so room_id -> node_idx stays stable.
    if isinstance(graph, nx.DiGraph):
        try:
            node_order = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            node_order = sorted(graph.nodes(), key=_stable_node_sort_key)
    else:
        node_order = sorted(graph.nodes(), key=_stable_node_sort_key)

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_order)}
    num_nodes = len(node_order)

    node_features = torch.zeros(num_nodes, node_dim, device=pipeline.device, dtype=torch.float32)
    node_positions = torch.zeros(num_nodes, 2, device=pipeline.device, dtype=torch.float32)
    for node_id, idx in node_to_idx.items():
        node_features[idx] = pipeline._extract_node_feature_vector(graph.nodes[node_id])
        pos = pipeline._get_node_grid_position(graph, node_id)
        if pos is None:
            node_positions[idx] = torch.tensor((float(idx), 0.0), device=pipeline.device, dtype=torch.float32)
        else:
            node_positions[idx] = torch.tensor(
                (float(pos[0]), float(pos[1])),
                device=pipeline.device,
                dtype=torch.float32,
            )

    edge_pairs: List[Tuple[int, int]] = []
    edge_features_list: List[List[float]] = []
    for u, v, edge_data in graph.edges(data=True):
        if u not in node_to_idx or v not in node_to_idx:
            continue

        edge_pairs.append((node_to_idx[u], node_to_idx[v]))
        edge_features_list.append(pipeline._encode_edge_feature_vector(edge_data))

        # For undirected graphs we add reverse edges explicitly for message passing.
        if not graph.is_directed() and u != v:
            edge_pairs.append((node_to_idx[v], node_to_idx[u]))
            edge_features_list.append(pipeline._encode_edge_feature_vector(edge_data))

    if edge_pairs:
        edge_index = (
            torch.tensor(edge_pairs, dtype=torch.long, device=pipeline.device)
            .t()
            .contiguous()
        )
        edge_features = torch.tensor(
            edge_features_list, dtype=torch.float32, device=pipeline.device
        )
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=pipeline.device)
        edge_features = torch.zeros(0, edge_dim, dtype=torch.float32, device=pipeline.device)

    edge_rrwp = compute_rrwp_edge_features(
        edge_index,
        num_nodes=num_nodes,
        steps=int(GRAPH_TPE_DIM),
        device=pipeline.device,
        dtype=torch.float32,
    )

    if use_tpe:
        tpe = pipeline._compute_tpe_features(graph, node_order, node_to_idx, node_features)
    else:
        tpe = torch.zeros(num_nodes, 8, device=pipeline.device, dtype=torch.float32)

    start_node = get_physical_start_node(graph)
    if start_node is None or start_node not in node_to_idx:
        start_node = next(
            (
                node_id
                for node_id in node_order
                if pipeline._room_role_flags(dict(graph.nodes[node_id])).get("is_start", False)
            ),
            None,
        )
    target_node = next(
        (
            node_id
            for node_id in node_order
            if pipeline._room_role_flags(dict(graph.nodes[node_id])).get("has_goal", False)
        ),
        None,
    )

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'edge_rrwp': edge_rrwp,
        'tpe': tpe,
        'node_positions': node_positions,
        'node_mask': torch.ones(num_nodes, device=pipeline.device, dtype=torch.float32),
        'node_order': node_order,
        'node_to_idx': node_to_idx,
        'start_node_id': int(node_to_idx.get(start_node, 0)) if start_node is not None else 0,
        'target_idx': int(node_to_idx.get(target_node, -1)) if target_node is not None else -1,
    }


def _get_neighbor_latents(
    pipeline,
    room_id: int,
    graph: nx.Graph,
    generated_latents: Dict[int, torch.Tensor]
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Get directional neighbor latents for already-generated rooms.

    Best-effort direction inference uses node layout coordinates when present.
    """
    neighbor_dict: Dict[str, Optional[torch.Tensor]] = {
        'N': None, 'S': None, 'E': None, 'W': None
    }

    if room_id not in graph:
        return neighbor_dict

    if graph.is_directed():
        neighbor_ids = [n for n in graph.predecessors(room_id) if n in generated_latents]
    else:
        neighbor_ids = [n for n in graph.neighbors(room_id) if n in generated_latents]

    if not neighbor_ids:
        return neighbor_dict

    unresolved: List[int] = []
    for nid in sorted(neighbor_ids, key=_stable_node_sort_key):
        direction = pipeline._infer_direction(graph, source_node=nid, target_node=room_id)
        if direction is not None and neighbor_dict[direction] is None:
            neighbor_dict[direction] = generated_latents[nid].to(pipeline.device)
        else:
            unresolved.append(nid)

    # Stable fallback assignment when spatial metadata is missing or ambiguous.
    for direction, nid in zip(['N', 'W', 'E', 'S'], unresolved):
        if neighbor_dict[direction] is None:
            neighbor_dict[direction] = generated_latents[nid].to(pipeline.device)

    return neighbor_dict


def _get_neighbor_reference_room_maps(
    pipeline,
    room_id: int,
    graph: nx.Graph,
    generated_rooms: Dict[Any, Any],
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Get directional neighboring room maps for already-generated rooms.

    This mirrors the teacher-forced room-map signal used during training
    and closes the exemplar-conditioning gap at inference time.
    """
    neighbor_dict: Dict[str, Optional[torch.Tensor]] = {
        'N': None, 'S': None, 'E': None, 'W': None
    }

    if room_id not in graph:
        return neighbor_dict

    if graph.is_directed():
        neighbor_ids = [n for n in graph.predecessors(room_id) if n in generated_rooms]
    else:
        neighbor_ids = [n for n in graph.neighbors(room_id) if n in generated_rooms]

    if not neighbor_ids:
        return neighbor_dict

    def _coerce_room_map(room_value: Any) -> Optional[torch.Tensor]:
        if isinstance(room_value, RoomGenerationResult):
            room_map = room_value.room_grid
        elif hasattr(room_value, "room_grid"):
            room_map = getattr(room_value, "room_grid")
        else:
            room_map = room_value
        if room_map is None:
            return None
        if isinstance(room_map, torch.Tensor):
            tensor = room_map.detach().to(pipeline.device)
        else:
            tensor = torch.as_tensor(room_map, device=pipeline.device)
        if tensor.dim() == 4 and int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == 1:
            tensor = tensor.squeeze(0).squeeze(0)
        elif tensor.dim() == 3 and int(tensor.shape[0]) == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 2:
            raise ValueError(
                f"Reference room map must resolve to [H,W], got shape={tuple(tensor.shape)}."
            )
        return tensor.contiguous()

    unresolved: List[int] = []
    for nid in sorted(neighbor_ids, key=_stable_node_sort_key):
        room_map = _coerce_room_map(generated_rooms[nid])
        if room_map is None:
            continue
        direction = pipeline._infer_direction(graph, source_node=nid, target_node=room_id)
        if direction is not None and neighbor_dict[direction] is None:
            neighbor_dict[direction] = room_map
        else:
            unresolved.append(nid)

    for direction, nid in zip(['N', 'W', 'E', 'S'], unresolved):
        if neighbor_dict[direction] is None:
            room_map = _coerce_room_map(generated_rooms[nid])
            if room_map is not None:
                neighbor_dict[direction] = room_map

    return neighbor_dict


def _extract_room_start_goal(
    pipeline,
    graph: nx.Graph,
    room_id: int
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Extract room-local start/goal hints for symbolic repair.

    Uses node metadata when available, otherwise infers a sensible
    left-to-right flow based on graph predecessors/successors.
    """
    if room_id not in graph:
        return ((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1))

    attrs = graph.nodes[room_id]

    start = (
        pipeline._parse_room_coord(attrs.get('start_pos'))
        or pipeline._parse_room_coord(attrs.get('entry_pos'))
        or pipeline._parse_room_coord(attrs.get('entrance'))
    )
    goal = (
        pipeline._parse_room_coord(attrs.get('goal_pos'))
        or pipeline._parse_room_coord(attrs.get('exit_pos'))
        or pipeline._parse_room_coord(attrs.get('exit'))
    )

    if start is None:
        has_pred = graph.in_degree(room_id) > 0 if graph.is_directed() else graph.degree(room_id) > 0
        start = (ROOM_HEIGHT // 2, 0) if has_pred else (ROOM_HEIGHT // 2, ROOM_WIDTH // 4)

    if goal is None:
        has_succ = graph.out_degree(room_id) > 0 if graph.is_directed() else graph.degree(room_id) > 0
        goal = (ROOM_HEIGHT // 2, ROOM_WIDTH - 1) if has_succ else (ROOM_HEIGHT // 2, (3 * ROOM_WIDTH) // 4)

    start = pipeline._clamp_room_coord(start)
    goal = pipeline._clamp_room_coord(goal)

    if start == goal:
        goal = pipeline._clamp_room_coord((goal[0], goal[1] + 1))

    return (start, goal)


def _build_room_boundary_constraints(
    pipeline,
    graph: nx.Graph,
    room_id: Any,
) -> torch.Tensor:
    """Build [1, 8] boundary constraints from incident topology edges."""
    has_neighbor: Dict[str, bool] = {"N": False, "S": False, "E": False, "W": False}
    required_door: Dict[str, bool] = {"N": False, "S": False, "E": False, "W": False}

    if room_id not in graph:
        return torch.zeros(1, 8, device=pipeline.device, dtype=torch.float32)

    incident: List[Any] = []
    if graph.is_directed():
        incident.extend(list(graph.predecessors(room_id)))
        incident.extend(list(graph.successors(room_id)))
    else:
        incident.extend(list(graph.neighbors(room_id)))

    incident_unique = sorted(set(incident), key=_stable_node_sort_key)
    unresolved: List[Any] = []
    for nid in incident_unique:
        direction = pipeline._infer_direction(graph, source_node=nid, target_node=room_id)
        if direction is None:
            unresolved.append(nid)
            continue
        has_neighbor[direction] = True
        required_door[direction] = True

    for direction, _nid in zip(["N", "E", "S", "W"], unresolved):
        if not has_neighbor[direction]:
            has_neighbor[direction] = True
            required_door[direction] = True

    boundary = build_boundary_constraints(has_neighbor=has_neighbor, required_door=required_door)
    return boundary.to(device=pipeline.device, dtype=torch.float32).unsqueeze(0)


def _room_role_flags(pipeline, attrs: Dict[str, Any]) -> Dict[str, bool]:
    """Extract high-level room-role booleans from graph node metadata."""
    tokens = pipeline._parse_label_tokens(attrs.get("label"))
    raw_type = str(attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or "").strip().lower()
    role_tokens = set(tokens) | set(pipeline._parse_label_tokens(raw_type))
    difficulty_rating = str(attrs.get("difficulty_rating", "") or "").strip().upper()

    def _hint(name: str, *aliases: str) -> bool:
        return pipeline._coerce_bool(attrs.get(name)) or any(pipeline._coerce_bool(attrs.get(alias)) for alias in aliases)

    return {
        "is_start": _hint("is_start", "is_entry") or raw_type in {"start", "entry"} or "start" in role_tokens or "entry" in role_tokens,
        "has_enemy": _hint("has_enemy") or "e" in role_tokens or "enemy" in role_tokens,
        "has_key": _hint("has_key") or "k" in role_tokens or "key" in role_tokens,
        "has_item": _hint("has_item", "has_macro_item", "has_minor_item") or "i" in role_tokens or "item" in role_tokens or "treasure" in role_tokens,
        "has_goal": _hint("has_triforce", "is_triforce", "is_goal") or raw_type in {"goal", "triforce"} or "t" in role_tokens or "goal" in role_tokens or "triforce" in role_tokens,
        "has_boss": _hint("has_boss", "is_boss") or "b" in role_tokens or "boss" in role_tokens,
        "has_puzzle": _hint("has_puzzle") or "p" in role_tokens or "puzzle" in role_tokens or raw_type in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"} or "puzzle" in raw_type,
        "is_tutorial_puzzle": bool(_hint("is_tutorial") or raw_type == "tutorial_puzzle" or difficulty_rating == "SAFE"),
        "is_combat_puzzle": bool(raw_type == "combat_puzzle"),
        "is_complex_puzzle": bool(raw_type == "complex_puzzle" or difficulty_rating in {"HARD", "EXTREME"}),
        "is_switch_puzzle": bool(raw_type == "switch"),
    }


def _resolve_puzzle_room_scaffold_profile(
    pipeline,
    *,
    attrs: Dict[str, Any],
    role_flags: Dict[str, bool],
    semantics: Dict[str, Any],
    node_type: str,
) -> Dict[str, Any]:
    """
    Derive a per-room constructive puzzle profile from topology semantics.

    Hybrid PCG systems work best when local room structure reflects explicit
    mission intent. This profile keeps the existing scaffold mechanism but
    adapts its density to pedagogical puzzle subtype information.
    """
    archetype = pipeline._select_puzzle_room_scaffold_archetype(
        role_flags=role_flags,
        semantics=semantics,
        node_type=node_type,
    )
    gate_family = pipeline._classify_puzzle_gate_family(
        role_flags=role_flags,
        semantics=semantics,
        node_type=node_type,
    )
    branch_density = float(max(0.0, min(1.0, getattr(pipeline, "default_puzzle_room_branch_density", 0.75))))
    block_budget = int(max(0, getattr(pipeline, "default_puzzle_room_block_budget", 28)))
    preserve_margin = int(max(0, getattr(pipeline, "default_puzzle_room_preserve_route_margin", 0)))
    difficulty_rating = str(attrs.get("difficulty_rating", "") or "").strip().upper()

    edge_constraints = semantics.get("edge_constraints", {})
    flat_edge_tokens: Set[str] = set()
    for tokens in edge_constraints.values():
        flat_edge_tokens.update(str(token) for token in tokens)

    required_doors = semantics.get("required_doors", {})
    required_door_count = int(sum(1 for enabled in required_doors.values() if enabled))

    if role_flags.get("is_tutorial_puzzle", False):
        archetype = "gate"
        branch_density = min(branch_density, 0.25)
        block_budget = min(block_budget, 10)
    elif role_flags.get("is_combat_puzzle", False):
        archetype = "combat"
        branch_density = min(branch_density, 0.30)
        block_budget = min(block_budget, 12)
    elif role_flags.get("is_complex_puzzle", False):
        if archetype not in {"hub", "serpentine"}:
            archetype = "hub" if required_door_count >= 3 else "serpentine"
        branch_density = max(branch_density, 0.70)
        block_budget = max(block_budget, 26)
    elif gate_family in {"switch", "toggle"}:
        archetype = "gate"
        branch_density = max(branch_density, 0.50)
        block_budget = max(block_budget, 18)
    elif gate_family in {"bombable", "item_unlock", "key"} and archetype not in {"hub", "combat"}:
        archetype = "gate"
        if gate_family == "key":
            branch_density = max(branch_density, 0.40)
            block_budget = max(block_budget, 15)
        elif gate_family == "item_unlock":
            branch_density = max(branch_density, 0.42)
            block_budget = max(block_budget, 16)
        else:
            branch_density = max(branch_density, 0.45)
            block_budget = max(block_budget, 17)
    elif node_type in {"item", "protection_item", "minor_item", "treasure", "stair", "stairs_up", "stairs_down", "warp"}:
        archetype = "island"
        branch_density = min(max(branch_density, 0.40), 0.65)
        block_budget = max(block_budget, 16)
    elif difficulty_rating == "MODERATE" and archetype == "serpentine":
        branch_density = min(max(branch_density, 0.45), 0.65)

    return {
        "archetype": str(archetype),
        "gate_family": str(gate_family),
        "branch_density": float(max(0.0, min(1.0, branch_density))),
        "block_budget": int(max(0, block_budget)),
        "preserve_route_margin": int(max(0, preserve_margin)),
    }


def _extract_room_topology_semantics(
    pipeline,
    graph: nx.Graph,
    room_id: Any,
) -> Dict[str, Any]:
    """Extract doorway and edge semantics for one room from the mission graph."""
    required_doors: Dict[str, bool] = {"N": False, "S": False, "E": False, "W": False}
    edge_constraints: Dict[str, Set[str]] = {"N": set(), "S": set(), "E": set(), "W": set()}
    incoming_dirs: Set[str] = set()
    outgoing_dirs: Set[str] = set()

    if room_id not in graph:
        return {
            "required_doors": required_doors,
            "edge_constraints": edge_constraints,
            "incoming_dirs": incoming_dirs,
            "outgoing_dirs": outgoing_dirs,
        }

    incident: List[Tuple[Any, Dict[str, Any], str]] = []
    if graph.is_directed():
        incident.extend((nid, dict(graph.get_edge_data(nid, room_id, default={}) or {}), "incoming") for nid in graph.predecessors(room_id))
        incident.extend((nid, dict(graph.get_edge_data(room_id, nid, default={}) or {}), "outgoing") for nid in graph.successors(room_id))
    else:
        incident.extend((nid, dict(graph.get_edge_data(room_id, nid, default={}) or {}), "bidirectional") for nid in graph.neighbors(room_id))

    unresolved: List[Tuple[Any, Dict[str, Any], str]] = []
    for nid, edge_data, flow in sorted(incident, key=lambda item: _stable_node_sort_key(item[0])):
        direction = pipeline._infer_direction(graph, source_node=nid, target_node=room_id)
        if direction is None:
            unresolved.append((nid, edge_data, flow))
            continue
        required_doors[direction] = True
        if flow in {"incoming", "bidirectional"}:
            incoming_dirs.add(direction)
        if flow in {"outgoing", "bidirectional"}:
            outgoing_dirs.add(direction)
        edge_constraints[direction].update(
            parse_edge_type_tokens(
                label=str(edge_data.get("label", "") or ""),
                edge_type=str(edge_data.get("edge_type", edge_data.get("type", "")) or ""),
            )
        )

    for direction, (_nid, edge_data, _flow) in zip(["N", "E", "S", "W"], unresolved):
        if required_doors[direction]:
            continue
        required_doors[direction] = True
        incoming_dirs.add(direction)
        outgoing_dirs.add(direction)
        edge_constraints[direction].update(
            parse_edge_type_tokens(
                label=str(edge_data.get("label", "") or ""),
                edge_type=str(edge_data.get("edge_type", edge_data.get("type", "")) or ""),
            )
        )

    return {
        "required_doors": required_doors,
        "edge_constraints": edge_constraints,
        "incoming_dirs": incoming_dirs,
        "outgoing_dirs": outgoing_dirs,
    }


def _build_room_plan_trace(
    pipeline,
    graph: nx.Graph,
    room_id: Any,
    room_grid: np.ndarray,
    *,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> np.ndarray:
    """Build a concrete traversability plan mask from the current room grid."""
    if room_id not in graph:
        return np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    attrs = graph.nodes[room_id]
    if start_goal is None:
        start_goal = pipeline._extract_room_start_goal(graph, room_id)
    start, goal = start_goal if start_goal is not None else (None, None)
    semantics = pipeline._extract_room_topology_semantics(graph, room_id)
    budget = pipeline._resolve_validator_plan_state_budget(
        attrs=attrs,
        semantics=semantics,
    )
    return build_semantic_room_plan_trace(
        np.asarray(room_grid, dtype=np.int32),
        start=start,
        goal=goal,
        required_doors=semantics["required_doors"],
        incoming_dirs=semantics["incoming_dirs"],
        outgoing_dirs=semantics["outgoing_dirs"],
        edge_constraint_tokens=semantics["edge_constraints"],
        room_role_flags=pipeline._room_role_flags(attrs),
        validator_plan_max_states=budget,
    ).astype(np.float32, copy=False)


def _resolve_validator_plan_state_budget(
    pipeline,
    *,
    attrs: Mapping[str, Any],
    semantics: Mapping[str, Any],
) -> int:
    """
    Adapt the room-local validator plan budget to semantic complexity.

    A single tiny cap for every room is too brittle for richer puzzle rooms,
    but a truly unbounded local planner is not acceptable in this pipeline
    because it can explode memory/runtime during export. We therefore scale
    the budget with room complexity while keeping a hard cap.
    """
    base_budget = int(max(32, int(getattr(pipeline, "default_validator_plan_max_states", DEFAULT_VALIDATOR_PLAN_MAX_STATES))))
    role_flags = pipeline._room_role_flags(dict(attrs))
    active_roles = sum(1 for enabled in role_flags.values() if bool(enabled))
    required_doors = sum(
        1 for enabled in dict(semantics.get("required_doors", {})).values()
        if bool(enabled)
    )
    distinct_tokens: Set[str] = set()
    for token_set in dict(semantics.get("edge_constraints", {})).values():
        distinct_tokens.update(str(token).strip().lower() for token in set(token_set or set()))
    complex_gate_tokens = {
        "switch",
        "switch_locked",
        "state_block",
        "on_off_gate",
        "toggle",
        "bombable",
        "item_gate",
        "item_locked",
        "key_locked",
        "boss_locked",
        "combat",
    }
    complexity_score = int(active_roles)
    complexity_score += max(0, int(required_doors) - 2)
    complexity_score += int(len(distinct_tokens))
    if bool(distinct_tokens & complex_gate_tokens):
        complexity_score += 2
    if bool(role_flags.get("has_puzzle", False)):
        complexity_score += 1
    if bool(role_flags.get("has_enemy", False)):
        complexity_score += 1

    if complexity_score <= 0:
        return base_budget

    complexity_multiplier = 1.0 + min(3.0, 0.2 * float(complexity_score))
    hard_cap = max(base_budget, min(4096, int(base_budget * 4)))
    return int(max(base_budget, min(hard_cap, round(base_budget * complexity_multiplier))))


def _build_room_topology_condition_tensor(
    pipeline,
    graph: nx.Graph,
    room_id: Any,
    *,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Build explicit per-room topology prior [1, C, H, W] for the diffusion U-Net."""
    if room_id not in graph:
        return torch.zeros(
            1,
            ROOM_TOPOLOGY_CHANNEL_COUNT,
            ROOM_HEIGHT,
            ROOM_WIDTH,
            device=pipeline.device,
            dtype=torch.float32,
        )

    attrs = graph.nodes[room_id]
    if start_goal is None:
        start_goal = pipeline._extract_room_start_goal(graph, room_id)
    start, goal = start_goal if start_goal is not None else (None, None)
    semantics = pipeline._extract_room_topology_semantics(graph, room_id)
    budget = pipeline._resolve_validator_plan_state_budget(
        attrs=attrs,
        semantics=semantics,
    )

    topo_np = build_room_topology_condition_map(
        start=start,
        goal=goal,
        required_doors=semantics["required_doors"],
        incoming_dirs=semantics["incoming_dirs"],
        outgoing_dirs=semantics["outgoing_dirs"],
        edge_constraint_tokens=semantics["edge_constraints"],
        room_role_flags=pipeline._room_role_flags(attrs),
        semantic_role_prior_strength=pipeline.default_semantic_role_prior_strength,
        semantic_puzzle_offset=pipeline.default_semantic_puzzle_offset,
        validator_plan_max_states=budget,
        puzzle_stage_topology_enabled=pipeline.default_puzzle_stage_topology_enabled,
        puzzle_stage_trace_decay=pipeline.default_puzzle_stage_trace_decay,
    )
    return torch.from_numpy(topo_np).unsqueeze(0).to(device=pipeline.device, dtype=torch.float32)


def _extract_explicit_style_id(pipeline, graph: nx.Graph, *, room_id: Any) -> Optional[int]:
    """
    Extract an explicit style/theme token for one room.

    This prefers explicit numeric IDs, but it also accepts the repo's
    canonical symbolic sector-theme labels so generated mission graphs can
    drive the style path without inventing a broader visual taxonomy.
    """
    candidate_values: List[Any] = []
    if room_id in graph.nodes:
        node_attrs = dict(graph.nodes[room_id])
        candidate_values.extend(
            iter_style_metadata_candidates(
                node_attrs,
                keys=("style_id", "theme_id", "sector_theme_id", "sector_theme", "theme", "theme_name"),
            )
        )
    graph_attrs = getattr(graph, "graph", None)
    if isinstance(graph_attrs, dict):
        candidate_values.extend(
            iter_style_metadata_candidates(
                graph_attrs,
                keys=("style_id", "theme_id", "sector_theme_id", "sector_theme", "theme", "theme_name"),
            )
        )
    return resolve_style_token_id(*candidate_values)


def _build_room_graph_context(
    pipeline,
    *,
    graph_data: Dict[str, Any],
    mission_graph: nx.Graph,
    room_id: Any,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Build per-room graph context shared by condition encoding and diffusion."""
    node_to_idx = dict(graph_data.get('node_to_idx', {}) or {})
    current_node_idx = node_to_idx.get(room_id, 0)
    start_node = get_physical_start_node(mission_graph)
    if start_node is None or start_node not in node_to_idx:
        start_node = next(
            (
                node_id
                for node_id, attrs in mission_graph.nodes(data=True)
                if pipeline._room_role_flags(dict(attrs)).get("is_start", False)
            ),
            None,
        )
    target_node = next(
        (
            node_id
            for node_id, attrs in mission_graph.nodes(data=True)
            if pipeline._room_role_flags(dict(attrs)).get("has_goal", False)
        ),
        None,
    )
    style_id = pipeline._extract_explicit_style_id(mission_graph, room_id=room_id)
    current_node_distance = compute_current_node_distance_features(
        graph_data.get('edge_index'),
        int(graph_data.get('node_features').shape[0]) if isinstance(graph_data.get('node_features'), torch.Tensor) else 0,
        current_node_idx=current_node_idx,
        device=pipeline.device,
        dtype=torch.float32,
        max_distance=pipeline.current_node_distance_max,
    )
    attrs = dict(mission_graph.nodes[room_id]) if room_id in mission_graph else {}
    if start_goal is None:
        start_goal = pipeline._extract_room_start_goal(mission_graph, room_id)
    start, goal = start_goal if start_goal is not None else (None, None)
    semantics = (
        pipeline._extract_room_topology_semantics(mission_graph, room_id)
        if room_id in mission_graph
        else {
            "required_doors": {},
            "incoming_dirs": set(),
            "outgoing_dirs": set(),
            "edge_constraints": {},
        }
    )
    budget = pipeline._resolve_validator_plan_state_budget(
        attrs=attrs,
        semantics=semantics,
    )
    puzzle_stage_condition = build_puzzle_stage_condition_metadata(
        room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
        start=start,
        goal=goal,
        required_doors=semantics["required_doors"],
        incoming_dirs=semantics["incoming_dirs"],
        outgoing_dirs=semantics["outgoing_dirs"],
        edge_constraint_tokens=semantics["edge_constraints"],
        room_role_flags=pipeline._room_role_flags(attrs),
        validator_plan_max_states=budget,
        semantic_puzzle_offset=pipeline.default_semantic_puzzle_offset,
        stage_trace_decay=pipeline.default_puzzle_stage_trace_decay,
    )
    return {
        'graph_scope': 'room',
        'node_features': graph_data.get('node_features'),
        'edge_index': graph_data.get('edge_index'),
        'edge_features': graph_data.get('edge_features'),
        'tpe': graph_data.get('tpe'),
        'node_positions': graph_data.get('node_positions'),
        'node_mask': graph_data.get('node_mask'),
        'has_room_anchor': True,
        'mission_graph': mission_graph,
        'current_node_idx': current_node_idx,
        'start_node_id': int(node_to_idx.get(start_node, 0)) if start_node is not None else 0,
        'target_idx': int(node_to_idx.get(target_node, -1)) if target_node is not None else -1,
        'puzzle_room_structure_enabled': bool(pipeline.default_puzzle_room_structure_enabled),
        'puzzle_stage_condition': puzzle_stage_condition,
        **({'current_node_distance': current_node_distance} if pipeline.use_current_node_distance_features else {}),
        **({'style_id': int(style_id)} if style_id is not None else {}),
        'room_topology_map': pipeline._build_room_topology_condition_tensor(
            mission_graph,
            room_id,
            start_goal=start_goal,
        ),
    }


def _edge_tokens_to_door_tile(pipeline, tokens: Set[str]) -> int:
    """Map edge-semantic tokens to a concrete door tile ID."""
    normalized = {str(t).strip().lower() for t in set(tokens)}
    if {"boss_locked"} & normalized:
        return int(SEMANTIC_PALETTE["DOOR_BOSS"])
    if {"key_locked", "locked"} & normalized:
        return int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    if {"bombable"} & normalized:
        return int(SEMANTIC_PALETTE["DOOR_BOMB"])
    if {"switch", "switch_locked", "puzzle"} & normalized:
        return int(SEMANTIC_PALETTE["DOOR_PUZZLE"])
    if {"soft_locked", "one_way", "shutter"} & normalized:
        return int(SEMANTIC_PALETTE["DOOR_SOFT"])
    return int(SEMANTIC_PALETTE["DOOR_OPEN"])


def _build_masked_room_fixed_tokens(
    pipeline,
    graph: nx.Graph,
    room_id: Any,
    *,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build hard-known token layout for the discrete masked room generator.

    This encodes exact doors and explicit start/goal hints as fixed tokens
    that are never re-masked during iterative sampling.
    """
    fixed_tokens = torch.zeros((1, ROOM_HEIGHT, ROOM_WIDTH), device=pipeline.device, dtype=torch.long)
    fixed_mask = torch.zeros((1, ROOM_HEIGHT, ROOM_WIDTH), device=pipeline.device, dtype=torch.bool)

    if room_id not in graph:
        return fixed_tokens, fixed_mask

    semantics = pipeline._extract_room_topology_semantics(graph, room_id)
    for direction, enabled in semantics["required_doors"].items():
        if not bool(enabled):
            continue
        tile_id = pipeline._edge_tokens_to_door_tile(semantics["edge_constraints"].get(direction, set()))
        spec = DOOR_POSITIONS[str(direction)]
        if direction in {"N", "S"}:
            row = int(spec["row"])
            c0 = int(spec["col_start"])
            c1 = int(spec["col_end"]) + 1
            fixed_tokens[0, row, c0:c1] = tile_id
            fixed_mask[0, row, c0:c1] = True
        else:
            col = int(spec["col"])
            r0 = int(spec["row_start"])
            r1 = int(spec["row_end"]) + 1
            fixed_tokens[0, r0:r1, col] = tile_id
            fixed_mask[0, r0:r1, col] = True

    if start_goal is None:
        start_goal = pipeline._extract_room_start_goal(graph, room_id)
    if start_goal is not None:
        start, goal = start_goal
        role_flags = pipeline._room_role_flags(dict(graph.nodes[room_id]))
        semantic_anchors = build_room_semantic_anchor_points(
            room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
            start=start,
            goal=goal,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            room_role_flags=role_flags,
            semantic_puzzle_offset=pipeline.default_semantic_puzzle_offset,
        )
        sr, sc = pipeline._clamp_room_coord(semantic_anchors.get("start", start))
        gr, gc = pipeline._clamp_room_coord(semantic_anchors.get("goal", goal))

        # These anchors primarily encode local traversability hints. Only the
        # actual start / goal rooms should receive semantic START / TRIFORCE
        # tiles; all other rooms keep these anchors walkable as plain floor.
        start_tile = (
            int(SEMANTIC_PALETTE["START"])
            if role_flags.get("is_start", False)
            else int(SEMANTIC_PALETTE["FLOOR"])
        )
        goal_tile = (
            int(SEMANTIC_PALETTE["TRIFORCE"])
            if role_flags.get("has_goal", False)
            else int(SEMANTIC_PALETTE["FLOOR"])
        )
        fixed_tokens[0, sr, sc] = start_tile
        fixed_mask[0, sr, sc] = True
        fixed_tokens[0, gr, gc] = goal_tile
        fixed_mask[0, gr, gc] = True

        marker_to_anchor = {
            int(TileID.KEY_SMALL): "key",
            int(TileID.KEY_BOSS): "key",
            int(TileID.KEY_ITEM): "item",
            int(TileID.ITEM_MINOR): "item",
            int(TileID.BOSS): "boss",
            int(TileID.PUZZLE): "puzzle",
            int(TileID.STAIR): "item",
        }
        for tile_id in pipeline._resolve_room_graph_markers(graph, room_id):
            if int(tile_id) in {int(TileID.START), int(TileID.TRIFORCE), int(TileID.ENEMY)}:
                continue
            anchor_name = marker_to_anchor.get(int(tile_id))
            if anchor_name is None:
                continue
            point = semantic_anchors.get(anchor_name)
            if point is None:
                continue
            rr, cc = pipeline._clamp_room_coord(point)
            fixed_tokens[0, rr, cc] = int(tile_id)
            fixed_mask[0, rr, cc] = True

    return fixed_tokens, fixed_mask


def _build_room_position_tensor(
    pipeline,
    graph: nx.Graph,
    room_id: Any,
    fallback_order_index: int,
) -> torch.Tensor:
    """Build [1, 2] room position tensor from graph metadata."""
    pos = pipeline._get_node_grid_position(graph, room_id)
    if pos is None:
        pos = (int(fallback_order_index), 0)
    return torch.tensor([[float(pos[0]), float(pos[1])]], device=pipeline.device, dtype=torch.float32)

