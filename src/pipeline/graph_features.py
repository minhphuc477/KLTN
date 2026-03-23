"""Graph feature encoding helpers for condition encoder inputs."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from src.core.definitions import parse_edge_type_tokens

logger = logging.getLogger(__name__)


def condition_feature_dims(condition_encoder: Any) -> Tuple[int, int]:
    """Get active (node_dim, edge_dim) expected by the condition encoder."""
    node_dim = 6
    edge_dim = 8
    global_encoder = getattr(condition_encoder, "global_encoder", None)
    if global_encoder is not None:
        node_dim = int(getattr(global_encoder, "node_feature_dim", node_dim))
        edge_dim = int(getattr(global_encoder, "edge_feature_dim", edge_dim))
    return max(1, node_dim), max(1, edge_dim)


def fit_feature_vector(values: List[float], target_dim: int) -> List[float]:
    """Pad/truncate feature list to target dimension."""
    dim = max(1, int(target_dim))
    if len(values) >= dim:
        return [float(v) for v in values[:dim]]
    return [float(v) for v in values] + ([0.0] * (dim - len(values)))


def extract_node_feature_vector(
    attrs: Dict[str, Any],
    *,
    node_dim: int,
    device: torch.device,
    parse_label_tokens: Callable[[Any], set],
    coerce_bool: Callable[[Any], bool],
    coerce_difficulty: Callable[[Any], float],
) -> torch.Tensor:
    """Extract node feature vector for the active conditioning schema."""
    tokens = parse_label_tokens(attrs.get("label"))

    def _as_nonneg_int(value: Any) -> int:
        try:
            return int(max(0, int(value)))
        except Exception:
            return 0

    enemy_hint = _as_nonneg_int(attrs.get("enemy_count_hint", attrs.get("enemy_count", 0)))
    key_hint = _as_nonneg_int(attrs.get("key_count_hint", attrs.get("key_count", 0)))
    puzzle_hint = _as_nonneg_int(attrs.get("puzzle_count_hint", attrs.get("puzzle_count", 0)))
    item_hint = _as_nonneg_int(attrs.get("item_count_hint", attrs.get("item_count", 0)))

    has_enemy = (
        coerce_bool(attrs.get("has_enemy"))
        or (enemy_hint > 0)
        or "e" in tokens
        or "enemy" in tokens
        or "b" in tokens
        or "boss" in tokens
    )
    has_key = (
        coerce_bool(attrs.get("has_key"))
        or (key_hint > 0)
        or "k" in tokens
        or "key" in tokens
        or "small_key" in tokens
        or "key_small" in tokens
    )
    has_item = (
        coerce_bool(attrs.get("has_item"))
        or coerce_bool(attrs.get("has_macro_item"))
        or coerce_bool(attrs.get("has_minor_item"))
        or (item_hint > 0)
        or "i" in tokens
        or "item" in tokens
        or "macro_item" in tokens
        or "minor_item" in tokens
        or "key_item" in tokens
        or "m" in tokens
        or "treasure" in tokens
    )
    has_triforce = (
        coerce_bool(attrs.get("has_triforce"))
        or coerce_bool(attrs.get("is_triforce"))
        or coerce_bool(attrs.get("is_goal"))
        or "t" in tokens
        or "triforce" in tokens
        or "goal" in tokens
    )
    has_boss = (
        coerce_bool(attrs.get("has_boss"))
        or coerce_bool(attrs.get("is_boss"))
        or "b" in tokens
        or "boss" in tokens
    )
    has_puzzle = (
        coerce_bool(attrs.get("has_puzzle"))
        or (puzzle_hint > 0)
        or "p" in tokens
        or "puzzle" in tokens
    )
    is_start = (
        coerce_bool(attrs.get("is_start"))
        or coerce_bool(attrs.get("is_entry"))
        or "s" in tokens
        or "start" in tokens
    )
    has_gate_hint = (
        coerce_bool(attrs.get("is_lock"))
        or coerce_bool(attrs.get("requires_key"))
        or coerce_bool(attrs.get("has_gate"))
        or "l" in tokens
        or "lock" in tokens
        or "locked" in tokens
    )
    difficulty = coerce_difficulty(attrs.get("difficulty", attrs.get("difficulty_rating", 0.5)))

    enemy_signal = float(np.clip(max(float(has_enemy), enemy_hint / 3.0), 0.0, 1.0))
    key_signal = float(np.clip(max(float(has_key), key_hint / 2.0), 0.0, 1.0))
    item_signal = float(np.clip(max(float(has_item), item_hint / 2.0), 0.0, 1.0))
    puzzle_signal = float(np.clip(max(float(has_puzzle), puzzle_hint / 2.0), 0.0, 1.0))

    base_features: List[float] = [
        enemy_signal,
        key_signal,
        item_signal,
        float(has_triforce),
        float(has_boss),
        puzzle_signal,
    ]
    extended_features: List[float] = [
        float(np.clip(enemy_hint / 4.0, 0.0, 1.0)),
        float(np.clip(key_hint / 3.0, 0.0, 1.0)),
        float(np.clip(item_hint / 3.0, 0.0, 1.0)),
        float(np.clip(puzzle_hint / 3.0, 0.0, 1.0)),
        float(difficulty),
        float(is_start),
        float(has_gate_hint),
        float(coerce_bool(attrs.get("is_safe"))),
    ]
    values = fit_feature_vector(base_features + extended_features, node_dim)
    return torch.tensor(values, device=device, dtype=torch.float32)


def encode_edge_feature_vector(edge_data: Dict[str, Any], *, edge_dim: int) -> List[float]:
    """Encode edge attributes for the active conditioning schema."""
    raw_type = edge_data.get("edge_type", edge_data.get("type", ""))
    label = edge_data.get("label", "")
    constraints = set(parse_edge_type_tokens(label=str(label or ""), edge_type=str(raw_type or "")))
    metadata = edge_data.get("metadata", {}) if isinstance(edge_data.get("metadata", {}), dict) else {}
    vglc_constraints = metadata.get("vglc_constraints", edge_data.get("vglc_constraints"))
    if isinstance(vglc_constraints, (list, tuple, set)):
        constraints.update(str(t).strip().lower() for t in vglc_constraints if str(t).strip())
    elif isinstance(vglc_constraints, str) and vglc_constraints.strip():
        constraints.update(parse_edge_type_tokens(label=vglc_constraints, edge_type=""))

    def _has_any(*names: str) -> bool:
        return any(n in constraints for n in names)

    key_strength = float(np.clip(float(edge_data.get("requires_key_count", 1 if _has_any("key_locked", "locked") else 0)) / 3.0, 0.0, 1.0))
    token_strength = float(np.clip(float(edge_data.get("token_count", 1 if _has_any("multi_lock") else 0)) / 3.0, 0.0, 1.0))

    key_locked = _has_any("key_locked", "locked", "multi_lock")
    bombable = _has_any("bombable")
    soft_locked = _has_any("soft_locked", "one_way", "shutter")
    boss_locked = _has_any("boss_locked")
    item_locked = _has_any("item_locked", "item_gate")
    stair = _has_any("stair", "stairs", "warp")
    switch = _has_any("switch", "switch_locked", "state_block", "on_off_gate")
    hazard = _has_any("hazard")
    hidden = _has_any("hidden", "secret")
    shutter = _has_any("shutter")
    state_block = _has_any("state_block")

    base_vec: List[float] = [
        1.0 if (not constraints or _has_any("open", "path")) else 0.0,
        max(float(key_locked), key_strength),
        float(bombable),
        float(soft_locked),
        float(boss_locked),
        float(item_locked),
        float(stair),
        float(switch),
    ]
    if sum(base_vec[1:]) > 0.0:
        base_vec[0] = max(base_vec[0], 0.25)

    extended_vec: List[float] = [
        float(hazard),
        float(shutter),
        max(float(_has_any("multi_lock")), token_strength),
        float(state_block),
        float(hidden),
        key_strength,
    ]
    return fit_feature_vector(base_vec + extended_vec, edge_dim)


def compute_tpe_features(
    graph: nx.Graph,
    node_order: List[int],
    node_to_idx: Dict[int, int],
    node_features: torch.Tensor,
    *,
    device: torch.device,
    parse_label_tokens: Callable[[Any], set],
    coerce_bool: Callable[[Any], bool],
    coerce_difficulty: Callable[[Any], float],
    on_shortest_path_fallback: Optional[Callable[[], None]] = None,
) -> torch.Tensor:
    """Compute lightweight topological positional encodings [N, 8]."""
    num_nodes = len(node_order)
    tpe = torch.zeros(num_nodes, 8, device=device, dtype=torch.float32)
    if num_nodes == 0:
        return tpe

    start_id = next(
        (
            nid
            for nid in node_order
            if coerce_bool(graph.nodes[nid].get("is_start")) or coerce_bool(graph.nodes[nid].get("is_entry"))
        ),
        node_order[0],
    )
    goal_id = next(
        (
            nid
            for nid in node_order
            if coerce_bool(graph.nodes[nid].get("has_triforce"))
            or coerce_bool(graph.nodes[nid].get("is_triforce"))
            or coerce_bool(graph.nodes[nid].get("is_goal"))
        ),
        node_order[-1],
    )

    try:
        if graph.is_directed():
            dist_from_start = dict(nx.single_source_shortest_path_length(graph, start_id))
            dist_to_goal = dict(nx.single_source_shortest_path_length(graph.reverse(copy=False), goal_id))
            shortest_path_len = nx.shortest_path_length(graph, start_id, goal_id)
        else:
            dist_from_start = dict(nx.single_source_shortest_path_length(graph, start_id))
            dist_to_goal = dict(nx.single_source_shortest_path_length(graph, goal_id))
            shortest_path_len = nx.shortest_path_length(graph, start_id, goal_id)
    except Exception:
        if on_shortest_path_fallback is not None:
            on_shortest_path_fallback()
        logger.debug(
            "Falling back to minimal TPE distances (start=%s goal=%s)",
            start_id,
            goal_id,
            exc_info=True,
        )
        dist_from_start = {start_id: 0}
        dist_to_goal = {goal_id: 0}
        shortest_path_len = None

    max_start = max(dist_from_start.values(), default=1)
    max_goal = max(dist_to_goal.values(), default=1)

    for node_id in node_order:
        idx = node_to_idx[node_id]
        attrs = graph.nodes[node_id]
        label_tokens = parse_label_tokens(attrs.get("label"))

        d_start = dist_from_start.get(node_id, max_start + 1)
        d_goal = dist_to_goal.get(node_id, max_goal + 1)

        tpe[idx, 0] = float(d_start / max(1, max_start))
        tpe[idx, 1] = float(d_goal / max(1, max_goal))

        if graph.is_directed():
            degree = graph.in_degree(node_id) + graph.out_degree(node_id)
        else:
            degree = graph.degree(node_id)
        tpe[idx, 2] = min(float(degree) / 4.0, 1.0)

        if shortest_path_len is not None:
            on_main = int((d_start + d_goal) == shortest_path_len)
            tpe[idx, 3] = float(on_main)

        tpe[idx, 4] = float(node_features[idx, 1].item() > 0.0)

        has_lock = (
            coerce_bool(attrs.get("is_lock"))
            or coerce_bool(attrs.get("requires_key"))
            or "lock" in label_tokens
            or "l" in label_tokens
        )
        tpe[idx, 5] = float(has_lock)

        tpe[idx, 6] = coerce_difficulty(attrs.get("difficulty", attrs.get("difficulty_rating", 0.5)))
        tpe[idx, 7] = float(attrs.get("key_id") is not None or coerce_bool(attrs.get("requires_key")))

    return tpe
