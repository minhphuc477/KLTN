"""Graph feature encoding helpers for condition encoder inputs."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch

from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FLOOR_FEATURE_INDEX,
    GRAPH_NODE_FEATURE_DIM,
    GRAPH_TPE_DIM,
    parse_edge_type_tokens,
)

logger = logging.getLogger(__name__)


def build_key_lock_pairs(
    graph: nx.Graph,
    node_to_idx: Mapping[Any, int],
    *,
    start_node: Optional[Any],
) -> List[Tuple[int, int]]:
    """Pair key providers with gated destinations for LogicNet supervision."""
    small_keys: List[Any] = []
    boss_keys: List[Any] = []
    item_providers: List[Any] = []
    token_providers: List[Any] = []
    provider_ids: Dict[Any, Optional[str]] = {}
    for node_id, attrs in graph.nodes(data=True):
        attrs_dict = dict(attrs)
        raw_type = str(
            attrs_dict.get(
                "type",
                attrs_dict.get("node_type", attrs_dict.get("room_type", "")),
            )
            or ""
        ).strip().upper()
        label_tokens = {
            token.strip()
            for token in str(attrs_dict.get("label", "") or "").split(",")
            if token.strip()
        }
        is_boss_key = (
            raw_type in {"BIG_KEY", "BOSS_KEY"}
            or bool(attrs_dict.get("has_boss_key", False))
            or bool(label_tokens & {"BIG_KEY", "BOSS_KEY", "big_key", "boss_key", "K"})
        )
        is_small_key = (
            not is_boss_key
            and (
                raw_type in {"KEY", "SMALL_KEY", "KEY_SMALL"}
                or bool(attrs_dict.get("has_key", False))
                or bool(
                    label_tokens
                    & {
                        "KEY",
                        "SMALL_KEY",
                        "KEY_SMALL",
                        "key",
                        "small_key",
                        "key_small",
                        "k",
                    }
                )
            )
        )
        is_item_provider = (
            not is_small_key
            and not is_boss_key
            and (
                raw_type in {"ITEM", "KEY_ITEM", "MACRO_ITEM", "PROTECTION_ITEM"}
                or bool(attrs_dict.get("has_item", False))
                or bool(attrs_dict.get("has_macro_item", False))
                or bool(label_tokens & {"I", "ITEM", "KEY_ITEM", "MACRO_ITEM"})
            )
        )
        is_token_provider = (
            not is_small_key
            and not is_boss_key
            and not is_item_provider
            and (
                raw_type == "TOKEN"
                or bool(attrs_dict.get("has_token", False))
                or bool(label_tokens & {"TOKEN", "token"})
            )
        )
        if not (is_small_key or is_boss_key or is_item_provider or is_token_provider):
            continue
        raw_key_id = (
            attrs_dict.get("token_id")
            if is_token_provider
            else (
                attrs_dict.get("item_type", attrs_dict.get("required_item"))
                if is_item_provider
                else attrs_dict.get("key_id")
            )
        )
        provider_ids[node_id] = (
            str(raw_key_id).strip()
            if raw_key_id is not None and str(raw_key_id).strip()
            else None
        )
        if is_boss_key:
            boss_keys.append(node_id)
        elif is_small_key:
            small_keys.append(node_id)
        elif is_token_provider:
            token_providers.append(node_id)
        else:
            item_providers.append(node_id)

    if not small_keys and not boss_keys and not item_providers and not token_providers:
        return []

    try:
        start_distances = (
            nx.single_source_shortest_path_length(graph, start_node)
            if start_node in graph
            else {}
        )
    except (nx.NetworkXError, nx.NodeNotFound):
        start_distances = {}

    gate_records: List[
        Tuple[
            int,
            int,
            Tuple[str, str],
            str,
            Optional[str],
            int,
            Any,
            Any,
            nx.Graph,
        ]
    ] = []
    seen_gates: Set[Tuple[str, frozenset[Any], Optional[str]]] = set()
    for raw_source, raw_target, edge_attrs in graph.edges(data=True):
        constraints = {
            str(value).strip().lower()
            for value in parse_edge_type_tokens(
                label=str(edge_attrs.get("label", "") or ""),
                edge_type=str(
                    edge_attrs.get("edge_type", edge_attrs.get("type", "")) or ""
                ),
            )
        }
        if constraints & {"multi_lock"}:
            key_kind = "token"
        elif constraints & {"boss_locked", "boss"}:
            key_kind = "boss"
        elif constraints & {"item_locked", "item_gate"}:
            key_kind = "item"
        elif constraints & {"key_locked", "locked", "k"}:
            key_kind = "small"
        else:
            continue

        required_raw = edge_attrs.get(
            "token_id" if key_kind == "token" else (
                "item_required" if key_kind == "item" else "key_required"
            ),
            edge_attrs.get("required_item") if key_kind == "item" else None,
        )
        required_id = (
            str(required_raw).strip()
            if required_raw is not None and str(required_raw).strip()
            else None
        )
        count_field = "token_count" if key_kind == "token" else "requires_key_count"
        try:
            required_count = max(1, int(edge_attrs.get(count_field, 1) or 1))
        except (TypeError, ValueError, OverflowError):
            required_count = 1
        gate_key = (key_kind, frozenset((raw_source, raw_target)), required_id)
        if gate_key in seen_gates:
            continue
        seen_gates.add(gate_key)

        source, target = raw_source, raw_target
        if start_distances.get(raw_target, 10**9) < start_distances.get(raw_source, 10**9):
            source, target = raw_target, raw_source
        if target not in node_to_idx:
            continue

        graph_without_gate = graph.copy()
        if graph_without_gate.has_edge(raw_source, raw_target):
            graph_without_gate.remove_edge(raw_source, raw_target)
        if graph_without_gate.has_edge(raw_target, raw_source):
            graph_without_gate.remove_edge(raw_target, raw_source)

        gate_records.append(
            (
                int(start_distances.get(source, 10**9)),
                int(start_distances.get(target, 10**9)),
                (type(target).__name__, str(target)),
                key_kind,
                required_id,
                required_count,
                source,
                target,
                graph_without_gate,
            )
        )

    ordered_pairs: List[Tuple[int, int]] = []
    used_small_key_providers: Set[Any] = set()
    used_token_providers: Set[Any] = set()
    for (
        _source_distance,
        _target_distance,
        _target_sort_key,
        key_kind,
        required_id,
        required_count,
        source,
        target,
        graph_without_gate,
    ) in sorted(
        gate_records,
        key=lambda record: (
            record[0],
            record[1],
            record[2],
            record[3],
            str(record[4] or ""),
        ),
    ):
        if key_kind == "boss":
            providers = boss_keys
        elif key_kind == "item":
            providers = item_providers
        elif key_kind == "token":
            providers = token_providers
        else:
            providers = small_keys
        candidates = [
            provider
            for provider in providers
            if provider in node_to_idx
            and (required_id is None or provider_ids.get(provider) == required_id)
            and (key_kind != "small" or provider not in used_small_key_providers)
            and (key_kind != "token" or provider not in used_token_providers)
        ]
        ranked: List[Tuple[int, Tuple[str, str], Any]] = []
        for provider in candidates:
            try:
                if start_node in graph_without_gate:
                    nx.shortest_path_length(graph_without_gate, start_node, provider)
                    nx.shortest_path_length(graph_without_gate, start_node, source)
                elif provider not in start_distances:
                    continue
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            ranked.append(
                (
                    -int(start_distances.get(provider, -1)),
                    (type(provider).__name__, str(provider)),
                    provider,
                )
            )
        selected = [entry[-1] for entry in sorted(ranked)[:required_count]]
        if len(selected) == required_count:
            for provider in selected:
                ordered_pairs.append((int(node_to_idx[provider]), int(node_to_idx[target])))
                if key_kind == "small":
                    used_small_key_providers.add(provider)
                elif key_kind == "token":
                    used_token_providers.add(provider)

    return list(dict.fromkeys(ordered_pairs))


def condition_feature_dims(condition_encoder: Any) -> Tuple[int, int]:
    """Get active (node_dim, edge_dim) expected by the condition encoder."""
    node_dim = int(GRAPH_NODE_FEATURE_DIM)
    edge_dim = int(GRAPH_EDGE_FEATURE_DIM)
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


def build_default_node_positions(
    num_nodes: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic fallback positions [N, 2] on a normalized line."""
    n = max(0, int(num_nodes))
    if n == 0:
        return torch.zeros((0, 2), device=device, dtype=dtype)
    if n == 1:
        return torch.zeros((1, 2), device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=dtype)
    ys = torch.zeros(n, device=device, dtype=dtype)
    return torch.stack([xs, ys], dim=1)


def compute_rwse_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int = 8,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute Random Walk Structural Encodings (RWSE) [N, steps].

    GPS (NeurIPS 2022) identifies RWSE as a low-cost structural encoding that
    consistently improves graph transformer performance. We use it as a robust
    fallback when explicit topological positional encodings are unavailable.
    """
    n = max(0, int(num_nodes))
    k = max(1, int(steps))
    rwse = torch.zeros((n, k), device=device, dtype=dtype)
    if n == 0:
        return rwse

    if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
        return rwse

    adjacency = torch.zeros((n, n), device=device, dtype=dtype)
    if edge_index.numel() > 0:
        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)
        valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
        src = src[valid]
        dst = dst[valid]
        if src.numel() > 0:
            adjacency[src, dst] = 1.0
            adjacency[dst, src] = 1.0

    adjacency = adjacency + torch.eye(n, device=device, dtype=dtype)
    transition = adjacency / adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
    walk = transition
    for step_idx in range(k):
        rwse[:, step_idx] = torch.diagonal(walk, 0)
        walk = walk @ transition
    return rwse


def compute_rrwp_edge_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int = 8,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute relative random-walk probabilities for each directed edge [E, steps]."""
    k = max(1, int(steps))
    if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
        return torch.zeros((0, k), device=device, dtype=dtype)

    edge_count = int(edge_index.shape[1])
    rrwp = torch.zeros((edge_count, k), device=device, dtype=dtype)
    n = max(0, int(num_nodes))
    if n == 0 or edge_count == 0:
        return rrwp

    src_all = edge_index[0].to(device=device, dtype=torch.long)
    dst_all = edge_index[1].to(device=device, dtype=torch.long)
    valid = (src_all >= 0) & (src_all < n) & (dst_all >= 0) & (dst_all < n)
    if not bool(valid.any()):
        return rrwp

    adjacency = torch.zeros((n, n), device=device, dtype=dtype)
    adjacency[src_all[valid], dst_all[valid]] = 1.0
    adjacency = adjacency + torch.eye(n, device=device, dtype=dtype)
    transition = adjacency / adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)

    walk = transition
    for step_idx in range(k):
        rrwp[valid, step_idx] = walk[src_all[valid], dst_all[valid]]
        walk = walk @ transition
    return rrwp


def _single_source_graph_distances(
    adjacency: List[List[int]],
    source_idx: int,
) -> List[int]:
    """Breadth-first shortest-path distances over a Python adjacency list."""
    num_nodes = len(adjacency)
    if num_nodes == 0 or source_idx < 0 or source_idx >= num_nodes:
        return [-1] * num_nodes

    distances = [-1] * num_nodes
    distances[source_idx] = 0
    queue: deque[int] = deque([int(source_idx)])

    while queue:
        node = int(queue.popleft())
        next_distance = int(distances[node]) + 1
        for neighbor in adjacency[node]:
            if neighbor < 0 or neighbor >= num_nodes:
                continue
            if distances[neighbor] != -1:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)

    return distances


def compute_current_node_distance_features(
    edge_index: Optional[torch.Tensor],
    num_nodes: int,
    *,
    current_node_idx: Optional[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    max_distance: int = 8,
) -> torch.Tensor:
    """
    Compute current-room-aware distance features [N, 4].

    Feature layout:
    - [:, 0] normalized undirected shortest-path distance to current node
    - [:, 1] normalized directed forward distance (current -> node)
    - [:, 2] normalized directed backward distance (node -> current)
    - [:, 3] current-node indicator

    Unreachable nodes are assigned a sentinel distance value of -1.0 so they
    are distinguishable from reachable nodes at the max-distance clip.
    """
    n = max(0, int(num_nodes))
    features = torch.zeros((n, 4), device=device, dtype=dtype)
    if n == 0 or current_node_idx is None:
        return features

    anchor = int(current_node_idx)
    if anchor < 0 or anchor >= n:
        logger.warning(
            "compute_current_node_distance_features received current_node_idx=%d for num_nodes=%d. Returning zeros.",
            anchor,
            n,
        )
        return features

    max_d = max(1, int(max_distance))
    directed_adj: List[List[int]] = [[] for _ in range(n)]
    reverse_adj: List[List[int]] = [[] for _ in range(n)]
    undirected_adj: List[List[int]] = [[] for _ in range(n)]

    if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and int(edge_index.shape[0]) == 2:
        if edge_index.numel() > 0:
            src = edge_index[0].detach().to(device="cpu", dtype=torch.long)
            dst = edge_index[1].detach().to(device="cpu", dtype=torch.long)
            valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
            src = src[valid].tolist()
            dst = dst[valid].tolist()
            for s, d in zip(src, dst):
                directed_adj[s].append(d)
                reverse_adj[d].append(s)
                undirected_adj[s].append(d)
                if s != d:
                    undirected_adj[d].append(s)

    undirected_dist = _single_source_graph_distances(undirected_adj, anchor)
    forward_dist = _single_source_graph_distances(directed_adj, anchor)
    backward_dist = _single_source_graph_distances(reverse_adj, anchor)

    def _encode_distances(distances: List[int]) -> torch.Tensor:
        encoded = torch.full((n,), -1.0, device=device, dtype=dtype)
        for node_idx, raw_distance in enumerate(distances):
            if raw_distance < 0:
                continue
            encoded[node_idx] = float(min(raw_distance, max_d)) / float(max_d)
        return encoded

    features[:, 0] = _encode_distances(undirected_dist)
    features[:, 1] = _encode_distances(forward_dist)
    features[:, 2] = _encode_distances(backward_dist)
    features[anchor, 3] = 1.0
    return features


def align_nodewise_tensor(
    value: Optional[torch.Tensor],
    *,
    num_nodes: int,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    feature_name: str,
    default_value: torch.Tensor,
) -> torch.Tensor:
    """
    Pad/truncate node-aligned tensors to [N, D].

    This keeps graph conditioning robust when OOD graphs arrive with missing or
    schema-shifted metadata.
    """
    target = default_value.to(device=device, dtype=dtype)
    if tuple(target.shape) != (int(num_nodes), int(feature_dim)):
        raise ValueError(
            f"default_value for {feature_name} must have shape {(int(num_nodes), int(feature_dim))}, "
            f"got {tuple(target.shape)}."
        )
    if not isinstance(value, torch.Tensor):
        return target

    tensor = value.to(device=device, dtype=dtype)
    if tensor.dim() == 1:
        expected = int(num_nodes) * int(feature_dim)
        if int(tensor.numel()) == expected:
            tensor = tensor.view(int(num_nodes), int(feature_dim))
        else:
            tensor = tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        logger.warning(
            "%s must be 2D [N, D], got %s. Using fallback tensor.",
            feature_name,
            tuple(tensor.shape),
        )
        return target

    if tuple(tensor.shape) != tuple(target.shape):
        logger.warning(
            "%s shape mismatch: got %s, expected %s. Applying pad/truncate fallback.",
            feature_name,
            tuple(tensor.shape),
            tuple(target.shape),
        )
    rows = min(int(num_nodes), int(tensor.shape[0]))
    cols = min(int(feature_dim), int(tensor.shape[1]))
    if rows > 0 and cols > 0:
        target[:rows, :cols] = tensor[:rows, :cols]
    return target


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
    raw_type = str(attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or "").strip().lower()
    role_tokens = set(tokens) | set(parse_label_tokens(raw_type))

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
        or "e" in role_tokens
        or "enemy" in role_tokens
        or "b" in role_tokens
        or "boss" in role_tokens
    )
    has_key = (
        coerce_bool(attrs.get("has_key"))
        or (key_hint > 0)
        or "k" in role_tokens
        or "key" in role_tokens
        or "small_key" in role_tokens
        or "key_small" in role_tokens
    )
    has_item = (
        coerce_bool(attrs.get("has_item"))
        or coerce_bool(attrs.get("has_macro_item"))
        or coerce_bool(attrs.get("has_minor_item"))
        or (item_hint > 0)
        or "i" in role_tokens
        or "item" in role_tokens
        or "macro_item" in role_tokens
        or "minor_item" in role_tokens
        or "key_item" in role_tokens
        or "m" in role_tokens
        or "treasure" in role_tokens
    )
    has_triforce = (
        coerce_bool(attrs.get("has_triforce"))
        or coerce_bool(attrs.get("is_triforce"))
        or coerce_bool(attrs.get("is_goal"))
        or raw_type in {"goal", "triforce"}
        or "t" in role_tokens
        or "triforce" in role_tokens
        or "goal" in role_tokens
    )
    has_boss = (
        coerce_bool(attrs.get("has_boss"))
        or coerce_bool(attrs.get("is_boss"))
        or "b" in role_tokens
        or "boss" in role_tokens
    )
    has_puzzle = (
        coerce_bool(attrs.get("has_puzzle"))
        or (puzzle_hint > 0)
        or "p" in role_tokens
        or "puzzle" in role_tokens
    )
    is_start = (
        coerce_bool(attrs.get("is_start"))
        or coerce_bool(attrs.get("is_entry"))
        or raw_type in {"start", "entry"}
        or "s" in role_tokens
        or "start" in role_tokens
        or "entry" in role_tokens
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
    is_secret = (
        coerce_bool(attrs.get("is_secret"))
        or "secret" in tokens
        or "hidden" in tokens
    )
    is_hub = (
        coerce_bool(attrs.get("is_hub"))
        or (int(max(0, int(attrs.get("virtual_layer", 0) or 0))) > 0)
        or (int(max(0, int(attrs.get("sector_id", 0) or 0))) > 0)
    )

    extended_features: List[float] = [
        float(np.clip(enemy_hint / 4.0, 0.0, 1.0)),
        float(np.clip(key_hint / 3.0, 0.0, 1.0)),
        float(np.clip(item_hint / 3.0, 0.0, 1.0)),
        float(np.clip(puzzle_hint / 3.0, 0.0, 1.0)),
        float(difficulty),
        float(is_start),
        float(is_secret),
        float(is_hub),
    ]
    floor_value: Any = None
    for floor_key in ("floor", "floor_id", "floor_level", "z"):
        if floor_key in attrs and attrs.get(floor_key) is not None:
            floor_value = attrs.get(floor_key)
            break
    if floor_value is None:
        position = attrs.get("position", attrs.get("pos"))
        if isinstance(position, (list, tuple, np.ndarray)) and len(position) > 2:
            floor_value = position[2]
    try:
        normalized_floor = float(np.clip(float(floor_value or 0.0) / 5.0, -1.0, 1.0))
    except (TypeError, ValueError, OverflowError):
        normalized_floor = 0.0

    features = base_features + extended_features
    if int(node_dim) > int(GRAPH_NODE_FLOOR_FEATURE_INDEX):
        features.append(normalized_floor)
    values = fit_feature_vector(features, node_dim)
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

    def _safe_nonneg_int(value: Any, default: int = 0) -> int:
        try:
            return int(max(0, int(value)))
        except Exception:
            return int(default)

    key_strength = float(
        np.clip(
            float(edge_data.get("requires_key_count", 1 if _has_any("key_locked", "locked") else 0)) / 3.0,
            0.0,
            1.0,
        )
    )
    token_strength = float(
        np.clip(
            float(edge_data.get("token_count", 1 if _has_any("multi_lock") else 0)) / 3.0,
            0.0,
            1.0,
        )
    )

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
    one_way = _has_any("one_way")

    preferred_direction = str(
        edge_data.get("preferred_direction", metadata.get("preferred_direction", "")) or ""
    ).strip().lower()
    one_way_forward = 0.0
    one_way_backward = 0.0
    if one_way:
        if preferred_direction in {"forward", "east", "south", "down"}:
            one_way_forward = 1.0
        elif preferred_direction in {"backward", "west", "north", "up"}:
            one_way_backward = 1.0
        else:
            one_way_forward = 0.5
            one_way_backward = 0.5

    switches_required = edge_data.get("switches_required", metadata.get("switches_required", []))
    if isinstance(switches_required, (list, tuple, set)):
        switch_count = len([value for value in switches_required if value is not None])
    else:
        switch_count = _safe_nonneg_int(switches_required, default=0)
    if switch_count <= 0 and switch:
        switch_count = 1
    switch_count_strength = float(np.clip(float(switch_count) / 4.0, 0.0, 1.0))
    battery_signal = float(
        (_safe_nonneg_int(edge_data.get("battery_id", metadata.get("battery_id")), default=0) > 0)
        or (switch_count > 1)
    )

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
        float(hidden),
        one_way_forward,
        one_way_backward,
        max(float(_has_any("multi_lock")), token_strength),
        max(float(state_block), switch_count_strength),
        battery_signal,
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
    tpe = torch.zeros(num_nodes, int(GRAPH_TPE_DIM), device=device, dtype=torch.float32)
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
