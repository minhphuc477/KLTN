"""Advisory progression-pacing evidence for generated mission graphs.

These metrics describe the exact resource-aware solution path. They are not
hard playability constraints and must not be presented as measured enjoyment.
Discrete beats are measured before any resampling or smoothing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import networkx as nx
import numpy as np


_SETUP_ROLES = {"KEY", "BIG_KEY", "ITEM", "PROTECTION_ITEM", "TOKEN", "TUTORIAL_PUZZLE"}
_GATE_ROLES = {"LOCK", "BOSS_DOOR", "COMBAT_PUZZLE"}
_CLIMAX_ROLES = {"BOSS", "MINI_BOSS", "ARENA", "COMPLEX_PUZZLE", "GOAL"}
_REST_ROLES = {"SCENIC", "RESOURCE_FARM"}


def _role(attrs: Mapping[str, Any]) -> str:
    value = attrs.get("node_type", attrs.get("type", attrs.get("label", "")))
    if hasattr(value, "name"):
        value = value.name
    return str(value).strip().upper().split(".")[-1]


def _finite_unit(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return float(np.clip(result, 0.0, 1.0))


def _edge_traversal_cost(graph: nx.Graph, source: Any, target: Any) -> float:
    """Resolve a finite non-negative edge cost without inventing a gate rule."""
    raw = graph.get_edge_data(source, target, default={}) or {}
    candidates: list[Mapping[str, Any]]
    if graph.is_multigraph():
        candidates = [
            dict(attrs)
            for attrs in raw.values()
            if isinstance(attrs, Mapping)
        ]
    else:
        candidates = [dict(raw)] if isinstance(raw, Mapping) else []

    costs: list[float] = []
    for attrs in candidates:
        value = attrs.get(
            "traversal_cost",
            attrs.get("cost", attrs.get("weight", 1.0)),
        )
        try:
            cost = float(value)
        except (TypeError, ValueError):
            cost = 1.0
        if math.isfinite(cost) and cost >= 0.0:
            costs.append(cost)
    return min(costs) if costs else 1.0


def evaluate_solution_path_pacing(
    graph: nx.Graph,
    solution_path: Sequence[Any],
) -> Dict[str, Any]:
    """Return event-level pacing evidence for an exact solution path."""
    path = list(solution_path)
    if len(path) < 2:
        return {
            "pacing_contract_applicable": False,
            "pacing_path_edges": max(0, len(path) - 1),
            "pacing_failure_reason": "exact solution path has fewer than two nodes",
        }

    roles: list[str] = []
    tensions: list[float] = []
    landmark_indices: list[int] = []
    setup_indices: list[int] = []
    gate_indices: list[int] = []
    climax_indices: list[int] = []
    rest_indices: list[int] = []
    landmark_roles = _SETUP_ROLES | _GATE_ROLES | _CLIMAX_ROLES | _REST_ROLES
    for index, node_id in enumerate(path):
        attrs = dict(graph.nodes[node_id]) if node_id in graph else {}
        role = _role(attrs)
        roles.append(role)
        default_tension = 1.0 if role in _CLIMAX_ROLES else 0.5
        tensions.append(
            _finite_unit(
                attrs.get(
                    "tension_value",
                    attrs.get("tension", attrs.get("difficulty", default_tension)),
                ),
                default_tension,
            )
        )
        if role in _SETUP_ROLES:
            setup_indices.append(index)
        if role in _GATE_ROLES:
            gate_indices.append(index)
        if role in _CLIMAX_ROLES:
            climax_indices.append(index)
        if role in _REST_ROLES:
            rest_indices.append(index)
        if role in landmark_roles:
            landmark_indices.append(index)

    path_edges = len(path) - 1
    edge_costs = [
        _edge_traversal_cost(graph, source, target)
        for source, target in zip(path[:-1], path[1:])
    ]
    cumulative_costs = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            np.cumsum(np.asarray(edge_costs, dtype=np.float64)),
        ]
    )
    normalized_landmarks = [float(index / path_edges) for index in landmark_indices]
    spacings = (
        np.diff(np.asarray(landmark_indices, dtype=np.float32))
        if len(landmark_indices) > 1
        else np.asarray([], dtype=np.float32)
    )
    spacing_mean = float(np.mean(spacings)) if spacings.size else 0.0
    spacing_cv = (
        float(np.std(spacings) / spacing_mean)
        if spacings.size and spacing_mean > 0.0
        else 0.0
    )
    landmark_segments = []
    for start_index, end_index in zip(landmark_indices[:-1], landmark_indices[1:]):
        landmark_segments.append(
            {
                "source_index": int(start_index),
                "target_index": int(end_index),
                "source_role": roles[start_index],
                "target_role": roles[end_index],
                "corridor_edge_count": int(end_index - start_index),
                "intermediate_room_count": int(max(0, end_index - start_index - 1)),
                "path_cost": float(
                    cumulative_costs[end_index] - cumulative_costs[start_index]
                ),
            }
        )
    landmark_segment_costs = np.asarray(
        [segment["path_cost"] for segment in landmark_segments],
        dtype=np.float64,
    )

    ordered_pairs = 0
    valid_pairs = 0
    for setup in setup_indices:
        for gate in gate_indices:
            ordered_pairs += 1
            valid_pairs += int(setup < gate)
    for gate in gate_indices:
        for climax in climax_indices:
            ordered_pairs += 1
            valid_pairs += int(gate < climax)
    beat_order_score = float(valid_pairs / ordered_pairs) if ordered_pairs else None

    raw_delta = np.diff(np.asarray(tensions, dtype=np.float32))
    event_threshold = max(
        0.12,
        float(np.quantile(np.abs(raw_delta), 0.75)) if raw_delta.size else 0.12,
    )
    tension_event_positions = [
        float((index + 0.5) / max(1, raw_delta.size))
        for index, value in enumerate(raw_delta)
        if abs(float(value)) >= event_threshold
    ]

    first_seen: Dict[Any, int] = {}
    revisit_distance = 0
    revisit_steps = 0
    for index, node_id in enumerate(path):
        if node_id in first_seen:
            revisit_steps += 1
            revisit_distance += index - first_seen[node_id]
        else:
            first_seen[node_id] = index

    return {
        "pacing_contract_applicable": True,
        "pacing_path_edges": int(path_edges),
        "pacing_roles": roles,
        "pacing_landmark_count": int(len(landmark_indices)),
        "pacing_landmark_positions": normalized_landmarks,
        "pacing_landmark_spacing_mean_edges": spacing_mean,
        "pacing_landmark_spacing_cv": spacing_cv,
        "pacing_total_graph_path_cost": float(cumulative_costs[-1]),
        "pacing_landmark_segments": landmark_segments,
        "pacing_landmark_segment_cost_mean": (
            float(np.mean(landmark_segment_costs))
            if landmark_segment_costs.size
            else 0.0
        ),
        "pacing_landmark_segment_cost_cv": (
            float(np.std(landmark_segment_costs) / np.mean(landmark_segment_costs))
            if landmark_segment_costs.size
            and float(np.mean(landmark_segment_costs)) > 0.0
            else 0.0
        ),
        "pacing_setup_before_gate_before_climax_score": beat_order_score,
        "pacing_tension_event_count": int(len(tension_event_positions)),
        "pacing_tension_event_positions": tension_event_positions,
        "pacing_rest_count": int(len(rest_indices)),
        "pacing_revisit_step_ratio": float(revisit_steps / max(1, len(path))),
        "pacing_mean_revisit_depth_edges": float(
            revisit_distance / max(1, revisit_steps)
        ),
    }
