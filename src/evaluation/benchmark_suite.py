"""
Research-Oriented Benchmark Suite for Zelda Dungeon Generation.

This module provides a reproducible protocol to compare generated mission graphs
against VGLC references using literature-aligned factors:

1) Completeness / validity:
   - start-goal existence
   - connectivity
   - path existence
   - mission-grammar constraint validity (when representable)

2) Progression quality:
   - lock/key pressure
   - path pressure
   - progression complexity proxy

3) Quality-diversity / expressive range:
   - coverage in descriptor spaces
   - descriptor diversity
   - novelty against reference corpus
   - fidelity (distribution similarity) against reference corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from src.core.definitions import (
    CHAR_TO_SEMANTIC,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    TileID,
    parse_edge_type_tokens,
    parse_node_label_tokens,
)
from src.data.zelda_core import DOTParser
from src.data.vglc_utils import filter_virtual_nodes, get_physical_start_node
from src.generation.evolutionary_director import (
    EvolutionaryTopologyGenerator,
    networkx_to_mission_graph,
)
from src.generation.grammar import MissionGrammar
from src.generation.realism_profiles import (
    get_realism_tuning_profile,
    list_realism_tuning_profiles,
)
from src.generation.weighted_bayesian_wfc import (
    WeightedBayesianWFCConfig,
    extract_tile_priors_from_vqvae,
    integrate_weighted_wfc_into_pipeline,
)

logger = logging.getLogger(__name__)


LOCK_EDGE_TYPES = {
    "locked",
    "key_locked",
    "boss_locked",
    "item_locked",
    "switch",
    "switch_locked",
    "state_block",
    "item_gate",
}
SOFT_LOCK_EDGE_TYPES = {"soft_locked", "one_way", "shutter"}
BOMBABLE_EDGE_TYPES = {"bombable"}
ITEM_GATE_EDGE_TYPES = {"item_locked", "item_gate"}
SWITCH_EDGE_TYPES = {"switch", "switch_locked", "state_block", "on_off_gate"}
STAIR_EDGE_TYPES = {"stair", "stairs", "warp"}
# Shortcut excludes generic stair connectors so topology shortcut pressure
# is not conflated with vertical traversal semantics.
SHORTCUT_EDGE_TYPES = {"shortcut", "warp", "teleport", "portal"}


DEFAULT_WFC_PROBE_MASK_RATIOS: Tuple[float, ...] = (0.15, 0.35, 0.55)


@dataclass
class GraphDescriptor:
    """Descriptor + completeness fields for one mission graph."""

    num_nodes: int
    num_edges: int
    has_start: bool
    has_goal: bool
    connected: bool
    path_exists: bool
    path_length: int
    directed_path_exists: bool
    directed_path_length: int
    weak_path_exists: bool
    weak_path_length: int
    directionality_gap: float
    key_count: int
    lock_count: int
    enemy_count: int
    puzzle_count: int
    item_count: int
    bombable_count: int
    softlock_count: int
    item_gate_count: int
    switch_count: int
    stair_count: int
    gate_variety: float
    linearity: float
    leniency: float
    branching_factor: float
    cycle_density: float
    shortcut_density: float
    gating_density: float
    gate_depth_ratio: float
    path_depth_ratio: float
    progression_complexity: float
    topology_complexity: float
    constraint_valid: bool
    repair_applied: bool
    total_repairs: int
    lock_key_repairs: int
    progression_repairs: int
    wave3_repairs: int
    repair_rounds: int
    generation_constraint_rejections: int
    candidate_repairs_applied: int
    rule_applications: int
    rule_applied: int
    rule_skipped: int


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark statistics."""

    num_generated: int
    num_reference: int
    mean_generation_time_sec: float
    completeness: Dict[str, float]
    robustness: Dict[str, float]
    expressive_range: Dict[str, float]
    reference_comparison: Dict[str, float]
    generated_descriptor_means: Dict[str, float]
    reference_descriptor_means: Dict[str, float]
    generated_descriptor_stds: Dict[str, float]
    reference_descriptor_stds: Dict[str, float]
    confidence_intervals: Dict[str, Dict[str, float]]
    calibration: Dict[str, Any]
    data_audit: Dict[str, Any] = field(default_factory=dict)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _normalized_node_tokens(attrs: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []

    label = str(attrs.get("label", "") or "")
    if label:
        tokens.extend(parse_node_label_tokens(label))

    node_type = str(attrs.get("type", "") or "")
    if node_type:
        tokens.append(node_type.lower())

    if attrs.get("is_start"):
        tokens.extend(["s", "start"])
    if attrs.get("is_triforce") or attrs.get("is_goal"):
        tokens.extend(["t", "goal", "triforce"])
    if attrs.get("is_boss"):
        tokens.extend(["b", "boss"])

    out = []
    seen = set()
    for tok in tokens:
        t = str(tok).strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _is_start_node(attrs: Dict[str, Any]) -> bool:
    tokens = set(_normalized_node_tokens(attrs))
    if attrs.get("is_entry"):
        return True
    return any(t in tokens for t in ("s", "start"))


def _is_goal_node(attrs: Dict[str, Any]) -> bool:
    tokens = set(_normalized_node_tokens(attrs))
    return any(t in tokens for t in ("t", "goal", "triforce"))


def _is_key_node(attrs: Dict[str, Any]) -> bool:
    tokens = set(_normalized_node_tokens(attrs))
    if "key" in tokens and "boss_key" not in tokens:
        return True
    return "k" in tokens or "key_small" in tokens


def _is_enemy_node(attrs: Dict[str, Any]) -> bool:
    tokens = set(_normalized_node_tokens(attrs))
    return "e" in tokens or "enemy" in tokens or "boss" in tokens or "b" in tokens


def _is_puzzle_node(attrs: Dict[str, Any]) -> bool:
    tokens = set(_normalized_node_tokens(attrs))
    return "p" in tokens or "puzzle" in tokens


def _is_item_node(attrs: Dict[str, Any]) -> bool:
    tokens = set(_normalized_node_tokens(attrs))
    return bool(
        {"i", "item", "minor_item", "macro_item", "key_item", "m"}.intersection(tokens)
        or ("I" in set(parse_node_label_tokens(str(attrs.get("label", "") or ""))))
    )


def _count_hint(attrs: Dict[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        try:
            return int(max(0, int(value)))
        except Exception:
            continue
    return 0


def _edge_constraints(attrs: Dict[str, Any]) -> List[str]:
    label = str(attrs.get("label", "") or "")
    edge_type = str(attrs.get("edge_type", attrs.get("type", "")) or "")
    return [str(t).strip().lower() for t in parse_edge_type_tokens(label=label, edge_type=edge_type)]


def _extract_generation_stats(graph: nx.Graph) -> Dict[str, int]:
    stats = graph.graph.get("generation_stats", {})
    if not isinstance(stats, dict):
        return {
            "repair_applied": 0,
            "total_repairs": 0,
            "lock_key_repairs": 0,
            "progression_repairs": 0,
            "wave3_repairs": 0,
            "repair_rounds": 0,
            "generation_constraint_rejections": 0,
            "candidate_repairs_applied": 0,
            "rule_applications": 0,
            "rule_applied": 0,
            "rule_skipped": 0,
        }
    return {
        "repair_applied": int(bool(stats.get("repair_applied", False))),
        "total_repairs": int(max(0, int(stats.get("total_repairs", 0)))),
        "lock_key_repairs": int(max(0, int(stats.get("lock_key_repairs", 0)))),
        "progression_repairs": int(max(0, int(stats.get("progression_repairs", 0)))),
        "wave3_repairs": int(max(0, int(stats.get("wave3_repairs", 0)))),
        "repair_rounds": int(max(0, int(stats.get("repair_rounds", 0)))),
        "generation_constraint_rejections": int(max(0, int(stats.get("generation_constraint_rejections", 0)))),
        "candidate_repairs_applied": int(max(0, int(stats.get("candidate_repairs_applied", 0)))),
        "rule_applications": int(max(0, int(stats.get("rule_applications", 0)))),
        "rule_applied": int(max(0, int(stats.get("rule_applied", 0)))),
        "rule_skipped": int(max(0, int(stats.get("rule_skipped", 0)))),
    }


def _select_start_goal_nodes(G: nx.Graph) -> Tuple[Optional[Any], Optional[Any]]:
    start_candidates = []
    goal_candidates = []
    for node_id, attrs in G.nodes(data=True):
        if _is_start_node(attrs):
            start_candidates.append(node_id)
        if _is_goal_node(attrs):
            goal_candidates.append(node_id)

    start = min(start_candidates, key=lambda n: str(n)) if start_candidates else None
    goal = min(goal_candidates, key=lambda n: str(n)) if goal_candidates else None
    return start, goal


def _safe_shortest_path(graph: nx.Graph, start: Any, goal: Any) -> Tuple[bool, int, List[Any]]:
    """Best-effort shortest path helper that never throws."""
    if start not in graph or goal not in graph:
        return False, 0, []
    try:
        if not nx.has_path(graph, start, goal):
            return False, 0, []
        path = nx.shortest_path(graph, start, goal)
        return True, max(0, len(path) - 1), list(path)
    except Exception:
        return False, 0, []


def _path_metrics(
    G: nx.Graph,
    start: Optional[Any],
    goal: Optional[Any],
) -> Tuple[bool, int, float, bool, int, bool, int, float]:
    """
    Path metrics with directional semantics.

    Returns:
        (
            path_exists, path_length, linearity,             # primary (directed for DiGraph, weak for Graph)
            directed_path_exists, directed_path_length,      # strict directed reachability
            weak_path_exists, weak_path_length,              # undirected reachability
            directionality_gap,                              # 0..1 mismatch penalty between weak and directed paths
        )
    """
    if start is None or goal is None:
        return False, 0, 0.0, False, 0, False, 0, 0.0

    weak_graph = G.to_undirected()
    weak_exists, weak_len, weak_path = _safe_shortest_path(weak_graph, start, goal)

    if isinstance(G, nx.DiGraph):
        directed_exists, directed_len, directed_path = _safe_shortest_path(G, start, goal)
        path_exists = bool(directed_exists)
        path_length = int(directed_len if directed_exists else 0)
        path_nodes = directed_path if directed_exists else []
    else:
        directed_exists, directed_len = weak_exists, weak_len
        path_exists = bool(weak_exists)
        path_length = int(weak_len if weak_exists else 0)
        path_nodes = weak_path if weak_exists else []

    linearity = _clip01((len(path_nodes) / max(1, G.number_of_nodes())) if path_nodes else 0.0)

    # Penalty that quantifies how much undirected evaluation would overestimate
    # progression when edge directions/gates are respected.
    directionality_gap = 0.0
    if isinstance(G, nx.DiGraph) and weak_exists:
        if not directed_exists:
            directionality_gap = 1.0
        else:
            directionality_gap = _clip01(
                max(0.0, float(directed_len) - float(weak_len)) / max(1.0, float(weak_len))
            )

    return (
        bool(path_exists),
        int(path_length),
        float(linearity),
        bool(directed_exists),
        int(directed_len),
        bool(weak_exists),
        int(weak_len),
        float(directionality_gap),
    )


def _primary_path_nodes(G: nx.Graph, start: Optional[Any], goal: Optional[Any]) -> List[Any]:
    """Get the primary start-goal path honoring directionality for DiGraphs."""
    if start is None or goal is None:
        return []
    if start not in G or goal not in G:
        return []

    try:
        if isinstance(G, nx.DiGraph):
            if nx.has_path(G, start, goal):
                return list(nx.shortest_path(G, start, goal))
            return []
        U = G.to_undirected()
        if nx.has_path(U, start, goal):
            return list(nx.shortest_path(U, start, goal))
    except Exception:
        return []
    return []


def _step_constraints(G: nx.Graph, u: Any, v: Any) -> List[str]:
    """Collect normalized constraint tokens for a path step edge."""
    attrs = G.get_edge_data(u, v, default=None)
    if attrs is None and not G.is_directed():
        attrs = G.get_edge_data(v, u, default=None)
    if attrs is None:
        return []

    # MultiGraph / MultiDiGraph style (not common in this repo, but safe).
    if isinstance(attrs, dict) and attrs and all(isinstance(val, dict) for val in attrs.values()):
        out: List[str] = []
        for edge_attrs in attrs.values():
            out.extend(_edge_constraints(edge_attrs))
        return out

    if isinstance(attrs, dict):
        return _edge_constraints(attrs)
    return []


def _mission_constraints_valid(G: nx.Graph, grammar: MissionGrammar) -> bool:
    """
    Validate lock/key + progression constraints for generated graph representations.
    """
    try:
        mission_graph = networkx_to_mission_graph(G)
    except Exception:
        return False

    try:
        mission_graph.sanitize()
        return bool(
            grammar.validate_lock_key_ordering(mission_graph)
            and grammar.validate_progression_constraints(mission_graph)
        )
    except Exception:
        return False


def extract_graph_descriptor(
    graph: nx.Graph,
    grammar: Optional[MissionGrammar] = None,
) -> GraphDescriptor:
    """
    Extract completeness + quality descriptor from a mission graph.
    """
    n_nodes = int(graph.number_of_nodes())
    n_edges = int(graph.number_of_edges())
    U = graph.to_undirected()
    connected = bool(nx.is_connected(U)) if n_nodes > 0 else False

    start, goal = _select_start_goal_nodes(graph)
    has_start = start is not None
    has_goal = goal is not None
    (
        path_exists,
        path_length,
        linearity,
        directed_path_exists,
        directed_path_length,
        weak_path_exists,
        weak_path_length,
        directionality_gap,
    ) = _path_metrics(graph, start, goal)

    key_count = 0
    enemy_count = 0
    puzzle_count = 0
    item_count = 0
    for _, attrs in graph.nodes(data=True):
        key_hint = _count_hint(attrs, ("key_count_hint", "key_count"))
        enemy_hint = _count_hint(attrs, ("enemy_count_hint", "enemy_count"))
        puzzle_hint = _count_hint(attrs, ("puzzle_count_hint", "puzzle_count"))
        item_hint = _count_hint(attrs, ("item_count_hint", "item_count"))
        if _is_key_node(attrs):
            key_count += max(1, key_hint)
        else:
            key_count += key_hint
        if _is_enemy_node(attrs):
            enemy_count += max(1, enemy_hint)
        else:
            enemy_count += enemy_hint
        if _is_puzzle_node(attrs):
            puzzle_count += max(1, puzzle_hint)
        else:
            puzzle_count += puzzle_hint
        if _is_item_node(attrs):
            item_count += max(1, item_hint)
        else:
            item_count += item_hint

    lock_count = 0
    bombable_count = 0
    softlock_count = 0
    item_gate_count = 0
    switch_count = 0
    stair_count = 0
    shortcut_count = 0
    for _, _, attrs in graph.edges(data=True):
        constraints = set(_edge_constraints(attrs))
        if constraints.intersection(LOCK_EDGE_TYPES):
            lock_count += 1
        if constraints.intersection(BOMBABLE_EDGE_TYPES):
            bombable_count += 1
        if constraints.intersection(SOFT_LOCK_EDGE_TYPES):
            softlock_count += 1
        if constraints.intersection(ITEM_GATE_EDGE_TYPES):
            item_gate_count += 1
        if constraints.intersection(SWITCH_EDGE_TYPES):
            switch_count += 1
        if constraints.intersection(STAIR_EDGE_TYPES):
            stair_count += 1
        metadata = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
        try:
            path_savings = int(max(0, int(attrs.get("path_savings", metadata.get("path_savings", 0)))))
        except Exception:
            path_savings = 0
        if constraints.intersection(SHORTCUT_EDGE_TYPES) or path_savings >= 2:
            shortcut_count += 1

    leniency = _clip01(key_count / max(1, lock_count)) if lock_count > 0 else 1.0

    if isinstance(graph, nx.DiGraph):
        branching_nodes = sum(1 for n in graph.nodes() if graph.out_degree(n) >= 2)
    else:
        branching_nodes = sum(1 for n in graph.nodes() if graph.degree(n) >= 3)
    branching_factor = _clip01(branching_nodes / max(1, n_nodes))

    cycle_rank = max(0, U.number_of_edges() - U.number_of_nodes() + nx.number_connected_components(U)) if n_nodes > 0 else 0
    cycle_density = _clip01(cycle_rank / max(1, n_nodes // 2))
    shortcut_density = _clip01(shortcut_count / max(1, n_edges))
    gating_density = _clip01(lock_count / max(1, n_edges))
    path_depth_ratio = _clip01(path_length / max(1, n_nodes - 1))

    path_nodes = _primary_path_nodes(graph, start, goal)
    gate_edges_on_path = 0
    if path_nodes and len(path_nodes) >= 2:
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            cset = set(_step_constraints(graph, u, v))
            if cset.intersection(LOCK_EDGE_TYPES):
                gate_edges_on_path += 1
    gate_depth_ratio = _clip01(gate_edges_on_path / max(1, len(path_nodes) - 1))

    puzzle_density = _clip01(puzzle_count / max(1, n_nodes))
    item_density = _clip01(item_count / max(1, n_nodes))
    bombable_ratio = _clip01(bombable_count / max(1, n_edges))
    softlock_ratio = _clip01(softlock_count / max(1, n_edges))
    switch_ratio = _clip01(switch_count / max(1, n_edges))
    stair_ratio = _clip01(stair_count / max(1, n_edges))
    item_gate_ratio = _clip01(item_gate_count / max(1, n_edges))
    gate_variety = _clip01(
        sum(1 for c in [bombable_count, softlock_count, item_gate_count, switch_count, stair_count] if c > 0) / 5.0
    )

    lock_pressure = min(1.0, lock_count / max(1.0, float(max(1, key_count))))
    path_pressure = _clip01(path_length / max(1.0, float(n_nodes)))
    backtracking_proxy = _clip01((1.0 - linearity) * 0.65 + cycle_density * 0.35)
    feature_complexity = _clip01(
        0.35 * puzzle_density
        + 0.20 * item_density
        + 0.15 * bombable_ratio
        + 0.15 * softlock_ratio
        + 0.15 * gate_variety
    )

    progression_complexity = _clip01(
        0.28 * lock_pressure
        + 0.18 * backtracking_proxy
        + 0.14 * path_pressure
        + 0.12 * item_gate_ratio
        + 0.12 * feature_complexity
        + 0.08 * gate_depth_ratio
        + 0.08 * path_depth_ratio
    )
    topology_complexity = _clip01(
        0.28 * branching_factor
        + 0.26 * cycle_density
        + 0.16 * gating_density
        + 0.12 * gate_variety
        + 0.12 * shortcut_density
        + 0.03 * switch_ratio
        + 0.03 * stair_ratio
    )

    constraint_valid = False
    if grammar is not None:
        constraint_valid = _mission_constraints_valid(graph, grammar)
    repair_stats = _extract_generation_stats(graph)

    return GraphDescriptor(
        num_nodes=n_nodes,
        num_edges=n_edges,
        has_start=has_start,
        has_goal=has_goal,
        connected=connected,
        path_exists=path_exists,
        path_length=path_length,
        directed_path_exists=directed_path_exists,
        directed_path_length=directed_path_length,
        weak_path_exists=weak_path_exists,
        weak_path_length=weak_path_length,
        directionality_gap=directionality_gap,
        key_count=key_count,
        lock_count=lock_count,
        enemy_count=enemy_count,
        puzzle_count=puzzle_count,
        item_count=item_count,
        bombable_count=bombable_count,
        softlock_count=softlock_count,
        item_gate_count=item_gate_count,
        switch_count=switch_count,
        stair_count=stair_count,
        gate_variety=gate_variety,
        linearity=linearity,
        leniency=leniency,
        branching_factor=branching_factor,
        cycle_density=cycle_density,
        shortcut_density=shortcut_density,
        gating_density=gating_density,
        gate_depth_ratio=gate_depth_ratio,
        path_depth_ratio=path_depth_ratio,
        progression_complexity=progression_complexity,
        topology_complexity=topology_complexity,
        constraint_valid=constraint_valid,
        repair_applied=bool(repair_stats["repair_applied"]),
        total_repairs=int(repair_stats["total_repairs"]),
        lock_key_repairs=int(repair_stats["lock_key_repairs"]),
        progression_repairs=int(repair_stats["progression_repairs"]),
        wave3_repairs=int(repair_stats["wave3_repairs"]),
        repair_rounds=int(repair_stats["repair_rounds"]),
        generation_constraint_rejections=int(repair_stats["generation_constraint_rejections"]),
        candidate_repairs_applied=int(repair_stats["candidate_repairs_applied"]),
        rule_applications=int(repair_stats["rule_applications"]),
        rule_applied=int(repair_stats["rule_applied"]),
        rule_skipped=int(repair_stats["rule_skipped"]),
    )


def _descriptor_vector(d: GraphDescriptor) -> np.ndarray:
    return np.array(
        [
            d.linearity,
            d.leniency,
            d.progression_complexity,
            d.topology_complexity,
        ],
        dtype=np.float32,
    )


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _coverage_2d(points: np.ndarray, bins: int = 20) -> float:
    if points.size == 0:
        return 0.0
    x = np.clip(points[:, 0], 0.0, 1.0)
    y = np.clip(points[:, 1], 0.0, 1.0)
    H, _, _ = np.histogram2d(x, y, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]])
    return float(np.sum(H > 0) / (bins * bins))


def _overlap_2d(points_a: np.ndarray, points_b: np.ndarray, bins: int = 20) -> float:
    if points_a.size == 0 or points_b.size == 0:
        return 0.0
    Ha, _, _ = np.histogram2d(
        np.clip(points_a[:, 0], 0.0, 1.0),
        np.clip(points_a[:, 1], 0.0, 1.0),
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    Hb, _, _ = np.histogram2d(
        np.clip(points_b[:, 0], 0.0, 1.0),
        np.clip(points_b[:, 1], 0.0, 1.0),
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    occ_a = Ha > 0
    occ_b = Hb > 0
    union = np.sum(np.logical_or(occ_a, occ_b))
    if union == 0:
        return 0.0
    return float(np.sum(np.logical_and(occ_a, occ_b)) / union)


def _descriptor_summary(descriptors: Sequence[GraphDescriptor]) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not descriptors:
        empty = {
            "linearity": 0.0,
            "leniency": 0.0,
            "progression_complexity": 0.0,
            "topology_complexity": 0.0,
            "path_length": 0.0,
            "directed_path_length": 0.0,
            "weak_path_length": 0.0,
            "directionality_gap": 0.0,
            "directed_path_exists_rate": 0.0,
            "weak_path_exists_rate": 0.0,
            "branching_factor": 0.0,
            "cycle_density": 0.0,
            "shortcut_density": 0.0,
            "gating_density": 0.0,
            "gate_depth_ratio": 0.0,
            "path_depth_ratio": 0.0,
            "num_nodes": 0.0,
            "num_edges": 0.0,
            "enemy_count": 0.0,
            "key_count": 0.0,
            "puzzle_count": 0.0,
            "item_count": 0.0,
            "bombable_count": 0.0,
            "softlock_count": 0.0,
            "item_gate_count": 0.0,
            "switch_count": 0.0,
            "stair_count": 0.0,
            "gate_variety": 0.0,
            "repair_rate": 0.0,
            "total_repairs": 0.0,
            "generation_constraint_rejections": 0.0,
            "candidate_repairs_applied": 0.0,
            "rule_applications": 0.0,
            "rule_applied": 0.0,
            "rule_skipped": 0.0,
        }
        return dict(empty), dict(empty)

    arr = np.array(
        [
            [
                d.linearity,
                d.leniency,
                d.progression_complexity,
                d.topology_complexity,
                d.path_length,
                d.directed_path_length,
                d.weak_path_length,
                d.directionality_gap,
                float(d.directed_path_exists),
                float(d.weak_path_exists),
                d.branching_factor,
                d.cycle_density,
                d.shortcut_density,
                d.gating_density,
                d.gate_depth_ratio,
                d.path_depth_ratio,
                d.num_nodes,
                d.num_edges,
                d.enemy_count,
                d.key_count,
                d.puzzle_count,
                d.item_count,
                d.bombable_count,
                d.softlock_count,
                d.item_gate_count,
                d.switch_count,
                d.stair_count,
                d.gate_variety,
                float(d.repair_applied),
                d.total_repairs,
                d.lock_key_repairs,
                d.progression_repairs,
                d.wave3_repairs,
                d.repair_rounds,
                d.generation_constraint_rejections,
                d.candidate_repairs_applied,
                d.rule_applications,
                d.rule_applied,
                d.rule_skipped,
            ]
            for d in descriptors
        ],
        dtype=np.float64,
    )
    keys = [
        "linearity",
        "leniency",
        "progression_complexity",
        "topology_complexity",
        "path_length",
        "directed_path_length",
        "weak_path_length",
        "directionality_gap",
        "directed_path_exists_rate",
        "weak_path_exists_rate",
        "branching_factor",
        "cycle_density",
        "shortcut_density",
        "gating_density",
        "gate_depth_ratio",
        "path_depth_ratio",
        "num_nodes",
        "num_edges",
        "enemy_count",
        "key_count",
        "puzzle_count",
        "item_count",
        "bombable_count",
        "softlock_count",
        "item_gate_count",
        "switch_count",
        "stair_count",
        "gate_variety",
        "repair_rate",
        "total_repairs",
        "lock_key_repairs",
        "progression_repairs",
        "wave3_repairs",
        "repair_rounds",
        "generation_constraint_rejections",
        "candidate_repairs_applied",
        "rule_applications",
        "rule_applied",
        "rule_skipped",
    ]
    means = {k: float(np.mean(arr[:, i])) for i, k in enumerate(keys)}
    stds = {k: float(np.std(arr[:, i])) for i, k in enumerate(keys)}
    return means, stds


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 400,
    alpha: float = 0.05,
    seed: int = 12345,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for a sample mean.

    Returns dict with keys: mean, low, high.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}

    mean = float(np.mean(arr))
    if arr.size == 1 or n_bootstrap <= 0:
        return {"mean": mean, "low": mean, "high": mean}

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(int(n_bootstrap), arr.size))
    boot_means = np.mean(arr[indices], axis=1)
    low = float(np.quantile(boot_means, alpha / 2.0))
    high = float(np.quantile(boot_means, 1.0 - (alpha / 2.0)))
    return {"mean": mean, "low": low, "high": high}


def _target_curve_for_rooms(rng: np.random.Generator, n_rooms: int) -> List[float]:
    """
    Sample a smooth target curve with rising macro-trend and local undulation.
    """
    if n_rooms <= 1:
        return [0.5]
    xs = np.linspace(0.0, 1.0, n_rooms)
    slope = rng.uniform(0.45, 0.95)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    wave = 0.15 * np.sin((2.0 * np.pi * xs) + phase)
    curve = np.clip(0.15 + slope * xs + wave, 0.0, 1.0)
    return [float(v) for v in curve]


def generate_block_i_graphs(
    num_samples: int,
    seed: int = 42,
    min_rooms: int = 8,
    max_rooms: int = 16,
    population_size: int = 32,
    generations: int = 40,
    rule_space: str = "full",
    rule_weight_overrides: Optional[Dict[str, float]] = None,
    descriptor_targets: Optional[Dict[str, float]] = None,
    room_count_bias: float = 0.0,
    search_strategy: str = "ga",
    qd_archive_cells: int = 128,
    qd_init_random_fraction: float = 0.35,
    qd_emitter_mutation_rate: float = 0.18,
    realism_tuning: Optional[Dict[str, float]] = None,
) -> Tuple[List[nx.Graph], List[float]]:
    """
    Generate mission graphs with Block I evolutionary search.
    """
    rng = np.random.default_rng(seed)
    graphs: List[nx.Graph] = []
    times: List[float] = []

    for i in range(num_samples):
        if max_rooms <= min_rooms:
            room_count = int(max(1, min_rooms))
        else:
            bias = float(np.clip(room_count_bias, 0.0, 1.0))
            if bias <= 1e-9:
                room_count = int(rng.integers(min_rooms, max_rooms + 1))
            else:
                # Right-skewed sampling to better cover richer topologies when requested.
                u = float(rng.beta(1.0 + (2.0 * bias), 1.0))
                room_count = int(min_rooms + round(u * float(max_rooms - min_rooms)))
                room_count = int(np.clip(room_count, min_rooms, max_rooms))
        curve = _target_curve_for_rooms(rng, room_count)
        t0 = time.time()
        generator = EvolutionaryTopologyGenerator(
            target_curve=curve,
            population_size=population_size,
            generations=generations,
            max_nodes=room_count,
            rule_space=rule_space,
            rule_weight_overrides=rule_weight_overrides,
            descriptor_targets=descriptor_targets,
            search_strategy=search_strategy,
            qd_archive_cells=qd_archive_cells,
            qd_init_random_fraction=qd_init_random_fraction,
            qd_emitter_mutation_rate=qd_emitter_mutation_rate,
            realism_tuning=realism_tuning,
            seed=seed + i,
        )
        G = generator.evolve(directed_output=True)
        times.append(float(time.time() - t0))
        graphs.append(G)

    return graphs, times


def load_vglc_reference_graphs(
    data_root: Path,
    limit: Optional[int] = None,
    filter_virtual_start_pointers: bool = True,
    mark_physical_start: bool = True,
) -> List[nx.DiGraph]:
    """
    Load reference topology graphs from VGLC-style DOT files.
    """
    parser = DOTParser()
    dot_files = sorted(data_root.rglob("*.dot"))
    if limit is not None:
        dot_files = dot_files[: max(0, int(limit))]

    graphs: List[nx.DiGraph] = []
    for dot_path in dot_files:
        try:
            G = parser.parse(str(dot_path))
            if filter_virtual_start_pointers:
                physical_start = get_physical_start_node(G) if mark_physical_start else None
                G = filter_virtual_nodes(G)
                if mark_physical_start and physical_start in G.nodes:
                    G.nodes[physical_start]["is_start"] = True
                    G.nodes[physical_start]["is_entry"] = True
            graphs.append(G)
        except Exception as e:
            logger.debug("Skipping unreadable DOT file %s: %s", dot_path, e)

    return graphs


def _is_vglc_room_slot(slot: np.ndarray) -> bool:
    """
    Fast VGLC room detector for 16x11 slot extraction.
    """
    if slot.size == 0:
        return False
    gap_count = int(np.sum(slot == "-"))
    total = int(slot.size)
    if gap_count > int(0.7 * total):
        return False
    wall_count = int(np.sum((slot == "W") | (slot == "w")))
    door_count = int(np.sum((slot == "D") | (slot == "d")))
    if door_count > 0 and wall_count >= 20:
        return True
    interior_count = total - wall_count - gap_count
    return bool(wall_count >= 20 and interior_count >= 5)


def load_vglc_reference_rooms(
    data_root: Path,
    max_rooms: int = 96,
) -> List[np.ndarray]:
    """
    Load semantic room grids from VGLC text files for WFC robustness probes.
    """
    txt_files = sorted(data_root.rglob("*.txt"))
    rooms: List[np.ndarray] = []
    for txt_path in txt_files:
        try:
            lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if not lines:
                continue
            width = max(len(line) for line in lines)
            grid_chars = np.array(
                [list(line.ljust(width, "-")) for line in lines],
                dtype="<U1",
            )
            h, w = grid_chars.shape
            row_slots = h // ROOM_HEIGHT
            col_slots = w // ROOM_WIDTH
            for rs in range(row_slots):
                for cs in range(col_slots):
                    r0, r1 = rs * ROOM_HEIGHT, (rs + 1) * ROOM_HEIGHT
                    c0, c1 = cs * ROOM_WIDTH, (cs + 1) * ROOM_WIDTH
                    slot = grid_chars[r0:r1, c0:c1]
                    if slot.shape != (ROOM_HEIGHT, ROOM_WIDTH):
                        continue
                    if not _is_vglc_room_slot(slot):
                        continue
                    semantic = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
                    for r in range(ROOM_HEIGHT):
                        for c in range(ROOM_WIDTH):
                            ch = str(slot[r, c])
                            tile_id = CHAR_TO_SEMANTIC.get(ch)
                            if tile_id is None:
                                tile_id = CHAR_TO_SEMANTIC.get(ch.upper())
                            if tile_id is None:
                                tile_id = CHAR_TO_SEMANTIC.get(ch.lower())
                            semantic[r, c] = int(tile_id if tile_id is not None else int(TileID.VOID))
                    rooms.append(semantic)
                    if len(rooms) >= max(1, int(max_rooms)):
                        return rooms
        except Exception as e:
            logger.debug("Skipping unreadable VGLC txt file %s: %s", txt_path, e)
    return rooms


def audit_block0_dataset(
    data_root: Path,
    *,
    max_rooms: int = 5000,
    graph_limit: Optional[int] = None,
    filter_virtual_start_pointers: bool = True,
) -> Dict[str, Any]:
    """
    Build Block-0 data/process audit stats from local VGLC sources.

    Includes file counts, extracted room counts/shapes, tile distribution,
    and graph topology statistics. This gives reproducible evidence that
    benchmark inputs are grounded in the current repository data.
    """
    txt_files = sorted(data_root.rglob("*.txt"))
    dot_files = sorted(data_root.rglob("*.dot"))
    reference_graphs = load_vglc_reference_graphs(
        data_root,
        limit=graph_limit,
        filter_virtual_start_pointers=bool(filter_virtual_start_pointers),
    )
    reference_rooms = load_vglc_reference_rooms(data_root, max_rooms=max_rooms)

    room_shapes: Dict[str, int] = {}
    for room in reference_rooms:
        key = f"{int(room.shape[0])}x{int(room.shape[1])}"
        room_shapes[key] = room_shapes.get(key, 0) + 1

    node_counts = [int(g.number_of_nodes()) for g in reference_graphs]
    edge_counts = [int(g.number_of_edges()) for g in reference_graphs]

    def _summary(arr: Sequence[int]) -> Dict[str, float]:
        if not arr:
            return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        vals = np.asarray(arr, dtype=np.float64)
        return {
            "count": float(vals.size),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals)),
        }

    completeness_rates = {"has_start_rate": 0.0, "has_goal_rate": 0.0, "path_exists_rate": 0.0}
    if reference_graphs:
        desc = [extract_graph_descriptor(g, grammar=None) for g in reference_graphs]
        completeness_rates = {
            "has_start_rate": float(np.mean([float(d.has_start) for d in desc])),
            "has_goal_rate": float(np.mean([float(d.has_goal) for d in desc])),
            "path_exists_rate": float(np.mean([float(d.path_exists) for d in desc])),
        }

    tile_hist: Dict[str, int] = {}
    if reference_rooms:
        flat = np.concatenate([np.asarray(r, dtype=np.int32).ravel() for r in reference_rooms], axis=0)
        ids, freqs = np.unique(flat, return_counts=True)
        for tid, cnt in zip(ids.tolist(), freqs.tolist()):
            tile_hist[str(int(tid))] = int(cnt)

    return {
        "data_root": str(data_root),
        "txt_file_count": int(len(txt_files)),
        "dot_file_count": int(len(dot_files)),
        "reference_room_count": int(len(reference_rooms)),
        "reference_graph_count": int(len(reference_graphs)),
        "room_shape_histogram": room_shapes,
        "graph_nodes": _summary(node_counts),
        "graph_edges": _summary(edge_counts),
        "graph_completeness": completeness_rates,
        "tile_histogram": tile_hist,
    }


def run_wfc_robustness_probe(
    room_grids: Sequence[np.ndarray],
    seed: int = 12345,
    max_samples: int = 64,
    mask_ratios: Optional[Sequence[float]] = None,
    preserve_critical_tiles: bool = True,
) -> List[Dict[str, Any]]:
    """
    Probe Weighted Bayesian WFC robustness on VGLC room corpus.
    """
    if not room_grids:
        return []
    samples = [np.asarray(g, dtype=np.int32) for g in room_grids[: max(1, int(max_samples))]]
    if mask_ratios is None:
        mask_cycle = [float(v) for v in DEFAULT_WFC_PROBE_MASK_RATIOS]
    else:
        mask_cycle = [float(v) for v in mask_ratios if np.isfinite(v) and 0.0 <= float(v) < 1.0]
    if not mask_cycle:
        mask_cycle = [float(v) for v in DEFAULT_WFC_PROBE_MASK_RATIOS]

    rng = np.random.default_rng(seed + 17)

    critical_tile_ids = {
        int(TileID.START),
        int(TileID.TRIFORCE),
        int(TileID.DOOR_LOCKED),
        int(TileID.DOOR_BOSS),
        int(TileID.KEY_SMALL),
    }

    def _apply_partial_mask(room: np.ndarray, ratio: float) -> Tuple[np.ndarray, int]:
        arr = np.asarray(room, dtype=np.int32).copy()
        h, w = arr.shape
        maskable = np.zeros((h, w), dtype=bool)
        if h > 2 and w > 2:
            maskable[1:-1, 1:-1] = True
        else:
            maskable[:, :] = True
        if preserve_critical_tiles:
            critical_mask = np.isin(arr, list(critical_tile_ids))
            maskable &= ~critical_mask
        candidate_idx = np.flatnonzero(maskable.ravel())
        if candidate_idx.size == 0:
            return arr, 0
        target = int(round(float(np.clip(ratio, 0.0, 0.95)) * float(candidate_idx.size)))
        if target <= 0:
            return arr, 0
        chosen = rng.choice(candidate_idx, size=min(target, candidate_idx.size), replace=False)
        rows, cols = np.unravel_index(chosen, arr.shape)
        arr[rows, cols] = -1  # Unknown tile sentinel (not seeded in WFC priors).
        return arr, int(chosen.size)

    tile_priors = extract_tile_priors_from_vqvae(
        vqvae_codebook=np.zeros((1, 1), dtype=np.float32),
        training_grids=samples,
    )
    wfc_cfg = WeightedBayesianWFCConfig(
        use_vqvae_priors=True,
        enable_backtracking=True,
        max_backtracks=256,
        max_restarts=3,
        kl_divergence_threshold=2.5,
    )
    out: List[Dict[str, Any]] = []
    for i, room in enumerate(samples):
        mask_ratio_target = float(mask_cycle[i % len(mask_cycle)])
        masked_room, masked_count = _apply_partial_mask(room, ratio=mask_ratio_target)
        known_count = int(masked_room.size - masked_count)
        applied_ratio = float(masked_count / max(1, masked_room.size))
        try:
            result = integrate_weighted_wfc_into_pipeline(
                neural_room=masked_room,
                tile_priors=tile_priors,
                seed=seed + i,
                config=wfc_cfg,
            )
            diag = dict(result.get("wfc_diagnostics", {}))
            diag["kl_divergence"] = float(result.get("kl_divergence", 0.0))
            diag["distribution_preserved"] = bool(result.get("distribution_preserved", False))
            diag["mask_ratio_target"] = float(mask_ratio_target)
            diag["mask_ratio_applied"] = float(applied_ratio)
            diag["masked_tiles"] = int(masked_count)
            diag["known_tiles"] = int(known_count)
            out.append(diag)
        except Exception as e:
            logger.debug("WFC probe failed on sample %d: %s", i, e)
            out.append(
                {
                    "generation_succeeded": False,
                    "contradictions": 0,
                    "backtracks": 0,
                    "restarts": 0,
                    "zero_prob_resets": 0,
                    "fallback_fills": 0,
                    "required_fallback": True,
                    "kl_divergence": float("nan"),
                    "distribution_preserved": False,
                    "mask_ratio_target": float(mask_ratio_target),
                    "mask_ratio_applied": float(applied_ratio),
                    "masked_tiles": int(masked_count),
                    "known_tiles": int(known_count),
                }
            )
    return out


RULE_GROUPS: Dict[str, List[str]] = {
    "progression": [
        "InsertLockKey",
        "AddItemGate",
        "AddFungibleLock",
        "AddMultiLock",
        "AddGatekeeper",
        "AddHazardGate",
        "AddBossGauntlet",
    ],
    "topology": [
        "Branch",
        "MergeShortcut",
        "CreateHub",
        "AddEntangledBranches",
        "SplitRoom",
        "AddValve",
        "AddTeleport",
        "AddStairs",
        "AddSector",
        "FormBigRoom",
    ],
    "challenge": [
        "InsertChallenge_ENEMY",
        "InsertChallenge_PUZZLE",
        "AddGatekeeper",
        "AddBossGauntlet",
    ],
    "relief": [
        "AddPacingBreaker",
        "AddResourceLoop",
        "AddItemShortcut",
        "AddSecret",
        "AddCollectionChallenge",
    ],
    "linearity_up": [
        "InsertChallenge_ENEMY",
        "InsertChallenge_PUZZLE",
        "PruneDeadEnd",
        "PruneGraph",
    ],
    "linearity_down": [
        "Branch",
        "MergeShortcut",
        "CreateHub",
        "AddEntangledBranches",
        "AddTeleport",
    ],
    "expansion_up": [
        "Branch",
        "CreateHub",
        "AddEntangledBranches",
        "SplitRoom",
        "AddSector",
        "FormBigRoom",
        "AddBossGauntlet",
        "AddSkillChain",
        "AddSecret",
    ],
    "expansion_down": [
        "PruneDeadEnd",
        "PruneGraph",
        "MergeShortcut",
    ],
    "puzzle_features": [
        "InsertChallenge_PUZZLE",
        "AddSkillChain",
        "AddPacingBreaker",
    ],
    "item_features": [
        "AddItemGate",
        "AddItemShortcut",
        "AddResourceLoop",
        "AddCollectionChallenge",
    ],
    "softlock_features": [
        "AddValve",
        "AddHazardGate",
        "AddGatekeeper",
    ],
    "switch_features": [
        "AddMultiLock",
    ],
    "stair_features": [
        "AddStairs",
        "AddTeleport",
    ],
}


def _descriptor_gap_score(
    generated: Dict[str, float],
    target: Dict[str, float],
) -> float:
    keys = [
        ("linearity", 0.12),
        ("leniency", 0.12),
        ("progression_complexity", 0.20),
        ("topology_complexity", 0.20),
        ("cycle_density", 0.06),
        ("shortcut_density", 0.06),
        ("gate_depth_ratio", 0.06),
        ("path_depth_ratio", 0.04),
        ("gating_density", 0.05),
        ("puzzle_count", 0.07),
        ("item_count", 0.06),
        ("bombable_count", 0.05),
        ("softlock_count", 0.05),
        ("item_gate_count", 0.05),
        ("switch_count", 0.03),
        ("stair_count", 0.02),
        ("gate_variety", 0.03),
    ]
    score = 0.0
    for key, weight in keys:
        score += float(weight) * abs(float(generated.get(key, 0.0)) - float(target.get(key, 0.0)))
    path_target = max(1.0, float(target.get("path_length", 0.0)))
    node_target = max(1.0, float(target.get("num_nodes", 0.0)))
    edge_target = max(1.0, float(target.get("num_edges", 0.0)))
    score += 0.10 * abs(float(generated.get("path_length", 0.0)) - float(target.get("path_length", 0.0))) / path_target
    score += 0.08 * abs(float(generated.get("num_nodes", 0.0)) - float(target.get("num_nodes", 0.0))) / node_target
    score += 0.06 * abs(float(generated.get("num_edges", 0.0)) - float(target.get("num_edges", 0.0))) / edge_target
    return float(score)


def _scale_rule_group(
    weights: Dict[str, float],
    rule_names: Sequence[str],
    factor: float,
) -> None:
    f = float(np.clip(float(factor), 0.6, 1.6))
    for name in rule_names:
        if name in weights:
            weights[name] = float(weights[name]) * f


def _normalize_weight_scale(
    weights: Dict[str, float],
    baseline_mean: float,
    min_w: float = 0.05,
    max_w: float = 4.0,
) -> Dict[str, float]:
    clipped = {k: float(np.clip(v, min_w, max_w)) for k, v in weights.items()}
    if not clipped:
        return clipped
    cur_mean = float(np.mean(list(clipped.values())))
    if cur_mean <= 0.0:
        return clipped
    scale = baseline_mean / cur_mean
    return {k: float(np.clip(v * scale, min_w, max_w)) for k, v in clipped.items()}


def calibrate_rule_weights_to_vglc(
    reference_graphs: Sequence[nx.Graph],
    *,
    seed: int = 42,
    iterations: int = 5,
    sample_size: int = 12,
    population_size: int = 24,
    generations: int = 24,
    min_rooms: int = 8,
    max_rooms: int = 16,
) -> Dict[str, Any]:
    """
    Calibrate grammar rule scheduling toward VGLC descriptor targets.
    """
    if not reference_graphs:
        return {
            "calibrated_rule_weights": {},
            "target_descriptor_means": {},
            "history": [],
            "best_gap": 0.0,
        }

    ref_desc = [extract_graph_descriptor(g, grammar=None) for g in reference_graphs]
    target_means, _ = _descriptor_summary(ref_desc)

    base_grammar = MissionGrammar(seed=seed)
    weights = {
        str(rule.name): float(rule.weight)
        for rule in base_grammar.rules
        if str(rule.name) != "Start"
    }
    baseline_mean = float(np.mean(list(weights.values()))) if weights else 1.0
    best_weights = dict(weights)
    best_gap = float("inf")
    history: List[Dict[str, Any]] = []

    for step in range(max(1, int(iterations))):
        graphs, _ = generate_block_i_graphs(
            num_samples=max(2, int(sample_size)),
            seed=seed + (step * 97),
            min_rooms=min_rooms,
            max_rooms=max_rooms,
            population_size=population_size,
            generations=generations,
            rule_space="full",
            rule_weight_overrides=weights,
            descriptor_targets={
                "linearity": float(target_means.get("linearity", 0.45)),
                "leniency": float(target_means.get("leniency", 0.50)),
                "progression_complexity": float(target_means.get("progression_complexity", 0.65)),
                "topology_complexity": float(target_means.get("topology_complexity", 0.45)),
                "path_length": float(target_means.get("path_length", 8.0)),
                "num_nodes": float(target_means.get("num_nodes", 12.0)),
                "num_edges": float(target_means.get("num_edges", 24.0)),
                "cycle_density": float(target_means.get("cycle_density", 0.30)),
                "shortcut_density": float(target_means.get("shortcut_density", 0.10)),
                "gate_depth_ratio": float(target_means.get("gate_depth_ratio", 0.25)),
                "path_depth_ratio": float(target_means.get("path_depth_ratio", 0.40)),
                "gating_density": float(target_means.get("gating_density", 0.16)),
                "puzzle_density": float(
                    target_means.get("puzzle_count", 0.0) / max(1.0, float(target_means.get("num_nodes", 12.0)))
                ),
                "item_density": float(
                    target_means.get("item_count", 0.0) / max(1.0, float(target_means.get("num_nodes", 12.0)))
                ),
                "gate_variety": float(target_means.get("gate_variety", 0.30)),
                "bombable_ratio": float(
                    target_means.get("bombable_count", 0.0) / max(1.0, float(target_means.get("num_edges", 24.0)))
                ),
                "soft_lock_ratio": float(
                    target_means.get("softlock_count", 0.0) / max(1.0, float(target_means.get("num_edges", 24.0)))
                ),
                "switch_ratio": float(
                    target_means.get("switch_count", 0.0) / max(1.0, float(target_means.get("num_edges", 24.0)))
                ),
                "stair_ratio": float(
                    target_means.get("stair_count", 0.0) / max(1.0, float(target_means.get("num_edges", 24.0)))
                ),
            },
        )
        desc = [extract_graph_descriptor(g, grammar=MissionGrammar(seed=seed + 500 + step)) for g in graphs]
        means, _ = _descriptor_summary(desc)
        constraint_valid_rate = float(np.mean([float(d.constraint_valid) for d in desc])) if desc else 0.0
        validity_gap = abs(1.0 - constraint_valid_rate)
        gap = _descriptor_gap_score(means, target_means) + (0.15 * validity_gap)
        errors = {
            "linearity": float(target_means.get("linearity", 0.0) - means.get("linearity", 0.0)),
            "leniency": float(target_means.get("leniency", 0.0) - means.get("leniency", 0.0)),
            "progression_complexity": float(
                target_means.get("progression_complexity", 0.0) - means.get("progression_complexity", 0.0)
            ),
            "topology_complexity": float(
                target_means.get("topology_complexity", 0.0) - means.get("topology_complexity", 0.0)
            ),
            "cycle_density": float(
                target_means.get("cycle_density", 0.0) - means.get("cycle_density", 0.0)
            ),
            "shortcut_density": float(
                target_means.get("shortcut_density", 0.0) - means.get("shortcut_density", 0.0)
            ),
            "gate_depth_ratio": float(
                target_means.get("gate_depth_ratio", 0.0) - means.get("gate_depth_ratio", 0.0)
            ),
            "path_depth_ratio": float(
                target_means.get("path_depth_ratio", 0.0) - means.get("path_depth_ratio", 0.0)
            ),
            "path_length_ratio": float(
                (target_means.get("path_length", 0.0) - means.get("path_length", 0.0))
                / max(1.0, float(target_means.get("path_length", 0.0)))
            ),
            "num_nodes_ratio": float(
                (target_means.get("num_nodes", 0.0) - means.get("num_nodes", 0.0))
                / max(1.0, float(target_means.get("num_nodes", 0.0)))
            ),
            "num_edges_ratio": float(
                (target_means.get("num_edges", 0.0) - means.get("num_edges", 0.0))
                / max(1.0, float(target_means.get("num_edges", 0.0)))
            ),
            "puzzle_count_ratio": float(
                (target_means.get("puzzle_count", 0.0) - means.get("puzzle_count", 0.0))
                / max(1.0, float(target_means.get("puzzle_count", 0.0)))
            ),
            "item_count_ratio": float(
                (target_means.get("item_count", 0.0) - means.get("item_count", 0.0))
                / max(1.0, float(target_means.get("item_count", 0.0)))
            ),
            "bombable_count_ratio": float(
                (target_means.get("bombable_count", 0.0) - means.get("bombable_count", 0.0))
                / max(1.0, float(target_means.get("bombable_count", 0.0)))
            ),
            "softlock_count_ratio": float(
                (target_means.get("softlock_count", 0.0) - means.get("softlock_count", 0.0))
                / max(1.0, float(target_means.get("softlock_count", 0.0)))
            ),
            "switch_count_ratio": float(
                (target_means.get("switch_count", 0.0) - means.get("switch_count", 0.0))
                / max(1.0, float(target_means.get("switch_count", 0.0)))
            ),
            "stair_count_ratio": float(
                (target_means.get("stair_count", 0.0) - means.get("stair_count", 0.0))
                / max(1.0, float(target_means.get("stair_count", 0.0)))
            ),
            "gating_density": float(target_means.get("gating_density", 0.0) - means.get("gating_density", 0.0)),
            "gate_variety": float(target_means.get("gate_variety", 0.0) - means.get("gate_variety", 0.0)),
            "constraint_valid_rate": float(1.0 - constraint_valid_rate),
        }
        history.append(
            {
                "iteration": int(step),
                "descriptor_gap": float(gap),
                "generated_descriptor_means": {
                    "linearity": float(means.get("linearity", 0.0)),
                    "leniency": float(means.get("leniency", 0.0)),
                    "progression_complexity": float(means.get("progression_complexity", 0.0)),
                    "topology_complexity": float(means.get("topology_complexity", 0.0)),
                    "path_length": float(means.get("path_length", 0.0)),
                    "num_nodes": float(means.get("num_nodes", 0.0)),
                    "num_edges": float(means.get("num_edges", 0.0)),
                    "cycle_density": float(means.get("cycle_density", 0.0)),
                    "shortcut_density": float(means.get("shortcut_density", 0.0)),
                    "gate_depth_ratio": float(means.get("gate_depth_ratio", 0.0)),
                    "path_depth_ratio": float(means.get("path_depth_ratio", 0.0)),
                    "gating_density": float(means.get("gating_density", 0.0)),
                    "puzzle_count": float(means.get("puzzle_count", 0.0)),
                    "item_count": float(means.get("item_count", 0.0)),
                    "bombable_count": float(means.get("bombable_count", 0.0)),
                    "softlock_count": float(means.get("softlock_count", 0.0)),
                    "item_gate_count": float(means.get("item_gate_count", 0.0)),
                    "switch_count": float(means.get("switch_count", 0.0)),
                    "stair_count": float(means.get("stair_count", 0.0)),
                    "gate_variety": float(means.get("gate_variety", 0.0)),
                    "constraint_valid_rate": float(constraint_valid_rate),
                },
                "errors": errors,
            }
        )

        if gap < best_gap:
            best_gap = float(gap)
            best_weights = dict(weights)

        e_prog = errors["progression_complexity"]
        e_topo = errors["topology_complexity"]
        e_cycle = errors["cycle_density"]
        e_shortcut = errors["shortcut_density"]
        e_gate_depth = errors["gate_depth_ratio"]
        e_path_depth = errors["path_depth_ratio"]
        e_lin = errors["linearity"]
        e_len = errors["leniency"]
        e_path = errors["path_length_ratio"]
        e_nodes = errors["num_nodes_ratio"]
        e_edges = errors["num_edges_ratio"]
        e_valid = errors["constraint_valid_rate"]
        e_puzzle = errors["puzzle_count_ratio"]
        e_item = errors["item_count_ratio"]
        e_bombable = errors["bombable_count_ratio"]
        e_softlock = errors["softlock_count_ratio"]
        e_switch = errors["switch_count_ratio"]
        e_stair = errors["stair_count_ratio"]
        e_gating_density = errors["gating_density"]
        e_gate_variety = errors["gate_variety"]

        _scale_rule_group(
            weights,
            RULE_GROUPS["progression"],
            1.0 + (0.55 * e_prog) + (0.20 * np.clip(e_gate_depth + e_path_depth, -1.0, 1.0)),
        )
        _scale_rule_group(
            weights,
            RULE_GROUPS["topology"],
            1.0 + (0.50 * e_topo) + (0.20 * np.clip(e_cycle + e_shortcut, -1.0, 1.0)),
        )

        if e_lin >= 0.0:
            _scale_rule_group(weights, RULE_GROUPS["linearity_up"], 1.0 + (0.45 * e_lin))
            _scale_rule_group(weights, RULE_GROUPS["linearity_down"], 1.0 - (0.45 * e_lin))
        else:
            lin_mag = abs(e_lin)
            _scale_rule_group(weights, RULE_GROUPS["linearity_up"], 1.0 - (0.45 * lin_mag))
            _scale_rule_group(weights, RULE_GROUPS["linearity_down"], 1.0 + (0.45 * lin_mag))

        if e_len >= 0.0:
            _scale_rule_group(weights, RULE_GROUPS["relief"], 1.0 + (0.60 * e_len))
            _scale_rule_group(weights, RULE_GROUPS["challenge"], 1.0 - (0.40 * e_len))
        else:
            len_mag = abs(e_len)
            _scale_rule_group(weights, RULE_GROUPS["relief"], 1.0 - (0.45 * len_mag))
            _scale_rule_group(weights, RULE_GROUPS["challenge"], 1.0 + (0.55 * len_mag))

        expansion_signal = float(np.clip((0.45 * e_nodes) + (0.30 * e_path) + (0.25 * e_edges), -1.0, 1.0))
        if expansion_signal >= 0.0:
            _scale_rule_group(weights, RULE_GROUPS["expansion_up"], 1.0 + (0.80 * expansion_signal))
            _scale_rule_group(weights, RULE_GROUPS["expansion_down"], 1.0 - (0.60 * expansion_signal))
        else:
            mag = abs(expansion_signal)
            _scale_rule_group(weights, RULE_GROUPS["expansion_up"], 1.0 - (0.55 * mag))
            _scale_rule_group(weights, RULE_GROUPS["expansion_down"], 1.0 + (0.70 * mag))

        if e_valid > 0.0:
            # When validity drops, temper expansion and reinforce progression-consistent rules.
            _scale_rule_group(weights, RULE_GROUPS["progression"], 1.0 + (0.55 * e_valid))
            _scale_rule_group(weights, RULE_GROUPS["expansion_up"], 1.0 - (0.45 * e_valid))
            _scale_rule_group(weights, RULE_GROUPS["expansion_down"], 1.0 + (0.45 * e_valid))

        _scale_rule_group(weights, RULE_GROUPS["puzzle_features"], 1.0 + (0.40 * np.clip(e_puzzle, -1.0, 1.0)))
        _scale_rule_group(weights, RULE_GROUPS["item_features"], 1.0 + (0.35 * np.clip(e_item, -1.0, 1.0)))
        _scale_rule_group(weights, RULE_GROUPS["softlock_features"], 1.0 + (0.30 * np.clip(e_softlock + e_bombable, -1.0, 1.0)))
        _scale_rule_group(weights, RULE_GROUPS["switch_features"], 1.0 + (0.30 * np.clip(e_switch, -1.0, 1.0)))
        _scale_rule_group(weights, RULE_GROUPS["stair_features"], 1.0 + (0.25 * np.clip(e_stair, -1.0, 1.0)))
        _scale_rule_group(weights, RULE_GROUPS["topology"], 1.0 + (0.20 * np.clip(e_cycle, -1.0, 1.0)))
        _scale_rule_group(weights, RULE_GROUPS["linearity_down"], 1.0 + (0.22 * np.clip(e_shortcut, -1.0, 1.0)))
        _scale_rule_group(weights, RULE_GROUPS["progression"], 1.0 + (0.16 * np.clip(e_gate_depth + e_path_depth, -1.0, 1.0)))
        _scale_rule_group(weights, RULE_GROUPS["progression"], 1.0 + (0.16 * np.clip(e_gating_density, -1.0, 1.0)))
        if e_gate_variety >= 0.0:
            _scale_rule_group(weights, RULE_GROUPS["topology"], 1.0 + (0.18 * e_gate_variety))
        else:
            _scale_rule_group(weights, RULE_GROUPS["topology"], 1.0 - (0.12 * abs(e_gate_variety)))

        weights = _normalize_weight_scale(weights, baseline_mean=baseline_mean)

    return {
        "calibrated_rule_weights": best_weights,
        "target_descriptor_means": {
            "linearity": float(target_means.get("linearity", 0.0)),
            "leniency": float(target_means.get("leniency", 0.0)),
            "progression_complexity": float(target_means.get("progression_complexity", 0.0)),
            "topology_complexity": float(target_means.get("topology_complexity", 0.0)),
            "path_length": float(target_means.get("path_length", 0.0)),
            "num_nodes": float(target_means.get("num_nodes", 0.0)),
            "num_edges": float(target_means.get("num_edges", 0.0)),
            "cycle_density": float(target_means.get("cycle_density", 0.0)),
            "shortcut_density": float(target_means.get("shortcut_density", 0.0)),
            "gate_depth_ratio": float(target_means.get("gate_depth_ratio", 0.0)),
            "path_depth_ratio": float(target_means.get("path_depth_ratio", 0.0)),
            "gating_density": float(target_means.get("gating_density", 0.0)),
            "puzzle_count": float(target_means.get("puzzle_count", 0.0)),
            "item_count": float(target_means.get("item_count", 0.0)),
            "bombable_count": float(target_means.get("bombable_count", 0.0)),
            "softlock_count": float(target_means.get("softlock_count", 0.0)),
            "item_gate_count": float(target_means.get("item_gate_count", 0.0)),
            "switch_count": float(target_means.get("switch_count", 0.0)),
            "stair_count": float(target_means.get("stair_count", 0.0)),
            "gate_variety": float(target_means.get("gate_variety", 0.0)),
        },
        "history": history,
        "best_gap": float(best_gap if np.isfinite(best_gap) else 0.0),
    }


def run_block_i_benchmark(
    generated_graphs: Sequence[nx.Graph],
    reference_graphs: Sequence[nx.Graph],
    generation_times: Optional[Sequence[float]] = None,
    wfc_probe_results: Optional[Sequence[Dict[str, Any]]] = None,
    bootstrap_samples: int = 400,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: int = 12345,
) -> BenchmarkSummary:
    """
    Compare generated graphs against references with completeness and QD metrics.
    """
    grammar = MissionGrammar(seed=1234)
    gen_desc = [extract_graph_descriptor(g, grammar=grammar) for g in generated_graphs]
    ref_desc = [extract_graph_descriptor(g, grammar=None) for g in reference_graphs]

    gen_vec = np.array([_descriptor_vector(d) for d in gen_desc], dtype=np.float64) if gen_desc else np.zeros((0, 4))
    ref_vec = np.array([_descriptor_vector(d) for d in ref_desc], dtype=np.float64) if ref_desc else np.zeros((0, 4))

    start_goal_series = [float(d.has_start and d.has_goal) for d in gen_desc]
    connected_series = [float(d.connected) for d in gen_desc]
    path_exists_series = [float(d.path_exists) for d in gen_desc]
    directed_path_exists_series = [float(d.directed_path_exists) for d in gen_desc]
    weak_path_exists_series = [float(d.weak_path_exists) for d in gen_desc]
    directionality_gap_series = [float(d.directionality_gap) for d in gen_desc]
    constraint_valid_series = [float(d.constraint_valid) for d in gen_desc]

    completeness = {
        "start_goal_rate": float(np.mean(start_goal_series)) if start_goal_series else 0.0,
        "connected_rate": float(np.mean(connected_series)) if connected_series else 0.0,
        "path_exists_rate": float(np.mean(path_exists_series)) if path_exists_series else 0.0,
        "directed_path_exists_rate": float(np.mean(directed_path_exists_series)) if directed_path_exists_series else 0.0,
        "weak_path_exists_rate": float(np.mean(weak_path_exists_series)) if weak_path_exists_series else 0.0,
        "path_directionality_gap_mean": float(np.mean(directionality_gap_series)) if directionality_gap_series else 0.0,
        "constraint_valid_rate": float(np.mean(constraint_valid_series)) if constraint_valid_series else 0.0,
    }
    completeness["overall_completeness"] = float(
        0.25 * completeness["start_goal_rate"]
        + 0.25 * completeness["connected_rate"]
        + 0.25 * completeness["path_exists_rate"]
        + 0.25 * completeness["constraint_valid_rate"]
    )
    overall_series = [
        0.25 * s + 0.25 * c + 0.25 * p + 0.25 * v
        for s, c, p, v in zip(
            start_goal_series,
            connected_series,
            path_exists_series,
            constraint_valid_series,
        )
    ]
    repair_applied_series = [float(d.repair_applied) for d in gen_desc]
    total_repairs_series = [float(d.total_repairs) for d in gen_desc]
    lock_key_repairs_series = [float(d.lock_key_repairs) for d in gen_desc]
    progression_repairs_series = [float(d.progression_repairs) for d in gen_desc]
    wave3_repairs_series = [float(d.wave3_repairs) for d in gen_desc]
    repair_rounds_series = [float(d.repair_rounds) for d in gen_desc]
    generation_constraint_rejections_series = [float(d.generation_constraint_rejections) for d in gen_desc]
    candidate_repairs_applied_series = [float(d.candidate_repairs_applied) for d in gen_desc]
    rule_applications_series = [float(d.rule_applications) for d in gen_desc]
    rule_applied_series = [float(d.rule_applied) for d in gen_desc]
    rule_skipped_series = [float(d.rule_skipped) for d in gen_desc]

    robustness: Dict[str, float] = {
        "repair_rate": float(np.mean(repair_applied_series)) if repair_applied_series else 0.0,
        "mean_total_repairs": float(np.mean(total_repairs_series)) if total_repairs_series else 0.0,
        "mean_lock_key_repairs": float(np.mean(lock_key_repairs_series)) if lock_key_repairs_series else 0.0,
        "mean_progression_repairs": float(np.mean(progression_repairs_series)) if progression_repairs_series else 0.0,
        "mean_wave3_repairs": float(np.mean(wave3_repairs_series)) if wave3_repairs_series else 0.0,
        "mean_repair_rounds": float(np.mean(repair_rounds_series)) if repair_rounds_series else 0.0,
        "mean_generation_constraint_rejections": float(np.mean(generation_constraint_rejections_series)) if generation_constraint_rejections_series else 0.0,
        "mean_candidate_repairs_applied": float(np.mean(candidate_repairs_applied_series)) if candidate_repairs_applied_series else 0.0,
        "mean_rule_applications": float(np.mean(rule_applications_series)) if rule_applications_series else 0.0,
        "mean_rule_applied": float(np.mean(rule_applied_series)) if rule_applied_series else 0.0,
        "mean_rule_skipped": float(np.mean(rule_skipped_series)) if rule_skipped_series else 0.0,
        "wfc_probe_count": 0.0,
        "wfc_contradiction_rate": 0.0,
        "wfc_restart_rate": 0.0,
        "wfc_fallback_rate": 0.0,
        "wfc_mean_contradictions": 0.0,
        "wfc_mean_backtracks": 0.0,
        "wfc_mean_restarts": 0.0,
        "wfc_mean_zero_prob_resets": 0.0,
        "wfc_mean_fallback_fills": 0.0,
        "wfc_mean_kl_divergence": 0.0,
        "wfc_distribution_preserved_rate": 0.0,
        "wfc_mean_mask_ratio": 0.0,
        "wfc_mean_masked_tiles": 0.0,
        "wfc_probe_count_severity_low": 0.0,
        "wfc_probe_count_severity_mid": 0.0,
        "wfc_probe_count_severity_high": 0.0,
        "wfc_restart_rate_severity_low": 0.0,
        "wfc_restart_rate_severity_mid": 0.0,
        "wfc_restart_rate_severity_high": 0.0,
        "wfc_fallback_rate_severity_low": 0.0,
        "wfc_fallback_rate_severity_mid": 0.0,
        "wfc_fallback_rate_severity_high": 0.0,
        "wfc_distribution_preserved_rate_severity_low": 0.0,
        "wfc_distribution_preserved_rate_severity_mid": 0.0,
        "wfc_distribution_preserved_rate_severity_high": 0.0,
        "wfc_mean_kl_divergence_severity_low": 0.0,
        "wfc_mean_kl_divergence_severity_mid": 0.0,
        "wfc_mean_kl_divergence_severity_high": 0.0,
    }

    if wfc_probe_results:
        contradictions = [float(max(0, int(r.get("contradictions", 0)))) for r in wfc_probe_results]
        backtracks = [float(max(0, int(r.get("backtracks", 0)))) for r in wfc_probe_results]
        restarts = [float(max(0, int(r.get("restarts", 0)))) for r in wfc_probe_results]
        resets = [float(max(0, int(r.get("zero_prob_resets", 0)))) for r in wfc_probe_results]
        fallback_fills = [float(max(0, int(r.get("fallback_fills", 0)))) for r in wfc_probe_results]
        fallback_flags = [float(bool(r.get("required_fallback", False))) for r in wfc_probe_results]
        kl_values = [float(r.get("kl_divergence", 0.0)) for r in wfc_probe_results]
        preserved_flags = [float(bool(r.get("distribution_preserved", False))) for r in wfc_probe_results]
        mask_ratios = [float(np.clip(r.get("mask_ratio_applied", 0.0), 0.0, 1.0)) for r in wfc_probe_results]
        masked_tiles = [float(max(0, int(r.get("masked_tiles", 0)))) for r in wfc_probe_results]

        robustness.update(
            {
                "wfc_probe_count": float(len(wfc_probe_results)),
                "wfc_contradiction_rate": float(np.mean([v > 0 for v in contradictions])) if contradictions else 0.0,
                "wfc_restart_rate": float(np.mean([v > 0 for v in restarts])) if restarts else 0.0,
                "wfc_fallback_rate": float(np.mean(fallback_flags)) if fallback_flags else 0.0,
                "wfc_mean_contradictions": float(np.mean(contradictions)) if contradictions else 0.0,
                "wfc_mean_backtracks": float(np.mean(backtracks)) if backtracks else 0.0,
                "wfc_mean_restarts": float(np.mean(restarts)) if restarts else 0.0,
                "wfc_mean_zero_prob_resets": float(np.mean(resets)) if resets else 0.0,
                "wfc_mean_fallback_fills": float(np.mean(fallback_fills)) if fallback_fills else 0.0,
                "wfc_mean_kl_divergence": float(np.mean(kl_values)) if kl_values else 0.0,
                "wfc_distribution_preserved_rate": float(np.mean(preserved_flags)) if preserved_flags else 0.0,
                "wfc_mean_mask_ratio": float(np.mean(mask_ratios)) if mask_ratios else 0.0,
                "wfc_mean_masked_tiles": float(np.mean(masked_tiles)) if masked_tiles else 0.0,
            }
        )

        def _mask_bucket_label(v: float) -> str:
            return f"mask_{int(round(100.0 * float(np.clip(v, 0.0, 1.0)))):02d}"

        def _severity_label(v: float) -> str:
            vv = float(np.clip(v, 0.0, 1.0))
            if vv < 0.25:
                return "low"
            if vv < 0.50:
                return "mid"
            return "high"

        by_mask: Dict[str, List[Dict[str, Any]]] = {}
        by_severity: Dict[str, List[Dict[str, Any]]] = {"low": [], "mid": [], "high": []}
        for row in wfc_probe_results:
            ratio = float(np.clip(row.get("mask_ratio_target", row.get("mask_ratio_applied", 0.0)), 0.0, 1.0))
            by_mask.setdefault(_mask_bucket_label(ratio), []).append(row)
            by_severity.setdefault(_severity_label(ratio), []).append(row)

        def _add_group_stats(group_rows: Sequence[Dict[str, Any]], key_suffix: str) -> None:
            if not group_rows:
                robustness[f"wfc_probe_count_{key_suffix}"] = 0.0
                robustness[f"wfc_restart_rate_{key_suffix}"] = 0.0
                robustness[f"wfc_fallback_rate_{key_suffix}"] = 0.0
                robustness[f"wfc_distribution_preserved_rate_{key_suffix}"] = 0.0
                robustness[f"wfc_mean_kl_divergence_{key_suffix}"] = 0.0
                return
            g_restarts = [float(max(0, int(r.get("restarts", 0)))) for r in group_rows]
            g_fallback = [float(bool(r.get("required_fallback", False))) for r in group_rows]
            g_preserved = [float(bool(r.get("distribution_preserved", False))) for r in group_rows]
            g_kl = [float(r.get("kl_divergence", 0.0)) for r in group_rows]
            robustness[f"wfc_probe_count_{key_suffix}"] = float(len(group_rows))
            robustness[f"wfc_restart_rate_{key_suffix}"] = float(np.mean([v > 0 for v in g_restarts])) if g_restarts else 0.0
            robustness[f"wfc_fallback_rate_{key_suffix}"] = float(np.mean(g_fallback)) if g_fallback else 0.0
            robustness[f"wfc_distribution_preserved_rate_{key_suffix}"] = float(np.mean(g_preserved)) if g_preserved else 0.0
            robustness[f"wfc_mean_kl_divergence_{key_suffix}"] = float(np.mean(g_kl)) if g_kl else 0.0

        for mask_key in sorted(by_mask.keys()):
            _add_group_stats(by_mask[mask_key], key_suffix=mask_key)
        for severity_key in ("low", "mid", "high"):
            _add_group_stats(by_severity.get(severity_key, []), key_suffix=f"severity_{severity_key}")

    expressive = {
        "coverage_linearity_leniency": _coverage_2d(gen_vec[:, :2], bins=20) if gen_vec.size else 0.0,
        "coverage_progression_topology": _coverage_2d(gen_vec[:, 2:], bins=20) if gen_vec.size else 0.0,
        "descriptor_diversity": float(
            np.mean(np.std(gen_vec, axis=0)) if gen_vec.size else 0.0
        ),
    }

    novelty = 0.0
    if gen_vec.size and ref_vec.size:
        dists = []
        for g in gen_vec:
            nearest = np.min(np.linalg.norm(ref_vec - g, axis=1))
            dists.append(nearest / np.sqrt(gen_vec.shape[1]))
        novelty = float(np.mean(dists))

    fidelity_js = 0.0
    if gen_vec.size and ref_vec.size:
        js_scores = []
        for i in range(gen_vec.shape[1]):
            hg, _ = np.histogram(gen_vec[:, i], bins=20, range=(0.0, 1.0), density=False)
            hr, _ = np.histogram(ref_vec[:, i], bins=20, range=(0.0, 1.0), density=False)
            js_scores.append(_js_divergence(hg, hr))
        fidelity_js = float(np.mean(js_scores))

    overlap = _overlap_2d(gen_vec[:, :2], ref_vec[:, :2], bins=20) if gen_vec.size and ref_vec.size else 0.0

    gen_means, gen_stds = _descriptor_summary(gen_desc)
    ref_means, ref_stds = _descriptor_summary(ref_desc)

    ci: Dict[str, Dict[str, float]] = {
        "start_goal_rate": _bootstrap_mean_ci(
            start_goal_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 1,
        ),
        "connected_rate": _bootstrap_mean_ci(
            connected_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 2,
        ),
        "path_exists_rate": _bootstrap_mean_ci(
            path_exists_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 3,
        ),
        "directed_path_exists_rate": _bootstrap_mean_ci(
            directed_path_exists_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 33,
        ),
        "weak_path_exists_rate": _bootstrap_mean_ci(
            weak_path_exists_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 34,
        ),
        "path_directionality_gap_mean": _bootstrap_mean_ci(
            directionality_gap_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 35,
        ),
        "constraint_valid_rate": _bootstrap_mean_ci(
            constraint_valid_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 4,
        ),
        "overall_completeness": _bootstrap_mean_ci(
            overall_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 5,
        ),
        "repair_rate": _bootstrap_mean_ci(
            repair_applied_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 6,
        ),
        "mean_total_repairs": _bootstrap_mean_ci(
            total_repairs_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 7,
        ),
        "mean_generation_constraint_rejections": _bootstrap_mean_ci(
            generation_constraint_rejections_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 8,
        ),
        "mean_candidate_repairs_applied": _bootstrap_mean_ci(
            candidate_repairs_applied_series,
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 9,
        ),
    }

    ci_keys = [
        "linearity",
        "leniency",
        "progression_complexity",
        "topology_complexity",
        "cycle_density",
        "shortcut_density",
        "gate_depth_ratio",
        "path_depth_ratio",
        "path_length",
        "num_nodes",
    ]
    for i, key in enumerate(ci_keys):
        ci[f"generated_{key}_mean"] = _bootstrap_mean_ci(
            [float(getattr(d, key)) for d in gen_desc],
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 10 + i,
        )
        ci[f"reference_{key}_mean"] = _bootstrap_mean_ci(
            [float(getattr(d, key)) for d in ref_desc],
            n_bootstrap=bootstrap_samples,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 20 + i,
        )

    return BenchmarkSummary(
        num_generated=len(gen_desc),
        num_reference=len(ref_desc),
        mean_generation_time_sec=float(np.mean(generation_times)) if generation_times else 0.0,
        completeness=completeness,
        robustness=robustness,
        expressive_range=expressive,
        reference_comparison={
            "novelty_vs_reference": novelty,
            "fidelity_js_divergence": fidelity_js,
            "expressive_overlap_reference": overlap,
        },
        generated_descriptor_means=gen_means,
        reference_descriptor_means=ref_means,
        generated_descriptor_stds=gen_stds,
        reference_descriptor_stds=ref_stds,
        confidence_intervals=ci,
        calibration={},
    )


def run_block_i_benchmark_from_scratch(
    num_generated: int = 30,
    seed: int = 42,
    data_root: Optional[Path] = None,
    reference_limit: Optional[int] = None,
    population_size: int = 32,
    generations: int = 40,
    min_rooms: int = 8,
    max_rooms: int = 16,
    rule_space: str = "full",
    calibrate_rule_schedule: bool = False,
    calibration_iterations: int = 5,
    calibration_sample_size: int = 12,
    wfc_probe_samples: int = 0,
    wfc_probe_mask_ratios: Optional[Sequence[float]] = None,
    wfc_probe_preserve_critical_tiles: bool = True,
    align_rooms_to_reference: bool = True,
    room_budget_cap: int = 42,
    room_count_bias: float = 0.45,
    search_strategy: str = "ga",
    qd_archive_cells: int = 128,
    qd_init_random_fraction: float = 0.35,
    qd_emitter_mutation_rate: float = 0.18,
    realism_tuning: Optional[Dict[str, float]] = None,
    bootstrap_samples: int = 400,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: int = 12345,
    filter_virtual_start_pointers: bool = True,
    include_data_audit: bool = True,
    data_audit_max_rooms: int = 5000,
) -> BenchmarkSummary:
    """
    End-to-end benchmark runner:
    generate graphs, load references, compute summary.
    """
    if data_root is None:
        data_root = Path("Data") / "The Legend of Zelda"
    reference_graphs = load_vglc_reference_graphs(
        data_root=data_root,
        limit=reference_limit,
        filter_virtual_start_pointers=bool(filter_virtual_start_pointers),
    )
    descriptor_targets: Optional[Dict[str, float]] = None
    if reference_graphs:
        ref_desc = [extract_graph_descriptor(g, grammar=None) for g in reference_graphs]
        ref_means, _ = _descriptor_summary(ref_desc)
        descriptor_targets = {
            "linearity": float(ref_means.get("linearity", 0.45)),
            "leniency": float(ref_means.get("leniency", 0.50)),
            "progression_complexity": float(ref_means.get("progression_complexity", 0.65)),
            "topology_complexity": float(ref_means.get("topology_complexity", 0.45)),
            "path_length": float(ref_means.get("path_length", 8.0)),
            "num_nodes": float(ref_means.get("num_nodes", 12.0)),
            "num_edges": float(ref_means.get("num_edges", 24.0)),
            "cycle_density": float(ref_means.get("cycle_density", 0.30)),
            "shortcut_density": float(ref_means.get("shortcut_density", 0.10)),
            "gate_depth_ratio": float(ref_means.get("gate_depth_ratio", 0.25)),
            "path_depth_ratio": float(ref_means.get("path_depth_ratio", 0.40)),
            "gating_density": float(ref_means.get("gating_density", 0.16)),
            "puzzle_density": float(
                ref_means.get("puzzle_count", 0.0) / max(1.0, float(ref_means.get("num_nodes", 12.0)))
            ),
            "item_density": float(
                ref_means.get("item_count", 0.0) / max(1.0, float(ref_means.get("num_nodes", 12.0)))
            ),
            "gate_variety": float(ref_means.get("gate_variety", 0.30)),
            "bombable_ratio": float(
                ref_means.get("bombable_count", 0.0) / max(1.0, float(ref_means.get("num_edges", 24.0)))
            ),
            "soft_lock_ratio": float(
                ref_means.get("softlock_count", 0.0) / max(1.0, float(ref_means.get("num_edges", 24.0)))
            ),
            "switch_ratio": float(
                ref_means.get("switch_count", 0.0) / max(1.0, float(ref_means.get("num_edges", 24.0)))
            ),
            "stair_ratio": float(
                ref_means.get("stair_count", 0.0) / max(1.0, float(ref_means.get("num_edges", 24.0)))
            ),
        }
    rule_weight_overrides: Optional[Dict[str, float]] = None
    calibration_payload: Dict[str, Any] = {
        "enabled": bool(calibrate_rule_schedule),
        "rule_weight_overrides": {},
        "target_descriptor_means": descriptor_targets or {},
        "history": [],
        "best_gap": 0.0,
    }
    if calibrate_rule_schedule and reference_graphs:
        calibration = calibrate_rule_weights_to_vglc(
            reference_graphs,
            seed=seed,
            iterations=calibration_iterations,
            sample_size=calibration_sample_size,
            population_size=max(16, population_size),
            generations=max(12, generations // 2),
            min_rooms=min_rooms,
            max_rooms=max_rooms,
        )
        rule_weight_overrides = calibration.get("calibrated_rule_weights", {})
        calibration_payload.update(
            {
                "rule_weight_overrides": rule_weight_overrides,
                "target_descriptor_means": calibration.get("target_descriptor_means", {}),
                "history": calibration.get("history", []),
                "best_gap": float(calibration.get("best_gap", 0.0)),
            }
        )

    effective_min_rooms = int(min_rooms)
    effective_max_rooms = int(max_rooms)
    if align_rooms_to_reference and reference_graphs:
        ref_nodes = np.asarray([max(1, int(g.number_of_nodes())) for g in reference_graphs], dtype=np.float64)
        if ref_nodes.size > 0:
            q25 = int(np.floor(np.quantile(ref_nodes, 0.25)))
            q75 = int(np.ceil(np.quantile(ref_nodes, 0.75)))
            cap = max(6, int(room_budget_cap))
            aligned_min = int(np.clip(q25, 6, cap))
            aligned_max = int(np.clip(max(q75, aligned_min + 2), aligned_min + 1, cap))
            effective_min_rooms = max(effective_min_rooms, aligned_min)
            effective_max_rooms = max(effective_max_rooms, aligned_max)
            effective_max_rooms = int(min(effective_max_rooms, cap))
            if effective_min_rooms >= effective_max_rooms:
                effective_min_rooms = max(6, effective_max_rooms - 2)

    generated_graphs, generation_times = generate_block_i_graphs(
        num_samples=num_generated,
        seed=seed,
        min_rooms=effective_min_rooms,
        max_rooms=effective_max_rooms,
        population_size=population_size,
        generations=generations,
        rule_space=rule_space,
        rule_weight_overrides=rule_weight_overrides,
        descriptor_targets=descriptor_targets,
        room_count_bias=room_count_bias,
        search_strategy=search_strategy,
        qd_archive_cells=qd_archive_cells,
        qd_init_random_fraction=qd_init_random_fraction,
        qd_emitter_mutation_rate=qd_emitter_mutation_rate,
        realism_tuning=realism_tuning,
    )
    calibration_payload.update(
        {
            "effective_room_budget": {
                "min_rooms": int(effective_min_rooms),
                "max_rooms": int(effective_max_rooms),
                "aligned_to_reference": bool(align_rooms_to_reference),
                "room_budget_cap": int(room_budget_cap),
                "room_count_bias": float(room_count_bias),
                "search_strategy": str(search_strategy),
                "qd_archive_cells": int(qd_archive_cells),
                "qd_init_random_fraction": float(qd_init_random_fraction),
                "qd_emitter_mutation_rate": float(qd_emitter_mutation_rate),
                "realism_tuning": dict(realism_tuning or {}),
            }
        }
    )
    wfc_probe_results: List[Dict[str, Any]] = []
    if int(wfc_probe_samples) > 0:
        reference_rooms = load_vglc_reference_rooms(data_root=data_root, max_rooms=int(wfc_probe_samples))
        wfc_probe_results = run_wfc_robustness_probe(
            room_grids=reference_rooms,
            seed=seed + 7000,
            max_samples=int(wfc_probe_samples),
            mask_ratios=wfc_probe_mask_ratios,
            preserve_critical_tiles=bool(wfc_probe_preserve_critical_tiles),
        )

    summary = run_block_i_benchmark(
        generated_graphs=generated_graphs,
        reference_graphs=reference_graphs,
        generation_times=generation_times,
        wfc_probe_results=wfc_probe_results,
        bootstrap_samples=bootstrap_samples,
        bootstrap_alpha=bootstrap_alpha,
        bootstrap_seed=bootstrap_seed,
    )
    summary.calibration = calibration_payload
    if include_data_audit:
        summary.data_audit = audit_block0_dataset(
            data_root=data_root,
            max_rooms=max(1, int(data_audit_max_rooms)),
            graph_limit=reference_limit,
            filter_virtual_start_pointers=bool(filter_virtual_start_pointers),
        )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run research-oriented Block I benchmark against VGLC references."
    )
    parser.add_argument("--num-generated", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--reference-limit", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--min-rooms", type=int, default=8)
    parser.add_argument("--max-rooms", type=int, default=16)
    parser.add_argument("--rule-space", type=str, default="full", choices=["core", "full"])
    parser.add_argument("--calibrate-rule-schedule", action="store_true")
    parser.add_argument("--calibration-iterations", type=int, default=5)
    parser.add_argument("--calibration-sample-size", type=int, default=12)
    parser.add_argument("--wfc-probe-samples", type=int, default=0)
    parser.add_argument(
        "--wfc-probe-mask-ratios",
        type=str,
        default="0.15,0.35,0.55",
        help="Comma-separated masking ratios for WFC stress probe (e.g. 0.15,0.35,0.55).",
    )
    parser.add_argument("--wfc-probe-no-preserve-critical", action="store_true")
    parser.add_argument("--no-align-rooms-to-reference", action="store_true")
    parser.add_argument("--room-budget-cap", type=int, default=42)
    parser.add_argument("--room-count-bias", type=float, default=0.45)
    parser.add_argument("--search-strategy", type=str, default="ga", choices=["ga", "cvt_emitter"])
    parser.add_argument("--qd-archive-cells", type=int, default=128)
    parser.add_argument("--qd-init-random-fraction", type=float, default=0.35)
    parser.add_argument("--qd-emitter-mutation-rate", type=float, default=0.18)
    parser.add_argument(
        "--realism-profile",
        type=str,
        default="gate_quality_heavy",
        choices=list_realism_tuning_profiles(include_engine_default=True),
        help="Named realism profile (default: gate_quality_heavy); JSON/file overrides are applied on top.",
    )
    parser.add_argument(
        "--realism-tuning-json",
        type=str,
        default="",
        help="JSON object with realism tuning overrides, e.g. '{\"adapt_node_gain\":0.46}'.",
    )
    parser.add_argument(
        "--realism-tuning-file",
        type=Path,
        default=None,
        help="Path to JSON file containing realism tuning overrides object.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument(
        "--keep-virtual-start-pointers",
        action="store_true",
        help="Disable virtual start-pointer filtering for reference DOT graphs.",
    )
    parser.add_argument("--no-data-audit", action="store_true")
    parser.add_argument("--data-audit-max-rooms", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    mask_ratios: List[float] = []
    for chunk in str(args.wfc_probe_mask_ratios).split(","):
        token = str(chunk).strip()
        if not token:
            continue
        try:
            mask_ratios.append(float(token))
        except ValueError:
            logger.warning("Ignoring invalid mask ratio token: %s", token)
    if not mask_ratios:
        mask_ratios = [float(v) for v in DEFAULT_WFC_PROBE_MASK_RATIOS]
    realism_tuning: Optional[Dict[str, float]] = get_realism_tuning_profile(args.realism_profile)
    if not realism_tuning:
        realism_tuning = None
    if str(args.realism_tuning_json).strip():
        try:
            parsed = json.loads(str(args.realism_tuning_json))
            if isinstance(parsed, dict):
                json_tuning = {
                    str(k): float(v)
                    for k, v in parsed.items()
                }
                realism_tuning = dict(realism_tuning or {})
                realism_tuning.update(json_tuning)
            else:
                logger.warning("Ignoring --realism-tuning-json because payload is not an object")
        except Exception as exc:
            logger.warning("Ignoring invalid --realism-tuning-json payload: %s", exc)
    if args.realism_tuning_file is not None:
        try:
            raw = args.realism_tuning_file.read_text(encoding="utf-8")
            parsed_file = json.loads(raw)
            if isinstance(parsed_file, dict):
                file_tuning = {str(k): float(v) for k, v in parsed_file.items()}
                realism_tuning = (realism_tuning or {})
                realism_tuning.update(file_tuning)
            else:
                logger.warning("Ignoring --realism-tuning-file because payload is not an object: %s", args.realism_tuning_file)
        except Exception as exc:
            logger.warning("Ignoring invalid --realism-tuning-file payload '%s': %s", args.realism_tuning_file, exc)
    summary = run_block_i_benchmark_from_scratch(
        num_generated=args.num_generated,
        seed=args.seed,
        data_root=args.data_root,
        reference_limit=args.reference_limit,
        population_size=args.population_size,
        generations=args.generations,
        min_rooms=args.min_rooms,
        max_rooms=args.max_rooms,
        rule_space=args.rule_space,
        calibrate_rule_schedule=args.calibrate_rule_schedule,
        calibration_iterations=args.calibration_iterations,
        calibration_sample_size=args.calibration_sample_size,
        wfc_probe_samples=args.wfc_probe_samples,
        wfc_probe_mask_ratios=mask_ratios,
        wfc_probe_preserve_critical_tiles=not bool(args.wfc_probe_no_preserve_critical),
        align_rooms_to_reference=not bool(args.no_align_rooms_to_reference),
        room_budget_cap=args.room_budget_cap,
        room_count_bias=args.room_count_bias,
        search_strategy=args.search_strategy,
        qd_archive_cells=args.qd_archive_cells,
        qd_init_random_fraction=args.qd_init_random_fraction,
        qd_emitter_mutation_rate=args.qd_emitter_mutation_rate,
        realism_tuning=realism_tuning,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_alpha=args.bootstrap_alpha,
        bootstrap_seed=args.bootstrap_seed,
        filter_virtual_start_pointers=not bool(args.keep_virtual_start_pointers),
        include_data_audit=not bool(args.no_data_audit),
        data_audit_max_rooms=args.data_audit_max_rooms,
    )

    payload = asdict(summary)
    text = json.dumps(payload, indent=2 if args.pretty else None)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved benchmark report to %s", args.output)


if __name__ == "__main__":
    main()
