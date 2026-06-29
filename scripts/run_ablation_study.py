"""
Thesis-grade ablation runner with fixed-seed paired comparisons.

Implements:
1) Core ablations: FULL vs NO_EVOLUTION / NO_GRAPH / NO_WFC / NO_LOGIC
    + RANDOM_TOPOLOGY / PURE_WFC
2) Requested sweeps:
   - VQ codebook size (128/512/2048) via categorical codebook cap
   - latent diffusion vs categorical
   - conditioning with/without TPE
   - logic guidance strength sweep
   - WFC on/off
3) Significance reporting (paired bootstrap CI + random-sign permutation p-value)
4) Multiple-comparison control (Benjamini-Hochberg FDR-adjusted p-values)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

# Ensure repository root is importable when script is executed directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_system import load_resolved_config_for_artifact
from src.evaluation.benchmark_suite import (
    extract_graph_descriptor,
    load_vglc_reference_graphs,
    load_vglc_reference_rooms,
)
from src.evaluation.search_benchmark_utils import confusion_ratio_vs_oracle, run_astar_oracle
from src.generation.evolutionary_director import mission_graph_to_networkx
from src.generation.evolutionary_director import networkx_to_mission_graph
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
from src.generation.grammar import Difficulty, MissionGrammar
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.pipeline.dungeon_pipeline import RoomGenerationResult
from src.pipeline.dungeon_pipeline import pipeline_kwargs_from_resolved_config
from src.pipeline.room_stitching import build_stitched_room_layout
from src.generation.weighted_bayesian_wfc import (
    WeightedBayesianWFC,
    WeightedBayesianWFCConfig,
    extract_tile_priors_from_vqvae,
)
from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH
from src.core.definitions import TileID
from src.core.definitions import parse_edge_type_tokens
from src.core.latent_diffusion import CrossAttention
from src.simulation.cognitive_bounded_search import solve_with_cbs
from src.simulation.validator import ZeldaLogicEnv
from src.pipeline.spatial_utils import first_free_position, get_node_grid_position
from src.utils.stable_seed import stable_seed_offset

logger = logging.getLogger(__name__)


def _maybe_existing_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    candidate = Path(str(path_value))
    return candidate if candidate.exists() else None


def _infer_logic_checkpoint(diffusion_checkpoint: Optional[str]) -> Optional[str]:
    checkpoint_path = _maybe_existing_path(diffusion_checkpoint)
    if checkpoint_path is None:
        return None
    siblings = [
        checkpoint_path.with_name("best_logic_model.pth"),
        checkpoint_path.with_name("logic_net_best.pth"),
    ]
    for candidate in siblings:
        if candidate.exists():
            return str(candidate)
    return None


def _load_pipeline_resolved_config(*artifact_paths: Optional[str]) -> Optional[Dict[str, Any]]:
    for artifact in artifact_paths:
        resolved = load_resolved_config_for_artifact(artifact)
        if isinstance(resolved, dict):
            return resolved
    return None


def _stitch_with_pipeline(
    pipeline: NeuralSymbolicDungeonPipeline,
    rooms: Dict[Any, RoomGenerationResult],
    graph: nx.Graph,
) -> np.ndarray:
    """Use the public stitch API while remaining compatible with positional-only stitchers."""
    stitch_fn = getattr(pipeline, "stitch_rooms", None)
    if callable(stitch_fn):
        try:
            return stitch_fn(rooms=rooms, graph=graph)
        except TypeError:
            return stitch_fn(rooms, graph)
    return pipeline._stitch_rooms(rooms, graph)


def _stitch_symbolic_rooms(
    rooms: Dict[Any, RoomGenerationResult],
    graph: nx.Graph,
) -> np.ndarray:
    """Stitch a non-neural baseline without constructing model modules."""
    return build_stitched_room_layout(
        rooms=rooms,
        graph=graph,
        enforce_room_dimensions=(ROOM_HEIGHT, ROOM_WIDTH),
    ).dungeon_grid


def _apply_symbolic_room_scaffold(
    room_grid: np.ndarray,
    node_attrs: Mapping[str, Any],
) -> Tuple[np.ndarray, int]:
    """Add the minimal graph-role/corridor contract shared by both WFC priors."""
    grid = np.asarray(room_grid, dtype=np.int32).copy()
    before = grid.copy()
    center_row = int(grid.shape[0] // 2)
    center_col = int(grid.shape[1] // 2)
    floor_id = int(TileID.FLOOR)
    grid[center_row, 1:-1] = floor_id
    grid[1:-1, center_col] = floor_id

    node_type = str(node_attrs.get("type", node_attrs.get("node_type", ""))).strip().upper()
    label = str(node_attrs.get("label", "")).strip().upper()
    if bool(node_attrs.get("is_start")) or node_type == "START" or label == "START":
        marker = int(TileID.START)
    elif bool(node_attrs.get("is_goal")) or bool(node_attrs.get("is_triforce")) or node_type == "GOAL" or label == "GOAL":
        marker = int(TileID.TRIFORCE)
    elif node_type in {"BOSS", "MINI_BOSS"}:
        marker = int(TileID.BOSS)
    elif node_type == "BIG_KEY":
        marker = int(TileID.KEY_BOSS)
    elif node_type in {"KEY", "TOKEN"}:
        marker = int(TileID.KEY_SMALL)
    elif node_type in {"ITEM", "TREASURE"}:
        marker = int(TileID.KEY_ITEM)
    else:
        marker = floor_id
    grid[center_row, center_col] = marker
    return grid, int(np.count_nonzero(grid != before))


def _sanitized_exception_name(exc: BaseException) -> str:
    """Short exception summary safe to persist in experiment outputs."""
    return type(exc).__name__


@dataclass
class ExperimentConfig:
    name: str
    use_evolution: bool = True
    random_topology: bool = False
    use_wfc: bool = True
    pure_wfc: bool = False
    logic_guidance_scale: float = 1.0
    logic_guidance_active_fraction: Optional[float] = None
    latent_sampler: str = "diffusion"  # diffusion | categorical
    categorical_codebook_size: Optional[int] = None
    use_tpe: bool = True
    disable_graph_node_cross_attention: bool = False
    topology_refinement_mode: str = "gat2"  # none | lightweight | sparse*/gat2* | graphormer
    room_generator_mode: str = "latent_diffusion"  # latent_diffusion | discrete_masked
    use_reference_room_maps: Optional[bool] = None
    wfc_prior_mode: str = "weighted"  # weighted | flat
    diffusion_topology_conditioning_mode: Optional[str] = None  # additive | spade | None
    diffusion_checkpoint_override: Optional[str] = None


def bind_topology_ablation_checkpoints(
    configs: Sequence[ExperimentConfig],
    *,
    default_diffusion_checkpoint: Optional[str],
    additive_checkpoint: Optional[str],
    spade_checkpoint: Optional[str],
    require_existing: bool,
) -> None:
    """Bind architecture-matched checkpoints to topology-conditioning arms."""
    checkpoint_by_mode = {
        "additive": additive_checkpoint or default_diffusion_checkpoint,
        "spade": spade_checkpoint,
    }
    missing: List[str] = []
    for cfg in configs:
        mode = str(cfg.diffusion_topology_conditioning_mode or "").strip().lower()
        if mode not in checkpoint_by_mode:
            continue
        checkpoint = checkpoint_by_mode[mode]
        cfg.diffusion_checkpoint_override = str(checkpoint) if checkpoint else None
        if require_existing and (not checkpoint or not Path(checkpoint).is_file()):
            missing.append(f"{cfg.name} ({mode})")
    if missing:
        raise ValueError(
            "Topology-conditioning ablations require separately trained, existing checkpoints for every arm. "
            f"Missing: {', '.join(missing)}. Use --diffusion-additive-checkpoint and "
            "--diffusion-spade-checkpoint."
        )


def validate_loaded_topology_conditioning_mode(
    pipeline: NeuralSymbolicDungeonPipeline,
    *,
    expected_mode: Optional[str],
    checkpoint_path: Optional[str],
) -> None:
    """Fail closed when checkpoint metadata defeats an ablation override."""
    if expected_mode is None:
        return
    expected = str(expected_mode).strip().lower()
    diffusion = getattr(pipeline, "diffusion", None)
    actual = str(getattr(diffusion, "topology_conditioning_mode", "")).strip().lower()
    if actual != expected:
        raise ValueError(
            "Loaded diffusion architecture does not match the requested topology-conditioning ablation: "
            f"expected={expected!r}, actual={actual!r}, checkpoint={checkpoint_path!r}. "
            "Use a checkpoint trained with the requested mode; constructor fallback overrides are not evidence."
        )


def set_topology_refinement_mode_or_raise(diffusion: Any, mode: str) -> None:
    """Apply an attention-topology ablation and verify the active mode."""
    normalized = str(mode).strip().lower()
    setter = getattr(diffusion, "set_topology_refinement_mode", None)
    getter = getattr(diffusion, "get_topology_refinement_mode", None)
    if not callable(setter) or not callable(getter):
        raise RuntimeError(
            f"Diffusion model does not expose topology-refinement switching required for mode={normalized!r}."
        )
    setter(normalized)
    actual = str(getter()).strip().lower()
    if actual != normalized:
        raise RuntimeError(
            "Topology-refinement ablation did not activate the requested mode: "
            f"expected={normalized!r}, actual={actual!r}."
        )


PRIMARY_ABLATION_METRICS: Tuple[str, ...] = (
    "solvable",
    "confusion_ratio",
    "confusion_index",
    "path_optimal",
    "tile_prior_kl",
    "graph_edit_distance",
    "generation_time_sec",
    "novelty",
    "reconstruction_error",
    "constraint_valid",
    "room_repair_rate",
    "topology_preservation_score",
    "directed_edge_preservation_score",
    "topology_attention_pairs",
    "topology_shortest_path_bias_ops",
    "topology_relative_attention_pairs_to_gat2",
)


ABLATION_DESIGN_NOTES: Dict[str, Dict[str, str]] = {
    "FULL": {
        "tier": "full_stack",
        "component": "canonical hybrid stack",
        "comparison": "reference condition for paired seed deltas",
        "isolates": "none; all enabled production components",
        "interpretation": "Use as the denominator for component-necessity and runtime tradeoff claims.",
    },
    "TOPO_LIGHTWEIGHT": {
        "tier": "block_iv",
        "component": "topology-aware attention refinement",
        "comparison": "FULL",
        "isolates": "weaker topology refinement while keeping topology, neural generation, LogicNet, and WFC on",
        "interpretation": "Tests whether the heavier graph refinement path is justified over a cheaper topology signal.",
    },
    "TOPO_SPARSE_EDGE": {
        "tier": "block_iv",
        "component": "topology-aware attention refinement",
        "comparison": "FULL and TOPO_LIGHTWEIGHT",
        "isolates": "edge-sparse graph attention over mission edges plus self loops without dense all-pairs scores",
        "interpretation": "Tests whether sparse topology attention preserves graph fidelity at lower asymptotic cost than GAT2/Graphormer.",
    },
    "TOPO_SPARSE_DIRECTED": {
        "tier": "block_iv",
        "component": "topology-aware attention refinement",
        "comparison": "TOPO_SPARSE_EDGE",
        "isolates": "directed sparse graph attention that does not mirror mission edges",
        "interpretation": "Tests whether respecting one-way topology helps or hurts compared with the legacy undirected inductive bias.",
    },
    "TOPO_SPARSE_SEMANTIC": {
        "tier": "block_iv",
        "component": "topology-aware attention refinement",
        "comparison": "TOPO_SPARSE_EDGE and TOPO_SPARSE_DIRECTED",
        "isolates": "edge-type-aware sparse attention using deterministic gate-severity bias",
        "interpretation": "Tests whether door/lock semantics improve topology preservation without adding untrained checkpoint parameters.",
    },
    "NO_EVOLUTION": {
        "tier": "block_i",
        "component": "evolutionary topology search",
        "comparison": "FULL",
        "isolates": "direct grammar generation with no evolutionary optimization",
        "interpretation": "Separates grammar validity from search pressure toward the target tension curve.",
    },
    "RANDOM_TOPOLOGY": {
        "tier": "block_i",
        "component": "topology prior",
        "comparison": "FULL and NO_EVOLUTION",
        "isolates": "a seeded start-to-goal random DAG baseline with no grammar/evolution objective",
        "interpretation": "Strict null for topology realism and controllability; should not be conflated with NO_EVOLUTION.",
    },
    "NO_GRAPH": {
        "tier": "block_iii_iv",
        "component": "graph conditioning",
        "comparison": "FULL",
        "isolates": "graph-token cross-attention, TPE, and topology refinement disabled together",
        "interpretation": "Measures whether room generation depends on mission-graph context beyond local priors.",
    },
    "NO_WFC": {
        "tier": "block_vi",
        "component": "symbolic WFC repair",
        "comparison": "FULL",
        "isolates": "post-neural symbolic repair disabled",
        "interpretation": "Tests whether symbolic repair is necessary for playable and topology-consistent room layouts.",
    },
    "NO_LOGIC": {
        "tier": "block_v",
        "component": "LogicNet guidance",
        "comparison": "FULL",
        "isolates": "logic-guidance scale set to zero while keeping graph conditioning and WFC",
        "interpretation": "Tests whether runtime logic gradients add value beyond WFC and graph-conditioned decoding.",
    },
    "PURE_WFC": {
        "tier": "block_vi",
        "component": "standalone symbolic generator with graph-role scaffold",
        "comparison": "FULL and NO_WFC",
        "isolates": (
            "weighted WFC rooms stitched over the same generated topology, bypassing neural priors; "
            "a deterministic START/GOAL/key-role scaffold is applied so graph semantics are visible to validation"
        ),
        "interpretation": (
            "Heuristic-only scaffolded baseline for testing learned pattern priors. "
            "Do not report it as unconstrained pure WFC or as neural-room evidence."
        ),
    },
    "LATENT_DIFFUSION": {
        "tier": "block_iv",
        "component": "latent diffusion sampler",
        "comparison": "LATENT_CATEGORICAL and FULL",
        "isolates": "continuous denoising sampler under the same graph and repair stack",
        "interpretation": "Reference room-generation sampler for representation comparisons.",
    },
    "LATENT_CATEGORICAL": {
        "tier": "block_iv",
        "component": "categorical latent sampler",
        "comparison": "LATENT_DIFFUSION",
        "isolates": "categorical/codebook sampling without DDPM-style denoising",
        "interpretation": "Tests whether the diffusion process itself improves quality over direct latent-code sampling.",
    },
    "COND_NO_TPE": {
        "tier": "block_iii",
        "component": "topological positional encoding",
        "comparison": "FULL",
        "isolates": "TPE disabled while retaining graph-token conditioning",
        "interpretation": "Measures whether relative graph-position signals matter beyond node attributes.",
    },
    "COND_WEAK_GRAPH": {
        "tier": "block_iii_iv",
        "component": "weak graph conditioning",
        "comparison": "FULL and COND_NO_TPE",
        "isolates": "TPE disabled and graph-node cross-attention removed",
        "interpretation": "Intermediate condition between full graph conditioning and fully graph-free generation.",
    },
}


def _tile_distribution(grids: Sequence[np.ndarray]) -> Dict[int, float]:
    counts: Dict[int, int] = {}
    total = 0
    for grid in grids:
        arr = np.asarray(grid, dtype=np.int32)
        unique, freq = np.unique(arr, return_counts=True)
        for k, v in zip(unique.tolist(), freq.tolist()):
            counts[int(k)] = counts.get(int(k), 0) + int(v)
            total += int(v)
    if total <= 0:
        return {}
    return {k: float(v / total) for k, v in counts.items()}


def _kl_divergence(reference: Dict[int, float], generated: Dict[int, float], eps: float = 1e-9) -> float:
    if not reference:
        return 0.0
    keys = sorted(set(reference.keys()) | set(generated.keys()))
    p = np.array([float(reference.get(k, 0.0)) + eps for k in keys], dtype=np.float64)
    q = np.array([float(generated.get(k, 0.0)) + eps for k in keys], dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _descriptor_vector(G: nx.Graph) -> np.ndarray:
    d = extract_graph_descriptor(G, grammar=None)
    return np.array(
        [d.linearity, d.leniency, d.progression_complexity, d.topology_complexity],
        dtype=np.float64,
    )


def _graph_edit_distance_proxy(Ga: nx.Graph, Gb: nx.Graph) -> float:
    na = float(Ga.number_of_nodes())
    nb = float(Gb.number_of_nodes())
    ea = float(Ga.number_of_edges())
    eb = float(Gb.number_of_edges())
    node_term = abs(na - nb) / max(1.0, max(na, nb))
    edge_term = abs(ea - eb) / max(1.0, max(ea, eb))

    def _type_hist(G: nx.Graph) -> Dict[str, int]:
        h: Dict[str, int] = {}
        for _, attrs in G.nodes(data=True):
            t = str(attrs.get("type", attrs.get("label", "unknown"))).lower()
            h[t] = h.get(t, 0) + 1
        return h

    ha = _type_hist(Ga)
    hb = _type_hist(Gb)
    keys = sorted(set(ha.keys()) | set(hb.keys()))
    if keys:
        va = np.array([ha.get(k, 0) for k in keys], dtype=np.float64)
        vb = np.array([hb.get(k, 0) for k in keys], dtype=np.float64)
        va /= max(np.sum(va), 1.0)
        vb /= max(np.sum(vb), 1.0)
        type_term = float(np.mean(np.abs(va - vb)))
    else:
        type_term = 0.0

    return float(0.4 * node_term + 0.35 * edge_term + 0.25 * type_term)


def _nearest_graph_edit_distance(G: nx.Graph, refs: Sequence[nx.Graph], max_refs: int = 20) -> float:
    if not refs:
        return 0.0
    candidates = list(refs[: max(1, int(max_refs))])
    dists = [_graph_edit_distance_proxy(G, R) for R in candidates]
    return float(min(dists)) if dists else 0.0


def _pairwise_diversity(vectors: Sequence[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    arr = np.stack(vectors, axis=0)
    total = 0.0
    count = 0
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            total += float(np.linalg.norm(arr[i] - arr[j]) / np.sqrt(arr.shape[1]))
            count += 1
    return float(total / max(1, count))


def _build_room_placement(graph: nx.Graph, room_ids: Sequence[Any]) -> Dict[Any, Tuple[int, int]]:
    """Reconstruct the room placement policy used by the pipeline stitch step."""
    placement: Dict[Any, Tuple[int, int]] = {}
    occupied = set()

    for room_id in room_ids:
        pos = get_node_grid_position(graph, room_id)
        if pos is None:
            continue
        resolved = first_free_position(pos, occupied)
        placement[room_id] = resolved
        occupied.add(resolved)

    if graph.is_directed():
        try:
            order = [n for n in nx.topological_sort(graph) if n in room_ids]
        except nx.NetworkXUnfeasible:
            order = sorted(list(room_ids), key=lambda x: str(x))
    else:
        order = sorted(list(room_ids), key=lambda x: str(x))

    next_row = max((r for r, _ in occupied), default=-1) + 1
    for room_id in order:
        if room_id in placement:
            continue
        while (next_row, 0) in occupied:
            next_row += 1
        placement[room_id] = (next_row, 0)
        occupied.add((next_row, 0))
        next_row += 1

    min_r = min(r for r, _ in placement.values()) if placement else 0
    min_c = min(c for _, c in placement.values()) if placement else 0
    return {rid: (r - min_r, c - min_c) for rid, (r, c) in placement.items()}


def _boundary_connection_exists(global_grid: np.ndarray, src_pos: Tuple[int, int], dst_pos: Tuple[int, int]) -> bool:
    """Check whether adjacent rooms share at least one traversable boundary opening."""
    dr = int(dst_pos[0] - src_pos[0])
    dc = int(dst_pos[1] - src_pos[1])
    if abs(dr) + abs(dc) != 1:
        return False

    blocked = {
        int(TileID.VOID),
        int(TileID.WALL),
        int(TileID.BLOCK),
        int(TileID.ELEMENT),
    }

    def _is_passable(val: Any) -> bool:
        try:
            return int(val) not in blocked
        except Exception:
            return False

    if dr != 0:
        src_row = (src_pos[0] + (1 if dr > 0 else 0)) * ROOM_HEIGHT - (1 if dr > 0 else 0)
        dst_row = src_row + (1 if dr > 0 else -1)
        c0 = src_pos[1] * ROOM_WIDTH
        c1 = c0 + ROOM_WIDTH
        for col in range(c0, c1):
            if 0 <= src_row < global_grid.shape[0] and 0 <= dst_row < global_grid.shape[0] and 0 <= col < global_grid.shape[1]:
                if _is_passable(global_grid[src_row, col]) and _is_passable(global_grid[dst_row, col]):
                    return True
        return False

    src_col = (src_pos[1] + (1 if dc > 0 else 0)) * ROOM_WIDTH - (1 if dc > 0 else 0)
    dst_col = src_col + (1 if dc > 0 else -1)
    r0 = src_pos[0] * ROOM_HEIGHT
    r1 = r0 + ROOM_HEIGHT
    for row in range(r0, r1):
        if 0 <= row < global_grid.shape[0] and 0 <= src_col < global_grid.shape[1] and 0 <= dst_col < global_grid.shape[1]:
            if _is_passable(global_grid[row, src_col]) and _is_passable(global_grid[row, dst_col]):
                return True
    return False


def _boundary_has_directional_marker(
    global_grid: np.ndarray,
    src_pos: Tuple[int, int],
    dst_pos: Tuple[int, int],
) -> bool:
    """Detect source-side directional/gating markers on a shared room boundary."""
    dr = int(dst_pos[0] - src_pos[0])
    dc = int(dst_pos[1] - src_pos[1])
    if abs(dr) + abs(dc) != 1:
        return False

    directional_marker_ids = {
        int(TileID.DOOR_SOFT),
        int(TileID.DOOR_LOCKED),
        int(TileID.DOOR_BOMB),
        int(TileID.DOOR_PUZZLE),
        int(TileID.DOOR_BOSS),
    }

    if dr != 0:
        src_row = (src_pos[0] + (1 if dr > 0 else 0)) * ROOM_HEIGHT - (1 if dr > 0 else 0)
        c0 = src_pos[1] * ROOM_WIDTH
        c1 = c0 + ROOM_WIDTH
        for col in range(c0, c1):
            if 0 <= src_row < global_grid.shape[0] and 0 <= col < global_grid.shape[1]:
                if int(global_grid[src_row, col]) in directional_marker_ids:
                    return True
        return False

    src_col = (src_pos[1] + (1 if dc > 0 else 0)) * ROOM_WIDTH - (1 if dc > 0 else 0)
    r0 = src_pos[0] * ROOM_HEIGHT
    r1 = r0 + ROOM_HEIGHT
    for row in range(r0, r1):
        if 0 <= row < global_grid.shape[0] and 0 <= src_col < global_grid.shape[1]:
            if int(global_grid[row, src_col]) in directional_marker_ids:
                return True
    return False


def _topology_information_scorecard(
    *,
    graph: nx.Graph,
    rooms: Dict[Any, RoomGenerationResult],
    dungeon_grid: np.ndarray,
) -> Dict[str, float]:
    """Measure how well stitched room connectivity preserves mission-graph topology."""
    room_ids = list(rooms.keys()) if rooms else list(graph.nodes())
    placement = _build_room_placement(graph, room_ids)

    directed_edges = [(u, v) for u, v in graph.edges() if u in placement and v in placement]
    undirected_edge_set = {tuple(sorted((u, v), key=lambda x: str(x))) for u, v in directed_edges if u != v}

    adjacent_pairs = set()
    ids = list(placement.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            pa, pb = placement[a], placement[b]
            if abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) == 1:
                adjacent_pairs.add(tuple(sorted((a, b), key=lambda x: str(x))))

    representable_edges = [e for e in undirected_edge_set if e in adjacent_pairs]
    representable_edge_rate = float(len(representable_edges) / max(1, len(undirected_edge_set)))

    connected_representable = 0
    for a, b in representable_edges:
        if _boundary_connection_exists(dungeon_grid, placement[a], placement[b]):
            connected_representable += 1
    edge_connection_recall = float(connected_representable / max(1, len(representable_edges)))

    phantom_pairs = [p for p in adjacent_pairs if p not in undirected_edge_set]
    phantom_connections = 0
    for a, b in phantom_pairs:
        if _boundary_connection_exists(dungeon_grid, placement[a], placement[b]):
            phantom_connections += 1
    phantom_connection_rate = float(phantom_connections / max(1, len(phantom_pairs)))

    topology_preservation_score = float(np.clip(
        (0.45 * representable_edge_rate)
        + (0.45 * edge_connection_recall)
        + (0.10 * (1.0 - phantom_connection_rate)),
        0.0,
        1.0,
    ))

    # Strict directed/gating preservation checks.
    # These metrics are intentionally strict: stitched room boundary carving is
    # currently symmetric, so one-way semantics from Block I can leak.
    directional_tokens = {
        "soft_locked",
        "one_way",
        "state_block",
        "switch",
        "switch_locked",
        "on_off_gate",
        "item_gate",
        "item_locked",
        "boss_locked",
        "key_locked",
        "locked",
        "multi_lock",
        "hazard",
        "shutter",
    }

    directed_candidates: List[Tuple[Any, Any]] = []
    directed_realized = 0
    directed_leaks = 0

    if graph.is_directed():
        for u, v, attrs in graph.edges(data=True):
            if u not in placement or v not in placement or u == v:
                continue
            if abs(placement[u][0] - placement[v][0]) + abs(placement[u][1] - placement[v][1]) != 1:
                continue

            label = str(attrs.get("label", "") or "")
            edge_type = str(attrs.get("edge_type", attrs.get("type", "")) or "")
            tokens = set(parse_edge_type_tokens(label=label, edge_type=edge_type))
            reverse_exists = bool(graph.has_edge(v, u))
            is_directional = (not reverse_exists) or bool(tokens.intersection(directional_tokens))
            if not is_directional:
                continue

            directed_candidates.append((u, v))
            opened = _boundary_connection_exists(dungeon_grid, placement[u], placement[v])
            if opened:
                directed_realized += 1
                # If reverse edge is absent, symmetric opening leaks one-way intent.
                if not reverse_exists:
                    has_marker = _boundary_has_directional_marker(
                        dungeon_grid,
                        placement[u],
                        placement[v],
                    )
                    if not has_marker:
                        directed_leaks += 1

    directed_representable_edge_rate = float(
        len(directed_candidates) / max(1, len(directed_edges))
    )
    directed_edge_realization_rate = float(
        directed_realized / max(1, len(directed_candidates))
    )
    directed_directionality_leak_rate = float(
        directed_leaks / max(1, directed_realized)
    )
    directed_edge_preservation_score = float(np.clip(
        directed_edge_realization_rate * (1.0 - directed_directionality_leak_rate),
        0.0,
        1.0,
    ))

    return {
        "topology_representable_edge_rate": representable_edge_rate,
        "topology_edge_connection_recall": edge_connection_recall,
        "topology_phantom_connection_rate": phantom_connection_rate,
        "topology_preservation_score": topology_preservation_score,
        "directed_representable_edge_rate": directed_representable_edge_rate,
        "directed_edge_realization_rate": directed_edge_realization_rate,
        "directed_directionality_leak_rate": directed_directionality_leak_rate,
        "directed_edge_preservation_score": directed_edge_preservation_score,
    }


def _paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    if deltas.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    means = np.mean(deltas[idx], axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def _paired_sign_permutation_pvalue(
    deltas: np.ndarray,
    *,
    n_perm: int = 5000,
    seed: int = 0,
) -> float:
    if deltas.size == 0:
        return 1.0
    observed = abs(float(np.mean(deltas)))
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    abs_d = np.abs(deltas)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, deltas.size))
    perm_means = np.mean(signs * abs_d[None, :], axis=1)
    p = (1.0 + float(np.sum(np.abs(perm_means) >= observed))) / float(n_perm + 1)
    return float(p)


def _benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR-adjusted p-values (q-values).
    """
    arr = np.asarray([float(p) for p in p_values], dtype=np.float64)
    n = int(arr.size)
    if n <= 0:
        return []
    order = np.argsort(arr)
    ranked = arr[order]
    q = np.zeros(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = float(i + 1)
        raw = float(ranked[i]) * float(n) / rank
        prev = min(prev, raw)
        q[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return [float(v) for v in out.tolist()]


def _json_sanitize(value: Any) -> Any:
    """Convert numpy/pandas scalars and non-finite floats to strict JSON values."""
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, np.ndarray):
        return [_json_sanitize(v) for v in value.tolist()]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_sanitize(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_sanitize(v) for v in value]
    return value


def _design_notes_for_config(cfg: ExperimentConfig) -> Dict[str, Any]:
    notes = dict(ABLATION_DESIGN_NOTES.get(cfg.name, {}))
    if not notes:
        if cfg.name.startswith("VQ_CODEBOOK_"):
            notes = {
                "tier": "block_ii",
                "component": "VQ codebook capacity",
                "comparison": "FULL and other VQ_CODEBOOK variants",
                "isolates": "categorical sampler codebook cap while keeping the rest of the stack matched",
                "interpretation": "Screens tokenizer capacity effects on reconstruction, diversity, and solvability proxies.",
            }
        elif cfg.name.startswith("LOGIC_G_"):
            notes = {
                "tier": "block_v",
                "component": "LogicNet guidance scale",
                "comparison": "FULL and NO_LOGIC",
                "isolates": "runtime logic-guidance strength",
                "interpretation": "Identifies whether guidance has a useful range or destabilizes decoding.",
            }
        elif cfg.name.startswith("LOGIC_ACTIVE_"):
            notes = {
                "tier": "block_v",
                "component": "LogicNet guidance timing",
                "comparison": "FULL and NO_LOGIC",
                "isolates": "fraction of reverse diffusion timesteps receiving logic guidance",
                "interpretation": "Tests whether late-timestep guidance stabilizes samples better than full-trajectory guidance.",
            }
        elif cfg.name.startswith("DIFFUSION_TOPO_"):
            notes = {
                "tier": "block_iii",
                "component": "diffusion topology conditioning",
                "comparison": "DIFFUSION_TOPO_ADDITIVE and DIFFUSION_TOPO_SPADE",
                "isolates": "conditioning injection style while keeping topology, sampler, and repair stack matched",
                "interpretation": "Tests whether SPADE-style affine topology modulation carries more useful structural signal than additive maps.",
            }
        else:
            notes = {
                "tier": "unspecified",
                "component": "custom ablation",
                "comparison": "FULL",
                "isolates": "configuration-defined difference",
                "interpretation": "Inspect paired deltas and confidence intervals before making a claim.",
            }

    return {
        "name": cfg.name,
        "tier": notes["tier"],
        "component": notes["component"],
        "comparison": notes["comparison"],
        "isolates": notes["isolates"],
        "primary_metrics": list(PRIMARY_ABLATION_METRICS),
        "interpretation": notes["interpretation"],
        "config": asdict(cfg),
    }


def build_ablation_plan(
    *,
    configs: Sequence[ExperimentConfig],
    seeds: Sequence[int],
    target_curve: Sequence[float],
    num_rooms: int,
    diffusion_steps: int,
    cbs_timeout: int,
    evolution_population: int,
    evolution_generations: int,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return a reproducible ablation-study plan without executing generation."""
    return {
        "protocol": "fixed_seed_paired_ablation",
        "design_goal": (
            "Estimate component necessity by changing one interpretable subsystem at a time "
            "where possible, using shared seeds and paired significance tests against FULL."
        ),
        "config_path": str(config_path) if config_path is not None else None,
        "seeds": [int(s) for s in seeds],
        "runtime_budget": {
            "num_rooms": int(num_rooms),
            "target_curve": [float(v) for v in target_curve],
            "diffusion_steps": int(diffusion_steps),
            "cbs_timeout": int(cbs_timeout),
            "evolution_population": int(evolution_population),
            "evolution_generations": int(evolution_generations),
        },
        "paired_statistics": {
            "baseline": "FULL",
            "confidence_interval": "paired bootstrap over seed deltas",
            "p_value": "random-sign permutation over paired seed deltas",
            "multiple_comparison_control": "Benjamini-Hochberg FDR over exported p-values",
        },
        "metrics": list(PRIMARY_ABLATION_METRICS),
        "experiments": [_design_notes_for_config(cfg) for cfg in configs],
        "claim_boundaries": [
            "RANDOM_TOPOLOGY is the strict topology null; NO_EVOLUTION is direct grammar generation.",
            "PURE_WFC bypasses neural room priors but still applies a deterministic graph-role scaffold; it is a scaffolded symbolic control, not unconstrained pure WFC.",
            "Single-seed or quick-profile results are screening evidence; thesis claims should use paired multi-seed runs.",
        ],
    }


def _format_ablation_plan_markdown(plan: Dict[str, Any]) -> str:
    lines = [
        "# Ablation Study Plan",
        "",
        f"Protocol: `{plan['protocol']}`",
        "",
        plan["design_goal"],
        "",
        "## Runtime Budget",
        "",
    ]
    for key, value in plan["runtime_budget"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Paired Statistics", ""])
    for key, value in plan["paired_statistics"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Experiments", ""])
    for exp in plan["experiments"]:
        lines.extend(
            [
                f"### {exp['name']}",
                f"- Tier: `{exp['tier']}`",
                f"- Component: {exp['component']}",
                f"- Comparison: {exp['comparison']}",
                f"- Isolates: {exp['isolates']}",
                f"- Interpretation: {exp['interpretation']}",
                "",
            ]
        )

    lines.extend(["## Claim Boundaries", ""])
    for item in plan["claim_boundaries"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


class AblationStudy:
    def __init__(
        self,
        *,
        output_dir: Path,
        data_root: Path,
        num_rooms: int,
        target_curve: Sequence[float],
        diffusion_steps: int,
        astar_timeout: int = 200000,
        cbs_timeout: int,
        evolution_population: int,
        evolution_generations: int,
        config_path: Optional[Path] = None,
        vqvae_checkpoint: Optional[str] = None,
        diffusion_checkpoint: Optional[str] = None,
        masked_room_checkpoint: Optional[str] = None,
        logic_net_checkpoint: Optional[str] = None,
        condition_encoder_checkpoint: Optional[str] = None,
        max_runtime_sec: Optional[float] = None,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_root = data_root
        self.config_path = Path(config_path) if config_path is not None else None
        self.num_rooms = int(num_rooms)
        self.target_curve = list(float(v) for v in target_curve)
        self.diffusion_steps = int(diffusion_steps)
        self.astar_timeout = int(astar_timeout)
        self.cbs_timeout = int(cbs_timeout)
        self.evolution_population = int(evolution_population)
        self.evolution_generations = int(evolution_generations)
        self.vqvae_checkpoint = str(vqvae_checkpoint) if vqvae_checkpoint else None
        self.diffusion_checkpoint = str(diffusion_checkpoint) if diffusion_checkpoint else None
        self.masked_room_checkpoint = str(masked_room_checkpoint) if masked_room_checkpoint else None
        inferred_logic_checkpoint = _infer_logic_checkpoint(self.diffusion_checkpoint)
        self.logic_net_checkpoint = str(logic_net_checkpoint) if logic_net_checkpoint else inferred_logic_checkpoint
        self.condition_encoder_checkpoint = (
            str(condition_encoder_checkpoint) if condition_encoder_checkpoint else None
        )
        self.max_runtime_sec = float(max_runtime_sec) if max_runtime_sec is not None else None
        self.resolved_config = _load_pipeline_resolved_config(
            str(self.config_path) if self.config_path is not None else None,
            self.diffusion_checkpoint,
            self.masked_room_checkpoint,
            self.logic_net_checkpoint,
            self.condition_encoder_checkpoint,
            self.vqvae_checkpoint,
        )
        self.pipeline_runtime_kwargs: Dict[str, Any] = {}
        if isinstance(self.resolved_config, dict):
            self.pipeline_runtime_kwargs.update(pipeline_kwargs_from_resolved_config(self.resolved_config))

        self.reference_graphs = load_vglc_reference_graphs(self.data_root, limit=64)
        ref_rooms = load_vglc_reference_rooms(self.data_root, max_rooms=256)
        self.reference_rooms = list(ref_rooms)
        self.reference_tile_dist = _tile_distribution(ref_rooms)
        self.reference_vectors = (
            np.stack([_descriptor_vector(g) for g in self.reference_graphs], axis=0)
            if self.reference_graphs
            else np.zeros((0, 4), dtype=np.float64)
        )

        self._pipeline_cache: Dict[str, NeuralSymbolicDungeonPipeline] = {}
        self._constraint_grammar = MissionGrammar(seed=2026)
        self._wfc_tile_priors: Optional[Dict[int, Any]] = None

    def _get_pipeline(self, cfg: ExperimentConfig) -> NeuralSymbolicDungeonPipeline:
        room_generator_mode = str(getattr(cfg, "room_generator_mode", "latent_diffusion")).strip().lower()
        diffusion_checkpoint = str(
            getattr(cfg, "diffusion_checkpoint_override", None) or self.diffusion_checkpoint
        )
        topology_conditioning_mode = getattr(cfg, "diffusion_topology_conditioning_mode", None)
        if topology_conditioning_mode is not None:
            topology_conditioning_mode = str(topology_conditioning_mode).strip().lower()
        cache_key = "|".join(
            [
                room_generator_mode,
                str(diffusion_checkpoint),
                str(topology_conditioning_mode or ""),
            ]
        )
        if cache_key not in self._pipeline_cache:
            pipeline_kwargs = dict(self.pipeline_runtime_kwargs)
            pipeline_kwargs.update(
                {
                    "vqvae_checkpoint": self.vqvae_checkpoint,
                    "diffusion_checkpoint": diffusion_checkpoint,
                    "masked_room_checkpoint": self.masked_room_checkpoint,
                    "logic_net_checkpoint": self.logic_net_checkpoint,
                    "condition_encoder_checkpoint": self.condition_encoder_checkpoint,
                    "room_generator_mode": room_generator_mode,
                    "condition_use_reference_room_maps": True,
                    "device": "auto",
                    "use_learned_refiner_rules": True,
                    "enable_logging": False,
                }
            )
            if topology_conditioning_mode:
                fallback_config = dict(pipeline_kwargs.get("diffusion_fallback_config") or {})
                fallback_config["topology_conditioning_mode"] = topology_conditioning_mode
                pipeline_kwargs["diffusion_fallback_config"] = fallback_config
            pipeline = NeuralSymbolicDungeonPipeline(
                **pipeline_kwargs,
            )
            validate_loaded_topology_conditioning_mode(
                pipeline,
                expected_mode=topology_conditioning_mode,
                checkpoint_path=diffusion_checkpoint,
            )
            self._pipeline_cache[cache_key] = pipeline
        return self._pipeline_cache[cache_key]

    def _build_non_evolution_graph(self, seed: int) -> nx.Graph:
        grammar = MissionGrammar(seed=seed)
        graph = grammar.generate(
            difficulty=Difficulty.MEDIUM,
            num_rooms=self.num_rooms,
            max_keys=max(1, self.num_rooms // 4),
            validate_all=True,
        )
        return mission_graph_to_networkx(graph)

    def _build_evolution_graph(self, seed: int) -> nx.Graph:
        generator = EvolutionaryTopologyGenerator(
            target_curve=self.target_curve,
            population_size=self.evolution_population,
            generations=self.evolution_generations,
            max_nodes=self.num_rooms,
            seed=seed,
        )
        return generator.evolve(directed_output=True)

    def _build_random_topology_graph(self, seed: int) -> nx.DiGraph:
        rng = np.random.default_rng(seed)
        n = int(max(3, self.num_rooms))
        graph = nx.DiGraph()

        for node_id in range(n):
            if node_id == 0:
                graph.add_node(
                    node_id,
                    label="S",
                    type="start",
                    is_start=True,
                    is_entry=True,
                )
            elif node_id == n - 1:
                graph.add_node(
                    node_id,
                    label="T",
                    type="goal",
                    is_goal=True,
                    is_triforce=True,
                )
            else:
                node_type = str(rng.choice(["enemy", "puzzle", "item", "empty"]))
                graph.add_node(node_id, label=node_type.upper(), type=node_type)

        # Backbone chain ensures at least one directed start->goal path.
        for u in range(n - 1):
            graph.add_edge(u, u + 1, label="open", edge_type="open")

        # Add random forward edges for branching without introducing directed cycles.
        p_extra = float(np.clip(2.5 / max(1, n), 0.08, 0.45))
        for u in range(n - 2):
            for v in range(u + 2, n):
                if float(rng.random()) < p_extra:
                    graph.add_edge(u, v, label="open", edge_type="open")

        return graph

    def _build_mission_graph(self, cfg: ExperimentConfig, seed: int) -> nx.Graph:
        if bool(cfg.random_topology):
            return self._build_random_topology_graph(seed=seed)
        if bool(cfg.use_evolution):
            return self._build_evolution_graph(seed=seed)
        return self._build_non_evolution_graph(seed=seed)

    def _get_wfc_tile_priors(self) -> Dict[int, Any]:
        if self._wfc_tile_priors is None:
            training_rooms = self.reference_rooms if self.reference_rooms else [
                np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
            ]
            self._wfc_tile_priors = extract_tile_priors_from_vqvae(
                vqvae_codebook=np.zeros((1, 1), dtype=np.float32),
                training_grids=training_rooms,
            )
        return self._wfc_tile_priors

    def _generate_dungeon_pure_wfc(
        self,
        *,
        graph: nx.Graph,
        seed: int,
        prior_mode: str = "weighted",
    ) -> Any:
        prior_mode = str(prior_mode or "weighted").strip().lower()
        if prior_mode not in {"weighted", "flat"}:
            raise ValueError(f"wfc_prior_mode must be 'weighted' or 'flat', got {prior_mode!r}.")
        tile_priors = self._get_wfc_tile_priors()
        wfc_cfg = WeightedBayesianWFCConfig(
            use_vqvae_priors=(prior_mode == "weighted"),
            enable_backtracking=True,
            max_backtracks=192,
            max_restarts=2,
        )

        rooms: Dict[int, RoomGenerationResult] = {}
        latent_h = max(1, ROOM_HEIGHT // 4)
        latent_w = max(1, (ROOM_WIDTH + 3) // 4)

        for room_id in sorted(list(graph.nodes()), key=lambda x: str(x)):
            room_seed = int(seed + stable_seed_offset(room_id, modulo=100000))
            wfc = WeightedBayesianWFC(
                width=ROOM_WIDTH,
                height=ROOM_HEIGHT,
                tile_priors=tile_priors,
                config=wfc_cfg,
                seed=room_seed,
            )
            room_grid = wfc.generate(seed=room_seed)
            room_grid, scaffold_tiles = _apply_symbolic_room_scaffold(
                room_grid,
                dict(graph.nodes[room_id]),
            )
            latent = torch.zeros((1, 1, latent_h, latent_w), dtype=torch.float32)
            rooms[int(room_id) if isinstance(room_id, int) else room_id] = RoomGenerationResult(
                room_id=int(room_id) if isinstance(room_id, int) else 0,
                room_grid=np.asarray(room_grid, dtype=np.int32),
                latent=latent,
                neural_grid=np.asarray(room_grid, dtype=np.int32),
                was_repaired=False,
                repair_mask=None,
                metrics={
                    "wfc_only": 1.0,
                    "wfc_prior_mode": prior_mode,
                    "symbolic_scaffold_tiles": float(scaffold_tiles),
                },
            )

        dungeon_grid = _stitch_symbolic_rooms(rooms, graph)
        metrics = {
            "num_rooms": len(rooms),
            "total_tiles_repaired": 0.0,
            "repair_rate": 0.0,
            "dungeon_shape": dungeon_grid.shape,
            "generation_time_sec": float("nan"),
            "wfc_only": 1.0,
            "wfc_prior_mode": prior_mode,
            "symbolic_scaffold_tiles": float(
                sum(room.metrics.get("symbolic_scaffold_tiles", 0.0) for room in rooms.values())
            ),
        }
        return SimpleNamespace(
            dungeon_grid=dungeon_grid,
            rooms=rooms,
            mission_graph=graph,
            metrics=metrics,
            map_elites_score=None,
            generation_time=float("nan"),
        )

    def _optimal_and_cbs_metrics(self, grid: np.ndarray, seed: int) -> Tuple[bool, float, float, float]:
        optimal_success = False
        optimal_len = 0
        oracle_status = "failed"
        cbs_success = False
        cbs_len = 0
        confusion_index = float("nan")
        try:
            env = ZeldaLogicEnv(semantic_grid=grid)
            oracle = run_astar_oracle(env, timeout=int(getattr(self, "astar_timeout", 200000)))
            optimal_success = bool(oracle["success"])
            optimal_len = int(oracle["path_length"])
            oracle_status = str(oracle["status"])
        except Exception:
            optimal_success = False
            optimal_len = 0
            oracle_status = "failed"

        try:
            cbs_success, cbs_path, _, cbs_metrics = solve_with_cbs(
                grid,
                persona="balanced",
                timeout=self.cbs_timeout,
                seed=seed,
            )
            cbs_len = max(0, len(cbs_path or []) - 1)
            confusion_index = float(cbs_metrics.confusion_index)
            if not cbs_success:
                cbs_len = 0
        except Exception:
            cbs_len = 0

        confusion_ratio = confusion_ratio_vs_oracle(
            optimal_len,
            cbs_len,
            oracle_status=oracle_status,
            candidate_success=cbs_success,
        )

        if optimal_success and cbs_success and cbs_len == 0:
            path_optimal = 1.0 if optimal_len == 0 else 0.0
        elif optimal_success and cbs_success and cbs_len > 0:
            path_optimal = float(max(0.0, min(1.0, optimal_len / cbs_len)))
        else:
            path_optimal = 0.0
        return bool(optimal_success), float(confusion_ratio), float(path_optimal), float(confusion_index)

    def _vq_reconstruction_error(self, pipeline: NeuralSymbolicDungeonPipeline, room_grid: np.ndarray) -> float:
        try:
            num_classes = int(getattr(pipeline.vqvae, "num_classes", 44))
            h, w = room_grid.shape
            x = np.zeros((1, num_classes, h, w), dtype=np.float32)
            clipped = np.clip(room_grid.astype(np.int64), 0, num_classes - 1)
            for r in range(h):
                for c in range(w):
                    x[0, int(clipped[r, c]), r, c] = 1.0
            xt = torch.from_numpy(x).to(pipeline.device)
            with torch.no_grad():
                z_q, _ = pipeline.vqvae.encode(xt)
                logits = pipeline.vqvae.decode(z_q, target_size=(h, w))
                recon = logits.argmax(dim=1).detach().cpu().numpy()[0]
            return float(np.mean(recon != clipped))
        except Exception:
            return float("nan")

    def _run_single(self, cfg: ExperimentConfig, seed: int) -> Dict[str, Any]:
        started = time.time()
        row: Dict[str, Any] = {
            "config": cfg.name,
            "seed": int(seed),
            "success": False,
            "solvable": False,
            "confusion_ratio": np.nan,
            "confusion_index": np.nan,
            "path_optimal": 0.0,
            "tile_prior_kl": np.nan,
            "graph_edit_distance": np.nan,
            "generation_time_sec": np.nan,
            "novelty": np.nan,
            "reconstruction_error": np.nan,
            "constraint_valid": np.nan,
            "room_repair_rate": np.nan,
            "tiles_repaired": np.nan,
            "symbolic_scaffold_tiles": np.nan,
            "topology_representable_edge_rate": np.nan,
            "topology_edge_connection_recall": np.nan,
            "topology_phantom_connection_rate": np.nan,
            "topology_preservation_score": np.nan,
            "directed_representable_edge_rate": np.nan,
            "directed_edge_realization_rate": np.nan,
            "directed_directionality_leak_rate": np.nan,
            "directed_edge_preservation_score": np.nan,
            "raw_topology_preservation_score": np.nan,
            "raw_directed_directionality_leak_rate": np.nan,
            "raw_topology_failed": False,
            "raw_topology_error_count": 0,
            "raw_topology_error": "",
            "topology_attention_pairs": np.nan,
            "topology_message_pairs": np.nan,
            "topology_shortest_path_bias_ops": np.nan,
            "topology_relative_attention_pairs_to_gat2": np.nan,
            "error": "",
        }

        try:
            mission_graph = self._build_mission_graph(cfg, seed=seed)
            pipeline = None if bool(cfg.pure_wfc) else self._get_pipeline(cfg)
            topology_cost = CrossAttention.topology_refinement_metrics(
                num_nodes=int(mission_graph.number_of_nodes()),
                num_edges=int(mission_graph.number_of_edges()),
                mode=str(cfg.topology_refinement_mode),
            )
            row.update(
                {
                    "topology_attention_pairs": float(topology_cost["attention_pairs"]),
                    "topology_message_pairs": float(topology_cost["message_pairs"]),
                    "topology_shortest_path_bias_ops": float(topology_cost["shortest_path_bias_ops"]),
                    "topology_relative_attention_pairs_to_gat2": float(
                        topology_cost["relative_attention_pairs_to_gat2"]
                    ),
                }
            )

            if bool(cfg.pure_wfc):
                result = self._generate_dungeon_pure_wfc(
                    graph=mission_graph,
                    seed=seed,
                    prior_mode=cfg.wfc_prior_mode,
                )
            else:
                assert pipeline is not None
                original_graph_token_flag = bool(getattr(pipeline, "use_graph_node_cross_attention", True))
                original_topology_mode = str(getattr(pipeline.diffusion, "get_topology_refinement_mode", lambda: "gat2")())
                original_reference_flag = bool(getattr(getattr(pipeline, "condition_encoder", None), "use_reference_room_maps", False))
                original_guidance_active_fraction = float(
                    getattr(getattr(pipeline.diffusion, "guidance", object()), "active_fraction", 1.0)
                )
                pipeline.use_graph_node_cross_attention = not bool(cfg.disable_graph_node_cross_attention)
                if cfg.use_reference_room_maps is not None and getattr(pipeline, "condition_encoder", None) is not None:
                    reference_encoder = getattr(pipeline.condition_encoder, "reference_room_encoder", None)
                    if bool(cfg.use_reference_room_maps) and reference_encoder is None:
                        raise RuntimeError(
                            "Room-branch benchmark requested reference-room conditioning, "
                            "but the loaded condition encoder has no reference encoder."
                        )
                    pipeline.condition_encoder.use_reference_room_maps = bool(cfg.use_reference_room_maps)
                set_topology_refinement_mode_or_raise(
                    pipeline.diffusion,
                    cfg.topology_refinement_mode,
                )
                if cfg.logic_guidance_active_fraction is not None:
                    pipeline.diffusion.guidance.active_fraction = float(
                        max(0.05, min(1.0, float(cfg.logic_guidance_active_fraction)))
                    )
                result = pipeline.generate_dungeon(
                    mission_graph=mission_graph,
                    generate_topology=False,
                    target_curve=self.target_curve,
                    num_rooms=self.num_rooms,
                    population_size=self.evolution_population,
                    generations=self.evolution_generations,
                    guidance_scale=7.5,
                    logic_guidance_scale=float(cfg.logic_guidance_scale),
                    num_diffusion_steps=self.diffusion_steps,
                    latent_sampler=cfg.latent_sampler,
                    categorical_codebook_size=cfg.categorical_codebook_size,
                    use_topological_positional_encoding=bool(cfg.use_tpe),
                    apply_repair=bool(cfg.use_wfc),
                    enable_map_elites=False,
                    seed=seed,
                )

                pipeline.use_graph_node_cross_attention = original_graph_token_flag
                if getattr(pipeline, "condition_encoder", None) is not None:
                    pipeline.condition_encoder.use_reference_room_maps = original_reference_flag
                set_topology_refinement_mode_or_raise(
                    pipeline.diffusion,
                    original_topology_mode,
                )
                pipeline.diffusion.guidance.active_fraction = original_guidance_active_fraction

            grid = np.asarray(result.dungeon_grid, dtype=np.int32)
            graph = result.mission_graph
            # Build a stitched dungeon from raw neural outputs (before symbolic repair)
            try:
                neural_rooms: Dict[Any, RoomGenerationResult] = {}
                for rid, r in result.rooms.items():
                    neural_rooms[rid] = RoomGenerationResult(
                        room_id=r.room_id,
                        room_grid=np.asarray(r.neural_grid, dtype=np.int32),
                        latent=r.latent,
                        neural_grid=np.asarray(r.neural_grid, dtype=np.int32),
                        was_repaired=False,
                        repair_mask=None,
                        neural_probs=getattr(r, 'neural_probs', None),
                        metrics=(r.metrics if isinstance(r.metrics, dict) else {}),
                    )
                neural_grid_global = np.asarray(
                    _stitch_symbolic_rooms(neural_rooms, graph)
                    if pipeline is None
                    else _stitch_with_pipeline(pipeline, neural_rooms, graph),
                    dtype=np.int32,
                )
                raw_topology_scorecard = _topology_information_scorecard(
                    graph=graph,
                    rooms=neural_rooms,
                    dungeon_grid=neural_grid_global,
                )
            except (AttributeError, RuntimeError, ValueError, TypeError, KeyError, IndexError) as e:
                neural_grid_global = None
                raw_topology_scorecard = None
                row["raw_topology_failed"] = True
                row["raw_topology_error_count"] = int(row.get("raw_topology_error_count", 0)) + 1
                row["raw_topology_error"] = _sanitized_exception_name(e)
                logger.warning(
                    "Failed to compute raw topology scorecard for config=%s seed=%d (%s)",
                    cfg.name,
                    int(seed),
                    _sanitized_exception_name(e),
                )
            desc_vec = _descriptor_vector(graph)

            tile_kl = _kl_divergence(self.reference_tile_dist, _tile_distribution([grid]))
            graph_ged = _nearest_graph_edit_distance(graph, self.reference_graphs, max_refs=24)

            novelty = 0.0
            if self.reference_vectors.size > 0:
                nearest = np.min(np.linalg.norm(self.reference_vectors - desc_vec[None, :], axis=1))
                novelty = float(nearest / np.sqrt(desc_vec.shape[0]))

            solvable, confusion_ratio, path_optimal, confusion_index = self._optimal_and_cbs_metrics(grid, seed=seed)

            first_room = next(iter(result.rooms.values())).room_grid if result.rooms else grid
            recon_error = (
                float("nan")
                if pipeline is None
                else self._vq_reconstruction_error(pipeline, np.asarray(first_room, dtype=np.int32))
            )
            constraint_valid = float("nan")
            try:
                mission = networkx_to_mission_graph(graph)
                mission.sanitize()
                constraint_valid = float(
                    self._constraint_grammar.validate_lock_key_ordering(mission)
                    and self._constraint_grammar.validate_progression_constraints(mission)
                )
            except Exception:
                constraint_valid = float("nan")

            room_repair_rate = float(result.metrics.get("repair_rate", float("nan")))
            tiles_repaired = float(result.metrics.get("total_tiles_repaired", float("nan")))
            symbolic_scaffold_tiles = float(result.metrics.get("symbolic_scaffold_tiles", float("nan")))
            topology_scorecard = _topology_information_scorecard(
                graph=graph,
                rooms=result.rooms,
                dungeon_grid=grid,
            )
            diffusion_model = getattr(pipeline, "diffusion", None) if pipeline is not None else None
            model_parameter_count = (
                int(sum(parameter.numel() for parameter in diffusion_model.parameters()))
                if diffusion_model is not None
                else 0
            )

            row.update(
                {
                    "success": True,
                    "solvable": bool(solvable),
                    "confusion_ratio": float(confusion_ratio),
                    "confusion_index": float(confusion_index),
                    "path_optimal": float(path_optimal),
                    "tile_prior_kl": float(tile_kl),
                    "graph_edit_distance": float(graph_ged),
                    "generation_time_sec": float(time.time() - started),
                    "novelty": float(novelty),
                    "reconstruction_error": float(recon_error),
                    "constraint_valid": float(constraint_valid),
                    "room_repair_rate": float(room_repair_rate),
                    "tiles_repaired": float(tiles_repaired),
                    "symbolic_scaffold_tiles": symbolic_scaffold_tiles,
                    "model_parameter_count": model_parameter_count,
                    "topology_representable_edge_rate": float(topology_scorecard["topology_representable_edge_rate"]),
                    "topology_edge_connection_recall": float(topology_scorecard["topology_edge_connection_recall"]),
                    "topology_phantom_connection_rate": float(topology_scorecard["topology_phantom_connection_rate"]),
                    "topology_preservation_score": float(topology_scorecard["topology_preservation_score"]),
                    "directed_representable_edge_rate": float(topology_scorecard["directed_representable_edge_rate"]),
                    "directed_edge_realization_rate": float(topology_scorecard["directed_edge_realization_rate"]),
                    "directed_directionality_leak_rate": float(topology_scorecard["directed_directionality_leak_rate"]),
                    "directed_edge_preservation_score": float(topology_scorecard["directed_edge_preservation_score"]),
                    "raw_topology_preservation_score": float(raw_topology_scorecard["topology_preservation_score"]) if raw_topology_scorecard is not None else float("nan"),
                    "raw_directed_directionality_leak_rate": float(raw_topology_scorecard["directed_directionality_leak_rate"]) if raw_topology_scorecard is not None else float("nan"),
                    "_descriptor_vec": desc_vec.tolist(),
                }
            )
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            row["generation_time_sec"] = float(time.time() - started)
            row["error"] = f"{type(e).__name__}: {e}"
            try:
                if "pipeline" in locals():
                    pipeline.use_graph_node_cross_attention = original_graph_token_flag
                    if getattr(pipeline, "condition_encoder", None) is not None:
                        pipeline.condition_encoder.use_reference_room_maps = original_reference_flag
                    try:
                        set_topology_refinement_mode_or_raise(
                            pipeline.diffusion,
                            original_topology_mode,
                        )
                    except (AttributeError, RuntimeError, ValueError, TypeError) as restore_error:
                        for key, cached in list(self._pipeline_cache.items()):
                            if cached is pipeline:
                                del self._pipeline_cache[key]
                        logger.warning(
                            "Discarded cached ablation pipeline after topology-mode restoration failed: %s",
                            restore_error,
                        )
                    pipeline.diffusion.guidance.active_fraction = original_guidance_active_fraction
            except (AttributeError, RuntimeError, ValueError, TypeError) as restore_error:
                logger.debug("Failed to restore ablation runner state after error: %s", restore_error)
        return row

    def run(self, configs: Sequence[ExperimentConfig], seeds: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        records: List[Dict[str, Any]] = []
        descriptor_store: Dict[str, List[np.ndarray]] = {cfg.name: [] for cfg in configs}
        started = time.time()
        stop_early = False

        for cfg in configs:
            if stop_early:
                break
            logger.info("Running config=%s (%d seeds)", cfg.name, len(seeds))
            for seed in seeds:
                if self.max_runtime_sec is not None:
                    elapsed = float(time.time() - started)
                    if elapsed >= self.max_runtime_sec:
                        logger.warning(
                            "Stopping ablation early due to runtime budget (elapsed=%.1fs, budget=%.1fs)",
                            elapsed,
                            self.max_runtime_sec,
                        )
                        stop_early = True
                        break
                row = self._run_single(cfg, int(seed))
                vec = row.pop("_descriptor_vec", None)
                if vec is not None:
                    descriptor_store[cfg.name].append(np.asarray(vec, dtype=np.float64))
                records.append(row)

        df = pd.DataFrame(records)
        summary_rows: List[Dict[str, Any]] = []
        for cfg in configs:
            sub = df[df["config"] == cfg.name]
            successful = sub[sub["success"].astype(bool)] if len(sub) > 0 else sub
            def _mean_col(name: str, default: float = float("nan")) -> float:
                if len(sub) == 0 or name not in sub:
                    return float(default)
                return float(sub[name].mean(skipna=True))

            summary_rows.append(
                {
                    "config": cfg.name,
                    "n": int(len(sub)),
                    "success_rate": float(sub["success"].mean()) if len(sub) > 0 else 0.0,
                    "failure_rate": float(1.0 - sub["success"].mean()) if len(sub) > 0 else 0.0,
                    "solvability_rate": float(sub["solvable"].mean()) if len(sub) > 0 else 0.0,
                    "solvability_rate_successful_generations": (
                        float(successful["solvable"].mean()) if len(successful) > 0 else 0.0
                    ),
                    "confusion_ratio": float(sub["confusion_ratio"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "confusion_index": float(sub["confusion_index"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "path_optimal": float(sub["path_optimal"].mean(skipna=True)) if len(sub) > 0 else 0.0,
                    "tile_prior_kl": float(sub["tile_prior_kl"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "graph_edit_distance": float(sub["graph_edit_distance"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "generation_time_sec": float(sub["generation_time_sec"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "model_parameter_count": _mean_col("model_parameter_count", default=0.0),
                    "novelty": float(sub["novelty"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "reconstruction_error": float(sub["reconstruction_error"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "constraint_valid_rate": float(sub["constraint_valid"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "room_repair_rate": float(sub["room_repair_rate"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "tiles_repaired": float(sub["tiles_repaired"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "symbolic_scaffold_tiles": _mean_col("symbolic_scaffold_tiles"),
                    "topology_representable_edge_rate": float(sub["topology_representable_edge_rate"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "topology_edge_connection_recall": float(sub["topology_edge_connection_recall"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "topology_phantom_connection_rate": float(sub["topology_phantom_connection_rate"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "topology_preservation_score": float(sub["topology_preservation_score"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "directed_representable_edge_rate": float(sub["directed_representable_edge_rate"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "directed_edge_realization_rate": float(sub["directed_edge_realization_rate"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "directed_directionality_leak_rate": float(sub["directed_directionality_leak_rate"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "directed_edge_preservation_score": float(sub["directed_edge_preservation_score"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "topology_attention_pairs": _mean_col("topology_attention_pairs"),
                    "topology_message_pairs": _mean_col("topology_message_pairs"),
                    "topology_shortest_path_bias_ops": _mean_col("topology_shortest_path_bias_ops"),
                    "topology_relative_attention_pairs_to_gat2": _mean_col("topology_relative_attention_pairs_to_gat2"),
                    "diversity": float(_pairwise_diversity(descriptor_store.get(cfg.name, []))),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        return df, summary_df

    @staticmethod
    def significance_report(
        df: pd.DataFrame,
        *,
        baseline: str = "FULL",
        metrics: Optional[Sequence[str]] = None,
        seed: int = 0,
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = [
                "solvable",
                "confusion_ratio",
                "confusion_index",
                "path_optimal",
                "tile_prior_kl",
                "graph_edit_distance",
                "generation_time_sec",
                "novelty",
                "reconstruction_error",
                "constraint_valid",
                "room_repair_rate",
                "tiles_repaired",
                "topology_representable_edge_rate",
                "topology_edge_connection_recall",
                "topology_phantom_connection_rate",
                "topology_preservation_score",
                "directed_representable_edge_rate",
                "directed_edge_realization_rate",
                "directed_directionality_leak_rate",
                "directed_edge_preservation_score",
            ]

        rows: List[Dict[str, Any]] = []
        base = df[df["config"] == baseline]
        other_configs = [c for c in sorted(df["config"].unique()) if c != baseline]

        for cfg in other_configs:
            other = df[df["config"] == cfg]
            merged = base.merge(other, on="seed", suffixes=("_base", "_cfg"))
            if merged.empty:
                continue
            for i, metric in enumerate(metrics):
                bcol = f"{metric}_base"
                ccol = f"{metric}_cfg"
                if bcol not in merged.columns or ccol not in merged.columns:
                    continue
                left = merged[ccol].astype(np.float64)
                right = merged[bcol].astype(np.float64)
                deltas = (left - right).to_numpy(dtype=np.float64)
                deltas = deltas[np.isfinite(deltas)]
                if deltas.size == 0:
                    continue
                mean_delta = float(np.mean(deltas))
                ci_low, ci_high = _paired_bootstrap_ci(
                    deltas,
                    n_boot=2000,
                    alpha=0.05,
                    seed=seed + 17 * (i + 1),
                )
                p_value = _paired_sign_permutation_pvalue(
                    deltas,
                    n_perm=4000,
                    seed=seed + 31 * (i + 1),
                )
                std = float(np.std(deltas))
                effect = float(mean_delta / std) if std > 1e-9 else 0.0
                rows.append(
                    {
                        "config": cfg,
                        "metric": metric,
                        "n_pairs": int(deltas.size),
                        "delta_mean_cfg_minus_full": mean_delta,
                        "delta_ci_low": ci_low,
                        "delta_ci_high": ci_high,
                        "p_value": p_value,
                        "effect_size_d": effect,
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        q_values = _benjamini_hochberg(out["p_value"].astype(float).tolist())
        out["p_value_bh_fdr"] = q_values
        out["significant_fdr_0_05"] = out["p_value_bh_fdr"] < 0.05
        return out

    def export(
        self,
        *,
        configs: Sequence[ExperimentConfig],
        seeds: Sequence[int],
        raw_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        sig_df: pd.DataFrame,
    ) -> None:
        def _fmt_table(df: pd.DataFrame) -> str:
            try:
                return df.to_markdown(index=False)
            except Exception:
                return df.to_string(index=False)

        raw_path = self.output_dir / "ablation_raw.csv"
        summary_path = self.output_dir / "ablation_summary.csv"
        sig_path = self.output_dir / "ablation_significance.csv"
        json_path = self.output_dir / "ablation_report.json"
        md_path = self.output_dir / "ablation_report.md"
        plan_json_path = self.output_dir / "ablation_plan.json"
        plan_md_path = self.output_dir / "ablation_plan.md"

        raw_df.to_csv(raw_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        sig_df.to_csv(sig_path, index=False)

        plan = build_ablation_plan(
            configs=configs,
            seeds=seeds,
            target_curve=self.target_curve,
            num_rooms=self.num_rooms,
            diffusion_steps=self.diffusion_steps,
            cbs_timeout=self.cbs_timeout,
            evolution_population=self.evolution_population,
            evolution_generations=self.evolution_generations,
            config_path=self.config_path,
        )
        plan_json_path.write_text(
            json.dumps(_json_sanitize(plan), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        plan_md_path.write_text(_format_ablation_plan_markdown(plan), encoding="utf-8")

        payload = {
            "plan": plan,
            "configs": [asdict(c) for c in configs],
            "seeds": list(int(s) for s in seeds),
            "summary": summary_df.to_dict(orient="records"),
            "significance": sig_df.to_dict(orient="records"),
        }
        json_path.write_text(
            json.dumps(_json_sanitize(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )

        lines = [
            "# Ablation Study Report",
            "",
            "## Configurations",
        ]
        lines.extend(
            [
                "",
                "Study design is exported separately as `ablation_plan.md` and `ablation_plan.json`.",
                "",
            ]
        )
        for cfg in configs:
            lines.append(f"- `{cfg.name}`: {asdict(cfg)}")
        lines.extend(
            [
                "",
                "## Summary Metrics",
                "",
                _fmt_table(summary_df),
                "",
                "## Paired Significance (vs FULL)",
                "",
                _fmt_table(sig_df) if not sig_df.empty else "_No paired comparisons available_",
            ]
        )
        md_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("Saved ablation outputs to %s", self.output_dir)


def build_experiment_set(include_extended: bool = True) -> List[ExperimentConfig]:
    core = [
        ExperimentConfig(name="FULL", topology_refinement_mode="gat2"),
        ExperimentConfig(name="TOPO_LIGHTWEIGHT", topology_refinement_mode="lightweight"),
        ExperimentConfig(name="TOPO_SPARSE_EDGE", topology_refinement_mode="sparse_edge"),
        ExperimentConfig(name="TOPO_SPARSE_DIRECTED", topology_refinement_mode="sparse_directed"),
        ExperimentConfig(name="TOPO_SPARSE_SEMANTIC", topology_refinement_mode="sparse_directed_semantic"),
        ExperimentConfig(name="NO_EVOLUTION", use_evolution=False),
        ExperimentConfig(name="RANDOM_TOPOLOGY", use_evolution=False, random_topology=True),
        ExperimentConfig(
            name="NO_GRAPH",
            use_tpe=False,
            disable_graph_node_cross_attention=True,
            topology_refinement_mode="none",
        ),
        ExperimentConfig(name="NO_WFC", use_wfc=False),
        ExperimentConfig(name="NO_LOGIC", logic_guidance_scale=0.0),
        ExperimentConfig(name="PURE_WFC", use_evolution=True, pure_wfc=True, use_wfc=False, logic_guidance_scale=0.0),
        ExperimentConfig(
            name="PURE_WFC_FLAT_PRIOR",
            use_evolution=True,
            pure_wfc=True,
            use_wfc=False,
            logic_guidance_scale=0.0,
            wfc_prior_mode="flat",
        ),
    ]
    if not include_extended:
        return core

    extended = [
        ExperimentConfig(name="TOPO_GRAPHORMER_STATIC", topology_refinement_mode="graphormer"),
        ExperimentConfig(name="TOPO_GRAPHORMER_LEARNED", topology_refinement_mode="graphormer_learned"),
        ExperimentConfig(name="VQ_CODEBOOK_128", latent_sampler="categorical", categorical_codebook_size=128),
        ExperimentConfig(name="VQ_CODEBOOK_512", latent_sampler="categorical", categorical_codebook_size=512),
        ExperimentConfig(name="VQ_CODEBOOK_2048", latent_sampler="categorical", categorical_codebook_size=2048),
        ExperimentConfig(name="LATENT_DIFFUSION", latent_sampler="diffusion"),
        ExperimentConfig(name="LATENT_CATEGORICAL", latent_sampler="categorical"),
        ExperimentConfig(name="COND_NO_TPE", use_tpe=False),
        ExperimentConfig(name="COND_WEAK_GRAPH", use_tpe=False, disable_graph_node_cross_attention=True),
        ExperimentConfig(name="LOGIC_G_0.25", logic_guidance_scale=0.25),
        ExperimentConfig(name="LOGIC_G_0.50", logic_guidance_scale=0.50),
        ExperimentConfig(name="LOGIC_G_1.50", logic_guidance_scale=1.50),
        ExperimentConfig(name="LOGIC_G_2.00", logic_guidance_scale=2.00),
        ExperimentConfig(name="LOGIC_ACTIVE_0.25", logic_guidance_active_fraction=0.25),
        ExperimentConfig(name="LOGIC_ACTIVE_0.50", logic_guidance_active_fraction=0.50),
        ExperimentConfig(name="LOGIC_ACTIVE_0.75", logic_guidance_active_fraction=0.75),
        ExperimentConfig(
            name="DIFFUSION_TOPO_ADDITIVE",
            diffusion_topology_conditioning_mode="additive",
        ),
        ExperimentConfig(
            name="DIFFUSION_TOPO_SPADE",
            diffusion_topology_conditioning_mode="spade",
        ),
    ]
    return core + extended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-seed thesis ablation protocol.")
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=Path("results/ablation"))
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional resolved/training YAML whose pipeline defaults should seed the ablation runtime.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--num-samples", type=int, default=8, help="Seeds per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for fixed-seed protocol")
    parser.add_argument("--num-rooms", type=int, default=8)
    parser.add_argument("--target-curve", type=str, default="0.2,0.4,0.6,0.8,0.7,0.5,0.3,0.2")
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--astar-timeout", type=int, default=200000)
    parser.add_argument("--cbs-timeout", type=int, default=120000)
    parser.add_argument("--evolution-population", type=int, default=24)
    parser.add_argument("--evolution-generations", type=int, default=30)
    parser.add_argument("--vqvae-checkpoint", type=str, default=None)
    parser.add_argument("--diffusion-checkpoint", type=str, default=None)
    parser.add_argument(
        "--diffusion-additive-checkpoint",
        type=str,
        default=None,
        help="Checkpoint trained with additive topology conditioning for the additive-vs-SPADE ablation.",
    )
    parser.add_argument(
        "--diffusion-spade-checkpoint",
        type=str,
        default=None,
        help="Checkpoint trained with SPADE topology conditioning for the additive-vs-SPADE ablation.",
    )
    parser.add_argument("--masked-room-checkpoint", type=str, default=None)
    parser.add_argument("--logic-net-checkpoint", type=str, default=None)
    parser.add_argument("--condition-encoder-checkpoint", type=str, default=None)
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        help="Optional wall-clock budget. If exceeded, stop and export partial results.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tractable quick profile for iterative thesis experiments.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write ablation_plan.json/md and exit without running generation.",
    )
    parser.add_argument(
        "--kaggle-t4x2",
        action="store_true",
        help="Apply a Kaggle T4 x2 preset for larger fixed-seed ablation runs.",
    )
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--configs", type=str, default="", help="Comma-separated subset of config names")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    target_curve = [float(v) for v in str(args.target_curve).split(",") if str(v).strip()]
    if args.quick:
        args.num_samples = min(int(args.num_samples), 2)
        args.diffusion_steps = min(int(args.diffusion_steps), 10)
        args.evolution_population = min(int(args.evolution_population), 12)
        args.evolution_generations = min(int(args.evolution_generations), 8)
        args.cbs_timeout = min(int(args.cbs_timeout), 30000)
        if args.max_runtime_sec is None:
            args.max_runtime_sec = 420.0
        logger.info(
            "Quick profile active: samples=%d, diffusion_steps=%d, pop=%d, gens=%d, cbs_timeout=%d, max_runtime_sec=%s",
            args.num_samples,
            args.diffusion_steps,
            args.evolution_population,
            args.evolution_generations,
            args.cbs_timeout,
            str(args.max_runtime_sec),
        )
    if args.kaggle_t4x2:
        args.num_samples = max(int(args.num_samples), 12)
        args.diffusion_steps = max(int(args.diffusion_steps), 25)
        args.evolution_population = max(int(args.evolution_population), 32)
        args.evolution_generations = max(int(args.evolution_generations), 40)
        args.cbs_timeout = max(int(args.cbs_timeout), 60000)
        if args.max_runtime_sec is None:
            args.max_runtime_sec = 10800.0
        logger.info(
            "Kaggle T4 x2 profile active: samples=%d, diffusion_steps=%d, pop=%d, gens=%d, cbs_timeout=%d, max_runtime_sec=%s",
            args.num_samples,
            args.diffusion_steps,
            args.evolution_population,
            args.evolution_generations,
            args.cbs_timeout,
            str(args.max_runtime_sec),
        )
    configs = build_experiment_set(include_extended=not args.core_only)
    if args.configs.strip():
        selected = {c.strip() for c in args.configs.split(",") if c.strip()}
        configs = [cfg for cfg in configs if cfg.name in selected]
        if not configs:
            raise ValueError("No matching configs after --configs filtering.")
    bind_topology_ablation_checkpoints(
        configs,
        default_diffusion_checkpoint=args.diffusion_checkpoint,
        additive_checkpoint=args.diffusion_additive_checkpoint,
        spade_checkpoint=args.diffusion_spade_checkpoint,
        require_existing=not args.plan_only,
    )

    seeds = [int(args.seed) + i for i in range(int(args.num_samples))]
    if args.plan_only:
        args.output.mkdir(parents=True, exist_ok=True)
        plan = build_ablation_plan(
            configs=configs,
            seeds=seeds,
            target_curve=target_curve,
            num_rooms=args.num_rooms,
            diffusion_steps=args.diffusion_steps,
            cbs_timeout=args.cbs_timeout,
            evolution_population=args.evolution_population,
            evolution_generations=args.evolution_generations,
            config_path=args.config,
        )
        (args.output / "ablation_plan.json").write_text(
            json.dumps(_json_sanitize(plan), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (args.output / "ablation_plan.md").write_text(
            _format_ablation_plan_markdown(plan),
            encoding="utf-8",
        )
        logger.info("Ablation plan written to %s", args.output)
        return 0

    study = AblationStudy(
        output_dir=args.output,
        data_root=args.data_root,
        config_path=args.config,
        num_rooms=args.num_rooms,
        target_curve=target_curve,
        diffusion_steps=args.diffusion_steps,
        astar_timeout=args.astar_timeout,
        cbs_timeout=args.cbs_timeout,
        evolution_population=args.evolution_population,
        evolution_generations=args.evolution_generations,
        vqvae_checkpoint=args.vqvae_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        masked_room_checkpoint=args.masked_room_checkpoint,
        logic_net_checkpoint=args.logic_net_checkpoint,
        condition_encoder_checkpoint=args.condition_encoder_checkpoint,
        max_runtime_sec=args.max_runtime_sec,
    )

    raw_df, summary_df = study.run(configs=configs, seeds=seeds)
    sig_df = study.significance_report(raw_df, baseline="FULL", seed=args.seed + 999)
    study.export(
        configs=configs,
        seeds=seeds,
        raw_df=raw_df,
        summary_df=summary_df,
        sig_df=sig_df,
    )

    logger.info("Ablation complete. Output: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
