"""Standardized publication benchmark and 4-fold aggregator for G-LMDM.

Computes Schema 2.0 4-stage attribution (raw -> constrained -> repaired -> compiled),
domain-specific interior tile distribution metrics (JS-divergence & TV vs VGLC),
memorization/novelty against the 45 unique VGLC training interiors (including V4
symmetry canonicalization), Expressive Range Analysis (ERA), topology realizability,
and matched inference compute efficiency (valid dungeons/sec).
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.definitions import DOOR_POSITIONS, ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE as P

INTERIOR_TRACKED_TILES: Tuple[Tuple[str, int], ...] = (
    ("FLOOR", int(P["FLOOR"])),
    ("BLOCK", int(P["BLOCK"])),
    ("ELEMENT", int(P["ELEMENT"])),
    ("ELEMENT_FLOOR", int(P["ELEMENT_FLOOR"])),
    ("ENEMY", int(P["ENEMY"])),
    ("WALL", int(P["WALL"])),
    ("OTHER", -1),
)

WALKABLE_TILES: frozenset[int] = frozenset(
    int(P[name])
    for name in (
        "FLOOR",
        "DOOR_OPEN",
        "DOOR_LOCKED",
        "DOOR_BOMB",
        "DOOR_PUZZLE",
        "DOOR_BOSS",
        "DOOR_SOFT",
        "START",
        "TRIFORCE",
        "KEY_SMALL",
        "KEY_BOSS",
        "KEY_ITEM",
        "ITEM_MINOR",
        "ENEMY",
        "BOSS",
        "PUZZLE",
        "STAIR",
        "ELEMENT_FLOOR",
    )
    if name in P
)


def extract_interior(room_grid: np.ndarray) -> np.ndarray:
    """Return the interior tile crop inside the 2-tile VGLC perimeter wall."""
    arr = np.asarray(room_grid, dtype=np.int32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] > 4 and arr.shape[1] > 4:
        return arr[2:-2, 2:-2]
    return arr


def canonical_v4_bytes(interior: np.ndarray) -> bytes:
    """Return the canonical byte key of an interior grid under the Klein V4 group."""
    arr = np.asarray(interior, dtype=np.int32)
    candidates = (
        arr.tobytes(),
        np.flip(arr, axis=1).copy().tobytes(),
        np.flip(arr, axis=0).copy().tobytes(),
        np.flip(arr, axis=(0, 1)).copy().tobytes(),
    )
    return min(candidates)


def compute_interior_distribution(rooms: Sequence[np.ndarray]) -> Dict[str, float]:
    """Compute normalized interior tile frequencies across tracked tile classes."""
    counts = {name: 0 for name, _ in INTERIOR_TRACKED_TILES}
    known_ids = {tid: name for name, tid in INTERIOR_TRACKED_TILES if tid >= 0}
    total = 0
    for grid in rooms:
        inner = extract_interior(grid)
        total += int(inner.size)
        unique_ids, id_counts = np.unique(inner, return_counts=True)
        for tid, cnt in zip(unique_ids.tolist(), id_counts.tolist()):
            bucket = known_ids.get(int(tid), "OTHER")
            counts[bucket] += int(cnt)
    denom = float(max(1, total))
    return {name: float(counts[name]) / denom for name, _ in INTERIOR_TRACKED_TILES}


def _distribution_vector(dist: Mapping[str, float], keys: Sequence[str]) -> np.ndarray:
    vec = np.array([max(0.0, float(dist.get(k, 0.0))) for k in keys], dtype=np.float64)
    total = float(vec.sum())
    if total > 0.0:
        vec /= total
    return vec


def jensen_shannon_divergence(
    dist_p: Mapping[str, float],
    dist_q: Mapping[str, float],
    *,
    keys: Optional[Sequence[str]] = None,
) -> float:
    """Compute base-2 Jensen-Shannon divergence in [0, 1] between two tile distributions."""
    used_keys = list(keys or [name for name, _ in INTERIOR_TRACKED_TILES])
    p = _distribution_vector(dist_p, used_keys)
    q = _distribution_vector(dist_q, used_keys)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = float(np.sum(np.where(p > eps, p * np.log2((p + eps) / (m + eps)), 0.0)))
    kl_qm = float(np.sum(np.where(q > eps, q * np.log2((q + eps) / (m + eps)), 0.0)))
    return max(0.0, min(1.0, 0.5 * (kl_pm + kl_qm)))


def total_variation_distance(
    dist_p: Mapping[str, float],
    dist_q: Mapping[str, float],
    *,
    keys: Optional[Sequence[str]] = None,
) -> float:
    """Compute Total Variation distance 0.5 * ||p - q||_1 in [0, 1]."""
    used_keys = list(keys or [name for name, _ in INTERIOR_TRACKED_TILES])
    p = _distribution_vector(dist_p, used_keys)
    q = _distribution_vector(dist_q, used_keys)
    return float(0.5 * np.abs(p - q).sum())


def raw_doors_connected(grid: np.ndarray, required_doors: Mapping[str, bool]) -> bool:
    """Return True iff all active doorway sockets in ``required_doors`` are connected by walkable tiles."""
    arr = np.asarray(grid, dtype=np.int32)
    targets: List[List[Tuple[int, int]]] = []
    for direction, enabled in required_doors.items():
        if not enabled:
            continue
        spec = DOOR_POSITIONS[direction]
        if direction in ("N", "S"):
            cells = [(spec["row"], c) for c in range(spec["col_start"], spec["col_end"] + 1)]
        else:
            cells = [(r, spec["col"]) for r in range(spec["row_start"], spec["row_end"] + 1)]
        targets.append(cells)
    if len(targets) <= 1:
        return True

    q: deque[Tuple[int, int]] = deque(targets[0])
    visited = set(targets[0])
    h, w = int(arr.shape[0]), int(arr.shape[1])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and int(arr[nr, nc]) in WALKABLE_TILES:
                visited.add((nr, nc))
                q.append((nr, nc))
    return all(any(cell in visited for cell in group) for group in targets[1:])


def compute_novelty_metrics(
    generated_rooms: Sequence[np.ndarray],
    train_interiors: Sequence[np.ndarray],
) -> Dict[str, float]:
    """Measure memorization, normalized nearest-neighbor Hamming novelty, and uniqueness."""
    if not generated_rooms:
        return {
            "train_exact_match_rate": 0.0,
            "train_v4_match_rate": 0.0,
            "mean_min_hamming_to_train": 0.0,
            "generated_unique_interior_ratio": 0.0,
            "generated_unique_v4_ratio": 0.0,
        }

    gen_inners = [extract_interior(g) for g in generated_rooms]
    train_inners = [extract_interior(t) for t in train_interiors]
    train_exact = {t.tobytes() for t in train_inners}
    train_v4 = {canonical_v4_bytes(t) for t in train_inners}

    exact_matches = 0
    v4_matches = 0
    gen_exact_set = set()
    gen_v4_set = set()
    min_hamming_distances: List[float] = []

    train_flat = np.stack([t.reshape(-1) for t in train_inners], axis=0) if train_inners else None
    for inner in gen_inners:
        b = inner.tobytes()
        vb = canonical_v4_bytes(inner)
        gen_exact_set.add(b)
        gen_v4_set.add(vb)
        if b in train_exact:
            exact_matches += 1
        if vb in train_v4:
            v4_matches += 1
        if train_flat is not None and train_flat.shape[1] == inner.size:
            diffs = np.mean(train_flat != inner.reshape(1, -1), axis=1)
            min_hamming_distances.append(float(np.min(diffs)))
        else:
            min_hamming_distances.append(1.0)

    n = float(len(gen_inners))
    return {
        "train_exact_match_rate": float(exact_matches) / n,
        "train_v4_match_rate": float(v4_matches) / n,
        "mean_min_hamming_to_train": float(np.mean(min_hamming_distances)),
        "generated_unique_interior_ratio": float(len(gen_exact_set)) / n,
        "generated_unique_v4_ratio": float(len(gen_v4_set)) / n,
    }


def _load_vglc_train_reference(
    data_dir: str = "Data/The Legend of Zelda",
    train_dungeons: Sequence[int] = tuple(range(1, 9)),
    variants: Sequence[int] = (1, 2),
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    from src.zelda_data.zelda_loader import ZeldaRoomDataset

    ds = ZeldaRoomDataset(
        data_dir,
        normalize=False,
        load_graphs=False,
        dungeon_ids=list(train_dungeons),
        variants=list(variants),
    )
    rooms = []
    for i in range(len(ds)):
        item = ds[i]
        arr = item.numpy() if hasattr(item, "numpy") else np.asarray(item)
        if arr.ndim == 3:
            arr = arr[0]
        rooms.append(arr.astype(np.int32))
    return rooms, compute_interior_distribution(rooms)


def _extract_era_and_topology(res: Any, raw_rooms: Sequence[np.ndarray]) -> Tuple[float, float, float, float]:
    metrics = dict(getattr(res, "metrics", None) or {})
    stages = dict(metrics.get("end_to_end_validation_stages", {}) or {})
    tile_stage = dict(stages.get("tile_solvability", {}) or {})
    tile_metrics = dict(tile_stage.get("metrics", {}) or {})
    path_len = float(tile_metrics.get("path_length", metrics.get("path_length", 0.0)) or 0.0)

    # Compute structural linearity from critical path ratio and room leniency from interior obstacles.
    g = getattr(res, "physical_mission_graph", None)
    bipartite_ok = 0.0
    if isinstance(g, nx.Graph) and g.number_of_nodes() > 0:
        und = g.to_undirected()
        max_deg = max((d for _, d in und.degree()), default=0)
        bipartite_ok = float(nx.is_bipartite(und) and max_deg <= 4)

    advisory = dict(metrics.get("end_to_end_advisory_metrics", {}) or {})
    linearity = float(advisory.get("critical_path_linearity", advisory.get("linearity", 0.0)) or 0.0)
    if linearity == 0.0 and isinstance(g, nx.Graph) and g.number_of_nodes() > 1:
        linearity = float(min(1.0, g.number_of_nodes() / max(1.0, float(g.number_of_edges() + 1))))

    if raw_rooms:
        obstacle_fracs = []
        for rm in raw_rooms:
            inner = extract_interior(rm)
            obstacle = (inner == int(P["BLOCK"])) | (inner == int(P["ELEMENT"])) | (inner == int(P["ENEMY"])) | (inner == int(P["WALL"]))
            obstacle_fracs.append(1.0 - float(np.mean(obstacle)))
        leniency = float(np.mean(obstacle_fracs))
    else:
        leniency = 1.0
    return path_len, linearity, leniency, bipartite_ok


CHECKPOINT_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("best", "masked_room_best.pth"),
    ("best_rare_recall", "masked_room_best_rare_recall.pth"),
    ("final", "masked_room_final.pth"),
)


class _BaselineRoomProposalModel:
    """Non-neural deterministic room-proposal drop-in for fair baseline attribution."""

    def __init__(
        self,
        *,
        baseline_name: str,
        train_rooms: Sequence[np.ndarray],
        hidden_dim: int = 32,
        num_classes: int = 44,
    ) -> None:
        normalized = str(baseline_name).strip().lower()
        if normalized not in {"empty_floor", "vglc_template"}:
            raise ValueError(
                f"baseline_name must be 'empty_floor' or 'vglc_template', got {baseline_name!r}."
            )
        self.baseline_name = normalized
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self._train_interiors = [extract_interior(r) for r in train_rooms]
        self._call_counter = 0

    def sample(
        self,
        *,
        context: Any,
        graph_data: Optional[Mapping[str, Any]] = None,
        fixed_tokens: Optional[Any] = None,
        fixed_mask: Optional[Any] = None,
        seed: Optional[int] = None,
        return_sampling_metrics: bool = False,
        **kwargs: Any,
    ) -> Any:
        import torch

        _ = (graph_data, kwargs)
        batch_size = int(context.shape[0]) if hasattr(context, "shape") and len(context.shape) >= 1 else 1
        device = context.device if isinstance(context, torch.Tensor) else torch.device("cpu")
        wall_id = int(P["WALL"])
        floor_id = int(P["FLOOR"])
        tokens = torch.full(
            (batch_size, ROOM_HEIGHT, ROOM_WIDTH),
            fill_value=floor_id,
            device=device,
            dtype=torch.long,
        )
        tokens[:, :2, :] = wall_id
        tokens[:, -2:, :] = wall_id
        tokens[:, :, :2] = wall_id
        tokens[:, :, -2:] = wall_id

        base_seed = 0 if seed is None else int(seed)
        if self.baseline_name == "vglc_template" and self._train_interiors:
            for b in range(batch_size):
                rng = np.random.default_rng(abs(base_seed + b * 1009 + self._call_counter * 37))
                pick = int(rng.integers(0, len(self._train_interiors)))
                interior = torch.as_tensor(self._train_interiors[pick], device=device, dtype=torch.long)
                if ROOM_HEIGHT > 4 and ROOM_WIDTH > 4 and tuple(interior.shape) == (ROOM_HEIGHT - 4, ROOM_WIDTH - 4):
                    tokens[b, 2:-2, 2:-2] = interior
        self._call_counter += 1

        if isinstance(fixed_tokens, torch.Tensor) and isinstance(fixed_mask, torch.Tensor):
            ft = fixed_tokens.to(device=device, dtype=torch.long)
            fm = fixed_mask.to(device=device, dtype=torch.bool)
            if ft.dim() == 2:
                ft = ft.unsqueeze(0)
            if fm.dim() == 2:
                fm = fm.unsqueeze(0)
            if int(ft.shape[0]) == 1 and batch_size > 1:
                ft = ft.expand(batch_size, -1, -1)
            if int(fm.shape[0]) == 1 and batch_size > 1:
                fm = fm.expand(batch_size, -1, -1)
            tokens[fm] = ft[fm]

        # Open 2nd-ring door aprons through the 2-tile VGLC wall shell.
        tokens[:, 1, :][tokens[:, 0, :] != wall_id] = floor_id
        tokens[:, ROOM_HEIGHT - 2, :][tokens[:, ROOM_HEIGHT - 1, :] != wall_id] = floor_id
        tokens[:, :, 1][tokens[:, :, 0] != wall_id] = floor_id
        tokens[:, :, ROOM_WIDTH - 2][tokens[:, :, ROOM_WIDTH - 1] != wall_id] = floor_id

        logits = torch.full(
            (batch_size, self.num_classes, ROOM_HEIGHT, ROOM_WIDTH),
            fill_value=-1e4,
            device=device,
            dtype=torch.float32,
        )
        logits.scatter_(1, tokens.clamp(0, self.num_classes - 1).unsqueeze(1), 1e4)
        hidden = torch.zeros(batch_size, self.hidden_dim, ROOM_HEIGHT, ROOM_WIDTH, device=device, dtype=torch.float32)
        metrics = {"masked_unmask_order_is_random": 0.0, "baseline_proposal": 1.0}
        if return_sampling_metrics:
            return tokens, logits, hidden, metrics
        return tokens, logits, hidden


def _run_pipeline_benchmark(
    pipe: Any,
    *,
    arm_name: str,
    config_path: str,
    checkpoint_path: str,
    best_epoch: int,
    ckpt_metrics: Mapping[str, Any],
    unmask_order: str,
    is_baseline: bool,
    seeds: Sequence[int],
    num_rooms: int,
    train_rooms: Sequence[np.ndarray],
    vglc_dist: Mapping[str, float],
    topology_population_size: Optional[int] = None,
    topology_generations: Optional[int] = None,
) -> Dict[str, Any]:
    raw_rooms: List[np.ndarray] = []
    final_rooms: List[np.ndarray] = []
    raw_conn_hits = 0
    repaired_rooms = 0
    total_tiles_repaired = 0
    accepted_dungeons = 0
    zero_repair_accepted = 0
    wall_times: List[float] = []
    path_lengths: List[float] = []
    linearities: List[float] = []
    leniencies: List[float] = []
    bipartite_hits: List[float] = []

    for seed in seeds:
        t0 = time.perf_counter()
        gen_kwargs: Dict[str, Any] = {
            "mission_graph": None,
            "generate_topology": True,
            "num_rooms": int(num_rooms),
            "seed": int(seed),
            "enable_map_elites": False,
        }
        if topology_population_size is not None:
            gen_kwargs["population_size"] = int(topology_population_size)
        if topology_generations is not None:
            gen_kwargs["generations"] = int(topology_generations)
        res = pipe.generate_dungeon(**gen_kwargs)
        elapsed = time.perf_counter() - t0
        wall_times.append(float(elapsed))

        met = dict(res.metrics or {})
        accepted = bool(met.get("end_to_end_validation_accepted", False))
        accepted_dungeons += int(accepted)

        dungeon_raw_rooms: List[np.ndarray] = []
        dungeon_repaired = 0
        g = res.physical_mission_graph
        for rid, room in res.rooms.items():
            sem = pipe._extract_room_topology_semantics(g, rid)
            raw_grid = np.asarray(room.raw_neural_grid, dtype=np.int32)
            final_grid = np.asarray(room.room_grid, dtype=np.int32)
            raw_rooms.append(raw_grid)
            final_rooms.append(final_grid)
            dungeon_raw_rooms.append(raw_grid)
            raw_conn_hits += int(raw_doors_connected(raw_grid, sem["required_doors"]))
            was_rep = int(bool(room.was_repaired))
            repaired_rooms += was_rep
            dungeon_repaired += was_rep
            total_tiles_repaired += int(room.metrics.get("tiles_changed", 0))

        if accepted and dungeon_repaired == 0:
            zero_repair_accepted += 1

        plen, lin, leni, bip = _extract_era_and_topology(res, dungeon_raw_rooms)
        path_lengths.append(plen)
        linearities.append(lin)
        leniencies.append(leni)
        bipartite_hits.append(bip)

    n_dungeons = max(1, len(seeds))
    n_rooms = max(1, len(raw_rooms))
    total_time = max(1e-9, float(sum(wall_times)))
    raw_dist = compute_interior_distribution(raw_rooms)
    rep_dist = compute_interior_distribution(final_rooms)
    novelty = compute_novelty_metrics(raw_rooms, train_rooms)
    block_rooms = sum(1 for r in raw_rooms if (extract_interior(r) == int(P["BLOCK"])).any())

    js_raw = jensen_shannon_divergence(raw_dist, vglc_dist)
    js_rep = jensen_shannon_divergence(rep_dist, vglc_dist)
    tv_raw = total_variation_distance(raw_dist, vglc_dist)
    tv_rep = total_variation_distance(rep_dist, vglc_dist)

    return {
        "arm": arm_name,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "unmask_order": unmask_order,
        "is_baseline": bool(is_baseline),
        "best_epoch": int(best_epoch),
        "checkpoint_metrics": {
            "val_base_loss": float(ckpt_metrics.get("val_base_loss", float("nan"))),
            "val_loss": float(ckpt_metrics.get("val_loss", float("nan"))),
            "val_masked_token_accuracy": float(ckpt_metrics.get("val_masked_token_accuracy", float("nan"))),
            "val_logic_loss": float(ckpt_metrics.get("val_logic_loss", float("nan"))),
            "val_rare_tile_recall": float(ckpt_metrics.get("val_rare_tile_recall", float("nan"))),
        },
        "num_dungeons": len(seeds),
        "num_rooms": len(raw_rooms),
        "end_to_end_accepted_rate": float(accepted_dungeons) / float(n_dungeons),
        "raw_door_connectivity_rate": float(raw_conn_hits) / float(n_rooms),
        "repair_rate": float(repaired_rooms) / float(n_rooms),
        "tiles_repaired_mean": float(total_tiles_repaired) / float(n_rooms),
        "tiles_repaired_total": int(total_tiles_repaired),
        "raw_rooms_with_block_rate": float(block_rooms) / float(n_rooms),
        "raw_interior_distribution": raw_dist,
        "repaired_interior_distribution": rep_dist,
        "js_div_raw_vs_vglc": js_raw,
        "js_div_repaired_vs_vglc": js_rep,
        "repair_distribution_shift_js": float(js_rep - js_raw),
        "tv_raw_vs_vglc": tv_raw,
        "tv_repaired_vs_vglc": tv_rep,
        **novelty,
        "mean_optimal_path_length": float(np.mean(path_lengths)),
        "mean_linearity": float(np.mean(linearities)),
        "mean_leniency": float(np.mean(leniencies)),
        "strict_bipartite_rate": float(np.mean(bipartite_hits)),
        "mean_wall_time_sec": float(np.mean(wall_times)),
        "accepted_dungeons_per_sec": float(accepted_dungeons) / total_time,
        "zero_repair_dungeons_per_sec": float(zero_repair_accepted) / total_time,
    }


def evaluate_arm(
    *,
    arm_name: str,
    config_path: str,
    checkpoint_path: str,
    seeds: Sequence[int],
    num_rooms: int,
    device: str,
    train_rooms: Sequence[np.ndarray],
    vglc_dist: Mapping[str, float],
    unmask_order_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Run standardized multi-seed end-to-end evaluation for a single G-LMDM checkpoint."""
    import torch
    from src.config_system import merge_config
    from src.pipeline.config_bridge import pipeline_kwargs_from_resolved_config
    from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline

    ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_metrics = dict(ckpt_meta.get("metrics", {}) or {})
    resolved = merge_config(yaml_path=config_path)
    mr = resolved["masked_room"]
    cond_mode = str(mr.get("graph_conditioning_mode", "node_sequence")).strip().lower()
    p_kwargs = pipeline_kwargs_from_resolved_config(resolved)
    if unmask_order_override is not None:
        normalized_order = str(unmask_order_override).strip().lower()
        if normalized_order not in {"random", "confidence"}:
            raise ValueError(
                f"unmask_order_override must be 'random' or 'confidence', got {unmask_order_override!r}."
            )
        p_kwargs["default_masked_room_unmask_order"] = normalized_order
    p_kwargs.update(
        {
            "masked_room_checkpoint": checkpoint_path,
            "device": device,
            "strict_checkpoint_mode": True,
            "enable_logging": False,
            "use_graph_node_cross_attention": bool(cond_mode == "node_sequence"),
        }
    )
    pipe = NeuralSymbolicDungeonPipeline(**p_kwargs)
    pipe.graph_conditioning_mode = cond_mode
    effective_order = str(p_kwargs.get("default_masked_room_unmask_order", "confidence"))
    return _run_pipeline_benchmark(
        pipe,
        arm_name=arm_name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        best_epoch=int(ckpt_meta.get("epoch", -1)) + 1,
        ckpt_metrics=ckpt_metrics,
        unmask_order=effective_order,
        is_baseline=False,
        seeds=seeds,
        num_rooms=num_rooms,
        train_rooms=train_rooms,
        vglc_dist=vglc_dist,
        topology_population_size=topology_population_size,
        topology_generations=topology_generations,
    )


def evaluate_non_neural_baseline(
    baseline_name: str,
    *,
    seeds: Sequence[int],
    num_rooms: int,
    train_rooms: Sequence[np.ndarray],
    vglc_dist: Mapping[str, float],
    config_path: str = "configs/glmdm_canonical.yaml",
    device: str = "cpu",
    topology_population_size: Optional[int] = None,
    topology_generations: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate deterministic non-neural room proposals through the exact same symbolic pipeline."""
    from src.config_system import merge_config
    from src.pipeline.config_bridge import pipeline_kwargs_from_resolved_config
    from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline

    resolved = merge_config(yaml_path=config_path)
    mr = resolved["masked_room"]
    cond_mode = str(mr.get("graph_conditioning_mode", "node_sequence")).strip().lower()
    p_kwargs = pipeline_kwargs_from_resolved_config(resolved)
    p_kwargs.update(
        {
            "masked_room_checkpoint": None,
            "device": device,
            "strict_checkpoint_mode": False,
            "enable_logging": False,
            "use_graph_node_cross_attention": bool(cond_mode == "node_sequence"),
        }
    )
    pipe = NeuralSymbolicDungeonPipeline(**p_kwargs)
    pipe.graph_conditioning_mode = cond_mode
    pipe.masked_room_model = _BaselineRoomProposalModel(
        baseline_name=baseline_name,
        train_rooms=train_rooms,
    )
    return _run_pipeline_benchmark(
        pipe,
        arm_name=f"baseline:{str(baseline_name).strip().lower()}",
        config_path=config_path,
        checkpoint_path=f"baseline://{str(baseline_name).strip().lower()}",
        best_epoch=0,
        ckpt_metrics={},
        unmask_order="n/a",
        is_baseline=True,
        seeds=seeds,
        num_rooms=num_rooms,
        train_rooms=train_rooms,
        vglc_dist=vglc_dist,
        topology_population_size=topology_population_size,
        topology_generations=topology_generations,
    )


def _collect_numeric_leafs(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            out[full_key] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[full_key] = float(value)
        elif isinstance(value, Mapping):
            out.update(_collect_numeric_leafs(value, prefix=full_key))
    return out


def aggregate_fold_reports(fold_reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-fold benchmark JSON reports into per-arm mean +/- std summaries."""
    by_arm: Dict[str, List[Dict[str, float]]] = {}
    for report in fold_reports:
        arms = report.get("arms", []) if isinstance(report, Mapping) else []
        for entry in arms:
            arm_name = str(entry.get("arm", "unknown"))
            by_arm.setdefault(arm_name, []).append(_collect_numeric_leafs(entry))

    summary: Dict[str, Any] = {"num_folds": len(fold_reports), "arms": {}}
    for arm_name, rows in by_arm.items():
        all_keys = sorted({k for row in rows for k in row})
        arm_summary: Dict[str, Dict[str, float]] = {}
        for k in all_keys:
            vals = [row[k] for row in rows if k in row]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            arm_summary[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1 if len(arr) > 1 else 0)),
                "n": int(len(arr)),
            }
        summary["arms"][arm_name] = arm_summary
    return summary


def _parse_arm_spec(spec: str) -> Tuple[str, str, str]:
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise ValueError(
            f"Expected --arm-spec in NAME=CONFIG=CHECKPOINT format, got {spec!r}."
        )
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def expand_checkpoint_dir_spec(spec: str) -> List[Tuple[str, str, str]]:
    """Expand NAME=CONFIG=CKPT_DIR into available best/best_rare_recall/final specs."""
    name, cfg_path, ckpt_dir_str = _parse_arm_spec(spec)
    ckpt_dir = Path(ckpt_dir_str)
    found: List[Tuple[str, str, str]] = []
    for tag, filename in CHECKPOINT_CANDIDATES:
        candidate = ckpt_dir / filename
        if candidate.is_file():
            found.append((f"{name}:{tag}", cfg_path, str(candidate)))
    if not found:
        expected = ", ".join(fname for _tag, fname in CHECKPOINT_CANDIDATES)
        raise FileNotFoundError(
            f"No masked-room checkpoints found in {ckpt_dir} (expected one of: {expected})."
        )
    return found


def run_falsification_checks(
    arm_results: Sequence[Mapping[str, Any]],
    vglc_dist: Mapping[str, float],
) -> Dict[str, Any]:
    """Run automated falsification diagnostics aligned with H10/H20/H30/H40."""
    neural_arms = [
        a for a in arm_results
        if not bool(a.get("is_baseline", False)) and not str(a.get("arm", "")).startswith("baseline:")
    ]
    empty_floor = next(
        (a for a in arm_results if str(a.get("arm", "")) == "baseline:empty_floor"),
        None,
    )

    h20_flagged: List[Dict[str, Any]] = []
    h10_flagged: List[Dict[str, Any]] = []
    h40_repair_flagged: List[Dict[str, Any]] = []
    beats_empty_floor: Dict[str, Any] = {}

    for a in neural_arms:
        name = str(a.get("arm", "unknown"))
        block_room_rate = float(a.get("raw_rooms_with_block_rate", 0.0))
        block_share = float((a.get("raw_interior_distribution") or {}).get("BLOCK", 0.0))
        if block_room_rate < 0.15 or block_share < 0.02:
            h20_flagged.append(
                {
                    "arm": name,
                    "raw_rooms_with_block_rate": block_room_rate,
                    "raw_block_share": block_share,
                }
            )

        v4_match = float(a.get("train_v4_match_rate", 0.0))
        min_ham = float(a.get("mean_min_hamming_to_train", 1.0))
        if v4_match > 0.80 or min_ham < 0.05:
            h10_flagged.append(
                {
                    "arm": name,
                    "train_v4_match_rate": v4_match,
                    "mean_min_hamming_to_train": min_ham,
                }
            )

        rep_rate = float(a.get("repair_rate", 0.0))
        rep_shift = float(a.get("repair_distribution_shift_js", 0.0))
        if rep_rate > 0.50 or abs(rep_shift) > 0.05:
            h40_repair_flagged.append(
                {
                    "arm": name,
                    "repair_rate": rep_rate,
                    "repair_distribution_shift_js": rep_shift,
                }
            )

        if empty_floor is not None:
            empty_js = float(empty_floor.get("js_div_raw_vs_vglc", 1.0))
            empty_block = float(empty_floor.get("raw_rooms_with_block_rate", 0.0))
            arm_js = float(a.get("js_div_raw_vs_vglc", 1.0))
            beats_empty_floor[name] = {
                "beats_js": bool(arm_js < empty_js),
                "beats_block_presence": bool(block_room_rate > empty_block),
                "passed": bool(arm_js < empty_js and block_room_rate > empty_block),
                "arm_js": arm_js,
                "empty_floor_js": empty_js,
            }

    # Pair unmask orders (@random vs @confidence) on identical checkpoints.
    paired_orders: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for a in neural_arms:
        name = str(a.get("arm", ""))
        order = str(a.get("unmask_order", "")).strip().lower()
        base_name = name.split("@", 1)[0] if "@" in name else str(a.get("checkpoint_path", name))
        if order in {"random", "confidence"}:
            paired_orders.setdefault(base_name, {})[order] = a

    h30_pairs: Dict[str, Dict[str, float]] = {}
    for base_name, variants in paired_orders.items():
        if "random" in variants and "confidence" in variants:
            r = variants["random"]
            c = variants["confidence"]
            h30_pairs[base_name] = {
                "js_div_random_minus_confidence": float(r["js_div_raw_vs_vglc"]) - float(c["js_div_raw_vs_vglc"]),
                "block_rate_random_minus_confidence": float(r["raw_rooms_with_block_rate"]) - float(c["raw_rooms_with_block_rate"]),
                "repair_rate_random_minus_confidence": float(r["repair_rate"]) - float(c["repair_rate"]),
            }

    return {
        "vglc_block_share_reference": float(vglc_dist.get("BLOCK", 0.0)),
        "H20_floor_shortcut_collapse": {
            "passed": len(h20_flagged) == 0,
            "flagged_arms": h20_flagged,
        },
        "H40_beats_empty_floor_baseline": {
            "evaluated": empty_floor is not None and len(neural_arms) > 0,
            "passed": (
                all(v["passed"] for v in beats_empty_floor.values())
                if (empty_floor is not None and beats_empty_floor)
                else None
            ),
            "per_arm": beats_empty_floor,
        },
        "H10_memorization_alarm": {
            "passed": len(h10_flagged) == 0,
            "flagged_arms": h10_flagged,
        },
        "H30_unmask_order_delta": {
            "paired_arms": h30_pairs,
        },
        "H40_repair_dominance": {
            "passed": len(h40_repair_flagged) == 0,
            "flagged_arms": h40_repair_flagged,
        },
    }


def _format_markdown_summary(arms: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Arm | Accepted | RawConn | RepairRate | TilesRep | BlockRooms | JS_raw | JS_rep | "
        "TrainMatch | NovelHamming | BipRate | Sec/Dung | Acc/Sec |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for a in arms:
        lines.append(
            f"| {a['arm']} | {a['end_to_end_accepted_rate']:.3f} | "
            f"{a['raw_door_connectivity_rate']:.3f} | {a['repair_rate']:.3f} | "
            f"{a['tiles_repaired_mean']:.2f} | {a['raw_rooms_with_block_rate']:.3f} | "
            f"{a['js_div_raw_vs_vglc']:.4f} | {a['js_div_repaired_vs_vglc']:.4f} | "
            f"{a['train_v4_match_rate']:.3f} | {a['mean_min_hamming_to_train']:.3f} | "
            f"{a['strict_bipartite_rate']:.3f} | {a['mean_wall_time_sec']:.2f} | "
            f"{a['accepted_dungeons_per_sec']:.3f} |"
        )
    return "\n".join(lines)


def _format_falsification_summary(checks: Mapping[str, Any]) -> str:
    lines = ["\nFalsification Checks:"]
    for key in (
        "H20_floor_shortcut_collapse",
        "H40_beats_empty_floor_baseline",
        "H10_memorization_alarm",
        "H40_repair_dominance",
    ):
        entry = checks.get(key, {})
        lines.append(f"- {key}: passed={entry.get('passed')} details={entry}")
    pairs = checks.get("H30_unmask_order_delta", {}).get("paired_arms", {})
    if pairs:
        lines.append(f"- H30_unmask_order_delta: {pairs}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Standardized G-LMDM publication benchmark and k-fold aggregator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arm-spec",
        action="append",
        default=[],
        help="Arm specification formatted as NAME=CONFIG_YAML=CHECKPOINT_PATH (repeatable).",
    )
    parser.add_argument(
        "--checkpoint-dir-spec",
        action="append",
        default=[],
        help="Directory specification NAME=CONFIG_YAML=CKPT_DIR auto-expanding best/best_rare_recall/final.",
    )
    parser.add_argument(
        "--eval-both-unmask-orders",
        action="store_true",
        help="Evaluate each neural checkpoint under both 'random' and 'confidence' unmask orders.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=("empty_floor", "vglc_template"),
        default=[],
        help="Run deterministic non-neural reference baseline through the exact same symbolic pipeline.",
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default="configs/glmdm_canonical.yaml",
        help="Pipeline config used for symbolic repair/validation during --baseline runs.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    parser.add_argument("--num-rooms", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-dir", type=str, default="Data/The Legend of Zelda")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--aggregate-folds-dir",
        type=str,
        default="",
        help="Optional directory of per-fold JSON reports to aggregate into mean +/- std.",
    )
    args = parser.parse_args(argv)

    if args.aggregate_folds_dir:
        fold_dir = Path(args.aggregate_folds_dir)
        json_paths = sorted(p for p in fold_dir.glob("*.json") if p.name != "aggregate_summary.json")
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in json_paths]
        agg = aggregate_fold_reports(reports)
        out_path = Path(args.output_json) if args.output_json else fold_dir / "aggregate_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"Aggregated {len(reports)} fold reports -> {out_path}")
        return

    if not args.arm_spec and not args.checkpoint_dir_spec and not args.baseline:
        parser.error("Provide --arm-spec, --checkpoint-dir-spec, --baseline, or --aggregate-folds-dir.")

    train_rooms, vglc_dist = _load_vglc_train_reference(data_dir=args.data_dir)
    arm_results: List[Dict[str, Any]] = []

    for baseline_name in args.baseline:
        arm_results.append(
            evaluate_non_neural_baseline(
                baseline_name,
                seeds=args.seeds,
                num_rooms=args.num_rooms,
                train_rooms=train_rooms,
                vglc_dist=vglc_dist,
                config_path=args.baseline_config,
                device=args.device,
            )
        )

    expanded_specs: List[Tuple[str, str, str]] = [
        _parse_arm_spec(spec) for spec in args.arm_spec
    ]
    for dir_spec in args.checkpoint_dir_spec:
        expanded_specs.extend(expand_checkpoint_dir_spec(dir_spec))

    for name, cfg_path, ckpt_path in expanded_specs:
        if args.eval_both_unmask_orders:
            for order in ("random", "confidence"):
                arm_results.append(
                    evaluate_arm(
                        arm_name=f"{name}@{order}",
                        config_path=cfg_path,
                        checkpoint_path=ckpt_path,
                        seeds=args.seeds,
                        num_rooms=args.num_rooms,
                        device=args.device,
                        train_rooms=train_rooms,
                        vglc_dist=vglc_dist,
                        unmask_order_override=order,
                    )
                )
        else:
            arm_results.append(
                evaluate_arm(
                    arm_name=name,
                    config_path=cfg_path,
                    checkpoint_path=ckpt_path,
                    seeds=args.seeds,
                    num_rooms=args.num_rooms,
                    device=args.device,
                    train_rooms=train_rooms,
                    vglc_dist=vglc_dist,
                )
            )

    falsification = run_falsification_checks(arm_results, vglc_dist)
    payload = {
        "vglc_reference_distribution": vglc_dist,
        "seeds": [int(s) for s in args.seeds],
        "num_rooms": int(args.num_rooms),
        "arms": arm_results,
        "falsification_checks": falsification,
    }
    print(_format_markdown_summary(arm_results))
    print(_format_falsification_summary(falsification))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report -> {out_path}")


if __name__ == "__main__":
    main()
