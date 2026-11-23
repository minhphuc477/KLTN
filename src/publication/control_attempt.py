"""Independent source-room retrieval and adjacency-WFC generation controls.

These controls fit only outer-fold training rooms. Graph conditioning is confined
to the common repair/assembly pipeline; retrieval itself is uniform over rooms.
WFC uses this repository's categorical adjacency implementation, not Gumin's
overlapping-pattern implementation. Contradictions remain failed attempts.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from collections import Counter, defaultdict
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from src.config_system import seed_everything, validate_config
from src.core.definitions import SEMANTIC_PALETTE
from src.pipeline.config_bridge import pipeline_kwargs_from_resolved_config

from .dataset_provenance import verify_dataset_manifest
from .pipeline_attempt import CanonicalAttemptPipeline
from .protocol import CONTROL_MODELS, payload_sha256
from .room_evidence import TrainingRoomReference


def _compile_count_distribution(counts: Counter) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-normalize a tile histogram into immutable arrays for O(1) weighted sampling."""
    tiles = np.fromiter(counts.keys(), dtype=np.int32)
    weights = np.fromiter(counts.values(), dtype=np.float64)
    probs = weights / weights.sum()
    return tiles, probs


def _sample_from_compiled(rng: np.random.Generator, dist: Tuple[np.ndarray, np.ndarray]) -> int:
    tiles, probs = dist
    if len(tiles) == 1:
        return int(tiles[0])
    return int(rng.choice(tiles, p=probs))


def _extract_pod_tables(training_rooms: np.ndarray) -> Dict[str, Any]:
    """Extract and pre-normalize Path-of-Destruction 3x3 repair tables once per fold."""
    n_rooms, height, width = training_rooms.shape
    pad_val = int(SEMANTIC_PALETTE["WALL"])
    padded = np.pad(training_rooms, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=pad_val)

    context_counts = defaultdict(Counter)
    pos_counts = [[Counter() for _ in range(width)] for _ in range(height)]
    global_counts: Counter = Counter()

    for idx in range(n_rooms):
        for r in range(height):
            for c in range(width):
                center = int(training_rooms[idx, r, c])
                global_counts[center] += 1
                pos_counts[r][c][center] += 1
                patch = padded[idx, r:r + 3, c:c + 3]
                ctx = (
                    int(patch[0, 0]), int(patch[0, 1]), int(patch[0, 2]),
                    int(patch[1, 0]),                    int(patch[1, 2]),
                    int(patch[2, 0]), int(patch[2, 1]), int(patch[2, 2]),
                )
                context_counts[ctx][center] += 1

    return {
        "height": int(height),
        "width": int(width),
        "pad_val": pad_val,
        "global_dist": _compile_count_distribution(global_counts),
        "pos_dists": [[_compile_count_distribution(pos_counts[r][c]) for c in range(width)] for r in range(height)],
        "context_dists": {ctx: _compile_count_distribution(cnt) for ctx, cnt in context_counts.items()},
        "unique_3x3_contexts": int(len(context_counts)),
    }


def _sample_pod_room(pod_tables: Mapping[str, Any], *, seed: int):
    """Sample one room using precompiled Path-of-Destruction 3x3 repair distributions."""
    rng = np.random.default_rng(seed)
    height = int(pod_tables["height"])
    width = int(pod_tables["width"])
    pad_val = int(pod_tables["pad_val"])
    global_tiles, global_probs = pod_tables["global_dist"]
    pos_dists = pod_tables["pos_dists"]
    context_dists = pod_tables["context_dists"]

    grid = rng.choice(global_tiles, size=(height, width), p=global_probs).astype(np.int32)
    coords = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(coords)
    for r, c in coords:
        ctx = (
            int(grid[r - 1, c - 1]) if (r > 0 and c > 0) else pad_val,
            int(grid[r - 1, c]) if r > 0 else pad_val,
            int(grid[r - 1, c + 1]) if (r > 0 and c + 1 < width) else pad_val,
            int(grid[r, c - 1]) if c > 0 else pad_val,
            int(grid[r, c + 1]) if c + 1 < width else pad_val,
            int(grid[r + 1, c - 1]) if (r + 1 < height and c > 0) else pad_val,
            int(grid[r + 1, c]) if r + 1 < height else pad_val,
            int(grid[r + 1, c + 1]) if (r + 1 < height and c + 1 < width) else pad_val,
        )
        dist = context_dists.get(ctx)
        grid[r, c] = _sample_from_compiled(rng, dist if dist is not None else pos_dists[r][c])

    return grid, {
        "baseline_family": "path_of_destruction_iterative_repair",
        "repair_steps": int(height * width),
        "unique_3x3_contexts": int(pod_tables["unique_3x3_contexts"]),
        "generation_succeeded": True,
    }


def _extract_ar_markov_tables(training_rooms: np.ndarray) -> Dict[str, Any]:
    """Extract and pre-normalize causal 1D raster Markov transition tables once per fold."""
    n_rooms, height, width = training_rooms.shape
    pad_val = int(SEMANTIC_PALETTE["WALL"])

    causal_counts = defaultdict(Counter)
    pos_counts = [[Counter() for _ in range(width)] for _ in range(height)]

    for idx in range(n_rooms):
        for r in range(height):
            for c in range(width):
                tile = int(training_rooms[idx, r, c])
                pos_counts[r][c][tile] += 1
                left = int(training_rooms[idx, r, c - 1]) if c > 0 else pad_val
                up = int(training_rooms[idx, r - 1, c]) if r > 0 else pad_val
                up_left = int(training_rooms[idx, r - 1, c - 1]) if (r > 0 and c > 0) else pad_val
                causal_counts[(left, up, up_left)][tile] += 1

    return {
        "height": int(height),
        "width": int(width),
        "pad_val": pad_val,
        "pos_dists": [[_compile_count_distribution(pos_counts[r][c]) for c in range(width)] for r in range(height)],
        "causal_dists": {ctx: _compile_count_distribution(cnt) for ctx, cnt in causal_counts.items()},
        "causal_contexts": int(len(causal_counts)),
    }


def _sample_ar_markov_room(ar_tables: Mapping[str, Any], *, seed: int):
    """Synthesize one room in causal 1D raster order using precompiled Markov tables."""
    rng = np.random.default_rng(seed)
    height = int(ar_tables["height"])
    width = int(ar_tables["width"])
    pad_val = int(ar_tables["pad_val"])
    pos_dists = ar_tables["pos_dists"]
    causal_dists = ar_tables["causal_dists"]

    grid = np.zeros((height, width), dtype=np.int32)
    for r in range(height):
        for c in range(width):
            left = int(grid[r, c - 1]) if c > 0 else pad_val
            up = int(grid[r - 1, c]) if r > 0 else pad_val
            up_left = int(grid[r - 1, c - 1]) if (r > 0 and c > 0) else pad_val
            dist = causal_dists.get((left, up, up_left))
            grid[r, c] = _sample_from_compiled(rng, dist if dist is not None else pos_dists[r][c])

    return grid, {
        "baseline_family": "raster_autoregressive_markov",
        "causal_contexts": int(ar_tables["causal_contexts"]),
        "generation_succeeded": True,
    }


def generate_control_rooms(reference, node_ids, *, model_id, seed):
    """Generate once per room with independent deterministic local RNG streams."""
    if model_id not in CONTROL_MODELS:
        raise ValueError("Unknown independent room control.")
    from src.generation.weighted_bayesian_wfc import (
        WeightedBayesianWFC, WeightedBayesianWFCConfig, extract_tile_priors_from_grids,
    )
    training = reference.rooms()
    priors = extract_tile_priors_from_grids(training) if model_id == "wfc_training" else None
    pod_tables = _extract_pod_tables(training) if model_id == "pod_training" else None
    ar_tables = _extract_ar_markov_tables(training) if model_id == "ar_markov_training" else None
    nodes = list(node_ids)
    rooms, diagnostics = {}, {}
    for node, child in zip(nodes, np.random.SeedSequence(seed).spawn(len(nodes))):
        child_seed = int(child.generate_state(1)[0])
        if model_id == "retrieval_training":
            index = int(np.random.default_rng(child_seed).integers(len(training)))
            grid = training[index].copy()
            detail = {"source_room_sha256": reference.fingerprints[index]}
        elif model_id == "pod_training":
            grid, detail = _sample_pod_room(pod_tables, seed=child_seed)
        elif model_id == "ar_markov_training":
            grid, detail = _sample_ar_markov_room(ar_tables, seed=child_seed)
        else:
            height, width = training.shape[1:]
            wfc = WeightedBayesianWFC(width, height, priors, config=WeightedBayesianWFCConfig(
                max_iterations=10_000, max_backtracks=256, max_restarts=3))
            grid = wfc.generate(seed=child_seed)
            detail = wfc.get_diagnostics()
            if not detail["generation_succeeded"] or detail["required_fallback"]:
                raise RuntimeError(f"WFC did not complete its constraints for room {node!r}: {detail}")
        rooms[node] = grid
        diagnostics[node] = detail
    return rooms, diagnostics


class ControlAttemptPipeline(CanonicalAttemptPipeline):
    """Use the same exact validator with a pipeline containing no neural models."""

    def _load(self, spec):
        if spec.metadata["model_id"] not in CONTROL_MODELS:
            raise ValueError("Control adapter requires a declared independent control.")
        if set(spec.artifacts) != {"resolved_config", "mission_graph"}:
            raise ValueError("Independent controls must not consume model checkpoints.")
        config = validate_config(yaml.safe_load(Path(spec.artifacts["resolved_config"]["path"]).read_text(encoding="utf-8")))
        if payload_sha256(config) != spec.metadata["config_sha256"]:
            raise ValueError("Control configuration differs from its sealed digest.")
        if (config["runtime"]["seed"] != spec.metadata["train_seed"]
                or config["dataset"]["test_dungeons"] != [int(spec.metadata["outer_fold"].removeprefix("holdout_dungeon_"))]):
            raise ValueError("Control configuration differs from its declared fold or repeat seed.")
        if config["generation"]["end_to_end_validation_mode"] != "report":
            raise ValueError("Controls must retain invalid attempts using report mode.")
        key = spec.metadata["config_sha256"]
        if key != self._model_key:
            from src.pipeline.models.model_manager import create_refiner
            from src.pipeline.types import PipelineComponents, SymbolicGenerationComponents
            provenance = verify_dataset_manifest(config["dataset"]["data_dir"])
            if provenance["content_sha256"] != spec.metadata["dataset_manifest_sha256"]:
                raise ValueError("Control training corpus changed after sealing.")
            self._reference = TrainingRoomReference.from_config(config)
            kwargs = pipeline_kwargs_from_resolved_config(config)
            stage = config["generation"]
            kwargs.update(components=PipelineComponents(symbolic=SymbolicGenerationComponents(
                refiner=create_refiner(False, max_repair_attempts=stage["symbolic_max_repair_attempts"],
                    margin=stage["symbolic_repair_margin"], adjacency_threshold=stage["symbolic_adjacency_threshold"]))),
                strict_checkpoint_mode=False, require_logic_net=False, use_learned_refiner_rules=False,
                vqvae_checkpoint=None, diffusion_checkpoint=None, condition_encoder_checkpoint=None,
                logic_net_checkpoint=None, masked_room_checkpoint=None, fast_sampling_checkpoint=None,
                device=config["runtime"]["device"], enable_logging=False)
            self._pipeline = self._factory(**kwargs)
            self._model_key = key
        return self._pipeline

    def generate(self, spec):
        spec.verify_artifacts()
        pipeline = self._load(spec)
        kwargs = spec.generation_kwargs
        if (spec.metadata["inference_variant"] != "unguided"
                or float(kwargs.get("logic_guidance_scale", 0.0)) != 0.0
                or any(name in kwargs for name in ("seed", "mission_graph"))
                or kwargs.get("generate_topology") or kwargs.get("enable_map_elites")):
            raise ValueError("Independent controls require a fixed mission and unguided generation.")
        seed = spec.metadata["sampling_seed"]
        seed_everything(seed, cudnn_benchmark=False, cudnn_deterministic=True, deterministic_algorithms=True)
        graph = nx.node_link_graph(json.loads(Path(spec.artifacts["mission_graph"]["path"]).read_text(encoding="utf-8")), edges="links")
        started = time.perf_counter()
        prepared = pipeline.prepare_dungeon_generation(mission_graph=graph, seed=seed, prepare_neural_context=False)
        physical = prepared.mission_graph_physical
        rooms, diagnostics = generate_control_rooms(self._reference, physical.nodes,
                                                    model_id=spec.metadata["model_id"], seed=seed)
        result = pipeline.repair_and_stitch_dungeon(rooms=rooms, mission_graph=physical,
            apply_repair=kwargs.get("apply_repair", pipeline.default_apply_repair), enable_map_elites=False, seed=seed)
        result.mission_graph = graph
        result.generation_time = time.perf_counter() - started
        result.metrics["generation_time_sec"] = result.generation_time
        for node, room in result.rooms.items():
            room.metrics["training_reference"] = self._reference.compare(room.raw_neural_grid)
            room.metrics["control_diagnostics"] = diagnostics[node]
        return result
