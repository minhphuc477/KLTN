"""Dungeon assembly helpers for graph preparation and room stitching."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from src.core import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
from src.pipeline.generation.sampler import _stable_node_seed_offset
from src.pipeline.room_stitching import (
    StitchedRoomLayout,
    build_stitched_room_layout,
    carve_room_connection_between_bboxes,
)
from src.pipeline.spatial_utils import stable_node_sort_key
from src.pipeline.types import (
    DungeonGenerationResult,
    GeneratedRoomSet,
    PreparedDungeonGeneration,
    RoomGenerationResult,
)
from src.utils.graph_utils import validate_graph_topology
from src.zelda_data.vglc_utils import filter_virtual_nodes, get_physical_start_node

logger = logging.getLogger(__name__)
_stable_node_sort_key = stable_node_sort_key


def _public_pipeline_hook(name: str, fallback: Any) -> Any:
    """Resolve monkeypatchable public facade hooks without importing it eagerly."""
    module = sys.modules.get("src.pipeline.dungeon_pipeline")
    if module is None:
        return fallback
    return getattr(module, name, fallback)


def _record_stage_time(stage_times: Dict[str, float], key: str, started_at: float) -> None:
    """Accumulate wall-clock stage timings in seconds."""
    stage_times[key] = float(stage_times.get(key, 0.0)) + float(time.perf_counter() - started_at)


def _aggregate_room_stage_times(room_metric_dicts: List[Dict[str, Any]]) -> Dict[str, float]:
    """Sum room-level timing metrics into a compact dungeon-level stage ledger."""
    timing_prefixes = (
        "batch_",
        "boundary_",
        "categorical_",
        "condition_",
        "diffusion_",
        "masked_",
        "post_decode_",
        "precomputed_",
        "room_generation_",
        "teacher_fallback_",
        "vqvae_",
    )
    timing_keys = {
        str(key)
        for metrics in room_metric_dicts
        for key in metrics
        if str(key).endswith("_time_sec")
        and (str(key).startswith(timing_prefixes) or str(key) == "repair_time_sec")
    }
    aggregated: Dict[str, float] = {}
    for key in sorted(timing_keys):
        total = 0.0
        for metrics in room_metric_dicts:
            try:
                total += float(metrics.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        aggregated[key] = float(total)
    return aggregated


def _get_topology_generator_class() -> Any:
    return _public_pipeline_hook("EvolutionaryTopologyGenerator", EvolutionaryTopologyGenerator)


def _get_validate_graph_topology() -> Any:
    return _public_pipeline_hook("validate_graph_topology", validate_graph_topology)


def stitch_room_layout(
    pipeline: Any,
    rooms: Dict[Any, Any],
    graph: nx.Graph,
    *,
    enforce_room_dimensions: Optional[Tuple[int, int]] = (ROOM_HEIGHT, ROOM_WIDTH),
    carve_connections: bool = True,
) -> StitchedRoomLayout:
    """Build a stitched dungeon layout and room bbox metadata."""
    return build_stitched_room_layout(
        rooms=rooms,
        graph=graph,
        fill_tile=int(SEMANTIC_PALETTE.get("VOID", 0)),
        sort_key=stable_node_sort_key,
        node_position_getter=pipeline._get_node_grid_position,
        first_free_position_fn=pipeline._first_free_position,
        enforce_room_dimensions=enforce_room_dimensions,
        carve_connections=carve_connections,
        diagnostic_callback=pipeline._bump_diagnostic,
    )


def stitch_rooms(
    pipeline: Any,
    rooms: Dict[int, RoomGenerationResult],
    graph: nx.Graph,
) -> np.ndarray:
    """Public room stitching entry point with optional injected stitcher support."""
    stitcher = getattr(pipeline, "stitcher", None)
    if stitcher is not None:
        stitch_fn = getattr(stitcher, "stitch_rooms", None)
        if callable(stitch_fn):
            try:
                stitched = stitch_fn(rooms=rooms, graph=graph)
            except TypeError:
                stitched = stitch_fn(rooms, graph)
            if isinstance(stitched, StitchedRoomLayout):
                return stitched.dungeon_grid
            return stitched
    return stitch_room_layout(pipeline, rooms=rooms, graph=graph).dungeon_grid


def stitch_room_grid(
    pipeline: Any,
    rooms: Dict[int, RoomGenerationResult],
    graph: nx.Graph,
) -> np.ndarray:
    """Internal grid-only stitch helper."""
    if not rooms:
        return np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
    return stitch_room_layout(pipeline, rooms, graph).dungeon_grid


def carve_room_connection_with_fallback(
    pipeline: Any,
    global_grid: np.ndarray,
    src_pos: Tuple[int, int],
    dst_pos: Tuple[int, int],
    edge_data: Optional[Dict[str, Any]] = None,
    has_reverse_edge: bool = False,
) -> None:
    """Carve a bbox-aware connection between two room grid positions."""
    src_bbox = (
        int(src_pos[1] * ROOM_WIDTH),
        int(src_pos[0] * ROOM_HEIGHT),
        int((src_pos[1] + 1) * ROOM_WIDTH - 1),
        int((src_pos[0] + 1) * ROOM_HEIGHT - 1),
    )
    dst_bbox = (
        int(dst_pos[1] * ROOM_WIDTH),
        int(dst_pos[0] * ROOM_HEIGHT),
        int((dst_pos[1] + 1) * ROOM_WIDTH - 1),
        int((dst_pos[0] + 1) * ROOM_HEIGHT - 1),
    )
    carve_room_connection_between_bboxes(
        global_grid,
        src_bbox,
        dst_bbox,
        edge_data=edge_data,
        has_reverse_edge=has_reverse_edge,
        fill_tile=int(SEMANTIC_PALETTE.get("VOID", 0)),
        diagnostic_callback=pipeline._bump_diagnostic,
    )


def prepare_dungeon_generation(
    pipeline,
    mission_graph: Optional[nx.Graph] = None,
    *,
    use_topological_positional_encoding: bool = True,
    generate_topology: bool = False,
    target_curve: Optional[List[float]] = None,
    num_rooms: Optional[int] = None,
    population_size: Optional[int] = None,
    generations: Optional[int] = None,
    mutation_rate: Optional[float] = None,
    crossover_rate: Optional[float] = None,
    genome_length: Optional[int] = None,
    rule_space: Optional[str] = None,
    transition_mix: Optional[float] = None,
    search_strategy: Optional[str] = None,
    qd_archive_cells: Optional[int] = None,
    qd_init_random_fraction: Optional[float] = None,
    qd_emitter_mutation_rate: Optional[float] = None,
    qd_archive_path: Optional[str] = None,
    qd_load_archive: Optional[bool] = None,
    qd_autosave_archive: Optional[bool] = None,
    max_lock_key_rules: Optional[int] = None,
    enable_rule_credit_assignment: Optional[bool] = None,
    enforce_generation_constraints: Optional[bool] = None,
    allow_candidate_repairs: Optional[bool] = None,
    seed: Optional[int] = None,
) -> PreparedDungeonGeneration:
    """
    Prepare mission-graph inputs for room generation.

    This isolates topology synthesis, virtual-node filtering, and graph
    tensor preparation so callers can run those phases independently.
    """
    graph = mission_graph
    if graph is None:
        if not generate_topology:
            raise ValueError(
                "mission_graph is None but generate_topology=False. "
                "Either provide a mission_graph or set generate_topology=True"
            )

        logger.info("Block I: Generating dungeon topology via evolutionary search")
        resolved_target_curve = (
            [float(v) for v in target_curve]
            if target_curve is not None
            else list(pipeline.topology_default_target_curve)
        )
        resolved_num_rooms = int(pipeline.topology_num_rooms if num_rooms is None else max(1, int(num_rooms)))
        resolved_population_size = int(
            pipeline.topology_population_size if population_size is None else max(1, int(population_size))
        )
        resolved_generations = int(
            pipeline.topology_generations if generations is None else max(1, int(generations))
        )
        resolved_mutation_rate = float(
            pipeline.topology_mutation_rate if mutation_rate is None else np.clip(float(mutation_rate), 0.0, 1.0)
        )
        resolved_crossover_rate = float(
            pipeline.topology_crossover_rate if crossover_rate is None else np.clip(float(crossover_rate), 0.0, 1.0)
        )
        resolved_genome_length = pipeline.topology_genome_length if genome_length is None else int(max(0, int(genome_length)))
        resolved_rule_space = (
            pipeline.topology_rule_space if rule_space is None else str(rule_space).strip().lower()
        )
        resolved_transition_mix = float(
            pipeline.topology_transition_mix if transition_mix is None else np.clip(float(transition_mix), 0.0, 1.0)
        )
        resolved_search_strategy = (
            pipeline.topology_search_strategy if search_strategy is None else str(search_strategy).strip().lower()
        )
        resolved_qd_archive_cells = int(
            pipeline.topology_qd_archive_cells if qd_archive_cells is None else max(32, int(qd_archive_cells))
        )
        resolved_qd_init_random_fraction = float(
            pipeline.topology_qd_init_random_fraction
            if qd_init_random_fraction is None
            else np.clip(float(qd_init_random_fraction), 0.05, 0.95)
        )
        resolved_qd_emitter_mutation_rate = float(
            pipeline.topology_qd_emitter_mutation_rate
            if qd_emitter_mutation_rate is None
            else np.clip(float(qd_emitter_mutation_rate), 0.01, 0.95)
        )
        resolved_qd_archive_path = (
            pipeline.topology_qd_archive_path
            if qd_archive_path is None
            else (str(qd_archive_path) if qd_archive_path else None)
        )
        resolved_qd_load_archive = bool(
            pipeline.topology_qd_load_archive if qd_load_archive is None else qd_load_archive
        )
        resolved_qd_autosave_archive = bool(
            pipeline.topology_qd_autosave_archive if qd_autosave_archive is None else qd_autosave_archive
        )
        resolved_max_lock_key_rules = int(
            pipeline.topology_max_lock_key_rules
            if max_lock_key_rules is None
            else max(0, int(max_lock_key_rules))
        )
        resolved_enable_rule_credit_assignment = bool(
            pipeline.topology_enable_rule_credit_assignment
            if enable_rule_credit_assignment is None
            else enable_rule_credit_assignment
        )
        resolved_enforce_generation_constraints = bool(
            pipeline.topology_enforce_generation_constraints
            if enforce_generation_constraints is None
            else enforce_generation_constraints
        )
        resolved_allow_candidate_repairs = bool(
            pipeline.topology_allow_candidate_repairs
            if allow_candidate_repairs is None
            else allow_candidate_repairs
        )
        target_genome_length = int(resolved_genome_length)
        if target_genome_length <= 0:
            target_genome_length = max(10, int(resolved_num_rooms * 0.7))
        topology_generator = _get_topology_generator_class()(
            target_curve=resolved_target_curve,
            population_size=resolved_population_size,
            generations=resolved_generations,
            mutation_rate=resolved_mutation_rate,
            crossover_rate=resolved_crossover_rate,
            genome_length=target_genome_length,
            max_nodes=resolved_num_rooms,
            rule_space=resolved_rule_space,
            transition_mix=resolved_transition_mix,
            search_strategy=resolved_search_strategy,
            qd_archive_cells=resolved_qd_archive_cells,
            qd_init_random_fraction=resolved_qd_init_random_fraction,
            qd_emitter_mutation_rate=resolved_qd_emitter_mutation_rate,
            qd_archive_path=resolved_qd_archive_path,
            qd_load_archive=resolved_qd_load_archive,
            qd_autosave_archive=resolved_qd_autosave_archive,
            max_lock_key_rules=resolved_max_lock_key_rules,
            enable_rule_credit_assignment=resolved_enable_rule_credit_assignment,
            enforce_generation_constraints=resolved_enforce_generation_constraints,
            allow_candidate_repairs=resolved_allow_candidate_repairs,
            seed=seed,
        )

        graph = topology_generator.evolve(directed_output=True)
        logger.info("Block I: Generated topology with %d rooms", graph.number_of_nodes())

        is_valid, errors = _get_validate_graph_topology()(graph)
        if not is_valid:
            if pipeline.strict_checkpoint_mode:
                raise ValueError(
                    f"Block I: Generated topology failed validation in strict mode: {errors}"
                )
            logger.warning("Block I: Generated topology has validation errors: %s", errors)

    if graph is None:
        raise ValueError("mission_graph is still None after topology generation attempt")

    logger.debug("Applying VGLC compliance: filtering virtual nodes from mission graph")
    mission_graph_physical = filter_virtual_nodes(graph)
    physical_start = get_physical_start_node(graph)
    if physical_start is not None:
        logger.debug("Physical start node: %s", physical_start)

    logger.info(
        "Generating dungeon with %d rooms (filtered %d virtual nodes)",
        len(mission_graph_physical.nodes),
        len(graph.nodes) - len(mission_graph_physical.nodes),
    )

    graph_data = pipeline._prepare_graph_context(
        mission_graph_physical,
        use_tpe=use_topological_positional_encoding,
    )
    return PreparedDungeonGeneration(
        mission_graph=graph,
        mission_graph_physical=mission_graph_physical,
        graph_data=graph_data,
    )


def generate_rooms_for_graph(
    pipeline,
    prepared: PreparedDungeonGeneration,
    *,
    guidance_scale: Optional[float] = None,
    logic_guidance_scale: Optional[float] = None,
    num_diffusion_steps: Optional[int] = None,
    use_fast_sampling: Optional[bool] = None,
    latent_sampler: Optional[str] = None,
    categorical_codebook_size: Optional[int] = None,
    apply_repair: Optional[bool] = None,
    seed: Optional[int] = None,
    batch_independent_rooms: bool = True,
    max_batch_size: int = 8,
) -> GeneratedRoomSet:
    """
    Generate all room grids/latents for a prepared mission graph.

    This allows room generation to be tested or reused without immediately
    stitching the final dungeon grid.
    """
    guidance_scale = pipeline.default_guidance_scale if guidance_scale is None else float(guidance_scale)
    logic_guidance_scale = (
        pipeline.default_logic_guidance_scale
        if logic_guidance_scale is None
        else float(logic_guidance_scale)
    )
    num_diffusion_steps = (
        pipeline.default_num_diffusion_steps if num_diffusion_steps is None else int(num_diffusion_steps)
    )
    use_fast_sampling = (
        pipeline.default_use_fast_sampling if use_fast_sampling is None else bool(use_fast_sampling)
    )
    latent_sampler = pipeline.default_latent_sampler if latent_sampler is None else str(latent_sampler)
    pipeline._require_room_generation_components(
        "generate_rooms_for_graph",
        latent_sampler=latent_sampler,
    )
    if categorical_codebook_size is None and pipeline.default_categorical_codebook_size is not None:
        categorical_codebook_size = int(pipeline.default_categorical_codebook_size)
    apply_repair = pipeline.default_apply_repair if apply_repair is None else bool(apply_repair)
    pipeline._puzzle_novelty_history = []
    pipeline._puzzle_variant_cache = {}
    pipeline._puzzle_novelty_committed = set()

    mission_graph_physical = prepared.mission_graph_physical
    graph_data = prepared.graph_data
    rooms: Dict[Any, RoomGenerationResult] = {}
    room_latents: Dict[Any, torch.Tensor] = {}
    batch_runtime_diagnostics: List[Dict[str, Any]] = []

    if batch_independent_rooms and mission_graph_physical.is_directed():
        layers = pipeline._topological_generation_layers(mission_graph_physical)
        offset = 0
        for layer_idx, layer in enumerate(layers):
            if not layer:
                continue
            buckets = pipeline._bucket_room_ids_by_latent_shape(
                room_ids=list(layer),
                mission_graph_physical=mission_graph_physical,
                room_latents=room_latents,
            )
            for bucket_key, bucket_room_ids in buckets.items():
                latent_shape_chw = (int(bucket_key[0]), int(bucket_key[1]), int(bucket_key[2]))
                target_h = int(bucket_key[3])
                target_w = int(bucket_key[4])

                if (target_h, target_w) != (ROOM_HEIGHT, ROOM_WIDTH):
                    logger.warning(
                        "Room-size bucket (%d,%d) is non-canonical; using sequential fallback for %d room(s).",
                        target_h,
                        target_w,
                        len(bucket_room_ids),
                    )
                    for room_id in bucket_room_ids:
                        idx = offset
                        neighbor_latents = pipeline._get_neighbor_latents(
                            room_id, mission_graph_physical, room_latents
                        )
                        reference_room_maps = (
                            pipeline._get_neighbor_reference_room_maps(room_id, mission_graph_physical, rooms)
                            if bool(getattr(pipeline.condition_encoder, "use_reference_room_maps", False))
                            else None
                        )
                        start_goal = pipeline._extract_room_start_goal(mission_graph_physical, room_id)
                        boundary_constraints = pipeline._build_room_boundary_constraints(
                            graph=mission_graph_physical,
                            room_id=room_id,
                        )
                        room_position = pipeline._build_room_position_tensor(
                            graph=mission_graph_physical,
                            room_id=room_id,
                            fallback_order_index=idx,
                        )
                        room_seed = None
                        if seed is not None:
                            room_seed = int(seed) + int(_stable_node_seed_offset(room_id))
                        room_graph_context = pipeline._build_room_graph_context(
                            graph_data=graph_data,
                            mission_graph=mission_graph_physical,
                            room_id=room_id,
                            start_goal=start_goal,
                        )
                        room_result = pipeline.generate_room(
                            neighbor_latents=neighbor_latents,
                            graph_context=room_graph_context,
                            room_id=room_id,
                            boundary_constraints=boundary_constraints,
                            position=room_position,
                            reference_room_maps=reference_room_maps,
                            guidance_scale=guidance_scale,
                            logic_guidance_scale=logic_guidance_scale,
                            num_diffusion_steps=num_diffusion_steps,
                            use_fast_sampling=use_fast_sampling,
                            latent_sampler=latent_sampler,
                            categorical_codebook_size=categorical_codebook_size,
                            apply_repair=apply_repair,
                            start_goal_coords=start_goal,
                            seed=room_seed,
                        )
                        rooms[room_id] = room_result
                        room_latents[room_id] = pipeline._encode_room_grid_to_latent(room_result.room_grid)
                        offset += 1
                    continue

                requested = max(1, int(max_batch_size))
                safe_chunk = pipeline._estimate_safe_batch_size(
                    requested_batch_size=requested,
                    latent_shape_chw=latent_shape_chw,
                )
                cuda_free_mb = None
                if torch.cuda.is_available():
                    try:
                        free_bytes, _total_bytes = torch.cuda.mem_get_info(device=pipeline.device)
                        cuda_free_mb = float(free_bytes) / (1024.0 * 1024.0)
                    except Exception:
                        cuda_free_mb = None

                logger.info(
                    "Batch planner: layer=%d bucket(latent=%s,target=%dx%d) rooms=%d requested=%d safe_chunk=%d cuda_free_mb=%s",
                    int(layer_idx),
                    str(latent_shape_chw),
                    int(target_h),
                    int(target_w),
                    int(len(bucket_room_ids)),
                    int(requested),
                    int(safe_chunk),
                    (f"{cuda_free_mb:.1f}" if isinstance(cuda_free_mb, float) else "n/a"),
                )

                batch_runtime_diagnostics.append(
                    {
                        'layer_index': int(layer_idx),
                        'latent_shape_chw': [int(latent_shape_chw[0]), int(latent_shape_chw[1]), int(latent_shape_chw[2])],
                        'target_room_size_hw': [int(target_h), int(target_w)],
                        'bucket_room_count': int(len(bucket_room_ids)),
                        'requested_batch_size': int(requested),
                        'safe_chunk_size': int(safe_chunk),
                        'cuda_free_mb': float(cuda_free_mb) if isinstance(cuda_free_mb, float) else float('nan'),
                    }
                )

                for k in range(0, len(bucket_room_ids), safe_chunk):
                    batch_room_ids = list(bucket_room_ids[k:k + safe_chunk])
                    logger.debug(
                        "Batch execute: layer=%d bucket=%s chunk_start=%d chunk_size=%d",
                        int(layer_idx),
                        str(bucket_key),
                        int(k),
                        int(len(batch_room_ids)),
                    )
                    batch_runtime_diagnostics.append(
                        {
                            'layer_index': int(layer_idx),
                            'bucket_key': str(bucket_key),
                            'chunk_start': int(k),
                            'actual_chunk_size': int(len(batch_room_ids)),
                        }
                    )
                    try:
                        batch_results = pipeline._generate_room_batch(
                            room_ids=batch_room_ids,
                            mission_graph_physical=mission_graph_physical,
                            graph_data=graph_data,
                            generated_rooms=rooms,
                            room_latents=room_latents,
                            guidance_scale=guidance_scale,
                            logic_guidance_scale=logic_guidance_scale,
                            num_diffusion_steps=num_diffusion_steps,
                            use_fast_sampling=use_fast_sampling,
                            latent_sampler=latent_sampler,
                            categorical_codebook_size=categorical_codebook_size,
                            apply_repair=apply_repair,
                            seed=seed,
                            layer_offset=offset,
                            latent_shape_chw=latent_shape_chw,
                        )
                    except (RuntimeError, ValueError) as exc:
                        pipeline._bump_diagnostic("batched_room_generation_fallback")
                        logger.warning(
                            "Batched room generation failed for chunk %s at layer %d; falling back to sequential generation. Error: %s",
                            batch_room_ids,
                            int(layer_idx),
                            exc,
                        )
                        if torch.cuda.is_available():
                            pipeline._synchronize_cuda_device()
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                logger.debug("torch.cuda.empty_cache() skipped after batch failure.", exc_info=True)
                        batch_runtime_diagnostics.append(
                            {
                                'layer_index': int(layer_idx),
                                'bucket_key': str(bucket_key),
                                'chunk_start': int(k),
                                'actual_chunk_size': int(len(batch_room_ids)),
                                'sequential_fallback_after_batch_error': 1,
                                'batch_error': str(exc),
                            }
                        )
                        batch_results = {}
                        for seq_offset, room_id in enumerate(batch_room_ids):
                            idx = offset + seq_offset
                            neighbor_latents = pipeline._get_neighbor_latents(
                                room_id, mission_graph_physical, room_latents
                            )
                            reference_room_maps = (
                                pipeline._get_neighbor_reference_room_maps(room_id, mission_graph_physical, rooms)
                                if bool(getattr(pipeline.condition_encoder, "use_reference_room_maps", False))
                                else None
                            )
                            start_goal = pipeline._extract_room_start_goal(mission_graph_physical, room_id)
                            boundary_constraints = pipeline._build_room_boundary_constraints(
                                graph=mission_graph_physical,
                                room_id=room_id,
                            )
                            room_position = pipeline._build_room_position_tensor(
                                graph=mission_graph_physical,
                                room_id=room_id,
                                fallback_order_index=idx,
                            )
                            room_seed = None
                            if seed is not None:
                                room_seed = int(seed) + int(_stable_node_seed_offset(room_id))
                            room_graph_context = pipeline._build_room_graph_context(
                                graph_data=graph_data,
                                mission_graph=mission_graph_physical,
                                room_id=room_id,
                                start_goal=start_goal,
                            )
                            batch_results[room_id] = pipeline.generate_room(
                                neighbor_latents=neighbor_latents,
                                graph_context=room_graph_context,
                                room_id=room_id,
                                boundary_constraints=boundary_constraints,
                                position=room_position,
                                reference_room_maps=reference_room_maps,
                                guidance_scale=guidance_scale,
                                logic_guidance_scale=logic_guidance_scale,
                                num_diffusion_steps=num_diffusion_steps,
                                use_fast_sampling=use_fast_sampling,
                                latent_sampler=latent_sampler,
                                categorical_codebook_size=categorical_codebook_size,
                                apply_repair=apply_repair,
                                start_goal_coords=start_goal,
                                seed=room_seed,
                            )
                    for room_id in batch_room_ids:
                        room_result = batch_results[room_id]
                        rooms[room_id] = room_result
                        room_latents[room_id] = pipeline._encode_room_grid_to_latent(room_result.room_grid)
                    offset += len(batch_room_ids)
    else:
        generation_order = sorted(
            mission_graph_physical.nodes(),
            key=_stable_node_sort_key,
        )
        for idx, room_id in enumerate(generation_order):
            logger.debug("Generating room %s (%d/%d)", room_id, idx + 1, len(generation_order))
            neighbor_latents = pipeline._get_neighbor_latents(
                room_id, mission_graph_physical, room_latents
            )
            reference_room_maps = (
                pipeline._get_neighbor_reference_room_maps(room_id, mission_graph_physical, rooms)
                if bool(getattr(pipeline.condition_encoder, "use_reference_room_maps", False))
                else None
            )
            start_goal = pipeline._extract_room_start_goal(mission_graph_physical, room_id)
            boundary_constraints = pipeline._build_room_boundary_constraints(
                graph=mission_graph_physical,
                room_id=room_id,
            )
            room_position = pipeline._build_room_position_tensor(
                graph=mission_graph_physical,
                room_id=room_id,
                fallback_order_index=idx,
            )
            room_seed = None
            if seed is not None:
                room_seed = int(seed) + int(_stable_node_seed_offset(room_id))
            room_graph_context = pipeline._build_room_graph_context(
                graph_data=graph_data,
                mission_graph=mission_graph_physical,
                room_id=room_id,
                start_goal=start_goal,
            )

            room_result = pipeline.generate_room(
                neighbor_latents=neighbor_latents,
                graph_context=room_graph_context,
                room_id=room_id,
                boundary_constraints=boundary_constraints,
                position=room_position,
                reference_room_maps=reference_room_maps,
                guidance_scale=guidance_scale,
                logic_guidance_scale=logic_guidance_scale,
                num_diffusion_steps=num_diffusion_steps,
                use_fast_sampling=use_fast_sampling,
                latent_sampler=latent_sampler,
                categorical_codebook_size=categorical_codebook_size,
                apply_repair=apply_repair,
                start_goal_coords=start_goal,
                seed=room_seed,
            )

            rooms[room_id] = room_result
            room_latents[room_id] = pipeline._encode_room_grid_to_latent(room_result.room_grid)

    return GeneratedRoomSet(
        rooms=rooms,
        room_latents=room_latents,
        batch_runtime_diagnostics=batch_runtime_diagnostics,
    )


@torch.no_grad()
def generate_dungeon(
    pipeline,
    mission_graph: Optional[nx.Graph] = None,
    guidance_scale: Optional[float] = None,
    logic_guidance_scale: Optional[float] = None,
    num_diffusion_steps: Optional[int] = None,
    use_fast_sampling: Optional[bool] = None,
    latent_sampler: Optional[str] = None,
    categorical_codebook_size: Optional[int] = None,
    use_topological_positional_encoding: Optional[bool] = None,
    apply_repair: Optional[bool] = None,
    seed: Optional[int] = None,
    enable_map_elites: Optional[bool] = None,
    # Block I: Evolutionary generation parameters
    generate_topology: bool = False,
    target_curve: Optional[List[float]] = None,
    num_rooms: Optional[int] = None,
    population_size: Optional[int] = None,
    generations: Optional[int] = None,
    mutation_rate: Optional[float] = None,
    crossover_rate: Optional[float] = None,
    genome_length: Optional[int] = None,
    rule_space: Optional[str] = None,
    transition_mix: Optional[float] = None,
    search_strategy: Optional[str] = None,
    qd_archive_cells: Optional[int] = None,
    qd_init_random_fraction: Optional[float] = None,
    qd_emitter_mutation_rate: Optional[float] = None,
    qd_archive_path: Optional[str] = None,
    qd_load_archive: Optional[bool] = None,
    qd_autosave_archive: Optional[bool] = None,
    max_lock_key_rules: Optional[int] = None,
    enable_rule_credit_assignment: Optional[bool] = None,
    enforce_generation_constraints: Optional[bool] = None,
    allow_candidate_repairs: Optional[bool] = None,
    batch_independent_rooms: bool = True,
    max_batch_size: int = 8,
) -> DungeonGenerationResult:
    """
    Generate a complete multi-room dungeon using graph-guided generation.

    This integrates all 7 blocks of the H-MOLQD pipeline:
    - Block I: Evolutionary Topology Director (optional, if generate_topology=True)
    - Block II: VQ-VAE latent encoding/decoding
    - Block III: Dual-stream condition encoding
    - Block IV: Latent diffusion with guidance
    - Block V: LogicNet differentiable solvability
    - Block VI: Symbolic WaveFunctionCollapse repair
    - Block VII: MAP-Elites quality-diversity evaluation

    VGLC Compliance: Filters virtual nodes before generation.

    Args:
        mission_graph: NetworkX graph with room nodes and door edges
                      If None and generate_topology=True, will generate automatically
        guidance_scale: Classifier-free guidance scale
        logic_guidance_scale: LogicNet gradient guidance scale
        num_diffusion_steps: Number of diffusion steps per room
        latent_sampler: "diffusion" (default) or "categorical"
        categorical_codebook_size: Optional cap for categorical codebook sampling
        use_topological_positional_encoding: Include TPE features in graph context
        apply_repair: Apply symbolic repair to each room
        seed: Random seed for reproducibility
        enable_map_elites: Compute MAP-Elites metrics
        generate_topology: Use Block I to evolve mission graph (if mission_graph=None)
        target_curve: Difficulty curve for evolutionary search [0.0-1.0]
        num_rooms: Number of rooms for generated topology
        population_size: Evolution population size
        generations: Number of evolutionary generations
        mutation_rate: Per-gene mutation probability for Block I
        crossover_rate: Crossover probability for Block I
        genome_length: Optional fixed genome length (0/None => auto)
        rule_space: Grammar rule-space (`core` or `full`)
        transition_mix: Mix between transition-biased and global rule priors
        search_strategy: Search backend (`ga` or `cvt_emitter` aliases)
        qd_archive_cells: CVT archive cells when using QD search
        qd_init_random_fraction: Bootstrap random fraction for QD search
        qd_emitter_mutation_rate: Emitter mutation rate for QD search
        max_lock_key_rules: Soft cap on InsertLockKey use per genome
        enable_rule_credit_assignment: Enable adaptive rule-credit assignment
        enforce_generation_constraints: Reject invalid intermediate candidates
        allow_candidate_repairs: Attempt local candidate repairs when constraints fail

    Returns:
        DungeonGenerationResult with complete dungeon and metrics
    """
    start_time = time.time()
    stage_times: Dict[str, float] = {}
    runtime_diagnostics_before = dict(pipeline.runtime_diagnostics)
    guidance_scale = pipeline.default_guidance_scale if guidance_scale is None else float(guidance_scale)
    logic_guidance_scale = (
        pipeline.default_logic_guidance_scale
        if logic_guidance_scale is None
        else float(logic_guidance_scale)
    )
    num_diffusion_steps = (
        pipeline.default_num_diffusion_steps if num_diffusion_steps is None else int(num_diffusion_steps)
    )
    use_fast_sampling = (
        pipeline.default_use_fast_sampling if use_fast_sampling is None else bool(use_fast_sampling)
    )
    latent_sampler = pipeline.default_latent_sampler if latent_sampler is None else str(latent_sampler)
    pipeline._require_room_generation_components(
        "generate_dungeon",
        latent_sampler=latent_sampler,
    )
    if categorical_codebook_size is None and pipeline.default_categorical_codebook_size is not None:
        categorical_codebook_size = int(pipeline.default_categorical_codebook_size)
    use_topological_positional_encoding = (
        pipeline.default_use_topological_positional_encoding
        if use_topological_positional_encoding is None
        else bool(use_topological_positional_encoding)
    )
    apply_repair = pipeline.default_apply_repair if apply_repair is None else bool(apply_repair)
    enable_map_elites = (
        pipeline.default_enable_map_elites if enable_map_elites is None else bool(enable_map_elites)
    )

    if seed is not None:
        torch.manual_seed(seed)

    if apply_repair and pipeline.refiner is None:
        pipeline._bump_diagnostic("repair_disabled_missing_component")
        logger.warning(
            "Dungeon generation requested symbolic repair, but no refiner component is configured; disabling repair."
        )
        apply_repair = False
    if enable_map_elites and pipeline.map_elites is None:
        pipeline._bump_diagnostic("map_elites_disabled_missing_component")
        logger.warning(
            "Dungeon generation requested MAP-Elites evaluation, but no map_elites component is configured; disabling evaluation."
        )
        enable_map_elites = False

    stage_started_at = time.perf_counter()
    prepared = pipeline.prepare_dungeon_generation(
        mission_graph=mission_graph,
        use_topological_positional_encoding=use_topological_positional_encoding,
        generate_topology=generate_topology,
        target_curve=target_curve,
        num_rooms=num_rooms,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        genome_length=genome_length,
        rule_space=rule_space,
        transition_mix=transition_mix,
        search_strategy=search_strategy,
        qd_archive_cells=qd_archive_cells,
        qd_init_random_fraction=qd_init_random_fraction,
        qd_emitter_mutation_rate=qd_emitter_mutation_rate,
        qd_archive_path=qd_archive_path,
        qd_load_archive=qd_load_archive,
        qd_autosave_archive=qd_autosave_archive,
        max_lock_key_rules=max_lock_key_rules,
        enable_rule_credit_assignment=enable_rule_credit_assignment,
        enforce_generation_constraints=enforce_generation_constraints,
        allow_candidate_repairs=allow_candidate_repairs,
        seed=seed,
    )
    _record_stage_time(stage_times, "prepare_dungeon_generation_time_sec", stage_started_at)

    stage_started_at = time.perf_counter()
    room_set = pipeline.generate_rooms_for_graph(
        prepared,
        guidance_scale=guidance_scale,
        logic_guidance_scale=logic_guidance_scale,
        num_diffusion_steps=num_diffusion_steps,
        use_fast_sampling=use_fast_sampling,
        latent_sampler=latent_sampler,
        categorical_codebook_size=categorical_codebook_size,
        apply_repair=apply_repair,
        seed=seed,
        batch_independent_rooms=batch_independent_rooms,
        max_batch_size=max_batch_size,
    )
    _record_stage_time(stage_times, "generate_rooms_for_graph_time_sec", stage_started_at)

    stage_started_at = time.perf_counter()
    stitched_layout = pipeline.stitch_room_layout(room_set.rooms, prepared.mission_graph_physical)
    dungeon_grid = np.asarray(stitched_layout.dungeon_grid, dtype=np.int32)
    _record_stage_time(stage_times, "stitch_room_layout_time_sec", stage_started_at)

    stage_started_at = time.perf_counter()
    puzzle_metadata = pipeline._globalize_room_puzzle_metadata(
        rooms=room_set.rooms,
        stitched_layout=stitched_layout,
    )
    _record_stage_time(stage_times, "globalize_puzzle_metadata_time_sec", stage_started_at)

    stage_started_at = time.perf_counter()
    map_elites_score = pipeline.evaluate_generated_dungeon(
        dungeon_grid,
        prepared.mission_graph_physical,
        enable_map_elites=enable_map_elites,
    )
    _record_stage_time(stage_times, "evaluate_generated_dungeon_time_sec", stage_started_at)

    stage_started_at = time.perf_counter()
    try:
        logic_solvability = pipeline.evaluate_dungeon_solvability(
            room_set.rooms,
            prepared.mission_graph_physical,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.debug("LogicNet dungeon solvability metrics failed: %s", exc)
        logic_solvability = {}
    _record_stage_time(stage_times, "evaluate_dungeon_solvability_time_sec", stage_started_at)

    # Compute overall metrics
    generation_time = time.time() - start_time
    num_rooms_generated = len(room_set.rooms)
    room_metric_dicts = [dict(r.metrics) for r in room_set.rooms.values()]
    alignment_metrics = pipeline._aggregate_room_alignment_metrics(room_metric_dicts)
    room_stage_times = _aggregate_room_stage_times(room_metric_dicts)
    stage_times["generation_total_time_sec"] = float(generation_time)
    metrics = {
        'num_rooms': num_rooms_generated,
        'repair_count': int(sum(int(r.metrics.get('repair_count', int(bool(r.was_repaired)))) for r in room_set.rooms.values())),
        'repair_time_sec': float(sum(float(r.metrics.get('repair_time_sec', 0.0)) for r in room_set.rooms.values())),
        'total_tiles_repaired': sum(r.metrics.get('tiles_changed', 0) for r in room_set.rooms.values()),
        'repair_rate': (
            sum(r.was_repaired for r in room_set.rooms.values()) / max(1, num_rooms_generated)
            if num_rooms_generated > 0
            else 0.0
        ),
        'dungeon_shape': dungeon_grid.shape,
        'generation_time_sec': generation_time,
        'stage_timing_sec': dict(stage_times),
        'room_stage_timing_sec': room_stage_times,
        'batch_generation_diagnostics': room_set.batch_runtime_diagnostics,
        'runtime_diagnostics': dict(pipeline.runtime_diagnostics),
        'runtime_diagnostics_delta': {
            key: int(value) - int(runtime_diagnostics_before.get(key, 0))
            for key, value in sorted(pipeline.runtime_diagnostics.items())
            if int(value) != int(runtime_diagnostics_before.get(key, 0))
        },
        'puzzle_plan_count': int(len(dict(puzzle_metadata.get('plans', {}) or {}))),
        'puzzle_stage_count': int(
            sum(
                len(list(dict(plan or {}).get('stage_sequence', []) or []))
                for plan in dict(puzzle_metadata.get('plans', {}) or {}).values()
            )
        ),
        'logicnet_dungeon_solvability': float(logic_solvability.get('solvability_score', 0.0)),
        'logicnet_room_solvability': float(logic_solvability.get('room_solvability_score', 0.0)),
        'logicnet_graph_reach_loss': float(logic_solvability.get('graph_reach_loss', 0.0)),
        'logicnet_lock_loss': float(logic_solvability.get('lock_loss', 0.0)),
        'logicnet_global_logic_loss': float(logic_solvability.get('global_logic_loss', 0.0)),
        'logicnet_num_failing_rooms': float(logic_solvability.get('num_failing', 0.0)),
        **alignment_metrics,
    }

    logger.info(f"Dungeon generated in {generation_time:.2f}s "
               f"(repair_rate={metrics['repair_rate']:.1%})")

    return DungeonGenerationResult(
        dungeon_grid=dungeon_grid,
        rooms=room_set.rooms,
        mission_graph=prepared.mission_graph,  # Preserve caller graph identity
        metrics=metrics,
        map_elites_score=map_elites_score,
        generation_time=generation_time,
        stitched_layout=stitched_layout,
        puzzle_metadata=puzzle_metadata,
    )


class DungeonAssembler:
    """
    Boundary for graph preparation, room stitching, and dungeon assembly.

    Owns the high-level graph preparation and dungeon generation loops.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def prepare_dungeon_generation(self, *args: Any, **kwargs: Any) -> Any:
        return prepare_dungeon_generation(self.engine, *args, **kwargs)

    def generate_rooms_for_graph(self, *args: Any, **kwargs: Any) -> Any:
        return generate_rooms_for_graph(self.engine, *args, **kwargs)

    def generate_dungeon(self, *args: Any, **kwargs: Any) -> Any:
        return generate_dungeon(self.engine, *args, **kwargs)

    def stitch_rooms(self, *args: Any, **kwargs: Any) -> Any:
        return stitch_rooms(self.engine, *args, **kwargs)

    def stitch_room_layout(self, *args: Any, **kwargs: Any) -> StitchedRoomLayout:
        return stitch_room_layout(self.engine, *args, **kwargs)


__all__ = [
    "DungeonAssembler",
    "prepare_dungeon_generation",
    "generate_rooms_for_graph",
    "generate_dungeon",
    "stitch_room_layout",
    "stitch_rooms",
    "stitch_room_grid",
    "carve_room_connection_with_fallback",
]
