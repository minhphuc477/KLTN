"""Evaluation, repair, and symbolic-only assembly helpers for the pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from src.core import SEMANTIC_PALETTE
from src.evaluation.tile_distribution import compare_tile_pattern_distributions
from src.pipeline.spatial_utils import coerce_bool, fit_room_grid
from src.pipeline.types import DungeonGenerationResult, MissingPipelineComponentError, RoomGenerationResult
from src.zelda_data.vglc_utils import get_physical_start_node

logger = logging.getLogger(__name__)


def _stitched_validation_context(
    graph: Optional[nx.Graph],
    stitched_layout: Optional[Any],
) -> Dict[str, Any]:
    """Build the exact graph/room mapping contract consumed by ZeldaLogicEnv."""
    if graph is None or stitched_layout is None:
        return {}
    slot_positions = dict(getattr(stitched_layout, "slot_positions", {}) or {})
    room_offsets = dict(getattr(stitched_layout, "room_offsets", {}) or {})
    if not slot_positions or not room_offsets:
        return {"graph": graph}
    room_to_node = {
        tuple(slot): node_id
        for node_id, slot in slot_positions.items()
        if node_id in graph and node_id in room_offsets
    }
    room_positions = {
        tuple(slot_positions[node_id]): tuple(room_offsets[node_id])
        for node_id in room_to_node.values()
    }
    node_to_room = {node_id: room for room, node_id in room_to_node.items()}
    return {
        "graph": graph,
        "room_to_node": room_to_node,
        "room_positions": room_positions,
        "node_to_room": node_to_room,
    }


def _hard_oracle_verdict(validation: Dict[str, Any]) -> Optional[bool]:
    """Return a definitive hard verdict, preserving indeterminate outcomes."""
    if not validation or not bool(validation.get("is_exact", False)):
        return None
    status = str(validation.get("termination_status", "unknown")).strip().lower()
    if status in {"budget_exhausted", "unknown", "validator_error"}:
        return None
    if bool(validation.get("solvable", False)):
        return True
    if status == "invalid" or bool(validation.get("proven_unsolvable", False)):
        return False
    return None


def _logicnet_hard_agreement(
    logic_solvability: Dict[str, Any],
    hard_validation: Dict[str, Any],
) -> Optional[bool]:
    """Compare models only when both produced semantically comparable verdicts."""
    hard_verdict = _hard_oracle_verdict(hard_validation)
    if hard_verdict is None or not bool(logic_solvability.get("is_graph_level", False)):
        return None
    soft_verdict = float(logic_solvability.get("solvability_score", 0.0)) >= 0.5
    return bool(soft_verdict == hard_verdict)


def _reference_room_grids_for_distribution(pipeline) -> List[np.ndarray]:
    """Return optional reference grids configured on the pipeline for realism metrics."""
    for attr_name in (
        "evaluation_reference_rooms",
        "reference_room_grids",
        "reference_rooms",
        "vglc_reference_rooms",
    ):
        value = getattr(pipeline, attr_name, None)
        if value is None:
            continue
        if isinstance(value, dict):
            candidates = value.values()
        else:
            candidates = value
        grids: List[np.ndarray] = []
        try:
            for item in candidates:
                array = np.asarray(item)
                if array.ndim == 2:
                    grids.append(array.astype(np.int64, copy=False))
        except TypeError:
            continue
        if grids:
            return grids
    return []

def repair_room(
    pipeline,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    required_floor_mask: Optional[np.ndarray] = None,
    feedback_callback: Optional[Any] = None,
    max_feedback_rounds: int = 0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Public symbolic-only room repair entry point.

    This is intentionally usable on pipelines created via
    `create_symbolic_repair_pipeline()` without loading the neural stack.
    """
    refiner = pipeline._require_component("refiner", "repair_room")
    repaired_grid, success, diagnostics = refiner.repair_room_with_feedback(
        grid=np.asarray(grid, dtype=np.int32),
        start=pipeline._normalize_room_coord(start, field_name="start"),
        goal=pipeline._normalize_room_coord(goal, field_name="goal"),
        required_floor_mask=(
            np.asarray(required_floor_mask, dtype=bool)
            if isinstance(required_floor_mask, np.ndarray)
            else None
        ),
        feedback_callback=feedback_callback,
        max_feedback_rounds=max_feedback_rounds,
        seed=seed,
    )
    return repaired_grid, bool(success), diagnostics


def evaluate_generated_dungeon(
    pipeline,
    dungeon_grid: np.ndarray,
    mission_graph_physical: nx.Graph,
    *,
    enable_map_elites: bool = True,
    room_puzzle_metadata: Optional[Dict[str, Any]] = None,
    stitched_layout: Optional[Any] = None,
    generated_rooms: Optional[Dict[Any, RoomGenerationResult]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a stitched dungeon grid with MAP-Elites when available.

    Returns `None` when evaluation is disabled or not applicable.
    """
    map_elites = getattr(pipeline, "map_elites", None)
    map_elites_available = bool(enable_map_elites and map_elites is not None)
    reference_rooms = _reference_room_grids_for_distribution(pipeline)
    if not map_elites_available and not reference_rooms:
        return None
    validator_fn = getattr(pipeline, "_validate_dungeon", None)
    validation_context = _stitched_validation_context(
        mission_graph_physical,
        stitched_layout,
    )
    solver_result = (
        validator_fn(
            dungeon_grid,
            room_puzzle_metadata=room_puzzle_metadata,
            **validation_context,
        )
        if callable(validator_fn)
        else _validate_dungeon(
            pipeline,
            dungeon_grid,
            room_puzzle_metadata=room_puzzle_metadata,
            **validation_context,
        )
    )
    map_elites_score: Dict[str, Any] = dict(solver_result or {})
    if map_elites_available:
        try:
            descriptor_metrics = map_elites.add_dungeon(
                dungeon=dungeon_grid,
                grid=dungeon_grid,
                solver_result=solver_result or {},
                mission_graph=mission_graph_physical,
                generated_rooms=generated_rooms,
            )
            if solver_result and solver_result.get('solvable'):
                descriptor_metrics = descriptor_metrics or {}
                map_elites_score.update({
                'linearity': descriptor_metrics.get('linearity', solver_result.get('linearity', 0.0)),
                'leniency': descriptor_metrics.get('leniency', solver_result.get('leniency', 0.0)),
                'progression_complexity': descriptor_metrics.get('progression_complexity', 0.0),
                'topology_complexity': descriptor_metrics.get('topology_complexity', 0.0),
                'path_length': descriptor_metrics.get('path_length', solver_result.get('path_length', 0)),
                })
                if hasattr(map_elites, 'advanced_archive_stats'):
                    advanced_stats = map_elites.advanced_archive_stats()
                    if advanced_stats is not None:
                        map_elites_score['advanced_archive'] = advanced_stats
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"MAP-Elites evaluation failed: {e}")

    if reference_rooms:
        try:
            tile_metrics = compare_tile_pattern_distributions(
                generated_grids=[np.asarray(dungeon_grid, dtype=np.int64)],
                reference_grids=reference_rooms,
                pattern_size=int(max(1, int(getattr(pipeline, "tile_pattern_size", 2)))),
            ).to_dict()
            map_elites_score["tile_pattern_distribution"] = tile_metrics
            map_elites_score.update(
                {
                    "tile_pattern_js_divergence": float(tile_metrics["js_divergence"]),
                    "tile_pattern_total_variation": float(tile_metrics["total_variation"]),
                    "tile_pattern_coverage": float(tile_metrics["pattern_coverage"]),
                }
            )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning("Tile-pattern distribution evaluation failed: %s", e)

    return map_elites_score or None


@torch.no_grad()
def evaluate_dungeon_solvability(
    pipeline,
    rooms: Dict[Any, 'RoomGenerationResult'],
    mission_graph_physical: nx.Graph,
) -> Dict[str, Any]:
    """
    Evaluate dungeon-level solvability using LogicNet at the correct scope.

    This is the graph-level evaluation that was previously (incorrectly) embedded
    in the per-room training loss. A single room's z [B, 64, 4, 3] cannot encode
    inter-room key-lock ordering; only a dungeon-level assessment with the actual
    generated room connectivity can meaningfully evaluate graph solvability.

    Args:
        rooms: Dict of room_id -> RoomGenerationResult from generate_rooms_for_graph
        mission_graph_physical: The physical mission graph with edge types

    Returns:
        Dict with solvability_score, graph_reach_loss, lock_loss, and
        failing_rooms (list of room IDs where local solvability failed)
    """
    result: Dict[str, Any] = {
        'solvability_score': 0.0,
        'room_solvability_score': 0.0,
        'graph_reach_loss': 0.0,
        'lock_loss': 0.0,
        'global_logic_loss': 0.0,
        'score_type': 'logicnet_surrogate',
        'is_exact': False,
        'is_graph_level': False,
        'evaluation_status': 'unavailable',
    }
    failing_rooms: List[Any] = []

    if pipeline.logic_net is None:
        logger.debug("evaluate_dungeon_solvability skipped: no logic_net component")
        return result

    graph_context = pipeline._prepare_graph_context(mission_graph_physical, use_tpe=True)
    node_to_idx = dict(graph_context.get('node_to_idx', {}) or {})

    def _first_matching_node(*keys: str) -> Optional[Any]:
        for node_id, attrs in mission_graph_physical.nodes(data=True):
            attrs_dict = dict(attrs)
            role_flags = pipeline._room_role_flags(attrs_dict)
            if any(coerce_bool(attrs_dict.get(key)) for key in keys):
                return node_id
            if {"is_start", "is_entry"} & set(keys) and role_flags.get("is_start", False):
                return node_id
            if {"has_triforce", "is_triforce", "is_goal"} & set(keys) and role_flags.get("has_goal", False):
                return node_id
        return None

    start_node = get_physical_start_node(mission_graph_physical)
    if start_node is None or start_node not in node_to_idx:
        start_node = _first_matching_node("is_start", "is_entry")
    if start_node is None or start_node not in node_to_idx:
        start_node = next(iter(node_to_idx.keys()), None)
    start_idx = int(node_to_idx.get(start_node, 0)) if start_node is not None else 0

    target_node = _first_matching_node("has_triforce", "is_triforce", "is_goal")
    target_idx = int(node_to_idx.get(target_node, -1)) if target_node is not None else -1

    dungeon_latents: List[torch.Tensor] = []
    current_node_indices: List[int] = []
    evaluated_room_ids: List[Any] = []
    room_topology_maps: List[torch.Tensor] = []
    room_boundaries: List[torch.Tensor] = []
    room_logic_masks: Dict[str, List[torch.Tensor]] = {
        "logic_source_mask": [],
        "logic_target_mask": [],
    }
    reference_shape: Optional[Tuple[int, ...]] = None

    # Evaluate per-room walkability via LogicNet (grid-level only)
    total_grid_reach = 0.0
    num_rooms = 0
    for room_id, room_result in rooms.items():
        if room_result.latent is None:
            continue
        z = room_result.latent
        if not isinstance(z, torch.Tensor) or z.numel() == 0:
            continue
        z = z.to(pipeline.device)
        if z.dim() == 3:
            z = z.unsqueeze(0)
        if z.dim() != 4:
            logger.debug("Room %s solvability eval skipped: latent shape %s is not rank-4", room_id, tuple(z.shape))
            continue
        if reference_shape is None:
            reference_shape = tuple(z.shape[1:])
        elif tuple(z.shape[1:]) != reference_shape:
            logger.debug(
                "Room %s solvability eval skipped: latent shape %s does not match %s",
                room_id,
                tuple(z.shape[1:]),
                reference_shape,
            )
            continue

        node_idx = node_to_idx.get(room_id)
        if node_idx is not None:
            dungeon_latents.append(z[:1])
            current_node_indices.append(int(node_idx))
            evaluated_room_ids.append(room_id)

        try:
            start_goal = pipeline._extract_room_start_goal(mission_graph_physical, room_id)
            room_graph_data = pipeline._build_room_graph_context(
                graph_data=graph_context,
                mission_graph=mission_graph_physical,
                room_id=room_id,
                start_goal=start_goal,
            )
            room_graph_data["boundary_constraints"] = pipeline._build_room_boundary_constraints(
                mission_graph_physical,
                room_id,
            )
            loss, info = pipeline.logic_net(z, graph_data=room_graph_data)
            grid_reach = float(info.get('grid_reachability', 0.0))
            total_grid_reach += grid_reach
            num_rooms += 1
            if grid_reach < 0.5:
                failing_rooms.append(room_id)
            if node_idx is not None:
                topology = room_graph_data.get("room_topology_map")
                boundary = room_graph_data.get("boundary_constraints")
                if isinstance(topology, torch.Tensor):
                    room_topology_maps.append(topology)
                if isinstance(boundary, torch.Tensor):
                    room_boundaries.append(boundary)
                for mask_name, mask_rows in room_logic_masks.items():
                    mask = room_graph_data.get(mask_name)
                    if isinstance(mask, torch.Tensor):
                        mask_rows.append(mask)
        except (RuntimeError, ValueError) as e:
            logger.debug("Room %s solvability eval failed: %s", room_id, e)
            failing_rooms.append(room_id)

    if num_rooms > 0:
        result['room_solvability_score'] = total_grid_reach / num_rooms

    if dungeon_latents:
        z_dungeon = torch.cat(dungeon_latents, dim=0)
        graph_data = {
            'graph_scope': 'dungeon',
            'node_features': graph_context.get('node_features'),
            'edge_index': graph_context.get('edge_index'),
            'edge_features': graph_context.get('edge_features'),
            'tpe': graph_context.get('tpe'),
            'node_positions': graph_context.get('node_positions'),
            'node_mask': graph_context.get('node_mask'),
            'current_node_idx': torch.tensor(current_node_indices, device=pipeline.device, dtype=torch.long),
            'start_node_id': int(start_idx),
            'target_idx': int(target_idx),
            'key_lock_pairs': list(graph_context.get('key_lock_pairs', []) or []),
        }
        if len(room_topology_maps) == len(dungeon_latents):
            graph_data["room_topology_map"] = torch.cat(room_topology_maps, dim=0)
        if len(room_boundaries) == len(dungeon_latents):
            graph_data["boundary_constraints"] = torch.cat(room_boundaries, dim=0)
        for mask_name, mask_rows in room_logic_masks.items():
            if len(mask_rows) == len(dungeon_latents):
                graph_data[mask_name] = torch.cat(mask_rows, dim=0)
        try:
            global_loss, global_info = pipeline.logic_net(z_dungeon, graph_data=graph_data)
            global_logic_loss = global_info.get('global_logic_loss', global_loss)
            if isinstance(global_logic_loss, torch.Tensor):
                global_logic_loss_value = float(global_logic_loss.detach().mean().item())
            else:
                global_logic_loss_value = float(global_logic_loss)

            reachability_value = global_info.get('global_graph_reachability')
            if isinstance(reachability_value, torch.Tensor):
                graph_score = float(reachability_value.detach().mean().item())
            elif reachability_value is not None:
                graph_score = float(reachability_value)
            else:
                graph_score = float(torch.exp(torch.tensor(-max(0.0, global_logic_loss_value))).item())

            result.update({
                'solvability_score': max(0.0, min(1.0, graph_score)),
                'graph_reach_loss': float(global_info.get('graph_reach_loss', global_info.get('global_graph_reach_loss', 0.0))) if not isinstance(global_info.get('graph_reach_loss'), torch.Tensor) else float(global_info['graph_reach_loss'].detach().mean().item()),
                'lock_loss': float(global_info.get('lock_loss', 0.0)) if not isinstance(global_info.get('lock_loss'), torch.Tensor) else float(global_info['lock_loss'].detach().mean().item()),
                'global_logic_loss': global_logic_loss_value,
                'global_loss_score': float(torch.exp(torch.tensor(-max(0.0, global_logic_loss_value))).item()),
                'global_room_passability': float(global_info.get('global_room_passability', 0.0)) if not isinstance(global_info.get('global_room_passability'), torch.Tensor) else float(global_info['global_room_passability'].detach().mean().item()),
                'global_num_rooms_scored': float(len(evaluated_room_ids)),
                'is_graph_level': True,
                'evaluation_status': 'evaluated',
            })
        except (RuntimeError, ValueError) as e:
            logger.debug("Dungeon-scope LogicNet solvability eval failed: %s", e)
            result['solvability_score'] = result['room_solvability_score']
            result['evaluation_status'] = 'room_only_fallback'
            result['error_message'] = str(e)
    else:
        result['solvability_score'] = result['room_solvability_score']
        result['evaluation_status'] = 'room_only'

    result['failing_rooms'] = failing_rooms  # type: ignore[assignment]
    result['num_rooms_evaluated'] = float(num_rooms)
    result['num_failing'] = float(len(failing_rooms))

    logger.info(
        "Dungeon solvability: %.3f global, %.3f room-local (%d/%d rooms passing)",
        result['solvability_score'],
        result['room_solvability_score'],
        num_rooms - len(failing_rooms),
        num_rooms,
    )
    return result


def repair_and_stitch_dungeon(
    pipeline,
    *,
    rooms: Dict[Any, Any],
    mission_graph: nx.Graph,
    apply_repair: bool = True,
    enable_map_elites: bool = False,
) -> DungeonGenerationResult:
    """
    Public symbolic-only dungeon assembly entry point.

    This path accepts pre-existing room grids or `RoomGenerationResult`
    objects, optionally repairs each room, then stitches and evaluates the
    final dungeon without requiring the neural generation stack.
    """
    if mission_graph is None:
        raise ValueError("repair_and_stitch_dungeon requires a mission_graph.")
    if not rooms:
        raise ValueError("repair_and_stitch_dungeon requires at least one room.")

    if apply_repair and pipeline.refiner is None:
        raise MissingPipelineComponentError(
            "repair_and_stitch_dungeon requires a refiner component when apply_repair=True."
        )

    normalized_rooms: Dict[Any, RoomGenerationResult] = {}
    for room_id, room_value in rooms.items():
        if isinstance(room_value, RoomGenerationResult):
            room_grid = fit_room_grid(room_value.room_grid)
            neural_grid = fit_room_grid(room_value.neural_grid)
            latent = room_value.latent.detach().cpu() if isinstance(room_value.latent, torch.Tensor) else torch.empty(0)
            puzzle_metadata = dict(room_value.puzzle_metadata or {})
        else:
            room_grid = fit_room_grid(room_value)
            neural_grid = room_grid.copy()
            latent = torch.empty(0)
            puzzle_metadata = {}

        room_plan_mask = None
        repaired_grid = room_grid.copy()
        was_repaired = False
        repair_diagnostics: Dict[str, Any] = {}

        if apply_repair:
            start_goal = pipeline._extract_room_start_goal(mission_graph, room_id)
            room_plan_mask = pipeline._build_room_plan_trace(
                mission_graph,
                room_id,
                repaired_grid,
                start_goal=start_goal,
            )
            repaired_grid, was_repaired, repair_diagnostics = pipeline.repair_room(
                repaired_grid,
                start=start_goal[0],
                goal=start_goal[1],
                required_floor_mask=room_plan_mask,
                seed=None,
            )

        normalized_rooms[room_id] = RoomGenerationResult(
            room_id=room_id,
            room_grid=np.asarray(repaired_grid, dtype=np.int32),
            latent=latent,
            neural_grid=np.asarray(neural_grid, dtype=np.int32),
            was_repaired=bool(was_repaired),
            raw_neural_grid=(
                fit_room_grid(room_value.raw_neural_grid)
                if isinstance(room_value, RoomGenerationResult)
                and getattr(room_value, "raw_neural_grid", None) is not None
                else np.asarray(neural_grid, dtype=np.int32)
            ),
            repair_mask=None,
            room_plan_mask=room_plan_mask,
            neural_probs=None,
            puzzle_metadata=puzzle_metadata,
            metrics={
                "room_id": room_id,
                "was_repaired": bool(was_repaired),
                "planned_traversability_pixels": float(np.sum(room_plan_mask)) if isinstance(room_plan_mask, np.ndarray) else 0.0,
                **repair_diagnostics,
            },
        )

    stitched_layout = pipeline.stitch_room_layout(normalized_rooms, mission_graph)
    dungeon_grid = np.asarray(stitched_layout.dungeon_grid, dtype=np.int32)
    puzzle_metadata = pipeline._globalize_room_puzzle_metadata(
        rooms=normalized_rooms,
        stitched_layout=stitched_layout,
    )
    map_elites_score = pipeline.evaluate_generated_dungeon(
        dungeon_grid,
        mission_graph,
        enable_map_elites=enable_map_elites,
        room_puzzle_metadata=puzzle_metadata,
        stitched_layout=stitched_layout,
        generated_rooms=normalized_rooms,
    )
    hard_validation = (
        dict(map_elites_score)
        if isinstance(map_elites_score, dict) and bool(map_elites_score.get("is_exact", False))
        else dict(
            pipeline._validate_dungeon(
                dungeon_grid,
                room_puzzle_metadata=puzzle_metadata,
                **_stitched_validation_context(mission_graph, stitched_layout),
            )
            or {}
        )
    )
    hard_verdict = _hard_oracle_verdict(hard_validation)
    try:
        logic_solvability = pipeline.evaluate_dungeon_solvability(normalized_rooms, mission_graph)
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.debug("LogicNet symbolic dungeon solvability metrics failed: %s", exc)
        logic_solvability = {}
    metrics = {
        "num_rooms": len(normalized_rooms),
        "total_tiles_repaired": sum(r.metrics.get("tiles_changed", 0) for r in normalized_rooms.values()),
        "repair_rate": (
            sum(bool(r.was_repaired) for r in normalized_rooms.values()) / max(1, len(normalized_rooms))
        ),
        "dungeon_shape": dungeon_grid.shape,
        "symbolic_only": True,
        "puzzle_plan_count": int(len(dict(puzzle_metadata.get("plans", {}) or {}))),
        "logicnet_dungeon_solvability": float(logic_solvability.get("solvability_score", 0.0)),
        "logicnet_room_solvability": float(logic_solvability.get("room_solvability_score", 0.0)),
        "logicnet_graph_reach_loss": float(logic_solvability.get("graph_reach_loss", 0.0)),
        "logicnet_lock_loss": float(logic_solvability.get("lock_loss", 0.0)),
        "logicnet_global_logic_loss": float(logic_solvability.get("global_logic_loss", 0.0)),
        "logicnet_num_failing_rooms": float(logic_solvability.get("num_failing", 0.0)),
        "final_hard_solvable": hard_verdict,
        "final_hard_validation_status": str(
            hard_validation.get("termination_status", "unknown")
        ),
        "final_hard_proven_unsolvable": bool(
            hard_validation.get("proven_unsolvable", False)
        ),
        "final_hard_states_explored": int(
            hard_validation.get("states_explored", 0) or 0
        ),
        "logicnet_hard_agreement": _logicnet_hard_agreement(
            logic_solvability,
            hard_validation,
        ),
        **dict(getattr(stitched_layout, "realization_metrics", {}) or {}),
    }
    return DungeonGenerationResult(
        dungeon_grid=dungeon_grid,
        rooms=normalized_rooms,
        mission_graph=mission_graph,
        metrics=metrics,
        map_elites_score=map_elites_score,
        generation_time=0.0,
        stitched_layout=stitched_layout,
        puzzle_metadata=puzzle_metadata,
    )


def _validate_dungeon(
    pipeline,
    dungeon_grid: np.ndarray,
    *,
    room_puzzle_metadata: Optional[Dict[str, Any]] = None,
    graph: Optional[nx.Graph] = None,
    room_to_node: Optional[Dict[Any, Any]] = None,
    room_positions: Optional[Dict[Any, Tuple[int, int]]] = None,
    node_to_room: Optional[Dict[Any, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate dungeon solvability and compute MAP-Elites descriptors.

    Uses the project validator when available, with graceful fallback.
    """
    map_elites = getattr(pipeline, "map_elites", None)
    floor_id = int(SEMANTIC_PALETTE.get('FLOOR', 1))
    enemy_id = int(SEMANTIC_PALETTE.get('ENEMY', 7))
    boss_id = int(SEMANTIC_PALETTE.get('BOSS', -1))
    key_id = int(SEMANTIC_PALETTE.get('KEY_SMALL', SEMANTIC_PALETTE.get('KEY', 8)))
    lock_id = int(SEMANTIC_PALETTE.get('DOOR_LOCKED', 11))
    playable_area = int((dungeon_grid == floor_id).sum())
    enemy_count = int((dungeon_grid == enemy_id).sum())
    if boss_id >= 0:
        enemy_count += int((dungeon_grid == boss_id).sum())
    leniency = (
        float(map_elites.calculate_leniency(dungeon_grid))
        if map_elites is not None
        else float(np.clip(1.0 - (enemy_count / max(1, playable_area)), 0.0, 1.0))
    )
    key_count = int((dungeon_grid == key_id).sum())
    lock_count = int((dungeon_grid == lock_id).sum())

    try:
        from src.simulation.validator import ZeldaValidator

        validator = ZeldaValidator()
        result = validator.validate_single(
            dungeon_grid,
            graph=graph,
            room_to_node=room_to_node,
            room_positions=room_positions,
            node_to_room=node_to_room,
            solver_timeout=int(max(1, int(getattr(pipeline, "tile_oracle_max_states", 200_000)))),
            room_puzzle_metadata=room_puzzle_metadata,
        )

        path_length = int(result.path_length) if result.is_solvable else 0
        if result.is_solvable and result.path:
            if map_elites is not None:
                linearity = float(map_elites.calculate_linearity(result.path))
            else:
                start, goal = result.path[0], result.path[-1]
                displacement = abs(int(start[0]) - int(goal[0])) + abs(int(start[1]) - int(goal[1]))
                linearity = float(np.clip(displacement / max(1, len(result.path) - 1), 0.0, 1.0))
        else:
            linearity = 0.0
        backtracking = float(np.clip(getattr(result, 'backtracking_score', 0.0), 0.0, 1.0))
        reachability = float(np.clip(getattr(result, 'reachability', 0.0), 0.0, 1.0))
        lock_pressure = min(1.0, lock_count / max(1.0, float(max(1, key_count))))
        path_pressure = min(1.0, float(path_length) / max(1.0, np.sqrt(max(1, playable_area)) * 2.5))
        progression_complexity = float(np.clip(
            (0.45 * lock_pressure) + (0.35 * backtracking) + (0.20 * path_pressure),
            0.0,
            1.0,
        ))
        topology_complexity = float(np.clip(
            (0.55 * float(np.clip(linearity, 0.0, 1.0))) + (0.45 * progression_complexity),
            0.0,
            1.0,
        ))
        quality_score = float(np.clip(
            (0.35 * reachability) +
            (0.25 * (1.0 - abs(backtracking - 0.25) / 0.25 if backtracking <= 0.5 else 0.0)) +
            (0.20 * float(result.is_valid_syntax)) +
            (0.20 * path_pressure),
            0.0,
            1.0,
        ))
        return {
            'solvable': bool(result.is_solvable),
            'path': list(result.path or []),
            'path_length': path_length,
            'linearity': linearity,
            'leniency': leniency,
            'progression_complexity': progression_complexity,
            'topology_complexity': topology_complexity,
            'quality_score': quality_score,
            'backtracking_score': backtracking,
            'reachability': reachability,
            'key_count': key_count,
            'lock_count': lock_count,
            'enemy_count': enemy_count,
            'is_valid_syntax': bool(result.is_valid_syntax),
            'error_message': str(result.error_message) if result.error_message else "",
            'solver_used': str(getattr(result, 'solver_used', 'astar')),
            'states_explored': int(getattr(result, 'states_explored', 0) or 0),
            'termination_status': str(getattr(result, 'termination_status', 'unknown')),
            'proven_unsolvable': bool(getattr(result, 'proven_unsolvable', False)),
            'final_inventory': dict(getattr(result, 'final_inventory', {}) or {}),
            'path_interactions': dict(getattr(result, 'path_interactions', {}) or {}),
            'is_exact': True,
        }
    except (AttributeError, RuntimeError, ValueError, TypeError) as e:
        bump_diagnostic = getattr(pipeline, "_bump_diagnostic", None)
        if callable(bump_diagnostic):
            bump_diagnostic("dungeon_validation_fallback")
        logger.warning(f"Dungeon validation failed: {e}")
        return {
            'solvable': None,
            'path_length': 0,
            'linearity': 0.0,
            'leniency': leniency,
            'progression_complexity': 0.0,
            'topology_complexity': 0.0,
            'quality_score': 0.0,
            'backtracking_score': 0.0,
            'reachability': 0.0,
            'key_count': key_count,
            'lock_count': lock_count,
            'enemy_count': enemy_count,
            'is_valid_syntax': False,
            'error_message': f"validator_error: {e}",
            'solver_used': 'validator_error',
            'states_explored': 0,
            'termination_status': 'validator_error',
            'proven_unsolvable': False,
            'is_exact': False,
        }

