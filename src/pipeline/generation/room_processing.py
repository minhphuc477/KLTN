"""Room-level semantic processing, puzzle scaffolding, and sampling utilities."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import torch

from src.core import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.definitions import DOOR_POSITIONS, TileID
from src.core.vqvae import canonical_latent_shape
from src.pipeline.block_contracts import BlockShapeContract, validate_feature_dims, validate_tensor_contract
from src.pipeline.repair_feedback import build_latent_edit_mask, logicnet_guided_inpaint_room
from src.pipeline.room_stitching import StitchedRoomLayout
from src.pipeline.room_topology_conditioning import (
    apply_puzzle_stage_control_to_conditioning,
    apply_puzzle_structure_control_to_conditioning,
    build_room_semantic_anchor_points,
)
from src.pipeline.spatial_utils import causal_generation_layers, stable_node_sort_key
from src.pipeline.types import RoomGenerationResult
from src.utils.stable_seed import stable_seed_offset

logger = logging.getLogger(__name__)
DEFAULT_ROOM_LATENT_HW: Tuple[int, int] = canonical_latent_shape((ROOM_HEIGHT, ROOM_WIDTH))
_stable_node_sort_key = stable_node_sort_key

def _sanitize_semantic_grid(
    pipeline,
    grid: np.ndarray,
    *,
    fallback_grid: Optional[np.ndarray] = None,
    strip_void: bool = False,
) -> Tuple[np.ndarray, int, List[int]]:
    """
    Clamp/repair semantic IDs to the canonical palette.

    Invalid tile IDs are replaced with fallback_grid values when available;
    otherwise they are replaced with FLOOR.
    """
    out = np.asarray(grid, dtype=np.int32).copy()
    invalid_mask = ~np.isin(out, pipeline._valid_semantic_tile_ids_np)
    if bool(strip_void):
        invalid_mask |= out == int(TileID.VOID)
    invalid_count = int(np.sum(invalid_mask))
    invalid_ids: List[int] = []
    if invalid_count <= 0:
        return out, 0, invalid_ids

    invalid_ids = [int(v) for v in np.unique(out[invalid_mask])]
    floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
    if fallback_grid is not None and np.shape(fallback_grid) == np.shape(out):
        fb = np.asarray(fallback_grid, dtype=np.int32)
        fb_invalid = ~np.isin(fb, pipeline._valid_semantic_tile_ids_np)
        if bool(strip_void):
            fb_invalid |= fb == int(TileID.VOID)
        if bool(np.any(fb_invalid)):
            fb = fb.copy()
            fb[fb_invalid] = floor_id
        out[invalid_mask] = fb[invalid_mask]
    else:
        out[invalid_mask] = floor_id
    return out, invalid_count, invalid_ids


def _strip_room_void_tiles(
    pipeline,
    grid: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Remove VOID tiles from a generated room grid.

    Room-local semantic tensors should never retain stitched-world VOID.
    If VOID survives decoding or repair, exported rooms can show black
    holes inside otherwise valid geometry. Boundary VOID becomes WALL;
    interior VOID becomes FLOOR.
    """
    out = np.asarray(grid, dtype=np.int32).copy()
    void_mask = out == int(TileID.VOID)
    total_void = int(np.sum(void_mask))
    if total_void <= 0:
        return out, {
            "boundary_void_tiles_removed": 0,
            "interior_void_tiles_removed": 0,
        }

    boundary_mask = np.zeros_like(out, dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[ROOM_HEIGHT - 1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, ROOM_WIDTH - 1] = True

    boundary_void_mask = void_mask & boundary_mask
    interior_void_mask = void_mask & ~boundary_mask
    if bool(np.any(boundary_void_mask)):
        out[boundary_void_mask] = int(TileID.WALL)
    if bool(np.any(interior_void_mask)):
        out[interior_void_mask] = int(TileID.FLOOR)
    return out, {
        "boundary_void_tiles_removed": int(np.sum(boundary_void_mask)),
        "interior_void_tiles_removed": int(np.sum(interior_void_mask)),
    }


def _strip_volatile_room_semantics(
    pipeline,
    grid: np.ndarray,
    *,
    graph: Optional[nx.Graph] = None,
    room_id: Any = None,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    salvage_graph_markers: bool = True,
    salvage_max_manhattan_distance: int = 2,
) -> Tuple[np.ndarray, int, List[int], int, List[int]]:
    """
    Remove volatile gameplay semantics from a generated room.

    The neural room generator is responsible for layout/geometry. Graph-owned
    semantics such as keys, enemies, stairs, and puzzle markers are placed
    deterministically after room structure is finalized.
    """
    out = np.asarray(grid, dtype=np.int32).copy()
    preserved_count = 0
    preserved_ids: List[int] = []
    keep_mask = np.zeros_like(out, dtype=bool)
    if bool(salvage_graph_markers) and isinstance(graph, nx.Graph) and room_id in graph:
        structural_view = out.copy()
        structural_view[np.isin(structural_view, pipeline._volatile_room_semantic_tile_ids_np)] = int(
            SEMANTIC_PALETTE.get("FLOOR", 1)
        )
        planned = pipeline._plan_room_graph_marker_layout(
            structural_view,
            graph=graph,
            room_id=room_id,
            start_goal=start_goal,
        )
        max_distance = int(max(0, salvage_max_manhattan_distance))
        for tile_id, slot in planned:
            hits = np.argwhere(out == int(tile_id))
            if hits.size <= 0:
                continue
            distances = np.abs(hits[:, 0] - int(slot[0])) + np.abs(hits[:, 1] - int(slot[1]))
            best_idx = int(np.argmin(distances))
            best_distance = int(distances[best_idx])
            if best_distance > max_distance:
                continue
            best_row = int(hits[best_idx][0])
            best_col = int(hits[best_idx][1])
            if (best_row, best_col) != (int(slot[0]), int(slot[1])):
                out[best_row, best_col] = int(SEMANTIC_PALETTE.get("FLOOR", 1))
            out[int(slot[0]), int(slot[1])] = int(tile_id)
            keep_mask[int(slot[0]), int(slot[1])] = True
            preserved_count += 1
            preserved_ids.append(int(tile_id))

    volatile_mask = np.isin(out, pipeline._volatile_room_semantic_tile_ids_np) & ~keep_mask
    volatile_count = int(np.sum(volatile_mask))
    if volatile_count <= 0:
        return out, 0, [], preserved_count, preserved_ids

    volatile_ids = [int(v) for v in np.unique(out[volatile_mask])]
    out[volatile_mask] = int(SEMANTIC_PALETTE.get("FLOOR", 1))
    return out, volatile_count, volatile_ids, preserved_count, preserved_ids


def _apply_semantic_constrained_decoding(
    pipeline,
    logits: torch.Tensor,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> Dict[str, int]:
    """
    Bias decode logits toward graph-owned semantic markers before argmax.

    This keeps semantic alignment inside the learned decode path instead of
    relying exclusively on post-hoc deterministic overlay. The bias is
    intentionally softer than hard door forcing: graph markers are nudged
    toward planned slots while stray volatile semantic channels are
    suppressed elsewhere.
    """
    if not bool(pipeline.default_semantic_constrained_decoding_enabled):
        return {"planned_markers": 0, "biased_slots": 0}
    if not isinstance(graph, nx.Graph) or room_id not in graph:
        return {"planned_markers": 0, "biased_slots": 0}
    if not isinstance(logits, torch.Tensor) or logits.dim() != 4 or int(logits.shape[0]) != 1:
        return {"planned_markers": 0, "biased_slots": 0}

    try:
        preview_grid = logits.argmax(dim=1).detach().cpu().numpy()[0]
        preview_grid, _, _ = pipeline._sanitize_semantic_grid(preview_grid, strip_void=True)
        preview_grid[np.isin(preview_grid, pipeline._volatile_room_semantic_tile_ids_np)] = int(
            SEMANTIC_PALETTE.get("FLOOR", 1)
        )
        planned = pipeline._plan_room_graph_marker_layout(
            preview_grid,
            graph=graph,
            room_id=room_id,
            start_goal=start_goal,
        )
        if not planned:
            return {"planned_markers": 0, "biased_slots": 0}

        suppression_bias = float(pipeline.default_semantic_marker_suppression_bias)
        positive_bias = float(pipeline.default_semantic_marker_logit_bias)
        marker_channels = [
            int(tile_id)
            for tile_id in pipeline._volatile_room_semantic_tile_ids_np.tolist()
            if 0 <= int(tile_id) < int(logits.shape[1])
        ]
        if suppression_bias > 0.0 and marker_channels:
            logits[0, marker_channels, :, :] = logits[0, marker_channels, :, :] - suppression_bias

        biased_slots = 0
        if positive_bias > 0.0:
            for tile_id, slot in planned:
                tile_index = int(tile_id)
                if tile_index < 0 or tile_index >= int(logits.shape[1]):
                    continue
                row, col = pipeline._clamp_room_coord(slot)
                logits[0, tile_index, row, col] = logits[0, tile_index, row, col] + positive_bias
                biased_slots += 1

        return {"planned_markers": int(len(planned)), "biased_slots": int(biased_slots)}
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Semantic constrained decoding skipped for room %s: %s", room_id, exc)
        return {"planned_markers": 0, "biased_slots": 0}


def _all_room_door_slots_mask(pipeline) -> np.ndarray:
    """Return a boolean mask covering every canonical doorway slot in a room."""
    mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    for direction, spec in DOOR_POSITIONS.items():
        if direction in {"N", "S"}:
            row = int(spec["row"])
            c0 = int(spec["col_start"])
            c1 = int(spec["col_end"]) + 1
            mask[row, c0:c1] = True
        else:
            col = int(spec["col"])
            r0 = int(spec["row_start"])
            r1 = int(spec["row_end"]) + 1
            mask[r0:r1, col] = True
    return mask


def _required_room_door_slots_mask(
    pipeline,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
) -> np.ndarray:
    """
    Return a mask of boundary door slots that are legal for this room.

    When graph metadata is unavailable, fall back to all canonical doorway
    slots so standalone room generation does not over-strip doors.
    """
    if not isinstance(graph, nx.Graph) or room_id not in graph:
        return pipeline._all_room_door_slots_mask()

    semantics = pipeline._extract_room_topology_semantics(graph, room_id)
    mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    for direction, enabled in semantics["required_doors"].items():
        if not bool(enabled):
            continue
        spec = DOOR_POSITIONS[str(direction)]
        if direction in {"N", "S"}:
            row = int(spec["row"])
            c0 = int(spec["col_start"])
            c1 = int(spec["col_end"]) + 1
            mask[row, c0:c1] = True
        else:
            col = int(spec["col"])
            r0 = int(spec["row_start"])
            r1 = int(spec["row_end"]) + 1
            mask[r0:r1, col] = True
    return mask


def _enforce_room_boundary_shell(
    pipeline,
    grid: np.ndarray,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Seal room boundaries deterministically and reopen only valid doorway slots.

    Neural generation and symbolic repair can both leave boundary floor leaks,
    which later read as "rooms" expanding into stitched void. The exported
    contract should be simple: boundary tiles are walls unless the graph
    explicitly requires a doorway there.
    """
    out = np.asarray(grid, dtype=np.int32).copy()
    before = out.copy()
    wall_id = int(TileID.WALL)
    floor_id = int(TileID.FLOOR)
    door_tile_values = np.array(
        [
            int(TileID.DOOR_OPEN),
            int(TileID.DOOR_LOCKED),
            int(TileID.DOOR_BOMB),
            int(TileID.DOOR_PUZZLE),
            int(TileID.DOOR_BOSS),
            int(TileID.DOOR_SOFT),
        ],
        dtype=np.int32,
    )

    boundary_mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[ROOM_HEIGHT - 1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, ROOM_WIDTH - 1] = True

    # Default to a closed shell first, then reopen only legal canonical doors.
    out[boundary_mask] = wall_id

    required_doors: Dict[str, bool] = {}
    edge_constraints: Dict[str, Set[str]] = {}
    if isinstance(graph, nx.Graph) and room_id in graph:
        semantics = pipeline._extract_room_topology_semantics(graph, room_id)
        required_doors = {str(k): bool(v) for k, v in semantics["required_doors"].items()}
        edge_constraints = {
            str(k): {str(tok) for tok in set(v)}
            for k, v in semantics["edge_constraints"].items()
        }

    # Standalone generation has no graph semantics; preserve any canonical
    # door tiles that the model placed in legal doorway strips.
    if not required_doors:
        for direction, spec in DOOR_POSITIONS.items():
            if direction in {"N", "S"}:
                row = int(spec["row"])
                c0 = int(spec["col_start"])
                c1 = int(spec["col_end"]) + 1
                strip = before[row, c0:c1]
                if bool(np.any(np.isin(strip, door_tile_values))):
                    out[row, c0:c1] = strip
            else:
                col = int(spec["col"])
                r0 = int(spec["row_start"])
                r1 = int(spec["row_end"]) + 1
                strip = before[r0:r1, col]
                if bool(np.any(np.isin(strip, door_tile_values))):
                    out[r0:r1, col] = strip
    else:
        for direction, enabled in required_doors.items():
            if not bool(enabled):
                continue
            tile_id = int(pipeline._edge_tokens_to_door_tile(edge_constraints.get(str(direction), set())))
            spec = DOOR_POSITIONS[str(direction)]
            if direction in {"N", "S"}:
                row = int(spec["row"])
                c0 = int(spec["col_start"])
                c1 = int(spec["col_end"]) + 1
                out[row, c0:c1] = tile_id
                interior_row = 1 if direction == "N" else ROOM_HEIGHT - 2
                out[interior_row, c0:c1] = floor_id
            else:
                col = int(spec["col"])
                r0 = int(spec["row_start"])
                r1 = int(spec["row_end"]) + 1
                out[r0:r1, col] = tile_id
                interior_col = 1 if direction == "W" else ROOM_WIDTH - 2
                out[r0:r1, interior_col] = floor_id

    boundary_wall_tiles_forced = int(np.sum(boundary_mask & (out == wall_id) & (before != wall_id)))
    boundary_door_tiles_forced = int(
        np.sum(boundary_mask & np.isin(out, door_tile_values) & (before != out))
    )
    interior_door_apron_tiles_forced = int(
        np.sum((~boundary_mask) & (out == floor_id) & (before != floor_id))
    )

    return out, {
        "boundary_wall_tiles_forced": int(boundary_wall_tiles_forced),
        "boundary_door_tiles_forced": int(boundary_door_tiles_forced),
        "interior_door_apron_tiles_forced": int(interior_door_apron_tiles_forced),
    }


def _strip_structural_room_artifacts(
    pipeline,
    grid: np.ndarray,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
    max_interior_component_tiles: int = 4,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Remove impossible structural artifacts from generated rooms.

    The diffusion branch is currently prone to producing:
    - door tiles floating inside the room interior or on the wrong wall slot
    - tiny isolated wall/block islands that read as decode noise rather than
      intentional room structure

    These artifacts are deterministic to detect from room topology and can
    be cleaned without changing mission-graph semantics.
    """
    out = np.asarray(grid, dtype=np.int32).copy()
    floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
    door_tiles = np.array(
        [
            int(TileID.DOOR_OPEN),
            int(TileID.DOOR_LOCKED),
            int(TileID.DOOR_BOMB),
            int(TileID.DOOR_PUZZLE),
            int(TileID.DOOR_BOSS),
            int(TileID.DOOR_SOFT),
        ],
        dtype=np.int32,
    )
    allowed_door_mask = pipeline._required_room_door_slots_mask(graph=graph, room_id=room_id)
    invalid_door_mask = np.isin(out, door_tiles) & ~allowed_door_mask
    invalid_door_tiles_removed = int(np.sum(invalid_door_mask))
    if invalid_door_tiles_removed > 0:
        out[invalid_door_mask] = floor_id

    wall_like_mask = np.isin(out, np.array([int(TileID.WALL), int(TileID.BLOCK)], dtype=np.int32))
    visited = np.zeros_like(wall_like_mask, dtype=bool)
    interior_obstacle_tiles_removed = 0
    interior_obstacle_components_removed = 0

    for row in range(ROOM_HEIGHT):
        for col in range(ROOM_WIDTH):
            if not bool(wall_like_mask[row, col]) or bool(visited[row, col]):
                continue

            component: List[Tuple[int, int]] = []
            stack: List[Tuple[int, int]] = [(row, col)]
            visited[row, col] = True
            touches_allowed_door = False

            while stack:
                cur_r, cur_c = stack.pop()
                component.append((cur_r, cur_c))
                if bool(allowed_door_mask[cur_r, cur_c]):
                    touches_allowed_door = True
                for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_r = cur_r + d_r
                    next_c = cur_c + d_c
                    if not (0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH):
                        continue
                    if not bool(wall_like_mask[next_r, next_c]) or bool(visited[next_r, next_c]):
                        continue
                    visited[next_r, next_c] = True
                    stack.append((next_r, next_c))

            if touches_allowed_door or len(component) > int(max_interior_component_tiles):
                continue

            for comp_r, comp_c in component:
                out[comp_r, comp_c] = floor_id
            interior_obstacle_tiles_removed += int(len(component))
            interior_obstacle_components_removed += 1

    return out, {
        "invalid_door_tiles_removed": int(invalid_door_tiles_removed),
        "interior_obstacle_tiles_removed": int(interior_obstacle_tiles_removed),
        "interior_obstacle_components_removed": int(interior_obstacle_components_removed),
    }


def _strip_room_block_structure(
    pipeline,
    grid: np.ndarray,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Strip interior BLOCK tiles for strict no-puzzle ablations.

    Disabling the runtime puzzle scaffold alone does not guarantee a true
    no-puzzle export because learned decode noise can still leave interior
    brown BLOCK tiles behind. This helper enforces the stronger ablation
    contract by removing those tiles deterministically.
    """
    out = np.asarray(grid, dtype=np.int32).copy()
    block_id = int(TileID.BLOCK)
    floor_id = int(TileID.FLOOR)
    block_mask = out == block_id
    removed_tiles = int(np.sum(block_mask))
    if removed_tiles <= 0:
        return out, {
            "applied": 0,
            "block_tiles_removed": 0,
            "block_components_removed": 0,
        }

    visited = np.zeros_like(block_mask, dtype=bool)
    removed_components = 0
    for row in range(ROOM_HEIGHT):
        for col in range(ROOM_WIDTH):
            if not bool(block_mask[row, col]) or bool(visited[row, col]):
                continue
            removed_components += 1
            stack: List[Tuple[int, int]] = [(row, col)]
            visited[row, col] = True
            while stack:
                cur_r, cur_c = stack.pop()
                for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_r = cur_r + d_r
                    next_c = cur_c + d_c
                    if not (0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH):
                        continue
                    if not bool(block_mask[next_r, next_c]) or bool(visited[next_r, next_c]):
                        continue
                    visited[next_r, next_c] = True
                    stack.append((next_r, next_c))

    out[block_mask] = floor_id
    logger.debug(
        "Room %s stripped %d BLOCK tiles across %d components for strict no-puzzle cleanup.",
        room_id,
        removed_tiles,
        removed_components,
    )
    return out, {
        "applied": 1,
        "block_tiles_removed": int(removed_tiles),
        "block_components_removed": int(removed_components),
    }


def _dilate_room_mask(pipeline, mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Lightweight 4-neighbour dilation for room-scale boolean masks."""
    out = np.asarray(mask, dtype=bool).copy()
    steps = int(max(0, int(radius)))
    for _ in range(steps):
        prev = out.copy()
        out[1:, :] |= prev[:-1, :]
        out[:-1, :] |= prev[1:, :]
        out[:, 1:] |= prev[:, :-1]
        out[:, :-1] |= prev[:, 1:]
    return out


def _paint_room_line_mask(
    pipeline,
    canvas: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    value: bool = True,
) -> None:
    """Paint a Manhattan polyline between two room-local points."""
    r0, c0 = int(start[0]), int(start[1])
    r1, c1 = int(end[0]), int(end[1])
    r, c = r0, c0
    while r != r1:
        canvas[r, c] = bool(value)
        r += 1 if r1 > r else -1
    while c != c1:
        canvas[r, c] = bool(value)
        c += 1 if c1 > c else -1
    canvas[r1, c1] = bool(value)



# =========================================================================
# Re-exported from src.generation.puzzle_generator for full backwards compatibility
# =========================================================================
from src.generation.puzzle_generator import (
    PuzzleGenerator,
    _apply_puzzle_room_scaffold,
    _build_puzzle_room_route_template,
    _build_puzzle_room_segments,
    _build_puzzle_room_variant_specs,
    _build_room_walkable_mask,
    _classify_puzzle_gate_family,
    _commit_puzzle_novelty_choice,
    _count_room_path_turns,
    _evaluate_puzzle_candidate_contract,
    _evaluate_puzzle_candidate_interaction_geometry,
    _evaluate_puzzle_candidate_interaction_sequence,
    _evaluate_puzzle_candidate_route_quality,
    _nearest_walkable_room_coord,
    _puzzle_descriptor_distance,
    _resolve_puzzle_interaction_sequence,
    _score_puzzle_candidate,
    _select_puzzle_room_scaffold_archetype,
    _shortest_room_path,
    _strip_small_interior_structure_components,
    _summarize_puzzle_candidate_descriptor,
    _switch_push_witnesses,
)


def _count_small_interior_structure_components(
    pipeline,
    grid: np.ndarray,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
    max_component_tiles: int = 6,
) -> int:
    """
    Count small interior wall/block islands that survive structural cleanup.

    These components are a good proxy for fast-sampler drift: they usually
    show up as isolated 1x1 / 2x2 block clusters inside otherwise plain
    rooms. We use the count as a quality gate for teacher fallback without
    touching topology semantics.
    """
    out = np.asarray(grid, dtype=np.int32)
    wall_like_mask = np.isin(out, np.array([int(TileID.WALL), int(TileID.BLOCK)], dtype=np.int32))
    allowed_door_mask = pipeline._required_room_door_slots_mask(graph=graph, room_id=room_id)
    visited = np.zeros_like(wall_like_mask, dtype=bool)
    suspicious_components = 0

    for row in range(ROOM_HEIGHT):
        for col in range(ROOM_WIDTH):
            if not bool(wall_like_mask[row, col]) or bool(visited[row, col]):
                continue

            component: List[Tuple[int, int]] = []
            stack: List[Tuple[int, int]] = [(row, col)]
            visited[row, col] = True
            touches_boundary = False
            touches_allowed_door = False

            while stack:
                cur_r, cur_c = stack.pop()
                component.append((cur_r, cur_c))
                if cur_r in {0, ROOM_HEIGHT - 1} or cur_c in {0, ROOM_WIDTH - 1}:
                    touches_boundary = True
                if bool(allowed_door_mask[cur_r, cur_c]):
                    touches_allowed_door = True
                for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_r = cur_r + d_r
                    next_c = cur_c + d_c
                    if not (0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH):
                        continue
                    if not bool(wall_like_mask[next_r, next_c]) or bool(visited[next_r, next_c]):
                        continue
                    visited[next_r, next_c] = True
                    stack.append((next_r, next_c))

            if touches_boundary or touches_allowed_door:
                continue
            if len(component) <= int(max_component_tiles):
                suspicious_components += 1

    return int(suspicious_components)


def _should_retry_room_with_teacher(
    pipeline,
    *,
    final_grid: np.ndarray,
    graph: Optional[nx.Graph],
    room_id: Any,
    metrics: Dict[str, float],
    source_mode: str = "fast_sampler",
) -> bool:
    """
    Decide whether a non-teacher room should be regenerated with the full teacher.

    The goal is pragmatic: keep cheaper / weaker generators for normal rooms,
    but avoid exporting rooms that still contain obvious structural decode
    noise after cleanup/repair.
    """
    suspicious_components = pipeline._count_small_interior_structure_components(
        final_grid,
        graph=graph,
        room_id=room_id,
    )
    if suspicious_components >= 1:
        return True

    block_id = int(TileID.BLOCK)
    interior_block_tiles = int(np.sum(np.asarray(final_grid, dtype=np.int32) == block_id))
    graph_markers = set(int(v) for v in pipeline._resolve_room_graph_markers(graph, room_id))
    if interior_block_tiles > 0 and not graph_markers.intersection({int(TileID.PUZZLE), int(TileID.STAIR)}):
        return True

    door_like_tiles = np.isin(
        np.asarray(final_grid, dtype=np.int32),
        np.array(
            [
                int(TileID.DOOR_OPEN),
                int(TileID.DOOR_LOCKED),
                int(TileID.DOOR_BOMB),
                int(TileID.DOOR_PUZZLE),
                int(TileID.DOOR_BOSS),
                int(TileID.DOOR_SOFT),
            ],
            dtype=np.int32,
        ),
    )
    allowed_door_mask = pipeline._required_room_door_slots_mask(graph=graph, room_id=room_id)
    unexpected_door_tiles = int(np.sum(door_like_tiles & ~allowed_door_mask))
    if unexpected_door_tiles >= 1:
        return True

    repair_tiles = int(metrics.get("tiles_changed", 0))
    repair_obstacles = int(metrics.get("repair_interior_obstacle_tiles_removed", 0))
    neural_obstacles = int(metrics.get("neural_interior_obstacle_tiles_removed", 0))
    if repair_tiles >= 18 and (repair_obstacles > 0 or neural_obstacles > 0):
        return True

    source_mode = str(source_mode or "fast_sampler").strip().lower()
    if source_mode == "masked_room":
        if float(metrics.get("final_graph_marker_overwrite_rate", 0.0)) > 0.34:
            return True
        if repair_tiles >= 12:
            return True

    return False


def _resolve_effective_sampling_guidance(
    pipeline,
    *,
    use_fast_sampling: bool,
    guidance_scale: float,
    logic_guidance_scale: float,
) -> Tuple[float, float]:
    """
    Clamp runtime guidance to the regime the distilled fast sampler was trained for.

    The fast-sampler student is distilled from the base diffusion model using
    the diffusion checkpoint's CFG setting and without LogicNet gradient
    guidance. Reusing more aggressive diffusion-time overrides than the
    teacher was trained with pushes the short-step student out of
    distribution and hurts room quality.
    """
    effective_guidance_scale = float(guidance_scale)
    effective_logic_guidance_scale = float(logic_guidance_scale)
    if not bool(use_fast_sampling) or not pipeline.diffusion.supports_fast_sampling():
        if (
            effective_logic_guidance_scale > 0.0
            and pipeline.logic_net is not None
            and not bool(getattr(pipeline, "logic_net_checkpoint_loaded", False))
        ):
            pipeline._bump_diagnostic("logic_guidance_disabled_untrained_logic_net")
            logger.warning(
                "LogicNet guidance requested with scale %.3f, but LogicNet is randomly initialized; "
                "disabling runtime logic guidance. Provide a checkpoint-backed LogicNet to enable it.",
                effective_logic_guidance_scale,
            )
            effective_logic_guidance_scale = 0.0
        if (
            effective_logic_guidance_scale > 0.0
            and pipeline.logic_net is not None
            and not bool(getattr(pipeline.logic_net, "_hmolqd_guidance_calibrated", False))
        ):
            pipeline._bump_diagnostic("logic_guidance_disabled_uncalibrated_surrogate")
            accuracy = getattr(pipeline.logic_net, "_hmolqd_logic_tile_accuracy", None)
            threshold = float(
                getattr(pipeline.logic_net, "_hmolqd_min_logic_tile_accuracy", 0.4)
            )
            logger.warning(
                "LogicNet guidance requested at scale %.3f, but latent-to-tile surrogate "
                "accuracy is %s (required %.3f); disabling runtime guidance.",
                effective_logic_guidance_scale,
                "unreported" if accuracy is None else f"{float(accuracy):.3f}",
                threshold,
            )
            effective_logic_guidance_scale = 0.0
        return effective_guidance_scale, effective_logic_guidance_scale

    trained_cfg_scale = float(
        max(
            0.0,
            getattr(
                pipeline.diffusion,
                "training_cfg_scale",
                pipeline.diffusion_fallback_config.get(
                    "cfg_scale",
                    getattr(pipeline.diffusion, "cfg_scale", effective_guidance_scale),
                ),
            ),
        )
    )
    if effective_guidance_scale > trained_cfg_scale + 1e-6:
        pipeline._bump_diagnostic("fast_sampling_cfg_clamped")
        logger.debug(
            "Fast sampling clamped CFG from %.3f to %.3f to match the distilled teacher regime.",
            effective_guidance_scale,
            trained_cfg_scale,
        )
        effective_guidance_scale = trained_cfg_scale

    if effective_logic_guidance_scale > 0.0:
        pipeline._bump_diagnostic("fast_sampling_logic_guidance_disabled")
        logger.debug(
            "Fast sampling disabled LogicNet runtime guidance (requested %.3f) because the student was "
            "not distilled with gradient guidance enabled.",
            effective_logic_guidance_scale,
        )
        effective_logic_guidance_scale = 0.0

    return effective_guidance_scale, effective_logic_guidance_scale


def _resolve_room_graph_markers(
    pipeline,
    graph: Optional[nx.Graph],
    room_id: Any,
) -> List[int]:
    """Infer deterministic per-room semantic markers from mission-graph metadata."""
    if not isinstance(graph, nx.Graph) or room_id not in graph:
        return []

    attrs = dict(graph.nodes[room_id])
    role_flags = pipeline._room_role_flags(attrs)
    label_tokens = pipeline._parse_label_tokens(attrs.get("label"))
    node_type = str(
        attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or ""
    ).strip().lower()

    markers: List[int] = []

    if role_flags.get("is_start", False) or node_type == "start":
        markers.append(int(TileID.START))
    if role_flags.get("has_goal", False) or node_type in {"goal", "triforce"}:
        markers.append(int(TileID.TRIFORCE))

    if node_type in {"boss", "mini_boss"} or role_flags.get("has_boss", False):
        markers.append(int(TileID.BOSS))
    elif (
        role_flags.get("has_enemy", False)
        or node_type in {"enemy", "arena", "combat_puzzle"}
        or "e" in label_tokens
    ):
        markers.append(int(TileID.ENEMY))

    if node_type in {"big_key", "boss_key"}:
        markers.append(int(TileID.KEY_BOSS))
    elif role_flags.get("has_key", False) or node_type == "key" or "k" in label_tokens:
        markers.append(int(TileID.KEY_SMALL))

    if node_type in {"item", "protection_item"} or role_flags.get("has_item", False):
        markers.append(int(TileID.KEY_ITEM))
    elif node_type in {"minor_item", "treasure"}:
        markers.append(int(TileID.ITEM_MINOR))

    if node_type in {"stairs_up", "stairs_down", "stair", "warp"}:
        markers.append(int(TileID.STAIR))

    if (
        role_flags.get("has_puzzle", False)
        or node_type in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"}
        or "p" in label_tokens
    ):
        markers.append(int(TileID.PUZZLE))

    return markers


def _build_room_graph_marker_preferences(
    pipeline,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> Dict[int, Tuple[int, int]]:
    """
    Build gate-aware preferred semantic slots for deterministic room markers.

    Generic room-role anchors are often too weak for puzzle readability:
    the scaffold may create a clear key pocket or item alcove while the
    overlay still drops the marker near the room center. This helper keeps
    marker placement aligned with the same gate-family semantics used by the
    constructive scaffold.
    """
    if start_goal is None:
        start_goal = pipeline._extract_room_start_goal(graph, room_id) if isinstance(graph, nx.Graph) else None
    start_coord, goal_coord = start_goal if start_goal is not None else (
        (ROOM_HEIGHT // 2, 0),
        (ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
    )
    attrs = dict(graph.nodes[room_id]) if isinstance(graph, nx.Graph) and room_id in graph else {}
    role_flags = pipeline._room_role_flags(attrs)
    semantics = pipeline._extract_room_topology_semantics(graph, room_id) if isinstance(graph, nx.Graph) and room_id in graph else {
        "required_doors": {},
        "incoming_dirs": set(),
        "outgoing_dirs": set(),
    }
    semantic_anchors = build_room_semantic_anchor_points(
        room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
        start=start_coord,
        goal=goal_coord,
        required_doors=semantics["required_doors"],
        incoming_dirs=semantics["incoming_dirs"],
        outgoing_dirs=semantics["outgoing_dirs"],
        room_role_flags=role_flags,
        semantic_puzzle_offset=pipeline.default_semantic_puzzle_offset,
    )
    node_type = str(
        attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or ""
    ).strip().lower()
    gate_family = pipeline._classify_puzzle_gate_family(
        role_flags=role_flags,
        semantics=semantics,
        node_type=node_type,
    )
    if gate_family in {"switch", "toggle", "bombable"}:
        stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
    elif gate_family == "item_unlock":
        stateful_anchor_name = "item" if "item" in semantic_anchors else ("puzzle" if "puzzle" in semantic_anchors else None)
    elif gate_family == "key":
        stateful_anchor_name = "key" if "key" in semantic_anchors else None
    elif gate_family == "combat":
        stateful_anchor_name = "enemy" if "enemy" in semantic_anchors else None
    else:
        stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
    stateful_anchor = pipeline._clamp_room_coord(semantic_anchors.get(stateful_anchor_name)) if stateful_anchor_name and semantic_anchors.get(stateful_anchor_name) is not None else None
    puzzle_anchor = pipeline._clamp_room_coord(semantic_anchors.get("puzzle", (max(1, ROOM_HEIGHT // 2 - 2), ROOM_WIDTH // 2)))
    enemy_anchor = pipeline._clamp_room_coord(semantic_anchors.get("enemy", (ROOM_HEIGHT // 2 - 2, ROOM_WIDTH // 2)))
    item_anchor = pipeline._clamp_room_coord(semantic_anchors.get("item", (min(ROOM_HEIGHT - 2, ROOM_HEIGHT // 2 + 2), ROOM_WIDTH // 2)))
    key_anchor = pipeline._clamp_room_coord(semantic_anchors.get("key", (ROOM_HEIGHT // 2, max(1, ROOM_WIDTH // 2 - 2))))
    boss_anchor = pipeline._clamp_room_coord(semantic_anchors.get("boss", (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)))

    preferred_positions: Dict[int, Tuple[int, int]] = {
        int(TileID.START): pipeline._clamp_room_coord(semantic_anchors.get("start", start_coord)),
        int(TileID.TRIFORCE): pipeline._clamp_room_coord(semantic_anchors.get("goal", goal_coord)),
        int(TileID.BOSS): boss_anchor,
        int(TileID.ENEMY): enemy_anchor,
        int(TileID.KEY_SMALL): key_anchor,
        int(TileID.KEY_BOSS): key_anchor,
        int(TileID.KEY_ITEM): item_anchor,
        int(TileID.ITEM_MINOR): item_anchor,
        int(TileID.STAIR): item_anchor,
        int(TileID.PUZZLE): puzzle_anchor,
    }

    if stateful_anchor is not None:
        if gate_family == "key":
            preferred_positions[int(TileID.KEY_SMALL)] = stateful_anchor
            preferred_positions[int(TileID.KEY_BOSS)] = stateful_anchor
            preferred_positions[int(TileID.PUZZLE)] = puzzle_anchor
        elif gate_family == "item_unlock":
            preferred_positions[int(TileID.KEY_ITEM)] = stateful_anchor
            preferred_positions[int(TileID.ITEM_MINOR)] = stateful_anchor
            preferred_positions[int(TileID.STAIR)] = stateful_anchor
            preferred_positions[int(TileID.PUZZLE)] = puzzle_anchor
        elif gate_family in {"switch", "toggle", "bombable"}:
            preferred_positions[int(TileID.PUZZLE)] = stateful_anchor
        elif gate_family == "combat":
            preferred_positions[int(TileID.ENEMY)] = stateful_anchor
            if role_flags.get("has_puzzle", False):
                preferred_positions[int(TileID.PUZZLE)] = puzzle_anchor

    return preferred_positions


def _find_room_graph_marker_slot(
    pipeline,
    grid: np.ndarray,
    *,
    preferred: Tuple[int, int],
    occupied: Set[Tuple[int, int]],
    tile_id: Optional[int] = None,
) -> Tuple[int, int]:
    """Find a stable in-room placement slot near a preferred coordinate."""
    floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
    preferred = pipeline._clamp_room_coord(preferred)
    wanted_tile = None if tile_id is None else int(tile_id)

    if (
        wanted_tile is not None
        and preferred not in occupied
        and int(grid[preferred[0], preferred[1]]) == wanted_tile
    ):
        return preferred

    if wanted_tile is not None:
        for radius in range(0, max(ROOM_HEIGHT, ROOM_WIDTH)):
            row_min = max(1, preferred[0] - radius)
            row_max = min(ROOM_HEIGHT - 2, preferred[0] + radius)
            col_min = max(1, preferred[1] - radius)
            col_max = min(ROOM_WIDTH - 2, preferred[1] + radius)
            for r in range(row_min, row_max + 1):
                for c in range(col_min, col_max + 1):
                    if max(abs(r - preferred[0]), abs(c - preferred[1])) != radius:
                        continue
                    if (r, c) in occupied:
                        continue
                    if int(grid[r, c]) == wanted_tile:
                        return (r, c)

    def _search(valid_tiles: Set[int]) -> Optional[Tuple[int, int]]:
        for radius in range(0, max(ROOM_HEIGHT, ROOM_WIDTH)):
            row_min = max(1, preferred[0] - radius)
            row_max = min(ROOM_HEIGHT - 2, preferred[0] + radius)
            col_min = max(1, preferred[1] - radius)
            col_max = min(ROOM_WIDTH - 2, preferred[1] + radius)
            for r in range(row_min, row_max + 1):
                for c in range(col_min, col_max + 1):
                    if max(abs(r - preferred[0]), abs(c - preferred[1])) != radius:
                        continue
                    if (r, c) in occupied:
                        continue
                    if int(grid[r, c]) in valid_tiles:
                        return (r, c)
        return None

    slot = _search({floor_id})
    if slot is not None:
        return slot

    # Fallback: any non-boundary position that is not a hard wall.
    non_blocking_tiles = {
        int(TileID.FLOOR),
        int(TileID.VOID),
        int(TileID.BLOCK),
        int(TileID.DOOR_OPEN),
        int(TileID.DOOR_LOCKED),
        int(TileID.DOOR_BOMB),
        int(TileID.DOOR_PUZZLE),
        int(TileID.DOOR_BOSS),
        int(TileID.DOOR_SOFT),
    }
    slot = _search(non_blocking_tiles)
    if slot is not None:
        return slot

    if preferred not in occupied:
        return preferred

    for r in range(1, ROOM_HEIGHT - 1):
        for c in range(1, ROOM_WIDTH - 1):
            if (r, c) not in occupied:
                return (r, c)

    return preferred


def _overlay_room_graph_markers(
    pipeline,
    grid: np.ndarray,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, int, List[int]]:
    """
    Reintroduce graph-owned semantics after structural room generation.

    This keeps the generator focused on layout while still producing rooms
    with the mission-critical markers the graph requires.
    """
    out = np.asarray(grid, dtype=np.int32).copy()
    markers = pipeline._resolve_room_graph_markers(graph, room_id)
    if not markers:
        return out, 0, []

    if start_goal is None:
        start_goal = pipeline._extract_room_start_goal(graph, room_id) if isinstance(graph, nx.Graph) else None
    start_coord, goal_coord = start_goal if start_goal is not None else (
        (ROOM_HEIGHT // 2, 0),
        (ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
    )
    preferred_positions = pipeline._build_room_graph_marker_preferences(
        graph=graph,
        room_id=room_id,
        start_goal=(start_coord, goal_coord),
    )

    occupied: Set[Tuple[int, int]] = set()
    placed: List[int] = []

    for tile_id in markers:
        preferred = preferred_positions.get(int(tile_id), (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
        slot = pipeline._find_room_graph_marker_slot(
            out,
            preferred=preferred,
            occupied=occupied,
            tile_id=int(tile_id),
        )
        occupied.add(slot)
        out[slot[0], slot[1]] = int(tile_id)
        placed.append(int(tile_id))

    return out, len(placed), placed


def _plan_room_graph_marker_layout(
    pipeline,
    grid: np.ndarray,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Compute the deterministic graph-marker placement plan for a room without
    mutating the room grid.

    The returned slots are the same positions `_overlay_room_graph_markers`
    would target on the same pre-overlay grid. This lets audits measure how
    much semantic placement is learned vs. forced by the symbolic overlay.
    """
    base_grid = np.asarray(grid, dtype=np.int32).copy()
    markers = pipeline._resolve_room_graph_markers(graph, room_id)
    if not markers:
        return []

    if start_goal is None:
        start_goal = pipeline._extract_room_start_goal(graph, room_id) if isinstance(graph, nx.Graph) else None
    start_coord, goal_coord = start_goal if start_goal is not None else (
        (ROOM_HEIGHT // 2, 0),
        (ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
    )
    preferred_positions = pipeline._build_room_graph_marker_preferences(
        graph=graph,
        room_id=room_id,
        start_goal=(start_coord, goal_coord),
    )

    occupied: Set[Tuple[int, int]] = set()
    placements: List[Tuple[int, Tuple[int, int]]] = []
    for tile_id in markers:
        preferred = preferred_positions.get(int(tile_id), (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
        slot = pipeline._find_room_graph_marker_slot(
            base_grid,
            preferred=preferred,
            occupied=occupied,
            tile_id=int(tile_id),
        )
        occupied.add(slot)
        placements.append((int(tile_id), slot))
    return placements



# =========================================================================
# Re-exported from src.generation.puzzle_generator for full backwards compatibility
# =========================================================================
from src.generation.puzzle_generator import (
    _build_room_puzzle_metadata,
    _globalize_room_puzzle_metadata,
)


def _measure_room_graph_marker_alignment(
    pipeline,
    grid: np.ndarray,
    *,
    placements: List[Tuple[int, Tuple[int, int]]],
    prefix: str,
) -> Dict[str, float]:
    """Measure how well a room already matches the planned graph markers."""
    grid_np = np.asarray(grid, dtype=np.int32)
    expected = int(len(placements))
    if expected <= 0:
        return {
            f"{prefix}graph_marker_expected": 0.0,
            f"{prefix}graph_marker_exact_matches": 0.0,
            f"{prefix}graph_marker_tile_present": 0.0,
            f"{prefix}graph_marker_exact_match_rate": 1.0,
            f"{prefix}graph_marker_presence_rate": 1.0,
            f"{prefix}semantic_anchor_avg_manhattan_error": 0.0,
            f"{prefix}semantic_anchor_max_manhattan_error": 0.0,
        }

    exact_matches = 0
    tile_present = 0
    distances: List[float] = []
    for tile_id, slot in placements:
        sr, sc = int(slot[0]), int(slot[1])
        if int(grid_np[sr, sc]) == int(tile_id):
            exact_matches += 1
            tile_present += 1
            distances.append(0.0)
            continue

        positions = np.argwhere(grid_np == int(tile_id))
        if positions.size == 0:
            distances.append(float(ROOM_HEIGHT + ROOM_WIDTH))
            continue

        tile_present += 1
        min_dist = min(
            abs(int(pos[0]) - sr) + abs(int(pos[1]) - sc)
            for pos in positions
        )
        distances.append(float(min_dist))

    exact_rate = float(exact_matches) / float(expected)
    presence_rate = float(tile_present) / float(expected)
    avg_error = float(sum(distances) / len(distances)) if distances else 0.0
    max_error = float(max(distances)) if distances else 0.0
    return {
        f"{prefix}graph_marker_expected": float(expected),
        f"{prefix}graph_marker_exact_matches": float(exact_matches),
        f"{prefix}graph_marker_tile_present": float(tile_present),
        f"{prefix}graph_marker_exact_match_rate": exact_rate,
        f"{prefix}graph_marker_presence_rate": presence_rate,
        f"{prefix}semantic_anchor_avg_manhattan_error": avg_error,
        f"{prefix}semantic_anchor_max_manhattan_error": max_error,
    }


def _aggregate_room_alignment_metrics(pipeline, room_metric_dicts: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-room semantic alignment metrics into dungeon-level summaries."""
    if not room_metric_dicts:
        return {
            "total_graph_marker_expected": 0.0,
            "total_graph_marker_overwrites": 0.0,
            "avg_neural_graph_marker_exact_match_rate": 1.0,
            "avg_final_pre_overlay_graph_marker_exact_match_rate": 1.0,
            "avg_final_post_overlay_graph_marker_exact_match_rate": 1.0,
            "avg_final_graph_marker_overwrite_rate": 0.0,
            "avg_neural_semantic_anchor_error": 0.0,
            "avg_final_pre_overlay_semantic_anchor_error": 0.0,
            "avg_final_post_overlay_semantic_anchor_error": 0.0,
        }

    def _mean(metric_key: str, default: float) -> float:
        return float(np.mean([float(m.get(metric_key, default)) for m in room_metric_dicts]))

    return {
        "total_graph_marker_expected": float(
            sum(float(m.get("final_pre_overlay_graph_marker_expected", 0.0)) for m in room_metric_dicts)
        ),
        "total_graph_marker_overwrites": float(
            sum(float(m.get("final_graph_marker_overwrites", 0.0)) for m in room_metric_dicts)
        ),
        "avg_neural_graph_marker_exact_match_rate": _mean(
            "neural_graph_marker_exact_match_rate",
            1.0,
        ),
        "avg_final_pre_overlay_graph_marker_exact_match_rate": _mean(
            "final_pre_overlay_graph_marker_exact_match_rate",
            1.0,
        ),
        "avg_final_post_overlay_graph_marker_exact_match_rate": _mean(
            "final_post_overlay_graph_marker_exact_match_rate",
            1.0,
        ),
        "avg_final_graph_marker_overwrite_rate": _mean(
            "final_graph_marker_overwrite_rate",
            0.0,
        ),
        "avg_neural_semantic_anchor_error": _mean(
            "neural_semantic_anchor_avg_manhattan_error",
            0.0,
        ),
        "avg_final_pre_overlay_semantic_anchor_error": _mean(
            "final_pre_overlay_semantic_anchor_avg_manhattan_error",
            0.0,
        ),
        "avg_final_post_overlay_semantic_anchor_error": _mean(
            "final_post_overlay_semantic_anchor_avg_manhattan_error",
            0.0,
        ),
    }



# =========================================================================
# Re-exported from src.generation.puzzle_generator for full backwards compatibility
# =========================================================================
from src.generation.puzzle_generator import (
    _aggregate_puzzle_stage_semantics_metrics,
)


def _build_latent_edit_mask(
    pipeline,
    room_mask: np.ndarray,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """Compatibility wrapper around extracted feedback helper."""
    return build_latent_edit_mask(
        room_mask,
        latent_h=latent_h,
        latent_w=latent_w,
        device=pipeline.device,
    )


def _logicnet_guided_inpaint_room(
    pipeline,
    current_grid: np.ndarray,
    dead_end_mask: np.ndarray,
    condition: torch.Tensor,
    graph_data: Optional[Dict[str, Any]],
    num_diffusion_steps: int,
    seed: Optional[int] = None,
    noise_strength: float = 0.5,
    guidance_scale_multiplier: float = 1.0,
) -> np.ndarray:
    """Compatibility wrapper around extracted feedback helper."""
    pipeline._require_room_generation_components("_logicnet_guided_inpaint_room")
    return logicnet_guided_inpaint_room(
        current_grid=current_grid,
        dead_end_mask=dead_end_mask,
        condition=condition,
        graph_data=graph_data,
        num_diffusion_steps=num_diffusion_steps,
        seed=seed,
        device=pipeline.device,
        vqvae=pipeline.vqvae,
        diffusion=pipeline.diffusion,
        num_classes=int(getattr(pipeline.vqvae, "num_classes", int(np.max(pipeline._valid_semantic_tile_ids_np)) + 1)),
        noise_strength=noise_strength,
        guidance_scale_multiplier=guidance_scale_multiplier,
    )


def _wfc_guided_inpaint_room(
    pipeline,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """Backward-compatible alias for _logicnet_guided_inpaint_room."""
    return _logicnet_guided_inpaint_room(pipeline, *args, **kwargs)


def _compute_room_condition(
    pipeline,
    *,
    neighbor_latents: Dict[str, Optional[torch.Tensor]],
    reference_room_maps: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    graph_context: Dict[str, Any],
    boundary_constraints: Optional[torch.Tensor],
    position: Optional[torch.Tensor],
    room_generator_mode: Optional[str] = None,
) -> torch.Tensor:
    """Build Block-III conditioning tensor for a room."""
    if boundary_constraints is None:
        boundary_constraints = torch.zeros(1, 8, device=pipeline.device)
    if position is None:
        position = torch.zeros(1, 2, device=pipeline.device)

    node_tokens: Optional[torch.Tensor] = None
    condition_dim = int(getattr(pipeline.condition_encoder, "output_dim", 256))
    style_id = graph_context.get("style_id")
    active_generator_mode = str(
        room_generator_mode or getattr(pipeline, "room_generator_mode", "latent_diffusion")
    ).strip().lower()
    global_graph_conditioning_enabled = bool(
        active_generator_mode == "discrete_masked"
        or getattr(pipeline, "diffusion_condition_global_graph_enabled", True)
    )
    use_graph_node_cross_attention = bool(
        global_graph_conditioning_enabled
        and getattr(pipeline, "use_graph_node_cross_attention", False)
    )
    try:
        if global_graph_conditioning_enabled:
            node_dim, edge_dim = pipeline._condition_feature_dims()
            validate_feature_dims(
                node_features=graph_context.get('node_features'),
                edge_features=graph_context.get('edge_features'),
                expected_node_dim=node_dim,
                expected_edge_dim=edge_dim,
            )
            condition_out = pipeline.condition_encoder(
                neighbor_latents=neighbor_latents,
                reference_room_maps=reference_room_maps,
                boundary_constraints=boundary_constraints,
                position=position,
                node_features=graph_context.get('node_features'),
                edge_index=graph_context.get('edge_index'),
                edge_features=graph_context.get('edge_features'),
                edge_rrwp=graph_context.get('edge_rrwp'),
                tpe=graph_context.get('tpe'),
                current_node_distance=graph_context.get('current_node_distance'),
                node_mask=graph_context.get('node_mask'),
                current_node_idx=graph_context.get('current_node_idx'),
                style_id=style_id,
                return_global_tokens=use_graph_node_cross_attention,
            )
        else:
            condition_out = pipeline.condition_encoder.encode_local_only(
                neighbor_latents=neighbor_latents,
                boundary_constraints=boundary_constraints,
                position=position,
            )
        if use_graph_node_cross_attention:
            if not isinstance(condition_out, tuple) or len(condition_out) != 2:
                raise ValueError(
                    "Condition encoder must return (conditioning, global_tokens) when "
                    "graph-node cross-attention is enabled."
                )
            condition, node_tokens = condition_out
        else:
            condition = condition_out
        validate_tensor_contract(
            condition,
            BlockShapeContract(name='block_iii_condition_output', dims=2, batch_dim=1, channel_dim=condition_dim),
        )
    except (AttributeError, RuntimeError, ValueError, TypeError) as e:
        pipeline._bump_diagnostic("condition_encoder_fallback")
        if bool(getattr(pipeline, "condition_strict_schema", True)):
            raise RuntimeError(
                f"Condition encoding failed under the strict schema: {e}"
            ) from e
        logger.warning(f"Condition encoding failed: {e}, using zero condition")
        if use_graph_node_cross_attention:
            num_nodes = 0
            node_features = graph_context.get("node_features")
            if isinstance(node_features, torch.Tensor) and node_features.dim() >= 2:
                num_nodes = int(node_features.shape[0])
            required_seq_len = max(1, num_nodes + 1)
            condition = torch.zeros(1, required_seq_len, condition_dim, device=pipeline.device)
        else:
            condition = torch.zeros(1, condition_dim, device=pipeline.device)
        node_tokens = None

    if use_graph_node_cross_attention and isinstance(node_tokens, torch.Tensor):
        try:
            if node_tokens.dim() == 2:
                node_tokens = node_tokens.unsqueeze(0)
            condition = torch.cat([condition.unsqueeze(1), node_tokens], dim=1)
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            pipeline._bump_diagnostic("graph_node_cross_attention_fallback")
            if bool(getattr(pipeline, "condition_strict_schema", True)):
                raise RuntimeError(
                    f"Graph-node conditioning assembly failed under the strict schema: {e}"
                ) from e
            logger.debug("Falling back to single conditioning vector: %s", e)

    puzzle_structure_condition_enabled = (
        pipeline.masked_room_puzzle_structure_condition_enabled
        if active_generator_mode == "discrete_masked"
        else pipeline.diffusion_puzzle_structure_condition_enabled
    )
    if puzzle_structure_condition_enabled and isinstance(condition, torch.Tensor):
        if condition.dim() == 3 and int(condition.shape[0]) == 1:
            condition = apply_puzzle_structure_control_to_conditioning(
                condition.squeeze(0),
                puzzle_structure_enabled=bool(graph_context.get("puzzle_room_structure_enabled", True)),
                graph_conditioning_mode="node_sequence",
            ).unsqueeze(0)
        elif condition.dim() == 2:
            if use_graph_node_cross_attention:
                condition = apply_puzzle_structure_control_to_conditioning(
                    condition,
                    puzzle_structure_enabled=bool(graph_context.get("puzzle_room_structure_enabled", True)),
                    graph_conditioning_mode="node_sequence",
                ).unsqueeze(0)
            else:
                condition = apply_puzzle_structure_control_to_conditioning(
                    condition,
                    puzzle_structure_enabled=bool(graph_context.get("puzzle_room_structure_enabled", True)),
                    graph_conditioning_mode="pooled",
                )

    puzzle_stage_condition_enabled = (
        bool(getattr(pipeline, "masked_room_puzzle_stage_conditioning_enabled", False))
        if active_generator_mode == "discrete_masked"
        else bool(getattr(pipeline, "diffusion_puzzle_stage_conditioning_enabled", False))
    )
    puzzle_stage_token_scale = (
        float(getattr(pipeline, "masked_room_puzzle_stage_token_scale", 0.20))
        if active_generator_mode == "discrete_masked"
        else float(getattr(pipeline, "diffusion_puzzle_stage_token_scale", 0.20))
    )
    if puzzle_stage_condition_enabled and isinstance(condition, torch.Tensor):
        # Training adds these tokens after structural controls. Preserve that
        # exact order at inference so stage-conditioned checkpoints see the
        # same conditioning distribution they optimized against.
        if condition.dim() == 3 and int(condition.shape[0]) == 1:
            condition = apply_puzzle_stage_control_to_conditioning(
                condition.squeeze(0),
                puzzle_stage_condition=graph_context.get("puzzle_stage_condition"),
                graph_conditioning_mode="node_sequence",
                scale=puzzle_stage_token_scale,
            ).unsqueeze(0)
        elif condition.dim() == 2:
            if use_graph_node_cross_attention:
                condition = apply_puzzle_stage_control_to_conditioning(
                    condition,
                    puzzle_stage_condition=graph_context.get("puzzle_stage_condition"),
                    graph_conditioning_mode="node_sequence",
                    scale=puzzle_stage_token_scale,
                ).unsqueeze(0)
            else:
                condition = apply_puzzle_stage_control_to_conditioning(
                    condition,
                    puzzle_stage_condition=graph_context.get("puzzle_stage_condition"),
                    graph_conditioning_mode="pooled",
                    scale=puzzle_stage_token_scale,
                )

    return condition


def _topological_generation_layers(pipeline, graph: nx.Graph) -> List[List[Any]]:
    """Return dependency-safe generation waves for directed or undirected graphs."""
    if graph.is_directed() and not nx.is_directed_acyclic_graph(graph):
        logger.debug(
            "Mission graph contains cycles; falling back to independent-set wave batching."
        )
    return causal_generation_layers(graph)


def _infer_room_latent_shape(
    pipeline,
    *,
    neighbor_latents: Dict[str, Optional[Any]],
) -> Tuple[int, int, int]:
    """Infer per-room latent (C,H,W) shape from neighbors or defaults."""
    if pipeline.room_generator_mode == "discrete_masked":
        hidden_dim = int(getattr(pipeline.masked_room_model, "hidden_dim", 64))
        default_shape = (hidden_dim, ROOM_HEIGHT, ROOM_WIDTH)
    else:
        diffusion = getattr(pipeline, "diffusion", None)
        vqvae = getattr(pipeline, "vqvae", None)
        latent_dim = getattr(diffusion, "latent_dim", None)
        if latent_dim is None:
            latent_dim = getattr(vqvae, "latent_dim", None)
        if latent_dim is None:
            latent_dim = getattr(getattr(vqvae, "quantizer", object()), "embedding_dim", None)
        if latent_dim is None:
            diffusion = pipeline._require_component("diffusion", "_infer_room_latent_shape")
            latent_dim = getattr(diffusion, "latent_dim")
        default_shape = (
            int(latent_dim),
            int(DEFAULT_ROOM_LATENT_HW[0]),
            int(DEFAULT_ROOM_LATENT_HW[1]),
        )
    for latent in neighbor_latents.values():
        if isinstance(latent, torch.Tensor) and latent.dim() == 4:
            return (int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))
        if isinstance(latent, np.ndarray) and latent.ndim == 4:
            return (int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))
    return default_shape


def _normalize_neighbor_latents(
    pipeline,
    neighbor_latents: Dict[str, Optional[Any]],
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Normalize neighbor latents to detached float32 tensors on pipeline device.

    Accepts torch tensors and numpy arrays with expected shape [1, C, H, W].
    """
    normalized: Dict[str, Optional[torch.Tensor]] = {}
    for direction, latent in dict(neighbor_latents).items():
        direction_key = str(direction)
        if latent is None:
            normalized[direction_key] = None
            continue
        if isinstance(latent, torch.Tensor):
            if latent.dim() != 4:
                raise ValueError(
                    f"Neighbor latent '{direction_key}' must be rank-4 tensor, got shape={tuple(latent.shape)}"
                )
            normalized[direction_key] = latent.detach().to(pipeline.device, dtype=torch.float32).contiguous()
            continue
        if isinstance(latent, np.ndarray):
            if latent.ndim != 4:
                raise ValueError(
                    f"Neighbor latent '{direction_key}' must be rank-4 ndarray, got shape={tuple(latent.shape)}"
                )
            normalized[direction_key] = (
                torch.from_numpy(latent)
                .detach()
                .to(pipeline.device, dtype=torch.float32)
                .contiguous()
            )
            continue
        raise TypeError(
            f"Neighbor latent '{direction_key}' has unsupported type: {type(latent).__name__}"
        )
    return normalized


def _cast_latent_for_vqvae_decode(pipeline, latent: torch.Tensor) -> torch.Tensor:
    """Match sampled latent dtype/device to the VQ-VAE decoder contract."""
    try:
        reference = next(pipeline.vqvae.parameters())
        target_device = reference.device
        target_dtype = reference.dtype
    except StopIteration:
        target_device = pipeline.device
        target_dtype = latent.dtype

    prepared = latent.contiguous()
    if prepared.device == target_device and prepared.dtype == target_dtype:
        return prepared
    return prepared.to(device=target_device, dtype=target_dtype)


def _synchronize_cuda_device(pipeline) -> None:
    """Conservatively drain queued CUDA work before cross-branch fallback handoffs."""
    if not torch.cuda.is_available():
        return
    device = pipeline.device if isinstance(pipeline.device, torch.device) else torch.device(pipeline.device)
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        logger.debug("CUDA synchronize skipped during room-generation fallback handoff.", exc_info=True)


def _decode_latent_with_vqvae(pipeline, latent: torch.Tensor) -> torch.Tensor:
    """Decode a latent with a defensive retry for rare cuDNN stream-handoff failures."""
    diffusion = getattr(pipeline, "diffusion", None)
    if diffusion is not None and hasattr(diffusion, "unscale_first_stage_latent"):
        latent = diffusion.unscale_first_stage_latent(latent)
    prepared = pipeline._cast_latent_for_vqvae_decode(latent)
    try:
        return pipeline.vqvae.decode(prepared)
    except RuntimeError as exc:
        message = str(exc)
        if "stream_mismatch" not in message.lower() and "stream mismatch" not in message.lower():
            raise
        pipeline._bump_diagnostic("vqvae_decode_stream_mismatch_retry")
        logger.warning(
            "VQ-VAE decode hit a CUDA stream mismatch; synchronizing device and retrying with cuDNN disabled."
        )
        pipeline._synchronize_cuda_device()
        safe_latent = prepared.detach().clone().contiguous()
        with torch.backends.cudnn.flags(enabled=False):
            return pipeline.vqvae.decode(safe_latent)


def _estimate_safe_batch_size(
    pipeline,
    *,
    requested_batch_size: int,
    latent_shape_chw: Tuple[int, int, int],
) -> int:
    """Estimate VRAM-safe batch size and clamp requested size accordingly."""
    requested = max(1, int(requested_batch_size))
    if not torch.cuda.is_available():
        return requested
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device=pipeline.device)
        free_bytes = int(max(0, free_bytes))
    except Exception:
        return requested

    c, h, w = [max(1, int(v)) for v in latent_shape_chw]
    # Conservative estimate: latent + activations + guidance/intermediate buffers.
    bytes_per_sample = int(c * h * w * 4 * 18)
    reserve = int(1024 * 1024 * 256)  # 256MB reserve for fragmentation/runtime overhead
    usable = max(0, free_bytes - reserve)
    if bytes_per_sample <= 0 or usable <= 0:
        return 1
    safe = max(1, int(usable // bytes_per_sample))
    return max(1, min(requested, safe))


def _stack_room_topology_maps(
    pipeline,
    topology_maps: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Normalize per-room topology maps to a batched [B,C,H,W] tensor."""
    stacked: List[torch.Tensor] = []
    for topo in topology_maps:
        if not isinstance(topo, torch.Tensor):
            raise TypeError(f"room_topology_map must be a tensor, got {type(topo).__name__}")
        tensor = topo.to(pipeline.device, dtype=torch.float32)
        if tensor.dim() == 4:
            if int(tensor.shape[0]) != 1:
                raise ValueError(
                    f"Per-room room_topology_map must have batch size 1 when rank-4, got {tuple(tensor.shape)}."
                )
            tensor = tensor.squeeze(0)
        elif tensor.dim() != 3:
            raise ValueError(
                f"Per-room room_topology_map must have shape [1,C,H,W] or [C,H,W], got {tuple(tensor.shape)}."
            )
        stacked.append(tensor.contiguous())
    if not stacked:
        raise ValueError("Cannot batch room_topology_map for an empty room set.")
    return torch.stack(stacked, dim=0)


def _select_graph_guidance_batch(
    graph_ctx: Dict[str, Any],
    batch_indices: List[int],
) -> Dict[str, Any]:
    """Subset a batched graph-guidance payload to a list of active batch indices."""
    batched_per_room_keys = {
        "boundary_constraints",
        "room_topology_map",
        "logic_source_mask",
        "logic_target_mask",
        "current_node_idx",
        "current_node_distance",
    }
    max_idx = max(batch_indices)
    selected: Dict[str, Any] = {}
    for key, value in graph_ctx.items():
        if not isinstance(value, torch.Tensor):
            selected[key] = value
            continue
        if key in batched_per_room_keys and value.dim() >= 1 and int(value.shape[0]) > max_idx:
            selected[key] = value[batch_indices]
        else:
            selected[key] = value
    return selected


def _slice_graph_guidance_batch(pipeline, graph_ctx: Dict[str, Any], batch_index: int) -> Dict[str, Any]:
    """Slice a batched graph-guidance payload down to one room/sample.

    Only per-room batched tensors (boundary constraints, topology maps, per-room masks/indices)
    should be indexed by batch_index. Whole-graph tensors (node_features, edge_index,
    node_mask, tpe, node_positions, edge_rrwp) must be preserved intact.
    """
    return _select_graph_guidance_batch(graph_ctx, [int(batch_index)])


def _bucket_room_ids_by_latent_shape(
    pipeline,
    *,
    room_ids: List[Any],
    mission_graph_physical: nx.Graph,
    room_latents: Dict[int, torch.Tensor],
) -> Dict[Tuple[int, int, int, int, int], List[Any]]:
    """Bucket independent rooms by latent shape and target room size."""
    buckets: Dict[Tuple[int, int, int, int, int], List[Any]] = {}
    for room_id in room_ids:
        neighbor_latents = pipeline._get_neighbor_latents(room_id, mission_graph_physical, room_latents)
        shape_chw = pipeline._infer_room_latent_shape(neighbor_latents=neighbor_latents)
        attrs = mission_graph_physical.nodes[room_id] if room_id in mission_graph_physical else {}
        target_h = int(attrs.get('room_height', attrs.get('height', ROOM_HEIGHT)))
        target_w = int(attrs.get('room_width', attrs.get('width', ROOM_WIDTH)))
        shape_key = (int(shape_chw[0]), int(shape_chw[1]), int(shape_chw[2]), target_h, target_w)
        if shape_key not in buckets:
            buckets[shape_key] = []
        buckets[shape_key].append(room_id)
    return buckets

