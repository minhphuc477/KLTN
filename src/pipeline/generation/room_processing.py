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
from src.pipeline.spatial_utils import stable_node_sort_key
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


def _build_puzzle_room_route_template(
    pipeline,
    *,
    archetype: str,
    gate_family: str,
    variant_spec: Optional[Mapping[str, Any]] = None,
    stateful_anchor: Optional[Tuple[int, int]],
    interaction_sequence: Optional[Sequence[Tuple[str, Tuple[int, int]]]] = None,
    flow_is_horizontal: bool,
    source_anchor: Tuple[int, int],
    destination_anchor: Tuple[int, int],
    puzzle_anchor: Tuple[int, int],
    role_flags: Dict[str, bool],
    semantics: Dict[str, Any],
) -> np.ndarray:
    """
    Build an archetype-specific route skeleton for constructive puzzle rooms.

    The default semantic room trace is intentionally permissive; it keeps
    traversal valid, but it does not impose much puzzle readability. For
    puzzle rooms we instead reserve a more explicit route skeleton first,
    then place obstacles around it. This yields rooms that read like
    gates/hubs/serpentines instead of random block bars.
    """
    mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    source = pipeline._clamp_room_coord(source_anchor)
    destination = pipeline._clamp_room_coord(destination_anchor)
    puzzle = pipeline._clamp_room_coord(puzzle_anchor)
    stateful = pipeline._clamp_room_coord(stateful_anchor) if stateful_anchor is not None else puzzle
    center = (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)
    switch_depth = int(max(1, getattr(pipeline, "default_puzzle_room_switch_pocket_depth", 3)))
    resource_offset = int(max(1, getattr(pipeline, "default_puzzle_room_resource_bypass_offset", 2)))
    key_depth = int(max(1, getattr(pipeline, "default_puzzle_room_key_pocket_depth", 3)))
    item_slot_depth = int(max(1, getattr(pipeline, "default_puzzle_room_item_slot_depth", 3)))
    toggle_offset = int(max(1, getattr(pipeline, "default_puzzle_room_toggle_corridor_offset", 2)))
    variant = dict(variant_spec or {})
    variant_style = str(variant.get("style", "baseline") or "baseline").strip().lower()
    variant_side_bias = int(max(-1, min(1, int(variant.get("side_bias", 0) or 0))))

    def _pick_side_lane(anchor_value: int, reference_value: int, *, low: int, high: int, offset: int) -> int:
        anchor_value = int(anchor_value)
        reference_value = int(reference_value)
        if variant_side_bias != 0:
            direction = int(variant_side_bias)
        else:
            direction = 1 if anchor_value >= reference_value else -1
        candidates = []
        if abs(anchor_value - reference_value) >= 1:
            candidates.append(anchor_value)
        candidates.append(anchor_value + direction * max(1, offset))
        candidates.append(reference_value + direction * max(2, offset))
        candidates.append(reference_value + (2 if direction >= 0 else -2))
        for candidate in candidates:
            candidate = max(low, min(high, int(candidate)))
            if abs(candidate - reference_value) >= 1:
                return candidate
        fallback = reference_value + (1 if reference_value < ((low + high) // 2) else -1)
        return max(low, min(high, int(fallback)))

    def _add_polyline(points: List[Tuple[int, int]]) -> None:
        if len(points) < 2:
            return
        for start, end in zip(points, points[1:]):
            pipeline._paint_room_line_mask(mask, pipeline._clamp_room_coord(start), pipeline._clamp_room_coord(end))

    if archetype == "gate":
        if flow_is_horizontal:
            gate_col = max(3, min(ROOM_WIDTH - 4, int(puzzle[1])))
            pocket_row = max(
                2,
                min(
                    ROOM_HEIGHT - 3,
                    int(stateful[0]) if variant_side_bias == 0 else int(center[0] + variant_side_bias * max(2, abs(int(stateful[0]) - int(center[0])) or 2)),
                ),
            )
            entry = (source[0], max(2, gate_col - 2))
            if gate_family == "switch":
                pocket = (pocket_row, max(2, gate_col - switch_depth))
                gate_open = (pocket_row, gate_col)
                exit_point = (
                    center[0] if variant_style == "bridge" else pocket_row,
                    min(ROOM_WIDTH - 3, gate_col + (3 if variant_style == "bridge" else 2)),
                )
            elif gate_family == "toggle":
                toggle_row = max(
                    2,
                    min(
                        ROOM_HEIGHT - 3,
                        int(stateful[0]) + (-toggle_offset if stateful[0] > ROOM_HEIGHT // 2 else toggle_offset),
                    ),
                )
                if variant_style == "weave":
                    pocket = (toggle_row, max(2, gate_col - 3))
                else:
                    pocket = (toggle_row, max(2, gate_col - 2))
                gate_open = (stateful[0], gate_col)
                exit_point = (toggle_row if variant_style == "weave" else stateful[0], min(ROOM_WIDTH - 3, gate_col + 2))
            elif gate_family == "bombable":
                bypass_row = _pick_side_lane(
                    int(stateful[0]),
                    int(center[0]),
                    low=2,
                    high=ROOM_HEIGHT - 3,
                    offset=resource_offset,
                )
                pocket = (bypass_row, max(2, gate_col - (resource_offset + (2 if variant_style == "wrap" else 1))))
                gate_open = (bypass_row, gate_col)
                exit_point = (bypass_row, min(ROOM_WIDTH - 3, gate_col + (4 if variant_style == "wrap" else 2)))
            elif gate_family == "item_unlock":
                item_row = max(2, min(ROOM_HEIGHT - 3, int(stateful[0])))
                item_col = max(
                    2,
                    min(
                        ROOM_WIDTH - 3,
                        int(stateful[1]) + (
                            variant_side_bias * max(1, item_slot_depth - 1)
                            if variant_style in {"slot", "ring"} else 0
                        ),
                    ),
                )
                pocket = (item_row, item_col)
                gate_open = (center[0], gate_col)
                exit_point = (item_row if variant_style == "ring" else center[0], max(min(ROOM_WIDTH - 3, item_col), min(ROOM_WIDTH - 3, gate_col + 2)))
            elif gate_family == "key" and stateful_anchor is not None:
                key_row = max(
                    2,
                    min(
                        ROOM_HEIGHT - 3,
                        int(stateful[0]) if variant_side_bias == 0 else int(center[0] + variant_side_bias * max(2, abs(int(stateful[0]) - int(center[0])) or 2)),
                    ),
                )
                pocket = (key_row, max(2, gate_col - key_depth))
                gate_open = (center[0], gate_col)
                exit_point = (key_row if variant_style == "split" else center[0], min(ROOM_WIDTH - 3, gate_col + 2))
            else:
                pocket = (pocket_row, max(2, gate_col - 2))
                gate_open = (pocket_row, gate_col)
                exit_point = (pocket_row, min(ROOM_WIDTH - 3, gate_col + (4 if variant_style == "wrap" else 2)))
            destination_hook = (destination[0], int(exit_point[1]))
            _add_polyline([source, entry, pocket, gate_open, exit_point, destination_hook, destination])
        else:
            gate_row = max(3, min(ROOM_HEIGHT - 4, int(puzzle[0])))
            pocket_col = max(
                2,
                min(
                    ROOM_WIDTH - 3,
                    int(stateful[1]) if variant_side_bias == 0 else int(center[1] + variant_side_bias * max(2, abs(int(stateful[1]) - int(center[1])) or 2)),
                ),
            )
            entry = (max(2, gate_row - 2), source[1])
            if gate_family == "switch":
                pocket = (max(2, gate_row - switch_depth), pocket_col)
                gate_open = (gate_row, pocket_col)
                exit_point = (min(ROOM_HEIGHT - 3, gate_row + (3 if variant_style == "bridge" else 2)), center[1] if variant_style == "bridge" else pocket_col)
            elif gate_family == "toggle":
                toggle_col = max(
                    2,
                    min(
                        ROOM_WIDTH - 3,
                        int(stateful[1]) + (-toggle_offset if stateful[1] > ROOM_WIDTH // 2 else toggle_offset),
                    ),
                )
                pocket = (max(2, gate_row - (3 if variant_style == "weave" else 2)), toggle_col)
                gate_open = (gate_row, stateful[1])
                exit_point = (min(ROOM_HEIGHT - 3, gate_row + 2), toggle_col if variant_style == "weave" else stateful[1])
            elif gate_family == "bombable":
                bypass_col = _pick_side_lane(
                    int(stateful[1]),
                    int(center[1]),
                    low=2,
                    high=ROOM_WIDTH - 3,
                    offset=resource_offset,
                )
                pocket = (max(2, gate_row - (resource_offset + (2 if variant_style == "wrap" else 1))), bypass_col)
                gate_open = (gate_row, bypass_col)
                exit_point = (min(ROOM_HEIGHT - 3, gate_row + (4 if variant_style == "wrap" else 2)), bypass_col)
            elif gate_family == "item_unlock":
                item_row = max(
                    2,
                    min(
                        ROOM_HEIGHT - 3,
                        int(stateful[0]) + (
                            variant_side_bias * max(1, item_slot_depth - 1)
                            if variant_style in {"slot", "ring"} else 0
                        ),
                    ),
                )
                item_col = max(2, min(ROOM_WIDTH - 3, int(stateful[1])))
                pocket = (item_row, item_col)
                gate_open = (gate_row, center[1])
                exit_point = (max(min(ROOM_HEIGHT - 3, item_row), min(ROOM_HEIGHT - 3, gate_row + 2)), item_col if variant_style == "ring" else center[1])
            elif gate_family == "key" and stateful_anchor is not None:
                key_col = max(
                    2,
                    min(
                        ROOM_WIDTH - 3,
                        int(stateful[1]) if variant_side_bias == 0 else int(center[1] + variant_side_bias * max(2, abs(int(stateful[1]) - int(center[1])) or 2)),
                    ),
                )
                pocket = (max(2, gate_row - key_depth), key_col)
                gate_open = (gate_row, center[1])
                exit_point = (min(ROOM_HEIGHT - 3, gate_row + 2), key_col if variant_style == "split" else center[1])
            else:
                pocket = (max(2, gate_row - 2), pocket_col)
                gate_open = (gate_row, pocket_col)
                exit_point = (min(ROOM_HEIGHT - 3, gate_row + (4 if variant_style == "wrap" else 2)), pocket_col)
            destination_hook = (int(exit_point[0]), destination[1])
            _add_polyline([source, entry, pocket, gate_open, exit_point, destination_hook, destination])
    elif archetype == "hub":
        hub_row = int(round((puzzle[0] + center[0]) / 2.0))
        hub_col = int(round((puzzle[1] + center[1]) / 2.0))
        if variant_style == "offset":
            hub_row += int(variant_side_bias * 2)
        hub = pipeline._clamp_room_coord((hub_row, hub_col))
        _add_polyline([source, hub, destination])
        for direction, enabled in semantics.get("required_doors", {}).items():
            if not bool(enabled):
                continue
            door_anchor = pipeline._clamp_room_coord(
                build_room_semantic_anchor_points(
                    room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
                    required_doors={str(direction): True},
                    incoming_dirs=set(),
                    outgoing_dirs=set(),
                    room_role_flags={},
                    semantic_puzzle_offset=pipeline.default_semantic_puzzle_offset,
                ).get(f"door:{direction}", hub)
            )
            pipeline._paint_room_line_mask(mask, hub, door_anchor)
        if variant_style == "cross":
            pipeline._paint_room_line_mask(mask, hub, (hub[0], max(1, hub[1] - 3)))
            pipeline._paint_room_line_mask(mask, hub, (hub[0], min(ROOM_WIDTH - 2, hub[1] + 3)))
        mask[max(1, hub[0] - 1): min(ROOM_HEIGHT - 1, hub[0] + 2), max(1, hub[1] - 1): min(ROOM_WIDTH - 1, hub[1] + 2)] = True
    elif archetype == "combat":
        arena_center = pipeline._clamp_room_coord(
            (
                int(round((source[0] + destination[0] + puzzle[0]) / 3.0)) + (variant_side_bias * 2 if variant_style == "offset" else 0),
                int(round((source[1] + destination[1] + puzzle[1]) / 3.0)),
            )
        )
        _add_polyline([source, arena_center, destination])
        pipeline._paint_room_line_mask(mask, arena_center, puzzle)
        if variant_style == "cross":
            pipeline._paint_room_line_mask(mask, (arena_center[0], max(1, arena_center[1] - 3)), (arena_center[0], min(ROOM_WIDTH - 2, arena_center[1] + 3)))
        mask[max(1, arena_center[0] - 1): min(ROOM_HEIGHT - 1, arena_center[0] + 2), max(1, arena_center[1] - 1): min(ROOM_WIDTH - 1, arena_center[1] + 2)] = True
    elif archetype == "island":
        waypoint = pipeline._clamp_room_coord(
            (
                int(round((puzzle[0] + destination[0]) / 2.0)) + (variant_side_bias * 2 if variant_style == "bridge" else 0),
                int(round((puzzle[1] + destination[1]) / 2.0)),
            )
        )
        _add_polyline([source, puzzle, waypoint, destination])
        if variant_style == "staggered":
            _add_polyline([source, (puzzle[0], max(1, puzzle[1] - 2)), puzzle])
        mask[max(1, puzzle[0] - 1): min(ROOM_HEIGHT - 1, puzzle[0] + 2), max(1, puzzle[1] - 1): min(ROOM_WIDTH - 1, puzzle[1] + 2)] = True
    else:  # serpentine
        if flow_is_horizontal:
            if variant_style == "split":
                side_row = max(2, min(ROOM_HEIGHT - 3, center[0] + variant_side_bias * 3))
                waypoints = [source, (2, 3), (side_row, 3), (side_row, ROOM_WIDTH - 4), (ROOM_HEIGHT - 4, ROOM_WIDTH - 4), destination]
            else:
                left_first = variant_style != "mirror"
                waypoints = [
                    source,
                    (2, 3 if left_first else ROOM_WIDTH - 4),
                    (4, 3 if left_first else ROOM_WIDTH - 4),
                    (4, ROOM_WIDTH - 4 if left_first else 3),
                    (8, ROOM_WIDTH - 4 if left_first else 3),
                    (8, 3 if left_first else ROOM_WIDTH - 4),
                    (12, 3 if left_first else ROOM_WIDTH - 4),
                    (12, ROOM_WIDTH - 4 if left_first else 3),
                    destination,
                ]
        else:
            if variant_style == "split":
                side_col = max(2, min(ROOM_WIDTH - 3, center[1] + variant_side_bias * 2))
                waypoints = [source, (3, 2), (3, side_col), (ROOM_HEIGHT - 4, side_col), (ROOM_HEIGHT - 4, ROOM_WIDTH - 3), destination]
            else:
                top_first = variant_style != "mirror"
                waypoints = [
                    source,
                    (3 if top_first else ROOM_HEIGHT - 4, 2),
                    (3 if top_first else ROOM_HEIGHT - 4, 4),
                    (ROOM_HEIGHT - 4 if top_first else 3, 4),
                    (ROOM_HEIGHT - 4 if top_first else 3, 6),
                    (3 if top_first else ROOM_HEIGHT - 4, 6),
                    (3 if top_first else ROOM_HEIGHT - 4, ROOM_WIDTH - 3),
                    destination,
                ]
        _add_polyline(waypoints)
        pipeline._paint_room_line_mask(mask, puzzle, waypoints[min(len(waypoints) - 2, 3)])

    if role_flags.get("has_puzzle", False):
        mask[max(1, puzzle[0] - 1): min(ROOM_HEIGHT - 1, puzzle[0] + 2), max(1, puzzle[1] - 1): min(ROOM_WIDTH - 1, puzzle[1] + 2)] = True

    sequence = list(interaction_sequence or [])
    if len(sequence) >= 2:
        ordered_points: List[Tuple[int, int]] = [source]
        ordered_points.extend(pipeline._clamp_room_coord(anchor) for _name, anchor in sequence)
        ordered_points.append(destination)
        _add_polyline(ordered_points)
        for _name, anchor in sequence:
            seq_r, seq_c = pipeline._clamp_room_coord(anchor)
            mask[
                max(1, seq_r - 1): min(ROOM_HEIGHT - 1, seq_r + 2),
                max(1, seq_c - 1): min(ROOM_WIDTH - 1, seq_c + 2),
            ] = True

    if gate_family in {"switch", "toggle"}:
        # A push puzzle has a temporal route, not one permanently walkable
        # corridor: the block's initial cell is occupied before the push and
        # becomes traversable afterwards. Do not reserve that cell as a
        # static path. The scaffold uses the same witnesses when placing the
        # block, so its geometry cannot contradict this route contract.
        for block_cell, player_cell, target_cell in _switch_push_witnesses(
            target_anchor=puzzle,
            source_anchor=source,
        ):
            block_r, block_c = block_cell
            player_r, player_c = player_cell
            target_r, target_c = target_cell
            mask[block_r, block_c] = False
            mask[player_r, player_c] = True
            mask[target_r, target_c] = True

    return mask


def _switch_push_witnesses(
    *,
    target_anchor: Tuple[int, int],
    source_anchor: Tuple[int, int],
) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Return valid one-push block/player/target arrangements for a marker.

    Each witness is ``(block_origin, player_staging, target)``. Ordering is
    deterministic and favours the staging position closest to the entry. This
    is deliberately geometry-only: callers still verify floor availability
    and actual reachability on the rendered candidate grid.
    """
    target_r, target_c = int(target_anchor[0]), int(target_anchor[1])
    source_r, source_c = int(source_anchor[0]), int(source_anchor[1])
    witnesses: List[
        Tuple[int, int, Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ] = []
    for order, (delta_r, delta_c) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
        block_r = target_r - int(delta_r)
        block_c = target_c - int(delta_c)
        player_r = block_r - int(delta_r)
        player_c = block_c - int(delta_c)
        if not (
            2 <= target_r <= ROOM_HEIGHT - 3
            and 2 <= target_c <= ROOM_WIDTH - 3
            and 2 <= block_r <= ROOM_HEIGHT - 3
            and 2 <= block_c <= ROOM_WIDTH - 3
            and 1 <= player_r < ROOM_HEIGHT - 1
            and 1 <= player_c < ROOM_WIDTH - 1
        ):
            continue
        source_distance = abs(player_r - source_r) + abs(player_c - source_c)
        witnesses.append(
            (
                int(source_distance),
                int(order),
                (int(block_r), int(block_c)),
                (int(player_r), int(player_c)),
                (int(target_r), int(target_c)),
            )
        )
    witnesses.sort(key=lambda item: (item[0], item[1]))
    return [(block, player, target) for _distance, _order, block, player, target in witnesses]


def _resolve_puzzle_interaction_sequence(
    pipeline,
    *,
    archetype: str,
    gate_family: str,
    role_flags: Mapping[str, bool],
    semantic_anchors: Mapping[str, Tuple[int, int]],
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Resolve a staged in-room interaction sequence for complex puzzle rooms.

    This is intentionally a lightweight constructive grammar. It does not
    simulate a full state machine, but it gives the room generator a
    progression order across local anchors such as key, item, puzzle, and
    enemy instead of treating every puzzle room as a single interaction.
    """
    archetype = str(archetype or "serpentine").strip().lower()
    gate_family = str(gate_family or "generic").strip().lower()
    flags = {str(key): bool(value) for key, value in dict(role_flags or {}).items()}
    anchors = {
        str(name): pipeline._clamp_room_coord(coord)
        for name, coord in dict(semantic_anchors or {}).items()
        if coord is not None
    }
    sequence: List[Tuple[str, Tuple[int, int]]] = []
    seen: Set[str] = set()

    def _add(name: str) -> None:
        key = str(name)
        if key in seen:
            return
        anchor = anchors.get(key)
        if anchor is None:
            return
        sequence.append((key, anchor))
        seen.add(key)

    multi_step_room = bool(flags.get("is_complex_puzzle", False))
    multi_step_room = multi_step_room or archetype in {"hub", "combat"}
    multi_step_room = multi_step_room or sum(
        int(flags.get(name, False))
        for name in ("has_key", "has_item", "has_enemy", "has_puzzle")
    ) >= 3
    if multi_step_room:
        for name, flag_key in (
            ("key", "has_key"),
            ("item", "has_item"),
            ("puzzle", "has_puzzle"),
            ("enemy", "has_enemy"),
            ("boss", "has_boss"),
        ):
            if bool(flags.get(flag_key, False)):
                _add(name)
        return list(sequence)

    if gate_family == "key":
        _add("key")
        _add("puzzle")
    elif gate_family == "item_unlock":
        _add("item")
        _add("puzzle")
    elif gate_family in {"switch", "toggle", "bombable"}:
        _add("puzzle")
    elif gate_family == "combat":
        if flags.get("has_puzzle", False):
            _add("puzzle")
        _add("enemy")

    return list(sequence)


def _evaluate_puzzle_candidate_interaction_sequence(
    pipeline,
    *,
    grid: np.ndarray,
    route_mask: np.ndarray,
    source_anchor: Tuple[int, int],
    destination_anchor: Tuple[int, int],
    interaction_sequence: Sequence[Tuple[str, Tuple[int, int]]],
) -> Dict[str, Any]:
    """
    Validate simple staged room progression across multiple local anchors.

    This remains a room-local proxy for multi-step puzzle structure. A
    candidate is rewarded when the reserved route actually visits the
    intermediate anchors and when each consecutive stage is walkably
    connectable in the final geometry.
    """
    sequence = [
        (str(name), pipeline._clamp_room_coord(anchor))
        for name, anchor in list(interaction_sequence or [])
        if anchor is not None
    ]
    if len(sequence) <= 1:
        return {
            "required": 0,
            "valid": 1,
            "score": 0.0,
            "failure_reasons": [],
            "sequence_length": int(len(sequence)),
            "route_anchor_coverage": 1.0 if sequence else 0.0,
            "pairwise_path_ratio": 1.0,
            "sequence_names": [name for name, _anchor in sequence],
        }

    walkable = pipeline._build_room_walkable_mask(grid)
    route_arr = np.asarray(route_mask, dtype=bool)

    covered = 0
    for _name, anchor in sequence:
        anchor_r, anchor_c = anchor
        route_r0 = max(0, anchor_r - 1)
        route_r1 = min(ROOM_HEIGHT, anchor_r + 2)
        route_c0 = max(0, anchor_c - 1)
        route_c1 = min(ROOM_WIDTH, anchor_c + 2)
        if bool(np.any(route_arr[route_r0:route_r1, route_c0:route_c1])):
            covered += 1
    route_anchor_coverage = float(covered) / float(max(1, len(sequence)))

    path_nodes: List[Tuple[int, int]] = [pipeline._clamp_room_coord(source_anchor)]
    path_nodes.extend(anchor for _name, anchor in sequence)
    path_nodes.append(pipeline._clamp_room_coord(destination_anchor))
    pairwise_success = 0
    pairwise_total = max(0, len(path_nodes) - 1)
    for start, goal in zip(path_nodes, path_nodes[1:]):
        if pipeline._shortest_room_path(walkable, start, goal):
            pairwise_success += 1
    pairwise_path_ratio = float(pairwise_success) / float(max(1, pairwise_total))

    failure_reasons: List[str] = []
    if route_anchor_coverage < 1.0:
        failure_reasons.append("incomplete_sequence_route")
    if pairwise_path_ratio < 1.0:
        failure_reasons.append("broken_sequence_connectivity")

    score = 0.0
    score += 0.90 * route_anchor_coverage
    score += 0.70 * pairwise_path_ratio
    score += 0.20 * float(min(1.0, float(len(sequence) - 1) / 3.0))
    valid = int(len(failure_reasons) == 0)
    if not valid:
        score -= 1.75 + (0.10 * len(failure_reasons))

    return {
        "required": 1,
        "valid": int(valid),
        "score": float(score),
        "failure_reasons": list(failure_reasons),
        "sequence_length": int(len(sequence)),
        "route_anchor_coverage": float(route_anchor_coverage),
        "pairwise_path_ratio": float(pairwise_path_ratio),
        "sequence_names": [name for name, _anchor in sequence],
    }


def _select_puzzle_room_scaffold_archetype(
    pipeline,
    *,
    role_flags: Dict[str, bool],
    semantics: Dict[str, Any],
    node_type: str,
) -> str:
    """Select the constructive puzzle archetype from graph-local semantics."""
    forced = str(getattr(pipeline, "default_puzzle_room_archetype_mode", "auto") or "auto").strip().lower()
    valid = {"auto", "gate", "serpentine", "hub", "island", "combat"}
    if forced not in valid:
        forced = "auto"
    if forced != "auto":
        return forced

    edge_constraints = semantics.get("edge_constraints", {})
    flat_edge_tokens: Set[str] = set()
    for tokens in edge_constraints.values():
        flat_edge_tokens.update(str(token) for token in tokens)

    required_doors = semantics.get("required_doors", {})
    required_door_count = int(sum(1 for enabled in required_doors.values() if enabled))

    if node_type == "combat_puzzle" or bool(role_flags.get("has_enemy", False)):
        return "combat"
    if {"switch", "switch_locked", "state_block", "on_off_gate"} & flat_edge_tokens:
        return "gate"
    if node_type in {"item", "protection_item", "minor_item", "treasure", "stair", "stairs_up", "stairs_down", "warp"} or bool(role_flags.get("has_item", False)):
        return "island"
    if required_door_count >= 3:
        return "hub"
    return "serpentine"


def _classify_puzzle_gate_family(
    pipeline,
    *,
    role_flags: Dict[str, bool],
    semantics: Dict[str, Any],
    node_type: str,
) -> str:
    """Classify the local puzzle as a concrete gate semantic family."""
    edge_constraints = semantics.get("edge_constraints", {})
    flat_edge_tokens: Set[str] = set()
    for tokens in edge_constraints.values():
        flat_edge_tokens.update(str(token) for token in tokens)

    if {"state_block", "on_off_gate"} & flat_edge_tokens:
        return "toggle"
    if role_flags.get("is_switch_puzzle", False) or {"switch", "switch_locked"} & flat_edge_tokens:
        return "switch"
    if role_flags.get("is_combat_puzzle", False) or node_type == "combat_puzzle":
        return "combat"
    if {"bombable"} & flat_edge_tokens:
        return "bombable"
    if {"item_locked", "item_gate"} & flat_edge_tokens:
        return "item_unlock"
    if {"key_locked", "locked", "boss_locked"} & flat_edge_tokens:
        return "key"
    if {"shutter", "soft_locked"} & flat_edge_tokens:
        return "combat"
    return "generic"


def _build_puzzle_room_variant_specs(
    pipeline,
    *,
    archetype: str,
    gate_family: str,
) -> List[Dict[str, Any]]:
    """Enumerate small, valid scaffold variants for novelty-aware puzzle selection."""
    if not bool(getattr(pipeline, "default_puzzle_room_novelty_enabled", True)):
        return [{"name": "baseline", "style": "baseline", "side_bias": 0, "branch_density_delta": 0.0, "block_budget_delta": 0}]

    specs: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def _add(
        name: str,
        *,
        style: str,
        side_bias: int = 0,
        branch_density_delta: float = 0.0,
        block_budget_delta: int = 0,
    ) -> None:
        variant_name = str(name).strip().lower()
        if not variant_name or variant_name in seen:
            return
        seen.add(variant_name)
        specs.append(
            {
                "name": variant_name,
                "style": str(style).strip().lower(),
                "side_bias": int(max(-1, min(1, int(side_bias)))),
                "branch_density_delta": float(branch_density_delta),
                "block_budget_delta": int(block_budget_delta),
            }
        )

    if archetype == "gate":
        if gate_family == "switch":
            _add("switch_upper_pocket", style="pocket", side_bias=-1, branch_density_delta=0.00, block_budget_delta=0)
            _add("switch_lower_pocket", style="pocket", side_bias=1, branch_density_delta=0.00, block_budget_delta=0)
            _add("switch_upper_bridge", style="bridge", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
            _add("switch_lower_bridge", style="bridge", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)
        elif gate_family == "toggle":
            _add("toggle_upper_corridor", style="corridor", side_bias=-1, branch_density_delta=0.04, block_budget_delta=1)
            _add("toggle_lower_corridor", style="corridor", side_bias=1, branch_density_delta=0.04, block_budget_delta=1)
            _add("toggle_upper_weave", style="weave", side_bias=-1, branch_density_delta=0.10, block_budget_delta=3)
            _add("toggle_lower_weave", style="weave", side_bias=1, branch_density_delta=0.10, block_budget_delta=3)
        elif gate_family == "bombable":
            _add("bomb_upper_bypass", style="bypass", side_bias=-1, branch_density_delta=0.02, block_budget_delta=0)
            _add("bomb_lower_bypass", style="bypass", side_bias=1, branch_density_delta=0.02, block_budget_delta=0)
            _add("bomb_upper_wrap", style="wrap", side_bias=-1, branch_density_delta=0.10, block_budget_delta=2)
            _add("bomb_lower_wrap", style="wrap", side_bias=1, branch_density_delta=0.10, block_budget_delta=2)
        elif gate_family == "item_unlock":
            _add("item_slot_left", style="slot", side_bias=-1, branch_density_delta=0.05, block_budget_delta=0)
            _add("item_slot_right", style="slot", side_bias=1, branch_density_delta=0.05, block_budget_delta=0)
            _add("item_ring_left", style="ring", side_bias=-1, branch_density_delta=0.12, block_budget_delta=3)
            _add("item_ring_right", style="ring", side_bias=1, branch_density_delta=0.12, block_budget_delta=3)
        elif gate_family == "key":
            _add("key_upper_alcove", style="alcove", side_bias=-1, branch_density_delta=0.00, block_budget_delta=0)
            _add("key_lower_alcove", style="alcove", side_bias=1, branch_density_delta=0.00, block_budget_delta=0)
            _add("key_upper_split", style="split", side_bias=-1, branch_density_delta=0.10, block_budget_delta=2)
            _add("key_lower_split", style="split", side_bias=1, branch_density_delta=0.10, block_budget_delta=2)
        else:
            _add("gate_upper_offset", style="offset", side_bias=-1, branch_density_delta=0.00, block_budget_delta=0)
            _add("gate_lower_offset", style="offset", side_bias=1, branch_density_delta=0.00, block_budget_delta=0)
            _add("gate_upper_wrap", style="wrap", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
            _add("gate_lower_wrap", style="wrap", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)
    elif archetype == "hub":
        _add("hub_ring", style="ring", branch_density_delta=0.00, block_budget_delta=0)
        _add("hub_cross", style="cross", branch_density_delta=0.06, block_budget_delta=2)
        _add("hub_upper_offset", style="offset", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
        _add("hub_lower_offset", style="offset", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)
    elif archetype == "combat":
        _add("combat_cross", style="cross", branch_density_delta=0.00, block_budget_delta=0)
        _add("combat_corners", style="corners", branch_density_delta=0.06, block_budget_delta=2)
        _add("combat_upper_lane", style="offset", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
        _add("combat_lower_lane", style="offset", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)
    elif archetype == "island":
        _add("island_quad", style="quad", branch_density_delta=0.00, block_budget_delta=0)
        _add("island_staggered", style="staggered", branch_density_delta=0.06, block_budget_delta=2)
        _add("island_upper_bridge", style="bridge", side_bias=-1, branch_density_delta=0.10, block_budget_delta=3)
        _add("island_lower_bridge", style="bridge", side_bias=1, branch_density_delta=0.10, block_budget_delta=3)
    else:
        _add("serpentine_classic", style="classic", branch_density_delta=0.00, block_budget_delta=0)
        _add("serpentine_mirror", style="mirror", branch_density_delta=0.02, block_budget_delta=0)
        _add("serpentine_upper_split", style="split", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
        _add("serpentine_lower_split", style="split", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)

    candidate_limit = int(max(1, min(6, int(getattr(pipeline, "default_puzzle_room_candidate_count", 4)))))
    return specs[:candidate_limit]


def _summarize_puzzle_candidate_descriptor(
    pipeline,
    *,
    grid: np.ndarray,
    stats: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compact numeric descriptor used for novelty-aware scaffold selection."""
    block_mask = np.asarray(grid, dtype=np.int32) == int(TileID.BLOCK)
    rows = np.sum(block_mask, axis=1)
    cols = np.sum(block_mask, axis=0)
    coords = np.argwhere(block_mask)
    if coords.size > 0:
        mean_row = float(np.mean(coords[:, 0]))
        mean_col = float(np.mean(coords[:, 1]))
    else:
        mean_row = float(ROOM_HEIGHT // 2)
        mean_col = float(ROOM_WIDTH // 2)
    half_r = ROOM_HEIGHT // 2
    half_c = ROOM_WIDTH // 2
    quadrants = [
        int(np.sum(block_mask[:half_r, :half_c])),
        int(np.sum(block_mask[:half_r, half_c:])),
        int(np.sum(block_mask[half_r:, :half_c])),
        int(np.sum(block_mask[half_r:, half_c:])),
    ]
    return {
        "variant_name": str(stats.get("variant_name", "") or ""),
        "archetype": str(stats.get("archetype", "") or ""),
        "gate_family": str(stats.get("gate_family", "") or ""),
        "tiles_added": int(stats.get("tiles_added", 0)),
        "segments_added": int(stats.get("segments_added", 0)),
        "row_coverage": int(np.sum(rows > 0)),
        "col_coverage": int(np.sum(cols > 0)),
        "center_row": float(mean_row),
        "center_col": float(mean_col),
        "quadrants": quadrants,
    }


def _build_room_walkable_mask(pipeline, grid: np.ndarray) -> np.ndarray:
    """Return walkable cells for room-local structural scoring."""
    grid_arr = np.asarray(grid, dtype=np.int32)
    blocked = np.isin(
        grid_arr,
        np.array([int(TileID.WALL), int(TileID.BLOCK)], dtype=np.int32),
    )
    return ~blocked


def _count_room_path_turns(pipeline, path: List[Tuple[int, int]]) -> int:
    """Count Manhattan direction changes along a discrete room path."""
    if len(path) < 3:
        return 0
    turns = 0
    prev_delta: Optional[Tuple[int, int]] = None
    for prev, cur in zip(path, path[1:]):
        delta = (int(cur[0]) - int(prev[0]), int(cur[1]) - int(prev[1]))
        if prev_delta is not None and delta != prev_delta:
            turns += 1
        prev_delta = delta
    return int(turns)


def _nearest_walkable_room_coord(
    pipeline,
    walkable: np.ndarray,
    anchor: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """Project an anchor onto the nearest walkable room tile."""
    mask = np.asarray(walkable, dtype=bool)
    if mask.shape != (ROOM_HEIGHT, ROOM_WIDTH):
        return None
    target_r, target_c = pipeline._clamp_room_coord(anchor)
    if bool(mask[target_r, target_c]):
        return (int(target_r), int(target_c))

    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[Tuple[int, int]] = deque([(int(target_r), int(target_c))])
    visited[target_r, target_c] = True
    while queue:
        row, col = queue.popleft()
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row + d_row
            next_col = col + d_col
            if not (0 <= next_row < ROOM_HEIGHT and 0 <= next_col < ROOM_WIDTH):
                continue
            if bool(visited[next_row, next_col]):
                continue
            if bool(mask[next_row, next_col]):
                return (int(next_row), int(next_col))
            visited[next_row, next_col] = True
            queue.append((int(next_row), int(next_col)))
    return None


def _shortest_room_path(
    pipeline,
    walkable: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Compute a room-local 4-neighbour shortest path."""
    mask = np.asarray(walkable, dtype=bool)
    if mask.shape != (ROOM_HEIGHT, ROOM_WIDTH):
        return []
    start_cell = pipeline._nearest_walkable_room_coord(mask, start)
    goal_cell = pipeline._nearest_walkable_room_coord(mask, goal)
    if start_cell is None or goal_cell is None:
        return []
    if start_cell == goal_cell:
        return [start_cell]

    queue: deque[Tuple[int, int]] = deque([start_cell])
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_cell: None}
    while queue:
        row, col = queue.popleft()
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row + d_row
            next_col = col + d_col
            next_cell = (int(next_row), int(next_col))
            if not (0 <= next_row < ROOM_HEIGHT and 0 <= next_col < ROOM_WIDTH):
                continue
            if not bool(mask[next_row, next_col]) or next_cell in parents:
                continue
            parents[next_cell] = (int(row), int(col))
            if next_cell == goal_cell:
                queue.clear()
                break
            queue.append(next_cell)

    if goal_cell not in parents:
        return []
    path: List[Tuple[int, int]] = []
    cursor: Optional[Tuple[int, int]] = goal_cell
    while cursor is not None:
        path.append((int(cursor[0]), int(cursor[1])))
        cursor = parents[cursor]
    path.reverse()
    return path


def _evaluate_puzzle_candidate_route_quality(
    pipeline,
    *,
    grid: np.ndarray,
    source_anchor: Tuple[int, int],
    destination_anchor: Tuple[int, int],
    stateful_anchor: Optional[Tuple[int, int]],
    route_mask: np.ndarray,
    gate_family: str,
    baseline_path_length: Optional[int],
) -> Dict[str, Any]:
    """
    Score puzzle readability from the actual walkable route, not novelty alone.

    The scaffold should not just be different; it should create a readable
    route with a visible detour or stateful pocket that matches the edge
    semantics implied by the topology.
    """
    walkable = pipeline._build_room_walkable_mask(grid)
    path = pipeline._shortest_room_path(walkable, source_anchor, destination_anchor)
    if not path:
        return {
            "path_exists": 0,
            "path_length": 0,
            "turn_count": 0,
            "route_overlap_ratio": 0.0,
            "detour_gain": 0.0,
            "stateful_distance_to_path": None,
            "stateful_via_path_length": None,
            "stateful_branch_gain": None,
            "stateful_on_path": 0,
            "score": -4.0,
        }

    path_length = max(0, len(path) - 1)
    turn_count = pipeline._count_room_path_turns(path)
    route_arr = np.asarray(route_mask, dtype=bool)
    route_overlap_ratio = (
        float(np.mean([1.0 if bool(route_arr[row, col]) else 0.0 for row, col in path]))
        if path else 0.0
    )

    reference_path_length = (
        int(baseline_path_length)
        if baseline_path_length is not None and int(baseline_path_length) > 0
        else int(path_length)
    )
    detour_gain = max(0.0, float(path_length - reference_path_length))

    stateful_distance_to_path: Optional[int] = None
    stateful_via_path_length: Optional[int] = None
    stateful_branch_gain: Optional[float] = None
    stateful_on_path = 0
    stateful_required = gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"}

    if stateful_anchor is not None:
        projected_stateful = pipeline._nearest_walkable_room_coord(walkable, stateful_anchor)
        if projected_stateful is not None:
            stateful_distance_to_path = int(
                min(
                    abs(int(projected_stateful[0]) - int(cell[0])) + abs(int(projected_stateful[1]) - int(cell[1]))
                    for cell in path
                )
            )
            stateful_on_path = int(stateful_distance_to_path == 0)
            source_to_stateful = pipeline._shortest_room_path(walkable, source_anchor, projected_stateful)
            stateful_to_goal = pipeline._shortest_room_path(walkable, projected_stateful, destination_anchor)
            if source_to_stateful and stateful_to_goal:
                stateful_via_path_length = max(0, len(source_to_stateful) - 1) + max(0, len(stateful_to_goal) - 1)
                stateful_branch_gain = float(max(0, stateful_via_path_length - path_length))

    route_quality_score = 0.0
    route_quality_score += 1.20 * float(max(0.0, min(1.0, route_overlap_ratio)))
    route_quality_score += 0.85 * float(min(1.0, detour_gain / 6.0))
    route_quality_score += 0.40 * float(min(1.0, turn_count / 4.0))

    if path_length > reference_path_length + 18:
        route_quality_score -= 0.35

    if stateful_required:
        if stateful_distance_to_path is None:
            route_quality_score -= 1.15
        else:
            proximity_score = max(0.0, 1.0 - (float(stateful_distance_to_path) / 5.0))
            route_quality_score += 1.10 * proximity_score

            desired_low = 0.0 if gate_family in {"switch", "toggle"} else 1.0
            desired_high = 4.0 if gate_family in {"switch", "toggle"} else 6.0
            branch_gain = float(stateful_branch_gain or 0.0)
            if desired_low <= branch_gain <= desired_high:
                route_quality_score += 0.60
            elif branch_gain < desired_low:
                route_quality_score += max(0.0, 0.60 - (0.35 * (desired_low - branch_gain)))
            else:
                route_quality_score += max(0.0, 0.60 - (0.10 * (branch_gain - desired_high)))

    return {
        "path_exists": 1,
        "path_length": int(path_length),
        "turn_count": int(turn_count),
        "route_overlap_ratio": float(route_overlap_ratio),
        "detour_gain": float(detour_gain),
        "stateful_distance_to_path": (
            int(stateful_distance_to_path) if stateful_distance_to_path is not None else None
        ),
        "stateful_via_path_length": (
            int(stateful_via_path_length) if stateful_via_path_length is not None else None
        ),
        "stateful_branch_gain": (
            float(stateful_branch_gain) if stateful_branch_gain is not None else None
        ),
        "stateful_on_path": int(stateful_on_path),
        "score": float(route_quality_score),
    }


def _evaluate_puzzle_candidate_contract(
    pipeline,
    *,
    grid: np.ndarray,
    gate_family: str,
    source_anchor: Tuple[int, int],
    destination_anchor: Tuple[int, int],
    stateful_anchor: Optional[Tuple[int, int]],
    route_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Validate that a puzzle scaffold expresses a readable local interaction contract.

    The scaffold should do more than add clutter. For stateful gate families,
    the room should expose a walkable interaction pocket near the stateful
    anchor, frame it with some local structure, and keep it legibly connected
    to the main route according to the intended gate semantics.
    """
    gate_family = str(gate_family or "generic").strip().lower()
    walkable = pipeline._build_room_walkable_mask(grid)
    stateful_required = gate_family in {"switch", "toggle", "bombable", "item_unlock", "key", "combat"}
    projected_stateful: Optional[Tuple[int, int]] = None
    if stateful_anchor is not None:
        projected_stateful = pipeline._nearest_walkable_room_coord(walkable, stateful_anchor)

    pocket_floor_tiles = 0
    frame_block_tiles = 0
    anchor_adjacent_walkable = 0
    if projected_stateful is not None:
        anchor_r, anchor_c = pipeline._clamp_room_coord(projected_stateful)
        for row in range(max(1, anchor_r - 1), min(ROOM_HEIGHT - 1, anchor_r + 2)):
            for col in range(max(1, anchor_c - 1), min(ROOM_WIDTH - 1, anchor_c + 2)):
                if bool(walkable[row, col]):
                    pocket_floor_tiles += 1
        for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_r = anchor_r + d_r
            next_c = anchor_c + d_c
            if 0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH and bool(walkable[next_r, next_c]):
                anchor_adjacent_walkable += 1
        for row in range(max(1, anchor_r - 2), min(ROOM_HEIGHT - 1, anchor_r + 3)):
            for col in range(max(1, anchor_c - 2), min(ROOM_WIDTH - 1, anchor_c + 3)):
                if abs(row - anchor_r) <= 1 and abs(col - anchor_c) <= 1:
                    continue
                if int(grid[row, col]) in {int(TileID.BLOCK), int(TileID.WALL)}:
                    frame_block_tiles += 1

    path_exists = int(route_quality.get("path_exists", 0) or 0)
    stateful_distance = route_quality.get("stateful_distance_to_path", None)
    stateful_via_path_length = route_quality.get("stateful_via_path_length", None)
    stateful_branch_gain = float(route_quality.get("stateful_branch_gain", 0.0) or 0.0)

    failure_reasons: List[str] = []
    if path_exists <= 0:
        failure_reasons.append("no_path")
    if stateful_required:
        if projected_stateful is None:
            failure_reasons.append("missing_stateful_anchor")
        if pocket_floor_tiles < 5:
            failure_reasons.append("weak_stateful_pocket")
        if anchor_adjacent_walkable < 1:
            failure_reasons.append("sealed_stateful_anchor")
        if gate_family in {"switch", "toggle", "combat", "key"}:
            min_frame_blocks = 2 if gate_family == "combat" else 4
            if frame_block_tiles < min_frame_blocks:
                failure_reasons.append("weak_local_structure")
        if stateful_via_path_length is None:
            failure_reasons.append("stateful_anchor_unreachable")
        if gate_family in {"key"} and stateful_branch_gain < 1.0:
            failure_reasons.append("missing_stateful_detour")

    contract_score = 0.0
    contract_score += 0.30 if path_exists > 0 else -1.50
    contract_score += 0.25 if projected_stateful is not None else -0.80
    contract_score += 0.35 * float(min(1.0, pocket_floor_tiles / 6.0))
    contract_score += 0.35 * float(min(1.0, frame_block_tiles / 6.0))
    contract_score += 0.20 * float(min(1.0, anchor_adjacent_walkable / 2.0))
    if stateful_via_path_length is not None:
        contract_score += 0.25
    if gate_family in {"item_unlock", "key", "bombable"}:
        contract_score += 0.25 * float(min(1.0, stateful_branch_gain / 2.0))
    valid = int(len(failure_reasons) == 0)
    if not valid:
        contract_score -= 1.5 + (0.10 * len(failure_reasons))

    return {
        "valid": int(valid),
        "score": float(contract_score),
        "failure_reasons": list(failure_reasons),
        "stateful_anchor_present": int(projected_stateful is not None),
        "projected_stateful_anchor": (
            [int(projected_stateful[0]), int(projected_stateful[1])]
            if projected_stateful is not None else None
        ),
        "pocket_floor_tiles": int(pocket_floor_tiles),
        "frame_block_tiles": int(frame_block_tiles),
        "anchor_adjacent_walkable": int(anchor_adjacent_walkable),
        "stateful_distance_to_path": (
            int(stateful_distance) if stateful_distance is not None else None
        ),
        "stateful_via_path_length": (
            int(stateful_via_path_length)
            if stateful_via_path_length is not None else None
        ),
        "stateful_branch_gain": float(stateful_branch_gain),
    }


def _evaluate_puzzle_candidate_interaction_geometry(
    pipeline,
    *,
    grid: np.ndarray,
    gate_family: str,
    source_anchor: Tuple[int, int],
    destination_anchor: Tuple[int, int],
    stateful_anchor: Optional[Tuple[int, int]],
    route_mask: np.ndarray,
    route_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate whether a candidate exposes a readable local interaction grammar.

    The constructive scaffold should not only add obstacles. For stateful
    gate families, the candidate should imply a concrete local action:

    - `switch` / `toggle`: a push-style interaction near the state anchor
    - `bombable`: a visible seam/barrier with a bypassed route
    - `item_unlock` / `key`: a gated alcove or pocket around the anchor

    This remains a local structural proxy, not a full causal simulator, but
    it is much closer to puzzle intent than generic block density.
    """
    gate_family = str(gate_family or "generic").strip().lower()
    interaction_required = gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"}
    walkable = pipeline._build_room_walkable_mask(grid)
    projected_stateful = (
        pipeline._nearest_walkable_room_coord(walkable, stateful_anchor)
        if stateful_anchor is not None
        else None
    )
    if not interaction_required:
        return {
            "valid": 1,
            "score": 0.0,
            "failure_reasons": [],
            "required": 0,
            "projected_stateful_anchor": (
                [int(projected_stateful[0]), int(projected_stateful[1])]
                if projected_stateful is not None
                else None
            ),
            "push_slot_count": 0,
            "targeted_push_slot_count": 0,
            "anchor_openings": 0,
            "local_block_tiles": 0,
            "barrier_axis_tiles": 0,
            "route_divergence": 0.0,
        }

    failure_reasons: List[str] = []
    if projected_stateful is None:
        return {
            "valid": 0,
            "score": -2.0,
            "failure_reasons": ["missing_stateful_anchor"],
            "required": 1,
            "projected_stateful_anchor": None,
            "push_slot_count": 0,
            "targeted_push_slot_count": 0,
            "anchor_openings": 0,
            "local_block_tiles": 0,
            "barrier_axis_tiles": 0,
            "route_divergence": 0.0,
        }

    anchor_r, anchor_c = pipeline._clamp_room_coord(projected_stateful)
    route_arr = np.asarray(route_mask, dtype=bool)
    block_like = np.isin(
        np.asarray(grid, dtype=np.int32),
        np.array([int(TileID.BLOCK), int(TileID.WALL)], dtype=np.int32),
    )
    flow_is_horizontal = abs(int(destination_anchor[1]) - int(source_anchor[1])) >= abs(
        int(destination_anchor[0]) - int(source_anchor[0])
    )

    anchor_openings = 0
    for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        next_r = anchor_r + d_r
        next_c = anchor_c + d_c
        if 0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH and bool(walkable[next_r, next_c]):
            anchor_openings += 1

    local_block_tiles = 0
    for row in range(max(1, anchor_r - 2), min(ROOM_HEIGHT - 1, anchor_r + 3)):
        for col in range(max(1, anchor_c - 2), min(ROOM_WIDTH - 1, anchor_c + 3)):
            if bool(block_like[row, col]):
                local_block_tiles += 1

    barrier_axis_tiles = 0
    if flow_is_horizontal:
        for row in range(max(1, anchor_r - 2), min(ROOM_HEIGHT - 1, anchor_r + 3)):
            if bool(block_like[row, anchor_c]):
                barrier_axis_tiles += 1
    else:
        for col in range(max(1, anchor_c - 2), min(ROOM_WIDTH - 1, anchor_c + 3)):
            if bool(block_like[anchor_r, col]):
                barrier_axis_tiles += 1

    push_slot_count = 0
    targeted_push_slot_count = 0
    seen_push_blocks: Set[Tuple[int, int]] = set()
    for row in range(max(1, anchor_r - 3), min(ROOM_HEIGHT - 1, anchor_r + 4)):
        for col in range(max(1, anchor_c - 3), min(ROOM_WIDTH - 1, anchor_c + 4)):
            if int(grid[row, col]) != int(TileID.BLOCK):
                continue
            block_has_valid_push = False
            block_targets_anchor = False
            for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                push_dest_r = row + d_r
                push_dest_c = col + d_c
                player_r = row - d_r
                player_c = col - d_c
                if not (
                    1 <= push_dest_r < ROOM_HEIGHT - 1
                    and 1 <= push_dest_c < ROOM_WIDTH - 1
                    and 1 <= player_r < ROOM_HEIGHT - 1
                    and 1 <= player_c < ROOM_WIDTH - 1
                ):
                    continue
                if not bool(walkable[push_dest_r, push_dest_c]):
                    continue
                if not bool(walkable[player_r, player_c]):
                    continue
                if (int(push_dest_r), int(push_dest_c)) == (int(anchor_r), int(anchor_c)):
                    if pipeline._shortest_room_path(
                        walkable,
                        source_anchor,
                        (int(player_r), int(player_c)),
                    ):
                        block_targets_anchor = True
                route_r0 = max(0, min(player_r, push_dest_r) - 1)
                route_r1 = min(ROOM_HEIGHT, max(player_r, push_dest_r) + 2)
                route_c0 = max(0, min(player_c, push_dest_c) - 1)
                route_c1 = min(ROOM_WIDTH, max(player_c, push_dest_c) + 2)
                if not bool(np.any(route_arr[route_r0:route_r1, route_c0:route_c1])):
                    continue
                block_has_valid_push = True
            if block_has_valid_push:
                seen_push_blocks.add((int(row), int(col)))
            if block_targets_anchor:
                targeted_push_slot_count += 1
    push_slot_count = int(len(seen_push_blocks))

    route_overlap_ratio = float(route_quality.get("route_overlap_ratio", 1.0) or 1.0)
    route_divergence = float(max(0.0, 1.0 - route_overlap_ratio))
    stateful_branch_gain = float(route_quality.get("stateful_branch_gain", 0.0) or 0.0)

    score = 0.0
    score += 0.25 * float(min(1.0, local_block_tiles / 5.0))
    score += 0.20 * float(min(1.0, barrier_axis_tiles / 3.0))

    if gate_family in {"switch", "toggle"}:
        score += 0.70 * float(min(1.0, push_slot_count / 1.0))
        score += 0.45 * float(min(1.0, targeted_push_slot_count / 1.0))
        if push_slot_count < 1:
            failure_reasons.append("missing_push_interaction")
        if targeted_push_slot_count < 1:
            failure_reasons.append("missing_targeted_push_interaction")
        if local_block_tiles < 3:
            failure_reasons.append("weak_interaction_geometry")
        if gate_family == "toggle":
            score += 0.20 * float(min(1.0, barrier_axis_tiles / 2.0))
    elif gate_family == "bombable":
        score += 0.70 * float(min(1.0, route_divergence / 0.20))
        if barrier_axis_tiles < 1 and local_block_tiles < 3:
            failure_reasons.append("weak_bomb_seam")
        if route_divergence < 0.10:
            failure_reasons.append("missing_bypass_divergence")
    elif gate_family in {"item_unlock", "key"}:
        if local_block_tiles < 4:
            failure_reasons.append("weak_alcove_frame")
        if gate_family == "key" and stateful_branch_gain < 1.0:
            failure_reasons.append("missing_key_detour")
        score += 0.45 * float(min(1.0, local_block_tiles / 6.0))
        score += 0.30 * float(max(0.0, 1.0 - (abs(anchor_openings - 1) / 2.0)))

    valid = int(len(failure_reasons) == 0)
    if not valid:
        score -= 1.75 + (0.10 * len(failure_reasons))

    return {
        "valid": int(valid),
        "score": float(score),
        "failure_reasons": list(failure_reasons),
        "required": 1,
        "projected_stateful_anchor": [int(anchor_r), int(anchor_c)],
        "push_slot_count": int(push_slot_count),
        "targeted_push_slot_count": int(targeted_push_slot_count),
        "anchor_openings": int(anchor_openings),
        "local_block_tiles": int(local_block_tiles),
        "barrier_axis_tiles": int(barrier_axis_tiles),
        "route_divergence": float(route_divergence),
    }


def _puzzle_descriptor_distance(
    pipeline,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    """Lightweight diversity distance for puzzle scaffold descriptors."""
    left_quadrants = list(left.get("quadrants", [0, 0, 0, 0]))
    right_quadrants = list(right.get("quadrants", [0, 0, 0, 0]))
    distance = 0.0
    distance += 1.0 if str(left.get("variant_name", "")) != str(right.get("variant_name", "")) else 0.0
    distance += 0.35 if str(left.get("gate_family", "")) != str(right.get("gate_family", "")) else 0.0
    distance += 0.20 if str(left.get("archetype", "")) != str(right.get("archetype", "")) else 0.0
    distance += abs(int(left.get("tiles_added", 0)) - int(right.get("tiles_added", 0))) / 24.0
    distance += abs(int(left.get("segments_added", 0)) - int(right.get("segments_added", 0))) / 6.0
    distance += abs(int(left.get("row_coverage", 0)) - int(right.get("row_coverage", 0))) / float(max(1, ROOM_HEIGHT - 2))
    distance += abs(int(left.get("col_coverage", 0)) - int(right.get("col_coverage", 0))) / float(max(1, ROOM_WIDTH - 2))
    distance += abs(float(left.get("center_row", ROOM_HEIGHT // 2)) - float(right.get("center_row", ROOM_HEIGHT // 2))) / float(max(1, ROOM_HEIGHT - 1))
    distance += abs(float(left.get("center_col", ROOM_WIDTH // 2)) - float(right.get("center_col", ROOM_WIDTH // 2))) / float(max(1, ROOM_WIDTH - 1))
    distance += sum(abs(int(a) - int(b)) for a, b in zip(left_quadrants, right_quadrants)) / 24.0
    return float(distance)


def _score_puzzle_candidate(
    pipeline,
    *,
    descriptor: Mapping[str, Any],
    stats: Mapping[str, Any],
    room_id: Any,
) -> float:
    """Score one scaffold candidate by structural quality plus novelty."""
    history = list(getattr(pipeline, "_puzzle_novelty_history", []) or [])
    novelty_weight = float(max(0.0, min(2.0, float(getattr(pipeline, "default_puzzle_room_novelty_weight", 0.45)))))
    memory_window = history[-8:]
    if memory_window:
        novelty_score = min(pipeline._puzzle_descriptor_distance(descriptor, prev) for prev in memory_window)
    else:
        novelty_score = 1.0

    same_variant_count = sum(
        1 for prev in memory_window if str(prev.get("variant_name", "")) == str(descriptor.get("variant_name", ""))
    )
    same_family_count = sum(
        1 for prev in memory_window if str(prev.get("gate_family", "")) == str(descriptor.get("gate_family", ""))
    )
    current_gate_family = str(descriptor.get("gate_family", "") or "")
    current_variant_name = str(descriptor.get("variant_name", "") or "")
    same_family_variants = {
        str(prev.get("variant_name", "") or "")
        for prev in memory_window
        if str(prev.get("gate_family", "") or "") == current_gate_family
    }
    repeat_family_variant = current_variant_name in same_family_variants
    unseen_family_variant_bonus = 0.35 if same_family_variants and not repeat_family_variant else 0.0
    block_budget = max(1, int(stats.get("profile_block_budget", 1)))
    structural_score = min(1.25, float(stats.get("tiles_added", 0)) / float(max(8, block_budget // 2)))
    structural_score += min(0.75, float(stats.get("segments_added", 0)) / 4.0)
    structural_score += min(0.35, float(stats.get("optional_segments_applied", 0)) / 3.0)
    structural_score += 0.15 * (
        float(descriptor.get("row_coverage", 0)) / float(max(1, ROOM_HEIGHT - 2))
        + float(descriptor.get("col_coverage", 0)) / float(max(1, ROOM_WIDTH - 2))
    )
    route_quality_score = float(stats.get("route_quality_score", 0.0) or 0.0)
    contract_score = float(stats.get("contract_score", 0.0) or 0.0)
    contract_valid = int(stats.get("contract_valid", 0) or 0)
    interaction_score = float(stats.get("interaction_score", 0.0) or 0.0)
    interaction_valid = int(stats.get("interaction_valid", 0) or 0)
    interaction_sequence_score = float(stats.get("interaction_sequence_score", 0.0) or 0.0)
    interaction_sequence_valid = int(stats.get("interaction_sequence_valid", 0) or 0)
    interaction_sequence_required = int(stats.get("interaction_sequence_required", 0) or 0)
    density_ratio = float(stats.get("tiles_added", 0)) / float(max(1, block_budget))
    optional_segments_applied = int(stats.get("optional_segments_applied", 0) or 0)
    route_overlap_ratio = float(stats.get("route_quality_overlap_ratio", 0.0) or 0.0)
    detour_gain = float(stats.get("route_quality_detour_gain", 0.0) or 0.0)
    stateful_branch_gain = float(stats.get("route_quality_stateful_branch_gain", 0.0) or 0.0)
    gate_family = str(descriptor.get("gate_family", "") or "")

    clutter_penalty = max(0.0, density_ratio - 0.55) * 1.5
    clutter_penalty += max(0, optional_segments_applied - 1) * 0.18
    if route_overlap_ratio > 0.90 and detour_gain < 1.0:
        clutter_penalty += 0.45
    if gate_family in {"switch", "toggle", "item_unlock", "key"} and stateful_branch_gain <= 0.0:
        clutter_penalty += 0.60

    tie_break = float(
        stable_seed_offset((room_id, str(descriptor.get("variant_name", ""))), modulo=1000)
    ) / 1000.0
    return float(
        structural_score
        + route_quality_score
        + contract_score
        + interaction_score
        + interaction_sequence_score
        + (min(0.25, novelty_weight) * novelty_score)
        + unseen_family_variant_bonus
        - (1.05 if repeat_family_variant else 0.0)
        - (0.30 * float(same_variant_count))
        - (0.10 * float(same_family_count))
        - clutter_penalty
        - (2.25 if contract_valid <= 0 else 0.0)
        - (2.50 if interaction_valid <= 0 and gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"} else 0.0)
        - (2.10 if interaction_sequence_required > 0 and interaction_sequence_valid <= 0 else 0.0)
        + (tie_break * 1e-3)
    )


def _commit_puzzle_novelty_choice(
    pipeline,
    *,
    room_id: Any,
    scaffold_stats: Mapping[str, Any],
) -> None:
    """Remember the selected scaffold descriptor once per room for later novelty scoring."""
    committed = getattr(pipeline, "_puzzle_novelty_committed", None)
    if not isinstance(committed, set):
        pipeline._puzzle_novelty_committed = set()
        committed = pipeline._puzzle_novelty_committed
    if room_id in committed:
        return
    descriptor = scaffold_stats.get("novelty_descriptor")
    if isinstance(descriptor, dict) and descriptor:
        history = getattr(pipeline, "_puzzle_novelty_history", None)
        if not isinstance(history, list):
            pipeline._puzzle_novelty_history = []
            history = pipeline._puzzle_novelty_history
        history.append(dict(descriptor))
    committed.add(room_id)


def _build_puzzle_room_segments(
    pipeline,
    *,
    archetype: str,
    gate_family: str,
    variant_spec: Optional[Mapping[str, Any]] = None,
    stateful_anchor: Optional[Tuple[int, int]],
    flow_is_horizontal: bool,
    puzzle_anchor: Tuple[int, int],
) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
    """
    Return required and optional puzzle scaffold segments.

    Each segment is a list of room coordinates that will be painted as BLOCK
    tiles when they do not collide with reserved route/anchor cells.
    """
    center_r = max(3, min(ROOM_HEIGHT - 4, int(puzzle_anchor[0])))
    center_c = max(3, min(ROOM_WIDTH - 4, int(puzzle_anchor[1])))
    if stateful_anchor is not None:
        stateful_r = max(3, min(ROOM_HEIGHT - 4, int(stateful_anchor[0])))
        stateful_c = max(3, min(ROOM_WIDTH - 4, int(stateful_anchor[1])))
    else:
        stateful_r = center_r
        stateful_c = center_c
    left_col = 3
    right_col = ROOM_WIDTH - 4
    top_row = 3
    bottom_row = ROOM_HEIGHT - 4
    resource_offset = int(max(1, getattr(pipeline, "default_puzzle_room_resource_bypass_offset", 2)))
    key_depth = int(max(1, getattr(pipeline, "default_puzzle_room_key_pocket_depth", 3)))
    toggle_offset = int(max(1, getattr(pipeline, "default_puzzle_room_toggle_corridor_offset", 2)))
    variant = dict(variant_spec or {})
    variant_style = str(variant.get("style", "baseline") or "baseline").strip().lower()
    variant_side_bias = int(max(-1, min(1, int(variant.get("side_bias", 0) or 0))))

    required: List[List[Tuple[int, int]]] = []
    optional: List[List[Tuple[int, int]]] = []

    if archetype == "gate":
        if flow_is_horizontal:
            gate_col = max(left_col + 1, min(right_col - 1, center_c))
            if gate_family == "bombable":
                bypass_row = max(
                    2,
                    min(
                        ROOM_HEIGHT - 3,
                        stateful_r + (-resource_offset if stateful_r > ROOM_HEIGHT // 2 else resource_offset),
                    ),
                )
                gap_rows = {bypass_row}
            elif gate_family == "toggle":
                gap_rows = {stateful_r}
            elif gate_family == "key" and stateful_anchor is not None:
                gap_rows = {center_r}
            else:
                gap_rows = {max(2, center_r - 1), center_r, min(ROOM_HEIGHT - 3, center_r + 1)}
            required.append([(row, gate_col) for row in range(2, ROOM_HEIGHT - 2) if row not in gap_rows])
            pocket_side = variant_side_bias if variant_side_bias != 0 else (-1 if center_r <= ROOM_HEIGHT // 2 else 1)
            pocket_row = max(2, min(ROOM_HEIGHT - 3, center_r + pocket_side * 2))
            if gate_family == "switch":
                required.append([(stateful_r, col) for col in range(max(2, gate_col - 3), min(ROOM_WIDTH - 2, gate_col + 1))])
                required.append([(row, max(2, gate_col - 3)) for row in range(min(center_r, stateful_r), max(center_r, stateful_r) + 1)])
                if variant_style == "bridge":
                    required.append([(row, min(ROOM_WIDTH - 3, gate_col + 2)) for row in range(3, ROOM_HEIGHT - 3) if abs(row - center_r) > 0])
                optional.append([(row, min(ROOM_WIDTH - 3, gate_col + 2)) for row in range(3, ROOM_HEIGHT - 3) if abs(row - center_r) > 1])
            elif gate_family == "toggle":
                corridor_top = max(2, stateful_r - toggle_offset)
                corridor_bottom = min(ROOM_HEIGHT - 3, stateful_r + toggle_offset)
                required.append([(corridor_top, col) for col in range(max(2, gate_col - 3), min(ROOM_WIDTH - 2, gate_col + 2))])
                required.append([(corridor_bottom, col) for col in range(max(2, gate_col - 3), min(ROOM_WIDTH - 2, gate_col + 2))])
                if variant_style == "weave":
                    required.append([(row, max(2, gate_col - 1)) for row in range(corridor_top + 1, corridor_bottom)])
                optional.append([(row, max(2, gate_col - 2)) for row in range(corridor_top + 1, corridor_bottom)])
            elif gate_family == "bombable":
                bypass_row = max(
                    2,
                    min(
                        ROOM_HEIGHT - 3,
                        stateful_r + (-resource_offset if stateful_r > ROOM_HEIGHT // 2 else resource_offset),
                    ),
                )
                resource_row = stateful_r
                required.append([(resource_row, col) for col in range(max(2, gate_col - 4), max(3, gate_col - 1))])
                required.append([(row, max(2, gate_col - 4)) for row in range(min(resource_row, bypass_row), max(resource_row, bypass_row) + 1)])
                if variant_style == "wrap":
                    required.append([(bypass_row, col) for col in range(max(2, gate_col - 1), min(ROOM_WIDTH - 2, gate_col + 4)) if col != gate_col])
                optional.append([(row, max(2, gate_col - 2)) for row in range(min(resource_row, bypass_row), max(resource_row, bypass_row) + 1)])
                optional.append([(bypass_row, col) for col in range(min(gate_col + 1, ROOM_WIDTH - 3), min(ROOM_WIDTH - 2, gate_col + 4))])
            elif gate_family == "item_unlock":
                item_row = stateful_r
                item_col = max(gate_col + 2, min(ROOM_WIDTH - 3, stateful_c))
                left_col = max(2, item_col - 1)
                right_col = min(ROOM_WIDTH - 3, item_col + 1)
                top_row = max(2, item_row - 1)
                bottom_row = min(ROOM_HEIGHT - 3, item_row + 1)
                required.append([(row, left_col) for row in range(top_row, bottom_row + 1) if row != item_row])
                required.append([(row, right_col) for row in range(top_row, bottom_row + 1) if row != item_row])
                required.append([(top_row, col) for col in range(left_col, right_col + 1) if col != item_col])
                required.append([(bottom_row, col) for col in range(left_col, right_col + 1) if col != item_col])
                if variant_style == "ring":
                    required.append([(item_row, left_col)])
                    required.append([(item_row, right_col)])
                optional.append([(center_r - 1, col) for col in range(min(gate_col + 1, item_col), max(gate_col + 1, item_col) + 1)])
                optional.append([(center_r + 1, col) for col in range(min(gate_col + 1, item_col), max(gate_col + 1, item_col) + 1)])
            elif gate_family == "key" and stateful_anchor is not None:
                key_row = stateful_r
                required.append([(key_row, col) for col in range(max(2, gate_col - key_depth - 1), max(3, gate_col - 1))])
                required.append([(row, max(2, gate_col - key_depth - 1)) for row in range(min(center_r, key_row), max(center_r, key_row) + 1)])
                if variant_style == "split":
                    optional.append([(row, max(2, gate_col - key_depth + 1)) for row in range(min(center_r, key_row), max(center_r, key_row) + 1)])
                optional.append([(row, min(ROOM_WIDTH - 3, gate_col + 2)) for row in range(3, ROOM_HEIGHT - 3) if abs(row - center_r) > 1])
            else:
                required.append([(pocket_row, col) for col in range(max(2, gate_col - 2), min(ROOM_WIDTH - 2, gate_col + 1))])
                optional.append([(row, max(2, gate_col - 2)) for row in range(min(center_r, pocket_row), max(center_r, pocket_row) + 1)])
                optional.append([(row, min(ROOM_WIDTH - 3, gate_col + 2)) for row in range(3, ROOM_HEIGHT - 3) if abs(row - center_r) > 1])
        else:
            gate_row = max(top_row + 1, min(bottom_row - 1, center_r))
            if gate_family == "bombable":
                bypass_col = max(
                    2,
                    min(
                        ROOM_WIDTH - 3,
                        stateful_c + (-resource_offset if stateful_c > ROOM_WIDTH // 2 else resource_offset),
                    ),
                )
                gap_cols = {bypass_col}
            elif gate_family == "toggle":
                gap_cols = {stateful_c}
            elif gate_family == "key" and stateful_anchor is not None:
                gap_cols = {center_c}
            else:
                gap_cols = {max(2, center_c - 1), center_c, min(ROOM_WIDTH - 3, center_c + 1)}
            required.append([(gate_row, col) for col in range(2, ROOM_WIDTH - 2) if col not in gap_cols])
            pocket_side = variant_side_bias if variant_side_bias != 0 else (-1 if center_c <= ROOM_WIDTH // 2 else 1)
            pocket_col = max(2, min(ROOM_WIDTH - 3, center_c + pocket_side * 2))
            if gate_family == "switch":
                required.append([(row, stateful_c) for row in range(max(2, gate_row - 3), min(ROOM_HEIGHT - 2, gate_row + 1))])
                required.append([(max(2, gate_row - 3), col) for col in range(min(center_c, stateful_c), max(center_c, stateful_c) + 1)])
                if variant_style == "bridge":
                    required.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(3, ROOM_WIDTH - 3) if abs(col - center_c) > 0])
                optional.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(3, ROOM_WIDTH - 3) if abs(col - center_c) > 1])
            elif gate_family == "toggle":
                corridor_left = max(2, stateful_c - toggle_offset)
                corridor_right = min(ROOM_WIDTH - 3, stateful_c + toggle_offset)
                required.append([(row, corridor_left) for row in range(max(2, gate_row - 3), min(ROOM_HEIGHT - 2, gate_row + 2))])
                required.append([(row, corridor_right) for row in range(max(2, gate_row - 3), min(ROOM_HEIGHT - 2, gate_row + 2))])
                if variant_style == "weave":
                    required.append([(max(2, gate_row - 1), col) for col in range(corridor_left + 1, corridor_right)])
                optional.append([(max(2, gate_row - 2), col) for col in range(corridor_left + 1, corridor_right)])
            elif gate_family == "bombable":
                bypass_col = max(
                    2,
                    min(
                        ROOM_WIDTH - 3,
                        stateful_c + (-resource_offset if stateful_c > ROOM_WIDTH // 2 else resource_offset),
                    ),
                )
                resource_col = stateful_c
                ledge_row = max(2, min(ROOM_HEIGHT - 3, gate_row - 2))
                required.append([(row, resource_col) for row in range(max(2, gate_row - 4), max(3, gate_row - 1))])
                required.append([(max(2, gate_row - 4), col) for col in range(min(resource_col, bypass_col), max(resource_col, bypass_col) + 1)])
                required.append([(ledge_row, col) for col in range(min(resource_col, bypass_col), max(resource_col, bypass_col) + 1)])
                if variant_style == "wrap":
                    required.append([(row, bypass_col) for row in range(max(2, gate_row - 1), min(ROOM_HEIGHT - 2, gate_row + 4)) if row != gate_row])
                    required.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(min(resource_col, bypass_col), max(resource_col, bypass_col) + 1)])
                optional.append([(max(2, gate_row - 2), col) for col in range(min(resource_col, bypass_col), max(resource_col, bypass_col) + 1)])
                optional.append([(row, bypass_col) for row in range(min(gate_row + 1, ROOM_HEIGHT - 3), min(ROOM_HEIGHT - 2, gate_row + 4))])
            elif gate_family == "item_unlock":
                item_row = max(gate_row + 2, min(ROOM_HEIGHT - 3, stateful_r))
                item_col = stateful_c
                top_row = max(2, item_row - 1)
                bottom_row = min(ROOM_HEIGHT - 3, item_row + 1)
                left_col = max(2, item_col - 1)
                right_col = min(ROOM_WIDTH - 3, item_col + 1)
                required.append([(row, left_col) for row in range(top_row, bottom_row + 1) if row != item_row])
                required.append([(row, right_col) for row in range(top_row, bottom_row + 1) if row != item_row])
                required.append([(top_row, col) for col in range(left_col, right_col + 1) if col != item_col])
                required.append([(bottom_row, col) for col in range(left_col, right_col + 1) if col != item_col])
                if variant_style == "ring":
                    required.append([(top_row, item_col)])
                    required.append([(bottom_row, item_col)])
                optional.append([(row, center_c - 1) for row in range(min(gate_row + 1, item_row), max(gate_row + 1, item_row) + 1)])
                optional.append([(row, center_c + 1) for row in range(min(gate_row + 1, item_row), max(gate_row + 1, item_row) + 1)])
            elif gate_family == "key" and stateful_anchor is not None:
                key_col = stateful_c
                required.append([(row, key_col) for row in range(max(2, gate_row - key_depth - 1), max(3, gate_row - 1))])
                required.append([(max(2, gate_row - key_depth - 1), col) for col in range(min(center_c, key_col), max(center_c, key_col) + 1)])
                if variant_style == "split":
                    optional.append([(max(2, gate_row - key_depth + 1), col) for col in range(min(center_c, key_col), max(center_c, key_col) + 1)])
                optional.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(3, ROOM_WIDTH - 3) if abs(col - center_c) > 1])
            else:
                required.append([(row, pocket_col) for row in range(max(2, gate_row - 2), min(ROOM_HEIGHT - 2, gate_row + 1))])
                optional.append([(max(2, gate_row - 2), col) for col in range(min(center_c, pocket_col), max(center_c, pocket_col) + 1)])
                optional.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(3, ROOM_WIDTH - 3) if abs(col - center_c) > 1])
    elif archetype == "hub":
        top = max(2, center_r - 3)
        bottom = min(ROOM_HEIGHT - 3, center_r + 3)
        left = max(2, center_c - 3)
        right = min(ROOM_WIDTH - 3, center_c + 3)
        required.append([(top, col) for col in range(left, right + 1) if abs(col - center_c) > 1])
        required.append([(bottom, col) for col in range(left, right + 1) if abs(col - center_c) > 1])
        required.append([(row, left) for row in range(top, bottom + 1) if abs(row - center_r) > 1])
        required.append([(row, right) for row in range(top, bottom + 1) if abs(row - center_r) > 1])
        if variant_style == "cross":
            required.append([(center_r, col) for col in range(left + 1, right) if abs(col - center_c) > 0])
            required.append([(row, center_c) for row in range(top + 1, bottom) if abs(row - center_r) > 0])
        elif variant_style == "offset":
            offset_row = max(2, min(ROOM_HEIGHT - 3, center_r + variant_side_bias * 2))
            required.append([(offset_row, col) for col in range(left + 1, right) if abs(col - center_c) > 1])
        optional.append([(center_r - 2, col) for col in range(left + 1, center_c - 1)])
        optional.append([(center_r - 2, col) for col in range(center_c + 2, right)])
        optional.append([(center_r + 2, col) for col in range(left + 1, center_c - 1)])
        optional.append([(center_r + 2, col) for col in range(center_c + 2, right)])
    elif archetype == "island":
        if variant_style == "staggered":
            optional.extend(
                [
                    [(center_r - 3, center_c - 2), (center_r - 3, center_c - 1), (center_r - 2, center_c - 2)],
                    [(center_r - 1, center_c + 1), (center_r, center_c + 1), (center_r, center_c + 2)],
                    [(center_r + 2, center_c - 1), (center_r + 3, center_c - 2), (center_r + 3, center_c - 1)],
                    [(center_r + 1, center_c + 2), (center_r + 2, center_c + 1), (center_r + 2, center_c + 2)],
                ]
            )
        else:
            optional.extend(
                [
                    [(center_r - 3, center_c - 2), (center_r - 3, center_c - 1), (center_r - 2, center_c - 2), (center_r - 2, center_c - 1)],
                    [(center_r - 1, center_c + 1), (center_r - 1, center_c + 2), (center_r, center_c + 1), (center_r, center_c + 2)],
                    [(center_r + 2, center_c - 2), (center_r + 2, center_c - 1), (center_r + 3, center_c - 2), (center_r + 3, center_c - 1)],
                    [(center_r + 1, center_c + 1), (center_r + 1, center_c + 2), (center_r + 2, center_c + 1), (center_r + 2, center_c + 2)],
                ]
            )
        if variant_style == "bridge":
            optional.append([(center_r + variant_side_bias * 2, col) for col in range(left_col + 1, right_col) if abs(col - center_c) > 1])
        if flow_is_horizontal:
            required.append([(center_r, col) for col in range(left_col + 1, right_col) if abs(col - center_c) > 2])
        else:
            required.append([(row, center_c) for row in range(top_row + 1, bottom_row) if abs(row - center_r) > 2])
    elif archetype == "combat":
        if variant_style == "corners":
            optional.extend(
                [
                    [(center_r - 3, center_c - 2), (center_r - 3, center_c - 1), (center_r - 2, center_c - 2), (center_r - 2, center_c - 1)],
                    [(center_r - 3, center_c + 1), (center_r - 3, center_c + 2), (center_r - 2, center_c + 1), (center_r - 2, center_c + 2)],
                    [(center_r + 2, center_c - 2), (center_r + 2, center_c - 1), (center_r + 3, center_c - 2), (center_r + 3, center_c - 1)],
                    [(center_r + 2, center_c + 1), (center_r + 2, center_c + 2), (center_r + 3, center_c + 1), (center_r + 3, center_c + 2)],
                ]
            )
        else:
            optional.extend(
                [
                    [(center_r - 2, center_c - 1), (center_r - 2, center_c), (center_r - 1, center_c - 1)],
                    [(center_r - 2, center_c + 1), (center_r - 2, center_c + 2), (center_r - 1, center_c + 2)],
                    [(center_r + 1, center_c - 1), (center_r + 2, center_c - 1), (center_r + 2, center_c)],
                    [(center_r + 1, center_c + 2), (center_r + 2, center_c + 1), (center_r + 2, center_c + 2)],
                ]
            )
        if variant_style == "cross":
            required.append([(center_r - 3, center_c), (center_r - 2, center_c)])
            required.append([(center_r + 2, center_c), (center_r + 3, center_c)])
        required.append([(center_r, center_c - 3), (center_r, center_c - 2)])
        required.append([(center_r, center_c + 2), (center_r, center_c + 3)])
    else:  # serpentine
        if flow_is_horizontal:
            rows = [3, 6, 9, 12] if variant_style != "split" else [4, 8, 12]
            for idx, row in enumerate(rows):
                if row >= ROOM_HEIGHT - 2:
                    continue
                gap_on_left = ((idx % 2) == 0) if variant_style != "mirror" else ((idx % 2) != 0)
                segment = [
                    (row, col)
                    for col in range(2, ROOM_WIDTH - 2)
                    if not (2 <= col <= 4 and gap_on_left)
                    and not (ROOM_WIDTH - 5 <= col <= ROOM_WIDTH - 3 and not gap_on_left)
                ]
                if idx < 2:
                    required.append(segment)
                else:
                    optional.append(segment)
            optional.append([(center_r + variant_side_bias, center_c - 1), (center_r + variant_side_bias, center_c + 1)])
        else:
            cols = [3, 5, 7] if variant_style != "split" else [3, 6, 8]
            for idx, col in enumerate(cols):
                if col >= ROOM_WIDTH - 2:
                    continue
                gap_on_top = ((idx % 2) == 0) if variant_style != "mirror" else ((idx % 2) != 0)
                segment = [
                    (row, col)
                    for row in range(2, ROOM_HEIGHT - 2)
                    if not (2 <= row <= 4 and gap_on_top)
                    and not (ROOM_HEIGHT - 5 <= row <= ROOM_HEIGHT - 3 and not gap_on_top)
                ]
                if idx < 2:
                    required.append(segment)
                else:
                    optional.append(segment)
            optional.append([(center_r - 1, center_c + variant_side_bias), (center_r + 1, center_c + variant_side_bias)])

    return required, optional


def _strip_small_interior_structure_components(
    pipeline,
    grid: np.ndarray,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
    max_component_tiles: int = 6,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Remove tiny isolated interior wall/block islands that read as noise."""
    out = np.asarray(grid, dtype=np.int32).copy()
    wall_like_mask = np.isin(out, np.array([int(TileID.WALL), int(TileID.BLOCK)], dtype=np.int32))
    allowed_door_mask = pipeline._required_room_door_slots_mask(graph=graph, room_id=room_id)
    visited = np.zeros_like(wall_like_mask, dtype=bool)
    removed_components = 0
    removed_tiles = 0

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

            if touches_boundary or touches_allowed_door or len(component) > int(max_component_tiles):
                continue
            for comp_r, comp_c in component:
                out[comp_r, comp_c] = int(TileID.FLOOR)
            removed_components += 1
            removed_tiles += len(component)

    return out, {
        "removed_components": int(removed_components),
        "removed_tiles": int(removed_tiles),
    }


def _apply_puzzle_room_scaffold(
    pipeline,
    grid: np.ndarray,
    *,
    graph: Optional[nx.Graph],
    room_id: Any,
    room_plan_mask: Optional[np.ndarray] = None,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Inject a lightweight constructive scaffold into under-structured puzzle rooms.

    Research-backed room-generation systems tend to keep structure explicit for
    constrained layouts instead of expecting small-data generators to discover
    reliable puzzle geometry on their own. We follow the same pragmatic
    hybrid approach here: preserve the planned traversability route, then add a
    small deterministic block-maze scaffold only when the current room is still
    overly empty.
    """
    out = np.asarray(grid, dtype=np.int32).copy()
    stats: Dict[str, Any] = {
        "applied": 0,
        "tiles_added": 0,
        "segments_added": 0,
        "existing_structure_tiles": 0,
        "planned_route_pixels": 0,
    }
    if not bool(pipeline.default_puzzle_room_scaffold_enabled):
        return out, stats
    if not isinstance(graph, nx.Graph) or room_id not in graph:
        return out, stats

    attrs = dict(graph.nodes[room_id])
    role_flags = pipeline._room_role_flags(attrs)
    semantics = pipeline._extract_room_topology_semantics(graph, room_id)
    has_puzzle_gate = any(
        {"switch", "switch_locked", "state_block", "on_off_gate", "puzzle"} & set(tokens)
        for tokens in semantics["edge_constraints"].values()
    )
    node_type = str(
        attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or ""
    ).strip().lower()
    if not (
        role_flags.get("has_puzzle", False)
        or has_puzzle_gate
        or node_type in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"}
    ):
        return out, stats

    out, structure_cleanup = pipeline._strip_small_interior_structure_components(
        out,
        graph=graph,
        room_id=room_id,
    )
    stats["noise_components_removed"] = int(structure_cleanup["removed_components"])
    stats["noise_tiles_removed"] = int(structure_cleanup["removed_tiles"])

    block_id = int(TileID.BLOCK)
    wall_id = int(TileID.WALL)
    floor_id = int(TileID.FLOOR)
    structure_mask = np.isin(out, np.array([wall_id, block_id], dtype=np.int32))
    interior_mask = np.zeros_like(out, dtype=bool)
    interior_mask[2:ROOM_HEIGHT - 2, 2:ROOM_WIDTH - 2] = True
    existing_structure_tiles = int(np.sum(structure_mask & interior_mask))
    stats["existing_structure_tiles"] = existing_structure_tiles
    if (
        existing_structure_tiles >= int(pipeline.default_puzzle_room_scaffold_min_structure_tiles)
        and int(structure_cleanup["removed_components"]) <= 0
    ):
        return out, stats

    normalized_start_goal = start_goal
    if normalized_start_goal is None:
        normalized_start_goal = pipeline._extract_room_start_goal(graph, room_id)
    if normalized_start_goal is None:
        normalized_start_goal = ((ROOM_HEIGHT // 2, 1), (ROOM_HEIGHT // 2, ROOM_WIDTH - 2))
    start_coord, goal_coord = pipeline._normalize_start_goal_coords(normalized_start_goal)

    if isinstance(room_plan_mask, np.ndarray) and room_plan_mask.shape == (ROOM_HEIGHT, ROOM_WIDTH):
        route_mask = np.asarray(room_plan_mask, dtype=np.float32) > 0.0
    else:
        try:
            route_mask = pipeline._build_room_plan_trace(
                graph,
                room_id,
                out,
                start_goal=(start_coord, goal_coord),
            ) > 0.0
        except Exception:
            route_mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    stats["planned_route_pixels"] = int(np.sum(route_mask))
    scaffold_profile = pipeline._resolve_puzzle_room_scaffold_profile(
        attrs=attrs,
        role_flags=role_flags,
        semantics=semantics,
        node_type=node_type,
    )
    preserve_margin = int(max(0, scaffold_profile.get("preserve_route_margin", getattr(pipeline, "default_puzzle_room_preserve_route_margin", 0))))

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

    planned_markers = pipeline._plan_room_graph_marker_layout(
        out,
        graph=graph,
        room_id=room_id,
        start_goal=(start_coord, goal_coord),
    )

    puzzle_anchor = semantic_anchors.get("puzzle", (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
    source_anchor = semantic_anchors.get("start", start_coord)
    destination_anchor = semantic_anchors.get("goal", goal_coord)
    flow_is_horizontal = abs(int(destination_anchor[1]) - int(source_anchor[1])) >= abs(
        int(destination_anchor[0]) - int(source_anchor[0])
    )
    archetype = str(scaffold_profile.get("archetype", "serpentine") or "serpentine").strip().lower()
    gate_family = str(scaffold_profile.get("gate_family", "generic") or "generic").strip().lower()
    if gate_family == "switch":
        stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
    elif gate_family == "toggle":
        stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
    elif gate_family == "bombable":
        stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
    elif gate_family == "item_unlock":
        stateful_anchor_name = "item" if "item" in semantic_anchors else ("puzzle" if "puzzle" in semantic_anchors else None)
    elif gate_family == "key":
        stateful_anchor_name = "key" if "key" in semantic_anchors else None
    elif gate_family == "combat":
        stateful_anchor_name = "enemy" if "enemy" in semantic_anchors else None
    else:
        stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
    stateful_anchor = semantic_anchors.get(stateful_anchor_name) if stateful_anchor_name is not None else None
    interaction_sequence = pipeline._resolve_puzzle_interaction_sequence(
        archetype=archetype,
        gate_family=gate_family,
        role_flags=role_flags,
        semantic_anchors=semantic_anchors,
    )
    variant_specs = pipeline._build_puzzle_room_variant_specs(
        archetype=archetype,
        gate_family=gate_family,
    )
    variant_cache = getattr(pipeline, "_puzzle_variant_cache", None)
    cached_variant = variant_cache.get(room_id) if isinstance(variant_cache, dict) else None
    base_grid = out.copy()
    baseline_path_metrics = pipeline._evaluate_puzzle_candidate_route_quality(
        grid=base_grid,
        source_anchor=source_anchor,
        destination_anchor=destination_anchor,
        stateful_anchor=stateful_anchor,
        route_mask=route_mask,
        gate_family=gate_family,
        baseline_path_length=None,
    )
    baseline_path_length = (
        int(baseline_path_metrics.get("path_length", 0))
        if int(baseline_path_metrics.get("path_exists", 0) or 0) > 0
        else None
    )

    def _mark_reserved(route_mask_candidate: np.ndarray) -> np.ndarray:
        reserved_candidate = (
            pipeline._dilate_room_mask(route_mask_candidate, radius=preserve_margin)
            if preserve_margin > 0 else route_mask_candidate.copy()
        )
        for point in semantic_anchors.values():
            rr, cc = pipeline._clamp_room_coord(point)
            reserved_candidate[int(rr), int(cc)] = True
        for _tile_id, slot in planned_markers:
            rr, cc = pipeline._clamp_room_coord(slot)
            reserved_candidate[int(rr), int(cc)] = True
        for direction, enabled in semantics["required_doors"].items():
            if not bool(enabled):
                continue
            spec = DOOR_POSITIONS[str(direction)]
            if direction in {"N", "S"}:
                apron_row = 2 if direction == "N" else ROOM_HEIGHT - 3
                c0 = int(spec["col_start"])
                c1 = int(spec["col_end"]) + 1
                reserved_candidate[apron_row, c0:c1] = True
            else:
                apron_col = 2 if direction == "W" else ROOM_WIDTH - 3
                r0 = int(spec["row_start"])
                r1 = int(spec["row_end"]) + 1
                reserved_candidate[r0:r1, apron_col] = True
        return reserved_candidate

    def _render_candidate(variant_spec: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        candidate_grid = base_grid.copy()
        route_candidate = pipeline._build_puzzle_room_route_template(
            archetype=archetype,
            gate_family=gate_family,
            variant_spec=variant_spec,
            stateful_anchor=stateful_anchor,
            interaction_sequence=interaction_sequence,
            flow_is_horizontal=flow_is_horizontal,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            puzzle_anchor=puzzle_anchor,
            role_flags=role_flags,
            semantics=semantics,
        )
        if bool(np.any(route_candidate)):
            route_mask_candidate = route_candidate
            route_template_used = 1
        else:
            route_mask_candidate = route_mask.copy()
            route_template_used = 0
        reserved_candidate = _mark_reserved(route_mask_candidate)

        branch_density = float(
            max(
                0.0,
                min(
                    1.0,
                    float(scaffold_profile.get("branch_density", getattr(pipeline, "default_puzzle_room_branch_density", 0.75)))
                    + float(variant_spec.get("branch_density_delta", 0.0)),
                ),
            )
        )
        block_budget = int(
            max(
                0,
                int(scaffold_profile.get("block_budget", getattr(pipeline, "default_puzzle_room_block_budget", 28)))
                + int(variant_spec.get("block_budget_delta", 0)),
            )
        )
        budget_remaining = int(block_budget)

        def _can_place(row: int, col: int) -> bool:
            if not (2 <= int(row) <= ROOM_HEIGHT - 3 and 2 <= int(col) <= ROOM_WIDTH - 3):
                return False
            if bool(reserved_candidate[int(row), int(col)]):
                return False
            return int(candidate_grid[int(row), int(col)]) == floor_id

        def _paint_block_line(points: List[Tuple[int, int]]) -> int:
            nonlocal budget_remaining
            added = 0
            for row, col in points:
                if budget_remaining <= 0:
                    break
                if _can_place(int(row), int(col)):
                    candidate_grid[int(row), int(col)] = block_id
                    added += 1
                    budget_remaining -= 1
            return int(added)

        def _apply_anchor_template(
            *,
            anchor: Optional[Tuple[int, int]],
            open_toward: Tuple[int, int],
        ) -> Tuple[int, int]:
            """
            Force a small readable puzzle pocket around the interaction anchor.

            This is the local grammar layer: before generic stochastic
            scaffold lines are added, stabilize the interaction site as a
            floor pocket framed by blocks with one intentional opening.
            """
            if anchor is None:
                return 0, 0
            anchor_r, anchor_c = pipeline._clamp_room_coord(anchor)
            open_r, open_c = pipeline._clamp_room_coord(open_toward)

            pocket_tiles_forced = 0
            frame_tiles_added = 0
            row_start = max(2, int(anchor_r) - 1)
            row_end = min(ROOM_HEIGHT - 3, int(anchor_r) + 1)
            col_start = max(2, int(anchor_c) - 1)
            col_end = min(ROOM_WIDTH - 3, int(anchor_c) + 1)

            for row in range(row_start, row_end + 1):
                for col in range(col_start, col_end + 1):
                    if int(candidate_grid[row, col]) != floor_id:
                        candidate_grid[row, col] = floor_id
                        pocket_tiles_forced += 1
                    reserved_candidate[row, col] = True

            delta_r = int(open_r) - int(anchor_r)
            delta_c = int(open_c) - int(anchor_c)
            if abs(delta_c) >= abs(delta_r):
                opening_side = "E" if delta_c >= 0 else "W"
            else:
                opening_side = "S" if delta_r >= 0 else "N"

            frame_cells: List[Tuple[int, int]] = []
            for row in range(row_start - 1, row_end + 2):
                for col in range(col_start - 1, col_end + 2):
                    if not (2 <= row <= ROOM_HEIGHT - 3 and 2 <= col <= ROOM_WIDTH - 3):
                        continue
                    on_border = row in {row_start - 1, row_end + 1} or col in {col_start - 1, col_end + 1}
                    if not on_border:
                        continue
                    if opening_side == "N" and row == row_start - 1 and col == int(anchor_c):
                        continue
                    if opening_side == "S" and row == row_end + 1 and col == int(anchor_c):
                        continue
                    if opening_side == "W" and col == col_start - 1 and row == int(anchor_r):
                        continue
                    if opening_side == "E" and col == col_end + 1 and row == int(anchor_r):
                        continue
                    frame_cells.append((int(row), int(col)))

            frame_tiles_added += _paint_block_line(frame_cells)
            return int(pocket_tiles_forced), int(frame_tiles_added)

        def _place_push_block_prop() -> int:
            """
            Add one isolated BLOCK near the interaction zone.

            The scaffold already builds blocking structure, but dense bars
            often read as generic noise instead of an intentional pushable
            block puzzle. This helper adds a single readable BLOCK prop near
            the planned route when the local geometry supports a valid push
            interaction.
            """
            nonlocal budget_remaining
            if budget_remaining <= 0:
                return 0
            if gate_family not in {"switch", "toggle", "key", "item_unlock", "generic"} and archetype not in {
                "hub",
                "serpentine",
            }:
                return 0

            anchor_r, anchor_c = (
                pipeline._clamp_room_coord(stateful_anchor)
                if stateful_anchor is not None
                else pipeline._clamp_room_coord(puzzle_anchor)
            )
            puzzle_r, puzzle_c = pipeline._clamp_room_coord(puzzle_anchor)

            # A staged switch only completes when a block reaches the puzzle
            # marker. Prefer a one-push witness for that exact destination
            # over an arbitrary movable block somewhere else in the room.
            if gate_family in {"switch", "toggle"}:
                for (block_r, block_c), (player_r, player_c), (target_r, target_c) in _switch_push_witnesses(
                    target_anchor=(puzzle_r, puzzle_c),
                    source_anchor=source_anchor,
                ):
                    if (
                        int(candidate_grid[target_r, target_c])
                        not in {floor_id, int(TileID.PUZZLE)}
                        or int(candidate_grid[block_r, block_c]) != floor_id
                        or int(candidate_grid[player_r, player_c]) != floor_id
                    ):
                        continue
                    candidate_grid[block_r, block_c] = block_id
                    budget_remaining -= 1
                    return 1

            candidate_slots: List[Tuple[int, int]] = []
            if flow_is_horizontal:
                candidate_slots.extend(
                    [
                        (anchor_r - 1, anchor_c - 2),
                        (anchor_r + 1, anchor_c - 2),
                        (anchor_r - 1, anchor_c + 2),
                        (anchor_r + 1, anchor_c + 2),
                        (puzzle_r - 2, puzzle_c - 1),
                        (puzzle_r + 2, puzzle_c + 1),
                    ]
                )
            else:
                candidate_slots.extend(
                    [
                        (anchor_r - 2, anchor_c - 1),
                        (anchor_r - 2, anchor_c + 1),
                        (anchor_r + 2, anchor_c - 1),
                        (anchor_r + 2, anchor_c + 1),
                        (puzzle_r - 1, puzzle_c - 2),
                        (puzzle_r + 1, puzzle_c + 2),
                    ]
                )

            seen_slots: Set[Tuple[int, int]] = set()
            for raw_row, raw_col in candidate_slots:
                row, col = pipeline._clamp_room_coord((raw_row, raw_col))
                slot = (int(row), int(col))
                if slot in seen_slots:
                    continue
                seen_slots.add(slot)
                if not _can_place(int(row), int(col)):
                    continue
                if bool(route_mask_candidate[int(row), int(col)]):
                    continue

                route_r0 = max(0, int(row) - 1)
                route_r1 = min(ROOM_HEIGHT, int(row) + 2)
                route_c0 = max(0, int(col) - 1)
                route_c1 = min(ROOM_WIDTH, int(col) + 2)
                if not bool(np.any(route_mask_candidate[route_r0:route_r1, route_c0:route_c1])):
                    continue

                for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    push_dest_r = int(row) + d_r
                    push_dest_c = int(col) + d_c
                    player_r = int(row) - d_r
                    player_c = int(col) - d_c
                    if not (
                        1 <= push_dest_r < ROOM_HEIGHT - 1
                        and 1 <= push_dest_c < ROOM_WIDTH - 1
                        and 1 <= player_r < ROOM_HEIGHT - 1
                        and 1 <= player_c < ROOM_WIDTH - 1
                    ):
                        continue
                    if bool(reserved_candidate[player_r, player_c]):
                        continue
                    if int(candidate_grid[push_dest_r, push_dest_c]) != floor_id:
                        continue
                    if int(candidate_grid[player_r, player_c]) != floor_id:
                        continue
                    candidate_grid[int(row), int(col)] = block_id
                    budget_remaining -= 1
                    return 1

            fallback_slots: List[Tuple[int, int]] = []
            if flow_is_horizontal:
                fallback_slots.extend(
                    [
                        (anchor_r, anchor_c - 3),
                        (anchor_r, anchor_c + 3),
                        (anchor_r - 2, anchor_c),
                        (anchor_r + 2, anchor_c),
                    ]
                )
            else:
                fallback_slots.extend(
                    [
                        (anchor_r - 3, anchor_c),
                        (anchor_r + 3, anchor_c),
                        (anchor_r, anchor_c - 2),
                        (anchor_r, anchor_c + 2),
                    ]
                )

            for raw_row, raw_col in fallback_slots:
                row, col = pipeline._clamp_room_coord((raw_row, raw_col))
                if not _can_place(int(row), int(col)):
                    continue
                if bool(route_mask_candidate[int(row), int(col)]):
                    continue
                route_r0 = max(0, int(row) - 1)
                route_r1 = min(ROOM_HEIGHT, int(row) + 2)
                route_c0 = max(0, int(col) - 1)
                route_c1 = min(ROOM_WIDTH, int(col) + 2)
                if not bool(np.any(route_mask_candidate[route_r0:route_r1, route_c0:route_c1])):
                    continue
                candidate_grid[int(row), int(col)] = block_id
                budget_remaining -= 1
                return 1
            return 0

        segments_added = 0
        tiles_added = 0
        optional_segments_applied = 0
        pocket_tiles_forced = 0
        required_segments, optional_segments = pipeline._build_puzzle_room_segments(
            archetype=archetype,
            gate_family=gate_family,
            variant_spec=variant_spec,
            stateful_anchor=stateful_anchor,
            flow_is_horizontal=flow_is_horizontal,
            puzzle_anchor=puzzle_anchor,
        )

        grammar_anchor = stateful_anchor if stateful_anchor is not None else puzzle_anchor
        if gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"}:
            forced_floor_tiles, anchor_frame_tiles = _apply_anchor_template(
                anchor=grammar_anchor,
                open_toward=source_anchor,
            )
            pocket_tiles_forced += int(forced_floor_tiles)
            if anchor_frame_tiles > 0:
                segments_added += 1
                tiles_added += int(anchor_frame_tiles)

        for segment in required_segments:
            added = _paint_block_line(segment)
            if added > 0:
                segments_added += 1
                tiles_added += added

        optional_quota = int(round(branch_density * len(optional_segments)))
        if branch_density > 0.0 and optional_segments and optional_quota <= 0:
            optional_quota = 1
        optional_quota = min(len(optional_segments), max(0, optional_quota))
        for segment in optional_segments[:optional_quota]:
            added = _paint_block_line(segment)
            if added > 0:
                segments_added += 1
                optional_segments_applied += 1
                tiles_added += added

        push_block_props_added = int(_place_push_block_prop())
        tiles_added += int(push_block_props_added)

        candidate_stats: Dict[str, Any] = {
            "applied": int(tiles_added > 0),
            "tiles_added": int(tiles_added),
            "segments_added": int(segments_added),
            "pocket_floor_tiles_forced": int(pocket_tiles_forced),
            "push_block_props_added": int(push_block_props_added),
            "optional_segments_requested": int(optional_quota),
            "optional_segments_applied": int(optional_segments_applied),
            "route_template_used": int(route_template_used),
            "planned_route_pixels": int(np.sum(route_mask_candidate)),
            "archetype": str(archetype),
            "gate_family": str(gate_family),
            "stateful_anchor_name": str(stateful_anchor_name or ""),
            "variant_name": str(variant_spec.get("name", "baseline") or "baseline"),
            "variant_style": str(variant_spec.get("style", "baseline") or "baseline"),
            "variant_side_bias": int(variant_spec.get("side_bias", 0) or 0),
            "profile_branch_density": float(branch_density),
            "profile_block_budget": int(block_budget),
            "profile_preserve_route_margin": int(preserve_margin),
        }
        route_quality = pipeline._evaluate_puzzle_candidate_route_quality(
            grid=candidate_grid,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            stateful_anchor=stateful_anchor,
            route_mask=route_mask_candidate,
            gate_family=gate_family,
            baseline_path_length=baseline_path_length,
        )
        contract = pipeline._evaluate_puzzle_candidate_contract(
            grid=candidate_grid,
            gate_family=gate_family,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            stateful_anchor=stateful_anchor,
            route_quality=route_quality,
        )
        interaction = pipeline._evaluate_puzzle_candidate_interaction_geometry(
            grid=candidate_grid,
            gate_family=gate_family,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            stateful_anchor=stateful_anchor,
            route_mask=route_mask_candidate,
            route_quality=route_quality,
        )
        sequence_eval = pipeline._evaluate_puzzle_candidate_interaction_sequence(
            grid=candidate_grid,
            route_mask=route_mask_candidate,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            interaction_sequence=interaction_sequence,
        )
        candidate_stats["route_quality_score"] = float(route_quality.get("score", 0.0) or 0.0)
        candidate_stats["route_quality_path_exists"] = int(route_quality.get("path_exists", 0) or 0)
        candidate_stats["route_quality_path_length"] = int(route_quality.get("path_length", 0) or 0)
        candidate_stats["route_quality_turn_count"] = int(route_quality.get("turn_count", 0) or 0)
        candidate_stats["route_quality_overlap_ratio"] = float(
            route_quality.get("route_overlap_ratio", 0.0) or 0.0
        )
        candidate_stats["route_quality_detour_gain"] = float(
            route_quality.get("detour_gain", 0.0) or 0.0
        )
        candidate_stats["route_quality_stateful_distance_to_path"] = route_quality.get(
            "stateful_distance_to_path",
            None,
        )
        candidate_stats["route_quality_stateful_via_path_length"] = route_quality.get(
            "stateful_via_path_length",
            None,
        )
        candidate_stats["route_quality_stateful_branch_gain"] = route_quality.get(
            "stateful_branch_gain",
            None,
        )
        candidate_stats["route_quality_stateful_on_path"] = int(
            route_quality.get("stateful_on_path", 0) or 0
        )
        candidate_stats["contract_valid"] = int(contract.get("valid", 0) or 0)
        candidate_stats["contract_score"] = float(contract.get("score", 0.0) or 0.0)
        candidate_stats["contract_failure_reasons"] = list(contract.get("failure_reasons", []) or [])
        candidate_stats["contract_stateful_anchor_present"] = int(
            contract.get("stateful_anchor_present", 0) or 0
        )
        candidate_stats["contract_projected_stateful_anchor"] = contract.get(
            "projected_stateful_anchor",
            None,
        )
        candidate_stats["contract_pocket_floor_tiles"] = int(
            contract.get("pocket_floor_tiles", 0) or 0
        )
        candidate_stats["contract_frame_block_tiles"] = int(
            contract.get("frame_block_tiles", 0) or 0
        )
        candidate_stats["contract_anchor_adjacent_walkable"] = int(
            contract.get("anchor_adjacent_walkable", 0) or 0
        )
        candidate_stats["interaction_valid"] = int(interaction.get("valid", 0) or 0)
        candidate_stats["interaction_score"] = float(interaction.get("score", 0.0) or 0.0)
        candidate_stats["interaction_failure_reasons"] = list(
            interaction.get("failure_reasons", []) or []
        )
        candidate_stats["interaction_push_slot_count"] = int(
            interaction.get("push_slot_count", 0) or 0
        )
        candidate_stats["interaction_targeted_push_slot_count"] = int(
            interaction.get("targeted_push_slot_count", 0) or 0
        )
        candidate_stats["interaction_anchor_openings"] = int(
            interaction.get("anchor_openings", 0) or 0
        )
        candidate_stats["interaction_local_block_tiles"] = int(
            interaction.get("local_block_tiles", 0) or 0
        )
        candidate_stats["interaction_barrier_axis_tiles"] = int(
            interaction.get("barrier_axis_tiles", 0) or 0
        )
        candidate_stats["interaction_route_divergence"] = float(
            interaction.get("route_divergence", 0.0) or 0.0
        )
        candidate_stats["interaction_sequence_valid"] = int(
            sequence_eval.get("valid", 0) or 0
        )
        candidate_stats["interaction_sequence_score"] = float(
            sequence_eval.get("score", 0.0) or 0.0
        )
        candidate_stats["interaction_sequence_required"] = int(
            sequence_eval.get("required", 0) or 0
        )
        candidate_stats["interaction_sequence_length"] = int(
            sequence_eval.get("sequence_length", 0) or 0
        )
        candidate_stats["interaction_sequence_route_anchor_coverage"] = float(
            sequence_eval.get("route_anchor_coverage", 0.0) or 0.0
        )
        candidate_stats["interaction_sequence_pairwise_path_ratio"] = float(
            sequence_eval.get("pairwise_path_ratio", 0.0) or 0.0
        )
        candidate_stats["interaction_sequence_names"] = list(
            sequence_eval.get("sequence_names", []) or []
        )
        candidate_stats["interaction_sequence_failure_reasons"] = list(
            sequence_eval.get("failure_reasons", []) or []
        )
        descriptor = pipeline._summarize_puzzle_candidate_descriptor(
            grid=candidate_grid,
            stats=candidate_stats,
        )
        candidate_stats["novelty_descriptor"] = descriptor
        candidate_stats["novelty_score"] = float(
            pipeline._score_puzzle_candidate(
                descriptor=descriptor,
                stats=candidate_stats,
                room_id=room_id,
            )
        )
        return candidate_grid, candidate_stats

    selected_grid = base_grid
    selected_stats: Dict[str, Any] = {
        "applied": 0,
        "tiles_added": 0,
        "segments_added": 0,
        "pocket_floor_tiles_forced": 0,
        "push_block_props_added": 0,
        "optional_segments_requested": 0,
        "optional_segments_applied": 0,
        "route_template_used": 0,
        "planned_route_pixels": int(np.sum(route_mask)),
        "archetype": str(archetype),
        "gate_family": str(gate_family),
        "stateful_anchor_name": str(stateful_anchor_name or ""),
        "variant_name": "baseline",
        "variant_style": "baseline",
        "variant_side_bias": 0,
        "profile_branch_density": float(scaffold_profile.get("branch_density", getattr(pipeline, "default_puzzle_room_branch_density", 0.75))),
        "profile_block_budget": int(scaffold_profile.get("block_budget", getattr(pipeline, "default_puzzle_room_block_budget", 28))),
        "profile_preserve_route_margin": int(preserve_margin),
        "novelty_descriptor": {},
        "novelty_score": 0.0,
        "route_quality_score": float(baseline_path_metrics.get("score", 0.0) or 0.0),
        "route_quality_path_exists": int(baseline_path_metrics.get("path_exists", 0) or 0),
        "route_quality_path_length": int(baseline_path_metrics.get("path_length", 0) or 0),
        "route_quality_turn_count": int(baseline_path_metrics.get("turn_count", 0) or 0),
        "route_quality_overlap_ratio": float(
            baseline_path_metrics.get("route_overlap_ratio", 0.0) or 0.0
        ),
        "route_quality_detour_gain": float(
            baseline_path_metrics.get("detour_gain", 0.0) or 0.0
        ),
        "route_quality_stateful_distance_to_path": baseline_path_metrics.get(
            "stateful_distance_to_path",
            None,
        ),
        "route_quality_stateful_via_path_length": baseline_path_metrics.get(
            "stateful_via_path_length",
            None,
        ),
        "route_quality_stateful_branch_gain": baseline_path_metrics.get(
            "stateful_branch_gain",
            None,
        ),
        "route_quality_stateful_on_path": int(
            baseline_path_metrics.get("stateful_on_path", 0) or 0
        ),
    }
    baseline_contract = pipeline._evaluate_puzzle_candidate_contract(
        grid=base_grid,
        gate_family=gate_family,
        source_anchor=source_anchor,
        destination_anchor=destination_anchor,
        stateful_anchor=stateful_anchor,
        route_quality=baseline_path_metrics,
    )
    baseline_interaction = pipeline._evaluate_puzzle_candidate_interaction_geometry(
        grid=base_grid,
        gate_family=gate_family,
        source_anchor=source_anchor,
        destination_anchor=destination_anchor,
        stateful_anchor=stateful_anchor,
        route_mask=route_mask,
        route_quality=baseline_path_metrics,
    )
    baseline_sequence = pipeline._evaluate_puzzle_candidate_interaction_sequence(
        grid=base_grid,
        route_mask=route_mask,
        source_anchor=source_anchor,
        destination_anchor=destination_anchor,
        interaction_sequence=interaction_sequence,
    )
    selected_stats["contract_valid"] = int(baseline_contract.get("valid", 0) or 0)
    selected_stats["contract_score"] = float(baseline_contract.get("score", 0.0) or 0.0)
    selected_stats["contract_failure_reasons"] = list(
        baseline_contract.get("failure_reasons", []) or []
    )
    selected_stats["contract_stateful_anchor_present"] = int(
        baseline_contract.get("stateful_anchor_present", 0) or 0
    )
    selected_stats["contract_projected_stateful_anchor"] = baseline_contract.get(
        "projected_stateful_anchor",
        None,
    )
    selected_stats["contract_pocket_floor_tiles"] = int(
        baseline_contract.get("pocket_floor_tiles", 0) or 0
    )
    selected_stats["contract_frame_block_tiles"] = int(
        baseline_contract.get("frame_block_tiles", 0) or 0
    )
    selected_stats["contract_anchor_adjacent_walkable"] = int(
        baseline_contract.get("anchor_adjacent_walkable", 0) or 0
    )
    selected_stats["interaction_valid"] = int(baseline_interaction.get("valid", 0) or 0)
    selected_stats["interaction_score"] = float(baseline_interaction.get("score", 0.0) or 0.0)
    selected_stats["interaction_failure_reasons"] = list(
        baseline_interaction.get("failure_reasons", []) or []
    )
    selected_stats["interaction_push_slot_count"] = int(
        baseline_interaction.get("push_slot_count", 0) or 0
    )
    selected_stats["interaction_targeted_push_slot_count"] = int(
        baseline_interaction.get("targeted_push_slot_count", 0) or 0
    )
    selected_stats["interaction_anchor_openings"] = int(
        baseline_interaction.get("anchor_openings", 0) or 0
    )
    selected_stats["interaction_local_block_tiles"] = int(
        baseline_interaction.get("local_block_tiles", 0) or 0
    )
    selected_stats["interaction_barrier_axis_tiles"] = int(
        baseline_interaction.get("barrier_axis_tiles", 0) or 0
    )
    selected_stats["interaction_route_divergence"] = float(
        baseline_interaction.get("route_divergence", 0.0) or 0.0
    )
    selected_stats["interaction_sequence_valid"] = int(
        baseline_sequence.get("valid", 0) or 0
    )
    selected_stats["interaction_sequence_score"] = float(
        baseline_sequence.get("score", 0.0) or 0.0
    )
    selected_stats["interaction_sequence_required"] = int(
        baseline_sequence.get("required", 0) or 0
    )
    selected_stats["interaction_sequence_length"] = int(
        baseline_sequence.get("sequence_length", 0) or 0
    )
    selected_stats["interaction_sequence_route_anchor_coverage"] = float(
        baseline_sequence.get("route_anchor_coverage", 0.0) or 0.0
    )
    selected_stats["interaction_sequence_pairwise_path_ratio"] = float(
        baseline_sequence.get("pairwise_path_ratio", 0.0) or 0.0
    )
    selected_stats["interaction_sequence_names"] = list(
        baseline_sequence.get("sequence_names", []) or []
    )
    selected_stats["interaction_sequence_failure_reasons"] = list(
        baseline_sequence.get("failure_reasons", []) or []
    )
    baseline_descriptor = pipeline._summarize_puzzle_candidate_descriptor(
        grid=base_grid,
        stats=selected_stats,
    )
    selected_stats["novelty_descriptor"] = baseline_descriptor
    baseline_selection_score = float(
        pipeline._score_puzzle_candidate(
            descriptor=baseline_descriptor,
            stats=selected_stats,
            room_id=room_id,
        )
    )
    selected_stats["selection_score"] = baseline_selection_score
    baseline_stats = dict(selected_stats)
    selected_score = float("-inf")

    if not variant_specs:
        variant_specs = [{"name": "baseline", "style": "baseline", "side_bias": 0, "branch_density_delta": 0.0, "block_budget_delta": 0}]

    if isinstance(cached_variant, dict):
        selected_grid, selected_stats = _render_candidate(cached_variant)
        selected_stats["selection_score"] = float(selected_stats.get("novelty_score", 0.0))
    else:
        for variant_spec in variant_specs:
            candidate_grid, candidate_stats = _render_candidate(variant_spec)
            candidate_score = float(candidate_stats.get("novelty_score", 0.0))
            candidate_stats["selection_score"] = candidate_score
            if candidate_score > selected_score:
                selected_score = candidate_score
                selected_grid = candidate_grid
                selected_stats = candidate_stats
                if isinstance(variant_cache, dict):
                    variant_cache[room_id] = dict(variant_spec)

    min_quality_gain = float(
        max(0.0, float(getattr(pipeline, "default_puzzle_room_min_quality_gain", 0.5)))
    )
    contract_required = gate_family in {"switch", "toggle", "item_unlock", "key", "combat"}
    interaction_required = gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"}
    sequence_required = int(selected_stats.get("interaction_sequence_required", 0) or 0) > 0
    if contract_required and int(selected_stats.get("contract_valid", 0) or 0) <= 0:
        selected_grid = base_grid
        selected_stats = dict(baseline_stats)
        selected_stats["selection_score"] = float(baseline_selection_score)
        selected_stats["quality_gate_skipped"] = 0
        selected_stats["contract_gate_skipped"] = 1
        selected_stats["interaction_gate_skipped"] = 0
        selected_stats["sequence_gate_skipped"] = 0
        stats.update(selected_stats)
        return selected_grid, stats
    if interaction_required and int(selected_stats.get("interaction_valid", 0) or 0) <= 0:
        selected_grid = base_grid
        selected_stats = dict(baseline_stats)
        selected_stats["selection_score"] = float(baseline_selection_score)
        selected_stats["quality_gate_skipped"] = 0
        selected_stats["contract_gate_skipped"] = 0
        selected_stats["interaction_gate_skipped"] = 1
        selected_stats["sequence_gate_skipped"] = 0
        stats.update(selected_stats)
        return selected_grid, stats
    if sequence_required and int(selected_stats.get("interaction_sequence_valid", 0) or 0) <= 0:
        selected_grid = base_grid
        selected_stats = dict(baseline_stats)
        selected_stats["selection_score"] = float(baseline_selection_score)
        selected_stats["quality_gate_skipped"] = 0
        selected_stats["contract_gate_skipped"] = 0
        selected_stats["interaction_gate_skipped"] = 0
        selected_stats["sequence_gate_skipped"] = 1
        stats.update(selected_stats)
        return selected_grid, stats

    if (
        float(selected_stats.get("selection_score", float("-inf")))
        < (baseline_selection_score + min_quality_gain)
    ):
        selected_grid = base_grid
        selected_stats = dict(baseline_stats)
        selected_stats["selection_score"] = float(baseline_selection_score)
        selected_stats["quality_gate_skipped"] = 1
        selected_stats["contract_gate_skipped"] = 0
        selected_stats["interaction_gate_skipped"] = 0
        selected_stats["sequence_gate_skipped"] = 0
    else:
        selected_stats["quality_gate_skipped"] = 0
        selected_stats["contract_gate_skipped"] = 0
        selected_stats["interaction_gate_skipped"] = 0
        selected_stats["sequence_gate_skipped"] = 0

    stats.update(selected_stats)
    return selected_grid, stats


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


def _build_room_puzzle_metadata(
    pipeline,
    *,
    grid: np.ndarray,
    graph: Optional[nx.Graph],
    room_id: Any,
    start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    marker_plan: Optional[Sequence[Tuple[int, Tuple[int, int]]]] = None,
    scaffold_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a structured room-local puzzle plan for validator-side state machines.

    The room scaffold already knows the intended local progression order.
    This helper serializes that order into explicit stateful stages so the
    stitched validator can enforce multi-step mechanics instead of treating
    puzzle rooms as purely visual clutter.
    """
    if not isinstance(graph, nx.Graph) or room_id not in graph:
        return {}

    room_grid = np.asarray(grid, dtype=np.int32)
    attrs = dict(graph.nodes[room_id])
    role_flags = pipeline._room_role_flags(attrs)
    semantics = pipeline._extract_room_topology_semantics(graph, room_id)
    node_type = str(
        attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or ""
    ).strip().lower()
    if start_goal is None:
        start_goal = pipeline._extract_room_start_goal(graph, room_id)
    start_coord, goal_coord = start_goal if start_goal is not None else (
        (ROOM_HEIGHT // 2, 0),
        (ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
    )
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
    gate_family = str(
        (scaffold_stats or {}).get(
            "gate_family",
            pipeline._classify_puzzle_gate_family(
                role_flags=role_flags,
                semantics=semantics,
                node_type=node_type,
            ),
        )
        or "generic"
    ).strip().lower()
    archetype = str(
        (scaffold_stats or {}).get(
            "archetype",
            pipeline._select_puzzle_room_scaffold_archetype(
                role_flags=role_flags,
                semantics=semantics,
                node_type=node_type,
            ),
        )
        or "serpentine"
    ).strip().lower()
    interaction_sequence = pipeline._resolve_puzzle_interaction_sequence(
        archetype=archetype,
        gate_family=gate_family,
        role_flags=role_flags,
        semantic_anchors=semantic_anchors,
    )
    door_rows, door_cols = np.where(room_grid == int(TileID.DOOR_PUZZLE))
    controlled_doors_local = [
        [int(row), int(col)]
        for row, col in zip(door_rows.tolist(), door_cols.tolist())
    ]
    if not interaction_sequence and controlled_doors_local:
        fallback_anchor = pipeline._clamp_room_coord(
            semantic_anchors.get("puzzle", semantic_anchors.get("goal", goal_coord))
        )
        interaction_sequence = [("puzzle", fallback_anchor)]
    if not interaction_sequence and not controlled_doors_local:
        return {}

    marker_slots: Dict[int, List[Tuple[int, int]]] = {}
    for tile_id, slot in list(marker_plan or []):
        marker_slots.setdefault(int(tile_id), []).append(pipeline._clamp_room_coord(slot))

    def _resolve_anchor(
        name: str,
        fallback: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Optional[int]]:
        normalized = str(name).strip().lower()
        if normalized == "key":
            candidates = [int(TileID.KEY_SMALL), int(TileID.KEY_BOSS)]
        elif normalized == "item":
            candidates = [int(TileID.KEY_ITEM), int(TileID.ITEM_MINOR), int(TileID.STAIR)]
        elif normalized == "puzzle":
            candidates = [int(TileID.PUZZLE)]
        elif normalized == "enemy":
            candidates = [int(TileID.ENEMY), int(TileID.BOSS)]
        elif normalized == "boss":
            candidates = [int(TileID.BOSS)]
        else:
            candidates = []
        for tile_id in candidates:
            if marker_slots.get(int(tile_id)):
                planned_slot = marker_slots[int(tile_id)][0]
                planned_row, planned_col = pipeline._clamp_room_coord(planned_slot)
                if int(room_grid[planned_row, planned_col]) == int(tile_id):
                    return (int(planned_row), int(planned_col)), int(tile_id)
        # Symbolic-only and legacy callers do not have the pre-overlay marker
        # plan. Prefer the final artifact's observed marker over a synthetic
        # anchor so the validator enforces the puzzle the player can see.
        fallback_row, fallback_col = pipeline._clamp_room_coord(fallback)
        observed: List[Tuple[int, int, int, int]] = []
        for tile_id in candidates:
            for row, col in np.argwhere(room_grid == int(tile_id)):
                distance = abs(int(row) - int(fallback_row)) + abs(int(col) - int(fallback_col))
                observed.append((int(distance), int(row), int(col), int(tile_id)))
        if observed:
            _distance, row, col, tile_id = min(observed)
            return (int(row), int(col)), int(tile_id)
        return pipeline._clamp_room_coord(fallback), None

    def _stage_kind(name: str) -> str:
        normalized = str(name).strip().lower()
        if normalized == "key":
            return "collect_key"
        if normalized == "item":
            return "collect_item"
        if normalized in {"enemy", "boss"}:
            return "defeat_enemy"
        if normalized == "puzzle" and gate_family in {"switch", "toggle"}:
            return "push_block_to_switch"
        return "step_on_puzzle"

    stage_sequence: List[Dict[str, Any]] = []
    for stage_index, (name, anchor) in enumerate(interaction_sequence):
        resolved_anchor, tile_id = _resolve_anchor(str(name), anchor)
        stage_sequence.append(
            {
                "stage_index": int(stage_index),
                "name": str(name),
                "kind": _stage_kind(str(name)),
                "local_anchor": [int(resolved_anchor[0]), int(resolved_anchor[1])],
                "trigger_tile_id": int(tile_id) if tile_id is not None else None,
            }
        )

    return {
        "plan_id": f"room_{room_id}",
        "room_id": room_id,
        "gate_family": str(gate_family),
        "archetype": str(archetype),
        "sequence_required": bool(int((scaffold_stats or {}).get("interaction_sequence_required", 0) or 0) > 0),
        "sequence_valid": bool(int((scaffold_stats or {}).get("interaction_sequence_valid", 0) or 0) > 0),
        "interaction_valid": bool(int((scaffold_stats or {}).get("interaction_valid", 0) or 0) > 0),
        "contract_valid": bool(int((scaffold_stats or {}).get("contract_valid", 0) or 0) > 0),
        "controlled_doors_local": list(controlled_doors_local),
        "stage_sequence": stage_sequence,
    }


def _globalize_room_puzzle_metadata(
    pipeline,
    *,
    rooms: Mapping[Any, RoomGenerationResult],
    stitched_layout: Optional[StitchedRoomLayout],
) -> Dict[str, Any]:
    """Lift room-local puzzle plans into stitched global coordinates."""
    if stitched_layout is None:
        return {"version": "stateful_v1", "plans": {}, "room_to_plan": {}}

    plans: Dict[str, Dict[str, Any]] = {}
    room_to_plan: Dict[str, str] = {}
    for room_id, room in dict(rooms or {}).items():
        local_meta = dict(getattr(room, "puzzle_metadata", {}) or {})
        if not local_meta:
            continue
        offset = stitched_layout.room_offsets.get(room_id)
        if offset is None:
            continue
        offset_r, offset_c = int(offset[0]), int(offset[1])
        plan_id = str(local_meta.get("plan_id", f"room_{room_id}"))
        room_to_plan[str(room_id)] = str(plan_id)

        global_stages: List[Dict[str, Any]] = []
        for raw_stage in list(local_meta.get("stage_sequence", []) or []):
            local_anchor = raw_stage.get("local_anchor")
            if not isinstance(local_anchor, (list, tuple)) or len(local_anchor) != 2:
                continue
            local_r, local_c = int(local_anchor[0]), int(local_anchor[1])
            global_stages.append(
                {
                    **dict(raw_stage),
                    "local_anchor": [local_r, local_c],
                    "global_anchor": [offset_r + local_r, offset_c + local_c],
                }
            )

        global_doors: List[List[int]] = []
        for local_door in list(local_meta.get("controlled_doors_local", []) or []):
            if not isinstance(local_door, (list, tuple)) or len(local_door) != 2:
                continue
            door_r, door_c = int(local_door[0]), int(local_door[1])
            global_doors.append([offset_r + door_r, offset_c + door_c])

        plans[str(plan_id)] = {
            **local_meta,
            "plan_id": str(plan_id),
            "room_id": room_id,
            "room_offset": [offset_r, offset_c],
            "stage_sequence": global_stages,
            "controlled_doors_global": global_doors,
        }

    return {
        "version": "stateful_v1",
        "plans": plans,
        "room_to_plan": room_to_plan,
    }


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


def _aggregate_puzzle_stage_semantics_metrics(
    pipeline,
    room_metric_dicts: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Aggregate model-stage scores without conflating them with constraints.

    A room can be structurally repaired after decoding.  Reporting only the
    repaired result would make an ablation appear to measure neural capability
    when it actually measures the combined neural-plus-symbolic system.  Keep
    raw and constrained scores separate, and make missing heads/conditions
    explicit through coverage fields rather than treating them as failures.
    """
    metric_keys = (
        "puzzle_stage_semantics_raw_joint_confidence",
        "puzzle_stage_semantics_constrained_joint_confidence",
        "puzzle_stage_semantics_constraint_confidence_delta",
        "puzzle_stage_semantics_raw_condition_available",
        "puzzle_stage_semantics_raw_head_loaded",
        "puzzle_stage_semantics_constrained_condition_available",
        "puzzle_stage_semantics_constrained_head_loaded",
    )
    output: Dict[str, float] = {
        "puzzle_stage_semantics_rooms_total": float(len(room_metric_dicts)),
        "puzzle_stage_semantics_rooms_scored": 0.0,
        "puzzle_stage_semantics_score_coverage": 0.0,
    }
    scored = [
        metrics
        for metrics in room_metric_dicts
        if "puzzle_stage_semantics_raw_joint_confidence" in metrics
    ]
    output["puzzle_stage_semantics_rooms_scored"] = float(len(scored))
    if room_metric_dicts:
        output["puzzle_stage_semantics_score_coverage"] = float(
            len(scored) / len(room_metric_dicts)
        )

    for key in metric_keys:
        values: List[float] = []
        for metrics in room_metric_dicts:
            if key not in metrics:
                continue
            try:
                value = float(metrics[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            output[f"avg_{key}"] = float(np.mean(values))
    return output


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
    try:
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
            return_global_tokens=pipeline.use_graph_node_cross_attention,
        )
        if pipeline.use_graph_node_cross_attention:
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
        if pipeline.strict_checkpoint_mode:
            raise RuntimeError(
                f"Condition encoding failed in strict mode: {e}"
            ) from e
        logger.warning(f"Condition encoding failed: {e}, using zero condition")
        if pipeline.use_graph_node_cross_attention:
            num_nodes = 0
            node_features = graph_context.get("node_features")
            if isinstance(node_features, torch.Tensor) and node_features.dim() >= 2:
                num_nodes = int(node_features.shape[0])
            required_seq_len = max(1, num_nodes + 1)
            condition = torch.zeros(1, required_seq_len, condition_dim, device=pipeline.device)
        else:
            condition = torch.zeros(1, condition_dim, device=pipeline.device)
        node_tokens = None

    if pipeline.use_graph_node_cross_attention and isinstance(node_tokens, torch.Tensor):
        try:
            if node_tokens.dim() == 2:
                node_tokens = node_tokens.unsqueeze(0)
            condition = torch.cat([condition.unsqueeze(1), node_tokens], dim=1)
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            pipeline._bump_diagnostic("graph_node_cross_attention_fallback")
            logger.debug("Falling back to single conditioning vector: %s", e)

    active_generator_mode = str(
        room_generator_mode or getattr(pipeline, "room_generator_mode", "latent_diffusion")
    ).strip().lower()
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
            if bool(pipeline.use_graph_node_cross_attention):
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
            if bool(pipeline.use_graph_node_cross_attention):
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
    """Return dependency-safe topological layers for directed graphs."""
    if not graph.is_directed():
        return [sorted(list(graph.nodes()), key=_stable_node_sort_key)]
    try:
        layers = [
            sorted(list(layer), key=_stable_node_sort_key)
            for layer in nx.topological_generations(graph)
        ]
        return [layer for layer in layers if layer]
    except nx.NetworkXUnfeasible:
        logger.warning(
            "Mission graph contains cycles; disabling layer batching and using sorted node order."
        )
        return [sorted(list(graph.nodes()), key=_stable_node_sort_key)]


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


def _slice_graph_guidance_batch(pipeline, graph_ctx: Dict[str, Any], batch_index: int) -> Dict[str, Any]:
    """Slice a batched graph-guidance payload down to one room/sample."""
    sliced: Dict[str, Any] = {}
    for key, value in graph_ctx.items():
        if not isinstance(value, torch.Tensor):
            sliced[key] = value
            continue
        if value.dim() >= 1 and int(value.shape[0]) > batch_index:
            if key == "edge_index" and value.dim() == 2:
                sliced[key] = value
            elif key == "node_features" and value.dim() == 2:
                sliced[key] = value
            elif key == "edge_features" and value.dim() == 2:
                sliced[key] = value
            else:
                sliced[key] = value[batch_index:batch_index + 1]
        else:
            sliced[key] = value
    return sliced


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

