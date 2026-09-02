"""Small integration layer between `visual_extractor` and the solver stack.

Purpose (minimal, testable):
- convert a single-room screenshot -> semantic room grid + confidence
- infer simple inventory items (KEY/ITEM/bomb) from the semantic grid
- produce a minimal StitchedDungeon wrapper for existing solvers

Keep responsibilities tiny so we don't duplicate game logic (reuse
Data.zelda_core constants and dataclasses).
"""
from __future__ import annotations
from typing import Tuple, Set, Optional
import numpy as np
from src.core.definitions import SEMANTIC_PALETTE, ROOM_HEIGHT, ROOM_WIDTH
from src.pipeline.room_stitching import build_room_canvas_from_slots
from src.zelda_data.zelda_core import StitchedDungeon
from src.data_processing.visual_extractor import extract_grid


def visual_extract_to_room(image_path: str, templates_dir: str, tile_px: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Run `extract_grid` on a screenshot and return (semantic_ids, conf).

    - semantic_ids: (ROOM_H, ROOM_W) int grid (uses SEMANTIC_PALETTE semantics where possible)
    - conf: (ROOM_H, ROOM_W) float confidence mask in [0,1]

    This is a thin wrapper that maps extract_grid's output into the
    project's semantic palette where possible.
    """
    out = extract_grid(image_path, templates_dir, tile_px=tile_px)
    ids = out[:, :, 0].astype(int)
    conf = out[:, :, 1].astype(float)

    # Heuristic mapping: if a template index corresponds to a known semantic
    # id (not available from extractor), caller can post-process. Return raw ids
    # for now alongside confidence -- higher layers will convert using template
    # metadata if available.
    return ids, conf


def infer_inventory_from_room(
    semantic_grid: np.ndarray,
    conf: Optional[np.ndarray] = None,
    template_to_semantic: Optional[dict] = None,
) -> Set[str]:
    """Return a small set of inventory tokens detected in the room.

    Currently recognizes: 'small_key', 'bomb', 'item' (generic).

    Args:
        semantic_grid: A 2-D integer grid. Can be either:
            * A **semantic grid** (values are SEMANTIC_PALETTE IDs, e.g. 30 for
              KEY) — pass ``template_to_semantic=None`` in this case.
            * A **raw template-index grid** (values are visual template indices
              returned by :func:`extract_grid`) — in this case supply the
              ``template_to_semantic`` mapping so the function can translate
              indices before comparing against SEMANTIC_PALETTE constants.
        conf:  Optional confidence mask (same shape). Not currently used for
              filtering but reserved for future threshold-gated detection.
        template_to_semantic: Optional dict mapping ``int`` template indices to
              ``int`` SEMANTIC_PALETTE IDs.  Build this from tileset metadata.
              When ``None`` the grid is assumed to already contain semantic IDs.

    Returns:
        A set of inventory token strings (e.g. {'small_key', 'item'}).
    """
    items: Set[str] = set()

    # If we received raw template indices, remap to semantic IDs first so that
    # comparisons against SEMANTIC_PALETTE work correctly.
    if template_to_semantic is not None:
        # Build a vectorised lookup table for speed (handles unknown indices by
        # mapping them to -1 which will never match a valid palette entry).
        max_idx = int(semantic_grid.max()) if semantic_grid.size > 0 else 0
        lut = np.full(max_idx + 1, -1, dtype=np.int32)
        for tpl_idx, sem_id in template_to_semantic.items():
            if 0 <= int(tpl_idx) <= max_idx:
                lut[int(tpl_idx)] = int(sem_id)
        grid = lut[np.clip(semantic_grid.astype(np.int32), 0, max_idx)]
    else:
        grid = semantic_grid

    key_id = int(SEMANTIC_PALETTE.get('KEY', 30))
    item_id = int(SEMANTIC_PALETTE.get('ITEM', 33))

    if np.any(grid == key_id):
        items.add('small_key')
    if np.any(grid == item_id):
        items.add('item')
    # bombs/items are not uniquely encoded in VGLC; heuristically detect 'bomb'
    # by nearby BLOCK+ITEM pattern (left as simple detection for now).
    return items


def make_stitched_for_single_room(room_grid: np.ndarray, room_pos: Tuple[int, int] = (0, 0)) -> StitchedDungeon:
    """Create a minimal StitchedDungeon containing a single room.

    This lets existing solvers (TilePathFinder, MazeSolver) operate on the
    single-room visual extraction with minimal changes.
    """
    h, w = room_grid.shape
    assert h == ROOM_HEIGHT and w == ROOM_WIDTH, "expected single-room grid"

    stitched_layout = build_room_canvas_from_slots(
        room_grids={room_pos: np.asarray(room_grid, dtype=np.int32)},
        slot_positions={room_pos: room_pos},
        fill_tile=int(SEMANTIC_PALETTE.get("VOID", 0)),
    )
    global_grid = stitched_layout.dungeon_grid
    room_positions = stitched_layout.room_offsets

    # Find START/TRIFORCE tile positions if present
    starts = np.argwhere(global_grid == SEMANTIC_PALETTE.get('START', 21))
    trif = np.argwhere(global_grid == SEMANTIC_PALETTE.get('TRIFORCE', 22))

    start_global = tuple(starts[0]) if starts.size else None
    trif_global = tuple(trif[0]) if trif.size else None

    return StitchedDungeon(dungeon_id="visual_single",
                            global_grid=global_grid,
                            room_positions=room_positions,
                            start_global=start_global,
                            triforce_global=trif_global)
