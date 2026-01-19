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
from Data.zelda_core import (
    SEMANTIC_PALETTE,
    StitchedDungeon,
    ROOM_HEIGHT,
    ROOM_WIDTH,
)
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
    # for now alongside confidence — higher layers will convert using template
    # metadata if available.
    return ids, conf


def infer_inventory_from_room(semantic_grid: np.ndarray, conf: Optional[np.ndarray] = None) -> Set[str]:
    """Return a small set of inventory tokens detected in the room.

    Currently recognizes: 'small_key', 'bomb', 'item' (generic).
    Uses SEMANTIC_PALETTE constants.
    """
    items = set()
    if np.any(semantic_grid == SEMANTIC_PALETTE.get('KEY', 30)):
        items.add('small_key')
    if np.any(semantic_grid == SEMANTIC_PALETTE.get('ITEM', 33)):
        items.add('item')
    # bombs/items are not uniquely encoded in VGLC; heuristically detect 'bomb' by nearby BLOCK+ITEM pattern
    # (left as simple detection for now)
    return items


def make_stitched_for_single_room(room_grid: np.ndarray, room_pos: Tuple[int, int] = (0, 0)) -> StitchedDungeon:
    """Create a minimal StitchedDungeon containing a single room.

    This lets existing solvers (TilePathFinder, MazeSolver) operate on the
    single-room visual extraction with minimal changes.
    """
    h, w = room_grid.shape
    assert h == ROOM_HEIGHT and w == ROOM_WIDTH, "expected single-room grid"

    # Place the room at global origin (0,0)
    global_grid = np.array(room_grid, dtype=np.int32)
    room_positions = {room_pos: (0, 0)}

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
