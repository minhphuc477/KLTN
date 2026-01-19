import numpy as np
from src.data_processing.visual_integration import make_stitched_for_single_room
from graph_solver import TilePathFinder
from Data.zelda_core import SEMANTIC_PALETTE, ROOM_HEIGHT, ROOM_WIDTH


def _make_room_with_element_barrier():
    """Create a single-room semantic grid where the only route from left->right
    must cross an ELEMENT (water) tile in the center column.
    Layout (11 cols): start at col 2, element barrier at col 5, goal at col 8.
    """
    g = np.full((ROOM_HEIGHT, ROOM_WIDTH), SEMANTIC_PALETTE['WALL'], dtype=np.int32)
    # clear a single corridor row
    r = ROOM_HEIGHT // 2
    for c in range(1, ROOM_WIDTH - 1):
        g[r, c] = SEMANTIC_PALETTE['FLOOR']
    # place START and TRIFORCE
    g[r, 2] = SEMANTIC_PALETTE['START']
    g[r, 8] = SEMANTIC_PALETTE['TRIFORCE']
    # place ELEMENT barrier (water) as the only pass-through cell
    g[r, 5] = SEMANTIC_PALETTE['ELEMENT']
    return g


def test_tile_path_requires_inventory_for_element():
    room = _make_room_with_element_barrier()
    stitched = make_stitched_for_single_room(room)
    tp = TilePathFinder(stitched, None)

    # Without raft -> pathfinder should NOT step on ELEMENT (returns fallback)
    path_no_inv = tp.find_tile_path([(0, 0)], inventory=set())
    # if algorithm cannot traverse water, it should either return fallback length 2 or a path that avoids element
    assert path_no_inv, "pathfinder must return a path (fallback or real)"
    # ensure ELEMENT tile not in path when no inventory (or that path is trivial fallback)
    assert not any(stitched.global_grid[pos] == SEMANTIC_PALETTE['ELEMENT'] for pos in path_no_inv), \
        "ELEMENT should not be traversed without inventory"

    # With raft in inventory -> path must include the ELEMENT tile
    path_with_raft = tp.find_tile_path([(0, 0)], inventory={'raft'})
    assert any(stitched.global_grid[pos] == SEMANTIC_PALETTE['ELEMENT'] for pos in path_with_raft), \
        "ELEMENT should be traversable when 'raft' is in inventory"
