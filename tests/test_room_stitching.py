import numpy as np

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.pipeline.spatial_utils import carve_room_connection
from src.pipeline.room_stitching import build_room_canvas_from_slots
from src.data_processing.visual_integration import make_stitched_for_single_room
from src.zelda_data.stitching.connectivity import connect_doors
from src.zelda_data.stitching.stitch_orchestration import (
    build_global_grid_from_rooms,
    build_stitched_room_layout_from_rooms,
)


def test_build_room_canvas_from_slots_returns_offsets_and_layout():
    room_grids = {
        "a": np.full((4, 3), 1, dtype=np.int32),
        "b": np.full((4, 5), 2, dtype=np.int32),
    }
    slot_positions = {
        "a": (0, 0),
        "b": (0, 1),
    }

    stitched = build_room_canvas_from_slots(
        room_grids=room_grids,
        slot_positions=slot_positions,
        fill_tile=0,
    )

    assert stitched.dungeon_grid.shape == (4, 8)
    assert stitched.room_offsets["a"] == (0, 0)
    assert stitched.room_offsets["b"] == (0, 3)
    assert stitched.layout_map["a"] == (0, 0, 2, 3)
    assert stitched.layout_map["b"] == (3, 0, 7, 3)


def test_build_global_grid_from_rooms_uses_shared_canvas_builder():
    class _Room:
        def __init__(self, semantic_grid):
            self.semantic_grid = semantic_grid

    rooms = {
        (0, 0): _Room(np.full((16, 11), 1, dtype=np.int32)),
        (0, 1): _Room(np.full((16, 11), 2, dtype=np.int32)),
    }

    grid, room_positions = build_global_grid_from_rooms(
        rooms_remapped=rooms,
        room_height=16,
        room_width=11,
    )

    assert grid.shape == (16, 22)
    assert room_positions[(0, 0)] == (0, 0)
    assert room_positions[(0, 1)] == (0, 11)
    assert int(grid[0, 0]) == 1
    assert int(grid[0, 11]) == 2


def test_build_stitched_room_layout_from_rooms_returns_canonical_layout_object():
    class _Room:
        def __init__(self, semantic_grid):
            self.semantic_grid = semantic_grid

    rooms = {
        (0, 0): _Room(np.full((16, 11), 3, dtype=np.int32)),
        (1, 0): _Room(np.full((16, 11), 4, dtype=np.int32)),
    }

    stitched = build_stitched_room_layout_from_rooms(
        rooms_remapped=rooms,
        room_height=16,
        room_width=11,
    )

    assert stitched.dungeon_grid.shape == (32, 11)
    assert stitched.room_offsets[(0, 0)] == (0, 0)
    assert stitched.room_offsets[(1, 0)] == (16, 0)
    assert stitched.layout_map[(0, 0)] == (0, 0, 10, 15)
    assert stitched.layout_map[(1, 0)] == (0, 16, 10, 31)


def test_make_stitched_for_single_room_uses_shared_stitch_canvas():
    room_grid = np.full((16, 11), 5, dtype=np.int32)
    room_grid[7, 5] = 21
    room_grid[8, 5] = 22

    stitched = make_stitched_for_single_room(room_grid, room_pos=(2, 3))

    assert stitched.global_grid.shape == (16, 11)
    assert stitched.room_positions[(2, 3)] == (0, 0)
    assert stitched.start_global == (7, 5)
    assert stitched.triforce_global == (8, 5)


def test_connect_doors_uses_shared_connector_for_one_way_and_reciprocal_boundaries():
    class _Room:
        def __init__(self, doors):
            self.doors = doors

    wall = int(SEMANTIC_PALETTE["WALL"])
    floor = int(SEMANTIC_PALETTE["FLOOR"])
    door_open = int(SEMANTIC_PALETTE["DOOR_OPEN"])
    door_soft = int(SEMANTIC_PALETTE["DOOR_SOFT"])

    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH * 2), wall, dtype=np.int32)
    rooms = {
        (0, 0): _Room({"E": True}),
        (0, 1): _Room({"W": False}),
    }

    connect_doors(
        grid=grid,
        rooms=rooms,
        semantic_palette=SEMANTIC_PALETTE,
        room_height=ROOM_HEIGHT,
        room_width=ROOM_WIDTH,
    )

    shared_rows = np.where(grid[:, ROOM_WIDTH - 1] == door_soft)[0]
    assert len(shared_rows) > 0
    assert np.all(grid[shared_rows, ROOM_WIDTH] == door_open)
    assert np.any(grid[:, ROOM_WIDTH // 2] == floor)


def test_spatial_utils_carve_room_connection_delegates_to_shared_bbox_connector():
    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH * 2), int(SEMANTIC_PALETTE["VOID"]), dtype=np.int32)

    carve_room_connection(
        global_grid=grid,
        src_pos=(0, 0),
        dst_pos=(0, 1),
        edge_data={"edge_type": "locked"},
        has_reverse_edge=True,
    )

    locked = int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    boundary_rows = np.where(grid[:, ROOM_WIDTH - 1] == locked)[0]
    assert len(boundary_rows) > 0
    assert np.all(grid[boundary_rows, ROOM_WIDTH] == locked)
