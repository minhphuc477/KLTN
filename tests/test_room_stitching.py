import networkx as nx
import numpy as np

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.pipeline.spatial_utils import carve_room_connection
from src.pipeline.room_stitching import (
    build_room_canvas_from_slots,
    carve_room_connection_between_bboxes,
    compute_layout_quality_metrics,
    compute_graph_aware_room_slots,
)
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


def test_non_adjacent_bbox_connector_walls_off_relaxed_corridor():
    void_id = int(SEMANTIC_PALETTE["VOID"])
    wall_id = int(SEMANTIC_PALETTE["WALL"])
    floor_id = int(SEMANTIC_PALETTE["FLOOR"])

    grid = np.full((24, 40), void_id, dtype=np.int32)
    # Paint two separated room footprints so the connector must route through void.
    grid[4:16, 2:13] = wall_id
    grid[4:16, 24:35] = wall_id

    src_bbox = (2, 4, 12, 15)
    dst_bbox = (24, 4, 34, 15)

    carve_room_connection_between_bboxes(
        grid,
        src_bbox,
        dst_bbox,
        fill_tile=void_id,
    )

    gap_mask = np.zeros_like(grid, dtype=bool)
    gap_mask[:, 13:24] = True
    corridor_floor = np.argwhere((grid == floor_id) & gap_mask)
    corridor_walls = np.argwhere((grid == wall_id) & gap_mask)

    assert len(corridor_floor) > 0
    assert len(corridor_walls) > 0


def test_compute_graph_aware_room_slots_handles_duplicate_preferred_positions():
    graph = nx.Graph()
    graph.add_node(0, position=(0, 0, 0))
    graph.add_node(1, position=(1, 0, 0))
    graph.add_node(2, position=(1, 1, 0))
    # Different floor, same row/col preference once z is ignored.
    graph.add_node(3, position=(1, 1, 1))
    graph.add_edges_from(((0, 1), (1, 2), (1, 3)))

    slot_positions = compute_graph_aware_room_slots(graph, [0, 1, 2, 3])

    assert len(set(slot_positions.values())) == 4
    for src, dst in graph.edges():
        src_pos = slot_positions[src]
        dst_pos = slot_positions[dst]
        assert abs(src_pos[0] - dst_pos[0]) + abs(src_pos[1] - dst_pos[1]) == 1

    metrics = compute_layout_quality_metrics(graph, slot_positions)
    assert metrics["graph_edge_slot_adjacency_rate"] == 1.0
    assert metrics["graph_edge_slot_mean_distance"] == 1.0
    assert metrics["graph_preferred_position_duplicate_rate"] is not None
    assert metrics["graph_preferred_position_duplicate_rate"] > 0.0


def test_compute_graph_aware_room_slots_uses_tree_fallback_for_cyclic_progression_graph():
    graph = nx.Graph()
    positions = {
        0: (0, 0, 0),
        1: (5, 5, 0),
        2: (2, 2, 0),
        3: (2, 2, 0),
        4: (2, 2, 1),
        5: (4, 2, 1),
        6: (2, 2, 1),
        7: (2, 2, 1),
        8: (3, 5, 0),
        9: (4, 5, 0),
        10: (2, 3, 1),
    }
    labels = {
        0: "START",
        1: "GOAL",
        2: "COMBAT_PUZZLE",
        3: "COMPLEX_PUZZLE",
        4: "STAIRS_UP",
        5: "TUTORIAL_PUZZLE",
        6: "ITEM",
        7: "EMPTY",
        8: "BOSS_DOOR",
        9: "BOSS",
        10: "BIG_KEY",
    }
    for node_id, pos in positions.items():
        graph.add_node(node_id, position=pos, label=labels[node_id], type=labels[node_id])
    graph.add_edges_from(
        (
            (0, 2),
            (0, 5),
            (2, 3),
            (2, 8),
            (3, 4),
            (3, 5),
            (4, 7),
            (5, 6),
            (6, 7),
            (7, 10),
            (8, 9),
            (9, 1),
        )
    )

    slot_positions = compute_graph_aware_room_slots(graph, sorted(graph.nodes()))

    assert abs(slot_positions[0][0] - slot_positions[2][0]) + abs(slot_positions[0][1] - slot_positions[2][1]) == 1
    assert abs(slot_positions[0][0] - slot_positions[5][0]) + abs(slot_positions[0][1] - slot_positions[5][1]) == 1
    assert len({col for _, col in slot_positions.values()}) >= 3


def test_layout_quality_metrics_do_not_rely_on_exact_graph_slot_matches_for_noisy_coordinates():
    graph = nx.Graph()
    graph.add_node(0, position=(0, 0, 0), label="START", type="START")
    graph.add_node(1, position=(4, 4, 0), label="ITEM", type="ITEM")
    graph.add_node(2, position=(4, 4, 0), label="COMBAT_PUZZLE", type="COMBAT_PUZZLE")
    graph.add_node(3, position=(5, 5, 0), label="GOAL", type="GOAL")
    graph.add_edges_from(((0, 1), (1, 2), (2, 3)))

    slot_positions = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 2),
    }

    metrics = compute_layout_quality_metrics(graph, slot_positions)

    assert metrics["graph_slot_match_rate"] is not None
    assert metrics["graph_slot_match_rate"] < 1.0
    assert metrics["graph_edge_slot_adjacency_rate"] == 1.0
    assert metrics["graph_edge_slot_mean_excess_distance"] == 0.0
    assert metrics["graph_preferred_position_duplicate_rate"] is not None
    assert metrics["graph_preferred_position_duplicate_rate"] > 0.0
