from types import SimpleNamespace

import networkx as nx
import numpy as np

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.pipeline.spatial_utils import fit_room_grid
from src.pipeline.room_topology_conditioning import ROOM_TOPOLOGY_CHANNELS
from src.zelda_data.zelda_loader import (
    ZeldaDungeonDataset,
    _build_room_graph_sample,
    _extract_graph_from_dungeon,
)


def test_dungeon_dataset_getitem_preserves_spatial_graph_fields():
    dataset = ZeldaDungeonDataset.__new__(ZeldaDungeonDataset)
    dataset.samples = [np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)]
    dataset.files = []
    dataset.transform = None
    dataset.normalize = True
    dataset.target_size = None
    dataset.load_graphs = True
    dataset.graphs = [{
        "node_features": np.zeros((2, 6), dtype=np.float32),
        "edge_index": np.array([[0], [1]], dtype=np.int64),
        "edge_attr": np.array([1], dtype=np.int64),
        "tpe": np.ones((2, 8), dtype=np.float32),
        "node_positions": np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "num_nodes": 2,
        "num_edges": 1,
        "start_node_id": 0,
        "node_to_idx": {10: 0, 20: 1},
    }]

    _room_tensor, graph = ZeldaDungeonDataset.__getitem__(dataset, 0)

    assert "tpe" in graph
    assert "node_positions" in graph
    assert tuple(graph["tpe"].shape) == (2, 8)
    assert tuple(graph["node_positions"].shape) == (2, 2)
    assert graph["node_to_idx"] == {10: 0, 20: 1}


def test_room_graph_sample_builds_room_topology_from_dataset_graph():
    graph = nx.DiGraph()
    graph.add_node(10, label="s", is_start=True)
    graph.add_node(20, label="t", is_triforce=True)
    graph.add_edge(10, 20, label="k", edge_type="key_locked")

    start_room = SimpleNamespace(
        semantic_grid=np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32),
        doors={"N": False, "S": False, "E": True, "W": False},
        has_boss=False,
        has_triforce=False,
        is_start=True,
        graph_node_id=10,
        node_label="s",
    )
    goal_room = SimpleNamespace(
        semantic_grid=np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32),
        doors={"N": False, "S": False, "E": False, "W": True},
        has_boss=False,
        has_triforce=True,
        is_start=False,
        graph_node_id=20,
        node_label="t",
    )
    dungeon = SimpleNamespace(
        graph=graph,
        rooms={
            (0, 0): start_room,
            (0, 1): goal_room,
        },
    )

    base_graph = _extract_graph_from_dungeon(dungeon)
    sample = _build_room_graph_sample(dungeon, (0, 0), start_room, base_graph)

    assert sample["current_node_idx"] == base_graph["node_to_idx"][10]
    assert sample["boundary_constraints"].shape == (8,)
    assert sample["boundary_constraints"][4] == 1.0
    assert sample["boundary_constraints"][5] == 1.0
    assert sample["room_topology_map"].shape == (18, ROOM_HEIGHT, ROOM_WIDTH)
    assert float(sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["door_e"]].sum()) > 0.0
    assert float(sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["gated_e"]].sum()) > 0.0


def test_fit_room_grid_transposes_swapped_room_shape_instead_of_cropping():
    swapped = np.arange(ROOM_WIDTH * ROOM_HEIGHT, dtype=np.int32).reshape(ROOM_WIDTH, ROOM_HEIGHT)

    fitted = fit_room_grid(swapped)

    assert fitted.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert np.array_equal(fitted, swapped.transpose())


def test_room_graph_sample_uses_actual_room_trace_for_traversability():
    graph = nx.DiGraph()
    graph.add_node(10, label="s", is_start=True)
    graph.add_node(20, label="e")
    graph.add_edge(10, 20, label="", edge_type="open")

    wall = int(SEMANTIC_PALETTE["WALL"])
    floor = int(SEMANTIC_PALETTE["FLOOR"])
    start_tile = int(SEMANTIC_PALETTE["START"])
    room_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), wall, dtype=np.int32)
    room_grid[1, 1:ROOM_WIDTH] = floor
    room_grid[1, 1] = start_tile

    start_room = SimpleNamespace(
        semantic_grid=room_grid,
        doors={"N": False, "S": False, "E": True, "W": False},
        has_boss=False,
        has_triforce=False,
        is_start=True,
        graph_node_id=10,
        node_label="s",
    )
    next_room = SimpleNamespace(
        semantic_grid=np.full((ROOM_HEIGHT, ROOM_WIDTH), floor, dtype=np.int32),
        doors={"N": False, "S": False, "E": False, "W": True},
        has_boss=False,
        has_triforce=False,
        is_start=False,
        graph_node_id=20,
        node_label="e",
    )
    dungeon = SimpleNamespace(
        graph=graph,
        rooms={
            (0, 0): start_room,
            (0, 1): next_room,
        },
    )

    base_graph = _extract_graph_from_dungeon(dungeon)
    sample = _build_room_graph_sample(dungeon, (0, 0), start_room, base_graph)
    traversability = sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["traversability"]]

    assert float(traversability[1, 1:ROOM_WIDTH].sum()) >= float(ROOM_WIDTH - 1)
    assert traversability[ROOM_HEIGHT // 2, ROOM_WIDTH // 2] == 0.0


def test_room_graph_sample_uses_validator_plan_for_locked_exit_key_room():
    graph = nx.DiGraph()
    graph.add_node(10, label="s", is_start=True)
    graph.add_node(20, label="")
    graph.add_edge(10, 20, label="k", edge_type="key_locked")

    wall = int(SEMANTIC_PALETTE["WALL"])
    floor = int(SEMANTIC_PALETTE["FLOOR"])
    start_tile = int(SEMANTIC_PALETTE["START"])
    key_tile = int(SEMANTIC_PALETTE["KEY_SMALL"])

    room_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), wall, dtype=np.int32)
    room_grid[1, 1:ROOM_WIDTH] = floor
    room_grid[1:7, 3] = floor
    room_grid[1, 1] = start_tile
    room_grid[6, 3] = key_tile

    key_room = SimpleNamespace(
        semantic_grid=room_grid,
        doors={"N": False, "S": False, "E": True, "W": False},
        has_boss=False,
        has_triforce=False,
        is_start=True,
        graph_node_id=10,
        node_label="s,k",
    )
    next_room = SimpleNamespace(
        semantic_grid=np.full((ROOM_HEIGHT, ROOM_WIDTH), floor, dtype=np.int32),
        doors={"N": False, "S": False, "E": False, "W": True},
        has_boss=False,
        has_triforce=False,
        is_start=False,
        graph_node_id=20,
        node_label="",
    )
    dungeon = SimpleNamespace(
        graph=graph,
        rooms={
            (0, 0): key_room,
            (0, 1): next_room,
        },
    )

    base_graph = _extract_graph_from_dungeon(dungeon)
    sample = _build_room_graph_sample(dungeon, (0, 0), key_room, base_graph)
    traversability = sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["traversability"]]

    assert traversability[6, 3] == 1.0


def test_room_graph_sample_uses_validator_plan_for_soft_locked_enemy_room():
    graph = nx.DiGraph()
    graph.add_node(5, label="")
    graph.add_node(10, label="e")
    graph.add_node(20, label="")
    graph.add_edge(5, 10, label="", edge_type="open")
    graph.add_edge(10, 20, label="l", edge_type="soft_locked")

    wall = int(SEMANTIC_PALETTE["WALL"])
    floor = int(SEMANTIC_PALETTE["FLOOR"])
    enemy_tile = int(SEMANTIC_PALETTE["ENEMY"])

    room_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), wall, dtype=np.int32)
    room_grid[1, :] = floor
    room_grid[1:7, 3] = floor
    room_grid[6, 3] = enemy_tile

    prev_room = SimpleNamespace(
        semantic_grid=np.full((ROOM_HEIGHT, ROOM_WIDTH), floor, dtype=np.int32),
        doors={"N": False, "S": False, "E": True, "W": False},
        has_boss=False,
        has_triforce=False,
        is_start=False,
        graph_node_id=5,
        node_label="",
    )
    combat_room = SimpleNamespace(
        semantic_grid=room_grid,
        doors={"N": False, "S": False, "E": True, "W": True},
        has_boss=False,
        has_triforce=False,
        is_start=False,
        graph_node_id=10,
        node_label="e",
    )
    next_room = SimpleNamespace(
        semantic_grid=np.full((ROOM_HEIGHT, ROOM_WIDTH), floor, dtype=np.int32),
        doors={"N": False, "S": False, "E": False, "W": True},
        has_boss=False,
        has_triforce=False,
        is_start=False,
        graph_node_id=20,
        node_label="",
    )
    dungeon = SimpleNamespace(
        graph=graph,
        rooms={
            (0, 0): prev_room,
            (0, 1): combat_room,
            (0, 2): next_room,
        },
    )

    base_graph = _extract_graph_from_dungeon(dungeon)
    sample = _build_room_graph_sample(dungeon, (0, 1), combat_room, base_graph)
    traversability = sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["traversability"]]

    assert traversability[6, 3] == 1.0
