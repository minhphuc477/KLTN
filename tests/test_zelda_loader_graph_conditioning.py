from types import SimpleNamespace

import networkx as nx
import numpy as np

from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
)
from src.pipeline.spatial_utils import fit_room_grid
from src.pipeline.room_topology_conditioning import ROOM_TOPOLOGY_CHANNELS, build_room_topology_condition_map
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
        "node_features": np.zeros((2, GRAPH_NODE_FEATURE_DIM), dtype=np.float32),
        "edge_index": np.array([[0], [1]], dtype=np.int64),
        "edge_attr": np.array([1], dtype=np.int64),
        "edge_features": np.zeros((1, GRAPH_EDGE_FEATURE_DIM), dtype=np.float32),
        "tpe": np.ones((2, 8), dtype=np.float32),
        "node_positions": np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "num_nodes": 2,
        "num_edges": 1,
        "start_node_id": 0,
        "node_to_idx": {10: 0, 20: 1},
    }]

    _room_tensor, graph = ZeldaDungeonDataset.__getitem__(dataset, 0)

    assert "tpe" in graph
    assert "edge_features" in graph
    assert "node_positions" in graph
    assert tuple(graph["node_features"].shape) == (2, GRAPH_NODE_FEATURE_DIM)
    assert tuple(graph["edge_features"].shape) == (1, GRAPH_EDGE_FEATURE_DIM)
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
    assert sample["edge_features"].shape[1] == GRAPH_EDGE_FEATURE_DIM
    assert sample["boundary_constraints"].shape == (8,)
    assert sample["boundary_constraints"][4] == 1.0
    assert sample["boundary_constraints"][5] == 1.0
    assert sample["room_topology_map"].shape == (ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH)
    assert float(sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["door_e"]].sum()) > 0.0
    assert float(sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["gated_e"]].sum()) > 0.0
    assert float(sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["gate_key_e"]].sum()) > 0.0


def test_room_graph_sample_resolves_symbolic_sector_theme_into_style_id():
    graph = nx.DiGraph()
    graph.graph["sector_theme"] = "ice_cavern"
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
    dungeon = SimpleNamespace(graph=graph, rooms={(0, 0): start_room, (0, 1): goal_room})

    base_graph = _extract_graph_from_dungeon(dungeon)
    sample = _build_room_graph_sample(dungeon, (0, 0), start_room, base_graph)

    assert sample["style_id"] == 2


def test_graph_extraction_preserves_one_way_direction_and_battery_features():
    graph = nx.DiGraph()
    graph.add_node(10, label="s", is_start=True, is_hub=True)
    graph.add_node(20, label="t", is_triforce=True, is_secret=True)
    graph.add_edge(
        10,
        20,
        edge_type="one_way",
        preferred_direction="forward",
        battery_id=2,
        switches_required=[7, 8],
    )

    room_a = SimpleNamespace(
        semantic_grid=np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32),
        doors={"N": False, "S": False, "E": True, "W": False},
        has_boss=False,
        has_triforce=False,
        is_start=True,
        graph_node_id=10,
        node_label="s",
    )
    room_b = SimpleNamespace(
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
            (0, 0): room_a,
            (0, 1): room_b,
        },
    )

    extracted = _extract_graph_from_dungeon(dungeon)

    assert tuple(extracted["node_features"].shape) == (2, GRAPH_NODE_FEATURE_DIM)
    assert tuple(extracted["edge_features"].shape) == (1, GRAPH_EDGE_FEATURE_DIM)
    # Directional one-way channels should be active.
    assert float(extracted["edge_features"][0, 11]) == 1.0
    assert float(extracted["edge_features"][0, 12]) == 0.0
    # Battery/switch cardinality channels should survive extraction.
    assert float(extracted["edge_features"][0, 14]) > 0.0
    assert float(extracted["edge_features"][0, 15]) == 1.0
    # Hub + secret node semantics should be exposed in the richer node schema.
    assert float(extracted["node_features"][0, 13]) == 1.0
    assert float(extracted["node_features"][1, 12]) == 1.0


def test_room_topology_condition_map_preserves_typed_gate_channels():
    topo = build_room_topology_condition_map(
        required_doors={"N": False, "S": False, "E": True, "W": True},
        edge_constraint_tokens={
            "E": {"bombable", "secret"},
            "W": {"switch", "state_block"},
        },
    )

    assert topo.shape == (ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH)
    assert float(topo[ROOM_TOPOLOGY_CHANNELS["gated_e"]].sum()) > 0.0
    assert float(topo[ROOM_TOPOLOGY_CHANNELS["gate_bomb_e"]].sum()) > 0.0
    assert float(topo[ROOM_TOPOLOGY_CHANNELS["gate_secret_e"]].sum()) > 0.0
    assert float(topo[ROOM_TOPOLOGY_CHANNELS["gated_w"]].sum()) > 0.0
    assert float(topo[ROOM_TOPOLOGY_CHANNELS["gate_switch_w"]].sum()) > 0.0


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
