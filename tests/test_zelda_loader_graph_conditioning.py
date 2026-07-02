from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    GRAPH_TPE_DIM,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
)
from src.pipeline.graph_features import compute_rrwp_edge_features
from src.pipeline.spatial_utils import fit_room_grid
from src.pipeline.room_topology_conditioning import (
    ROOM_TOPOLOGY_CHANNELS,
    _state_key,
    _room_local_state_search,
    _initial_state_for_sequence,
    build_puzzle_stage_condition_metadata,
    apply_puzzle_structure_control_to_conditioning,
    apply_puzzle_structure_dropout_batch,
    build_topology_loss_focus_map,
    build_room_semantic_anchor_points,
    build_semantic_room_plan_trace,
    build_room_topology_condition_map,
    infer_puzzle_room_structure_enabled,
)
from src.simulation.validator import GameState, ZeldaLogicEnv
from src.zelda_data.zelda_loader import (
    DungeonBatchSampler,
    ZeldaDungeonDataset,
    _build_room_graph_sample,
    _extract_graph_from_dungeon,
    graph_collate_fn,
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
    assert "edge_rrwp" in graph
    assert "node_positions" in graph
    assert tuple(graph["node_features"].shape) == (2, GRAPH_NODE_FEATURE_DIM)
    assert tuple(graph["edge_features"].shape) == (1, GRAPH_EDGE_FEATURE_DIM)
    assert tuple(graph["edge_rrwp"].shape) == (1, GRAPH_TPE_DIM)
    assert tuple(graph["tpe"].shape) == (2, 8)
    assert tuple(graph["node_positions"].shape) == (2, 2)
    assert graph["node_to_idx"] == {10: 0, 20: 1}
    assert graph["target_idx"] == -1
    assert graph["key_lock_pairs"] == []


def test_compute_rrwp_edge_features_preserves_edge_order_and_invalid_rows():
    edge_index = torch.tensor([[0, 99, 1], [1, 0, 2]], dtype=torch.long)

    rrwp = compute_rrwp_edge_features(
        edge_index,
        num_nodes=3,
        steps=2,
        device=torch.device("cpu"),
    )

    assert tuple(rrwp.shape) == (3, 2)
    assert rrwp[0, 0].item() == pytest.approx(0.5)
    assert rrwp[1].sum().item() == pytest.approx(0.0)
    assert rrwp[2, 0].item() == pytest.approx(0.5)


def test_dungeon_batch_sampler_groups_room_samples_by_dungeon_variant():
    class _Dataset(torch.utils.data.Dataset):
        sample_metadata = [
            {"dungeon_id": "d1_v1", "current_node_idx": 1},
            {"dungeon_id": "d2_v1", "current_node_idx": 0},
            {"dungeon_id": "d1_v1", "current_node_idx": 0},
        ]

        def __len__(self):
            return len(self.sample_metadata)

        def __getitem__(self, idx):
            return idx

    sampler = DungeonBatchSampler.from_dataset(_Dataset(), shuffle=False)
    batches = list(iter(sampler))

    assert batches == [[2, 0], [1]]


def test_dungeon_batch_sampler_preserves_graph_fields_in_batches():
    class _Dataset(torch.utils.data.Dataset):
        sample_metadata = [
            {"dungeon_id": "d1_v1", "current_node_idx": 0},
            {"dungeon_id": "d1_v1", "current_node_idx": 1},
        ]

        def __len__(self):
            return len(self.sample_metadata)

        def __getitem__(self, idx):
            graph = {
                "node_features": torch.zeros(2, GRAPH_NODE_FEATURE_DIM),
                "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
                "edge_features": torch.zeros(1, GRAPH_EDGE_FEATURE_DIM),
                "edge_rrwp": torch.zeros(1, GRAPH_TPE_DIM),
                "tpe": torch.zeros(2, GRAPH_TPE_DIM),
            }
            room = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH)
            return room, graph

    dataset = _Dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=DungeonBatchSampler.from_dataset(dataset, shuffle=False),
        collate_fn=graph_collate_fn,
    )

    _rooms, graph_list = next(iter(loader))

    assert len(graph_list) == 2
    for graph in graph_list:
        assert graph["edge_index"] is not None
        assert tuple(graph["edge_index"].shape) == (2, 1)
        assert tuple(graph["node_features"].shape) == (2, GRAPH_NODE_FEATURE_DIM)
        assert tuple(graph["edge_rrwp"].shape) == (1, GRAPH_TPE_DIM)
        assert tuple(graph["tpe"].shape) == (2, GRAPH_TPE_DIM)
        assert tuple(graph["batch_idx"].shape) == (2,)
        assert torch.equal(graph["batch_idx"], torch.zeros(2, dtype=torch.long))


def test_graph_collate_adds_per_graph_batch_idx_without_mutating_source():
    graph = {
        "node_features": torch.zeros(3, GRAPH_NODE_FEATURE_DIM),
        "edge_index": torch.empty(2, 0, dtype=torch.long),
    }

    _rooms, graph_list = graph_collate_fn([(torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH), graph)])

    assert "batch_idx" not in graph
    assert torch.equal(graph_list[0]["batch_idx"], torch.zeros(3, dtype=torch.long))


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
    assert float(sample["logic_source_mask"].sum()) == 0.0
    assert float(sample["logic_target_mask"].sum()) > 0.0


def test_dataset_graph_extraction_pairs_identified_key_with_locked_gate():
    graph = nx.DiGraph()
    graph.add_node(10, label="s", type="START", is_start=True)
    graph.add_node(20, label="k", type="KEY", has_key=True, key_id=7)
    graph.add_node(30, label="e", type="ROOM")
    graph.add_node(40, label="t", type="GOAL", is_triforce=True)
    graph.add_edge(10, 20, edge_type="PATH")
    graph.add_edge(
        20,
        30,
        edge_type="LOCKED",
        key_required=7,
        requires_key_count=1,
    )
    graph.add_edge(30, 40, edge_type="PATH")

    rooms = {}
    for col, node_id in enumerate((10, 20, 30, 40)):
        rooms[(0, col)] = SimpleNamespace(graph_node_id=node_id)
    dungeon = SimpleNamespace(graph=graph, rooms=rooms)

    extracted = _extract_graph_from_dungeon(dungeon)

    assert extracted["target_idx"] == extracted["node_to_idx"][40]
    assert extracted["key_lock_pairs"] == [
        (extracted["node_to_idx"][20], extracted["node_to_idx"][30])
    ]


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


def test_room_graph_sample_exposes_puzzle_structure_flag():
    graph = nx.DiGraph()
    graph.add_node(10, label="p", type="puzzle", has_puzzle=True)

    room_grid = np.ones((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
    room_grid[4:8, 4] = int(SEMANTIC_PALETTE["BLOCK"])
    puzzle_room = SimpleNamespace(
        semantic_grid=room_grid,
        doors={"N": False, "S": False, "E": False, "W": False},
        has_boss=False,
        has_triforce=False,
        is_start=False,
        graph_node_id=10,
        node_label="p",
    )
    dungeon = SimpleNamespace(graph=graph, rooms={(0, 0): puzzle_room})

    base_graph = _extract_graph_from_dungeon(dungeon)
    sample = _build_room_graph_sample(dungeon, (0, 0), puzzle_room, base_graph)

    assert sample["has_puzzle"] is True
    assert sample["puzzle_room_structure_enabled"] is True
    assert infer_puzzle_room_structure_enabled(room_grid, {"has_puzzle": True}) is True


def test_room_graph_sample_emits_ordered_puzzle_stage_condition():
    graph = nx.DiGraph()
    graph.add_node(10, label="p", type="puzzle", has_puzzle=True)
    graph.add_node(20, label="g", is_triforce=True, has_goal=True)
    graph.add_edge(10, 20, edge_type="switch_locked", label="switch")

    room_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room_grid[ROOM_HEIGHT // 2, ROOM_WIDTH // 2] = int(SEMANTIC_PALETTE["PUZZLE"])
    puzzle_room = SimpleNamespace(
        semantic_grid=room_grid,
        doors={"N": False, "S": False, "E": True, "W": True},
        has_boss=False,
        has_triforce=False,
        is_start=False,
        graph_node_id=10,
        node_label="p",
    )
    goal_room = SimpleNamespace(
        semantic_grid=np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32),
        doors={"N": False, "S": False, "E": False, "W": True},
        has_boss=False,
        has_triforce=True,
        is_start=False,
        graph_node_id=20,
        node_label="g",
    )
    dungeon = SimpleNamespace(graph=graph, rooms={(0, 0): puzzle_room, (0, 1): goal_room})

    base_graph = _extract_graph_from_dungeon(dungeon)
    sample = _build_room_graph_sample(
        dungeon,
        (0, 0),
        puzzle_room,
        base_graph,
        puzzle_stage_topology_enabled=True,
    )

    stage_condition = sample["puzzle_stage_condition"]
    assert stage_condition["sequence_required"] is True
    assert stage_condition["gate_family"] == "switch"
    assert len(stage_condition["stage_sequence"]) >= 1
    assert stage_condition["stage_sequence"][0]["kind"] == "push_block_to_switch"


def test_build_puzzle_stage_condition_metadata_builds_weighted_stage_trace():
    room_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room_grid[ROOM_HEIGHT // 2, ROOM_WIDTH // 2] = int(SEMANTIC_PALETTE["PUZZLE"])
    metadata = build_puzzle_stage_condition_metadata(
        room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
        start=(ROOM_HEIGHT // 2, 1),
        goal=(ROOM_HEIGHT // 2, ROOM_WIDTH - 2),
        required_doors={"W": True, "E": True},
        incoming_dirs={"W"},
        outgoing_dirs={"E"},
        edge_constraint_tokens={"E": {"switch_locked"}},
        room_role_flags={"has_puzzle": True},
        room_grid=room_grid,
        stage_trace_decay=0.5,
    )

    trace = metadata["stage_trace_mask"]
    assert metadata["gate_family"] == "switch"
    assert metadata["sequence_required"] is True
    assert len(metadata["stage_sequence"]) >= 1
    assert trace.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert float(trace.max()) > 0.0


def test_apply_puzzle_structure_dropout_batch_strips_blocks_and_flips_flag():
    real_maps = torch.zeros(1, 1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
    real_maps[:, :, :, :] = float(SEMANTIC_PALETTE["FLOOR"]) / 43.0
    real_maps[:, :, 6, 4] = float(SEMANTIC_PALETTE["BLOCK"]) / 43.0
    graph_list = [{"has_puzzle": True, "puzzle_room_structure_enabled": True}]

    augmented_maps, augmented_graphs = apply_puzzle_structure_dropout_batch(
        real_maps,
        graph_list,
        num_classes=44,
        dropout_prob=1.0,
    )

    tile_ids = (augmented_maps.squeeze(1) * 43.0).round().long()
    assert int(tile_ids[0, 6, 4].item()) == int(SEMANTIC_PALETTE["FLOOR"])
    assert augmented_graphs is not None
    assert augmented_graphs[0]["puzzle_room_structure_enabled"] is False
    assert augmented_graphs[0]["puzzle_structure_dropout_applied"] is True


def test_apply_puzzle_structure_control_to_conditioning_is_explicit():
    conditioning = torch.zeros(3, 8, dtype=torch.float32)
    enabled = apply_puzzle_structure_control_to_conditioning(
        conditioning,
        puzzle_structure_enabled=True,
        graph_conditioning_mode="node_sequence",
    )
    disabled = apply_puzzle_structure_control_to_conditioning(
        conditioning,
        puzzle_structure_enabled=False,
        graph_conditioning_mode="node_sequence",
    )
    pooled = apply_puzzle_structure_control_to_conditioning(
        conditioning[:1],
        puzzle_structure_enabled=False,
        graph_conditioning_mode="pooled",
    )

    assert tuple(enabled.shape) == (4, 8)
    assert tuple(disabled.shape) == (4, 8)
    assert not torch.allclose(enabled[-1], disabled[-1])
    assert tuple(pooled.shape) == (1, 8)
    assert not torch.allclose(pooled, conditioning[:1])


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
        incoming_dirs={"W"},
        outgoing_dirs={"E"},
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


def test_topology_loss_focus_map_emphasizes_markers_gates_and_trace():
    topo = build_room_topology_condition_map(
        start=(8, 1),
        goal=(8, ROOM_WIDTH - 2),
        required_doors={"N": False, "S": False, "E": True, "W": True},
        incoming_dirs={"W"},
        outgoing_dirs={"E"},
        edge_constraint_tokens={"E": {"bombable"}, "W": {"switch_locked"}},
        room_role_flags={"has_puzzle": True, "is_switch_puzzle": True},
    )

    focus = build_topology_loss_focus_map(
        torch.from_numpy(topo),
        marker_weight=2.0,
        trace_weight=0.75,
        dilation=1,
    )

    assert focus is not None
    assert tuple(focus.shape) == (1, ROOM_HEIGHT, ROOM_WIDTH)
    assert float(focus.max().item()) >= 2.0
    assert float(focus.sum().item()) > 0.0
    assert float(focus[0, 8, 1].item()) >= 2.0
    assert float(focus[0, ROOM_HEIGHT // 2, ROOM_WIDTH // 2].item()) > 0.0


def test_validator_initial_state_treats_non_local_item_gate_as_carried_item():
    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)

    state = _initial_state_for_sequence(
        room,
        (ROOM_HEIGHT // 2, 1),
        ["start", "door:E"],
        {"E": {"item_gate"}},
    )

    assert state.has_item is True


def test_validator_state_key_includes_current_floor():
    state_a = GameState(position=(1, 1), current_floor=0)
    state_b = GameState(position=(1, 1), current_floor=1)

    assert _state_key(state_a) != _state_key(state_b)


def test_room_local_state_search_respects_state_budget():
    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    env = ZeldaLogicEnv(room, render_mode=False)
    start_state = GameState(position=(1, 1))

    result = _room_local_state_search(
        env,
        start_state,
        (ROOM_HEIGHT - 2, ROOM_WIDTH - 2),
        max_states=1,
    )

    assert result is None


def test_semantic_room_plan_trace_falls_back_when_validator_budget_is_too_small():
    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)

    trace = build_semantic_room_plan_trace(
        room,
        required_doors={"N": False, "S": False, "E": True, "W": True},
        incoming_dirs={"W"},
        outgoing_dirs={"E"},
        edge_constraint_tokens={"E": {"switch_locked"}},
        room_role_flags={"has_puzzle": True},
        start=(ROOM_HEIGHT // 2, 1),
        goal=(ROOM_HEIGHT // 2, ROOM_WIDTH - 2),
        validator_plan_max_states=1,
    )

    assert trace.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert float(trace.sum()) > 0.0


def test_semantic_room_plan_trace_discards_partial_validator_sequences_before_fallback():
    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    fallback_trace = build_semantic_room_plan_trace(
        room,
        required_doors={"N": False, "S": False, "E": True, "W": False},
        incoming_dirs=set(),
        outgoing_dirs={"E"},
        edge_constraint_tokens={"E": {"switch_locked"}},
        room_role_flags={"has_puzzle": True},
        start=(ROOM_HEIGHT // 2, 1),
        goal=(ROOM_HEIGHT // 2, ROOM_WIDTH - 2),
        validator_plan_max_states=1,
    )
    partial_budget_trace = build_semantic_room_plan_trace(
        room,
        required_doors={"N": False, "S": False, "E": True, "W": False},
        incoming_dirs=set(),
        outgoing_dirs={"E"},
        edge_constraint_tokens={"E": {"switch_locked"}},
        room_role_flags={"has_puzzle": True},
        start=(ROOM_HEIGHT // 2, 1),
        goal=(ROOM_HEIGHT // 2, ROOM_WIDTH - 2),
        validator_plan_max_states=60,
    )

    assert np.array_equal(partial_budget_trace, fallback_trace)


def test_room_topology_condition_map_respects_validator_budget_for_synthetic_trace():
    low_budget = build_room_topology_condition_map(
        room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
        start=(ROOM_HEIGHT // 2, 1),
        goal=(ROOM_HEIGHT // 2, ROOM_WIDTH - 2),
        required_doors={"N": False, "S": False, "E": True, "W": False},
        incoming_dirs=set(),
        outgoing_dirs={"E"},
        edge_constraint_tokens={"E": {"switch_locked"}},
        room_role_flags={"has_puzzle": True},
        validator_plan_max_states=1,
    )
    partial_budget = build_room_topology_condition_map(
        room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
        start=(ROOM_HEIGHT // 2, 1),
        goal=(ROOM_HEIGHT // 2, ROOM_WIDTH - 2),
        required_doors={"N": False, "S": False, "E": True, "W": False},
        incoming_dirs=set(),
        outgoing_dirs={"E"},
        edge_constraint_tokens={"E": {"switch_locked"}},
        room_role_flags={"has_puzzle": True},
        validator_plan_max_states=60,
    )

    traversability_channel = ROOM_TOPOLOGY_CHANNELS["traversability"]
    assert not np.array_equal(
        partial_budget[traversability_channel],
        low_budget[traversability_channel],
    )
    assert float(np.sum(low_budget[traversability_channel])) > float(
        np.sum(partial_budget[traversability_channel])
    )


def test_room_topology_condition_map_localizes_semantic_role_anchors():
    role_flags = {
        "is_start": True,
        "has_goal": True,
        "has_key": True,
        "has_item": True,
        "has_puzzle": True,
    }
    anchors = build_room_semantic_anchor_points(
        start=(8, 1),
        goal=(8, ROOM_WIDTH - 2),
        required_doors={"W": True, "E": True},
        incoming_dirs={"W"},
        outgoing_dirs={"E"},
        room_role_flags=role_flags,
    )
    topo = build_room_topology_condition_map(
        start=(8, 1),
        goal=(8, ROOM_WIDTH - 2),
        required_doors={"W": True, "E": True},
        incoming_dirs={"W"},
        outgoing_dirs={"E"},
        room_role_flags=role_flags,
    )

    for role_name, anchor_name in (
        ("role_start", "start"),
        ("role_goal", "goal"),
        ("role_key", "key"),
        ("role_item", "item"),
        ("role_puzzle", "puzzle"),
    ):
        channel = topo[ROOM_TOPOLOGY_CHANNELS[role_name]]
        anchor = anchors[anchor_name]
        assert float(channel[anchor[0], anchor[1]]) == 1.0
        assert float(np.sum(channel == 1.0)) == 1.0
        assert float(channel[ROOM_HEIGHT // 2, ROOM_WIDTH // 2]) >= 0.15


def test_room_topology_condition_map_exposes_puzzle_subtype_channels():
    role_flags = {
        "has_puzzle": True,
        "is_tutorial_puzzle": True,
        "is_combat_puzzle": True,
        "is_complex_puzzle": True,
        "is_switch_puzzle": True,
    }
    topo = build_room_topology_condition_map(
        start=(8, 1),
        goal=(8, ROOM_WIDTH - 2),
        required_doors={"W": True, "E": True},
        incoming_dirs={"W"},
        outgoing_dirs={"E"},
        room_role_flags=role_flags,
    )
    puzzle_anchor = build_room_semantic_anchor_points(
        start=(8, 1),
        goal=(8, ROOM_WIDTH - 2),
        required_doors={"W": True, "E": True},
        incoming_dirs={"W"},
        outgoing_dirs={"E"},
        room_role_flags=role_flags,
    )["puzzle"]

    for role_name in (
        "role_puzzle",
        "role_tutorial_puzzle",
        "role_combat_puzzle",
        "role_complex_puzzle",
        "role_switch_puzzle",
    ):
        channel = topo[ROOM_TOPOLOGY_CHANNELS[role_name]]
        assert float(channel[puzzle_anchor[0], puzzle_anchor[1]]) == 1.0
        assert float(channel[ROOM_HEIGHT // 2, ROOM_WIDTH // 2]) >= 0.15


def test_room_graph_sample_promotes_puzzle_subtype_metadata_into_topology_channels():
    graph = nx.DiGraph()
    graph.add_node(
        10,
        label="p",
        type="complex_puzzle",
        has_puzzle=True,
        difficulty_rating="HARD",
    )

    puzzle_room = SimpleNamespace(
        semantic_grid=np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32),
        doors={"N": False, "S": False, "E": False, "W": False},
        has_boss=False,
        has_triforce=False,
        is_start=False,
        graph_node_id=10,
        node_label="p",
    )
    dungeon = SimpleNamespace(graph=graph, rooms={(0, 0): puzzle_room})

    base_graph = _extract_graph_from_dungeon(dungeon)
    sample = _build_room_graph_sample(dungeon, (0, 0), puzzle_room, base_graph)
    topo = sample["room_topology_map"]

    assert float(topo[ROOM_TOPOLOGY_CHANNELS["role_puzzle"]].sum()) > 0.0
    assert float(topo[ROOM_TOPOLOGY_CHANNELS["role_complex_puzzle"]].sum()) > 0.0


def test_fit_room_grid_transposes_swapped_room_shape_instead_of_cropping():
    swapped = np.arange(ROOM_WIDTH * ROOM_HEIGHT, dtype=np.int32).reshape(ROOM_WIDTH, ROOM_HEIGHT)

    fitted = fit_room_grid(swapped)

    assert fitted.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert np.array_equal(fitted, swapped.transpose())


def test_room_graph_sample_runtime_aligned_topology_avoids_room_grid_trace_leakage():
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

    assert sample["topology_supervision_mode"] == "runtime_aligned"
    assert traversability[ROOM_HEIGHT // 2, ROOM_WIDTH // 2] == 1.0
    assert float(traversability[1, 1:ROOM_WIDTH].sum()) < float(ROOM_WIDTH - 1)


def test_room_graph_sample_oracle_mode_uses_actual_room_trace_for_traversability():
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
    sample = _build_room_graph_sample(
        dungeon,
        (0, 0),
        start_room,
        base_graph,
        topology_supervision_mode="oracle_room_grid",
    )
    traversability = sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["traversability"]]

    assert float(traversability[1, 1:ROOM_WIDTH].sum()) >= float(ROOM_WIDTH - 1)
    assert traversability[ROOM_HEIGHT // 2, ROOM_WIDTH // 2] == 0.0


def test_room_graph_sample_oracle_mode_uses_validator_plan_for_locked_exit_key_room():
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
    sample = _build_room_graph_sample(
        dungeon,
        (0, 0),
        key_room,
        base_graph,
        topology_supervision_mode="oracle_room_grid",
    )
    traversability = sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["traversability"]]

    assert traversability[6, 3] == 1.0


def test_room_graph_sample_oracle_mode_uses_validator_plan_for_soft_locked_enemy_room():
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
    sample = _build_room_graph_sample(
        dungeon,
        (0, 1),
        combat_room,
        base_graph,
        topology_supervision_mode="oracle_room_grid",
    )
    traversability = sample["room_topology_map"][ROOM_TOPOLOGY_CHANNELS["traversability"]]

    assert traversability[6, 3] == 1.0
