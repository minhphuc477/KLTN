from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np

from src.core.definitions import (
    CHAR_TO_SEMANTIC,
    ID_TO_NAME,
    ROOM_TOPOLOGY_CHANNELS,
    SEMANTIC_TO_CHAR,
    TileID,
    semantic_grid_to_vglc_lines,
)
from src.core.logic_net import ROOM_TOPOLOGY_CHANNELS as LOGICNET_ROOM_TOPOLOGY_CHANNELS
from src.evaluation.pcbs_validation import (
    PCBS_BOUNDED_RATIONALITY_WEIGHTS,
    PCBS_READABILITY_WEIGHT_SOURCE,
    compute_pcbs_readability_metrics,
)
from src.evaluation.validator import AgentSimulator, SolvabilityChecker, ValidationState
from src.simulation.validation_helpers import SanityChecker


def test_semantic_export_preserves_entity_tiles_roundtrip():
    grid = np.asarray(
        [
            [
                int(TileID.START),
                int(TileID.TRIFORCE),
                int(TileID.KEY_SMALL),
                int(TileID.KEY_BOSS),
                int(TileID.KEY_ITEM),
                int(TileID.ITEM_MINOR),
                int(TileID.BOSS),
            ]
        ],
        dtype=np.int32,
    )

    line = semantic_grid_to_vglc_lines(grid)[0]
    restored = np.asarray([[int(CHAR_TO_SEMANTIC[ch]) for ch in line]], dtype=np.int32)

    assert restored.tolist() == grid.tolist()
    assert SEMANTIC_TO_CHAR[int(TileID.START)] != "F"
    assert SEMANTIC_TO_CHAR[int(TileID.TRIFORCE)] != "F"
    assert SEMANTIC_TO_CHAR[int(TileID.KEY_SMALL)] != "F"
    assert ID_TO_NAME[int(TileID.KEY_SMALL)] == "KEY_SMALL"
    assert ID_TO_NAME[int(TileID.ITEM_MINOR)] == "ITEM_MINOR"


def test_logicnet_uses_canonical_room_topology_channel_schema():
    assert LOGICNET_ROOM_TOPOLOGY_CHANNELS == ROOM_TOPOLOGY_CHANNELS
    assert len(LOGICNET_ROOM_TOPOLOGY_CHANNELS) > 11
    assert "role_tutorial_puzzle" in LOGICNET_ROOM_TOPOLOGY_CHANNELS


def test_agent_simulator_closed_set_uses_full_state_not_raw_hash(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node(0, label="s")
    graph.add_node(1, label="k")
    graph.add_node(2, label="")
    graph.add_node(3, label="t")
    graph.add_edge(0, 2, edge_type="open")
    graph.add_edge(0, 1, edge_type="open")
    graph.add_edge(1, 2, edge_type="open")
    graph.add_edge(2, 3, edge_type="key_locked")

    monkeypatch.setattr(ValidationState, "__hash__", lambda _self: 0)

    result = AgentSimulator(graph).find_path(max_states=100)

    assert result.is_solvable
    assert result.solution_path[-1] == 3


def test_agent_simulator_heuristic_uses_bound_undirected_cache(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node(0, label="s")
    graph.add_node(1, label="")
    graph.add_node(2, label="t")
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    simulator = AgentSimulator(graph)

    monkeypatch.setattr(
        simulator.graph,
        "to_undirected",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("uncached to_undirected call")),
    )

    assert simulator.heuristic(0, 2) == 2.0
    assert simulator.heuristic(1, 2) == 1.0


def test_agent_simulator_explicit_zero_node_override_is_not_treated_as_missing():
    graph = nx.DiGraph()
    graph.add_node(0, label="s")
    graph.add_node(1, label="")
    graph.add_node(2, label="t")
    graph.add_edge(0, 1, edge_type="open")
    graph.add_edge(1, 2, edge_type="open")

    simulator = AgentSimulator(graph)
    simulator.start_node = 1

    result = simulator.find_path(start_node=0, goal_node=2, max_states=20)

    assert result.is_solvable
    assert result.solution_path == [0, 1, 2]


def test_reachability_check_explicit_zero_node_override_is_not_treated_as_missing():
    graph = nx.DiGraph()
    graph.add_node(0, label="s")
    graph.add_node(1, label="")
    graph.add_node(2, label="t")
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    checker = SolvabilityChecker()
    ok, unreachable = checker.check_all_rooms_reachable(graph, start_node=0)

    assert ok
    assert unreachable == set()


def test_pcbs_readability_weights_are_reported_as_hand_tuned_metadata():
    metrics = compute_pcbs_readability_metrics(
        oracle={"success": True, "path_length": 10},
        pcbs_success=True,
        pcbs_solution_length=12,
        pcbs_trajectory_length=20,
        pcbs_states=10,
        timeout_pcbs=100,
        pcbs_metrics=SimpleNamespace(
            total_steps=20,
            unique_tiles_visited=15,
            confusion_index=0.5,
            navigation_entropy=0.4,
            cognitive_load=0.3,
        ),
        puzzle_stall_steps=2,
    )

    assert metrics["weight_source"] == PCBS_READABILITY_WEIGHT_SOURCE
    assert metrics["bounded_rationality_weights"] == PCBS_BOUNDED_RATIONALITY_WEIGHTS


def test_sanity_checker_rejects_sparse_walkable_maps_that_old_threshold_allowed():
    grid = np.full((10, 10), int(TileID.WALL), dtype=np.int32)
    walkable = [
        (0, 0, TileID.START),
        (0, 1, TileID.FLOOR),
        (0, 2, TileID.FLOOR),
        (0, 3, TileID.FLOOR),
        (0, 4, TileID.FLOOR),
        (0, 5, TileID.TRIFORCE),
    ]
    for row, col, tile_id in walkable:
        grid[row, col] = int(tile_id)

    is_valid, errors = SanityChecker(grid).check_all()

    assert not is_valid
    assert any("mostly blocked" in error for error in errors)
