from __future__ import annotations

import sys
import types

import networkx as nx
import numpy as np

from src.core.condition_encoder import create_condition_encoder
from src.evaluation.map_elites import CBSFeatureExtractor
from src.generation.entity_spawner import (
    EntitySpawner,
    create_room_semantics_from_graph,
    spawn_all_entities,
)
from scripts.export_manual_rich_topology_compare import build_manual_rich_topology_graph
from src.generation.evolutionary_director import networkx_to_mission_graph
from src.generation.grammar import EdgeType
from src.evaluation.validator import ExternalValidator
from src.simulation.validator import GraphGuidedValidator
from src.utils.stable_seed import stable_seed_offset
from src.utils.graph_utils import validate_goal_subgraph


def test_condition_encoder_factory_default_matches_config_default():
    encoder = create_condition_encoder(latent_dim=16, output_dim=32)

    assert encoder.global_encoder.gnn_type == "gcn"


def test_networkx_to_mission_graph_deduplicates_implied_reverse_edges_for_string_ids():
    graph = nx.DiGraph()
    graph.add_node("start", type="start", label="s")
    graph.add_node("goal", type="goal", label="t")
    graph.add_edge("start", "goal", edge_type="path")
    graph.add_edge(
        "goal",
        "start",
        edge_type="path",
        metadata={"implied_reverse": True},
    )

    converted = networkx_to_mission_graph(graph)

    assert len(converted.nodes) == 2
    assert len(converted.edges) == 1
    assert converted.edges[0].source == "start"
    assert converted.edges[0].target == "goal"
    assert converted.edges[0].edge_type == EdgeType.PATH


def test_built_in_manual_rich_topology_graph_preserves_direction():
    graph = build_manual_rich_topology_graph()

    assert graph.is_directed()
    assert nx.is_directed_acyclic_graph(graph)
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(1, 0)
    assert graph.has_edge(10, 11)
    assert not graph.has_edge(11, 10)


def test_built_in_manual_rich_topology_graph_passes_strict_goal_gauntlet():
    graph = build_manual_rich_topology_graph()

    is_valid, errors = validate_goal_subgraph(graph)

    assert is_valid, errors


def test_built_in_manual_rich_topology_graph_is_solvable_for_graph_validator():
    graph = build_manual_rich_topology_graph()

    result = ExternalValidator().validate(graph)

    assert result.is_solvable, result.failure_reason


def test_cbs_feature_cache_key_tracks_node_attributes(monkeypatch):
    fake_module = types.ModuleType("src.evaluation.cbs_fitness")

    def _fake_compute_cbs_fitness(graph, persona="balanced"):
        goal_count = sum(
            1
            for _, data in graph.nodes(data=True)
            if str(data.get("label", "")).strip().lower() == "t"
        )
        return {
            "confusion_ratio": float(goal_count),
            "room_entropy": 0.2 + (0.5 * goal_count),
        }

    fake_module.compute_cbs_fitness = _fake_compute_cbs_fitness
    monkeypatch.setitem(sys.modules, "src.evaluation.cbs_fitness", fake_module)

    extractor = CBSFeatureExtractor()

    graph_a = nx.DiGraph()
    graph_a.add_node("room_0", label="s")
    graph_a.add_node("room_1", label="t")
    graph_a.add_edge("room_0", "room_1", edge_type="open")

    graph_b = nx.DiGraph()
    graph_b.add_node("room_0", label="s")
    graph_b.add_node("room_1", label="")
    graph_b.add_edge("room_0", "room_1", edge_type="open")

    features_a_first = extractor.extract(graph_a)
    features_a_second = extractor.extract(graph_a)
    features_b = extractor.extract(graph_b)

    assert features_a_first == features_a_second
    assert features_a_first != features_b
    assert len(extractor._cache) == 2


def test_create_room_semantics_accepts_string_node_ids():
    semantics = create_room_semantics_from_graph(
        mission_graph={
            "nodes": {
                "boss-room": {
                    "type": "boss",
                    "difficulty": 0.85,
                    "is_boss": True,
                }
            }
        },
        node_id="boss-room",
        tension_curve=[0.1, 0.2, 0.3],
    )

    assert semantics.node_id == "boss-room"
    assert semantics.room_type == "boss"
    assert abs(float(semantics.difficulty) - 0.85) < 1e-9


def test_create_room_semantics_keeps_final_boss_classification_when_goal_is_present():
    """A final room can be both the triforce goal and the boss arena."""
    semantics = create_room_semantics_from_graph(
        mission_graph={
            "nodes": {
                "final": {
                    "type": "boss",
                    "is_boss": True,
                    "is_triforce": True,
                }
            }
        },
        node_id="final",
    )

    assert semantics.room_type == "boss"


def test_spawn_all_entities_uses_stable_room_seeds_for_string_node_ids(monkeypatch):
    recorded = []

    def _fake_spawn_entities(self, room_grid, room_semantics, room_bounds, seed=None):
        recorded.append((room_semantics.node_id, seed, tuple(room_grid.shape), room_bounds))
        return []

    monkeypatch.setattr(EntitySpawner, "spawn_entities", _fake_spawn_entities)

    dungeon_grid = np.zeros((2, 4), dtype=np.int64)
    mission_graph = {
        "nodes": {
            "room_a": {"type": "start", "label": "s"},
            "room_b": {"type": "boss", "label": "b"},
        }
    }
    layout_map = {
        "room_a": (0, 0, 1, 1),
        "room_b": (2, 0, 3, 1),
    }

    spawn_all_entities(
        dungeon_grid=dungeon_grid,
        mission_graph=mission_graph,
        layout_map=layout_map,
        seed=123,
    )

    assert recorded == [
        ("room_a", 123 + stable_seed_offset("room_a", modulo=100000), (2, 2), (0, 0, 1, 1)),
        ("room_b", 123 + stable_seed_offset("room_b", modulo=100000), (2, 2), (2, 0, 3, 1)),
    ]


def test_graph_guided_validator_normalizes_tuple_room_keys_deterministically():
    validator = GraphGuidedValidator()

    rooms = {
        (2, 0): {"name": "room_b"},
        (0, 0): {"name": "room_a"},
        "7": {"name": "room_existing"},
    }

    normalized_a = validator._normalize_rooms(rooms)
    normalized_b = validator._normalize_rooms(dict(reversed(list(rooms.items()))))

    assert normalized_a == normalized_b
    assert 7 in normalized_a
    assert normalized_a[7]["name"] == "room_existing"
    tuple_room_ids = sorted(room_id for room_id in normalized_a if room_id != 7)
    assert tuple_room_ids == [8, 9]
