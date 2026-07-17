import numpy as np
import pytest

from src.generation.entity_spawner import (
    EntitySpawner,
    EntityType,
    RoomSemantics,
)
from src.generation.graph_constraint_enforcer import GraphConstraintEnforcer, RoomBoundary


def _enforcer_config(**overrides):
    config = {"wall": 1, "floor": 0, "door": 2}
    config.update(overrides)
    return config


def test_adjacency_only_connections_fail_closed_by_default():
    enforcer = GraphConstraintEnforcer(_enforcer_config())

    with pytest.raises(ValueError, match="missing canonical edge metadata"):
        enforcer._get_valid_neighbors("a", {"adjacency": {"a": {"E": "b"}}})


def test_adjacency_only_connections_require_explicit_legacy_open_mode():
    enforcer = GraphConstraintEnforcer(
        _enforcer_config(allow_implicit_open_adjacency=True)
    )

    neighbors = enforcer._get_valid_neighbors(
        "a",
        {"adjacency": {"a": {"E": "b"}}},
    )

    assert neighbors == {
        "b": {"edge_type": "open", "_spatial_gate_here": True}
    }


def test_canonical_gate_metadata_wins_over_parallel_adjacency():
    enforcer = GraphConstraintEnforcer(_enforcer_config())

    neighbors = enforcer._get_valid_neighbors(
        "a",
        {
            "adjacency": {"a": {"E": "b"}},
            "edges": [("a", "b", {"edge_type": "locked"})],
        },
    )

    assert neighbors["b"]["edge_type"] == "locked"


@pytest.mark.parametrize("edge_type", ["hidden", "secret"])
def test_discovery_only_edges_do_not_crash_or_create_ordinary_doors(edge_type):
    enforcer = GraphConstraintEnforcer(_enforcer_config())

    neighbors = enforcer._get_valid_neighbors(
        "a",
        {"edges": [("a", "b", {"edge_type": edge_type})]},
    )

    assert neighbors == {}


def test_required_anchor_carves_a_minimal_interior_foothold_in_all_wall_room():
    enforcer = GraphConstraintEnforcer(_enforcer_config(start=3))
    grid = np.full((5, 5), 1, dtype=np.int64)
    boundary = RoomBoundary(0, 0, 4, 4, "start")

    anchored = enforcer._place_room_anchor(
        grid,
        boundary,
        {"node_type": "start"},
    )

    assert int(anchored[2, 2]) == 3


def test_boss_room_honors_has_treasure_independently():
    spawner = EntitySpawner({"floor_tile_id": 0, "min_enemy_distance": 0})
    semantics = RoomSemantics(
        node_id="boss",
        room_type="boss",
        difficulty=1.0,
        has_treasure=True,
    )

    entities = spawner.spawn_entities(
        np.zeros((9, 9), dtype=np.int64),
        semantics,
        (10, 20, 18, 28),
        seed=7,
    )

    bosses = [entity for entity in entities if entity.entity_type == EntityType.ENEMY_BOSS]
    chests = [entity for entity in entities if entity.entity_type == EntityType.CHEST]
    assert len(bosses) == 1
    assert len(chests) == 1
    assert (bosses[0].x, bosses[0].y) != (chests[0].x, chests[0].y)
