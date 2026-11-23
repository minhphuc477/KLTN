"""Unassembled evidence and independent controls use only synthetic rooms."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.publication.control_attempt import generate_control_rooms
from src.publication.room_evidence import TrainingRoomReference, floor_structure, heldout_overlap, room_fingerprint


def test_reference_is_immutable_and_nearest_distance_is_exact():
    room = np.array([[1, 2, 1], [1, 2, 2]], dtype=np.int32)
    reference = TrainingRoomReference([room])
    room[:] = 2
    reference.rooms()[:] = 2
    assert reference.compare(np.array([[1, 2, 1], [1, 2, 2]]))["exact_training_match"]
    metrics = reference.compare(np.full((2, 3), 2))
    assert metrics["nearest_training_changed_tiles"] == 3
    assert metrics["nearest_training_hamming_fraction"] == 0.5
    assert room_fingerprint(room) != room_fingerprint(room.T)


def test_floor_components_are_descriptive_and_do_not_connect_through_gates():
    grid = np.array([[1, 10, 1], [1, 2, 999]])
    metrics = floor_structure(grid)
    assert metrics["floor_components"] == 2
    assert metrics["largest_floor_component_fraction"] == 2 / 3
    assert metrics["unknown_tile_fraction"] == 1 / 6


def test_training_float_storage_accepts_exact_ids_but_never_rounds_outputs():
    reference = TrainingRoomReference([np.ones((3, 4), dtype=np.float32)])
    assert reference.compare(np.ones((3, 4), dtype=np.int32))["exact_training_match"]
    for value in (1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite integral"):
            TrainingRoomReference([np.full((3, 4), value)])


def test_heldout_strata_keep_duplicates_without_removing_requested_rooms():
    seen = np.ones((3, 4), dtype=np.int32)
    unseen = seen.copy()
    unseen[0, 0] = 2
    evidence = heldout_overlap(TrainingRoomReference([seen]), [seen, seen, unseen])
    assert evidence["heldout_room_count"] == 3
    assert evidence["repeated_layout_count"] == 2
    assert evidence["unseen_layout_count"] == 1


@pytest.mark.parametrize("model", ["retrieval_training", "wfc_training", "pod_training", "ar_markov_training"])
def test_controls_repeat_without_using_global_rng_or_modifying_training_rooms(model):
    reference = TrainingRoomReference([np.ones((4, 5), dtype=np.int32)])
    first, _ = generate_control_rooms(reference, [0, 1], model_id=model, seed=19)
    np.random.seed(999)
    np.random.random(100)
    second, diagnostics = generate_control_rooms(reference, [0, 1], model_id=model, seed=19)
    for node in first:
        np.testing.assert_array_equal(first[node], second[node])
        assert diagnostics[node]
    first[0][:] = 42
    assert np.all(reference.rooms() == 1)


def test_wfc_fallback_cannot_be_reported_as_a_successful_control(monkeypatch):
    monkeypatch.setattr("src.generation.weighted_bayesian_wfc.WeightedBayesianWFC",
        lambda *args, **kwargs: SimpleNamespace(generate=lambda **kw: np.ones((2, 2), dtype=int),
            get_diagnostics=lambda: {"generation_succeeded": False, "required_fallback": True}))
    with pytest.raises(RuntimeError, match="did not complete"):
        generate_control_rooms(TrainingRoomReference([np.ones((2, 2), dtype=int)]), [0],
                               model_id="wfc_training", seed=7)


def test_compilation_reuses_anchors_without_creating_duplicate_starts():
    from src.core.definitions import SEMANTIC_PALETTE
    from src.generation.graph_constraint_enforcer import GraphConstraintEnforcer, RoomBoundary
    grid = np.ones((6, 7), dtype=np.int32)
    start, goal = SEMANTIC_PALETTE["START"], SEMANTIC_PALETTE["TRIFORCE"]
    grid[1, 1] = grid[1, 2] = start
    grid[4, 5] = goal
    enforcer = GraphConstraintEnforcer({"wall": 2, "floor": 1, "door": 10, "start": start, "goal": goal})
    boundary = RoomBoundary(0, 0, 6, 5, 0)
    first = enforcer._place_room_anchor(grid.copy(), boundary, {"is_start": True, "is_goal": True})
    second = enforcer._place_room_anchor(first.copy(), boundary, {"is_start": True, "is_goal": True})
    assert np.count_nonzero(first == start) == np.count_nonzero(first == goal) == 1
    np.testing.assert_array_equal(first, second)


def test_symmetry_invariant_nearest_neighbor_detects_reflected_and_rotated_copies():
    room = np.array([
        [2, 1, 1, 1],
        [2, 3, 1, 1],
        [1, 1, 20, 1],
    ], dtype=np.int32)
    reference = TrainingRoomReference([room])
    for name, candidate in (
        ("flip_h", np.flip(room, axis=1)),
        ("flip_v", np.flip(room, axis=0)),
        ("rot180", np.flip(room, axis=(0, 1))),
    ):
        metrics = reference.compare(candidate)
        assert metrics["symmetry_group"] == "V4_klein_four"
        assert metrics["exact_training_match"] is False
        assert metrics["symmetry_exact_training_match"] is True
        assert metrics["symmetry_nearest_transform"] == name
        assert metrics["symmetry_nearest_training_changed_tiles"] == 0
        assert metrics["symmetry_nearest_training_hamming_fraction"] == 0.0


def test_pod_and_ar_markov_controls_exercise_multi_tile_context_tables():
    from src.core.definitions import TileID
    wall, floor, block, door, enemy = (
        int(TileID.WALL),
        int(TileID.FLOOR),
        int(TileID.BLOCK),
        int(TileID.DOOR_OPEN),
        int(TileID.ENEMY),
    )
    rooms = []
    for k in range(3):
        g = np.full((6, 7), floor, dtype=np.int32)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = wall
        g[0, 3] = door
        g[2 + (k % 2), 2 + k] = block
        g[4, 4] = enemy
        rooms.append(g)
    reference = TrainingRoomReference(rooms)
    vocab = {wall, floor, block, door, enemy}

    for model in ("pod_training", "ar_markov_training"):
        out_a, diag_a = generate_control_rooms(reference, [0, 1], model_id=model, seed=101)
        out_b, _ = generate_control_rooms(reference, [0, 1], model_id=model, seed=101)
        for nid in (0, 1):
            np.testing.assert_array_equal(out_a[nid], out_b[nid])
            assert set(np.unique(out_a[nid]).tolist()).issubset(vocab)
            assert diag_a[nid]["generation_succeeded"] is True
        assert diag_a[0].get("unique_3x3_contexts", diag_a[0].get("causal_contexts", 0)) > 1
