import numpy as np
import networkx as nx
import pytest

from src.core.definitions import SEMANTIC_PALETTE
from src.evaluation.validator import ExternalValidator
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv, ZeldaValidator
from src.validation.end_to_end import build_end_to_end_validation_report
from src.validation.pacing import evaluate_solution_path_pacing


def _valid_grid() -> np.ndarray:
    grid = np.full((5, 5), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    grid[1, 1] = int(SEMANTIC_PALETTE["START"])
    grid[3, 3] = int(SEMANTIC_PALETTE["TRIFORCE"])
    return grid


def test_end_to_end_contract_requires_all_applicable_exact_stages():
    report = build_end_to_end_validation_report(
        dungeon_grid=_valid_grid(),
        graph_validation={
            "solvable": True,
            "all_rooms_reachable": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
        spatial_validation={"final_spatial_edge_records_broken": 0},
        tile_validation={
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
        logicnet_agreement=True,
    )

    assert report.accepted is True
    assert report.failed_stages == []
    assert report.indeterminate_stages == []
    report.require_accepted()


def test_end_to_end_contract_rejects_invalid_representation_and_indeterminate_oracle():
    grid = _valid_grid().astype(np.float32)
    grid[0, 0] = 999.5
    report = build_end_to_end_validation_report(
        dungeon_grid=grid,
        graph_validation={
            "solvable": True,
            "all_rooms_reachable": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
        spatial_validation={"final_spatial_edge_records_broken": 0},
        tile_validation={
            "solvable": None,
            "is_exact": False,
            "termination_status": "budget_exhausted",
        },
    )

    assert report.accepted is False
    assert report.failed_stages == ["representation"]
    assert report.indeterminate_stages == ["tile_solvability"]
    with pytest.raises(RuntimeError, match="representation"):
        report.require_accepted()


def test_graph_stage_is_not_required_for_untyped_topology_api_inputs():
    report = build_end_to_end_validation_report(
        dungeon_grid=_valid_grid(),
        graph_validation={
            "solvable": None,
            "termination_status": "not_applicable_missing_roles",
        },
        spatial_validation={"final_spatial_edge_records_broken": 0},
        tile_validation={
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
    )

    assert report.graph_progression.applicable is False
    assert report.accepted is True


def test_graph_oracle_replays_consumable_key_route():
    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("key", type="KEY", key_id="small-1")
    graph.add_node("lock", type="LOCK")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "key", edge_type="path")
    graph.add_edge(
        "key",
        "lock",
        edge_type="key_locked",
        key_required="small-1",
    )
    graph.add_edge("lock", "goal", edge_type="path")

    result = ExternalValidator(mode="full").validate(graph)

    assert result.is_solvable is True
    assert result.solution_path == ["start", "key", "lock", "goal"]
    assert result.route_replay_status == "verified"
    assert result.route_replay_error == ""


def test_solved_graph_without_route_certificate_is_indeterminate():
    report = build_end_to_end_validation_report(
        dungeon_grid=_valid_grid(),
        graph_validation={
            "solvable": True,
            "all_rooms_reachable": True,
            "termination_status": "solved",
        },
        spatial_validation={"final_spatial_edge_records_broken": 0},
        tile_validation={
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
    )

    assert report.accepted is False
    assert report.indeterminate_stages == ["graph_progression"]


def test_requested_solver_consistency_is_fail_closed_and_distinguishes_budget():
    mismatch = build_end_to_end_validation_report(
        dungeon_grid=_valid_grid(),
        graph_validation={
            "solvable": True,
            "all_rooms_reachable": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
        spatial_validation={"final_spatial_edge_records_broken": 0},
        tile_validation={
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
            "path_length": 8,
            "path_cost": 11.5,
            "solver_consistency_status": "path_cost_mismatch",
            "solver_consistency_path_length": 7,
            "solver_consistency_path_cost": 10.0,
        },
    )
    assert mismatch.accepted is False
    assert mismatch.failed_stages == ["solver_consistency"]
    assert mismatch.solver_consistency.details["astar_path_cost"] == 11.5
    assert mismatch.solver_consistency.details["dijkstra_path_cost"] == 10.0

    indeterminate = build_end_to_end_validation_report(
        dungeon_grid=_valid_grid(),
        graph_validation={
            "solvable": True,
            "all_rooms_reachable": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
        spatial_validation={"final_spatial_edge_records_broken": 0},
        tile_validation={
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
            "path_length": 8,
            "solver_consistency_status": "indeterminate_budget_exhausted",
        },
    )
    assert indeterminate.accepted is False
    assert indeterminate.indeterminate_stages == ["solver_consistency"]


def test_solver_consistency_compares_accumulated_state_space_cost():
    palette = SEMANTIC_PALETTE
    wall = int(palette["WALL"])
    floor = int(palette["FLOOR"])
    grid = np.full((5, 7), wall, dtype=np.int64)
    grid[2, 1:6] = floor
    grid[1, 1:6] = floor
    grid[2, 1] = int(palette["START"])
    grid[2, 3] = int(palette["ENEMY"])
    grid[2, 5] = int(palette["TRIFORCE"])

    result = ZeldaValidator().validate_single(
        grid,
        solver_timeout=10_000,
        verify_dijkstra_consistency=True,
    )

    assert result.is_solvable is True
    assert result.path_cost == pytest.approx(6.0)
    assert result.solver_consistency_path_cost == pytest.approx(6.0)
    assert result.solver_consistency_status == "consistent"
    assert result.route_replay_status == "verified"
    assert result.route_replay_path_cost == pytest.approx(result.path_cost)


def test_route_replay_rejects_a_reconstructed_path_through_a_wall():
    palette = SEMANTIC_PALETTE
    wall = int(palette["WALL"])
    floor = int(palette["FLOOR"])
    grid = np.full((5, 5), wall, dtype=np.int64)
    grid[2, 1] = int(palette["START"])
    grid[2, 2] = wall
    grid[2, 3] = int(palette["TRIFORCE"])
    grid[1, 1:4] = floor

    env = ZeldaLogicEnv(grid)
    try:
        solver = StateSpaceAStar(env, timeout=1_000)
        verified, error, _final_state, replay_cost = solver.verify_position_path(
            [(2, 1), (2, 2), (2, 3)]
        )
    finally:
        env.close()

    assert verified is False
    assert "no legal transition" in error
    assert replay_cost is None


@pytest.mark.parametrize("mechanic", ["locked_door", "push_block"])
def test_public_validator_replays_stateful_tile_mechanics(mechanic):
    palette = SEMANTIC_PALETTE
    wall = int(palette["WALL"])
    floor = int(palette["FLOOR"])
    grid = np.full((7, 9), floor, dtype=np.int64)
    grid[[0, -1], :] = wall
    grid[:, [0, -1]] = wall
    grid[1:6, 4] = wall
    grid[3, 1] = int(palette["START"])
    grid[3, 7] = int(palette["TRIFORCE"])

    if mechanic == "locked_door":
        grid[2, 2] = int(palette["KEY_SMALL"])
        grid[3, 4] = int(palette["DOOR_LOCKED"])
    else:
        grid[3, 4] = int(palette["BLOCK"])

    result = ZeldaValidator().validate_single(grid, solver_timeout=10_000)

    assert result.is_solvable is True
    assert result.route_replay_status == "verified"
    assert result.route_replay_error == ""
    assert result.route_replay_path_cost == pytest.approx(result.path_cost)
    if mechanic == "locked_door":
        assert result.final_inventory["items_collected"] == 1
        assert result.final_inventory["doors_opened"] == 1
    else:
        assert (3, 4) in result.path


def test_public_validator_replays_cross_floor_stair_transition():
    palette = SEMANTIC_PALETTE
    wall = int(palette["WALL"])
    floor = int(palette["FLOOR"])
    stair = int(palette["STAIR"])
    grid = np.full((16, 22), wall, dtype=np.int64)
    grid[1:15, 1:10] = floor
    grid[1:15, 12:21] = floor
    grid[7, 5] = int(palette["START"])
    grid[7, 9] = stair
    grid[7, 12] = stair
    grid[7, 17] = int(palette["TRIFORCE"])

    graph = nx.DiGraph()
    graph.add_node(0, position=(0, 0, 0))
    graph.add_node(1, position=(0, 1, 1))
    graph.add_edge(0, 1, edge_type="STAIRS")
    result = ZeldaValidator().validate_single(
        grid,
        graph=graph,
        room_positions={(0, 0): (0, 0), (0, 1): (0, 11)},
        room_to_node={(0, 0): 0, (0, 1): 1},
        node_to_room={0: (0, 0), 1: (0, 1)},
        solver_timeout=10_000,
    )

    assert result.is_solvable is True
    assert result.route_replay_status == "verified"
    assert result.route_replay_path_cost == pytest.approx(result.path_cost)
    assert result.final_inventory["current_floor"] == 1
    assert (7, 12) in result.path


def test_solved_tile_oracle_without_route_certificate_is_indeterminate():
    report = build_end_to_end_validation_report(
        dungeon_grid=_valid_grid(),
        graph_validation={
            "solvable": True,
            "all_rooms_reachable": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
        spatial_validation={"final_spatial_edge_records_broken": 0},
        tile_validation={
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
        },
    )

    assert report.accepted is False
    assert report.indeterminate_stages == ["route_replay"]


def test_pacing_reports_unsmoothed_landmark_corridor_costs():
    graph = nx.DiGraph()
    graph.add_node("start", node_type="START")
    graph.add_node("key", node_type="KEY")
    graph.add_node("corridor", node_type="ROOM")
    graph.add_node("lock", node_type="LOCK")
    graph.add_node("boss", node_type="BOSS")
    graph.add_edge("start", "key", traversal_cost=1.0)
    graph.add_edge("key", "corridor", traversal_cost=2.0)
    graph.add_edge("corridor", "lock", traversal_cost=3.0)
    graph.add_edge("lock", "boss", traversal_cost=4.0)

    metrics = evaluate_solution_path_pacing(
        graph,
        ["start", "key", "corridor", "lock", "boss"],
    )

    assert metrics["pacing_total_graph_path_cost"] == pytest.approx(10.0)
    assert metrics["pacing_landmark_segments"] == [
        {
            "source_index": 1,
            "target_index": 3,
            "source_role": "KEY",
            "target_role": "LOCK",
            "corridor_edge_count": 2,
            "intermediate_room_count": 1,
            "path_cost": 5.0,
        },
        {
            "source_index": 3,
            "target_index": 4,
            "source_role": "LOCK",
            "target_role": "BOSS",
            "corridor_edge_count": 1,
            "intermediate_room_count": 0,
            "path_cost": 4.0,
        },
    ]


def test_spatial_topology_mismatch_cannot_hide_behind_edge_integrity():
    report = build_end_to_end_validation_report(
        dungeon_grid=_valid_grid(),
        graph_validation={
            "solvable": True,
            "all_rooms_reachable": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
        spatial_validation={
            "final_spatial_edge_records_broken": 0,
            "spatial_topology_invariant_preservation_score": 0.8,
            "spatial_topology_exact_invariants_preserved": False,
        },
        tile_validation={
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
    )

    assert report.accepted is False
    assert report.failed_stages == ["spatial_realization"]


def test_attached_global_state_proof_is_optional_but_fail_closed_when_present():
    common = {
        "dungeon_grid": _valid_grid(),
        "graph_validation": {
            "solvable": True,
            "all_rooms_reachable": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
        "spatial_validation": {"final_spatial_edge_records_broken": 0},
        "tile_validation": {
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
            "route_replay_status": "verified",
        },
    }
    ordinary = build_end_to_end_validation_report(**common)
    assert ordinary.global_state_progression.applicable is False
    assert ordinary.accepted is True

    deadlocked = build_end_to_end_validation_report(
        **common,
        global_state_validation={
            "accepted": False,
            "complete": True,
            "termination_status": "unreachable_contract",
        },
    )
    assert deadlocked.global_state_progression.applicable is True
    assert deadlocked.failed_stages == ["global_state_progression"]
    assert deadlocked.accepted is False

    exhausted = build_end_to_end_validation_report(
        **common,
        global_state_validation={
            "accepted": False,
            "complete": False,
            "termination_status": "budget_exhausted",
        },
    )
    assert exhausted.indeterminate_stages == ["global_state_progression"]
