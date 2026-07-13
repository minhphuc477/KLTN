import numpy as np
import pytest

from src.core.definitions import SEMANTIC_PALETTE
from src.validation.end_to_end import build_end_to_end_validation_report


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
        },
        spatial_validation={"final_spatial_edge_records_broken": 0},
        tile_validation={
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
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
        },
    )

    assert report.graph_progression.applicable is False
    assert report.accepted is True


def test_requested_solver_consistency_is_fail_closed_and_distinguishes_budget():
    mismatch = build_end_to_end_validation_report(
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
            "path_length": 8,
            "solver_consistency_status": "path_cost_mismatch",
            "solver_consistency_path_length": 7,
        },
    )
    assert mismatch.accepted is False
    assert mismatch.failed_stages == ["solver_consistency"]

    indeterminate = build_end_to_end_validation_report(
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
            "path_length": 8,
            "solver_consistency_status": "indeterminate_budget_exhausted",
        },
    )
    assert indeterminate.accepted is False
    assert indeterminate.indeterminate_stages == ["solver_consistency"]


def test_spatial_topology_mismatch_cannot_hide_behind_edge_integrity():
    report = build_end_to_end_validation_report(
        dungeon_grid=_valid_grid(),
        graph_validation={
            "solvable": True,
            "all_rooms_reachable": True,
            "termination_status": "solved",
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
        },
        "spatial_validation": {"final_spatial_edge_records_broken": 0},
        "tile_validation": {
            "solvable": True,
            "is_exact": True,
            "termination_status": "solved",
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
