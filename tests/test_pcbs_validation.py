from __future__ import annotations

import json

import numpy as np

from src.core.definitions import TileID
from src.evaluation.pcbs_validation import (
    build_ieee_markdown_table,
    evaluate_astar_vs_pcbs,
    prepare_dungeon_grid_for_validation,
)


def test_prepare_dungeon_grid_for_validation_fixes_invalid_and_enclosed_void() -> None:
    grid = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 21, 1, 999, 0],
            [0, 1, 0, 22, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    prepared = prepare_dungeon_grid_for_validation(grid)

    assert int(prepared.grid[1, 3]) == int(TileID.FLOOR)
    assert int(prepared.grid[2, 2]) == int(TileID.WALL)
    assert prepared.diagnostics["invalid_tile_count"] == 1
    assert prepared.diagnostics["enclosed_void_tiles_filled"] == 1
    assert prepared.start == (1, 1)
    assert prepared.goal == (2, 3)


def test_prepare_dungeon_grid_for_validation_normalizes_duplicate_terminals() -> None:
    grid = np.array(
        [
            [2, 2, 2, 2, 2],
            [2, 21, 1, 21, 2],
            [2, 1, 1, 1, 2],
            [2, 22, 1, 22, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=np.int32,
    )

    prepared = prepare_dungeon_grid_for_validation(grid, start=(1, 1), goal=(3, 3))

    assert int(np.sum(prepared.grid == int(TileID.START))) == 1
    assert int(np.sum(prepared.grid == int(TileID.TRIFORCE))) == 1
    assert prepared.diagnostics["duplicate_start_tiles_removed"] == 1
    assert prepared.diagnostics["duplicate_goal_tiles_removed"] == 1


def test_evaluate_astar_vs_pcbs_emits_paper_metrics_and_markdown() -> None:
    grid = np.array(
        [
            [2, 2, 2, 2, 2],
            [2, 21, 43, 22, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=np.int32,
    )

    result = evaluate_astar_vs_pcbs(grid, persona="novice", timeout_astar=500, timeout_pcbs=200, seed=7)
    table = build_ieee_markdown_table(result, map_name="UnitTest")

    assert "validation_handoff" in result
    assert "oracle" in result
    assert "pcbs" in result
    assert "comparison" in result
    assert "puzzle_stall_steps" in result["pcbs"]
    assert "room_entropy" in result["pcbs"]
    assert "unique_rooms_visited" in result["pcbs"]
    assert result["pcbs"]["status"] in {"success", "failed", "budget_exhausted"}
    assert result["comparison"]["pcbs_status"] == result["pcbs"]["status"]
    assert "affordance_reactivations" in result["pcbs"]
    assert "inventory_change_events" in result["pcbs"]
    assert "bounded_rationality_index" in result["pcbs"]
    assert "readability_score" in result["pcbs"]
    assert "cognitive_effort_index" in result["pcbs"]
    assert "pcbs_outcome_class" in result["pcbs"]
    assert "pcbs_failure_driver" in result["pcbs"]
    assert "pcbs_dominant_pressure" in result["pcbs"]
    assert result["pcbs"]["pcbs_calibration_bucket"] in {
        "readable",
        "readable_but_costly",
        "bounded_gap",
        "hard_invalid",
    }
    assert "pcbs_outcome_class" in result["comparison"]
    assert "oracle_pcbs_path_delta" in result["comparison"]
    assert "| Map | Solver |" in table
    assert "Outcome" in table
    assert "P-CBS (novice)" in table
    json.dumps(result)


def test_evaluate_astar_vs_pcbs_separates_failed_trajectory_from_solution_path() -> None:
    grid = np.array(
        [
            [2, 2, 2, 2, 2],
            [2, 21, 43, 22, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=np.int32,
    )

    result = evaluate_astar_vs_pcbs(grid, persona="novice", timeout_astar=500, timeout_pcbs=1, seed=7)

    assert result["oracle"]["success"] is True
    assert result["pcbs"]["success"] is False
    assert result["pcbs"]["path_length"] == 0
    assert result["pcbs"]["trajectory_length"] >= 1
    assert result["pcbs"]["path_efficiency_ratio"] == 0.0
    assert result["pcbs"]["pcbs_calibration_bucket"] == "bounded_gap"
    assert result["pcbs"]["pcbs_outcome_class"] in {
        "bounded_budget_exhausted",
        "puzzle_readability_failure",
        "navigation_confusion_failure",
        "cognitive_load_failure",
        "bounded_failure_unclassified",
    }


def test_evaluate_astar_vs_pcbs_uses_calibration_artifact(tmp_path) -> None:
    grid = np.array(
        [
            [2, 2, 2, 2, 2],
            [2, 21, 43, 22, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=np.int32,
    )
    calibration = {
        "weight_source": "unit_test_calibrated_weights",
        "bounded_rationality_weights": {
            "confusion": 1.0,
            "navigation_entropy": 0.0,
            "cognitive_load": 0.0,
            "state_budget": 0.0,
            "puzzle_stall": 0.0,
        },
        "cognitive_effort_weights": {
            "cognitive_load": 1.0,
            "confusion": 0.0,
            "revisit_rate": 0.0,
            "state_budget": 0.0,
            "puzzle_stall": 0.0,
        },
        "novice": {
            "calibrated_config": {
                "memory_capacity": 9,
                "memory_decay_rate": 0.99,
            }
        },
    }
    calibration_path = tmp_path / "pcbs_persona_overrides.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    result = evaluate_astar_vs_pcbs(
        grid,
        persona="novice",
        timeout_astar=500,
        timeout_pcbs=200,
        seed=7,
        calibration_path=calibration_path,
    )

    assert result["pcbs"]["persona_source"] == "telemetry_calibrated_persona"
    assert result["pcbs"]["weight_source"] == "unit_test_calibrated_weights"
    assert result["pcbs"]["metric_interpretation"] == "telemetry_calibrated"
    assert result["comparison"]["pcbs_persona_source"] == "telemetry_calibrated_persona"
