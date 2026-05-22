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
    assert "| Map | Solver |" in table
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
