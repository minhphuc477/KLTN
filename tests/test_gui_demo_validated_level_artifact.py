import json
from pathlib import Path

import numpy as np
import pytest

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.simulation.validator import Action, CONDITIONAL_IDS, StateSpaceAStar, WALKABLE_IDS, ZeldaLogicEnv


_BOSS_KEY_OLD_POS = (88, 15)
_BOSS_KEY_FAR_POS = (65, 18)
_OPEN_BRIDGE_CELLS = (
    (23, 32), (24, 32), (25, 32),
    (39, 22), (40, 22), (41, 22),
    (39, 33), (40, 33), (41, 33),
    (55, 22), (56, 22), (57, 22),
    (71, 22), (72, 22), (73, 22),
)
_BOSS_GATE_BRIDGE_CELLS = (
    (71, 10), (72, 10), (73, 10),
    (87, 10), (88, 10), (89, 10),
)


def _load_generated_json_artifact(path: Path):
    if not path.exists():
        pytest.skip(f"generated artifact is not present in the cleaned repo: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_demo_grid_or_skip(path: str) -> np.ndarray:
    demo_path = Path(path)
    if not demo_path.exists():
        pytest.skip(f"demo artifact is not present in the cleaned repo: {demo_path}")
    return np.loadtxt(demo_path, dtype=np.int32)


def _close_exterior_room_openings(grid: np.ndarray) -> np.ndarray:
    """Match the GUI demo cleanup: close room-edge doors that lead into void."""
    fixed = np.asarray(grid, dtype=np.int32).copy()
    room_rows = (fixed.shape[0] + ROOM_HEIGHT - 1) // ROOM_HEIGHT
    room_cols = (fixed.shape[1] + ROOM_WIDTH - 1) // ROOM_WIDTH
    occupied = []
    for room_r in range(room_rows):
        occupied_row = []
        for room_c in range(room_cols):
            block = fixed[
                room_r * ROOM_HEIGHT:min((room_r + 1) * ROOM_HEIGHT, fixed.shape[0]),
                room_c * ROOM_WIDTH:min((room_c + 1) * ROOM_WIDTH, fixed.shape[1]),
            ]
            occupied_row.append(bool((block != 0).any()))
        occupied.append(occupied_row)

    for room_r in range(room_rows):
        for room_c in range(room_cols):
            if not occupied[room_r][room_c]:
                continue
            r0, c0 = room_r * ROOM_HEIGHT, room_c * ROOM_WIDTH
            r1 = min(r0 + ROOM_HEIGHT, fixed.shape[0])
            c1 = min(c0 + ROOM_WIDTH, fixed.shape[1])
            exterior_sides = [
                (room_r == 0 or not occupied[room_r - 1][room_c], [(r0, c) for c in range(c0, c1)]),
                (
                    room_r == room_rows - 1 or not occupied[room_r + 1][room_c],
                    [(r1 - 1, c) for c in range(c0, c1)],
                ),
                (room_c == 0 or not occupied[room_r][room_c - 1], [(r, c0) for r in range(r0, r1)]),
                (
                    room_c == room_cols - 1 or not occupied[room_r][room_c + 1],
                    [(r, c1 - 1) for r in range(r0, r1)],
                ),
            ]
            for is_exterior, coords in exterior_sides:
                if not is_exterior:
                    continue
                for r, c in coords:
                    if int(fixed[r, c]) not in (0, 2):
                        fixed[r, c] = 2
    return fixed


def _apply_validated_demo_progression_edits(grid: np.ndarray) -> np.ndarray:
    """Move the boss key farther out and make paired room borders traversable."""
    fixed = np.asarray(grid, dtype=np.int32).copy()
    fixed[_BOSS_KEY_OLD_POS] = int(SEMANTIC_PALETTE["FLOOR"])
    fixed[_BOSS_KEY_FAR_POS] = int(SEMANTIC_PALETTE["KEY_BOSS"])
    for pos in _OPEN_BRIDGE_CELLS:
        fixed[pos] = int(SEMANTIC_PALETTE["DOOR_OPEN"])
    for pos in _BOSS_GATE_BRIDGE_CELLS:
        fixed[pos] = int(SEMANTIC_PALETTE["DOOR_BOSS"])
    return fixed


def _interior_room_border_passability_mismatches(grid: np.ndarray):
    passable = {int(tile) for tile in WALKABLE_IDS | CONDITIONAL_IDS}
    mismatches = []
    room_rows = (grid.shape[0] + ROOM_HEIGHT - 1) // ROOM_HEIGHT
    room_cols = (grid.shape[1] + ROOM_WIDTH - 1) // ROOM_WIDTH
    occupied = []
    for room_r in range(room_rows):
        occupied_row = []
        for room_c in range(room_cols):
            block = grid[
                room_r * ROOM_HEIGHT:min((room_r + 1) * ROOM_HEIGHT, grid.shape[0]),
                room_c * ROOM_WIDTH:min((room_c + 1) * ROOM_WIDTH, grid.shape[1]),
            ]
            occupied_row.append(bool((block != 0).any()))
        occupied.append(occupied_row)

    for room_r in range(room_rows):
        for room_c in range(room_cols):
            if not occupied[room_r][room_c]:
                continue
            if room_c + 1 < room_cols and occupied[room_r][room_c + 1]:
                left_c = (room_c + 1) * ROOM_WIDTH - 1
                right_c = (room_c + 1) * ROOM_WIDTH
                for row in range(room_r * ROOM_HEIGHT, min((room_r + 1) * ROOM_HEIGHT, grid.shape[0])):
                    left_open = int(grid[row, left_c]) in passable
                    right_open = int(grid[row, right_c]) in passable
                    if left_open != right_open:
                        mismatches.append(((row, left_c), (row, right_c)))
            if room_r + 1 < room_rows and occupied[room_r + 1][room_c]:
                top_r = (room_r + 1) * ROOM_HEIGHT - 1
                bottom_r = (room_r + 1) * ROOM_HEIGHT
                for col in range(room_c * ROOM_WIDTH, min((room_c + 1) * ROOM_WIDTH, grid.shape[1])):
                    top_open = int(grid[top_r, col]) in passable
                    bottom_open = int(grid[bottom_r, col]) in passable
                    if top_open != bottom_open:
                        mismatches.append(((top_r, col), (bottom_r, col)))
    return mismatches


def test_gui_demo_validated_level_matches_tuned_real_model_output_artifact():
    output_artifact = Path(
        "outputs/zelda_hmolqd_downstream_stageconditioned_semantics_v3_puzzlefix/"
        "protocol_manual_compare_stageconditioned_semantics_v3_puzzlefix/"
        "masked_room_full/dungeon_grid_ids.json"
    )
    demo_txt = Path("examples/gui_demo_validated_level.txt")

    output_grid = np.asarray(_load_generated_json_artifact(output_artifact), dtype=np.int32)
    demo_grid = _load_demo_grid_or_skip(str(demo_txt))

    expected_grid = _apply_validated_demo_progression_edits(_close_exterior_room_openings(output_grid))
    assert int(np.sum(expected_grid != output_grid)) == 73
    assert tuple(np.argwhere(demo_grid == SEMANTIC_PALETTE["KEY_BOSS"])[0]) == _BOSS_KEY_FAR_POS
    assert np.array_equal(demo_grid, expected_grid)


def test_gui_demo_validated_level_has_no_blocked_room_openings():
    demo_grid = _load_demo_grid_or_skip("examples/gui_demo_validated_level.txt")

    assert _interior_room_border_passability_mismatches(demo_grid) == []


def test_gui_demo_validated_level_is_solver_validated():
    demo_grid = _load_demo_grid_or_skip("examples/gui_demo_validated_level.txt")

    success, path, diagnostics = StateSpaceAStar(ZeldaLogicEnv(demo_grid), timeout=200_000).solve_with_diagnostics()

    assert success is True, diagnostics
    assert demo_grid.shape == (112, 55)
    assert len(path) == 97
    assert diagnostics.final_inventory["doors_opened"] >= 1
    assert diagnostics.final_inventory["items_collected"] >= 1


def test_gui_hard_demo_level_matches_chapter4_real_generated_artifact():
    output_artifact = Path(
        "results/ch4_generated_topology_real_pdrop035_seed20260418_fixedvalidator/"
        "diffusion_cfg3_logic0_steps50/dungeon_grid_ids.json"
    )
    demo_txt = Path("examples/gui_demo_hard_real_model_level.txt")

    output_grid = np.asarray(_load_generated_json_artifact(output_artifact), dtype=np.int32)
    demo_grid = _load_demo_grid_or_skip(str(demo_txt))

    assert output_grid.shape == (32, 77)
    assert np.array_equal(demo_grid, output_grid)


def test_gui_hard_demo_route_is_real_pcbs_trace_and_replays_to_victory():
    route_file = Path("examples/gui_demo_hard_real_model_route.json")
    case_study = Path("results/ch4_generated_topology_real_pdrop035_seed20260418_fixedvalidator/solver_case_study_budget250000.json")
    grid = _load_demo_grid_or_skip("examples/gui_demo_hard_real_model_level.txt")

    route = json.loads(route_file.read_text(encoding="utf-8"))
    source = _load_generated_json_artifact(case_study)
    source_solver = source["contexts"]["full"]["solvers"]["pcbs_balanced"]
    path = [tuple(point) for point in route["path"]]

    assert route["source_artifact"].endswith("diffusion_cfg3_logic0_steps50/dungeon_grid_ids.json")
    assert route["algorithm"] == "P-CBS balanced"
    assert route["path"] == source_solver["path"]
    assert route["path_length"] == 364
    assert route["demo_metrics"]["unique_rooms_visited"] >= 7
    assert route["demo_metrics"]["room_transition_count"] >= 12
    assert source_solver["success"] is True
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))

    action_for_delta = {
        (-1, 0): Action.UP,
        (1, 0): Action.DOWN,
        (0, -1): Action.LEFT,
        (0, 1): Action.RIGHT,
    }
    env = ZeldaLogicEnv(grid)
    assert env.state.position == path[0]

    for current, target in zip(path, path[1:]):
        delta = (target[0] - current[0], target[1] - current[1])
        env.step(action_for_delta[delta])
        assert env.state.position == target

    assert env.done is True
    assert env.state.position == path[-1]


def test_gui_hard_demo_level_solves_live_with_pcbs_balanced():
    from src.gui.solver.process_worker import solve_in_subprocess

    grid = _load_demo_grid_or_skip("examples/gui_demo_hard_real_model_level.txt")
    start = tuple(int(v) for v in np.argwhere(grid == 21)[0])
    goal = tuple(int(v) for v in np.argwhere(grid == 22)[0])

    result = solve_in_subprocess(
        grid,
        start,
        goal,
        7,
        {"strict_original_mode": False},
        {
            "timeout": 100_000,
            "representation": "tile",
            "allow_diagonals": False,
            "rules_profile": "vglc_strict",
        },
    )
    path = result.get("path") or []
    visited_rooms = {(int(r) // 16, int(c) // 11) for r, c in path}

    assert result["success"] is True, result
    assert len(path) == 102
    assert len(visited_rooms) >= 6
    assert result["solver_result"]["algorithm"] == "P-CBS"
    assert result["solver_result"]["trajectory_len"] == 364
    assert result["solver_result"]["pcbs_route_compressed"] is True


def test_gui_real_full_pipeline_pdrop035_demo_matches_source_artifact():
    output_artifact = Path(
        "results/thesis_ch4_evals/pdrop_sweep/pdrop035/"
        "diffusion_cfg3_logic0_steps50/dungeon_grid_ids.json"
    )
    demo_txt = Path("examples/gui_demo_real_full_pipeline_pdrop035_level.txt")

    output_grid = np.asarray(_load_generated_json_artifact(output_artifact), dtype=np.int32)
    demo_grid = _load_demo_grid_or_skip(str(demo_txt))

    assert output_grid.shape == (64, 44)
    assert np.array_equal(demo_grid, output_grid)


def test_gui_real_full_pipeline_pdrop035_demo_solves_live_with_pcbs_balanced():
    from src.gui.solver.process_worker import solve_in_subprocess

    grid = _load_demo_grid_or_skip("examples/gui_demo_real_full_pipeline_pdrop035_level.txt")
    start = tuple(int(v) for v in np.argwhere(grid == 21)[0])
    goal = tuple(int(v) for v in np.argwhere(grid == 22)[0])

    result = solve_in_subprocess(
        grid,
        start,
        goal,
        7,
        {"strict_original_mode": False},
        {
            "timeout": 100_000,
            "representation": "tile",
            "allow_diagonals": False,
            "rules_profile": "vglc_strict",
        },
    )
    path = result.get("path") or []
    visited_rooms = {(int(r) // 16, int(c) // 11) for r, c in path}
    solver_result = result["solver_result"]

    assert result["success"] is True, result
    assert len(path) == solver_result["display_path_len"]
    assert len(path) < solver_result["trajectory_len"]
    assert len(visited_rooms) >= 6
    assert solver_result["algorithm"] == "P-CBS"
    assert solver_result["pcbs_route_compressed"] is True
    assert solver_result["pcbs_loops_removed"] == solver_result["trajectory_len"] - len(path)

    action_for_delta = {
        (-1, 0): Action.UP,
        (1, 0): Action.DOWN,
        (0, -1): Action.LEFT,
        (0, 1): Action.RIGHT,
    }
    env = ZeldaLogicEnv(grid)
    assert env.state.position == path[0]
    for current, target in zip(path, path[1:]):
        delta = (target[0] - current[0], target[1] - current[1])
        env.step(action_for_delta[delta])
        assert env.state.position == target
    assert env.done is True
