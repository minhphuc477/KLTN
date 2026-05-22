import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.gui.gameplay.path_strategies import smart_grid_path
from src.gui.runtime.route_payload import apply_loaded_route_data
from src.gui.solver.pcbs_route import compress_pcbs_route_for_replay
from src.gui.solver.process_worker import solve_in_subprocess
from src.simulation.validator import Action, ZeldaLogicEnv


HARD_DEMO_GRID = Path("examples/gui_demo_hard_real_model_level.txt")
HARD_DEMO_ROUTE = Path("examples/gui_demo_hard_real_model_route.json")


def _replay_to_done(grid, path):
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


def test_pcbs_route_compression_removes_only_replay_safe_loops():
    grid = np.loadtxt(HARD_DEMO_GRID, dtype=np.int32)
    route_data = json.loads(HARD_DEMO_ROUTE.read_text(encoding="utf-8"))
    raw_path = [tuple(point) for point in route_data["path"]]

    compressed, stats = compress_pcbs_route_for_replay(grid=grid, path=raw_path)

    assert len(raw_path) == 364
    assert len(compressed) == 102
    assert stats["compressed"] is True
    assert stats["loops_removed"] == 262
    _replay_to_done(grid, compressed)


def test_process_worker_pcbs_returns_short_replay_path_with_raw_trace_metrics():
    grid = np.loadtxt(HARD_DEMO_GRID, dtype=np.int32)
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
            "seed": 123,
            "representation": "tile",
            "allow_diagonals": False,
            "rules_profile": "vglc_strict",
        },
    )

    solver_result = result["solver_result"]
    path = [tuple(point) for point in result["path"]]

    assert result["success"] is True
    assert len(path) == 102
    assert solver_result["trajectory_len"] == 364
    assert solver_result["display_path_len"] == 102
    assert solver_result["pcbs_route_compressed"] is True
    assert solver_result["pcbs_loops_removed"] == 262
    assert solver_result["cbs_metrics"]["total_steps"] == 364
    assert solver_result["cbs_metrics"]["unique_rooms"] >= 1
    assert "room_entropy" in solver_result["cbs_metrics"]
    _replay_to_done(grid, path)


def test_loaded_pcbs_route_is_compressed_when_environment_is_available():
    grid = np.loadtxt(HARD_DEMO_GRID, dtype=np.int32)
    route_data = json.loads(HARD_DEMO_ROUTE.read_text(encoding="utf-8"))
    gui = SimpleNamespace(
        env=ZeldaLogicEnv(grid),
        auto_path=[],
        solution_path=[],
        start_pos=None,
        goal_pos=None,
        auto_step_idx=0,
        auto_mode=False,
    )

    path_len = apply_loaded_route_data(gui, route_data)

    assert path_len == 102
    assert len(gui.auto_path) == 102
    assert gui.solver_result["trajectory_len"] == 364
    assert gui.solver_result["pcbs_route_compressed"] is True
    _replay_to_done(grid, gui.auto_path)


def test_completionist_and_novice_skip_quick_grid_path():
    grid = np.zeros((3, 3), dtype=np.int32)
    grid[1, 1] = 21
    grid[1, 2] = 22
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    for algorithm_idx in (13, 14):
        gui = SimpleNamespace(
            maps=[grid],
            current_map_idx=0,
            env=SimpleNamespace(start_pos=(1, 1), goal_pos=(1, 2)),
            algorithm_idx=algorithm_idx,
            show_heatmap=False,
            feature_flags={},
            _algorithm_name=lambda idx: f"P-CBS {idx}",
        )

        success, path, teleports = smart_grid_path(
            gui,
            logger,
            convert_diagonal_to_4dir=lambda p, grid=None: p,
            semantic_palette={},
            np_module=np,
            path_cls=Path,
            os_module=SimpleNamespace(environ={}),
        )

        assert success is False
        assert path == []
        assert teleports == 0


def test_gui_pcbs_default_persona_seeds_solve_non_balanced_demo_routes():
    grid = np.loadtxt(HARD_DEMO_GRID, dtype=np.int32)
    start = tuple(int(v) for v in np.argwhere(grid == 21)[0])
    goal = tuple(int(v) for v in np.argwhere(grid == 22)[0])
    expected_seed = {
        9: 0,
        13: 42,
        14: 42,
    }

    for algorithm_idx, seed in expected_seed.items():
        result = solve_in_subprocess(
            grid,
            start,
            goal,
            algorithm_idx,
            {"strict_original_mode": False},
            {
                "timeout": 8_000,
                "representation": "tile",
                "allow_diagonals": False,
                "rules_profile": "vglc_strict",
            },
        )

        solver_result = result["solver_result"]
        assert result["success"] is True, result
        assert solver_result["pcbs_seed"] == seed
        assert solver_result["trajectory_len"] > len(result["path"])
        assert solver_result["pcbs_route_compressed"] is True
