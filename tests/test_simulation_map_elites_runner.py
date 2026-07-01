import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.pipeline import NeuralSymbolicDungeonPipeline
from src.simulation.map_elites import MAPElitesEvaluator, run_map_elites_on_maps


def _simple_solvable_grid():
    floor = int(SEMANTIC_PALETTE["FLOOR"])
    start = int(SEMANTIC_PALETTE["START"])
    goal = int(SEMANTIC_PALETTE["TRIFORCE"])

    grid = np.full((16, 11), floor, dtype=np.int64)
    grid[1, 1] = start
    grid[14, 9] = goal
    return grid


def test_run_map_elites_on_maps_accepts_raw_grids():
    grid = _simple_solvable_grid()

    evaluator, occ = run_map_elites_on_maps([grid], resolution=8)
    assert occ.shape == (8, 8)
    assert len(evaluator.grid) >= 1


def test_map_elites_archive_round_trips(tmp_path):
    grid = _simple_solvable_grid()
    archive_path = tmp_path / "runtime_map_elites.pkl"
    evaluator = MAPElitesEvaluator(resolution=8, enable_advanced_archive=False)
    evaluator.add_dungeon(
        dungeon=grid,
        grid=grid,
        solver_result={
            "solvable": True,
            "path_length": 3,
            "path": [(1, 1), (1, 2), (1, 3)],
            "quality_score": 0.75,
        },
    )

    evaluator.save_archive(archive_path)

    loaded = MAPElitesEvaluator(resolution=8, enable_advanced_archive=False)
    loaded.load_archive(archive_path)

    assert loaded.occupancy_grid().tolist() == evaluator.occupancy_grid().tolist()
    assert set(loaded.grid.keys()) == set(evaluator.grid.keys())
    key = next(iter(evaluator.grid))
    assert loaded.grid[key].score == evaluator.grid[key].score


def test_run_map_elites_on_maps_can_warm_start_archive(tmp_path):
    grid = _simple_solvable_grid()
    archive_path = tmp_path / "runtime_map_elites.pkl"

    evaluator, _ = run_map_elites_on_maps(
        [grid],
        resolution=8,
        archive_path=archive_path,
        autosave_archive=True,
        enable_advanced_archive=False,
    )
    assert archive_path.exists()
    assert evaluator.grid

    loaded, occ = run_map_elites_on_maps(
        [],
        resolution=8,
        archive_path=archive_path,
        load_existing_archive=True,
        enable_advanced_archive=False,
    )
    assert occ.shape == (8, 8)
    assert set(loaded.grid.keys()) == set(evaluator.grid.keys())


def test_symbolic_pipeline_can_warm_start_map_elites_archive(tmp_path):
    grid = _simple_solvable_grid()
    archive_path = tmp_path / "runtime_map_elites.pkl"
    evaluator = MAPElitesEvaluator(resolution=8, enable_advanced_archive=False)
    evaluator.add_dungeon(
        dungeon=grid,
        grid=grid,
        solver_result={
            "solvable": True,
            "path_length": 3,
            "path": [(1, 1), (1, 2), (1, 3)],
            "quality_score": 0.75,
        },
    )
    evaluator.save_archive(archive_path)

    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device="cpu",
        enable_logging=False,
        enable_map_elites=True,
        map_elites_resolution=8,
        map_elites_archive_path=str(archive_path),
        map_elites_load_archive=True,
    )

    assert pipeline.map_elites is not None
    assert set(pipeline.map_elites.grid.keys()) == set(evaluator.grid.keys())
