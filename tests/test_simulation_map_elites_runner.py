import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.map_elites import run_map_elites_on_maps


def test_run_map_elites_on_maps_accepts_raw_grids():
    floor = int(SEMANTIC_PALETTE["FLOOR"])
    start = int(SEMANTIC_PALETTE["START"])
    goal = int(SEMANTIC_PALETTE["TRIFORCE"])

    grid = np.full((16, 11), floor, dtype=np.int64)
    grid[1, 1] = start
    grid[14, 9] = goal

    evaluator, occ = run_map_elites_on_maps([grid], resolution=8)
    assert occ.shape == (8, 8)
    assert len(evaluator.grid) >= 1
