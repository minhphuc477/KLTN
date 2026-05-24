"""Focused WFC integration checks."""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.weighted_bayesian_wfc import TilePrior


def test_wfc_direction_mapping_and_tile_prior_probability():
    """Verify cardinal adjacency extraction and TilePrior probabilities."""
    test_grid = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )

    r, c = 1, 1
    adjacencies = []
    if r > 0:
        adjacencies.append((test_grid[r - 1, c], "N"))
    if r < test_grid.shape[0] - 1:
        adjacencies.append((test_grid[r + 1, c], "S"))
    if c < test_grid.shape[1] - 1:
        adjacencies.append((test_grid[r, c + 1], "E"))
    if c > 0:
        adjacencies.append((test_grid[r, c - 1], "W"))

    assert adjacencies == [(0, "N"), (0, "S"), (0, "E"), (0, "W")]

    prior = TilePrior(
        tile_id=1,
        frequency=0.5,
        adjacency_counts={(2, "N"): 10, (2, "S"): 5, (3, "N"): 5},
    )

    assert abs(prior.get_adjacency_probability(2, "N") - (10 / 15)) < 0.01
