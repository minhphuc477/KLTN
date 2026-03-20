import numpy as np

from src.generation.weighted_bayesian_wfc import (
    TilePrior,
    WeightedBayesianWFC,
    WeightedBayesianWFCConfig,
    integrate_weighted_wfc_into_pipeline,
)


def _simple_priors():
    return {
        1: TilePrior(tile_id=1, frequency=0.7, adjacency_counts={(1, "N"): 10, (1, "S"): 10, (1, "E"): 10, (1, "W"): 10}),
        2: TilePrior(tile_id=2, frequency=0.3, adjacency_counts={(2, "N"): 10, (2, "S"): 10, (2, "E"): 10, (2, "W"): 10}),
    }


def test_weighted_wfc_exposes_diagnostics():
    priors = _simple_priors()
    wfc = WeightedBayesianWFC(
        width=4,
        height=4,
        tile_priors=priors,
        config=WeightedBayesianWFCConfig(max_iterations=128),
        seed=123,
    )
    grid = wfc.generate(seed=456)
    assert grid.shape == (4, 4)
    diag = wfc.get_diagnostics()
    assert "contradictions" in diag
    assert "restarts" in diag
    assert "fallback_fills" in diag
    assert diag["total_cells"] == 16


def test_integrate_weighted_wfc_returns_probe_metrics():
    priors = _simple_priors()
    room = np.ones((4, 4), dtype=np.int32)
    out = integrate_weighted_wfc_into_pipeline(
        neural_room=room,
        tile_priors=priors,
        seed=99,
    )
    assert "wfc_diagnostics" in out
    assert "kl_divergence" in out
    assert "distribution_preserved" in out
