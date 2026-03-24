import numpy as np

from src.utils.explainability import compute_neuro_symbolic_discrepancy_heatmap


def test_compute_neuro_symbolic_discrepancy_heatmap_zero_when_no_change():
    neural_grid = np.array([[1, 2], [3, 4]], dtype=np.int64)
    symbolic_grid = neural_grid.copy()

    probs = np.zeros((5, 2, 2), dtype=np.float32)
    probs[1, 0, 0] = 1.0
    probs[2, 0, 1] = 1.0
    probs[3, 1, 0] = 1.0
    probs[4, 1, 1] = 1.0

    heatmap, stats = compute_neuro_symbolic_discrepancy_heatmap(
        neural_probs=probs,
        neural_grid=neural_grid,
        symbolic_grid=symbolic_grid,
    )

    assert heatmap.shape == (2, 2)
    assert np.allclose(heatmap, 0.0)
    assert stats["changed_tiles"] == 0.0


def test_compute_neuro_symbolic_discrepancy_heatmap_positive_when_symbolic_overrides():
    neural_grid = np.array([[1, 2], [3, 4]], dtype=np.int64)
    symbolic_grid = np.array([[1, 2], [0, 4]], dtype=np.int64)

    # [H, W, C] style is also accepted.
    probs_hwc = np.full((2, 2, 5), 1e-4, dtype=np.float32)
    probs_hwc[0, 0, 1] = 0.99
    probs_hwc[0, 1, 2] = 0.99
    probs_hwc[1, 0, 3] = 0.95
    probs_hwc[1, 0, 0] = 0.02
    probs_hwc[1, 1, 4] = 0.99

    heatmap, stats = compute_neuro_symbolic_discrepancy_heatmap(
        neural_probs=probs_hwc,
        neural_grid=neural_grid,
        symbolic_grid=symbolic_grid,
    )

    assert heatmap[1, 0] > 0.0
    assert stats["changed_tiles"] == 1.0
    assert stats["changed_ratio"] > 0.0
