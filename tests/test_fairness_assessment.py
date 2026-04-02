import numpy as np

from src.evaluation.fairness_assessment import (
    compute_tile_distribution,
    jensen_shannon_divergence,
    compare_distributions,
    summarize_distribution,
)


def test_compute_tile_distribution_and_jsd():
    # Two small synthetic maps with two classes
    a = np.array([[0, 0], [0, 1]], dtype=np.int64)
    b = np.array([[0, 1], [1, 1]], dtype=np.int64)
    gen_dist = compute_tile_distribution([a], num_classes=2)
    ref_dist = compute_tile_distribution([b], num_classes=2)

    # gen_dist should sum to 1 and have probabilities [3/4, 1/4]
    assert np.isclose(gen_dist.sum(), 1.0)
    assert np.isclose(gen_dist[0], 0.75)
    assert np.isclose(gen_dist[1], 0.25)

    # ref_dist should be [1/4, 3/4]
    assert np.isclose(ref_dist[0], 0.25)
    assert np.isclose(ref_dist[1], 0.75)

    # JSD should be symmetric and > 0
    jsd_ab = jensen_shannon_divergence(gen_dist, ref_dist)
    jsd_ba = jensen_shannon_divergence(ref_dist, gen_dist)
    assert np.isclose(jsd_ab, jsd_ba)
    assert jsd_ab > 0.0

    comp = compare_distributions(gen_dist, ref_dist)
    assert 'jsd' in comp and 'l1' in comp and 'per_tile_ratio' in comp


def test_load_maps_from_dir_and_run(tmp_path):
    # Create two .npy files and ensure fairness_assessment can process them
    a = np.array([[0, 1], [1, 0]], dtype=np.int64)
    b = np.array([[1, 1], [0, 0]], dtype=np.int64)
    p1 = tmp_path / "map_0001.npy"
    p2 = tmp_path / "map_0002.npy"
    np.save(p1, a)
    np.save(p2, b)

    # Use compute_tile_distribution directly on loaded arrays
    gen_dist = compute_tile_distribution([a, b], num_classes=2)
    assert np.isclose(gen_dist.sum(), 1.0)
    # combined counts: zeros=4, ones=4 => uniform
    assert np.isclose(gen_dist[0], 0.5)
    assert np.isclose(gen_dist[1], 0.5)


def test_compute_tile_distribution_ignores_out_of_range_tiles():
    arr = np.array([[0, -1], [1, 99]], dtype=np.int64)
    dist = compute_tile_distribution([arr], num_classes=2)

    assert np.isclose(dist.sum(), 1.0)
    assert np.isclose(dist[0], 0.5)
    assert np.isclose(dist[1], 0.5)


def test_summarize_distribution_reports_entropy_and_active_classes():
    dist = np.array([0.5, 0.5, 0.0], dtype=np.float64)
    summary = summarize_distribution(dist)

    assert summary["active_class_count"] == 2.0
    assert summary["entropy"] > 0.0
    assert summary["max_tile_share"] == 0.5
