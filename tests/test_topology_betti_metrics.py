import numpy as np

from src.evaluation.topology_betti import (
    betti_curve,
    digital_betti_numbers,
    normalized_betti_curve_distance,
)


def test_digital_betti_numbers_distinguish_components_and_holes():
    two_components = np.zeros((7, 7), dtype=bool)
    two_components[1:3, 1:3] = True
    two_components[4:6, 4:6] = True
    assert digital_betti_numbers(two_components) == (2, 0)

    ring = np.zeros((7, 7), dtype=bool)
    ring[1:6, 1:6] = True
    ring[2:5, 2:5] = False
    assert digital_betti_numbers(ring) == (1, 1)


def test_betti_curve_distance_reports_topology_change():
    connected = np.zeros((7, 7), dtype=np.float32)
    connected[1:6, 1:6] = 1.0
    ring = connected.copy()
    ring[2:5, 2:5] = 0.0

    connected_curve = betti_curve(connected, thresholds=(0.25, 0.5, 0.75))
    ring_curve = betti_curve(ring, thresholds=(0.25, 0.5, 0.75))

    assert normalized_betti_curve_distance(
        connected_curve,
        connected_curve,
        grid_size=connected.size,
    ) == 0.0
    assert normalized_betti_curve_distance(
        connected_curve,
        ring_curve,
        grid_size=connected.size,
    ) > 0.0
