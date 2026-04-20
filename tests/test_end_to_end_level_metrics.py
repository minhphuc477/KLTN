from __future__ import annotations

import pytest

from src.evaluation.end_to_end_level_metrics import (
    compute_end_to_end_structural_metrics,
    normalized_compression_distance,
    pairwise_ncd_stats,
)


def test_normalized_compression_distance_prefers_identical_strings():
    a = "FFFF\nFFFF\n"
    b = "FFFF\nFFFF\n"
    c = "WWWW\nWWWW\n"

    same = normalized_compression_distance(a, b)
    different = normalized_compression_distance(a, c)

    assert same <= different


def test_pairwise_ncd_stats_counts_all_pairs():
    stats = pairwise_ncd_stats(
        [
            "FFFF\nFFFF\n",
            "FFFW\nFFFW\n",
            "WWWW\nWWWW\n",
        ]
    )

    assert stats["count"] == 3
    assert stats["mean"] is not None
    assert stats["max"] is not None


def test_end_to_end_structural_metrics_track_uniqueness_and_reference_distance():
    metrics = compute_end_to_end_structural_metrics(
        room_texts={
            0: "FFFF\nFFFF\n",
            1: "FFFW\nFFFW\n",
            2: "FFFF\nFFFF\n",
        },
        dungeon_text="FFFFWWWW\n",
        reference_room_texts=[
            "FFFF\nFFFF\n",
            "WWWW\nWWWW\n",
        ],
    )

    assert metrics["room_count"] == 3
    assert metrics["unique_room_count"] == 2
    assert metrics["room_unique_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["room_pairwise_ncd"]["count"] == 3
    assert metrics["room_nearest_reference_ncd"]["count"] == 3
    assert metrics["room_symbol_entropy_mean"] is not None
    assert metrics["dungeon_symbol_entropy_non_void"] is not None
