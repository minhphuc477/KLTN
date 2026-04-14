import pytest

from scripts.run_fixed_graph_multi_seed_audit import _aggregate_variant
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline


def test_room_alignment_aggregation_includes_post_overlay_semantic_error():
    aggregated = NeuralSymbolicDungeonPipeline._aggregate_room_alignment_metrics(
        [
            {
                "final_pre_overlay_graph_marker_expected": 2.0,
                "final_graph_marker_overwrites": 1.0,
                "neural_graph_marker_exact_match_rate": 0.5,
                "final_pre_overlay_graph_marker_exact_match_rate": 0.25,
                "final_post_overlay_graph_marker_exact_match_rate": 1.0,
                "final_graph_marker_overwrite_rate": 0.5,
                "neural_semantic_anchor_avg_manhattan_error": 4.0,
                "final_pre_overlay_semantic_anchor_avg_manhattan_error": 2.0,
                "final_post_overlay_semantic_anchor_avg_manhattan_error": 0.0,
            },
            {
                "final_pre_overlay_graph_marker_expected": 1.0,
                "final_graph_marker_overwrites": 0.0,
                "neural_graph_marker_exact_match_rate": 1.0,
                "final_pre_overlay_graph_marker_exact_match_rate": 0.5,
                "final_post_overlay_graph_marker_exact_match_rate": 1.0,
                "final_graph_marker_overwrite_rate": 0.0,
                "neural_semantic_anchor_avg_manhattan_error": 2.0,
                "final_pre_overlay_semantic_anchor_avg_manhattan_error": 1.0,
                "final_post_overlay_semantic_anchor_avg_manhattan_error": 0.5,
            },
        ]
    )

    assert aggregated["total_graph_marker_expected"] == pytest.approx(3.0)
    assert aggregated["total_graph_marker_overwrites"] == pytest.approx(1.0)
    assert aggregated["avg_final_post_overlay_graph_marker_exact_match_rate"] == pytest.approx(1.0)
    assert aggregated["avg_final_post_overlay_semantic_anchor_error"] == pytest.approx(0.25)


def test_fixed_graph_audit_aggregate_tracks_post_overlay_semantic_error():
    entries = [
        {
            "metrics": {
                "repair_rate": 1.0,
                "total_tiles_repaired": 10,
                "generation_time_sec": 3.0,
                "avg_neural_graph_marker_exact_match_rate": 0.75,
                "avg_final_pre_overlay_graph_marker_exact_match_rate": 0.5,
                "avg_final_post_overlay_graph_marker_exact_match_rate": 1.0,
                "avg_final_graph_marker_overwrite_rate": 0.25,
                "avg_neural_semantic_anchor_error": 4.0,
                "avg_final_pre_overlay_semantic_anchor_error": 2.0,
                "avg_final_post_overlay_semantic_anchor_error": 0.0,
            },
            "room_hashes": {"0": "aaa"},
        },
        {
            "metrics": {
                "repair_rate": 0.5,
                "total_tiles_repaired": 6,
                "generation_time_sec": 5.0,
                "avg_neural_graph_marker_exact_match_rate": 1.0,
                "avg_final_pre_overlay_graph_marker_exact_match_rate": 0.75,
                "avg_final_post_overlay_graph_marker_exact_match_rate": 1.0,
                "avg_final_graph_marker_overwrite_rate": 0.5,
                "avg_neural_semantic_anchor_error": 2.0,
                "avg_final_pre_overlay_semantic_anchor_error": 1.0,
                "avg_final_post_overlay_semantic_anchor_error": 0.5,
            },
            "room_hashes": {"0": "bbb"},
        },
    ]

    aggregate = _aggregate_variant(entries)

    assert aggregate["avg_repair_rate"] == pytest.approx(0.75)
    assert aggregate["avg_final_post_overlay_graph_marker_exact_match_rate"] == pytest.approx(1.0)
    assert aggregate["avg_final_post_overlay_semantic_anchor_error"] == pytest.approx(0.25)
    assert aggregate["unique_layout_count"] == 2
