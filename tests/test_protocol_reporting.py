import pytest
import networkx as nx
import numpy as np

from scripts.run_fixed_graph_multi_seed_audit import _aggregate_variant
from scripts.run_fast_sampler_visual_audit import (
    _json_sanitize as _audit_json_sanitize,
    build_validation_context_from_generation_result,
    build_validation_search_stats_payload,
)
from scripts.run_stateful_puzzle_hparam_sweep import _json_sanitize as _sweep_json_sanitize
from scripts.run_stateful_puzzle_hparam_sweep import _profile_score
from src.pipeline.dungeon_pipeline import DungeonGenerationResult, NeuralSymbolicDungeonPipeline
from src.pipeline.room_stitching import StitchedRoomLayout
from src.simulation.validator import GraphGuidedValidator


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
            "validation": {
                "graph_guided_oracle": {"solvable": True},
                "mechanical_contract": {"hybrid_oracle_pass": True},
                "search_algorithms": {
                    "tile_state_space": {
                        "astar": {
                            "success": True,
                            "fallback_used": False,
                            "path_length": 12,
                            "states_explored": 40,
                            "time_sec": 0.5,
                            "path_ratio_vs_astar": 1.0,
                            "states_ratio_vs_astar": 1.0,
                        },
                        "greedy": {
                            "success": True,
                            "fallback_used": False,
                            "path_length": 14,
                            "states_explored": 18,
                            "time_sec": 0.2,
                            "path_ratio_vs_astar": 14.0 / 12.0,
                            "states_ratio_vs_astar": 18.0 / 40.0,
                        },
                    },
                    "agreement": {
                        "astar_success": True,
                        "all_algorithms_solved": True,
                    },
                },
            },
            "end_to_end_evaluation": {
                "room_unique_ratio": 1.0,
                "room_pairwise_ncd": {"mean": 0.15},
                "room_nearest_reference_ncd": {"mean": 0.10},
                "room_symbol_entropy_mean": 1.2,
                "dungeon_symbol_entropy_non_void": 1.4,
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
            "validation": {
                "graph_guided_oracle": {"solvable": False},
                "mechanical_contract": {"hybrid_oracle_pass": False},
                "search_algorithms": {
                    "tile_state_space": {
                        "astar": {
                            "success": True,
                            "fallback_used": False,
                            "path_length": 10,
                            "states_explored": 32,
                            "time_sec": 0.4,
                            "path_ratio_vs_astar": 1.0,
                            "states_ratio_vs_astar": 1.0,
                        },
                        "greedy": {
                            "success": False,
                            "fallback_used": True,
                            "path_length": 0,
                            "states_explored": 0,
                            "time_sec": 0.1,
                            "path_ratio_vs_astar": None,
                            "states_ratio_vs_astar": None,
                        },
                    },
                    "agreement": {
                        "astar_success": True,
                        "all_algorithms_solved": False,
                    },
                },
            },
            "end_to_end_evaluation": {
                "room_unique_ratio": 0.5,
                "room_pairwise_ncd": {"mean": 0.30},
                "room_nearest_reference_ncd": {"mean": 0.18},
                "room_symbol_entropy_mean": 1.0,
                "dungeon_symbol_entropy_non_void": 1.1,
            },
            "room_hashes": {"0": "bbb"},
        },
    ]

    aggregate = _aggregate_variant(entries)

    assert aggregate["avg_repair_rate"] == pytest.approx(0.75)
    assert aggregate["avg_final_post_overlay_graph_marker_exact_match_rate"] == pytest.approx(1.0)
    assert aggregate["avg_final_post_overlay_semantic_anchor_error"] == pytest.approx(0.25)
    assert aggregate["avg_room_unique_ratio"] == pytest.approx(0.75)
    assert aggregate["avg_room_pairwise_ncd_mean"] == pytest.approx(0.225)
    assert aggregate["avg_room_nearest_reference_ncd_mean"] == pytest.approx(0.14)
    assert aggregate["avg_room_symbol_entropy_mean"] == pytest.approx(1.1)
    assert aggregate["avg_dungeon_symbol_entropy_non_void"] == pytest.approx(1.25)
    assert aggregate["unique_layout_count"] == 2
    assert aggregate["graph_guided_oracle_solvable_rate"] == pytest.approx(0.5)
    assert aggregate["hybrid_oracle_pass_rate"] == pytest.approx(0.5)
    assert aggregate["search_algorithm_aggregate"]["tile_state_space"]["astar"]["success_rate"] == pytest.approx(1.0)
    assert aggregate["search_algorithm_aggregate"]["tile_state_space"]["astar"]["independent_success_rate"] == pytest.approx(1.0)
    assert aggregate["search_algorithm_aggregate"]["tile_state_space"]["greedy"]["success_rate"] == pytest.approx(0.5)
    assert aggregate["search_algorithm_aggregate"]["tile_state_space"]["greedy"]["fallback_rate"] == pytest.approx(0.5)
    assert aggregate["search_algorithm_aggregate"]["agreement"]["all_algorithms_solved"] == pytest.approx(0.5)
    assert aggregate["search_algorithm_aggregate"]["oracle_stack"]["graph_guided_oracle_solvable_rate"] == pytest.approx(0.5)
    assert aggregate["search_algorithm_aggregate"]["oracle_stack"]["hybrid_oracle_pass_rate"] == pytest.approx(0.5)
    assert aggregate["search_algorithm_aggregate"]["behavioral_probe"]["cbs_balanced"]["success_rate"] == pytest.approx(0.0)


def test_validation_search_sidecar_includes_oracle_and_behavioral_layers():
    payload = build_validation_search_stats_payload(
        {
            "search_algorithms": {"tile_state_space": {"astar": {"success": True}}, "agreement": {"astar_success": True}},
            "astar_grid": {"solvable": True},
            "softlock_check": {"is_safe": True},
            "graph_progression": {"goal_gauntlet_valid": True},
            "cbs_balanced": {"success": False, "confusion_ratio_vs_astar": float("inf")},
        }
    )

    assert payload["search_algorithms"]["tile_state_space"]["astar"]["success"] is True
    assert payload["hard_oracle"]["astar_grid"]["solvable"] is True
    assert payload["hard_oracle"]["graph_progression"]["goal_gauntlet_valid"] is True
    assert payload["behavioral_probe"]["cbs_balanced"]["success"] is False


def test_stateful_puzzle_profile_score_stays_finite_when_cbs_confusion_is_infinite():
    score = _profile_score(
        {
            "metrics": {
                "repair_rate": 0.8,
                "avg_final_graph_marker_overwrite_rate": 0.1,
                "avg_final_post_overlay_semantic_anchor_error": 0.0,
                "generation_time_sec": 100.0,
                "puzzle_stage_count": 7,
                "puzzle_plan_count": 5,
            },
            "runtime_diagnostics": {
                "puzzle_room_contract_valid": 2,
                "puzzle_room_interaction_valid": 2,
                "puzzle_room_sequence_valid": 3,
            },
            "validation": {
                "mechanical_contract": {"hybrid_oracle_pass": True},
                "cbs_balanced": {
                    "success": False,
                    "status": "budget_exhausted",
                    "confusion_ratio_vs_astar": float("inf"),
                    "confusion_index": 20.0,
                    "cognitive_load": 1.0,
                    "peak_frustration": 0.8,
                },
            },
        }
    )

    assert score == pytest.approx(127.05)


def test_stateful_puzzle_summary_json_sanitizes_non_finite_payloads():
    payload = {
        "raw_summaries": {
            "baseline_default": {
                "validation": {
                    "cbs_balanced": {
                        "confusion_ratio_vs_astar": float("inf"),
                    }
                }
            }
        }
    }

    sanitized = _sweep_json_sanitize(payload)

    assert sanitized["raw_summaries"]["baseline_default"]["validation"]["cbs_balanced"]["confusion_ratio_vs_astar"] is None


def test_export_summary_json_sanitizes_non_finite_payloads():
    payload = {
        "validation": {
            "cbs_balanced": {
                "confusion_ratio_vs_astar": float("inf"),
            }
        }
    }

    sanitized = _audit_json_sanitize(payload)

    assert sanitized["validation"]["cbs_balanced"]["confusion_ratio_vs_astar"] is None


def test_graph_guided_validator_accepts_goal_typed_nodes():
    graph = nx.DiGraph()
    graph.add_node(0, type="START", label="START")
    graph.add_node(1, type="GOAL", label="GOAL")
    graph.add_edge(0, 1)

    class _Room:
        def __init__(self):
            self.grid = np.array([[1]], dtype=int)

    dungeon_data = type(
        "DungeonDataStub",
        (),
        {
            "graph": graph,
            "rooms": {"0": _Room(), "1": _Room()},
        },
    )()

    result = GraphGuidedValidator().validate_dungeon_with_graph(dungeon_data)

    assert result.is_solvable is True
    assert result.start_node == 0
    assert result.triforce_node == 1


def test_validation_context_uses_stitched_slot_keys_for_puzzle_metadata():
    result = DungeonGenerationResult(
        dungeon_grid=[[21, 1, 22]],
        rooms={},
        mission_graph=nx.DiGraph(),
        generation_time=0.0,
        metrics={},
        stitched_layout=StitchedRoomLayout(
            dungeon_grid=[[21, 1, 22]],
            slot_positions={7: (1, 2)},
            room_offsets={7: (20, 30)},
            layout_map={},
        ),
        puzzle_metadata={"plans": [{"room_id": 7}]},
    )

    context = build_validation_context_from_generation_result(result)

    assert context["room_puzzle_metadata"]["plans"][0]["room_id"] == 7
    assert context["room_positions"] == {(1, 2): (20, 30)}
    assert context["room_to_node"] == {(1, 2): 7}
    assert context["node_to_room"] == {7: (1, 2)}
