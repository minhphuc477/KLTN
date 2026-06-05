import json

import pytest
import networkx as nx
import numpy as np

from scripts.run_ablation_study import (
    AblationStudy,
    ExperimentConfig,
    _json_sanitize,
    build_ablation_plan,
    build_experiment_set,
)
from scripts.compare_protocol_to_baselines import build_report as build_protocol_baseline_report
from scripts.compare_protocol_to_baselines import _render_markdown as render_protocol_baseline_markdown
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


def test_ablation_core_plan_documents_random_and_pure_wfc_baselines():
    configs = build_experiment_set(include_extended=False)
    plan = build_ablation_plan(
        configs=configs,
        seeds=[42, 43],
        target_curve=[0.2, 0.5, 0.8],
        num_rooms=8,
        diffusion_steps=5,
        cbs_timeout=1000,
        evolution_population=6,
        evolution_generations=4,
    )

    experiments = {entry["name"]: entry for entry in plan["experiments"]}

    assert experiments["RANDOM_TOPOLOGY"]["tier"] == "block_i"
    assert "strict topology null" in " ".join(plan["claim_boundaries"])
    assert experiments["PURE_WFC"]["component"] == "standalone symbolic generator"
    assert "topology_preservation_score" in plan["metrics"]


def test_ablation_extended_plan_documents_logic_guidance_timing_sweep():
    configs = build_experiment_set(include_extended=True)
    plan = build_ablation_plan(
        configs=configs,
        seeds=[42],
        target_curve=[0.2, 0.5, 0.8],
        num_rooms=8,
        diffusion_steps=5,
        cbs_timeout=1000,
        evolution_population=6,
        evolution_generations=4,
    )

    experiments = {entry["name"]: entry for entry in plan["experiments"]}

    assert experiments["LOGIC_ACTIVE_0.25"]["component"] == "LogicNet guidance timing"
    assert experiments["LOGIC_ACTIVE_0.25"]["config"]["logic_guidance_active_fraction"] == pytest.approx(0.25)


def test_ablation_json_sanitize_outputs_strict_json_values():
    payload = {
        "nan": float("nan"),
        "inf": float("inf"),
        "np_float": np.float32(1.25),
        "array": np.array([1, np.nan]),
        "config": ExperimentConfig(name="SMOKE"),
    }

    sanitized = _json_sanitize(payload)

    assert sanitized["nan"] is None
    assert sanitized["inf"] is None
    assert sanitized["np_float"] == pytest.approx(1.25)
    assert sanitized["array"] == [1, None]
    assert json.dumps(sanitized, allow_nan=False)


def test_core_ablation_set_includes_sparse_edge_topology_variant():
    from scripts.run_ablation_study import build_experiment_set

    configs = build_experiment_set(include_extended=False)
    by_name = {cfg.name: cfg for cfg in configs}

    assert by_name["FULL"].topology_refinement_mode == "gat2"
    assert by_name["TOPO_SPARSE_EDGE"].topology_refinement_mode == "sparse_edge"


def test_ablation_summary_separates_failure_rate_from_conditional_solvability():
    study = AblationStudy.__new__(AblationStudy)
    study.max_runtime_sec = None
    rows = {
        ("FULL", 1): {
            "config": "FULL",
            "seed": 1,
            "success": True,
            "solvable": True,
            "confusion_ratio": 0.0,
            "confusion_index": 0.0,
            "path_optimal": 1.0,
            "tile_prior_kl": 0.0,
            "graph_edit_distance": 0.0,
            "generation_time_sec": 1.0,
            "novelty": 0.0,
            "reconstruction_error": 0.0,
            "constraint_valid": 1.0,
            "room_repair_rate": 0.0,
            "tiles_repaired": 0.0,
            "topology_representable_edge_rate": 1.0,
            "topology_edge_connection_recall": 1.0,
            "topology_phantom_connection_rate": 0.0,
            "topology_preservation_score": 1.0,
            "directed_representable_edge_rate": 1.0,
            "directed_edge_realization_rate": 1.0,
            "directed_directionality_leak_rate": 0.0,
            "directed_edge_preservation_score": 1.0,
        },
        ("FULL", 2): {
            "config": "FULL",
            "seed": 2,
            "success": False,
            "solvable": False,
            "confusion_ratio": np.nan,
            "confusion_index": np.nan,
            "path_optimal": 0.0,
            "tile_prior_kl": np.nan,
            "graph_edit_distance": np.nan,
            "generation_time_sec": 0.5,
            "novelty": np.nan,
            "reconstruction_error": np.nan,
            "constraint_valid": np.nan,
            "room_repair_rate": np.nan,
            "tiles_repaired": np.nan,
            "topology_representable_edge_rate": np.nan,
            "topology_edge_connection_recall": np.nan,
            "topology_phantom_connection_rate": np.nan,
            "topology_preservation_score": np.nan,
            "directed_representable_edge_rate": np.nan,
            "directed_edge_realization_rate": np.nan,
            "directed_directionality_leak_rate": np.nan,
            "directed_edge_preservation_score": np.nan,
        },
    }
    study._run_single = lambda cfg, seed: dict(rows[(cfg.name, seed)])

    _raw, summary = study.run(configs=[ExperimentConfig(name="FULL")], seeds=[1, 2])
    full = summary.iloc[0].to_dict()

    assert full["success_rate"] == pytest.approx(0.5)
    assert full["failure_rate"] == pytest.approx(0.5)
    assert full["solvability_rate"] == pytest.approx(0.5)
    assert full["solvability_rate_successful_generations"] == pytest.approx(1.0)


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
                "raw_neural_to_cleaned_tiles_changed": 3,
                "raw_neural_to_final_tiles_changed": 8,
            },
            "runtime_diagnostics": {
                "fast_sampler_teacher_fallback": 2,
            },
            "validation": {
                "raw_astar_grid": {"solvable": False},
                "astar_grid": {"solvable": True},
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
            "cleanup_totals": {
                "neural_invalid_door_tiles_removed": 2,
                "neural_interior_obstacle_tiles_removed": 1,
                "repair_invalid_door_tiles_removed": 4,
                "repair_interior_obstacle_tiles_removed": 2,
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
                "raw_neural_to_cleaned_tiles_changed": 1,
                "raw_neural_to_final_tiles_changed": 4,
            },
            "runtime_diagnostics": {
                "masked_room_teacher_fallback": 1,
            },
            "validation": {
                "raw_astar_grid": {"solvable": False},
                "astar_grid": {"solvable": True},
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
            "cleanup_totals": {
                "neural_invalid_door_tiles_removed": 1,
                "neural_interior_obstacle_tiles_removed": 0,
                "repair_invalid_door_tiles_removed": 0,
                "repair_interior_obstacle_tiles_removed": 1,
            },
            "room_hashes": {"0": "bbb"},
        },
    ]

    aggregate = _aggregate_variant(entries)

    assert aggregate["avg_repair_rate"] == pytest.approx(0.75)
    assert aggregate["teacher_fallback_run_rate"] == pytest.approx(1.0)
    assert aggregate["total_teacher_fallback_count"] == pytest.approx(3.0)
    assert aggregate["avg_teacher_fallback_count_per_run"] == pytest.approx(1.5)
    assert aggregate["total_teacher_fallback_source_fast_sampler_count"] == pytest.approx(2.0)
    assert aggregate["total_teacher_fallback_source_masked_room_count"] == pytest.approx(1.0)
    assert aggregate["avg_raw_neural_to_cleaned_tiles_changed"] == pytest.approx(2.0)
    assert aggregate["avg_raw_neural_to_final_tiles_changed"] == pytest.approx(6.0)
    assert aggregate["avg_neural_cleanup_tiles_removed"] == pytest.approx(2.0)
    assert aggregate["avg_repair_cleanup_tiles_removed"] == pytest.approx(3.5)
    assert aggregate["avg_final_post_overlay_graph_marker_exact_match_rate"] == pytest.approx(1.0)
    assert aggregate["avg_final_post_overlay_semantic_anchor_error"] == pytest.approx(0.25)
    assert aggregate["avg_overlay_graph_marker_match_rate_delta"] == pytest.approx(0.375)
    assert aggregate["avg_overlay_semantic_anchor_error_reduction"] == pytest.approx(1.25)
    assert aggregate["overlay_semantic_anchor_improvement_rate"] == pytest.approx(1.0)
    assert aggregate["post_repair_astar_grid_solvable_rate"] == pytest.approx(1.0)
    assert aggregate["raw_astar_grid_solvable_rate"] == pytest.approx(0.0)
    assert aggregate["raw_hard_oracle_available_rate"] == pytest.approx(1.0)
    assert aggregate["raw_to_post_repair_astar_grid_solvability_delta"] == pytest.approx(1.0)
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
    assert aggregate["search_algorithm_aggregate"]["oracle_stack"]["post_repair_astar_grid_solvable_rate"] == pytest.approx(1.0)
    assert aggregate["search_algorithm_aggregate"]["oracle_stack"]["raw_astar_grid_solvable_rate"] == pytest.approx(0.0)
    assert aggregate["search_algorithm_aggregate"]["oracle_stack"]["raw_to_post_repair_astar_grid_solvability_delta"] == pytest.approx(1.0)
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


def test_protocol_baseline_report_flags_fallback_repair_and_overlay_evidence(tmp_path):
    fixed_graph = {
        "aggregate": {
            "strict_masked_room": {
                "avg_repair_rate": 0.4,
                "avg_total_tiles_repaired": 14.0,
                "avg_teacher_fallback_used": 0.25,
                "avg_teacher_fallback_source_masked_room": 0.25,
                "avg_generation_time_sec": 2.0,
                "avg_final_graph_marker_overwrite_rate": 0.1,
                "avg_final_pre_overlay_semantic_anchor_error": 3.0,
                "avg_final_post_overlay_semantic_anchor_error": 1.0,
                "search_algorithm_aggregate": {
                    "oracle_stack": {
                        "astar_grid_solvable_rate": 1.0,
                        "graph_guided_oracle_solvable_rate": 1.0,
                        "softlock_safe_rate": 1.0,
                        "goal_gauntlet_valid_rate": 1.0,
                        "hybrid_oracle_pass_rate": 1.0,
                    },
                    "behavioral_probe": {"cbs_balanced": {"success_rate": 0.5}},
                },
            }
        }
    }
    matched_budget = {"summary": [{"method": "wfc", "fitness": 0.1, "overall_completeness": 0.2}]}
    pcg = {
        "summary": [
            {
                "problem_name": "zelda",
                "method": "wfc",
                "external_quality_pass_rate": 0.1,
                "external_controlability_pass_rate": 0.2,
                "external_diversity_pass_rate": 0.3,
                "internal_overall_completeness": 0.4,
            }
        ]
    }
    fixed_path = tmp_path / "fixed.json"
    matched_path = tmp_path / "matched.json"
    pcg_path = tmp_path / "pcg.json"
    fixed_path.write_text(json.dumps(fixed_graph), encoding="utf-8")
    matched_path.write_text(json.dumps(matched_budget), encoding="utf-8")
    pcg_path.write_text(json.dumps(pcg), encoding="utf-8")

    report = build_protocol_baseline_report(
        fixed_graph_summary=fixed_path,
        matched_budget_report=matched_path,
        pcg_benchmark_report=pcg_path,
    )

    audit = report["strict_evidence_audit"]["by_variant"]["strict_masked_room"]
    assert audit["publication_ready_standalone_neural_evidence"] is False
    assert set(audit["issues"]) == {
        "teacher_fallback_used",
        "raw_hard_oracle_metrics_missing",
        "repair_heavy",
        "overlay_assisted_semantics",
    }
    assert report["claim_status"]["can_claim_surpasses_publications"] is False
    markdown = render_protocol_baseline_markdown(report)
    assert "Strict Evidence Audit" in markdown
    assert "BLOCKING: strict_masked_room: teacher fallback rate" in markdown


def test_protocol_baseline_report_blocks_missing_hard_oracle_metrics(tmp_path):
    fixed_graph = {
        "aggregate": {
            "neural_no_oracle": {
                "avg_repair_rate": 0.0,
                "avg_total_tiles_repaired": 0.0,
                "avg_teacher_fallback_used": 0.0,
                "avg_generation_time_sec": 2.0,
                "avg_final_graph_marker_overwrite_rate": 0.0,
            }
        }
    }
    matched_budget = {"summary": [{"method": "wfc", "fitness": 0.1, "overall_completeness": 0.2}]}
    pcg = {"summary": []}
    fixed_path = tmp_path / "fixed.json"
    matched_path = tmp_path / "matched.json"
    pcg_path = tmp_path / "pcg.json"
    fixed_path.write_text(json.dumps(fixed_graph), encoding="utf-8")
    matched_path.write_text(json.dumps(matched_budget), encoding="utf-8")
    pcg_path.write_text(json.dumps(pcg), encoding="utf-8")

    report = build_protocol_baseline_report(
        fixed_graph_summary=fixed_path,
        matched_budget_report=matched_path,
        pcg_benchmark_report=pcg_path,
    )

    audit = report["strict_evidence_audit"]["by_variant"]["neural_no_oracle"]
    assert audit["publication_ready_standalone_neural_evidence"] is False
    assert audit["issues"] == ["hard_oracle_metrics_missing"]
    assert report["strict_evidence_audit"]["blocking"] == [
        "neural_no_oracle: no hard oracle metrics present; cannot support standalone neural solvability claims."
    ]
    markdown = render_protocol_baseline_markdown(report)
    assert "hard_oracle_present=False" in markdown


def test_protocol_baseline_report_blocks_post_repair_only_hard_solvability(tmp_path):
    fixed_graph = {
        "aggregate": {
            "post_repair_only": {
                "avg_repair_rate": 0.1,
                "avg_total_tiles_repaired": 2.0,
                "avg_teacher_fallback_used": 0.0,
                "search_algorithm_aggregate": {
                    "oracle_stack": {
                        "post_repair_astar_grid_solvable_rate": 1.0,
                    }
                },
            }
        }
    }
    matched_budget = {"summary": []}
    pcg = {"summary": []}
    fixed_path = tmp_path / "fixed.json"
    matched_path = tmp_path / "matched.json"
    pcg_path = tmp_path / "pcg.json"
    fixed_path.write_text(json.dumps(fixed_graph), encoding="utf-8")
    matched_path.write_text(json.dumps(matched_budget), encoding="utf-8")
    pcg_path.write_text(json.dumps(pcg), encoding="utf-8")

    report = build_protocol_baseline_report(
        fixed_graph_summary=fixed_path,
        matched_budget_report=matched_path,
        pcg_benchmark_report=pcg_path,
    )

    audit = report["strict_evidence_audit"]["by_variant"]["post_repair_only"]
    assert audit["publication_ready_standalone_neural_evidence"] is False
    assert audit["issues"] == ["post_repair_only_hard_solvability"]
    assert audit["post_repair_hard_solvability_rate"] == pytest.approx(1.0)
    assert audit["standalone_hard_solvability_rate"] is None
    markdown = render_protocol_baseline_markdown(report)
    assert "post_repair_hard=1.0000" in markdown


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
