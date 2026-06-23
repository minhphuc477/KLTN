"""
Run a fixed-graph multi-seed audit for the diffusion / fast-sampler branch.

This keeps the mission graph constant and reruns the same graph-conditioned
generation with multiple random seeds. It is useful after sampler/runtime
fixes because it separates visual stability from Block I topology variance.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from networkx.readwrite import json_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_fast_sampler_visual_audit import (
    VALIDATION_SEARCH_SUITE_VERSION,
    add_generation_override_args,
    export_variant,
    generation_overrides_from_namespace,
)
from scripts.export_semantic_anchor_end_to_end import export_masked_variant
from scripts.export_manual_rich_topology_compare import (
    _ensure_directed_progression_graph,
    build_manual_rich_topology_graph,
)


def _load_mission_graph(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = json_graph.node_link_graph(payload, edges="links")
    return _ensure_directed_progression_graph(graph, source=str(path))


def _safe_mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _safe_median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _safe_mean_or_none(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _safe_optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_bool_as_float(value: Any) -> float | None:
    if value is None:
        return None
    return 1.0 if bool(value) else 0.0


def _nested_get(payload: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_nested(payload: Dict[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value = _nested_get(payload, path)
        if value is not None:
            return value
    return None


def _entry_diagnostic_count(entry: Dict[str, Any], key: str) -> float:
    metrics = dict(entry.get("metrics", {}) or {})
    candidates = (
        _nested_get(metrics, ("runtime_diagnostics_delta", key)),
        _nested_get(entry, ("runtime_diagnostics", key)),
        _nested_get(metrics, ("runtime_diagnostics", key)),
    )
    for candidate in candidates:
        numeric = _safe_optional_float(candidate)
        if numeric is not None:
            return numeric
    return 0.0


def _entry_teacher_fallback_source_count(entry: Dict[str, Any], source: str) -> float:
    diagnostics_key = f"{source}_teacher_fallback"
    metric_key = f"teacher_fallback_source_{source}"
    diagnostics_count = _entry_diagnostic_count(entry, diagnostics_key)
    metric_value = _safe_optional_float(dict(entry.get("metrics", {}) or {}).get(metric_key))
    return max(diagnostics_count, metric_value or 0.0)


def _entry_numeric(entry: Dict[str, Any], *paths: Sequence[str]) -> float | None:
    for path in paths:
        value = _first_nested(entry, (path,))
        numeric = _safe_optional_float(value)
        if numeric is not None:
            return numeric
    return None


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    return value


def _load_existing_variant_summary(seed_dir: Path, variant_name: str) -> Dict[str, Any]:
    summary_path = seed_dir / str(variant_name) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing cached variant summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _try_load_existing_seed_summary(seed_dir: Path, variants: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    cached_variants: Dict[str, Any] = {}
    for variant in variants:
        variant_name = str(variant["variant_name"])
        summary_path = seed_dir / variant_name / "summary.json"
        if not summary_path.exists():
            return None
        cached_variants[variant_name] = json.loads(summary_path.read_text(encoding="utf-8"))
    return cached_variants


def _aggregate_variant(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {}

    repair_rates = [float(entry["metrics"]["repair_rate"]) for entry in entries]
    repaired_tiles = [int(entry["metrics"]["total_tiles_repaired"]) for entry in entries]
    generation_times = [float(entry["metrics"]["generation_time_sec"]) for entry in entries]
    masked_metric_names = (
        "avg_masked_refinement_steps_requested",
        "avg_masked_refinement_steps_executed",
        "avg_masked_corrector_rounds_requested",
        "avg_masked_corrector_rounds_executed",
        "avg_masked_corrector_refinement_steps_executed",
        "avg_masked_initial_editable_tokens",
        "avg_masked_initial_tokens_committed",
        "avg_masked_corrector_tokens_committed",
        "avg_masked_mean_tokens_committed_per_step",
        "avg_masked_final_unresolved_tokens",
    )
    masked_refinement_summary = {
        name: _safe_mean_or_none([
            value
            for entry in entries
            if (value := _safe_optional_float(entry["metrics"].get(name))) is not None
        ])
        for name in masked_metric_names
    }
    teacher_fallback_fast_counts = [
        _entry_teacher_fallback_source_count(entry, "fast_sampler")
        for entry in entries
    ]
    teacher_fallback_masked_counts = [
        _entry_teacher_fallback_source_count(entry, "masked_room")
        for entry in entries
    ]
    teacher_fallback_counts = [
        max(
            fast_count + masked_count,
            _safe_optional_float(entry["metrics"].get("teacher_fallback_used")) or 0.0,
        )
        for entry, fast_count, masked_count in zip(
            entries,
            teacher_fallback_fast_counts,
            teacher_fallback_masked_counts,
        )
    ]
    teacher_fallback_used = [
        1.0 if fallback_count > 0.0 else 0.0
        for fallback_count in teacher_fallback_counts
    ]
    teacher_fallback_fast = [
        1.0 if fallback_count > 0.0 else 0.0
        for fallback_count in teacher_fallback_fast_counts
    ]
    teacher_fallback_masked = [
        1.0 if fallback_count > 0.0 else 0.0
        for fallback_count in teacher_fallback_masked_counts
    ]
    neural_match_rates = [
        float(entry["metrics"].get("avg_neural_graph_marker_exact_match_rate", 1.0))
        for entry in entries
    ]
    final_pre_overlay_match_rates = [
        float(entry["metrics"].get("avg_final_pre_overlay_graph_marker_exact_match_rate", 1.0))
        for entry in entries
    ]
    final_post_overlay_match_rates = [
        float(entry["metrics"].get("avg_final_post_overlay_graph_marker_exact_match_rate", 1.0))
        for entry in entries
    ]
    marker_overwrite_rates = [
        float(entry["metrics"].get("avg_final_graph_marker_overwrite_rate", 0.0))
        for entry in entries
    ]
    neural_anchor_errors = [
        float(entry["metrics"].get("avg_neural_semantic_anchor_error", 0.0))
        for entry in entries
    ]
    final_pre_overlay_anchor_errors = [
        float(entry["metrics"].get("avg_final_pre_overlay_semantic_anchor_error", 0.0))
        for entry in entries
    ]
    final_post_overlay_anchor_errors = [
        float(entry["metrics"].get("avg_final_post_overlay_semantic_anchor_error", 0.0))
        for entry in entries
    ]
    overlay_match_rate_deltas = [
        post_rate - pre_rate
        for pre_rate, post_rate in zip(final_pre_overlay_match_rates, final_post_overlay_match_rates)
    ]
    overlay_anchor_error_reductions = [
        pre_error - post_error
        for pre_error, post_error in zip(final_pre_overlay_anchor_errors, final_post_overlay_anchor_errors)
    ]
    overlay_improved_semantic_anchor = [
        1.0 if reduction > 0.0 else 0.0
        for reduction in overlay_anchor_error_reductions
    ]
    astar_solvable = [
        1.0 if bool(entry.get("validation", {}).get("astar_grid", {}).get("solvable", False)) else 0.0
        for entry in entries
    ]
    raw_astar_values: List[float] = []
    raw_astar_available: List[float] = []
    raw_to_post_repair_astar_deltas: List[float] = []
    raw_astar_paths = (
        ("raw_astar_grid", "solvable"),
        ("raw_hard_oracle", "astar_grid", "solvable"),
        ("pre_repair_astar_grid", "solvable"),
        ("pre_repair_hard_oracle", "astar_grid", "solvable"),
    )
    for entry, post_astar in zip(entries, astar_solvable):
        raw_value = _first_nested(dict(entry.get("validation", {}) or {}), raw_astar_paths)
        raw_astar = _optional_bool_as_float(raw_value)
        raw_astar_available.append(1.0 if raw_astar is not None else 0.0)
        if raw_astar is not None:
            raw_astar_values.append(raw_astar)
            raw_to_post_repair_astar_deltas.append(post_astar - raw_astar)
    graph_guided_solvable = [
        1.0 if bool(entry.get("validation", {}).get("graph_guided_oracle", {}).get("solvable", False)) else 0.0
        for entry in entries
    ]
    softlock_safe = [
        1.0 if bool(entry.get("validation", {}).get("softlock_check", {}).get("is_safe", False)) else 0.0
        for entry in entries
    ]
    cbs_success = [
        1.0 if bool(entry.get("validation", {}).get("cbs_balanced", {}).get("success", False)) else 0.0
        for entry in entries
    ]
    cbs_path_lengths = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("path_length", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_states = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("states_explored", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_confusion_ratios = [
        confusion_ratio
        for entry in entries
        if (
            confusion_ratio := _safe_optional_float(
                entry.get("validation", {}).get("cbs_balanced", {}).get("confusion_ratio_vs_astar")
            )
        ) is not None
    ]
    cbs_confusion_index = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("confusion_index", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_navigation_entropy = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("navigation_entropy", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_cognitive_load = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("cognitive_load", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_deliberation_events = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("deliberation_events", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_budget_exhaustion_events = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("budget_exhaustion_events", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_peak_frustration = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("peak_frustration", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_affordance_reactivations = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("affordance_reactivations", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_affordance_guided_steps = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("affordance_guided_steps", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_inventory_change_events = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("inventory_change_events", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_focus_switches = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("focus_switches", 0.0) or 0.0)
        for entry in entries
    ]
    cbs_focus_guided_steps = [
        float(entry.get("validation", {}).get("cbs_balanced", {}).get("focus_guided_steps", 0.0) or 0.0)
        for entry in entries
    ]
    goal_gauntlet_valid = [
        1.0 if bool(entry.get("validation", {}).get("graph_progression", {}).get("goal_gauntlet_valid", False)) else 0.0
        for entry in entries
    ]
    hybrid_oracle_pass = [
        1.0 if bool(entry.get("validation", {}).get("mechanical_contract", {}).get("hybrid_oracle_pass", False)) else 0.0
        for entry in entries
    ]
    room_unique_ratios = [
        room_unique_ratio
        for entry in entries
        if (
            room_unique_ratio := _safe_optional_float(
                entry.get("end_to_end_evaluation", {}).get("room_unique_ratio")
            )
        ) is not None
    ]
    room_pairwise_ncd_means = [
        room_pairwise_ncd_mean
        for entry in entries
        if (
            room_pairwise_ncd_mean := _safe_optional_float(
                entry.get("end_to_end_evaluation", {}).get("room_pairwise_ncd", {}).get("mean")
            )
        ) is not None
    ]
    room_reference_ncd_means = [
        room_reference_ncd_mean
        for entry in entries
        if (
            room_reference_ncd_mean := _safe_optional_float(
                entry.get("end_to_end_evaluation", {}).get("room_nearest_reference_ncd", {}).get("mean")
            )
        ) is not None
    ]
    room_symbol_entropy_means = [
        room_symbol_entropy_mean
        for entry in entries
        if (
            room_symbol_entropy_mean := _safe_optional_float(
                entry.get("end_to_end_evaluation", {}).get("room_symbol_entropy_mean")
            )
        ) is not None
    ]
    dungeon_symbol_entropy = [
        dungeon_symbol_entropy_non_void
        for entry in entries
        if (
            dungeon_symbol_entropy_non_void := _safe_optional_float(
                entry.get("end_to_end_evaluation", {}).get("dungeon_symbol_entropy_non_void")
            )
        ) is not None
    ]
    raw_neural_to_cleaned_tiles_changed = [
        value
        for entry in entries
        if (
            value := _entry_numeric(
                entry,
                ("metrics", "raw_neural_to_cleaned_tiles_changed"),
                ("metrics", "total_raw_neural_to_cleaned_tiles_changed"),
            )
        ) is not None
    ]
    raw_neural_to_final_tiles_changed = [
        value
        for entry in entries
        if (
            value := _entry_numeric(
                entry,
                ("metrics", "raw_neural_to_final_tiles_changed"),
                ("metrics", "total_raw_neural_to_final_tiles_changed"),
            )
        ) is not None
    ]
    neural_cleanup_tiles_removed = [
        value
        for entry in entries
        if (
            value := _safe_optional_float(
                sum(
                    float(entry.get("cleanup_totals", {}).get(key, 0.0) or 0.0)
                    for key in (
                        "neural_invalid_door_tiles_removed",
                        "neural_interior_obstacle_tiles_removed",
                    )
                )
            )
        ) is not None
        and "cleanup_totals" in entry
    ]
    repair_cleanup_tiles_removed = [
        value
        for entry in entries
        if (
            value := _safe_optional_float(
                sum(
                    float(entry.get("cleanup_totals", {}).get(key, 0.0) or 0.0)
                    for key in (
                        "repair_invalid_door_tiles_removed",
                        "repair_interior_obstacle_tiles_removed",
                    )
                )
            )
        ) is not None
        and "cleanup_totals" in entry
    ]
    room_hash_signatures = [
        "|".join(f"{room_id}:{digest}" for room_id, digest in sorted(entry["room_hashes"].items()))
        for entry in entries
    ]

    search_algorithm_aggregate: Dict[str, Any] = {
        "search_suite_version": VALIDATION_SEARCH_SUITE_VERSION,
        "tile_state_space": {},
        "agreement": {},
        "behavioral_probe": {},
        "oracle_stack": {},
        "excluded_algorithms": {},
    }
    algorithm_buckets: Dict[str, Dict[str, List[float]]] = {}
    agreement_buckets: Dict[str, List[float]] = {}
    excluded_algorithms: Dict[str, str] = {}
    for entry in entries:
        suite = entry.get("validation", {}).get("search_algorithms", {})
        excluded_algorithms.update({str(k): str(v) for k, v in dict(suite.get("excluded_algorithms", {})).items()})
        for algorithm_name, result in dict(suite.get("tile_state_space", {})).items():
            bucket = algorithm_buckets.setdefault(
                str(algorithm_name),
                {
                    "success": [],
                    "independent_success": [],
                    "fallback_used": [],
                    "path_length": [],
                    "states_explored": [],
                    "time_sec": [],
                    "path_ratio_vs_astar": [],
                    "states_ratio_vs_astar": [],
                },
            )
            success_flag = bool(result.get("success", False))
            fallback_flag = bool(result.get("fallback_used", False))
            bucket["success"].append(1.0 if success_flag else 0.0)
            bucket["independent_success"].append(1.0 if (success_flag and not fallback_flag) else 0.0)
            bucket["fallback_used"].append(1.0 if fallback_flag else 0.0)
            bucket["path_length"].append(float(result.get("path_length", 0.0) or 0.0))
            bucket["states_explored"].append(float(result.get("states_explored", 0.0) or 0.0))
            bucket["time_sec"].append(float(result.get("time_sec", 0.0) or 0.0))
            path_ratio = _safe_optional_float(result.get("path_ratio_vs_astar"))
            if path_ratio is not None:
                bucket["path_ratio_vs_astar"].append(path_ratio)
            states_ratio = _safe_optional_float(result.get("states_ratio_vs_astar"))
            if states_ratio is not None:
                bucket["states_ratio_vs_astar"].append(states_ratio)
        for agreement_name, value in dict(suite.get("agreement", {})).items():
            if isinstance(value, bool):
                agreement_buckets.setdefault(str(agreement_name), []).append(1.0 if value else 0.0)
            else:
                numeric_value = _safe_optional_float(value)
                if numeric_value is not None:
                    agreement_buckets.setdefault(str(agreement_name), []).append(numeric_value)

    for algorithm_name, bucket in algorithm_buckets.items():
        search_algorithm_aggregate["tile_state_space"][algorithm_name] = {
            "success_rate": _safe_mean(bucket["success"]),
            "independent_success_rate": _safe_mean(bucket["independent_success"]),
            "fallback_rate": _safe_mean(bucket["fallback_used"]),
            "avg_path_length": _safe_mean(bucket["path_length"]),
            "avg_states_explored": _safe_mean(bucket["states_explored"]),
            "avg_time_sec": _safe_mean(bucket["time_sec"]),
            "avg_path_ratio_vs_astar": _safe_mean_or_none(bucket["path_ratio_vs_astar"]),
            "avg_states_ratio_vs_astar": _safe_mean_or_none(bucket["states_ratio_vs_astar"]),
        }
    for agreement_name, values in agreement_buckets.items():
        search_algorithm_aggregate["agreement"][agreement_name] = _safe_mean(values)
    search_algorithm_aggregate["behavioral_probe"] = {
        "cbs_balanced": {
            "success_rate": _safe_mean(cbs_success),
            "avg_path_length": _safe_mean(cbs_path_lengths),
            "avg_states_explored": _safe_mean(cbs_states),
            "avg_confusion_ratio_vs_astar": _safe_mean(cbs_confusion_ratios),
            "avg_confusion_index": _safe_mean(cbs_confusion_index),
            "avg_navigation_entropy": _safe_mean(cbs_navigation_entropy),
            "avg_cognitive_load": _safe_mean(cbs_cognitive_load),
            "avg_deliberation_events": _safe_mean(cbs_deliberation_events),
            "avg_budget_exhaustion_events": _safe_mean(cbs_budget_exhaustion_events),
            "avg_peak_frustration": _safe_mean(cbs_peak_frustration),
            "avg_affordance_reactivations": _safe_mean(cbs_affordance_reactivations),
            "avg_affordance_guided_steps": _safe_mean(cbs_affordance_guided_steps),
            "avg_inventory_change_events": _safe_mean(cbs_inventory_change_events),
            "avg_focus_switches": _safe_mean(cbs_focus_switches),
            "avg_focus_guided_steps": _safe_mean(cbs_focus_guided_steps),
        }
    }
    search_algorithm_aggregate["oracle_stack"] = {
        "astar_grid_solvable_rate": _safe_mean(astar_solvable),
        "post_repair_astar_grid_solvable_rate": _safe_mean(astar_solvable),
        "raw_astar_grid_solvable_rate": _safe_mean_or_none(raw_astar_values),
        "raw_hard_oracle_available_rate": _safe_mean(raw_astar_available),
        "raw_to_post_repair_astar_grid_solvability_delta": _safe_mean_or_none(raw_to_post_repair_astar_deltas),
        "graph_guided_oracle_solvable_rate": _safe_mean(graph_guided_solvable),
        "softlock_safe_rate": _safe_mean(softlock_safe),
        "goal_gauntlet_valid_rate": _safe_mean(goal_gauntlet_valid),
        "hybrid_oracle_pass_rate": _safe_mean(hybrid_oracle_pass),
    }
    search_algorithm_aggregate["excluded_algorithms"] = excluded_algorithms

    return {
        "num_runs": len(entries),
        "avg_repair_rate": _safe_mean(repair_rates),
        "median_repair_rate": _safe_median(repair_rates),
        "avg_total_tiles_repaired": _safe_mean(repaired_tiles),
        "median_total_tiles_repaired": _safe_median(repaired_tiles),
        "avg_teacher_fallback_used": _safe_mean(teacher_fallback_used),
        "avg_teacher_fallback_source_fast_sampler": _safe_mean(teacher_fallback_fast),
        "avg_teacher_fallback_source_masked_room": _safe_mean(teacher_fallback_masked),
        "teacher_fallback_run_rate": _safe_mean(teacher_fallback_used),
        "total_teacher_fallback_count": float(sum(teacher_fallback_counts)),
        "avg_teacher_fallback_count_per_run": _safe_mean(teacher_fallback_counts),
        "total_teacher_fallback_source_fast_sampler_count": float(sum(teacher_fallback_fast_counts)),
        "avg_teacher_fallback_source_fast_sampler_count_per_run": _safe_mean(teacher_fallback_fast_counts),
        "total_teacher_fallback_source_masked_room_count": float(sum(teacher_fallback_masked_counts)),
        "avg_teacher_fallback_source_masked_room_count_per_run": _safe_mean(teacher_fallback_masked_counts),
        "avg_generation_time_sec": _safe_mean(generation_times),
        "median_generation_time_sec": _safe_median(generation_times),
        **masked_refinement_summary,
        "avg_raw_neural_to_cleaned_tiles_changed": _safe_mean_or_none(raw_neural_to_cleaned_tiles_changed),
        "avg_raw_neural_to_final_tiles_changed": _safe_mean_or_none(raw_neural_to_final_tiles_changed),
        "avg_neural_cleanup_tiles_removed": _safe_mean_or_none(neural_cleanup_tiles_removed),
        "avg_repair_cleanup_tiles_removed": _safe_mean_or_none(repair_cleanup_tiles_removed),
        "avg_neural_graph_marker_exact_match_rate": _safe_mean(neural_match_rates),
        "median_neural_graph_marker_exact_match_rate": _safe_median(neural_match_rates),
        "avg_final_pre_overlay_graph_marker_exact_match_rate": _safe_mean(final_pre_overlay_match_rates),
        "median_final_pre_overlay_graph_marker_exact_match_rate": _safe_median(final_pre_overlay_match_rates),
        "avg_final_post_overlay_graph_marker_exact_match_rate": _safe_mean(final_post_overlay_match_rates),
        "median_final_post_overlay_graph_marker_exact_match_rate": _safe_median(final_post_overlay_match_rates),
        "avg_overlay_graph_marker_match_rate_delta": _safe_mean(overlay_match_rate_deltas),
        "avg_final_graph_marker_overwrite_rate": _safe_mean(marker_overwrite_rates),
        "median_final_graph_marker_overwrite_rate": _safe_median(marker_overwrite_rates),
        "avg_neural_semantic_anchor_error": _safe_mean(neural_anchor_errors),
        "median_neural_semantic_anchor_error": _safe_median(neural_anchor_errors),
        "avg_final_pre_overlay_semantic_anchor_error": _safe_mean(final_pre_overlay_anchor_errors),
        "median_final_pre_overlay_semantic_anchor_error": _safe_median(final_pre_overlay_anchor_errors),
        "avg_final_post_overlay_semantic_anchor_error": _safe_mean(final_post_overlay_anchor_errors),
        "median_final_post_overlay_semantic_anchor_error": _safe_median(final_post_overlay_anchor_errors),
        "avg_overlay_semantic_anchor_error_reduction": _safe_mean(overlay_anchor_error_reductions),
        "overlay_semantic_anchor_improvement_rate": _safe_mean(overlay_improved_semantic_anchor),
        "astar_grid_solvable_rate": _safe_mean(astar_solvable),
        "post_repair_astar_grid_solvable_rate": _safe_mean(astar_solvable),
        "raw_astar_grid_solvable_rate": _safe_mean_or_none(raw_astar_values),
        "raw_hard_oracle_available_rate": _safe_mean(raw_astar_available),
        "raw_to_post_repair_astar_grid_solvability_delta": _safe_mean_or_none(raw_to_post_repair_astar_deltas),
        "graph_guided_oracle_solvable_rate": _safe_mean(graph_guided_solvable),
        "softlock_safe_rate": _safe_mean(softlock_safe),
        "hybrid_oracle_pass_rate": _safe_mean(hybrid_oracle_pass),
        "cbs_success_rate": _safe_mean(cbs_success),
        "avg_cbs_confusion_ratio_vs_astar": _safe_mean(cbs_confusion_ratios),
        "avg_cbs_cognitive_load": _safe_mean(cbs_cognitive_load),
        "avg_cbs_deliberation_events": _safe_mean(cbs_deliberation_events),
        "avg_cbs_budget_exhaustion_events": _safe_mean(cbs_budget_exhaustion_events),
        "avg_cbs_peak_frustration": _safe_mean(cbs_peak_frustration),
        "avg_cbs_affordance_reactivations": _safe_mean(cbs_affordance_reactivations),
        "avg_cbs_affordance_guided_steps": _safe_mean(cbs_affordance_guided_steps),
        "avg_cbs_inventory_change_events": _safe_mean(cbs_inventory_change_events),
        "avg_cbs_focus_switches": _safe_mean(cbs_focus_switches),
        "avg_cbs_focus_guided_steps": _safe_mean(cbs_focus_guided_steps),
        "goal_gauntlet_valid_rate": _safe_mean(goal_gauntlet_valid),
        "avg_room_unique_ratio": _safe_mean_or_none(room_unique_ratios),
        "avg_room_pairwise_ncd_mean": _safe_mean_or_none(room_pairwise_ncd_means),
        "avg_room_nearest_reference_ncd_mean": _safe_mean_or_none(room_reference_ncd_means),
        "avg_room_symbol_entropy_mean": _safe_mean_or_none(room_symbol_entropy_means),
        "avg_dungeon_symbol_entropy_non_void": _safe_mean_or_none(dungeon_symbol_entropy),
        "unique_layout_count": len(set(room_hash_signatures)),
        "all_layouts_identical": len(set(room_hash_signatures)) == 1,
        "search_algorithm_aggregate": search_algorithm_aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-graph multi-seed audit for diffusion / fast-sampler.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing checkpoints and resolved_config.json",
    )
    parser.add_argument(
        "--mission-graph",
        type=Path,
        default=None,
        help="Optional path to a fixed mission_graph.json file. If omitted, the built-in rich manual topology is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-seed exports and aggregate summary",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260404, 20260405, 20260406],
        help="Seeds to audit",
    )
    parser.add_argument(
        "--lcm-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit fast-sampler/LCM checkpoint to use for fast-sampling variants.",
    )
    parser.add_argument(
        "--reuse-existing-seed-summaries",
        action="store_true",
        help=(
            "Reuse existing per-seed variant summary.json files when they already exist "
            "under --output-dir instead of regenerating those variants."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Skip generation and rebuild the aggregate summary from existing per-seed "
            "summary.json files under --output-dir. Fails if any expected seed/variant "
            "summary is missing."
        ),
    )
    parser.add_argument(
        "--include-no-fallback-ablations",
        action="store_true",
        help=(
            "Also export strict no-fallback variants for fast-sampler and masked-room, "
            "plus pure-neural no-overlay variants used by the neural-semantics audit."
        ),
    )
    parser.add_argument(
        "--include-puzzle-ablations",
        action="store_true",
        help=(
            "Also export puzzle-off variants with puzzle_room_scaffold_enabled=False "
            "for diffusion, fast-sampler, and masked-room."
        ),
    )
    add_generation_override_args(parser)
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> Dict[str, str]:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")

    if args.mission_graph is None:
        mission_graph = build_manual_rich_topology_graph()
        mission_graph_source = "built_in_manual_rich_topology"
    else:
        mission_graph = _load_mission_graph(Path(args.mission_graph))
        mission_graph_source = str(Path(args.mission_graph))

    generation_overrides = generation_overrides_from_namespace(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mission_graph.json").write_text(
        json.dumps(json_graph.node_link_data(mission_graph, edges="links"), indent=2),
        encoding="utf-8",
    )
    print(
        f"[topology-audit-fixed-graph] run_dir={run_dir} graph={mission_graph_source} "
        f"seeds={[int(seed) for seed in args.seeds]} overrides={generation_overrides}",
        flush=True,
    )

    variants = [
        {
            "exporter": "diffusion",
            "variant_name": "diffusion_cfg3_logic0_steps50",
            "guidance_scale": 3.0,
            "logic_guidance_scale": 0.0,
            "num_diffusion_steps": 50,
            "use_fast_sampling": False,
        },
        {
            "exporter": "diffusion",
            "variant_name": "fast_cfg3_logic0_steps4",
            "guidance_scale": 3.0,
            "logic_guidance_scale": 0.0,
            "num_diffusion_steps": 4,
            "use_fast_sampling": True,
        },
        {
            "exporter": "masked_room",
            "variant_name": "masked_room_full",
        },
    ]
    if bool(getattr(args, "include_no_fallback_ablations", False)):
        variants.extend(
            [
                {
                    "exporter": "diffusion",
                    "variant_name": "fast_cfg3_logic0_steps4_no_fallback",
                    "guidance_scale": 3.0,
                    "logic_guidance_scale": 0.0,
                    "num_diffusion_steps": 4,
                    "use_fast_sampling": True,
                    "variant_generation_overrides": {
                        "fast_sampler_teacher_fallback_enabled": False,
                    },
                },
                {
                    "exporter": "masked_room",
                    "variant_name": "masked_room_full_no_fallback",
                    "variant_generation_overrides": {
                        "masked_room_teacher_fallback_enabled": False,
                    },
                },
                {
                    "exporter": "diffusion",
                    "variant_name": "diffusion_cfg3_logic0_steps50_pure_neural",
                    "guidance_scale": 3.0,
                    "logic_guidance_scale": 0.0,
                    "num_diffusion_steps": 50,
                    "use_fast_sampling": False,
                    "variant_generation_overrides": {
                        "deterministic_graph_marker_overlay_enabled": False,
                    },
                },
                {
                    "exporter": "diffusion",
                    "variant_name": "fast_cfg3_logic0_steps4_pure_neural_no_fallback",
                    "guidance_scale": 3.0,
                    "logic_guidance_scale": 0.0,
                    "num_diffusion_steps": 4,
                    "use_fast_sampling": True,
                    "variant_generation_overrides": {
                        "deterministic_graph_marker_overlay_enabled": False,
                        "fast_sampler_teacher_fallback_enabled": False,
                    },
                },
                {
                    "exporter": "masked_room",
                    "variant_name": "masked_room_full_pure_neural_no_fallback",
                    "variant_generation_overrides": {
                        "deterministic_graph_marker_overlay_enabled": False,
                        "masked_room_teacher_fallback_enabled": False,
                    },
                },
            ]
        )
    if bool(getattr(args, "include_puzzle_ablations", False)):
        variants.extend(
            [
                {
                    "exporter": "diffusion",
                    "variant_name": "diffusion_cfg3_logic0_steps50_no_puzzle",
                    "guidance_scale": 3.0,
                    "logic_guidance_scale": 0.0,
                    "num_diffusion_steps": 50,
                    "use_fast_sampling": False,
                    "variant_generation_overrides": {
                        "puzzle_room_scaffold_enabled": False,
                        "puzzle_room_structure_enabled": False,
                    },
                },
                {
                    "exporter": "diffusion",
                    "variant_name": "fast_cfg3_logic0_steps4_no_puzzle",
                    "guidance_scale": 3.0,
                    "logic_guidance_scale": 0.0,
                    "num_diffusion_steps": 4,
                    "use_fast_sampling": True,
                    "variant_generation_overrides": {
                        "puzzle_room_scaffold_enabled": False,
                        "puzzle_room_structure_enabled": False,
                    },
                },
                {
                    "exporter": "masked_room",
                    "variant_name": "masked_room_full_no_puzzle",
                    "variant_generation_overrides": {
                        "puzzle_room_scaffold_enabled": False,
                        "puzzle_room_structure_enabled": False,
                    },
                },
            ]
        )

    per_seed: List[Dict[str, Any]] = []
    aggregate_inputs: Dict[str, List[Dict[str, Any]]] = {
        str(variant["variant_name"]): [] for variant in variants
    }

    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_summary: Dict[str, Any] = {"seed": int(seed), "variants": {}}
        cached_variants = _try_load_existing_seed_summary(seed_dir, variants)
        if bool(getattr(args, "aggregate_only", False)):
            if cached_variants is None:
                raise FileNotFoundError(
                    "Cannot use --aggregate-only because one or more cached per-seed "
                    f"summaries are missing under {seed_dir}"
                )
            for variant_name, summary in cached_variants.items():
                seed_summary["variants"][variant_name] = summary
                aggregate_inputs[variant_name].append(summary)
            per_seed.append(seed_summary)
            continue
        if bool(getattr(args, "reuse_existing_seed_summaries", False)) and cached_variants is not None:
            print(
                f"[topology-audit-fixed-graph] seed={int(seed)} reusing cached variant summaries",
                flush=True,
            )
            for variant_name, summary in cached_variants.items():
                seed_summary["variants"][variant_name] = summary
                aggregate_inputs[variant_name].append(summary)
            per_seed.append(seed_summary)
            continue
        for variant in variants:
            variant_generation_overrides = dict(generation_overrides)
            variant_generation_overrides.update(dict(variant.get("variant_generation_overrides", {})))
            print(
                f"[topology-audit-fixed-graph] seed={int(seed)} variant={variant['variant_name']} starting",
                flush=True,
            )
            if str(variant.get("exporter", "diffusion")) == "masked_room":
                summary = export_masked_variant(
                    run_dir=run_dir,
                    mission_graph=copy.deepcopy(mission_graph),
                    variant_name=str(variant["variant_name"]),
                    out_dir=seed_dir,
                    seed=int(seed),
                    generation_overrides=variant_generation_overrides,
                )
            else:
                export_kwargs = {k: v for k, v in variant.items() if k not in {"exporter", "variant_generation_overrides"}}
                summary = export_variant(
                    run_dir=run_dir,
                    mission_graph=copy.deepcopy(mission_graph),
                    out_dir=seed_dir,
                    seed=int(seed),
                    generation_overrides=variant_generation_overrides,
                    fast_sampling_checkpoint=args.lcm_checkpoint,
                    **export_kwargs,
                )
            variant_name = str(variant["variant_name"])
            seed_summary["variants"][variant_name] = summary
            aggregate_inputs[variant_name].append(summary)
            metrics = summary.get("metrics", {})
            print(
                f"[topology-audit-fixed-graph] seed={int(seed)} variant={variant_name} done "
                f"repair_rate={metrics.get('repair_rate')} "
                f"overwrite={metrics.get('avg_final_graph_marker_overwrite_rate')}",
                flush=True,
            )
        per_seed.append(seed_summary)

    aggregate = {
        variant_name: _aggregate_variant(entries)
        for variant_name, entries in aggregate_inputs.items()
    }

    output = {
        "run_dir": str(run_dir),
        "mission_graph": mission_graph_source,
        "seeds": [int(seed) for seed in args.seeds],
        "generation_overrides": generation_overrides,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_sanitize(output), indent=2), encoding="utf-8")
    (args.output_dir / "search_algorithm_comparison.json").write_text(
        json.dumps(
            _json_sanitize(
                {
                    variant_name: aggregate_payload.get("search_algorithm_aggregate", {})
                    for variant_name, aggregate_payload in aggregate.items()
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[topology-audit-fixed-graph] wrote {summary_path}", flush=True)
    return {"output": str(summary_path)}


def main() -> None:
    args = parse_args()
    print(json.dumps(run_from_args(args), indent=2))


if __name__ == "__main__":
    main()
