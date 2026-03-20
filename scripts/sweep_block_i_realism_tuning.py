"""Reproducible sweep for Block-I realism tuning coefficients.

Usage:
    f:/KLTN/.venv/Scripts/python.exe scripts/sweep_block_i_realism_tuning.py
    f:/KLTN/.venv/Scripts/python.exe scripts/sweep_block_i_realism_tuning.py --num-generated 24 --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import run_block_i_benchmark_from_scratch
from src.generation.realism_profiles import get_realism_tuning_profile


def _objective_general(
    summary_payload: Dict[str, Any],
    *,
    min_constraint_valid_rate: float,
    min_overall_completeness: float,
) -> Tuple[float, Dict[str, float]]:
    comp = summary_payload.get("completeness", {}) or {}
    rob = summary_payload.get("robustness", {}) or {}
    expr = summary_payload.get("expressive_range", {}) or {}
    ref_cmp = summary_payload.get("reference_comparison", {}) or {}
    gen = summary_payload.get("generated_descriptor_means", {}) or {}
    ref = summary_payload.get("reference_descriptor_means", {}) or {}

    overall_completeness = float(comp.get("overall_completeness", 0.0))
    constraint_valid_rate = float(comp.get("constraint_valid_rate", 0.0))
    directionality_gap = float(comp.get("path_directionality_gap_mean", 1.0))
    generation_rejections = float(rob.get("mean_generation_constraint_rejections", 0.0))
    descriptor_diversity = float(expr.get("descriptor_diversity", 0.0))
    expressive_overlap = float(ref_cmp.get("expressive_overlap_reference", 0.0))

    def _ratio_gap(metric: str) -> float:
        g = float(gen.get(metric, 0.0))
        r = float(ref.get(metric, 0.0))
        if r <= 1e-6:
            return abs(g - r)
        return abs(g - r) / max(1e-6, r)

    node_gap = _ratio_gap("num_nodes")
    edge_gap = _ratio_gap("num_edges")
    topo_gap = _ratio_gap("topology_complexity")
    path_gap = _ratio_gap("path_length")

    # Lower is better.
    score = (
        (2.4 * node_gap)
        + (2.2 * edge_gap)
        + (1.4 * topo_gap)
        + (0.8 * path_gap)
        + (0.9 * directionality_gap)
        + (0.9 * generation_rejections)
        + (0.8 * max(0.0, (0.10 - descriptor_diversity) / 0.10))
        + (0.6 * max(0.0, (0.08 - expressive_overlap) / 0.08))
        + (8.0 * max(0.0, 1.0 - overall_completeness))
        + (10.0 * max(0.0, 1.0 - constraint_valid_rate))
    )
    if constraint_valid_rate < float(min_constraint_valid_rate):
        score += 1000.0 * float(min_constraint_valid_rate - constraint_valid_rate)
    if overall_completeness < float(min_overall_completeness):
        score += 700.0 * float(min_overall_completeness - overall_completeness)

    return float(score), {
        "overall_completeness": overall_completeness,
        "constraint_valid_rate": constraint_valid_rate,
        "path_directionality_gap_mean": directionality_gap,
        "mean_generation_constraint_rejections": generation_rejections,
        "descriptor_diversity": descriptor_diversity,
        "expressive_overlap_reference": expressive_overlap,
        "num_nodes_gap_ratio": float(node_gap),
        "num_edges_gap_ratio": float(edge_gap),
        "topology_complexity_gap_ratio": float(topo_gap),
        "path_length_gap_ratio": float(path_gap),
    }


def _objective_node_scale(
    summary_payload: Dict[str, Any],
    *,
    min_constraint_valid_rate: float,
    min_overall_completeness: float,
) -> Tuple[float, Dict[str, float]]:
    comp = summary_payload.get("completeness", {}) or {}
    rob = summary_payload.get("robustness", {}) or {}
    expr = summary_payload.get("expressive_range", {}) or {}
    ref_cmp = summary_payload.get("reference_comparison", {}) or {}
    gen = summary_payload.get("generated_descriptor_means", {}) or {}
    ref = summary_payload.get("reference_descriptor_means", {}) or {}

    overall_completeness = float(comp.get("overall_completeness", 0.0))
    constraint_valid_rate = float(comp.get("constraint_valid_rate", 0.0))
    directionality_gap = float(comp.get("path_directionality_gap_mean", 1.0))
    generation_rejections = float(rob.get("mean_generation_constraint_rejections", 0.0))
    descriptor_diversity = float(expr.get("descriptor_diversity", 0.0))
    expressive_overlap = float(ref_cmp.get("expressive_overlap_reference", 0.0))

    def _ratio_gap(metric: str) -> float:
        g = float(gen.get(metric, 0.0))
        r = float(ref.get(metric, 0.0))
        if r <= 1e-6:
            return abs(g - r)
        return abs(g - r) / max(1e-6, r)

    node_gap = _ratio_gap("num_nodes")
    edge_gap = _ratio_gap("num_edges")
    topo_gap = _ratio_gap("topology_complexity")
    path_gap = _ratio_gap("path_length")
    gating_gap = _ratio_gap("gating_density")
    gate_depth_gap = _ratio_gap("gate_depth_ratio")

    score = (
        (3.4 * node_gap)
        + (3.1 * edge_gap)
        + (1.2 * topo_gap)
        + (0.6 * path_gap)
        + (0.8 * gating_gap)
        + (0.6 * gate_depth_gap)
        + (0.8 * directionality_gap)
        + (1.1 * generation_rejections)
        + (0.9 * max(0.0, (0.10 - descriptor_diversity) / 0.10))
        + (0.7 * max(0.0, (0.08 - expressive_overlap) / 0.08))
        + (8.0 * max(0.0, 1.0 - overall_completeness))
        + (12.0 * max(0.0, 1.0 - constraint_valid_rate))
    )
    if constraint_valid_rate < float(min_constraint_valid_rate):
        score += 1300.0 * float(min_constraint_valid_rate - constraint_valid_rate)
    if overall_completeness < float(min_overall_completeness):
        score += 900.0 * float(min_overall_completeness - overall_completeness)

    return float(score), {
        "overall_completeness": overall_completeness,
        "constraint_valid_rate": constraint_valid_rate,
        "path_directionality_gap_mean": directionality_gap,
        "mean_generation_constraint_rejections": generation_rejections,
        "descriptor_diversity": descriptor_diversity,
        "expressive_overlap_reference": expressive_overlap,
        "num_nodes_gap_ratio": float(node_gap),
        "num_edges_gap_ratio": float(edge_gap),
        "topology_complexity_gap_ratio": float(topo_gap),
        "path_length_gap_ratio": float(path_gap),
        "gating_density_gap_ratio": float(gating_gap),
        "gate_depth_gap_ratio": float(gate_depth_gap),
    }


def _candidate_configs() -> List[Tuple[str, Dict[str, float]]]:
    return [
        ("baseline", {}),
        (
            "moderate_node_edge_push",
            {
                "adapt_node_gain": 0.46,
                "adapt_edge_density_gain": 0.44,
                "adapt_edge_budget_gain": 0.30,
                "prior_node_boost_gain": 0.40,
                "prior_edge_boost_gain": 0.34,
            },
        ),
        (
            "aggressive_structural_push",
            {
                "adapt_node_gain": 0.56,
                "adapt_edge_density_gain": 0.56,
                "adapt_edge_budget_gain": 0.38,
                "prior_node_boost_gain": 0.52,
                "prior_edge_boost_gain": 0.42,
                "node_cap_floor_ratio": 0.98,
                "node_cap_expand_ratio": 1.16,
                "node_cap_hard_cap_ratio": 1.45,
            },
        ),
        (
            "high_node_low_edge",
            {
                "adapt_node_gain": 0.60,
                "adapt_edge_density_gain": 0.30,
                "adapt_edge_budget_gain": 0.20,
                "prior_node_boost_gain": 0.50,
                "node_cap_floor_ratio": 0.99,
                "node_cap_expand_ratio": 1.18,
            },
        ),
        (
            "balanced_high_with_caps",
            {
                "adapt_node_gain": 0.52,
                "adapt_edge_density_gain": 0.48,
                "adapt_edge_budget_gain": 0.34,
                "prior_node_boost_gain": 0.46,
                "prior_edge_boost_gain": 0.38,
                "prior_node_boost_max": 1.34,
                "prior_edge_boost_max": 1.28,
                "node_cap_floor_ratio": 1.00,
                "node_cap_expand_ratio": 1.20,
                "node_cap_hard_cap_ratio": 1.50,
            },
        ),
        (
            "density_focus",
            {
                "adapt_node_gain": 0.36,
                "adapt_edge_density_gain": 0.62,
                "adapt_edge_budget_gain": 0.44,
                "prior_node_boost_gain": 0.28,
                "prior_edge_boost_gain": 0.48,
            },
        ),
    ]


def _candidate_configs_node_scale() -> List[Tuple[str, Dict[str, float]]]:
    return [
        ("baseline", {}),
        (
            "recommended_gate_quality_heavy",
            get_realism_tuning_profile("gate_quality_heavy"),
        ),
        (
            "current_density_focus",
            {
                "adapt_node_gain": 0.36,
                "adapt_edge_density_gain": 0.62,
                "adapt_edge_budget_gain": 0.44,
                "prior_node_boost_gain": 0.28,
                "prior_edge_boost_gain": 0.48,
            },
        ),
        (
            "node_edge_scale_push_a",
            {
                "adapt_node_gain": 0.44,
                "adapt_edge_density_gain": 0.70,
                "adapt_edge_budget_gain": 0.52,
                "prior_node_boost_gain": 0.34,
                "prior_edge_boost_gain": 0.56,
                "node_cap_floor_ratio": 0.98,
                "node_cap_expand_ratio": 1.18,
                "node_cap_hard_cap_ratio": 1.45,
            },
        ),
        (
            "node_edge_scale_push_b",
            {
                "adapt_node_gain": 0.50,
                "adapt_edge_density_gain": 0.78,
                "adapt_edge_budget_gain": 0.58,
                "prior_node_boost_gain": 0.40,
                "prior_edge_boost_gain": 0.64,
                "node_cap_floor_ratio": 1.00,
                "node_cap_expand_ratio": 1.22,
                "node_cap_hard_cap_ratio": 1.55,
            },
        ),
        (
            "node_push_guarded_edges",
            {
                "adapt_node_gain": 0.58,
                "adapt_edge_density_gain": 0.64,
                "adapt_edge_budget_gain": 0.46,
                "prior_node_boost_gain": 0.50,
                "prior_edge_boost_gain": 0.50,
                "node_cap_floor_ratio": 1.02,
                "node_cap_expand_ratio": 1.24,
                "node_cap_hard_cap_ratio": 1.60,
            },
        ),
        (
            "balanced_caps_high",
            {
                "adapt_node_gain": 0.48,
                "adapt_edge_density_gain": 0.74,
                "adapt_edge_budget_gain": 0.56,
                "prior_node_boost_gain": 0.38,
                "prior_edge_boost_gain": 0.60,
                "prior_node_boost_max": 1.38,
                "prior_edge_boost_max": 1.32,
                "node_cap_floor_ratio": 1.00,
                "node_cap_expand_ratio": 1.20,
                "node_cap_hard_cap_ratio": 1.52,
            },
        ),
        (
            "edge_density_bias",
            {
                "adapt_node_gain": 0.36,
                "adapt_edge_density_gain": 0.86,
                "adapt_edge_budget_gain": 0.66,
                "prior_node_boost_gain": 0.28,
                "prior_edge_boost_gain": 0.72,
                "node_cap_floor_ratio": 0.97,
                "node_cap_expand_ratio": 1.15,
                "node_cap_hard_cap_ratio": 1.40,
            },
        ),
    ]


def _evaluate_candidate(
    *,
    name: str,
    realism_tuning: Dict[str, float],
    seed: int,
    num_generated: int,
    population_size: int,
    generations: int,
    min_rooms: int,
    max_rooms: int,
    room_budget_cap: int,
    room_count_bias: float,
    bootstrap_samples: int,
    include_data_audit: bool,
    mode: str,
    min_constraint_valid_rate: float,
    min_overall_completeness: float,
) -> Dict[str, Any]:
    summary = run_block_i_benchmark_from_scratch(
        num_generated=num_generated,
        seed=seed,
        population_size=population_size,
        generations=generations,
        min_rooms=min_rooms,
        max_rooms=max_rooms,
        room_budget_cap=room_budget_cap,
        room_count_bias=room_count_bias,
        rule_space="full",
        calibrate_rule_schedule=False,
        wfc_probe_samples=0,
        search_strategy="ga",
        qd_archive_cells=128,
        qd_init_random_fraction=0.35,
        qd_emitter_mutation_rate=0.18,
        realism_tuning=realism_tuning,
        bootstrap_samples=bootstrap_samples,
        include_data_audit=include_data_audit,
    )
    payload = asdict(summary)
    if str(mode).strip().lower() == "node-scale":
        score, diagnostics = _objective_node_scale(
            payload,
            min_constraint_valid_rate=float(min_constraint_valid_rate),
            min_overall_completeness=float(min_overall_completeness),
        )
    else:
        score, diagnostics = _objective_general(
            payload,
            min_constraint_valid_rate=float(min_constraint_valid_rate),
            min_overall_completeness=float(min_overall_completeness),
        )
    return {
        "name": name,
        "realism_tuning": dict(realism_tuning),
        "objective_score": float(score),
        "diagnostics": diagnostics,
        "summary": payload,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep Block-I realism tuning and pick best config.")
    parser.add_argument("--mode", type=str, default="general", choices=["general", "node-scale"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generated", type=int, default=16)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--min-rooms", type=int, default=8)
    parser.add_argument("--max-rooms", type=int, default=16)
    parser.add_argument("--room-budget-cap", type=int, default=42)
    parser.add_argument("--room-count-bias", type=float, default=0.45)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--min-constraint-valid-rate", type=float, default=1.0)
    parser.add_argument("--min-overall-completeness", type=float, default=1.0)
    parser.add_argument("--include-data-audit", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "block_i_realism_tuning_sweep_2026_03_08.json",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    mode = str(args.mode).strip().lower()
    candidates = _candidate_configs_node_scale() if mode == "node-scale" else _candidate_configs()
    rows: List[Dict[str, Any]] = []
    for idx, (name, realism_tuning) in enumerate(candidates):
        run_seed = int(args.seed) + (idx * 97)
        row = _evaluate_candidate(
            name=name,
            realism_tuning=realism_tuning,
            seed=run_seed,
            num_generated=int(args.num_generated),
            population_size=int(args.population_size),
            generations=int(args.generations),
            min_rooms=int(args.min_rooms),
            max_rooms=int(args.max_rooms),
            room_budget_cap=int(args.room_budget_cap),
            room_count_bias=float(args.room_count_bias),
            bootstrap_samples=int(args.bootstrap_samples),
            include_data_audit=bool(args.include_data_audit),
            mode=mode,
            min_constraint_valid_rate=float(args.min_constraint_valid_rate),
            min_overall_completeness=float(args.min_overall_completeness),
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "name": row["name"],
                    "objective_score": row["objective_score"],
                    "diagnostics": row["diagnostics"],
                },
                ensure_ascii=False,
            )
        )

    rows = sorted(rows, key=lambda item: float(item.get("objective_score", 1e9)))
    best = rows[0] if rows else None

    output_payload = {
        "mode": mode,
        "seed": int(args.seed),
        "num_generated": int(args.num_generated),
        "population_size": int(args.population_size),
        "generations": int(args.generations),
        "min_rooms": int(args.min_rooms),
        "max_rooms": int(args.max_rooms),
        "room_budget_cap": int(args.room_budget_cap),
        "room_count_bias": float(args.room_count_bias),
        "bootstrap_samples": int(args.bootstrap_samples),
        "min_constraint_valid_rate": float(args.min_constraint_valid_rate),
        "min_overall_completeness": float(args.min_overall_completeness),
        "best": best,
        "candidates_sorted": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(json.dumps({"best_name": (best or {}).get("name"), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
