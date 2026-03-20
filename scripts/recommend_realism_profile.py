"""Run multi-profile Block-I benchmark and rank profiles by weighted objective.

Usage:
    python scripts/recommend_realism_profile.py --num-generated 16 --output-dir results/profile_recommendation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import run_block_i_benchmark_from_scratch
from src.generation.realism_profiles import (
    get_realism_tuning_profile,
    list_realism_tuning_profiles,
)


def _ratio_gap(generated: float, reference: float) -> float:
    g = float(generated)
    r = float(reference)
    if r <= 1e-6:
        return abs(g - r)
    return abs(g - r) / max(1e-6, r)


def _score(summary: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, float]:
    comp = summary.get("completeness", {})
    expr = summary.get("expressive_range", {})
    cmp_ = summary.get("reference_comparison", {})
    gen = summary.get("generated_descriptor_means", {})
    ref = summary.get("reference_descriptor_means", {})

    fidelity = float(cmp_.get("fidelity_js_divergence", 1.0))
    overlap = float(cmp_.get("expressive_overlap_reference", 0.0))
    diversity = float(expr.get("descriptor_diversity", 0.0))
    node_gap = _ratio_gap(float(gen.get("num_nodes", 0.0)), float(ref.get("num_nodes", 0.0)))
    edge_gap = _ratio_gap(float(gen.get("num_edges", 0.0)), float(ref.get("num_edges", 0.0)))
    completeness_penalty = max(0.0, 1.0 - float(comp.get("overall_completeness", 0.0)))
    validity_penalty = max(0.0, 1.0 - float(comp.get("constraint_valid_rate", 0.0)))

    objective = float(
        (weights["fidelity"] * fidelity)
        + (weights["overlap"] * max(0.0, 1.0 - overlap))
        + (weights["diversity"] * max(0.0, 1.0 - diversity))
        + (weights["node_gap"] * node_gap)
        + (weights["edge_gap"] * edge_gap)
        + (weights["completeness"] * completeness_penalty)
        + (weights["validity"] * validity_penalty)
    )

    return {
        "objective_score": objective,
        "fidelity_js_divergence": fidelity,
        "expressive_overlap_reference": overlap,
        "descriptor_diversity": diversity,
        "num_nodes_gap_ratio": float(node_gap),
        "num_edges_gap_ratio": float(edge_gap),
        "overall_completeness": float(comp.get("overall_completeness", 0.0)),
        "constraint_valid_rate": float(comp.get("constraint_valid_rate", 0.0)),
    }


def _parse_weights(raw: str) -> Dict[str, float]:
    default = {
        "fidelity": 0.35,
        "overlap": 0.20,
        "diversity": 0.20,
        "node_gap": 0.10,
        "edge_gap": 0.10,
        "completeness": 6.0,
        "validity": 8.0,
    }
    if not str(raw).strip():
        return default
    parsed = dict(default)
    for chunk in str(raw).split(","):
        token = str(chunk).strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        key_name = str(key).strip().lower()
        if key_name not in parsed:
            continue
        try:
            parsed[key_name] = float(value)
        except ValueError:
            continue
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recommend best realism profile from weighted benchmark comparison")
    parser.add_argument(
        "--profiles",
        type=str,
        default="gate_quality_heavy,engine_default",
        help="Comma-separated profile names from src/generation/realism_profiles.py",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generated", type=int, default=16)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--min-rooms", type=int, default=8)
    parser.add_argument("--max-rooms", type=int, default=16)
    parser.add_argument("--room-budget-cap", type=int, default=42)
    parser.add_argument("--room-count-bias", type=float, default=0.45)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help=(
            "Optional comma-separated weights, e.g. "
            "fidelity=0.4,overlap=0.25,diversity=0.2,node_gap=0.08,edge_gap=0.07"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "profile_recommendation",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    requested_profiles = [p.strip() for p in str(args.profiles).split(",") if p.strip()]
    available = set(list_realism_tuning_profiles(include_engine_default=True))
    profiles: List[str] = []
    for profile in requested_profiles:
        if profile in available:
            profiles.append(profile)
    if not profiles:
        raise ValueError("No valid profiles provided. Use --profiles with known profile names.")

    weights = _parse_weights(str(args.weights))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, profile in enumerate(profiles):
        realism_tuning = get_realism_tuning_profile(profile)
        summary_obj = run_block_i_benchmark_from_scratch(
            num_generated=int(args.num_generated),
            seed=int(args.seed) + (idx * 97),
            population_size=int(args.population_size),
            generations=int(args.generations),
            min_rooms=int(args.min_rooms),
            max_rooms=int(args.max_rooms),
            room_budget_cap=int(args.room_budget_cap),
            room_count_bias=float(args.room_count_bias),
            rule_space="full",
            calibrate_rule_schedule=False,
            wfc_probe_samples=0,
            search_strategy="ga",
            qd_archive_cells=128,
            qd_init_random_fraction=0.35,
            qd_emitter_mutation_rate=0.18,
            realism_tuning=(realism_tuning or None),
            bootstrap_samples=int(args.bootstrap_samples),
            include_data_audit=False,
        )
        summary = summary_obj.__dict__
        metrics = _score(summary, weights)
        payload = {
            "name": profile,
            "realism_tuning": dict(realism_tuning),
            "metrics": metrics,
            "summary": summary,
        }
        rows.append(payload)
        (output_dir / f"{profile}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows.sort(key=lambda item: float(item.get("metrics", {}).get("objective_score", 1e9)))
    final = {
        "weights": weights,
        "best": rows[0] if rows else None,
        "results": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "best": ((rows[0] if rows else {}) or {}).get("name"),
                "summary": str(output_dir / "summary.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
