"""Run three focused Block-I improvement experiments and summarize results.

Experiments:
1) baseline
2) scale_heavy (node/edge scale pressure)
3) gate_quality_heavy (gate realism + rejection pressure)
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
from src.generation.realism_profiles import get_realism_tuning_profile


def _score(summary: Dict[str, Any]) -> float:
    comp = summary.get("completeness", {})
    gen = summary.get("generated_descriptor_means", {})
    ref = summary.get("reference_descriptor_means", {})
    rob = summary.get("robustness", {})
    expr = summary.get("expressive_range", {})
    cmp_ = summary.get("reference_comparison", {})

    def rg(name: str) -> float:
        g = float(gen.get(name, 0.0))
        r = float(ref.get(name, 0.0))
        if r <= 1e-6:
            return abs(g - r)
        return abs(g - r) / max(1e-6, r)

    node_gap = rg("num_nodes")
    edge_gap = rg("num_edges")
    gate_gap = rg("gating_density")
    path_gap = rg("path_length")

    overall = float(comp.get("overall_completeness", 0.0))
    valid = float(comp.get("constraint_valid_rate", 0.0))
    rej = float(rob.get("mean_generation_constraint_rejections", 0.0))
    div = float(expr.get("descriptor_diversity", 0.0))
    overlap = float(cmp_.get("expressive_overlap_reference", 0.0))

    return float(
        (2.8 * node_gap)
        + (2.6 * edge_gap)
        + (1.2 * gate_gap)
        + (0.9 * path_gap)
        + (1.0 * rej)
        + (0.8 * max(0.0, (0.10 - div) / 0.10))
        + (0.6 * max(0.0, (0.08 - overlap) / 0.08))
        + (10.0 * max(0.0, 1.0 - overall))
        + (12.0 * max(0.0, 1.0 - valid))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused Block-I improvement experiments")
    parser.add_argument("--output", type=Path, default=Path("results") / "block_i_improvement_experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generated", type=int, default=64)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--min-rooms", type=int, default=8)
    parser.add_argument("--max-rooms", type=int, default=16)
    parser.add_argument("--room-budget-cap", type=int, default=42)
    parser.add_argument("--room-count-bias", type=float, default=0.45)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs: List[Dict[str, Any]] = [
        {"name": "baseline", "realism_tuning": {}},
        {
            "name": "scale_heavy",
            "realism_tuning": get_realism_tuning_profile("scale_heavy"),
        },
        {
            "name": "gate_quality_heavy",
            "realism_tuning": get_realism_tuning_profile("gate_quality_heavy"),
        },
    ]

    rows: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(configs):
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
            realism_tuning=dict(cfg.get("realism_tuning", {})),
            bootstrap_samples=int(args.bootstrap_samples),
            include_data_audit=False,
        )
        payload = {
            "name": str(cfg["name"]),
            "realism_tuning": dict(cfg.get("realism_tuning", {})),
            "summary": summary_obj.__dict__,
        }
        payload["objective_score"] = _score(payload["summary"])
        rows.append(payload)

        (out_dir / f"{cfg['name']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows.sort(key=lambda x: float(x.get("objective_score", 1e9)))
    final = {
        "best": rows[0] if rows else None,
        "results": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_dir / "summary.json"), "best": final.get("best", {}).get("name")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
