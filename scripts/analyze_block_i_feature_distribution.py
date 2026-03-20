"""
Generate many Block-I topologies and analyze feature-frequency distribution.

Primary use:
- "Generate 100 dungeons for statistical analysis"
- "Analyze feature frequency distribution"
- optional empirical weight tuning against VGLC targets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import (
    calibrate_rule_weights_to_vglc,
    extract_graph_descriptor,
    generate_block_i_graphs,
    load_vglc_reference_graphs,
)
from src.generation.grammar import MissionGrammar

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Block-I feature frequency at scale.")
    parser.add_argument("--output", type=Path, default=Path("results") / "block_i_feature_distribution")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--reference-limit", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rooms", type=int, default=8)
    parser.add_argument("--max-rooms", type=int, default=16)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--rule-space", type=str, default="full", choices=["core", "full"])
    parser.add_argument("--room-count-bias", type=float, default=0.45)
    parser.add_argument("--search-strategy", type=str, default="ga", choices=["ga", "cvt_emitter"])
    parser.add_argument("--qd-archive-cells", type=int, default=128)
    parser.add_argument("--qd-init-random-fraction", type=float, default=0.35)
    parser.add_argument("--qd-emitter-mutation-rate", type=float, default=0.18)
    parser.add_argument("--auto-tune", action="store_true", help="Run empirical rule-weight tuning against VGLC.")
    parser.add_argument("--calibration-iterations", type=int, default=4)
    parser.add_argument("--calibration-sample-size", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _descriptor_targets(reference_graphs: List[Any]) -> Dict[str, float]:
    if not reference_graphs:
        return {
            "linearity": 0.45,
            "leniency": 0.50,
            "progression_complexity": 0.65,
            "topology_complexity": 0.45,
        }
    ref_desc = [extract_graph_descriptor(g, grammar=None) for g in reference_graphs]
    return {
        "linearity": float(np.mean([d.linearity for d in ref_desc])),
        "leniency": float(np.mean([d.leniency for d in ref_desc])),
        "progression_complexity": float(np.mean([d.progression_complexity for d in ref_desc])),
        "topology_complexity": float(np.mean([d.topology_complexity for d in ref_desc])),
        "path_length": float(np.mean([d.path_length for d in ref_desc])),
        "num_nodes": float(np.mean([d.num_nodes for d in ref_desc])),
        "cycle_density": float(np.mean([d.cycle_density for d in ref_desc])),
        "shortcut_density": float(np.mean([d.shortcut_density for d in ref_desc])),
        "gate_depth_ratio": float(np.mean([d.gate_depth_ratio for d in ref_desc])),
        "path_depth_ratio": float(np.mean([d.path_depth_ratio for d in ref_desc])),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if not args.verbose:
        logging.getLogger("src.data.vglc_utils").setLevel(logging.ERROR)
        logging.getLogger("src.generation.grammar").setLevel(logging.ERROR)
        logging.getLogger("src.generation.evolutionary_director").setLevel(logging.ERROR)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_graphs = load_vglc_reference_graphs(
        data_root=args.data_root,
        limit=args.reference_limit,
        filter_virtual_start_pointers=True,
    )
    descriptor_targets = _descriptor_targets(reference_graphs)

    tuned_weights: Optional[Dict[str, float]] = None
    calibration_payload: Dict[str, Any] = {"enabled": bool(args.auto_tune), "history": []}
    if args.auto_tune:
        cal = calibrate_rule_weights_to_vglc(
            reference_graphs,
            seed=int(args.seed),
            iterations=int(args.calibration_iterations),
            sample_size=int(args.calibration_sample_size),
            min_rooms=int(args.min_rooms),
            max_rooms=int(args.max_rooms),
            population_size=int(args.population_size),
            generations=int(args.generations),
            rule_space=str(args.rule_space),
            room_count_bias=float(args.room_count_bias),
            search_strategy=str(args.search_strategy),
            qd_archive_cells=int(args.qd_archive_cells),
            qd_init_random_fraction=float(args.qd_init_random_fraction),
            qd_emitter_mutation_rate=float(args.qd_emitter_mutation_rate),
        )
        tuned_weights = {
            str(k): float(v)
            for k, v in dict(cal.get("calibrated_rule_weights", {})).items()
            if np.isfinite(v) and float(v) >= 0.0
        }
        calibration_payload = {
            "enabled": True,
            "best_gap": float(cal.get("best_gap", 0.0)),
            "history": cal.get("history", []),
        }

    graphs, times = generate_block_i_graphs(
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        min_rooms=int(args.min_rooms),
        max_rooms=int(args.max_rooms),
        population_size=int(args.population_size),
        generations=int(args.generations),
        rule_space=str(args.rule_space),
        rule_weight_overrides=tuned_weights,
        descriptor_targets=descriptor_targets,
        room_count_bias=float(args.room_count_bias),
        search_strategy=str(args.search_strategy),
        qd_archive_cells=int(args.qd_archive_cells),
        qd_init_random_fraction=float(args.qd_init_random_fraction),
        qd_emitter_mutation_rate=float(args.qd_emitter_mutation_rate),
    )

    grammar = MissionGrammar(seed=int(args.seed) + 5000)
    rows: List[Dict[str, Any]] = []
    for idx, g in enumerate(graphs):
        d = extract_graph_descriptor(g, grammar=grammar)
        rows.append(
            {
                "sample": int(idx),
                "generation_time_sec": float(times[idx]) if idx < len(times) else float("nan"),
                "num_nodes": int(d.num_nodes),
                "num_edges": int(d.num_edges),
                "path_length": int(d.path_length),
                "linearity": float(d.linearity),
                "leniency": float(d.leniency),
                "progression_complexity": float(d.progression_complexity),
                "topology_complexity": float(d.topology_complexity),
                "cycle_density": float(d.cycle_density),
                "shortcut_density": float(d.shortcut_density),
                "gate_depth_ratio": float(d.gate_depth_ratio),
                "path_depth_ratio": float(d.path_depth_ratio),
                "directionality_gap": float(d.directionality_gap),
                "constraint_valid": float(d.constraint_valid),
                "repair_applied": float(d.repair_applied),
                "total_repairs": int(d.total_repairs),
                "lock_key_repairs": int(d.lock_key_repairs),
                "progression_repairs": int(d.progression_repairs),
                "wave3_repairs": int(d.wave3_repairs),
                "rule_applied": int(d.rule_applied),
                "rule_skipped": int(d.rule_skipped),
                "puzzle_count": int(d.puzzle_count),
                "item_count": int(d.item_count),
                "switch_count": int(d.switch_count),
                "stair_count": int(d.stair_count),
                "gate_variety": float(d.gate_variety),
                "has_switch_feature": float(d.switch_count > 0),
                "has_stair_feature": float(d.stair_count > 0),
                "has_shortcut": float(d.shortcut_density > 0.0),
                "has_cycle": float(d.cycle_density > 0.0),
                "has_wave3_repair": float(d.wave3_repairs > 0),
            }
        )

    raw_df = pd.DataFrame(rows)
    summary = {
        "num_samples": int(len(raw_df)),
        "mean_generation_time_sec": float(raw_df["generation_time_sec"].mean(skipna=True)) if not raw_df.empty else 0.0,
        "constraint_valid_rate": float(raw_df["constraint_valid"].mean(skipna=True)) if not raw_df.empty else 0.0,
        "repair_rate": float(raw_df["repair_applied"].mean(skipna=True)) if not raw_df.empty else 0.0,
        "feature_presence_rate": {
            "switch": float(raw_df["has_switch_feature"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "stair": float(raw_df["has_stair_feature"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "shortcut": float(raw_df["has_shortcut"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "cycle": float(raw_df["has_cycle"].mean(skipna=True)) if not raw_df.empty else 0.0,
        },
        "descriptor_means": {
            "linearity": float(raw_df["linearity"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "leniency": float(raw_df["leniency"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "progression_complexity": float(raw_df["progression_complexity"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "topology_complexity": float(raw_df["topology_complexity"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "cycle_density": float(raw_df["cycle_density"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "shortcut_density": float(raw_df["shortcut_density"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "gate_depth_ratio": float(raw_df["gate_depth_ratio"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "path_depth_ratio": float(raw_df["path_depth_ratio"].mean(skipna=True)) if not raw_df.empty else 0.0,
            "directionality_gap": float(raw_df["directionality_gap"].mean(skipna=True)) if not raw_df.empty else 0.0,
        },
        "descriptor_targets": descriptor_targets,
        "calibration": calibration_payload,
        "tuned_rule_weights": tuned_weights or {},
    }

    raw_path = out_dir / "feature_distribution_raw.csv"
    summary_path = out_dir / "feature_distribution_summary.json"
    md_path = out_dir / "feature_distribution_report.md"
    raw_df.to_csv(raw_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        mean_row = raw_df.mean(numeric_only=True).to_frame("mean").reset_index().rename(columns={"index": "metric"})
        mean_table = mean_row.to_markdown(index=False)
    except Exception:
        mean_table = raw_df.mean(numeric_only=True).to_string()

    lines = [
        "# Block I Feature Distribution Report",
        "",
        f"- samples: {summary['num_samples']}",
        f"- constraint_valid_rate: {summary['constraint_valid_rate']:.4f}",
        f"- repair_rate: {summary['repair_rate']:.4f}",
        f"- mean_generation_time_sec: {summary['mean_generation_time_sec']:.4f}",
        "",
        "## Feature Presence Rates",
        "",
        f"- switch: {summary['feature_presence_rate']['switch']:.4f}",
        f"- stair: {summary['feature_presence_rate']['stair']:.4f}",
        f"- shortcut: {summary['feature_presence_rate']['shortcut']:.4f}",
        f"- cycle: {summary['feature_presence_rate']['cycle']:.4f}",
        "",
        "## Mean Numeric Metrics",
        "",
        mean_table,
        "",
        f"- auto_tune_enabled: {bool(args.auto_tune)}",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved feature-distribution outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
