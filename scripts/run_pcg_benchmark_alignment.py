"""Evaluate generated topology graphs against the external PCG Benchmark Zelda task."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import (
    extract_graph_descriptor,
    generate_block_i_graphs,
    load_vglc_reference_graphs,
    run_block_i_benchmark,
)
from src.evaluation.pcg_benchmark_alignment import (
    PCG_BENCHMARK_ZELDA_VARIANTS,
    evaluate_graphs_with_pcg_benchmark_zelda,
)
from src.utils.stable_seed import stable_seed_offset


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    rule_space: str
    search_strategy: str


METHODS: Dict[str, MethodConfig] = {
    "FULL_GA": MethodConfig(name="FULL_GA", rule_space="full", search_strategy="ga"),
    "FULL_CVT": MethodConfig(name="FULL_CVT", rule_space="full", search_strategy="cvt_emitter"),
    "CORE_GA": MethodConfig(name="CORE_GA", rule_space="core", search_strategy="ga"),
}


def _method_list(raw: str) -> List[MethodConfig]:
    out: List[MethodConfig] = []
    for token in [t.strip().upper() for t in str(raw).split(",") if t.strip()]:
        if token not in METHODS:
            raise ValueError(f"Unknown method '{token}'. Valid: {sorted(METHODS.keys())}")
        out.append(METHODS[token])
    return out or [METHODS["FULL_GA"], METHODS["FULL_CVT"]]


def _problem_list(raw: str) -> List[str]:
    problems = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not problems:
        problems = ["zelda-v0"]
    invalid = [name for name in problems if name not in PCG_BENCHMARK_ZELDA_VARIANTS]
    if invalid:
        raise ValueError(
            f"Unsupported PCG benchmark Zelda problems: {invalid}. "
            f"Valid: {sorted(PCG_BENCHMARK_ZELDA_VARIANTS.keys())}"
        )
    return problems


def _problem_room_budget(problem_name: str, room_budget_cap: int) -> Tuple[int, int]:
    cap = max(12, int(room_budget_cap))
    if str(problem_name) == "zelda-large-v0":
        return (18, int(min(cap, 32)))
    return (8, int(min(cap, 16)))


def _descriptor_targets(reference_graphs: Sequence[Any]) -> Dict[str, float]:
    desc = [extract_graph_descriptor(g, grammar=None) for g in reference_graphs]
    if not desc:
        return {
            "linearity": 0.45,
            "leniency": 0.5,
            "progression_complexity": 0.68,
            "topology_complexity": 0.45,
            "path_length": 9.0,
            "num_nodes": 20.0,
        }
    return {
        "linearity": float(np.mean([d.linearity for d in desc])),
        "leniency": float(np.mean([d.leniency for d in desc])),
        "progression_complexity": float(np.mean([d.progression_complexity for d in desc])),
        "topology_complexity": float(np.mean([d.topology_complexity for d in desc])),
        "path_length": float(np.mean([d.path_length for d in desc])),
        "num_nodes": float(np.mean([d.num_nodes for d in desc])),
    }


def _safe_mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _build_external_summary(problem_name: str, external: Dict[str, Any]) -> Dict[str, float]:
    variant = PCG_BENCHMARK_ZELDA_VARIANTS[problem_name]
    rows = list(external.get("rows", []))
    return {
        "external_semantic_valid_rate": _safe_mean(rows, "semantic_valid"),
        "external_quality_pass_rate": float(external.get("quality_mean", 0.0)),
        "external_quality_detail_mean": _safe_mean(rows, "quality"),
        "external_diversity_pass_rate": float(external.get("diversity_mean", 0.0)),
        "external_diversity_detail_mean": _safe_mean(rows, "diversity"),
        "external_controlability_pass_rate": float(external.get("controlability_mean", 0.0)),
        "external_controlability_detail_mean": _safe_mean(rows, "controlability"),
        "external_solution_length_pass_rate": _safe_mean(rows, "solution_length_pass"),
        "external_enemy_band_pass_rate": _safe_mean(rows, "enemy_band_pass"),
        "external_mean_solution_length": _safe_mean(rows, "solution_length"),
        "external_solution_length_target": float(variant.solution_length),
        "external_mean_enemy_count": _safe_mean(rows, "enemies"),
        "external_enemy_target": float(variant.enemies),
        "external_control_fallback_rate": _safe_mean(rows, "control_fallback_applied"),
        "external_mean_initial_solution_length": _safe_mean(rows, "initial_solution_length"),
        "external_mean_abs_player_key_error": _safe_mean(rows, "player_key_abs_error"),
        "external_mean_abs_key_door_error": _safe_mean(rows, "key_door_abs_error"),
        "external_mean_initial_abs_player_key_error": _safe_mean(rows, "player_key_abs_error_initial"),
        "external_mean_initial_abs_key_door_error": _safe_mean(rows, "key_door_abs_error_initial"),
        "external_mean_graph_player_key_raw": _safe_mean(rows, "graph_player_key_raw"),
        "external_mean_graph_key_door_raw": _safe_mean(rows, "graph_key_door_raw"),
        "external_mean_graph_player_key_aligned": _safe_mean(rows, "graph_player_key"),
        "external_mean_graph_key_door_aligned": _safe_mean(rows, "graph_key_door"),
        "external_mean_content_player_key_initial": _safe_mean(rows, "content_player_key_initial"),
        "external_mean_content_key_door_initial": _safe_mean(rows, "content_key_door_initial"),
        "external_mean_content_player_key": _safe_mean(rows, "content_player_key"),
        "external_mean_content_key_door": _safe_mean(rows, "content_key_door"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align generated topologies with the external PCG Benchmark Zelda task.")
    parser.add_argument("--output", type=Path, default=Path("results") / "pcg_benchmark_alignment")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--reference-limit", type=int, default=None)
    parser.add_argument("--methods", type=str, default="FULL_GA,FULL_CVT")
    parser.add_argument("--problems", type=str, default="zelda-v0,zelda-enemies-v0,zelda-large-v0")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--room-budget-cap", type=int, default=42)
    parser.add_argument("--qd-archive-cells", type=int, default=128)
    parser.add_argument("--qd-init-random-fraction", type=float, default=0.35)
    parser.add_argument("--qd-emitter-mutation-rate", type=float, default=0.18)
    parser.add_argument("--control-mode", type=str, default="graph", choices=["graph", "content"])
    parser.add_argument("--pcg-benchmark-repo", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    methods = _method_list(args.methods)
    problems = _problem_list(args.problems)

    refs = load_vglc_reference_graphs(data_root=args.data_root, limit=args.reference_limit)
    if not refs:
        raise RuntimeError(f"No reference graphs loaded from {args.data_root}")

    descriptor_targets = _descriptor_targets(refs)
    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    benchmark_payloads: Dict[str, Any] = {}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for problem_name in problems:
        min_rooms, max_rooms = _problem_room_budget(problem_name, int(args.room_budget_cap))
        for method in methods:
            run_seed = int(args.seed) + stable_seed_offset((problem_name, method.name), modulo=100000)
            logger.info(
                "PCG benchmark alignment: problem=%s method=%s room_budget=[%d,%d] samples=%d",
                problem_name,
                method.name,
                min_rooms,
                max_rooms,
                int(args.num_samples),
            )
            t0 = time.time()
            graphs, gen_times = generate_block_i_graphs(
                num_samples=int(args.num_samples),
                seed=run_seed,
                min_rooms=int(min_rooms),
                max_rooms=int(max_rooms),
                population_size=int(args.population_size),
                generations=int(args.generations),
                rule_space=str(method.rule_space),
                descriptor_targets=descriptor_targets,
                room_count_bias=0.45,
                search_strategy=str(method.search_strategy),
                qd_archive_cells=int(args.qd_archive_cells),
                qd_init_random_fraction=float(args.qd_init_random_fraction),
                qd_emitter_mutation_rate=float(args.qd_emitter_mutation_rate),
            )
            internal_bench = run_block_i_benchmark(generated_graphs=graphs, reference_graphs=refs, generation_times=gen_times)
            external = evaluate_graphs_with_pcg_benchmark_zelda(
                graphs,
                problem_name=problem_name,
                control_mode=str(args.control_mode),
                repo_path=args.pcg_benchmark_repo,
                seed=run_seed + 5000,
            )
            benchmark_payloads[f"{problem_name}:{method.name}"] = {
                "internal": asdict(internal_bench),
                "external": {
                    "problem_name": str(external["problem_name"]),
                    "control_mode": str(external["control_mode"]),
                    "quality_pass_rate": float(external["quality_mean"]),
                    "diversity_pass_rate": float(external["diversity_mean"]),
                    "controlability_pass_rate": float(external["controlability_mean"]),
                    "detail_summary": _build_external_summary(problem_name, external),
                    "rows": external["rows"],
                },
            }

            external_rows = external["rows"]
            for idx, row in enumerate(external_rows):
                raw_rows.append(
                    {
                        "problem_name": problem_name,
                        "method": method.name,
                        "index": int(idx),
                        "control_mode": str(args.control_mode),
                        "quality": float(row.get("quality", 0.0)),
                        "diversity": float(row.get("diversity", 0.0)),
                        "controlability": float(row.get("controlability", 0.0)),
                        "mapper_mode": str(row.get("mapper_mode", "free_routed")),
                        "semantic_valid": float(row.get("semantic_valid", 1.0)),
                        "semantic_error": str(row.get("semantic_error", "")),
                        "control_fallback_applied": float(row.get("control_fallback_applied", 0.0)),
                        "quality_pass": float(row.get("quality_pass", 0.0)),
                        "diversity_pass": float(row.get("diversity_pass", 0.0)),
                        "controlability_pass": float(row.get("controlability_pass", 0.0)),
                        "regions": float(row.get("regions", 0.0)),
                        "players": float(row.get("players", 0.0)),
                        "keys": float(row.get("keys", 0.0)),
                        "doors": float(row.get("doors", 0.0)),
                        "enemies": float(row.get("enemies", 0.0)),
                        "player_key_info": float(row.get("player_key_info", 0.0)),
                        "key_door_info": float(row.get("key_door_info", 0.0)),
                        "player_key_control": float(row.get("player_key_control", 0.0)),
                        "key_door_control": float(row.get("key_door_control", 0.0)),
                        "player_key_abs_error_initial": float(row.get("player_key_abs_error_initial", 0.0)),
                        "key_door_abs_error_initial": float(row.get("key_door_abs_error_initial", 0.0)),
                        "player_key_abs_error": float(row.get("player_key_abs_error", 0.0)),
                        "key_door_abs_error": float(row.get("key_door_abs_error", 0.0)),
                        "initial_solution_length": float(row.get("initial_solution_length", 0.0)),
                        "solution_length": float(row.get("solution_length", 0.0)),
                        "solution_length_target": float(row.get("solution_length_target", 0.0)),
                        "solution_length_pass": float(row.get("solution_length_pass", 0.0)),
                        "enemy_band_pass": float(row.get("enemy_band_pass", 0.0)),
                        "graph_player_key": float(row.get("graph_player_key", 0.0)),
                        "graph_key_door": float(row.get("graph_key_door", 0.0)),
                        "graph_player_key_raw": float(row.get("graph_player_key_raw", 0.0)),
                        "graph_key_door_raw": float(row.get("graph_key_door_raw", 0.0)),
                        "content_player_key_initial": float(row.get("content_player_key_initial", 0.0)),
                        "content_key_door_initial": float(row.get("content_key_door_initial", 0.0)),
                        "content_player_key": float(row.get("content_player_key", 0.0)),
                        "content_key_door": float(row.get("content_key_door", 0.0)),
                        "enemy_target": float(row.get("enemy_target", 0.0)),
                        "mapped_enemy_count": float(row.get("mapped_enemy_count", 0.0)),
                    }
                )

            external_summary = _build_external_summary(problem_name, external)
            summary_rows.append(
                {
                    "problem_name": problem_name,
                    "method": method.name,
                    "n": int(len(graphs)),
                    "control_mode": str(args.control_mode),
                    "min_rooms": int(min_rooms),
                    "max_rooms": int(max_rooms),
                    "internal_overall_completeness": float(internal_bench.completeness.get("overall_completeness", 0.0)),
                    "internal_constraint_valid_rate": float(internal_bench.completeness.get("constraint_valid_rate", 0.0)),
                    "internal_key_before_lock_rate": float(internal_bench.completeness.get("key_before_lock_rate", 0.0)),
                    "internal_switch_before_gate_rate": float(internal_bench.completeness.get("switch_before_gate_rate", 0.0)),
                    "internal_battery_satisfaction_rate": float(
                        internal_bench.completeness.get("battery_satisfaction_rate", 0.0)
                    ),
                    **external_summary,
                    "internal_novelty_vs_reference": float(
                        internal_bench.reference_comparison.get("novelty_vs_reference", 0.0)
                    ),
                    "internal_expressive_overlap_reference": float(
                        internal_bench.reference_comparison.get("expressive_overlap_reference", 0.0)
                    ),
                    "linearity": float(internal_bench.generated_descriptor_means.get("linearity", 0.0)),
                    "progression_complexity": float(
                        internal_bench.generated_descriptor_means.get("progression_complexity", 0.0)
                    ),
                    "topology_complexity": float(
                        internal_bench.generated_descriptor_means.get("topology_complexity", 0.0)
                    ),
                    "path_redundancy": float(internal_bench.generated_descriptor_means.get("path_redundancy", 0.0)),
                    "articulation_ratio": float(internal_bench.generated_descriptor_means.get("articulation_ratio", 0.0)),
                    "branch_utility_rate": float(
                        internal_bench.generated_descriptor_means.get("branch_utility_rate", 0.0)
                    ),
                    "secret_content_discoverability_rate": float(
                        internal_bench.generated_descriptor_means.get("secret_content_discoverability_rate", 0.0)
                    ),
                    "mean_generation_time_sec": float(np.mean(gen_times)) if gen_times else 0.0,
                    "wall_time_sec": float(time.time() - t0),
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(summary_rows)
    raw_path = out_dir / "pcg_benchmark_alignment_raw.csv"
    summary_path = out_dir / "pcg_benchmark_alignment_summary.csv"
    json_path = out_dir / "pcg_benchmark_alignment_report.json"
    md_path = out_dir / "pcg_benchmark_alignment_report.md"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "methods": [m.name for m in methods],
        "problems": problems,
        "control_mode": str(args.control_mode),
        "settings": {
            "num_samples": int(args.num_samples),
            "population_size": int(args.population_size),
            "generations": int(args.generations),
            "room_budget_cap": int(args.room_budget_cap),
            "pcg_benchmark_repo": str(args.pcg_benchmark_repo) if args.pcg_benchmark_repo else None,
        },
        "summary": summary_df.to_dict(orient="records"),
        "payload_by_problem_method": benchmark_payloads,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _fmt(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except (TypeError, ValueError, AttributeError, ImportError):
            return df.to_string(index=False)

    external_report_columns = [
        "problem_name",
        "method",
        "n",
        "control_mode",
        "external_semantic_valid_rate",
        "external_quality_pass_rate",
        "external_quality_detail_mean",
        "external_diversity_pass_rate",
        "external_diversity_detail_mean",
        "external_controlability_pass_rate",
        "external_controlability_detail_mean",
        "external_solution_length_pass_rate",
        "external_mean_solution_length",
        "external_solution_length_target",
        "external_enemy_band_pass_rate",
        "external_mean_enemy_count",
        "external_enemy_target",
        "external_control_fallback_rate",
        "external_mean_initial_solution_length",
    ]
    mapper_report_columns = [
        "problem_name",
        "method",
        "external_mean_graph_player_key_aligned",
        "external_mean_graph_key_door_aligned",
        "external_mean_content_player_key_initial",
        "external_mean_content_key_door_initial",
        "external_mean_content_player_key",
        "external_mean_content_key_door",
        "external_mean_initial_abs_player_key_error",
        "external_mean_initial_abs_key_door_error",
        "external_mean_abs_player_key_error",
        "external_mean_abs_key_door_error",
    ]
    internal_report_columns = [
        "problem_name",
        "method",
        "min_rooms",
        "max_rooms",
        "internal_overall_completeness",
        "internal_constraint_valid_rate",
        "internal_key_before_lock_rate",
        "internal_switch_before_gate_rate",
        "internal_battery_satisfaction_rate",
        "internal_novelty_vs_reference",
        "internal_expressive_overlap_reference",
        "linearity",
        "progression_complexity",
        "topology_complexity",
        "path_redundancy",
        "articulation_ratio",
        "branch_utility_rate",
        "secret_content_discoverability_rate",
        "mean_generation_time_sec",
        "wall_time_sec",
    ]

    lines = [
        "# PCG Benchmark Zelda Alignment",
        "",
        "## Problems",
    ]
    lines.extend([f"- `{name}`" for name in problems])
    lines.extend(
        [
            "",
            "## Methods",
        ]
    )
    lines.extend([f"- `{method.name}`" for method in methods])
    lines.extend(
        [
            "",
            "## Settings",
            "",
            f"- `control_mode`: {args.control_mode}",
            f"- `num_samples`: {int(args.num_samples)}",
            f"- `population_size`: {int(args.population_size)}",
            f"- `generations`: {int(args.generations)}",
            "",
            "## External Benchmark",
            "",
            _fmt(summary_df[external_report_columns]),
            "",
            "## Mapper Diagnostics",
            "",
            _fmt(summary_df[mapper_report_columns]),
            "",
            "## Internal Topology",
            "",
            _fmt(summary_df[internal_report_columns]),
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved PCG benchmark alignment outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
