"""Generate or execute Round-5 scientific-gap ablation commands.

This closes publication-gap items that were not represented as explicit
experimental factors:

- additive topology conditioning vs SPADE topology conditioning
- diffusion teacher vs fast sampler latency-quality tradeoff
- weighted Bayesian WFC vs flat-prior WFC
- A* vs P-CBS validation on final generated branches, with raw/pre-repair and
  post-repair metrics kept separate by the downstream pipeline export
- 100-room and 500-room designer-controllability stress rows
- matched-budget P-CBS component ablation
- paired-seed statistical significance
- target-response checks that graph semantic changes affect room semantics

Default mode is plan-only. Use ``--execute`` only when checkpoints, data, and
compute budget are ready. The output manifest is intentionally machine-readable
so reported paper tables can distinguish planned experiments from completed
runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class GapExperiment:
    name: str
    family: str
    hypothesis: str
    required_metrics: Sequence[str]
    command: Sequence[str]


def _parse_seeds(raw: str) -> List[int]:
    seeds = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    return seeds or [42]


def _base_fast_sampler_args(args: argparse.Namespace, seed: int, output_dir: Path, variant: str) -> List[str]:
    command = [
        args.python,
        "scripts/run_fixed_graph_multi_seed_audit.py",
        "--run-dir",
        str(args.run_dir),
        "--output-dir",
        str(output_dir / "fast_sampler_latency_quality" / variant / f"seed_{seed}"),
        "--seeds",
        str(seed),
    ]
    if args.mission_graph_json:
        command.extend(["--mission-graph", str(args.mission_graph_json)])
    if args.lcm_checkpoint:
        command.extend(["--lcm-checkpoint", str(args.lcm_checkpoint)])
    return command


def build_experiments(args: argparse.Namespace) -> List[GapExperiment]:
    output_dir = Path(args.output_dir)
    seeds = _parse_seeds(args.seeds)
    experiments: List[GapExperiment] = []

    topology_metrics = (
        "validation_tile_kl",
        "validation_graph_edit_distance",
        "logicnet_dungeon_solvability",
        "teacher_fallback_used",
        "peak_memory_mb",
        "wall_clock_time_per_step",
    )
    for mode in ("additive", "spade"):
        for seed in seeds:
            checkpoint_dir = output_dir / "topology_conditioning" / mode / f"seed_{seed}" / "checkpoints"
            experiments.append(
                GapExperiment(
                    name=f"topology_{mode}_seed_{seed}",
                    family="topology_conditioning",
                    hypothesis=(
                        "SPADE-style affine topology modulation should outperform or match additive maps "
                        "on structural validity if topology conditioning is carrying semantic signal."
                        if mode == "spade"
                        else "Additive topology maps are the baseline conditioning path."
                    ),
                    required_metrics=topology_metrics,
                    command=[
                        args.python,
                        "src/train_diffusion.py",
                        "--config",
                        str(args.config),
                        "--topology-conditioning-mode",
                        mode,
                        "--checkpoint-dir",
                        str(checkpoint_dir),
                        "--epochs",
                        str(int(args.epochs)),
                        "--seed",
                        str(seed),
                        *list(args.extra_train_args or []),
                    ],
                )
            )

    fast_metrics = (
        "generation_time_sec",
        "room_time_sec_mean",
        "post_oracle_solved_rate",
        "post_pcbs_valid_rate",
        "teacher_fallback_used",
        "ncd_diversity",
        "tile_entropy",
    )
    for variant, extra_args in (
        ("paired_diffusion_fast_sampler", []),
    ):
        for seed in seeds:
            command = _base_fast_sampler_args(args, seed, output_dir, variant)
            command.extend(extra_args)
            experiments.append(
                GapExperiment(
                    name=f"fixed_graph_diffusion_vs_fast_sampler_seed_{seed}",
                    family="fast_sampler_latency_quality",
                    hypothesis=(
                        "The fixed-graph audit exports full diffusion and fast-sampler variants under the same "
                        "seed/topology, enabling paired latency-quality comparison."
                    ),
                    required_metrics=fast_metrics,
                    command=command,
                )
            )

    wfc_metrics = (
        "oracle_solved_rate",
        "pcbs_valid_rate",
        "repair_count_mean",
        "total_tiles_repaired_mean",
        "tile_distribution_kl",
        "contradiction_rate",
        "generation_time_sec",
    )
    for variant, config_name in (
        ("weighted_bayesian_wfc", "PURE_WFC"),
        ("flat_prior_wfc", "PURE_WFC_FLAT_PRIOR"),
    ):
        for seed in seeds:
            experiments.append(
                GapExperiment(
                    name=f"{variant}_seed_{seed}",
                    family="wfc_prior",
                    hypothesis=(
                        "Weighted Bayesian WFC should preserve learned tile statistics better than flat priors "
                        "without increasing contradiction rate."
                        if variant == "weighted_bayesian_wfc"
                        else "Flat-prior WFC is the symbolic control for checking whether learned priors matter."
                    ),
                    required_metrics=wfc_metrics,
                    command=[
                        args.python,
                        "scripts/run_ablation_study.py",
                        "--output",
                        str(output_dir / "wfc_prior" / variant / f"seed_{seed}"),
                        "--num-samples",
                        "1",
                        "--seed",
                        str(seed),
                        "--configs",
                        config_name,
                        "--quick",
                    ],
                )
            )

    generated_branch_metrics = (
        "raw_oracle_solved_rate",
        "post_oracle_solved_rate",
        "raw_pcbs_valid_rate",
        "post_pcbs_valid_rate",
        "tiles_repaired_mean",
        "teacher_fallback_used",
        "paired_logicnet_delta",
    )
    for seed in seeds:
        experiments.append(
            GapExperiment(
                name=f"generated_branch_astar_pcbs_prepost_seed_{seed}",
                family="generated_branch_astar_pcbs",
                hypothesis=(
                    "A* and P-CBS must be reported side-by-side on the same generated branches, "
                    "with raw/pre-repair validity separated from post-repair validity."
                ),
                required_metrics=generated_branch_metrics,
                command=[
                    args.python,
                    "scripts/run_generated_graph_full_pipeline_eval.py",
                    "--run-dir",
                    str(args.run_dir),
                    "--output-dir",
                    str(output_dir / "generated_branch_astar_pcbs" / f"seed_{seed}"),
                    "--seeds",
                    str(seed),
                    "--variants",
                    str(args.generated_branch_variants),
                    "--min-rooms",
                    str(int(args.generated_branch_min_rooms)),
                    "--max-rooms",
                    str(int(args.generated_branch_max_rooms)),
                    "--regime-name",
                    "round7_generated_branch_prepost",
                    "--room-budget-cap",
                    str(int(args.generated_branch_room_budget_cap)),
                ],
            )
        )

    controllability_metrics = (
        "target_match_score",
        "target_response_slope",
        "room_count_error",
        "linearity_error",
        "gate_pressure_error",
        "stress_100_success_rate",
        "stress_500_success_rate",
        "wall_clock_sec",
    )
    experiments.append(
        GapExperiment(
            name="designer_controllability_100_500_stress",
            family="designer_controllability",
            hypothesis=(
                "Designer controllability claims require explicit target-response rows and "
                "100-room/500-room stress targets, not only ordinary-size averages."
            ),
            required_metrics=controllability_metrics,
            command=[
                args.python,
                "scripts/run_designer_controllability_proof.py",
                "--execute",
                "--output",
                str(output_dir / "designer_controllability" / "stress_100_500"),
                "--seed",
                str(seeds[0]),
                "--samples-per-target",
                str(int(args.controllability_samples_per_target)),
                "--population-size",
                str(int(args.controllability_population_size)),
                "--generations",
                str(int(args.controllability_generations)),
                "--write-graphs",
            ],
        )
    )

    pcbs_component_metrics = (
        "pcbs_success_rate",
        "astar_success_rate",
        "pcbs_failure_driver",
        "pcbs_outcome_class",
        "component_delta_vs_full",
        "matched_budget_runtime_sec",
    )
    for persona in [p.strip() for p in str(args.pcbs_personas).split(",") if p.strip()]:
        experiments.append(
            GapExperiment(
                name=f"pcbs_component_matched_budget_{persona}",
                family="pcbs_component_ablation",
                hypothesis=(
                    "P-CBS component claims require matched-budget ablations by persona rather than "
                    "aggregate full-model means."
                ),
                required_metrics=pcbs_component_metrics,
                command=[
                    args.python,
                    "scripts/run_pcbs_component_ablation.py",
                    "--levels",
                    str(args.pcbs_levels),
                    "--variants",
                    str(args.pcbs_variants),
                    "--persona",
                    persona,
                    "--timeout-astar",
                    str(int(args.timeout_astar)),
                    "--timeout-pcbs",
                    str(int(args.timeout_pcbs)),
                    "--seed",
                    str(seeds[0]),
                    "--output-dir",
                    str(output_dir / "pcbs_component_ablation" / persona),
                    "--quiet",
                ],
            )
        )

    paired_significance_metrics = (
        "paired_n",
        "mean_delta",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "permutation_p_value",
        "effect_size",
    )
    experiments.append(
        GapExperiment(
            name="matched_budget_paired_seed_significance",
            family="statistical_significance",
            hypothesis=(
                "Architecture and topology comparisons need paired-seed deltas, confidence intervals, "
                "and permutation p-values rather than unpaired aggregate means."
            ),
            required_metrics=paired_significance_metrics,
            command=[
                args.python,
                "scripts/run_matched_budget_topology_benchmark.py",
                "--output",
                str(output_dir / "matched_budget_significance"),
                "--seed",
                str(seeds[0]),
                "--num-samples",
                str(int(args.significance_num_samples)),
                "--eval-budget",
                str(int(args.significance_eval_budget)),
                "--min-rooms",
                str(int(args.significance_min_rooms)),
                "--max-rooms",
                str(int(args.significance_max_rooms)),
                "--methods",
                str(args.significance_methods),
            ],
        )
    )

    target_response_metrics = (
        "semantic_target_delta",
        "pre_repair_anchor_error_delta",
        "pre_repair_role_match_delta",
        "post_repair_role_match_delta",
        "target_response_monotonicity",
    )
    for seed in seeds:
        target_response_command = [
            args.python,
            "scripts/run_conditioning_logicnet_repair_ablation.py",
            "--execute",
            "--config",
            str(args.config),
            "--output",
            str(output_dir / "target_response_semantics" / f"seed_{seed}"),
            "--seeds",
            str(seed),
            "--num-rooms",
            str(int(args.target_response_num_rooms)),
            "--timeout-astar",
            str(int(args.timeout_astar)),
            "--timeout-pcbs",
            str(int(args.timeout_pcbs)),
        ]
        if args.vqvae_checkpoint:
            target_response_command.extend(["--vqvae-checkpoint", str(args.vqvae_checkpoint)])
        if args.diffusion_checkpoint:
            target_response_command.extend(["--diffusion-checkpoint", str(args.diffusion_checkpoint)])
        experiments.append(
            GapExperiment(
                name=f"target_response_semantic_pre_repair_seed_{seed}",
                family="target_response_semantics",
                hypothesis=(
                    "Changing graph semantics must measurably change generated room semantics before "
                    "symbolic repair; otherwise the repair layer, not the model, may be carrying the claim."
                ),
                required_metrics=target_response_metrics,
                command=target_response_command,
            )
        )

    return experiments


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    experiments = build_experiments(args)
    return {
        "script": Path(__file__).name,
        "mode": "execute" if args.execute else "plan",
        "config": str(args.config),
        "output_dir": str(args.output_dir),
        "scientific_boundary": (
            "These are ablations, not claims. Treat a row as evidence only when status is passed "
            "and its output metrics file is archived with the same seed/config."
        ),
        "runs": [
            {
                "name": exp.name,
                "family": exp.family,
                "hypothesis": exp.hypothesis,
                "required_metrics": list(exp.required_metrics),
                "command": list(exp.command),
                "status": "planned",
                "elapsed_sec": 0.0,
            }
            for exp in experiments
        ],
    }


def write_manifest(payload: Dict[str, Any], manifest_path: Path, csv_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "family", "status", "elapsed_sec", "command", "required_metrics"],
        )
        writer.writeheader()
        for run in payload["runs"]:
            writer.writerow(
                {
                    "name": run["name"],
                    "family": run["family"],
                    "status": run["status"],
                    "elapsed_sec": run.get("elapsed_sec", 0.0),
                    "command": " ".join(str(part) for part in run["command"]),
                    "required_metrics": ",".join(run["required_metrics"]),
                }
            )


def execute_manifest(payload: Dict[str, Any]) -> None:
    for run in payload["runs"]:
        command = list(run["command"])
        if "--plan-only" in command:
            run["status"] = "planned_only"
            continue
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        run["elapsed_sec"] = float(time.perf_counter() - start)
        run["status"] = "passed" if completed.returncode == 0 else "failed"
        run["returncode"] = int(completed.returncode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Round-5 scientific-gap ablation manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/zelda_hmolqd.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/round5_scientific_gaps"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/zelda_hmolqd_fulltrain_rerun"))
    parser.add_argument("--vqvae-checkpoint", type=Path, default=None)
    parser.add_argument("--diffusion-checkpoint", type=Path, default=None)
    parser.add_argument("--lcm-checkpoint", type=Path, default=None)
    parser.add_argument("--masked-room-checkpoint", type=Path, default=None)
    parser.add_argument("--mission-graph-json", type=Path, default=None)
    parser.add_argument("--generated-branch-variants", type=str, default="diffusion,fast_sampler,masked_room")
    parser.add_argument("--generated-branch-min-rooms", type=int, default=18)
    parser.add_argument("--generated-branch-max-rooms", type=int, default=32)
    parser.add_argument("--generated-branch-room-budget-cap", type=int, default=64)
    parser.add_argument("--controllability-samples-per-target", type=int, default=5)
    parser.add_argument("--controllability-population-size", type=int, default=64)
    parser.add_argument("--controllability-generations", type=int, default=20)
    parser.add_argument("--pcbs-levels", type=str, default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--pcbs-variants", type=str, default="1,2")
    parser.add_argument("--pcbs-personas", type=str, default="novice,balanced,expert")
    parser.add_argument("--timeout-astar", type=int, default=200000)
    parser.add_argument("--timeout-pcbs", type=int, default=50000)
    parser.add_argument("--significance-num-samples", type=int, default=30)
    parser.add_argument("--significance-eval-budget", type=int, default=200)
    parser.add_argument("--significance-min-rooms", type=int, default=18)
    parser.add_argument("--significance-max-rooms", type=int, default=32)
    parser.add_argument("--significance-methods", type=str, default="VGLC_REF,FULL_GA,FULL_MAP_ELITES")
    parser.add_argument("--target-response-num-rooms", type=int, default=24)
    parser.add_argument("--extra-train-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest or (Path(args.output_dir) / "round5_scientific_gap_manifest.json")
    csv_path = args.csv or manifest_path.with_suffix(".csv")
    payload = build_manifest(args)
    if args.execute:
        execute_manifest(payload)
    write_manifest(payload, manifest_path, csv_path)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
