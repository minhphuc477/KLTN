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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
    required_inputs: Sequence[str] = ()
    output_paths: Sequence[str] = ()


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
        "solvability_rate",
        "tile_prior_kl",
        "graph_edit_distance",
        "generation_time_sec",
        "topology_preservation_score",
        "constraint_valid_rate",
        "model_parameter_count",
    )
    for seed in seeds:
        arm_checkpoints: Dict[str, Path] = {}
        for mode in ("additive", "spade"):
            checkpoint_dir = output_dir / "topology_conditioning" / mode / f"seed_{seed}" / "checkpoints"
            final_checkpoint = checkpoint_dir / "final_model.pth"
            arm_checkpoints[mode] = final_checkpoint
            experiments.append(
                GapExperiment(
                    name=f"train_topology_{mode}_seed_{seed}",
                    family="topology_conditioning_training",
                    hypothesis="Train one architecture-matched topology-conditioning arm.",
                    required_metrics=(),
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
                        *(
                            ["--vqvae-checkpoint", str(args.vqvae_checkpoint)]
                            if args.vqvae_checkpoint is not None
                            else []
                        ),
                        *list(args.extra_train_args or []),
                    ],
                    required_inputs=(str(args.config), str(args.vqvae_checkpoint or "")),
                    output_paths=(str(final_checkpoint),),
                )
            )
        eval_dir = output_dir / "topology_conditioning" / "paired" / f"seed_{seed}"
        experiments.append(
            GapExperiment(
                name=f"topology_spade_seed_{seed}",
                family="topology_conditioning",
                hypothesis=(
                    "SPADE-style affine topology modulation should improve topology preservation "
                    "relative to an architecture- and seed-matched additive arm."
                ),
                required_metrics=topology_metrics,
                command=[
                    args.python,
                    "scripts/run_ablation_study.py",
                    "--output",
                    str(eval_dir),
                    "--data-root",
                    str(args.data_root),
                    "--num-samples",
                    str(int(args.topology_eval_samples)),
                    "--seed",
                    str(seed),
                    "--configs",
                    "DIFFUSION_TOPO_ADDITIVE,DIFFUSION_TOPO_SPADE",
                    "--vqvae-checkpoint",
                    str(args.vqvae_checkpoint or ""),
                    "--diffusion-checkpoint",
                    str(arm_checkpoints["additive"]),
                    "--diffusion-additive-checkpoint",
                    str(arm_checkpoints["additive"]),
                    "--diffusion-spade-checkpoint",
                    str(arm_checkpoints["spade"]),
                ],
                required_inputs=(
                    str(args.data_root),
                    str(args.vqvae_checkpoint or ""),
                    str(arm_checkpoints["additive"]),
                    str(arm_checkpoints["spade"]),
                ),
                output_paths=(str(eval_dir / "ablation_summary.csv"),),
            )
        )

    fast_metrics = (
        "paired_n",
        "diffusion_over_fast_speedup",
        "fast_minus_diffusion_room_pairwise_ncd_mean",
        "fast_minus_diffusion_room_symbol_entropy_mean",
        "fast_minus_diffusion_astar_grid_solvable",
        "fast_minus_diffusion_cbs_success",
    )
    for variant, extra_args in (
        ("paired_diffusion_fast_sampler", []),
    ):
        for seed in seeds:
            command = _base_fast_sampler_args(args, seed, output_dir, variant)
            command.append("--paired-diffusion-fast-only")
            command.extend(extra_args)
            fast_output = output_dir / "fast_sampler_latency_quality" / variant / f"seed_{seed}"
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
                    required_inputs=(str(args.run_dir), str(args.lcm_checkpoint or "")),
                    output_paths=(str(fast_output / "paired_diffusion_fast_sampler.json"),),
                )
            )

    wfc_metrics = (
        "solvability_rate",
        "constraint_valid_rate",
        "room_repair_rate",
        "tiles_repaired",
        "symbolic_scaffold_tiles",
        "tile_prior_kl",
        "generation_time_sec",
        "diversity",
    )
    for seed in seeds:
        wfc_output = output_dir / "wfc_prior" / f"paired_seed_{seed}"
        experiments.append(
            GapExperiment(
                name=f"weighted_vs_flat_wfc_seed_{seed}",
                family="wfc_prior",
                hypothesis=(
                    "Weighted Bayesian WFC should preserve learned tile statistics better than "
                    "a flat-prior control under identical seeds and generation budgets. "
                    "Both arms are neural-free symbolic controls with the same deterministic graph-role scaffold."
                ),
                required_metrics=wfc_metrics,
                command=[
                    args.python,
                    "scripts/run_ablation_study.py",
                    "--output",
                    str(wfc_output),
                    "--data-root",
                    str(args.data_root),
                    "--num-samples",
                    str(int(args.wfc_samples_per_seed)),
                    "--seed",
                    str(seed),
                    "--configs",
                    "PURE_WFC,PURE_WFC_FLAT_PRIOR",
                    "--astar-timeout",
                    str(int(args.timeout_astar)),
                    "--cbs-timeout",
                    str(int(args.timeout_pcbs)),
                ],
                required_inputs=(str(args.data_root),),
                output_paths=(str(wfc_output / "ablation_summary.csv"),),
            )
        )

    generated_branch_metrics = (
        "hybrid_oracle_pass_rate",
        "cbs_success_rate",
        "repair_rate_mean",
        "tiles_repaired_mean",
        "generation_time_sec_mean",
    )
    for seed in seeds:
        experiments.append(
            GapExperiment(
                name=f"generated_branch_astar_pcbs_final_seed_{seed}",
                family="generated_branch_astar_pcbs",
                hypothesis=(
                    "A* and P-CBS must be reported side-by-side on the same final generated "
                    "artifacts for each room-generator branch. Raw/pre-repair effects belong "
                    "to the separate conditioning/repair ablation."
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
                    "--rule-space",
                    "spatial",
                ],
                required_inputs=(str(args.run_dir),),
                output_paths=(
                    str(
                        output_dir
                        / "generated_branch_astar_pcbs"
                        / f"seed_{seed}"
                        / "full_pipeline_aggregate.csv"
                    ),
                ),
            )
        )

    controllability_metrics = (
        "macro_norm_error",
        "pass_all_rate",
        "actual_num_nodes_mean",
        "pass_rate_num_nodes",
        "generation_time_sec_mean",
    )
    for seed in seeds:
        stress_output = output_dir / "designer_controllability" / f"stress_100_250_500_seed_{seed}"
        experiments.append(
            GapExperiment(
                name=f"designer_controllability_100_500_stress_seed_{seed}",
                family="designer_controllability",
                hypothesis=(
                    "Designer controllability claims require explicit target tracking at 100, 250, "
                    "and 500 rooms rather than extrapolation from ordinary-size averages."
                ),
                required_metrics=controllability_metrics,
                command=[
                    args.python,
                    "scripts/run_designer_controllability_proof.py",
                    "--execute",
                    "--output",
                    str(stress_output),
                    "--data-root",
                    str(args.data_root),
                    "--seed",
                    str(seed),
                    "--samples-per-target",
                    str(int(args.controllability_samples_per_target)),
                    "--population-size",
                    str(int(args.controllability_population_size)),
                    "--generations",
                    str(int(args.controllability_generations)),
                    "--target-names",
                    "p_large_stress_100,p_large_stress_250,p_large_stress_500",
                    "--write-graphs",
                ],
                required_inputs=(str(args.data_root),),
                output_paths=(str(stress_output / "designer_controllability_summary.csv"),),
            )
        )

    pcbs_component_metrics = (
        "experiment_valid",
        "success_rate",
        "success_rate_given_oracle_solved",
        "outcome_class_counts",
        "avg_time_ms",
        "matched_budget_contract",
    )
    for persona in [p.strip() for p in str(args.pcbs_personas).split(",") if p.strip()]:
        for seed in seeds:
            pcbs_output = output_dir / "pcbs_component_ablation" / persona / f"seed_{seed}"
            experiments.append(
                GapExperiment(
                    name=f"pcbs_component_matched_budget_{persona}_seed_{seed}",
                    family="pcbs_component_ablation",
                    hypothesis=(
                        "P-CBS component claims require identical state budgets, maps, and seeds "
                        "for full and ablated variants within each persona."
                    ),
                    required_metrics=pcbs_component_metrics,
                    command=[
                        args.python,
                        "scripts/run_pcbs_component_ablation.py",
                        "--levels",
                        str(args.pcbs_levels),
                        "--data-root",
                        str(args.data_root),
                        "--variants",
                        str(args.pcbs_variants),
                        "--persona",
                        persona,
                        "--timeout-astar",
                        str(int(args.timeout_astar)),
                        "--timeout-pcbs",
                        str(int(args.timeout_pcbs)),
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(pcbs_output),
                        "--quiet",
                    ],
                    required_inputs=(str(args.data_root),),
                    output_paths=(str(pcbs_output / "summary.json"),),
                )
            )

    paired_significance_metrics = (
        "n_pairs",
        "delta_mean_cfg_minus_full",
        "delta_ci_low",
        "delta_ci_high",
        "p_value",
        "effect_size_d",
        "p_value_bh_fdr",
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
            output_paths=(
                str(output_dir / "matched_budget_significance" / "matched_budget_significance.csv"),
            ),
        )
    )

    target_response_metrics = (
        "target_family",
        "metric",
        "target_mean",
        "actual_mean",
        "mean_norm_error",
        "pass_rate",
        "monotonic_non_decreasing_from_previous",
    )
    for seed in seeds:
        target_response_output = output_dir / "graph_target_response" / f"seed_{seed}"
        target_response_command = [
            args.python,
            "scripts/run_designer_controllability_proof.py",
            "--execute",
            "--output",
            str(target_response_output),
            "--data-root",
            str(args.data_root),
            "--seed",
            str(seed),
            "--samples-per-target",
            str(int(args.controllability_samples_per_target)),
            "--population-size",
            str(int(args.controllability_population_size)),
            "--generations",
            str(int(args.controllability_generations)),
            "--target-names",
            (
                "axis_linearity_0p36,axis_linearity_0p52,axis_linearity_0p68,"
                "axis_keylock_k5_l2,axis_keylock_k3_l3,axis_keylock_k2_l6,"
                "axis_size_12,axis_size_24,axis_size_36"
            ),
        ]
        experiments.append(
            GapExperiment(
                name=f"graph_target_response_seed_{seed}",
                family="graph_target_response",
                hypothesis=(
                    "Changing one graph target at a time should produce a monotonic response in "
                    "the generated mission-graph descriptor under matched search budgets."
                ),
                required_metrics=target_response_metrics,
                command=target_response_command,
                required_inputs=(str(args.data_root),),
                output_paths=(
                    str(target_response_output / "designer_target_response_monotonicity.csv"),
                ),
            )
        )

    selected = {token.strip() for token in str(args.families).split(",") if token.strip()}
    if not selected or "all" in selected:
        return experiments
    if "topology_conditioning" in selected:
        selected.add("topology_conditioning_training")
    available = {experiment.family for experiment in experiments}
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(f"Unknown experiment families: {unknown}. Available: {sorted(available)}")
    return [experiment for experiment in experiments if experiment.family in selected]


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    experiments = build_experiments(args)
    payload = {
        "script": Path(__file__).name,
        "mode": "execute" if args.execute else "plan",
        "config": str(args.config),
        "output_dir": str(args.output_dir),
        "scientific_boundary": (
            "These are ablations, not claims. Treat a row as evidence only when status is passed "
            "and its output metrics file is archived with the same seed/config."
        ),
        "research_basis": [
            {
                "topic": "SPADE conditioning",
                "source": "Semantic Image Synthesis with Spatially-Adaptive Normalization",
                "url": "https://arxiv.org/abs/1903.07291",
                "protocol_implication": "Compare learned spatial affine modulation against an additive conditioning control.",
            },
            {
                "topic": "few-step latent consistency",
                "source": "Latent Consistency Models",
                "url": "https://arxiv.org/abs/2310.04378",
                "protocol_implication": (
                    "Use paired 50-step versus 4-step quality/latency tests, while labeling the "
                    "repository artifact as consistency_lora rather than paper-faithful LCM-LoRA."
                ),
            },
            {
                "topic": "controllable graph generation",
                "source": "G-PCGRL: Procedural Graph Data Generation via Reinforcement Learning",
                "url": "https://arxiv.org/abs/2407.10483",
                "protocol_implication": "Report target node-count error and pass rates, not only generation success.",
            },
            {
                "topic": "out-of-distribution scale",
                "source": "PCGRL+: Scaling, Control and Generalization in Reinforcement Learning Level Generators",
                "url": "https://arxiv.org/abs/2408.12525",
                "protocol_implication": "Evaluate explicit 100, 250, and 500 room targets as separate stress rows.",
            },
            {
                "topic": "WFC control",
                "source": "WaveFunctionCollapse is Constraint Solving in the Wild",
                "url": "https://doi.org/10.1145/3102071.3110566",
                "protocol_implication": "Hold constraints and seeds fixed while ablating learned pattern priors.",
            },
        ],
        "runs": [
            {
                "name": exp.name,
                "family": exp.family,
                "hypothesis": exp.hypothesis,
                "required_metrics": list(exp.required_metrics),
                "command": list(exp.command),
                "required_inputs": list(exp.required_inputs),
                "output_paths": list(exp.output_paths),
                "status": "planned",
                "elapsed_sec": 0.0,
            }
            for exp in experiments
        ],
    }
    produced_outputs: set[Path] = set()
    for run in payload["runs"]:
        missing_inputs: List[str] = []
        dependency_inputs: List[str] = []
        for raw_path in run.get("required_inputs", []):
            if not raw_path:
                missing_inputs.append(str(raw_path))
                continue
            resolved = _resolve_artifact_path(raw_path).resolve()
            if resolved.exists():
                continue
            if resolved in produced_outputs:
                dependency_inputs.append(str(raw_path))
            else:
                missing_inputs.append(str(raw_path))
        if missing_inputs:
            run["status"] = "blocked_missing_inputs"
            run["missing_inputs"] = missing_inputs
        elif dependency_inputs:
            run["status"] = "waiting_for_manifest_dependencies"
            run["dependency_inputs"] = dependency_inputs
        else:
            run["status"] = "ready_to_execute"
        produced_outputs.update(
            _resolve_artifact_path(path).resolve()
            for path in run.get("output_paths", [])
            if path
        )
    return payload


def write_manifest(payload: Dict[str, Any], manifest_path: Path, csv_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "family",
                "status",
                "elapsed_sec",
                "command",
                "required_inputs",
                "output_paths",
                "required_metrics",
            ],
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
                    "required_inputs": ",".join(run.get("required_inputs", [])),
                    "output_paths": ",".join(run.get("output_paths", [])),
                    "required_metrics": ",".join(run["required_metrics"]),
                }
            )


def _resolve_artifact_path(raw_path: str) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else ROOT / path


def _collect_metric_names(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return set(next(csv.reader(handle), []))
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        names: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    names.add(str(key))
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
        return names
    return set()


def execute_manifest(payload: Dict[str, Any]) -> None:
    for run in payload["runs"]:
        command = list(run["command"])
        if "--plan-only" in command:
            run["status"] = "planned_only"
            continue
        missing_inputs = [
            raw_path
            for raw_path in run.get("required_inputs", [])
            if not raw_path or not _resolve_artifact_path(raw_path).exists()
        ]
        if missing_inputs:
            run["status"] = "blocked_missing_inputs"
            run["missing_inputs"] = missing_inputs
            continue
        if run.get("family") == "fast_sampler_latency_quality":
            try:
                from src.optimization.lcm_lora import load_fast_sampler_checkpoint

                checkpoint_arg = command[command.index("--lcm-checkpoint") + 1]
                _, checkpoint_info = load_fast_sampler_checkpoint(checkpoint_arg)
                run["fast_sampler_distillation_type"] = checkpoint_info.distillation_type
            except (ImportError, ValueError, OSError, IndexError) as exc:
                run["status"] = "blocked_invalid_fast_sampler_checkpoint"
                run["input_error"] = f"{type(exc).__name__}: {exc}"
                continue
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        run["elapsed_sec"] = float(time.perf_counter() - start)
        run["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            run["status"] = "failed_process"
            continue
        output_paths = [_resolve_artifact_path(path) for path in run.get("output_paths", [])]
        missing_outputs = [str(path) for path in output_paths if not path.exists()]
        if missing_outputs:
            run["status"] = "failed_missing_outputs"
            run["missing_outputs"] = missing_outputs
            continue
        if not output_paths:
            run["status"] = "completed_needs_metric_artifact"
            continue
        metric_names: set[str] = set()
        for output_path in output_paths:
            metric_names.update(_collect_metric_names(output_path))
        missing_metrics = sorted(set(run.get("required_metrics", [])) - metric_names)
        if missing_metrics:
            run["status"] = "failed_missing_metrics"
            run["missing_metrics"] = missing_metrics
            continue
        run["status"] = "passed"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Round-5 scientific-gap ablation manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/zelda_hmolqd.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("Data/The Legend of Zelda"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/round5_scientific_gaps"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument(
        "--families",
        type=str,
        default=(
            "topology_conditioning,fast_sampler_latency_quality,"
            "pcbs_component_ablation,wfc_prior,designer_controllability"
        ),
        help="Comma-separated experiment families to plan or execute; use 'all' for legacy extras.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--topology-eval-samples", type=int, default=8)
    parser.add_argument("--wfc-samples-per-seed", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/zelda_hmolqd_fulltrain_rerun"))
    parser.add_argument("--vqvae-checkpoint", type=Path, default=None)
    parser.add_argument("--diffusion-checkpoint", type=Path, default=None)
    parser.add_argument("--lcm-checkpoint", type=Path, default=None)
    parser.add_argument("--masked-room-checkpoint", type=Path, default=None)
    parser.add_argument("--mission-graph-json", type=Path, default=None)
    parser.add_argument("--generated-branch-variants", type=str, default="diffusion,fast,masked")
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
    selected_families = {token.strip() for token in str(args.families).split(",") if token.strip()}
    if args.execute:
        if ("all" in selected_families or "topology_conditioning" in selected_families) and args.vqvae_checkpoint is None:
            raise ValueError(
                "Executing topology conditioning requires --vqvae-checkpoint so diffusion ablations "
                "cannot silently train against a random tokenizer."
            )
        if args.vqvae_checkpoint is not None and not args.vqvae_checkpoint.exists():
            raise FileNotFoundError(
                f"VQ-VAE checkpoint does not exist: {args.vqvae_checkpoint}"
            )
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
