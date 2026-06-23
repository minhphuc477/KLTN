"""Generate or execute Round-5 scientific-gap ablation commands.

This closes three publication-gap items that were not represented as explicit
experimental factors:

- additive topology conditioning vs SPADE topology conditioning
- diffusion teacher vs fast sampler latency-quality tradeoff
- weighted Bayesian WFC vs flat-prior WFC

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
