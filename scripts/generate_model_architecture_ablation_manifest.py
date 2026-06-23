"""Generate or execute model-architecture ablation commands.

This script closes the experiment-planning gap for:

- U-Net vs. DiT backbone comparison.
- DiT LayerNorm/GELU vs RMSNorm/GELU vs LayerNorm/SwiGLU vs RMSNorm/SwiGLU.

Default mode is plan-only. Use ``--execute`` only when checkpoints, data, and
compute budget are ready. The script writes a machine-readable manifest so
paper tables can report which ablations were planned versus actually run.
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

from src.config_system import merge_config  # noqa: E402


@dataclass(frozen=True)
class ArchitectureVariant:
    name: str
    denoiser_backbone: str
    dit_activation_type: str
    dit_norm_type: str
    comparison_family: str
    hypothesis: str


def build_variants() -> List[ArchitectureVariant]:
    return [
        ArchitectureVariant(
            name="unet_baseline",
            denoiser_backbone="unet",
            dit_activation_type="gelu",
            dit_norm_type="layer",
            comparison_family="backbone",
            hypothesis="Canonical U-Net denoiser baseline.",
        ),
        ArchitectureVariant(
            name="dit_layer_gelu",
            denoiser_backbone="dit",
            dit_activation_type="gelu",
            dit_norm_type="layer",
            comparison_family="backbone+dit_norm_activation",
            hypothesis="DiT baseline with LayerNorm and GELU.",
        ),
        ArchitectureVariant(
            name="dit_rms_gelu",
            denoiser_backbone="dit",
            dit_activation_type="gelu",
            dit_norm_type="rms",
            comparison_family="dit_norm_activation",
            hypothesis="RMSNorm may reduce normalization cost and improve stability.",
        ),
        ArchitectureVariant(
            name="dit_layer_swiglu",
            denoiser_backbone="dit",
            dit_activation_type="swiglu",
            dit_norm_type="layer",
            comparison_family="dit_norm_activation",
            hypothesis="SwiGLU may improve parameter efficiency versus GELU.",
        ),
        ArchitectureVariant(
            name="dit_rms_swiglu",
            denoiser_backbone="dit",
            dit_activation_type="swiglu",
            dit_norm_type="rms",
            comparison_family="dit_norm_activation",
            hypothesis="Combined RMSNorm/SwiGLU modern DiT ablation.",
        ),
    ]


def _parse_seeds(raw: str) -> List[int]:
    seeds = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    return seeds or [42]


def _command_for(
    *,
    python_exe: str,
    config: Path,
    output_dir: Path,
    variant: ArchitectureVariant,
    seed: int,
    epochs: int,
    extra_args: Sequence[str],
) -> List[str]:
    checkpoint_dir = output_dir / variant.name / f"seed_{seed}" / "checkpoints"
    command = [
        python_exe,
        "src/train_diffusion.py",
        "--config",
        str(config),
        "--denoiser-backbone",
        variant.denoiser_backbone,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--epochs",
        str(int(epochs)),
        "--seed",
        str(int(seed)),
    ]
    if variant.denoiser_backbone == "dit":
        command.extend(
            [
                "--dit-activation-type",
                variant.dit_activation_type,
                "--dit-norm-type",
                variant.dit_norm_type,
            ]
        )
    command.extend(extra_args)
    return command


def write_manifest(
    *,
    manifest_path: Path,
    csv_path: Path,
    payload: Dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    rows = payload["runs"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "seed",
                "comparison_family",
                "status",
                "elapsed_sec",
                "command",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row["variant"]["name"],
                    "seed": row["seed"],
                    "comparison_family": row["variant"]["comparison_family"],
                    "status": row["status"],
                    "elapsed_sec": row.get("elapsed_sec", 0.0),
                    "command": " ".join(row["command"]),
                }
            )


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    resolved = merge_config(yaml_path=str(config_path), cli_overrides=None)
    output_dir = Path(args.output_dir)
    seeds = _parse_seeds(args.seeds)
    variants = build_variants()
    runs: List[Dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            command = _command_for(
                python_exe=args.python,
                config=config_path,
                output_dir=output_dir,
                variant=variant,
                seed=seed,
                epochs=args.epochs,
                extra_args=list(args.extra_args or []),
            )
            runs.append(
                {
                    "variant": asdict(variant),
                    "seed": int(seed),
                    "command": command,
                    "status": "planned",
                    "elapsed_sec": 0.0,
                }
            )
    return {
        "script": Path(__file__).name,
        "mode": "execute" if args.execute else "plan",
        "config": str(config_path),
        "resolved_output_dir": str(resolved["runtime"]["output_dir"]),
        "output_dir": str(output_dir),
        "required_metrics": [
            "wall_clock_time_per_step",
            "loss_mean",
            "loss_variance",
            "parameter_count",
            "validation_tile_kl",
            "validation_graph_edit_distance",
            "teacher_fallback_used",
            "peak_memory_mb",
        ],
        "runs": runs,
    }


def execute_plan(payload: Dict[str, Any]) -> None:
    for run in payload["runs"]:
        start = time.perf_counter()
        completed = subprocess.run(run["command"], cwd=str(ROOT), check=False)
        run["elapsed_sec"] = float(time.perf_counter() - start)
        run["status"] = "passed" if completed.returncode == 0 else "failed"
        run["returncode"] = int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute architecture ablation runs.")
    parser.add_argument("--config", default="configs/zelda_hmolqd.yaml")
    parser.add_argument("--output-dir", default="results/model_architecture_ablations")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments appended to every train_diffusion.py command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_plan(args)
    if args.execute:
        execute_plan(payload)
    out_dir = Path(args.output_dir)
    write_manifest(
        manifest_path=out_dir / "model_architecture_ablation_manifest.json",
        csv_path=out_dir / "model_architecture_ablation_manifest.csv",
        payload=payload,
    )
    failed = [run for run in payload["runs"] if run["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
