"""Generate or execute model-architecture ablation commands.

This script closes the experiment-planning gap for:

- U-Net vs. DiT backbone comparison.
- DiT LayerNorm/GELU vs RMSNorm/GELU vs LayerNorm/SwiGLU vs RMSNorm/SwiGLU.
- Softmax vs. Hedgehog linear attention.
- GATv2 vs. directed semantic sparse attention vs. learned Graphormer bias.
- SPADE vs. additive topology conditioning.

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
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_system import merge_config  # noqa: E402
from src.utils.checkpoint import safe_torch_load  # noqa: E402


@dataclass(frozen=True)
class ArchitectureVariant:
    name: str
    denoiser_backbone: str
    dit_activation_type: str
    dit_norm_type: str
    attention_mode: str
    topology_refinement_mode: str
    topology_conditioning_mode: str
    comparison_family: str
    hypothesis: str


def build_variants() -> List[ArchitectureVariant]:
    return [
        ArchitectureVariant(
            name="unet_baseline",
            denoiser_backbone="unet",
            dit_activation_type="gelu",
            dit_norm_type="layer",
            attention_mode="softmax",
            topology_refinement_mode="gat2",
            topology_conditioning_mode="spade",
            comparison_family="backbone",
            hypothesis="Canonical U-Net denoiser baseline.",
        ),
        ArchitectureVariant(
            name="dit_layer_gelu",
            denoiser_backbone="dit",
            dit_activation_type="gelu",
            dit_norm_type="layer",
            attention_mode="softmax",
            topology_refinement_mode="gat2",
            topology_conditioning_mode="spade",
            comparison_family="backbone+dit_norm_activation",
            hypothesis="DiT baseline with LayerNorm and GELU.",
        ),
        ArchitectureVariant(
            name="dit_rms_gelu",
            denoiser_backbone="dit",
            dit_activation_type="gelu",
            dit_norm_type="rms",
            attention_mode="softmax",
            topology_refinement_mode="gat2",
            topology_conditioning_mode="spade",
            comparison_family="dit_norm_activation",
            hypothesis="RMSNorm may reduce normalization cost and improve stability.",
        ),
        ArchitectureVariant(
            name="dit_layer_swiglu",
            denoiser_backbone="dit",
            dit_activation_type="swiglu",
            dit_norm_type="layer",
            attention_mode="softmax",
            topology_refinement_mode="gat2",
            topology_conditioning_mode="spade",
            comparison_family="dit_norm_activation",
            hypothesis="SwiGLU may improve parameter efficiency versus GELU.",
        ),
        ArchitectureVariant(
            name="dit_rms_swiglu",
            denoiser_backbone="dit",
            dit_activation_type="swiglu",
            dit_norm_type="rms",
            attention_mode="softmax",
            topology_refinement_mode="gat2",
            topology_conditioning_mode="spade",
            comparison_family="dit_norm_activation",
            hypothesis="Combined RMSNorm/SwiGLU modern DiT ablation.",
        ),
        ArchitectureVariant(
            name="unet_linear_hedgehog",
            denoiser_backbone="unet",
            dit_activation_type="gelu",
            dit_norm_type="layer",
            attention_mode="linear_hedgehog",
            topology_refinement_mode="gat2",
            topology_conditioning_mode="spade",
            comparison_family="attention_kernel",
            hypothesis="Linear attention trades exact softmax interactions for graph-scaling efficiency.",
        ),
        ArchitectureVariant(
            name="unet_sparse_directed_semantic",
            denoiser_backbone="unet",
            dit_activation_type="gelu",
            dit_norm_type="layer",
            attention_mode="softmax",
            topology_refinement_mode="sparse_directed_semantic",
            topology_conditioning_mode="spade",
            comparison_family="topology_refinement",
            hypothesis="Directed edge-semantic sparse attention preserves topology without all-pairs preprocessing.",
        ),
        ArchitectureVariant(
            name="unet_graphormer_learned_directed_semantic",
            denoiser_backbone="unet",
            dit_activation_type="gelu",
            dit_norm_type="layer",
            attention_mode="softmax",
            topology_refinement_mode="graphormer_learned_directed_semantic",
            topology_conditioning_mode="spade",
            comparison_family="topology_refinement",
            hypothesis="Learned distance and edge bias may improve structural fidelity on small graphs.",
        ),
        ArchitectureVariant(
            name="unet_additive_topology",
            denoiser_backbone="unet",
            dit_activation_type="gelu",
            dit_norm_type="layer",
            attention_mode="softmax",
            topology_refinement_mode="gat2",
            topology_conditioning_mode="additive",
            comparison_family="topology_conditioning",
            hypothesis="Additive topology injection is the parameter-matched baseline for SPADE conditioning.",
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
    vqvae_checkpoint: Path,
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
        "--vqvae-checkpoint",
        str(vqvae_checkpoint),
        "--attention-mode",
        variant.attention_mode,
        "--topology-refinement-mode",
        variant.topology_refinement_mode,
        "--topology-conditioning-mode",
        variant.topology_conditioning_mode,
        "--no-auto-resume",
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


def _resolve_artifact_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else ROOT / path


def _collect_metric_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return set(next(csv.reader(handle), []))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".pt", ".pth"}:
        payload = safe_torch_load(path, map_location="cpu")
    else:
        return set()
    if isinstance(payload, (list, Mapping)):
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


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    resolved = merge_config(yaml_path=str(config_path), cli_overrides=None)
    output_dir = Path(args.output_dir)
    seeds = _parse_seeds(args.seeds)
    variants = build_variants()
    configured_vqvae = (
        getattr(args, "vqvae_checkpoint", None)
        or resolved.get("diffusion", {}).get("vqvae_checkpoint")
    )
    vqvae_checkpoint = Path(configured_vqvae or "REQUIRED_VQVAE_CHECKPOINT")
    if args.execute and not configured_vqvae:
        raise ValueError(
            "--execute requires --vqvae-checkpoint or "
            "diffusion.vqvae_checkpoint in the base config."
        )
    if args.execute and not vqvae_checkpoint.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint does not exist: {vqvae_checkpoint}")
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
                vqvae_checkpoint=vqvae_checkpoint,
                extra_args=list(args.extra_args or []),
            )
            runs.append(
                {
                    "variant": asdict(variant),
                    "seed": int(seed),
                    "command": command,
                    "status": "planned",
                    "elapsed_sec": 0.0,
                    "output_paths": [
                        str(output_dir / variant.name / f"seed_{seed}" / "checkpoints" / "final_model.pth"),
                    ],
                }
            )
    return {
        "script": Path(__file__).name,
        "mode": "execute" if args.execute else "plan",
        "config": str(config_path),
        "vqvae_checkpoint": str(vqvae_checkpoint),
        "resolved_output_dir": str(resolved["runtime"]["output_dir"]),
        "output_dir": str(output_dir),
        "required_metrics": [
            "epoch",
            "loss",
            "diffusion_loss",
            "val_diffusion_loss",
            "val_total_loss",
            "lr",
        ],
        "evaluation_metrics_required_for_paper": [
            "wall_clock_time_per_step",
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
        run["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            run["status"] = "failed_process"
            continue
        output_paths = [_resolve_artifact_path(path) for path in run.get("output_paths", [])]
        if not output_paths:
            run["status"] = "completed_needs_metric_artifact"
            continue
        missing_outputs = [str(path) for path in output_paths if not path.exists()]
        if missing_outputs:
            run["status"] = "failed_missing_outputs"
            run["missing_outputs"] = missing_outputs
            continue
        metric_names: set[str] = set()
        for output_path in output_paths:
            metric_names.update(_collect_metric_names(output_path))
        missing_metrics = sorted(set(payload.get("required_metrics", [])) - metric_names)
        if missing_metrics:
            run["status"] = "failed_missing_metrics"
            run["missing_metrics"] = missing_metrics
            continue
        run["status"] = "passed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute architecture ablation runs.")
    parser.add_argument("--config", default="configs/zelda_hmolqd.yaml")
    parser.add_argument("--output-dir", default="results/model_architecture_ablations")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--vqvae-checkpoint", type=Path, default=None)
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
    failed = [run for run in payload["runs"] if str(run["status"]).startswith("failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
