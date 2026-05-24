"""
Static training hyperparameter and batch preflight checks.

This script validates the resolved training configuration beyond hard schema
checks. It is intentionally lightweight: by default it only inspects config
contracts, and with --probe-data it also counts the configured Zelda training
corpus to report effective train/validation batch counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_system import merge_config


@dataclass(frozen=True)
class CheckIssue:
    level: str
    check_id: str
    path: str
    message: str
    recommendation: str
    evidence: Dict[str, Any]

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["evidence"] = json.dumps(self.evidence, sort_keys=True)
        return row


def _stage_split(total_samples: Optional[int], validation_fraction: float) -> Tuple[Optional[int], Optional[int]]:
    if total_samples is None:
        return None, None
    total = int(total_samples)
    fraction = float(max(0.0, min(0.5, validation_fraction)))
    if total < 2 or fraction <= 0.0:
        return total, 0
    val_size = int(round(total * fraction))
    val_size = max(1, min(total - 1, val_size))
    return total - val_size, val_size


def _batch_count(samples: Optional[int], batch_size: int, drop_last: bool) -> Optional[int]:
    if samples is None:
        return None
    samples = int(samples)
    batch_size = max(1, int(batch_size))
    if bool(drop_last):
        return samples // batch_size
    return int(math.ceil(samples / float(batch_size)))


def _issue(
    issues: List[CheckIssue],
    level: str,
    check_id: str,
    path: str,
    message: str,
    recommendation: str,
    **evidence: Any,
) -> None:
    issues.append(
        CheckIssue(
            level=str(level).upper(),
            check_id=str(check_id),
            path=str(path),
            message=str(message),
            recommendation=str(recommendation),
            evidence=dict(evidence),
        )
    )


def probe_training_corpus(config: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[CheckIssue]]:
    """Count configured train/internal-validation samples without building models."""
    issues: List[CheckIssue] = []
    dataset = dict(config["dataset"])
    try:
        from src.zelda_data.zelda_loader import create_dataloader

        loader = create_dataloader(
            data_dir=dataset["data_dir"],
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            use_vglc=bool(dataset["use_vglc"]),
            normalize=bool(dataset["normalize"]),
            room_level=bool(dataset["room_level"]),
            load_graphs=False,
            dungeon_ids=dataset.get("train_dungeons", list(range(1, 9))),
            variants=dataset.get("variants", [1, 2]),
        )
        total_samples = int(len(loader.dataset))
        return {
            "probed": True,
            "train_internal_total_samples": total_samples,
            "sample_kind": "rooms" if bool(dataset["room_level"]) else "dungeons",
            "data_dir": str(dataset["data_dir"]),
            "train_dungeons": [int(v) for v in dataset.get("train_dungeons", [])],
            "test_dungeons": [int(v) for v in dataset.get("test_dungeons", [])],
            "variants": [int(v) for v in dataset.get("variants", [])],
        }, issues
    except Exception as exc:
        _issue(
            issues,
            "WARNING",
            "data_probe_failed",
            "dataset.data_dir",
            f"Could not count configured training samples: {type(exc).__name__}: {exc}",
            "Fix the dataset path or rerun with --no-probe-data for static-only config checks.",
            data_dir=str(dataset.get("data_dir", "")),
        )
        return {
            "probed": False,
            "train_internal_total_samples": None,
            "sample_kind": "rooms" if bool(dataset.get("room_level", True)) else "dungeons",
            "data_dir": str(dataset.get("data_dir", "")),
        }, issues


def _check_batch_plan(
    issues: List[CheckIssue],
    *,
    stage_name: str,
    total_samples: Optional[int],
    validation_fraction: float,
    batch_size: int,
    drop_last: bool,
    min_samples_per_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    train_samples, val_samples = _stage_split(total_samples, validation_fraction)
    effective_train = train_samples
    if min_samples_per_epoch is not None and train_samples is not None:
        effective_train = max(int(train_samples), int(min_samples_per_epoch))
    train_batches = _batch_count(effective_train, batch_size, drop_last)
    val_batches = _batch_count(val_samples, batch_size, False)
    plan = {
        "stage": stage_name,
        "total_samples": total_samples,
        "train_samples": train_samples,
        "validation_samples": val_samples,
        "effective_train_samples": effective_train,
        "batch_size": int(batch_size),
        "drop_last": bool(drop_last),
        "train_batches_per_epoch": train_batches,
        "validation_batches": val_batches,
    }

    if train_batches == 0:
        _issue(
            issues,
            "ERROR",
            "zero_train_batches",
            f"{stage_name}.batch_plan",
            f"{stage_name} would have zero train batches with drop_last={bool(drop_last)}.",
            "Reduce dataset.batch_size, disable dataset.drop_last, or increase the effective sample count.",
            **plan,
        )
    elif train_batches is not None and train_batches < 4:
        _issue(
            issues,
            "WARNING",
            "very_few_train_batches",
            f"{stage_name}.batch_plan",
            f"{stage_name} has fewer than four train batches per epoch.",
            "Use a smaller batch size or a larger effective epoch if this is not a deliberate smoke profile.",
            **plan,
        )

    if bool(drop_last) and effective_train is not None and int(effective_train) % int(batch_size) != 0:
        _issue(
            issues,
            "INFO",
            "dropped_epoch_samples",
            f"{stage_name}.batch_plan",
            f"{stage_name} drops the final partial training batch.",
            "This is fine when intentional; set dataset.drop_last=false if every sampled room should contribute.",
            dropped_samples=int(effective_train) % int(batch_size),
            **plan,
        )
    return plan


def build_training_hyperparameter_report(
    config: Mapping[str, Any],
    *,
    data_probe: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    issues: List[CheckIssue] = []
    dataset = dict(config["dataset"])
    runtime = dict(config["runtime"])
    vqvae = dict(config["vqvae"])
    diffusion = dict(config["diffusion"])
    fast_sampler = dict(config["fast_sampler"])
    masked_room = dict(config["masked_room"])
    generation = dict(config["generation"])

    probed_total = None
    if data_probe:
        value = data_probe.get("train_internal_total_samples")
        probed_total = None if value is None else int(value)

    batch_size = int(dataset["batch_size"])
    drop_last = bool(dataset["drop_last"])
    batch_plans = [
        _check_batch_plan(
            issues,
            stage_name="vqvae",
            total_samples=probed_total,
            validation_fraction=float(vqvae["validation_fraction"]),
            batch_size=batch_size,
            drop_last=drop_last,
            min_samples_per_epoch=int(dataset["min_samples_per_epoch"]),
        ),
        _check_batch_plan(
            issues,
            stage_name="diffusion",
            total_samples=probed_total,
            validation_fraction=float(diffusion["validation_fraction"]),
            batch_size=batch_size,
            drop_last=drop_last,
        ),
        _check_batch_plan(
            issues,
            stage_name="fast_sampler",
            total_samples=probed_total,
            validation_fraction=float(fast_sampler["validation_fraction"]),
            batch_size=batch_size,
            drop_last=drop_last,
        ),
        _check_batch_plan(
            issues,
            stage_name="masked_room",
            total_samples=probed_total,
            validation_fraction=float(masked_room["validation_fraction"]),
            batch_size=batch_size,
            drop_last=drop_last,
        ),
    ]

    if str(runtime["device"]).lower() == "cuda":
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        if not cuda_available:
            _issue(
                issues,
                "ERROR",
                "cuda_requested_unavailable",
                "runtime.device",
                "runtime.device is set to cuda but CUDA is not available in this environment.",
                "Use runtime.device=auto/cpu for local checks or run training on a CUDA-enabled host.",
                configured_device=str(runtime["device"]),
            )

    if int(dataset["num_workers"]) > 0:
        _issue(
            issues,
            "INFO",
            "windows_worker_guardrail",
            "dataset.num_workers",
            "Non-zero DataLoader workers can be fragile on Windows notebooks and IDE launches.",
            "Keep dataset.num_workers=0 on Windows unless the launch path has been tested.",
            num_workers=int(dataset["num_workers"]),
        )

    if int(diffusion["latent_dim"]) != int(vqvae["latent_dim"]):
        _issue(
            issues,
            "ERROR",
            "latent_dim_mismatch",
            "diffusion.latent_dim",
            "Diffusion latent_dim does not match the tokenizer latent_dim.",
            "Keep diffusion.latent_dim == vqvae.latent_dim before training or loading checkpoints.",
            diffusion_latent_dim=int(diffusion["latent_dim"]),
            vqvae_latent_dim=int(vqvae["latent_dim"]),
        )

    for section_name, section in (("diffusion", diffusion), ("masked_room", masked_room)):
        context_dim = int(section["context_dim"])
        condition_heads = int(section["condition_num_attention_heads"])
        condition_hidden = int(section["condition_hidden_dim"])
        if context_dim % condition_heads != 0:
            _issue(
                issues,
                "ERROR",
                "condition_output_heads_not_divisible",
                f"{section_name}.context_dim",
                f"{section_name}.context_dim must be divisible by condition_num_attention_heads.",
                "Choose a context_dim/head-count pair accepted by CrossAttentionFusion.",
                context_dim=context_dim,
                condition_num_attention_heads=condition_heads,
            )
        if str(section["condition_gnn_type"]).lower() == "gps" and condition_hidden % condition_heads != 0:
            _issue(
                issues,
                "ERROR",
                "gps_hidden_heads_not_divisible",
                f"{section_name}.condition_hidden_dim",
                f"{section_name} GPS conditioning requires hidden_dim divisible by attention heads.",
                "Adjust condition_hidden_dim or condition_num_attention_heads before constructing the conditioner.",
                condition_hidden_dim=condition_hidden,
                condition_num_attention_heads=condition_heads,
            )

    for section_name, section in (("diffusion", diffusion), ("masked_room", masked_room)):
        model_channels = int(section["model_channels"])
        unet_heads = int(section["unet_num_heads"])
        channel_mult = [int(v) for v in section["unet_channel_mult"]]
        bad_widths = [model_channels * mult for mult in channel_mult if (model_channels * mult) % unet_heads != 0]
        if bad_widths:
            _issue(
                issues,
                "ERROR",
                "unet_heads_not_divisible",
                f"{section_name}.unet_num_heads",
                f"{section_name} U-Net attention widths are not divisible by unet_num_heads.",
                "Adjust model_channels, unet_channel_mult, or unet_num_heads.",
                model_channels=model_channels,
                unet_channel_mult=channel_mult,
                unet_num_heads=unet_heads,
                bad_widths=bad_widths,
            )

    if int(generation["num_diffusion_steps"]) > int(diffusion["num_timesteps"]):
        _issue(
            issues,
            "ERROR",
            "generation_steps_exceed_training_timesteps",
            "generation.num_diffusion_steps",
            "Runtime diffusion steps exceed the trained diffusion timestep count.",
            "Set generation.num_diffusion_steps <= diffusion.num_timesteps.",
            generation_num_diffusion_steps=int(generation["num_diffusion_steps"]),
            diffusion_num_timesteps=int(diffusion["num_timesteps"]),
        )

    if str(vqvae["architecture"]).lower() == "vqvae2":
        top_codebook_size = vqvae.get("top_codebook_size") or vqvae["codebook_size"]
        if int(top_codebook_size) > int(vqvae["codebook_size"]):
            _issue(
                issues,
                "WARNING",
                "vqvae2_top_codebook_larger_than_bottom",
                "vqvae.top_codebook_size",
                "The VQ-VAE-2 top codebook is larger than the bottom codebook.",
                "This is allowed, but use it only as an explicit capacity ablation and report utilization per level.",
                top_codebook_size=int(top_codebook_size),
                bottom_codebook_size=int(vqvae["codebook_size"]),
            )

    if probed_total is not None:
        train_samples, _ = _stage_split(probed_total, float(vqvae["validation_fraction"]))
        if train_samples is not None and int(vqvae["codebook_size"]) > max(512, int(train_samples)):
            _issue(
                issues,
                "WARNING",
                "codebook_large_for_corpus",
                "vqvae.codebook_size",
                "Tokenizer codebook size is large relative to the probed training corpus.",
                "Treat this as an ablation and require codebook utilization/entropy evidence before using it as the main branch.",
                codebook_size=int(vqvae["codebook_size"]),
                train_samples=int(train_samples),
            )

    if float(vqvae["validation_fraction"]) <= 0.0 and str(vqvae["best_checkpoint_metric"]) == "val_loss":
        _issue(
            issues,
            "WARNING",
            "val_metric_without_val_split",
            "vqvae.best_checkpoint_metric",
            "VQ-VAE asks for val_loss selection while validation_fraction is zero.",
            "Enable validation_fraction or explicitly set best_checkpoint_metric=train_loss for smoke runs.",
            validation_fraction=float(vqvae["validation_fraction"]),
            best_checkpoint_metric=str(vqvae["best_checkpoint_metric"]),
        )

    if float(diffusion["alpha_logic"]) > 0.0 and int(diffusion["warmup_epochs"]) >= int(diffusion["epochs"]):
        _issue(
            issues,
            "WARNING",
            "logic_loss_never_activates",
            "diffusion.warmup_epochs",
            "Logic loss is configured but warmup_epochs is not smaller than total diffusion epochs.",
            "Reduce diffusion.warmup_epochs or set alpha_logic=0 for a visual-only ablation.",
            alpha_logic=float(diffusion["alpha_logic"]),
            warmup_epochs=int(diffusion["warmup_epochs"]),
            epochs=int(diffusion["epochs"]),
        )

    if bool(diffusion["puzzle_stage_conditioning_enabled"]) != bool(generation["puzzle_stage_topology_enabled"]):
        _issue(
            issues,
            "INFO",
            "stage_conditioning_runtime_mismatch",
            "generation.puzzle_stage_topology_enabled",
            "Training and runtime stage-topology flags differ.",
            "Enable runtime stage topology only with stage-conditioned checkpoints; disable it for base checkpoints.",
            diffusion_puzzle_stage_conditioning_enabled=bool(diffusion["puzzle_stage_conditioning_enabled"]),
            generation_puzzle_stage_topology_enabled=bool(generation["puzzle_stage_topology_enabled"]),
        )

    level_counts = {
        "ERROR": sum(1 for issue in issues if issue.level == "ERROR"),
        "WARNING": sum(1 for issue in issues if issue.level == "WARNING"),
        "INFO": sum(1 for issue in issues if issue.level == "INFO"),
    }
    return {
        "summary": {
            "status": "fail" if level_counts["ERROR"] else "pass",
            "issue_counts": level_counts,
            "config_name": str(config.get("runtime", {}).get("experiment_name", "")),
        },
        "data_probe": dict(data_probe or {"probed": False, "train_internal_total_samples": None}),
        "batch_plans": batch_plans,
        "issues": [asdict(issue) for issue in issues],
    }


def write_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "training_hyperparameter_check.json"
    csv_path = output_dir / "training_hyperparameter_check.csv"
    md_path = output_dir / "training_hyperparameter_check.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    issues = [CheckIssue(**issue) for issue in report.get("issues", [])]
    fieldnames = ["level", "check_id", "path", "message", "recommendation", "evidence"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.to_row())

    lines = [
        "# Training Hyperparameter Check",
        "",
        f"- status: `{report['summary']['status']}`",
        f"- errors: `{report['summary']['issue_counts']['ERROR']}`",
        f"- warnings: `{report['summary']['issue_counts']['WARNING']}`",
        f"- infos: `{report['summary']['issue_counts']['INFO']}`",
        "",
        "## Batch Plan",
        "",
        "| Stage | Total | Train | Val | Effective Train | Batch | Drop Last | Train Batches | Val Batches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for plan in report.get("batch_plans", []):
        lines.append(
            f"| `{plan['stage']}` | {plan.get('total_samples')} | {plan.get('train_samples')} | "
            f"{plan.get('validation_samples')} | {plan.get('effective_train_samples')} | "
            f"{plan.get('batch_size')} | {int(bool(plan.get('drop_last')))} | "
            f"{plan.get('train_batches_per_epoch')} | {plan.get('validation_batches')} |"
        )
    lines.extend(["", "## Issues", ""])
    if not issues:
        lines.append("No issues found.")
    else:
        lines.extend(["| Level | Check | Path | Message | Recommendation |", "|---|---|---|---|---|"])
        for issue in issues:
            lines.append(
                f"| `{issue.level}` | `{issue.check_id}` | `{issue.path}` | "
                f"{issue.message} | {issue.recommendation} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check resolved training hyperparameters and batch math.")
    parser.add_argument("--config", type=Path, default=Path("configs") / "zelda_hmolqd.yaml")
    parser.add_argument("--output", type=Path, default=Path("results") / "training_hyperparameter_check")
    parser.add_argument("--probe-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-errors", action="store_true", help="Always exit 0 after writing the report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = merge_config(yaml_path=str(args.config), cli_overrides=None)
    data_probe: Dict[str, Any] = {"probed": False, "train_internal_total_samples": None}
    probe_issues: List[CheckIssue] = []
    if bool(args.probe_data):
        data_probe, probe_issues = probe_training_corpus(config)
    report = build_training_hyperparameter_report(config, data_probe=data_probe)
    if probe_issues:
        report["issues"].extend(asdict(issue) for issue in probe_issues)
        for issue in probe_issues:
            report["summary"]["issue_counts"][issue.level] += 1
        report["summary"]["status"] = "fail" if report["summary"]["issue_counts"]["ERROR"] else "pass"
    write_report(args.output, report)
    print(
        "Training hyperparameter check: "
        f"{report['summary']['status']} "
        f"(errors={report['summary']['issue_counts']['ERROR']}, "
        f"warnings={report['summary']['issue_counts']['WARNING']}, "
        f"infos={report['summary']['issue_counts']['INFO']}) -> {args.output}"
    )
    if report["summary"]["issue_counts"]["ERROR"] and not bool(args.allow_errors):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
