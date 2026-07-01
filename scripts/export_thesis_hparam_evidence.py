"""
Export thesis-facing empirical evidence summaries for VQ-VAE ablations and
in-flight diffusion branches.

This script is intentionally narrow: it reads the repository's current training
artifacts and emits JSON + Markdown tables that can be cited in Chapter 4.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class VQVaeRunSpec:
    run_id: str
    label: str
    output_dir: str
    codebook_size: int
    hidden_dim: int
    latent_dim: int
    coordconv: bool
    mrf_penalty_weight: float


@dataclass
class DiffusionRunSpec:
    run_id: str
    label: str
    output_dir: str
    tokenizer_label: str
    tokenizer_codebook_size: int


VQVAE_RUNS: List[VQVaeRunSpec] = [
    VQVaeRunSpec(
        run_id="baseline256_coord_mrf",
        label="Baseline (256, CoordConv, MRF)",
        output_dir="outputs/vqvae_audit_baseline_v2",
        codebook_size=256,
        hidden_dim=96,
        latent_dim=64,
        coordconv=True,
        mrf_penalty_weight=0.05,
    ),
    VQVaeRunSpec(
        run_id="codebook128",
        label="Codebook-128",
        output_dir="outputs/vqvae_ablation_codebook128_v2",
        codebook_size=128,
        hidden_dim=96,
        latent_dim=64,
        coordconv=True,
        mrf_penalty_weight=0.05,
    ),
    VQVaeRunSpec(
        run_id="codebook512",
        label="Codebook-512",
        output_dir="outputs/vqvae_ablation_codebook512_v2",
        codebook_size=512,
        hidden_dim=96,
        latent_dim=64,
        coordconv=True,
        mrf_penalty_weight=0.05,
    ),
    VQVaeRunSpec(
        run_id="hidden64",
        label="Hidden-64",
        output_dir="outputs/vqvae_ablation_hidden64_v2",
        codebook_size=256,
        hidden_dim=64,
        latent_dim=64,
        coordconv=True,
        mrf_penalty_weight=0.05,
    ),
    VQVaeRunSpec(
        run_id="no_coordconv",
        label="No CoordConv",
        output_dir="outputs/vqvae_ablation_no_coordconv_v2",
        codebook_size=256,
        hidden_dim=96,
        latent_dim=64,
        coordconv=False,
        mrf_penalty_weight=0.05,
    ),
    VQVaeRunSpec(
        run_id="no_mrf",
        label="No MRF",
        output_dir="outputs/vqvae_ablation_no_mrf_v2",
        codebook_size=256,
        hidden_dim=96,
        latent_dim=64,
        coordconv=True,
        mrf_penalty_weight=0.0,
    ),
]

DIFFUSION_RUNS: List[DiffusionRunSpec] = [
    DiffusionRunSpec(
        run_id="baseline_branch_diffusion",
        label="Baseline branch diffusion",
        output_dir="outputs/zelda_hmolqd_downstream_baseline_puzzle_subtype_v2_rerun_heldout_20260419_182314",
        tokenizer_label="Baseline VQ-VAE",
        tokenizer_codebook_size=256,
    ),
    DiffusionRunSpec(
        run_id="codebook512_branch_diffusion",
        label="Codebook-512 branch diffusion",
        output_dir="outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v2_rerun_heldout_20260419_182314",
        tokenizer_label="Codebook-512 VQ-VAE",
        tokenizer_codebook_size=512,
    ),
]


EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+): loss=([0-9.]+), diffusion=([0-9.]+), "
    r"val_diffusion_loss=([0-9.]+), val_logic_loss=([0-9.]+), "
    r"val_total_loss=([0-9.]+), val_solvability_proxy=([0-9.]+), "
    r"logic_loss_(enabled|disabled)"
)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_vqvae_summary(spec: VQVaeRunSpec) -> Dict[str, Any]:
    history_path = REPO_ROOT / spec.output_dir / "checkpoints" / "vqvae" / "vqvae_training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing VQ-VAE history: {history_path}")
    history = _load_json(history_path)
    if not isinstance(history, list) or not history:
        raise ValueError(f"Unexpected VQ-VAE history format: {history_path}")
    best = min(history, key=lambda row: float(row.get("val_loss", float("inf"))))
    last = history[-1]
    return {
        **asdict(spec),
        "history_path": str(history_path.relative_to(REPO_ROOT)),
        "epochs_completed": int(len(history)),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "best_val_recon_loss": float(best.get("val_recon_loss", best.get("recon_loss", 0.0))),
        "best_val_accuracy": float(best.get("val_accuracy", 0.0)),
        "best_val_perplexity": float(best.get("val_perplexity", 0.0)),
        "best_codebook_utilization": float(best.get("codebook_utilization", 0.0)),
        "best_codebook_entropy_norm": float(best.get("codebook_usage_entropy_normalized", 0.0)),
        "best_ema_live_rate": float(best.get("ema_live_rate", 0.0)),
        "last_val_loss": float(last.get("val_loss", float("inf"))),
    }


def _load_diffusion_run_metadata(spec: DiffusionRunSpec) -> Dict[str, Any]:
    metadata_path = REPO_ROOT / spec.output_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    payload = _load_json(metadata_path)
    return payload if isinstance(payload, dict) else {}


def build_diffusion_summary(spec: DiffusionRunSpec) -> Dict[str, Any]:
    log_path = REPO_ROOT / spec.output_dir / "training.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing diffusion log: {log_path}")
    epochs: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = EPOCH_RE.search(line)
            if not match:
                continue
            epochs.append(
                {
                    "epoch": int(match.group(1)),
                    "epochs": int(match.group(2)),
                    "loss": float(match.group(3)),
                    "diffusion_loss": float(match.group(4)),
                    "val_diffusion_loss": float(match.group(5)),
                    "val_logic_loss": float(match.group(6)),
                    "val_total_loss": float(match.group(7)),
                    "val_solvability_proxy": float(match.group(8)),
                    "logic_status": str(match.group(9)),
                }
            )
    if not epochs:
        raise ValueError(f"No epoch summaries found in diffusion log: {log_path}")
    best = min(epochs, key=lambda row: float(row["val_total_loss"]))
    latest = epochs[-1]
    metadata = _load_diffusion_run_metadata(spec)
    return {
        **asdict(spec),
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "epochs_observed": len(epochs),
        "latest_epoch": int(latest["epoch"]),
        "latest_train_loss": float(latest["loss"]),
        "latest_val_total_loss": float(latest["val_total_loss"]),
        "latest_val_diffusion_loss": float(latest["val_diffusion_loss"]),
        "latest_val_logic_loss": float(latest["val_logic_loss"]),
        "latest_val_solvability_proxy": float(latest["val_solvability_proxy"]),
        "best_epoch": int(best["epoch"]),
        "best_val_total_loss": float(best["val_total_loss"]),
        "best_val_diffusion_loss": float(best["val_diffusion_loss"]),
        "best_val_logic_loss": float(best["val_logic_loss"]),
        "best_val_solvability_proxy": float(best["val_solvability_proxy"]),
        "run_metadata": metadata,
        "protocol_warning": (
            "This branch was launched before the 2026-04-19 diffusion held-out-validation patch; "
            "its logged validation metrics are interim and must not be treated as thesis-final checkpoint-selection evidence."
        ),
    }


def _md_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    header_list = list(headers)
    lines = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join(["---"] * len(header_list)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_markdown(payload: Dict[str, Any]) -> str:
    vqvae_rows = []
    for row in payload["vqvae"]:
        vqvae_rows.append(
            [
                row["label"],
                row["codebook_size"],
                row["hidden_dim"],
                _fmt(row["coordconv"]),
                _fmt(row["mrf_penalty_weight"], digits=3),
                row["best_epoch"],
                _fmt(row["best_val_loss"]),
                _fmt(row["best_val_perplexity"]),
                _fmt(row["best_codebook_utilization"]),
                _fmt(row["best_ema_live_rate"]),
            ]
        )

    diffusion_rows = []
    for row in payload["diffusion"]:
        diffusion_rows.append(
            [
                row["label"],
                row["tokenizer_label"],
                row["best_epoch"],
                row["latest_epoch"],
                _fmt(row["best_val_total_loss"]),
                _fmt(row["best_val_diffusion_loss"]),
                _fmt(row["best_val_logic_loss"]),
                _fmt(row["best_val_solvability_proxy"]),
            ]
        )

    top_vq = payload["vqvae_ranked"][0]
    md_lines = [
        "# Thesis Hyperparameter Evidence Summary",
        "",
        f"Generated from local artifacts on {payload['generated_at_local']}.",
        "",
        "## Protocol Alerts",
        "",
        "- Diffusion branches currently running were launched before the held-out diffusion validation patch added on 2026-04-19.",
        "- Completed VQ-VAE ablations use a deterministic 10% hold-out split and support tokenizer screening, but only single-seed results are available so far.",
        "- For thesis-final Chapter 4 claims, rerun diffusion branches under the patched trainer and keep final comparisons on the fixed-graph multi-seed audit, not on in-training validation alone.",
        "",
        "## VQ-VAE Ablations",
        "",
        _md_table(
            [
                "Variant",
                "Codebook",
                "Hidden",
                "CoordConv",
                "MRF",
                "Best Epoch",
                "Best Val Loss",
                "Val Perplexity",
                "Utilization",
                "EMA Live Rate",
            ],
            vqvae_rows,
        ),
        "",
        f"Best validation loss among the six completed VQ-VAE runs: **{top_vq['label']}** at epoch {top_vq['best_epoch']} with val_loss={_fmt(top_vq['best_val_loss'])}.",
        "",
        "## Diffusion Interim Status",
        "",
        _md_table(
            [
                "Branch",
                "Tokenizer",
                "Best Epoch",
                "Latest Epoch",
                "Best Val Total",
                "Best Val Diff",
                "Best Val Logic",
                "Best Solvability Proxy",
            ],
            diffusion_rows,
        ),
        "",
        "These diffusion values are **interim only** because the runs were started before diffusion adopted a held-out validation split.",
        "",
    ]
    return "\n".join(md_lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis-facing evidence summaries from current artifacts.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results") / "thesis_hparam_evidence_2026_04_19",
        help="Output path prefix without extension. Produces both .json and .md.",
    )
    args = parser.parse_args()

    output_prefix = (REPO_ROOT / args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    vqvae = [build_vqvae_summary(spec) for spec in VQVAE_RUNS]
    vqvae_ranked = sorted(vqvae, key=lambda row: float(row["best_val_loss"]))
    diffusion = [build_diffusion_summary(spec) for spec in DIFFUSION_RUNS]

    payload = {
        "generated_at_local": output_prefix.parent.stat().st_mtime_ns,
        "repo_root": str(REPO_ROOT),
        "vqvae": vqvae,
        "vqvae_ranked": vqvae_ranked,
        "diffusion": diffusion,
    }
    # Replace opaque timestamp with ISO-ish string after payload construction.
    from datetime import datetime

    payload["generated_at_local"] = datetime.now().astimezone().isoformat(timespec="seconds")

    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
