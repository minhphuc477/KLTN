"""Generate publication-style Chapter 4 figures for the thesis report.

This script reads the current repository artifacts and exports a compact set of
high-signal figures into ``REPORT_LATEX/figures/ch4``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "REPORT_LATEX" / "figures" / "ch4"


METHOD_COLORS: Dict[str, str] = {
    "FULL_GA": "#1f77b4",
    "FULL_CVT": "#2ca02c",
    "CORE_GA": "#d62728",
}

REGIME_MARKERS: Dict[str, str] = {
    "in_dist": "o",
    "ood_small": "s",
    "ood_large": "^",
}

BRANCH_MARKERS: Dict[str, str] = {
    "diffusion_cfg3_logic0_steps50": "o",
    "fast_cfg3_logic0_steps4": "^",
    "masked_room_full": "s",
}

BRANCH_COLORS: Dict[str, str] = {
    "diffusion_cfg3_logic0_steps50": "#1f77b4",
    "fast_cfg3_logic0_steps4": "#ff7f0e",
    "masked_room_full": "#2ca02c",
}

VQVAE_LABELS: Dict[str, str] = {
    "Baseline (256, CoordConv, MRF)": "Baseline",
    "Codebook-128": "Codebook 128",
    "Codebook-512": "Codebook 512",
    "Hidden-64": "H64",
    "No CoordConv": "No CoordConv",
    "No MRF": "No MRF",
}


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> None:
    out_path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_vqvae_rows() -> List[Mapping[str, Any]]:
    path = ROOT / "results" / "thesis_hparam_evidence_current.json"
    if not path.exists():
        path = ROOT / "results" / "thesis_hparam_evidence_2026_04_19.json"
    payload = _load_json(path)
    return list(payload.get("vqvae", []))


def plot_vqvae_overview() -> None:
    rows = _load_vqvae_rows()
    if not rows:
        return

    order = [
        "Baseline (256, CoordConv, MRF)",
        "Codebook-128",
        "Codebook-512",
        "Hidden-64",
        "No CoordConv",
        "No MRF",
    ]
    row_map = {str(row["label"]): row for row in rows}
    selected = [row_map[label] for label in order if label in row_map]
    labels = [VQVAE_LABELS[str(row["label"])] for row in selected]
    best_loss = np.asarray([float(row["best_val_loss"]) for row in selected], dtype=float)
    last_loss = np.asarray([float(row["last_val_loss"]) for row in selected], dtype=float)
    perplexity = np.asarray([float(row["best_val_perplexity"]) for row in selected], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    x = np.arange(len(labels))
    width = 0.38
    axes[0].bar(x - width / 2, best_loss, width=width, color="#1f77b4", label="Best validation loss")
    axes[0].bar(x + width / 2, last_loss, width=width, color="#9ecae1", label="Final validation loss")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18)
    axes[0].set_ylabel("Validation loss (log scale)")
    axes[0].set_title("VQ-VAE tokenizer convergence")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].legend(frameon=False, fontsize=9)

    bars = axes[1].bar(x, perplexity, color="#2ca02c", width=0.62)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18)
    axes[1].set_ylabel("Validation perplexity")
    axes[1].set_title("Active codebook usage")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    for bar, value in zip(bars, perplexity):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    _save(fig, "vqvae_ablation_overview.png")


def _short_room_label(row: Mapping[str, Any]) -> str:
    group = str(row["group"])
    label = str(row["label"])
    variant = str(row["variant"])
    if group == "branch_compare_pdrop035":
        return {
            "diffusion_cfg3_logic0_steps50": "Diff",
            "fast_cfg3_logic0_steps4": "Fast",
            "masked_room_full": "Mask",
        }.get(variant, variant)
    if group == "pdrop_sweep":
        digits = label.replace("pdrop", "")
        return f"p={int(digits) / 100:.2f}"
    if group == "tokenizer_compare":
        prefix = "512" if "512" in label else "256"
        suffix = "D" if "diffusion" in variant else "M"
        return f"{prefix}-{suffix}"
    if group == "stage_semantics_compare":
        prefix = "Stage+" if "stageconditioned" in label else "Base"
        suffix = "D" if "diffusion" in variant else "M"
        return f"{prefix}{suffix}"
    if group == "structure_control_compare":
        prefix = "Struct+" if "masked_branch" in label else "Base"
        suffix = "D" if "diffusion" in variant else "M"
        return f"{prefix}{suffix}"
    return label


def _scatter_panel(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    for _, row in frame.iterrows():
        variant = str(row["variant"])
        face = BRANCH_COLORS.get(variant, "#4c566a")
        success = bool(row["cbs_success"])
        marker = BRANCH_MARKERS.get(variant, "o")
        ax.scatter(
            float(row["generation_time_sec"]),
            float(row["room_nearest_reference_ncd_mean"]),
            marker=marker,
            s=82,
            c=[face if success else "white"],
            edgecolors=face,
            linewidths=1.8,
            zorder=3,
        )
        ax.text(
            float(row["generation_time_sec"]) + 0.35,
            float(row["room_nearest_reference_ncd_mean"]) + 0.0015,
            _short_room_label(row),
            fontsize=8,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("End-to-end time (s)")
    ax.set_ylabel("Nearest-reference NCD")
    ax.grid(alpha=0.25, linestyle="--")


def plot_room_generation_landscape() -> None:
    frame = pd.read_csv(ROOT / "results" / "thesis_ch4_evals" / "chapter4_eval_rows.csv")
    groups = [
        ("branch_compare_pdrop035", "Room-generation branches"),
        ("pdrop_sweep", "Dropout sweep"),
        ("tokenizer_compare", "Codebook comparison"),
        ("stage_semantics_compare", "Stage-semantic conditioning"),
        ("structure_control_compare", "Structure conditioning"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2))
    axes_flat = list(axes.flatten())
    for ax, (group_name, title) in zip(axes_flat, groups):
        subset = frame.loc[frame["group"] == group_name].copy()
        _scatter_panel(ax, subset, title)

    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    legend_ax.scatter([], [], marker="o", c="#1f77b4", edgecolors="#1f77b4", s=82, label="Diffusion")
    legend_ax.scatter([], [], marker="^", c="#ff7f0e", edgecolors="#ff7f0e", s=82, label="Fast sampler")
    legend_ax.scatter([], [], marker="s", c="#2ca02c", edgecolors="#2ca02c", s=82, label="Masked room")
    legend_ax.scatter([], [], marker="o", c="white", edgecolors="#4c566a", s=82, label="P-CBS rejected")
    legend_ax.scatter([], [], marker="o", c="#4c566a", edgecolors="#4c566a", s=82, label="P-CBS accepted")
    legend_ax.legend(loc="center", frameon=False, fontsize=10)
    legend_ax.text(
        0.05,
        0.08,
        "Better points lie closer to the lower-left corner: faster and closer to reference.",
        transform=legend_ax.transAxes,
        fontsize=9,
    )

    _save(fig, "room_generation_landscape.png")


def plot_topology_robustness() -> None:
    frame = pd.read_csv(ROOT / "results" / "ood_blinded_eval_current" / "ood_scaling_summary.csv")
    regime_order = ["in_dist", "ood_small", "ood_large"]
    method_order = ["FULL_GA", "FULL_CVT", "CORE_GA"]
    regime_labels = {
        "in_dist": "In-distribution",
        "ood_small": "OOD-small",
        "ood_large": "OOD-large",
    }
    method_labels = {
        "FULL_GA": "Full + GA",
        "FULL_CVT": "Full + CVT",
        "CORE_GA": "Core rules + GA",
    }
    regime_short = {"in_dist": "ID", "ood_small": "S", "ood_large": "L"}
    method_short = {"FULL_GA": "F-GA", "FULL_CVT": "F-CVT", "CORE_GA": "C-GA"}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))

    x = np.arange(len(regime_order))
    width = 0.24
    for offset, method in zip([-width, 0.0, width], method_order):
        vals = []
        completeness = []
        for regime in regime_order:
            row = frame.loc[(frame["regime"] == regime) & (frame["method"] == method)].iloc[0]
            vals.append(float(row["constraint_valid_rate"]))
            completeness.append(float(row["overall_completeness"]))
        bars = axes[0].bar(
            x + offset,
            vals,
            width=width,
            color=METHOD_COLORS[method],
            alpha=0.82,
            label=method_labels[method],
        )
        axes[0].plot(
            x + offset,
            completeness,
            color="#111827",
            marker="D",
            linewidth=1.1,
            markersize=4,
        )
        for bar, value in zip(bars, vals):
            if value > 0.0:
                continue
            bar_center = bar.get_x() + bar.get_width() / 2
            axes[0].scatter(
                [bar_center],
                [0.012],
                s=26,
                marker="o",
                c=[METHOD_COLORS[method]],
                edgecolors="#111827",
                linewidths=0.5,
                zorder=4,
            )
            axes[0].text(
                bar_center,
                0.022,
                "0.00",
                ha="center",
                va="bottom",
                fontsize=7,
                color=METHOD_COLORS[method],
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([regime_labels[r] for r in regime_order])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Constraint validity rate")
    axes[0].set_title("Constraint robustness across scale regimes")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].legend(frameon=False, fontsize=8)

    for _, row in frame.iterrows():
        method = str(row["method"])
        regime = str(row["regime"])
        axes[1].scatter(
            float(row["generation_time_sec"]),
            float(row["novelty_vs_reference"]),
            marker=REGIME_MARKERS[regime],
            s=95,
            c=[METHOD_COLORS[method]],
            edgecolors="#111827",
            linewidths=0.6,
        )
        axes[1].text(
            float(row["generation_time_sec"]) + 0.12,
            float(row["novelty_vs_reference"]) + 0.0016,
            f"{regime_short[regime]}-{method_short[method]}",
            fontsize=7.5,
        )
    axes[1].set_xlabel("Mean generation time (s)")
    axes[1].set_ylabel("Novelty vs. reference")
    axes[1].set_title("Novelty-cost trade-off")
    axes[1].grid(alpha=0.25, linestyle="--")

    _save(fig, "topology_robustness_tradeoff.png")


def plot_mission_graph_matched_budget() -> None:
    path = ROOT / "results" / "old_result" / "matched_budget_topology_v1" / "matched_budget_summary.csv"
    if not path.exists():
        return

    frame = pd.read_csv(path)
    method_order = ["RANDOM", "ES", "GA", "MAP_ELITES", "FULL"]
    method_labels = {
        "RANDOM": "Random",
        "ES": "ES",
        "GA": "GA",
        "MAP_ELITES": "MAP-Elites",
        "FULL": "Full",
    }
    method_colors = {
        "RANDOM": "#9ca3af",
        "ES": "#1f77b4",
        "GA": "#ff7f0e",
        "MAP_ELITES": "#2ca02c",
        "FULL": "#d62728",
    }
    frame = frame.set_index("method").loc[method_order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    x = np.arange(len(frame))
    bars = axes[0].bar(
        x,
        frame["fitness"].to_numpy(dtype=float),
        color=[method_colors[m] for m in frame["method"]],
        width=0.68,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([method_labels[m] for m in frame["method"]], rotation=10)
    axes[0].set_ylabel("Mean fitness")
    axes[0].set_title("Matched-budget mission-graph fitness")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    for bar, valid in zip(bars, frame["feasible_operational_rate"].to_numpy(dtype=float)):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"valid={valid:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for _, row in frame.iterrows():
        method = str(row["method"])
        axes[1].scatter(
            float(row["num_nodes"]),
            float(row["secret_component_count"]),
            s=18.0 * float(row["generation_time_sec"]),
            c=[method_colors[method]],
            edgecolors="#111827",
            linewidths=0.6,
        )
        axes[1].text(
            float(row["num_nodes"]) + 0.22,
            float(row["secret_component_count"]) + 0.01,
            method_labels[method],
            fontsize=8,
        )
    axes[1].set_xlabel("Mean node count")
    axes[1].set_ylabel("Mean secret-component count")
    axes[1].set_title("Structural richness vs. search cost")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].text(
        0.04,
        0.05,
        "Bubble size is proportional to mean generation time.",
        transform=axes[1].transAxes,
        fontsize=8.5,
    )

    _save(fig, "mission_graph_matched_budget.png")


def _pcbs_paths() -> List[Path]:
    candidates = [
        ROOT / "results" / "pcbs_component_ablation_novice_l123_v1" / "summary.json",
        ROOT / "results" / "pcbs_component_ablation_balanced_l123_v4" / "summary.json",
        ROOT / "results" / "pcbs_component_ablation_speedrunner_l123_v1" / "summary.json",
    ]
    return [path for path in candidates if path.exists()]


def _persona_name_from_path(path: Path) -> str:
    name = path.parent.name
    if "novice" in name:
        return "Novice"
    if "balanced" in name:
        return "Balanced"
    if "speedrunner" in name:
        return "Speedrunner"
    return name


def plot_pcbs_persona_ablation() -> None:
    paths = _pcbs_paths()
    if len(paths) < 2:
        return

    payloads = [(_persona_name_from_path(path), _load_json(path)) for path in paths]
    variant_order = ["full", "no_revisit", "no_uncertainty", "no_deliberation", "no_affordance", "no_focus"]
    variant_labels = ["Full", "No revisit", "No uncertainty", "No deliberation", "No affordance", "No focus"]
    persona_labels = [name for name, _ in payloads]

    success = np.full((len(persona_labels), len(variant_order)), np.nan, dtype=float)
    confusion = np.full((len(persona_labels), len(variant_order)), np.nan, dtype=float)

    for i, (_, payload) in enumerate(payloads):
        variants = payload.get("variants", {})
        for j, key in enumerate(variant_order):
            if key not in variants:
                continue
            row = variants[key]
            success[i, j] = float(row.get("success_rate", math.nan))
            confusion[i, j] = math.log10(1.0 + float(row.get("avg_confusion_index", 0.0)))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    im0 = axes[0].imshow(success, cmap="YlGnBu", vmin=0.0, vmax=max(0.34, np.nanmax(success)))
    axes[0].set_xticks(np.arange(len(variant_labels)))
    axes[0].set_xticklabels(variant_labels, rotation=24, ha="right")
    axes[0].set_yticks(np.arange(len(persona_labels)))
    axes[0].set_yticklabels(persona_labels)
    axes[0].set_title("P-CBS success rate")
    for i in range(success.shape[0]):
        for j in range(success.shape[1]):
            if math.isnan(success[i, j]):
                continue
            axes[0].text(j, i, f"{success[i, j]*100:.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(confusion, cmap="magma")
    axes[1].set_xticks(np.arange(len(variant_labels)))
    axes[1].set_xticklabels(variant_labels, rotation=24, ha="right")
    axes[1].set_yticks(np.arange(len(persona_labels)))
    axes[1].set_yticklabels(persona_labels)
    axes[1].set_title("log10(1 + confusion index)")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            if math.isnan(confusion[i, j]):
                continue
            axes[1].text(j, i, f"{confusion[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    _save(fig, "pcbs_persona_ablation_heatmap.png")


def main() -> None:
    _ensure_out_dir()
    plot_vqvae_overview()
    plot_room_generation_landscape()
    plot_mission_graph_matched_budget()
    plot_topology_robustness()
    plot_pcbs_persona_ablation()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
