"""Compare diffusion training runs for logic-loss A/B experiments.

This script compares two MetricsLogger JSON histories (baseline vs variant),
aligns epochs, computes per-epoch deltas, and writes summary artifacts.

Typical use:
    python scripts/compare_logic_loss_ab.py \
        --baseline checkpoints/ab_legacy/logs \
        --variant checkpoints/ab_predicted/logs \
        --output results/logic_loss_ab
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _resolve_metrics_json(path_like: str) -> Path:
    """Resolve either a direct metrics JSON path or latest JSON in a directory."""
    path = Path(path_like)
    if path.is_file():
        return path
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Path not found or not a directory: {path}")

    candidates = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .json metrics files found in {path}")
    return candidates[-1]


def _load_history(metrics_path: Path) -> List[Dict[str, Any]]:
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Metrics file must contain a JSON list: {metrics_path}")
    return [row for row in payload if isinstance(row, dict)]


def _to_epoch_map(history: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """Build epoch -> numeric metrics map (last row wins for duplicate epochs)."""
    epoch_map: Dict[int, Dict[str, float]] = {}
    for row in history:
        epoch_raw = row.get("epoch")
        if not isinstance(epoch_raw, (int, float)):
            continue
        epoch = int(epoch_raw)

        numeric: Dict[str, float] = {}
        for k, v in row.items():
            if isinstance(v, (int, float)):
                numeric[k] = float(v)
        epoch_map[epoch] = numeric
    return epoch_map


def _safe_get(d: Dict[str, float], key: str) -> float:
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) else float("nan")


def _mean(values: List[float]) -> float:
    usable = [v for v in values if v == v]
    if not usable:
        return float("nan")
    return sum(usable) / float(len(usable))


def _best_max(epoch_map: Dict[int, Dict[str, float]], key: str) -> Tuple[int, float]:
    best_epoch = -1
    best_val = float("-inf")
    for e, m in epoch_map.items():
        v = _safe_get(m, key)
        if v == v and v > best_val:
            best_val = v
            best_epoch = e
    return best_epoch, best_val


def _best_min(epoch_map: Dict[int, Dict[str, float]], key: str) -> Tuple[int, float]:
    best_epoch = -1
    best_val = float("inf")
    for e, m in epoch_map.items():
        v = _safe_get(m, key)
        if v == v and v < best_val:
            best_val = v
            best_epoch = e
    return best_epoch, best_val


def compare_runs(
    baseline_path: Path,
    variant_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    baseline_hist = _load_history(baseline_path)
    variant_hist = _load_history(variant_path)

    baseline_epochs = _to_epoch_map(baseline_hist)
    variant_epochs = _to_epoch_map(variant_hist)

    common_epochs = sorted(set(baseline_epochs.keys()) & set(variant_epochs.keys()))
    if not common_epochs:
        raise ValueError("No common epoch values found between baseline and variant metrics.")

    output_dir.mkdir(parents=True, exist_ok=True)

    delta_csv = output_dir / "epoch_delta.csv"
    with delta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "baseline_loss",
                "variant_loss",
                "delta_loss_variant_minus_baseline",
                "baseline_val_solvability",
                "variant_val_solvability",
                "delta_val_solvability_variant_minus_baseline",
                "baseline_logic_loss",
                "variant_logic_loss",
                "delta_logic_loss_variant_minus_baseline",
            ]
        )

        delta_loss_vals: List[float] = []
        delta_solv_vals: List[float] = []
        delta_logic_vals: List[float] = []

        for epoch in common_epochs:
            b = baseline_epochs[epoch]
            v = variant_epochs[epoch]

            b_loss = _safe_get(b, "loss")
            v_loss = _safe_get(v, "loss")
            d_loss = v_loss - b_loss

            b_solv = _safe_get(b, "val_solvability")
            v_solv = _safe_get(v, "val_solvability")
            d_solv = v_solv - b_solv

            b_logic = _safe_get(b, "logic_loss")
            v_logic = _safe_get(v, "logic_loss")
            d_logic = v_logic - b_logic

            delta_loss_vals.append(d_loss)
            delta_solv_vals.append(d_solv)
            delta_logic_vals.append(d_logic)

            writer.writerow(
                [
                    epoch,
                    b_loss,
                    v_loss,
                    d_loss,
                    b_solv,
                    v_solv,
                    d_solv,
                    b_logic,
                    v_logic,
                    d_logic,
                ]
            )

    b_last_epoch = max(baseline_epochs.keys())
    v_last_epoch = max(variant_epochs.keys())
    b_last = baseline_epochs[b_last_epoch]
    v_last = variant_epochs[v_last_epoch]

    b_best_solv_epoch, b_best_solv = _best_max(baseline_epochs, "val_solvability")
    v_best_solv_epoch, v_best_solv = _best_max(variant_epochs, "val_solvability")

    b_best_loss_epoch, b_best_loss = _best_min(baseline_epochs, "loss")
    v_best_loss_epoch, v_best_loss = _best_min(variant_epochs, "loss")

    summary: Dict[str, Any] = {
        "baseline_metrics_file": str(baseline_path),
        "variant_metrics_file": str(variant_path),
        "aligned_epoch_count": int(len(common_epochs)),
        "aligned_epoch_min": int(common_epochs[0]),
        "aligned_epoch_max": int(common_epochs[-1]),
        "mean_delta": {
            "loss_variant_minus_baseline": _mean(delta_loss_vals),
            "val_solvability_variant_minus_baseline": _mean(delta_solv_vals),
            "logic_loss_variant_minus_baseline": _mean(delta_logic_vals),
        },
        "last": {
            "baseline": {
                "epoch": int(b_last_epoch),
                "loss": _safe_get(b_last, "loss"),
                "val_solvability": _safe_get(b_last, "val_solvability"),
                "logic_loss": _safe_get(b_last, "logic_loss"),
            },
            "variant": {
                "epoch": int(v_last_epoch),
                "loss": _safe_get(v_last, "loss"),
                "val_solvability": _safe_get(v_last, "val_solvability"),
                "logic_loss": _safe_get(v_last, "logic_loss"),
            },
        },
        "best": {
            "baseline": {
                "best_val_solvability": {"epoch": int(b_best_solv_epoch), "value": float(b_best_solv)},
                "best_loss": {"epoch": int(b_best_loss_epoch), "value": float(b_best_loss)},
            },
            "variant": {
                "best_val_solvability": {"epoch": int(v_best_solv_epoch), "value": float(v_best_solv)},
                "best_loss": {"epoch": int(v_best_loss_epoch), "value": float(v_best_loss)},
            },
        },
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_md = output_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Logic-Loss A/B Comparison",
                "",
                f"- Baseline: {baseline_path}",
                f"- Variant: {variant_path}",
                f"- Aligned epochs: {len(common_epochs)} ({common_epochs[0]}..{common_epochs[-1]})",
                "",
                "## Mean Delta (Variant - Baseline)",
                f"- loss: {summary['mean_delta']['loss_variant_minus_baseline']:.6f}",
                f"- val_solvability: {summary['mean_delta']['val_solvability_variant_minus_baseline']:.6f}",
                f"- logic_loss: {summary['mean_delta']['logic_loss_variant_minus_baseline']:.6f}",
                "",
                "## Best Val Solvability",
                f"- baseline: epoch {b_best_solv_epoch}, value {b_best_solv:.6f}",
                f"- variant: epoch {v_best_solv_epoch}, value {v_best_solv:.6f}",
                "",
                "## Best Loss",
                f"- baseline: epoch {b_best_loss_epoch}, value {b_best_loss:.6f}",
                f"- variant: epoch {v_best_loss_epoch}, value {v_best_loss:.6f}",
                "",
                "Artifacts:",
                f"- {delta_csv}",
                f"- {summary_json}",
                f"- {summary_md}",
            ]
        ),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare logic-loss A/B diffusion runs.")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline metrics JSON file or directory containing MetricsLogger JSON files.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Variant metrics JSON file or directory containing MetricsLogger JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/logic_loss_ab",
        help="Output directory for comparison artifacts.",
    )
    args = parser.parse_args()

    baseline_json = _resolve_metrics_json(args.baseline)
    variant_json = _resolve_metrics_json(args.variant)

    summary = compare_runs(
        baseline_path=baseline_json,
        variant_path=variant_json,
        output_dir=Path(args.output),
    )

    print("Logic-loss A/B comparison complete")
    print(f"Baseline metrics: {baseline_json}")
    print(f"Variant metrics: {variant_json}")
    print(f"Aligned epochs: {summary['aligned_epoch_count']}")
    print(f"Mean delta val_solvability (variant-baseline): {summary['mean_delta']['val_solvability_variant_minus_baseline']:.6f}")
    print(f"Artifacts saved under: {args.output}")


if __name__ == "__main__":
    main()
