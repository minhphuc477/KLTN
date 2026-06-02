#!/usr/bin/env python3
"""Analyze and visualize 2D/4D MAP-Elites or CVT archive diversity."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, is_dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURE_NAMES = ("linearity", "leniency", "progression_complexity", "topology_complexity")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    return vars(value) if hasattr(value, "__dict__") else {}


def _record_from_elite(elite: Any) -> Dict[str, Any] | None:
    payload = _as_mapping(elite)
    features = payload.get("features")
    if features is None:
        metrics = _as_mapping(payload.get("metrics"))
        features = [metrics.get(name) for name in FEATURE_NAMES]
    if not isinstance(features, (list, tuple, np.ndarray)):
        return None
    try:
        clean_features = [float(value) for value in features]
    except (TypeError, ValueError):
        return None
    if not clean_features or not all(np.isfinite(clean_features)):
        return None
    return {
        "features": clean_features,
        "fitness": float(payload.get("fitness", payload.get("score", 0.0)) or 0.0),
        "cell": payload.get("cell"),
    }


def _records_from_runtime_grid(grid: Mapping[Any, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for cell, entry in grid.items():
        payload = dict(_as_mapping(entry))
        payload["cell"] = payload.get("cell", cell)
        record = _record_from_elite(payload)
        if record is not None:
            records.append(record)
    return records


def _records_from_archive_object(archive: Any) -> List[Dict[str, Any]]:
    payload = _as_mapping(archive)
    inner = payload.get("archive")
    if isinstance(inner, Mapping):
        records = []
        for cell, elite in inner.items():
            elite_payload = dict(_as_mapping(elite))
            elite_payload["cell"] = elite_payload.get("cell", cell)
            record = _record_from_elite(elite_payload)
            if record is not None:
                records.append(record)
        return records
    return []


def extract_archive_records(payload: Any) -> List[Dict[str, Any]]:
    """Normalize supported runtime, CVT, and JSON archive layouts."""
    if isinstance(payload, list):
        return [record for item in payload if (record := _record_from_elite(item)) is not None]
    mapping = _as_mapping(payload)
    advanced = mapping.get("advanced_archive")
    records = _records_from_archive_object(advanced)
    if records:
        return records
    if isinstance(mapping.get("elites"), list):
        return extract_archive_records(mapping["elites"])
    archive_records = _records_from_archive_object(mapping)
    if archive_records:
        return archive_records
    grid = mapping.get("grid")
    if isinstance(grid, Mapping):
        return _records_from_runtime_grid(grid)
    return []


def load_archive_payload(path: Path, *, trust_pickle: bool = False) -> Any:
    """Load JSON by default; require an explicit flag for unsafe pickle input."""
    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return json.loads(path.read_text(encoding="utf-8"))
    if not trust_pickle:
        raise ValueError("Pickle input can execute code. Re-run with --trust-pickle for a trusted local archive.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def analyze_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("Archive contains no readable elites.")
    features = np.asarray([record["features"] for record in records], dtype=np.float64)
    fitness = np.asarray([record.get("fitness", 0.0) for record in records], dtype=np.float64)
    dim = int(features.shape[1])
    names = list(FEATURE_NAMES[:dim]) if dim <= len(FEATURE_NAMES) else [f"feature_{i}" for i in range(dim)]
    return {
        "num_elites": int(features.shape[0]),
        "feature_dims": dim,
        "feature_names": names,
        "fitness": {
            "mean": float(np.mean(fitness)),
            "min": float(np.min(fitness)),
            "max": float(np.max(fitness)),
            "sum": float(np.sum(fitness)),
        },
        "features": {
            name: {
                "mean": float(np.mean(features[:, idx])),
                "std": float(np.std(features[:, idx])),
                "min": float(np.min(features[:, idx])),
                "max": float(np.max(features[:, idx])),
            }
            for idx, name in enumerate(names)
        },
        "mean_feature_variance": float(np.var(features, axis=0).mean()),
        "covariance": np.cov(features, rowvar=False).tolist() if len(records) > 1 else [],
    }


def render_pairwise_heatmaps(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    bins: int = 16,
) -> None:
    features = np.asarray([record["features"] for record in records], dtype=np.float64)
    dim = int(features.shape[1])
    names = list(FEATURE_NAMES[:dim]) if dim <= len(FEATURE_NAMES) else [f"feature_{i}" for i in range(dim)]
    pairs = list(combinations(range(dim), 2)) or [(0, 0)]
    cols = min(3, len(pairs))
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.1 * cols, 4.2 * rows), squeeze=False)
    for ax, (left, right) in zip(axes.flat, pairs):
        if left == right:
            counts, edges = np.histogram(features[:, left], bins=bins, range=(0.0, 1.0))
            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
        else:
            counts, x_edges, y_edges = np.histogram2d(
                features[:, left],
                features[:, right],
                bins=bins,
                range=((0.0, 1.0), (0.0, 1.0)),
            )
            image = ax.imshow(
                counts.T,
                origin="lower",
                extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                aspect="auto",
                cmap="viridis",
            )
            fig.colorbar(image, ax=ax, label="occupied elites")
        ax.set_xlabel(names[left])
        ax.set_ylabel(names[right] if left != right else "count")
        ax.set_title(f"{names[left]} vs {names[right]}" if left != right else names[left])
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle("QD archive pairwise behavioral-space occupancy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "qd_archive_analysis")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--trust-pickle", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_archive_payload(args.archive, trust_pickle=bool(args.trust_pickle))
    records = extract_archive_records(payload)
    summary = analyze_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.output_dir / "qd_archive_pairwise_heatmaps.png"
    render_pairwise_heatmaps(records, heatmap_path, bins=max(2, int(args.bins)))
    report = {
        "source_archive": str(args.archive),
        "heatmap": str(heatmap_path),
        **summary,
    }
    report_path = args.output_dir / "qd_archive_diversity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "heatmap": str(heatmap_path), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
