r"""
Fairness / bias assessment utilities for generated Zelda dungeons.

This module provides light-weight, reproducible measures comparing generated
maps against a reference set (training data or held-out maps). Current
metrics:
- per-tile frequency distribution comparison (Jensen-Shannon divergence)
- L1 and L2 differences per tile

CLI usage:
python -m src.evaluation.fairness_assessment --generated-dir outputs/generated --reference-dir Data/The\ Legend\ of\ Zelda/processed_npy --output report.json

The script purposely avoids heavy external dependencies and is safe to run
in CI as a lightweight check over a small number of samples.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_maps_from_dir(directory: Path, *, extensions: Tuple[str, ...] = (".npy",), max_samples: Optional[int] = None) -> List[np.ndarray]:
    """Load semantic tile grids from `directory` by supported extensions.

    Only simple formats (.npy) are supported by default. Returns a list of
    2D numpy arrays (H, W) containing integer semantic IDs.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Generated/reference maps directory not found: {directory}")

    paths: List[Path] = []
    for ext in extensions:
        paths.extend(sorted(directory.glob(f"*{ext}")))
    if not paths:
        raise RuntimeError(f"No maps found in {directory} with extensions={extensions}")

    if max_samples is not None:
        paths = paths[: int(max_samples)]

    maps: List[np.ndarray] = []
    for p in paths:
        try:
            arr = np.load(p)
        except Exception as e:
            logger.warning("Skipping %s: failed to load (%s)", p, e)
            continue
        if arr.ndim == 3 and arr.shape[0] in (1,):
            arr = arr.squeeze(0)
        if arr.ndim != 2:
            logger.warning("Skipping %s: expected 2D semantic grid, got shape=%s", p, arr.shape)
            continue
        maps.append(arr.astype(np.int64, copy=False))
    if not maps:
        raise RuntimeError(f"No valid maps loaded from {directory}")
    return maps


def compute_tile_distribution(maps: Iterable[np.ndarray], num_classes: int) -> np.ndarray:
    """Compute normalized tile-frequency distribution over `num_classes`.

    Returns a 1D numpy array of length `num_classes` summing to 1.0.
    """
    counts = np.zeros(int(num_classes), dtype=np.float64)
    total = 0
    for arr in maps:
        flat = arr.ravel().astype(np.int64, copy=False)
        valid_mask = (flat >= 0) & (flat < int(num_classes))
        flat = flat[valid_mask]
        if flat.size == 0:
            continue
        hist = np.bincount(flat, minlength=int(num_classes)).astype(np.float64)
        counts += hist
        total += flat.size
    if total <= 0:
        raise ValueError("No tiles found in provided maps")
    probs = counts / float(total)
    return probs


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two discrete distributions.

    Uses natural logarithm. Returns a float >= 0.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("Distribution shapes must match for JSD")
    m = 0.5 * (p + q)
    return 0.5 * (_kl_divergence(p, m) + _kl_divergence(q, m))


def compare_distributions(generated: np.ndarray, reference: np.ndarray) -> Dict[str, object]:
    """Compare two tile distributions and return summary metrics."""
    if generated.shape != reference.shape:
        raise ValueError("generated and reference must have same shape")
    jsd = jensen_shannon_divergence(generated, reference)
    l1 = float(np.linalg.norm(generated - reference, ord=1))
    l2 = float(np.linalg.norm(generated - reference, ord=2))
    per_tile_ratio = (generated + 1e-12) / (reference + 1e-12)
    return {
        "jsd": jsd,
        "l1": l1,
        "l2": l2,
        "per_tile_ratio": per_tile_ratio.tolist(),
        "generated": generated.tolist(),
        "reference": reference.tolist(),
    }


def distribution_entropy(distribution: np.ndarray) -> float:
    """Compute Shannon entropy over a discrete distribution."""
    probs = np.asarray(distribution, dtype=np.float64)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def summarize_distribution(distribution: np.ndarray) -> Dict[str, float]:
    """Return small, reproducible summary statistics for one tile distribution."""
    probs = np.asarray(distribution, dtype=np.float64)
    active = int(np.count_nonzero(probs > 0.0))
    entropy = float(distribution_entropy(probs))
    max_share = float(np.max(probs)) if probs.size > 0 else 0.0
    return {
        "entropy": entropy,
        "active_class_count": float(active),
        "max_tile_share": max_share,
    }


def run_fairness_assessment(
    generated_dir: Path,
    reference_dir: Optional[Path],
    *,
    num_classes: int = 44,
    max_samples: Optional[int] = 100,
) -> Dict[str, object]:
    """Run fairness check and return a JSON-serializable report.

    If `reference_dir` is None, the function will only return the generated
    distribution (no comparison) to allow lightweight usage.
    """
    generated_maps = load_maps_from_dir(Path(generated_dir), max_samples=max_samples)
    gen_dist = compute_tile_distribution(generated_maps, num_classes=num_classes)
    invalid_generated = int(sum(int(np.size(arr)) - int(np.count_nonzero((arr >= 0) & (arr < int(num_classes)))) for arr in generated_maps))

    report: Dict[str, object] = {"generated_count": len(generated_maps), "num_classes": int(num_classes)}
    report["generated_distribution"] = gen_dist.tolist()
    report["generated_invalid_tile_count"] = invalid_generated
    report.update({f"generated_{k}": v for k, v in summarize_distribution(gen_dist).items()})

    if reference_dir is not None:
        reference_maps = load_maps_from_dir(Path(reference_dir), max_samples=max_samples)
        ref_dist = compute_tile_distribution(reference_maps, num_classes=num_classes)
        invalid_reference = int(sum(int(np.size(arr)) - int(np.count_nonzero((arr >= 0) & (arr < int(num_classes)))) for arr in reference_maps))
        comp = compare_distributions(gen_dist, ref_dist)
        report["reference_count"] = len(reference_maps)
        report["reference_distribution"] = ref_dist.tolist()
        report["reference_invalid_tile_count"] = invalid_reference
        report.update({f"reference_{k}": v for k, v in summarize_distribution(ref_dist).items()})
        report.update(comp)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fairness / bias assessment for generated maps")
    parser.add_argument("--generated-dir", type=Path, required=True, help="Directory with generated .npy maps")
    parser.add_argument("--reference-dir", type=Path, default=None, help="Directory with reference .npy maps (optional)")
    parser.add_argument("--num-classes", type=int, default=44, help="Number of semantic tile classes")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples to load from each dir")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "fairness_report.json", help="Output JSON report path")

    args = parser.parse_args(argv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    report = run_fairness_assessment(args.generated_dir, args.reference_dir, num_classes=int(args.num_classes), max_samples=(int(args.max_samples) if args.max_samples else None))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote fairness report to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
