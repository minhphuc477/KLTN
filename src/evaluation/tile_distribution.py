"""Tile-pattern distribution metrics for discrete dungeon rooms.

These metrics compare generated categorical grids against VGLC/reference grids
without projecting them through natural-image feature extractors. They are a
discrete analogue to distributional realism checks such as FID: lower
tile-pattern divergence means generated local structure is closer to the
reference corpus.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

Pattern = Tuple[int, ...]


@dataclass(frozen=True)
class TilePatternDistributionResult:
    """Summary of generated/reference local-pattern similarity."""

    pattern_size: int
    generated_unique_patterns: int
    reference_unique_patterns: int
    shared_patterns: int
    generated_total_patterns: int
    reference_total_patterns: int
    js_divergence: float
    kl_generated_to_reference: float
    kl_reference_to_generated: float
    total_variation: float
    pattern_coverage: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "pattern_size": float(self.pattern_size),
            "generated_unique_patterns": float(self.generated_unique_patterns),
            "reference_unique_patterns": float(self.reference_unique_patterns),
            "shared_patterns": float(self.shared_patterns),
            "generated_total_patterns": float(self.generated_total_patterns),
            "reference_total_patterns": float(self.reference_total_patterns),
            "js_divergence": float(self.js_divergence),
            "kl_generated_to_reference": float(self.kl_generated_to_reference),
            "kl_reference_to_generated": float(self.kl_reference_to_generated),
            "total_variation": float(self.total_variation),
            "pattern_coverage": float(self.pattern_coverage),
        }


def _as_2d_grid(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D tile grid, got shape {tuple(arr.shape)}.")
    return arr.astype(np.int64, copy=False)


def iter_tile_patterns(grid: np.ndarray, *, pattern_size: int = 2) -> Iterable[Pattern]:
    """Yield flattened k x k tile patterns from a single room/grid."""
    k = int(pattern_size)
    if k <= 0:
        raise ValueError("pattern_size must be positive.")
    arr = _as_2d_grid(grid)
    h, w = arr.shape
    if h < k or w < k:
        return
    for row in range(0, h - k + 1):
        for col in range(0, w - k + 1):
            yield tuple(int(v) for v in arr[row:row + k, col:col + k].reshape(-1))


def tile_pattern_counts(grids: Sequence[np.ndarray], *, pattern_size: int = 2) -> Counter[Pattern]:
    """Count local tile patterns over a room corpus."""
    counts: Counter[Pattern] = Counter()
    for grid in grids:
        counts.update(iter_tile_patterns(grid, pattern_size=pattern_size))
    return counts


def _probabilities(
    counts: Mapping[Pattern, int],
    support: Sequence[Pattern],
    *,
    smoothing: float,
) -> np.ndarray:
    alpha = float(max(0.0, smoothing))
    values = np.array([float(counts.get(pattern, 0)) + alpha for pattern in support], dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return np.full((len(support),), 1.0 / max(1, len(support)), dtype=np.float64)
    return values / total


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))


def compare_tile_pattern_distributions(
    generated_grids: Sequence[np.ndarray],
    reference_grids: Sequence[np.ndarray],
    *,
    pattern_size: int = 2,
    smoothing: float = 1e-6,
) -> TilePatternDistributionResult:
    """Compare generated and reference grids with local tile-pattern distances."""
    gen_counts = tile_pattern_counts(generated_grids, pattern_size=pattern_size)
    ref_counts = tile_pattern_counts(reference_grids, pattern_size=pattern_size)
    support = sorted(set(gen_counts) | set(ref_counts))
    if not support:
        raise ValueError("No tile patterns could be extracted from generated/reference grids.")

    p = _probabilities(gen_counts, support, smoothing=smoothing)
    q = _probabilities(ref_counts, support, smoothing=smoothing)
    m = 0.5 * (p + q)
    kl_pq = _kl(p, q)
    kl_qp = _kl(q, p)
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    tv = 0.5 * float(np.abs(p - q).sum())
    shared = set(gen_counts) & set(ref_counts)

    return TilePatternDistributionResult(
        pattern_size=int(pattern_size),
        generated_unique_patterns=len(gen_counts),
        reference_unique_patterns=len(ref_counts),
        shared_patterns=len(shared),
        generated_total_patterns=int(sum(gen_counts.values())),
        reference_total_patterns=int(sum(ref_counts.values())),
        js_divergence=float(js),
        kl_generated_to_reference=float(kl_pq),
        kl_reference_to_generated=float(kl_qp),
        total_variation=float(tv),
        pattern_coverage=float(len(shared) / max(1, len(ref_counts))),
    )
