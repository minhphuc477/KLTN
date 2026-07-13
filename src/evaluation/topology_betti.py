"""Lightweight cubical-topology descriptors for room-generation ablations.

This module computes digital Betti curves over superlevel sets. It is an
evaluation metric, not a differentiable persistent-homology loss and not a
replacement for the inventory-aware validator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

DEFAULT_BETTI_THRESHOLDS = tuple(np.linspace(0.1, 0.9, 9).tolist())


@dataclass(frozen=True)
class BettiCurve:
    thresholds: tuple[float, ...]
    beta0: tuple[int, ...]
    beta1: tuple[int, ...]

    @property
    def beta0_auc(self) -> float:
        return float(np.mean(self.beta0)) if self.beta0 else 0.0

    @property
    def beta1_auc(self) -> float:
        return float(np.mean(self.beta1)) if self.beta1 else 0.0


def _component_count(mask: np.ndarray, offsets: Sequence[tuple[int, int]]) -> tuple[int, list[bool]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    boundary_flags: list[bool] = []
    for row, col in np.argwhere(mask):
        row_i, col_i = int(row), int(col)
        if visited[row_i, col_i]:
            continue
        count += 1
        touches_boundary = False
        visited[row_i, col_i] = True
        queue = deque([(row_i, col_i)])
        while queue:
            current_row, current_col = queue.popleft()
            if current_row in {0, height - 1} or current_col in {0, width - 1}:
                touches_boundary = True
            for row_offset, col_offset in offsets:
                next_row = current_row + row_offset
                next_col = current_col + col_offset
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                if visited[next_row, next_col] or not mask[next_row, next_col]:
                    continue
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))
        boundary_flags.append(touches_boundary)
    return count, boundary_flags


def digital_betti_numbers(foreground: np.ndarray) -> tuple[int, int]:
    """Return (beta0, beta1) using a 4/8 digital-topology convention."""
    mask = np.asarray(foreground, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"foreground must be rank-2, got {tuple(mask.shape)}.")
    foreground_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    background_offsets = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    beta0, _ = _component_count(mask, foreground_offsets)
    _, background_boundary_flags = _component_count(~mask, background_offsets)
    beta1 = sum(not touches_boundary for touches_boundary in background_boundary_flags)
    return int(beta0), int(beta1)


def betti_curve(
    walkability_probability: np.ndarray,
    *,
    thresholds: Iterable[float] = DEFAULT_BETTI_THRESHOLDS,
) -> BettiCurve:
    """Compute Betti numbers for probability superlevel sets."""
    probability = np.asarray(walkability_probability, dtype=np.float64)
    if probability.ndim != 2:
        raise ValueError(
            f"walkability_probability must be rank-2, got {tuple(probability.shape)}."
        )
    if not np.isfinite(probability).all():
        raise ValueError("walkability_probability contains non-finite values.")
    normalized_thresholds = tuple(float(value) for value in thresholds)
    if not normalized_thresholds or any(not 0.0 <= value <= 1.0 for value in normalized_thresholds):
        raise ValueError("Betti thresholds must be a non-empty sequence in [0, 1].")
    beta0: list[int] = []
    beta1: list[int] = []
    clipped = np.clip(probability, 0.0, 1.0)
    for threshold in normalized_thresholds:
        components, holes = digital_betti_numbers(clipped >= threshold)
        beta0.append(components)
        beta1.append(holes)
    return BettiCurve(normalized_thresholds, tuple(beta0), tuple(beta1))


def normalized_betti_curve_distance(left: BettiCurve, right: BettiCurve, *, grid_size: int) -> float:
    """Normalized L1 distance between aligned H0/H1 Betti curves."""
    if left.thresholds != right.thresholds:
        raise ValueError("Betti curves must use identical thresholds.")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    delta = np.abs(np.asarray(left.beta0) - np.asarray(right.beta0))
    delta += np.abs(np.asarray(left.beta1) - np.asarray(right.beta1))
    return float(np.mean(delta) / float(grid_size)) if delta.size else 0.0
