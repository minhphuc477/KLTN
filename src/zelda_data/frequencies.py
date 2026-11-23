"""Tile class frequency computation for room datasets.

Provides empirical class distribution calculations to support rare-tile loss
weighting, focal loss balancing, and dataset distribution inspection.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from src.core.definitions import SEMANTIC_PALETTE

__all__ = ["compute_room_tile_frequencies"]


def compute_room_tile_frequencies(
    dataset: Any,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Compute empirical tile-class frequencies across a room dataset or subset.

    Returns a float32 tensor of shape ``(num_classes,)`` summing to ``1.0``
    (or all zeros if ``dataset`` is empty). Reads un-normalized semantic IDs
    directly when ``dataset`` wraps a ``ZeldaRoomDataset``.
    """
    resolved_classes = (
        int(num_classes)
        if num_classes is not None
        else max(int(v) for v in SEMANTIC_PALETTE.values()) + 1
    )
    if resolved_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {resolved_classes}.")

    counts = torch.zeros(resolved_classes, dtype=torch.long)

    # Fast path for ZeldaRoomDataset or Subset(ZeldaRoomDataset) to avoid
    # graph/symmetry overhead and normalized float rounding drift.
    raw_rooms = None
    indices: Optional[Iterable[int]] = None
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "rooms"):
        raw_rooms = dataset.dataset.rooms
        indices = dataset.indices
    elif hasattr(dataset, "rooms"):
        raw_rooms = dataset.rooms
        indices = range(len(raw_rooms))

    if raw_rooms is not None and indices is not None:
        for idx in indices:
            arr = np.asarray(raw_rooms[int(idx)])
            flat = torch.as_tensor(arr, dtype=torch.long).reshape(-1)
            valid = flat[(flat >= 0) & (flat < resolved_classes)]
            if valid.numel() > 0:
                counts += torch.bincount(valid, minlength=resolved_classes)
    else:
        for sample in dataset:
            grid = sample[0] if isinstance(sample, (tuple, list)) else sample
            t = torch.as_tensor(grid)
            if t.is_floating_point() and t.numel() > 0 and float(t.max().item()) <= 1.0:
                flat = torch.round(t * 43.0).long().reshape(-1)
            else:
                flat = t.long().reshape(-1)
            valid = flat[(flat >= 0) & (flat < resolved_classes)]
            if valid.numel() > 0:
                counts += torch.bincount(valid, minlength=resolved_classes)

    total = counts.sum()
    if int(total.item()) == 0:
        return torch.zeros(resolved_classes, dtype=torch.float32)
    return counts.to(dtype=torch.float32) / float(total.item())
