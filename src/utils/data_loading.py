"""Shared DataLoader construction helpers for training scripts."""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy RNGs inside a PyTorch DataLoader worker."""
    _ = worker_id
    worker_seed = int(torch.initial_seed() % 2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_runtime_kwargs(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> Dict[str, Any]:
    """Return safe DataLoader runtime kwargs.

    PyTorch only accepts ``persistent_workers`` and ``prefetch_factor`` when
    worker processes are enabled. Keeping that rule in one place prevents each
    training script from hand-rolling slightly different loader setup.
    """
    workers = int(max(0, num_workers))
    kwargs: Dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(pin_memory),
    }
    if workers > 0:
        kwargs["worker_init_fn"] = seed_dataloader_worker
        kwargs["persistent_workers"] = True if persistent_workers is None else bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(max(1, prefetch_factor))
    return kwargs
