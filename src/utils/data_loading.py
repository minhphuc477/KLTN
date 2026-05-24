"""Shared DataLoader construction helpers for training scripts."""

from __future__ import annotations

from typing import Any, Dict


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
        kwargs["persistent_workers"] = True if persistent_workers is None else bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(max(1, prefetch_factor))
    return kwargs
