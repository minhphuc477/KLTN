"""Optimizer parameter grouping helpers."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn


def adamw_decay_param_groups(
    named_parameters: Iterable[Tuple[str, nn.Parameter]],
    *,
    weight_decay: float,
    base_name: str = "model",
) -> List[dict]:
    """Split AdamW params into decayed matrices and non-decayed biases/scales.

    AdamW decay on one-dimensional parameters such as bias, LayerNorm, GroupNorm,
    and BatchNorm scale vectors is a standard training foot-gun. This helper
    keeps decay on trainable matrices/convolution kernels and disables it for
    1D parameters.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    seen = set()

    for param_name, param in named_parameters:
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        if param.ndim <= 1 or str(param_name).endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: List[dict] = []
    if decay_params:
        groups.append(
            {
                "name": f"{base_name}_decay",
                "params": decay_params,
                "weight_decay": float(max(0.0, weight_decay)),
            }
        )
    if no_decay_params:
        groups.append(
            {
                "name": f"{base_name}_no_decay",
                "params": no_decay_params,
                "weight_decay": 0.0,
            }
        )
    return groups


def adamw_decay_param_groups_for_modules(
    modules: Iterable[Tuple[str, nn.Module]],
    *,
    weight_decay: float,
) -> List[dict]:
    """Build deduplicated AdamW groups across multiple named modules."""
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    seen = set()

    for module_name, module in modules:
        for param_name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if param.ndim <= 1 or str(param_name).endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    groups: List[dict] = []
    if decay_params:
        groups.append(
            {
                "name": "decay",
                "params": decay_params,
                "weight_decay": float(max(0.0, weight_decay)),
            }
        )
    if no_decay_params:
        groups.append({"name": "no_decay", "params": no_decay_params, "weight_decay": 0.0})
    return groups


__all__ = ["adamw_decay_param_groups", "adamw_decay_param_groups_for_modules"]
