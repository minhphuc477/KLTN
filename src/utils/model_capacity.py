"""
Small-data model-capacity guardrails for training entrypoints.

These warnings do not block execution. They surface audit-backed situations
where the configured model is likely oversized relative to the available
dataset, which is especially relevant for the repository's small Zelda corpus.
"""

from __future__ import annotations

import logging
from typing import Mapping

import torch.nn as nn


SMALL_DATASET_WARN_SAMPLES = 1_000
VERY_SMALL_DATASET_WARN_SAMPLES = 500
LARGE_MODEL_WARN_PARAMS = 50_000_000
VERY_LARGE_MODEL_WARN_PARAMS = 100_000_000
PARAMS_PER_SAMPLE_WARN = 50_000.0
PARAMS_PER_SAMPLE_HIGH = 100_000.0


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count module parameters."""
    return sum(
        int(param.numel())
        for param in module.parameters()
        if not trainable_only or bool(param.requires_grad)
    )


def format_parameter_count(count: int) -> str:
    """Render large parameter counts in a readable short form."""
    value = int(max(0, count))
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def log_capacity_guardrails(
    logger: logging.Logger,
    *,
    stage_name: str,
    dataset_size: int,
    param_groups: Mapping[str, int],
    recommended_config: str = "configs/zelda_hmolqd.yaml",
    capacity_knobs: str,
) -> None:
    """
    Log trainable parameter totals and warn when the model looks oversized
    relative to the available data.
    """
    resolved_dataset_size = int(max(0, dataset_size))
    groups = {str(name): int(max(0, count)) for name, count in param_groups.items()}
    total_trainable = sum(groups.values())
    params_per_sample = (
        float(total_trainable) / float(resolved_dataset_size)
        if resolved_dataset_size > 0
        else None
    )

    group_summary = ", ".join(
        f"{name}={format_parameter_count(count)}"
        for name, count in groups.items()
    ) or "none"
    if params_per_sample is None:
        logger.info(
            "%s trainable parameters: total=%s (%s); dataset_size unavailable.",
            stage_name,
            format_parameter_count(total_trainable),
            group_summary,
        )
        return

    logger.info(
        "%s trainable parameters: total=%s (%s); dataset_size=%d; params_per_sample=%.0f.",
        stage_name,
        format_parameter_count(total_trainable),
        group_summary,
        resolved_dataset_size,
        params_per_sample,
    )

    if (
        resolved_dataset_size < SMALL_DATASET_WARN_SAMPLES
        and total_trainable >= LARGE_MODEL_WARN_PARAMS
    ):
        logger.warning(
            "%s is training %s parameters on only %d samples. "
            "This repository's audit flagged that combination as high risk for overfitting "
            "and unstable conclusions on the Zelda corpus. Prefer the reduced canonical "
            "profile in %s or lower %s.",
            stage_name,
            format_parameter_count(total_trainable),
            resolved_dataset_size,
            recommended_config,
            capacity_knobs,
        )

    if (
        resolved_dataset_size <= VERY_SMALL_DATASET_WARN_SAMPLES
        and (
            total_trainable >= VERY_LARGE_MODEL_WARN_PARAMS
            or params_per_sample >= PARAMS_PER_SAMPLE_HIGH
        )
    ):
        logger.warning(
            "%s is deep into the small-data danger zone: %.0f trainable parameters per sample. "
            "Treat reported gains carefully and prioritize ablations plus reduced-capacity runs.",
            stage_name,
            params_per_sample,
        )
    elif (
        resolved_dataset_size < SMALL_DATASET_WARN_SAMPLES
        and params_per_sample >= PARAMS_PER_SAMPLE_WARN
    ):
        logger.warning(
            "%s currently allocates %.0f trainable parameters per sample. "
            "That is acceptable for experimentation, but it is a reproducibility risk on a tiny corpus.",
            stage_name,
            params_per_sample,
        )
