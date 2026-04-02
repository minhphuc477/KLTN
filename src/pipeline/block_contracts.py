"""Runtime block I/O contracts for neural-symbolic pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class BlockShapeContract:
    """Expected tensor shape contract for a pipeline stage."""

    name: str
    dims: int
    batch_dim: Optional[int] = None
    channel_dim: Optional[int] = None
    spatial_hw: Optional[Tuple[int, int]] = None


class BlockContractError(ValueError):
    """Raised when a pipeline tensor violates a declared stage contract."""


def validate_tensor_contract(tensor: torch.Tensor, contract: BlockShapeContract) -> None:
    """Validate a tensor against a stage shape contract."""
    if not isinstance(tensor, torch.Tensor):
        raise BlockContractError(f"{contract.name}: expected torch.Tensor, got {type(tensor)!r}")

    if tensor.dim() != contract.dims:
        raise BlockContractError(
            f"{contract.name}: expected {contract.dims} dims, got {tensor.dim()} with shape {tuple(tensor.shape)}"
        )

    if contract.batch_dim is not None and int(tensor.shape[0]) != int(contract.batch_dim):
        raise BlockContractError(
            f"{contract.name}: expected batch={contract.batch_dim}, got {int(tensor.shape[0])}"
        )

    if contract.channel_dim is not None and int(tensor.shape[1]) != int(contract.channel_dim):
        raise BlockContractError(
            f"{contract.name}: expected channels={contract.channel_dim}, got {int(tensor.shape[1])}"
        )

    if contract.spatial_hw is not None:
        h_expected, w_expected = contract.spatial_hw
        h_actual = int(tensor.shape[-2])
        w_actual = int(tensor.shape[-1])
        if h_actual != int(h_expected) or w_actual != int(w_expected):
            raise BlockContractError(
                f"{contract.name}: expected spatial=({h_expected},{w_expected}), got ({h_actual},{w_actual})"
            )


def validate_feature_dims(
    *,
    node_features: Optional[torch.Tensor],
    edge_features: Optional[torch.Tensor],
    expected_node_dim: int,
    expected_edge_dim: int,
) -> None:
    """
    Validate basic graph feature tensor shape sanity before condition encoding.

    Exact width alignment is handled inside `GlobalStreamEncoder`, which
    automatically pads/truncates feature tensors for backward compatibility.
    This helper therefore only rejects obviously malformed tensors, not schema
    width mismatches.
    """
    if isinstance(node_features, torch.Tensor) and node_features.dim() == 2:
        actual = int(node_features.shape[1])
        if actual <= 0 or int(expected_node_dim) <= 0:
            raise BlockContractError(
                f"condition_encoder.node_features: expected positive dim, got expected={expected_node_dim} actual={actual}"
            )

    if isinstance(edge_features, torch.Tensor) and edge_features.dim() == 2:
        actual = int(edge_features.shape[1])
        if actual <= 0 or int(expected_edge_dim) <= 0:
            raise BlockContractError(
                f"condition_encoder.edge_features: expected positive dim, got expected={expected_edge_dim} actual={actual}"
            )


def validate_checkpoint_metadata(
    *,
    metadata: dict,
    model_name: str,
    expected_version: str = "1.0",
    accepted_model_types: Optional[Sequence[str]] = None,
) -> None:
    """Validate lightweight checkpoint metadata schema."""
    fmt = str(metadata.get("format_version", "")).strip()
    if fmt and fmt != expected_version:
        raise BlockContractError(
            f"{model_name}: unsupported checkpoint metadata version {fmt!r} (expected {expected_version!r})"
        )

    declared_model = str(metadata.get("model_type", "")).strip().lower()
    accepted = {
        str(model_name).strip().lower(),
        *(
            str(model_type).strip().lower()
            for model_type in (accepted_model_types or [])
            if str(model_type).strip()
        ),
    }
    if declared_model and declared_model not in accepted:
        raise BlockContractError(
            f"{model_name}: metadata model_type={declared_model!r} does not match loader"
        )


def summarize_missing_keys(keys: Sequence[str], max_items: int = 8) -> str:
    """Short summary string for missing/unexpected key diagnostics."""
    keys = list(keys)
    if not keys:
        return "[]"
    shown = keys[: max(1, int(max_items))]
    suffix = " ..." if len(keys) > len(shown) else ""
    return f"{shown}{suffix}"
