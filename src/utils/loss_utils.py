"""Loss and classification utilities hardened for deterministic execution."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def spatial_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with guaranteed deterministic execution on CUDA.

    In PyTorch CUDA implementations, standard `F.cross_entropy` on 4D spatial tensors
    `[Batch, Channels, Height, Width]` invokes `nll_loss2d_forward_out_cuda_template`,
    which lacks a deterministic implementation when reduction is 'mean' or 'sum'.
    Under `torch.use_deterministic_algorithms(True)`, this raises a RuntimeError.

    This function safely permutes and flattens spatial dimensions to 2D `[N, Channels]`
    with 1D targets `[N]`, routing computation through `nll_loss_forward_out_cuda_template`,
    which is fully deterministic across all reduction modes while preserving exact
    mathematical equivalence.

    Args:
        logits: Unnormalized class scores of shape `[B, C, H, W]`, `[B, C]`, or `[N, C]`.
        targets: Ground-truth class indices of shape `[B, H, W]` or `[B]`.
        weight: Optional manual rescaling weight given to each class `[C]`.
        ignore_index: Target value that is ignored and does not contribute to gradient.
        reduction: Reduction mode: 'none', 'mean', or 'sum'.
        label_smoothing: Amount of smoothing in [0.0, 1.0].

    Returns:
        Loss tensor with reduction applied (or `[B, H, W]` when reduction='none' on 4D inputs).
    """
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Invalid reduction mode: {reduction!r}. Expected 'mean', 'sum', or 'none'.")

    # If already 1D/2D classification logits [B, C], standard F.cross_entropy is already deterministic.
    if logits.ndim <= 2:
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    # For 4D spatial tensors [B, C, H, W] (or any D >= 3 spatial tensors [B, C, d1, d2, ...]):
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    spatial_shape = logits.shape[2:]

    # Permute channels to the trailing dimension: [B, H, W, C] -> [N, C]
    permute_dims = (0,) + tuple(range(2, logits.ndim)) + (1,)
    flat_logits = logits.permute(permute_dims).reshape(-1, num_classes)
    flat_targets = targets.reshape(-1)

    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    if reduction == "none":
        # Reshape back to original spatial shape [B, H, W, ...]
        return loss.reshape(batch_size, *spatial_shape)

    return loss


def spatial_classification_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute top-1 classification accuracy across spatial dimensions.

    Args:
        logits: Unnormalized class scores of shape `[B, C, H, W]` or `[B, C]`.
        targets: Ground-truth class indices of shape `[B, H, W]` or `[B]`.
        ignore_index: Optional class index to mask out from accuracy calculation.

    Returns:
        Scalar float tensor representing accuracy in [0.0, 1.0].
    """
    channel_dim = 1
    predictions = logits.argmax(dim=channel_dim)

    if ignore_index is not None:
        mask = targets != int(ignore_index)
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        correct = (predictions == targets) & mask
        return correct.sum().float() / mask.sum().float()

    return (predictions == targets).float().mean()
