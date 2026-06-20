"""Shared attention kernels for token and graph-grid conditioning."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class HedgehogFeatureMap(nn.Module):
    """
    Trainable Hedgehog feature map for linear attention.

    Uses symmetric positive/negative channel pairs so the linear kernel keeps
    more of the softmax attention geometry than rectified feature maps.
    """

    def __init__(self, num_heads: int, head_dim: int, feature_dim: int = 32):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.feature_dim = int(max(4, feature_dim))
        self.weights = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim, self.feature_dim)
        )
        nn.init.normal_(self.weights, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Map [B, heads, seq, head_dim] -> [B, heads, seq, 2 * feature_dim]."""
        projected = torch.einsum("hdf,bhld->bhlf", self.weights, x)
        projected_fp32 = projected.float()
        mapped = torch.cat(
            [
                torch.softmax(projected_fp32, dim=-1),
                torch.softmax(-projected_fp32, dim=-1),
            ],
            dim=-1,
        )
        return mapped.to(dtype=projected.dtype)


def expand_attention_mask(
    mask: Optional[Tensor],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    """Normalize a token-validity mask to [B, 1, L, 1] for feature-map attention."""
    if mask is None:
        return None
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    if mask.shape[1] > seq_len:
        mask = mask[:, :seq_len]
    elif mask.shape[1] < seq_len:
        mask = torch.nn.functional.pad(mask, (0, seq_len - mask.shape[1]), value=0)
    return mask[:, None, :, None].to(device=device, dtype=dtype)


def hedgehog_linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    q_map: HedgehogFeatureMap,
    k_map: HedgehogFeatureMap,
    token_mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute linear attention with Hedgehog feature maps.

    Args:
        q: [B, heads, Nq, D]
        k: [B, heads, Nk, D]
        v: [B, heads, Nk, D]
        token_mask: Optional [B, Nk] valid-token mask
    """
    output_dtype = v.dtype
    f_q = q_map(q).float()
    f_k = k_map(k).float()
    v_accum = v.float()
    if token_mask is not None:
        expanded = expand_attention_mask(
            token_mask,
            batch_size=int(v.shape[0]),
            seq_len=int(v.shape[2]),
            device=v.device,
            dtype=torch.float32,
        )
        if expanded is not None:
            f_k = f_k * expanded
            v_accum = v_accum * expanded

    kv = torch.einsum("bhlf,bhld->bhfd", f_k, v_accum)
    numer = torch.einsum("bhnf,bhfd->bhnd", f_q, kv)
    k_sum = f_k.sum(dim=2)
    denom = torch.einsum("bhnf,bhf->bhn", f_q, k_sum).unsqueeze(-1).clamp_min(eps)
    return (numer / denom).to(dtype=output_dtype)
