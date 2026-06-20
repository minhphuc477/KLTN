from __future__ import annotations

import torch

from src.core.attention_kernels import HedgehogFeatureMap, hedgehog_linear_attention


def test_hedgehog_linear_attention_accumulates_long_half_sequences_in_fp32():
    q_map = HedgehogFeatureMap(num_heads=2, head_dim=4, feature_dim=8).half()
    k_map = HedgehogFeatureMap(num_heads=2, head_dim=4, feature_dim=8).half()
    q = torch.zeros(1, 2, 4, 4, dtype=torch.float16)
    k = torch.zeros(1, 2, 128, 4, dtype=torch.float16)
    v = torch.full((1, 2, 128, 4), 60000.0, dtype=torch.float16)

    output = hedgehog_linear_attention(q, k, v, q_map=q_map, k_map=k_map)

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.allclose(output.float(), torch.full_like(output.float(), 60000.0), rtol=1e-3, atol=1.0)


def test_hedgehog_linear_attention_all_masked_context_is_finite_zero():
    q_map = HedgehogFeatureMap(num_heads=1, head_dim=4, feature_dim=8)
    k_map = HedgehogFeatureMap(num_heads=1, head_dim=4, feature_dim=8)
    q = torch.randn(2, 1, 3, 4)
    k = torch.randn(2, 1, 5, 4)
    v = torch.randn(2, 1, 5, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)

    output = hedgehog_linear_attention(q, k, v, q_map=q_map, k_map=k_map, token_mask=mask)

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0
