"""Tests for deterministic spatial loss and accuracy utilities."""

import pytest
import torch
import torch.nn.functional as F

from src.utils.loss_utils import (
    spatial_classification_accuracy,
    spatial_cross_entropy,
)


def test_spatial_cross_entropy_mathematical_equivalence():
    torch.manual_seed(42)
    B, C, H, W = 4, 44, 16, 11
    logits = torch.randn(B, C, H, W, dtype=torch.float32)
    targets = torch.randint(0, C, (B, H, W), dtype=torch.long)

    # Reduction mean
    expected_mean = F.cross_entropy(logits, targets, reduction="mean")
    actual_mean = spatial_cross_entropy(logits, targets, reduction="mean")
    assert torch.allclose(actual_mean, expected_mean, atol=1e-6)

    # Reduction sum
    expected_sum = F.cross_entropy(logits, targets, reduction="sum")
    actual_sum = spatial_cross_entropy(logits, targets, reduction="sum")
    assert torch.allclose(actual_sum, expected_sum, atol=1e-5)

    # Reduction none
    expected_none = F.cross_entropy(logits, targets, reduction="none")
    actual_none = spatial_cross_entropy(logits, targets, reduction="none")
    assert actual_none.shape == (B, H, W)
    assert torch.allclose(actual_none, expected_none, atol=1e-6)


def test_spatial_cross_entropy_parameters_support():
    torch.manual_seed(123)
    B, C, H, W = 2, 10, 8, 8
    logits = torch.randn(B, C, H, W, dtype=torch.float32)
    targets = torch.randint(0, C, (B, H, W), dtype=torch.long)
    targets[0, 2:4, 2:4] = -100  # ignore index

    weights = torch.rand(C, dtype=torch.float32)

    # With weights and ignore_index
    expected = F.cross_entropy(
        logits, targets, weight=weights, ignore_index=-100, reduction="mean", label_smoothing=0.1
    )
    actual = spatial_cross_entropy(
        logits, targets, weight=weights, ignore_index=-100, reduction="mean", label_smoothing=0.1
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_spatial_cross_entropy_gradient_flow():
    torch.manual_seed(42)
    B, C, H, W = 2, 5, 4, 4
    logits1 = torch.randn(B, C, H, W, requires_grad=True)
    logits2 = logits1.detach().clone().requires_grad_(True)
    targets = torch.randint(0, C, (B, H, W), dtype=torch.long)

    loss1 = F.cross_entropy(logits1, targets)
    loss1.backward()

    loss2 = spatial_cross_entropy(logits2, targets)
    loss2.backward()

    assert torch.allclose(logits1.grad, logits2.grad, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spatial_cross_entropy_deterministic_on_cuda():
    """Verify that spatial_cross_entropy runs without error under strict CUDA determinism."""
    old_det = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        B, C, H, W = 4, 44, 16, 11
        logits = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)
        targets = torch.randint(0, C, (B, H, W), device="cuda", dtype=torch.long)

        # Standard F.cross_entropy on 4D CUDA tensor fails under deterministic algorithms
        with pytest.raises(RuntimeError, match="nll_loss2d_forward_out_cuda_template"):
            _ = F.cross_entropy(logits, targets, reduction="mean")

        # spatial_cross_entropy succeeds without error
        loss_mean = spatial_cross_entropy(logits, targets, reduction="mean")
        assert torch.isfinite(loss_mean)

        loss_sum = spatial_cross_entropy(logits, targets, reduction="sum")
        assert torch.isfinite(loss_sum)

        loss_none = spatial_cross_entropy(logits, targets, reduction="none")
        assert loss_none.shape == (B, H, W)
        assert torch.isfinite(loss_none).all()
    finally:
        torch.use_deterministic_algorithms(old_det)


def test_spatial_classification_accuracy():
    B, C, H, W = 2, 4, 3, 3
    # Craft perfect predictions
    targets = torch.tensor([
        [[0, 1, 2], [3, 0, 1], [2, 3, 0]],
        [[1, 2, 3], [0, 1, 2], [3, 0, 1]],
    ], dtype=torch.long)
    logits = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float() * 10.0

    acc = spatial_classification_accuracy(logits, targets)
    assert acc.item() == pytest.approx(1.0)

    # Corrupt one position
    corrupted_targets = targets.clone()
    corrupted_targets[0, 0, 0] = (corrupted_targets[0, 0, 0] + 1) % C
    acc_corrupted = spatial_classification_accuracy(logits, corrupted_targets)
    total_elements = B * H * W
    assert acc_corrupted.item() == pytest.approx((total_elements - 1) / total_elements)

    # With ignore_index
    targets_with_ignore = targets.clone()
    targets_with_ignore[0, 0, 0] = -100
    acc_ignore = spatial_classification_accuracy(logits, targets_with_ignore, ignore_index=-100)
    assert acc_ignore.item() == pytest.approx(1.0)


def test_spatial_cross_entropy_invalid_reduction():
    logits = torch.randn(2, 3, 4, 4)
    targets = torch.randint(0, 3, (2, 4, 4))
    with pytest.raises(ValueError, match="Invalid reduction mode"):
        spatial_cross_entropy(logits, targets, reduction="invalid")
