"""Feasibility probes for LogicNet-guided diffusion sampling."""

import pytest
import torch

from scripts.gradient_probe import (
    NoiseGradientStats,
    _extract_logicnet_state,
    _infer_logicnet_architecture,
    probe_logicnet_noisy_gradients,
    recommend_guidance_window,
)
from src.core.logic_net import LogicNet


def test_noisy_logicnet_gradient_probe_records_finite_statistics():
    logic_net = LogicNet(
        latent_dim=44,
        hidden_dim=16,
        num_classes=44,
        num_iterations=3,
        grid_pathfinder_type="bellman_ford",
    )

    rows = probe_logicnet_noisy_gradients(
        logic_net,
        noise_levels=[0.0, 0.5, 1.0],
        samples_per_level=2,
        seed=123,
        device=torch.device("cpu"),
    )

    assert [row.noise_level for row in rows] == [0.0, 0.5, 1.0]
    for row in rows:
        assert row.finite_rate == pytest.approx(1.0)
        assert row.grad_norm_mean > 0.0
        assert row.grad_abs_mean > 0.0
        assert torch.isfinite(torch.tensor(row.loss_mean))
        assert torch.isfinite(torch.tensor(row.score_mean))
        assert 0.0 <= row.walkability_mean <= 1.0


def test_guidance_window_recommends_late_stage_when_high_noise_gradients_collapse():
    rows = [
        NoiseGradientStats(
            noise_level=0.0,
            score_mean=0.9,
            loss_mean=0.1,
            grad_norm_mean=1.0,
            grad_norm_std=0.0,
            grad_abs_mean=0.1,
            finite_rate=1.0,
            relative_grad_norm=1.0,
            walkability_mean=0.5,
            walkability_std=0.1,
        ),
        NoiseGradientStats(
            noise_level=1.0,
            score_mean=0.5,
            loss_mean=0.5,
            grad_norm_mean=1e-8,
            grad_norm_std=0.0,
            grad_abs_mean=1e-9,
            finite_rate=1.0,
            relative_grad_norm=1e-8,
            walkability_mean=0.5,
            walkability_std=0.1,
        ),
    ]

    recommendation = recommend_guidance_window(rows)

    assert recommendation["strategy"] == "late"
    assert recommendation["max_stable_noise"] == pytest.approx(0.0)


def test_gradient_probe_extracts_checkpoint_architecture():
    logic_net = LogicNet(
        latent_dim=12,
        hidden_dim=24,
        num_classes=44,
        num_iterations=2,
        grid_pathfinder_type="bellman_ford",
    )
    checkpoint = {"logic_net_state_dict": logic_net.state_dict()}

    state = _extract_logicnet_state(checkpoint)
    inferred = _infer_logicnet_architecture(state)

    assert inferred["latent_dim"] == 12
    assert inferred["hidden_dim"] == 24
    assert inferred["num_classes"] == 44
