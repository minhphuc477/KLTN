#!/usr/bin/env python
"""Focused validation tests for LogicNet guidance fixes."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.latent_diffusion import GradientGuidance
from src.core.logic_net import LogicNet
from src.pipeline.graph_features import extract_node_feature_vector
from src.pipeline.spatial_utils import parse_label_tokens


def test_logicnet_gradient_magnitude():
    """LogicNet should produce a nonzero latent gradient with dungeon node routing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logic_net = LogicNet(
        latent_dim=64,
        num_tile_classes=5,
    ).to(device)
    logic_net.eval()

    b, c, h, w = 4, 64, 32, 32
    z = torch.randn(b, c, h, w, device=device, dtype=torch.float32, requires_grad=True)

    graph_data = {
        "node_features": torch.randn(10, 16, device=device, dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], device=device, dtype=torch.long),
        "edge_features": torch.randn(3, 8, device=device, dtype=torch.float32),
        "graph_scope": "dungeon",
        "current_node_idx": torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long),
        "start_idx": 0,
        "target_idx": 9,
    }

    with torch.enable_grad():
        loss = logic_net(z, graph_data=graph_data)
        loss_scalar = loss[0].mean() if isinstance(loss, tuple) else loss.mean()

    assert loss_scalar.requires_grad
    loss_scalar.backward()
    grad_norm = z.grad.norm().item() if z.grad is not None else 0.0
    assert grad_norm > 1e-6


def test_logicnet_guidance_schedule_is_late_process_active():
    """LogicNet is a clean-latent solver, so scheduled guidance should be strongest near t=0."""
    guidance = GradientGuidance(logic_net=None, guidance_scale=1.0, active_fraction=0.3)

    assert guidance._scheduled_scale(t=999, num_timesteps=1000) == 0.0
    assert guidance._scheduled_scale(t=700, num_timesteps=1000) == 0.0
    assert guidance._scheduled_scale(t=100, num_timesteps=1000) > 0.0
    assert guidance._scheduled_scale(t=0, num_timesteps=1000) == pytest.approx(1.0)


def test_graph_feature_roles_include_type_fields():
    """Mission graphs that use type/room_type START/GOAL should still feed LogicNet roles."""

    def _coerce_bool(value):
        return bool(value)

    def _coerce_difficulty(_value):
        return 0.5

    goal_features = extract_node_feature_vector(
        {"room_type": "goal"},
        node_dim=14,
        device=torch.device("cpu"),
        parse_label_tokens=parse_label_tokens,
        coerce_bool=_coerce_bool,
        coerce_difficulty=_coerce_difficulty,
    )
    start_features = extract_node_feature_vector(
        {"type": "START"},
        node_dim=14,
        device=torch.device("cpu"),
        parse_label_tokens=parse_label_tokens,
        coerce_bool=_coerce_bool,
        coerce_difficulty=_coerce_difficulty,
    )

    assert float(goal_features[3].item()) == 1.0
    assert float(start_features[11].item()) == 1.0


if __name__ == "__main__":
    try:
        test_logicnet_gradient_magnitude()
        test_logicnet_guidance_schedule_is_late_process_active()
        test_graph_feature_roles_include_type_fields()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
