"""Regression checks for LogicNet optimizer registration."""

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.latent_diffusion import create_latent_diffusion
from src.core.logic_net import LogicNet


def _param_ids(module):
    return {id(param) for param in module.parameters()}


def _optimizer_param_ids(optimizer):
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


def test_logicnet_params_are_in_standalone_optimizer():
    logic_net = LogicNet(latent_dim=64, num_classes=44)
    optimizer = torch.optim.AdamW(logic_net.parameters(), lr=0.001)

    assert _param_ids(logic_net) <= _optimizer_param_ids(optimizer)


def test_logicnet_params_are_in_diffusion_optimizer_after_assignment():
    logic_net = LogicNet(latent_dim=64, num_classes=44)
    diffusion = create_latent_diffusion(
        latent_dim=64,
        model_channels=128,
        context_dim=256,
    )
    diffusion.guidance.logic_net = logic_net

    optimizer = torch.optim.AdamW(list(diffusion.parameters()), lr=0.001)

    assert _param_ids(logic_net) <= _param_ids(diffusion)
    assert _param_ids(logic_net) <= _optimizer_param_ids(optimizer)
