"""Regression checks for LogicNet registration inside GradientGuidance."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.latent_diffusion import GradientGuidance
from src.core.logic_net import LogicNet


def _param_ids(module):
    return {id(param) for param in module.parameters()}


def test_gradient_guidance_registers_logicnet_assigned_after_creation():
    guidance = GradientGuidance(logic_net=None)
    logic_net = LogicNet(latent_dim=64, num_classes=44)

    guidance.logic_net = logic_net

    assert _param_ids(logic_net) <= _param_ids(guidance)


def test_gradient_guidance_registers_logicnet_at_creation():
    logic_net = LogicNet(latent_dim=64, num_classes=44)
    guidance = GradientGuidance(logic_net=logic_net)

    assert _param_ids(logic_net) <= _param_ids(guidance)
