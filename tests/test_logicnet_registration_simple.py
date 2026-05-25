"""Regression checks for LogicNet runtime wiring inside GradientGuidance."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.latent_diffusion import GradientGuidance
from src.core.logic_net import LogicNet


def _param_ids(module):
    return {id(param) for param in module.parameters()}


def test_gradient_guidance_keeps_assigned_logicnet_out_of_state_dict():
    guidance = GradientGuidance(logic_net=None)
    logic_net = LogicNet(latent_dim=64, num_classes=44)

    guidance.logic_net = logic_net

    assert guidance.logic_net is logic_net
    assert _param_ids(logic_net).isdisjoint(_param_ids(guidance))
    assert not any("logic_net" in key for key in guidance.state_dict())


def test_gradient_guidance_keeps_constructor_logicnet_out_of_state_dict():
    logic_net = LogicNet(latent_dim=64, num_classes=44)
    guidance = GradientGuidance(logic_net=logic_net)

    assert guidance.logic_net is logic_net
    assert _param_ids(logic_net).isdisjoint(_param_ids(guidance))
    assert not any("logic_net" in key for key in guidance.state_dict())
