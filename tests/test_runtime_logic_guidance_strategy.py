from types import SimpleNamespace

import pytest

from src.core.latent_diffusion import GradientGuidance
from src.pipeline.generation.sampler import _configure_runtime_logic_guidance


class _Pipeline:
    def __init__(self, *, strategy: str, active_fraction: float = 0.2):
        self.logic_net = object()
        self.diffusion = SimpleNamespace(guidance=GradientGuidance(logic_net=None, guidance_scale=0.0))
        self.default_logic_guidance_strategy = strategy
        self.default_logic_guidance_active_fraction = active_fraction
        self.diagnostics = {}

    def _bump_diagnostic(self, key: str) -> None:
        self.diagnostics[key] = self.diagnostics.get(key, 0) + 1


def test_late_runtime_logic_guidance_sets_late_active_window():
    pipeline = _Pipeline(strategy="late", active_fraction=0.25)

    scale = _configure_runtime_logic_guidance(pipeline, 1.5)

    assert scale == pytest.approx(1.5)
    assert pipeline.diffusion.guidance.logic_net is pipeline.logic_net
    assert pipeline.diffusion.guidance.guidance_scale == pytest.approx(1.5)
    assert pipeline.diffusion.guidance.schedule_enabled is True
    assert pipeline.diffusion.guidance.active_fraction == pytest.approx(0.25)
    assert pipeline.diagnostics == {"logic_guidance_late_dpps_used": 1}


def test_full_runtime_logic_guidance_sets_full_active_window():
    pipeline = _Pipeline(strategy="full", active_fraction=0.2)

    scale = _configure_runtime_logic_guidance(pipeline, 1.0)

    assert scale == pytest.approx(1.0)
    assert pipeline.diffusion.guidance.logic_net is pipeline.logic_net
    assert pipeline.diffusion.guidance.active_fraction == pytest.approx(1.0)
    assert pipeline.diagnostics == {"logic_guidance_full_dpps_used": 1}


def test_none_runtime_logic_guidance_disables_logicnet_even_with_positive_scale():
    pipeline = _Pipeline(strategy="none", active_fraction=0.2)

    scale = _configure_runtime_logic_guidance(pipeline, 1.0)

    assert scale == pytest.approx(0.0)
    assert pipeline.diffusion.guidance.logic_net is None
    assert pipeline.diffusion.guidance.guidance_scale == pytest.approx(0.0)
    assert pipeline.diagnostics == {}


def test_runtime_logic_guidance_disables_when_logicnet_missing():
    pipeline = _Pipeline(strategy="late", active_fraction=0.2)
    pipeline.logic_net = None

    scale = _configure_runtime_logic_guidance(pipeline, 1.0)

    assert scale == pytest.approx(0.0)
    assert pipeline.diffusion.guidance.logic_net is None
    assert pipeline.diffusion.guidance.guidance_scale == pytest.approx(0.0)
