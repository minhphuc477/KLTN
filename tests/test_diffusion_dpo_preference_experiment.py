from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import experiments.diffusion_dpo_preference as dpo_exp


class _DummyDPOModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, preferred, rejected, context, **kwargs):
        _ = (preferred, rejected, context, kwargs)
        loss = self.weight * 0.5
        return loss, {
            "dpo_margin": self.weight.detach() * 0.25,
            "dpo_accuracy": torch.ones((), dtype=torch.float32),
        }


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint="dummy.pt",
        reference_checkpoint=None,
        preference_pairs="pairs.pt",
        output_checkpoint=None,
        device="cpu",
        steps=1,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        learning_rate=1e-1,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        beta=0.1,
        log_every=100,
        denoiser_backbone="unet",
        latent_dim=4,
        model_channels=8,
        context_dim=8,
        num_heads=2,
        dit_depth=1,
        dit_patch_size=1,
        num_timesteps=8,
        prediction_type="epsilon",
    )


def _payload(device: torch.device):
    return {
        "preferred": torch.zeros(1, 4, 2, 2, device=device),
        "rejected": torch.ones(1, 4, 2, 2, device=device),
        "context": torch.zeros(1, 8, device=device),
    }


def test_dpo_preference_script_skips_nonfinite_gradients_without_optimizer_step(monkeypatch):
    model = _DummyDPOModel()
    initial_weight = model.weight.detach().clone()
    monkeypatch.setattr(dpo_exp, "Accelerator", None)
    monkeypatch.setattr(dpo_exp, "_load_model_from_checkpoint", lambda path, config: model)
    monkeypatch.setattr(dpo_exp, "_load_preference_payload", lambda path, device: _payload(device))
    monkeypatch.setattr(dpo_exp, "_gradients_are_finite", lambda parameters: False)

    metrics = dpo_exp.run_dpo(_args())

    assert metrics["skipped_nonfinite_batch"] == pytest.approx(1.0)
    assert torch.allclose(model.weight.detach(), initial_weight)


def test_dpo_preference_script_skips_nonfinite_clipped_grad_norm_without_optimizer_step(monkeypatch):
    model = _DummyDPOModel()
    initial_weight = model.weight.detach().clone()
    monkeypatch.setattr(dpo_exp, "Accelerator", None)
    monkeypatch.setattr(dpo_exp, "_load_model_from_checkpoint", lambda path, config: model)
    monkeypatch.setattr(dpo_exp, "_load_preference_payload", lambda path, device: _payload(device))
    monkeypatch.setattr(dpo_exp, "_gradients_are_finite", lambda parameters: True)
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(float("inf")),
    )

    metrics = dpo_exp.run_dpo(_args())

    assert metrics["skipped_nonfinite_batch"] == pytest.approx(1.0)
    assert torch.allclose(model.weight.detach(), initial_weight)
