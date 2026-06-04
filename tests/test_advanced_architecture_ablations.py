import math

import numpy as np
import pytest
import torch

from src.config_system import merge_config
from src.core.latent_diffusion import create_latent_diffusion
from src.evaluation.perturb_and_map import perturb_and_map_reachability
from src.train_diffusion import (
    DiffusionTrainingConfig,
    diffusion_training_kwargs_from_resolved_config,
)


def test_flow_matching_loss_is_finite_and_backpropagates():
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=4,
    )
    model.train()
    z_0 = torch.randn(2, 8, 3, 4, requires_grad=True)
    context = torch.randn(2, 16)

    loss = model.flow_matching_loss(z_0, context)
    assert loss.shape == ()
    assert torch.isfinite(loss)

    loss.backward()
    grad_norm = sum(
        float(param.grad.detach().abs().sum().item())
        for param in model.parameters()
        if param.grad is not None
    )
    assert grad_norm > 0.0
    assert z_0.grad is not None
    assert torch.isfinite(z_0.grad).all()


def test_flow_matching_loss_ignores_ddpm_min_snr_weighting(monkeypatch):
    class _ZeroDenoiser(torch.nn.Module):
        def forward(self, x, t, context, **_kwargs):
            _ = (t, context)
            return torch.zeros_like(x)

    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=8,
        context_dim=8,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        min_snr_gamma=5.0,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=2,
    )
    model.denoiser = _ZeroDenoiser()

    def _fixed_rand(*shape, **kwargs):
        assert shape == (2,)
        return torch.tensor([0.01, 0.9], device=kwargs.get("device"), dtype=kwargs.get("dtype"))

    monkeypatch.setattr(torch, "rand", _fixed_rand)
    z_0 = torch.zeros(2, 4, 2, 2, requires_grad=True)
    noise = torch.ones_like(z_0)
    context = torch.zeros(2, 8)

    loss = model.flow_matching_loss(z_0, context, noise=noise)

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(1.0)


def test_dit_backbone_flow_matching_loss_is_finite_and_backpropagates():
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        denoiser_backbone="dit",
        dit_depth=2,
        dit_patch_size=1,
        dit_mlp_ratio=2.0,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        unet_num_heads=4,
        unet_dropout=0.0,
    )
    model.eval()
    assert model.denoiser_backbone == "dit"
    model.train()
    z_0 = torch.randn(2, 8, 3, 4, requires_grad=True)
    context = torch.randn(2, 5, 16)

    loss = model.flow_matching_loss(z_0, context, graph_data={"node_mask": torch.ones(2, 5, dtype=torch.bool)})
    assert loss.shape == ()
    assert torch.isfinite(loss)

    loss.backward()
    grad_norm = sum(
        float(param.grad.detach().abs().sum().item())
        for param in model.parameters()
        if param.grad is not None
    )
    assert grad_norm > 0.0
    assert z_0.grad is not None
    assert torch.isfinite(z_0.grad).all()


def test_dit_backbone_uses_context_edge_index_for_token_topology():
    torch.manual_seed(7)
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        denoiser_backbone="dit",
        topology_refinement_mode="lightweight",
        dit_depth=1,
        dit_patch_size=1,
        dit_mlp_ratio=2.0,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        unet_num_heads=4,
        unet_dropout=0.0,
    )
    model.eval()
    x_t = torch.randn(1, 8, 3, 4)
    t = torch.tensor([3], dtype=torch.long)
    context = torch.randn(1, 3, 16)
    node_mask = torch.ones(1, 3, dtype=torch.bool)
    edge_chain = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_skip = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)

    with torch.no_grad():
        out_chain = model.denoiser(
            x_t,
            t,
            context,
            context_edge_index=edge_chain,
            context_node_mask=node_mask,
        )
        out_skip = model.denoiser(
            x_t,
            t,
            context,
            context_edge_index=edge_skip,
            context_node_mask=node_mask,
        )

    assert tuple(out_chain.shape) == tuple(x_t.shape)
    assert not torch.allclose(out_chain, out_skip)


def test_dit_backbone_uses_spatial_graph_data():
    torch.manual_seed(11)
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        denoiser_backbone="dit",
        dit_depth=1,
        dit_patch_size=1,
        dit_mlp_ratio=2.0,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        unet_num_heads=4,
        unet_dropout=0.0,
        room_topology_channels=18,
    )
    model.eval()
    x_t = torch.randn(1, 8, 3, 4)
    t = torch.tensor([2], dtype=torch.long)
    context = torch.randn(1, 3, 16)
    base_spatial = {
        "graph_nodes": context,
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "node_positions": torch.zeros(1, 3, 2),
        "node_tpe": torch.zeros(1, 3, 8),
        "current_node_distance": torch.zeros(1, 3, 4),
        "node_mask": torch.ones(1, 3, dtype=torch.bool),
        "room_topology_map": torch.zeros(1, 18, 6, 6),
    }
    changed_topology = dict(base_spatial)
    changed_topology["room_topology_map"] = torch.ones(1, 18, 6, 6)

    with torch.no_grad():
        out_base = model.denoiser(x_t, t, context, spatial_graph_data=base_spatial)
        out_topology = model.denoiser(x_t, t, context, spatial_graph_data=changed_topology)

    assert tuple(out_base.shape) == tuple(x_t.shape)
    assert not torch.allclose(out_base, out_topology)


def test_flow_ode_sampler_matches_flow_matching_objective_shape_and_finiteness():
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        denoiser_backbone="dit",
        dit_depth=1,
        dit_patch_size=1,
        dit_mlp_ratio=2.0,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        cfg_scale=1.0,
        unet_num_heads=4,
        unet_dropout=0.0,
    )
    model.eval()
    context = torch.randn(2, 4, 16)
    graph_data = {"node_mask": torch.ones(2, 4, dtype=torch.bool)}

    sample, intermediates = model.flow_ode_sample(
        context,
        shape=(2, 8, 3, 4),
        graph_data=graph_data,
        num_steps=4,
        return_intermediates=True,
    )
    routed = model.sample(
        context,
        shape=(2, 8, 3, 4),
        graph_data=graph_data,
        sampler="flow_ode",
        num_steps=4,
    )

    assert sample.shape == (2, 8, 3, 4)
    assert routed.shape == (2, 8, 3, 4)
    assert len(intermediates) == 5
    assert torch.isfinite(sample).all()
    assert torch.isfinite(routed).all()


def test_flow_matching_loss_accepts_min_snr_config_without_discrete_timestep_indexing():
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        denoiser_backbone="dit",
        dit_depth=1,
        dit_patch_size=1,
        dit_mlp_ratio=2.0,
        num_timesteps=8,
        min_snr_gamma=5.0,
        cfg_dropout_prob=0.0,
        unet_num_heads=4,
        unet_dropout=0.0,
    )
    model.train()
    z_0 = torch.randn(2, 8, 3, 4, requires_grad=True)
    context = torch.randn(2, 16)

    loss = model.flow_matching_loss(z_0, context)
    assert torch.isfinite(loss)
    loss.backward()
    assert z_0.grad is not None
    assert torch.isfinite(z_0.grad).all()


def test_dit_cross_attention_uses_graph_tokens_and_masks():
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        denoiser_backbone="dit",
        dit_depth=1,
        dit_patch_size=1,
        dit_mlp_ratio=2.0,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        unet_num_heads=4,
        unet_dropout=0.0,
    )
    model.eval()
    x_t = torch.randn(1, 8, 3, 4)
    t = torch.tensor([2], dtype=torch.long)
    context = torch.randn(1, 4, 16)
    graph_data = {"node_mask": torch.tensor([[True, True, False, False]])}

    baseline = model.denoiser(x_t, t, context, context_node_mask=graph_data["node_mask"])
    changed_masked_tokens = context.clone()
    changed_masked_tokens[:, 2:] = changed_masked_tokens[:, 2:] + 100.0
    masked_same = model.denoiser(x_t, t, changed_masked_tokens, context_node_mask=graph_data["node_mask"])
    changed_valid_tokens = context.clone()
    changed_valid_tokens[:, :2] = changed_valid_tokens[:, :2] + 1.0
    valid_changed = model.denoiser(x_t, t, changed_valid_tokens, context_node_mask=graph_data["node_mask"])

    assert torch.allclose(baseline, masked_same, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(baseline, valid_changed)


def test_pag_reaches_dit_attention_blocks_and_resets_mode():
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        denoiser_backbone="dit",
        dit_depth=1,
        dit_patch_size=1,
        dit_mlp_ratio=2.0,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        cfg_scale=1.0,
        pag_scale=0.5,
        unet_num_heads=4,
    )
    x_t = torch.randn(1, 8, 3, 4)
    t = torch.tensor([3], dtype=torch.long)
    context = torch.randn(1, 3, 16)
    node_mask = torch.ones(1, 3, dtype=torch.bool)

    model.pag_scale = 0.0
    baseline = model._predict_noise_cfg(x_t, t, context, graph_data={"node_mask": node_mask})
    model.pag_scale = 0.5
    guided = model._predict_noise_cfg(x_t, t, context, graph_data={"node_mask": node_mask})

    assert not torch.allclose(baseline, guided)
    for module in model.modules():
        if hasattr(module, "perturbation_mode"):
            assert module.perturbation_mode == "none"


def test_pag_uses_self_attention_perturbation_and_resets_mode():
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        cfg_scale=1.0,
        pag_scale=0.75,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(0,),
        unet_num_heads=4,
    )
    x_t = torch.randn(1, 8, 3, 4)
    t = torch.tensor([3], dtype=torch.long)
    context = torch.randn(1, 16)

    model.pag_scale = 0.0
    baseline = model._predict_noise_cfg(x_t, t, context)
    model.pag_scale = 0.75
    guided = model._predict_noise_cfg(x_t, t, context)

    assert not torch.allclose(baseline, guided)
    for module in model.modules():
        if hasattr(module, "perturbation_mode"):
            assert module.perturbation_mode == "none"


def test_pag_uses_conditional_prediction_not_post_cfg_branch(monkeypatch):
    class _FormulaProbeDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.perturbed = False

        def forward(self, _x_t, _t, context, **_kwargs):
            if self.perturbed:
                return torch.full((1, 2, 2, 2), 2.0)
            if torch.count_nonzero(context).item() == 0:
                return torch.full((1, 2, 2, 2), 10.0)
            return torch.full((1, 2, 2, 2), 3.0)

    model = create_latent_diffusion(
        latent_dim=2,
        model_channels=8,
        context_dim=4,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        cfg_scale=2.0,
        pag_scale=0.5,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=1,
    )
    probe = _FormulaProbeDenoiser()
    model.denoiser = probe

    def _set_perturbation(mode):
        probe.perturbed = mode == "identity"
        return 1

    monkeypatch.setattr(model, "set_self_attention_perturbation", _set_perturbation)

    out = model._predict_noise_cfg(
        torch.zeros(1, 2, 2, 2),
        torch.tensor([3], dtype=torch.long),
        torch.ones(1, 4),
    )

    assert torch.allclose(out, torch.full_like(out, -3.5))
    assert probe.perturbed is False


def test_diffusion_training_config_exposes_flow_matching_ablation():
    resolved = merge_config(yaml_path=None, cli_overrides=None)
    kwargs = diffusion_training_kwargs_from_resolved_config(resolved)
    assert kwargs["diffusion_training_objective"] == "diffusion"
    assert kwargs["denoiser_backbone"] == "unet"
    assert kwargs["pag_scale"] == pytest.approx(0.0)

    config = DiffusionTrainingConfig(diffusion_training_objective="flow_matching", denoiser_backbone="dit")
    assert config.diffusion_training_objective == "flow_matching"
    pag_config = DiffusionTrainingConfig(pag_scale=1.25)
    assert pag_config.pag_scale == pytest.approx(1.25)
    dit_config = DiffusionTrainingConfig(denoiser_backbone="dit", dit_depth=2, dit_patch_size=1)
    assert dit_config.denoiser_backbone == "dit"
    with pytest.raises(ValueError, match="diffusion_training_objective"):
        DiffusionTrainingConfig(diffusion_training_objective="pag")
    with pytest.raises(ValueError, match="denoiser_backbone='dit'"):
        DiffusionTrainingConfig(diffusion_training_objective="flow_matching", denoiser_backbone="unet")
    with pytest.raises(ValueError, match="denoiser_backbone"):
        DiffusionTrainingConfig(denoiser_backbone="mlp")


def test_latent_diffusion_compute_loss_dispatches_configured_objective(monkeypatch):
    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=8,
        context_dim=8,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=1,
        training_objective="flow_matching",
    )
    assert model.training_objective == "flow_matching"

    calls = []
    z_0 = torch.randn(1, 4, 2, 2)
    context = torch.randn(1, 8)

    def _diffusion_loss(*args, **kwargs):
        calls.append("diffusion")
        return torch.tensor(2.0)

    def _flow_loss(*args, **kwargs):
        calls.append("flow")
        return torch.tensor(3.0)

    monkeypatch.setattr(model, "training_loss", _diffusion_loss)
    monkeypatch.setattr(model, "flow_matching_loss", _flow_loss)

    loss = model.compute_loss(z_0, context)
    assert loss.item() == pytest.approx(3.0)
    assert calls == ["flow"]

    model.training_objective = "diffusion"
    loss = model.compute_loss(z_0, context)
    assert loss.item() == pytest.approx(2.0)
    assert calls == ["flow", "diffusion"]

    model.training_objective = "unknown"
    with pytest.raises(ValueError, match="training_objective"):
        model.compute_loss(z_0, context)


def test_dpo_preference_loss_trains_preferred_over_rejected_pairs():
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=4,
    )
    reference = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=4,
    )
    reference.load_state_dict(model.state_dict())
    for param in reference.parameters():
        param.requires_grad_(False)

    preferred = torch.randn(2, 8, 3, 4)
    rejected = torch.randn(2, 8, 3, 4)
    context = torch.randn(2, 16)
    noise = torch.randn_like(preferred)
    timesteps = torch.tensor([1, 5], dtype=torch.long)

    rng_state = torch.random.get_rng_state()
    loss, metrics = model.dpo_preference_loss(
        preferred,
        rejected,
        context,
        reference_model=reference,
        beta=0.2,
        noise=noise,
        timesteps=timesteps,
    )
    torch.random.set_rng_state(rng_state)
    called_loss, called_metrics = model(
        preferred,
        rejected,
        context,
        reference_model=reference,
        beta=0.2,
        noise=noise,
        timesteps=timesteps,
        forward_mode="dpo_preference_loss",
    )

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert torch.allclose(called_loss, loss)
    assert set(metrics) >= {
        "dpo_model_margin",
        "dpo_reference_margin",
        "dpo_preferred_score",
        "dpo_rejected_score",
    }
    assert torch.allclose(called_metrics["dpo_model_margin"], metrics["dpo_model_margin"])
    loss.backward()
    model_grad = sum(
        float(param.grad.detach().abs().sum().item())
        for param in model.parameters()
        if param.grad is not None
    )
    ref_grad = sum(
        float(param.grad.detach().abs().sum().item())
        for param in reference.parameters()
        if param.grad is not None
    )
    assert model_grad > 0.0
    assert ref_grad == 0.0


def test_perturb_and_map_reachability_open_grid_is_deterministic():
    walkability = np.ones((5, 5), dtype=np.float32)
    result_a = perturb_and_map_reachability(
        walkability,
        (0, 0),
        (4, 4),
        num_samples=8,
        noise_scale=0.1,
        seed=7,
    )
    result_b = perturb_and_map_reachability(
        walkability,
        (0, 0),
        (4, 4),
        num_samples=8,
        noise_scale=0.1,
        seed=7,
    )

    assert result_a.reachability == pytest.approx(1.0)
    assert result_a.num_successes == 8
    assert math.isfinite(result_a.mean_cost)
    assert result_a.path_frequency.shape == (5, 5)
    assert result_a.path_frequency[0, 0] == pytest.approx(1.0)
    assert result_a.path_frequency[4, 4] == pytest.approx(1.0)
    assert np.allclose(result_a.path_frequency, result_b.path_frequency)


def test_perturb_and_map_reachability_respects_hard_blockers():
    walkability = np.ones((5, 5), dtype=np.float32)
    walkability[2, :] = 0.0

    result = perturb_and_map_reachability(
        walkability,
        (0, 0),
        (4, 4),
        num_samples=6,
        noise_scale=0.5,
        seed=11,
    )

    assert result.reachability == pytest.approx(0.0)
    assert result.num_successes == 0
    assert math.isinf(result.mean_cost)
    assert np.count_nonzero(result.path_frequency) == 0
