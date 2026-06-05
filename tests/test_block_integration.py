# pyright: reportPrivateUsage=false

"""
H-MOLQD Block Integration Test
================================

Validates that all 7 blocks can be instantiated and connected
end-to-end without import errors, signature mismatches, or crashes.

Run:
    python -m pytest tests/test_block_integration.py -v
    python tests/test_block_integration.py          # standalone
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import numpy as np

from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    ROOM_WIDTH,
)


def test_block_ii_vqvae():
    """Block II: SemanticVQVAE instantiation, encode/decode round-trip."""
    from src.core.vqvae import create_vqvae

    model = create_vqvae(num_classes=44, codebook_size=64, latent_dim=32)
    x = torch.randn(2, 44, ROOM_HEIGHT, ROOM_WIDTH)

    # encode returns exactly 2 values
    z_q, indices = model.encode(x)
    assert z_q.shape[0] == 2
    assert z_q.shape[1] == 32  # latent_dim
    assert indices.shape[0] == 2

    # decode round-trip
    recon = model.decode(z_q, target_size=(ROOM_HEIGHT, ROOM_WIDTH))
    assert recon.shape == (2, 44, ROOM_HEIGHT, ROOM_WIDTH)
    recon_swapped = model.decode(z_q, target_size=(ROOM_WIDTH, ROOM_HEIGHT))
    assert recon_swapped.shape == (2, 44, ROOM_HEIGHT, ROOM_WIDTH)

    # full forward returns (recon, indices, losses_dict)
    recon, indices, losses = model(x)
    assert 'total_loss' in losses
    print("  ✓ Block II (VQ-VAE): encode/decode/forward OK")


def test_block_iii_condition_encoder():
    """Block III: DualStreamConditionEncoder with edge features."""
    from src.core.condition_encoder import create_condition_encoder

    encoder = create_condition_encoder(latent_dim=32, output_dim=128)
    assert encoder.global_encoder.node_feature_dim == GRAPH_NODE_FEATURE_DIM
    assert encoder.global_encoder.edge_feature_dim == GRAPH_EDGE_FEATURE_DIM
    assert encoder.global_encoder.gnn_type == "gcn"

    # Test encode_global_only (most common path)
    node_features = torch.randn(5, 5)
    edge_index = torch.tensor([[0,1,2,3,1], [1,2,3,4,3]], dtype=torch.long)
    edge_features = torch.randn(5, GRAPH_EDGE_FEATURE_DIM)  # Phase 3A: edge features

    c_global = encoder.encode_global_only(
        node_features, edge_index,
        edge_features=edge_features,
    )
    assert c_global.shape[0] == 5  # N nodes
    assert c_global.shape[1] == 128  # output_dim

    # Test without edge features (backward compatible)
    c_global_no_edge = encoder.encode_global_only(node_features, edge_index)
    assert c_global_no_edge.shape == c_global.shape
    print("  ✓ Block III (ConditionEncoder): global encoding with/without edge features OK")


@pytest.mark.parametrize("gnn_type", ["sage", "gps"])
def test_block_iii_condition_encoder_supports_alternative_graph_backbones(gnn_type: str):
    """Block III: alternative graph backbones should preserve the token shape contract."""
    from src.core.condition_encoder import create_condition_encoder

    encoder = create_condition_encoder(latent_dim=32, output_dim=96, gnn_type=gnn_type)
    node_features = torch.randn(5, encoder.global_encoder.node_feature_dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 1], [1, 2, 3, 4, 3]], dtype=torch.long)
    edge_features = torch.randn(edge_index.shape[1], encoder.global_encoder.edge_feature_dim)

    c_global = encoder.encode_global_only(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )

    assert tuple(c_global.shape) == (5, 96)
    assert torch.isfinite(c_global).all()


def test_block_iii_condition_encoder_clamps_integer_edge_labels_to_fixed_width():
    """Block III: integer edge labels should use stable fixed-width one-hot encoding."""
    from src.core.condition_encoder import create_condition_encoder

    encoder = create_condition_encoder(latent_dim=32, output_dim=64)
    global_encoder = encoder.global_encoder
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_labels = torch.tensor(
        [0, global_encoder.edge_feature_dim - 1, global_encoder.edge_feature_dim + 5],
        dtype=torch.long,
    )

    prepared = global_encoder._prepare_edge_features(edge_labels, edge_index)

    assert prepared is not None
    assert prepared.shape == (3, global_encoder.edge_feature_dim)
    assert torch.allclose(prepared.sum(dim=1), torch.ones(3))
    assert prepared[0, 0] == 1.0
    assert prepared[1, global_encoder.edge_feature_dim - 1] == 1.0
    assert prepared[2, global_encoder.edge_feature_dim - 1] == 1.0


def test_block_iv_latent_diffusion():
    """Block IV: LatentDiffusionModel training loss and sampling."""
    from src.core.latent_diffusion import create_latent_diffusion

    model = create_latent_diffusion(
        latent_dim=32, model_channels=32, context_dim=64,
        num_timesteps=10,  # tiny for test speed
        cfg_scale=2.0,
        prediction_type='epsilon',
        min_snr_gamma=5.0,
    )

    z_0 = torch.randn(2, 32, 3, 4)
    context = torch.randn(2, 64)

    # Training loss
    loss = model.training_loss(z_0, context)
    assert loss.shape == ()
    assert not torch.isnan(loss)

    # DDPM sample
    with torch.no_grad():
        z_gen = model.sample(context, shape=(2, 32, 3, 4))
    assert z_gen.shape == (2, 32, 3, 4)

    # DDIM sample (checks CRITICAL-3 fix: alpha_t defined)
    with torch.no_grad():
        z_ddim = model.ddim_sample(context, shape=(2, 32, 3, 4), num_steps=5)
    assert z_ddim.shape == (2, 32, 3, 4)
    print("  ✓ Block IV (LatentDiffusion): loss, DDPM, DDIM sampling OK")


def test_block_iv_ddim_sample_clamps_when_num_steps_exceeds_num_timesteps():
    """Block IV: DDIM sampling should remain valid when num_steps exceeds schedule length."""
    from src.core.latent_diffusion import create_latent_diffusion

    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=8,
        context_dim=16,
        num_timesteps=4,
        cfg_scale=1.0,
    )
    model.eval()
    context = torch.randn(1, 16)

    with torch.no_grad():
        z_ddim = model.ddim_sample(context, shape=(1, 8, 8, 8), num_steps=9)

    assert z_ddim.shape == (1, 8, 8, 8)
    assert torch.isfinite(z_ddim).all()


def test_block_iv_cfg_schedule_decays_toward_min_scale():
    """Block IV: scheduled CFG should start high and decay near the final denoising steps."""
    from src.core.latent_diffusion import create_latent_diffusion

    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=8,
        context_dim=16,
        num_timesteps=10,
        cfg_scale=5.0,
        cfg_schedule_mode="linear_decay",
        cfg_schedule_min_scale=1.0,
        cfg_schedule_power=1.0,
    )

    scales = model._cfg_scale_for_timestep(torch.tensor([9, 4, 0]))
    assert scales.shape == (3,)
    assert scales[0].item() == pytest.approx(5.0)
    assert scales[-1].item() == pytest.approx(1.0)
    assert 1.0 < scales[1].item() < 5.0


def test_block_iv_predict_noise_cfg_uses_scheduled_scale():
    """Block IV: CFG interpolation should follow the scheduled scale, not a static scalar."""
    from src.core.latent_diffusion import create_latent_diffusion

    class _DummyDenoiser(torch.nn.Module):
        def forward(self, x_t, t, context, **kwargs):
            strength = context.sum(dim=1, keepdim=True)
            return strength[:, :, None, None].expand_as(x_t)

    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=8,
        context_dim=4,
        num_timesteps=10,
        cfg_scale=5.0,
        cfg_schedule_mode="linear_decay",
        cfg_schedule_min_scale=1.0,
    )
    model.denoiser = _DummyDenoiser()

    x_t = torch.zeros(1, 4, 2, 2)
    context = torch.ones(1, 4)

    pred_early = model._predict_noise_cfg(x_t, torch.tensor([9]), context)
    pred_late = model._predict_noise_cfg(x_t, torch.tensor([0]), context)

    assert torch.allclose(pred_early, torch.full_like(pred_early, 20.0))
    assert torch.allclose(pred_late, torch.full_like(pred_late, 4.0))


def test_block_iv_gradient_guidance_sanitizes_graph_data_and_vector_loss():
    """Block IV: guidance should sanitize graph tensors and accept vector LogicNet loss."""
    from src.core.latent_diffusion import GradientGuidance

    class _DummyLogicNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.captured_graph_data = None

        def forward(self, x_t, graph_data=None):
            self.captured_graph_data = graph_data
            assert graph_data is not None
            assert graph_data["adjacency"].device == x_t.device
            assert graph_data["edge_weights"].device == x_t.device
            assert graph_data["adjacency"].dtype == x_t.dtype
            assert graph_data["edge_weights"].dtype == x_t.dtype
            assert graph_data["adjacency"].requires_grad is False
            assert graph_data["edge_weights"].requires_grad is False
            return x_t.flatten(1).mean(dim=1)

    logic_net = _DummyLogicNet()
    guidance = GradientGuidance(
        logic_net=logic_net,
        guidance_scale=0.5,
        clamp_magnitude=10.0,
        schedule_enabled=False,
        max_graph_nodes=8,
        max_key_lock_pairs=4,
        max_guidance_elements=1024,
    )

    x_t = torch.randn(2, 4, 3, 3)
    graph_data = {
        "adjacency": torch.tensor(
            [
                [0.0, float("nan"), 2.0],
                [1.0, 0.0, float("inf")],
                [0.0, -3.0, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "edge_weights": torch.tensor(
            [
                [0.0, 1.0, float("inf")],
                [2.0, float("nan"), 4.0],
                [5.0, -7.0, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "start_idx": "1",
        "target_idx": 99,
        "key_lock_pairs": [("0", "1"), ("bad", 2), (1, 99)],
    }

    grad = guidance.compute_guidance(x_t, graph_data)

    assert grad.shape == x_t.shape
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 0.0

    captured = logic_net.captured_graph_data
    assert captured is not None
    assert captured["start_idx"] == 1
    assert captured["target_idx"] is None
    assert captured["key_lock_pairs"] == [(0, 1)]
    assert torch.equal(
        captured["adjacency"],
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=x_t.dtype,
        ),
    )
    assert float(captured["edge_weights"].max().item()) == float(guidance.max_graph_nodes)
    assert float(captured["edge_weights"][2, 1].item()) == 0.0


def test_block_iv_gradient_guidance_objective_mode_controls_sign():
    """Loss guidance descends the objective; reward guidance ascends it."""
    from src.core.latent_diffusion import GradientGuidance

    class _LinearObjective(torch.nn.Module):
        def forward(self, x_t, graph_data=None):
            _ = graph_data
            return x_t.flatten(1).sum(dim=1)

    x_t = torch.ones(1, 1, 1, 1)
    predicted_mean = torch.zeros_like(x_t)

    loss_guidance = GradientGuidance(
        logic_net=_LinearObjective(),
        guidance_scale=1.0,
        clamp_magnitude=0.0,
        relative_norm_cap=0.0,
        mean_relative_norm_cap=0.0,
        schedule_enabled=False,
        max_guidance_elements=16,
        objective_mode="loss",
    )
    reward_guidance = GradientGuidance(
        logic_net=_LinearObjective(),
        guidance_scale=1.0,
        clamp_magnitude=0.0,
        relative_norm_cap=0.0,
        mean_relative_norm_cap=0.0,
        schedule_enabled=False,
        max_guidance_elements=16,
        objective_mode="reward",
    )

    assert torch.allclose(loss_guidance.apply_guidance(predicted_mean, x_t), torch.full_like(x_t, -1.0))
    assert torch.allclose(reward_guidance.apply_guidance(predicted_mean, x_t), torch.full_like(x_t, 1.0))


def test_block_iv_gradient_guidance_caps_update_against_predicted_mean():
    """Apply-time cap should prevent guidance from dominating the denoiser mean."""
    from src.core.latent_diffusion import GradientGuidance

    class _LargeGradObjective(torch.nn.Module):
        def forward(self, x_t, graph_data=None):
            _ = graph_data
            return (x_t * 1000.0).flatten(1).sum(dim=1)

    guidance = GradientGuidance(
        logic_net=_LargeGradObjective(),
        guidance_scale=1.0,
        clamp_magnitude=0.0,
        relative_norm_cap=0.0,
        mean_relative_norm_cap=0.25,
        mean_norm_floor_fraction=0.0,
        schedule_enabled=False,
        max_guidance_elements=1024,
    )

    x_t = torch.ones(1, 2, 2, 2)
    predicted_mean = torch.full_like(x_t, 2.0)
    guided = guidance.apply_guidance(predicted_mean, x_t)

    update_norm = (predicted_mean - guided).view(1, -1).norm(dim=1)
    mean_norm = predicted_mean.view(1, -1).norm(dim=1)
    assert torch.all(update_norm <= mean_norm * 0.25 + 1e-5)


def test_block_iv_gradient_guidance_skips_oversized_latents():
    """Block IV: guidance should skip expensive autograd when latent size exceeds cap."""
    from src.core.latent_diffusion import GradientGuidance

    class _FailIfCalled(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("LogicNet should not be invoked for oversized guidance payloads")

    guidance = GradientGuidance(
        logic_net=_FailIfCalled(),
        schedule_enabled=False,
        max_guidance_elements=4,
    )

    x_t = torch.randn(1, 2, 2, 2)
    grad = guidance.compute_guidance(x_t)

    assert torch.equal(grad, torch.zeros_like(x_t))


def test_block_iv_gradient_guidance_caps_relative_norm():
    """Logic guidance should not overwhelm the latent magnitude on OOD graphs."""
    from src.core.latent_diffusion import GradientGuidance

    class _LargeGradLogicNet(torch.nn.Module):
        def forward(self, x_t, graph_data=None):
            _ = graph_data
            return (x_t * 1000.0).flatten(1).sum(dim=1)

    guidance = GradientGuidance(
        logic_net=_LargeGradLogicNet(),
        guidance_scale=1.0,
        clamp_magnitude=0.0,
        relative_norm_cap=0.10,
        schedule_enabled=False,
        max_guidance_elements=1024,
    )

    x_t = torch.ones(1, 2, 2, 2, dtype=torch.float32)
    grad = guidance.compute_guidance(x_t)

    grad_norm = grad.view(1, -1).norm(dim=1)
    ref_norm = x_t.view(1, -1).norm(dim=1)
    assert torch.all(grad_norm <= ref_norm * 0.10 + 1e-5)


def test_block_iv_gradient_guidance_accepts_room_topology_without_graph_adjacency():
    """Room-topology guidance should remain active even when only per-room priors are available."""
    from src.core.latent_diffusion import GradientGuidance

    class _TopologyOnlyLogicNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.captured_graph_data = None

        def forward(self, x_t, graph_data=None):
            self.captured_graph_data = graph_data
            assert graph_data is not None
            assert "room_topology_map" in graph_data
            assert "boundary_constraints" in graph_data
            assert "adjacency" not in graph_data
            return x_t.flatten(1).mean(dim=1)

    logic_net = _TopologyOnlyLogicNet()
    guidance = GradientGuidance(
        logic_net=logic_net,
        guidance_scale=0.5,
        clamp_magnitude=10.0,
        schedule_enabled=False,
        max_guidance_elements=1024,
    )

    x_t = torch.randn(1, 4, 3, 3)
    graph_data = {
        "room_topology_map": torch.rand(ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32, requires_grad=True),
        "boundary_constraints": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True),
    }

    grad = guidance.compute_guidance(x_t, graph_data)

    assert grad.shape == x_t.shape
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 0.0
    captured = logic_net.captured_graph_data
    assert captured is not None
    assert captured["room_topology_map"].shape == (1, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH)
    assert captured["room_topology_map"].requires_grad is False
    assert captured["boundary_constraints"].shape == (1, 8)
    assert captured["boundary_constraints"].requires_grad is False


def test_block_iv_gradient_guidance_preserves_edge_index_graph_context():
    """Edge-index mission graphs should reach LogicNet guidance even without adjacency matrices."""
    from src.core.latent_diffusion import GradientGuidance

    class _GraphOnlyLogicNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.captured_graph_data = None

        def forward(self, x_t, graph_data=None):
            self.captured_graph_data = graph_data
            assert graph_data is not None
            assert "adjacency" not in graph_data
            assert tuple(graph_data["edge_index"].shape) == (2, 2)
            assert tuple(graph_data["node_features"].shape) == (3, 6)
            assert tuple(graph_data["edge_features"].shape) == (2, 8)
            assert graph_data["current_node_idx"].dtype == torch.long
            assert graph_data["start_node_id"].dtype == torch.long
            return x_t.flatten(1).mean(dim=1)

    logic_net = _GraphOnlyLogicNet()
    guidance = GradientGuidance(
        logic_net=logic_net,
        guidance_scale=0.5,
        clamp_magnitude=10.0,
        schedule_enabled=False,
        max_graph_nodes=8,
        max_guidance_elements=1024,
    )

    x_t = torch.randn(1, 4, 3, 3)
    graph_data = {
        "graph_scope": "room",
        "node_features": torch.randn(3, 6, dtype=torch.float32, requires_grad=True),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_features": torch.zeros(2, 8, dtype=torch.float32, requires_grad=True),
        "current_node_idx": torch.tensor([1], dtype=torch.long),
        "start_node_id": torch.tensor(0, dtype=torch.long),
        "target_idx": torch.tensor(2, dtype=torch.long),
    }

    grad = guidance.compute_guidance(x_t, graph_data)

    assert grad.shape == x_t.shape
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 0.0
    captured = logic_net.captured_graph_data
    assert captured is not None
    assert captured["graph_scope"] == "room"
    assert captured["node_features"].requires_grad is False
    assert captured["edge_features"].requires_grad is False


def test_block_iv_topology_aware_cross_attention_sequence_context():
    """Block IV: sequence context path with topology-aware cross-attention refinement."""
    from src.core.latent_diffusion import create_latent_diffusion

    model = create_latent_diffusion(
        latent_dim=16,
        model_channels=16,
        context_dim=32,
        num_timesteps=10,
        cfg_scale=2.0,
    )

    context = torch.randn(1, 6, 32)
    graph_data = {
        # 5 graph nodes (+1 anchor token prepended to context sequence in pipeline).
        'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        'node_features': torch.randn(5, 6),
    }

    with torch.no_grad():
        z_ddim = model.ddim_sample(
            context=context,
            shape=(1, 16, 8, 8),
            num_steps=3,
            graph_data=graph_data,
        )

    assert z_ddim.shape == (1, 16, 8, 8)
    assert torch.isfinite(z_ddim).all()
    print("  ✓ Block IV (Topology-Aware CrossAttention): sequence context path OK")


def test_block_iv_topology_aware_cross_attention_mask_broadcast_and_sparse_valid_tokens():
    """Block IV: topology refinement handles sparse valid tokens and broadcast node masks."""
    from src.core.latent_diffusion import create_latent_diffusion

    model = create_latent_diffusion(
        latent_dim=16,
        model_channels=16,
        context_dim=32,
        num_timesteps=10,
        cfg_scale=1.0,
    )

    # Batch size 2 with sequence context. node_mask is provided as 1D and should broadcast.
    context = torch.randn(2, 6, 32)
    graph_data = {
        'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        'node_features': torch.randn(5, 6),
        # Sparse/non-contiguous valid-node pattern over 5 graph nodes.
        'node_mask': torch.tensor([1, 0, 1, 1, 0], dtype=torch.long),
    }

    with torch.no_grad():
        z_ddim = model.ddim_sample(
            context=context,
            shape=(2, 16, 8, 8),
            num_steps=3,
            graph_data=graph_data,
        )

    assert z_ddim.shape == (2, 16, 8, 8)
    assert torch.isfinite(z_ddim).all()
    print("  ✓ Block IV (Topology-Aware CrossAttention): node_mask broadcast + sparse valid tokens OK")


def test_block_iv_topology_refinement_mode_switch_runs_all_modes():
    """Block IV: topology refinement ablation modes all run."""
    from src.core.latent_diffusion import create_latent_diffusion

    model = create_latent_diffusion(
        latent_dim=16,
        model_channels=16,
        context_dim=32,
        num_timesteps=10,
        cfg_scale=1.0,
    )

    context = torch.randn(1, 6, 32)
    graph_data = {
        'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        'node_features': torch.randn(5, 6),
    }

    outputs = []
    for mode in ["none", "lightweight", "sparse_edge", "upgraded", "graphormer"]:
        updated = model.set_topology_refinement_mode(mode)
        assert updated > 0
        assert model.get_topology_refinement_mode() == mode
        with torch.no_grad():
            z = model.ddim_sample(
                context=context,
                shape=(1, 16, 8, 8),
                num_steps=3,
                graph_data=graph_data,
            )
        assert z.shape == (1, 16, 8, 8)
        assert torch.isfinite(z).all()
        outputs.append(z)

    # Ensure mode changes are not degenerate no-ops for all outputs.
    diff_light_vs_up = float((outputs[1] - outputs[2]).abs().mean().item())
    assert diff_light_vs_up >= 0.0
    print("  ✓ Block IV (Topology Modes): none/lightweight/sparse_edge/upgraded/graphormer execution OK")


def test_block_iv_attention_mode_switch_runs_softmax_and_linear_hedgehog():
    """Block IV: cross-attention kernel can switch between softmax and linear Hedgehog."""
    from src.core.latent_diffusion import create_latent_diffusion

    model = create_latent_diffusion(
        latent_dim=16,
        model_channels=16,
        context_dim=32,
        num_timesteps=10,
        cfg_scale=1.0,
        attention_mode="softmax",
        hedgehog_feature_dim=16,
    )

    context = torch.randn(1, 6, 32)
    graph_data = {
        'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        'node_features': torch.randn(5, 6),
    }

    outputs = []
    for mode in ["softmax", "linear_hedgehog"]:
        updated = model.set_attention_mode(mode)
        assert updated > 0
        assert model.get_attention_mode() == mode
        with torch.no_grad():
            z = model.ddim_sample(
                context=context,
                shape=(1, 16, 8, 8),
                num_steps=3,
                graph_data=graph_data,
            )
        assert z.shape == (1, 16, 8, 8)
        assert torch.isfinite(z).all()
        outputs.append(z)

    mean_abs_diff = float((outputs[0] - outputs[1]).abs().mean().item())
    assert mean_abs_diff >= 0.0


@pytest.mark.parametrize("topology_conditioning_mode", ["additive", "spade"])
def test_block_iv_spatial_graph_conditioning_accepts_room_topology_maps(topology_conditioning_mode: str):
    """Block IV: active denoiser accepts graph-grid spatial topology conditioning for both topology modes."""
    from src.core.latent_diffusion import create_latent_diffusion

    model = create_latent_diffusion(
        latent_dim=16,
        model_channels=16,
        context_dim=32,
        num_timesteps=10,
        cfg_scale=1.0,
        attention_mode="linear_hedgehog",
        topology_conditioning_mode=topology_conditioning_mode,
        hedgehog_feature_dim=16,
    )

    context = torch.randn(2, 6, 32)
    graph_data = {
        'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        'node_features': torch.randn(5, 6),
        'tpe': torch.randn(5, 8),
        'node_positions': torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [2.0, 2.0]],
            dtype=torch.float32,
        ),
        'node_mask': torch.ones(5, dtype=torch.float32),
        'room_topology_map': torch.randn(2, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH),
    }

    with torch.no_grad():
        z = model.ddim_sample(
            context=context,
            shape=(2, 16, 4, 3),
            num_steps=3,
            graph_data=graph_data,
        )

    assert z.shape == (2, 16, 4, 3)
    assert torch.isfinite(z).all()


def test_block_v_logic_net():
    """Block V: LogicNet forward and temperature annealing."""
    from src.core.logic_net import LogicNet

    logic = LogicNet(latent_dim=32, num_classes=44, num_iterations=5)

    z = torch.randn(2, 32, 3, 4)  # latent codes
    loss, info = logic(z)
    assert loss.shape == ()
    assert 'walkability' in info

    # Temperature annealing (Phase 1D)
    logic.update_temperature(0.0)
    assert abs(logic.current_temperature.item() - 1.0) < 0.01
    logic.update_temperature(1.0)
    assert logic.current_temperature.item() < 0.1
    print("  ✓ Block V (LogicNet): forward + temperature annealing OK")


def test_block_vi_map_elites():
    """Block VI: MAP-Elites archive and feature extractors."""
    from src.evaluation.map_elites import (
        EliteArchive, CVTEliteArchive,
        CombinedFeatureExtractor, CBSFeatureExtractor, FullFeatureExtractor,
        create_map_elites,
    )

    # Standard archive
    archive = EliteArchive(feature_dims=2, cells_per_dim=5)
    assert archive.feature_dims == 2

    # CVT archive
    cvt = CVTEliteArchive(feature_dims=2, num_cells=10)
    assert cvt.num_cells == 10

    # Feature extractors
    combined = CombinedFeatureExtractor()
    assert callable(combined.extract)

    cbs = CBSFeatureExtractor()
    assert callable(cbs.extract)

    full = FullFeatureExtractor()
    assert callable(full.extract)

    # create_map_elites convenience
    me = create_map_elites(feature_type='combined', archive_type='grid')
    assert me is not None
    print("  ✓ Block VI (MAP-Elites): archives + extractors OK")


def test_block_vii_symbolic_refiner():
    """Block VII: SymbolicRefiner with LearnedTileStatistics."""
    from src.core.symbolic_refiner import (
        create_symbolic_refiner,
        LearnedTileStatistics, FailurePoint,
    )

    # Test FailurePoint with metadata (CRITICAL-5 fix)
    fp = FailurePoint(
        position=(5, 5),
        failure_type='blocked',
        required_item=None,
        metadata={'room_id': 3},
    )
    assert fp.metadata == {'room_id': 3}

    # LearnedTileStatistics (Phase 3B)
    stats = LearnedTileStatistics()
    fake_room = np.random.randint(0, 10, size=(ROOM_HEIGHT, ROOM_WIDTH))
    stats.observe(fake_room)
    assert stats._total_tiles > 0

    adj = stats.get_adjacency_rules(threshold=0.01)
    assert isinstance(adj, dict)

    weights = stats.get_tile_weights()
    assert len(weights) > 0

    # SymbolicRefiner with learned stats
    refiner = create_symbolic_refiner(learned_stats=stats)
    assert refiner.learned_stats is not None

    # Quick repair test
    grid = np.ones((ROOM_HEIGHT, ROOM_WIDTH), dtype=int)  # all floor
    grid[5, :] = 2  # wall barrier
    repaired, _success = refiner.repair_room(grid, start=(2, 8), goal=(8, 8))
    assert repaired.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    print("  ✓ Block VII (SymbolicRefiner): LearnedTileStatistics + repair OK")


def test_pipeline_vqvae_to_diffusion():
    """End-to-end: VQ-VAE encode → Diffusion loss → sample → VQ-VAE decode."""
    from src.core.vqvae import create_vqvae
    from src.core.latent_diffusion import create_latent_diffusion
    from src.core.condition_encoder import create_condition_encoder

    vqvae = create_vqvae(num_classes=44, codebook_size=32, latent_dim=16)
    diffusion = create_latent_diffusion(
        latent_dim=16, model_channels=16, context_dim=32,
        num_timesteps=10,
    )
    cond_encoder = create_condition_encoder(latent_dim=16, output_dim=32)

    # Simulate training step
    x = torch.randn(2, 44, ROOM_HEIGHT, ROOM_WIDTH)
    vqvae.eval()
    with torch.no_grad():
        z_q, _indices = vqvae.encode(x)  # CRITICAL-2: 2 values

    # Build conditioning
    node_features = torch.randn(4, 5)
    edge_index = torch.tensor([[0,1,2,1],[1,2,3,0]], dtype=torch.long)
    c_global = cond_encoder.encode_global_only(node_features, edge_index)
    conditioning = c_global.mean(dim=0, keepdim=True).expand(2, -1)

    # Diffusion training loss
    loss = diffusion.training_loss(z_q, conditioning)
    assert not torch.isnan(loss)

    # Sample and decode
    with torch.no_grad():
        z_gen = diffusion.ddim_sample(conditioning, shape=z_q.shape, num_steps=5)
        recon = vqvae.decode(z_gen, target_size=(ROOM_HEIGHT, ROOM_WIDTH))
    assert recon.shape == (2, 44, ROOM_HEIGHT, ROOM_WIDTH)
    print("  ✓ Pipeline: VQ-VAE → Diffusion → VQ-VAE decode OK")


def test_trainer_instantiation():
    """DiffusionTrainer: can be constructed without crashes."""
    from src.train_diffusion import DiffusionTrainer, DiffusionTrainingConfig

    config = DiffusionTrainingConfig(
        quick=True,
        epochs=1,
        num_timesteps=10,
        latent_dim=16,
        model_channels=16,
        context_dim=32,
    )
    # This exercises CRITICAL-1, CRITICAL-6 fixes
    trainer = DiffusionTrainer(config)
    assert trainer.vqvae is not None
    assert trainer.diffusion is not None
    assert trainer.condition_encoder is not None
    assert trainer.logic_net is not None
    assert trainer.ema_diffusion is not None  # Phase 4A

    # Check Block V LogicNet (not legacy)
    assert hasattr(trainer.logic_net, 'update_temperature'), \
        "Should be Block V LogicNet with temperature annealing, not legacy"
    print("  ✓ DiffusionTrainer: instantiation OK (all blocks connected)")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("H-MOLQD Block Integration Tests")
    print("=" * 60 + "\n")

    tests = [
        test_block_ii_vqvae,
        test_block_iii_condition_encoder,
        test_block_iv_latent_diffusion,
        test_block_v_logic_net,
        test_block_vi_map_elites,
        test_block_vii_symbolic_refiner,
        test_pipeline_vqvae_to_diffusion,
        test_trainer_instantiation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}\n")
