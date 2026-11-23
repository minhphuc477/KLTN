"""Regression tests for model architecture and training/sampling optimization fixes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.config_system import merge_config
from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion
from src.core.vqvae import VectorQuantizer
from src.train_diffusion import DiffusionTrainer


def test_vq_dead_code_replacement_jitter_scales_with_manifold_spread():
    torch.manual_seed(0)
    z_small = torch.randn(512, 64) * 0.05
    repl_small = VectorQuantizer._sample_dead_code_replacements(z_small, 32)
    assert repl_small.shape == (32, 64)
    # Center-to-replacement offset should stay small relative to manifold spread,
    # rather than ejecting codes via an oversized min=0.05 per-axis noise floor.
    assert float(repl_small.std().item()) < 0.08

    z_zero = torch.zeros(64, 16)
    repl_zero = VectorQuantizer._sample_dead_code_replacements(z_zero, 8)
    assert float(repl_zero.abs().max().item()) > 0.0
    assert float(repl_zero.std().item()) < 0.01


def test_latent_diffusion_preserves_scaled_latents_above_unit_bound(monkeypatch):
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=32,
        context_dim=16,
        num_timesteps=20,
        unet_channel_mult=(1,),
        unet_attention_resolutions=(),
        latent_scale_factor=12.0,
        latent_clip_bound=5.0,
    ).eval()

    x_t = torch.full((2, 8, 4, 3), 2.5)
    t = torch.tensor([0, 0], dtype=torch.long)
    ctx = torch.zeros((2, 1, 16))
    # Force zero prediction so pred_x0 equals x_t / sqrt(alpha_bar_0) > 2.4
    monkeypatch.setattr(model, "_predict_noise_cfg", lambda x, *args, **kwargs: torch.zeros_like(x))
    mean, _, _ = model.p_mean_variance(x_t, t, ctx, clip_denoised=True)
    assert float(mean.min().item()) > 1.5

    # Disable clipping completely via latent_clip_bound=None
    unclipped = LatentDiffusionModel(
        latent_dim=8,
        model_channels=32,
        context_dim=16,
        num_timesteps=20,
        unet_channel_mult=(1,),
        unet_attention_resolutions=(),
        latent_clip_bound=None,
    ).eval()
    assert unclipped.get_latent_clamp_range() is None


def test_flow_ode_sample_does_not_truncate_gaussian_noise_states(monkeypatch):
    model = LatentDiffusionModel(
        latent_dim=4,
        model_channels=16,
        context_dim=8,
        num_timesteps=10,
        unet_channel_mult=(1,),
        unet_attention_resolutions=(),
        training_objective="flow_matching",
        latent_clip_bound=5.0,
    ).eval()
    # Zero velocity keeps x_{t - dt} == x_1 ~ N(0, 1); before our fix, step 1
    # clamped x_next to [-1.0, 1.0], destroying ~31.7% of the Gaussian noise mass.
    monkeypatch.setattr(model, "_predict_noise_cfg", lambda x, *a, **k: torch.zeros_like(x))

    torch.manual_seed(42)
    ctx = torch.zeros((1, 1, 8))
    out, intermediates = model.flow_ode_sample(
        ctx,
        shape=(4, 4, 4, 3),
        num_steps=4,
        return_intermediates=True,
    )
    assert float(intermediates[1].abs().max().item()) > 1.5
    assert float(out.abs().max().item()) > 1.5


def test_auxiliary_pred_x0_supports_flow_matching_and_bounded_timesteps(monkeypatch):
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(aux_max_timestep_fraction=0.5, use_amp=False, amp_mixed_precision="fp16")
    trainer._autocast_context = lambda: torch.autocast("cpu", enabled=False)
    trainer.distributed_context = None
    trainer._tensor_is_finite = lambda t: bool(torch.isfinite(t).all().item())
    trainer._handle_nonfinite = lambda *a, **k: None

    seen = {}

    class _DummyFlowDiffusion:
        num_timesteps = 100
        training_objective = "flow_matching"

        @staticmethod
        def _extract_context_topology(cond, gd):
            return None, None, None

        @staticmethod
        def _extract_spatial_graph_context(cond, gd):
            return None

        @staticmethod
        def denoiser(x_t, model_t, cond, **kwargs):
            seen["x_t"] = x_t
            seen["model_t"] = model_t
            # Exact oracle velocity v = noise - z_0 = (x_t - z_0) / t_cont for non-zero t
            t_cont = model_t.view(-1, 1, 1, 1) / float(99)
            return torch.where(t_cont > 0, (x_t - seen["z_0"]) / t_cont.clamp_min(1e-6), torch.zeros_like(x_t))

        @staticmethod
        def _clamp_clean_latent(pred):
            return torch.clamp(pred, -5.0, 5.0)

    trainer.diffusion = _DummyFlowDiffusion()
    z_0 = torch.full((4, 4, 4, 3), 2.5)
    seen["z_0"] = z_0

    pred_x0 = trainer._predict_x0_for_auxiliary_supervision(
        z_0=z_0,
        conditioning=torch.zeros((4, 1, 8)),
        diffusion_graph_data=None,
    )
    assert pred_x0 is not None
    # Bound timestep sampling to <= 50% of num_timesteps
    assert float(seen["model_t"].max().item()) <= 50.0
    # Oracle velocity recovers z_0=2.5 without getting truncated to 1.0
    assert torch.allclose(pred_x0, z_0, atol=1e-4)


def test_warmup_does_not_tick_cosine_scheduler_before_completion():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(
        epochs=10,
        global_lr_warmup_epochs=2,
        logic_lr_warmup_epochs=0,
        logic_net_trainable=False,
        learning_rate=1e-3,
    )
    trainer._estimated_total_steps = 10  # 2 epochs warmup => 2 warmup steps
    param = torch.nn.Parameter(torch.ones(2, 2))
    trainer.optimizer = torch.optim.SGD([{"params": [param], "lr": 1e-3, "base_lr": 1e-3, "name": "diffusion"}])
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=8, eta_min=0.0)

    trainer.global_step = 0
    trainer._apply_lr_warmup(completed_steps=0)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)

    # Step 1 completes while global warmup is finishing: scheduler must not advance early
    trainer.optimizer.step()
    trainer.global_step = 1
    trainer._step_scheduler_and_warmup()
    assert trainer.scheduler.last_epoch == 0
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    # Step 2 finishes after warmup completed: cosine decay begins cleanly
    trainer.optimizer.step()
    trainer.global_step = 2
    trainer._step_scheduler_and_warmup()
    assert trainer.scheduler.last_epoch == 1


def test_zelda_hmolqd_config_and_mission_bank_paths_valid():
    resolved = merge_config(yaml_path="configs/zelda_hmolqd.yaml", cli_overrides=None)
    diff = resolved["diffusion"]
    assert diff["unet_channel_mult"] == [1, 2]
    assert diff["unet_attention_resolutions"] == [0, 1]
    assert diff["scheduler_t0"] == 100
    assert diff["scheduler_t_mult"] == 1
    assert diff["use_amp"] is True
    assert diff["latent_clip_bound"] == pytest.approx(5.0)
    assert diff["aux_max_timestep_fraction"] == pytest.approx(0.5)

    design_path = Path("configs/publication_mission_bank/evaluation_design_pending_support.json")
    payload = json.loads(design_path.read_text(encoding="utf-8"))
    for split_entries in payload["mission_bank"].values():
        for raw_path in split_entries.values():
            assert Path(raw_path).is_file(), f"Missing mission bank file: {raw_path}"


def test_extract_context_topology_anchor_and_padding_without_node_features():
    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=16,
        context_dim=8,
        num_timesteps=10,
        unet_channel_mult=(1,),
        unet_attention_resolutions=(),
        unet_num_heads=2,
    ).eval()

    ctx = torch.zeros((2, 5, 8))
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    node_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    adj_ei, _adj_ea, adj_mask = model._extract_context_topology(
        ctx,
        {
            "edge_index": edge_index,
            "node_mask": node_mask,
            "has_room_anchor": True,
        },
    )
    assert torch.equal(adj_ei, edge_index + 1)
    assert adj_mask is not None
    assert tuple(adj_mask.shape) == (2, 5)
    assert torch.equal(adj_mask[:, 0], torch.ones(2))
    assert torch.equal(adj_mask[:, 1:4], node_mask)
    assert torch.equal(adj_mask[:, 4], torch.ones(2))

    # Also verify extra puzzle/prefix tokens beyond num_nodes + 1 still pad cleanly
    adj_ei2, _, adj_mask2 = model._extract_context_topology(
        ctx,
        {
            "node_features": torch.zeros((2, 3, 8)),
            "edge_index": edge_index,
            "node_mask": node_mask,
            "has_room_anchor": True,
        },
    )
    assert torch.equal(adj_ei2, edge_index + 1)
    assert tuple(adj_mask2.shape) == (2, 5)


def test_dit_denoiser_supports_odd_half_channels_and_cfg_all_masked_rows():
    from src.core.latent_diffusion import DiTDenoiser

    denoiser = DiTDenoiser(
        in_channels=4,
        model_channels=10,  # 10 // 2 = 5 (odd d_half, model_channels % 4 != 0)
        out_channels=4,
        context_dim=8,
        depth=2,
        patch_size=1,
        num_heads=2,
    ).eval()
    x = torch.randn(2, 4, 4, 3)
    t = torch.tensor([1.0, 5.0])
    ctx = torch.randn(2, 3, 8)
    node_mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])  # row 1 all-masked (CFG drop)

    out = denoiser(x, t, ctx, context_node_mask=node_mask)
    assert tuple(out.shape) == tuple(x.shape)
    assert torch.isfinite(out).all()


def test_local_stream_encoder_matches_bfloat16_dtype():
    from src.core.condition_encoder import LocalStreamEncoder

    encoder = LocalStreamEncoder(latent_dim=4, output_dim=16, hidden_dim=16).to(dtype=torch.bfloat16).eval()
    neighbors = {k: torch.zeros((2, 4, 4, 3), dtype=torch.float32) for k in ("N", "S", "E", "W")}
    boundary = torch.zeros((2, 8), dtype=torch.float32)
    position = torch.ones((2, 2), dtype=torch.float32)
    out = encoder(neighbors, boundary, position)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_vqvae_and_gaussian_vae_weighted_recon_accepts_float_scalar_targets():
    from src.core.gaussian_vae import SemanticGaussianVAE
    from src.core.vqvae import SemanticVQVAE, SemanticVQVAE2

    logits = torch.randn(2, 44, 16, 11)
    float_targets = torch.randint(0, 44, (2, 1, 16, 11)).to(dtype=torch.float32)

    vq1 = SemanticVQVAE(num_classes=44, codebook_size=16, latent_dim=8, hidden_dim=16, use_codebook=False)
    vq2 = SemanticVQVAE2(num_classes=44, codebook_size=16, latent_dim=8, hidden_dim=16, use_codebook=False)
    gvae = SemanticGaussianVAE(num_classes=44, latent_dim=8, hidden_dim=16)

    for model in (vq1, vq2, gvae):
        loss = model._weighted_reconstruction_loss(logits, float_targets)
        assert torch.isfinite(loss)


def test_lora_adapters_toggle_cleanly_and_disable_fast_sampling(tmp_path):
    from src.optimization.lcm_lora import (
        LoRALinear,
        extract_lora_state_dict,
        inject_lora_into_model,
        save_fast_sampler_checkpoint,
    )

    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=16,
        context_dim=8,
        num_timesteps=8,
        unet_channel_mult=(1,),
        unet_attention_resolutions=(0,),
        unet_num_heads=2,
    ).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.denoiser.output_proj.weight, std=0.1)

    x = torch.randn(1, 4, 4, 3)
    t = torch.tensor([2], dtype=torch.long)
    ctx = torch.randn(1, 2, 8)
    with torch.no_grad():
        base_out = model.denoiser(x, t, ctx).clone()

    # Build a valid adapter checkpoint with non-zero lora_B weights
    temp_model = create_latent_diffusion(
        latent_dim=4,
        model_channels=16,
        context_dim=8,
        num_timesteps=8,
        unet_channel_mult=(1,),
        unet_attention_resolutions=(0,),
        unet_num_heads=2,
    ).eval()
    inject_lora_into_model(temp_model.denoiser, rank=2, alpha=4.0)
    with torch.no_grad():
        for name, param in temp_model.named_parameters():
            if ".lora_B" in name:
                param.fill_(0.25)
    ckpt = tmp_path / "fast_adapter.pth"
    save_fast_sampler_checkpoint(
        str(ckpt),
        lora_state_dict=extract_lora_state_dict(temp_model),
        base_diffusion_checkpoint="unused.pth",
        num_inference_steps=2,
        lora_rank=2,
        lora_alpha=4.0,
    )

    model.enable_fast_sampling(adapter_checkpoint=str(ckpt), num_inference_steps=2, strict=True)
    # Re-enabling must unwrap and avoid double-wrapping LoRALinear.base_layer
    model.enable_fast_sampling(adapter_checkpoint=str(ckpt), num_inference_steps=2, strict=True)
    assert model.has_active_adapters() is True
    for mod in model.denoiser.modules():
        if isinstance(mod, LoRALinear):
            assert not isinstance(mod.base_layer, LoRALinear)

    with torch.no_grad():
        adapted_out = model.denoiser(x, t, ctx).clone()
    assert not torch.allclose(adapted_out, base_out, atol=1e-5)

    model.disable_adapters()
    assert model.has_active_adapters() is False
    with torch.no_grad():
        disabled_out = model.denoiser(x, t, ctx).clone()
    assert torch.allclose(disabled_out, base_out, atol=1e-6)

    model.disable_fast_sampling()
    assert model.supports_fast_sampling() is False
    assert model.has_active_adapters() is False


def test_latent_diffusion_inpaint_supports_flow_matching_and_preserves_unmasked_region(monkeypatch):
    model = LatentDiffusionModel(
        latent_dim=4,
        model_channels=16,
        context_dim=8,
        num_timesteps=20,
        unet_channel_mult=(1,),
        unet_attention_resolutions=(),
        unet_num_heads=2,
        training_objective="flow_matching",
    ).eval()

    x_0 = torch.full((1, 4, 4, 3), 2.5)
    mask = torch.zeros((1, 1, 4, 3))
    mask[:, :, 1:3, 1:2] = 1.0
    ctx = torch.zeros((1, 1, 8))

    called = {"convert": 0, "t_models": []}
    monkeypatch.setattr(
        model,
        "_convert_prediction",
        lambda *a, **k: called.__setitem__("convert", called["convert"] + 1),
    )

    def _fake_pred(x_t, t, context, **kwargs):
        called["t_models"].append(float(t[0].item()))
        return torch.zeros_like(x_t)

    monkeypatch.setattr(model, "_predict_noise_cfg", _fake_pred)

    out = model.inpaint(x_0=x_0, mask=mask, context=ctx, num_steps=4, noise_strength=0.5)
    assert called["convert"] == 0
    assert len(called["t_models"]) == 4
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:, :, 0, :], x_0[:, :, 0, :], atol=1e-6)


def test_consistency_lora_trainer_respects_teacher_latent_clip_bound():
    from src.train_lcm import ConsistencyLoRATrainer

    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=16,
        context_dim=8,
        num_timesteps=10,
        unet_channel_mult=(1,),
        unet_attention_resolutions=(),
        unet_num_heads=2,
        latent_clip_bound=5.0,
    )
    trainer = ConsistencyLoRATrainer.__new__(ConsistencyLoRATrainer)
    trainer.ode_teacher = model
    clamped = model._clamp_clean_latent(torch.tensor([2.5, -2.5, 9.0]))
    assert torch.allclose(clamped, torch.tensor([2.5, -2.5, 5.0]))


def test_enhanced_attention_block_forwards_graph_args_by_keyword():
    from src.core.graph_grid_attention import EnhancedAttentionBlock

    torch.manual_seed(0)
    block = EnhancedAttentionBlock(grid_dim=16, graph_dim=24, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 16, 4, 4)
    graph_nodes = torch.randn(2, 3, 24)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    node_positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]] * 2)
    node_tpe = torch.randn(2, 3, 8)
    dist = torch.tensor([[[0.0], [1.0], [2.0]]] * 2)
    node_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    out = block(
        x,
        graph_nodes=graph_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_positions=node_positions,
        node_tpe=node_tpe,
        current_node_distance=dist,
        node_mask=node_mask,
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_sinusoidal_position_encoding_small_dims_and_dynamic_growth():
    from src.core.graph_grid_attention import SinusoidalPositionEncoding2D

    for dim in (1, 2, 3, 4):
        enc = SinusoidalPositionEncoding2D(dim=dim, max_size=(4, 4))
        out = enc(torch.zeros(1, dim, 6, 7))
        assert out.shape == (1, dim, 6, 7)
        assert enc.pe.shape == (6, 7, dim)
        assert torch.isfinite(out).all()


def test_hedgehog_and_graph_node_pe_bfloat16_alignment():
    from src.core.attention_kernels import HedgehogFeatureMap
    from src.core.graph_grid_attention import GraphNodePositionEncoding

    hmap = HedgehogFeatureMap(num_heads=2, head_dim=8, feature_dim=4)
    out = hmap(torch.randn(1, 2, 5, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 2, 5, 8)

    gpe = GraphNodePositionEncoding(dim=16).to(dtype=torch.bfloat16)
    gout = gpe(torch.randn(2, 4, 16, dtype=torch.float32), node_positions=torch.zeros(2, 4, 2))
    assert gout.dtype == torch.bfloat16
    assert gout.shape == (2, 4, 16)


def test_cross_attention_learned_graphormer_and_empty_context():
    from src.core.latent_diffusion import CrossAttention

    ca = CrossAttention(query_dim=16, context_dim=16, num_heads=2).to(dtype=torch.bfloat16)
    ca.set_topology_refinement_mode("graphormer_learned")
    assert ca.graphormer_spatial_bias.weight.dtype == torch.bfloat16
    assert ca.graphormer_in_degree.weight.dtype == torch.bfloat16

    x = torch.randn(2, 6, 16, dtype=torch.bfloat16)
    empty_out = ca(x, torch.zeros(2, 0, 16, dtype=torch.bfloat16))
    assert empty_out.shape == x.shape
    assert torch.all(empty_out == 0)


def test_latent_diffusion_sample_bfloat16_preserves_dtype():
    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=16,
        context_dim=8,
        num_timesteps=4,
        unet_channel_mult=(1,),
        unet_attention_resolutions=(),
        unet_num_heads=2,
    ).to(dtype=torch.bfloat16).eval()

    ctx = torch.randn(1, 8, dtype=torch.float32)
    sample = model.sample(ctx, shape=(1, 4, 4, 3), sampler="diffusion")
    assert sample.dtype == torch.bfloat16
    assert sample.shape == (1, 4, 4, 3)


def test_global_stream_gin_and_dual_stream_bfloat16():
    from src.core.condition_encoder import (
        DualStreamConditionEncoder,
        GlobalStreamEncoder,
        HAS_TORCH_GEOMETRIC,
    )
    from src.core.definitions import GRAPH_EDGE_FEATURE_DIM, GRAPH_NODE_FEATURE_DIM

    g_enc = GlobalStreamEncoder(
        node_feature_dim=GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=GRAPH_EDGE_FEATURE_DIM,
        hidden_dim=16,
        output_dim=16,
        num_layers=1,
        gnn_type="gin",
    )
    if HAS_TORCH_GEOMETRIC:
        assert any(type(layer).__name__ == "GINConv" for layer in g_enc.gnn_layers)

    dual = DualStreamConditionEncoder(
        latent_dim=4,
        output_dim=16,
        hidden_dim=16,
        num_gnn_layers=1,
        num_attention_heads=2,
        gnn_type="gin",
        use_reference_room_maps=False,
    ).to(dtype=torch.bfloat16).eval()

    neighbors = {"N": torch.randn(1, 4, 4, 3), "S": None, "E": None, "W": None}
    boundary = torch.zeros(1, 8)
    pos = torch.zeros(1, 2)
    node_feats = torch.randn(3, GRAPH_NODE_FEATURE_DIM)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    out = dual(neighbors, boundary, pos, node_feats, edge_index)
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 16)


def test_logicnet_pathfinders_and_vqvae_bfloat16_compatibility():
    from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH
    from src.core.logic_net import (
        LearnableGridPathfinder,
        LogicNet,
        PerturbAndMAPGridPathfinder,
        SoftBellmanFordGridPathfinder,
        WalkabilityPredictor,
    )

    logits = torch.randn(1, 44, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.bfloat16)
    src = torch.zeros(1, 1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
    src[:, :, 1, 1] = 1.0
    goal = torch.zeros(1, 1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
    goal[:, :, 4, 4] = 1.0

    for pf in (
        SoftBellmanFordGridPathfinder(num_iterations=2, num_classes=44).to(dtype=torch.bfloat16),
        LearnableGridPathfinder(num_iterations=2, num_classes=44).to(dtype=torch.bfloat16),
        PerturbAndMAPGridPathfinder(num_iterations=2, num_classes=44, num_samples=1).to(dtype=torch.bfloat16),
    ):
        d = pf(logits, src.to(dtype=torch.bfloat16))
        assert d.shape == (1, 1, ROOM_HEIGHT, ROOM_WIDTH)

    wp = WalkabilityPredictor(num_classes=44).to(dtype=torch.bfloat16)
    assert wp(logits).dtype == torch.bfloat16

    ln = LogicNet(latent_dim=4, num_classes=44, hidden_dim=8, num_iterations=2).to(dtype=torch.bfloat16)
    scores = ln(logits, src, goal)
    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()

    vq = VectorQuantizer(num_embeddings=16, embedding_dim=8).eval()
    z_q, vq_loss, idx = vq(torch.randn(1, 8, 4, 4, dtype=torch.bfloat16))
    assert z_q.dtype == torch.bfloat16
    assert idx.shape == (1, 4, 4)
    assert torch.isfinite(vq_loss)


def test_validation_package_collision_alignment_consolidated():
    import numpy as np
    import src.validation as validation_pkg
    from src.validation.collision_alignment_validator import CollisionAlignmentValidator

    assert validation_pkg.CollisionAlignmentValidator is CollisionAlignmentValidator
    validator = CollisionAlignmentValidator()
    # Disconnected open door surrounded by walls triggers door + phantom mismatch on the same tile
    grid = np.full((5, 5), 2, dtype=np.int32)
    grid[1, 1] = 1
    grid[3, 3] = 10
    res = validator.validate_room(grid, start_pos=(1, 1))
    assert 0.0 <= res.alignment_score <= 1.0
    assert res.mismatched_pixels == 1
    # Out-of-bounds start_pos returns empty reachability safely without IndexError
    res_oob = validator.validate_room(grid, start_pos=(99, 99))
    assert res_oob.total_pixels == 25


def test_path_analyzer_never_returns_empty_failures_on_blocked_endpoints():
    import numpy as np
    from src.core.definitions import SEMANTIC_PALETTE
    from src.core.symbolic_refiner import PathAnalyzer, LearnedTileStatistics

    floor = int(SEMANTIC_PALETTE["FLOOR"])
    wall = int(SEMANTIC_PALETTE["WALL"])
    key = int(SEMANTIC_PALETTE["KEY_SMALL"])

    grid = np.full((5, 5), floor, dtype=np.int32)
    # Blocked start coordinate must report a FailurePoint rather than empty [].
    grid[0, 0] = wall
    analyzer = PathAnalyzer()
    failures = analyzer.analyze_grid(grid, start=(0, 0), goal=(4, 4))
    assert len(failures) >= 1
    assert failures[0].failure_type == "disconnected"
    assert (0, 0) in failures[0].blocking_tiles

    # Blocked goal coordinate must also report a FailurePoint.
    grid2 = np.full((5, 5), floor, dtype=np.int32)
    grid2[4, 4] = wall
    failures_goal = analyzer.analyze_grid(grid2, start=(0, 0), goal=(4, 4))
    assert len(failures_goal) >= 1
    assert (4, 4) in failures_goal[0].blocking_tiles

    # Observed singleton non-self-adjacent tile must still have a non-empty legal neighbor set.
    stats = LearnedTileStatistics()
    sample = np.full((3, 3), wall, dtype=np.int32)
    sample[1, 1] = key
    sample[1, 2] = floor
    stats.observe(sample)
    rules = stats.get_adjacency_rules(threshold=0.95)
    assert rules.get(key), "Thresholding must not leave an observed tile with an empty WFC domain"


def test_vqvae2_supports_fsq_and_guards_get_codebook():
    from src.core.vqvae import FSQuantizer, SemanticVQVAE2, create_vqvae

    model = create_vqvae(
        num_classes=44,
        latent_dim=16,
        hidden_dim=16,
        hidden_dims=[16, 32],
        num_res_blocks=1,
        architecture="vqvae2",
        quantizer_type="fsq",
        fsq_levels=[5, 5, 5],
        fsq_num_dims=3,
    )
    assert isinstance(model, SemanticVQVAE2)
    assert isinstance(model.top_quantizer, FSQuantizer)
    assert isinstance(model.bottom_quantizer, FSQuantizer)
    assert model.codebook_size == 125
    assert model.top_codebook_size == 125

    x = torch.randn(1, 44, 16, 11)
    recon, _vq_loss, losses = model(x)
    assert recon.shape == x.shape
    assert "indices" in losses

    decoded = model.decode_indices(losses["indices"], target_size=(16, 11))
    assert decoded.shape == x.shape

    with pytest.raises(RuntimeError, match="implicit scalar codebook"):
        model.get_codebook()

    with pytest.raises(ValueError, match="quantizer_type"):
        SemanticVQVAE2(num_classes=44, latent_dim=16, hidden_dim=16, quantizer_type="invalid")


def test_ml_heuristic_dtype_and_locked_door_cache_parity():
    import numpy as np
    from src.core.definitions import SEMANTIC_PALETTE
    from src.ml.heuristic_learning import (
        HeuristicNetwork,
        HeuristicTrainer,
        MLHeuristicAStar,
        TrainingExample,
    )

    class _DummyEnv:
        def __init__(self, grid, start_pos=(0, 0), goal_pos=(3, 3)):
            self.grid = grid
            self.height, self.width = grid.shape
            self.start_pos = start_pos
            self.goal_pos = goal_pos

    class _DummyState:
        def __init__(self, position, keys=1, opened_doors=None, collected_items=None):
            self.position = position
            self.keys = keys
            self.bomb_count = 0
            self.has_bomb = False
            self.has_boss_key = False
            self.has_item = True
            self.opened_doors = set(opened_doors or ())
            self.collected_items = set(collected_items or ())

    floor = int(SEMANTIC_PALETTE["FLOOR"])
    door_locked = int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    grid = np.full((4, 4), floor, dtype=np.int32)
    grid[1, 2] = door_locked
    grid[2, 2] = door_locked
    env = _DummyEnv(grid)

    trainer = HeuristicTrainer(4, 4)
    ex = TrainingExample(
        position=(1, 1),
        keys=1,
        has_bomb=False,
        has_boss_key=False,
        has_item=True,
        remaining_cost=4.0,
        opened_doors={(1, 2)},
        collected_items={(0, 1)},
    )
    trainer.train([ex] * 4, env, epochs=2, batch_size=2)
    trainer_feats = trainer.featurize_state(ex, env)

    solver = MLHeuristicAStar(env)
    solver.model = trainer.model.to(dtype=torch.bfloat16)
    state = _DummyState(position=(1, 1), keys=1, opened_doors={(1, 2)}, collected_items={(0, 1)})
    runtime_feats = solver._featurize_runtime_state(state)
    assert np.allclose(trainer_feats, runtime_feats, atol=1e-6)
    h = solver.heuristic(state)
    assert np.isfinite(h) and 0.0 <= h <= 4.0

    # Mixed-precision calibration & prediction must not fail on float64/bfloat16 models.
    net64 = HeuristicNetwork(4, 4).to(dtype=torch.float64)
    assert np.isfinite(net64.predict_cost(runtime_feats))


def test_graph_constraint_enforcer_virtual_start_and_mixed_topology():
    import numpy as np
    from src.generation.graph_constraint_enforcer import enforce_all_rooms, verify_topology_match

    tile_config = {
        "wall": 2,
        "floor": 1,
        "door": 10,
        "door_open": 10,
        "start": 21,
    }
    grid = np.full((6, 12), tile_config["wall"], dtype=np.int32)
    grid[1:5, 1:5] = tile_config["floor"]
    grid[1:5, 7:11] = tile_config["floor"]
    layout_map = {1: (0, 0, 5, 5), "room_b": (6, 0, 11, 5)}
    mission_graph = {
        "nodes": {
            "v_start": {"is_start": True, "is_virtual": True},
            1: {"label": "key"},
            "room_b": {"label": "goal"},
        },
        "edges": [
            ("v_start", 1, {"edge_type": "open"}),
            (1, "room_b", {"edge_type": "open"}),
            (1, "room_b", {"edge_type": "visual_link"}),
            (1, 1, {"edge_type": "open"}),
        ],
    }

    out = enforce_all_rooms(grid.copy(), mission_graph, layout_map, tile_config)
    # Input dict must not be mutated and adjacent physical room must receive start marker.
    assert "is_start" not in mission_graph["nodes"][1]
    assert int(np.sum(out == tile_config["start"])) >= 1

    # Non-physical edges & mixed int/str node IDs should match physical doors without TypeError.
    assert verify_topology_match(out, mission_graph, layout_map, tile_config) is True


def test_lora_layer_handles_mixed_precision_inputs():
    from src.optimization.lcm_lora import LoRALayer

    layer = LoRALayer(in_features=8, out_features=8, rank=2, alpha=4.0, dtype=torch.float32)
    with torch.no_grad():
        layer.lora_B.fill_(0.5)
    x_bf16 = torch.randn(2, 8, dtype=torch.bfloat16)
    y = layer(x_bf16)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()
