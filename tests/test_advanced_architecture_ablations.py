import math

import numpy as np
import pytest
import torch

import src.core.latent_diffusion as latent_diffusion_module
from src.config_system import merge_config
from src.core.latent_diffusion import create_latent_diffusion
from src.evaluation.perturb_and_map import perturb_and_map_reachability
from src.train_diffusion import (
    DiffusionTrainingConfig,
    diffusion_training_kwargs_from_resolved_config,
)


def _enable_fresh_denoiser_signal(denoiser, *, self_gate: bool = False, cross_gate: bool = False) -> None:
    """Open zero-initialized ablation heads only when a test needs observable routing."""
    with torch.no_grad():
        output_proj = getattr(denoiser, "output_proj", None)
        if output_proj is not None:
            torch.nn.init.normal_(output_proj.weight, std=0.02)
            if output_proj.bias is not None:
                output_proj.bias.zero_()
        out_proj = getattr(denoiser, "out_proj", None)
        if out_proj is not None:
            torch.nn.init.normal_(out_proj.weight, std=0.02)
            if out_proj.bias is not None:
                out_proj.bias.zero_()
        for block in getattr(denoiser, "blocks", []):
            bias = block.adaLN[-1].bias
            hidden = int(block.norm1.normalized_shape[0])
            if self_gate:
                bias[2 * hidden:3 * hidden].fill_(1.0)
            if cross_gate:
                bias[5 * hidden:6 * hidden].fill_(1.0)


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


def test_dit_adaln_zero_block_starts_as_identity():
    from src.core.latent_diffusion import DiTBlock

    torch.manual_seed(101)
    block = DiTBlock(hidden_dim=16, cond_dim=16, num_heads=4, dropout=0.0)
    x = torch.randn(2, 5, 16)
    cond = torch.randn(2, 16)
    context = torch.randn(2, 3, 16)

    out = block(x, cond, context_tokens=context)

    assert torch.allclose(out, x, atol=1e-6)


def test_fresh_diffusion_denoiser_heads_start_as_zero_predictors():
    from src.core.latent_diffusion import DiTDenoiser, UNetDenoiser

    torch.manual_seed(102)
    x = torch.randn(2, 4, 4, 4)
    t = torch.tensor([0, 3], dtype=torch.long)
    context = torch.randn(2, 8)
    unet = UNetDenoiser(
        in_channels=4,
        model_channels=8,
        out_channels=4,
        context_dim=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_resolutions=(),
        num_heads=2,
        dropout=0.0,
    )
    dit = DiTDenoiser(
        in_channels=4,
        model_channels=8,
        out_channels=4,
        context_dim=8,
        depth=1,
        patch_size=1,
        num_heads=2,
        dropout=0.0,
    )

    assert torch.allclose(unet(x, t, context), torch.zeros_like(x), atol=1e-6)
    assert torch.allclose(dit(x, t, context), torch.zeros_like(x), atol=1e-6)


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
    _enable_fresh_denoiser_signal(model.denoiser, cross_gate=True)
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


def test_graphormer_topology_bias_encodes_shortest_path_distance():
    from src.core.latent_diffusion import CrossAttention

    attention = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="graphormer",
        dropout=0.0,
    )
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    node_mask = torch.ones(1, 3, dtype=torch.bool)
    norm_adj, valid = attention._batched_normalized_adjacency(
        batch_size=1,
        seq_len=3,
        edge_index=edge_index,
        node_mask=node_mask,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    bias = attention._shortest_path_attention_bias(norm_adj > 0.0, valid, dtype=torch.float32)

    assert bias.shape == (1, 3, 3)
    assert bias[0, 0, 0].item() == pytest.approx(0.0)
    assert bias[0, 0, 1].item() == pytest.approx(-1.0)
    assert bias[0, 0, 2].item() == pytest.approx(-2.0)


def test_graphormer_topology_refinement_reports_cost_metrics_against_gat2():
    from src.core.latent_diffusion import CrossAttention

    gat2_metrics = CrossAttention.topology_refinement_metrics(num_nodes=6, num_edges=5, mode="gat2")
    sparse_metrics = CrossAttention.topology_refinement_metrics(
        num_nodes=6,
        num_edges=5,
        mode="sparse_edge",
    )
    graphormer_metrics = CrossAttention.topology_refinement_metrics(
        num_nodes=6,
        num_edges=5,
        mode="graphormer",
    )
    lightweight_metrics = CrossAttention.topology_refinement_metrics(
        num_nodes=6,
        num_edges=5,
        mode="lightweight",
    )

    assert gat2_metrics["attention_pairs"] == pytest.approx(36.0)
    assert gat2_metrics["shortest_path_bias_ops"] == pytest.approx(0.0)
    assert sparse_metrics["attention_pairs"] == pytest.approx(16.0)
    assert sparse_metrics["relative_attention_pairs_to_gat2"] < gat2_metrics["relative_attention_pairs_to_gat2"]
    assert sparse_metrics["shortest_path_bias_ops"] == pytest.approx(0.0)
    assert graphormer_metrics["attention_pairs"] == pytest.approx(gat2_metrics["attention_pairs"])
    assert graphormer_metrics["shortest_path_bias_ops"] == pytest.approx(216.0)
    assert lightweight_metrics["attention_pairs"] == pytest.approx(0.0)
    assert lightweight_metrics["message_pairs"] == pytest.approx(11.0)


def test_learned_graphormer_uses_centrality_spatial_and_edge_encodings():
    from src.core.latent_diffusion import CrossAttention

    torch.manual_seed(29)
    attention = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="graphormer_learned",
        dropout=0.0,
    )
    context = torch.randn(1, 3, 16)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    node_mask = torch.ones(1, 3, dtype=torch.bool)

    open_edges = torch.tensor([0, 0], dtype=torch.long)
    boss_edges = torch.tensor([4, 4], dtype=torch.long)
    out_open = attention._refine_context_topology(
        context,
        edge_index=edge_index,
        edge_attr=open_edges,
        node_mask=node_mask,
    )
    out_boss = attention._refine_context_topology(
        context,
        edge_index=edge_index,
        edge_attr=boss_edges,
        node_mask=node_mask,
    )
    out_open.sum().backward()

    assert tuple(out_open.shape) == tuple(context.shape)
    assert torch.isfinite(out_open).all()
    assert not torch.allclose(out_open, out_boss)
    assert attention.graphormer_spatial_bias is not None
    assert attention.graphormer_edge_bias is not None
    assert attention.graphormer_in_degree is not None
    assert attention.graphormer_out_degree is not None
    assert attention.graphormer_spatial_bias.weight.grad is not None
    assert attention.graphormer_edge_bias.weight.grad is not None


def test_learned_graphormer_centrality_excludes_synthetic_self_loops():
    from src.core.latent_diffusion import CrossAttention

    torch.manual_seed(30)
    attention = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="graphormer_learned",
        dropout=0.0,
    )
    context = torch.randn(1, 2, 16)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    node_mask = torch.ones(1, 2, dtype=torch.bool)

    out = attention._refine_context_topology(
        context,
        edge_index=edge_index,
        node_mask=node_mask,
    )
    out.sum().backward()

    assert attention.graphormer_in_degree is not None
    assert attention.graphormer_out_degree is not None
    assert attention.graphormer_in_degree.weight.grad[0].abs().sum() > 0
    assert attention.graphormer_out_degree.weight.grad[0].abs().sum() > 0
    assert attention.graphormer_in_degree.weight.grad[1].abs().sum() == pytest.approx(0.0)
    assert attention.graphormer_out_degree.weight.grad[1].abs().sum() == pytest.approx(0.0)


def test_learned_graphormer_directed_variant_respects_edge_direction():
    from src.core.latent_diffusion import CrossAttention

    torch.manual_seed(38)
    directed = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="graphormer_learned_directed",
        dropout=0.0,
    )
    undirected = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="graphormer_learned",
        dropout=0.0,
    )
    undirected.load_state_dict(directed.state_dict())
    context = torch.randn(1, 3, 16)
    one_way = torch.tensor([[0], [1]], dtype=torch.long)
    node_mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        out_directed = directed._refine_context_topology(context, edge_index=one_way, node_mask=node_mask)
        out_undirected = undirected._refine_context_topology(context, edge_index=one_way, node_mask=node_mask)

    assert torch.isfinite(out_directed).all()
    assert not torch.allclose(out_directed, out_undirected)


def test_sparse_edge_topology_refinement_mode_runs_as_large_graph_ablation():
    from src.core.latent_diffusion import CrossAttention

    torch.manual_seed(23)
    attention = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="sparse_edge",
        dropout=0.0,
    )
    context = torch.randn(1, 4, 16)
    chain_edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    skip_edges = torch.tensor([[0, 3], [3, 1]], dtype=torch.long)
    node_mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        out_chain = attention._refine_context_topology(context, edge_index=chain_edges, node_mask=node_mask)
        out_skip = attention._refine_context_topology(context, edge_index=skip_edges, node_mask=node_mask)

    assert tuple(out_chain.shape) == tuple(context.shape)
    assert torch.isfinite(out_chain).all()
    assert not torch.allclose(out_chain, out_skip)


def test_sparse_directed_topology_refinement_respects_edge_direction():
    from src.core.latent_diffusion import CrossAttention

    torch.manual_seed(31)
    directed = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="sparse_directed",
        dropout=0.0,
    )
    undirected = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="sparse_edge",
        dropout=0.0,
    )
    undirected.load_state_dict(directed.state_dict())
    context = torch.randn(1, 3, 16)
    one_way = torch.tensor([[0], [1]], dtype=torch.long)
    node_mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        out_directed = directed._refine_context_topology(context, edge_index=one_way, node_mask=node_mask)
        out_undirected = undirected._refine_context_topology(context, edge_index=one_way, node_mask=node_mask)

    assert not torch.allclose(out_directed, out_undirected)


def test_sparse_semantic_topology_refinement_uses_edge_attr_as_ablation():
    from src.core.latent_diffusion import CrossAttention

    torch.manual_seed(37)
    semantic = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="sparse_directed_semantic",
        dropout=0.0,
    )
    blind = CrossAttention(
        query_dim=16,
        context_dim=16,
        num_heads=4,
        topology_refinement_mode="sparse_directed",
        dropout=0.0,
    )
    blind.load_state_dict(semantic.state_dict())
    context = torch.randn(1, 3, 16)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    open_edges = torch.tensor([0, 0], dtype=torch.long)
    locked_edges = torch.tensor([4, 4], dtype=torch.long)
    node_mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        out_open = semantic._refine_context_topology(
            context,
            edge_index=edge_index,
            edge_attr=open_edges,
            node_mask=node_mask,
        )
        out_locked = semantic._refine_context_topology(
            context,
            edge_index=edge_index,
            edge_attr=locked_edges,
            node_mask=node_mask,
        )
        out_blind_open = blind._refine_context_topology(
            context,
            edge_index=edge_index,
            edge_attr=open_edges,
            node_mask=node_mask,
        )
        out_blind_locked = blind._refine_context_topology(
            context,
            edge_index=edge_index,
            edge_attr=locked_edges,
            node_mask=node_mask,
        )

    assert not torch.allclose(out_open, out_locked)
    assert torch.allclose(out_blind_open, out_blind_locked)


def test_latent_diffusion_graphormer_topology_refinement_mode_runs_as_ablation():
    torch.manual_seed(17)
    model = create_latent_diffusion(
        latent_dim=8,
        model_channels=16,
        context_dim=16,
        denoiser_backbone="dit",
        topology_refinement_mode="graphormer",
        dit_depth=1,
        dit_patch_size=1,
        dit_mlp_ratio=2.0,
        num_timesteps=8,
        cfg_dropout_prob=0.0,
        unet_num_heads=4,
        unet_dropout=0.0,
    )
    assert model.get_topology_refinement_mode() == "graphormer"

    x_t = torch.randn(1, 8, 3, 4)
    t = torch.tensor([3], dtype=torch.long)
    context = torch.randn(1, 3, 16)
    graph_data = {
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "node_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    with torch.no_grad():
        out = model.denoiser(
            x_t,
            t,
            context,
            context_edge_index=graph_data["edge_index"],
            context_node_mask=graph_data["node_mask"],
        )

    assert tuple(out.shape) == tuple(x_t.shape)
    assert torch.isfinite(out).all()


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
    _enable_fresh_denoiser_signal(model.denoiser)
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
    _enable_fresh_denoiser_signal(model.denoiser, cross_gate=True)
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
    _enable_fresh_denoiser_signal(model.denoiser, self_gate=True)
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
    _enable_fresh_denoiser_signal(model.denoiser)
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


def test_unet_gradient_checkpointing_invokes_activation_checkpoint(monkeypatch):
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
    )
    for module in model.denoiser.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = True
    model.train()

    calls = 0

    def _counting_checkpoint(function, *args, **kwargs):
        _ = kwargs
        nonlocal calls
        calls += 1
        return function(*args)

    monkeypatch.setattr(latent_diffusion_module, "activation_checkpoint", _counting_checkpoint)
    out = model.denoiser(
        torch.randn(1, 4, 2, 2, requires_grad=True),
        torch.tensor([1], dtype=torch.long),
        torch.randn(1, 1, 8),
    )
    out.mean().backward()

    assert calls > 0


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
