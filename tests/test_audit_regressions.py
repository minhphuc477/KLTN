import numpy as np
import torch

from src.config_system import merge_config
from src.core.graph_grid_attention import GraphToGridCrossAttention
from src.core.latent_diffusion import create_latent_diffusion
from src.core.logic_net import LogicNet, SoftBellmanFordGridPathfinder
from src.evaluation import compare_tile_pattern_distributions
from src.train_diffusion import DiffusionTrainingConfig


def test_graph_to_grid_attention_can_capture_softmax_maps():
    module = GraphToGridCrossAttention(
        grid_dim=8,
        graph_dim=6,
        num_heads=2,
        dropout=0.0,
        attention_mode="softmax",
    )
    module.eval()
    module.set_attention_capture(True)

    grid = torch.randn(1, 8, 4, 3)
    graph_nodes = torch.randn(1, 5, 6)
    output = module(grid, graph_nodes)
    attention = module.get_last_attention_map()

    assert output.shape == grid.shape
    assert attention is not None
    assert attention.shape == (1, 4, 3, 5)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(1, 4, 3), atol=1e-5)


def test_logic_net_supports_explicit_bellman_ford_grid_pathfinder():
    logic_net = LogicNet(
        latent_dim=8,
        hidden_dim=16,
        num_classes=44,
        num_iterations=3,
        grid_pathfinder_type="bellman_ford",
    )

    assert logic_net.grid_pathfinder_type == "bellman_ford"
    assert isinstance(logic_net.grid_pathfinder, SoftBellmanFordGridPathfinder)

    room_grid = torch.zeros(1, 44, 16, 11)
    room_grid[:, 1, :, :] = 8.0
    source_mask = torch.zeros(1, 1, 16, 11)
    source_mask[:, :, 8, 5] = 1.0

    distances = logic_net.grid_pathfinder(room_grid, source_mask)
    assert distances.shape == (1, 1, 16, 11)
    assert torch.isfinite(distances).all()


def test_disable_logic_net_config_zeroes_guidance_and_logic_loss():
    config = DiffusionTrainingConfig(
        logic_net_enabled=False,
        logic_net_trainable=True,
        guidance_scale=1.0,
        alpha_logic=0.1,
    )

    assert config.logic_net_enabled is False
    assert config.logic_net_trainable is False
    assert config.guidance_scale == 0.0
    assert config.alpha_logic == 0.0


def test_config_system_disabling_logic_net_forces_non_trainable():
    config = merge_config(
        cli_overrides={
            "diffusion": {
                "logic_net_enabled": False,
                "logic_net_trainable": True,
                "logic_grid_pathfinder": "bellman_ford",
            }
        }
    )

    assert config["diffusion"]["logic_net_enabled"] is False
    assert config["diffusion"]["logic_net_trainable"] is False
    assert config["diffusion"]["logic_grid_pathfinder"] == "bellman_ford"


def test_tile_pattern_distribution_identical_samples_have_zero_js():
    reference = [np.array([[1, 1, 2], [1, 2, 2], [3, 3, 2]], dtype=np.int64)]
    generated = [reference[0].copy()]

    result = compare_tile_pattern_distributions(generated, reference, pattern_size=2)

    assert result.js_divergence == 0.0
    assert result.total_variation == 0.0
    assert result.pattern_coverage == 1.0


def test_latent_diffusion_compile_for_inference_is_opt_in(monkeypatch):
    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=8,
        context_dim=8,
        num_timesteps=4,
    )
    calls = []

    def fake_compile(module, mode=None):
        calls.append((module, mode))
        return module

    monkeypatch.setattr(torch, "compile", fake_compile, raising=False)

    compiled = model.compile_for_inference(mode="reduce-overhead")

    assert compiled is True
    assert model._compiled_for_inference is True
    assert calls and calls[0][1] == "reduce-overhead"
