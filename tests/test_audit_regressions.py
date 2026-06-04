from pathlib import Path

import numpy as np
import torch
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.config_system import merge_config
from src.core.graph_grid_attention import GraphToGridCrossAttention
from src.core.latent_diffusion import CrossAttention, ResBlock, create_latent_diffusion
from src.core.logic_net import LogicNet, SoftBellmanFordGridPathfinder
from src.evaluation import compare_tile_pattern_distributions
from src.pipeline.config_schema import validate_config_payload
from src.pipeline.evaluation import evaluate_generated_dungeon
from src.train_diffusion import DiffusionTrainingConfig
from src.utils.attention_visualization import save_attention_map_images


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


def test_context_topology_refinement_uses_batched_padded_adjacency():
    attention = CrossAttention(
        query_dim=8,
        context_dim=8,
        num_heads=2,
        topology_refinement_mode="lightweight",
    )
    context = torch.randn(2, 4, 8)
    edge_index = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3]],
            [[0, 1, -1], [1, 0, -1]],
        ],
        dtype=torch.long,
    )
    node_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)

    refined = attention._refine_context_topology(
        context,
        edge_index=edge_index,
        node_mask=node_mask,
    )

    assert refined.shape == context.shape
    assert torch.isfinite(refined).all()
    norm_adj, valid = attention._batched_normalized_adjacency(
        batch_size=2,
        seq_len=4,
        edge_index=edge_index,
        node_mask=node_mask,
        device=context.device,
        dtype=context.dtype,
    )
    assert valid.tolist() == [[True, True, True, True], [True, True, False, False]]
    assert torch.all(norm_adj[1, 2:] == 0)
    assert torch.all(norm_adj[1, :, 2:] == 0)


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
    assert distances[0, 0, 8, 5].item() == 0.0
    assert distances[0, 0, 0, 0].item() > distances[0, 0, 8, 5].item()


def test_logic_net_defaults_to_bellman_ford_grid_pathfinder():
    logic_net = LogicNet(latent_dim=8, hidden_dim=16, num_classes=44, num_iterations=3)

    assert logic_net.grid_pathfinder_type == "bellman_ford"
    assert isinstance(logic_net.grid_pathfinder, SoftBellmanFordGridPathfinder)


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


def test_pydantic_config_schema_rejects_invalid_diffusion_choice():
    with pytest.raises(ValidationError):
        validate_config_payload({"diffusion": {"logic_grid_pathfinder": "invalid"}})


@pytest.mark.parametrize(
    "pathfinder",
    ["soft-bellman-ford", "soft_bellman_ford", "perturb-and-map", "value_iteration"],
)
def test_pydantic_config_schema_accepts_public_logic_pathfinder_aliases(pathfinder):
    config = validate_config_payload({"diffusion": {"logic_grid_pathfinder": pathfinder}})

    assert config["diffusion"]["logic_grid_pathfinder"] == pathfinder


def test_pydantic_config_schema_returns_cross_field_normalization():
    config = validate_config_payload(
        {"diffusion": {"logic_net_enabled": False, "logic_net_trainable": True}}
    )

    assert config["diffusion"]["logic_net_enabled"] is False
    assert config["diffusion"]["logic_net_trainable"] is False
    assert config["diffusion"]["guidance_scale"] == 0.0
    assert config["diffusion"]["alpha_logic"] == 0.0
    assert config["diffusion"]["alpha_logic_tile"] == 0.0


def test_tile_pattern_distribution_identical_samples_have_zero_js():
    reference = [np.array([[1, 1, 2], [1, 2, 2], [3, 3, 2]], dtype=np.int64)]
    generated = [reference[0].copy()]

    result = compare_tile_pattern_distributions(generated, reference, pattern_size=2)

    assert result.js_divergence == 0.0
    assert result.total_variation == 0.0
    assert result.pattern_coverage == 1.0


def test_pipeline_evaluation_reports_tile_pattern_distribution_without_map_elites():
    grid = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 2]], dtype=np.int64)
    pipeline = SimpleNamespace(
        map_elites=None,
        evaluation_reference_rooms=[grid.copy()],
        tile_pattern_size=2,
    )

    result = evaluate_generated_dungeon(
        pipeline,
        dungeon_grid=grid.copy(),
        mission_graph_physical=SimpleNamespace(),
        enable_map_elites=False,
    )

    assert result is not None
    assert result["tile_pattern_js_divergence"] == 0.0
    assert result["tile_pattern_coverage"] == 1.0


def test_attention_visualization_saves_numpy_and_heatmap_artifacts(tmp_path):
    attention = torch.tensor(
        [[[[0.75, 0.25], [0.50, 0.50]], [[0.25, 0.75], [0.10, 0.90]]]],
        dtype=torch.float32,
    )

    result = save_attention_map_images(
        attention,
        tmp_path,
        prefix="graph_attention",
        node_labels=["start", "goal"],
    )

    assert Path(result["npy"]).exists()
    assert result["shape"] == (1, 2, 2, 2)
    assert len(result["pngs"]) == 2
    assert all(Path(path).exists() for path in result["pngs"])


def test_resblock_groupnorm_selection_supports_tiny_channel_widths():
    assert ResBlock.num_groups(3) == 3
    assert ResBlock.num_groups(1) == 1


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
