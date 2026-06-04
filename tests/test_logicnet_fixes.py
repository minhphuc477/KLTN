#!/usr/bin/env python
"""Focused validation tests for LogicNet guidance fixes."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.latent_diffusion import GradientGuidance
from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.logic_net import (
    DifferentiablePathfinder,
    LogicNet,
    PerturbAndMAPGridPathfinder,
    SemanticEdgeEncoder,
    SoftBellmanFordGridPathfinder,
    ValueIterationGridPathfinder,
    WalkabilityPredictor,
)
from src.pipeline.graph_features import extract_node_feature_vector
from src.pipeline.spatial_utils import parse_label_tokens


def test_logicnet_gradient_magnitude():
    """LogicNet should produce a nonzero latent gradient with dungeon node routing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logic_net = LogicNet(
        latent_dim=64,
        num_tile_classes=5,
    ).to(device)
    logic_net.eval()

    b, c, h, w = 4, 64, 32, 32
    z = torch.randn(b, c, h, w, device=device, dtype=torch.float32, requires_grad=True)

    graph_data = {
        "node_features": torch.randn(10, 16, device=device, dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], device=device, dtype=torch.long),
        "edge_features": torch.randn(3, 8, device=device, dtype=torch.float32),
        "graph_scope": "dungeon",
        "current_node_idx": torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long),
        "start_idx": 0,
        "target_idx": 9,
    }

    with torch.enable_grad():
        loss = logic_net(z, graph_data=graph_data)
        loss_scalar = loss[0].mean() if isinstance(loss, tuple) else loss.mean()

    assert loss_scalar.requires_grad
    loss_scalar.backward()
    grad_norm = z.grad.norm().item() if z.grad is not None else 0.0
    assert grad_norm > 1e-6


def test_logicnet_walkable_ids_derive_from_semantic_palette():
    expected = {
        int(SEMANTIC_PALETTE[name])
        for name in (
            "FLOOR",
            "DOOR_OPEN",
            "DOOR_LOCKED",
            "DOOR_BOMB",
            "DOOR_PUZZLE",
            "DOOR_BOSS",
            "DOOR_SOFT",
            "START",
            "TRIFORCE",
            "KEY_SMALL",
            "KEY_BOSS",
            "KEY_ITEM",
            "ITEM_MINOR",
            "ELEMENT_FLOOR",
            "STAIR",
            "ENEMY",
            "BOSS",
            "PUZZLE",
        )
    }

    assert set(WalkabilityPredictor.WALKABLE_IDS) == expected
    assert set(SoftBellmanFordGridPathfinder.WALKABLE_IDS) == expected


def test_logicnet_guidance_schedule_is_late_process_active():
    """LogicNet is a clean-latent solver, so scheduled guidance should be strongest near t=0."""
    guidance = GradientGuidance(logic_net=None, guidance_scale=1.0, active_fraction=0.3)

    assert guidance._scheduled_scale(t=999, num_timesteps=1000) == 0.0
    assert guidance._scheduled_scale(t=700, num_timesteps=1000) == 0.0
    assert guidance._scheduled_scale(t=100, num_timesteps=1000) > 0.0
    assert guidance._scheduled_scale(t=0, num_timesteps=1000) == pytest.approx(1.0)


def test_logicnet_temperature_updates_grid_pathfinder():
    logic_net = LogicNet(
        latent_dim=4,
        num_tile_classes=5,
        grid_pathfinder_type="bellman_ford",
        initial_temperature=1.0,
        final_temperature=0.1,
    )

    logic_net.update_temperature(1.0)

    assert logic_net.graph_pathfinder.temperature == pytest.approx(0.1)
    assert logic_net.grid_pathfinder.pathfinder.temperature == pytest.approx(0.1)


def test_differentiable_pathfinder_grid_uses_edge_weights():
    pathfinder = DifferentiablePathfinder(num_iterations=8, temperature=0.05, inf_distance=20.0)
    walkability = torch.ones(1, 3, 4)
    source = torch.zeros(1, 3, 4)
    source[:, 1, 0] = 1.0
    unit_cost = torch.ones_like(walkability)
    high_cost = unit_cost.clone()
    high_cost[:, 1, 1] = 8.0

    distances_unit = pathfinder(walkability, unit_cost, source)
    distances_high = pathfinder(walkability, high_cost, source)

    assert distances_high[0, 1, 1] > distances_unit[0, 1, 1] + 1.0


def test_differentiable_pathfinder_graph_soft_update_keeps_edge_weight_gradients():
    pathfinder = DifferentiablePathfinder(num_iterations=3, temperature=0.5, inf_distance=20.0)
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    edge_weights = torch.tensor(
        [
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    source = torch.tensor([1.0, 0.0, 0.0])

    distances = pathfinder(adjacency, edge_weights, source)
    distances[2].backward()

    assert edge_weights.grad is not None
    assert torch.isfinite(edge_weights.grad).all()
    assert float(edge_weights.grad[0, 2].abs().item()) > 0.0


def test_logicnet_compatibility_mode_routes_through_grid_pathfinder(monkeypatch):
    logic_net = LogicNet(latent_dim=4, num_tile_classes=5)

    def _fail_graph_pathfinder(*_args, **_kwargs):
        raise AssertionError("compatibility mode should not call graph_pathfinder")

    monkeypatch.setattr(logic_net.graph_pathfinder, "forward", _fail_graph_pathfinder)

    tile_logits = torch.zeros(1, 5, ROOM_HEIGHT, ROOM_WIDTH)
    tile_logits[:, 1] = 8.0
    start = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH)
    start[:, 0, 0] = 1.0
    goal = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH)
    goal[:, ROOM_HEIGHT - 1, ROOM_WIDTH - 1] = 1.0

    scores = logic_net(tile_logits, start, goal)

    assert tuple(scores.shape) == (1,)
    assert torch.isfinite(scores).all()


def test_logicnet_without_topology_uses_single_cell_source_not_all_doors():
    logic_net = LogicNet(latent_dim=4, num_tile_classes=5)
    z = torch.zeros(2, 5, 16, 11)
    z[:, 1] = 8.0

    loss, info = logic_net(z, graph_data=None)

    assert loss.ndim == 0
    assert info["source_mask_mode"] == "single_walkable_cell"
    source_mask = logic_net._create_single_cell_source_mask(info["walkability"])
    assert torch.allclose(source_mask.sum(dim=(1, 2, 3)), torch.ones(2))


def test_logicnet_global_room_passability_preserves_index_copy_gradient():
    logic_net = LogicNet(latent_dim=4, num_tile_classes=5)
    room_passability = torch.tensor([0.25], requires_grad=True)
    graph_data = {
        "adjacency": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "edge_weights": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "node_features": torch.zeros(2, 4),
        "start_idx": 0,
        "target_idx": 1,
        "current_node_idx": 0,
    }

    total, _reach, _lock, _info = logic_net._compute_one_global_graph_loss(
        node_count=2,
        edge_index=None,
        adjacency=graph_data["adjacency"],
        edge_weights=graph_data["edge_weights"],
        edge_features=None,
        edge_attr=None,
        node_features=graph_data["node_features"],
        node_mask=None,
        start_idx=0,
        target_idx=1,
        key_lock_pairs=[],
        current_node_idx=0,
        room_passability=room_passability,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    grad = torch.autograd.grad(total, room_passability, allow_unused=False)[0]

    assert grad is not None
    assert torch.isfinite(grad).all()


def test_semantic_edge_encoder_defaults_and_receives_gradients():
    encoder = SemanticEdgeEncoder(num_edge_types=8)
    edge_attr = torch.tensor([0, 1, 2, 4, 7], dtype=torch.long)

    penalties = encoder(edge_attr)
    penalties.sum().backward()

    assert penalties.tolist() == pytest.approx([0.0, 1.0, 0.5, 2.0, 0.5])
    assert encoder.residual_logits.grad is not None
    assert encoder.residual_logits.grad.abs().sum().item() > 0.0


def test_vin_pathfinder_is_selectable_and_backpropagates():
    net = LogicNet(
        latent_dim=8,
        num_classes=5,
        num_iterations=3,
        grid_pathfinder_type="vin",
    )
    assert net.grid_pathfinder_type == "vin"
    assert isinstance(net.grid_pathfinder, ValueIterationGridPathfinder)

    room_logits = torch.randn(2, 5, 16, 11, requires_grad=True)
    source = torch.zeros(2, 1, 16, 11)
    source[:, :, 1, 1] = 1.0
    walkability = torch.sigmoid(torch.randn(2, 1, 16, 11))
    distances = net.grid_pathfinder(room_logits, source, walkability)
    assert tuple(distances.shape) == (2, 1, 16, 11)
    distances.mean().backward()
    assert room_logits.grad is not None
    assert room_logits.grad.abs().sum().item() > 0.0


def test_perturb_and_map_pathfinder_is_selectable_and_backpropagates():
    net = LogicNet(
        latent_dim=8,
        num_classes=5,
        num_iterations=3,
        grid_pathfinder_type="perturb_and_map",
    )
    assert net.grid_pathfinder_type == "perturb_and_map"
    assert isinstance(net.grid_pathfinder, PerturbAndMAPGridPathfinder)

    room_logits = torch.randn(1, 5, 6, 6, requires_grad=True)
    source = torch.zeros(1, 1, 6, 6)
    source[:, :, 0, 0] = 1.0
    walkability = torch.full((1, 1, 6, 6), 0.75, requires_grad=True)
    distances = net.grid_pathfinder(room_logits, source, walkability)
    assert tuple(distances.shape) == (1, 1, 6, 6)
    assert torch.isfinite(distances[:, :, 0, 0]).all()
    distances.mean().backward()
    assert walkability.grad is not None
    assert walkability.grad.abs().sum().item() > 0.0


def test_logicnet_perturb_and_map_propagates_to_latents_and_classifier():
    net = LogicNet(
        latent_dim=8,
        num_classes=5,
        hidden_dim=16,
        num_iterations=3,
        grid_pathfinder_type="perturb_and_map",
    )
    z = torch.randn(1, 8, 6, 6, requires_grad=True)

    loss, info = net(z, graph_data=None)
    assert torch.isfinite(loss)
    assert net.grid_pathfinder_type == "perturb_and_map"
    assert isinstance(info, dict)

    loss.backward()
    classifier_grad = sum(
        float(param.grad.detach().abs().sum().item())
        for param in net.tile_classifier.parameters()
        if param.grad is not None
    )
    assert z.grad is not None
    assert z.grad.abs().sum().item() > 0.0
    assert classifier_grad > 0.0


def test_logicnet_edge_attr_penalties_follow_valid_edge_filter():
    logic_net = LogicNet(latent_dim=4, num_tile_classes=5)
    edge_index = torch.tensor([[0, 99, 1], [1, 2, 2]], dtype=torch.long)
    edge_attr = torch.tensor([1, 4, 2], dtype=torch.long)

    adj, weights = logic_net._build_adjacency_and_weights(
        node_count=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    assert adj[0, 1].item() == pytest.approx(1.0)
    assert adj[1, 2].item() == pytest.approx(1.0)
    assert weights[0, 1].item() == pytest.approx(2.0)
    assert weights[1, 2].item() == pytest.approx(1.5)
    assert weights[0, 2].item() == pytest.approx(0.0)


def test_logicnet_skips_dungeon_scope_graph_loss_without_full_node_passability():
    logic_net = LogicNet(latent_dim=4, num_tile_classes=5)
    room_passability = torch.tensor([0.25, 0.5], requires_grad=True)
    graph_data = {
        "graph_scope": "dungeon",
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "node_features": torch.zeros(4, 4),
        "start_idx": 0,
        "target_idx": 3,
        "current_node_idx": torch.tensor([0, 1]),
    }

    total, reach, lock, info = logic_net._compute_global_graph_losses(
        graph_data,
        room_passability=room_passability,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert total.item() == pytest.approx(0.0)
    assert reach.item() == pytest.approx(0.0)
    assert lock.item() == pytest.approx(0.0)
    assert info["global_graph_skipped"] == "dungeon_scope_requires_full_room_passability"


def test_graph_feature_roles_include_type_fields():
    """Mission graphs that use type/room_type START/GOAL should still feed LogicNet roles."""

    def _coerce_bool(value):
        return bool(value)

    def _coerce_difficulty(_value):
        return 0.5

    goal_features = extract_node_feature_vector(
        {"room_type": "goal"},
        node_dim=14,
        device=torch.device("cpu"),
        parse_label_tokens=parse_label_tokens,
        coerce_bool=_coerce_bool,
        coerce_difficulty=_coerce_difficulty,
    )
    start_features = extract_node_feature_vector(
        {"type": "START"},
        node_dim=14,
        device=torch.device("cpu"),
        parse_label_tokens=parse_label_tokens,
        coerce_bool=_coerce_bool,
        coerce_difficulty=_coerce_difficulty,
    )

    assert float(goal_features[3].item()) == 1.0
    assert float(start_features[11].item()) == 1.0


if __name__ == "__main__":
    try:
        test_logicnet_gradient_magnitude()
        test_logicnet_guidance_schedule_is_late_process_active()
        test_graph_feature_roles_include_type_fields()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
