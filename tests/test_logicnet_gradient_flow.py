"""Gradient-flow probes for LogicNet's measurable training signal."""

import pytest
import torch

from src.core.definitions import SEMANTIC_PALETTE
from src.core.logic_net import LogicNet, WalkabilityPredictor


@pytest.mark.parametrize("pathfinder", ["bellman_ford", "cnn"])
def test_logicnet_grid_loss_backpropagates_to_latent(pathfinder):
    net = LogicNet(
        latent_dim=8,
        num_classes=44,
        hidden_dim=16,
        num_iterations=3,
        grid_pathfinder_type=pathfinder,
    )
    z = torch.randn(2, 8, 4, 4, requires_grad=True)

    loss, info = net(z, graph_data=None)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(info["grid_reach_loss"])
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum().item()) > 1e-8


def test_logicnet_graph_loss_backpropagates_to_room_passability():
    net = LogicNet(latent_dim=8, num_classes=44, hidden_dim=16, num_iterations=3)
    room_passability = torch.tensor([0.25, 0.9, 0.6], requires_grad=True)
    node_features = torch.zeros(3, 6)
    node_features[2, 3] = 1.0

    total, reach, _lock, info = net._compute_one_global_graph_loss(
        node_count=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        adjacency=None,
        edge_weights=None,
        edge_features=torch.zeros(2, 8),
        edge_attr=None,
        node_features=node_features,
        node_mask=None,
        start_idx=0,
        target_idx=2,
        key_lock_pairs=[],
        current_node_idx=None,
        room_passability=room_passability,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    grad = torch.autograd.grad(total, room_passability, allow_unused=False)[0]

    assert torch.isfinite(total)
    assert torch.isfinite(reach)
    assert "global_graph_skipped" not in info
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 1e-8


def test_walkability_ids_from_palette():
    predictor = WalkabilityPredictor(num_classes=44)

    for name in ("FLOOR", "DOOR_OPEN", "START", "TRIFORCE", "STAIR", "ELEMENT_FLOOR"):
        tile_id = int(SEMANTIC_PALETTE[name])
        assert predictor.walkability_weights[tile_id].item() == pytest.approx(1.0)


def test_temperature_annealing_is_monotone_decreasing():
    net = LogicNet(
        latent_dim=8,
        num_classes=44,
        initial_temperature=2.0,
        final_temperature=0.05,
    )

    temps = []
    for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
        net.update_temperature(progress)
        temps.append(float(net.current_temperature.item()))

    assert temps == sorted(temps, reverse=True)
    assert temps[0] == pytest.approx(2.0)
    assert temps[-1] == pytest.approx(0.05)
