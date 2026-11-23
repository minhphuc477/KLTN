"""Behavioral tests for the local-connectivity hypothesis, without checkpoints."""

import pytest
import torch

from src.core.definitions import ROOM_HEIGHT as H, ROOM_WIDTH as W, SEMANTIC_PALETTE as P
from src.core.logic_net import LogicNet


def test_mixed_resource_room_keeps_small_key_and_typed_item_supervision():
    import networkx as nx
    from src.pipeline.graph_features import build_key_lock_pairs
    graph = nx.DiGraph()
    graph.add_node(0, label='s')
    graph.add_node(1, label='k,I', key_id='small', item_type='LADDER')
    graph.add_node(2)
    graph.add_node(3, label='t')
    graph.add_edge(0, 1)
    graph.add_edge(1, 2, edge_type='key_locked', key_required='small')
    graph.add_edge(2, 3, edge_type='item_locked', item_required='LADDER')
    assert build_key_lock_pairs(graph, {n: n for n in graph}, start_node=0) == [(1, 2), (1, 3)]


def test_zero_global_weights_remove_all_global_logic_gradients():
    net = LogicNet(
        latent_dim=4,
        num_classes=5,
        global_reach_weight=0.0,
        global_room_weight=0.0,
        lock_weight=0.0,
    )
    room_passability = torch.tensor([0.4, 0.8], requires_grad=True)
    total, reach, lock, _ = net._compute_one_global_graph_loss(
        node_count=2,
        edge_index=torch.tensor([[0], [1]]), adjacency=None, edge_weights=None,
        edge_features=torch.tensor([[0.0, 1.0]]), edge_attr=None,
        node_features=torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        node_mask=None, start_idx=0, target_idx=1, key_lock_pairs=[(1, 1)],
        current_node_idx=None, room_passability=room_passability,
        device=torch.device('cpu'), dtype=torch.float32,
    )
    gradient, = torch.autograd.grad(total, room_passability)
    assert reach > 0 and lock > 0
    assert total == 0 and torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize('explicit', [False, True])
def test_multiple_entrances_cannot_hide_disconnected_room(explicit):
    net = LogicNet(latent_dim=8, hidden_dim=16, num_classes=44, num_iterations=4)
    boundary = torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0.]])
    doors = net._build_boundary_door_mask(boundary, batch_size=1, device=torch.device('cpu'),
                                         dtype=torch.float32, spatial_hw=(H, W))
    graph = ({'logic_source_mask': doors, 'logic_target_mask': doors} if explicit
             else {'boundary_constraints': boundary})
    blocked = torch.full((1, H, W), P['WALL'], dtype=torch.long)
    blocked[doors[:, 0].bool()] = P['FLOOR']
    connected = blocked.clone()
    connected[:, :, W // 2 - 1:W // 2 + 2] = P['FLOOR']

    def logits(ids):
        return (torch.nn.functional.one_hot(ids, 44).permute(0, 3, 1, 2).float() * 12 - 6)

    blocked_logits = logits(blocked).requires_grad_(True)
    blocked_loss, _ = net(blocked_logits, graph_data=graph)
    connected_loss, _ = net(logits(connected), graph_data=graph)
    assert connected_loss < blocked_loss - 0.01
    gradient, = torch.autograd.grad(blocked_loss, blocked_logits)
    assert torch.isfinite(gradient).all()
    # Gradient descent must reduce wall confidence along the separating barrier.
    assert gradient[:, P['WALL'], H // 2].sum() > 0


def test_empty_topology_masks_get_per_room_nonvacuous_targets():
    net = LogicNet(latent_dim=8, hidden_dim=16)
    source = torch.zeros(2, 1, H, W)
    target = torch.zeros_like(source)
    source[0, 0, 0, 2] = 1
    target[0, 0, H - 1, 2] = 1
    masks = net._resolve_room_logic_targets(
        {'logic_source_mask': source, 'logic_target_mask': target}, batch_size=2,
        device=torch.device('cpu'), dtype=torch.float32, spatial_hw=(H, W))
    assert torch.equal(masks['source_mask'].sum((1, 2, 3)), torch.ones(2))
    assert masks['source_mask'][1, 0, H // 2, W // 2] == 1
    assert masks['target_mask'][1].sum() == H * W
