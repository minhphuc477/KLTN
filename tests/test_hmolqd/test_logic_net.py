"""
Tests for H-MOLQD Block V: LogicNet
====================================

Tests for differentiable pathfinding and solvability checking.
"""

import pytest
import torch
from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH
from src.pipeline.room_topology_conditioning import ROOM_TOPOLOGY_CHANNELS


class TestDifferentiablePathfinder:
    """Tests for differentiable pathfinding."""
    
    def test_soft_operations(self):
        """Test soft min/max operations."""
        from src.core.logic_net import soft_min, soft_max
        
        x = torch.tensor([1.0, 2.0, 3.0])
        
        s_min = soft_min(x, temperature=0.1)
        s_max = soft_max(x, temperature=0.1)
        
        # Should be close to hard min/max
        assert s_min < 1.5
        assert s_max > 2.5
    
    def test_pathfinder_forward(self):
        """Test pathfinder forward pass."""
        from src.core.logic_net import DifferentiablePathfinder
        
        pathfinder = DifferentiablePathfinder(iterations=10)
        
        # Create simple walkability map
        walkability = torch.ones(1, 16, 11)
        walkability[0, 0:3, :] = 0  # Block top rows
        
        # Start at bottom-left, goal at bottom-right
        start = torch.zeros(1, 16, 11)
        start[0, 15, 0] = 1.0
        
        goal = torch.zeros(1, 16, 11)
        goal[0, 15, 10] = 1.0
        
        distances = pathfinder(walkability, torch.ones_like(walkability), start)
        
        assert distances.shape == (1, 16, 11)
        # Goal should have small distance
        assert distances[0, 15, 10] < distances[0, 0, 0]

    def test_pathfinder_grid_mode_requires_all_rank3_tensors(self):
        """Mixed-rank grid inputs should raise a clean ValueError instead of attribute errors."""
        from src.core.logic_net import DifferentiablePathfinder

        pathfinder = DifferentiablePathfinder(iterations=4)
        walkability = torch.ones(1, 8, 8)
        start = torch.zeros(1, 8, 8)
        source_mask = torch.zeros(8)

        with pytest.raises(ValueError, match="DifferentiablePathfinder grid mode expects"):
            pathfinder(walkability, start, source_mask)


class TestReachabilityScorer:
    """Tests for reachability scoring."""
    
    def test_scorer_forward(self):
        """Test reachability scorer."""
        from src.core.logic_net import ReachabilityScorer
        
        scorer = ReachabilityScorer()
        
        # Create distance map
        distances = torch.rand(2, 16, 11) * 10
        distances[0, 8, 5] = 0.5  # Goal reached
        distances[1, 8, 5] = 100  # Goal not reached
        
        goal = torch.zeros(2, 16, 11)
        goal[:, 8, 5] = 1.0
        
        scores = scorer(distances, goal)
        
        assert scores.shape == (2,)
        assert scores[0] > scores[1]

    def test_negative_distances_do_not_produce_negative_loss(self):
        """Reachability loss should remain a penalty even if upstream distances go negative."""
        from src.core.logic_net import ReachabilityScorer

        scorer = ReachabilityScorer()

        distances = torch.full((1, 16, 11), -25.0)
        goal = torch.zeros(1, 16, 11)
        goal[:, 8, 5] = 1.0

        scores, loss = scorer(distances, goal, return_loss=True)

        assert torch.isfinite(scores).all()
        assert torch.isfinite(loss)
        assert torch.all(scores <= 1.0 + 1e-6)
        assert float(loss.item()) >= 0.0


class TestKeyLockChecker:
    """Tests for key-lock constraint checking."""
    
    def test_checker_basic(self):
        """Test key-lock checker."""
        from src.core.logic_net import KeyLockChecker
        
        checker = KeyLockChecker()
        
        # More keys than locks = solvable
        key_probs = torch.tensor([0.9, 0.8, 0.7])  # 3 keys
        lock_probs = torch.tensor([0.9, 0.8])      # 2 locks
        
        score = checker(key_probs, lock_probs, mode="legacy_probability")
        
        assert 0 <= score <= 1

    def test_checker_requires_explicit_legacy_probability_mode(self):
        """Ambiguous two-vector calls must not silently bypass distance checks."""
        from src.core.logic_net import KeyLockChecker

        checker = KeyLockChecker()
        key_probs = torch.tensor([0.9, 0.8, 0.7])
        lock_probs = torch.tensor([0.9, 0.8, 0.1])

        with pytest.raises(ValueError, match="Ambiguous two-tensor"):
            checker(key_probs, lock_probs)


def test_dense_adjacency_preserves_key_lock_edge_penalties():
    from src.core.logic_net import LogicNet

    logic_net = LogicNet(latent_dim=8, hidden_dim=16, num_classes=44, num_iterations=3)
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    edge_features = torch.zeros(3, 3, 8)
    edge_features[0, 1, 1] = 1.0
    edge_attr = torch.zeros(3, 3, dtype=torch.long)
    edge_attr[0, 2] = 4

    adj, weights = logic_net._build_adjacency_and_weights(
        node_count=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        adjacency=adjacency,
        edge_features=edge_features,
        edge_attr=edge_attr,
    )

    assert torch.equal(adj, adjacency)
    assert weights[0, 1].item() > 1.0
    assert weights[0, 2].item() > weights[0, 1].item()


def test_key_lock_checker_uses_resource_gated_reachability_for_cyclic_graphs():
    from src.core.logic_net import LogicNet

    logic_net = LogicNet(latent_dim=8, hidden_dim=16, num_classes=44, num_iterations=6)
    room_passability = torch.ones(3)

    edge_index_ok = torch.tensor(
        [
            [0, 0, 2, 2],
            [1, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    edge_attr_ok = torch.tensor([1, 0, 0, 0], dtype=torch.long)
    ok_total, _ok_reach, ok_lock, ok_info = logic_net._compute_one_global_graph_loss(
        node_count=3,
        edge_index=edge_index_ok,
        adjacency=None,
        edge_weights=None,
        edge_features=None,
        edge_attr=edge_attr_ok,
        node_features=None,
        node_mask=None,
        start_idx=0,
        target_idx=None,
        key_lock_pairs=[(2, 1)],
        current_node_idx=None,
        room_passability=room_passability,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    edge_index_blocked = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 0],
        ],
        dtype=torch.long,
    )
    edge_attr_blocked = torch.tensor([1, 0, 0], dtype=torch.long)
    blocked_total, _blocked_reach, blocked_lock, blocked_info = logic_net._compute_one_global_graph_loss(
        node_count=3,
        edge_index=edge_index_blocked,
        adjacency=None,
        edge_weights=None,
        edge_features=None,
        edge_attr=edge_attr_blocked,
        node_features=None,
        node_mask=None,
        start_idx=0,
        target_idx=None,
        key_lock_pairs=[(2, 1)],
        current_node_idx=None,
        room_passability=room_passability,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert ok_info["key_lock_mode"] == "resource_gated"
    assert blocked_info["key_lock_mode"] == "resource_gated"
    assert ok_info["locked_edge_count"] == pytest.approx(1.0)
    assert torch.isfinite(ok_total)
    assert torch.isfinite(blocked_total)
    assert blocked_lock.item() > ok_lock.item()


def test_logicnet_resource_gating_preserves_ordered_multi_key_progression():
    from src.core.logic_net import LogicNet

    logic_net = LogicNet(latent_dim=8, hidden_dim=16, num_classes=44, num_iterations=8)
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.tensor([0, 1, 1], dtype=torch.long)
    common = dict(
        node_count=4,
        edge_index=edge_index,
        adjacency=None,
        edge_weights=None,
        edge_features=None,
        edge_attr=edge_attr,
        node_features=None,
        node_mask=None,
        start_idx=0,
        target_idx=3,
        current_node_idx=None,
        room_passability=torch.ones(4),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    _valid_total, valid_reach_loss, valid_lock_loss, valid_info = (
        logic_net._compute_one_global_graph_loss(
            key_lock_pairs=[(1, 2), (2, 3)],
            **common,
        )
    )
    _invalid_total, invalid_reach_loss, invalid_lock_loss, invalid_info = (
        logic_net._compute_one_global_graph_loss(
            key_lock_pairs=[(2, 3), (1, 2)],
            **common,
        )
    )

    assert valid_info["resource_gate_ordering"] == "ordered"
    assert valid_info["blocked_resource_stage_count"] == pytest.approx(0.0)
    assert invalid_info["blocked_resource_stage_count"] >= 1.0
    assert valid_lock_loss.item() < invalid_lock_loss.item()
    assert valid_reach_loss.item() < invalid_reach_loss.item()


class TestLogicNet:
    """Tests for complete LogicNet module."""
    
    def test_logicnet_forward(self):
        """Test LogicNet forward pass."""
        from src.core.logic_net import LogicNet
        
        logic_net = LogicNet(
            num_tile_classes=44,
            hidden_dim=32,
        )
        
        # One-hot encoded room
        room = torch.randn(2, 44, 16, 11)
        room = torch.softmax(room, dim=1)  # Valid distribution
        
        # Start and goal positions
        start = torch.zeros(2, 16, 11)
        start[:, 15, 5] = 1.0
        
        goal = torch.zeros(2, 16, 11)
        goal[:, 0, 5] = 1.0
        
        solvability = logic_net(room, start, goal)
        
        assert solvability.shape == (2,)
        assert torch.all(solvability >= 0)
        assert torch.all(solvability <= 1)
    
    def test_logicnet_gradient_flow(self):
        """Test that gradients flow through LogicNet."""
        from src.core.logic_net import LogicNet
        
        logic_net = LogicNet(
            num_tile_classes=44,
            hidden_dim=32,
        )
        
        room = torch.randn(1, 44, 16, 11, requires_grad=True)
        room_soft = torch.softmax(room, dim=1)
        
        start = torch.zeros(1, 16, 11)
        start[0, 15, 5] = 1.0
        
        goal = torch.zeros(1, 16, 11)
        goal[0, 0, 5] = 1.0
        
        solvability = logic_net(room_soft, start, goal)
        solvability.sum().backward()
        
        assert room.grad is not None
        assert room.grad.abs().sum() > 0

    def test_logicnet_projects_latent_room_pathfinding_to_canonical_room_size(self):
        """Latent inputs should be lifted to room resolution before door/path checks."""
        from src.core.logic_net import LogicNet

        logic_net = LogicNet(
            num_tile_classes=44,
            hidden_dim=32,
        )

        z = torch.randn(1, 64, 4, 3)
        loss, info = logic_net(z)

        assert loss.ndim == 0
        assert tuple(info["latent_tile_logits"].shape[-2:]) == (4, 3)
        assert tuple(info["tile_logits"].shape[-2:]) == (ROOM_HEIGHT, ROOM_WIDTH)
        assert tuple(info["walkability"].shape[-2:]) == (ROOM_HEIGHT, ROOM_WIDTH)
        assert tuple(info["grid_distances"].shape[-2:]) == (ROOM_HEIGHT, ROOM_WIDTH)

    def test_logicnet_room_topology_losses_are_included_in_total_loss(self):
        """Room-topology traces and anchors should contribute directly to the optimized loss."""
        from src.core.logic_net import LogicNet

        logic_net = LogicNet(
            latent_dim=64,
            num_classes=44,
            num_iterations=4,
            topology_trace_weight=0.6,
            topology_anchor_weight=0.4,
        )

        topology = torch.zeros(1, len(ROOM_TOPOLOGY_CHANNELS), ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        topology[:, ROOM_TOPOLOGY_CHANNELS["traversability"], 4:12, 2:9] = 1.0
        topology[:, ROOM_TOPOLOGY_CHANNELS["start"], ROOM_HEIGHT // 2, 1] = 1.0
        topology[:, ROOM_TOPOLOGY_CHANNELS["goal"], ROOM_HEIGHT // 2, ROOM_WIDTH - 2] = 1.0
        topology[:, ROOM_TOPOLOGY_CHANNELS["door_w"], ROOM_HEIGHT // 2 - 1:ROOM_HEIGHT // 2 + 2, 0] = 1.0
        topology[:, ROOM_TOPOLOGY_CHANNELS["door_e"], ROOM_HEIGHT // 2 - 1:ROOM_HEIGHT // 2 + 2, ROOM_WIDTH - 1] = 1.0

        boundary = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        z = torch.randn(1, 64, 4, 3)

        loss, info = logic_net(
            z,
            graph_data={
                "room_topology_map": topology,
                "boundary_constraints": boundary,
            },
        )

        expected = (
            logic_net.reach_weight * info["grid_reach_loss"]
            + logic_net.topology_trace_weight * info["topology_trace_loss"]
            + logic_net.topology_anchor_weight * info["topology_anchor_loss"]
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert torch.isfinite(info["topology_trace_loss"])
        assert torch.isfinite(info["topology_anchor_loss"])
        assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-5)

    def test_logicnet_global_graph_loss_depends_on_current_room_latent(self):
        """Mission-graph reachability should have a gradient path through room passability."""
        from src.core.logic_net import LogicNet

        logic_net = LogicNet(
            latent_dim=64,
            num_classes=44,
            num_iterations=4,
            global_reach_weight=1.0,
            global_room_weight=0.25,
        )

        z = torch.randn(1, 64, 4, 3, requires_grad=True)
        node_features = torch.zeros(3, 6, dtype=torch.float32)
        node_features[2, 3] = 1.0  # target/triforce node
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_features = torch.zeros(2, 8, dtype=torch.float32)

        loss, info = logic_net(
            z,
            graph_data={
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_features": edge_features,
                "current_node_idx": torch.tensor([1], dtype=torch.long),
                "start_node_id": 0,
            },
        )
        loss.backward()

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert torch.isfinite(info["global_logic_loss"])
        assert torch.isfinite(info["graph_reach_loss"])
        assert "global_graph_reachability" in info
        assert z.grad is not None
        assert float(z.grad.abs().sum().item()) > 0.0

    def test_logicnet_resolves_typed_gate_channels_as_door_anchors(self):
        """Typed gate-only topology maps should still register as doorway anchors."""
        from src.core.logic_net import LogicNet

        logic_net = LogicNet(
            latent_dim=64,
            num_classes=44,
            num_iterations=4,
        )

        topology = torch.zeros(1, len(ROOM_TOPOLOGY_CHANNELS), ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        topology[:, ROOM_TOPOLOGY_CHANNELS["gate_switch_e"], ROOM_HEIGHT // 2 - 1:ROOM_HEIGHT // 2 + 2, ROOM_WIDTH - 1] = 1.0

        targets = logic_net._resolve_room_logic_targets(
            {"room_topology_map": topology},
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert targets["source_mask"] is not None
        assert targets["target_mask"] is not None
        assert targets["anchor_target"] is not None
        assert float(targets["anchor_target"].sum().item()) > 0.0


class TestTileClassifier:
    """Tests for tile classification."""
    
    def test_classifier_forward(self):
        """Test tile classifier."""
        from src.core.logic_net import TileClassifier
        
        classifier = TileClassifier(
            in_channels=32,
            num_classes=44,
        )
        
        features = torch.randn(2, 32, 16, 11)
        
        logits = classifier(features)
        
        assert logits.shape == (2, 44, 16, 11)
        assert not torch.allclose(
            logits.sum(dim=1),
            torch.ones_like(logits[:, 0]),
            atol=1e-5,
        )

        probs = TileClassifier(
            in_channels=32,
            num_classes=44,
            output_mode="probs",
        )(features)
        assert probs.shape == (2, 44, 16, 11)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestWalkabilityPredictor:
    """Tests for walkability prediction."""
    
    def test_predictor_from_tiles(self):
        """Test walkability from tile probabilities."""
        from src.core.logic_net import WalkabilityPredictor
        
        predictor = WalkabilityPredictor(num_tile_classes=44)
        
        # Create tile distribution
        tiles = torch.zeros(1, 44, 16, 11)
        tiles[0, 1, :, :] = 1.0  # All floor
        
        walkability = predictor(tiles, is_probs=True)
        
        assert walkability.shape == (1, 16, 11)
        # Floor should be walkable
        assert walkability.mean() > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
