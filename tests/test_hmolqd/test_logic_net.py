"""
Tests for H-MOLQD Block V: LogicNet
====================================

Tests for differentiable pathfinding and solvability checking.
"""

import pytest

torch = pytest.importorskip("torch")
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
        
        distances = pathfinder(walkability, start, goal)
        
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


class TestKeyLockChecker:
    """Tests for key-lock constraint checking."""
    
    def test_checker_basic(self):
        """Test key-lock checker."""
        from src.core.logic_net import KeyLockChecker
        
        checker = KeyLockChecker()
        
        # More keys than locks = solvable
        key_probs = torch.tensor([0.9, 0.8, 0.7])  # 3 keys
        lock_probs = torch.tensor([0.9, 0.8])      # 2 locks
        
        score = checker(key_probs, lock_probs)
        
        assert 0 <= score <= 1


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
        
        probs = classifier(features)
        
        assert probs.shape == (2, 44, 16, 11)
        # Should sum to 1 along class dim
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
        
        walkability = predictor(tiles)
        
        assert walkability.shape == (1, 16, 11)
        # Floor should be walkable
        assert walkability.mean() > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
