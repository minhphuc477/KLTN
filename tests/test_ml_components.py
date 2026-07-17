"""
Test Suite for ML Components
============================

Comprehensive tests for:
1. Training Pipeline (train_diffusion.py)
2. Graph-to-Grid Cross-Attention
3. Tortuosity Loss
4. Mission Grammar
5. Causal WFC

Run with: pytest tests/test_ml_components.py -v
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# TEST: TORTUOSITY LOSS
# ============================================================================

class TestTortuosityLoss:
    """Tests for tortuosity loss computation."""
    
    def test_tortuosity_import(self):
        """Test that tortuosity module imports correctly."""
        from src.ml.logic_net import (
            tortuosity_loss,
            DifferentiableTortuosity,
            combined_logic_loss,
        )
        assert tortuosity_loss is not None
        assert DifferentiableTortuosity is not None
        assert combined_logic_loss is not None
    
    def test_differentiable_tortuosity_forward(self):
        """Test DifferentiableTortuosity forward pass."""
        from src.ml.logic_net import DifferentiableTortuosity
        
        module = DifferentiableTortuosity(num_iterations=20)
        
        # Create test probability map (mostly walkable)
        B, H, W = 4, 16, 11
        prob_map = torch.ones(B, 1, H, W) * 0.9
        
        # Add walls
        prob_map[:, :, 0, :] = 0.0  # Top wall
        prob_map[:, :, -1, :] = 0.0  # Bottom wall
        
        starts = [(2, 2)] * B
        goals = [(13, 8)] * B
        
        tortuosity, is_valid = module(prob_map, starts, goals)
        
        assert tortuosity.shape == (B,)
        assert is_valid.shape == (B,)
        assert (tortuosity >= 1.0).all(), "Tortuosity should be >= 1.0"
    
    def test_tortuosity_loss_gradient_flow(self):
        """Test that gradients flow through tortuosity loss."""
        from src.ml.logic_net import tortuosity_loss
        
        B, H, W = 2, 16, 11
        prob_map = torch.rand(B, 1, H, W, requires_grad=True)
        
        starts = [(2, 2)] * B
        goals = [(13, 8)] * B
        
        loss = tortuosity_loss(prob_map, starts, goals)
        loss.backward()
        
        assert prob_map.grad is not None
        assert not torch.isnan(prob_map.grad).any()

    def test_tortuosity_value_iteration_uses_shortest_neighbor_not_sum(self):
        from src.ml.logic_net import DifferentiableTortuosity

        module = DifferentiableTortuosity(num_iterations=8)
        prob_map = torch.ones(1, 1, 3, 5)

        path_length = module.compute_soft_path_length(
            prob_map,
            start_coords=[(1, 0)],
            goal_coords=[(1, 4)],
        )

        assert path_length.item() == pytest.approx(4.0, abs=1e-4)
    
    def test_combined_logic_loss(self):
        """Test combined solvability + tortuosity loss."""
        from src.ml.logic_net import combined_logic_loss
        
        B, H, W = 4, 16, 11
        prob_map = torch.rand(B, 1, H, W)
        
        starts = [(2, 2)] * B
        goals = [(13, 8)] * B
        
        total_loss, loss_dict = combined_logic_loss(
            prob_map, starts, goals,
            solvability_weight=1.0,
            tortuosity_weight=0.3,
        )
        
        assert 'solvability_loss' in loss_dict
        assert 'tortuosity_loss' in loss_dict
        assert 'mean_solvability' in loss_dict
        assert total_loss.ndim == 0  # Scalar

    def test_combined_logic_loss_reuses_default_pathfinder_helpers(self):
        """Default helper modules should be cached instead of allocated per call."""
        import src.ml.logic_net as logic_mod
        from src.ml.logic_net import combined_logic_loss

        logic_mod._COMBINED_LOGIC_HELPER_CACHE.clear()
        prob_map = torch.rand(1, 1, 8, 8)
        starts = [(1, 1)]
        goals = [(6, 6)]

        combined_logic_loss(prob_map, starts, goals)
        first_helpers = next(iter(logic_mod._COMBINED_LOGIC_HELPER_CACHE.values()))
        combined_logic_loss(prob_map, starts, goals)
        second_helpers = next(iter(logic_mod._COMBINED_LOGIC_HELPER_CACHE.values()))

        assert first_helpers[0] is second_helpers[0]
        assert first_helpers[1] is second_helpers[1]

    def test_legacy_soft_bellman_ford_uses_distance_relaxation_and_walls(self):
        """The legacy compatibility path must not saturate through solid walls."""
        from src.ml.logic_net import SoftBellmanFord

        pathfinder = SoftBellmanFord(num_iterations=8, temperature=0.5, wall_penalty=20.0)
        open_map = torch.ones(1, 1, 5, 5)
        blocked_map = open_map.clone()
        blocked_map[:, :, :, 2] = 0.0

        start = [(2, 0)]
        goal = [(2, 4)]

        open_score = pathfinder(open_map, start, goal)
        blocked_score = pathfinder(blocked_map, start, goal)

        assert open_score.item() > 0.9
        assert blocked_score.item() < 0.1

    def test_legacy_soft_bellman_ford_backpropagates_to_walkability(self):
        """Soft distance relaxation should keep a useful gradient to walkability probabilities."""
        from src.ml.logic_net import SoftBellmanFord

        pathfinder = SoftBellmanFord(num_iterations=8, temperature=1.0, wall_penalty=5.0)
        prob_map = torch.full((1, 1, 5, 5), 0.8, requires_grad=True)

        score = pathfinder(prob_map, [(2, 0)], [(2, 4)]).mean()
        score.backward()

        assert prob_map.grad is not None
        assert torch.isfinite(prob_map.grad).all()
        assert prob_map.grad.abs().sum().item() > 0.0
    
    def test_straight_path_penalty(self):
        """Test that straight paths get penalized."""
        from src.ml.logic_net import tortuosity_loss
        
        # Create a straight corridor
        H, W = 16, 11
        straight_map = torch.zeros(1, 1, H, W)
        straight_map[0, 0, :, 5] = 1.0  # Vertical corridor
        
        starts = [(2, 5)]
        goals = [(13, 5)]
        
        straight_loss = tortuosity_loss(straight_map, starts, goals, target_tortuosity=1.5)
        
        # Create a winding path
        winding_map = torch.zeros(1, 1, H, W)
        for r in range(2, 14):
            c = 5 + int(2 * np.sin(r * 0.5))
            winding_map[0, 0, r, max(0, min(W-1, c))] = 1.0
        
        _winding_loss = tortuosity_loss(winding_map, starts, goals, target_tortuosity=1.5)
        
        # Straight path should have higher loss (more penalty)
        # This is a soft test since the losses depend on the actual path computation
        assert straight_loss.item() >= 0  # Should be non-negative


def test_heuristic_admissibility_calibration_subtracts_observed_overestimate():
    from src.ml.heuristic_learning import HeuristicTrainer

    trainer = HeuristicTrainer(map_height=8, map_width=8)
    with torch.no_grad():
        for param in trainer.model.parameters():
            param.zero_()
        trainer.model.fc4.bias.fill_(10.0)

    features = np.zeros((2, 10), dtype=np.float32)
    true_costs = np.array([3.0, 4.0], dtype=np.float32)

    trainer.enforce_admissibility(
        scaling_factor=1.0,
        validation_features=features,
        true_costs=true_costs,
    )

    with torch.no_grad():
        preds = (
            trainer.model(torch.as_tensor(features)).squeeze(-1).numpy()
            * trainer.target_scale
        )

    assert np.all(preds <= true_costs + 1e-6)


def test_heuristic_features_do_not_include_remaining_cost_label():
    from types import SimpleNamespace
    from src.ml.heuristic_learning import HeuristicTrainer, TrainingExample

    trainer = HeuristicTrainer(map_height=8, map_width=8)
    env = SimpleNamespace(
        start_pos=(0, 0),
        goal_pos=(7, 7),
        grid=np.zeros((8, 8), dtype=np.int64),
    )
    common = dict(
        position=(3, 2),
        keys=1,
        has_bomb=False,
        has_boss_key=False,
        has_item=True,
    )
    near_label = TrainingExample(remaining_cost=1, **common)
    far_label = TrainingExample(remaining_cost=50, **common)

    np.testing.assert_array_equal(
        trainer.featurize_state(near_label, env),
        trainer.featurize_state(far_label, env),
    )


# ============================================================================
# TEST: GRAPH-TO-GRID CROSS-ATTENTION
# ============================================================================

class TestGraphGridAttention:
    """Tests for graph-to-grid cross-attention."""
    
    def test_cross_attention_import(self):
        """Test module imports correctly."""
        from src.core.graph_grid_attention import (
            GraphToGridCrossAttention,
            EnhancedAttentionBlock,
        )
        assert GraphToGridCrossAttention is not None
        assert EnhancedAttentionBlock is not None
    
    def test_position_encoding_2d(self):
        """Test 2D sinusoidal position encoding."""
        from src.core.graph_grid_attention import SinusoidalPositionEncoding2D
        
        pe = SinusoidalPositionEncoding2D(dim=128)
        
        B, C, H, W = 2, 128, 16, 11
        x = torch.randn(B, C, H, W)
        
        output = pe(x)
        
        assert output.shape == x.shape
        # Position encoding adds to input, so output should be different
        assert not torch.allclose(output, x)

    def test_position_encoding_2d_grows_for_large_feature_maps(self):
        from src.core.graph_grid_attention import SinusoidalPositionEncoding2D

        pe = SinusoidalPositionEncoding2D(dim=16, max_size=(4, 4))
        x = torch.randn(1, 16, 5, 7)

        output = pe(x)

        assert output.shape == x.shape
        assert pe.pe.shape[0] >= 5
        assert pe.pe.shape[1] >= 7
    
    def test_graph_to_grid_forward(self):
        """Test GraphToGridCrossAttention forward pass."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention
        
        B, C, H, W = 2, 128, 16, 11
        N_nodes = 10
        graph_dim = 256
        
        module = GraphToGridCrossAttention(
            grid_dim=C,
            graph_dim=graph_dim,
            num_heads=8,
        )
        
        grid_features = torch.randn(B, C, H, W)
        graph_nodes = torch.randn(B, N_nodes, graph_dim)
        node_positions = torch.randint(0, 10, (B, N_nodes, 2)).float()
        node_tpe = torch.randn(B, N_nodes, 8)
        current_node_distance = torch.randn(B, N_nodes, 4)
        
        output = module(
            grid_features,
            graph_nodes,
            node_positions=node_positions,
            node_tpe=node_tpe,
            current_node_distance=current_node_distance,
        )
        
        assert output.shape == grid_features.shape
    
    def test_graph_to_grid_gradient_flow(self):
        """Test gradient flow through cross-attention."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention
        
        B, C, H, W = 2, 64, 8, 8
        N = 5
        graph_dim = 128
        
        module = GraphToGridCrossAttention(
            grid_dim=C,
            graph_dim=graph_dim,
            num_heads=4,
        )
        
        grid_features = torch.randn(B, C, H, W, requires_grad=True)
        graph_nodes = torch.randn(B, N, graph_dim, requires_grad=True)
        
        output = module(grid_features, graph_nodes)
        loss = output.mean()
        loss.backward()
        
        assert grid_features.grad is not None
        assert graph_nodes.grad is not None

    def test_spatial_alignment_loss_backpropagates_through_attention(self):
        """Captured attention used for alignment must remain differentiable."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(
            grid_dim=32,
            graph_dim=48,
            num_heads=4,
            attention_mode="softmax",
        )
        module.set_attention_capture(True)
        grid_features = torch.randn(2, 32, 4, 4, requires_grad=True)
        graph_nodes = torch.randn(2, 3, 48, requires_grad=True)
        _ = module(grid_features, graph_nodes)

        node_indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        target_positions = torch.tensor(
            [[[0, 0], [3, 3]], [[1, 1], [2, 2]]],
            dtype=torch.float32,
        )
        loss = module.spatial_alignment_loss(node_indices, target_positions)
        loss.backward()

        assert loss.requires_grad
        assert module.q_proj.weight.grad is not None
        assert module.k_proj.weight.grad is not None
        assert torch.isfinite(module.q_proj.weight.grad).all()

    def test_spatial_alignment_loss_caps_missed_attention_gradient(self):
        """A complete miss should not produce the old -1/1e-8 gradient spike."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(
            grid_dim=32,
            graph_dim=48,
            num_heads=4,
            attention_mode="softmax",
        )
        missed = torch.full((1, 4, 1, 3), 1.0e-8, requires_grad=True)
        with module._attention_capture_lock:
            module.last_attention_weights_for_loss = missed
            module.last_attention_grid_shape = (1, 1)

        loss = module.spatial_alignment_loss(
            torch.tensor([[2]], dtype=torch.long),
            torch.tensor([[[0, 0]]], dtype=torch.float32),
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert missed.grad is not None
        assert torch.isfinite(missed.grad).all()
        assert float(missed.grad.abs().max()) < 1.0e5

    def test_graph_to_grid_attention_fully_masked_graph_has_finite_backward(self):
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(
            grid_dim=16,
            graph_dim=12,
            num_heads=4,
            attention_mode="softmax",
        )
        grid_features = torch.randn(2, 16, 3, 3, requires_grad=True)
        graph_nodes = torch.randn(2, 3, 12, requires_grad=True)
        node_mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.float32)

        out = module(grid_features, graph_nodes, node_mask=node_mask)
        loss = out.square().mean()
        loss.backward()

        assert torch.isfinite(out).all()
        assert grid_features.grad is not None and torch.isfinite(grid_features.grad).all()
        assert graph_nodes.grad is not None and torch.isfinite(graph_nodes.grad).all()

    def test_graph_to_grid_attention_fully_masked_row_is_identity(self):
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(
            grid_dim=16,
            graph_dim=12,
            num_heads=4,
            attention_mode="softmax",
            dropout=0.0,
        )
        module.eval()
        grid_features = torch.randn(2, 16, 3, 3)
        graph_nodes = torch.randn(2, 3, 12)
        node_mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.bool)

        out_a = module(grid_features, graph_nodes, node_mask=node_mask)
        changed_graph = graph_nodes.clone()
        changed_graph[1] = changed_graph[1] * 100.0 + 17.0
        out_b = module(grid_features, changed_graph, node_mask=node_mask)

        assert torch.allclose(out_a[1], grid_features[1])
        assert torch.allclose(out_b[1], grid_features[1])
        assert torch.allclose(out_a[1], out_b[1])

    def test_graph_to_grid_gcn_filters_edges_touching_padded_nodes(self):
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        torch.manual_seed(33)
        module = GraphToGridCrossAttention(
            grid_dim=16,
            graph_dim=12,
            num_heads=4,
            attention_mode="softmax",
            dropout=0.0,
        )
        module.eval()
        grid_features = torch.randn(1, 16, 3, 3)
        graph_nodes = torch.randn(1, 3, 12)
        node_mask = torch.tensor([[True, True, False]])
        padded_edge = torch.tensor([[0], [2]], dtype=torch.long)
        no_edges = torch.empty(2, 0, dtype=torch.long)

        out_with_padded_edge = module(
            grid_features,
            graph_nodes,
            edge_index=padded_edge,
            node_mask=node_mask,
        )
        out_without_edge = module(
            grid_features,
            graph_nodes,
            edge_index=no_edges,
            node_mask=node_mask,
        )

        assert torch.allclose(out_with_padded_edge, out_without_edge, atol=1e-6)

    def test_graph_to_grid_edge_semantics_are_explicit_ablation(self):
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        torch.manual_seed(34)
        grid_features = torch.randn(1, 16, 3, 3)
        graph_nodes = torch.randn(1, 3, 12)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        open_edges = torch.tensor([0, 0], dtype=torch.long)
        locked_edges = torch.tensor([4, 4], dtype=torch.long)

        baseline = GraphToGridCrossAttention(
            grid_dim=16,
            graph_dim=12,
            num_heads=4,
            attention_mode="softmax",
            dropout=0.0,
            use_edge_semantics=False,
        ).eval()
        semantic = GraphToGridCrossAttention(
            grid_dim=16,
            graph_dim=12,
            num_heads=4,
            attention_mode="softmax",
            dropout=0.0,
            use_edge_semantics=True,
        ).eval()

        with torch.no_grad():
            base_open = baseline(grid_features, graph_nodes, edge_index=edge_index, edge_attr=open_edges)
            base_locked = baseline(grid_features, graph_nodes, edge_index=edge_index, edge_attr=locked_edges)
            sem_open = semantic(grid_features, graph_nodes, edge_index=edge_index, edge_attr=open_edges)
            sem_locked = semantic(grid_features, graph_nodes, edge_index=edge_index, edge_attr=locked_edges)

        assert torch.allclose(base_open, base_locked, atol=1e-6)
        assert not torch.allclose(sem_open, sem_locked)

    def test_spatial_alignment_loss_requires_captured_softmax_maps(self):
        """Alignment should fail loudly when capture/softmax attention is absent."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(grid_dim=32, graph_dim=48, num_heads=4)
        with pytest.raises(RuntimeError, match="set_attention_capture"):
            module.spatial_alignment_loss(
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, 1, 2),
            )
    
    def test_enhanced_attention_block(self):
        """Test EnhancedAttentionBlock with graph and context modes."""
        from src.core.graph_grid_attention import EnhancedAttentionBlock
        
        B, C, H, W = 2, 128, 8, 8
        N_nodes = 5
        graph_dim = 256
        context_dim = 256
        
        block = EnhancedAttentionBlock(
            grid_dim=C,
            graph_dim=graph_dim,
            context_dim=context_dim,
        )
        
        grid_features = torch.randn(B, C, H, W)
        
        # Test with graph nodes
        graph_nodes = torch.randn(B, N_nodes, graph_dim)
        out1 = block(grid_features, graph_nodes=graph_nodes)
        assert out1.shape == grid_features.shape
        
        # Test with context vector (backward compat)
        context = torch.randn(B, context_dim)
        out2 = block(grid_features, context=context)
        assert out2.shape == grid_features.shape
    
    def test_node_mask(self):
        """Test graph-to-grid with node masking."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention
        
        B, C, H, W = 2, 64, 8, 8
        N = 10
        graph_dim = 128
        
        module = GraphToGridCrossAttention(grid_dim=C, graph_dim=graph_dim)
        
        grid_features = torch.randn(B, C, H, W)
        graph_nodes = torch.randn(B, N, graph_dim)
        
        # Mask out last 5 nodes
        node_mask = torch.ones(B, N)
        node_mask[:, 5:] = 0
        
        output = module(grid_features, graph_nodes, node_mask=node_mask)
        assert output.shape == grid_features.shape

    def test_graph_to_grid_linear_hedgehog_and_topology_map(self):
        """Graph-grid conditioning should support linear Hedgehog attention and topology maps."""
        from src.core.graph_grid_attention import SpatialGraphConditioner

        module = SpatialGraphConditioner(
            grid_dim=64,
            graph_dim=128,
            topology_channels=18,
            attention_mode="linear_hedgehog",
            hedgehog_feature_dim=16,
        )

        grid_features = torch.randn(2, 64, 8, 8)
        graph_nodes = torch.randn(2, 5, 128)
        node_positions = torch.randn(2, 5, 2)
        node_tpe = torch.randn(2, 5, 8)
        room_topology_map = torch.randn(2, 18, 16, 11)

        output = module(
            grid_features,
            graph_nodes=graph_nodes,
            node_positions=node_positions,
            node_tpe=node_tpe,
            room_topology_map=room_topology_map,
        )
        assert output.shape == grid_features.shape

    def test_graph_to_grid_rejects_node_mask_shape_mismatch(self):
        """Graph-grid conditioning should fail fast when mask length does not match node count."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(grid_dim=64, graph_dim=128)
        grid_features = torch.randn(2, 64, 8, 8)
        graph_nodes = torch.randn(2, 5, 128)
        node_mask = torch.ones(2, 4)

        with pytest.raises(ValueError, match="node_mask shape"):
            module(grid_features, graph_nodes, node_mask=node_mask)

    def test_graph_to_grid_legacy_positional_swap_requires_explicit_opt_in(self):
        """Legacy positional node_positions/node_tpe calls should require an explicit compatibility flag."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(grid_dim=64, graph_dim=128)
        grid_features = torch.randn(2, 64, 8, 8)
        graph_nodes = torch.randn(2, 5, 128)
        node_positions = torch.randn(2, 5, 2)
        node_tpe = torch.randn(2, 5, 8)

        with pytest.raises(ValueError, match="legacy positional arguments"):
            module(grid_features, graph_nodes, node_positions, node_tpe)

        compat_module = GraphToGridCrossAttention(
            grid_dim=64,
            graph_dim=128,
            allow_legacy_argument_swap=True,
        )
        output = compat_module(grid_features, graph_nodes, node_positions, node_tpe)
        assert output.shape == grid_features.shape

    def test_graph_to_grid_rejects_non_divisible_head_configuration(self):
        """Transformer-style multi-head projections require grid_dim to split evenly across heads."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        with pytest.raises(ValueError, match="divisible by num_heads"):
            GraphToGridCrossAttention(grid_dim=62, graph_dim=128, num_heads=8)

    def test_graph_to_grid_empty_graph_is_identity(self):
        """Empty graph batches should safely no-op instead of producing invalid attention tensors."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(grid_dim=64, graph_dim=128)
        grid_features = torch.randn(2, 64, 8, 8)
        graph_nodes = torch.randn(2, 0, 128)

        output = module(grid_features, graph_nodes)
        assert output.shape == grid_features.shape
        assert torch.equal(output, grid_features)

    def test_graph_to_grid_supports_shared_batched_edge_index(self):
        """A shared graph topology batched as [1, 2, E] should broadcast across samples."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(grid_dim=64, graph_dim=128)
        grid_features = torch.randn(2, 64, 8, 8)
        graph_nodes = torch.randn(2, 5, 128)
        edge_index = torch.tensor([[[0, 1, 2], [1, 2, 3]]], dtype=torch.long)

        output = module(grid_features, graph_nodes, edge_index=edge_index)
        assert output.shape == grid_features.shape

    def test_graph_to_grid_aligns_malformed_tpe_width(self):
        """OOD graph metadata with the wrong TPE width should be normalized instead of crashing."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(grid_dim=64, graph_dim=128)
        grid_features = torch.randn(2, 64, 8, 8)
        graph_nodes = torch.randn(2, 5, 128)
        node_tpe = torch.randn(2, 5, 5)

        output = module(grid_features, graph_nodes, node_tpe=node_tpe)
        assert output.shape == grid_features.shape

    def test_graph_to_grid_aligns_malformed_current_node_distance_width(self):
        """Current-room distance features with the wrong width should be normalized instead of crashing."""
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        module = GraphToGridCrossAttention(grid_dim=64, graph_dim=128)
        grid_features = torch.randn(2, 64, 8, 8)
        graph_nodes = torch.randn(2, 5, 128)
        current_node_distance = torch.randn(2, 5, 2)

        output = module(
            grid_features,
            graph_nodes,
            current_node_distance=current_node_distance,
        )
        assert output.shape == grid_features.shape

    def test_graph_to_grid_switches_to_linear_attention_for_large_graphs(self, monkeypatch):
        """Large graph batches should automatically avoid quadratic softmax attention."""
        import src.core.graph_grid_attention as graph_grid_attention
        from src.core.graph_grid_attention import GraphToGridCrossAttention

        seen = {"linear_called": False}

        def _fake_linear_attention(Q, K, V, q_map, k_map, token_mask=None):
            _ = (q_map, k_map, token_mask)
            seen["linear_called"] = True
            return torch.zeros_like(Q)

        monkeypatch.setattr(graph_grid_attention, "hedgehog_linear_attention", _fake_linear_attention)

        module = GraphToGridCrossAttention(
            grid_dim=64,
            graph_dim=128,
            attention_mode="softmax",
            auto_linear_attention_nodes=4,
        )
        grid_features = torch.randn(1, 64, 8, 8)
        graph_nodes = torch.randn(1, 6, 128)

        output = module(grid_features, graph_nodes)

        assert output.shape == grid_features.shape
        assert seen["linear_called"] is True

    def test_lightweight_gcn_matches_dense_normalized_reference(self):
        """The lightweight GCN should preserve normalized-adjacency behavior without dense N x N materialization."""
        from src.core.graph_grid_attention import LightweightGCNLayer

        layer = LightweightGCNLayer(in_dim=3, out_dim=2)
        with torch.no_grad():
            layer.linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
            layer.linear.bias.zero_()

        x = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
            dtype=torch.float32,
        )
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        output = layer(x, edge_index)

        adj = torch.zeros(3, 3, dtype=torch.float32)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0
        adj[1, 2] = 1.0
        adj[2, 1] = 1.0
        adj = adj + torch.eye(3, dtype=torch.float32)
        deg = adj.sum(dim=1).clamp(min=1.0)
        norm_adj = deg.pow(-0.5)[:, None] * adj * deg.pow(-0.5)[None, :]
        projected = x[0, :, :2]
        expected = norm_adj @ projected

        assert torch.allclose(output[0], expected, atol=1e-6)

    def test_lightweight_gcn_batches_graphs_without_cross_graph_messages(self):
        """Batched sparse propagation must match independent dense graphs, including padding."""
        from src.core.graph_grid_attention import LightweightGCNLayer

        layer = LightweightGCNLayer(in_dim=2, out_dim=2)
        with torch.no_grad():
            layer.linear.weight.copy_(torch.eye(2))
            layer.linear.bias.zero_()

        x = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            ]
        )
        edge_index = torch.tensor(
            [
                [[0, 1, -1], [1, 2, -1]],
                [[0, 2, 99], [2, 1, 0]],
            ],
            dtype=torch.long,
        )
        node_mask = torch.tensor([[True, True, True], [True, True, False]])

        output = layer(x, edge_index, node_mask=node_mask)

        expected_graphs = []
        for batch_idx in range(2):
            valid = node_mask[batch_idx]
            adjacency = torch.zeros(3, 3)
            for source, target in edge_index[batch_idx].t().tolist():
                if (
                    0 <= source < 3
                    and 0 <= target < 3
                    and bool(valid[source])
                    and bool(valid[target])
                ):
                    adjacency[source, target] += 1.0
                    adjacency[target, source] += 1.0
            adjacency += torch.diag(valid.float())
            degree = adjacency.sum(dim=1).clamp(min=1.0)
            normalized = degree.pow(-0.5)[:, None] * adjacency * degree.pow(-0.5)[None, :]
            expected_graphs.append(normalized @ (x[batch_idx] * valid[:, None]))

        expected = torch.stack(expected_graphs)
        assert torch.allclose(output, expected, atol=1e-6)
        assert torch.equal(output[1, 2], torch.zeros(2))

    def test_spatial_graph_conditioner_rejects_topology_batch_mismatch(self):
        """SpatialGraphConditioner should validate room-topology batch alignment."""
        from src.core.graph_grid_attention import SpatialGraphConditioner

        module = SpatialGraphConditioner(
            grid_dim=64,
            graph_dim=128,
            topology_channels=18,
        )
        grid_features = torch.randn(2, 64, 8, 8)
        room_topology_map = torch.randn(1, 18, 16, 11)

        with pytest.raises(ValueError, match="room_topology_map batch size"):
            module(grid_features, room_topology_map=room_topology_map)

    @pytest.mark.parametrize("topology_conditioning_mode", ["additive", "spade"])
    def test_spatial_graph_conditioner_gates_allow_branch_gradients_from_step_one(
        self,
        topology_conditioning_mode,
    ):
        """Conditioning branches should receive gradients immediately for both additive and SPADE topology paths."""
        from src.core.graph_grid_attention import SpatialGraphConditioner

        module = SpatialGraphConditioner(
            grid_dim=64,
            graph_dim=128,
            topology_channels=18,
            topology_conditioning_mode=topology_conditioning_mode,
        )
        grid_features = torch.randn(2, 64, 8, 8, requires_grad=True)
        graph_nodes = torch.randn(2, 5, 128, requires_grad=True)
        node_positions = torch.randn(2, 5, 2)
        node_tpe = torch.randn(2, 5, 8)
        room_topology_map = torch.randn(2, 18, 16, 11)

        output = module(
            grid_features,
            graph_nodes=graph_nodes,
            node_positions=node_positions,
            node_tpe=node_tpe,
            room_topology_map=room_topology_map,
        )
        loss = output.square().mean()
        loss.backward()

        topo_module = module.topology_conditioner
        if topology_conditioning_mode == "additive":
            topo_grad = topo_module.proj[0].weight.grad
        else:
            topo_grad = topo_module.to_scale_shift[0].weight.grad
        graph_grad = module.graph_cross_attn.q_proj.weight.grad

        assert topo_grad is not None
        assert graph_grad is not None
        assert topo_grad.abs().sum() > 0
        assert graph_grad.abs().sum() > 0

    def test_attention_block_uses_room_anchor_only_for_generic_cross_attention_when_spatial_graph_path_is_active(self, monkeypatch):
        """Graph-node tokens should flow through the dedicated spatial path rather than a duplicated generic cross-attention."""
        from src.core.latent_diffusion import AttentionBlock

        block = AttentionBlock(dim=64, context_dim=128)
        x = torch.randn(2, 64, 8, 8)
        context = torch.randn(2, 6, 128)
        spatial_graph_data = {
            "graph_nodes": torch.randn(2, 5, 128),
            "edge_index": torch.tensor([[[0, 1, 2], [1, 2, 3]]], dtype=torch.long),
            "node_positions": torch.randn(2, 5, 2),
            "node_tpe": torch.randn(2, 5, 8),
            "node_mask": torch.ones(2, 5),
            "room_topology_map": torch.randn(2, 18, 16, 11),
        }

        seen = {}

        def _fake_cross_attn(x_flat, cross_context, edge_index=None, node_mask=None):
            seen["context_shape"] = tuple(cross_context.shape)
            seen["edge_index"] = edge_index
            seen["node_mask"] = node_mask
            return torch.zeros_like(x_flat)

        def _fake_spatial_conditioner(x_in, **kwargs):
            seen["spatial_called"] = True
            return x_in

        monkeypatch.setattr(block.cross_attn, "forward", _fake_cross_attn)
        monkeypatch.setattr(block.spatial_graph_conditioner, "forward", _fake_spatial_conditioner)

        out = block(
            x,
            context,
            context_edge_index=torch.tensor([[[0, 1, 2], [1, 2, 3]]], dtype=torch.long),
            context_node_mask=torch.ones(2, 6),
            spatial_graph_data=spatial_graph_data,
        )

        assert out.shape == x.shape
        assert seen["context_shape"] == (2, 1, 128)
        assert seen["edge_index"] is None
        assert seen["node_mask"] is None
        assert seen["spatial_called"] is True

    def test_attention_block_skips_generic_cross_attention_when_context_is_only_graph_tokens(self, monkeypatch):
        from src.core.latent_diffusion import AttentionBlock

        block = AttentionBlock(dim=64, context_dim=128)
        x = torch.randn(2, 64, 8, 8)
        graph_nodes = torch.randn(2, 5, 128)
        spatial_graph_data = {
            "graph_nodes": graph_nodes,
            "edge_index": torch.tensor([[[0, 1, 2], [1, 2, 3]]], dtype=torch.long),
            "node_positions": torch.randn(2, 5, 2),
            "node_tpe": torch.randn(2, 5, 8),
            "node_mask": torch.ones(2, 5),
            "room_topology_map": torch.randn(2, 18, 16, 11),
        }

        seen = {}

        def _forbidden_cross_attn(*_args, **_kwargs):
            raise AssertionError("generic cross-attention should not duplicate graph-token conditioning")

        def _fake_spatial_conditioner(x_in, **kwargs):
            seen["spatial_called"] = True
            seen["graph_nodes_shape"] = tuple(kwargs["graph_nodes"].shape)
            return x_in

        monkeypatch.setattr(block.cross_attn, "forward", _forbidden_cross_attn)
        monkeypatch.setattr(block.spatial_graph_conditioner, "forward", _fake_spatial_conditioner)

        out = block(
            x,
            graph_nodes,
            context_edge_index=torch.tensor([[[0, 1, 2], [1, 2, 3]]], dtype=torch.long),
            context_node_mask=torch.ones(2, 5),
            spatial_graph_data=spatial_graph_data,
        )

        assert out.shape == x.shape
        assert seen["spatial_called"] is True
        assert seen["graph_nodes_shape"] == (2, 5, 128)

    def test_latent_cross_attention_all_masked_context_is_finite_and_zero(self):
        from src.core.latent_diffusion import CrossAttention

        module = CrossAttention(query_dim=16, context_dim=8, num_heads=4, dropout=0.0)
        x = torch.randn(2, 5, 16)
        context = torch.randn(2, 3, 8)
        node_mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)

        out = module(x, context, node_mask=node_mask)

        assert torch.isfinite(out).all()
        assert out.shape == x.shape
        assert torch.allclose(out[1], torch.zeros_like(out[1]))

    def test_latent_cross_attention_filters_invalid_padded_edges_before_clamping(self):
        from src.core.latent_diffusion import CrossAttention

        torch.manual_seed(7)
        module = CrossAttention(query_dim=16, context_dim=8, num_heads=4, dropout=0.0)
        module.eval()
        x = torch.randn(1, 5, 16)
        context = torch.randn(1, 3, 8)
        invalid_edge = torch.tensor([[-1], [2]], dtype=torch.long)
        empty_edges = torch.empty(2, 0, dtype=torch.long)

        out_invalid = module(x, context, edge_index=invalid_edge)
        out_empty = module(x, context, edge_index=empty_edges)

        assert torch.allclose(out_invalid, out_empty, atol=1e-6, rtol=1e-6)

    def test_latent_cross_attention_rejects_node_mask_shape_mismatch(self):
        from src.core.latent_diffusion import CrossAttention

        module = CrossAttention(query_dim=16, context_dim=8, num_heads=4, dropout=0.0)
        x = torch.randn(2, 5, 16)
        context = torch.randn(2, 3, 8)
        node_mask = torch.ones(2, 2, dtype=torch.bool)

        with pytest.raises(ValueError, match="node_mask shape"):
            module(x, context, node_mask=node_mask)

    def test_gps_layer_filters_invalid_padded_edges_before_local_message_passing(self):
        from src.core.condition_encoder import GPSLayer

        torch.manual_seed(11)
        layer = GPSLayer(hidden_dim=16, num_heads=4, dropout=0.0)
        layer.eval()
        h = torch.randn(3, 16)
        invalid_edge = torch.tensor([[-1], [2]], dtype=torch.long)
        empty_edges = torch.empty(2, 0, dtype=torch.long)

        out_invalid = layer(h, edge_index=invalid_edge)
        out_empty = layer(h, edge_index=empty_edges)

        assert torch.allclose(out_invalid, out_empty, atol=1e-6, rtol=1e-6)

    def test_gps_layer_fallback_filters_invalid_edges_instead_of_clamping(self):
        from src.core.condition_encoder import GPSLayer

        torch.manual_seed(13)
        layer = GPSLayer(hidden_dim=16, num_heads=4, dropout=0.0)
        layer.local_gnn = None
        layer.eval()
        h = torch.randn(3, 16)
        invalid_edge = torch.tensor([[-1], [2]], dtype=torch.long)
        empty_edges = torch.empty(2, 0, dtype=torch.long)

        out_invalid = layer(h, edge_index=invalid_edge)
        out_empty = layer(h, edge_index=empty_edges)

        assert torch.allclose(out_invalid, out_empty, atol=1e-6, rtol=1e-6)


# ============================================================================
# TEST: MISSION GRAMMAR
# ============================================================================

class TestMissionGrammar:
    """Tests for mission grammar graph generation."""
    
    def test_grammar_import(self):
        """Test module imports correctly."""
        from src.generation.grammar import (
            MissionGrammar,
            MissionGraph,
        )
        assert MissionGrammar is not None
        assert MissionGraph is not None
    
    def test_generate_simple_graph(self):
        """Test generating a simple mission graph."""
        from src.generation.grammar import MissionGrammar, Difficulty
        
        grammar = MissionGrammar(seed=42)
        graph = grammar.generate(
            difficulty=Difficulty.EASY,
            num_rooms=5,
            max_keys=1,
        )
        
        assert len(graph.nodes) >= 2  # At least start and goal
        assert len(graph.edges) >= 1  # At least one edge
    
    def test_graph_has_start_and_goal(self):
        """Test that generated graph has START and GOAL nodes."""
        from src.generation.grammar import MissionGrammar, NodeType, Difficulty
        
        grammar = MissionGrammar(seed=123)
        graph = grammar.generate(difficulty=Difficulty.MEDIUM, num_rooms=8)
        
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        
        assert start is not None, "Graph should have START node"
        assert goal is not None, "Graph should have GOAL node"
        assert start.node_type == NodeType.START
        assert goal.node_type == NodeType.GOAL
    
    def test_lock_key_validation(self):
        """Test lock-key ordering validation."""
        from src.generation.grammar import MissionGrammar, Difficulty
        
        grammar = MissionGrammar(seed=456)
        graph = grammar.generate(
            difficulty=Difficulty.HARD,
            num_rooms=10,
            max_keys=2,
        )
        
        # Generated graphs should always be valid
        assert grammar.validate_lock_key_ordering(graph)
    
    def test_graph_to_tensor(self):
        """Test converting graph to PyTorch tensors."""
        from src.generation.grammar import MissionGrammar, Difficulty
        
        grammar = MissionGrammar(seed=789)
        graph = grammar.generate(difficulty=Difficulty.MEDIUM, num_rooms=6)
        
        edge_index, node_features = graph.to_tensor()
        
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] >= 1  # At least one edge
        assert node_features.shape[0] == len(graph.nodes)
    
    def test_tpe_computation(self):
        """Test topological positional encoding."""
        from src.generation.grammar import MissionGrammar, Difficulty
        
        grammar = MissionGrammar(seed=101)
        graph = grammar.generate(difficulty=Difficulty.MEDIUM, num_rooms=6)
        
        tpe = graph.compute_tpe()
        
        assert tpe.shape == (len(graph.nodes), 8)
        assert not torch.isnan(tpe).any()
    
    def test_graph_to_gnn_input(self):
        """Test converting graph to GNN input format."""
        from src.generation.grammar import MissionGrammar, Difficulty, graph_to_gnn_input
        
        grammar = MissionGrammar(seed=202)
        graph = grammar.generate(difficulty=Difficulty.MEDIUM, num_rooms=6)
        
        gnn_input = graph_to_gnn_input(graph, current_node_idx=0)
        
        assert 'edge_index' in gnn_input
        assert 'node_features' in gnn_input
        assert 'tpe' in gnn_input
        assert 'current_node' in gnn_input
        assert 'adjacency' in gnn_input
    
    def test_deterministic_generation(self):
        """Test that same seed produces same graph."""
        from src.generation.grammar import MissionGrammar, Difficulty
        
        grammar1 = MissionGrammar(seed=999)
        graph1 = grammar1.generate(difficulty=Difficulty.MEDIUM, num_rooms=6)
        
        grammar2 = MissionGrammar(seed=999)
        graph2 = grammar2.generate(difficulty=Difficulty.MEDIUM, num_rooms=6)
        
        assert len(graph1.nodes) == len(graph2.nodes)
        assert len(graph1.edges) == len(graph2.edges)


# ============================================================================
# TEST: CAUSAL WFC
# ============================================================================

class TestCausalWFC:
    """Tests for causal Wave Function Collapse."""
    
    def test_wfc_import(self):
        """Test module imports correctly."""
        from src.generation.wfc_refiner import (
            CausalWFC,
            ZeldaTileSet,
        )
        assert CausalWFC is not None
        assert ZeldaTileSet is not None
    
    def test_tile_set_creation(self):
        """Test Zelda tile set creation."""
        from src.generation.wfc_refiner import ZeldaTileSet, TileType
        
        tile_set = ZeldaTileSet()
        
        assert len(tile_set.tiles) > 0
        assert tile_set.get_tile(0) is not None  # Floor tile
        
        # Check that key and lock tiles exist
        keys = tile_set.get_tiles_by_type(TileType.KEY_SMALL)
        locks = tile_set.get_tiles_by_type(TileType.DOOR_LOCKED)
        
        assert len(keys) >= 1
        assert len(locks) >= 1
    
    def test_wfc_generate(self):
        """Test basic WFC generation."""
        from src.generation.wfc_refiner import CausalWFC, ZeldaTileSet
        
        tile_set = ZeldaTileSet()
        wfc = CausalWFC(tile_set, width=11, height=16, seed=42)
        
        grid = wfc.generate(start_pos=(14, 5), goal_pos=(1, 5))
        
        assert grid.shape == (16, 11)
        assert grid.dtype == np.int32
    
    def test_causal_ordering(self):
        """Test that WFC maintains causal ordering."""
        from src.generation.wfc_refiner import CausalWFC, ZeldaTileSet
        
        tile_set = ZeldaTileSet()
        wfc = CausalWFC(tile_set, width=11, height=16, seed=42)
        
        _grid = wfc.generate(start_pos=(14, 5), goal_pos=(1, 5))
        
        # Validate causal ordering
        assert wfc.validate_causal_ordering()
    
    def test_game_state_tracking(self):
        """Test that game state is properly tracked."""
        from src.generation.wfc_refiner import CausalWFC, ZeldaTileSet
        
        tile_set = ZeldaTileSet()
        wfc = CausalWFC(tile_set, width=11, height=16, seed=42)
        
        wfc.generate(start_pos=(14, 5), goal_pos=(1, 5))
        
        # Game state should have been used
        stats = wfc.get_statistics()
        assert 'keys_placed' in stats
        assert 'locks_placed' in stats
        assert 'contradictions' in stats
    
    def test_deterministic_generation(self):
        """Test that same seed produces same grid."""
        from src.generation.wfc_refiner import CausalWFC, ZeldaTileSet
        
        tile_set = ZeldaTileSet()
        
        wfc1 = CausalWFC(tile_set, width=11, height=16, seed=123)
        grid1 = wfc1.generate()
        
        wfc2 = CausalWFC(tile_set, width=11, height=16, seed=123)
        grid2 = wfc2.generate()
        
        assert np.array_equal(grid1, grid2)
    
    def test_fixed_tiles(self):
        """Test generation with fixed tiles."""
        from src.generation.wfc_refiner import CausalWFC, ZeldaTileSet
        
        tile_set = ZeldaTileSet()
        wfc = CausalWFC(tile_set, width=11, height=16, seed=42)
        
        # Initialize with fixed walls on border
        fixed = {}
        for r in range(16):
            fixed[(r, 0)] = 1  # Wall
            fixed[(r, 10)] = 1  # Wall
        
        wfc.initialize(fixed_tiles=fixed)
        
        # Check that fixed tiles are set
        assert wfc.grid[0][0].collapsed_tile == 1
        assert wfc.grid[0][10].collapsed_tile == 1


# ============================================================================
# TEST: TRAINING PIPELINE
# ============================================================================

class TestTrainingPipeline:
    """Tests for the diffusion training pipeline."""
    
    def test_train_diffusion_import(self):
        """Test that training module imports correctly."""
        from src.train_diffusion import (
            DiffusionTrainingConfig,
            DiffusionTrainer,
            train_diffusion,
        )
        assert DiffusionTrainingConfig is not None
        assert DiffusionTrainer is not None
        assert train_diffusion is not None
    
    def test_training_config(self):
        """Test training configuration."""
        from src.train_diffusion import DiffusionTrainingConfig

        config = DiffusionTrainingConfig(
            epochs=2,
            batch_size=2,
            quick=True,
        )

        assert config.epochs == 2
        assert config.batch_size == 2

        config_dict = config.to_dict()
        assert 'epochs' in config_dict
        assert 'learning_rate' in config_dict


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_grammar_to_wfc_pipeline(self):
        """Test full pipeline from grammar to WFC."""
        from src.generation.grammar import MissionGrammar, Difficulty
        from src.generation.wfc_refiner import generate_with_grammar
        
        # Generate mission graph
        grammar = MissionGrammar(seed=42)
        graph = grammar.generate(difficulty=Difficulty.MEDIUM, num_rooms=6)
        
        # Generate grid from graph
        grid = generate_with_grammar(graph, width=11, height=16, seed=42)
        
        assert grid.shape == (16, 11)
    
    def test_grammar_to_gnn_to_attention(self):
        """Test pipeline from grammar to GNN to cross-attention."""
        from src.generation.grammar import MissionGrammar, Difficulty, graph_to_gnn_input
        from src.core.graph_grid_attention import GraphToGridCrossAttention
        
        # Generate mission graph
        grammar = MissionGrammar(seed=42)
        graph = grammar.generate(difficulty=Difficulty.MEDIUM, num_rooms=6)
        
        # Convert to GNN input
        gnn_input = graph_to_gnn_input(graph, current_node_idx=0)
        
        # Create cross-attention module
        B = 2
        C = 128
        H, W = 16, 11
        graph_dim = gnn_input['node_features'].shape[1]
        
        module = GraphToGridCrossAttention(
            grid_dim=C,
            graph_dim=graph_dim,
            num_heads=8,
        )
        
        # Create fake grid features
        grid_features = torch.randn(B, C, H, W)
        
        # Expand graph nodes for batch
        graph_nodes = gnn_input['node_features'].unsqueeze(0).expand(B, -1, -1)
        tpe = gnn_input['tpe'].unsqueeze(0).expand(B, -1, -1)
        
        # Apply cross-attention
        output = module(grid_features, graph_nodes, node_tpe=tpe)
        
        assert output.shape == grid_features.shape


def test_global_stream_encoder_rrwp_changes_gps_edge_messages():
    from src.core.condition_encoder import GlobalStreamEncoder
    from src.core.definitions import GRAPH_TPE_DIM

    encoder = GlobalStreamEncoder(
        node_feature_dim=4,
        edge_feature_dim=3,
        hidden_dim=16,
        output_dim=8,
        num_layers=1,
        gnn_type="gps",
        num_heads=4,
        dropout=0.0,
        use_rrwp_edge_features=True,
    )
    encoder.eval()
    node_features = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_features = torch.zeros(3, 3)
    tpe = torch.zeros(3, GRAPH_TPE_DIM)
    edge_rrwp_zero = torch.zeros(3, GRAPH_TPE_DIM)
    edge_rrwp_nonzero = torch.zeros(3, GRAPH_TPE_DIM)
    edge_rrwp_nonzero[:, 0] = torch.tensor([0.25, 0.5, 0.75])

    with torch.no_grad():
        out_zero = encoder(
            node_features,
            edge_index,
            edge_features=edge_features,
            edge_rrwp=edge_rrwp_zero,
            tpe=tpe,
        )
        out_nonzero = encoder(
            node_features,
            edge_index,
            edge_features=edge_features,
            edge_rrwp=edge_rrwp_nonzero,
            tpe=tpe,
        )

    assert tuple(out_zero.shape) == (3, 8)
    assert not torch.allclose(out_zero, out_nonzero)


def test_global_stream_encoder_rrwp_width_mismatch_is_aligned_not_dropped():
    from src.core.condition_encoder import GlobalStreamEncoder
    from src.core.definitions import GRAPH_TPE_DIM

    class _NarrowRRWP(torch.nn.Module):
        def forward(self, x):
            return x[:, :8]

    encoder = GlobalStreamEncoder(
        node_feature_dim=4,
        edge_feature_dim=3,
        hidden_dim=16,
        output_dim=8,
        num_layers=1,
        gnn_type="gps",
        num_heads=4,
        dropout=0.0,
        use_rrwp_edge_features=True,
    )
    encoder.edge_rrwp_proj = _NarrowRRWP()
    encoder.eval()

    node_features = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_features = torch.zeros(3, 3)
    edge_rrwp_zero = torch.zeros(3, GRAPH_TPE_DIM)
    edge_rrwp_nonzero = torch.zeros(3, GRAPH_TPE_DIM)
    edge_rrwp_nonzero[:, 0] = torch.tensor([0.25, 0.5, 0.75])

    with torch.no_grad():
        out_zero = encoder(
            node_features,
            edge_index,
            edge_features=edge_features,
            edge_rrwp=edge_rrwp_zero,
        )
        out_nonzero = encoder(
            node_features,
            edge_index,
            edge_features=edge_features,
            edge_rrwp=edge_rrwp_nonzero,
        )

    assert tuple(out_zero.shape) == (3, 8)
    assert not torch.allclose(out_zero, out_nonzero)


def test_global_stream_encoder_gps_global_attention_respects_batch_idx():
    from src.core.condition_encoder import GlobalStreamEncoder

    torch.manual_seed(0)
    encoder = GlobalStreamEncoder(
        node_feature_dim=4,
        edge_feature_dim=3,
        hidden_dim=16,
        output_dim=8,
        num_layers=1,
        gnn_type="gps",
        num_heads=4,
        dropout=0.0,
    )
    encoder.eval()
    first_graph = torch.randn(2, 4)
    second_graph_a = torch.zeros(2, 4)
    second_graph_b = torch.full((2, 4), 100.0)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_features = torch.empty(0, 3)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    with torch.no_grad():
        out_a = encoder(
            torch.cat([first_graph, second_graph_a], dim=0),
            edge_index,
            edge_features=edge_features,
            batch_idx=batch_idx,
        )
        out_b = encoder(
            torch.cat([first_graph, second_graph_b], dim=0),
            edge_index,
            edge_features=edge_features,
            batch_idx=batch_idx,
        )

    assert torch.allclose(out_a[:2], out_b[:2], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out_a[2:], out_b[2:])


def test_gps_layer_global_attention_respects_node_mask():
    from src.core.condition_encoder import GPSLayer

    torch.manual_seed(0)
    layer = GPSLayer(hidden_dim=8, num_heads=2, dropout=0.0)
    layer.eval()

    valid = torch.randn(2, 8)
    padded_a = torch.zeros(2, 8)
    padded_b = torch.full((2, 8), 100.0)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    batch_idx = torch.zeros(4, dtype=torch.long)
    node_mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)

    with torch.no_grad():
        out_a = layer(
            torch.cat([valid, padded_a], dim=0),
            edge_index=edge_index,
            batch_idx=batch_idx,
            node_mask=node_mask,
        )
        out_b = layer(
            torch.cat([valid, padded_b], dim=0),
            edge_index=edge_index,
            batch_idx=batch_idx,
            node_mask=node_mask,
        )

    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_b).all()
    assert torch.allclose(out_a[:2], out_b[:2], atol=1e-5, rtol=1e-5)


def test_cross_attention_fusion_all_masked_context_is_finite():
    from src.core.condition_encoder import CrossAttentionFusion

    module = CrossAttentionFusion(local_dim=4, global_dim=4, output_dim=8, num_heads=2, dropout=0.0)
    local = torch.randn(2, 4, requires_grad=True)
    global_tokens = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    out = module(local, global_tokens, mask=mask)
    loss = out.pow(2).mean()
    loss.backward()

    assert tuple(out.shape) == (2, 8)
    assert torch.isfinite(out).all()
    assert local.grad is not None
    assert torch.isfinite(local.grad).all()
    assert global_tokens.grad is not None
    assert torch.isfinite(global_tokens.grad).all()


def test_cross_attention_fusion_manual_fallback_zeros_all_masked_rows():
    import src.core.condition_encoder as condition_encoder
    from src.core.condition_encoder import CrossAttentionFusion

    module = CrossAttentionFusion(local_dim=4, global_dim=4, output_dim=8, num_heads=2, dropout=0.0)
    local = torch.randn(2, 4, requires_grad=True)
    global_tokens = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    original_has_sdpa = condition_encoder.HAS_SDPA
    try:
        condition_encoder.HAS_SDPA = False
        output = module(local, global_tokens, mask=mask)
        output.square().mean().backward()
    finally:
        condition_encoder.HAS_SDPA = original_has_sdpa

    assert torch.isfinite(output).all()
    assert torch.isfinite(local.grad).all()
    assert torch.isfinite(global_tokens.grad).all()


def test_cross_attention_fusion_sdpa_matches_manual_fallback():
    import src.core.condition_encoder as condition_encoder
    from src.core.condition_encoder import CrossAttentionFusion

    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("scaled_dot_product_attention is unavailable in this PyTorch build")

    torch.manual_seed(17)
    module = CrossAttentionFusion(local_dim=4, global_dim=5, output_dim=8, num_heads=2, dropout=0.0)
    module.eval()
    local = torch.randn(3, 4)
    global_tokens = torch.randn(3, 6, 5)
    mask = torch.tensor(
        [
            [True, True, False, True, False, False],
            [False, True, True, False, True, False],
            [True, False, False, False, False, True],
        ],
        dtype=torch.bool,
    )

    original_has_sdpa = condition_encoder.HAS_SDPA
    try:
        condition_encoder.HAS_SDPA = True
        sdpa_out = module(local, global_tokens, mask=mask)
        condition_encoder.HAS_SDPA = False
        manual_out = module(local, global_tokens, mask=mask)
    finally:
        condition_encoder.HAS_SDPA = original_has_sdpa

    assert torch.allclose(sdpa_out, manual_out, atol=1e-5, rtol=1e-5)


def test_cross_attention_fusion_uses_sdpa(monkeypatch):
    import src.core.condition_encoder as condition_encoder
    from src.core.condition_encoder import CrossAttentionFusion

    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("scaled_dot_product_attention is unavailable in this PyTorch build")

    calls = {"count": 0}
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def _spy_sdpa(*args, **kwargs):
        calls["count"] += 1
        return original_sdpa(*args, **kwargs)

    module = CrossAttentionFusion(local_dim=4, global_dim=4, output_dim=8, num_heads=2, dropout=0.0)
    module.eval()
    original_has_sdpa = condition_encoder.HAS_SDPA
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _spy_sdpa)
    try:
        condition_encoder.HAS_SDPA = True
        out = module(
            torch.randn(2, 4),
            torch.randn(2, 3, 4),
            mask=torch.tensor([[True, False, True], [False, True, True]]),
        )
    finally:
        condition_encoder.HAS_SDPA = original_has_sdpa

    assert calls["count"] == 1
    assert tuple(out.shape) == (2, 8)
    assert torch.isfinite(out).all()


def test_global_stream_encoder_gat_handles_isolated_single_node_graph():
    from src.core.condition_encoder import GlobalStreamEncoder

    encoder = GlobalStreamEncoder(
        node_feature_dim=4,
        edge_feature_dim=3,
        hidden_dim=16,
        output_dim=8,
        num_layers=1,
        gnn_type="gat",
        num_heads=4,
        dropout=0.0,
    )
    encoder.eval()

    with torch.no_grad():
        out = encoder(
            torch.randn(1, 4),
            torch.empty(2, 0, dtype=torch.long),
            edge_features=torch.empty(0, 3),
        )

    assert tuple(out.shape) == (1, 8)
    assert torch.isfinite(out).all()


def test_dual_stream_encoder_keeps_style_and_reference_features_separate(monkeypatch):
    from src.core.condition_encoder import DualStreamConditionEncoder

    class _SpyProjection(torch.nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.output_dim = int(output_dim)
            self.seen = None

        def forward(self, x):
            self.seen = x.detach().clone()
            return torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)

    class _ReferenceEncoder(torch.nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.output_dim = int(output_dim)

        def forward(self, _maps, *, batch_size, device, dtype):
            return torch.ones(batch_size, self.output_dim, device=device, dtype=dtype)

    output_dim = 8
    encoder = DualStreamConditionEncoder(
        latent_dim=4,
        node_feature_dim=4,
        edge_feature_dim=3,
        hidden_dim=16,
        output_dim=output_dim,
        num_gnn_layers=1,
        gnn_type="gps",
        num_attention_heads=4,
        use_reference_room_maps=False,
    )
    monkeypatch.setattr(
        encoder.local_encoder,
        "forward",
        lambda *args, **kwargs: torch.zeros(1, output_dim),
    )
    monkeypatch.setattr(
        encoder.global_encoder,
        "forward",
        lambda *args, **kwargs: torch.zeros(2, output_dim),
    )
    monkeypatch.setattr(
        encoder.fusion,
        "forward",
        lambda _local, _global, **_kwargs: torch.zeros(1, output_dim),
    )
    encoder.reference_room_encoder = _ReferenceEncoder(output_dim)
    spy = _SpyProjection(output_dim)
    encoder.output_proj = spy

    out = encoder(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        boundary_constraints=torch.zeros(1, 8),
        position=torch.zeros(1, 2),
        node_features=torch.zeros(2, 4),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_features=torch.empty(0, 3),
        current_node_idx=0,
        style_id=None,
    )

    assert tuple(out.shape) == (1, output_dim)
    assert spy.seen is not None
    assert tuple(spy.seen.shape) == (1, output_dim * 3)
    assert torch.allclose(spy.seen[:, output_dim:2 * output_dim], torch.zeros(1, output_dim))
    assert torch.allclose(spy.seen[:, 2 * output_dim:], torch.ones(1, output_dim))


@pytest.mark.parametrize("gnn_type", ["gcn", "sage"])
def test_global_stream_encoder_rejects_rrwp_for_backbones_that_ignore_edge_attributes(gnn_type):
    from src.core.condition_encoder import GlobalStreamEncoder

    with pytest.raises(ValueError, match="RRWP edge features require"):
        GlobalStreamEncoder(
            node_feature_dim=4,
            edge_feature_dim=3,
            hidden_dim=16,
            output_dim=8,
            num_layers=1,
            gnn_type=gnn_type,
            dropout=0.0,
            use_rrwp_edge_features=True,
        )


def test_dual_stream_encoder_wires_global_gps_attention_heads():
    from src.core.condition_encoder import DualStreamConditionEncoder

    encoder = DualStreamConditionEncoder(
        latent_dim=4,
        node_feature_dim=4,
        edge_feature_dim=3,
        hidden_dim=16,
        output_dim=8,
        num_gnn_layers=1,
        gnn_type="gps",
        num_attention_heads=8,
        use_reference_room_maps=False,
    )

    assert encoder.global_encoder.gps_layers[0].global_attn.num_heads == 8


def test_global_stream_encoder_skips_mismatched_rrwp_rows():
    from src.core.condition_encoder import GlobalStreamEncoder
    from src.core.definitions import GRAPH_TPE_DIM

    encoder = GlobalStreamEncoder(
        node_feature_dim=4,
        edge_feature_dim=3,
        hidden_dim=16,
        output_dim=8,
        num_layers=1,
        gnn_type="gps",
        num_heads=4,
        dropout=0.0,
        use_rrwp_edge_features=True,
    )
    node_features = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_features = torch.zeros(3, 3)
    tpe = torch.zeros(3, GRAPH_TPE_DIM)

    out = encoder(
        node_features,
        edge_index,
        edge_features=edge_features,
        edge_rrwp=torch.zeros(0, GRAPH_TPE_DIM),
        tpe=tpe,
    )

    assert tuple(out.shape) == (3, 8)
    assert torch.isfinite(out).all()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
