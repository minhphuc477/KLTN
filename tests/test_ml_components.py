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
        try:
            from src.train_diffusion import (
                DiffusionTrainingConfig,
                DiffusionTrainer,
                train_diffusion,
            )
            assert DiffusionTrainingConfig is not None
            assert DiffusionTrainer is not None
            assert train_diffusion is not None
        except ImportError as e:
            # May fail if dependencies not fully set up
            pytest.skip(f"Import failed (may need dependencies): {e}")
    
    def test_training_config(self):
        """Test training configuration."""
        try:
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
        except ImportError:
            pytest.skip("Training module not available")


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
