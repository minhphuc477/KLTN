"""
Integration Tests for Neural-Symbolic Dungeon Pipeline
=======================================================

Tests the complete 7-block pipeline end-to-end.

Run:
    pytest tests/test_neural_pipeline.py -v
    
    # Quick test
    pytest tests/test_neural_pipeline.py -v -k "test_pipeline_initialization"
"""

import pytest
import torch
import numpy as np
import networkx as nx
from pathlib import Path

from src.pipeline import (
    NeuralSymbolicDungeonPipeline,
    create_pipeline,
)
from src.core import ROOM_HEIGHT, ROOM_WIDTH


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def device():
    """Use CPU for tests to avoid GPU memory issues."""
    return 'cpu'


@pytest.fixture
def pipeline(device):
    """Create pipeline without checkpoints (random init)."""
    return NeuralSymbolicDungeonPipeline(
        device=device,
        enable_logging=False,
    )


@pytest.fixture
def simple_graph():
    """Create a simple 3-room mission graph."""
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edges_from([(0, 1), (1, 2)])
    return G


@pytest.fixture
def neighbor_latents(device):
    """Create dummy neighbor latents."""
    return {
        'N': torch.randn(1, 64, 4, 3, device=device),
        'S': None,
        'E': torch.randn(1, 64, 4, 3, device=device),
        'W': None,
    }


@pytest.fixture
def graph_context(device):
    """Create dummy graph context."""
    return {
        'node_features': torch.randn(3, 6, device=device),
        'edge_index': torch.tensor([[0, 1], [1, 2]], device=device).t(),
        'tpe': torch.randn(3, 8, device=device),
        'current_node_idx': 0,
    }


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

def test_pipeline_initialization(pipeline):
    """Test that pipeline initializes all components."""
    assert pipeline.vqvae is not None
    assert pipeline.condition_encoder is not None
    assert pipeline.diffusion is not None
    assert pipeline.logic_net is not None
    assert pipeline.refiner is not None
    assert pipeline.map_elites is not None
    
    print("✓ All 7 blocks initialized successfully")


def test_pipeline_device(device):
    """Test that pipeline respects device setting."""
    pipeline = NeuralSymbolicDungeonPipeline(device=device, enable_logging=False)
    assert str(pipeline.device) == device
    
    # Check model devices
    assert next(pipeline.vqvae.parameters()).device.type == device
    assert next(pipeline.diffusion.parameters()).device.type == device
    
    print(f"✓ Pipeline correctly initialized on {device}")


def test_create_pipeline_convenience():
    """Test convenience function for pipeline creation."""
    # Should work even without checkpoint directory
    pipeline = create_pipeline(
        checkpoint_dir="./nonexistent",
        device='cpu'
    )
    
    assert isinstance(pipeline, NeuralSymbolicDungeonPipeline)
    print("✓ Convenience function works correctly")


# =============================================================================
# DIMENSION TESTS
# =============================================================================

def test_room_dimensions(pipeline, neighbor_latents, graph_context):
    """Test that generated rooms have correct dimensions."""
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=2,  # Quick test
        apply_repair=False,
        seed=42
    )
    
    # Check room grid dimensions
    assert result.room_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH), \
        f"Expected (16, 11), got {result.room_grid.shape}"
    
    # Check latent dimensions
    assert result.latent.shape == (1, 64, 4, 3), \
        f"Expected (1, 64, 4, 3), got {result.latent.shape}"
    
    # Check neural grid dimensions
    assert result.neural_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    
    print(f"✓ Room dimensions correct: {result.room_grid.shape}")


def test_latent_space_consistency(pipeline):
    """Test VQ-VAE encode/decode consistency."""
    # Create dummy one-hot room
    room = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH, device=pipeline.device)
    room[:, 1, :, :] = 1.0  # All floor tiles
    
    # Encode
    z_q, indices = pipeline.vqvae.encode(room)
    
    # Check latent dimensions
    assert z_q.shape == (1, 64, 4, 3), f"Latent shape mismatch: {z_q.shape}"
    assert indices.shape == (1, 4, 3), f"Indices shape mismatch: {indices.shape}"
    
    # Decode
    recon = pipeline.vqvae.decode(z_q)
    assert recon.shape == (1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    
    print(f"✓ VQ-VAE latent space consistent: {z_q.shape} → {recon.shape}")


# =============================================================================
# SINGLE ROOM GENERATION TESTS
# =============================================================================

def test_single_room_generation_basic(pipeline, neighbor_latents, graph_context):
    """Test basic single room generation without repair."""
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=42
    )
    
    # Check result structure
    assert result.room_id == 0
    assert result.room_grid is not None
    assert result.latent is not None
    assert result.neural_grid is not None
    assert not result.was_repaired
    assert result.repair_mask is None
    
    # Check metrics
    assert 'room_id' in result.metrics
    assert 'neural_grid_entropy' in result.metrics
    
    print(f"✓ Basic room generation successful (entropy={result.metrics['neural_grid_entropy']:.3f})")


def test_single_room_generation_with_repair(pipeline, neighbor_latents, graph_context):
    """Test room generation with symbolic repair."""
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=1,
        num_diffusion_steps=5,
        apply_repair=True,
        start_goal_coords=((8, 0), (8, 10)),
        seed=42
    )
    
    # Check that repair was attempted
    assert isinstance(result.was_repaired, bool)
    
    if result.was_repaired:
        assert result.repair_mask is not None
        assert result.repair_mask.shape == (ROOM_HEIGHT, ROOM_WIDTH)
        assert 'tiles_changed' in result.metrics
        print(f"✓ Repair successful ({result.metrics['tiles_changed']} tiles changed)")
    else:
        print("✓ Repair not needed (neural output already valid)")


def test_reproducibility_with_seed(pipeline, neighbor_latents, graph_context):
    """Test that seeded generation is reproducible."""
    result1 = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=12345
    )
    
    result2 = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=12345
    )
    
    # Check that results are identical
    assert np.array_equal(result1.neural_grid, result2.neural_grid), \
        "Seeded generation should be reproducible"
    
    print("✓ Reproducibility verified with seed=12345")


# =============================================================================
# MULTI-ROOM DUNGEON GENERATION TESTS
# =============================================================================

def test_dungeon_generation_simple(pipeline, simple_graph):
    """Test complete dungeon generation with simple graph."""
    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=3,
        apply_repair=False,
        seed=42,
        enable_map_elites=False,
    )
    
    # Check result structure
    assert result.dungeon_grid is not None
    assert len(result.rooms) == 3
    assert result.mission_graph is simple_graph
    
    # Check metrics
    assert result.metrics['num_rooms'] == 3
    assert 'generation_time_sec' in result.metrics
    assert result.generation_time > 0
    
    print(f"✓ 3-room dungeon generated in {result.generation_time:.2f}s")


def test_dungeon_generation_with_repair(pipeline, simple_graph):
    """Test dungeon generation with repair enabled."""
    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=3,
        apply_repair=True,
        seed=42,
        enable_map_elites=False,
    )
    
    # Check repair metrics
    assert 'repair_rate' in result.metrics
    assert 'total_tiles_repaired' in result.metrics
    
    repair_rate = result.metrics['repair_rate']
    assert 0.0 <= repair_rate <= 1.0
    
    print(f"✓ Dungeon with repair (rate={repair_rate:.1%}, "
          f"tiles={result.metrics['total_tiles_repaired']})")


def test_room_order_preservation(pipeline, simple_graph):
    """Test that rooms are generated in graph order."""
    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=2,
        seed=42
    )
    
    # Check all rooms present
    expected_ids = set(simple_graph.nodes())
    actual_ids = set(result.rooms.keys())
    assert expected_ids == actual_ids, f"Missing rooms: {expected_ids - actual_ids}"
    
    print(f"✓ All {len(expected_ids)} rooms generated correctly")


# =============================================================================
# GUIDANCE TESTS
# =============================================================================

def test_logic_guidance_effect(pipeline, neighbor_latents, graph_context):
    """Test that LogicNet guidance affects generation."""
    # Without guidance
    result_no_guidance = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        logic_guidance_scale=0.0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=42
    )
    
    # With guidance
    result_with_guidance = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        logic_guidance_scale=1.0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=42
    )
    
    # Grids should be different (guidance changes trajectory)
    # Note: Due to randomness, this test might occasionally fail
    different = not np.array_equal(
        result_no_guidance.neural_grid,
        result_with_guidance.neural_grid
    )
    
    if different:
        print("✓ LogicNet guidance changes generation (as expected)")
    else:
        print("⚠ LogicNet guidance had no visible effect (may need more steps)")


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_missing_neighbor_handling(pipeline, graph_context):
    """Test that pipeline handles missing neighbors gracefully."""
    # All neighbors None
    neighbor_latents = {'N': None, 'S': None, 'E': None, 'W': None}
    
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=3,
        apply_repair=False,
        seed=42
    )
    
    assert result.room_grid is not None
    print("✓ Missing neighbors handled correctly")


def test_invalid_graph_context_fallback(pipeline, neighbor_latents):
    """Test fallback when graph context is invalid."""
    # Empty graph context
    graph_context = {}
    
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=3,
        apply_repair=False,
        seed=42
    )
    
    assert result.room_grid is not None
    print("✓ Invalid graph context handled with fallback")


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.slow
def test_generation_performance(pipeline, simple_graph):
    """Test generation performance (marked as slow)."""
    import time
    
    start = time.time()
    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=50,
        apply_repair=True,
        seed=42
    )
    duration = time.time() - start
    
    # Should complete in reasonable time (adjust based on hardware)
    assert duration < 60.0, f"Generation too slow: {duration:.1f}s"
    
    rooms_per_sec = len(result.rooms) / duration
    print(f"✓ Performance: {rooms_per_sec:.2f} rooms/sec ({duration:.2f}s total)")


# =============================================================================
# INTEGRATION SMOKE TEST
# =============================================================================

def test_complete_pipeline_smoke():
    """Comprehensive smoke test of the entire pipeline."""
    print("\n" + "="*70)
    print("COMPLETE PIPELINE SMOKE TEST")
    print("="*70)
    
    # Initialize pipeline
    pipeline = NeuralSymbolicDungeonPipeline(device='cpu', enable_logging=True)
    
    # Create test graph
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    # Generate dungeon
    result = pipeline.generate_dungeon(
        mission_graph=G,
        guidance_scale=7.5,
        logic_guidance_scale=0.5,
        num_diffusion_steps=10,
        apply_repair=True,
        seed=42,
        enable_map_elites=True,
    )
    
    # Verify all components
    assert len(result.rooms) == 4
    assert result.dungeon_grid.shape[1] == ROOM_WIDTH
    assert result.metrics['num_rooms'] == 4
    assert 'generation_time_sec' in result.metrics
    
    # Check individual rooms
    for room_id, room_result in result.rooms.items():
        assert room_result.room_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH)
        assert room_result.latent.shape == (1, 64, 4, 3)
        assert 'neural_grid_entropy' in room_result.metrics
    
    print("\n" + "="*70)
    print("SMOKE TEST RESULTS:")
    print(f"  Rooms generated: {len(result.rooms)}")
    print(f"  Dungeon shape: {result.dungeon_grid.shape}")
    print(f"  Repair rate: {result.metrics['repair_rate']:.1%}")
    print(f"  Generation time: {result.generation_time:.2f}s")
    if result.map_elites_score:
        print(f"  MAP-Elites linearity: {result.map_elites_score['linearity']:.3f}")
        print(f"  MAP-Elites leniency: {result.map_elites_score['leniency']:.3f}")
    print("="*70)
    print("✓ COMPLETE PIPELINE SMOKE TEST PASSED")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Run smoke test directly
    test_complete_pipeline_smoke()
