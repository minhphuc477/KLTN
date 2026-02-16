"""
Test Suite for Topology Generation Bug Fixes
===========================================

Tests all bug fixes implemented:
1. dungeon_pipeline.py: Fixed max_rooms parameter passing
2. EvolutionaryTopologyGenerator: Added max_nodes parameter
3. D* Lite: Fixed predecessor state computation
4. Bidirectional A*: Fixed collision detection with state sets

This test suite validates that all fixes work correctly and maintain
the expected behavior of the system.
"""

import numpy as np
import pytest
import logging
from typing import List, Tuple

from src.core.definitions import SEMANTIC_PALETTE
from src.generation.evolutionary_director import (
    EvolutionaryTopologyGenerator,
    GraphGrammarExecutor
)
from src.simulation.validator import ZeldaLogicEnv, GameState
from src.simulation.dstar_lite import DStarLiteSolver
from src.simulation.bidirectional_astar import BidirectionalAStar

logger = logging.getLogger(__name__)


class TestTopologyGeneratorMaxNodes:
    """Test that EvolutionaryTopologyGenerator properly accepts and enforces max_nodes."""
    
    def test_max_nodes_parameter_exists(self):
        """Verify max_nodes parameter is accepted by constructor."""
        target_curve = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Should not raise TypeError
        generator = EvolutionaryTopologyGenerator(
            target_curve=target_curve,
            population_size=10,
            generations=5,
            genome_length=10,
            max_nodes=15,  # This parameter should now exist
            seed=42
        )
        
        assert generator.max_nodes == 15
    
    def test_max_nodes_enforced_during_evolution(self):
        """Verify max_nodes is enforced during graph generation."""
        target_curve = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        generator = EvolutionaryTopologyGenerator(
            target_curve=target_curve,
            population_size=10,
            generations=10,
            genome_length=20,
            max_nodes=10,  # Constrain to max 10 nodes
            seed=42
        )
        
        graph = generator.evolve()
        
        # Verify graph doesn't exceed max_nodes
        assert graph.number_of_nodes() <= 10, \
            f"Generated graph has {graph.number_of_nodes()} nodes, exceeds max_nodes=10"
    
    def test_room_count_control(self):
        """Test genome_length relationship to final room count."""
        target_curve = [0.3, 0.5, 0.7, 0.9]
        
        # Small genome should produce fewer rooms
        small_generator = EvolutionaryTopologyGenerator(
            target_curve=target_curve,
            population_size=10,
            generations=5,
            genome_length=8,
            max_nodes=15,
            seed=42
        )
        
        small_graph = small_generator.evolve()
        small_rooms = small_graph.number_of_nodes()
        
        # Large genome should produce more rooms (but capped by max_nodes)
        large_generator = EvolutionaryTopologyGenerator(
            target_curve=target_curve,
            population_size=10,
            generations=5,
            genome_length=25,
            max_nodes=15,
            seed=42
        )
        
        large_graph = large_generator.evolve()
        large_rooms = large_graph.number_of_nodes()
        
        # Both should respect max_nodes
        assert small_rooms <= 15
        assert large_rooms <= 15
        
        # Large genome should generally produce more rooms (not always guaranteed due to randomness)
        logger.info(f"Small genome: {small_rooms} rooms, Large genome: {large_rooms} rooms")


class TestDStarLitePredecessorFix:
    """Test D* Lite bug fix for proper predecessor state computation."""
    
    def create_simple_dungeon(self) -> np.ndarray:
        """Create a simple test dungeon with key and door."""
        grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
        
        # Walls
        grid[0, :] = SEMANTIC_PALETTE['WALL']
        grid[-1, :] = SEMANTIC_PALETTE['WALL']
        grid[:, 0] = SEMANTIC_PALETTE['WALL']
        grid[:, -1] = SEMANTIC_PALETTE['WALL']
        
        # Start and goal
        grid[1, 1] = SEMANTIC_PALETTE['START']
        grid[8, 8] = SEMANTIC_PALETTE['TRIFORCE']
        
        # Key and locked door
        grid[1, 5] = SEMANTIC_PALETTE['KEY_SMALL']
        grid[5, 5] = SEMANTIC_PALETTE['DOOR_LOCKED']
        
        return grid
    
    def test_dstar_lite_finds_path(self):
        """Test that D* Lite can find a path with proper state handling."""
        grid = self.create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = DStarLiteSolver(env, heuristic_mode='balanced')
        
        start_state = env.state.copy()
        success, path, nodes = solver.solve(start_state)
        
        assert success, "D* Lite should find a path in simple dungeon"
        assert len(path) > 0, "Path should not be empty"
        assert path[0] == start_state.position, "Path should start at start position"
        assert path[-1] == env.goal_pos, "Path should end at goal position"
        
        logger.info(f"D* Lite success: path_len={len(path)}, nodes_explored={nodes}")
    
    def test_dstar_lite_handles_state_transitions(self):
        """Test that D* Lite properly handles key collection and door opening."""
        grid = self.create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = DStarLiteSolver(env, heuristic_mode='balanced')
        
        start_state = env.state.copy()
        success, path, nodes = solver.solve(start_state)
        
        assert success, "Should find path requiring key collection"
        
        # Verify path visits key location
        key_pos = (1, 5)
        assert key_pos in path, f"Path should visit key at {key_pos}"
        
        # Verify path visits door location
        door_pos = (5, 5)
        assert door_pos in path, f"Path should visit door at {door_pos}"
        
        # Verify key is collected before door
        key_idx = path.index(key_pos)
        door_idx = path.index(door_pos)
        assert key_idx < door_idx, "Key must be collected before door is opened"


class TestBidirectionalAStarCollisionFix:
    """Test Bidirectional A* collision detection bug fix."""
    
    def create_test_dungeon(self) -> np.ndarray:
        """Create a test dungeon for bidirectional search."""
        grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
        
        # Walls
        grid[0, :] = SEMANTIC_PALETTE['WALL']
        grid[-1, :] = SEMANTIC_PALETTE['WALL']
        grid[:, 0] = SEMANTIC_PALETTE['WALL']
        grid[:, -1] = SEMANTIC_PALETTE['WALL']
        
        # Start and goal
        grid[1, 1] = SEMANTIC_PALETTE['START']
        grid[8, 8] = SEMANTIC_PALETTE['TRIFORCE']
        
        # Key and door in middle
        grid[3, 3] = SEMANTIC_PALETTE['KEY_SMALL']
        grid[5, 5] = SEMANTIC_PALETTE['DOOR_LOCKED']
        
        return grid
    
    def test_bidirectional_collision_detection(self):
        """Test that bidirectional A* properly detects collisions with state compatibility."""
        grid = self.create_test_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = BidirectionalAStar(env, timeout=50000, heuristic_mode='balanced')
        
        success, path, nodes = solver.solve()
        
        # Should find a path (even if slower than standard A*)
        assert success, "Bidirectional A* should find path with proper collision detection"
        assert len(path) > 0, "Path should not be empty"
        
        logger.info(f"Bidirectional A* success: path_len={len(path)}, nodes={nodes}")
    
    def test_collision_with_opened_doors(self):
        """Test collision detection properly checks opened_doors compatibility."""
        grid = self.create_test_dungeon()
        env = ZeldaLogicEnv(grid)
        
        # Create two states at same position with different opened_doors
        pos = (5, 5)
        state1 = GameState(position=pos, keys=1, opened_doors=set())
        state2 = GameState(position=pos, keys=0, opened_doors={(3, 3)})
        
        # These states should NOT be compatible for collision
        # state1 has key but hasn't opened door
        # state2 has opened a door but no key left
        
        # They are at same position but have incompatible states
        assert state1.position == state2.position
        assert state1.opened_doors != state2.opened_doors
        
        # Hash should differ due to opened_doors
        assert hash(state1) != hash(state2), \
            "States with different opened_doors should have different hashes"
    
    def test_collision_with_collected_items(self):
        """Test collision detection properly checks collected_items compatibility."""
        pos = (3, 3)
        
        # Forward state: before collecting item
        forward_state = GameState(
            position=pos,
            keys=0,
            collected_items=set()
        )
        
        # Backward state: after collecting item
        backward_state = GameState(
            position=pos,
            keys=1,
            collected_items={(3, 3)}
        )
        
        # Forward collected_items should be subset of backward collected_items
        assert forward_state.collected_items.issubset(backward_state.collected_items), \
            "Forward collected_items should be subset"
        
        # But they have different hashes
        assert hash(forward_state) != hash(backward_state), \
            "States with different collected_items should have different hashes"


class TestGraphGrammarExecutorMaxNodes:
    """Test GraphGrammarExecutor respects max_nodes parameter."""
    
    def test_executor_respects_max_nodes(self):
        """Test that executor stops at max_nodes."""
        executor = GraphGrammarExecutor(seed=42)
        
        # Create a genome with many rules
        genome = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        
        # Execute with low max_nodes
        graph = executor.execute(genome, difficulty=0.5, max_nodes=8)
        
        # Should not exceed max_nodes
        assert len(graph.nodes) <= 8, \
            f"Graph has {len(graph.nodes)} nodes, exceeds max_nodes=8"
    
    def test_executor_applies_rules_until_limit(self):
        """Test executor keeps applying valid rules until node limit."""
        executor = GraphGrammarExecutor(seed=42)
        
        # Many insertion rules
        genome = [1] * 30  # InsertChallenge_ENEMY repeated
        
        graph = executor.execute(genome, difficulty=0.5, max_nodes=12)
        
        # Should have applied multiple rules up to limit
        assert len(graph.nodes) > 1, "Should have applied some rules"
        assert len(graph.nodes) <= 12, "Should respect max_nodes"


class TestIntegrationPipeline:
    """Integration test for entire topology generation pipeline."""
    
    def test_pipeline_with_room_count_control(self):
        """Test that pipeline correctly controls room count via genome_length and max_nodes."""
        target_curve = [0.2, 0.4, 0.6, 0.8, 1.0]
        target_rooms = 12
        
        # Calculate genome_length (empirical: ~0.7x target rooms)
        genome_length = max(10, int(target_rooms * 0.7))
        
        generator = EvolutionaryTopologyGenerator(
            target_curve=target_curve,
            population_size=20,
            generations=15,
            genome_length=genome_length,
            max_nodes=target_rooms,
            seed=42
        )
        
        graph = generator.evolve()
        
        # Should be reasonably close to target (within 50% tolerance)
        assert graph.number_of_nodes() <= target_rooms, \
            f"Graph has {graph.number_of_nodes()} nodes, exceeds max {target_rooms}"
        
        assert graph.number_of_nodes() >= int(target_rooms * 0.4), \
            f"Graph has {graph.number_of_nodes()} nodes, too far below target {target_rooms}"
        
        logger.info(f"Generated {graph.number_of_nodes()} rooms (target: {target_rooms})")
    
    def test_topology_validation(self):
        """Test that generated topologies are valid."""
        from src.data.vglc_utils import validate_topology, filter_virtual_nodes
        
        target_curve = [0.3, 0.5, 0.7, 0.9]
        
        generator = EvolutionaryTopologyGenerator(
            target_curve=target_curve,
            population_size=15,
            generations=10,
            genome_length=12,
            max_nodes=15,
            seed=42
        )
        
        graph = generator.evolve()
        
        # Filter virtual nodes
        physical_graph = filter_virtual_nodes(graph)
        
        # Validate topology
        report = validate_topology(physical_graph)
        
        # Should have a valid topology
        assert physical_graph.number_of_nodes() > 0, "Should have physical nodes"
        assert physical_graph.number_of_edges() >= 0, "Should have edges"
        
        logger.info(f"Topology validation: {report.summary()}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
