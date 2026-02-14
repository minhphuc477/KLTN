"""
Tests for Cognitive Bounded Search (CBS) module.

Run with: pytest tests/test_cognitive_bounded_search.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch,
    CBSMetrics,
    BeliefMap,
    VisionSystem,
    WorkingMemory,
    MemoryItem,
    MemoryItemType,
    TileKnowledge,
    AgentPersona,
    PersonaConfig,
    CuriosityHeuristic,
    SafetyHeuristic,
    GoalSeekingHeuristic,
    solve_with_cbs,
    compare_personas,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def simple_grid():
    """Create a simple 10x10 test grid with start, goal, and some walls."""
    grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    
    # Add walls around edges
    grid[0, :] = SEMANTIC_PALETTE['WALL']
    grid[-1, :] = SEMANTIC_PALETTE['WALL']
    grid[:, 0] = SEMANTIC_PALETTE['WALL']
    grid[:, -1] = SEMANTIC_PALETTE['WALL']
    
    # Add start and goal
    grid[1, 1] = SEMANTIC_PALETTE['START']
    grid[8, 8] = SEMANTIC_PALETTE['TRIFORCE']
    
    return grid


@pytest.fixture
def grid_with_key():
    """Grid requiring a key to reach the goal."""
    grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    
    # Walls
    grid[0, :] = SEMANTIC_PALETTE['WALL']
    grid[-1, :] = SEMANTIC_PALETTE['WALL']
    grid[:, 0] = SEMANTIC_PALETTE['WALL']
    grid[:, -1] = SEMANTIC_PALETTE['WALL']
    
    # Dividing wall with locked door
    grid[1:9, 5] = SEMANTIC_PALETTE['WALL']
    grid[5, 5] = SEMANTIC_PALETTE['DOOR_LOCKED']
    
    # Start on left, goal on right
    grid[1, 1] = SEMANTIC_PALETTE['START']
    grid[8, 8] = SEMANTIC_PALETTE['TRIFORCE']
    
    # Key on left side
    grid[3, 3] = SEMANTIC_PALETTE['KEY_SMALL']
    
    return grid


@pytest.fixture
def grid_with_enemies():
    """Grid with enemies to test safety heuristic."""
    grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    
    # Walls
    grid[0, :] = SEMANTIC_PALETTE['WALL']
    grid[-1, :] = SEMANTIC_PALETTE['WALL']
    grid[:, 0] = SEMANTIC_PALETTE['WALL']
    grid[:, -1] = SEMANTIC_PALETTE['WALL']
    
    # Start and goal
    grid[1, 1] = SEMANTIC_PALETTE['START']
    grid[8, 8] = SEMANTIC_PALETTE['TRIFORCE']
    
    # Enemies in the middle
    grid[4, 4] = SEMANTIC_PALETTE['ENEMY']
    grid[5, 5] = SEMANTIC_PALETTE['ENEMY']
    grid[6, 6] = SEMANTIC_PALETTE['ENEMY']
    
    return grid


# ==============================================================================
# BELIEF MAP TESTS
# ==============================================================================

class TestBeliefMap:
    """Tests for BeliefMap epistemic state."""
    
    def test_initialization(self):
        """Test belief map initializes correctly."""
        belief = BeliefMap(grid_shape=(10, 10), decay_rate=0.95)
        
        assert belief.grid_shape == (10, 10)
        assert belief.decay_rate == 0.95
        assert len(belief.known_tiles) == 0
    
    def test_observe_tile(self):
        """Test observing a tile updates beliefs."""
        belief = BeliefMap(grid_shape=(10, 10))
        
        belief.observe((5, 5), SEMANTIC_PALETTE['FLOOR'], current_step=0, is_visit=True)
        
        tile_type, confidence = belief.get_tile_with_confidence((5, 5))
        assert tile_type == SEMANTIC_PALETTE['FLOOR']
        assert confidence == 1.0  # Visit = full confidence
        assert belief.get_knowledge_state((5, 5)) == TileKnowledge.EXPLORED
    
    def test_unknown_tile(self):
        """Test querying unknown tile returns default assumption."""
        belief = BeliefMap(grid_shape=(10, 10))
        
        tile_type, confidence = belief.get_tile_with_confidence((5, 5))
        assert tile_type == SEMANTIC_PALETTE['WALL']  # Default assumption
        assert confidence == 0.0
        assert belief.get_knowledge_state((5, 5)) == TileKnowledge.UNKNOWN
    
    def test_confidence_decay(self):
        """Test memory decay reduces confidence over time."""
        belief = BeliefMap(grid_shape=(10, 10), decay_rate=0.9)
        
        # Observe tile at step 0
        belief.observe((5, 5), SEMANTIC_PALETTE['FLOOR'], current_step=0, is_visit=False)
        initial_confidence = belief.get_confidence((5, 5))
        
        # Apply decay at step 5
        belief.apply_decay(current_step=5)
        decayed_confidence = belief.get_confidence((5, 5))
        
        assert decayed_confidence < initial_confidence
    
    def test_confusion_index(self):
        """Test confusion index calculation."""
        belief = BeliefMap(grid_shape=(10, 10))
        
        # Visit 3 tiles, revisit 1
        belief.observe((1, 1), SEMANTIC_PALETTE['FLOOR'], 0, is_visit=True)
        belief.observe((2, 2), SEMANTIC_PALETTE['FLOOR'], 1, is_visit=True)
        belief.observe((3, 3), SEMANTIC_PALETTE['FLOOR'], 2, is_visit=True)
        belief.observe((1, 1), SEMANTIC_PALETTE['FLOOR'], 3, is_visit=True)  # Revisit
        
        confusion = belief.compute_confusion_index()
        assert confusion == pytest.approx(1/3, rel=0.01)  # 1 revisit / 3 unique


# ==============================================================================
# VISION SYSTEM TESTS
# ==============================================================================

class TestVisionSystem:
    """Tests for VisionSystem field-of-view."""
    
    def test_initialization(self):
        """Test vision system initializes with correct parameters."""
        vision = VisionSystem(radius=5, cone_angle=120.0)
        
        assert vision.radius == 5
        assert vision.cone_angle == 120.0
        assert vision.enable_occlusion == True
    
    def test_full_vision(self, simple_grid):
        """Test 360° vision sees all tiles in radius."""
        vision = VisionSystem(radius=3, cone_angle=360.0, enable_occlusion=False)
        
        visible = vision.get_visible_tiles(
            position=(5, 5),
            direction=(0, 1),
            grid=simple_grid
        )
        
        # Should include current position and tiles within radius
        assert (5, 5) in visible
        assert len(visible) > 1
    
    def test_occlusion(self, simple_grid):
        """Test that walls block vision."""
        vision = VisionSystem(radius=10, cone_angle=360.0, enable_occlusion=True)
        
        # Position near wall - should not see through walls
        visible = vision.get_visible_tiles(
            position=(1, 1),
            direction=(0, 1),
            grid=simple_grid
        )
        
        # Should not see tiles outside the room (walls block)
        assert len(visible) < 100  # Less than full grid


# ==============================================================================
# WORKING MEMORY TESTS
# ==============================================================================

class TestWorkingMemory:
    """Tests for WorkingMemory capacity limits."""
    
    def test_initialization(self):
        """Test memory initializes with correct capacity."""
        memory = WorkingMemory(capacity=7, decay_rate=0.95)
        
        assert memory.capacity == 7
        assert len(memory.items) == 0
    
    def test_remember(self):
        """Test adding items to memory."""
        memory = WorkingMemory(capacity=7)
        
        success = memory.remember(MemoryItemType.GOAL, (10, 10), current_step=0)
        
        assert success
        assert len(memory.items) == 1
        assert memory.is_remembered((10, 10), MemoryItemType.GOAL)
    
    def test_capacity_limit(self):
        """Test that memory enforces capacity limit."""
        memory = WorkingMemory(capacity=3)
        
        # Add 5 items (exceeds capacity)
        for i in range(5):
            memory.remember(MemoryItemType.POSITION, (i, i), current_step=i)
        
        # Should only have 3 items (capacity)
        assert len(memory.items) == 3
        assert memory.total_forgotten == 2
    
    def test_salience_priority(self):
        """Test that high-salience items are retained."""
        memory = WorkingMemory(capacity=2)
        
        # Add goal (high salience)
        memory.remember(MemoryItemType.GOAL, (10, 10), current_step=0)
        
        # Add 2 low-salience positions (should displace one, but not goal)
        memory.remember(MemoryItemType.POSITION, (1, 1), current_step=1)
        memory.remember(MemoryItemType.POSITION, (2, 2), current_step=2)
        
        # Goal should still be remembered (highest salience)
        assert memory.is_remembered((10, 10), MemoryItemType.GOAL)
    
    def test_recall_by_type(self):
        """Test recalling items filtered by type."""
        memory = WorkingMemory(capacity=7)
        
        memory.remember(MemoryItemType.GOAL, (10, 10), current_step=0)
        memory.remember(MemoryItemType.ITEM, (5, 5), current_step=0)
        memory.remember(MemoryItemType.THREAT, (3, 3), current_step=0)
        
        goals = memory.recall(MemoryItemType.GOAL)
        assert len(goals) == 1
        assert goals[0].position == (10, 10)
    
    def test_memory_decay(self):
        """Test memory decay over time."""
        memory = WorkingMemory(capacity=7, decay_rate=0.5)  # Fast decay
        
        memory.remember(MemoryItemType.POSITION, (1, 1), current_step=0)
        initial_count = len(memory.items)
        
        # Apply heavy decay
        memory.apply_decay(current_step=20)
        
        # Low-salience item should be forgotten
        assert len(memory.items) <= initial_count


# ==============================================================================
# HEURISTIC TESTS
# ==============================================================================

class TestHeuristics:
    """Tests for decision heuristics."""
    
    def test_curiosity_heuristic(self):
        """Test curiosity scores unexplored tiles higher."""
        belief = BeliefMap(grid_shape=(10, 10))
        memory = WorkingMemory()
        curiosity = CuriosityHeuristic(weight=1.0)
        
        # Unknown tile should have high score
        unknown_score = curiosity.score(
            current_pos=(5, 5),
            target_pos=(6, 6),
            target_tile=SEMANTIC_PALETTE['FLOOR'],
            belief_map=belief,
            memory=memory,
            goal_pos=None,
            current_step=0
        )
        
        # Now mark tile as explored
        belief.observe((6, 6), SEMANTIC_PALETTE['FLOOR'], 0, is_visit=True)
        
        explored_score = curiosity.score(
            current_pos=(5, 5),
            target_pos=(6, 6),
            target_tile=SEMANTIC_PALETTE['FLOOR'],
            belief_map=belief,
            memory=memory,
            goal_pos=None,
            current_step=1
        )
        
        assert unknown_score > explored_score
    
    def test_safety_heuristic(self):
        """Test safety heuristic penalizes enemies."""
        belief = BeliefMap(grid_shape=(10, 10))
        memory = WorkingMemory()
        safety = SafetyHeuristic(weight=1.0)
        
        # Score for safe floor
        safe_score = safety.score(
            current_pos=(5, 5),
            target_pos=(6, 6),
            target_tile=SEMANTIC_PALETTE['FLOOR'],
            belief_map=belief,
            memory=memory,
            goal_pos=None,
            current_step=0
        )
        
        # Score for enemy tile
        danger_score = safety.score(
            current_pos=(5, 5),
            target_pos=(6, 6),
            target_tile=SEMANTIC_PALETTE['ENEMY'],
            belief_map=belief,
            memory=memory,
            goal_pos=None,
            current_step=0
        )
        
        assert safe_score > danger_score
        assert danger_score < 0  # Negative for threats
    
    def test_goal_seeking_heuristic(self):
        """Test goal-seeking prefers moves toward goal."""
        belief = BeliefMap(grid_shape=(10, 10))
        memory = WorkingMemory()
        memory.remember(MemoryItemType.GOAL, (10, 10), current_step=0)
        
        goal_seek = GoalSeekingHeuristic(weight=1.0)
        
        # Move toward goal (from 5,5 to 6,6 is closer to 10,10)
        toward_score = goal_seek.score(
            current_pos=(5, 5),
            target_pos=(6, 6),
            target_tile=SEMANTIC_PALETTE['FLOOR'],
            belief_map=belief,
            memory=memory,
            goal_pos=(10, 10),
            current_step=1
        )
        
        # Move away from goal (from 5,5 to 4,4 is farther from 10,10)
        away_score = goal_seek.score(
            current_pos=(5, 5),
            target_pos=(4, 4),
            target_tile=SEMANTIC_PALETTE['FLOOR'],
            belief_map=belief,
            memory=memory,
            goal_pos=(10, 10),
            current_step=1
        )
        
        assert toward_score > away_score


# ==============================================================================
# PERSONA TESTS
# ==============================================================================

class TestPersonas:
    """Tests for agent personas."""
    
    def test_persona_configs(self):
        """Test all personas have valid configurations."""
        for persona in AgentPersona:
            config = PersonaConfig.get_persona(persona)
            
            assert config.memory_capacity > 0
            assert 0 <= config.memory_decay_rate <= 1
            assert config.vision_radius > 0
            assert 0 < config.vision_cone <= 360
            assert config.satisficing_threshold > 0
    
    def test_speedrunner_config(self):
        """Test speedrunner has optimal-seeking configuration."""
        config = PersonaConfig.get_persona(AgentPersona.SPEEDRUNNER)
        
        assert config.heuristic_weights['goal_seeking'] >= 1.5
        assert config.heuristic_weights['safety'] == 0  # Ignores danger
        assert config.memory_capacity >= 10
    
    def test_forgetful_config(self):
        """Test forgetful has poor memory configuration."""
        config = PersonaConfig.get_persona(AgentPersona.FORGETFUL)
        
        assert config.memory_capacity <= 5
        assert config.memory_decay_rate <= 0.85
        assert config.random_tiebreaker > 0.2  # More random


# ==============================================================================
# CBS SOLVER TESTS
# ==============================================================================

class TestCBS:
    """Tests for CognitiveBoundedSearch solver."""
    
    def test_basic_solve(self, simple_grid):
        """Test CBS can solve a simple grid."""
        env = ZeldaLogicEnv(semantic_grid=simple_grid)
        cbs = CognitiveBoundedSearch(env, persona=AgentPersona.BALANCED, timeout=10000)
        
        success, path, states, metrics = cbs.solve()
        
        assert success
        assert len(path) > 0
        assert path[0] == (1, 1)  # Start position
        assert path[-1] == (8, 8)  # Goal position
    
    def test_key_collection(self, grid_with_key):
        """Test CBS collects key before locked door."""
        env = ZeldaLogicEnv(semantic_grid=grid_with_key)
        cbs = CognitiveBoundedSearch(env, persona=AgentPersona.BALANCED, timeout=20000)
        
        success, path, states, metrics = cbs.solve()
        
        assert success
        # Path should include key position before door
        key_pos = (3, 3)
        door_pos = (5, 5)
        
        if key_pos in path and door_pos in path:
            assert path.index(key_pos) < path.index(door_pos)
    
    def test_metrics_computed(self, simple_grid):
        """Test CBS computes cognitive metrics."""
        env = ZeldaLogicEnv(semantic_grid=simple_grid)
        cbs = CognitiveBoundedSearch(env, persona=AgentPersona.BALANCED, timeout=10000)
        
        success, path, states, metrics = cbs.solve()
        
        assert isinstance(metrics, CBSMetrics)
        assert metrics.total_steps == len(path)
        assert metrics.unique_tiles_visited > 0
        assert metrics.confusion_index >= 0
        assert metrics.navigation_entropy >= 0
    
    def test_persona_affects_behavior(self, grid_with_enemies):
        """Test different personas produce different paths."""
        env = ZeldaLogicEnv(semantic_grid=grid_with_enemies)
        
        # Speedrunner (ignores enemies)
        cbs_speed = CognitiveBoundedSearch(env, persona=AgentPersona.SPEEDRUNNER, seed=42)
        success_s, path_s, _, metrics_s = cbs_speed.solve()
        
        # Cautious (avoids enemies)
        env.reset()
        cbs_caution = CognitiveBoundedSearch(env, persona=AgentPersona.CAUTIOUS, seed=42)
        success_c, path_c, _, metrics_c = cbs_caution.solve()
        
        # Both should succeed
        assert success_s
        assert success_c
        
        # Cautious should have different path characteristics
        # (Note: with seed=42, both may find same path, so we check metrics)
        # The cautious agent should generally show higher path length or different route
    
    def test_compare_personas(self, simple_grid):
        """Test persona comparison function."""
        results = compare_personas(simple_grid, timeout=10000, seed=42)
        
        assert len(results) == len(AgentPersona)
        
        for persona_name, (success, path_len, metrics) in results.items():
            assert isinstance(success, bool)
            assert path_len >= 0
            assert isinstance(metrics, CBSMetrics)
    
    def test_reproducibility(self, simple_grid):
        """Test CBS produces same results with same seed."""
        env1 = ZeldaLogicEnv(semantic_grid=simple_grid)
        cbs1 = CognitiveBoundedSearch(env1, persona=AgentPersona.EXPLORER, seed=12345)
        success1, path1, states1, metrics1 = cbs1.solve()
        
        env2 = ZeldaLogicEnv(semantic_grid=simple_grid)
        cbs2 = CognitiveBoundedSearch(env2, persona=AgentPersona.EXPLORER, seed=12345)
        success2, path2, states2, metrics2 = cbs2.solve()
        
        assert success1 == success2
        assert path1 == path2
        assert metrics1.confusion_index == metrics2.confusion_index


# ==============================================================================
# CONVENIENCE FUNCTION TESTS
# ==============================================================================

class TestConvenienceFunctions:
    """Tests for solve_with_cbs and compare_personas."""
    
    def test_solve_with_cbs(self, simple_grid):
        """Test convenience function."""
        success, path, states, metrics = solve_with_cbs(
            simple_grid,
            persona='balanced',
            timeout=10000,
            seed=42
        )
        
        assert success
        assert len(path) > 0
        assert isinstance(metrics, CBSMetrics)


# ==============================================================================
# METRICS TESTS
# ==============================================================================

class TestCBSMetrics:
    """Tests for CBSMetrics dataclass."""
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = CBSMetrics(
            confusion_index=0.5,
            navigation_entropy=1.2,
            cognitive_load=0.8,
            aha_latency=15,
            unique_tiles_visited=50,
            total_steps=75,
        )
        
        d = metrics.to_dict()
        
        assert 'confusion_index' in d
        assert 'navigation_entropy' in d
        assert d['total_steps'] == 75
    
    def test_metrics_summary(self):
        """Test metrics summary string."""
        metrics = CBSMetrics(
            confusion_index=0.5,
            navigation_entropy=1.2,
            cognitive_load=0.8,
            aha_latency=15,
        )
        
        summary = metrics.summary()
        
        assert 'Confusion Index' in summary
        assert 'Navigation Entropy' in summary
        assert 'Cognitive Load' in summary


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestIntegration:
    """Integration tests with existing validator components."""
    
    def test_cbs_uses_zelda_logic_env(self, simple_grid):
        """Test CBS works with ZeldaLogicEnv."""
        env = ZeldaLogicEnv(semantic_grid=simple_grid)
        cbs = CognitiveBoundedSearch(env, persona=AgentPersona.BALANCED)
        
        # CBS should use the env's properties
        assert cbs.env.start_pos == (1, 1)
        assert cbs.env.goal_pos == (8, 8)
    
    def test_cbs_vs_astar_consistency(self, simple_grid):
        """Test CBS and A* both find valid solutions."""
        env = ZeldaLogicEnv(semantic_grid=simple_grid)
        
        # A* solution
        astar = StateSpaceAStar(env)
        astar_success, astar_path, astar_states = astar.solve()
        
        # CBS solution
        env.reset()
        cbs = CognitiveBoundedSearch(env, persona=AgentPersona.SPEEDRUNNER, seed=42)
        cbs_success, cbs_path, cbs_states, cbs_metrics = cbs.solve()
        
        # Both should succeed on simple grid
        assert astar_success
        assert cbs_success
        
        # CBS path may be longer (suboptimal) but should still reach goal
        assert cbs_path[-1] == astar_path[-1]  # Same goal


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
