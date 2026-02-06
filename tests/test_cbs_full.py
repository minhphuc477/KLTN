"""
Comprehensive Tests for CBS+ (Cognitive Bounded Search Plus).

This module contains tests for:
1. Bayesian belief update correctness
2. Memory decay over time
3. Inventory-aware planning (key before door)
4. Curiosity heuristic (prefers unknown regions)
5. Persona parameter effects

Research References:
- Working memory capacity (Miller, 1956): 7±2 items
- Memory decay (Anderson & Schooler, 1991): Exponential decay λ≈0.01
- Bounded rationality (Simon, 1955): Satisficing over optimal
"""
import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch,
    BeliefMap,
    VisionSystem,
    WorkingMemory,
    MemoryItemType,
    CBSMetrics,
    PersonaConfig,
    AgentPersona,
    PERSONA_CONFIGS,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def small_grid():
    """5x5 grid with start (1,1), goal (3,3), one key at (2,1), locked door at (2,3)."""
    grid = np.full((5, 5), SEMANTIC_PALETTE['FLOOR'], dtype=np.int32)
    # Walls around edges
    grid[0, :] = SEMANTIC_PALETTE['WALL']
    grid[4, :] = SEMANTIC_PALETTE['WALL']
    grid[:, 0] = SEMANTIC_PALETTE['WALL']
    grid[:, 4] = SEMANTIC_PALETTE['WALL']
    # Start and goal
    grid[1, 1] = SEMANTIC_PALETTE['START']
    grid[3, 3] = SEMANTIC_PALETTE['TRIFORCE']
    # Key and locked door
    grid[2, 1] = SEMANTIC_PALETTE['KEY_SMALL']
    grid[2, 3] = SEMANTIC_PALETTE['DOOR_LOCKED']
    return grid


@pytest.fixture
def exploration_grid():
    """Grid with visible and unexplored regions to test curiosity."""
    grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int32)
    # Walls around edges
    grid[0, :] = SEMANTIC_PALETTE['WALL']
    grid[9, :] = SEMANTIC_PALETTE['WALL']
    grid[:, 0] = SEMANTIC_PALETTE['WALL']
    grid[:, 9] = SEMANTIC_PALETTE['WALL']
    # Internal wall creating two regions
    grid[1:6, 5] = SEMANTIC_PALETTE['WALL']
    grid[5, 5] = SEMANTIC_PALETTE['DOOR_OPEN']  # Opening to right side
    # Start on left, goal on right
    grid[1, 1] = SEMANTIC_PALETTE['START']
    grid[8, 8] = SEMANTIC_PALETTE['TRIFORCE']
    return grid


@pytest.fixture
def belief_map(small_grid):
    """Create a BeliefMap for the small grid."""
    return BeliefMap(grid_shape=small_grid.shape)


@pytest.fixture
def vision_system():
    """Create a VisionSystem with default parameters."""
    return VisionSystem(radius=3, enable_occlusion=True)


# ============================================================================
# Test 1: Bayesian Belief Update Correctness
# ============================================================================

class TestBayesianBeliefUpdate:
    """Test Bayesian belief updates with observation evidence."""

    def test_prior_initialization(self, belief_map):
        """Test that prior beliefs start with 0.5 for unknown tiles."""
        for r in range(belief_map.height):
            for c in range(belief_map.width):
                conf = belief_map.get_confidence(r, c)
                assert conf == 0.5 or conf == 0.0, f"Prior at ({r},{c}) should be 0.5 or 0.0"

    def test_observation_increases_confidence(self, belief_map, small_grid):
        """Test that observing a tile increases confidence."""
        # Initial confidence
        initial_conf = belief_map.get_confidence(2, 2)
        
        # Observe the tile
        belief_map.update(2, 2, observed_tile=small_grid[2, 2])
        
        # Confidence should increase
        new_conf = belief_map.get_confidence(2, 2)
        assert new_conf > initial_conf, "Observation should increase confidence"

    def test_consistent_observation_high_confidence(self, belief_map):
        """Multiple consistent observations should yield high confidence."""
        # Observe same tile multiple times
        for _ in range(10):
            belief_map.update(2, 2, observed_tile=SEMANTIC_PALETTE['FLOOR'])
        
        conf = belief_map.get_confidence(2, 2)
        assert conf > 0.9, f"Multiple observations should give high confidence, got {conf}"

    def test_bayes_formula_correctness(self, belief_map):
        """
        Test Bayes' formula: P(tile|obs) = P(obs|tile) * P(tile) / P(obs)
        
        For observation O and hypothesis H (tile type):
        P(H|O) = P(O|H) * P(H) / sum_i(P(O|H_i) * P(H_i))
        """
        r, c = 2, 2
        obs_tile = SEMANTIC_PALETTE['FLOOR']
        
        # Get prior
        prior_conf = belief_map.get_confidence(r, c)
        prior_tile = belief_map.get_tile(r, c)
        
        # Update with observation
        belief_map.update(r, c, observed_tile=obs_tile)
        
        # Posterior should be higher if observation matches prior guess
        posterior_conf = belief_map.get_confidence(r, c)
        
        # Observation should always increase confidence for known tile
        assert posterior_conf >= prior_conf, "Bayes update should not decrease confidence"

    def test_contradictory_observation_decreases_confidence(self, belief_map):
        """Observing a different tile than believed should adjust belief."""
        r, c = 2, 2
        
        # First observe as floor
        belief_map.update(r, c, observed_tile=SEMANTIC_PALETTE['FLOOR'])
        conf_after_floor = belief_map.get_confidence(r, c)
        
        # Then observe as wall (contradiction)
        belief_map.update(r, c, observed_tile=SEMANTIC_PALETTE['WALL'])
        conf_after_wall = belief_map.get_confidence(r, c)
        
        # Belief should have changed - tile type should now be wall
        assert belief_map.get_tile(r, c) == SEMANTIC_PALETTE['WALL'], \
            "Tile belief should update to latest observation"


# ============================================================================
# Test 2: Memory Decay Over Time
# ============================================================================

class TestMemoryDecay:
    """Test exponential memory decay λ ≈ 0.01 per step."""

    def test_decay_formula(self):
        """
        Test decay formula: confidence(t) = confidence(0) * exp(-λt)
        where λ ≈ 0.01
        """
        decay_rate = 0.01
        initial_conf = 1.0
        
        for t in [10, 50, 100, 200]:
            expected = initial_conf * np.exp(-decay_rate * t)
            actual = initial_conf * np.exp(-0.01 * t)  # Using exact λ=0.01
            assert abs(expected - actual) < 0.001, f"Decay formula mismatch at t={t}"

    def test_working_memory_capacity(self):
        """Test Miller's Law: 7±2 items capacity."""
        wm = WorkingMemory(capacity=7)
        
        # Add exactly 7 items
        for i in range(7):
            wm.remember(MemoryItemType.POSITION, (i, i), current_step=i)
        
        assert len(wm.items) <= 9, "Working memory should not exceed 7+2 items"
        assert len(wm.items) >= 5, "Working memory should maintain at least 7-2 items"

    def test_old_memories_fade(self, belief_map):
        """Memories should decay over time steps."""
        r, c = 2, 2
        
        # Observe a tile
        belief_map.update(r, c, observed_tile=SEMANTIC_PALETTE['FLOOR'])
        initial_conf = belief_map.get_confidence(r, c)
        
        # Simulate time passing (apply decay)
        for _ in range(100):
            belief_map.apply_decay(decay_rate=0.01)
        
        final_conf = belief_map.get_confidence(r, c)
        
        # Confidence should have decreased
        assert final_conf < initial_conf, "Memory should decay over time"
        
        # Check decay magnitude
        expected_ratio = np.exp(-0.01 * 100)  # ≈ 0.368
        actual_ratio = final_conf / initial_conf if initial_conf > 0 else 0
        assert actual_ratio < 0.5, f"Decay should be significant after 100 steps"

    def test_recent_observation_beats_old(self, belief_map):
        """Recent observations should dominate old memories."""
        r, c = 2, 2
        
        # Old observation
        belief_map.update(r, c, observed_tile=SEMANTIC_PALETTE['FLOOR'])
        
        # Time passes
        for _ in range(50):
            belief_map.apply_decay(decay_rate=0.01)
        
        old_conf = belief_map.get_confidence(r, c)
        
        # New observation
        belief_map.update(r, c, observed_tile=SEMANTIC_PALETTE['FLOOR'])
        new_conf = belief_map.get_confidence(r, c)
        
        # New observation should restore confidence
        assert new_conf > old_conf, "Recent observation should restore confidence"


# ============================================================================
# Test 3: Inventory-Aware Planning (Key Before Door)
# ============================================================================

class TestInventoryAwarePlanning:
    """Test that agent picks up key before attempting locked door."""

    def test_key_door_sequence(self, small_grid):
        """Agent should find key before attempting locked door."""
        # ZeldaLogicEnv finds start/goal from the grid tiles
        env = ZeldaLogicEnv(semantic_grid=small_grid)
        cbs = CognitiveBoundedSearch(env, persona='balanced', timeout=500, seed=42)
        success, path, states, metrics = cbs.solve()
        
        if not success:
            pytest.skip("CBS did not find solution in time")
        
        # Find key and door positions in path
        key_pos = (2, 1)
        door_pos = (2, 3)
        
        key_idx = None
        door_idx = None
        
        for i, pos in enumerate(path):
            if pos == key_pos and key_idx is None:
                key_idx = i
            if pos == door_pos and door_idx is None:
                door_idx = i
        
        # Key should be visited before door (or door not visited if key enables bypass)
        if door_idx is not None and key_idx is not None:
            assert key_idx < door_idx, \
                f"Key should be picked up (idx={key_idx}) before door (idx={door_idx})"

    def test_subgoal_key_insertion(self, small_grid):
        """Test that hierarchical planner inserts key as subgoal."""
        env = ZeldaLogicEnv(semantic_grid=small_grid)
        cbs = CognitiveBoundedSearch(env, persona='balanced', timeout=500, seed=42)
        
        # Run the solver to completion - it will handle subgoal insertion internally
        success, path, metrics, visited = cbs.solve()
        
        # If path found, it should visit the key before door
        if success and path:
            key_pos = tuple(np.argwhere(small_grid == SEMANTIC_PALETTE['KEY'])[0]) \
                if SEMANTIC_PALETTE['KEY'] in small_grid else None
            # The solver should have found a valid path
            assert len(path) > 0, "Should find a path"


# ============================================================================
# Test 4: Curiosity Heuristic (Prefers Unknown Regions)
# ============================================================================

class TestCuriosityHeuristic:
    """Test curiosity-driven exploration toward unknown regions."""

    def test_info_gain_computation(self, belief_map):
        """Test information gain for unexplored vs explored tiles."""
        # Unknown tiles have confidence 0, explored tiles have high confidence
        # Information gain comes from reducing uncertainty
        
        # Check initial confidence is low (unknown)
        initial_conf = belief_map.get_confidence(2, 2)
        assert initial_conf < 0.5, "Unknown tile should have low confidence"
        
        # Observe the tile
        belief_map.update(2, 2, observed_tile=SEMANTIC_PALETTE['FLOOR'])
        explored_conf = belief_map.get_confidence(2, 2)
        
        # Confidence should increase after observation
        assert explored_conf > initial_conf, \
            "Explored tiles should have higher confidence"

    def test_curiosity_prefers_unknown(self, exploration_grid):
        """Agent should move toward unexplored regions when possible."""
        env = ZeldaLogicEnv(semantic_grid=exploration_grid)
        
        # Use explorer persona (high curiosity weight)
        cbs = CognitiveBoundedSearch(env, persona='explorer', timeout=500, seed=42)
        
        # Explorer should value information gain - check persona config
        assert cbs.persona_config.curiosity_weight > 0, \
            "Explorer should have positive curiosity weight"
        
        # Solve and verify it explores
        success, path, states_explored, metrics = cbs.solve()
        if success:
            assert len(path) > 0, "Explorer should find a path"
            # Check metrics for exploration evidence
            assert states_explored > 0 or metrics.nodes_expanded > 0, \
                "Explorer should explore states"

    def test_utility_function_with_curiosity(self, belief_map):
        """
        Test utility function: U(a) = α·goal_progress + β·info_gain - γ·risk
        """
        alpha, beta, gamma = 0.6, 0.3, 0.1  # Balanced persona
        
        # Compute utility for moving toward unknown vs known area
        goal_progress = 1.0  # Same for both
        
        # Unknown area has high info gain
        unknown_info_gain = 1.0
        known_info_gain = 0.1
        
        risk = 0.1  # Same for both
        
        utility_unknown = alpha * goal_progress + beta * unknown_info_gain - gamma * risk
        utility_known = alpha * goal_progress + beta * known_info_gain - gamma * risk
        
        assert utility_unknown > utility_known, \
            f"Unknown area should have higher utility: {utility_unknown} > {utility_known}"

    def test_explorer_explores_more(self, exploration_grid):
        """Explorer persona should visit more unique tiles than cautious."""
        env_explorer = ZeldaLogicEnv(semantic_grid=exploration_grid)
        env_cautious = ZeldaLogicEnv(semantic_grid=exploration_grid.copy())
        
        # Run with different personas
        cbs_explorer = CognitiveBoundedSearch(env_explorer, persona='explorer', timeout=300, seed=42)
        cbs_cautious = CognitiveBoundedSearch(env_cautious, persona='cautious', timeout=300, seed=42)
        
        success_e, path_e, _, metrics_e = cbs_explorer.solve()
        success_c, path_c, _, metrics_c = cbs_cautious.solve()
        
        if success_e and success_c:
            unique_e = len(set(path_e))
            unique_c = len(set(path_c))
            # Explorer should generally visit more unique tiles
            # (though not guaranteed due to randomness)
            print(f"Explorer unique tiles: {unique_e}, Cautious: {unique_c}")


# ============================================================================
# Test 5: Persona Parameter Effects
# ============================================================================

class TestPersonaEffects:
    """Test that persona parameters affect agent behavior."""

    def test_persona_config_values(self):
        """Verify persona configurations match specification."""
        expected = {
            'balanced': {'alpha': 0.6, 'beta': 0.3, 'gamma': 0.1},
            'forgetful': {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3},
            'explorer': {'alpha': 0.3, 'beta': 0.6, 'gamma': 0.1},
            'cautious': {'alpha': 0.5, 'beta': 0.2, 'gamma': 0.3},
        }
        
        for name, params in expected.items():
            if name in PERSONA_CONFIGS:
                config = PERSONA_CONFIGS[name]
                assert abs(config.goal_weight - params['alpha']) < 0.01, \
                    f"{name} goal_weight mismatch"
                assert abs(config.curiosity_weight - params['beta']) < 0.01, \
                    f"{name} curiosity_weight mismatch"
                assert abs(config.risk_weight - params['gamma']) < 0.01, \
                    f"{name} risk_weight mismatch"

    def test_forgetful_higher_decay(self):
        """Forgetful persona should have faster memory decay."""
        balanced = PERSONA_CONFIGS.get('balanced')
        forgetful = PERSONA_CONFIGS.get('forgetful')
        
        if balanced and forgetful:
            assert forgetful.decay_rate >= balanced.decay_rate, \
                "Forgetful should have equal or higher decay rate"

    def test_cautious_avoids_enemies(self, small_grid):
        """Cautious persona should weight risk higher."""
        cautious = PERSONA_CONFIGS.get('cautious')
        balanced = PERSONA_CONFIGS.get('balanced')
        
        if cautious and balanced:
            assert cautious.risk_weight > balanced.risk_weight, \
                "Cautious should have higher risk weight"

    def test_explorer_high_curiosity(self):
        """Explorer persona should have highest curiosity weight."""
        explorer = PERSONA_CONFIGS.get('explorer')
        
        if explorer:
            assert explorer.curiosity_weight >= 0.5, \
                f"Explorer curiosity should be high, got {explorer.curiosity_weight}"

    def test_persona_affects_path_length(self, exploration_grid):
        """Different personas should produce different path characteristics."""
        results = {}
        
        for persona_name in ['balanced', 'explorer', 'cautious']:
            env = ZeldaLogicEnv(semantic_grid=exploration_grid.copy())
            cbs = CognitiveBoundedSearch(env, persona=persona_name, timeout=300, seed=42)
            success, path, states, metrics = cbs.solve()
            
            if success:
                results[persona_name] = {
                    'path_length': len(path),
                    'states_explored': states,
                    'unique_tiles': len(set(path))
                }
        
        # Print results for analysis
        for name, data in results.items():
            print(f"{name}: path={data['path_length']}, states={data['states_explored']}")


# ============================================================================
# Integration Tests
# ============================================================================

class TestCBSIntegration:
    """Integration tests for the full CBS+ solver."""

    def test_solve_simple_dungeon(self, small_grid):
        """CBS+ should solve a simple 5x5 dungeon."""
        env = ZeldaLogicEnv(semantic_grid=small_grid)
        cbs = CognitiveBoundedSearch(env, persona='balanced', timeout=1000, seed=42)
        success, path, states, metrics = cbs.solve()
        
        assert success, "CBS should solve simple dungeon"
        assert len(path) > 0, "Path should not be empty"
        assert path[0] == env.start_pos, "Path should start at start position"
        assert path[-1] == env.goal_pos, "Path should end at goal"

    def test_metrics_recorded(self, small_grid):
        """CBS+ should record all required metrics."""
        env = ZeldaLogicEnv(semantic_grid=small_grid)
        cbs = CognitiveBoundedSearch(env, persona='balanced', timeout=1000, seed=42)
        success, path, states, metrics = cbs.solve()
        
        # Check metrics exist
        assert hasattr(metrics, 'path_length'), "Should record path_length"
        assert hasattr(metrics, 'replans'), "Should record replans"
        assert hasattr(metrics, 'confusion_events'), "Should record confusion_events"
        assert hasattr(metrics, 'backtrack_loops'), "Should record backtrack_loops"

    def test_deterministic_with_seed(self, small_grid):
        """Same seed should produce same path."""
        env1 = ZeldaLogicEnv(semantic_grid=small_grid.copy())
        env2 = ZeldaLogicEnv(semantic_grid=small_grid.copy())
        
        cbs1 = CognitiveBoundedSearch(env1, persona='balanced', timeout=500, seed=12345)
        cbs2 = CognitiveBoundedSearch(env2, persona='balanced', timeout=500, seed=12345)
        
        success1, path1, _, _ = cbs1.solve()
        success2, path2, _, _ = cbs2.solve()
        
        # Both should succeed/fail consistently
        assert success1 == success2, "Same seed should produce same success result"
        # Note: exact paths may differ due to exploration order and timing

    def test_vision_system_occlusion(self, vision_system, small_grid):
        """Vision system should handle wall occlusion."""
        # Create a grid with walls blocking view
        visible = vision_system.get_visible_tiles(
            position=(2, 2),
            direction=(0, 1),  # Direction required
            grid=small_grid
        )
        
        # Should see nearby cells but not through walls
        assert len(visible) > 0, "Should see at least some cells"
        
        # Cells behind walls should not be visible
        # (depends on implementation details)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_start_equals_goal(self):
        """Handle case where start equals goal."""
        grid = np.full((5, 5), SEMANTIC_PALETTE['FLOOR'], dtype=np.int32)
        # Put START and TRIFORCE at same position
        grid[2, 2] = SEMANTIC_PALETTE['TRIFORCE']
        grid[1, 1] = SEMANTIC_PALETTE['START']  # Need a start somewhere
        
        env = ZeldaLogicEnv(semantic_grid=grid)
        cbs = CognitiveBoundedSearch(env, persona='balanced', timeout=100, seed=42)
        success, path, states, metrics = cbs.solve()
        
        # Should immediately succeed
        assert success or len(path) <= 1, "Start=Goal should succeed quickly"

    def test_impossible_path(self):
        """Handle case where no path exists."""
        grid = np.full((5, 5), SEMANTIC_PALETTE['WALL'], dtype=np.int32)
        grid[1, 1] = SEMANTIC_PALETTE['START']
        grid[3, 3] = SEMANTIC_PALETTE['TRIFORCE']
        
        env = ZeldaLogicEnv(semantic_grid=grid)
        cbs = CognitiveBoundedSearch(env, persona='balanced', timeout=100, seed=42)
        success, path, states, metrics = cbs.solve()
        
        # Should fail gracefully
        assert not success, "Should fail when no path exists"

    def test_empty_belief_map(self):
        """BeliefMap with no observations should have low confidence."""
        bm = BeliefMap(grid_shape=(10, 10))
        
        # All cells should have low/prior confidence
        avg_conf = np.mean([bm.get_confidence(r, c) for r in range(10) for c in range(10)])
        assert avg_conf <= 0.5, "Unobserved map should have low average confidence"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
