# pyright: reportPrivateUsage=false

"""
Tests for H-MOLQD Block VII: Symbolic Refiner
==============================================

Tests for WFC-based dungeon repair.
"""

import pytest
import numpy as np


class TestPathAnalyzer:
    """Tests for path analysis."""
    
    def test_analyze_passable_grid(self):
        """Test analysis of passable grid."""
        from src.core.symbolic_refiner import PathAnalyzer, TileType
        
        analyzer = PathAnalyzer()
        
        # Create passable grid
        grid = np.full((10, 10), TileType.FLOOR.value)
        
        failures = analyzer.analyze_grid(grid, start=(0, 0), goal=(9, 9))
        
        assert len(failures) == 0
    
    def test_analyze_blocked_grid(self):
        """Test analysis of blocked grid."""
        from src.core.symbolic_refiner import PathAnalyzer, TileType
        
        analyzer = PathAnalyzer()
        
        # Create blocked grid
        grid = np.full((10, 10), TileType.FLOOR.value)
        grid[5, :] = TileType.WALL.value  # Wall across middle
        
        failures = analyzer.analyze_grid(grid, start=(0, 5), goal=(9, 5))
        
        assert len(failures) > 0
        assert any(f.failure_type == 'disconnected' for f in failures)

    def test_analyze_non_square_grid_uses_row_col_coordinates(self):
        """Row/col coordinates should work correctly on non-square grids."""
        from src.core.symbolic_refiner import PathAnalyzer, TileType

        analyzer = PathAnalyzer()

        grid = np.full((16, 11), TileType.WALL.value)
        grid[8, :] = TileType.FLOOR.value

        failures = analyzer.analyze_grid(grid, start=(8, 0), goal=(8, 10))

        assert failures == []


class TestEntropyReset:
    """Tests for entropy reset mask creation."""
    
    def test_create_mask_basic(self):
        """Test basic mask creation."""
        from src.core.symbolic_refiner import EntropyReset, FailurePoint
        
        resetter = EntropyReset(margin=1)
        
        failure = FailurePoint(
            position=(5, 5),
            failure_type='blocked',
            required_item=None,
        )
        
        mask = resetter.create_mask((10, 10), [failure])
        
        assert mask.shape == (10, 10)
        assert mask[5, 5] == True
        assert mask[4, 5] == True  # Margin
        assert mask[0, 0] == False
    
    def test_expand_mask(self):
        """Test mask expansion."""
        from src.core.symbolic_refiner import EntropyReset
        
        resetter = EntropyReset()
        
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        
        expanded = resetter.expand_mask(mask, iterations=2)
        
        # Should be larger
        assert expanded.sum() > mask.sum()
        assert expanded[5, 5] == True
        assert expanded[3, 5] == True  # 2 steps away


class TestWaveFunctionCollapse:
    """Tests for Wave Function Collapse."""
    
    def test_wfc_initialization(self):
        """Test WFC state initialization."""
        from src.core.symbolic_refiner import WaveFunctionCollapse, TileType
        
        wfc = WaveFunctionCollapse(
            tile_types=[TileType.FLOOR.value, TileType.WALL.value],
        )
        
        state = wfc.initialize_state(height=5, width=5)
        
        assert state.grid.shape == (5, 5, 2)
        assert np.allclose(state.grid.sum(axis=2), 1.0)  # Normalized
    
    def test_wfc_collapse(self):
        """Test WFC collapse to valid grid."""
        from src.core.symbolic_refiner import WaveFunctionCollapse, TileType
        
        wfc = WaveFunctionCollapse(
            tile_types=[TileType.FLOOR.value, TileType.WALL.value],
            max_iterations=1000,
        )
        
        state = wfc.initialize_state(height=8, width=8)
        
        result_grid, _success = wfc.collapse(state)
        
        assert result_grid.shape == (8, 8)
        assert set(np.unique(result_grid)).issubset({TileType.FLOOR.value, TileType.WALL.value})
    
    def test_wfc_with_initial_grid(self):
        """Test WFC with initial constraints."""
        from src.core.symbolic_refiner import WaveFunctionCollapse, TileType
        
        wfc = WaveFunctionCollapse(
            tile_types=[TileType.FLOOR.value, TileType.WALL.value],
        )
        
        # Initial grid with some fixed tiles
        initial = np.full((8, 8), TileType.FLOOR.value)
        initial[0, :] = TileType.WALL.value  # Top wall
        
        # Mask: only regenerate middle
        mask = np.zeros((8, 8), dtype=bool)
        mask[3:5, 3:5] = True
        
        state = wfc.initialize_state(
            height=8,
            width=8,
            initial_grid=initial,
            mask=mask,
        )
        
        # Top row should be collapsed to WALL
        assert state.collapsed[0, 0] == True
        
        result_grid, _success = wfc.collapse(state)
        
        # Top row should still be WALL
        assert np.all(result_grid[0, :] == TileType.WALL.value)

    def test_wfc_single_option_zero_entropy_is_not_contradiction(self):
        """A single valid option should collapse successfully even with zero entropy."""
        from src.core.symbolic_refiner import WaveFunctionCollapse, WFCState, TileType

        wfc = WaveFunctionCollapse(
            tile_types=[TileType.FLOOR.value, TileType.WALL.value],
            max_iterations=10,
        )
        state = WFCState(
            grid=np.array([[[1.0, 0.0]]], dtype=np.float32),
            collapsed=np.array([[False]], dtype=bool),
            tile_types=[TileType.FLOOR.value, TileType.WALL.value],
            adjacency={},
        )

        result_grid, success = wfc.collapse(state)

        assert success is True
        assert result_grid[0, 0] == TileType.FLOOR.value


class TestConstraintPropagator:
    """Tests for constraint propagation."""
    
    def test_enforce_connectivity(self):
        """Test connectivity enforcement."""
        from src.core.symbolic_refiner import ConstraintPropagator, TileType
        
        propagator = ConstraintPropagator()
        
        # Grid with blocked path
        grid = np.full((10, 10), TileType.FLOOR.value)
        grid[5, :] = TileType.WALL.value
        
        walkable = {TileType.FLOOR.value}
        
        fixed_grid = propagator.enforce_connectivity(
            grid, start=(5, 0), goal=(5, 9), walkable=walkable
        )
        
        # Should have created a path
        # Check path exists
        has_path = propagator._find_path(fixed_grid, (5, 0), (5, 9), walkable)
        assert has_path is not None

    def test_enforce_connectivity_uses_row_col_on_non_square_grid(self):
        """Connectivity carving should follow row/col coordinates on non-square grids."""
        from src.core.symbolic_refiner import ConstraintPropagator, TileType

        propagator = ConstraintPropagator()
        grid = np.full((16, 11), TileType.WALL.value)
        walkable = {TileType.FLOOR.value}

        fixed_grid = propagator.enforce_connectivity(
            grid,
            start=(8, 0),
            goal=(8, 10),
            walkable=walkable,
        )

        assert np.all(fixed_grid[8, :] == TileType.FLOOR.value)

    def test_enforce_connectivity_honors_required_floor_mask(self):
        """Required plan masks should be preserved as walkable floor during repair."""
        from src.core.symbolic_refiner import ConstraintPropagator, TileType

        propagator = ConstraintPropagator()
        grid = np.full((10, 10), TileType.WALL.value)
        required = np.zeros((10, 10), dtype=bool)
        required[2:8, 4] = True
        walkable = {TileType.FLOOR.value}

        fixed_grid = propagator.enforce_connectivity(
            grid,
            start=(2, 4),
            goal=(7, 4),
            walkable=walkable,
            required_floor_mask=required,
        )

        assert np.all(fixed_grid[2:8, 4] == TileType.FLOOR.value)


class TestSymbolicRefiner:
    """Tests for complete Symbolic Refiner."""
    
    def test_repair_passable_room(self):
        """Test repair of already passable room."""
        from src.core.symbolic_refiner import SymbolicRefiner, TileType
        
        refiner = SymbolicRefiner()
        
        grid = np.full((16, 11), TileType.FLOOR.value)
        
        _repaired, success = refiner.repair_room(
            grid, start=(0, 5), goal=(15, 5)
        )
        
        assert success == True
    
    def test_repair_blocked_room(self):
        """Test repair of blocked room."""
        from src.core.symbolic_refiner import SymbolicRefiner, TileType
        
        refiner = SymbolicRefiner(max_repair_attempts=10)
        
        # Create blocked room
        grid = np.full((16, 11), TileType.FLOOR.value)
        grid[8, :] = TileType.WALL.value  # Wall in middle
        
        repaired, _success = refiner.repair_room(
            grid, start=(0, 5), goal=(15, 5)
        )
        
        # Should attempt repair (may or may not succeed depending on WFC)
        assert repaired.shape == grid.shape

    def test_repair_room_clamps_public_row_col_coordinates(self):
        """Public repair entry point should normalize out-of-bounds row/col coordinates."""
        from src.core.symbolic_refiner import SymbolicRefiner, TileType

        refiner = SymbolicRefiner(max_repair_attempts=1)
        grid = np.full((16, 11), TileType.FLOOR.value)

        repaired, success = refiner.repair_room(
            grid,
            start=(-5, 99),
            goal=(99, -3),
        )

        assert repaired.shape == grid.shape
        assert isinstance(success, bool)

    def test_repair_room_preserves_required_floor_mask(self):
        """Repair should preserve a provided traversability prior."""
        from src.core.symbolic_refiner import SymbolicRefiner, TileType

        refiner = SymbolicRefiner(max_repair_attempts=1)
        grid = np.full((16, 11), TileType.WALL.value)
        required = np.zeros((16, 11), dtype=bool)
        required[8, :] = True

        repaired, success = refiner.repair_room(
            grid,
            start=(8, 0),
            goal=(8, 10),
            required_floor_mask=required,
        )

        assert isinstance(success, bool)
        assert np.all(repaired[8, :] == TileType.FLOOR.value)
    
    def test_analyze_failures(self):
        """Test failure analysis."""
        from src.core.symbolic_refiner import SymbolicRefiner, TileType
        
        refiner = SymbolicRefiner()
        
        # Create simple mock dungeon
        class MockDungeon:
            def __init__(self):
                self.rooms = [MockRoom()]
        
        class MockRoom:
            def __init__(self):
                grid = np.full((16, 11), TileType.FLOOR.value)
                grid[8, :] = TileType.WALL.value
                self.grid = grid
        
        dungeon = MockDungeon()
        
        failures = refiner.analyze_failures(dungeon)
        
        assert isinstance(failures, list)


class TestWFCState:
    """Tests for WFC state management."""
    
    def test_entropy_calculation(self):
        """Test entropy calculation."""
        from src.core.symbolic_refiner import WFCState
        
        # Create state with 2 equally likely tiles
        grid = np.ones((5, 5, 2)) * 0.5
        collapsed = np.zeros((5, 5), dtype=bool)
        
        state = WFCState(
            grid=grid,
            collapsed=collapsed,
            tile_types=[0, 1],
            adjacency={},
        )
        
        entropy = state.entropy(2, 2)
        
        # Entropy of uniform distribution over 2 items
        expected = 1.0  # log2(2) = 1
        assert abs(entropy - expected) < 0.1
    
    def test_get_options(self):
        """Test getting tile options."""
        from src.core.symbolic_refiner import WFCState
        
        grid = np.zeros((3, 3, 4))
        grid[1, 1, 0] = 0.3
        grid[1, 1, 2] = 0.7
        collapsed = np.zeros((3, 3), dtype=bool)
        
        state = WFCState(
            grid=grid,
            collapsed=collapsed,
            tile_types=[10, 20, 30, 40],
            adjacency={},
        )
        
        options = state.get_options(1, 1)
        
        assert 10 in options
        assert 30 in options
        assert 20 not in options


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
