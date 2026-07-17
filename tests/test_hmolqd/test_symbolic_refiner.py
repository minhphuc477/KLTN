# pyright: reportPrivateUsage=false

"""
Tests for H-MOLQD Block VI: Symbolic Refiner
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

    def test_cost_map_normalization_preserves_astar_admissibility(self):
        """Cost guidance must not allow sub-unit steps with a Manhattan heuristic."""
        from src.core.symbolic_refiner import PathAnalyzer

        cost_map = np.array([[0.0, 0.5], [np.nan, np.inf]], dtype=np.float32)

        costs = PathAnalyzer._normalize_cost_map(cost_map, (2, 2))

        assert costs is not None
        assert float(costs.min()) >= 1.0
        assert float(costs[1, 1]) == pytest.approx(1e6)

    def test_analyze_non_square_grid_uses_row_col_coordinates(self):
        """Row/col coordinates should work correctly on non-square grids."""
        from src.core.symbolic_refiner import PathAnalyzer, TileType

        analyzer = PathAnalyzer()

        grid = np.full((16, 11), TileType.WALL.value)
        grid[8, :] = TileType.FLOOR.value

        failures = analyzer.analyze_grid(grid, start=(8, 0), goal=(8, 10))

        assert failures == []

    def test_analyze_graph_requires_boss_key_before_boss_locked_edge(self):
        """Boss-locked graph edges should fail unless the boss key was collected earlier."""
        import networkx as nx
        from src.core.symbolic_refiner import PathAnalyzer

        analyzer = PathAnalyzer()
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="")
        graph.add_node(2, label="t")
        graph.add_edge(0, 1, edge_type="open")
        graph.add_edge(1, 2, edge_type="boss_locked")

        failures = analyzer.analyze_graph(graph, 0, 2)

        assert any(f.failure_type == "missing_boss_key" for f in failures)

    def test_analyze_graph_accepts_boss_key_collected_before_boss_locked_edge(self):
        """Compact K boss-key labels should unlock later boss-locked graph edges."""
        import networkx as nx
        from src.core.symbolic_refiner import PathAnalyzer

        analyzer = PathAnalyzer()
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="K")
        graph.add_node(2, label="t")
        graph.add_edge(0, 1, edge_type="open")
        graph.add_edge(1, 2, edge_type="boss_locked")

        failures = analyzer.analyze_graph(graph, 0, 2)

        assert not any(f.failure_type == "missing_boss_key" for f in failures)

    def test_analyze_graph_consumes_small_keys_across_locked_edges(self):
        """One collected small key should not open two later locked edges."""
        import networkx as nx
        from src.core.symbolic_refiner import PathAnalyzer

        analyzer = PathAnalyzer()
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="key")
        graph.add_node(2, label="")
        graph.add_node(3, label="t")
        graph.add_edge(0, 1, edge_type="open")
        graph.add_edge(1, 2, edge_type="locked")
        graph.add_edge(2, 3, edge_type="locked")

        failures = analyzer.analyze_graph(graph, 0, 3)

        assert any(f.failure_type == "missing_key" for f in failures)

    def test_analyze_graph_accepts_path_with_enough_small_keys(self):
        """Two small keys should open two later locked edges in inventory order."""
        import networkx as nx
        from src.core.symbolic_refiner import PathAnalyzer

        analyzer = PathAnalyzer()
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="key")
        graph.add_node(2, label="small_key")
        graph.add_node(3, label="")
        graph.add_node(4, label="t")
        graph.add_edge(0, 1, edge_type="open")
        graph.add_edge(1, 2, edge_type="open")
        graph.add_edge(2, 3, edge_type="locked")
        graph.add_edge(3, 4, edge_type="locked")

        failures = analyzer.analyze_graph(graph, 0, 4)

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

    def test_enforce_connectivity_prefers_low_cost_carve_path(self):
        """LogicNet-style cost maps should guide symbolic wall carving."""
        from src.core.symbolic_refiner import ConstraintPropagator, TileType

        propagator = ConstraintPropagator()
        grid = np.full((5, 5), TileType.WALL.value)
        walkable = {TileType.FLOOR.value}
        cost_map = np.full((5, 5), 10.0, dtype=np.float32)
        cost_map[:, 0] = 1.0
        cost_map[4, :] = 1.0

        fixed_grid = propagator.enforce_connectivity(
            grid,
            start=(0, 0),
            goal=(4, 4),
            walkable=walkable,
            cost_map=cost_map,
        )

        assert np.all(fixed_grid[:, 0] == TileType.FLOOR.value)
        assert np.all(fixed_grid[4, :] == TileType.FLOOR.value)
        assert np.all(fixed_grid[0, 1:4] == TileType.WALL.value)

    def test_enforce_connectivity_derives_soft_costs_without_l_shape_fallback(self):
        """Missing cost maps should still follow existing structure, not hardcoded L-shapes."""
        from src.core.symbolic_refiner import ConstraintPropagator, TileType

        propagator = ConstraintPropagator()
        grid = np.full((5, 5), TileType.WALL.value)
        grid[1:5, 0] = TileType.FLOOR.value
        grid[4, 0:4] = TileType.FLOOR.value
        walkable = {TileType.FLOOR.value}

        fixed_grid = propagator.enforce_connectivity(
            grid,
            start=(0, 0),
            goal=(4, 4),
            walkable=walkable,
        )

        assert np.all(fixed_grid[:, 0] == TileType.FLOOR.value)
        assert np.all(fixed_grid[4, :] == TileType.FLOOR.value)
        assert np.all(fixed_grid[0, 1:4] == TileType.WALL.value)


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

    def test_repair_room_excludes_required_floor_mask_from_wfc_reset(self):
        """Local WFC regeneration must not alter tiles forced by LogicNet floor masks."""
        from src.core.symbolic_refiner import FailurePoint, SymbolicRefiner, TileType

        class _Analyzer:
            def __init__(self):
                self.calls = 0

            def analyze_grid(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return [
                        FailurePoint(
                            position=(8, 5),
                            failure_type="disconnected",
                            required_item=None,
                            blocking_tiles=[(8, 5)],
                        )
                    ]
                return []

        class _EntropyReset:
            def create_mask(self, shape, _failures):
                mask = np.zeros(shape, dtype=bool)
                mask[8, 5] = True
                mask[8, 6] = True
                return mask

            def expand_mask(self, mask, iterations=1):
                _ = iterations
                return mask

        class _WFC:
            def __init__(self):
                self.seen_mask = None

            def initialize_state(self, **kwargs):
                self.seen_mask = np.asarray(kwargs["mask"], dtype=bool).copy()
                return kwargs

            def collapse(self, state):
                return np.asarray(state["initial_grid"]).copy(), True

        refiner = SymbolicRefiner(max_repair_attempts=2)
        refiner.path_analyzer = _Analyzer()
        refiner.entropy_reset = _EntropyReset()
        refiner.wfc = _WFC()
        refiner.refresh_learned_rules = lambda: None
        grid = np.full((16, 11), TileType.WALL.value)
        required = np.zeros((16, 11), dtype=bool)
        required[8, 5] = True

        _repaired, _success = refiner.repair_room(
            grid,
            start=(8, 0),
            goal=(8, 10),
            required_floor_mask=required,
        )

        assert refiner.wfc.seen_mask is not None
        assert bool(refiner.wfc.seen_mask[8, 5]) is False
        assert bool(refiner.wfc.seen_mask[8, 6]) is True

    def test_repair_room_excludes_topology_tiles_from_wfc_reset(self):
        """A dilated contradiction region must not erase doors or route endpoints."""
        from src.core.symbolic_refiner import FailurePoint, SymbolicRefiner, TileType

        class _Analyzer:
            def analyze_grid(self, *_args, **_kwargs):
                return [
                    FailurePoint(
                        position=(8, 5),
                        failure_type="disconnected",
                        required_item=None,
                        blocking_tiles=[],
                    )
                ]

        class _EntropyReset:
            def create_mask(self, shape, _failures):
                return np.ones(shape, dtype=bool)

            def expand_mask(self, mask, iterations=1):
                _ = iterations
                return mask

        class _WFC:
            def __init__(self):
                self.seen_mask = None

            def initialize_state(self, **kwargs):
                self.seen_mask = np.asarray(kwargs["mask"], dtype=bool).copy()
                return kwargs

            def collapse(self, state):
                return np.asarray(state["initial_grid"]).copy(), False

        refiner = SymbolicRefiner(max_repair_attempts=1)
        refiner.path_analyzer = _Analyzer()
        refiner.entropy_reset = _EntropyReset()
        refiner.wfc = _WFC()
        refiner.refresh_learned_rules = lambda: None
        grid = np.full((16, 11), TileType.WALL.value)
        grid[0, 5] = TileType.DOOR_LOCKED.value
        grid[8, 1] = TileType.START.value
        grid[8, 9] = TileType.TRIFORCE.value
        grid[15, 5] = TileType.STAIR.value

        refiner.repair_room(grid, start=(8, 1), goal=(8, 9))

        assert refiner.wfc.seen_mask is not None
        for position in ((0, 5), (8, 1), (8, 9), (15, 5)):
            assert bool(refiner.wfc.seen_mask[position]) is False
        assert bool(refiner.wfc.seen_mask[8, 5]) is True

    def test_feedback_cannot_overwrite_original_topology_tiles(self):
        """A full-grid neural callback must still honor immutable topology anchors."""
        from src.core.symbolic_refiner import FailurePoint, SymbolicRefiner, TileType

        class _Analyzer:
            def __init__(self):
                self.calls = 0

            def analyze_grid(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return [FailurePoint((8, 5), "disconnected", None, [])]
                return []

        class _EntropyReset:
            def create_mask(self, shape, _failures):
                return np.ones(shape, dtype=bool)

            def expand_mask(self, mask, iterations=1):
                _ = iterations
                return mask

        class _FailingWFC:
            def initialize_state(self, **kwargs):
                return kwargs

            def collapse(self, state):
                return np.asarray(state["initial_grid"]).copy(), False

        refiner = SymbolicRefiner(max_repair_attempts=2)
        refiner.path_analyzer = _Analyzer()
        refiner.entropy_reset = _EntropyReset()
        refiner.wfc = _FailingWFC()
        refiner.refresh_learned_rules = lambda: None

        grid = np.full((16, 11), TileType.WALL.value)
        anchors = {
            (0, 5): TileType.DOOR_LOCKED.value,
            (8, 1): TileType.START.value,
            (8, 9): TileType.TRIFORCE.value,
            (15, 5): TileType.STAIR.value,
        }
        for position, tile in anchors.items():
            grid[position] = tile

        def _hostile_feedback(current, *_args):
            return np.full_like(current, TileType.FLOOR.value)

        repaired, success, diagnostics = refiner.repair_room_with_feedback(
            grid,
            start=(8, 1),
            goal=(8, 9),
            feedback_callback=_hostile_feedback,
            max_feedback_rounds=1,
        )

        assert success is True
        assert diagnostics["feedback_applied"] == 1
        for position, tile in anchors.items():
            assert int(repaired[position]) == int(tile)

    def test_repair_room_seed_makes_wfc_reproducible(self):
        """Seeded repair should not depend on NumPy's global random state."""
        from src.core.symbolic_refiner import SymbolicRefiner, TileType

        refiner = SymbolicRefiner(max_repair_attempts=3)
        grid = np.full((16, 11), TileType.WALL.value)
        grid[1, 1] = TileType.START.value
        grid[14, 9] = TileType.TRIFORCE.value

        np.random.seed(1)
        repaired_a, success_a = refiner.repair_room(grid, start=(1, 1), goal=(14, 9), seed=123)
        np.random.seed(999)
        repaired_b, success_b = refiner.repair_room(grid, start=(1, 1), goal=(14, 9), seed=123)

        assert success_a == success_b
        assert np.array_equal(repaired_a, repaired_b)

    def test_local_wfc_cannot_invent_graph_owned_topology_tiles(self):
        from src.core.symbolic_refiner import FailurePoint, SymbolicRefiner, TileType

        class _Analyzer:
            def __init__(self):
                self.calls = 0

            def analyze_grid(self, *_args, **_kwargs):
                self.calls += 1
                return (
                    [FailurePoint((8, 5), "disconnected", None, [])]
                    if self.calls == 1
                    else []
                )

        class _EntropyReset:
            def create_mask(self, shape, _failures):
                return np.ones(shape, dtype=bool)

            def expand_mask(self, mask, iterations=1):
                _ = iterations
                return mask

        class _TopologyInventingWFC:
            def initialize_state(self, **kwargs):
                return kwargs

            def collapse(self, state):
                invented = np.full_like(state["initial_grid"], TileType.START.value)
                return invented, True

        refiner = SymbolicRefiner(max_repair_attempts=2)
        refiner.path_analyzer = _Analyzer()
        refiner.entropy_reset = _EntropyReset()
        refiner.wfc = _TopologyInventingWFC()
        refiner.refresh_learned_rules = lambda: None

        grid = np.full((16, 11), TileType.FLOOR.value)
        grid[8, 1] = TileType.START.value
        grid[8, 9] = TileType.TRIFORCE.value
        repaired, success, _ = refiner.repair_room_with_feedback(
            grid,
            start=(8, 1),
            goal=(8, 9),
        )

        assert success is True
        assert np.count_nonzero(repaired == TileType.START.value) == 1
        assert np.count_nonzero(repaired == TileType.TRIFORCE.value) == 1
    
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

    def test_entropy_at_uses_row_col_order(self):
        from src.core.symbolic_refiner import WFCState

        grid = np.zeros((2, 3, 2), dtype=np.float32)
        grid[:, :, :] = [1.0, 0.0]
        grid[1, 2, :] = [0.5, 0.5]
        state = WFCState(
            grid=grid,
            collapsed=np.zeros((2, 3), dtype=bool),
            tile_types=[0, 1],
            adjacency={},
        )

        assert state.entropy_at(1, 2) == pytest.approx(1.0)
        assert state.entropy_at(0, 1) == pytest.approx(0.0)

    def test_default_adjacency_does_not_force_entity_self_adjacency(self):
        from src.core.symbolic_refiner import DEFAULT_ADJACENCY, TileType

        assert TileType.FLOOR.value in DEFAULT_ADJACENCY[TileType.FLOOR.value]
        assert TileType.START.value not in DEFAULT_ADJACENCY[TileType.START.value]
        assert TileType.TRIFORCE.value not in DEFAULT_ADJACENCY[TileType.TRIFORCE.value]
    
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
