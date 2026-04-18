"""
Tests for H-MOLQD Block VI: External Validator & MAP-Elites
============================================================

Tests for solvability validation and quality diversity.
"""

import pytest

# NetworkX required for these tests
nx = pytest.importorskip("networkx")


class TestAgentSimulator:
    """Tests for agent-based simulation."""
    
    def test_simulation_basic(self):
        """Test basic simulation."""
        from src.evaluation.validator import AgentSimulator
        
        simulator = AgentSimulator()
        
        # Create simple solvable dungeon graph
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="")
        graph.add_node(2, label="t")
        graph.add_edge(0, 1, edge_type="open")
        graph.add_edge(1, 2, edge_type="open")
        
        result = simulator.simulate(graph)
        
        assert result.is_solvable == True
        assert result.path is not None
    
    def test_simulation_locked_path(self):
        """Test simulation with locked doors."""
        from src.evaluation.validator import AgentSimulator
        
        simulator = AgentSimulator()
        
        # Dungeon: start -> key -> locked_door -> goal
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="k")  # Key
        graph.add_node(2, label="t")
        graph.add_edge(0, 1, edge_type="open")
        graph.add_edge(1, 2, edge_type="key_locked")
        
        result = simulator.simulate(graph)
        
        assert result.is_solvable == True
    
    def test_simulation_unsolvable(self):
        """Test unsolvable dungeon detection."""
        from src.evaluation.validator import AgentSimulator
        
        simulator = AgentSimulator()
        
        # Locked door but no key
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="t")
        graph.add_edge(0, 1, edge_type="key_locked")
        
        result = simulator.simulate(graph)
        
        assert result.is_solvable == False

    def test_simulation_respects_boss_key_and_item_gate(self):
        """Boss/item progression should not be flattened into open traversal."""
        from src.evaluation.validator import AgentSimulator

        simulator = AgentSimulator()

        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="K")  # boss key
        graph.add_node(2, label="I")  # key item / ladder / bomb item
        graph.add_node(3, label="")
        graph.add_node(4, label="t")
        graph.add_edge(0, 1, edge_type="open")
        graph.add_edge(1, 2, edge_type="boss_locked")
        graph.add_edge(2, 3, edge_type="item_gate", item_required="BOMB")
        graph.add_edge(3, 4, edge_type="open")

        result = simulator.simulate(graph)

        assert result.is_solvable is True

    def test_simulation_requires_switch_before_switch_locked_edge(self):
        """Switch-like nodes should be required before switch_locked progression edges."""
        from src.evaluation.validator import AgentSimulator

        simulator = AgentSimulator()

        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, type="SWITCH", has_puzzle=True)
        graph.add_node(2, label="t")
        graph.add_edge(0, 1, edge_type="open")
        graph.add_edge(1, 2, edge_type="switch_locked")

        result = simulator.simulate(graph)
        assert result.is_solvable is True

        unswitched = nx.DiGraph()
        unswitched.add_node(0, label="s")
        unswitched.add_node(1, label="")
        unswitched.add_node(2, label="t")
        unswitched.add_edge(0, 1, edge_type="open")
        unswitched.add_edge(1, 2, edge_type="switch_locked")

        result_unswitched = simulator.simulate(unswitched)
        assert result_unswitched.is_solvable is False


class TestSolvabilityChecker:
    """Tests for solvability checking."""
    
    def test_checker_solvable(self):
        """Test solvable dungeon."""
        from src.evaluation.validator import SolvabilityChecker
        
        checker = SolvabilityChecker()
        
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="t")
        graph.add_edge(0, 1, edge_type="open")
        
        is_solvable, path = checker.check(graph)
        
        assert is_solvable == True
        assert path is not None
    
    def test_checker_disconnected(self):
        """Test disconnected dungeon."""
        from src.evaluation.validator import SolvabilityChecker
        
        checker = SolvabilityChecker()
        
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="t")
        # No edge between start and goal
        
        is_solvable, _path = checker.check(graph)
        
        assert is_solvable == False


class TestExternalValidator:
    """Tests for complete External Validator."""
    
    def test_validate_graph(self):
        """Test validation of graph dungeon."""
        from src.evaluation.validator import ExternalValidator
        
        validator = ExternalValidator()
        
        graph = nx.DiGraph()
        graph.add_node(0, label="s,k")
        graph.add_node(1, label="")
        graph.add_node(2, label="t")
        graph.add_edge(0, 1, edge_type="open")
        graph.add_edge(1, 2, edge_type="key_locked")
        
        result = validator.validate(graph)
        
        assert hasattr(result, 'is_solvable')
        assert result.is_solvable == True


class TestCBSFitnessProxy:
    """Tests for graph-proxy CBS fitness semantics."""

    def test_graph_proxy_uses_explicit_start_and_goal_semantics(self):
        """Explicit start/goal nodes should win over degree-based fallbacks."""
        from src.evaluation.cbs_fitness import compute_cbs_fitness

        graph = nx.DiGraph()
        graph.add_node(10, label="s")
        graph.add_node(20, label="")
        graph.add_node(30, label="t")
        graph.add_node(40, label="")
        graph.add_edge(40, 10, edge_type="open")
        graph.add_edge(10, 20, edge_type="open")
        graph.add_edge(20, 30, edge_type="open")

        metrics = compute_cbs_fitness(graph)

        assert metrics["solvable_astar"] is True
        assert metrics["astar_path_length"] == 2


class TestFunMetrics:
    """Tests for fun metrics evaluator (including pacing analyzer)."""

    def test_pacing_changes_with_tension_profile(self):
        """Late-peak profile should produce later peak placement than flat profile."""
        from src.evaluation.fun_metrics import FunMetricsEvaluator

        evaluator = FunMetricsEvaluator()

        graph = nx.DiGraph()
        for i in range(5):
            label = "s" if i == 0 else ("t" if i == 4 else "")
            graph.add_node(i, label=label)
            if i > 0:
                graph.add_edge(i - 1, i, edge_type="open")

        solution_path = [0, 1, 2, 3, 4]
        critical_path = set(solution_path)

        late_peak_contents = {
            0: {"safe_room": True, "enemies": 0},
            1: {"enemies": 1, "puzzles": 1},
            2: {"enemies": 2, "puzzles": 1, "health_pickups": 1},
            3: {"enemies": 4, "boss": True},
            4: {"enemies": 1, "goal": True},
        }
        flat_contents = {room: {"enemies": 1} for room in solution_path}

        late_metrics = evaluator.evaluate(graph, solution_path, late_peak_contents, critical_path)
        flat_metrics = evaluator.evaluate(graph, solution_path, flat_contents, critical_path)

        assert 0.0 <= late_metrics.pacing.pacing_score <= 1.0
        assert 0.0 <= late_metrics.pacing.peak_placement <= 1.0
        assert late_metrics.pacing.rest_areas >= 0
        assert late_metrics.pacing.peak_placement > flat_metrics.pacing.peak_placement

    def test_pacing_handles_empty_path(self):
        """Empty path should return safe fallback pacing metrics."""
        from src.evaluation.fun_metrics import FunMetricsEvaluator

        evaluator = FunMetricsEvaluator()
        graph = nx.DiGraph()

        metrics = evaluator.evaluate(
            mission_graph=graph,
            solution_path=[],
            room_contents={},
            critical_path=set(),
        )

        assert metrics.pacing.tension_variance == 0.0
        assert metrics.pacing.rest_areas == 0
        assert 0.0 <= metrics.pacing.pacing_score <= 1.0


class TestEliteArchive:
    """Tests for Elite Archive."""
    
    def test_archive_add(self):
        """Test adding to archive."""
        from src.evaluation.map_elites import EliteArchive
        
        archive = EliteArchive(
            feature_dims=2,
            cells_per_dim=10,
        )
        
        # Add a solution
        was_added = archive.add(
            solution="dungeon_1",
            fitness=0.8,
            features=(0.5, 0.5),
        )
        
        assert was_added == True
        assert len(archive.archive) == 1
    
    def test_archive_replacement(self):
        """Test that better solutions replace worse ones."""
        from src.evaluation.map_elites import EliteArchive
        
        archive = EliteArchive(
            feature_dims=2,
            cells_per_dim=10,
        )
        
        # Add initial solution
        archive.add("dungeon_1", 0.5, (0.5, 0.5))
        
        # Add better solution in same cell
        was_added = archive.add("dungeon_2", 0.9, (0.5, 0.5))
        
        assert was_added == True
        assert archive.archive[(5, 5)].fitness == 0.9
        assert archive.archive[(5, 5)].solution == "dungeon_2"
    
    def test_archive_no_replacement_if_worse(self):
        """Test that worse solutions don't replace better ones."""
        from src.evaluation.map_elites import EliteArchive
        
        archive = EliteArchive(
            feature_dims=2,
            cells_per_dim=10,
        )
        
        archive.add("dungeon_1", 0.9, (0.5, 0.5))
        was_added = archive.add("dungeon_2", 0.5, (0.5, 0.5))
        
        assert was_added == False
        assert archive.archive[(5, 5)].fitness == 0.9


class TestFeatureExtractor:
    """Tests for feature extraction."""
    
    def test_linearity_leniency_extractor(self):
        """Test Linearity-Leniency feature extractor."""
        from src.evaluation.map_elites import LinearityLeniencyExtractor
        
        extractor = LinearityLeniencyExtractor()
        
        # Create linear dungeon
        graph = nx.DiGraph()
        for i in range(5):
            label = ""
            if i == 0:
                label = "s"
            elif i == 4:
                label = "t"
            graph.add_node(i, label=label)
            if i > 0:
                graph.add_edge(i-1, i, edge_type="open")
        
        linearity, leniency = extractor.extract(graph)
        
        assert 0 <= linearity <= 1
        assert 0 <= leniency <= 1


class TestMAPElites:
    """Tests for MAP-Elites algorithm."""
    
    def test_map_elites_add(self):
        """Test adding dungeons to MAP-Elites."""
        from src.evaluation.map_elites import MAPElites, LinearityLeniencyExtractor
        
        map_elites = MAPElites(
            feature_extractor=LinearityLeniencyExtractor(),
            fitness_fn=lambda g: 1.0,  # Always solvable
            cells_per_dim=5,
        )
        
        # Create test dungeon
        graph = nx.DiGraph()
        graph.add_node(0, label="s")
        graph.add_node(1, label="t")
        graph.add_edge(0, 1, edge_type="open")
        
        was_added, fitness, features = map_elites.add(graph)
        
        assert was_added == True
        assert fitness == 1.0
        assert len(features) == 2
    
    def test_map_elites_diversity(self):
        """Test diversity in MAP-Elites."""
        from src.evaluation.map_elites import MAPElites, LinearityLeniencyExtractor
        
        map_elites = MAPElites(
            feature_extractor=LinearityLeniencyExtractor(),
            fitness_fn=lambda g: 1.0,
            cells_per_dim=10,
        )
        
        # Add diverse dungeons
        for i in range(10):
            graph = nx.DiGraph()
            graph.add_node(0, label="s")
            
            # Add varying number of intermediate nodes
            for j in range(1, i + 2):
                label = "k" if j % 3 == 0 else ""
                if j == i + 1:
                    label = "t"
                graph.add_node(j, label=label)
                edge_type = "key_locked" if j % 3 == 0 else "open"
                graph.add_edge(j-1, j, edge_type=edge_type)
            
            map_elites.add(graph)
        
        # Should have some diversity
        stats = map_elites.get_archive_stats()
        assert stats.num_elites >= 3
    
    def test_map_elites_diverse_set(self):
        """Test getting diverse set of elites."""
        from src.evaluation.map_elites import MAPElites, LinearityLeniencyExtractor
        
        map_elites = MAPElites(
            feature_extractor=LinearityLeniencyExtractor(),
            fitness_fn=lambda g: 1.0,
            cells_per_dim=10,
        )
        
        # Add several dungeons
        for i in range(5):
            graph = nx.DiGraph()
            graph.add_node(0, label="s")
            graph.add_node(1, label="t")
            graph.add_edge(0, 1, edge_type="open")
            map_elites.add(graph, precomputed_features=(i*0.2, i*0.1))
        
        diverse_set = map_elites.get_diverse_set(n=3)
        
        assert len(diverse_set) <= 3


class TestDiversityMetrics:
    """Tests for diversity metrics."""
    
    def test_metrics_computation(self):
        """Test diversity metrics computation."""
        from src.evaluation.map_elites import EliteArchive, DiversityMetrics
        
        archive = EliteArchive(
            feature_dims=2,
            cells_per_dim=10,
        )
        
        # Add some solutions
        archive.add("a", 0.5, (0.1, 0.1))
        archive.add("b", 0.7, (0.5, 0.5))
        archive.add("c", 0.9, (0.9, 0.9))
        
        metrics = DiversityMetrics(archive)
        
        coverage = metrics.coverage()
        qd_score = metrics.qd_score()
        uniformity = metrics.uniformity()
        
        assert coverage > 0
        assert qd_score == 0.5 + 0.7 + 0.9
        assert 0 <= uniformity <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
