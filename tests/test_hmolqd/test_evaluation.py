"""
Tests for H-MOLQD Block VI: External Validator & MAP-Elites
============================================================

Tests for solvability validation and quality diversity.
"""

import pytest
import numpy as np

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
        
        is_solvable, path = checker.check(graph)
        
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
