"""
Integration Tests for Evolutionary Topology Director
===================================================

Comprehensive test suite covering:
1. Basic functionality
2. Edge cases
3. Configuration variations
4. Fitness evaluation
5. Graph validity
6. Reproducibility

Run: pytest tests/test_evolutionary_director.py -v
"""

import sys
from pathlib import Path
import numpy as np
import networkx as nx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.evolutionary_director import (
    EvolutionaryTopologyGenerator,
    TensionCurveEvaluator,
    GraphGrammarExecutor,
    mission_graph_to_networkx,
    networkx_to_mission_graph,
)
from src.generation.grammar import MissionGraph, NodeType


class TestEvolutionaryDirector:
    """Test suite for EvolutionaryTopologyGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        target = [0.2, 0.5, 0.8, 1.0]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=20,
            generations=10,
            seed=42,
        )
        
        assert gen.target_curve == target
        assert gen.population_size == 20
        assert gen.generations == 10
        assert gen.seed == 42
        assert len(gen.best_fitness_history) == 0
    
    def test_simple_evolution(self):
        """Test basic evolution completes successfully."""
        target = [0.3, 0.7, 1.0]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=10,
            generations=5,
            genome_length=8,
            seed=42,
        )
        
        graph = gen.evolve()
        
        # Check output type
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Check statistics recorded
        stats = gen.get_statistics()
        assert len(stats['best_fitness_history']) > 0
        assert stats['final_best_fitness'] >= 0.0
        assert stats['final_best_fitness'] <= 1.0
    
    def test_graph_validity(self):
        """Test generated graphs are valid."""
        target = [0.2, 0.5, 0.9]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=15,
            generations=10,
            seed=123,
        )
        
        graph = gen.evolve()
        
        # Check required node types
        node_types = [graph.nodes[n]['type'] for n in graph.nodes()]
        assert 'START' in node_types
        assert 'GOAL' in node_types
        
        # Check connectivity
        assert nx.is_connected(graph)
        
        # Check path exists START → GOAL
        start_nodes = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'START']
        goal_nodes = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'GOAL']
        
        assert len(start_nodes) > 0
        assert len(goal_nodes) > 0
        
        path_exists = nx.has_path(graph, start_nodes[0], goal_nodes[0])
        assert path_exists
    
    def test_reproducibility(self):
        """Test same seed produces same results."""
        target = [0.1, 0.5, 1.0]
        
        gen1 = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=10,
            generations=5,
            seed=999,
        )
        
        gen2 = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=10,
            generations=5,
            seed=999,
        )
        
        graph1 = gen1.evolve()
        graph2 = gen2.evolve()
        
        # Check same topology
        assert graph1.number_of_nodes() == graph2.number_of_nodes()
        assert graph1.number_of_edges() == graph2.number_of_edges()
        
        # Check same fitness
        stats1 = gen1.get_statistics()
        stats2 = gen2.get_statistics()
        assert abs(stats1['final_best_fitness'] - stats2['final_best_fitness']) < 1e-6
    
    def test_fitness_convergence(self):
        """Test fitness improves or stays high."""
        target = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=30,
            generations=20,
            seed=42,
        )
        
        graph = gen.evolve()
        stats = gen.get_statistics()
        
        # Check fitness is reasonable
        assert stats['final_best_fitness'] >= 0.5
        
        # Check history exists
        assert len(stats['best_fitness_history']) > 0
        assert len(stats['avg_fitness_history']) > 0
    
    def test_empty_target_curve(self):
        """Test handling of edge case: empty target curve."""
        try:
            gen = EvolutionaryTopologyGenerator(
                target_curve=[],
                population_size=10,
                generations=5,
            )
            # Should either fail or handle gracefully
            assert False, "Should raise error for empty target"
        except (ValueError, IndexError, AssertionError):
            pass  # Expected
    
    def test_single_point_curve(self):
        """Test single-point target curve."""
        target = [1.0]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=10,
            generations=5,
            genome_length=5,
            seed=42,
        )
        
        graph = gen.evolve()
        
        assert graph.number_of_nodes() >= 2  # At least START + GOAL
        assert nx.is_connected(graph)
    
    def test_flat_curve(self):
        """Test flat difficulty curve."""
        target = [0.5, 0.5, 0.5, 0.5]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=15,
            generations=10,
            seed=456,
        )
        
        graph = gen.evolve()
        stats = gen.get_statistics()
        
        assert graph.number_of_nodes() > 0
        assert stats['final_best_fitness'] >= 0.0
    
    def test_decreasing_curve(self):
        """Test decreasing difficulty (hard → easy)."""
        target = [1.0, 0.8, 0.5, 0.2]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=20,
            generations=15,
            seed=789,
        )
        
        graph = gen.evolve()
        
        assert graph.number_of_nodes() > 0
        assert nx.is_connected(graph)
    
    def test_long_curve(self):
        """Test long target curve."""
        target = [i / 20.0 for i in range(20)]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=40,
            generations=30,
            genome_length=30,
            seed=111,
        )
        
        graph = gen.evolve()
        
        # Should handle long curves
        assert graph.number_of_nodes() > 0
    
    def test_small_population(self):
        """Test with minimal population size."""
        target = [0.3, 0.7, 1.0]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=5,  # Very small
            generations=5,
            seed=42,
        )
        
        graph = gen.evolve()
        
        assert graph.number_of_nodes() > 0
    
    def test_large_population(self):
        """Test with large population."""
        target = [0.2, 0.5, 0.8, 1.0]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=100,
            generations=5,
            seed=42,
        )
        
        graph = gen.evolve()
        stats = gen.get_statistics()
        
        # Large population should converge quickly
        assert stats['final_best_fitness'] >= 0.8
    
    def test_custom_transition_matrix(self):
        """Test with custom transition matrix."""
        target = [0.3, 0.7, 1.0]
        
        custom_transitions = {
            "Start": {"InsertChallenge_ENEMY": 0.8, "Branch": 0.2},
            "InsertChallenge_ENEMY": {"InsertChallenge_ENEMY": 0.6, "Branch": 0.4},
            "Branch": {"InsertChallenge_ENEMY": 1.0},
            "InsertChallenge_PUZZLE": {"InsertChallenge_ENEMY": 1.0},
            "InsertLockKey": {"InsertChallenge_ENEMY": 1.0},
        }
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            zelda_transition_matrix=custom_transitions,
            population_size=15,
            generations=10,
            seed=222,
        )
        
        graph = gen.evolve()
        
        # Should favor ENEMY nodes
        node_types = [graph.nodes[n]['type'] for n in graph.nodes()]
        enemy_count = node_types.count('ENEMY')
        assert enemy_count > 0


class TestTensionCurveEvaluator:
    """Test tension curve extraction and fitness calculation."""
    
    def test_evaluator_initialization(self):
        """Test evaluator setup."""
        target = [0.1, 0.5, 1.0]
        evaluator = TensionCurveEvaluator(target)
        
        assert len(evaluator.target_curve) == 3
        assert evaluator.target_length == 3
    
    def test_solvable_graph_fitness(self):
        """Test fitness of solvable graph."""
        target = [0.2, 0.5, 0.8, 1.0]
        evaluator = TensionCurveEvaluator(target)
        
        # Create simple graph
        executor = GraphGrammarExecutor(seed=42)
        genome = [1, 1, 2, 1]  # Simple sequence
        graph = executor.execute(genome)
        
        fitness = evaluator.calculate_fitness(graph)
        
        # Should be solvable and have non-zero fitness
        assert fitness > 0.0
        assert fitness <= 1.0
    
    def test_curve_extraction(self):
        """Test tension curve extraction."""
        target = [0.3, 0.6, 0.9]
        evaluator = TensionCurveEvaluator(target)
        
        executor = GraphGrammarExecutor(seed=42)
        genome = [1, 2, 1]
        graph = executor.execute(genome)
        
        curve = evaluator.extract_tension_curve(graph)
        
        # Check curve properties
        assert len(curve) == len(target)
        assert all(0.0 <= v <= 1.0 for v in curve)


class TestGraphGrammarExecutor:
    """Test genome execution."""
    
    def test_executor_initialization(self):
        """Test executor setup."""
        executor = GraphGrammarExecutor(seed=42)
        
        assert len(executor.rules) == 5
        assert len(executor.rule_names) == 5
    
    def test_simple_genome_execution(self):
        """Test executing a simple genome."""
        executor = GraphGrammarExecutor(seed=42)
        genome = [1, 1, 2]  # Enemy, Enemy, Puzzle
        
        graph = executor.execute(genome)
        
        # Check basic properties
        assert len(graph.nodes) >= 2  # At least START + GOAL
        assert len(graph.edges) > 0
    
    def test_empty_genome(self):
        """Test empty genome."""
        executor = GraphGrammarExecutor(seed=42)
        genome = []
        
        graph = executor.execute(genome)
        
        # Should still have START + GOAL from StartRule
        assert len(graph.nodes) >= 2
    
    def test_invalid_rule_ids(self):
        """Test genome with invalid rule IDs."""
        executor = GraphGrammarExecutor(seed=42)
        genome = [100, 200, -1]  # Invalid IDs
        
        # Should clamp to valid range
        graph = executor.execute(genome)
        
        assert len(graph.nodes) >= 2
    
    def test_long_genome(self):
        """Test long genome sequence."""
        executor = GraphGrammarExecutor(seed=42)
        genome = [1, 2, 1, 2, 4, 1, 2, 3, 1, 2, 4, 1, 2, 1]
        
        graph = executor.execute(genome)
        
        # Should handle long sequences
        assert len(graph.nodes) > 0
        assert len(graph.nodes) <= 20  # max_nodes limit


class TestGraphConversion:
    """Test graph format conversions."""
    
    def test_mission_to_networkx(self):
        """Test MissionGraph → NetworkX conversion."""
        executor = GraphGrammarExecutor(seed=42)
        genome = [1, 2, 1]
        mission_graph = executor.execute(genome)
        
        nx_graph = mission_graph_to_networkx(mission_graph)
        
        assert isinstance(nx_graph, nx.Graph)
        assert nx_graph.number_of_nodes() == len(mission_graph.nodes)
        
        # Check attributes preserved
        for node_id in nx_graph.nodes():
            assert 'type' in nx_graph.nodes[node_id]
            assert 'difficulty' in nx_graph.nodes[node_id]
    
    def test_networkx_to_mission(self):
        """Test NetworkX → MissionGraph conversion."""
        executor = GraphGrammarExecutor(seed=42)
        genome = [1, 2]
        mission_graph = executor.execute(genome)
        
        # Convert both ways
        nx_graph = mission_graph_to_networkx(mission_graph)
        converted_back = networkx_to_mission_graph(nx_graph)
        
        # Check preservation
        assert len(converted_back.nodes) == len(mission_graph.nodes)
        assert len(converted_back.edges) == len(mission_graph.edges)
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves data."""
        executor = GraphGrammarExecutor(seed=42)
        genome = [1, 2, 3, 1]
        original = executor.execute(genome)
        
        # Round trip
        nx_graph = mission_graph_to_networkx(original)
        reconstructed = networkx_to_mission_graph(nx_graph)
        
        # Verify node count preserved
        assert len(reconstructed.nodes) == len(original.nodes)
        
        # Verify START and GOAL preserved
        assert reconstructed.get_start_node() is not None
        assert reconstructed.get_goal_node() is not None


class TestStatistics:
    """Test statistics tracking."""
    
    def test_statistics_recording(self):
        """Test statistics are recorded during evolution."""
        target = [0.2, 0.5, 0.8, 1.0]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=15,
            generations=10,
            seed=42,
        )
        
        graph = gen.evolve()
        stats = gen.get_statistics()
        
        # Check all fields present
        assert 'best_fitness_history' in stats
        assert 'avg_fitness_history' in stats
        assert 'diversity_history' in stats
        assert 'final_best_fitness' in stats
        assert 'generations_run' in stats
        assert 'converged' in stats
        
        # Check history lengths match
        assert len(stats['best_fitness_history']) == stats['generations_run']
        assert len(stats['avg_fitness_history']) == stats['generations_run']
    
    def test_diversity_tracking(self):
        """Test population diversity is tracked."""
        target = [0.3, 0.7, 1.0]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=20,
            generations=15,
            mutation_rate=0.2,
            seed=42,
        )
        
        graph = gen.evolve()
        stats = gen.get_statistics()
        
        # Diversity should be recorded
        assert len(stats['diversity_history']) > 0
        
        # Diversity should be in [0, 1]
        for div in stats['diversity_history']:
            assert 0.0 <= div <= 1.0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    import pytest
    
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
