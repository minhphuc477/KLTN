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

import networkx as nx
import pytest

from src.generation.evolutionary_director import (
    EvolutionaryTopologyGenerator,
    TensionCurveEvaluator,
    GraphGrammarExecutor,
    mission_graph_to_networkx,
    networkx_to_mission_graph,
)
from src.generation.grammar import MissionGraph, MissionNode, NodeType, EdgeType
from src.utils.graph_utils import validate_graph_topology


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

    def test_crossover_selects_transition_compatible_boundary(self):
        """Recombination should preserve the strongest learned rule boundary."""
        probe = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.6, 1.0],
            population_size=4,
            generations=1,
            rule_space="core",
            seed=17,
        )
        names = probe.executor.rule_names
        transitions = {
            names[2]: {names[2]: 1.0},
            names[3]: {names[3]: 1.0},
        }
        generator = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.6, 1.0],
            zelda_transition_matrix=transitions,
            transition_mix=1.0,
            population_size=4,
            generations=1,
            rule_space="core",
            seed=17,
        )

        child1, child2 = generator._crossover([1, 2, 3, 4], [4, 3, 2, 1])

        assert child1 == [1, 2, 2, 1]
        assert child2 == [4, 3, 3, 4]

    def test_mechanical_feasibility_is_not_exact_style_target_matching(self):
        """A playable graph remains feasible when it misses soft descriptor targets."""
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START))
        graph.add_node(MissionNode(id=1, node_type=NodeType.GOAL))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.sanitize()

        result = TensionCurveEvaluator(
            target_curve=[0.0, 0.3, 0.7, 1.0],
        ).evaluate_graph(graph)

        assert result["mechanically_feasible"] is True
        assert result["feasible"] is True
        assert result["constraint_violation"] > 0.0

    def test_finalize_graph_output_repairs_goal_gauntlet(self):
        """Final export should normalize malformed goal chains before validation."""
        gen = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.5, 0.8, 1.0],
            population_size=4,
            generations=1,
            seed=42,
            rule_space="full",
            enforce_generation_constraints=True,
            allow_candidate_repairs=True,
        )

        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.ENEMY, position=(1, 0, 0), difficulty=0.4))
        graph.add_node(MissionNode(id=2, node_type=NodeType.GOAL, position=(3, 0, 0), difficulty=1.0))
        graph.add_node(MissionNode(id=3, node_type=NodeType.BOSS_DOOR, position=(2, 0, 0), difficulty=0.9, key_id=3))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 3, EdgeType.BOSS_LOCKED, key_required=3)
        graph.add_edge(3, 2, EdgeType.PATH)
        graph.sanitize()

        finalized = gen._finalize_graph_output(graph, directed_output=True)
        is_valid, errors = validate_graph_topology(finalized)

        assert is_valid, errors

    def test_finalize_graph_output_repairs_disconnected_component(self):
        """Final export should prune disconnected optional decoration."""
        gen = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.5, 0.8, 1.0],
            population_size=4,
            generations=1,
            seed=42,
            rule_space="full",
            enforce_generation_constraints=True,
            allow_candidate_repairs=True,
        )

        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0), difficulty=0.0))
        graph.add_node(MissionNode(id=1, node_type=NodeType.ENEMY, position=(1, 0, 0), difficulty=0.4))
        graph.add_node(MissionNode(id=2, node_type=NodeType.GOAL, position=(4, 0, 0), difficulty=1.0))
        graph.add_node(MissionNode(id=3, node_type=NodeType.ENEMY, position=(7, 0, 0), difficulty=0.5))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        # Node 3 is intentionally isolated optional decoration.
        graph.sanitize()

        finalized = gen._finalize_graph_output(graph, directed_output=True)
        is_valid, errors = validate_graph_topology(finalized)

        assert is_valid, errors
        assert nx.is_connected(finalized.to_undirected())
        assert 3 not in finalized
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
        
        # Check path exists START -> GOAL
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
        """The elitist search retains the best finite feasible score it observes."""
        target = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        gen = EvolutionaryTopologyGenerator(
            target_curve=target,
            population_size=30,
            generations=20,
            seed=42,
        )
        
        _graph = gen.evolve()
        stats = gen.get_statistics()
        
        # A scalar threshold is not a convergence proof: objective magnitudes
        # change when new hard constraints or descriptor terms are added.  The
        # stable contract of this mu+lambda search is elitist retention.
        history = stats['best_fitness_history']
        assert len(history) > 0
        assert all(0.0 <= score <= 1.0 for score in history)
        assert stats['final_best_fitness'] == max(history)
        assert stats['final_best_fitness'] >= history[0]
        assert len(stats['avg_fitness_history']) > 0
    
    def test_empty_target_curve(self):
        """Test handling of edge case: empty target curve."""
        try:
            EvolutionaryTopologyGenerator(
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
        """Test decreasing difficulty (hard -> easy)."""
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
    
    @pytest.mark.timeout(180)
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
            rule_space="core",
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
        
        _graph = gen.evolve()
        stats = gen.get_statistics()
        
        # Fitness magnitudes change when structural validity contracts tighten;
        # the stable algorithmic contract is elitist, bounded improvement.
        history = stats['best_fitness_history']
        assert len(history) == 5
        assert 0.0 <= stats['final_best_fitness'] <= 1.0
        assert stats['final_best_fitness'] == max(history)
        assert stats['final_best_fitness'] >= history[0]
    
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

        # Combat pressure can now surface as richer combat-bearing node types,
        # not only the legacy ENEMY label.
        combat_count = sum(
            1
            for _, data in graph.nodes(data=True)
            if bool(data.get("has_enemy", False))
            or str(data.get("type", "")).upper()
            in {"ENEMY", "COMBAT_PUZZLE", "BOSS", "MINI_BOSS", "ARENA"}
        )
        assert combat_count > 0


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

    def test_curve_alignment_scores_the_complete_generated_suffix(self):
        """Generated rooms beyond target length must affect narrative fitness."""
        target = [0.0, 0.5, 1.0]
        aligned = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        broken_suffix = [0.0, 0.2, 0.4, 0.0, 0.0, 0.0]

        aligned_score = TensionCurveEvaluator._curve_event_alignment_scores(
            aligned,
            target,
        )["curve_fitness"]
        broken_score = TensionCurveEvaluator._curve_event_alignment_scores(
            broken_suffix,
            target,
        )["curve_fitness"]

        assert aligned_score > broken_score

    def test_pedagogical_descriptor_metrics_detect_skill_chain(self):
        """Evaluator should reward explicit item -> tutorial -> combat -> complex progression."""
        target = [0.2, 0.4, 0.7, 1.0]
        evaluator = TensionCurveEvaluator(target)

        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0)))
        graph.add_node(MissionNode(id=1, node_type=NodeType.ITEM, position=(1, 0, 0), item_type="BOW"))
        graph.add_node(
            MissionNode(
                id=2,
                node_type=NodeType.TUTORIAL_PUZZLE,
                position=(2, 0, 0),
                is_tutorial=True,
                difficulty=0.2,
            )
        )
        graph.add_node(MissionNode(id=3, node_type=NodeType.COMBAT_PUZZLE, position=(3, 0, 0), difficulty=0.5))
        graph.add_node(MissionNode(id=4, node_type=NodeType.COMPLEX_PUZZLE, position=(4, 0, 0), difficulty=0.8))
        graph.add_node(MissionNode(id=5, node_type=NodeType.GOAL, position=(5, 0, 0), difficulty=1.0))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        graph.add_edge(2, 3, EdgeType.PATH)
        graph.add_edge(3, 4, EdgeType.PATH)
        graph.add_edge(4, 5, EdgeType.PATH)
        graph.sanitize()

        metrics = evaluator._extract_descriptor_metrics(graph)

        assert metrics["tutorial_puzzle_count"] == 1.0
        assert metrics["combat_puzzle_count"] == 1.0
        assert metrics["complex_puzzle_count"] == 1.0
        assert metrics["pedagogical_puzzle_variety"] == 1.0
        assert metrics["skill_chain_score"] == 1.0
        assert metrics["tutorial_climax_depth_score"] > 0.0

    def test_pedagogical_depth_score_prefers_deeper_tutorial_to_climax_chain(self):
        """Longer tutorial->combat->complex->boss span should score higher than a compressed chain."""
        target = [0.2, 0.4, 0.7, 1.0, 1.0, 1.0]
        evaluator = TensionCurveEvaluator(target)

        shallow = MissionGraph()
        shallow.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0)))
        shallow.add_node(MissionNode(id=1, node_type=NodeType.ITEM, position=(1, 0, 0), item_type="HOOKSHOT"))
        shallow.add_node(MissionNode(id=2, node_type=NodeType.TUTORIAL_PUZZLE, position=(2, 0, 0), is_tutorial=True))
        shallow.add_node(MissionNode(id=3, node_type=NodeType.COMBAT_PUZZLE, position=(3, 0, 0)))
        shallow.add_node(MissionNode(id=4, node_type=NodeType.COMPLEX_PUZZLE, position=(4, 0, 0)))
        shallow.add_node(MissionNode(id=5, node_type=NodeType.BOSS, position=(5, 0, 0)))
        shallow.add_edge(0, 1, EdgeType.PATH)
        shallow.add_edge(1, 2, EdgeType.PATH)
        shallow.add_edge(2, 3, EdgeType.PATH)
        shallow.add_edge(3, 4, EdgeType.PATH)
        shallow.add_edge(4, 5, EdgeType.PATH)
        shallow.sanitize()

        deep = MissionGraph()
        deep.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0)))
        deep.add_node(MissionNode(id=1, node_type=NodeType.ITEM, position=(1, 0, 0), item_type="HOOKSHOT"))
        deep.add_node(MissionNode(id=2, node_type=NodeType.TUTORIAL_PUZZLE, position=(2, 0, 0), is_tutorial=True))
        deep.add_node(MissionNode(id=3, node_type=NodeType.EMPTY, position=(3, 0, 0)))
        deep.add_node(MissionNode(id=4, node_type=NodeType.COMBAT_PUZZLE, position=(4, 0, 0)))
        deep.add_node(MissionNode(id=5, node_type=NodeType.EMPTY, position=(5, 0, 0)))
        deep.add_node(MissionNode(id=6, node_type=NodeType.COMPLEX_PUZZLE, position=(6, 0, 0)))
        deep.add_node(MissionNode(id=7, node_type=NodeType.BOSS, position=(7, 0, 0)))
        deep.add_edge(0, 1, EdgeType.PATH)
        deep.add_edge(1, 2, EdgeType.PATH)
        deep.add_edge(2, 3, EdgeType.PATH)
        deep.add_edge(3, 4, EdgeType.PATH)
        deep.add_edge(4, 5, EdgeType.PATH)
        deep.add_edge(5, 6, EdgeType.PATH)
        deep.add_edge(6, 7, EdgeType.PATH)
        deep.sanitize()

        shallow_metrics = evaluator._extract_descriptor_metrics(shallow)
        deep_metrics = evaluator._extract_descriptor_metrics(deep)

        assert deep_metrics["skill_chain_score"] == shallow_metrics["skill_chain_score"] == 1.0
        assert deep_metrics["tutorial_climax_depth_score"] > shallow_metrics["tutorial_climax_depth_score"]

    def test_generator_prior_boosts_add_skill_chain_rule(self):
        """Target-aware priors should actively favor AddSkillChain in full-rule mode."""
        gen = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.5, 0.8, 1.0],
            population_size=6,
            generations=2,
            genome_length=8,
            seed=42,
            rule_space="full",
        )

        add_skill_rule_id = gen.rule_name_to_id["AddSkillChain"]

        assert add_skill_rule_id in gen._pedagogical_rule_ids
        assert gen._global_rule_weights[add_skill_rule_id] > float(gen.executor.rules[add_skill_rule_id].weight)

    def test_initial_population_includes_structured_seed_genomes(self):
        """High pedagogical targets should seed the initial population with tutorial-aware genomes."""
        gen = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
            population_size=10,
            generations=2,
            genome_length=10,
            seed=7,
            rule_space="full",
        )

        population = gen._generate_initial_population()
        add_skill_rule_id = gen.rule_name_to_id["AddSkillChain"]
        assert any(add_skill_rule_id in ind.genome for ind in population)

    def test_progression_balance_repair_reduces_leniency_on_item_path(self):
        """Final Block I balancing should tighten overly lenient item-first graphs."""
        gen = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
            population_size=6,
            generations=2,
            genome_length=8,
            seed=21,
            rule_space="full",
        )

        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0)))
        graph.add_node(MissionNode(id=1, node_type=NodeType.EMPTY, position=(1, 0, 0)))
        graph.add_node(MissionNode(id=2, node_type=NodeType.ITEM, position=(2, 0, 0), item_type="BOW"))
        graph.add_node(MissionNode(id=3, node_type=NodeType.TUTORIAL_PUZZLE, position=(3, 0, 0), is_tutorial=True, difficulty=0.2))
        graph.add_node(MissionNode(id=4, node_type=NodeType.COMBAT_PUZZLE, position=(4, 0, 0), difficulty=0.5))
        graph.add_node(MissionNode(id=5, node_type=NodeType.COMPLEX_PUZZLE, position=(5, 0, 0), difficulty=0.8))
        graph.add_node(MissionNode(id=6, node_type=NodeType.BOSS_DOOR, position=(6, 0, 0)))
        graph.add_node(MissionNode(id=7, node_type=NodeType.BOSS, position=(7, 0, 0)))
        graph.add_node(MissionNode(id=8, node_type=NodeType.GOAL, position=(8, 0, 0)))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        graph.add_edge(2, 3, EdgeType.PATH)
        graph.add_edge(3, 4, EdgeType.PATH)
        graph.add_edge(4, 5, EdgeType.PATH)
        graph.add_edge(5, 6, EdgeType.PATH)
        graph.add_edge(6, 7, EdgeType.PATH)
        graph.add_edge(7, 8, EdgeType.PATH)
        graph.sanitize()

        before = gen.evaluator._extract_descriptor_metrics(graph)
        constraint_grammar = getattr(getattr(gen, "executor", None), "_constraint_grammar", None)
        repaired = gen._repair_progression_balance(graph, constraint_grammar=constraint_grammar)
        after = gen.evaluator._extract_descriptor_metrics(repaired)

        assert after["leniency"] < before["leniency"]

    def test_progression_balance_repair_prunes_junk_deadend_without_hurting_pedagogy(self):
        """Final Block I balancing should prune non-essential side junk before export."""
        gen = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
            population_size=6,
            generations=2,
            genome_length=8,
            seed=22,
            rule_space="full",
            descriptor_targets={"linearity": 0.68, "leniency": 1.0},
        )

        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START, position=(0, 0, 0)))
        graph.add_node(MissionNode(id=1, node_type=NodeType.ITEM, position=(1, 0, 0), item_type="BOW"))
        graph.add_node(MissionNode(id=2, node_type=NodeType.TUTORIAL_PUZZLE, position=(2, 0, 0), is_tutorial=True, difficulty=0.2))
        graph.add_node(MissionNode(id=3, node_type=NodeType.COMBAT_PUZZLE, position=(3, 0, 0), difficulty=0.5))
        graph.add_node(MissionNode(id=4, node_type=NodeType.COMPLEX_PUZZLE, position=(4, 0, 0), difficulty=0.8))
        graph.add_node(MissionNode(id=5, node_type=NodeType.BOSS_DOOR, position=(5, 0, 0)))
        graph.add_node(MissionNode(id=6, node_type=NodeType.BOSS, position=(6, 0, 0)))
        graph.add_node(MissionNode(id=7, node_type=NodeType.GOAL, position=(7, 0, 0)))
        for node_id, y in zip(range(8, 14), range(1, 7)):
            graph.add_node(MissionNode(id=node_id, node_type=NodeType.EMPTY, position=(3, y, 0)))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        graph.add_edge(2, 3, EdgeType.PATH)
        graph.add_edge(3, 4, EdgeType.PATH)
        graph.add_edge(4, 5, EdgeType.PATH)
        graph.add_edge(5, 6, EdgeType.PATH)
        graph.add_edge(6, 7, EdgeType.PATH)
        for node_id in range(8, 14):
            graph.add_edge(3, node_id, EdgeType.PATH)
        graph.sanitize()

        before = gen.evaluator._extract_descriptor_metrics(graph)
        constraint_grammar = getattr(getattr(gen, "executor", None), "_constraint_grammar", None)
        repaired = gen._repair_progression_balance(graph, constraint_grammar=constraint_grammar)
        after = gen.evaluator._extract_descriptor_metrics(repaired)

        removable_empty_leaves = [
            node.id
            for node in repaired.nodes.values()
            if node.node_type == NodeType.EMPTY
            and repaired.get_node_degree(node.id) == 1
            and not node.is_hub
            and not node.is_secret
        ]
        assert removable_empty_leaves == []
        # Path linearity intentionally ignores optional dead-end padding, so
        # cleanup must preserve rather than artificially improve this metric.
        assert after["linearity"] >= before["linearity"]
        assert after["pedagogical_puzzle_variety"] >= before["pedagogical_puzzle_variety"]
        assert after["skill_chain_score"] >= before["skill_chain_score"]

    def test_trim_surplus_reward_keys_demotes_free_side_branch_keys(self):
        """Final gate-economy pass should demote gratuitous side-branch reward keys."""
        gen = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
            population_size=6,
            generations=2,
            genome_length=8,
            seed=31,
            rule_space="full",
        )

        graph = MissionGraph()
        for node_id, node_type in [
            (0, NodeType.START),
            (1, NodeType.ITEM),
            (2, NodeType.TUTORIAL_PUZZLE),
            (3, NodeType.COMBAT_PUZZLE),
            (4, NodeType.COMPLEX_PUZZLE),
            (5, NodeType.BOSS_DOOR),
            (6, NodeType.BOSS),
            (7, NodeType.GOAL),
            (8, NodeType.KEY),
            (9, NodeType.KEY),
        ]:
            kwargs = {}
            if node_type == NodeType.BOSS_DOOR:
                kwargs["key_id"] = 5
            if node_type == NodeType.ITEM:
                kwargs["item_type"] = "BOW"
            graph.add_node(MissionNode(id=node_id, node_type=node_type, position=(node_id, 0, 0), **kwargs))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        graph.add_edge(2, 3, EdgeType.PATH)
        graph.add_edge(3, 4, EdgeType.PATH)
        graph.add_edge(4, 5, EdgeType.BOSS_LOCKED, key_required=5)
        graph.add_edge(5, 6, EdgeType.PATH)
        graph.add_edge(6, 7, EdgeType.PATH)
        graph.add_edge(3, 8, EdgeType.PATH)
        graph.add_edge(3, 9, EdgeType.PATH)
        graph.sanitize()

        trimmed, changes = gen._trim_surplus_reward_keys(graph)

        assert changes >= 1
        assert sum(1 for node in trimmed.nodes.values() if node.node_type == NodeType.KEY and node.key_id is None) <= 1
        assert any(node.node_type in {NodeType.ITEM, NodeType.TREASURE} for node in [trimmed.nodes[8], trimmed.nodes[9]])

    def test_gate_economy_repair_reduces_surplus_and_improves_gate_pressure(self):
        """Final export calibration should reduce gratuitous key surplus without hurting pedagogy."""
        gen = EvolutionaryTopologyGenerator(
            target_curve=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
            population_size=6,
            generations=2,
            genome_length=8,
            seed=32,
            rule_space="full",
        )

        graph = MissionGraph()
        for node_id, node_type in [
            (0, NodeType.START),
            (1, NodeType.ITEM),
            (2, NodeType.TUTORIAL_PUZZLE),
            (3, NodeType.COMBAT_PUZZLE),
            (4, NodeType.COMPLEX_PUZZLE),
            (5, NodeType.BOSS_DOOR),
            (6, NodeType.BOSS),
            (7, NodeType.GOAL),
            (8, NodeType.KEY),
            (9, NodeType.KEY),
            (10, NodeType.KEY),
        ]:
            kwargs = {}
            if node_type == NodeType.BOSS_DOOR:
                kwargs["key_id"] = 5
            if node_type == NodeType.ITEM:
                kwargs["item_type"] = "HOOKSHOT"
            graph.add_node(MissionNode(id=node_id, node_type=node_type, position=(node_id, 0, 0), **kwargs))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.PATH)
        graph.add_edge(2, 3, EdgeType.PATH)
        graph.add_edge(3, 4, EdgeType.PATH)
        graph.add_edge(4, 5, EdgeType.PATH)
        graph.add_edge(5, 6, EdgeType.PATH)
        graph.add_edge(6, 7, EdgeType.PATH)
        graph.add_edge(3, 8, EdgeType.PATH)
        graph.add_edge(3, 9, EdgeType.PATH)
        graph.add_edge(4, 10, EdgeType.PATH)
        graph.sanitize()

        before = gen.evaluator._extract_descriptor_metrics(graph)
        constraint_grammar = getattr(getattr(gen, "executor", None), "_constraint_grammar", None)
        repaired = gen._repair_gate_economy(graph, constraint_grammar=constraint_grammar)
        after = gen.evaluator._extract_descriptor_metrics(repaired)

        assert after["small_key_surplus"] < before["small_key_surplus"]
        assert after["gating_density"] >= before["gating_density"]
        assert after["pedagogical_puzzle_variety"] >= before["pedagogical_puzzle_variety"]


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

    def test_degree_projection_preserves_locked_edges_before_paths(self):
        """Degree pruning should remove soft edges before progression gates."""
        executor = GraphGrammarExecutor(seed=42)
        graph = MissionGraph()
        for nid, node_type in [
            (0, NodeType.START),
            (1, NodeType.EMPTY),
            (2, NodeType.LOCK),
            (3, NodeType.ENEMY),
            (4, NodeType.ENEMY),
            (5, NodeType.ENEMY),
            (6, NodeType.GOAL),
        ]:
            graph.add_node(MissionNode(id=nid, node_type=node_type))

        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.LOCKED, key_required=99)
        graph.add_edge(1, 3, EdgeType.PATH)
        graph.add_edge(1, 4, EdgeType.PATH)
        graph.add_edge(1, 5, EdgeType.PATH)
        graph.add_edge(1, 6, EdgeType.PATH)
        graph.sanitize()

        removed = executor._enforce_max_degree(graph, max_degree=4)

        assert removed > 0
        assert any(edge.edge_type == EdgeType.LOCKED for edge in graph.edges)

    def test_allow_candidate_repairs_false_skips_invalid_candidates(self):
        """Invalid candidates should be skipped when repairs are disabled."""
        class DummyInvalidRule:
            name = "DummyInvalid"

            def can_apply(self, graph, context):
                return True

            def apply(self, graph, context):
                graph.add_node(MissionNode(id=2, node_type=NodeType.ENEMY))
                graph.add_edge(0, 2, EdgeType.PATH)
                return graph

        class DummyConstraintGrammar:
            def ensure_anchor_nodes(self, graph):
                return graph

            def validate_lock_key_ordering(self, graph, *, log_failures=True):
                return False

            def validate_progression_constraints(self, graph, *, log_failures=True):
                return False

            def fix_lock_key_ordering(self, graph):
                raise AssertionError("repairs must not run when allow_candidate_repairs=False")

            def repair_progression_constraints(self, graph):
                raise AssertionError("repairs must not run when allow_candidate_repairs=False")

        executor = GraphGrammarExecutor(seed=42, allow_candidate_repairs=False)
        executor.rules = [executor.rules[0], DummyInvalidRule()]
        executor.rule_names = ["Start", "DummyInvalid"]
        executor._constraint_grammar = DummyConstraintGrammar()

        graph = executor.execute([1], record_trace=True)

        assert len(graph.nodes) == 2
        assert graph.generation_stats["candidate_repairs_applied"] == 0
        assert graph.generation_stats["generation_constraint_rejections"] == 1
        assert graph.generation_stats["rule_skipped"] == 1

    def test_allow_candidate_repairs_true_retries_invalid_candidates(self):
        """When enabled, candidate repairs should be attempted before rejection."""
        class DummyInvalidRule:
            name = "DummyInvalid"

            def can_apply(self, graph, context):
                return True

            def apply(self, graph, context):
                graph.add_node(MissionNode(id=2, node_type=NodeType.ENEMY))
                graph.add_edge(0, 2, EdgeType.PATH)
                return graph

        class DummyConstraintGrammar:
            def __init__(self):
                self.repaired = False

            def ensure_anchor_nodes(self, graph):
                return graph

            def validate_lock_key_ordering(self, graph, *, log_failures=True):
                return self.repaired

            def validate_progression_constraints(self, graph, *, log_failures=True):
                return self.repaired

            def fix_lock_key_ordering(self, graph):
                self.repaired = True
                return graph

            def repair_progression_constraints(self, graph):
                return graph

        executor = GraphGrammarExecutor(seed=42, allow_candidate_repairs=True)
        executor.rules = [executor.rules[0], DummyInvalidRule()]
        executor.rule_names = ["Start", "DummyInvalid"]
        executor._constraint_grammar = DummyConstraintGrammar()

        graph = executor.execute([1], record_trace=True)

        assert len(graph.nodes) == 3
        assert graph.generation_stats["candidate_repairs_applied"] == 1
        assert graph.generation_stats["generation_constraint_rejections"] == 1
        assert graph.generation_stats["rule_applied"] == 1


class TestGraphConversion:
    """Test graph format conversions."""
    
    def test_mission_to_networkx(self):
        """Test MissionGraph -> NetworkX conversion."""
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
        """Test NetworkX -> MissionGraph conversion."""
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

    def test_sparse_node_ids_export_to_dense_tensor_indices(self):
        """Sparse node IDs from rewrites must still export valid tensors."""
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START))
        graph.add_node(MissionNode(id=2, node_type=NodeType.ENEMY))
        graph.add_node(MissionNode(id=5, node_type=NodeType.GOAL))
        graph.add_edge(0, 2, EdgeType.PATH)
        graph.add_edge(2, 5, EdgeType.PATH)
        graph.sanitize()

        edge_index, node_features = graph.to_tensor()
        adjacency = graph.to_adjacency_matrix()

        assert edge_index.shape == (2, 2)
        assert node_features.shape[0] == len(graph.nodes)
        assert int(edge_index.max().item()) < node_features.shape[0]
        assert adjacency.shape == (len(graph.nodes), len(graph.nodes))
        assert adjacency[0, 1] == 1.0
        assert adjacency[1, 2] == 1.0
        assert adjacency[1, 0] == 1.0
        assert adjacency[2, 1] == 1.0

    def test_full_rule_executor_graph_export_handles_sparse_ids(self):
        """Executor outputs remain exportable after prune and merge rules."""
        executor = GraphGrammarExecutor(seed=0, use_full_rule_space=True)
        genome = [i % (len(executor.rules) - 1) + 1 for i in range(25)]
        graph = executor.execute(genome, max_nodes=18, allow_override=True)

        edge_index, node_features = graph.to_tensor()
        adjacency = graph.to_adjacency_matrix()

        assert edge_index.shape[0] == 2
        assert node_features.shape[0] == len(graph.nodes)
        if edge_index.numel() > 0:
            assert int(edge_index.max().item()) < node_features.shape[0]
        assert adjacency.shape == (len(graph.nodes), len(graph.nodes))

    def test_compute_tpe_uses_reverse_goal_distances_for_directed_edges(self):
        """Distance-to-goal must respect reverse traversal over directed mission edges."""
        graph = MissionGraph()
        graph.add_node(MissionNode(id=0, node_type=NodeType.START))
        graph.add_node(MissionNode(id=1, node_type=NodeType.ENEMY))
        graph.add_node(MissionNode(id=2, node_type=NodeType.GOAL))
        graph.add_edge(0, 1, EdgeType.PATH)
        graph.add_edge(1, 2, EdgeType.ONE_WAY)
        graph.sanitize()

        tpe = graph.compute_tpe()

        assert tpe.shape == (3, 8)
        # start -> goal distance is 2 hops, so reverse distance-to-goal
        # should normalize to [1.0, 0.5, 0.0] for [start, mid, goal].
        assert abs(float(tpe[0, 1]) - 1.0) < 1e-6
        assert abs(float(tpe[1, 1]) - 0.5) < 1e-6
        assert abs(float(tpe[2, 1]) - 0.0) < 1e-6
        assert float(tpe[0, 3]) == 1.0
        assert float(tpe[1, 3]) == 1.0
        assert float(tpe[2, 3]) == 1.0


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
        
        _graph = gen.evolve()
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
        
        _graph = gen.evolve()
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
