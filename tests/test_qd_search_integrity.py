"""Synthetic checks for QD archive evidence and descriptor integrity."""

import networkx as nx
import numpy as np
import pytest

from src.evaluation.map_elites import CBSFeatureExtractor, CVTEliteArchive, EliteArchive, LinearityLeniencyExtractor


@pytest.mark.parametrize("kind", ["grid", "cvt"])
def test_archive_preserves_evaluated_candidate_when_emitters_mutate(kind):
    archive = (EliteArchive(seed=3) if kind == "grid" else
               CVTEliteArchive(num_cells=4, num_cvt_samples=40, seed=3))
    graph = nx.path_graph(3)
    graph.nodes[0]["inventory"] = ["key"]
    metadata = {"witness": [0, 1, 2]}
    assert archive.add(graph, 0.7, (0.3, 0.4), metadata)
    graph.nodes[0]["inventory"].clear()
    graph.remove_edge(1, 2)
    metadata["witness"].clear()
    parent = archive.get_random_elite()
    parent.solution.remove_node(1)
    parent.metadata["witness"].clear()
    parent.fitness = 100.0
    stored = archive.get_all_elites()[0]
    assert stored.solution.has_edge(1, 2)
    assert stored.solution.nodes[0]["inventory"] == ["key"]
    assert stored.metadata["witness"] == [0, 1, 2]
    assert archive.get_stats().total_fitness == 0.7


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan"])
def test_cvt_cannot_report_coverage_against_nonexistent_cells(monkeypatch, defect):
    import scipy.cluster.vq
    def broken_kmeans(samples, initial):
        centroids = initial.copy()
        if defect == "missing":
            centroids = centroids[:-1]
        elif defect == "duplicate":
            centroids[0] = centroids[1]
        else:
            centroids[0, 0] = np.nan
        return centroids, 0.0
    monkeypatch.setattr(scipy.cluster.vq, "kmeans", broken_kmeans)
    with pytest.raises(RuntimeError, match="declared number"):
        CVTEliteArchive(num_cells=4, num_cvt_samples=40, seed=3)


@pytest.mark.parametrize("defect", ["error", "nan", "missing"])
def test_failed_descriptors_are_not_fabricated_or_cached(monkeypatch, defect):
    import src.evaluation.cbs_fitness as cbs
    def compute(graph, **kwargs):
        if defect == "error":
            raise RuntimeError("descriptor failed")
        return {} if defect == "missing" else {"confusion_ratio": np.nan, "room_entropy": 0.5}
    monkeypatch.setattr(cbs, "compute_cbs_fitness", compute)
    extractor = CBSFeatureExtractor()
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        extractor.extract(nx.path_graph(3))
    assert not extractor._cache


def test_descriptor_cache_changes_with_global_rules_and_persona(monkeypatch):
    import src.evaluation.cbs_fitness as cbs
    def compute(graph, persona):
        return {"confusion_ratio": graph.graph["rule_value"],
                "room_entropy": 0.2 if persona == "balanced" else 0.8}
    monkeypatch.setattr(cbs, "compute_cbs_fitness", compute)
    graph = nx.path_graph(3)
    graph.graph["rule_value"] = 1
    extractor = CBSFeatureExtractor()
    first = extractor.extract(graph)
    graph.graph["rule_value"] = 2
    second = extractor.extract(graph)
    extractor.persona = "explorer"
    third = extractor.extract(graph)
    assert first != second != third


def test_leniency_counts_both_resources_in_explicit_composite_gate():
    graph = nx.DiGraph()
    graph.add_node(0, type="KEY")
    graph.add_node(1, type="GOAL")
    graph.add_edge(0, 1, edge_type="open", edge_constraints=["key_locked", "boss_locked"])
    assert LinearityLeniencyExtractor()._compute_leniency(graph) == 0.5


def test_leniency_counts_one_physical_gate_for_implied_reverse_arc():
    graph = nx.DiGraph()
    graph.add_node(0, type="KEY")
    graph.add_node(1, type="GOAL")
    graph.add_edge(0, 1, edge_type="key_locked")
    graph.add_edge(1, 0, edge_type="key_locked", metadata={"implied_reverse": True})
    assert LinearityLeniencyExtractor()._compute_leniency(graph) == 1.0


def test_benchmark_resource_pressure_matches_compound_gate_and_reverse_identity():
    from src.evaluation.benchmark_suite import extract_graph_descriptor
    graph = nx.DiGraph()
    graph.add_node(0, type="KEY", is_start=True)
    graph.add_node(1, type="GOAL", is_goal=True)
    attrs = {"edge_type": "open", "edge_constraints": ["key_locked", "boss_locked"]}
    graph.add_edge(0, 1, **attrs)
    graph.add_edge(1, 0, **attrs, metadata={"implied_reverse": True})
    assert extract_graph_descriptor(graph).leniency == 0.5


def test_runtime_archive_gating_descriptor_reads_explicit_constraints():
    from src.simulation.map_elites import MAPElitesEvaluator
    graph = nx.DiGraph()
    graph.add_edge(0, 1, edge_type="open", edge_constraints=["key_locked"])
    evaluator = MAPElitesEvaluator(enable_advanced_archive=False)
    assert evaluator._extract_graph_topology_metrics(graph)["gating_density"] == 1.0


@pytest.mark.parametrize("strategy", ["ga", "cvt_emitter"])
@pytest.mark.parametrize("budget", [1, 5, 8, 11])
def test_topology_search_uses_exact_candidate_budget_including_initialization(monkeypatch, strategy, budget):
    import copy
    from src.generation.evolutionary_director import EvolutionaryTopologyGenerator, Individual
    generator = EvolutionaryTopologyGenerator(
        [0.2, 0.8], population_size=4, generations=1, rule_space="core",
        evaluation_budget=budget, search_strategy=strategy, seed=3,
    )
    mission = generator.executor.execute([1], difficulty=0.5, max_nodes=8)
    assert mission.is_graph_connected()
    calls = []
    def evaluate(graph):
        calls.append(graph)
        # High fitness would have caused premature convergence in legacy GA.
        return {"fitness": 1.0, "feasible": True, "constraint_violation": 0.0,
                "descriptor_metrics": {"linearity": 0.5}}
    monkeypatch.setattr(generator.evaluator, "evaluate_graph", evaluate)
    monkeypatch.setattr(generator.executor, "execute", lambda *args, **kwargs: copy.deepcopy(mission))
    monkeypatch.setattr(generator, "_generate_initial_population", lambda: [Individual([1]) for _ in range(4)])
    monkeypatch.setattr(generator, "_generate_offspring", lambda population: [Individual([1]) for _ in range(4)])
    monkeypatch.setattr(generator, "_adapt_global_rule_prior_from_population", lambda population: None)
    monkeypatch.setattr(generator, "_finalize_graph_output", lambda *args, **kwargs: nx.path_graph(3))
    monkeypatch.setattr(generator, "_new_qd_archive", lambda: CVTEliteArchive(
        num_cells=4, feature_dims=4, num_cvt_samples=40, seed=3))
    assert generator.evolve().number_of_nodes() == 3
    assert len(calls) == budget
    assert generator.get_statistics()["candidate_evaluations"] == budget


@pytest.mark.parametrize("budget", [True, 1.5, 0, -1])
def test_topology_candidate_budget_rejects_lossy_values(budget):
    from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
    with pytest.raises(ValueError, match="positive integer"):
        EvolutionaryTopologyGenerator([0.2, 0.8], evaluation_budget=budget)
