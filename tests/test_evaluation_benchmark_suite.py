import networkx as nx

from src.evaluation.benchmark_suite import (
    extract_graph_descriptor,
    run_block_i_benchmark,
)
from src.generation.grammar import MissionGrammar


def _make_simple_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node(0, label="s", type="START")
    G.add_node(1, label="k", type="KEY")
    G.add_node(2, label="t", type="GOAL")
    G.add_edge(0, 1, edge_type="open")
    G.add_edge(1, 2, edge_type="key_locked")
    return G


def test_extract_graph_descriptor_basic():
    grammar = MissionGrammar(seed=123)
    G = _make_simple_graph()
    d = extract_graph_descriptor(G, grammar=grammar)

    assert d.has_start is True
    assert d.has_goal is True
    assert d.path_exists is True
    assert d.path_length >= 1
    assert 0.0 <= d.linearity <= 1.0
    assert 0.0 <= d.leniency <= 1.0
    assert d.key_count >= 1
    assert d.lock_count >= 1
    assert d.repair_applied is False
    assert d.total_repairs == 0


def test_run_block_i_benchmark_shapes():
    G1 = _make_simple_graph()
    G2 = _make_simple_graph()

    summary = run_block_i_benchmark(
        generated_graphs=[G1],
        reference_graphs=[G2],
        generation_times=[0.1],
    )

    assert summary.num_generated == 1
    assert summary.num_reference == 1
    assert "overall_completeness" in summary.completeness
    assert "repair_rate" in summary.robustness
    assert "coverage_linearity_leniency" in summary.expressive_range
    assert "novelty_vs_reference" in summary.reference_comparison


def test_extract_graph_descriptor_reads_generation_stats():
    grammar = MissionGrammar(seed=7)
    G = _make_simple_graph()
    G.graph["generation_stats"] = {
        "repair_applied": True,
        "total_repairs": 3,
        "lock_key_repairs": 1,
        "progression_repairs": 1,
        "wave3_repairs": 1,
        "repair_rounds": 2,
    }
    d = extract_graph_descriptor(G, grammar=grammar)
    assert d.repair_applied is True
    assert d.total_repairs == 3
    assert d.repair_rounds == 2


def test_run_block_i_benchmark_aggregates_wfc_probe_metrics():
    G = _make_simple_graph()
    summary = run_block_i_benchmark(
        generated_graphs=[G],
        reference_graphs=[G],
        generation_times=[0.1],
        wfc_probe_results=[
            {
                "contradictions": 2,
                "backtracks": 4,
                "restarts": 1,
                "zero_prob_resets": 3,
                "fallback_fills": 2,
                "required_fallback": True,
                "kl_divergence": 1.2,
                "distribution_preserved": True,
            }
        ],
    )
    assert summary.robustness["wfc_probe_count"] == 1.0
    assert summary.robustness["wfc_mean_contradictions"] == 2.0
    assert summary.robustness["wfc_restart_rate"] == 1.0
