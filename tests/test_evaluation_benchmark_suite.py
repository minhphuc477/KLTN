import networkx as nx

from src.evaluation.benchmark_suite import (
    _path_linearity,
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
    assert d.key_gate_count >= 1
    assert d.key_before_lock_rate == 1.0
    assert d.path_redundancy == 0.0
    assert d.branch_count == 0
    assert d.branch_utility_rate == 1.0
    assert d.repair_applied is False
    assert d.total_repairs == 0


def test_linearity_ignores_dead_end_padding_but_detects_alternate_routes():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    path = [0, 1, 2, 3]
    baseline = _path_linearity(graph, path)

    for node in range(4, 14):
        graph.add_edge(1, node)
    padded = _path_linearity(graph, path)

    graph.add_edges_from([(1, 14), (14, 15), (15, 2)])
    alternate_route = _path_linearity(graph, path)

    assert baseline == 1.0
    assert padded == baseline
    assert alternate_route < padded


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
    assert "key_before_lock_rate" in summary.completeness
    assert "repair_rate" in summary.robustness
    assert "coverage_linearity_leniency" in summary.expressive_range
    assert "coverage_redundancy_articulation" in summary.expressive_range
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


def test_extract_graph_descriptor_tracks_switch_and_secret_semantics():
    grammar = MissionGrammar(seed=99)
    G = nx.Graph()
    G.add_node(0, label="s", type="START")
    G.add_node(1, label="", type="EMPTY")
    G.add_node(2, label="t", type="GOAL")
    G.add_node(3, label="S1", type="SWITCH", switch_id=3)
    G.add_node(4, label="i", type="ITEM", is_secret=True, item_type="MAP")

    G.add_edge(0, 1, edge_type="open")
    G.add_edge(1, 2, edge_type="state_block", switches_required=[3], battery_id=7)
    G.add_edge(1, 3, edge_type="open")
    G.add_edge(1, 4, edge_type="hidden")

    d = extract_graph_descriptor(G, grammar=grammar)

    assert d.switch_gate_count == 1
    assert d.switch_before_gate_rate == 1.0
    assert d.battery_gate_count == 1
    assert d.battery_satisfaction_rate == 1.0
    assert d.branch_count == 2
    assert d.branch_utility_rate == 1.0
    assert d.secret_component_count == 1
    assert d.secret_content_discoverability_rate == 1.0
    assert d.articulation_count >= 1


def test_extract_graph_descriptor_flags_broken_key_gate_and_pointless_branch():
    grammar = MissionGrammar(seed=5)
    G = nx.Graph()
    G.add_node(0, label="s", type="START")
    G.add_node(1, label="", type="EMPTY")
    G.add_node(2, label="t", type="GOAL")
    G.add_node(3, label="", type="EMPTY")

    G.add_edge(0, 1, edge_type="open")
    G.add_edge(1, 2, edge_type="key_locked", key_required=99)
    G.add_edge(1, 3, edge_type="open")

    d = extract_graph_descriptor(G, grammar=grammar)

    assert d.key_gate_count >= 1
    assert d.key_before_lock_rate == 0.0
    assert d.branch_count == 1
    assert d.branch_utility_rate == 0.0
    assert d.secret_component_count == 0
    assert d.secret_content_discoverability_rate == 1.0
