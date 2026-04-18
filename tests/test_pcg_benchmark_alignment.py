import random
import sys

import networkx as nx
import pytest

from src.evaluation.pcg_benchmark_alignment import (
    PCG_BENCHMARK_ZELDA_VARIANTS,
    _control_error,
    _pcg_benchmark_repo_candidates,
    import_pcg_benchmark,
    map_graph_to_pcg_benchmark_zelda,
    select_pcg_benchmark_zelda_problem,
)
from scripts.run_ood_scaling_and_blinded_eval import _build_ood_summary_row
from scripts.run_pcg_benchmark_alignment import _build_external_summary, _problem_room_budget
from src.evaluation.benchmark_suite import run_block_i_benchmark


def _make_graph_for_alignment() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node(0, label="s", type="START", position=(0, 0))
    G.add_node(1, label="", type="EMPTY", position=(0, 1))
    G.add_node(2, label="k", type="KEY", position=(0, 2))
    G.add_node(3, label="t", type="GOAL", position=(1, 2))
    G.add_node(4, label="e", type="ENEMY", position=(1, 1), enemy_count_hint=1)
    G.add_edge(0, 1, edge_type="open")
    G.add_edge(1, 2, edge_type="open")
    G.add_edge(2, 3, edge_type="open")
    G.add_edge(1, 4, edge_type="open")
    return G


def _make_large_graph_triggering_control_fallback() -> nx.DiGraph:
    rng = random.Random(0)
    for trial in range(4):
        graph = nx.DiGraph()
        num_nodes = 24
        for idx in range(num_nodes):
            label = ""
            node_type = "EMPTY"
            if idx == 0:
                label = "s"
                node_type = "START"
            elif idx == num_nodes // 2:
                label = "k"
                node_type = "KEY"
            elif idx == num_nodes - 1:
                label = "t"
                node_type = "GOAL"
            elif idx % 7 == 0:
                label = "e"
                node_type = "ENEMY"
            graph.add_node(idx, label=label, type=node_type, position=(rng.randint(0, 20), rng.randint(0, 20)))

        for idx in range(num_nodes - 1):
            graph.add_edge(idx, idx + 1, edge_type="open")
        for _ in range(50):
            src = rng.randrange(num_nodes)
            dst = rng.randrange(num_nodes)
            if src != dst:
                graph.add_edge(src, dst, edge_type="open")

        mapping = map_graph_to_pcg_benchmark_zelda(
            graph,
            problem_name="zelda-large-v0",
            enemy_target=8,
            seed=trial,
        )
        if mapping.metadata.get("control_fallback_applied"):
            return graph

    raise AssertionError("Expected to find a deterministic large-variant fallback graph.")


def test_map_graph_to_pcg_benchmark_zelda_produces_valid_content():
    graph = _make_graph_for_alignment()
    mapping = map_graph_to_pcg_benchmark_zelda(
        graph,
        problem_name="zelda-v0",
        enemy_target=3,
        seed=17,
    )

    content = mapping.content
    assert content.shape == (7, 11)
    assert int((content == 2).sum()) == 1
    assert int((content == 3).sum()) == 1
    assert int((content == 4).sum()) == 1
    assert int((content == 5).sum()) == 3
    variant = PCG_BENCHMARK_ZELDA_VARIANTS["zelda-v0"]
    assert mapping.graph_control["player_key"] >= variant.control_min
    assert mapping.graph_control["key_door"] >= variant.control_min
    assert mapping.content_control["player_key"] == mapping.graph_control["player_key"]
    assert mapping.content_control["key_door"] == mapping.graph_control["key_door"]
    assert mapping.content_control["player_key"] + mapping.content_control["key_door"] >= variant.solution_length
    assert mapping.metadata["graph_control_raw"]["player_key"] > 0
    assert mapping.metadata["graph_control_raw"]["key_door"] > 0
    assert mapping.metadata["semantic_valid"] is True


def test_map_graph_to_pcg_benchmark_zelda_rejects_missing_explicit_key_semantics():
    graph = nx.DiGraph()
    graph.add_node(0, label="s", type="START", position=(0, 0))
    graph.add_node(1, label="", type="EMPTY", position=(0, 1))
    graph.add_node(2, label="t", type="GOAL", position=(0, 2))
    graph.add_edge(0, 1, edge_type="open")
    graph.add_edge(1, 2, edge_type="open")

    mapping = map_graph_to_pcg_benchmark_zelda(graph, problem_name="zelda-v0", seed=11)

    assert mapping.metadata["semantic_valid"] is False
    assert mapping.metadata["semantic_error"] == "missing_explicit_key"
    assert mapping.metadata["mapper_mode"] == "invalid_semantics"
    assert mapping.graph_control == {"player_key": 0, "key_door": 0}
    assert mapping.content_control == {"player_key": 0, "key_door": 0}
    assert int((mapping.content == 2).sum()) == 0
    assert int((mapping.content == 3).sum()) == 0
    assert int((mapping.content == 4).sum()) == 0


def test_large_variant_applies_control_fallback_when_free_routing_collapses_key_door_budget():
    graph = _make_large_graph_triggering_control_fallback()
    mapping = map_graph_to_pcg_benchmark_zelda(
        graph,
        problem_name="zelda-large-v0",
        enemy_target=8,
        seed=1,
    )

    assert mapping.metadata["control_fallback_applied"] is True
    assert mapping.metadata["mapper_mode"] == "corridor_fallback"
    initial_control = mapping.metadata["content_control_initial"]
    assert _control_error(initial_control, mapping.graph_control) > 0
    assert mapping.content_control["player_key"] == mapping.graph_control["player_key"]
    assert mapping.content_control["key_door"] == mapping.graph_control["key_door"]


def test_select_pcg_benchmark_zelda_problem_chooses_large_for_big_graph():
    G = nx.path_graph(30, create_using=nx.DiGraph)
    for idx in G.nodes():
        G.nodes[idx]["position"] = (idx, 0)
    variant = select_pcg_benchmark_zelda_problem(G)
    assert variant.name == "zelda-large-v0"


def test_external_alignment_uses_benchmark_shaped_room_budgets():
    assert _problem_room_budget("zelda-v0", 42) == (8, 16)
    assert _problem_room_budget("zelda-enemies-v0", 42) == (8, 16)
    assert _problem_room_budget("zelda-large-v0", 42) == (18, 32)


def test_external_summary_reports_strict_and_continuous_metrics_side_by_side():
    summary = _build_external_summary(
        "zelda-v0",
        {
            "quality_mean": 0.25,
            "diversity_mean": 0.5,
            "controlability_mean": 0.75,
            "rows": [
                {
                    "semantic_valid": 1.0,
                    "quality": 0.8,
                    "diversity": 1.0,
                    "controlability": 0.9,
                    "solution_length_pass": 1.0,
                    "enemy_band_pass": 1.0,
                    "solution_length": 22.0,
                    "enemies": 3.0,
                    "control_fallback_applied": 0.0,
                    "initial_solution_length": 20.0,
                    "player_key_abs_error_initial": 3.0,
                    "key_door_abs_error_initial": 4.0,
                    "player_key_abs_error": 1.0,
                    "key_door_abs_error": 2.0,
                    "graph_player_key_raw": 3.0,
                    "graph_key_door_raw": 4.0,
                    "graph_player_key": 11.0,
                    "graph_key_door": 11.0,
                    "content_player_key_initial": 8.0,
                    "content_key_door_initial": 12.0,
                    "content_player_key": 11.0,
                    "content_key_door": 11.0,
                },
                {
                    "semantic_valid": 0.0,
                    "quality": 1.0,
                    "diversity": 0.5,
                    "controlability": 0.7,
                    "solution_length_pass": 0.0,
                    "enemy_band_pass": 1.0,
                    "solution_length": 18.0,
                    "enemies": 4.0,
                    "control_fallback_applied": 1.0,
                    "initial_solution_length": 14.0,
                    "player_key_abs_error_initial": 6.0,
                    "key_door_abs_error_initial": 7.0,
                    "player_key_abs_error": 2.0,
                    "key_door_abs_error": 3.0,
                    "graph_player_key_raw": 2.0,
                    "graph_key_door_raw": 5.0,
                    "graph_player_key": 11.0,
                    "graph_key_door": 12.0,
                    "content_player_key_initial": 13.0,
                    "content_key_door_initial": 9.0,
                    "content_player_key": 10.0,
                    "content_key_door": 12.0,
                },
            ],
        },
    )

    assert summary["external_quality_pass_rate"] == 0.25
    assert summary["external_semantic_valid_rate"] == 0.5
    assert summary["external_diversity_pass_rate"] == 0.5
    assert summary["external_controlability_pass_rate"] == 0.75
    assert summary["external_quality_detail_mean"] == 0.9
    assert summary["external_controlability_detail_mean"] == 0.8
    assert summary["external_solution_length_pass_rate"] == 0.5
    assert summary["external_enemy_band_pass_rate"] == 1.0
    assert summary["external_control_fallback_rate"] == 0.5
    assert summary["external_mean_initial_solution_length"] == 17.0
    assert summary["external_mean_initial_abs_player_key_error"] == 4.5
    assert summary["external_mean_initial_abs_key_door_error"] == 5.5
    assert summary["external_mean_abs_player_key_error"] == 1.5
    assert summary["external_mean_abs_key_door_error"] == 2.5
    assert summary["external_mean_content_player_key_initial"] == 10.5
    assert summary["external_mean_content_key_door_initial"] == 10.5
    assert summary["external_solution_length_target"] == 18.0
    assert summary["external_enemy_target"] == 3.0


def test_ood_summary_row_includes_topology_semantics_metrics():
    graph = _make_graph_for_alignment()
    bench = run_block_i_benchmark(generated_graphs=[graph], reference_graphs=[graph], generation_times=[0.2])

    row = _build_ood_summary_row(
        regime_name="in_dist",
        method_name="FULL_GA",
        bench=bench,
        min_rooms=8,
        max_rooms=16,
        gen_times=[0.2],
        wall_time_sec=0.3,
        n_graphs=1,
    )

    assert row["key_before_lock_rate"] >= 0.0
    assert row["switch_before_gate_rate"] >= 0.0
    assert row["battery_satisfaction_rate"] >= 0.0
    assert "path_redundancy" in row
    assert "articulation_ratio" in row
    assert "branch_utility_rate" in row
    assert "secret_content_discoverability_rate" in row


def test_pcg_benchmark_candidate_detection_prefers_cwd_clone(monkeypatch, tmp_path):
    repo_root = tmp_path / "tmp" / "pcg_benchmark_upstream"
    (repo_root / "pcg_benchmark").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PCG_BENCHMARK_REPO", raising=False)

    candidates = _pcg_benchmark_repo_candidates()
    assert candidates
    assert candidates[0] == repo_root.resolve()


def test_import_pcg_benchmark_uses_explicit_repo(monkeypatch, tmp_path):
    repo_root = tmp_path / "local_repo"
    pkg_root = repo_root / "pcg_benchmark"
    pkg_root.mkdir(parents=True)
    (pkg_root / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")

    monkeypatch.delenv("PCG_BENCHMARK_REPO", raising=False)
    sys.modules.pop("pcg_benchmark", None)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in list(sys.path) if "pcg_benchmark" not in str(entry).lower()],
    )

    module = import_pcg_benchmark(repo_path=repo_root)
    assert getattr(module, "VALUE", None) == 7
    assert str(getattr(module, "__file__", "")).startswith(str(repo_root))
