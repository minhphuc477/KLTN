from dataclasses import dataclass
import json

import networkx as nx
import numpy as np
import pytest

from scripts.run_synthetic_metroidvania_ood_probe import (
    build_chain_control_graph,
    build_metroidvania_ood_graph,
)
from scripts.analyze_topology_repair_shift import summarize as summarize_topology_repair_shift
from scripts.validate_human_playtest_telemetry import validate_human_session
from scripts.visualize_qd_archive import analyze_records, extract_archive_records
from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.map_elites import MAPElitesEvaluator, run_map_elites_on_maps
from src.utils.playtest_telemetry import PlaytestTelemetryCollector


def _grid() -> np.ndarray:
    return np.full((16, 11), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int64)


def _chain_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, label="START", difficulty=0.0)
    graph.add_node(1, label="KEY", difficulty=0.3)
    graph.add_node(2, label="LOCK", difficulty=0.7)
    graph.add_node(3, label="GOAL", difficulty=1.0)
    graph.add_edge(0, 1, edge_type="PATH")
    graph.add_edge(1, 2, edge_type="LOCKED")
    graph.add_edge(2, 3, edge_type="PATH")
    return graph


def _branch_graph() -> nx.DiGraph:
    graph = _chain_graph()
    graph.add_node(4, label="ENEMY", difficulty=0.5)
    graph.add_edge(1, 4, edge_type="PATH")
    graph.add_edge(4, 2, edge_type="PATH")
    return graph


def test_topology_repair_shift_refuses_pickle_without_explicit_trust(tmp_path):
    graph_path = tmp_path / "graph.pkl"
    graph_path.write_bytes(b"not a trusted pickle")

    payload = summarize_topology_repair_shift([graph_path])

    assert payload["summary"]["error_count"] == 1
    assert "Refusing to unpickle" in payload["rows"][0]["error"]


def test_runtime_map_elites_prefers_macro_graph_descriptors():
    evaluator = MAPElitesEvaluator(enable_advanced_archive=False, descriptor_mode="hybrid")
    solver_result = {
        "solvable": True,
        "path_length": 3,
        "path": [(1, 1), (1, 2), (1, 3)],
    }
    chain_features, chain_metrics = evaluator._build_behavior_descriptor(_grid(), solver_result, _chain_graph())
    branch_features, branch_metrics = evaluator._build_behavior_descriptor(_grid(), solver_result, _branch_graph())

    assert chain_metrics["graph_descriptor_used"] == 1.0
    assert branch_metrics["graph_descriptor_used"] == 1.0
    assert chain_features != branch_features
    assert branch_metrics["branching_factor"] > chain_metrics["branching_factor"]
    assert chain_metrics["critical_path_length"] == 3.0
    assert chain_metrics["graph_descriptor_feasible"] == 1.0
    assert chain_metrics["graph_path_keys_consumed"] == 1.0


def test_runtime_map_elites_rejects_infeasible_macro_descriptor_path():
    graph = nx.DiGraph()
    graph.add_node(0, label="START")
    graph.add_node(1, label="GOAL")
    graph.add_edge(0, 1, edge_type="LOCKED", key_required="key_generic")
    evaluator = MAPElitesEvaluator(enable_advanced_archive=False, descriptor_mode="hybrid")

    _, metrics = evaluator._build_behavior_descriptor(
        _grid(),
        {"solvable": True, "path_length": 2, "path": [(1, 1), (1, 2)]},
        graph,
    )

    assert metrics["graph_descriptor_used"] == 0.0
    assert "graph_descriptor_feasible" not in metrics


def test_runtime_map_elites_does_not_archive_infeasible_mission_economy():
    graph = nx.DiGraph()
    graph.add_node(0, label="START")
    graph.add_node(1, label="GOAL")
    graph.add_edge(0, 1, edge_type="LOCKED", key_required="missing_key")
    evaluator = MAPElitesEvaluator(enable_advanced_archive=False, descriptor_mode="hybrid")

    metrics = evaluator.add_dungeon(
        dungeon=_grid(),
        grid=_grid(),
        solver_result={
            "solvable": True,
            "path_length": 2,
            "path": [(1, 1), (1, 2)],
            "quality_score": 1.0,
        },
        mission_graph=graph,
    )

    assert metrics["archive_rejected_graph_progression"] == 1.0
    assert evaluator.grid == {}


def test_runtime_map_elites_does_not_collect_required_item_as_provider():
    graph = nx.DiGraph()
    graph.add_node(0, label="START")
    graph.add_node(1, label="PUZZLE", required_item="BOMB")
    graph.add_node(2, label="GOAL")
    graph.add_edge(0, 1, edge_type="PATH")
    graph.add_edge(1, 2, edge_type="ITEM_GATE", item_required="BOMB")
    evaluator = MAPElitesEvaluator(enable_advanced_archive=False, descriptor_mode="hybrid")

    _, metrics = evaluator._build_behavior_descriptor(
        _grid(),
        {"solvable": True, "path_length": 2, "path": [(1, 1), (1, 2)]},
        graph,
    )

    assert metrics["graph_descriptor_used"] == 0.0
    assert "graph_descriptor_feasible" not in metrics


def test_runtime_map_elites_collects_real_item_provider_for_item_gate():
    graph = nx.DiGraph()
    graph.add_node(0, label="START")
    graph.add_node(1, label="ITEM", item_type="BOMB")
    graph.add_node(2, label="GOAL")
    graph.add_edge(0, 1, edge_type="PATH")
    graph.add_edge(1, 2, edge_type="ITEM_GATE", item_required="BOMB")
    evaluator = MAPElitesEvaluator(enable_advanced_archive=False, descriptor_mode="hybrid")

    _, metrics = evaluator._build_behavior_descriptor(
        _grid(),
        {"solvable": True, "path_length": 2, "path": [(1, 1), (1, 2)]},
        graph,
    )

    assert metrics["graph_descriptor_used"] == 1.0
    assert metrics["graph_descriptor_feasible"] == 1.0


def test_runtime_map_elites_legacy_mode_preserves_grid_ablation():
    evaluator = MAPElitesEvaluator(enable_advanced_archive=False, descriptor_mode="legacy")
    features, metrics = evaluator._build_behavior_descriptor(
        _grid(),
        {"solvable": True, "path_length": 5},
        _branch_graph(),
    )

    assert metrics["graph_descriptor_used"] == 0.0
    assert features[3] == metrics["density"]


@dataclass
class _DungeonWrapper:
    global_grid: np.ndarray
    mission_graph: nx.DiGraph


class _AlwaysSolves:
    def solve(self, _dungeon):
        return {
            "solvable": True,
            "path_length": 3,
            "path": [(1, 1), (1, 2), (1, 3)],
        }


def test_map_elites_list_runner_forwards_embedded_mission_graph():
    evaluator, _ = run_map_elites_on_maps(
        [_DungeonWrapper(_grid(), _branch_graph())],
        solver=_AlwaysSolves(),
        enable_advanced_archive=False,
    )
    entry = next(iter(evaluator.grid.values()))
    assert entry.metrics["graph_descriptor_used"] == 1.0


def test_human_session_requires_consent_and_records_provenance(tmp_path):
    collector = PlaytestTelemetryCollector(tmp_path, append_jsonl=False)
    with pytest.raises(ValueError, match="requires recorded consent"):
        collector.start_human_session(
            "session_bad",
            participant_id="P001",
            study_id="study",
            consent_recorded=False,
        )

    collector.start_human_session(
        "session_ok",
        participant_id="P001",
        study_id="study",
        consent_recorded=True,
    )
    collector.log_event("step", position=(1, 2))
    session = collector.current_session
    assert session is not None
    assert validate_human_session(collector._session_to_dict(session)) == []


def test_qd_visualizer_normalizes_runtime_json_grid():
    records = extract_archive_records(
        {
            "grid": {
                "0,0": {
                    "score": 0.75,
                    "metrics": {
                        "linearity": 0.1,
                        "leniency": 0.2,
                        "progression_complexity": 0.3,
                        "topology_complexity": 0.4,
                    },
                },
                "1,1": {
                    "score": 0.90,
                    "metrics": {
                        "linearity": 0.8,
                        "leniency": 0.7,
                        "progression_complexity": 0.6,
                        "topology_complexity": 0.5,
                    },
                },
            }
        }
    )
    summary = analyze_records(records)
    assert summary["num_elites"] == 2
    assert summary["feature_dims"] == 4
    assert summary["fitness"]["max"] == 0.9


def test_runtime_map_elites_exports_portable_json(tmp_path):
    evaluator = MAPElitesEvaluator(enable_advanced_archive=False)
    evaluator.add_dungeon(
        dungeon=_grid(),
        grid=_grid(),
        solver_result={"solvable": True, "path_length": 5, "quality_score": 0.8},
        mission_graph=_chain_graph(),
    )
    export_path = evaluator.export_archive_json(tmp_path / "archive.json")
    records = extract_archive_records(json.loads(export_path.read_text(encoding="utf-8")))
    assert len(records) == 1
    assert len(records[0]["features"]) == 4


def test_synthetic_ood_graph_is_structurally_distinct():
    control = build_chain_control_graph()
    ood = build_metroidvania_ood_graph()
    assert ood.number_of_nodes() > control.number_of_nodes()
    assert any(ood.out_degree(node) >= 2 for node in ood.nodes)
    assert not any(control.out_degree(node) >= 2 for node in control.nodes)
