"""Source-schema regressions; fixtures are synthetic, with no experiment artifacts."""

import networkx as nx
import numpy as np
import pytest

from src.core.definitions import SEMANTIC_PALETTE, parse_edge_type_tokens
from src.zelda_data.parsers.core_parsers import DOTParser, GridBasedRoomExtractor, VGLCParser
from src.zelda_data.splits import normalize_dungeon_ids, normalize_variants
from src.zelda_data.vglc_utils import parse_edge_attributes
from src.zelda_data.zelda_core import RoomGraphMatcher, StateSpaceGraphSolver
from src.data_processing.data_adapter import GraphvizParser


def test_vglc_visibility_is_retained_outside_passage_adjacency(tmp_path):
    source = tmp_path / "source.dot"
    source.write_text('digraph G {\n0 [label="s"];\n1 [label="e"];\n'
                      '2 [label="t"];\n0 -> 1;\n1 -> 2 [label="s"];\n}',
                      encoding="utf-8")
    graph = DOTParser().parse(str(source))
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(1, 2)
    assert not nx.has_path(graph, 0, 2)
    assert graph.graph["non_traversable_edges"] == [
        {"source": 1, "target": 2, "label": "s", "edge_type": "impassable",
         "edge_constraints": ["impassable"]}
    ]
    assert parse_edge_type_tokens("s") == ["impassable"]
    assert parse_edge_type_tokens("s,k") == ["impassable", "key_locked"]
    assert parse_edge_type_tokens("stair") == ["stair"]
    # The uppercase *tile* S remains a stair, independently of DOT labels.
    from src.core.definitions import CHAR_TO_SEMANTIC
    assert CHAR_TO_SEMANTIC["S"] == SEMANTIC_PALETTE["STAIR"]


def test_source_switch_does_not_become_a_start_during_normalization(tmp_path):
    source = tmp_path / "source.dot"
    source.write_text('digraph G {\n0 [label="s"];\n1 [label="p,S1"];\n'
                      '2 [label="S"];\n3 [label="K"];\n0 -> 1;\n}',
                      encoding="utf-8")
    graph = DOTParser().parse(str(source))
    RoomGraphMatcher()._normalize_graph(graph)
    assert graph.nodes[0]["is_start"]
    for node in (1, 2):
        assert graph.nodes[node]["has_switch"]
        assert not graph.nodes[node]["is_start"]
        assert not graph.nodes[node]["is_start_pointer"]
    assert graph.nodes[1]["source_label"] == "p,S1"
    assert graph.nodes[1]["switch_tokens"] == ["S1"]
    assert graph.nodes[3]["has_boss_key"]
    assert not graph.nodes[3]["has_item"]


def test_vglc_graph_utilities_do_not_classify_visibility_as_warp():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="s")
    edge = parse_edge_attributes(graph, 0, 1)
    assert edge.edge_type == "impassable"
    assert not edge.is_warp
    assert not edge.is_open


@pytest.mark.parametrize("label", ["s", "unknown_gate"])
def test_graph_inventory_solver_fails_closed_on_nonpassage_constraints(label):
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label=label)
    assert not StateSpaceGraphSolver(graph).solve(0, 1)["solvable"]


def test_graph_inventory_solver_accepts_unlabelled_open_edge():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    assert StateSpaceGraphSolver(graph).solve(0, 1)["solvable"]


def test_adapter_parser_shares_schema_without_overwriting_node_labels_from_edges():
    source = ('digraph G { 0 [label="s"]; 1 [label="S1,p"]; 2 [label="K"]; '
              '0 -> 1 [label="k"]; 1 -> 2 [label="s"]; }')
    graph = GraphvizParser().parse_string(source)
    assert graph.nodes[1]["source_label"] == "S1,p"
    assert graph.nodes[1]["has_switch"]
    assert not graph.nodes[1]["is_start"]
    assert graph.nodes[2]["has_boss_key"]
    assert not graph.has_edge(1, 2)
    assert graph.edges[0, 1]["edge_constraints"] == ["key_locked"]


def test_adapter_parser_preserves_composite_gate_requirements():
    graph = GraphvizParser().parse_string('digraph G { 0 -> 1 [label="I,S1"]; }')
    assert graph.edges[0, 1]["edge_constraints"] == ["item_locked", "switch"]


def test_numbered_switch_requires_its_actual_provider_not_an_unrelated_puzzle():
    from src.evaluation.validator import ExternalValidator
    graph = DOTParser().parse_string('digraph G { 0 [label="s"]; 1 [label="p"]; '
        '2 [label="S1"]; 3 [label="t"]; 0 -> 1; 1 -> 3 [label="S1"]; }')
    assert graph.edges[1, 3]["switches_required"] == [2]
    assert not ExternalValidator(mode="full").validate(graph).is_solvable
    graph.add_edge(1, 2, edge_type="open")
    graph.add_edge(2, 1, edge_type="open")
    result = ExternalValidator(mode="full").validate(graph)
    assert result.is_solvable
    assert 2 in result.solution_path


def test_unresolved_numbered_switch_cannot_be_unlocked_by_any_puzzle():
    from src.evaluation.validator import ExternalValidator
    graph = DOTParser().parse_string('digraph G { 0 [label="s,p"]; 1 [label="t"]; 0 -> 1 [label="S2"]; }')
    assert graph.edges[0, 1]["edge_constraints"] == ["switch"]
    assert graph.edges[0, 1]["unresolved_switch_tokens"] == ["S2"]
    assert not ExternalValidator(mode="full").validate(graph).is_solvable


@pytest.mark.parametrize("normalize", [normalize_dungeon_ids, normalize_variants])
@pytest.mark.parametrize("values", [[], [True], [1.5], ["1"], [float("nan")]])
def test_split_identities_never_silently_coerce_or_expand(normalize, values):
    with pytest.raises(ValueError):
        normalize(values)


def test_numpy_integer_split_identities_are_supported():
    assert normalize_dungeon_ids([np.int64(2), np.int64(1)]) == (1, 2)
    assert normalize_variants(None) == (1, 2)


@pytest.mark.parametrize("lines", [["W" * 11] * 15, ["W" * 12] * 16,
    ["W" * 11] * 15 + ["W" * 10], ["?" * 11] * 16])
def test_both_map_parsers_reject_lossy_input_recovery(tmp_path, lines):
    path = tmp_path / "bad.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError):
        GridBasedRoomExtractor().extract(str(path))
    with pytest.raises(ValueError):
        VGLCParser(room_cls=dict).parse(str(path))
