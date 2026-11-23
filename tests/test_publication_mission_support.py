"""A primary mission's representation must be supported before training."""

import json

import networkx as nx
import pytest

from src.publication.evaluation_design import seal_evaluation_design, validate_mission_support


@pytest.mark.parametrize("kind", ["state_block", "stair", "multi_lock"])
def test_unsupported_mission_blocks_training_design_instead_of_scoring_every_model_zero(tmp_path, kind):
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True)
    graph.add_node(1, is_goal=True)
    graph.add_edge(0, 1, edge_type=kind)
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(nx.node_link_data(graph, edges="links")), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="primary exact-validity endpoint cannot be certified"):
        seal_evaluation_design(mission_bank={"fold": {"mission": path}}, sampling_seeds=[1],
                               generation_kwargs={}, fold_ids=["fold"])
    assert path.read_bytes() == before


def test_supported_but_unsolvable_mission_is_not_filtered_out(tmp_path):
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True)
    graph.add_node(1, is_goal=True)
    graph.add_edge(0, 1, edge_type="key_locked")  # No key provider: support is not success.
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(nx.node_link_data(graph, edges="links")), encoding="utf-8")
    validate_mission_support(path)


@pytest.mark.parametrize('graph', [nx.star_graph(5), nx.complete_bipartite_graph(3, 3)])
def test_unrepresentable_layout_blocks_design_before_training(tmp_path, graph):
    path = tmp_path / 'mission.json'
    path.write_text(json.dumps(nx.node_link_data(graph, edges='links')), encoding='utf-8')
    with pytest.raises(ValueError, match='four-neighbor|nonplanar'):
        validate_mission_support(path)


def test_supported_physical_progression_contract_can_be_sealed(tmp_path):
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True, type='switch')
    graph.add_node(1, is_goal=True)
    graph.add_edge(0, 1, edge_constraints=['switch', 'soft_locked', 'key_locked', 'item_locked'],
                   switches_required=[0], requires_key_count=2)
    path = tmp_path / 'mission.json'
    path.write_text(json.dumps(nx.node_link_data(graph, edges='links')), encoding='utf-8')
    validate_mission_support(path)
