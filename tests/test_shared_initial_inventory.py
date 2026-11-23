"""Finite initial inventory must agree in graph, physical search and replay."""

import networkx as nx
import numpy as np
import pytest

from src.core.definitions import ROOM_HEIGHT as H, ROOM_WIDTH as W, SEMANTIC_PALETTE as P
from src.evaluation.validator import ExternalValidator
from src.pipeline.room_stitching import build_stitched_room_layout_from_slots
from src.pipeline.validation_context import build_stitched_validation_context
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv


def test_physical_entry_survives_virtual_start_removal_in_exact_oracle():
    from src.zelda_data.vglc_utils import filter_virtual_nodes
    graph = nx.DiGraph(initial_inventory={'bomb_count': 0})
    graph.add_node(0, label='s', is_start_pointer=True)
    graph.add_node(1, label='', has_key=True)
    graph.add_node(2, label='t')
    graph.add_edge(0, 1, edge_type='open')
    graph.add_edge(1, 2, edge_type='key_locked')
    physical = filter_virtual_nodes(graph)
    assert 0 not in physical and physical.nodes[1]['is_entry']
    result = ExternalValidator(mode='full').validate(physical)
    assert result.is_solvable and result.solution_path == [1, 2]


@pytest.mark.parametrize('kind,resource', [('key_locked', 'keys'), ('bombable', 'bomb_count')])
@pytest.mark.parametrize('source_door', [False, True])
def test_source_reciprocal_door_charges_once_but_authored_arcs_stay_independent(kind, resource, source_door):
    from types import SimpleNamespace
    from src.simulation.graph_progression import apply_graph_gate
    from src.simulation.state import GameState
    from src.zelda_data.adapter_io import bind_source_physical_doors
    graph = nx.DiGraph()
    graph.add_edge(0, 1, edge_type=kind)
    graph.add_edge(1, 0, edge_type=kind)
    if source_door:
        bind_source_physical_doors(graph)
    state = GameState(position=(0, 0), **{resource: 1})
    env = SimpleNamespace(graph=graph)
    allowed, state = apply_graph_gate(env, state, 0, 1)
    assert allowed and getattr(state, resource) == 0
    allowed, _ = apply_graph_gate(env, state, 1, 0)
    assert allowed == source_door


def test_source_door_binding_preserves_asymmetric_requirements():
    from src.simulation.edge_logic import graph_edge_state_key
    from src.zelda_data.adapter_io import bind_source_physical_doors
    graph = nx.DiGraph()
    graph.add_edge(0, 1, edge_type='key_locked', requires_key_count=1)
    graph.add_edge(1, 0, edge_type='key_locked', requires_key_count=2)
    bind_source_physical_doors(graph)
    assert graph_edge_state_key(graph, 0, 1) != graph_edge_state_key(graph, 1, 0)


@pytest.mark.parametrize('bombs', [0, 1, 2])
def test_graph_and_tile_oracles_spend_the_same_finite_bombs(bombs):
    graph = nx.DiGraph(initial_inventory={'bomb_count': bombs}, spatial_transition_policy='recorded')
    graph.add_node(0, is_start=True)
    graph.add_node(1)
    graph.add_node(2, is_goal=True)
    graph.add_edge(0, 1, edge_constraints=['stair', 'bombable'], edge_type='bombable')
    graph.add_edge(1, 2, edge_constraints=['stair', 'bombable'], edge_type='bombable')
    oracle = ExternalValidator(mode='full')
    result = oracle.validate(graph, max_states=1000)
    assert result.is_solvable == (bombs == 2)
    if result.is_solvable:
        assert result.metrics['doors_opened'] == 2
    rooms = {node: np.full((H, W), P['FLOOR'], dtype=np.int64) for node in graph}
    layout = build_stitched_room_layout_from_slots(rooms, graph, {0: (0, 0), 1: (0, 1), 2: (0, 2)})
    first = layout.connection_realizations[0, 1]['source_position']
    last = layout.connection_realizations[1, 2]['target_position']
    layout.dungeon_grid[first[0], first[1] - 1] = P['START']
    layout.dungeon_grid[last[0], last[1] + 1] = P['TRIFORCE']
    env = ZeldaLogicEnv(layout.dungeon_grid, **build_stitched_validation_context(graph, layout))
    assert env.state.bomb_count == bombs
    env.reset()
    assert env.state.bomb_count == bombs
    success, path, _ = StateSpaceAStar(env, timeout=10000).solve_with_verified_diagnostics()
    assert success == result.is_solvable
    if success:
        verified, _, final, _ = StateSpaceAStar(env).verify_position_path(path)
        assert verified and final.bomb_count == 0


@pytest.mark.parametrize('inventory', [{'bomb_count': True}, {'bomb_count': -1}, {'bomb_count': 1.5}, {'keys': 4}])
def test_initial_inventory_rejects_ambiguous_or_unsupported_fields(inventory):
    graph = nx.DiGraph(initial_inventory=inventory)
    graph.add_node(0, is_start=True)
    graph.add_node(1, is_goal=True)
    graph.add_edge(0, 1)
    with pytest.raises(ValueError, match='inventory|bomb_count'):
        ExternalValidator(mode='full').validate(graph)
    with pytest.raises(ValueError, match='inventory|bomb_count'):
        ZeldaLogicEnv(np.ones((H, W)), graph=graph)


def test_all_room_coverage_is_complete_when_every_node_has_a_witness_at_budget():
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True)
    graph.add_node(1, is_goal=True, has_key=True)
    graph.add_edge(0, 1, edge_type='open')
    graph.add_edge(1, 0, edge_type='open')
    oracle = ExternalValidator(mode='full')
    result = oracle.checker.check_all_rooms_reachable_detailed(graph, max_states=2)
    assert result['complete'] and not result['unreachable_nodes']
    assert result['termination_status'] == 'all_nodes_reached'
