"""Nonlocal graph transitions must retain gates, direction and route costs."""

import copy

import networkx as nx
import numpy as np
import pytest

from src.core.definitions import ROOM_HEIGHT as H, ROOM_WIDTH as W, SEMANTIC_PALETTE as P
from src.pipeline.room_stitching import build_stitched_room_layout_from_slots, validate_connection_realizations
from src.pipeline.stitching.dungeon_assembler import compile_stitched_graph
from src.pipeline.validation_context import build_stitched_validation_context
from src.simulation.search_base import GameStateSearchConfig
from src.simulation.search_factory import run_game_state_solver
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv


def transport_env(provide_key):
    graph = nx.DiGraph(spatial_transition_policy='recorded')
    graph.add_edge('a', 'b', edge_type='key_locked', edge_constraints=['stair', 'key_locked'])
    rooms = {node: np.full((H, W), P['FLOOR'], dtype=np.int64) for node in graph}
    layout = build_stitched_room_layout_from_slots(rooms, graph, {'a': (0, 0), 'b': (0, 2)})
    record = layout.connection_realizations['a', 'b']
    entry, exit_pos = record['source_position'], record['target_position']
    row, col = entry
    layout.dungeon_grid[row, col - 2] = P['START']
    if provide_key:
        layout.dungeon_grid[row, col - 1] = P['KEY_SMALL']
    layout.dungeon_grid[exit_pos[0], exit_pos[1] + 1] = P['TRIFORCE']
    return ZeldaLogicEnv(layout.dungeon_grid, **build_stitched_validation_context(graph, layout)), entry, exit_pos


@pytest.mark.parametrize('algorithm', range(8))
@pytest.mark.parametrize('provide_key', [False, True])
def test_search_and_independent_replay_obey_recorded_transport(algorithm, provide_key, tmp_path):
    env, entry, exit_pos = transport_env(provide_key)
    config = GameStateSearchConfig(timeout=20000, max_depth=8, allow_diagonals=False)
    if algorithm == 7:
        from src.ml.heuristic_learning import HeuristicTrainer
        checkpoint = tmp_path / 'fresh_transport_tiebreak.pth'
        HeuristicTrainer(map_height=env.height, map_width=env.width).save_model(str(checkpoint))
        config.learned_heuristic_model_path = str(checkpoint)
    result = run_game_state_solver(env, algorithm, config)
    assert result.success == provide_key
    if provide_key:
        assert (entry, exit_pos) in list(zip(result.path, result.path[1:]))
        valid, reason, final, cost = StateSpaceAStar(env).verify_position_path(result.path)
        assert valid, reason
        assert final.keys == 0
        assert cost == len(result.path) - 1 + 0.5  # One key pickup costs 1.5.
        if algorithm in (0, 1, 2, 4, 6, 7):
            assert cost == 4.5


def test_interactive_transport_cannot_reverse_or_skip_gate():
    env, entry, exit_pos = transport_env(False)
    env.state.position = entry
    assert env.step_graph_transition(exit_pos)[0].position == entry
    env.state.keys = 1
    state, _, _, _ = env.step_graph_transition(exit_pos)
    assert state.position == exit_pos and state.keys == 0
    assert env.step_graph_transition(entry)[0].position == exit_pos
    assert not env.try_move_pure(state, entry, P['STAIR'])[0]


def test_transition_endpoints_are_frozen_across_stage_grids_and_replayable_from_disk(tmp_path):
    from types import SimpleNamespace
    from src.publication.graph_evidence import write_graph_evidence, load_graph_evidence
    graph = nx.DiGraph(spatial_transition_policy='recorded')
    graph.add_edge(0, 1, edge_type='stair')
    rooms = {node: np.full((H, W), P['FLOOR'], dtype=np.int64) for node in graph}
    slots = {0: (0, 0), 1: (0, 1)}
    layout = build_stitched_room_layout_from_slots(rooms, graph, slots)
    original = layout.connection_realizations[0, 1]
    altered = {node: room.copy() for node, room in rooms.items()}
    row, col = original['source_position']
    altered[0][row, col] = P['WALL']
    stage = build_stitched_room_layout_from_slots(altered, graph, slots, fixed_graph_transitions=layout.connection_realizations)
    assert stage.connection_realizations == layout.connection_realizations
    result = SimpleNamespace(stitched_layout=layout, physical_mission_graph=graph)
    artifact = write_graph_evidence(result, tmp_path)
    context = load_graph_evidence(artifact)
    env = ZeldaLogicEnv(layout.dungeon_grid, **context)
    env.state.position = original['source_position']
    assert env.step_graph_transition(original['target_position'])[0].position == original['target_position']


@pytest.mark.parametrize('graph', [nx.star_graph(5), nx.complete_bipartite_graph(3, 3)])
def test_nonplanar_and_high_degree_missions_keep_all_edges_and_labels(graph):
    graph.graph['spatial_transition_policy'] = 'recorded'
    graph.nodes[0].update(is_start=True)
    graph.nodes[max(graph)].update(is_goal=True)
    for source, target in graph.edges:
        graph.edges[source, target]['edge_type'] = 'open'
    original = copy.deepcopy(nx.node_link_data(graph, edges='links'))
    rooms = {node: np.full((H, W), P['FLOOR'], dtype=np.int64) for node in graph}
    slots = {node: (0, node) for node in graph}
    layout = build_stitched_room_layout_from_slots(rooms, graph, slots)
    grid, _, _, integrity = compile_stitched_graph(layout.dungeon_grid, graph, layout, seed=42)
    assert nx.node_link_data(graph, edges='links') == original
    assert len(layout.connection_realizations) == graph.number_of_edges()
    assert layout.realization_metrics['spatial_graph_edges_uncarved'] == 0
    assert layout.realization_metrics['spatial_topology_exact_invariants_preserved']
    assert layout.realization_metrics['recorded_graph_transition_count'] > 0
    assert integrity['final_spatial_edge_records_broken'] == 0
    env = ZeldaLogicEnv(grid, **build_stitched_validation_context(graph, layout))
    success, path, _ = StateSpaceAStar(env, timeout=20000).solve_with_verified_diagnostics()
    assert success
    assert StateSpaceAStar(env).verify_position_path(path)[0]
    damaged = grid.copy()
    record = next(record for record in layout.connection_realizations.values() if record['mode'] == 'graph_transition')
    damaged[tuple(record['source_position'])] = P['FLOOR']
    assert validate_connection_realizations(damaged, layout.connection_realizations)['final_spatial_edge_records_broken'] > 0
