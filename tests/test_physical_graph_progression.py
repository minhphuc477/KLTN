"""Hand-built progression witnesses, independent of experiment artifacts."""

import networkx as nx
import numpy as np
import pytest

from src.core.definitions import ROOM_HEIGHT as H, ROOM_WIDTH as W, SEMANTIC_PALETTE as P
from src.simulation.graph_progression import apply_graph_gate
from src.simulation.search_base import GameStateSearchConfig
from src.simulation.search_factory import run_game_state_solver
from src.simulation.state import Action, dominates, game_state_key
from src.simulation.validator import SolverOptions, StateSpaceAStar, ZeldaLogicEnv


def passage(kind="key_locked", *, directed=False, wide=False, gap=0, **edge_attrs):
    grid = np.full((H, W * 2 + gap), P['WALL'], dtype=np.int64)
    row = H // 2
    grid[row, W - 3:W + gap + 3] = P['FLOOR']
    grid[row, W - 3] = P['START']
    grid[row, W + gap + 2] = P['TRIFORCE']
    cells = [(row, col) for col in range(W - 1, W + gap + 1)]
    if wide:
        grid[row + 1, W - 2:W + gap + 2] = P['FLOOR']
        cells += [(row + 1, col) for col in range(W - 1, W + gap + 1)]
    tile = {'key_locked': 'DOOR_LOCKED', 'soft_locked': 'DOOR_SOFT',
            'switch': 'DOOR_PUZZLE', 'item_locked': 'DOOR_PUZZLE'}.get(kind, 'DOOR_OPEN')
    for pos in cells:
        grid[pos] = P[tile]
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edge('a', 'b', edge_type=kind, **edge_attrs)
    return ZeldaLogicEnv(
        grid, graph=graph, room_to_node={(0, 0): 'a', (0, 1): 'b'},
        room_positions={(0, 0): (0, 0), (0, 1): (0, W + gap)},
        node_to_room={'a': (0, 0), 'b': (0, 1)},
        connection_realizations={('a', 'b'): {'cells': cells}},
        solver_options=SolverOptions(start_bombs=0),
    )


def move(env, state, target):
    allowed, candidate = env.try_move_pure(state, target, int(env.original_grid[target]))
    assert allowed, (state.position, target)
    return candidate


@pytest.mark.parametrize('gap', [0, 3])
def test_wide_and_routed_passages_charge_once_and_remain_open(gap):
    env = passage(wide=True, gap=gap)
    row = H // 2
    state = env.state.copy()
    state.keys = 1
    for col in range(W - 2, W + gap + 2):
        state = move(env, state, (row, col))
    assert state.keys == 0
    assert len(state.opened_graph_edges) == 1
    state = move(env, state, (row + 1, W + gap + 1))
    for col in range(W + gap, W - 3, -1):
        state = move(env, state, (row + 1, col))
    assert state.keys == 0
    assert state.active_graph_edge is None


def test_compound_gate_is_atomic_and_checks_named_permanent_item():
    env = passage(edge_constraints=['key_locked', 'item_locked'], item_required='RAFT', requires_key_count=2)
    state = env.state.copy()
    state.keys, state.has_item, state.item_names = 2, True, {'LADDER'}
    before = game_state_key(state)
    assert not apply_graph_gate(env, state, 'a', 'b')[0]
    assert game_state_key(state) == before
    state.item_names.add('RAFT')
    allowed, candidate = apply_graph_gate(env, state, 'a', 'b')
    assert allowed and candidate.keys == 0
    candidate.item_names = {'LADDER'}
    assert not apply_graph_gate(env, candidate, 'a', 'b')[0]


def test_named_keys_fail_closed_and_zero_count_hint_means_one_key():
    env = passage(requires_key_count=0)
    state = env.state.copy()
    state.keys = 1
    allowed, candidate = apply_graph_gate(env, state, 'a', 'b')
    assert allowed and candidate.keys == 0
    env.graph.edges['a', 'b']['key_required'] = 'specific_key'
    assert not apply_graph_gate(env, state, 'a', 'b')[0]


def test_named_small_and_boss_keys_match_provider_room_pickups():
    # Small key matched via KEY provider node with key_id=7 in room 'a'
    env = passage('key_locked', key_required=7)
    env.graph.nodes['a'].update(type='KEY', key_id=7, has_key=True)
    key_pos = (H // 2 - 1, W - 2)
    env.original_grid[key_pos] = P['KEY_SMALL']

    state = move(env, env.state.copy(), (H // 2, W - 2))
    state.keys = 1
    # Uncollected item in inventory cannot satisfy named key identity
    assert not apply_graph_gate(env, state, 'a', 'b')[0]

    # Moving onto KEY_SMALL in provider room collects it and unlocks edge
    collected = move(env, state, key_pos)
    assert key_pos in collected.collected_items
    allowed, candidate = apply_graph_gate(env, collected, 'a', 'b')
    assert allowed and candidate.keys == 1

    # Mismatched key_required still fails closed
    env.graph.edges['a', 'b']['key_required'] = 99
    assert not apply_graph_gate(env, collected, 'a', 'b')[0]

    # Boss key matched via BIG_KEY provider node with key_id=2 in room 'a'
    boss_env = passage('boss_locked', key_required=2)
    boss_env.graph.nodes['a'].update(type='BIG_KEY', key_id=2, has_boss_key=True)
    boss_env.original_grid[key_pos] = P['KEY_BOSS']
    boss_state = move(boss_env, boss_env.state.copy(), (H // 2, W - 2))
    boss_state.has_boss_key = True
    assert not apply_graph_gate(boss_env, boss_state, 'a', 'b')[0]

    boss_start = move(boss_env, boss_env.state.copy(), (H // 2, W - 2))
    boss_collected = move(boss_env, boss_start, key_pos)
    assert boss_collected.has_boss_key
    allowed_boss, _ = apply_graph_gate(boss_env, boss_collected, 'a', 'b')
    assert allowed_boss


def test_directed_passage_cannot_be_reversed_after_opening():
    env = passage('one_way', directed=True)
    state = env.state.copy()
    row = H // 2
    for col in range(W - 2, W + 2):
        state = move(env, state, (row, col))
    allowed, _ = env.try_move_pure(state, (row, W), int(env.original_grid[row, W]))
    assert not allowed


def test_directed_passage_cannot_reverse_from_destination_boundary():
    env = passage('one_way', directed=True)
    state = env.state.copy()
    row = H // 2
    for col in range(W - 2, W + 1):
        state = move(env, state, (row, col))
    assert not env.try_move_pure(state, (row, W - 1), P['DOOR_OPEN'])[0]


def test_phantom_opening_cannot_cross_rooms():
    env = passage('open')
    row = H // 2 - 1
    env.original_grid[row, W - 1:W + 1] = P['FLOOR']
    state = env.state.copy()
    state.position = (row, W - 1)
    assert not env.try_move_pure(state, (row, W), P['FLOOR'])[0]


def test_graph_soft_door_requires_source_enemies_in_dataset_profile():
    env = passage('soft_locked')
    row = H // 2
    env.original_grid[row - 1, W - 2] = P['ENEMY']
    env._build_room_enemy_cache()
    state = move(env, env.state.copy(), (row, W - 2))
    assert not env.try_move_pure(state, (row, W - 1), P['DOOR_SOFT'])[0]
    state = move(env, state, (row - 1, W - 2))
    assert (row - 1, W - 2) in state.defeated_enemies
    state = move(env, state, (row, W - 2))
    assert env.try_move_pure(state, (row, W - 1), P['DOOR_SOFT'])[0]


def test_switch_is_activated_by_its_tile_and_participates_in_state_identity():
    env = passage('switch', switches_required=['remote'])
    env.graph.nodes['a'].update(type='switch', switch_id='remote')
    row = H // 2
    env.original_grid[row - 1, W - 2] = P['PUZZLE']
    state = move(env, env.state.copy(), (row, W - 2))
    assert not env.try_move_pure(state, (row, W - 1), P['DOOR_PUZZLE'])[0]
    activated = move(env, state, (row - 1, W - 2))
    activated = move(env, activated, (row, W - 2))
    assert activated.switches_activated == {'a', 'remote'}
    assert game_state_key(activated) != game_state_key(state)
    assert dominates(activated, state) and not dominates(state, activated)
    assert env.try_move_pure(activated, (row, W - 1), P['DOOR_PUZZLE'])[0]
    env.graph.edges['a', 'b']['switches_required'] = ['another_switch']
    assert not env.try_move_pure(activated, (row, W - 1), P['DOOR_PUZZLE'])[0]


def test_interactive_step_and_legal_actions_obey_same_gate():
    env = passage()
    env.state.position = (H // 2, W - 2)
    assert int(Action.RIGHT) not in env.get_valid_actions()
    original = env.state.copy()
    state, _, _, _ = env.step(Action.RIGHT)
    assert state == original
    env.state.keys = 1
    assert int(Action.RIGHT) in env.get_valid_actions()
    state, _, _, _ = env.step(Action.RIGHT)
    assert state.position == (H // 2, W - 1) and state.keys == 0


def test_source_minor_item_does_not_grant_progression_item_and_undirected_doors_share_state():
    from src.zelda_data.parsers.core_parsers import DOTParser
    from src.simulation.edge_logic import graph_edge_state_key
    graph = DOTParser().parse_string('graph { 0 [label="e,i,k"]; 1 [label="I"]; 0 -- 1 [label="k"]; }')
    assert graph.nodes[0]['has_minor_item']
    assert not graph.nodes[0]['has_item']
    assert graph.nodes[1]['has_item']
    assert graph_edge_state_key(graph, 0, 1) == graph_edge_state_key(graph, 1, 0)


def test_mixed_source_key_and_progression_item_both_materialize():
    from src.generation.entity_spawner import spawn_all_entities, materialize_entities_on_grid
    grid = np.full((H, W), P['WALL'], dtype=np.int64)
    grid[1:-1, 1:-1] = P['FLOOR']
    mission = {'nodes': {0: {'label': 'e,p,k,I', 'has_key': True, 'has_item': True}}}
    entities = spawn_all_entities(grid, mission, {0: (0, 0, W - 1, H - 1)}, seed=42)
    result = materialize_entities_on_grid(grid, entities)
    assert np.count_nonzero(result == P['KEY_SMALL']) == 1
    assert np.count_nonzero(result == P['KEY_ITEM']) == 1


def test_compiler_places_remote_switch_and_solver_must_visit_it():
    from src.pipeline.room_stitching import build_stitched_room_layout_from_slots
    from src.pipeline.stitching.dungeon_assembler import compile_stitched_graph
    from src.pipeline.validation_context import build_stitched_validation_context
    graph = nx.DiGraph()
    graph.add_node(0, type='switch', switch_id='remote')
    graph.add_node(1, is_start=True, type='START')
    graph.add_node(2, is_goal=True, type='GOAL')
    graph.add_edge(1, 0, edge_type='open')
    graph.add_edge(0, 1, edge_type='open')
    graph.add_edge(1, 2, edge_type='switch', switches_required=['remote'])
    rooms = {node: np.full((H, W), P['FLOOR'], dtype=np.int64) for node in graph}
    layout = build_stitched_room_layout_from_slots(rooms, graph, {0: (0, 0), 1: (0, 1), 2: (0, 2)})
    grid, _, _, _ = compile_stitched_graph(layout.dungeon_grid, graph, layout, seed=42)
    switch_tiles = {tuple(pos) for pos in np.argwhere(grid == P['PUZZLE'])}
    assert switch_tiles and all(col < W for _, col in switch_tiles)
    assert np.count_nonzero(grid == P['START']) == 1
    env = ZeldaLogicEnv(grid, **build_stitched_validation_context(graph, layout))
    success, path, _ = StateSpaceAStar(env, timeout=20000).solve_with_verified_diagnostics()
    assert success and switch_tiles.intersection(path)
    replay = StateSpaceAStar(env).verify_position_path(path)
    assert replay[0] and 'remote' in replay[2].switches_activated


@pytest.mark.parametrize('algorithm', range(8))
@pytest.mark.parametrize('provide_key', [False, True])
def test_all_search_algorithms_and_replay_enforce_physical_gate(algorithm, provide_key, tmp_path):
    env = passage(wide=True)
    if provide_key:
        env.original_grid[H // 2, W - 2] = P['KEY_SMALL']
    config = GameStateSearchConfig(timeout=2000, max_depth=20, allow_diagonals=False)
    if algorithm == 7:
        from src.ml.heuristic_learning import HeuristicTrainer
        trainer = HeuristicTrainer(map_height=env.height, map_width=env.width)
        checkpoint = tmp_path / 'fresh_untrained_tiebreak.pth'
        trainer.save_model(str(checkpoint))
        config.learned_heuristic_model_path = str(checkpoint)
    result = run_game_state_solver(env, algorithm, config)
    assert result.success == provide_key
    if provide_key:
        valid, reason, final, cost = StateSpaceAStar(env).verify_position_path(result.path)
        assert valid, reason
        assert final.keys == 0
        assert np.isfinite(cost)


def test_entity_spawner_reuses_existing_boss_key_and_item_tiles():
    from src.generation.entity_spawner import spawn_all_entities, materialize_entities_on_grid

    grid = np.full((H, W), P['WALL'], dtype=np.int64)
    grid[1:-1, 1:-1] = P['FLOOR']
    grid[3, 3] = P['KEY_BOSS']
    grid[5, 5] = P['KEY_ITEM']
    grid[7, 7] = P['BOSS']

    mission_boss = {'nodes': {0: {'type': 'BOSS', 'is_boss': True}}}
    entities_boss = spawn_all_entities(grid.copy(), mission_boss, {0: (0, 0, W - 1, H - 1)}, seed=42)
    mat_boss = materialize_entities_on_grid(grid, entities_boss)
    assert np.count_nonzero(mat_boss == P['BOSS']) == 1

    mission_key_item = {'nodes': {0: {'type': 'BIG_KEY', 'has_key': True, 'has_boss_key': True, 'key_id': 2, 'item_type': 'LAVA_PROTECTION'}}}
    entities_ki = spawn_all_entities(grid.copy(), mission_key_item, {0: (0, 0, W - 1, H - 1)}, seed=42)
    mat_ki = materialize_entities_on_grid(grid, entities_ki)
    assert np.count_nonzero(mat_ki == P['KEY_BOSS']) == 1
    assert np.count_nonzero(mat_ki == P['KEY_ITEM']) == 1


def test_non_soft_locked_enemy_does_not_bloat_defeated_enemies():
    env = passage('open')
    enemy_pos = (H // 2 - 1, W - 3)
    env.original_grid[enemy_pos] = P['ENEMY']
    env._build_room_enemy_cache()
    state = move(env, env.state.copy(), enemy_pos)
    assert enemy_pos not in state.defeated_enemies
