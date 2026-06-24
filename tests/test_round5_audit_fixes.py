import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from src.core.definitions import ROOM_HEIGHT, ROOM_TOPOLOGY_CHANNELS, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_WIDTH, TileID
from src.core.discrete_masked_model import create_discrete_masked_model
from src.data_processing.data_adapter import IntelligentDataAdapter, VGLCParser as AdapterVGLCParser
from src.evaluation.map_elites import LinearityLeniencyExtractor
from src.evaluation.benchmark_suite import extract_graph_descriptor
from src.evaluation.search_benchmark_utils import confusion_ratio_vs_oracle, normalized_confusion_ratio
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
from src.ml.logic_net import DifferentiableTortuosity, InventoryAwareLogicNet, SoftBellmanFord
from src.core.logic_net import DifferentiablePathfinder
from src.pipeline.room_stitching import carve_room_connection_between_bboxes
from src.simulation.validator import GameState, SEMANTIC_PALETTE, StateSpaceAStar, ZeldaLogicEnv
from src.zelda_data.stitching.graph_placement import find_boundary_doors
from src.zelda_data.parsers.core_parsers import VGLCParser


def test_output_node_cap_keeps_required_item_provider():
    gen = EvolutionaryTopologyGenerator(
        target_curve=[0.2, 0.6, 1.0],
        population_size=2,
        generations=1,
        max_nodes=2,
        seed=1,
    )
    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("key", type="ITEM", item_type="HOOKSHOT")
    graph.add_node("gate", type="ENEMY")
    graph.add_node("goal", type="GOAL")
    graph.add_node("extra", type="TREASURE")
    graph.add_edges_from(
        [
            ("start", "key", {"edge_type": "PATH"}),
            ("key", "gate", {"edge_type": "ITEM_GATE", "item_required": "HOOKSHOT"}),
            ("gate", "goal", {"edge_type": "PATH"}),
            ("start", "extra", {"edge_type": "PATH"}),
        ]
    )

    capped = gen._enforce_output_node_cap(graph)

    assert "key" in capped.nodes
    assert "gate" in capped.nodes
    assert "goal" in capped.nodes


def test_output_node_cap_keeps_generic_item_provider_for_item_gate_without_requirement():
    gen = EvolutionaryTopologyGenerator(
        target_curve=[0.2, 0.6, 1.0],
        population_size=2,
        generations=1,
        max_nodes=2,
        seed=1,
    )
    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("item", type="ITEM")
    graph.add_node("gate", type="ENEMY")
    graph.add_node("goal", type="GOAL")
    graph.add_node("extra", type="TREASURE")
    graph.add_edges_from(
        [
            ("start", "item", {"edge_type": "PATH"}),
            ("item", "gate", {"edge_type": "ITEM_GATE"}),
            ("gate", "goal", {"edge_type": "PATH"}),
            ("start", "extra", {"edge_type": "PATH"}),
        ]
    )

    capped = gen._enforce_output_node_cap(graph)

    assert "item" in capped.nodes
    assert "gate" in capped.nodes


def test_output_node_cap_keeps_token_provider_for_multi_lock():
    gen = EvolutionaryTopologyGenerator(
        target_curve=[0.2, 0.6, 1.0],
        population_size=2,
        generations=1,
        max_nodes=2,
        seed=1,
    )
    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("token", type="TOKEN")
    graph.add_node("gate", type="ENEMY")
    graph.add_node("goal", type="GOAL")
    graph.add_node("extra", type="TREASURE")
    graph.add_edges_from(
        [
            ("start", "token", {"edge_type": "PATH"}),
            ("token", "gate", {"edge_type": "MULTI_LOCK", "token_count": 1}),
            ("gate", "goal", {"edge_type": "PATH"}),
            ("start", "extra", {"edge_type": "PATH"}),
        ]
    )

    capped = gen._enforce_output_node_cap(graph)

    assert "token" in capped.nodes


def test_output_node_cap_preserves_directed_reachability_to_progression_nodes():
    gen = EvolutionaryTopologyGenerator(
        target_curve=[0.2, 0.6, 1.0],
        population_size=2,
        generations=1,
        max_nodes=2,
        seed=1,
    )
    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("bridge", type="EMPTY")
    graph.add_node("key", type="KEY")
    graph.add_node("goal", type="GOAL")
    graph.add_node("extra_1", type="TREASURE")
    graph.add_node("extra_2", type="TREASURE")
    graph.add_edges_from(
        [
            ("start", "bridge", {"edge_type": "PATH"}),
            ("bridge", "key", {"edge_type": "PATH"}),
            ("start", "goal", {"edge_type": "PATH"}),
            ("start", "extra_1", {"edge_type": "PATH"}),
            ("start", "extra_2", {"edge_type": "PATH"}),
        ]
    )

    capped = gen._enforce_output_node_cap(graph)

    assert "bridge" in capped
    assert nx.has_path(capped, "start", "key")


def test_output_connectivity_repairs_goal_only_component():
    graph = nx.DiGraph()
    graph.add_node("start", type="START", position=(0, 0, 0))
    graph.add_node("middle", type="ENEMY", position=(1, 0, 0))
    graph.add_node("goal", type="GOAL", position=(3, 0, 0))
    graph.add_edge("start", "middle", edge_type="PATH")

    repaired = EvolutionaryTopologyGenerator._repair_output_connectivity(graph)

    assert nx.is_connected(repaired.to_undirected())
    assert any("goal" in edge for edge in repaired.edges())


def test_output_connectivity_does_not_add_raw_path_into_goal_component_without_boss_anchor():
    graph = nx.DiGraph()
    graph.add_node("start", type="START", position=(0, 0, 0))
    graph.add_node("middle", type="ENEMY", position=(1, 0, 0))
    graph.add_node("goal", type="GOAL", position=(3, 0, 0))
    graph.add_edge("start", "middle", edge_type="PATH")

    repaired = EvolutionaryTopologyGenerator._repair_output_connectivity(graph)
    incoming = list(repaired.in_edges("goal", data=True))

    assert incoming
    assert all(str(data.get("edge_type", "")).upper() != "PATH" for _src, _dst, data in incoming)


def test_leniency_counts_boss_key_and_boss_lock():
    graph = nx.DiGraph()
    graph.add_node("s", type="START")
    graph.add_node("g", type="GOAL")
    graph.add_edge("s", "g", edge_type="BOSS_LOCKED")
    extractor = LinearityLeniencyExtractor()

    assert extractor._compute_leniency(graph) == pytest.approx(0.0)

    graph.add_node("bk", type="BIG_KEY")
    assert extractor._compute_leniency(graph) == pytest.approx(1.0)


def test_leniency_does_not_pool_boss_keys_with_small_locks():
    graph = nx.DiGraph()
    graph.add_node("s", type="START")
    graph.add_node("g", type="GOAL")
    graph.add_node("bk1", type="BIG_KEY")
    graph.add_node("bk2", type="BIG_KEY")
    graph.add_edge("s", "g", edge_type="LOCKED")
    graph.add_edge("g", "s", edge_type="MULTI_LOCK", token_count=1)
    extractor = LinearityLeniencyExtractor()

    assert extractor._compute_leniency(graph) == pytest.approx(0.0)


def test_leniency_counts_token_nodes_and_key_count_for_multi_locks():
    graph = nx.DiGraph()
    graph.add_node("s", type="START")
    graph.add_node("token_bundle", type="TOKEN", key_count=2)
    graph.add_node("g", type="GOAL")
    graph.add_edge("s", "g", edge_type="MULTI_LOCK", token_count=2)
    extractor = LinearityLeniencyExtractor()

    assert extractor._compute_leniency(graph) == pytest.approx(1.0)


def test_benchmark_leniency_matches_token_multi_lock_economy():
    graph = nx.DiGraph()
    graph.add_node("s", type="START")
    graph.add_node("token_bundle", type="TOKEN", key_count=2)
    graph.add_node("g", type="GOAL")
    graph.add_edge("s", "token_bundle", edge_type="open")
    graph.add_edge("token_bundle", "g", edge_type="MULTI_LOCK", token_count=2)

    descriptor = extract_graph_descriptor(graph)

    assert descriptor.leniency == pytest.approx(1.0)


def test_confusion_ratios_handle_zero_length_oracle_paths():
    assert confusion_ratio_vs_oracle(0, 0, oracle_status="solved", candidate_success=True) == pytest.approx(0.0)
    assert math.isfinite(confusion_ratio_vs_oracle(0, 3, oracle_status="solved", candidate_success=True))
    assert normalized_confusion_ratio(0, 0, 0, oracle_status="solved", candidate_success=True) == pytest.approx(0.0)
    assert normalized_confusion_ratio(0, 3, 0, oracle_status="solved", candidate_success=True) == pytest.approx(1.0)


def test_differentiable_tortuosity_accepts_winding_paths_longer_than_h_plus_w():
    probability = torch.full((1, 1, 5, 5), 1e-4)
    path = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (3, 2),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 3),
        (0, 4),
    ]
    for row, col in path:
        probability[0, 0, row, col] = 1.0
    module = DifferentiableTortuosity(num_iterations=20)

    path_lengths = module.compute_soft_path_length(probability, [(0, 0)], [(0, 4)])

    assert float(path_lengths.item()) > 10.0
    assert float(path_lengths.item()) < 26.0


def test_soft_bellman_ford_uses_grid_area_distance_sentinel():
    module = SoftBellmanFord(num_iterations=1, wall_penalty=20.0)
    probability = torch.zeros(1, 1, 8, 8)

    distances = module.distance_map(probability, [(0, 0)])

    assert float(distances.max().item()) > 20.0 * (8 + 8)


def test_differentiable_pathfinder_accepts_batched_graph_inputs():
    pathfinder = DifferentiablePathfinder(num_iterations=4, temperature=0.05, inf_distance=50.0)
    adjacency = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    weights = torch.where(adjacency > 0, torch.ones_like(adjacency), torch.full_like(adjacency, 50.0))
    source = torch.tensor([[True, False, False], [True, False, False]])

    distances = pathfinder(adjacency, weights, source)

    assert tuple(distances.shape) == (2, 3)
    assert float(distances[0, 2].item()) < 3.0
    assert float(distances[1, 2].item()) < 2.0


def test_inventory_logic_net_does_not_recount_the_same_key_each_stage():
    floor = torch.ones(1, 1, 1, 3)
    keys = torch.zeros_like(floor)
    keys[0, 0, 0, 0] = 1.0
    locked = torch.zeros_like(floor)
    locked[0, 0, 0, 2] = 1.0

    two_stage = InventoryAwareLogicNet(num_iterations=3, num_key_stages=2)
    four_stage = InventoryAwareLogicNet(num_iterations=3, num_key_stages=4)
    four_stage.load_state_dict(two_stage.state_dict())

    score_two = two_stage(floor, keys, locked, [(0, 0)], [(0, 2)])
    score_four = four_stage(floor, keys, locked, [(0, 0)], [(0, 2)])

    assert torch.allclose(score_two, score_four, atol=1e-6)


def test_state_space_astar_pareto_frontier_keeps_incomparable_inventory_states():
    grid = np.full((3, 3), SEMANTIC_PALETTE["FLOOR"], dtype=np.int64)
    grid[1, 1] = SEMANTIC_PALETTE["START"]
    grid[1, 2] = SEMANTIC_PALETTE["TRIFORCE"]
    solver = StateSpaceAStar(ZeldaLogicEnv(grid), timeout=32)
    base = GameState(position=(1, 1))
    has_key = base.copy()
    has_key.keys = 1
    has_bomb = base.copy()
    has_bomb.bomb_count = 1

    frontier = solver._add_to_pareto_frontier([], has_key, 1.0)
    frontier = solver._add_to_pareto_frontier(frontier, has_bomb, 1.0)

    assert len(frontier) == 2
    assert not solver._pareto_frontier_dominates(frontier[:1], has_bomb, 1.0)


def test_validator_collects_hidden_item_exposed_by_block_push_in_pure_and_mutable_paths():
    grid = np.full((3, 6), SEMANTIC_PALETTE["WALL"], dtype=np.int64)
    grid[1, 1:5] = SEMANTIC_PALETTE["FLOOR"]
    grid[1, 1] = SEMANTIC_PALETTE["START"]
    grid[1, 2] = SEMANTIC_PALETTE["BLOCK"]
    grid[1, 4] = SEMANTIC_PALETTE["TRIFORCE"]
    underlay = {(1, 2): int(SEMANTIC_PALETTE["KEY_SMALL"])}

    env = ZeldaLogicEnv(grid, block_underlay_tiles=underlay)
    state = env.state.copy()
    ok, state = env.try_move_pure(state, (1, 2), int(grid[1, 2]))
    assert ok
    assert state.keys == 1

    env = ZeldaLogicEnv(grid, block_underlay_tiles=underlay)
    env.state.position = (1, 1)
    ok, state, _reward, _info = env._try_move((1, 2), int(grid[1, 2]))
    assert ok
    assert state.keys == 1


def test_puzzle_stage_progress_updates_on_element_and_collected_item_tiles():
    grid = np.full((3, 6), SEMANTIC_PALETTE["WALL"], dtype=np.int64)
    grid[1, 1:5] = SEMANTIC_PALETTE["FLOOR"]
    grid[1, 1] = SEMANTIC_PALETTE["START"]
    grid[1, 2] = SEMANTIC_PALETTE["KEY_ITEM"]
    grid[1, 3] = SEMANTIC_PALETTE["ELEMENT"]
    grid[1, 4] = SEMANTIC_PALETTE["TRIFORCE"]
    metadata = {
        "plans": {
            "room_0": {
                "controlled_doors_global": [],
                "stage_sequence": [
                    {"stage_index": 0, "kind": "collect_item", "global_anchor": [1, 2]},
                    {"stage_index": 1, "kind": "step_on_element", "global_anchor": [1, 3], "trigger_tile_id": int(SEMANTIC_PALETTE["ELEMENT"])},
                ],
            }
        }
    }
    env = ZeldaLogicEnv(grid, room_puzzle_metadata=metadata)
    state = env.state.copy()
    ok, state = env.try_move_pure(state, (1, 2), int(grid[1, 2]))
    assert ok
    assert ("room_0", 0) in state.completed_puzzle_stages
    ok, state = env.try_move_pure(state, (1, 3), int(grid[1, 3]))
    assert ok
    assert ("room_0", 1) in state.completed_puzzle_stages

    replay_state = state.copy()
    replay_state.completed_puzzle_stages = set()
    ok, replay_state = env.try_move_pure(replay_state, (1, 2), int(grid[1, 2]))
    assert ok
    assert ("room_0", 0) in replay_state.completed_puzzle_stages


def test_edge_aware_logit_bias_accumulates_corner_evidence():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=16,
        context_dim=16,
        num_steps=2,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=4,
    )
    logits = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    topology = torch.zeros(1, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH)
    topology[:, ROOM_TOPOLOGY_CHANNELS["door_n"], 0, :] = 1.0
    topology[:, ROOM_TOPOLOGY_CHANNELS["door_w"], :, 0] = 1.0

    biased = model._apply_edge_aware_logit_bias(logits, {"room_topology_map": topology}, bias_strength=2.0)

    corner_door_delta = float(biased[0, int(TileID.DOOR_OPEN), 0, 0].item())
    corner_wall_delta = float(biased[0, int(TileID.WALL), 0, 0].item())
    assert corner_door_delta == pytest.approx(4.0)
    assert corner_wall_delta == pytest.approx(-4.0)


def test_edge_aware_logit_bias_uses_gate_family_semantics():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=16,
        context_dim=16,
        num_steps=2,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=4,
    )
    logits = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    topology = torch.zeros(1, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH)
    topology[:, ROOM_TOPOLOGY_CHANNELS["door_n"], 0, 4:7] = 1.0
    topology[:, ROOM_TOPOLOGY_CHANNELS["gate_key_n"], 0, 4:7] = 1.0

    biased = model._apply_edge_aware_logit_bias(logits, {"room_topology_map": topology}, bias_strength=2.0)

    locked_delta = float(biased[0, int(TileID.DOOR_LOCKED), 0, 5].item())
    open_delta = float(biased[0, int(TileID.DOOR_OPEN), 0, 5].item())
    wall_delta = float(biased[0, int(TileID.WALL), 0, 5].item())
    assert locked_delta > open_delta
    assert locked_delta == pytest.approx(4.0)
    assert open_delta == pytest.approx(1.0)
    assert wall_delta == pytest.approx(-2.0)


def test_fixed_token_logits_preserve_editable_class_zero_logits():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=16,
        context_dim=16,
        num_steps=2,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=4,
    )
    logits = torch.randn(1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    original = logits.clone()
    fixed_tokens = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.long)
    fixed_tokens[0, 0, 0] = int(TileID.DOOR_OPEN)
    fixed_mask = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.bool)
    fixed_mask[0, 0, 0] = True

    constrained = model._apply_fixed_token_logits(logits, fixed_tokens=fixed_tokens, fixed_mask=fixed_mask)

    assert torch.equal(constrained[:, :, 1:, 1:], original[:, :, 1:, 1:])
    assert constrained[0, int(TileID.DOOR_OPEN), 0, 0] > 1000.0
    assert constrained[0, int(TileID.VOID), 0, 0] < -1000.0


def test_quest2_graph_lookup_prefers_vglc_loz2_naming(tmp_path: Path):
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    canonical = graph_dir / "LoZ2_3.dot"
    legacy = graph_dir / "LoZ_3_q2.dot"
    canonical.write_text("digraph {}", encoding="utf-8")
    legacy.write_text("digraph {}", encoding="utf-8")

    adapter = object.__new__(IntelligentDataAdapter)
    adapter.graph_dir = graph_dir

    assert adapter._find_graph_file("tloz3_2") == canonical


def test_core_parser_detects_outer_and_inner_wall_doors():
    parser = VGLCParser(room_cls=object)
    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), "W", dtype="<U1")
    grid[0, 4:7] = "D"
    grid[1, 4:7] = "W"
    grid[14, 4:7] = "D"
    grid[15, 4:7] = "W"

    doors = parser._detect_doors(grid)

    assert doors["N"] is True
    assert doors["S"] is True


@pytest.mark.parametrize("glyph", ["D", "d", "F", "f", "."])
def test_core_parser_detects_all_open_boundary_door_glyphs(glyph: str):
    parser = VGLCParser(room_cls=object)
    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), "W", dtype="<U1")
    grid[0, 4:7] = glyph

    doors = parser._detect_doors(grid)

    assert doors["N"] is True


def test_data_adapter_fills_interior_voids_for_rooms_with_doors():
    parser = AdapterVGLCParser()
    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), "-", dtype="<U1")
    grid[0, :] = "W"
    grid[-1, :] = "W"
    grid[:, 0] = "W"
    grid[:, -1] = "W"
    grid[0, 4:7] = "D"
    doors = parser._detect_doors(grid)

    semantic = parser._chars_to_semantic(grid, doors=doors)

    assert np.all(semantic[2:14, 2:9] != int(TileID.VOID))


def test_data_adapter_vglc_parser_uses_centralized_core_parser_semantics():
    parser = AdapterVGLCParser()
    assert isinstance(parser._core, VGLCParser)

    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), "-", dtype="<U1")
    grid[0, :] = "W"
    grid[-1, :] = "W"
    grid[:, 0] = "W"
    grid[:, -1] = "W"
    grid[0, 4:7] = "d"
    adapter_doors = parser._detect_doors(grid)
    core_doors = parser._core._detect_doors(grid)

    assert adapter_doors == {"N": "open"}
    np.testing.assert_array_equal(
        parser._chars_to_semantic(grid, adapter_doors),
        parser._core._to_semantic(grid, core_doors),
    )


def test_graph_placement_boundary_doors_use_canonical_positions():
    grid = np.zeros((ROOM_HEIGHT * 2, ROOM_WIDTH * 2), dtype=np.int32)

    north = find_boundary_doors(grid, (ROOM_HEIGHT, 0), (0, 0), (1, 0), (0, 0), ROOM_HEIGHT, ROOM_WIDTH)
    south = find_boundary_doors(grid, (0, 0), (ROOM_HEIGHT, 0), (0, 0), (1, 0), ROOM_HEIGHT, ROOM_WIDTH)
    west = find_boundary_doors(grid, (0, ROOM_WIDTH), (0, 0), (0, 1), (0, 0), ROOM_HEIGHT, ROOM_WIDTH)
    east = find_boundary_doors(grid, (0, 0), (0, ROOM_WIDTH), (0, 0), (0, 1), ROOM_HEIGHT, ROOM_WIDTH)

    assert north == [(ROOM_HEIGHT, 4), (ROOM_HEIGHT, 5), (ROOM_HEIGHT, 6)]
    assert south == [(ROOM_HEIGHT - 1, 4), (ROOM_HEIGHT - 1, 5), (ROOM_HEIGHT - 1, 6)]
    assert west == [(7, ROOM_WIDTH), (8, ROOM_WIDTH)]
    assert east == [(7, ROOM_WIDTH - 1), (8, ROOM_WIDTH - 1)]


def test_stitcher_uses_canonical_door_widths_for_adjacent_rooms():
    grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH * 2), dtype=np.int32)
    carve_room_connection_between_bboxes(
        grid,
        (0, 0, ROOM_WIDTH - 1, ROOM_HEIGHT - 1),
        (ROOM_WIDTH, 0, ROOM_WIDTH * 2 - 1, ROOM_HEIGHT - 1),
    )

    open_rows = np.flatnonzero(grid[:, ROOM_WIDTH - 1] != 0)
    assert open_rows.tolist() == [7, 8]
