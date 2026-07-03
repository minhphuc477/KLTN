from __future__ import annotations

from types import SimpleNamespace

import pytest
import networkx as nx
import numpy as np

from src.core.definitions import (
    CHAR_TO_SEMANTIC,
    ID_TO_NAME,
    ROOM_TOPOLOGY_CHANNELS,
    SEMANTIC_PALETTE,
    SEMANTIC_TO_CHAR,
    TileID,
    semantic_grid_to_vglc_lines,
)
from src.core.logic_net import ROOM_TOPOLOGY_CHANNELS as LOGICNET_ROOM_TOPOLOGY_CHANNELS
from src.evaluation.pcbs_validation import (
    PCBS_BOUNDED_RATIONALITY_WEIGHTS,
    PCBS_READABILITY_WEIGHT_SOURCE,
    compute_pcbs_readability_metrics,
)
from src.evaluation.validator import AgentSimulator, SolvabilityChecker, ValidationState
from src.simulation.validation_helpers import SanityChecker
from src.simulation.validator import GameState, StateSpaceAStar, ZeldaLogicEnv


def test_semantic_export_preserves_entity_tiles_roundtrip():
    grid = np.asarray(
        [
            [
                int(TileID.START),
                int(TileID.TRIFORCE),
                int(TileID.KEY_SMALL),
                int(TileID.KEY_BOSS),
                int(TileID.KEY_ITEM),
                int(TileID.ITEM_MINOR),
                int(TileID.BOSS),
            ]
        ],
        dtype=np.int32,
    )

    line = semantic_grid_to_vglc_lines(grid)[0]
    restored = np.asarray([[int(CHAR_TO_SEMANTIC[ch]) for ch in line]], dtype=np.int32)

    assert restored.tolist() == grid.tolist()
    assert SEMANTIC_TO_CHAR[int(TileID.START)] != "F"
    assert SEMANTIC_TO_CHAR[int(TileID.TRIFORCE)] != "F"
    assert SEMANTIC_TO_CHAR[int(TileID.KEY_SMALL)] != "F"
    assert ID_TO_NAME[int(TileID.KEY_SMALL)] == "KEY_SMALL"
    assert ID_TO_NAME[int(TileID.ITEM_MINOR)] == "ITEM_MINOR"


def test_semantic_export_preserves_puzzle_vs_element_roundtrip():
    grid = np.asarray(
        [[int(TileID.ELEMENT), int(TileID.PUZZLE), int(TileID.ELEMENT_FLOOR)]],
        dtype=np.int32,
    )

    line = semantic_grid_to_vglc_lines(grid)[0]
    restored = np.asarray([[int(CHAR_TO_SEMANTIC[ch]) for ch in line]], dtype=np.int32)

    assert SEMANTIC_TO_CHAR[int(TileID.ELEMENT)] != SEMANTIC_TO_CHAR[int(TileID.PUZZLE)]
    assert line == "PXO"
    assert restored.tolist() == grid.tolist()


def test_pipeline_checkpoint_loader_requests_weights_only(monkeypatch, tmp_path):
    import torch
    from src.pipeline.runtime import _load_checkpoint_and_metadata

    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"model_state_dict": {}}

    monkeypatch.setattr(torch, "load", fake_load)
    checkpoint_path = tmp_path / "checkpoint.pth"

    checkpoint, metadata = _load_checkpoint_and_metadata(
        SimpleNamespace(device="cpu", strict_checkpoint_mode=False),
        str(checkpoint_path),
        "test-model",
    )

    assert checkpoint == {"model_state_dict": {}}
    assert metadata == {}
    assert calls[0][1]["weights_only"] is True


def test_safe_torch_load_fails_closed_when_weights_only_is_unsupported(monkeypatch, tmp_path):
    import torch
    from src.utils.checkpoint import ALLOW_LEGACY_TORCH_LOAD_ENV, safe_torch_load

    def fake_load(*args, **kwargs):
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return {"unsafe": True}

    monkeypatch.delenv(ALLOW_LEGACY_TORCH_LOAD_ENV, raising=False)
    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(RuntimeError, match="Refusing unsafe legacy torch.load"):
        safe_torch_load(tmp_path / "checkpoint.pth", map_location="cpu")


def test_safe_torch_load_allows_legacy_only_with_explicit_trusted_opt_in(monkeypatch, tmp_path):
    import torch
    from src.utils.checkpoint import ALLOW_LEGACY_TORCH_LOAD_ENV, safe_torch_load

    calls = []

    def fake_load(*args, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return {"trusted_local": True}

    monkeypatch.setenv(ALLOW_LEGACY_TORCH_LOAD_ENV, "1")
    monkeypatch.setattr(torch, "load", fake_load)

    assert safe_torch_load(tmp_path / "checkpoint.pth", map_location="cpu") == {"trusted_local": True}
    assert calls[0]["weights_only"] is True
    assert "weights_only" not in calls[1]


def test_safe_torch_load_preserves_unrelated_type_errors(monkeypatch, tmp_path):
    import torch
    from src.utils.checkpoint import ALLOW_LEGACY_TORCH_LOAD_ENV, safe_torch_load

    def fake_load(*args, **kwargs):
        raise TypeError("corrupt checkpoint payload")

    monkeypatch.setenv(ALLOW_LEGACY_TORCH_LOAD_ENV, "1")
    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(TypeError, match="corrupt checkpoint payload"):
        safe_torch_load(tmp_path / "checkpoint.pth", map_location="cpu")


def test_ml_heuristic_astar_does_not_construct_trainer_per_query(monkeypatch):
    from src.ml import heuristic_learning
    from src.ml.heuristic_learning import MLHeuristicAStar

    grid = np.full((5, 5), SEMANTIC_PALETTE["FLOOR"], dtype=np.int64)
    grid[1, 1] = SEMANTIC_PALETTE["START"]
    grid[3, 3] = SEMANTIC_PALETTE["TRIFORCE"]
    env = ZeldaLogicEnv(grid)

    class DummyModel:
        def predict_cost(self, features):
            assert features.shape == (10,)
            return 2.0

    def fail_init(*args, **kwargs):
        raise AssertionError("HeuristicTrainer must not be constructed during heuristic inference")

    solver = MLHeuristicAStar(env)
    solver.model = DummyModel()
    monkeypatch.setattr(heuristic_learning.HeuristicTrainer, "__init__", fail_init)

    assert solver.heuristic(env.state.copy()) == pytest.approx(2.0)


def test_graph_linearity_uses_path_directness_not_optional_branch_coverage():
    from src.evaluation.benchmark_suite import extract_graph_descriptor

    graph = nx.DiGraph()
    graph.add_node(0, label="s", type="START", position=(0, 0))
    graph.add_node(1, type="EMPTY", position=(1, 0))
    graph.add_node(2, label="t", type="GOAL", position=(2, 0))
    graph.add_edge(0, 1, edge_type="open")
    graph.add_edge(1, 2, edge_type="open")
    for idx in range(3, 10):
        graph.add_node(idx, type="EMPTY", position=(1, idx))
        graph.add_edge(1, idx, edge_type="open")

    descriptor = extract_graph_descriptor(graph)

    assert descriptor.path_exists is True
    assert descriptor.linearity == pytest.approx(1.0)


def test_hazard_edges_require_the_permanent_protection_item():
    from src.simulation.edge_logic import can_traverse_edge_type, edge_type_from_data

    state = SimpleNamespace(position=(0, 0), keys=0, bomb_count=0, has_boss_key=False, has_item=False)

    assert edge_type_from_data({"edge_type": "hazard"}) == "hazard"
    assert not can_traverse_edge_type(
        "hazard",
        state,
        strict_original_mode=True,
        get_room_for_position=lambda _pos: None,
        is_room_cleared=lambda _room, _state: False,
    )
    state.has_item = True
    assert can_traverse_edge_type(
        "hazard",
        state,
        strict_original_mode=True,
        get_room_for_position=lambda _pos: None,
        is_room_cleared=lambda _room, _state: False,
    )


def test_key_economy_rejects_key_trapped_behind_its_own_lock_with_cyclic_start():
    from src.simulation.key_economy_validator import KeyEconomyValidator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("cycle")
    graph.add_node("locked_room", key_id="small")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "cycle", lock_type="open")
    graph.add_edge("cycle", "start", lock_type="open")
    graph.add_edge("start", "locked_room", lock_type="locked", key_id="small")
    graph.add_edge("locked_room", "goal", lock_type="open")

    result = KeyEconomyValidator(graph).validate()

    assert result.is_valid is False
    assert result.key_surplus["start->locked_room"] == -1


def test_key_economy_respects_edge_type_locked_schema():
    from src.simulation.key_economy_validator import KeyEconomyValidator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("locked_room", key_id="small")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "locked_room", edge_type="key_locked", key_id="small")
    graph.add_edge("locked_room", "goal", edge_type="open")

    result = KeyEconomyValidator(graph).validate()

    assert result.is_valid is False
    assert result.key_surplus["start->locked_room"] == -1


def test_key_economy_rejects_item_gate_with_item_locked_behind_gate():
    from src.simulation.key_economy_validator import KeyEconomyValidator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("item_room", type="ITEM", item_type="HOOKSHOT")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "item_room", edge_type="item_gate", item_required="HOOKSHOT")
    graph.add_edge("item_room", "goal", edge_type="open")

    result = KeyEconomyValidator(graph).validate()

    assert result.greedy_solvable is False
    assert result.is_valid is False


def test_key_economy_accepts_specific_boss_key_identifier():
    from src.simulation.key_economy_validator import KeyEconomyValidator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("big_key", type="BIG_KEY", key_id=7)
    graph.add_node("boss_door", type="BOSS_DOOR")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "big_key", edge_type="open")
    graph.add_edge("big_key", "boss_door", edge_type="boss_locked", key_required=7)
    graph.add_edge("boss_door", "goal", edge_type="open")

    result = KeyEconomyValidator(graph).validate()

    assert result.greedy_solvable
    assert result.adversarial_solvable
    assert result.is_valid


def test_key_economy_infers_boss_key_from_target_boss_door_node():
    from src.simulation.key_economy_validator import KeyEconomyValidator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("big_key", type="BIG_KEY", key_id=7)
    graph.add_node("boss_door", type="BOSS_DOOR", key_id=7)
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "big_key", edge_type="open")
    graph.add_edge("big_key", "boss_door", edge_type="boss_locked")
    graph.add_edge("boss_door", "goal", edge_type="open")

    result = KeyEconomyValidator(graph).validate()

    assert result.greedy_solvable
    assert result.adversarial_solvable
    assert result.is_valid


def test_key_economy_infers_item_gate_requirement_from_target_node():
    from src.simulation.key_economy_validator import KeyEconomyValidator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("item", type="ITEM", item_type="HOOKSHOT")
    graph.add_node("gate", type="PUZZLE", required_item="HOOKSHOT")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "item", edge_type="open")
    graph.add_edge("item", "gate", edge_type="item_gate")
    graph.add_edge("gate", "goal", edge_type="open")

    result = KeyEconomyValidator(graph).validate()

    assert result.greedy_solvable
    assert result.adversarial_solvable
    assert result.is_valid


def test_key_economy_does_not_collect_required_key_from_lock_nodes():
    from src.simulation.key_economy_validator import KeyEconomyValidator

    graph = nx.DiGraph()
    graph.add_node("start", type="START", items=["small_1"])
    graph.add_node("lock_1", type="LOCK", key_id="small_2")
    graph.add_node("lock_2", type="LOCK")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "lock_1", edge_type="key_locked", key_id="small_1")
    graph.add_edge("lock_1", "lock_2", edge_type="key_locked", key_id="small_2")
    graph.add_edge("lock_2", "goal", edge_type="open")

    result = KeyEconomyValidator(graph).validate()

    assert result.greedy_solvable is False
    assert result.is_valid is False


def test_map_elites_progression_does_not_collect_required_key_from_lock_nodes():
    from src.simulation.map_elites import MAPElitesEvaluator

    graph = nx.DiGraph()
    graph.add_node("start", type="START", items=["small_1"])
    graph.add_node("lock_1", type="LOCK", key_id="small_2")
    graph.add_node("lock_2", type="LOCK")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "lock_1", edge_type="key_locked", key_required="small_1")
    graph.add_edge("lock_1", "lock_2", edge_type="key_locked", key_required="small_2")
    graph.add_edge("lock_2", "goal", edge_type="open")

    assert MAPElitesEvaluator._progression_feasible_path(graph, "start", "goal") is None


def test_map_elites_progression_infers_gate_key_from_target_lock_node():
    from src.simulation.map_elites import MAPElitesEvaluator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("key", type="KEY", key_id=7)
    graph.add_node("lock", type="LOCK", key_id=7)
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "key", edge_type="open")
    graph.add_edge("key", "lock", edge_type="key_locked")
    graph.add_edge("lock", "goal", edge_type="open")

    result = MAPElitesEvaluator._progression_feasible_path(graph, "start", "goal")

    assert result is not None
    path, _inventory, keys_collected, keys_consumed = result
    assert path == ["start", "key", "lock", "goal"]
    assert keys_collected == 1
    assert keys_consumed == 1


def test_external_validator_infers_boss_key_from_target_boss_door_node():
    from src.evaluation.validator import AgentSimulator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("big_key", type="BIG_KEY", key_id=7)
    graph.add_node("boss_door", type="BOSS_DOOR", key_id=9)
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "big_key", edge_type="open")
    graph.add_edge("big_key", "boss_door", edge_type="boss_locked")
    graph.add_edge("boss_door", "goal", edge_type="open")

    assert AgentSimulator(graph).find_path().is_solvable is False

    graph.nodes["boss_door"]["key_id"] = 7
    assert AgentSimulator(graph).find_path().is_solvable is True


def test_external_validator_infers_item_gate_requirement_from_target_node():
    from src.evaluation.validator import AgentSimulator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("item", type="ITEM", item_type="RAFT")
    graph.add_node("gate", type="PUZZLE", required_item="BOW")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "item", edge_type="open")
    graph.add_edge("item", "gate", edge_type="item_gate")
    graph.add_edge("gate", "goal", edge_type="open")

    assert AgentSimulator(graph).find_path().is_solvable is False

    graph.nodes["item"]["item_type"] = "BOW"
    assert AgentSimulator(graph).find_path().is_solvable is True


def test_external_validator_does_not_collect_required_item_as_provider():
    from src.evaluation.validator import AgentSimulator

    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("consumer_metadata", type="ITEM", required_item="BOW")
    graph.add_node("gate", type="PUZZLE", required_item="BOW")
    graph.add_node("goal", type="GOAL")
    graph.add_edge("start", "consumer_metadata", edge_type="open")
    graph.add_edge("consumer_metadata", "gate", edge_type="item_gate")
    graph.add_edge("gate", "goal", edge_type="open")

    assert AgentSimulator(graph).find_path().is_solvable is False


def test_src_ml_logicnet_exports_canonical_block_v_logicnet():
    from src.core.logic_net import LogicNet as CoreLogicNet
    from src.ml.logic_net import LegacyLogicNet, LogicNet as MlLogicNet

    assert MlLogicNet is CoreLogicNet
    assert LegacyLogicNet is not CoreLogicNet


def test_local_stream_pools_boundary_facing_neighbor_edges():
    import torch
    from src.core.condition_encoder import LocalStreamEncoder

    encoder = LocalStreamEncoder(latent_dim=1, hidden_dim=4, output_dim=4)
    latent = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)

    assert encoder._pool_neighbor_latent(latent, "N").item() == torch.tensor([8, 9, 10, 11], dtype=torch.float32).mean().item()
    assert encoder._pool_neighbor_latent(latent, "S").item() == torch.tensor([0, 1, 2, 3], dtype=torch.float32).mean().item()
    assert encoder._pool_neighbor_latent(latent, "E").item() == torch.tensor([0, 4, 8], dtype=torch.float32).mean().item()
    assert encoder._pool_neighbor_latent(latent, "W").item() == torch.tensor([3, 7, 11], dtype=torch.float32).mean().item()


def test_fallback_gnn_uses_sparse_edge_index_and_handles_isolated_nodes():
    import torch
    from src.core.condition_encoder import FallbackGNN

    gnn = FallbackGNN(node_dim=3, hidden_dim=5, output_dim=7, num_layers=1)
    node_features = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    out = gnn(node_features, edge_index)

    assert out.shape == (4, 7)
    assert torch.isfinite(out).all()


def test_dataset_split_manifest_records_hashes_and_split(tmp_path):
    from scripts.lock_dataset_split import build_manifest

    data_root = tmp_path / "zelda"
    (data_root / "Processed").mkdir(parents=True)
    (data_root / "Processed" / "tloz1_1.txt").write_bytes(b"abc\n")
    (data_root / "Original").mkdir()
    (data_root / "Original" / "source.txt").write_text("source\n", encoding="utf-8")

    manifest = build_manifest(data_root, train_dungeon_ids=[1, 2], test_dungeon_ids=[9])

    assert manifest["split"]["train_dungeon_ids"] == [1, 2]
    assert manifest["split"]["test_dungeon_ids"] == [9]
    assert manifest["file_count"] == 2
    by_path = {record["path"]: record for record in manifest["files"]}
    assert by_path["Processed/tloz1_1.txt"]["sha256"] == "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb"


def test_logicnet_uses_canonical_room_topology_channel_schema():
    assert LOGICNET_ROOM_TOPOLOGY_CHANNELS == ROOM_TOPOLOGY_CHANNELS
    assert len(LOGICNET_ROOM_TOPOLOGY_CHANNELS) > 11
    assert "role_tutorial_puzzle" in LOGICNET_ROOM_TOPOLOGY_CHANNELS


def test_agent_simulator_closed_set_uses_full_state_not_raw_hash(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node(0, label="s")
    graph.add_node(1, label="k")
    graph.add_node(2, label="")
    graph.add_node(3, label="t")
    graph.add_edge(0, 2, edge_type="open")
    graph.add_edge(0, 1, edge_type="open")
    graph.add_edge(1, 2, edge_type="open")
    graph.add_edge(2, 3, edge_type="key_locked")

    monkeypatch.setattr(ValidationState, "__hash__", lambda _self: 0)

    result = AgentSimulator(graph).find_path(max_states=100)

    assert result.is_solvable
    assert result.solution_path[-1] == 3


def test_agent_simulator_heuristic_uses_bound_undirected_cache(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node(0, label="s")
    graph.add_node(1, label="")
    graph.add_node(2, label="t")
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    simulator = AgentSimulator(graph)

    monkeypatch.setattr(
        simulator.graph,
        "to_undirected",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("uncached to_undirected call")),
    )

    assert simulator.heuristic(0, 2) == 2.0
    assert simulator.heuristic(1, 2) == 1.0


def test_agent_simulator_explicit_zero_node_override_is_not_treated_as_missing():
    graph = nx.DiGraph()
    graph.add_node(0, label="s")
    graph.add_node(1, label="")
    graph.add_node(2, label="t")
    graph.add_edge(0, 1, edge_type="open")
    graph.add_edge(1, 2, edge_type="open")

    simulator = AgentSimulator(graph)
    simulator.start_node = 1

    result = simulator.find_path(start_node=0, goal_node=2, max_states=20)

    assert result.is_solvable
    assert result.solution_path == [0, 1, 2]


def test_reachability_check_explicit_zero_node_override_is_not_treated_as_missing():
    graph = nx.DiGraph()
    graph.add_node(0, label="s")
    graph.add_node(1, label="")
    graph.add_node(2, label="t")
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    checker = SolvabilityChecker()
    ok, unreachable = checker.check_all_rooms_reachable(graph, start_node=0)

    assert ok
    assert unreachable == set()


def test_pcbs_readability_weights_are_reported_as_hand_tuned_metadata():
    metrics = compute_pcbs_readability_metrics(
        oracle={"success": True, "path_length": 10},
        pcbs_success=True,
        pcbs_solution_length=12,
        pcbs_trajectory_length=20,
        pcbs_states=10,
        timeout_pcbs=100,
        pcbs_metrics=SimpleNamespace(
            total_steps=20,
            unique_tiles_visited=15,
            confusion_index=0.5,
            navigation_entropy=0.4,
            cognitive_load=0.3,
        ),
        puzzle_stall_steps=2,
    )

    assert metrics["weight_source"] == PCBS_READABILITY_WEIGHT_SOURCE
    assert metrics["bounded_rationality_weights"] == PCBS_BOUNDED_RATIONALITY_WEIGHTS


def test_sanity_checker_rejects_sparse_walkable_maps_that_old_threshold_allowed():
    grid = np.full((10, 10), int(TileID.WALL), dtype=np.int32)
    walkable = [
        (0, 0, TileID.START),
        (0, 1, TileID.FLOOR),
        (0, 2, TileID.FLOOR),
        (0, 3, TileID.FLOOR),
        (0, 4, TileID.FLOOR),
        (0, 5, TileID.TRIFORCE),
    ]
    for row, col, tile_id in walkable:
        grid[row, col] = int(tile_id)

    is_valid, errors = SanityChecker(grid).check_all()

    assert not is_valid
    assert any("mostly blocked" in error for error in errors)


def test_state_domination_requires_identical_pushed_block_geometry():
    from src.simulation.validator import GameState, dominates

    unpushed = GameState(position=(2, 2))
    pushed = GameState(position=(2, 2), pushed_blocks={((2, 3), (2, 4))})

    assert not dominates(unpushed, pushed)
    assert not dominates(pushed, unpushed)
    assert dominates(pushed, pushed.copy())


def test_state_domination_does_not_prune_saved_consumable_pickup():
    from src.simulation.validator import GameState, dominates

    consumed = GameState(
        position=(2, 2),
        keys=0,
        collected_items={(1, 1)},
    )
    saved = GameState(position=(2, 2), keys=0)

    assert not dominates(consumed, saved)
    assert not dominates(saved, consumed)


def test_map_elites_leniency_counts_bosses_as_hazards():
    from src.simulation.map_elites import MAPElitesEvaluator

    grid = np.full((3, 3), SEMANTIC_PALETTE["FLOOR"], dtype=np.int64)
    grid[1, 1] = SEMANTIC_PALETTE["BOSS"]

    leniency = MAPElitesEvaluator().calculate_leniency(grid)

    assert leniency < 1.0
    assert leniency == np.clip(1.0 - (1.0 / 8.0), 0.0, 1.0)


def test_structural_dead_ends_exclude_canonical_start_and_goal():
    from src.evaluation.structural_metrics import analyze_structural_topology

    graph = nx.DiGraph()
    graph.add_node(0, label="start")
    graph.add_node(1, label="middle")
    graph.add_node(2, label="goal")
    graph.add_edges_from([(0, 1), (1, 2)])

    metrics = analyze_structural_topology(graph)

    assert metrics.dead_end_ratio == 0.0


def test_structural_branching_factor_averages_all_non_terminal_nodes():
    from src.evaluation.structural_metrics import compute_branching_factor

    graph = nx.DiGraph()
    graph.add_node(0, label="start")
    graph.add_node(6, label="goal")
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7)])

    branching = compute_branching_factor(graph)

    assert 0.0 < branching < 2.0


def test_structural_metrics_collapse_reciprocal_corridor_arcs():
    from src.evaluation.structural_metrics import (
        analyze_structural_topology,
        compute_branching_factor,
        compute_cyclomatic_complexity,
    )

    graph = nx.DiGraph()
    graph.add_node(0, label="start")
    graph.add_node(1, label="middle")
    graph.add_node(2, label="goal")
    graph.add_edges_from([(0, 1), (1, 0), (1, 2), (2, 1)])

    assert compute_cyclomatic_complexity(graph) == 0.0
    assert compute_branching_factor(graph) == pytest.approx(1.0)
    assert analyze_structural_topology(graph).dead_end_ratio == 0.0


def test_perturb_and_map_astar_uses_min_cost_admissible_heuristic():
    from src.evaluation.perturb_and_map import _astar

    costs = np.asarray(
        [
            [0.0, 100.0, 0.0001],
            [0.0001, 0.0001, 0.0001],
        ],
        dtype=np.float32,
    )
    result = _astar(
        costs,
        np.ones_like(costs, dtype=bool),
        start=(0, 0),
        goal=(0, 2),
        min_step_cost=0.0001,
    )

    assert result is not None
    _, path = result
    assert path is not None
    assert (1, 1) in path
    assert (0, 1) not in path


def test_skill_chain_validator_accepts_valid_longer_path_not_only_shortest():
    from src.generation.grammar.graph_types import EdgeType, MissionGraph, MissionNode, NodeType
    from src.generation.grammar_validators import validate_skill_chains

    graph = MissionGraph()
    for node in [
        MissionNode(0, NodeType.ITEM),
        MissionNode(1, NodeType.TUTORIAL_PUZZLE, is_tutorial=True, difficulty=0.1),
        MissionNode(2, NodeType.GOAL),
        MissionNode(3, NodeType.COMBAT_PUZZLE, difficulty=0.4),
        MissionNode(4, NodeType.COMPLEX_PUZZLE, difficulty=0.8),
    ]:
        graph.add_node(node)
    graph.add_edge(0, 1, EdgeType.PATH)
    graph.add_edge(1, 2, EdgeType.PATH)
    graph.add_edge(1, 3, EdgeType.PATH)
    graph.add_edge(3, 4, EdgeType.PATH)
    graph.add_edge(4, 2, EdgeType.PATH)

    assert validate_skill_chains(graph)


def test_resource_loop_validator_accepts_one_reachable_provider_before_gate():
    from src.generation.grammar.graph_types import EdgeType, MissionGraph, MissionNode, NodeType
    from src.generation.grammar_validators import validate_resource_loops

    graph = MissionGraph()
    for node in [
        MissionNode(0, NodeType.START),
        MissionNode(1, NodeType.RESOURCE_FARM, drops_resource="BOMB"),
        MissionNode(2, NodeType.LOCK),
        MissionNode(3, NodeType.GOAL),
        MissionNode(4, NodeType.RESOURCE_FARM, drops_resource="BOMB"),
    ]:
        graph.add_node(node)
    graph.add_edge(0, 1, EdgeType.PATH)
    graph.add_edge(1, 2, EdgeType.ITEM_GATE, item_required="BOMB")
    graph.add_edge(2, 3, EdgeType.PATH)
    graph.add_edge(3, 4, EdgeType.PATH)

    assert validate_resource_loops(graph)


def test_resource_loop_validator_rejects_mutual_item_gate_softlock():
    from src.generation.grammar.graph_types import EdgeType, MissionGraph, MissionNode, NodeType
    from src.generation.grammar_validators import validate_resource_loops

    graph = MissionGraph()
    for node in [
        MissionNode(0, NodeType.START),
        MissionNode(1, NodeType.RESOURCE_FARM, drops_resource="BOMB"),
        MissionNode(2, NodeType.RESOURCE_FARM, drops_resource="HOOKSHOT"),
        MissionNode(3, NodeType.GOAL),
    ]:
        graph.add_node(node)
    graph.add_edge(0, 1, EdgeType.ITEM_GATE, item_required="HOOKSHOT")
    graph.add_edge(0, 2, EdgeType.ITEM_GATE, item_required="BOMB")
    graph.add_edge(1, 3, EdgeType.PATH)
    graph.add_edge(2, 3, EdgeType.PATH)

    assert not validate_resource_loops(graph)


def test_progression_validator_rejects_mutual_lock_key_softlock():
    from src.generation.grammar.graph_types import EdgeType, MissionGraph, MissionNode, NodeType
    from src.generation.grammar.mission_grammar import MissionGrammar

    graph = MissionGraph()
    for node in [
        MissionNode(0, NodeType.START),
        MissionNode(1, NodeType.KEY, key_id=1),
        MissionNode(2, NodeType.KEY, key_id=2),
        MissionNode(3, NodeType.GOAL),
    ]:
        graph.add_node(node)
    graph.add_edge(0, 1, EdgeType.LOCKED, key_required=2)
    graph.add_edge(0, 2, EdgeType.LOCKED, key_required=1)
    graph.add_edge(1, 3, EdgeType.PATH)
    graph.add_edge(2, 3, EdgeType.PATH)

    grammar = MissionGrammar(seed=7)

    assert not grammar.validate_progression_constraints(graph, log_failures=False)


def test_locked_mission_edges_are_bidirectional_for_physical_traversal():
    from src.generation.grammar.graph_types import EdgeType, MissionGraph, MissionNode, NodeType

    graph = MissionGraph()
    graph.add_node(MissionNode(0, NodeType.START))
    graph.add_node(MissionNode(1, NodeType.LOCK))
    graph.add_node(MissionNode(2, NodeType.GOAL))
    graph.add_edge(0, 1, EdgeType.LOCKED, key_required=99)
    graph.add_edge(1, 2, EdgeType.ITEM_GATE, item_required="BOMB")
    graph.sanitize()

    assert 0 in graph.get_neighbors(1)
    assert 1 in graph.get_neighbors(2)


def test_graph_cognitive_proxy_uses_physical_edges_not_directed_edge_count():
    from src.evaluation.cbs_fitness import _compute_graph_cognitive_proxy

    graph = nx.DiGraph()
    graph.add_node(0, type="start")
    graph.add_node(1, type="room")
    graph.add_node(2, type="goal")
    graph.add_edges_from([(0, 1), (1, 0), (1, 2), (2, 1)])

    metrics = _compute_graph_cognitive_proxy(graph, target_confusion_ratio=2.0)

    assert metrics["astar_path_length"] == 2
    assert metrics["astar_states"] >= 1
    assert metrics["oracle_status"] == "solved"
    assert metrics["confusion_ratio"] < 1.2


def test_pcbs_component_report_rejects_zero_oracle_evidence():
    from scripts.run_pcbs_component_ablation import _build_markdown, summarize

    summary = summarize([])
    report = _build_markdown(summary, persona="novice")

    assert summary["experiment_valid"] is False
    assert summary["invalid_reason"] == "no_oracle_solved_maps"
    assert "Invalid component comparison" in report


def test_insert_lock_key_places_key_on_side_branch_not_trunk():
    from src.generation.grammar.core_rules import InsertLockKeyRule, StartRule
    from src.generation.grammar.graph_types import EdgeType, MissionGraph, NodeType

    rng = np.random.default_rng(123)
    graph = MissionGraph()
    graph = StartRule().apply(graph, {"goal_row": 0, "goal_col": 4, "rng": rng})
    graph = InsertLockKeyRule().apply(graph, {"rng": rng})

    key_nodes = [node for node in graph.nodes.values() if node.node_type == NodeType.KEY]
    assert len(key_nodes) == 1
    key_id = key_nodes[0].id
    assert any(edge.edge_type == EdgeType.LOCKED and edge.key_required == key_id for edge in graph.edges)

    start = graph.get_start_node()
    goal = graph.get_goal_node()
    assert start is not None and goal is not None
    trunk = InsertLockKeyRule()._find_shortest_path_nodes(graph, start.id, goal.id)
    assert key_id not in trunk
    assert graph.get_neighbors(key_id)


def test_boss_door_preserves_boss_key_and_key_item_does_not_grant_bombs():
    from src.simulation.validator import GameState, ZeldaLogicEnv

    grid = np.full((3, 4), SEMANTIC_PALETTE["FLOOR"], dtype=np.int64)
    grid[1, 0] = SEMANTIC_PALETTE["START"]
    grid[1, 1] = SEMANTIC_PALETTE["DOOR_BOSS"]
    grid[1, 2] = SEMANTIC_PALETTE["KEY_ITEM"]
    grid[1, 3] = SEMANTIC_PALETTE["TRIFORCE"]
    env = ZeldaLogicEnv(grid)

    ok, opened = env.try_move_pure(GameState(position=(1, 0), has_boss_key=True), (1, 1), SEMANTIC_PALETTE["DOOR_BOSS"])
    assert ok
    assert opened.has_boss_key

    ok, picked = env.try_move_pure(GameState(position=(1, 1), bomb_count=0), (1, 2), SEMANTIC_PALETTE["KEY_ITEM"])
    assert ok
    assert picked.has_item
    assert picked.bomb_count == 0


def test_key_economy_treats_boss_keys_as_persistent_for_multiple_boss_doors():
    from src.simulation.key_economy_validator import KeyEconomyValidator

    graph = nx.DiGraph()
    graph.add_node(0, label="START", items=["key_boss"])
    graph.add_node(1, label="BOSS_DOOR")
    graph.add_node(2, label="BOSS_DOOR")
    graph.add_node(3, label="GOAL")
    graph.add_edge(0, 1, lock_type="boss")
    graph.add_edge(1, 2, lock_type="boss")
    graph.add_edge(2, 3, lock_type="open")

    result = KeyEconomyValidator(graph).validate()

    assert result.greedy_solvable
    assert result.adversarial_solvable
    assert result.is_valid


def test_graph_warp_consumes_each_small_key_once_per_edge():
    grid = np.full((48, 11), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int64)
    grid[0, 0] = int(SEMANTIC_PALETTE["START"])
    grid[-1, -1] = int(SEMANTIC_PALETTE["TRIFORCE"])
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"edge_type": "key_locked"}),
            (1, 2, {"edge_type": "key_locked"}),
        ]
    )
    room_positions = {(0, 0): (0, 0), (1, 0): (16, 0), (2, 0): (32, 0)}
    room_to_node = {(0, 0): 0, (1, 0): 1, (2, 0): 2}
    env = ZeldaLogicEnv(
        grid,
        graph=graph,
        room_positions=room_positions,
        room_to_node=room_to_node,
        node_to_room={0: (0, 0), 1: (1, 0), 2: (2, 0)},
    )
    solver = StateSpaceAStar(env)
    state = GameState(position=(1, 1), keys=1)

    opened, after_first = solver.apply_graph_edge_transition(
        state,
        (1, 1),
        (17, 1),
        "key_locked",
    )
    after_first.position = (17, 1)
    blocked, _after_second = solver.apply_graph_edge_transition(
        after_first,
        (17, 1),
        (33, 1),
        "key_locked",
    )

    assert opened is True
    assert after_first.keys == 0
    assert len(after_first.opened_graph_edges) == 1
    assert blocked is False


def test_solver_comparison_uses_canonical_bomb_and_item_transitions():
    from src.simulation.solver_comparison import SolverComparison
    from src.simulation.validator import GameState, ZeldaLogicEnv

    grid = np.full((3, 3), SEMANTIC_PALETTE["FLOOR"], dtype=np.int64)
    grid[1, 0] = SEMANTIC_PALETTE["START"]
    grid[1, 1] = SEMANTIC_PALETTE["DOOR_BOMB"]
    grid[1, 2] = SEMANTIC_PALETTE["TRIFORCE"]
    grid[0, 1] = SEMANTIC_PALETTE["ELEMENT"]
    comparison = SolverComparison(ZeldaLogicEnv(grid))

    blocked, _state = comparison._simple_move(
        GameState(position=(1, 0), bomb_count=0),
        (1, 1),
        SEMANTIC_PALETTE["DOOR_BOMB"],
    )
    opened, bomb_state = comparison._simple_move(
        GameState(position=(1, 0), bomb_count=1),
        (1, 1),
        SEMANTIC_PALETTE["DOOR_BOMB"],
    )
    water_blocked, _state = comparison._simple_move(
        GameState(position=(0, 0), has_item=False),
        (0, 1),
        SEMANTIC_PALETTE["ELEMENT"],
    )

    assert blocked is False
    assert opened is True
    assert bomb_state.bomb_count == 0
    assert water_blocked is False


def test_search_metrics_are_bounded_and_confusion_is_overhead():
    from src.evaluation.search_benchmark_utils import (
        confusion_ratio_vs_oracle,
        path_efficiency_ratio,
        safe_positive_int,
    )

    assert path_efficiency_ratio(5, 10) == 1.0
    assert confusion_ratio_vs_oracle(10, 10, oracle_status="solved", candidate_success=True) == 0.0
    assert confusion_ratio_vs_oracle(10, 15, oracle_status="solved", candidate_success=True) == 0.5
    assert safe_positive_int(float("inf")) > 1_000_000


def test_ncd_reports_zero_for_identical_level_texts():
    from src.evaluation.end_to_end_level_metrics import normalized_compression_distance

    room = "WWWW\nWSGW\nWWWW"

    assert normalized_compression_distance(room, room) == 0.0
