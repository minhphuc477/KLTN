"""Tests for src.generation.puzzle_generator module."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.core import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.definitions import TileID
from src.generation.puzzle_generator import (
    PuzzleGenerator,
    _apply_puzzle_room_scaffold,
    _build_puzzle_room_route_template,
    _build_room_puzzle_metadata,
    _classify_puzzle_gate_family,
    _globalize_room_puzzle_metadata,
    _select_puzzle_room_scaffold_archetype,
)
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline


def test_puzzle_generator_class_instantiation():
    """Verify PuzzleGenerator OOP wrapper instantiates cleanly."""
    pipeline = object.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 999
    pipeline.default_semantic_puzzle_offset = 2
    pipeline.domain_schema = None

    generator = PuzzleGenerator(pipeline)
    assert generator.pipeline is pipeline


def test_puzzle_generator_archetype_selection():
    """Verify puzzle archetype selection for various room attributes."""
    pipeline = object.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_archetype_mode = "auto"

    # Combat puzzle
    combat_archetype = _select_puzzle_room_scaffold_archetype(
        pipeline,
        role_flags={"is_combat_puzzle": True, "has_puzzle": True},
        semantics={"edge_constraints": {}},
        node_type="combat_puzzle",
    )
    assert combat_archetype == "combat"

    # Hub puzzle with 4 doors (simulated via semantics)
    gate_archetype = _select_puzzle_room_scaffold_archetype(
        pipeline,
        role_flags={"has_puzzle": True},
        semantics={"edge_constraints": {"0->1": ["locked", "gate"]}},
        node_type="puzzle",
    )
    assert gate_archetype in {"gate", "serpentine", "hub", "island", "combat"}


def test_puzzle_generator_gate_classification():
    """Verify gate family classification."""
    pipeline = object.__new__(NeuralSymbolicDungeonPipeline)

    gate_family = _classify_puzzle_gate_family(
        pipeline,
        role_flags={"is_switch_puzzle": True},
        semantics={"edge_constraints": {"0->1": ["switch_locked", "gate"]}},
        node_type="switch",
    )
    assert gate_family == "switch"


def test_puzzle_generator_apply_scaffold_integration():
    """Verify apply_scaffold generates valid structure without breaking contracts."""
    pipeline = object.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_scaffold_max_structure_tiles = 50
    pipeline.default_puzzle_room_scaffold_min_door_distance = 2
    pipeline.default_puzzle_room_scaffold_archetype_mode = "auto"
    pipeline.default_puzzle_room_scaffold_candidate_count = 4
    pipeline.default_puzzle_novelty_archive = []
    pipeline.default_semantic_puzzle_offset = 2
    pipeline.domain_schema = None

    generator = PuzzleGenerator(pipeline)

    graph = nx.DiGraph()
    graph.add_node(0, type="PUZZLE", has_puzzle=True, position=(0, 0, 0))
    graph.add_node(1, type="GOAL", position=(1, 0, 0))
    graph.add_edge(0, 1, edge_type="PATH")

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])
    room[0, 5] = int(TileID.DOOR_OPEN)
    room[-1, 5] = int(TileID.DOOR_OPEN)

    scaffolded_grid, stats = generator.apply_scaffold(
        room,
        graph=graph,
        room_id=0,
    )

    assert isinstance(scaffolded_grid, np.ndarray)
    assert scaffolded_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert stats["applied"] in {0, 1}


def test_glmdm_gradient_flow_zero_dead_params_and_logicnet_backward():
    """Verify full G-LMDM gradient propagation with anomaly detection and zero dead params."""
    import torch
    from src.core.definitions import GRAPH_EDGE_FEATURE_DIM, GRAPH_NODE_FEATURE_DIM
    from src.train_masked_room import MaskedRoomTrainer, MaskedRoomTrainingConfig

    cfg = MaskedRoomTrainingConfig(
        num_classes=44,
        model_channels=64,
        hidden_dim=24,
        context_dim=64,
        condition_hidden_dim=64,
        condition_num_gnn_layers=2,
        condition_num_attention_heads=4,
        condition_gnn_type="gps",
        condition_use_reference_room_maps=True,
        condition_use_rrwp_edge_features=True,
        unet_channel_mult=[1, 2],
        unet_num_res_blocks=1,
        unet_attention_resolutions=[0, 1],
        unet_num_heads=4,
        logic_net_enabled=True,
        logic_net_trainable=False,
        alpha_logic=0.1,
        device="cpu",
    )
    trainer = MaskedRoomTrainer(cfg)
    real_maps = torch.randint(0, 44, (2, 1, ROOM_HEIGHT, ROOM_WIDTH)).float() / 43.0
    edge_feats = torch.zeros(6, GRAPH_EDGE_FEATURE_DIM)
    edge_feats[:, 0] = 1.0
    graph_sample = {
        "node_features": torch.randn(4, GRAPH_NODE_FEATURE_DIM),
        "edge_index": torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]], dtype=torch.long),
        "edge_features": edge_feats,
        "edge_attr": torch.zeros(6, dtype=torch.long),
        "current_node_idx": 1,
        "start_node_id": 0,
        "target_idx": 3,
        "room_topology_map": torch.randn(54, ROOM_HEIGHT, ROOM_WIDTH),
        "logic_source_mask": torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH),
        "logic_target_mask": torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH),
        "boundary_constraints": torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "room_position": torch.tensor([1.0, 2.0]),
        "style_id": 1,
        "neighbor_maps": {"N": torch.randint(0, 44, (ROOM_HEIGHT, ROOM_WIDTH))},
    }
    graph_sample["logic_source_mask"][0, 1, 5] = 1.0
    graph_sample["logic_target_mask"][0, 14, 5] = 1.0
    graph_list = [
        dict(graph_sample, current_node_idx=0),
        dict(graph_sample, current_node_idx=1),
    ]

    trainer.model.train()
    trainer.condition_encoder.train()
    with torch.autograd.set_detect_anomaly(True):
        metrics = trainer._step(real_maps, graph_list, train=True)
    assert metrics["skipped_nonfinite_batch"] == 0.0

    for module in (trainer.model, trainer.condition_encoder):
        for param_name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            assert param.grad is not None, f"Missing gradient on {param_name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {param_name}"
            assert param.grad.abs().sum().item() > 0.0, f"Zero gradient on {param_name}"

    trainer.optimizer.zero_grad(set_to_none=True)
    token_ids = trainer._to_token_ids(real_maps, num_classes=cfg.num_classes)
    cond_vectors = [trainer._encode_graph_conditioning(g) for g in graph_list]
    conditioning = trainer._stack_conditioning_vectors(cond_vectors)
    graph_batch = trainer._stack_graph_batch(graph_list)
    _loss, _metrics, aux = trainer.model.training_loss(
        token_ids, conditioning, graph_data=graph_batch, return_aux=True
    )
    logic_loss, _logic_info = trainer.logic_net(aux["logits"], graph_data=graph_batch)
    with torch.autograd.set_detect_anomaly(True):
        logic_loss.backward()

    model_logic_grad = sum(
        p.grad.norm().item() for p in trainer.model.parameters() if p.grad is not None
    )
    cond_logic_grad = sum(
        p.grad.norm().item()
        for p in trainer.condition_encoder.parameters()
        if p.grad is not None
    )
    rrwp_grad = trainer.condition_encoder.global_encoder.edge_rrwp_proj.weight.grad.norm().item()
    assert model_logic_grad > 0.0
    assert cond_logic_grad > 0.0
    assert rrwp_grad > 0.0


def test_glmdm_discrete_masked_strict_dungeon_puzzle_generation(tmp_path):
    """Verify strict composite checkpoint loading and discrete_masked dungeon/puzzle synthesis."""
    from src.train_masked_room import MaskedRoomTrainer, MaskedRoomTrainingConfig

    ckpt = tmp_path / "masked_room_best.pth"
    cfg = MaskedRoomTrainingConfig(
        num_classes=44,
        model_channels=64,
        hidden_dim=24,
        context_dim=64,
        condition_hidden_dim=64,
        condition_num_gnn_layers=2,
        condition_num_attention_heads=4,
        condition_gnn_type="gps",
        condition_use_reference_room_maps=True,
        condition_use_rrwp_edge_features=True,
        unet_channel_mult=[1, 2],
        unet_num_res_blocks=1,
        unet_attention_resolutions=[0, 1],
        unet_num_heads=4,
        logic_net_enabled=True,
        logic_net_trainable=False,
        alpha_logic=0.1,
        device="cpu",
    )
    trainer = MaskedRoomTrainer(cfg)
    trainer.save_checkpoint(str(ckpt), {"val_loss": 1.0}, include_optimizer=False)

    pipe = NeuralSymbolicDungeonPipeline(
        room_generator_mode="discrete_masked",
        masked_room_checkpoint=str(ckpt),
        condition_gnn_type="gps",
        condition_use_reference_room_maps=True,
        condition_use_rrwp_edge_features=True,
        strict_checkpoint_mode=True,
        require_logic_net=True,
        device="cpu",
    )
    assert pipe.vqvae is None
    assert pipe.logic_net is not None

    graph = nx.DiGraph()
    graph.add_node(0, label="START", is_start=True, pos=(0, 0), room_type="start")
    graph.add_node(1, label="KEY,PUZZLE", has_key=True, has_puzzle=True, pos=(0, 1), room_type="puzzle")
    graph.add_node(2, label="TRIFORCE", is_triforce=True, pos=(0, 2), room_type="goal")
    graph.add_edge(0, 1, edge_type="open", direction="E")
    graph.add_edge(1, 2, edge_type="key_locked", direction="E")

    res = pipe.generate_dungeon(
        mission_graph=graph,
        num_diffusion_steps=2,
        apply_repair=True,
        enable_map_elites=False,
        seed=42,
    )
    assert res.dungeon_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH * 3)
    assert len(res.rooms) == 3
    assert len(res.puzzle_metadata.get("plans", [])) >= 1
    assert res.rooms[1].puzzle_metadata
    assert res.rooms[1].raw_neural_grid is not None
    assert pipe.runtime_diagnostics.get("masked_room_neighbor_boundary_tokens_applied", 0) >= 1
