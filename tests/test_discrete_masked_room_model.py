import logging
import networkx as nx
import numpy as np
import torch

from src.core.definitions import ROOM_HEIGHT, ROOM_TOPOLOGY_CHANNELS, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.discrete_masked_model import DiscreteMaskedRoomModel, create_discrete_masked_model
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.pipeline.room_topology_conditioning import build_room_semantic_anchor_points
from src.train_masked_room import MaskedRoomTrainer, MaskedRoomTrainingConfig, train_masked_room


class _DummyMaskedConditionEncoder:
    def __init__(self, output_dim: int = 8):
        self.output_dim = output_dim
        self.captured_reference_room_maps = None
        self.captured_style_id = None

    def __call__(
        self,
        *,
        neighbor_latents,
        boundary_constraints,
        position,
        node_features,
        edge_index,
        edge_features=None,
        tpe=None,
        current_node_distance=None,
        current_node_idx=None,
        reference_room_maps=None,
        style_id=None,
        return_global_tokens=False,
    ):
        _ = (
            neighbor_latents,
            boundary_constraints,
            position,
            node_features,
            edge_index,
            edge_features,
            tpe,
            current_node_distance,
            current_node_idx,
            style_id,
        )
        self.captured_reference_room_maps = reference_room_maps
        self.captured_style_id = style_id
        room_anchor = torch.full((1, self.output_dim), 7.0, dtype=torch.float32)
        if return_global_tokens:
            global_tokens = torch.full((1, int(node_features.shape[0]), self.output_dim), 3.0, dtype=torch.float32)
            return room_anchor, global_tokens
        return room_anchor

    def encode_global_only(self, *args, **kwargs):
        node_features = args[0]
        _ = kwargs
        return torch.full((int(node_features.shape[0]), self.output_dim), 3.0, dtype=torch.float32)

    def encode_local_only(self, *args, **kwargs):
        _ = (args, kwargs)
        return torch.full((1, self.output_dim), 5.0, dtype=torch.float32)


def test_discrete_masked_model_respects_fixed_tokens():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=64,
        model_channels=32,
        context_dim=256,
        num_steps=3,
    )
    context = torch.zeros(1, 1, 256)
    fixed_tokens = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.long)
    fixed_mask = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.bool)
    fixed_tokens[0, 0, 5] = int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    fixed_tokens[0, 8, 5] = int(SEMANTIC_PALETTE["START"])
    fixed_mask[0, 0, 5] = True
    fixed_mask[0, 8, 5] = True

    tokens, _logits, _hidden = model.sample(
        context=context,
        fixed_tokens=fixed_tokens,
        fixed_mask=fixed_mask,
        num_steps=2,
        seed=123,
    )

    assert int(tokens[0, 0, 5]) == int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    assert int(tokens[0, 8, 5]) == int(SEMANTIC_PALETTE["START"])


def test_pipeline_generate_room_uses_discrete_masked_mode(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )
    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))

    graph = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    graph_context = pipeline._build_room_graph_context(
        graph_data=graph,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )

    called = {"sample": 0}

    def _sample(**kwargs):
        called["sample"] += 1
        fixed_tokens = kwargs.get("fixed_tokens")
        fixed_mask = kwargs.get("fixed_mask")
        tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=torch.long)
        if isinstance(fixed_tokens, torch.Tensor) and isinstance(fixed_mask, torch.Tensor):
            tokens[fixed_mask] = fixed_tokens[fixed_mask]
        logits = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
        hidden = torch.zeros(1, 64, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        return tokens, logits, hidden

    monkeypatch.setattr(pipeline.masked_room_model, "sample", _sample)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        num_diffusion_steps=4,
        seed=7,
        start_goal_coords=((8, 0), (8, 10)),
    )

    assert called["sample"] == 1
    assert result.room_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["START"]))) == 1
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["TRIFORCE"]))) == 0


def test_pipeline_generate_room_masked_mode_falls_back_to_diffusion_teacher_on_noise(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
        default_masked_room_teacher_fallback_enabled=True,
    )
    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, pos=(0, 0))

    graph = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    graph_context = pipeline._build_room_graph_context(
        graph_data=graph,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )

    def _noisy_masked_sample(**kwargs):
        _ = kwargs
        tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=torch.long)
        tokens[:, 4:12, 4:7] = int(SEMANTIC_PALETTE["BLOCK"])
        logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
        logits[:, int(SEMANTIC_PALETTE["BLOCK"]), 4:12, 4:7] = 8.0
        hidden = torch.zeros(1, 64, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        return tokens, logits, hidden

    def _teacher_ddim_sample(*, context, shape, num_steps, graph_data=None):
        _ = (context, num_steps, graph_data)
        return torch.zeros(shape, dtype=torch.float32)

    def _teacher_decode(_latent):
        logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
        return logits

    monkeypatch.setattr(pipeline.masked_room_model, "sample", _noisy_masked_sample)
    monkeypatch.setattr(pipeline.diffusion, "ddim_sample", _teacher_ddim_sample)
    monkeypatch.setattr(pipeline.vqvae, "decode", _teacher_decode)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        num_diffusion_steps=4,
        seed=7,
        start_goal_coords=((8, 0), (8, 10)),
    )

    assert result.metrics["teacher_fallback_used"] == 1.0
    assert result.metrics["teacher_fallback_source_masked_room"] == 1.0
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["BLOCK"]))) == 0


def test_masked_room_fixed_tokens_only_emit_start_and_goal_in_true_role_rooms():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True, pos=(0, 0))
    graph.add_node(1, is_goal=True, pos=(0, 1))
    graph.add_edge(0, 1)

    start_tokens, start_mask = pipeline._build_masked_room_fixed_tokens(
        graph,
        0,
        start_goal=((8, 0), (8, 10)),
    )
    goal_tokens, goal_mask = pipeline._build_masked_room_fixed_tokens(
        graph,
        1,
        start_goal=((8, 0), (8, 10)),
    )

    start_role_flags = pipeline._room_role_flags(dict(graph.nodes[0]))
    start_semantics = pipeline._extract_room_topology_semantics(graph, 0)
    start_anchors = build_room_semantic_anchor_points(
        start=(8, 0),
        goal=(8, 10),
        required_doors=start_semantics["required_doors"],
        incoming_dirs=start_semantics["incoming_dirs"],
        outgoing_dirs=start_semantics["outgoing_dirs"],
        room_role_flags=start_role_flags,
    )
    sr, sc = start_anchors["start"]
    assert bool(start_mask[0, sr, sc])
    assert int(start_tokens[0, sr, sc]) == int(SEMANTIC_PALETTE["START"])

    goal_role_flags = pipeline._room_role_flags(dict(graph.nodes[1]))
    goal_semantics = pipeline._extract_room_topology_semantics(graph, 1)
    goal_anchors = build_room_semantic_anchor_points(
        start=(8, 0),
        goal=(8, 10),
        required_doors=goal_semantics["required_doors"],
        incoming_dirs=goal_semantics["incoming_dirs"],
        outgoing_dirs=goal_semantics["outgoing_dirs"],
        room_role_flags=goal_role_flags,
    )
    gr, gc = goal_anchors["goal"]
    assert bool(goal_mask[0, gr, gc])
    assert int(goal_tokens[0, gr, gc]) == int(SEMANTIC_PALETTE["TRIFORCE"])


def test_masked_room_fixed_tokens_place_shared_semantic_anchors_for_graph_roles():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )
    graph = nx.DiGraph()
    graph.add_node(0, label="BIG_KEY", type="BIG_KEY", has_key=True, pos=(0, 0))
    graph.add_node(1, label="ITEM", type="ITEM", has_item=True, pos=(0, 1))
    graph.add_node(2, label="SWITCH", type="SWITCH", has_puzzle=True, pos=(0, 2))
    graph.add_node(3, label="BOSS", type="BOSS", has_boss=True, pos=(0, 3))

    for room_id, anchor_name, tile_name in (
        (0, "key", "KEY_BOSS"),
        (1, "item", "KEY_ITEM"),
        (2, "puzzle", "PUZZLE"),
        (3, "boss", "BOSS"),
    ):
        fixed_tokens, fixed_mask = pipeline._build_masked_room_fixed_tokens(
            graph,
            room_id,
            start_goal=((8, 0), (8, 10)),
        )
        role_flags = pipeline._room_role_flags(dict(graph.nodes[room_id]))
        semantics = pipeline._extract_room_topology_semantics(graph, room_id)
        anchors = build_room_semantic_anchor_points(
            start=(8, 0),
            goal=(8, 10),
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            room_role_flags=role_flags,
        )
        rr, cc = anchors[anchor_name]
        assert bool(fixed_mask[0, rr, cc])
        assert int(fixed_tokens[0, rr, cc]) == int(SEMANTIC_PALETTE[tile_name])


def test_topology_fixed_mask_keeps_semantic_role_anchors_during_training():
    tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=torch.long)
    topo = torch.zeros(1, 50, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)

    key_anchor = (5, 3)
    boss_anchor = (10, 7)
    topo[0, ROOM_TOPOLOGY_CHANNELS["role_key"], key_anchor[0], key_anchor[1]] = 1.0
    topo[0, ROOM_TOPOLOGY_CHANNELS["role_boss"], boss_anchor[0], boss_anchor[1]] = 1.0
    tokens[0, key_anchor[0], key_anchor[1]] = int(SEMANTIC_PALETTE["KEY_BOSS"])
    tokens[0, boss_anchor[0], boss_anchor[1]] = int(SEMANTIC_PALETTE["BOSS"])

    fixed_tokens, fixed_mask = DiscreteMaskedRoomModel.build_fixed_mask_from_topology_map(
        tokens,
        topo,
        num_classes=44,
    )

    assert bool(fixed_mask[0, key_anchor[0], key_anchor[1]])
    assert bool(fixed_mask[0, boss_anchor[0], boss_anchor[1]])
    assert int(fixed_tokens[0, key_anchor[0], key_anchor[1]]) == int(SEMANTIC_PALETTE["KEY_BOSS"])
    assert int(fixed_tokens[0, boss_anchor[0], boss_anchor[1]]) == int(SEMANTIC_PALETTE["BOSS"])


def test_overlay_room_graph_markers_maps_generated_graph_types_to_semantic_tiles():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )
    base_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)

    graph = nx.DiGraph()
    graph.add_node(0, label="BIG_KEY", type="BIG_KEY", has_key=True, pos=(0, 0))
    graph.add_node(1, label="STAIRS_DOWN", type="STAIRS_DOWN", pos=(0, 1))
    graph.add_node(2, label="SWITCH", type="SWITCH", pos=(0, 2))

    key_grid, key_count, _ = pipeline._overlay_room_graph_markers(
        base_grid,
        graph=graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )
    stair_grid, stair_count, _ = pipeline._overlay_room_graph_markers(
        base_grid,
        graph=graph,
        room_id=1,
        start_goal=((8, 0), (8, 10)),
    )
    puzzle_grid, puzzle_count, _ = pipeline._overlay_room_graph_markers(
        base_grid,
        graph=graph,
        room_id=2,
        start_goal=((8, 0), (8, 10)),
    )

    assert key_count == 1
    assert int(np.sum(key_grid == int(SEMANTIC_PALETTE["KEY_BOSS"]))) == 1

    assert stair_count == 1
    assert int(np.sum(stair_grid == int(SEMANTIC_PALETTE["STAIR"]))) == 1

    assert puzzle_count == 1
    assert int(np.sum(puzzle_grid == int(SEMANTIC_PALETTE["PUZZLE"]))) == 1
    assert int(np.sum(puzzle_grid == int(SEMANTIC_PALETTE["START"]))) == 0


def test_room_role_flags_do_not_treat_switch_label_as_start():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )

    role_flags = pipeline._room_role_flags({"label": "S", "type": "SWITCH", "has_puzzle": True})

    assert role_flags["is_start"] is False
    assert role_flags["has_puzzle"] is True


def test_pipeline_generate_room_strips_hallucinated_semantics_and_replays_graph_markers(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
        default_masked_room_teacher_fallback_enabled=False,
    )
    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, label="GOAL", type="GOAL", is_goal=True, pos=(0, 0))

    graph = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    graph_context = pipeline._build_room_graph_context(
        graph_data=graph,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )

    def _sample(**kwargs):
        _ = kwargs
        tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=torch.long)
        tokens[:, 5, 2] = int(SEMANTIC_PALETTE["ENEMY"])
        tokens[:, 6, 3] = int(SEMANTIC_PALETTE["PUZZLE"])
        tokens[:, 7, 4] = int(SEMANTIC_PALETTE["STAIR"])
        tokens[:, 8, 5] = int(SEMANTIC_PALETTE["ELEMENT"])
        logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
        logits[:, int(SEMANTIC_PALETTE["ENEMY"]), 5, 2] = 7.0
        logits[:, int(SEMANTIC_PALETTE["PUZZLE"]), 6, 3] = 7.0
        logits[:, int(SEMANTIC_PALETTE["STAIR"]), 7, 4] = 7.0
        logits[:, int(SEMANTIC_PALETTE["ELEMENT"]), 8, 5] = 7.0
        hidden = torch.zeros(1, 64, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        return tokens, logits, hidden

    monkeypatch.setattr(pipeline.masked_room_model, "sample", _sample)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        num_diffusion_steps=4,
        seed=7,
        start_goal_coords=((8, 0), (8, 10)),
    )

    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["TRIFORCE"]))) == 1
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["ENEMY"]))) == 0
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["PUZZLE"]))) == 0
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["STAIR"]))) == 0
    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["ELEMENT"]))) == 0
    assert result.metrics["neural_semantic_tiles_stripped"] >= 3
    assert result.metrics["final_graph_markers_placed"] == 1


def test_discrete_masked_model_accepts_unbatched_edge_index_graph_context():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=32,
        model_channels=32,
        context_dim=8,
        num_steps=2,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(0,),
        unet_num_heads=1,
    )
    tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=44, dtype=torch.long)
    step = torch.zeros(1, dtype=torch.long)
    context = torch.randn(1, 2, 8)
    graph_data = {
        "node_features": torch.randn(2, 6),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "tpe": torch.randn(2, 8),
        "node_positions": torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        "current_node_distance": torch.randn(2, 4),
    }

    logits = model.forward(tokens, step, context, graph_data=graph_data)

    assert tuple(logits.shape) == (1, 44, ROOM_HEIGHT, ROOM_WIDTH)


def test_masked_room_trainer_passes_configurable_mask_schedule(monkeypatch):
    config = MaskedRoomTrainingConfig(
        device="cpu",
        quick=True,
        min_mask_ratio=0.25,
        max_mask_ratio=0.55,
        model_channels=32,
        hidden_dim=32,
        condition_hidden_dim=64,
        condition_num_attention_heads=4,
        unet_num_heads=4,
    )
    trainer = MaskedRoomTrainer(config)
    captured = {}

    def _fake_training_loss(*args, **kwargs):
        captured["min_mask_ratio"] = kwargs.get("min_mask_ratio")
        captured["max_mask_ratio"] = kwargs.get("max_mask_ratio")
        return torch.tensor(0.0, device=trainer.device), {
            "loss": 0.0,
            "mask_ratio": 0.0,
            "masked_fraction": 0.0,
        }

    monkeypatch.setattr(trainer.model, "training_loss", _fake_training_loss)

    real_maps = torch.zeros(1, 1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
    metrics = trainer._step(real_maps, graph_list=None, train=False)

    assert metrics["loss"] == 0.0
    assert captured["min_mask_ratio"] == 0.25
    assert captured["max_mask_ratio"] == 0.55


def test_masked_room_trainer_can_pass_reference_room_maps_into_condition_encoder():
    trainer = MaskedRoomTrainer.__new__(MaskedRoomTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = type(
        "Cfg",
        (),
        {
            "graph_conditioning_mode": "node_sequence",
            "condition_use_reference_room_maps": True,
            "use_current_node_distance_features": True,
            "current_node_distance_max": 8,
        },
    )()
    trainer.condition_encoder = _DummyMaskedConditionEncoder(output_dim=8)

    graph_dict = {
        "node_features": torch.randn(2, 6),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_attr": torch.tensor([0], dtype=torch.long),
        "tpe": torch.randn(2, 8),
        "boundary_constraints": torch.zeros(8, dtype=torch.float32),
        "room_position": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "current_node_idx": 0,
        "neighbor_maps": {
            "N": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.25, dtype=torch.float32),
            "S": None,
            "E": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.75, dtype=torch.float32),
            "W": None,
        },
    }

    _encoded = MaskedRoomTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert trainer.condition_encoder.captured_reference_room_maps is graph_dict["neighbor_maps"]


def test_masked_room_trainer_passes_explicit_style_id_into_condition_encoder():
    trainer = MaskedRoomTrainer.__new__(MaskedRoomTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = type(
        "Cfg",
        (),
        {
            "graph_conditioning_mode": "node_sequence",
            "condition_use_reference_room_maps": False,
            "use_current_node_distance_features": True,
            "current_node_distance_max": 8,
        },
    )()
    trainer.condition_encoder = _DummyMaskedConditionEncoder(output_dim=8)

    graph_dict = {
        "node_features": torch.randn(2, 6),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_attr": torch.tensor([0], dtype=torch.long),
        "tpe": torch.randn(2, 8),
        "boundary_constraints": torch.zeros(8, dtype=torch.float32),
        "room_position": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "current_node_idx": 0,
        "style_id": 3,
    }

    _encoded = MaskedRoomTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert trainer.condition_encoder.captured_style_id == 3


def test_masked_room_resume_checkpoint_round_trip(tmp_path):
    config = MaskedRoomTrainingConfig(
        device="cpu",
        quick=True,
        checkpoint_dir=str(tmp_path),
        model_channels=32,
        hidden_dim=32,
        condition_hidden_dim=64,
        condition_num_attention_heads=4,
        unet_num_heads=4,
    )
    trainer = MaskedRoomTrainer(config)
    trainer.epoch = 3
    trainer.global_step = 17

    tracked_param = next(trainer.model.parameters())
    original = tracked_param.detach().clone()

    resume_path = tmp_path / "masked_room_resume.pth"
    inference_path = tmp_path / "masked_room_inference.pth"

    trainer.save_checkpoint(str(resume_path), {"val_loss": 1.25}, include_optimizer=True)
    resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" in resume_payload
    assert "scheduler_state_dict" in resume_payload
    assert resume_payload["epoch"] == 3
    assert resume_payload["global_step"] == 17

    trainer.save_checkpoint(str(inference_path), {"val_loss": 1.25}, include_optimizer=False)
    inference_payload = torch.load(inference_path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" not in inference_payload
    assert "scheduler_state_dict" not in inference_payload

    with torch.no_grad():
        tracked_param.zero_()
    trainer.epoch = 0
    trainer.global_step = 0

    trainer.load_checkpoint(str(resume_path))

    assert trainer.epoch == 3
    assert trainer.global_step == 17
    assert torch.allclose(tracked_param, original)


def test_masked_room_auto_resume_skips_incompatible_latest_checkpoint(
    monkeypatch,
    tmp_path,
    caplog,
):
    latest_resume = tmp_path / "latest_resume.pth"
    latest_resume.write_bytes(b"stub")

    class _EmptyLoader:
        def __init__(self):
            self.dataset = [0]

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    monkeypatch.setattr("src.train_masked_room.create_dataloader", lambda *args, **kwargs: _EmptyLoader())

    def _boom(self, path: str):
        raise RuntimeError("size mismatch for masked-room checkpoint")

    monkeypatch.setattr(MaskedRoomTrainer, "load_checkpoint", _boom)

    config = MaskedRoomTrainingConfig(
        device="cpu",
        data_dir="unused",
        checkpoint_dir=str(tmp_path),
        epochs=1,
        batch_size=1,
        model_channels=32,
        hidden_dim=32,
        condition_hidden_dim=64,
        condition_num_attention_heads=4,
        unet_num_heads=4,
        auto_resume=True,
    )

    caplog.set_level(logging.WARNING)
    trainer = train_masked_room(config)

    assert trainer.epoch == 0
    assert "Skipping auto-resume masked-room checkpoint" in caplog.text
