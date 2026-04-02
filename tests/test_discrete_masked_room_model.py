import networkx as nx
import torch

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.discrete_masked_model import create_discrete_masked_model
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.train_masked_room import MaskedRoomTrainer, MaskedRoomTrainingConfig


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
    assert int(result.room_grid[8, 0]) == int(SEMANTIC_PALETTE["START"])
    assert int(result.room_grid[8, 10]) == int(SEMANTIC_PALETTE["TRIFORCE"])


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
