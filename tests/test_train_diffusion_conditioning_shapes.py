# pyright: reportPrivateUsage=false

from types import SimpleNamespace

import pytest
import torch

from src.core.definitions import GRAPH_EDGE_FEATURE_DIM, ROOM_HEIGHT, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_WIDTH
from src.train_diffusion import DiffusionTrainer

EMPTY_NEIGHBORS = {"N": None, "S": None, "E": None, "W": None}


class _DummyEvalModel(torch.nn.Module):
    def __init__(self, diffusion_loss: float = 0.5):
        super().__init__()
        self.last_conditioning = None
        self.last_graph_data = None
        self.diffusion_loss = float(diffusion_loss)

    def eval(self):
        return self

    def training_loss(self, z_0, conditioning, graph_data=None):
        self.last_conditioning = conditioning
        self.last_graph_data = graph_data
        _ = z_0
        return torch.tensor(self.diffusion_loss, dtype=torch.float32)

    def sample(self, conditioning, shape, graph_data=None):
        self.last_conditioning = conditioning
        self.last_graph_data = graph_data
        return torch.zeros(shape, dtype=torch.float32)


class _NaNEvalModel(_DummyEvalModel):
    def training_loss(self, z_0, conditioning, graph_data=None):
        self.last_conditioning = conditioning
        self.last_graph_data = graph_data
        _ = z_0
        return torch.full((), float("nan"), dtype=torch.float32)

    def sample(self, conditioning, shape, graph_data=None):
        self.last_conditioning = conditioning
        self.last_graph_data = graph_data
        return torch.full(shape, float("nan"), dtype=torch.float32)


class _DummyLogicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_graph_data = None

    def forward(self, z_latent, graph_data=None):
        self.last_graph_data = graph_data
        return torch.tensor(0.25, dtype=torch.float32), {}


class _TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class _DummyTrainingLossModule(_TinyModule):
    def __init__(self, loss_value: float):
        super().__init__()
        self.loss_value = float(loss_value)
        self.num_timesteps = 1000

    def train(self):
        return self

    def training_loss(self, z_0, conditioning, graph_data=None):
        _ = (z_0, conditioning, graph_data)
        return torch.tensor(self.loss_value, dtype=torch.float32)


class _DummyRoomAwareConditionEncoder:
    def __init__(self, output_dim: int = 8):
        self.output_dim = output_dim
        self.captured_current_node_idx = None
        self.captured_current_node_distance = None
        self.captured_return_global_tokens = None
        self.captured_neighbor_latents = None
        self.captured_reference_room_maps = None
        self.captured_style_id = None
        self.encode_global_only_calls = 0

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
        _ = (boundary_constraints, position, node_features, edge_index, edge_features, tpe, style_id)
        self.captured_current_node_idx = current_node_idx
        self.captured_current_node_distance = current_node_distance
        self.captured_return_global_tokens = return_global_tokens
        self.captured_neighbor_latents = neighbor_latents
        self.captured_reference_room_maps = reference_room_maps
        self.captured_style_id = style_id
        room_anchor = torch.full((1, self.output_dim), 7.0, dtype=torch.float32)
        if return_global_tokens:
            global_tokens = torch.full((1, int(node_features.shape[0]), self.output_dim), 3.0, dtype=torch.float32)
            return room_anchor, global_tokens
        return room_anchor

    def encode_global_only(
        self,
        node_features,
        edge_index,
        edge_features=None,
        tpe=None,
        current_node_distance=None,
    ):
        _ = (edge_index, edge_features, tpe, current_node_distance)
        self.encode_global_only_calls += 1
        return torch.full((int(node_features.shape[0]), self.output_dim), 3.0, dtype=torch.float32)

    def encode_local_only(
        self,
        neighbor_latents,
        boundary_constraints,
        position,
    ):
        _ = (neighbor_latents, boundary_constraints, position)
        return torch.full((1, self.output_dim), 5.0, dtype=torch.float32)


def _make_node_sequence_graph_list() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "n": 3,
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.tensor([[0.0] * 8, [0.5] * 8, [1.0] * 8], dtype=torch.float32),
            "node_positions": torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        },
        {
            "n": 5,
            "node_features": torch.randn(5, 6),
            "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            "tpe": torch.ones(5, 8, dtype=torch.float32),
            "node_positions": torch.tensor(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [2.0, 2.0]],
                dtype=torch.float32,
            ),
        },
    ]


def _assert_batched_graph_sequence(
    graph_data: dict[str, torch.Tensor],
    graph_list: list[dict[str, torch.Tensor]],
) -> None:
    assert tuple(graph_data["node_positions"].shape) == (2, 5, 2)
    assert tuple(graph_data["current_node_distance"].shape) == (2, 5, 4)
    assert tuple(graph_data["node_mask"].shape) == (2, 5)
    assert tuple(graph_data["edge_index"].shape[:2]) == (2, 2)
    assert tuple(graph_data["edge_features"].shape[:2]) == (2, 3)
    assert tuple(graph_data["edge_attr"].shape) == (2, 3)
    assert tuple(graph_data["current_node_idx"].shape) == (2,)
    assert tuple(graph_data["start_node_id"].shape) == (2,)
    assert tuple(graph_data["target_idx"].shape) == (2,)
    assert torch.allclose(graph_data["node_positions"][0, :3], graph_list[0]["node_positions"])
    assert torch.allclose(graph_data["tpe"][1, :5], graph_list[1]["tpe"])


def _make_room_condition_graph_dict(**overrides) -> dict[str, object]:
    graph_dict: dict[str, object] = {
        "node_features": torch.randn(2, 6),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_attr": torch.tensor([0], dtype=torch.long),
        "tpe": torch.randn(2, 8),
        "boundary_constraints": torch.zeros(8, dtype=torch.float32),
        "room_position": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "current_node_idx": 0,
    }
    graph_dict.update(overrides)
    return graph_dict


def _make_stub_trainer(context_dim: int = 8) -> DiffusionTrainer:
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(
        graph_conditioning_mode="node_sequence",
        context_dim=context_dim,
        edge_feature_dim=GRAPH_EDGE_FEATURE_DIM,
        warmup_epochs=0,
        alpha_visual=1.0,
        alpha_logic=0.1,
        validation_num_diffusion_samples=4,
    )
    trainer.device = torch.device("cpu")
    trainer.epoch = 0
    trainer.condition_encoder = object()

    trainer._encode_graph_conditioning = lambda graph_dict: torch.randn(
        int(graph_dict["n"]), context_dim, dtype=torch.float32
    )
    trainer._build_logic_graph_data = lambda graph_dict: None
    trainer.encode_to_latent = lambda real_maps: torch.zeros(
        (real_maps.shape[0], 4, 2, 2), dtype=torch.float32
    )
    trainer.get_dummy_conditioning = lambda batch_size: torch.randn(
        batch_size, 1, context_dim, dtype=torch.float32
    )

    trainer.logic_net = _DummyLogicNet()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.diffusion = _DummyEvalModel()

    trainer.scheduler = SimpleNamespace(step=lambda: None)

    def _train_step_stub(
        real_maps,
        conditioning=None,
        include_logic_loss=True,
        logic_graph_data=None,
        diffusion_graph_data=None,
    ):
        _ = (real_maps, include_logic_loss)
        trainer.last_train_conditioning_shape = tuple(conditioning.shape)
        trainer.last_train_logic_graph_data = logic_graph_data
        trainer.last_train_diffusion_graph_data = diffusion_graph_data
        return {
            "loss": 0.0,
            "diffusion_loss": 0.0,
            "logic_loss": 0.0,
            "solvability": 1.0,
        }

    trainer.train_step = _train_step_stub
    return trainer


def test_encode_edge_features_prefers_explicit_edge_vectors():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(edge_feature_dim=GRAPH_EDGE_FEATURE_DIM)

    explicit = torch.randn(2, GRAPH_EDGE_FEATURE_DIM)
    encoded = DiffusionTrainer._encode_edge_features(
        trainer,
        {
            "edge_features": explicit,
            "edge_attr": torch.tensor([1, 4], dtype=torch.long),
        },
    )

    assert encoded is not None
    assert torch.allclose(encoded, explicit)


def test_encode_edge_features_falls_back_to_schema_width_one_hot():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(edge_feature_dim=GRAPH_EDGE_FEATURE_DIM)

    encoded = DiffusionTrainer._encode_edge_features(
        trainer,
        {
            "edge_attr": torch.tensor([0, 7, 99], dtype=torch.long),
        },
    )

    assert encoded is not None
    assert tuple(encoded.shape) == (3, GRAPH_EDGE_FEATURE_DIM)
    assert torch.allclose(encoded.sum(dim=1), torch.ones(3))
    assert float(encoded[1, 7]) == 1.0
    assert float(encoded[2, GRAPH_EDGE_FEATURE_DIM - 1]) == 1.0


def test_train_epoch_node_sequence_conditioning_is_batched_and_padded():
    trainer = _make_stub_trainer(context_dim=8)

    real_maps = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)
    graph_list = _make_node_sequence_graph_list()
    dataloader = [(real_maps, graph_list)]

    DiffusionTrainer.train_epoch(trainer, dataloader)

    assert trainer.last_train_conditioning_shape == (2, 5, 8)
    _assert_batched_graph_sequence(trainer.last_train_diffusion_graph_data, graph_list)


def test_validate_node_sequence_conditioning_is_batched_and_padded():
    trainer = _make_stub_trainer(context_dim=8)

    real_maps = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)
    graph_list = _make_node_sequence_graph_list()
    dataloader = [(real_maps, graph_list)]

    _metrics = DiffusionTrainer.validate(trainer, dataloader, num_samples=2, num_diffusion_samples=2)

    assert tuple(trainer.ema_diffusion.last_conditioning.shape) == (2, 5, 8)
    _assert_batched_graph_sequence(trainer.ema_diffusion.last_graph_data, graph_list)
    assert _metrics["val_diffusion_loss"] == pytest.approx(0.5)
    assert _metrics["val_logic_loss"] == pytest.approx(0.25)
    assert _metrics["val_total_loss"] == pytest.approx(0.525)
    assert _metrics["val_solvability_proxy"] == pytest.approx(float(torch.exp(torch.tensor(-0.25)).item()))
    assert 0.0 <= _metrics["val_solvability_proxy"] <= 1.0


def test_validate_warmup_total_loss_excludes_logic_term():
    trainer = _make_stub_trainer(context_dim=8)
    trainer.epoch = -1

    real_maps = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)
    graph_list = _make_node_sequence_graph_list()
    dataloader = [(real_maps, graph_list)]

    metrics = DiffusionTrainer.validate(trainer, dataloader, num_samples=2, num_diffusion_samples=2)

    assert metrics["val_diffusion_loss"] == pytest.approx(0.5)
    assert metrics["val_logic_loss"] == pytest.approx(0.25)
    assert metrics["val_total_loss"] == pytest.approx(0.5)


def test_encode_graph_conditioning_prepends_room_anchor_for_room_samples():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(graph_conditioning_mode="node_sequence", context_dim=8)
    trainer.device = torch.device("cpu")
    trainer.condition_encoder = _DummyRoomAwareConditionEncoder(output_dim=8)

    graph_dict = {
        "node_features": torch.randn(3, 6),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_attr": torch.tensor([0, 1], dtype=torch.long),
        "tpe": torch.randn(3, 8),
        "boundary_constraints": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        "room_position": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "current_node_idx": 2,
    }

    encoded = DiffusionTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert tuple(encoded.shape) == (4, 8)
    assert torch.allclose(encoded[0], torch.full((8,), 7.0))
    assert torch.allclose(encoded[1:], torch.full((3, 8), 3.0))
    assert trainer.condition_encoder.captured_current_node_idx == 2
    assert tuple(trainer.condition_encoder.captured_current_node_distance.shape) == (3, 4)
    assert float(trainer.condition_encoder.captured_current_node_distance[2, 3]) == pytest.approx(1.0)
    assert trainer.condition_encoder.captured_return_global_tokens is True
    assert trainer.condition_encoder.encode_global_only_calls == 0
    assert trainer.condition_encoder.captured_neighbor_latents == EMPTY_NEIGHBORS


def test_encode_graph_conditioning_uses_teacher_forced_neighbor_latents_when_available():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(
        graph_conditioning_mode="node_sequence",
        context_dim=8,
        condition_use_reference_room_maps=False,
    )
    trainer.device = torch.device("cpu")
    trainer.condition_encoder = _DummyRoomAwareConditionEncoder(output_dim=8)

    def _encode_to_latent_stub(room_map):
        value = float(room_map.mean().item())
        return torch.full((int(room_map.shape[0]), 4, 2, 2), value, dtype=torch.float32)

    trainer.encode_to_latent = _encode_to_latent_stub

    graph_dict = _make_room_condition_graph_dict(
        neighbor_maps={
            "N": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.25, dtype=torch.float32),
            "S": None,
            "E": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.75, dtype=torch.float32),
            "W": None,
        },
    )

    _encoded = DiffusionTrainer._encode_graph_conditioning(trainer, graph_dict)

    captured = trainer.condition_encoder.captured_neighbor_latents
    assert captured is not None
    assert captured["S"] is None
    assert captured["W"] is None
    assert tuple(captured["N"].shape) == (1, 4, 2, 2)
    assert tuple(captured["E"].shape) == (1, 4, 2, 2)
    assert torch.allclose(captured["N"], torch.full((1, 4, 2, 2), 0.25))
    assert torch.allclose(captured["E"], torch.full((1, 4, 2, 2), 0.75))


def test_encode_graph_conditioning_can_disable_teacher_forced_neighbor_latents():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(
        graph_conditioning_mode="node_sequence",
        context_dim=8,
        use_teacher_forced_neighbor_latents=False,
    )
    trainer.device = torch.device("cpu")
    trainer.condition_encoder = _DummyRoomAwareConditionEncoder(output_dim=8)
    trainer.encode_to_latent = lambda room_map: torch.full((1, 4, 2, 2), 9.0, dtype=torch.float32)

    graph_dict = _make_room_condition_graph_dict(
        neighbor_maps={
            "N": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.25, dtype=torch.float32),
        },
    )

    _encoded = DiffusionTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert trainer.condition_encoder.captured_neighbor_latents == EMPTY_NEIGHBORS


def test_encode_graph_conditioning_can_pass_reference_room_maps_into_condition_encoder():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(
        graph_conditioning_mode="node_sequence",
        context_dim=8,
        condition_use_reference_room_maps=True,
    )
    trainer.device = torch.device("cpu")
    trainer.condition_encoder = _DummyRoomAwareConditionEncoder(output_dim=8)
    trainer.encode_to_latent = lambda room_map: torch.full((1, 4, 2, 2), 1.0, dtype=torch.float32)

    neighbor_maps = {
        "N": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.25, dtype=torch.float32),
        "S": None,
        "E": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.75, dtype=torch.float32),
        "W": None,
    }
    graph_dict = _make_room_condition_graph_dict(neighbor_maps=neighbor_maps)

    _encoded = DiffusionTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert trainer.condition_encoder.captured_reference_room_maps is neighbor_maps


def test_encode_graph_conditioning_passes_explicit_style_id_into_condition_encoder():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(
        graph_conditioning_mode="node_sequence",
        context_dim=8,
        condition_use_reference_room_maps=False,
    )
    trainer.device = torch.device("cpu")
    trainer.condition_encoder = _DummyRoomAwareConditionEncoder(output_dim=8)
    trainer.encode_to_latent = lambda room_map: torch.full((1, 4, 2, 2), 1.0, dtype=torch.float32)

    graph_dict = _make_room_condition_graph_dict(style_id=4)

    _encoded = DiffusionTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert trainer.condition_encoder.captured_style_id == 4


def test_encode_graph_conditioning_prepends_default_anchor_for_plain_graph_samples():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(graph_conditioning_mode="node_sequence", context_dim=8)
    trainer.device = torch.device("cpu")
    trainer.condition_encoder = _DummyRoomAwareConditionEncoder(output_dim=8)

    graph_dict = {
        "node_features": torch.randn(3, 6),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_attr": torch.tensor([0, 1], dtype=torch.long),
        "tpe": torch.randn(3, 8),
    }

    encoded = DiffusionTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert tuple(encoded.shape) == (4, 8)
    assert torch.allclose(encoded[0], torch.full((8,), 5.0))
    assert torch.allclose(encoded[1:], torch.full((3, 8), 3.0))
    assert trainer.condition_encoder.encode_global_only_calls == 1


def test_encode_graph_conditioning_appends_stage_tokens_when_enabled():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(
        graph_conditioning_mode="node_sequence",
        context_dim=8,
        condition_use_reference_room_maps=False,
        puzzle_stage_conditioning_enabled=True,
        puzzle_stage_token_scale=0.20,
    )
    trainer.device = torch.device("cpu")
    trainer.condition_encoder = _DummyRoomAwareConditionEncoder(output_dim=8)
    trainer.encode_to_latent = lambda room_map: torch.full((1, 4, 2, 2), 1.0, dtype=torch.float32)

    graph_dict = _make_room_condition_graph_dict(
        puzzle_stage_condition={
            "gate_family": "switch",
            "sequence_required": True,
            "controlled_doors": ["E"],
            "stage_sequence": [
                {"stage_index": 0, "kind": "push_block_to_switch", "local_anchor": [5, 5]},
                {"stage_index": 1, "kind": "reach_exit", "local_anchor": [5, 9]},
            ],
        },
    )

    encoded = DiffusionTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert tuple(encoded.shape) == (5, 8)
    assert not torch.allclose(encoded[-1], encoded[-2])


def test_get_dummy_conditioning_returns_deterministic_null_conditioning():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(graph_conditioning_mode="node_sequence", context_dim=8)
    trainer.device = torch.device("cpu")

    conditioning = DiffusionTrainer.get_dummy_conditioning(trainer, batch_size=3)

    assert tuple(conditioning.shape) == (3, 1, 8)
    assert torch.count_nonzero(conditioning) == 0


def test_stack_diffusion_graph_batch_canonicalizes_mixed_anchor_semantics():
    trainer = _make_stub_trainer(context_dim=8)

    graph_list = [
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
            "has_room_anchor": True,
        },
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
        },
    ]

    stacked = DiffusionTrainer._stack_diffusion_graph_batch(trainer, graph_list)

    assert stacked["has_room_anchor"] is True
    assert tuple(stacked["node_features"].shape) == (2, 3, 6)
    assert tuple(stacked["node_mask"].shape) == (2, 3)


def test_train_epoch_and_validate_pass_batched_room_topology_into_logicnet():
    trainer = _make_stub_trainer(context_dim=8)

    real_maps = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)
    graph_list = [
        {
            "n": 3,
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
            "room_topology_map": torch.randn(ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH),
            "boundary_constraints": torch.zeros(8, dtype=torch.float32),
        },
        {
            "n": 4,
            "node_features": torch.randn(4, 6),
            "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            "tpe": torch.randn(4, 8),
            "node_positions": torch.randn(4, 2),
            "room_topology_map": torch.randn(ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH),
            "boundary_constraints": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        },
    ]
    dataloader = [(real_maps, graph_list)]

    DiffusionTrainer.train_epoch(trainer, dataloader)
    _ = DiffusionTrainer.validate(trainer, dataloader, num_samples=2)

    assert trainer.last_train_logic_graph_data is not None
    assert trainer.last_train_diffusion_graph_data is not None
    assert tuple(trainer.last_train_logic_graph_data["room_topology_map"].shape) == (2, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH)
    assert tuple(trainer.last_train_logic_graph_data["boundary_constraints"].shape) == (2, 8)
    assert tuple(trainer.last_train_logic_graph_data["edge_features"].shape[:2]) == (2, 3)
    assert tuple(trainer.last_train_logic_graph_data["current_node_idx"].shape) == (2,)
    assert tuple(trainer.last_train_logic_graph_data["start_node_id"].shape) == (2,)
    assert tuple(trainer.logic_net.last_graph_data["room_topology_map"].shape) == (2, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH)
    assert tuple(trainer.logic_net.last_graph_data["boundary_constraints"].shape) == (2, 8)
    assert tuple(trainer.logic_net.last_graph_data["edge_features"].shape[:2]) == (2, 3)
    assert tuple(trainer.logic_net.last_graph_data["current_node_idx"].shape) == (2,)


def test_normalize_diffusion_graph_sample_uses_rwse_fallback_for_missing_tpe_and_positions():
    trainer = _make_stub_trainer(context_dim=8)

    graph_dict = {
        "node_features": torch.randn(4, 6),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "tpe": torch.randn(2, 5),
        "node_positions": torch.randn(3, 1),
    }

    sample = DiffusionTrainer._normalize_diffusion_graph_sample(trainer, graph_dict)

    assert tuple(sample["tpe"].shape) == (4, 8)
    assert tuple(sample["current_node_distance"].shape) == (4, 4)
    assert tuple(sample["node_positions"].shape) == (4, 2)
    assert torch.isfinite(sample["tpe"]).all()
    assert torch.isfinite(sample["current_node_distance"]).all()
    assert torch.isfinite(sample["node_positions"]).all()
    assert float(sample["tpe"].abs().sum().item()) > 0.0


def test_normalize_diffusion_graph_sample_builds_current_node_distance_from_anchor():
    trainer = _make_stub_trainer(context_dim=8)
    trainer.config.current_node_distance_max = 4

    graph_dict = {
        "node_features": torch.randn(4, 6),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "current_node_idx": 2,
    }

    sample = DiffusionTrainer._normalize_diffusion_graph_sample(trainer, graph_dict)

    assert tuple(sample["current_node_distance"].shape) == (4, 4)
    assert float(sample["current_node_distance"][2, 0]) == pytest.approx(0.0)
    assert float(sample["current_node_distance"][2, 1]) == pytest.approx(0.0)
    assert float(sample["current_node_distance"][2, 2]) == pytest.approx(0.0)
    assert float(sample["current_node_distance"][2, 3]) == pytest.approx(1.0)
    assert float(sample["current_node_distance"][3, 1]) == pytest.approx(0.25)


def test_stack_diffusion_graph_batch_rejects_mixed_topology_map_presence():
    trainer = _make_stub_trainer(context_dim=8)

    graph_list = [
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
            "room_topology_map": torch.randn(ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH),
        },
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
        },
    ]

    with pytest.raises(ValueError, match="room_topology_map must be present"):
        DiffusionTrainer._stack_diffusion_graph_batch(trainer, graph_list)


def test_stack_diffusion_graph_batch_warns_when_topology_shapes_disable_stacking(caplog):
    trainer = _make_stub_trainer(context_dim=8)

    graph_list = [
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
            "room_topology_map": torch.randn(ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH),
        },
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
            "room_topology_map": torch.randn(ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT - 1, ROOM_WIDTH),
        },
    ]

    with caplog.at_level("WARNING"):
        stacked = DiffusionTrainer._stack_diffusion_graph_batch(trainer, graph_list)

    assert "room_topology_map" not in stacked
    assert "Disabling batched room_topology_map stacking due to shape mismatch" in caplog.text


def test_train_step_skips_nonfinite_diffusion_loss():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        alpha_visual=1.0,
        alpha_logic=0.1,
        logic_loss_mode="predicted_latent",
        grad_clip_norm=1.0,
        epochs=1,
    )
    trainer.diffusion = _DummyTrainingLossModule(float("nan"))
    trainer.condition_encoder = _TinyModule()
    trainer.logic_net = _DummyLogicNet()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters()) + list(trainer.condition_encoder.parameters()),
        lr=1e-3,
    )
    trainer.global_step = 0
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda real_maps: torch.zeros((real_maps.shape[0], 4, 2, 2), dtype=torch.float32)
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)

    metrics = DiffusionTrainer.train_step(trainer, torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32))

    assert metrics["skipped_nonfinite_batch"] == pytest.approx(1.0)
    assert metrics["loss"] == pytest.approx(0.0)
    assert trainer.global_step == 1


def test_validate_skips_nonfinite_generated_samples():
    trainer = _make_stub_trainer(context_dim=8)
    trainer.ema_diffusion = _NaNEvalModel()
    trainer._nonfinite_warning_counts = {}

    graph_list = _make_node_sequence_graph_list()
    batch = (torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32), graph_list)

    metrics = DiffusionTrainer.validate(trainer, [batch], num_samples=2)

    assert metrics["val_diffusion_loss"] == pytest.approx(float("inf"))
    assert metrics["val_logic_loss"] == pytest.approx(float("inf"))
    assert metrics["val_total_loss"] == pytest.approx(float("inf"))
    assert metrics["val_solvability"] == pytest.approx(0.0)
    assert metrics["val_skipped_nonfinite"] == pytest.approx(4.0)


def test_state_dict_is_finite_rejects_nan_weights():
    state_dict = {"weight": torch.tensor([1.0, float("nan")])}
    assert DiffusionTrainer._state_dict_is_finite(state_dict) is False
