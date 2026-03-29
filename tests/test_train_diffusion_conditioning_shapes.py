# pyright: reportPrivateUsage=false

from types import SimpleNamespace

import pytest
import torch

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH
from src.train_diffusion import DiffusionTrainer


class _DummyEvalModel:
    def __init__(self):
        self.last_conditioning = None
        self.last_graph_data = None

    def eval(self):
        return self

    def sample(self, conditioning, shape, graph_data=None):
        self.last_conditioning = conditioning
        self.last_graph_data = graph_data
        return torch.zeros(shape, dtype=torch.float32)


class _DummyLogicNet(torch.nn.Module):
    def forward(self, z_latent, graph_data=None):
        _ = graph_data
        return torch.tensor(0.25, dtype=torch.float32), {}


class _DummyRoomAwareConditionEncoder:
    def __init__(self, output_dim: int = 8):
        self.output_dim = output_dim
        self.captured_current_node_idx = None

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
        current_node_idx=None,
        style_id=None,
    ):
        _ = (neighbor_latents, boundary_constraints, position, node_features, edge_index, edge_features, tpe, style_id)
        self.captured_current_node_idx = current_node_idx
        return torch.full((1, self.output_dim), 7.0, dtype=torch.float32)

    def encode_global_only(
        self,
        node_features,
        edge_index,
        edge_features=None,
        tpe=None,
    ):
        _ = (edge_index, edge_features, tpe)
        return torch.full((int(node_features.shape[0]), self.output_dim), 3.0, dtype=torch.float32)


def _make_stub_trainer(context_dim: int = 8) -> DiffusionTrainer:
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(
        graph_conditioning_mode="node_sequence",
        context_dim=context_dim,
        warmup_epochs=0,
        alpha_logic=0.1,
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
        _ = (real_maps, include_logic_loss, logic_graph_data)
        trainer.last_train_conditioning_shape = tuple(conditioning.shape)
        trainer.last_train_diffusion_graph_data = diffusion_graph_data
        return {
            "loss": 0.0,
            "diffusion_loss": 0.0,
            "logic_loss": 0.0,
            "solvability": 1.0,
        }

    trainer.train_step = _train_step_stub
    return trainer


def test_train_epoch_node_sequence_conditioning_is_batched_and_padded():
    trainer = _make_stub_trainer(context_dim=8)

    real_maps = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)
    graph_list = [
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
    dataloader = [(real_maps, graph_list)]

    DiffusionTrainer.train_epoch(trainer, dataloader)

    assert trainer.last_train_conditioning_shape == (2, 5, 8)
    assert tuple(trainer.last_train_diffusion_graph_data["node_positions"].shape) == (2, 5, 2)
    assert tuple(trainer.last_train_diffusion_graph_data["node_mask"].shape) == (2, 5)
    assert tuple(trainer.last_train_diffusion_graph_data["edge_index"].shape[:2]) == (2, 2)
    assert torch.allclose(
        trainer.last_train_diffusion_graph_data["node_positions"][0, :3],
        graph_list[0]["node_positions"],
    )
    assert torch.allclose(
        trainer.last_train_diffusion_graph_data["tpe"][1, :5],
        graph_list[1]["tpe"],
    )


def test_validate_node_sequence_conditioning_is_batched_and_padded():
    trainer = _make_stub_trainer(context_dim=8)

    real_maps = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)
    graph_list = [
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
    dataloader = [(real_maps, graph_list)]

    _metrics = DiffusionTrainer.validate(trainer, dataloader, num_samples=2)

    assert tuple(trainer.ema_diffusion.last_conditioning.shape) == (2, 5, 8)
    assert tuple(trainer.ema_diffusion.last_graph_data["node_positions"].shape) == (2, 5, 2)
    assert tuple(trainer.ema_diffusion.last_graph_data["node_mask"].shape) == (2, 5)
    assert torch.allclose(
        trainer.ema_diffusion.last_graph_data["node_positions"][0, :3],
        graph_list[0]["node_positions"],
    )
    assert _metrics["val_logic_loss"] == pytest.approx(0.25)
    assert _metrics["val_solvability_proxy"] == pytest.approx(float(torch.exp(torch.tensor(-0.25)).item()))
    assert 0.0 <= _metrics["val_solvability_proxy"] <= 1.0


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


def test_stack_diffusion_graph_batch_rejects_mixed_anchor_semantics():
    trainer = _make_stub_trainer(context_dim=8)

    graph_list = [
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
            "boundary_constraints": torch.zeros(8, dtype=torch.float32),
            "room_position": torch.zeros(2, dtype=torch.float32),
        },
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
        },
    ]

    with pytest.raises(ValueError, match="Mixed graph anchor semantics"):
        DiffusionTrainer._stack_diffusion_graph_batch(trainer, graph_list)


def test_stack_diffusion_graph_batch_rejects_mixed_topology_map_presence():
    trainer = _make_stub_trainer(context_dim=8)

    graph_list = [
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
            "room_topology_map": torch.randn(18, ROOM_HEIGHT, ROOM_WIDTH),
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
            "room_topology_map": torch.randn(18, ROOM_HEIGHT, ROOM_WIDTH),
        },
        {
            "node_features": torch.randn(3, 6),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "tpe": torch.randn(3, 8),
            "node_positions": torch.randn(3, 2),
            "room_topology_map": torch.randn(18, ROOM_HEIGHT - 1, ROOM_WIDTH),
        },
    ]

    with caplog.at_level("WARNING"):
        stacked = DiffusionTrainer._stack_diffusion_graph_batch(trainer, graph_list)

    assert "room_topology_map" not in stacked
    assert "Disabling batched room_topology_map stacking due to shape mismatch" in caplog.text
