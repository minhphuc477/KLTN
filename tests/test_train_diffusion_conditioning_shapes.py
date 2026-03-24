# pyright: reportPrivateUsage=false

from types import SimpleNamespace

import torch

from src.train_diffusion import DiffusionTrainer


class _DummyEvalModel:
    def __init__(self):
        self.last_conditioning = None

    def eval(self):
        return self

    def sample(self, conditioning, shape):
        self.last_conditioning = conditioning
        return torch.zeros(shape, dtype=torch.float32)


class _DummyLogicNet(torch.nn.Module):
    def forward(self, z_latent, graph_data=None):
        _ = graph_data
        return torch.tensor(0.25, dtype=torch.float32), {}


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

    def _train_step_stub(real_maps, conditioning=None, include_logic_loss=True, logic_graph_data=None):
        _ = (real_maps, include_logic_loss, logic_graph_data)
        trainer.last_train_conditioning_shape = tuple(conditioning.shape)
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

    real_maps = torch.zeros((2, 1, 11, 16), dtype=torch.float32)
    graph_list = [{"n": 3}, {"n": 5}]
    dataloader = [(real_maps, graph_list)]

    DiffusionTrainer.train_epoch(trainer, dataloader)

    assert trainer.last_train_conditioning_shape == (2, 5, 8)


def test_validate_node_sequence_conditioning_is_batched_and_padded():
    trainer = _make_stub_trainer(context_dim=8)

    real_maps = torch.zeros((2, 1, 11, 16), dtype=torch.float32)
    graph_list = [{"n": 3}, {"n": 5}]
    dataloader = [(real_maps, graph_list)]

    _metrics = DiffusionTrainer.validate(trainer, dataloader, num_samples=2)

    assert tuple(trainer.ema_diffusion.last_conditioning.shape) == (2, 5, 8)
