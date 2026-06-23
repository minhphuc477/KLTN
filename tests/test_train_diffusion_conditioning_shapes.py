# pyright: reportPrivateUsage=false

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.train_diffusion as train_diffusion_module
from src.core.definitions import GRAPH_EDGE_FEATURE_DIM, GRAPH_TPE_DIM, ROOM_HEIGHT, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.latent_diffusion import create_latent_diffusion
from src.core.logic_net import LogicNet
from src.core.vqvae import create_vqvae
from src.train_diffusion import DiffusionTrainer
from src.utils.frozen_latent_cache import FrozenLatentCache

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


class _CountingVQVAE:
    def __init__(self):
        self.num_classes = 44
        self.encode_calls = 0

    def encode(self, x_onehot):
        self.encode_calls += 1
        latent_value = x_onehot.mean(dim=(1, 2, 3), keepdim=True)
        return latent_value.expand(-1, 4, 2, 2).contiguous(), torch.zeros(x_onehot.shape[0], dtype=torch.long)


class _DecodeTrackingVQVAE:
    def __init__(self):
        self.decode_calls = 0
        self.last_latent_requires_grad = None

    def decode(self, latent, target_size=None):
        self.decode_calls += 1
        self.last_latent_requires_grad = bool(latent.requires_grad)
        batch_size = int(latent.shape[0])
        height, width = target_size or (ROOM_HEIGHT, ROOM_WIDTH)
        base = latent.mean(dim=1, keepdim=True)
        logits = torch.zeros(batch_size, 44, height, width, device=latent.device, dtype=latent.dtype)
        logits[:, 1:2] = torch.nn.functional.interpolate(
            base,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return logits


class _QuantizeTrackingVQVAE(_DecodeTrackingVQVAE):
    def __init__(self):
        super().__init__()
        self.quantize_calls = 0
        self.last_quantize_input_requires_grad = None

    def quantize(self, latent):
        self.quantize_calls += 1
        self.last_quantize_input_requires_grad = bool(latent.requires_grad)
        return latent + (latent.round() - latent).detach(), torch.zeros((), dtype=latent.dtype), torch.zeros(
            latent.shape[0],
            latent.shape[2],
            latent.shape[3],
            dtype=torch.long,
            device=latent.device,
        )


def test_configure_guidance_wires_logic_net_and_sampling_limits():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.logic_net = object()
    trainer.diffusion = SimpleNamespace(guidance=SimpleNamespace())
    trainer.config = SimpleNamespace(
        guidance_scale=1.75,
        guidance_clamp_magnitude=0.5,
        guidance_relative_norm_cap=0.125,
        guidance_schedule_enabled=False,
        guidance_active_fraction=0.6,
        guidance_decay_power=2.0,
        guidance_max_graph_nodes=64,
        guidance_max_key_lock_pairs=128,
        guidance_max_guidance_elements=4096,
    )

    DiffusionTrainer._configure_guidance(trainer)

    guidance = trainer.diffusion.guidance
    assert guidance.logic_net is trainer.logic_net
    assert guidance.guidance_scale == pytest.approx(1.75)
    assert guidance.clamp_magnitude == pytest.approx(0.5)
    assert guidance.relative_norm_cap == pytest.approx(0.125)
    assert guidance.schedule_enabled is False
    assert guidance.active_fraction == pytest.approx(0.6)
    assert guidance.decay_power == pytest.approx(2.0)
    assert guidance.max_graph_nodes == 64
    assert guidance.max_key_lock_pairs == 128
    assert guidance.max_guidance_elements == 4096


def test_configure_guidance_does_not_register_logic_net_inside_diffusion_state():
    guidance = torch.nn.Module()
    diffusion = torch.nn.Module()
    diffusion.guidance = guidance
    logic_net = torch.nn.Linear(2, 2)
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.logic_net = logic_net
    trainer.diffusion = diffusion
    trainer.config = SimpleNamespace()

    DiffusionTrainer._configure_guidance(trainer)

    assert guidance.logic_net is logic_net
    assert "logic_net" not in guidance._modules
    assert not any("logic_net" in key for key in diffusion.state_dict())


def test_predicted_latent_logic_branch_backpropagates_from_vqvae_decode_to_unet():
    torch.manual_seed(7)
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        num_classes=44,
        latent_dim=4,
        alpha_visual=1.0,
        alpha_logic=0.1,
        alpha_logic_tile=0.0,
        logic_net_enabled=True,
        logic_net_trainable=True,
        logic_loss_mode="predicted_latent",
        grad_clip_norm=0.0,
        latent_cache_enabled=False,
        latent_cache_max_items=0,
        graph_spatial_alignment_weight=0.0,
        epochs=1,
        learning_rate=1e-3,
    )
    trainer.vqvae = create_vqvae(
        num_classes=44,
        latent_dim=4,
        hidden_dim=8,
        codebook_size=8,
        use_coordconv=False,
    )
    trainer.vqvae.eval()
    for param in trainer.vqvae.parameters():
        param.requires_grad = False
    trainer.diffusion = create_latent_diffusion(
        latent_dim=4,
        context_dim=8,
        num_timesteps=4,
        model_channels=8,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=1,
        cfg_dropout_prob=0.0,
        room_topology_channels=ROOM_TOPOLOGY_CHANNEL_COUNT,
        min_snr_gamma=0.0,
    )
    trainer.model = trainer.diffusion
    trainer.condition_encoder = torch.nn.Linear(1, 1)
    trainer.logic_net = LogicNet(latent_dim=4, hidden_dim=8, num_classes=44, num_iterations=2)
    trainer.optimizer = torch.optim.AdamW(
        list(trainer.diffusion.parameters())
        + list(trainer.condition_encoder.parameters())
        + list(trainer.logic_net.parameters()),
        lr=1e-3,
    )
    for group in trainer.optimizer.param_groups:
        group.setdefault("base_lr", 1e-3)
    trainer.ema_diffusion = create_latent_diffusion(
        latent_dim=4,
        context_dim=8,
        num_timesteps=4,
        model_channels=8,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=1,
        cfg_dropout_prob=0.0,
        room_topology_channels=ROOM_TOPOLOGY_CHANNEL_COUNT,
        min_snr_gamma=0.0,
    )
    trainer.ema_decay = 0.0
    trainer.global_step = 0
    trainer._estimated_total_steps = 1
    trainer._latent_cache = FrozenLatentCache(enabled=False, max_items=0)
    trainer._accelerator = None
    trainer.distributed_context = None
    trainer._nonfinite_warning_counts = {}

    real_maps = torch.randint(0, 44, (1, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)
    conditioning = torch.zeros(1, 8)

    metrics = DiffusionTrainer.train_step(
        trainer,
        real_maps,
        conditioning=conditioning,
        include_logic_loss=True,
    )

    unet_grad_norm = sum(
        float(param.grad.detach().abs().sum().item())
        for param in trainer.diffusion.denoiser.parameters()
        if param.grad is not None
    )
    assert float(metrics.get("skipped_nonfinite_batch", 0.0)) == 0.0
    assert metrics["logic_loss"] >= 0.0
    assert unet_grad_norm > 0.0


class _TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class _TinyCheckpointModule(_TinyModule):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight.data.fill_(float(value))
        self.guidance = SimpleNamespace()


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


class _FiniteDifferentiableTrainingLossModule(_DummyTrainingLossModule):
    def __init__(self):
        super().__init__(0.25)

    def training_loss(self, z_0, conditioning, graph_data=None):
        _ = (z_0, conditioning, graph_data)
        return self.weight * 0.25


class _DPOTrainingModule(_TinyModule):
    def train(self):
        return self

    def diffusion_dpo_loss(
        self,
        chosen_z,
        rejected_z,
        conditioning,
        **kwargs,
    ):
        _ = (chosen_z, rejected_z, conditioning, kwargs)
        loss = self.weight * 0.5
        return loss, {
            "dpo_margin": self.weight.detach() * 0.25,
            "dpo_accuracy": torch.ones((), dtype=torch.float32),
            "chosen_score": torch.ones((), dtype=torch.float32),
            "rejected_score": torch.zeros((), dtype=torch.float32),
        }


class _ComputeLossModule(_TinyModule):
    def __init__(self):
        super().__init__()
        self.training_objective = "diffusion"
        self.calls: list[tuple[str, object]] = []

    def compute_loss(self, z_0, conditioning, graph_data=None):
        _ = (z_0, conditioning)
        self.calls.append((self.training_objective, graph_data))
        return torch.tensor(0.75, dtype=torch.float32)


class _TinyDiffusionWithDenoiser(_DummyTrainingLossModule):
    def __init__(self):
        super().__init__(0.25)
        self.num_timesteps = 4
        self.prediction_type = "epsilon"
        class _KwargDenoiser(torch.nn.Conv2d):
            def forward(self, x, t, context, **kwargs):
                _ = (t, context, kwargs)
                return super().forward(x)

        self.denoiser = _KwargDenoiser(4, 4, kernel_size=1)
        self.register_buffer("sqrt_alphas_cumprod", torch.ones(4))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.full((4,), 0.1))

    def q_sample(self, z_0, t, noise):
        _ = t
        return z_0 + 0.1 * noise

    def _extract_context_topology(self, conditioning, graph_data):
        _ = (conditioning, graph_data)
        return None, None

    def _extract_spatial_graph_context(self, conditioning, graph_data):
        _ = (conditioning, graph_data)
        return None


class _DummyRoomAwareConditionEncoder:
    def __init__(self, output_dim: int = 8):
        self.output_dim = output_dim
        self.captured_current_node_idx = None
        self.captured_current_node_distance = None
        self.captured_batch_idx = None
        self.captured_node_mask = None
        self.captured_edge_rrwp = None
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
        edge_rrwp=None,
        tpe=None,
        current_node_distance=None,
        batch_idx=None,
        node_mask=None,
        current_node_idx=None,
        reference_room_maps=None,
        style_id=None,
        return_global_tokens=False,
    ):
        _ = (boundary_constraints, position, node_features, edge_index, edge_features, tpe, batch_idx, style_id)
        self.captured_current_node_idx = current_node_idx
        self.captured_current_node_distance = current_node_distance
        self.captured_batch_idx = batch_idx
        self.captured_node_mask = node_mask
        self.captured_edge_rrwp = edge_rrwp
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
        edge_rrwp=None,
        tpe=None,
        current_node_distance=None,
        batch_idx=None,
        node_mask=None,
    ):
        _ = (edge_index, edge_features, tpe, current_node_distance, batch_idx)
        self.captured_batch_idx = batch_idx
        self.captured_node_mask = node_mask
        self.captured_edge_rrwp = edge_rrwp
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
        force_optimizer_step=False,
    ):
        _ = (real_maps, include_logic_loss, force_optimizer_step)
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


def test_diffusion_objective_loss_delegates_to_model_compute_loss():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(diffusion_training_objective="flow_matching")
    model = _ComputeLossModule()
    graph_data = {"node_features": torch.zeros(1, 2)}

    loss = DiffusionTrainer._diffusion_objective_loss(
        trainer,
        torch.zeros(1, 4, 2, 2),
        torch.zeros(1, 8),
        graph_data=graph_data,
        model=model,
    )

    assert loss.item() == pytest.approx(0.75)
    assert model.calls == [("flow_matching", graph_data)]
    assert model.training_objective == "diffusion"


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


def test_try_stack_dungeon_scope_graph_batch_collapses_full_room_set():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        edge_feature_dim=GRAPH_EDGE_FEATURE_DIM,
        current_node_distance_max=8,
        graph_conditioning_mode="node_sequence",
    )

    node_features = torch.zeros(3, 6)
    node_features[2, 3] = 1.0
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    base = {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": torch.zeros(2, GRAPH_EDGE_FEATURE_DIM),
        "edge_attr": torch.tensor([0, 1], dtype=torch.long),
        "edge_rrwp": torch.zeros(2, GRAPH_TPE_DIM),
        "tpe": torch.zeros(3, GRAPH_TPE_DIM),
        "node_positions": torch.zeros(3, 2),
        "node_mask": torch.ones(3),
        "num_nodes": 3,
        "num_edges": 2,
        "start_node_id": 0,
        "target_idx": 2,
        "node_to_idx": {"a": 0, "b": 1, "c": 2},
        "room_topology_map": torch.zeros(ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH),
        "boundary_constraints": torch.zeros(8),
    }
    graph_list = []
    for idx in (0, 1, 2):
        item = dict(base)
        item["current_node_idx"] = idx
        graph_list.append(item)

    graph_data = DiffusionTrainer._try_stack_dungeon_scope_graph_batch(trainer, graph_list)

    assert graph_data is not None
    assert graph_data["graph_scope"] == "dungeon"
    assert tuple(graph_data["node_features"].shape) == (3, 6)
    assert graph_data["current_node_idx"].tolist() == [0, 1, 2]
    assert tuple(graph_data["room_topology_map"].shape) == (
        3,
        ROOM_TOPOLOGY_CHANNEL_COUNT,
        ROOM_HEIGHT,
        ROOM_WIDTH,
    )


def test_wfc_pseudo_label_loss_is_opt_in_and_backpropagates(monkeypatch):
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.global_step = 0
    trainer.config = SimpleNamespace(
        alpha_wfc_pseudo=1.0,
        wfc_pseudo_max_samples=1,
        wfc_pseudo_confidence_threshold=0.99,
        num_classes=5,
        seed=123,
    )
    pred_tile_logits = torch.randn(1, 5, ROOM_HEIGHT, ROOM_WIDTH, requires_grad=True)
    real_maps = torch.zeros(1, 1, ROOM_HEIGHT, ROOM_WIDTH)
    monkeypatch.setattr(
        "src.train_diffusion.integrate_weighted_wfc_into_pipeline",
        lambda *_args, **_kwargs: {"grid": np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int64)},
    )

    loss, sample_count, repaired_mean = DiffusionTrainer._wfc_pseudo_label_loss(
        trainer,
        pred_tile_logits,
        real_maps,
    )
    loss.backward()

    assert sample_count == pytest.approx(1.0)
    assert loss.item() >= 0.0
    assert repaired_mean.item() >= 0.0
    assert pred_tile_logits.grad is not None
    assert pred_tile_logits.grad.abs().sum().item() > 0.0


def test_wfc_pseudo_label_loss_scales_subset_by_repaired_samples(monkeypatch):
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.global_step = 0
    trainer.config = SimpleNamespace(
        alpha_wfc_pseudo=1.0,
        wfc_pseudo_max_samples=1,
        wfc_pseudo_confidence_threshold=0.0,
        num_classes=5,
        seed=123,
    )
    pred_tile_logits = torch.zeros(4, 5, ROOM_HEIGHT, ROOM_WIDTH, requires_grad=True)
    real_maps = torch.zeros(4, 1, ROOM_HEIGHT, ROOM_WIDTH)

    monkeypatch.setattr(
        "src.train_diffusion.integrate_weighted_wfc_into_pipeline",
        lambda *_args, **_kwargs: {"grid": np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int64)},
    )

    scaled_loss, sample_count, repaired_mean = DiffusionTrainer._wfc_pseudo_label_loss(
        trainer,
        pred_tile_logits,
        real_maps,
    )

    assert sample_count == pytest.approx(1.0)
    assert scaled_loss.item() == pytest.approx(repaired_mean.item())


def test_diffusion_adamw_groups_exclude_bias_and_norm_from_weight_decay():
    module = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.LayerNorm(4),
    )

    groups = DiffusionTrainer._adamw_decay_param_groups("probe", module, weight_decay=0.1)
    by_name = {group["name"]: group for group in groups}

    assert by_name["probe_decay"]["weight_decay"] == pytest.approx(0.1)
    assert by_name["probe_no_decay"]["weight_decay"] == pytest.approx(0.0)
    decay_ids = {id(param) for param in by_name["probe_decay"]["params"]}
    no_decay_ids = {id(param) for param in by_name["probe_no_decay"]["params"]}
    assert id(module[0].weight) in decay_ids
    assert id(module[0].bias) in no_decay_ids
    assert id(module[1].weight) in no_decay_ids
    assert id(module[1].bias) in no_decay_ids


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
        "batch_idx": torch.tensor([0, 0, 1], dtype=torch.long),
        "node_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
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
    assert torch.equal(trainer.condition_encoder.captured_batch_idx, torch.tensor([0, 0, 1], dtype=torch.long))
    assert torch.equal(trainer.condition_encoder.captured_node_mask, graph_dict["node_mask"])
    assert tuple(trainer.condition_encoder.captured_edge_rrwp.shape) == (2, GRAPH_TPE_DIM)


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


def test_encode_graph_conditioning_passes_batch_idx_to_global_only_path():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(graph_conditioning_mode="node_sequence", context_dim=8)
    trainer.device = torch.device("cpu")
    trainer.condition_encoder = _DummyRoomAwareConditionEncoder(output_dim=8)

    graph_dict = {
        "node_features": torch.randn(4, 6),
        "edge_index": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        "edge_attr": torch.tensor([0, 1], dtype=torch.long),
        "tpe": torch.randn(4, 8),
        "batch_idx": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "node_mask": torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
    }

    _encoded = DiffusionTrainer._encode_graph_conditioning(trainer, graph_dict)

    assert trainer.condition_encoder.encode_global_only_calls == 1
    assert torch.equal(trainer.condition_encoder.captured_batch_idx, graph_dict["batch_idx"])
    assert torch.equal(trainer.condition_encoder.captured_node_mask, graph_dict["node_mask"])


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


def test_encode_to_latent_reuses_frozen_vqvae_cache_for_repeated_maps():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        latent_cache_enabled=True,
        latent_cache_max_items=8,
        vqvae_checkpoint="stub-vqvae.pth",
        vqvae_architecture="vqvae2",
        num_classes=44,
    )
    trainer.vqvae = _CountingVQVAE()
    trainer._latent_cache = FrozenLatentCache(enabled=True, max_items=8)

    maps = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)
    maps[1] = 0.5

    first = DiffusionTrainer.encode_to_latent(trainer, maps)
    second = DiffusionTrainer.encode_to_latent(trainer, maps)

    assert tuple(first.shape) == (2, 4, 2, 2)
    assert torch.allclose(first, second)
    assert trainer.vqvae.encode_calls == 1
    assert trainer._latent_cache.hits == 2


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
    trainer._accelerator = None
    trainer.global_step = 0
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda real_maps: torch.zeros((real_maps.shape[0], 4, 2, 2), dtype=torch.float32)
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)

    metrics = DiffusionTrainer.train_step(trainer, torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32))

    assert metrics["skipped_nonfinite_batch"] == pytest.approx(1.0)
    assert metrics["loss"] == pytest.approx(0.0)
    assert trainer.global_step == 0


def test_train_step_accumulates_gradients_before_optimizer_step():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        alpha_visual=1.0,
        alpha_logic=0.0,
        alpha_logic_tile=0.0,
        alpha_wfc_pseudo=0.0,
        logic_loss_mode="predicted_latent",
        logic_net_enabled=False,
        logic_net_trainable=False,
        grad_clip_norm=0.0,
        gradient_accumulation_steps=2,
        epochs=1,
        learning_rate=1e-3,
        global_lr_warmup_epochs=0,
        logic_lr_warmup_epochs=0,
    )
    trainer.diffusion = _FiniteDifferentiableTrainingLossModule()
    trainer.condition_encoder = _TinyModule()
    trainer.logic_net = _DummyLogicNet()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters()) + list(trainer.condition_encoder.parameters()),
        lr=1e-3,
    )
    trainer._accelerator = None
    trainer.global_step = 0
    trainer._accumulation_micro_steps = 0
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda real_maps: torch.zeros((real_maps.shape[0], 4, 2, 2), dtype=torch.float32)
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)
    trainer._gradients_are_finite = lambda: True

    step_calls = 0
    ema_calls = 0
    warmup_calls = 0

    def _step():
        nonlocal step_calls
        step_calls += 1

    def _update_ema():
        nonlocal ema_calls
        ema_calls += 1

    def _warmup(*, completed_steps=None):
        _ = completed_steps
        nonlocal warmup_calls
        warmup_calls += 1

    trainer.optimizer.step = _step
    trainer._update_ema = _update_ema
    trainer._apply_lr_warmup = _warmup

    first = DiffusionTrainer.train_step(
        trainer,
        torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
        include_logic_loss=False,
    )
    assert first["optimizer_step"] == pytest.approx(0.0)
    assert trainer.global_step == 0
    assert step_calls == 0
    assert ema_calls == 0
    assert trainer._accumulation_micro_steps == 1
    assert trainer.diffusion.weight.grad is not None

    second = DiffusionTrainer.train_step(
        trainer,
        torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
        include_logic_loss=False,
    )
    assert second["optimizer_step"] == pytest.approx(1.0)
    assert trainer.global_step == 1
    assert step_calls == 1
    assert ema_calls == 1
    assert warmup_calls == 1
    assert trainer._accumulation_micro_steps == 0
    assert trainer.diffusion.weight.grad is not None


def test_train_step_nonfinite_gradients_do_not_create_ghost_step():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        alpha_visual=1.0,
        alpha_logic=0.0,
        alpha_logic_tile=0.0,
        alpha_wfc_pseudo=0.0,
        logic_loss_mode="predicted_latent",
        logic_net_enabled=False,
        logic_net_trainable=False,
        grad_clip_norm=1.0,
        epochs=1,
    )
    trainer.diffusion = _FiniteDifferentiableTrainingLossModule()
    trainer.condition_encoder = _TinyModule()
    trainer.logic_net = _DummyLogicNet()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters()) + list(trainer.condition_encoder.parameters()),
        lr=1e-3,
    )
    trainer._accelerator = None
    trainer.global_step = 7
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda real_maps: torch.zeros((real_maps.shape[0], 4, 2, 2), dtype=torch.float32)
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)
    trainer._gradients_are_finite = lambda: False

    step_calls = 0
    ema_calls = 0
    warmup_calls = 0

    def _step():
        nonlocal step_calls
        step_calls += 1

    def _update_ema():
        nonlocal ema_calls
        ema_calls += 1

    def _warmup(*, completed_steps=None):
        _ = completed_steps
        nonlocal warmup_calls
        warmup_calls += 1

    trainer.optimizer.step = _step
    trainer._update_ema = _update_ema
    trainer._apply_lr_warmup = _warmup

    metrics = DiffusionTrainer.train_step(
        trainer,
        torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
        include_logic_loss=False,
    )

    assert metrics["skipped_nonfinite_batch"] == pytest.approx(1.0)
    assert trainer.global_step == 7
    assert step_calls == 0
    assert ema_calls == 0
    assert warmup_calls == 0


def test_train_step_nonfinite_clipped_grad_norm_does_not_create_ghost_step(monkeypatch):
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        alpha_visual=1.0,
        alpha_logic=0.0,
        alpha_logic_tile=0.0,
        alpha_wfc_pseudo=0.0,
        logic_loss_mode="predicted_latent",
        logic_net_enabled=False,
        logic_net_trainable=False,
        grad_clip_norm=1.0,
        epochs=1,
    )
    trainer.diffusion = _FiniteDifferentiableTrainingLossModule()
    trainer.condition_encoder = _TinyModule()
    trainer.logic_net = _DummyLogicNet()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters()) + list(trainer.condition_encoder.parameters()),
        lr=1e-3,
    )
    trainer._accelerator = None
    trainer.global_step = 11
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda real_maps: torch.zeros((real_maps.shape[0], 4, 2, 2), dtype=torch.float32)
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)
    trainer._gradients_are_finite = lambda: True

    step_calls = 0
    ema_calls = 0
    warmup_calls = 0

    def _step():
        nonlocal step_calls
        step_calls += 1

    def _update_ema():
        nonlocal ema_calls
        ema_calls += 1

    def _warmup(*, completed_steps=None):
        _ = completed_steps
        nonlocal warmup_calls
        warmup_calls += 1

    trainer.optimizer.step = _step
    trainer._update_ema = _update_ema
    trainer._apply_lr_warmup = _warmup
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(float("inf")),
    )

    metrics = DiffusionTrainer.train_step(
        trainer,
        torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
        include_logic_loss=False,
    )

    assert metrics["skipped_nonfinite_batch"] == pytest.approx(1.0)
    assert trainer.global_step == 11
    assert step_calls == 0
    assert ema_calls == 0
    assert warmup_calls == 0


def test_dpo_step_nonfinite_gradients_do_not_create_ghost_step():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        diffusion_training_objective="diffusion",
        grad_clip_norm=1.0,
        logic_net_trainable=False,
    )
    trainer.diffusion = _DPOTrainingModule()
    trainer.condition_encoder = _TinyModule()
    trainer.logic_net = _DummyLogicNet()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters()) + list(trainer.condition_encoder.parameters()),
        lr=1e-3,
    )
    trainer._accelerator = None
    trainer.distributed_context = None
    trainer.global_step = 13
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda maps: torch.zeros((maps.shape[0], 4, 2, 2), dtype=torch.float32)
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)
    trainer._gradients_are_finite = lambda: False

    step_calls = 0
    ema_calls = 0
    warmup_calls = 0

    def _step():
        nonlocal step_calls
        step_calls += 1

    def _update_ema():
        nonlocal ema_calls
        ema_calls += 1

    def _warmup(*, completed_steps=None):
        _ = completed_steps
        nonlocal warmup_calls
        warmup_calls += 1

    trainer.optimizer.step = _step
    trainer._update_ema = _update_ema
    trainer._apply_lr_warmup = _warmup

    metrics = DiffusionTrainer.dpo_step(
        trainer,
        torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
        torch.ones((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
    )

    assert metrics["skipped_nonfinite_batch"] == pytest.approx(1.0)
    assert trainer.global_step == 13
    assert step_calls == 0
    assert ema_calls == 0
    assert warmup_calls == 0


def test_dpo_step_accumulates_gradients_before_optimizer_step():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        diffusion_training_objective="diffusion",
        grad_clip_norm=0.0,
        logic_net_trainable=False,
        gradient_accumulation_steps=2,
    )
    trainer.diffusion = _DPOTrainingModule()
    trainer.condition_encoder = _TinyModule()
    trainer.logic_net = _DummyLogicNet()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters()) + list(trainer.condition_encoder.parameters()),
        lr=1e-3,
    )
    trainer._accelerator = None
    trainer.distributed_context = None
    trainer.global_step = 0
    trainer._accumulation_micro_steps = 0
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda maps: torch.zeros((maps.shape[0], 4, 2, 2), dtype=torch.float32)
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)
    trainer._gradients_are_finite = lambda: True

    step_calls = 0
    ema_calls = 0
    warmup_calls = 0

    def _step():
        nonlocal step_calls
        step_calls += 1

    def _update_ema():
        nonlocal ema_calls
        ema_calls += 1

    def _warmup(*, completed_steps=None):
        _ = completed_steps
        nonlocal warmup_calls
        warmup_calls += 1

    trainer.optimizer.step = _step
    trainer._update_ema = _update_ema
    trainer._apply_lr_warmup = _warmup

    first = DiffusionTrainer.dpo_step(
        trainer,
        torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
        torch.ones((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
    )
    second = DiffusionTrainer.dpo_step(
        trainer,
        torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
        torch.ones((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
    )

    assert first["optimizer_step"] == pytest.approx(0.0)
    assert first["gradient_accumulation_micro_steps"] == pytest.approx(1.0)
    assert second["optimizer_step"] == pytest.approx(1.0)
    assert trainer.global_step == 1
    assert step_calls == 1
    assert ema_calls == 1
    assert warmup_calls == 1
    assert trainer._accumulation_micro_steps == 0


def test_dpo_step_nonfinite_clipped_grad_norm_does_not_create_ghost_step(monkeypatch):
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        diffusion_training_objective="diffusion",
        grad_clip_norm=1.0,
        logic_net_trainable=False,
    )
    trainer.diffusion = _DPOTrainingModule()
    trainer.condition_encoder = _TinyModule()
    trainer.logic_net = _DummyLogicNet()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters()) + list(trainer.condition_encoder.parameters()),
        lr=1e-3,
    )
    trainer._accelerator = None
    trainer.distributed_context = None
    trainer.global_step = 17
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda maps: torch.zeros((maps.shape[0], 4, 2, 2), dtype=torch.float32)
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)
    trainer._gradients_are_finite = lambda: True

    step_calls = 0
    ema_calls = 0
    warmup_calls = 0

    def _step():
        nonlocal step_calls
        step_calls += 1

    def _update_ema():
        nonlocal ema_calls
        ema_calls += 1

    def _warmup(*, completed_steps=None):
        _ = completed_steps
        nonlocal warmup_calls
        warmup_calls += 1

    trainer.optimizer.step = _step
    trainer._update_ema = _update_ema
    trainer._apply_lr_warmup = _warmup
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(float("inf")),
    )

    metrics = DiffusionTrainer.dpo_step(
        trainer,
        torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
        torch.ones((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
    )

    assert metrics["skipped_nonfinite_batch"] == pytest.approx(1.0)
    assert trainer.global_step == 17
    assert step_calls == 0
    assert ema_calls == 0
    assert warmup_calls == 0


def test_train_step_predicted_latent_decodes_to_tile_logits_for_logic_loss():
    class _ShapeRecordingLogicNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.last_shape = None

        def forward(self, z_input, graph_data=None):
            _ = graph_data
            self.last_shape = tuple(z_input.shape)
            return z_input[:, 1].mean() * self.scale, {}

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        num_classes=44,
        alpha_visual=1.0,
        alpha_logic=0.1,
        logic_loss_mode="predicted_latent",
        logic_net_enabled=True,
        grad_clip_norm=1.0,
        epochs=1,
        learning_rate=1e-3,
        global_lr_warmup_epochs=0,
        logic_lr_warmup_epochs=0,
        logic_net_trainable=True,
    )
    trainer.diffusion = _TinyDiffusionWithDenoiser()
    trainer.condition_encoder = _TinyModule()
    trainer.logic_net = _ShapeRecordingLogicNet()
    trainer.vqvae = _DecodeTrackingVQVAE()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters())
        + list(trainer.condition_encoder.parameters())
        + list(trainer.logic_net.parameters()),
        lr=1e-3,
    )
    trainer._accelerator = None
    trainer.global_step = 0
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda real_maps: torch.zeros(
        (real_maps.shape[0], 4, 2, 2),
        dtype=torch.float32,
    )
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)

    metrics = DiffusionTrainer.train_step(
        trainer,
        torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32),
    )

    assert metrics.get("skipped_nonfinite_batch", 0.0) == pytest.approx(0.0)
    assert trainer.vqvae.decode_calls == 1
    assert trainer.vqvae.last_latent_requires_grad is True
    assert trainer.logic_net.last_shape == (2, 44, ROOM_HEIGHT, ROOM_WIDTH)


def test_decode_latent_for_logic_quantizes_before_vqvae_decode():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = SimpleNamespace(num_classes=44)
    trainer.vqvae = _QuantizeTrackingVQVAE()
    latent = torch.randn(2, 4, 2, 2, requires_grad=True)

    logits = DiffusionTrainer._decode_latent_for_logic(trainer, latent)
    loss = logits[:, 1].mean()
    loss.backward()

    assert trainer.vqvae.quantize_calls == 1
    assert trainer.vqvae.decode_calls == 1
    assert trainer.vqvae.last_quantize_input_requires_grad is True
    assert trainer.vqvae.last_latent_requires_grad is True
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_train_step_trains_logicnet_tile_classifier_when_enabled():
    class _TileClassifierLogicNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tile_classifier = torch.nn.Conv2d(4, 44, kernel_size=1)
            self.last_loss_input_shape = None

        def _project_tile_logits_to_room(self, logits):
            return torch.nn.functional.interpolate(
                logits,
                size=(ROOM_HEIGHT, ROOM_WIDTH),
                mode="bilinear",
                align_corners=False,
            )

        def forward(self, z_input, graph_data=None):
            _ = graph_data
            self.last_loss_input_shape = tuple(z_input.shape)
            return z_input[:, 1].mean(), {}

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        num_classes=44,
        alpha_visual=1.0,
        alpha_logic=0.1,
        alpha_logic_tile=0.5,
        logic_loss_mode="predicted_latent",
        logic_net_enabled=True,
        grad_clip_norm=1.0,
        epochs=1,
        learning_rate=1e-3,
        global_lr_warmup_epochs=0,
        logic_lr_warmup_epochs=0,
        logic_net_trainable=True,
    )
    trainer.diffusion = _TinyDiffusionWithDenoiser()
    trainer.condition_encoder = _TinyModule()
    trainer.logic_net = _TileClassifierLogicNet()
    trainer.vqvae = _DecodeTrackingVQVAE()
    trainer.ema_diffusion = _DummyEvalModel()
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters())
        + list(trainer.condition_encoder.parameters())
        + list(trainer.logic_net.parameters()),
        lr=1e-3,
    )
    trainer._accelerator = None
    trainer.global_step = 0
    trainer._nonfinite_warning_counts = {}
    trainer.encode_to_latent = lambda real_maps: torch.zeros(
        (real_maps.shape[0], 4, 2, 2),
        dtype=torch.float32,
    )
    trainer.get_dummy_conditioning = lambda batch_size: torch.zeros((batch_size, 1, 8), dtype=torch.float32)
    maps = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)
    maps[:, :, 2:6, 2:6] = 1.0 / 43.0

    metrics = DiffusionTrainer.train_step(trainer, maps)

    assert metrics["logic_tile_loss"] > 0.0
    assert 0.0 <= metrics["logic_tile_accuracy"] <= 1.0
    assert trainer.logic_net.tile_classifier.weight.grad is not None


def test_vqvae_codebook_stats_reports_usage_metrics():
    class _UsageVQVAE:
        def get_codebook_usage(self):
            return torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32)

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.vqvae = _UsageVQVAE()

    stats = DiffusionTrainer._vqvae_codebook_stats(trainer)

    assert stats["vqvae_codebook_active_codes"] == pytest.approx(2.0)
    assert stats["vqvae_codebook_total_codes"] == pytest.approx(4.0)
    assert stats["vqvae_codebook_active_fraction"] == pytest.approx(0.5)
    assert stats["vqvae_codebook_perplexity"] == pytest.approx(2.0)


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


def test_validate_reports_logic_tile_accuracy():
    class _PerfectTileLogicNet(_DummyLogicNet):
        def __init__(self):
            super().__init__()
            self.tile_classifier = torch.nn.Conv2d(4, 44, kernel_size=1)

        def _project_tile_logits_to_room(self, logits):
            batch_size = int(logits.shape[0])
            projected = torch.zeros(batch_size, 44, ROOM_HEIGHT, ROOM_WIDTH)
            projected[:, 0] = 10.0
            return projected

    trainer = _make_stub_trainer(context_dim=8)
    trainer.config.num_classes = 44
    trainer.logic_net = _PerfectTileLogicNet()
    trainer.encode_to_latent = lambda real_maps: torch.zeros((real_maps.shape[0], 4, 2, 2), dtype=torch.float32)
    batch = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)

    metrics = DiffusionTrainer.validate(trainer, [batch], num_samples=2, num_diffusion_samples=2)

    assert metrics["val_logic_tile_accuracy"] == pytest.approx(1.0)


def test_validate_suppresses_sampling_guidance_when_tile_accuracy_below_gate():
    class _WrongTileLogicNet(_DummyLogicNet):
        def __init__(self):
            super().__init__()
            self.tile_classifier = torch.nn.Conv2d(4, 44, kernel_size=1)

        def _project_tile_logits_to_room(self, logits):
            batch_size = int(logits.shape[0])
            projected = torch.zeros(batch_size, 44, ROOM_HEIGHT, ROOM_WIDTH)
            projected[:, 1] = 10.0
            return projected

    trainer = _make_stub_trainer(context_dim=8)
    trainer.config.num_classes = 44
    trainer.config.min_logic_tile_accuracy_for_guidance = 0.4
    trainer.logic_net = _WrongTileLogicNet()
    trainer.encode_to_latent = lambda real_maps: torch.zeros((real_maps.shape[0], 4, 2, 2), dtype=torch.float32)
    trainer.ema_diffusion.guidance = SimpleNamespace(guidance_scale=1.5)
    batch = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)

    metrics = DiffusionTrainer.validate(trainer, [batch], num_samples=2, num_diffusion_samples=2)

    assert metrics["val_logic_tile_accuracy"] == pytest.approx(0.0)
    assert metrics["val_logic_guidance_suppressed_low_tile_accuracy"] == pytest.approx(2.0)
    assert trainer.ema_diffusion.guidance.guidance_scale == pytest.approx(1.5)


def test_validate_reports_hard_solvability_from_decoded_samples():
    class _SolvableDecodeVQVAE:
        def decode(self, latent, target_size=None):
            batch_size = int(latent.shape[0])
            height, width = target_size or (ROOM_HEIGHT, ROOM_WIDTH)
            logits = torch.zeros(batch_size, 44, height, width)
            logits[:, int(SEMANTIC_PALETTE["WALL"])] = 5.0
            row = height // 2
            logits[:, :, row, :] = 0.0
            logits[:, int(SEMANTIC_PALETTE["FLOOR"]), row, :] = 5.0
            logits[:, :, row, 1] = 0.0
            logits[:, int(SEMANTIC_PALETTE["START"]), row, 1] = 6.0
            logits[:, :, row, width - 1] = 0.0
            logits[:, int(SEMANTIC_PALETTE["DOOR_OPEN"]), row, width - 1] = 6.0
            return logits

    trainer = _make_stub_trainer(context_dim=8)
    trainer.config.num_classes = 44
    trainer.vqvae = _SolvableDecodeVQVAE()
    batch = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)

    metrics = DiffusionTrainer.validate(trainer, [batch], num_samples=2, num_diffusion_samples=2)

    assert metrics["val_hard_solvability"] == pytest.approx(1.0)
    assert "val_grid_reach_loss" in metrics
    assert "val_graph_reach_loss" in metrics


def test_validate_hard_solvability_uses_only_counted_generated_samples():
    class _MixedDecodeVQVAE:
        def decode(self, latent, target_size=None):
            batch_size = int(latent.shape[0])
            height, width = target_size or (ROOM_HEIGHT, ROOM_WIDTH)
            logits = torch.zeros(batch_size, 44, height, width)
            logits[:, int(SEMANTIC_PALETTE["WALL"])] = 5.0
            if batch_size >= 1:
                row = height // 2
                logits[0, :, row, :] = 0.0
                logits[0, int(SEMANTIC_PALETTE["FLOOR"]), row, :] = 5.0
                logits[0, :, row, 1] = 0.0
                logits[0, int(SEMANTIC_PALETTE["START"]), row, 1] = 6.0
                logits[0, :, row, width - 1] = 0.0
                logits[0, int(SEMANTIC_PALETTE["DOOR_OPEN"]), row, width - 1] = 6.0
            return logits

    trainer = _make_stub_trainer(context_dim=8)
    trainer.config.num_classes = 44
    trainer.vqvae = _MixedDecodeVQVAE()
    batch = torch.zeros((2, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)

    metrics = DiffusionTrainer.validate(trainer, [batch], num_samples=1, num_diffusion_samples=1)

    assert metrics["val_hard_solvability"] == pytest.approx(1.0)


def test_validate_reports_post_repair_solvability_metrics():
    class _BlockedDecodeVQVAE:
        def decode(self, latent, target_size=None):
            batch_size = int(latent.shape[0])
            height, width = target_size or (ROOM_HEIGHT, ROOM_WIDTH)
            logits = torch.zeros(batch_size, 44, height, width)
            logits[:, int(SEMANTIC_PALETTE["WALL"])] = 5.0
            row = height // 2
            logits[:, :, row, 1] = 0.0
            logits[:, int(SEMANTIC_PALETTE["START"]), row, 1] = 6.0
            logits[:, :, row, width - 1] = 0.0
            logits[:, int(SEMANTIC_PALETTE["DOOR_OPEN"]), row, width - 1] = 6.0
            return logits

    class _Repairer:
        def __init__(self):
            self.calls = 0

        def repair_room_with_neural_guidance(self, grid, start, goal, tile_logits, **_kwargs):
            self.calls += 1
            _ = (start, goal, tile_logits)
            repaired = np.asarray(grid).copy()
            row = repaired.shape[0] // 2
            repaired[row, :] = int(SEMANTIC_PALETTE["FLOOR"])
            repaired[row, 1] = int(SEMANTIC_PALETTE["START"])
            repaired[row, repaired.shape[1] - 1] = int(SEMANTIC_PALETTE["DOOR_OPEN"])
            return repaired, True, {}

    trainer = _make_stub_trainer(context_dim=8)
    trainer.config.num_classes = 44
    trainer.vqvae = _BlockedDecodeVQVAE()
    trainer.validation_neural_guided_repair = _Repairer()
    batch = torch.zeros((1, 1, ROOM_HEIGHT, ROOM_WIDTH), dtype=torch.float32)

    metrics = DiffusionTrainer.validate(trainer, [batch], num_samples=1, num_diffusion_samples=1)

    assert metrics["val_hard_solvability"] == pytest.approx(0.0)
    assert metrics["val_hard_solvability_after_repair"] == pytest.approx(1.0)
    assert metrics["val_neural_repair_success_rate"] == pytest.approx(1.0)
    assert metrics["val_logicnet_score_after_repair"] > 0.0
    assert trainer.validation_neural_guided_repair.calls == 1


def test_validation_repair_slices_room_batch_graph_data():
    class _RecordingRepairer:
        def __init__(self):
            self.graph_data = None

        def repair_room_with_neural_guidance(self, grid, start, goal, tile_logits, graph_data=None, **_kwargs):
            _ = (start, goal, tile_logits)
            self.graph_data = graph_data
            return np.asarray(grid).copy(), True, {}

    trainer = _make_stub_trainer(context_dim=8)
    trainer.config.num_classes = 44
    trainer.validation_neural_guided_repair = _RecordingRepairer()

    logits = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    row = ROOM_HEIGHT // 2
    logits[:, int(SEMANTIC_PALETTE["WALL"])] = 5.0
    logits[:, :, row, 1] = 0.0
    logits[:, int(SEMANTIC_PALETTE["START"]), row, 1] = 6.0
    logits[:, :, row, ROOM_WIDTH - 1] = 0.0
    logits[:, int(SEMANTIC_PALETTE["DOOR_OPEN"]), row, ROOM_WIDTH - 1] = 6.0

    graph_data = {
        "graph_scope": "room_batch",
        "node_features": torch.zeros(1, 3, 6),
        "edge_index": torch.full((1, 2, 0), -1, dtype=torch.long),
        "current_node_idx": torch.tensor([2], dtype=torch.long),
    }

    repaired, success_rate = DiffusionTrainer._repair_validation_decoded_logits(
        trainer,
        logits,
        graph_data=graph_data,
    )

    assert repaired is not None
    assert success_rate == pytest.approx(1.0)
    assert trainer.validation_neural_guided_repair.graph_data["graph_scope"] == "room"
    assert tuple(trainer.validation_neural_guided_repair.graph_data["node_features"].shape) == (3, 6)
    assert int(trainer.validation_neural_guided_repair.graph_data["current_node_idx"].item()) == 2


def test_state_dict_is_finite_rejects_nan_weights():
    state_dict = {"weight": torch.tensor([1.0, float("nan")])}
    assert DiffusionTrainer._state_dict_is_finite(state_dict) is False


def test_load_checkpoint_strips_legacy_embedded_guidance_logicnet_state(tmp_path):
    class _TinyDiffusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Linear(2, 2)
            self.guidance = torch.nn.Module()

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace()
    trainer.diffusion = _TinyDiffusion()
    trainer.ema_diffusion = _TinyDiffusion()
    trainer.condition_encoder = torch.nn.Linear(2, 2)
    trainer.logic_net = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters())
        + list(trainer.condition_encoder.parameters())
        + list(trainer.logic_net.parameters()),
        lr=1e-3,
    )
    trainer.scheduler = SimpleNamespace(load_state_dict=lambda _state: None)

    diffusion_state = dict(trainer.diffusion.state_dict())
    ema_state = dict(trainer.ema_diffusion.state_dict())
    diffusion_state["guidance.logic_net.weight"] = torch.ones_like(trainer.logic_net.weight)
    ema_state["guidance.logic_net.bias"] = torch.ones_like(trainer.logic_net.bias)

    path = tmp_path / "legacy_logicnet_embedded.pth"
    torch.save(
        {
            "epoch": 2,
            "global_step": 17,
            "diffusion_state_dict": diffusion_state,
            "ema_diffusion_state_dict": ema_state,
            "condition_encoder_state_dict": trainer.condition_encoder.state_dict(),
            "logic_net_state_dict": trainer.logic_net.state_dict(),
        },
        path,
    )

    DiffusionTrainer.load_checkpoint(trainer, str(path))

    assert trainer.epoch == 2
    assert trainer.global_step == 17
    assert trainer.diffusion.guidance.logic_net is trainer.logic_net
    assert trainer.ema_diffusion.guidance.logic_net is trainer.logic_net
    assert "logic_net" not in trainer.diffusion.guidance._modules


def test_load_checkpoint_without_ema_state_initializes_ema_from_diffusion(tmp_path):
    class _TinyDiffusion(torch.nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.core = torch.nn.Linear(2, 2)
            self.guidance = torch.nn.Module()
            with torch.no_grad():
                self.core.weight.fill_(float(value))
                self.core.bias.fill_(float(value))

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace()
    trainer.diffusion = _TinyDiffusion(2.0)
    trainer.ema_diffusion = _TinyDiffusion(-5.0)
    trainer.condition_encoder = torch.nn.Linear(2, 2)
    trainer.logic_net = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters())
        + list(trainer.condition_encoder.parameters())
        + list(trainer.logic_net.parameters()),
        lr=1e-3,
    )
    trainer.scheduler = SimpleNamespace(load_state_dict=lambda _state: None)

    path = tmp_path / "missing_ema_checkpoint.pth"
    torch.save(
        {
            "epoch": 3,
            "global_step": 19,
            "diffusion_state_dict": trainer.diffusion.state_dict(),
            "condition_encoder_state_dict": trainer.condition_encoder.state_dict(),
            "logic_net_state_dict": trainer.logic_net.state_dict(),
        },
        path,
    )

    DiffusionTrainer.load_checkpoint(trainer, str(path))

    for ema_param, diffusion_param in zip(trainer.ema_diffusion.parameters(), trainer.diffusion.parameters()):
        assert torch.allclose(ema_param, diffusion_param)


def test_safetensors_sidecar_round_trips_inference_weights_without_optimizer(tmp_path):
    if not train_diffusion_module._HAS_SAFETENSORS:
        pytest.skip("safetensors is not installed")

    def _make_config():
        config = SimpleNamespace(
            latent_dim=4,
            context_dim=8,
            num_timesteps=8,
            schedule_type="cosine",
            diffusion_training_objective="diffusion",
            denoiser_backbone="unet",
            pag_scale=0.0,
            dit_depth=1,
            dit_patch_size=1,
            dit_mlp_ratio=4.0,
            num_classes=44,
            vqvae_hidden_dim=8,
            vqvae_codebook_size=16,
            vqvae_architecture="vqvae",
            vqvae_top_codebook_size=None,
            vqvae_top_latent_dim=None,
            vqvae_use_coordconv=True,
            vqvae_checkpoint=None,
            semantic_role_prior_strength=0.1,
            semantic_puzzle_offset=0,
            topology_supervision_mode="runtime_aligned",
            logic_net_enabled=True,
            guidance_scale=0.0,
            guidance_clamp_magnitude=1.0,
            guidance_relative_norm_cap=0.25,
            guidance_schedule_enabled=True,
            guidance_active_fraction=1.0,
            guidance_decay_power=1.0,
            guidance_max_graph_nodes=16,
            guidance_max_key_lock_pairs=16,
            guidance_max_guidance_elements=1024,
        )
        config.to_dict = lambda: {
            key: value
            for key, value in config.__dict__.items()
            if not callable(value)
        }
        return config

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.config = _make_config()
    trainer.diffusion = _TinyCheckpointModule(2.0)
    trainer.ema_diffusion = _TinyCheckpointModule(3.0)
    trainer.condition_encoder = _TinyCheckpointModule(4.0)
    trainer.logic_net = _TinyCheckpointModule(5.0)
    trainer.optimizer = torch.optim.SGD(
        list(trainer.diffusion.parameters())
        + list(trainer.condition_encoder.parameters())
        + list(trainer.logic_net.parameters()),
        lr=1e-3,
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(trainer.optimizer, T_0=1)
    trainer.epoch = 7
    trainer.global_step = 11

    path = tmp_path / "resume.pth"
    DiffusionTrainer.save_checkpoint(trainer, str(path), include_optimizer=True)
    safetensors_path = path.with_suffix(".safetensors")
    assert safetensors_path.exists()
    assert safetensors_path.with_suffix(".safetensors.meta.json").exists()

    loaded = DiffusionTrainer.__new__(DiffusionTrainer)
    loaded.config = _make_config()
    loaded.device = torch.device("cpu")
    loaded.diffusion = _TinyCheckpointModule(0.0)
    loaded.ema_diffusion = _TinyCheckpointModule(0.0)
    loaded.condition_encoder = _TinyCheckpointModule(0.0)
    loaded.logic_net = _TinyCheckpointModule(0.0)
    loaded.optimizer = torch.optim.SGD(
        list(loaded.diffusion.parameters())
        + list(loaded.condition_encoder.parameters())
        + list(loaded.logic_net.parameters()),
        lr=1e-3,
    )
    loaded.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(loaded.optimizer, T_0=1)

    DiffusionTrainer.load_checkpoint(loaded, str(safetensors_path))

    assert float(loaded.diffusion.weight.item()) == pytest.approx(2.0)
    assert float(loaded.ema_diffusion.weight.item()) == pytest.approx(3.0)
    assert float(loaded.condition_encoder.weight.item()) == pytest.approx(4.0)
    assert float(loaded.logic_net.weight.item()) == pytest.approx(5.0)
    assert loaded.epoch == 7
    assert loaded.global_step == 11
    assert loaded._accumulation_micro_steps == 0
    assert loaded.diffusion.guidance.logic_net is loaded.logic_net
