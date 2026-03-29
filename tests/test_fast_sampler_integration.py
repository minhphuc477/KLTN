import torch
import networkx as nx

from src.core.latent_diffusion import create_latent_diffusion
from src.optimization.lcm_lora import (
    FastSamplerCheckpointInfo,
    load_fast_sampler_checkpoint,
    save_fast_sampler_checkpoint,
)
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline


def test_fast_sampler_checkpoint_round_trip(tmp_path):
    checkpoint_path = tmp_path / "fast_sampler_test.pth"
    save_fast_sampler_checkpoint(
        str(checkpoint_path),
        lora_state_dict={"dummy.lora.weight": torch.randn(1)},
        base_diffusion_checkpoint="checkpoints/best_model.pth",
        num_inference_steps=4,
        lora_rank=8,
        lora_alpha=8.0,
    )

    state_dict, info = load_fast_sampler_checkpoint(str(checkpoint_path))

    assert "dummy.lora.weight" in state_dict
    assert isinstance(info, FastSamplerCheckpointInfo)
    assert info.distillation_type == "consistency_lora"
    assert info.num_inference_steps == 4
    assert info.base_diffusion_checkpoint == "checkpoints/best_model.pth"


def test_latent_diffusion_enable_fast_sampling_uses_adapter(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "adapter.pth"
    save_fast_sampler_checkpoint(
        str(checkpoint_path),
        lora_state_dict={"dummy.lora.weight": torch.randn(1)},
        base_diffusion_checkpoint="base.pth",
        num_inference_steps=4,
        lora_rank=8,
        lora_alpha=8.0,
    )

    created = {}

    class _FakeSampler:
        def __init__(self, diffusion_model, **kwargs):
            created["kwargs"] = kwargs
            self.diffusion_model = diffusion_model

        def sample_fast(self, **kwargs):
            return torch.zeros(kwargs["latent_shape"])

    monkeypatch.setattr("src.optimization.lcm_lora.GraphConditionedFastSampler", _FakeSampler)

    model = create_latent_diffusion(latent_dim=8, model_channels=8, context_dim=16, num_timesteps=10)
    model.enable_fast_sampling(adapter_checkpoint=str(checkpoint_path), num_inference_steps=3, strict=False)

    assert model.supports_fast_sampling() is True
    assert created["kwargs"]["adapter_checkpoint"] == str(checkpoint_path)
    assert created["kwargs"]["num_inference_steps"] == 3


def test_true_lcm_lora_metadata_is_rejected(tmp_path):
    checkpoint_path = tmp_path / "paper_lcm_lora.pth"
    save_fast_sampler_checkpoint(
        str(checkpoint_path),
        lora_state_dict={"dummy.lora.weight": torch.randn(1)},
        base_diffusion_checkpoint="base.pth",
        num_inference_steps=4,
        lora_rank=8,
        lora_alpha=8.0,
        distillation_type="lcm_lora",
    )

    try:
        load_fast_sampler_checkpoint(str(checkpoint_path))
    except ValueError as exc:
        assert "does not support paper-faithful LCM-LoRA" in str(exc)
    else:
        raise AssertionError("Expected paper-faithful lcm_lora metadata to be rejected.")


def test_pipeline_generate_room_routes_to_fast_sampler(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(device="cpu", enable_logging=False)
    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    graph = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    graph_context = pipeline._build_room_graph_context(
        graph_data=graph,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )

    called = {"fast": 0}

    def _supports():
        return True

    def _fast_sample(**kwargs):
        called["fast"] += 1
        return torch.zeros(kwargs["shape"], device=pipeline.device)

    monkeypatch.setattr(pipeline.diffusion, "supports_fast_sampling", _supports)
    monkeypatch.setattr(pipeline.diffusion, "fast_sample", _fast_sample)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        use_fast_sampling=True,
        num_diffusion_steps=5,
        seed=42,
    )

    assert called["fast"] == 1
    assert result.room_grid.shape == (16, 11)
