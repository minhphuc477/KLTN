import pytest
import torch
import networkx as nx

from src.core.graph_grid_attention import GraphNodePositionEncoding
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


def test_pipeline_casts_fast_sampled_latent_to_vqvae_decode_dtype(monkeypatch):
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

    decode_dtype = next(pipeline.vqvae.parameters()).dtype
    captured = {"dtype": None}

    def _supports():
        return True

    def _fast_sample(**kwargs):
        return torch.zeros(kwargs["shape"], device=pipeline.device, dtype=torch.float64)

    def _decode(latent):
        captured["dtype"] = latent.dtype
        assert latent.dtype == decode_dtype
        return torch.zeros(
            (latent.shape[0], pipeline.vqvae.num_classes, 16, 11),
            device=latent.device,
            dtype=decode_dtype,
        )

    monkeypatch.setattr(pipeline.diffusion, "supports_fast_sampling", _supports)
    monkeypatch.setattr(pipeline.diffusion, "fast_sample", _fast_sample)
    monkeypatch.setattr(pipeline.vqvae, "decode", _decode)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        use_fast_sampling=True,
        num_diffusion_steps=5,
        seed=7,
    )

    assert captured["dtype"] == decode_dtype
    assert result.room_grid.shape == (16, 11)


def test_pipeline_retries_vqvae_decode_on_stream_mismatch(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(device="cpu", enable_logging=False)
    latent = torch.zeros((1, int(pipeline.diffusion.latent_dim), 4, 3), dtype=torch.float32)
    calls = {"count": 0, "sync": 0}

    def _decode(z):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("cuDNN error: CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH")
        return torch.zeros((z.shape[0], pipeline.vqvae.num_classes, 16, 11), dtype=z.dtype, device=z.device)

    def _sync():
        calls["sync"] += 1

    monkeypatch.setattr(pipeline.vqvae, "decode", _decode)
    monkeypatch.setattr(pipeline, "_synchronize_cuda_device", _sync)

    logits = pipeline._decode_latent_with_vqvae(latent)

    assert tuple(logits.shape) == (1, pipeline.vqvae.num_classes, 16, 11)
    assert calls["count"] == 2
    assert calls["sync"] == 1


def test_pipeline_stacks_room_topology_maps_without_collapsing_batch_into_channels():
    pipeline = object.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.device = torch.device("cpu")
    topo_a = torch.zeros((1, 54, 16, 11), dtype=torch.float32)
    topo_b = torch.ones((54, 16, 11), dtype=torch.float32)

    stacked = NeuralSymbolicDungeonPipeline._stack_room_topology_maps(pipeline, [topo_a, topo_b])

    assert tuple(stacked.shape) == (2, 54, 16, 11)
    assert torch.allclose(stacked[0], topo_a.squeeze(0))
    assert torch.allclose(stacked[1], topo_b)


def test_pipeline_fast_sampler_clamps_cfg_and_disables_logic_guidance(monkeypatch):
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

    captured = {}
    pipeline.diffusion.training_cfg_scale = 3.0

    def _supports():
        return True

    def _fast_sample(**kwargs):
        captured["guidance_scale"] = float(kwargs["guidance_scale"])
        captured["logic_net"] = pipeline.diffusion.guidance.logic_net
        captured["logic_guidance_scale"] = float(pipeline.diffusion.guidance.guidance_scale)
        return torch.zeros(kwargs["shape"], device=pipeline.device)

    monkeypatch.setattr(pipeline.diffusion, "supports_fast_sampling", _supports)
    monkeypatch.setattr(pipeline.diffusion, "fast_sample", _fast_sample)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        use_fast_sampling=True,
        guidance_scale=7.5,
        logic_guidance_scale=1.0,
        num_diffusion_steps=5,
        seed=11,
    )

    assert captured["guidance_scale"] == pytest.approx(3.0)
    assert captured["logic_guidance_scale"] == pytest.approx(0.0)
    assert captured["logic_net"] is None
    assert result.room_grid.shape == (16, 11)


def test_pipeline_fast_sampler_teacher_fallback_is_default_off_and_opt_in(monkeypatch):
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

    calls = {"fast": 0, "teacher": 0}

    def _supports():
        return True

    def _fast_sample(**kwargs):
        calls["fast"] += 1
        return torch.zeros(kwargs["shape"], device=pipeline.device)

    def _ddim_sample(**kwargs):
        calls["teacher"] += 1
        return torch.ones(kwargs["shape"], device=pipeline.device)

    def _decode(latent):
        fill = 2.0 if float(latent.mean()) > 0.5 else 1.0
        logits = torch.zeros(
            (latent.shape[0], pipeline.vqvae.num_classes, 16, 11),
            device=latent.device,
            dtype=next(pipeline.vqvae.parameters()).dtype,
        )
        logits[:, int(fill), :, :] = 5.0
        return logits

    monkeypatch.setattr(pipeline.diffusion, "supports_fast_sampling", _supports)
    monkeypatch.setattr(pipeline.diffusion, "fast_sample", _fast_sample)
    monkeypatch.setattr(pipeline.diffusion, "ddim_sample", _ddim_sample)
    monkeypatch.setattr(pipeline.vqvae, "decode", _decode)
    monkeypatch.setattr(pipeline, "_should_retry_room_with_teacher", lambda **kwargs: True)

    default_result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        use_fast_sampling=True,
        num_diffusion_steps=4,
        seed=13,
    )

    assert calls["fast"] == 1
    assert calls["teacher"] == 0
    assert float(default_result.metrics.get("teacher_fallback_used", 0.0)) == pytest.approx(0.0)
    assert default_result.metrics["used_fast_sampling"] == pytest.approx(1.0)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        use_fast_sampling=True,
        num_diffusion_steps=4,
        seed=13,
        allow_teacher_fallback=True,
    )

    assert calls["fast"] == 2
    assert calls["teacher"] == 1
    assert result.metrics["teacher_fallback_used"] == pytest.approx(1.0)
    assert result.metrics["used_fast_sampling"] == pytest.approx(0.0)
    assert result.room_grid.shape == (16, 11)


def test_ddim_sample_matches_denoiser_dtype():
    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=8,
        context_dim=8,
        num_timesteps=8,
        cfg_scale=1.0,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=1,
    ).to(dtype=torch.float64)

    context = torch.randn(1, 8, dtype=torch.float32)

    sample = model.ddim_sample(
        context=context,
        shape=(1, 4, 4, 4),
        num_steps=2,
    )

    assert sample.dtype == torch.float64


def test_ddim_sample_eta_zero_stays_finite_when_alpha_rounds_to_one(monkeypatch):
    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=8,
        context_dim=8,
        num_timesteps=1000,
        cfg_scale=1.0,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=1,
    )
    context = torch.randn(1, 8, dtype=torch.float32)

    monkeypatch.setattr(model, "_sampling_dtype", lambda: torch.float16)
    monkeypatch.setattr(model, "_extract_context_topology", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(model, "_extract_spatial_graph_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model,
        "_predict_noise_cfg",
        lambda x_t, t, context, graph_data=None, cached_topology=None, cached_spatial=None: torch.zeros_like(x_t),
    )

    sample = model.ddim_sample(
        context=context,
        shape=(1, 4, 4, 4),
        num_steps=50,
        eta=0.0,
    )

    assert torch.isfinite(sample).all()


def test_graph_node_position_encoding_matches_feature_dtype():
    encoder = GraphNodePositionEncoding(dim=8).to(dtype=torch.float64)
    node_features = torch.randn(1, 3, 8, dtype=torch.float64)
    node_positions = torch.randint(0, 10, (1, 3, 2), dtype=torch.int64)
    tpe = torch.randn(1, 3, 8, dtype=torch.float32)
    current_node_distance = torch.randn(1, 3, 4, dtype=torch.float32)
    structure_features = torch.randn(1, 3, 2, dtype=torch.float32)

    output = encoder(
        node_features,
        node_positions=node_positions,
        tpe=tpe,
        current_node_distance=current_node_distance,
        structure_features=structure_features,
    )

    assert output.dtype == torch.float64
