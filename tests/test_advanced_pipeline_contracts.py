import numpy as np
import networkx as nx
from types import SimpleNamespace

from src.generation.style_transfer import ThemeType
import src.pipeline.advanced_pipeline as advanced_pipeline_module
from src.pipeline.advanced_pipeline import (
    AdvancedNeuralSymbolicPipeline,
    AdvancedPipelineConfig,
)


def _make_test_config() -> AdvancedPipelineConfig:
    return AdvancedPipelineConfig(
        use_lcm_lora=True,
        enable_seam_smoothing=False,
        enable_collision_validation=False,
        enable_big_rooms=False,
        enable_global_state=False,
        calculate_fun_metrics=False,
        enable_diversity_analysis=False,
        record_demo=False,
        enable_explainability=False,
    )


def test_advanced_pipeline_disables_requested_lcm_without_real_backend():
    """Requested LCM-LoRA should not activate when only the experimental path exists."""
    pipeline = AdvancedNeuralSymbolicPipeline(_make_test_config())

    assert pipeline.fast_sampling_active is False
    assert "no distilled consistency-LoRA checkpoint" in pipeline.fast_sampling_reason


def test_advanced_pipeline_uses_standard_diffusion_steps_without_real_lcm(monkeypatch):
    """Advanced pipeline should keep standard DDIM steps when no real LCM backend is active."""
    pipeline = AdvancedNeuralSymbolicPipeline(_make_test_config())
    captured = {}

    class _RoomResult:
        def __init__(self):
            self.room_grid = np.zeros((16, 11), dtype=np.int32)

    def fake_generate_room(**kwargs):
        captured["num_diffusion_steps"] = kwargs["num_diffusion_steps"]
        return _RoomResult()

    monkeypatch.setattr(pipeline.neural_pipeline, "generate_room", fake_generate_room)

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0)
    room = pipeline._generate_single_room_with_ml(
        node_id=0,
        mission_graph=mission_graph,
        graph_context={},
        neighbor_latents={},
        theme=ThemeType.ZELDA_CLASSIC,
    )

    assert room.shape == (16, 11)
    assert captured["num_diffusion_steps"] == 50


def test_advanced_pipeline_activates_compatible_consistency_lora_backend(monkeypatch, tmp_path):
    """A validated repo fast-sampler adapter should flow into the core generation pipeline."""
    adapter = tmp_path / "fast_sampler_best.pth"
    adapter.write_bytes(b"adapter")
    base = tmp_path / "best_model.pth"
    base.write_bytes(b"base")
    captured = {}

    def fake_load_fast_sampler_checkpoint(path):
        assert str(path) == str(adapter)
        return {}, SimpleNamespace(
            distillation_type="consistency_lora",
            base_diffusion_checkpoint=str(base),
            num_inference_steps=4,
            lora_rank=8,
            lora_alpha=8.0,
            target_modules=(),
        )

    class _FakeDiffusion:
        def supports_fast_sampling(self):
            return True

    class _FakePipeline:
        def __init__(self, **kwargs):
            captured["constructor_kwargs"] = kwargs
            self.diffusion = _FakeDiffusion()
            self.default_guidance_scale = 3.0
            self.default_logic_guidance_scale = 0.0
            self.default_apply_repair = False
            self.default_start_goal_coords = None

        def generate_room(self, **kwargs):
            captured["generate_room_kwargs"] = kwargs

            class _RoomResult:
                room_grid = np.zeros((16, 11), dtype=np.int32)

            return _RoomResult()

    monkeypatch.setattr(advanced_pipeline_module, "load_fast_sampler_checkpoint", fake_load_fast_sampler_checkpoint)
    monkeypatch.setattr(advanced_pipeline_module, "NeuralSymbolicDungeonPipeline", _FakePipeline)

    pipeline = AdvancedNeuralSymbolicPipeline(
        AdvancedPipelineConfig(
            use_lcm_lora=True,
            lcm_lora_checkpoint=adapter,
            lcm_steps=4,
            enable_seam_smoothing=False,
            enable_collision_validation=False,
            enable_big_rooms=False,
            enable_global_state=False,
            calculate_fun_metrics=False,
            enable_diversity_analysis=False,
            record_demo=False,
            enable_explainability=False,
        )
    )

    assert pipeline.fast_sampling_active is True
    constructor_kwargs = captured["constructor_kwargs"]
    assert constructor_kwargs["fast_sampling_checkpoint"] == str(adapter)
    assert constructor_kwargs["diffusion_checkpoint"] == str(base)
    assert constructor_kwargs["default_use_fast_sampling"] is True

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0)
    pipeline._generate_single_room_with_ml(
        node_id=0,
        mission_graph=mission_graph,
        graph_context={},
        neighbor_latents={},
        theme=ThemeType.ZELDA_CLASSIC,
    )

    generate_kwargs = captured["generate_room_kwargs"]
    assert generate_kwargs["use_fast_sampling"] is True
    assert generate_kwargs["num_diffusion_steps"] == 4


def test_advanced_pipeline_reports_no_lcm_speedup_without_real_backend():
    """Reported LCM speedup must remain neutral when no real fast backend is active."""
    pipeline = AdvancedNeuralSymbolicPipeline(_make_test_config())

    assert pipeline._compute_reported_lcm_speedup(room_count=8, gen_time=12.0) == 1.0
