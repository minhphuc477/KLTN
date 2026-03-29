import numpy as np
import networkx as nx

from src.generation.style_transfer import ThemeType
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
    assert "paper-faithful LCM-LoRA runtime" in pipeline.fast_sampling_reason


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


def test_advanced_pipeline_reports_no_lcm_speedup_without_real_backend():
    """Reported LCM speedup must remain neutral when no real fast backend is active."""
    pipeline = AdvancedNeuralSymbolicPipeline(_make_test_config())

    assert pipeline._compute_reported_lcm_speedup(room_count=8, gen_time=12.0) == 1.0
