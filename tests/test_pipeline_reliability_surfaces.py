import time
from types import SimpleNamespace

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.generation.global_state import StateAwareRoomGenerator
from src.pipeline.robust_pipeline import (
    BlockStatus,
    PipelineBlock,
    PipelineConfig,
)
from src.pipeline.types import PipelineComponentFactory


def test_pipeline_block_retries_then_succeeds():
    attempts = {"count": 0}

    def flaky_executor(_state):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ValueError("transient")
        return {"ok": True}

    block = PipelineBlock(
        name="flaky_block",
        executor=flaky_executor,
        validator=lambda out: bool(out.get("ok")),
        config=PipelineConfig(max_retries=3, base_backoff=0.0, enable_logging=False),
    )

    result = block.execute(state={})
    assert result.status == BlockStatus.SUCCESS
    assert result.attempts == 2
    assert attempts["count"] == 2


def test_pipeline_block_reports_failure_after_retries():
    def always_fail(_state):
        raise RuntimeError("boom")

    block = PipelineBlock(
        name="always_fail",
        executor=always_fail,
        config=PipelineConfig(max_retries=2, base_backoff=0.0, enable_logging=False),
    )

    result = block.execute(state={})
    assert result.status == BlockStatus.FAILED
    assert result.attempts == 2
    assert "RuntimeError" in (result.error or "")


def test_advanced_pipeline_module_imports_and_config_defaults():
    from src.pipeline.advanced_pipeline import AdvancedPipelineConfig

    cfg = AdvancedPipelineConfig()
    assert cfg.enable_seam_smoothing is True
    assert cfg.enable_global_state is True
    assert cfg.calculate_fun_metrics is True
    assert isinstance(cfg.boss_arena_size, tuple)


def test_state_aware_room_generator_applies_canonical_water_state():
    generator = StateAwareRoomGenerator(SimpleNamespace())
    room = np.full((6, 5), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int64)

    high = generator.apply_state_modifications(room, {"water_level": "high"})
    low = generator.apply_state_modifications(room, {"water_level": "low"})

    assert np.all(high[3:] == int(SEMANTIC_PALETTE["ELEMENT"]))
    assert np.array_equal(low, room)


def test_pipeline_block_timeout_path():
    def slow_executor(_state):
        time.sleep(0.05)
        return {"done": True}

    block = PipelineBlock(
        name="slow_block",
        executor=slow_executor,
        validator=lambda out: bool(out.get("done")),
        config=PipelineConfig(
            max_retries=1,
            base_backoff=0.0,
            enable_logging=False,
            timeout_per_block=0.001,
        ),
    )

    started = time.monotonic()
    result = block.execute(state={})
    elapsed = time.monotonic() - started
    assert result.status == BlockStatus.FAILED
    assert "TimeoutError" in (result.error or "")
    assert elapsed < 0.04


def test_component_factory_does_not_load_unused_maskgit_models():
    calls = []
    pipeline = SimpleNamespace(
        room_generator_mode="discrete_masked",
        default_latent_sampler="diffusion",
        default_masked_room_teacher_fallback_enabled=False,
        default_fast_sampler_teacher_fallback_enabled=False,
        _load_vqvae=lambda path: calls.append(("vqvae", path)),
        _load_condition_encoder=lambda path: calls.append(("condition", path)) or object(),
        _load_diffusion=lambda path: calls.append(("diffusion", path)),
        _load_logic_net=lambda path: calls.append(("logic", path)) or object(),
        _create_refiner=lambda *args, **kwargs: object(),
    )

    components = PipelineComponentFactory().build(pipeline)

    assert components.neural.vqvae is None
    assert components.neural.diffusion is None
    assert [name for name, _path in calls] == ["condition", "logic"]


def test_component_factory_categorical_mode_skips_diffusion_only():
    calls = []
    pipeline = SimpleNamespace(
        room_generator_mode="latent_diffusion",
        default_latent_sampler="categorical",
        default_masked_room_teacher_fallback_enabled=False,
        default_fast_sampler_teacher_fallback_enabled=False,
        _load_vqvae=lambda path: calls.append(("vqvae", path)) or object(),
        _load_condition_encoder=lambda path: calls.append(("condition", path)) or object(),
        _load_diffusion=lambda path: calls.append(("diffusion", path)),
        _load_logic_net=lambda path: calls.append(("logic", path)) or object(),
        _create_refiner=lambda *args, **kwargs: object(),
    )

    components = PipelineComponentFactory().build(pipeline)

    assert components.neural.vqvae is not None
    assert components.neural.diffusion is None
    assert [name for name, _path in calls] == ["vqvae", "logic"]


def test_component_factory_loads_teacher_stack_when_maskgit_fallback_is_enabled():
    calls = []
    pipeline = SimpleNamespace(
        room_generator_mode="discrete_masked",
        default_latent_sampler="diffusion",
        default_masked_room_teacher_fallback_enabled=True,
        default_fast_sampler_teacher_fallback_enabled=False,
        _load_vqvae=lambda path: calls.append(("vqvae", path)) or object(),
        _load_condition_encoder=lambda path: calls.append(("condition", path)) or object(),
        _load_diffusion=lambda path: calls.append(("diffusion", path)) or object(),
        _load_logic_net=lambda path: calls.append(("logic", path)) or object(),
        _create_refiner=lambda *args, **kwargs: object(),
    )

    components = PipelineComponentFactory().build(pipeline)

    assert components.neural.vqvae is not None
    assert components.neural.diffusion is not None
    assert [name for name, _path in calls] == ["vqvae", "condition", "diffusion", "logic"]
