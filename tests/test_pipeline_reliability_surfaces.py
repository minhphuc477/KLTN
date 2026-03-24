import time

from src.pipeline.robust_pipeline import (
    BlockStatus,
    PipelineBlock,
    PipelineConfig,
)


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

    result = block.execute(state={})
    assert result.status == BlockStatus.FAILED
    assert "TimeoutError" in (result.error or "")
