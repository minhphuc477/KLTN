"""Backward-compatible import surface for pipeline configuration dataclasses."""

from src.pipeline.config import GraphConfig, ModelConfig, PipelineConfig, SamplerConfig

TopologyConfig = GraphConfig

__all__ = [
    "PipelineConfig",
    "ModelConfig",
    "SamplerConfig",
    "GraphConfig",
    "TopologyConfig",
]
