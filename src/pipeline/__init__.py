"""
KLTN Pipeline Module
====================

Lazy exports for neural-symbolic dungeon generation pipeline components.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "NeuralSymbolicDungeonPipeline",
    "MissingPipelineComponentError",
    "NeuralGenerationComponents",
    "SymbolicGenerationComponents",
    "PipelineComponents",
    "PipelineComponentFactory",
    "RoomGenerationResult",
    "DungeonGenerationResult",
    "PreparedDungeonGeneration",
    "GeneratedRoomSet",
    "StitchedRoomLayout",
    "topology_generation_kwargs_from_resolved_config",
    "generation_runtime_kwargs_from_resolved_config",
    "pipeline_kwargs_from_resolved_config",
    "create_pipeline",
]

_DUNGEON_PIPELINE_EXPORTS = {
    "NeuralSymbolicDungeonPipeline",
    "create_pipeline",
}

_CONFIG_BRIDGE_EXPORTS = {
    "topology_generation_kwargs_from_resolved_config",
    "generation_runtime_kwargs_from_resolved_config",
    "pipeline_kwargs_from_resolved_config",
}

_PIPELINE_TYPE_EXPORTS = {
    "MissingPipelineComponentError",
    "NeuralGenerationComponents",
    "SymbolicGenerationComponents",
    "PipelineComponents",
    "PipelineComponentFactory",
    "RoomGenerationResult",
    "DungeonGenerationResult",
    "PreparedDungeonGeneration",
    "GeneratedRoomSet",
}

if TYPE_CHECKING:
    from src.pipeline.dungeon_pipeline import (
        NeuralSymbolicDungeonPipeline,
        create_pipeline,
    )
    from src.pipeline.config_bridge import (
        topology_generation_kwargs_from_resolved_config,
        generation_runtime_kwargs_from_resolved_config,
        pipeline_kwargs_from_resolved_config,
    )
    from src.pipeline.types import (
        DungeonGenerationResult,
        GeneratedRoomSet,
        MissingPipelineComponentError,
        NeuralGenerationComponents,
        PipelineComponentFactory,
        PipelineComponents,
        PreparedDungeonGeneration,
        RoomGenerationResult,
        SymbolicGenerationComponents,
    )
    from src.pipeline.room_stitching import StitchedRoomLayout


def __getattr__(name: str) -> Any:
    if name in _PIPELINE_TYPE_EXPORTS:
        module = import_module("src.pipeline.types")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _CONFIG_BRIDGE_EXPORTS:
        module = import_module("src.pipeline.config_bridge")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _DUNGEON_PIPELINE_EXPORTS:
        module = import_module("src.pipeline.dungeon_pipeline")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "StitchedRoomLayout":
        module = import_module("src.pipeline.room_stitching")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
