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
    "MissingPipelineComponentError",
    "NeuralGenerationComponents",
    "SymbolicGenerationComponents",
    "PipelineComponents",
    "PipelineComponentFactory",
    "RoomGenerationResult",
    "DungeonGenerationResult",
    "PreparedDungeonGeneration",
    "GeneratedRoomSet",
    "topology_generation_kwargs_from_resolved_config",
    "generation_runtime_kwargs_from_resolved_config",
    "pipeline_kwargs_from_resolved_config",
    "create_pipeline",
}

if TYPE_CHECKING:
    from src.pipeline.dungeon_pipeline import (
        DungeonGenerationResult,
        GeneratedRoomSet,
        MissingPipelineComponentError,
        NeuralGenerationComponents,
        NeuralSymbolicDungeonPipeline,
        PipelineComponentFactory,
        PipelineComponents,
        PreparedDungeonGeneration,
        RoomGenerationResult,
        SymbolicGenerationComponents,
        topology_generation_kwargs_from_resolved_config,
        generation_runtime_kwargs_from_resolved_config,
        pipeline_kwargs_from_resolved_config,
        create_pipeline,
    )
    from src.pipeline.room_stitching import StitchedRoomLayout


def __getattr__(name: str) -> Any:
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
