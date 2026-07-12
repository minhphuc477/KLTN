"""
Shared pipeline data structures and dependency-injection bundles.

This module is intentionally small and importable without pulling in the full
9k+ line pipeline orchestration class.  ``dungeon_pipeline`` re-exports these
names for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import torch

from src.core import (
    DualStreamConditionEncoder,
    LatentDiffusionModel,
    LogicNet,
    SemanticVQVAE,
    SymbolicRefiner,
)
from src.pipeline.room_stitching import StitchedRoomLayout
from src.simulation.map_elites import MAPElitesEvaluator


class MissingPipelineComponentError(RuntimeError):
    """Raised when a pipeline operation requires an unavailable injected component."""


@dataclass
class NeuralGenerationComponents:
    """Injectable neural stack used for room generation."""

    vqvae: Optional[SemanticVQVAE] = None
    condition_encoder: Optional[DualStreamConditionEncoder] = None
    diffusion: Optional[LatentDiffusionModel] = None
    logic_net: Optional[LogicNet] = None


@dataclass
class SymbolicGenerationComponents:
    """Injectable symbolic/evaluation stack used around neural generation."""

    refiner: Optional[SymbolicRefiner] = None
    stitcher: Optional[Any] = None
    map_elites: Optional[MAPElitesEvaluator] = None


@dataclass
class PipelineComponents:
    """
    Dependency-injection bundle for NeuralSymbolicDungeonPipeline.

    Any field may be left unset for partial pipelines. Public operations
    fail fast or disable optional stages when a required component is absent.
    """

    neural: NeuralGenerationComponents = field(default_factory=NeuralGenerationComponents)
    symbolic: SymbolicGenerationComponents = field(default_factory=SymbolicGenerationComponents)


@dataclass
class PipelineComponentFactory:
    """Factory for assembling the default component bundle for the pipeline."""

    vqvae_checkpoint: Optional[str] = None
    diffusion_checkpoint: Optional[str] = None
    logic_net_checkpoint: Optional[str] = None
    condition_encoder_checkpoint: Optional[str] = None
    use_learned_refiner_rules: bool = True
    map_elites_resolution: int = 20
    map_elites_archive_path: Optional[str] = None
    map_elites_load_archive: bool = False
    map_elites_autosave_archive: bool = False
    map_elites_preference_buffer_size: int = 0
    symbolic_max_repair_attempts: int = 5
    symbolic_repair_margin: int = 2
    symbolic_adjacency_threshold: float = 0.01

    def build(self, pipeline: Any) -> PipelineComponents:
        room_mode = str(getattr(pipeline, "room_generator_mode", "latent_diffusion")).strip().lower()
        sampler_mode = str(getattr(pipeline, "default_latent_sampler", "diffusion")).strip().lower()
        teacher_fallback_enabled = bool(
            getattr(pipeline, "default_masked_room_teacher_fallback_enabled", False)
            if room_mode == "discrete_masked"
            else getattr(pipeline, "default_fast_sampler_teacher_fallback_enabled", False)
        )
        needs_vqvae = room_mode != "discrete_masked" or teacher_fallback_enabled
        needs_diffusion = teacher_fallback_enabled or (
            room_mode != "discrete_masked" and sampler_mode != "categorical"
        )
        needs_condition_encoder = room_mode == "discrete_masked" or needs_diffusion
        return PipelineComponents(
            neural=NeuralGenerationComponents(
                vqvae=(
                    pipeline._load_vqvae(self.vqvae_checkpoint)
                    if needs_vqvae
                    else None
                ),
                condition_encoder=(
                    pipeline._load_condition_encoder(self.condition_encoder_checkpoint)
                    if needs_condition_encoder
                    else None
                ),
                diffusion=(
                    pipeline._load_diffusion(self.diffusion_checkpoint)
                    if needs_diffusion
                    else None
                ),
                logic_net=pipeline._load_logic_net(self.logic_net_checkpoint),
            ),
            symbolic=SymbolicGenerationComponents(
                refiner=pipeline._create_refiner(
                    self.use_learned_refiner_rules,
                    max_repair_attempts=self.symbolic_max_repair_attempts,
                    margin=self.symbolic_repair_margin,
                    adjacency_threshold=self.symbolic_adjacency_threshold,
                ),
                stitcher=None,
                map_elites=MAPElitesEvaluator(
                    resolution=self.map_elites_resolution,
                    tie_breaker="quality_score",
                    descriptor_mode="hybrid",
                    archive_path=self.map_elites_archive_path,
                    load_existing_archive=self.map_elites_load_archive,
                    autosave_archive=self.map_elites_autosave_archive,
                    preference_buffer_size=self.map_elites_preference_buffer_size,
                ),
            ),
        )


@dataclass
class RoomGenerationResult:
    """Result of generating a single room."""

    room_id: int
    room_grid: np.ndarray
    latent: torch.Tensor
    neural_grid: np.ndarray
    was_repaired: bool
    raw_neural_grid: Optional[np.ndarray] = None
    repair_mask: Optional[np.ndarray] = None
    room_plan_mask: Optional[np.ndarray] = None
    neural_probs: Optional[np.ndarray] = None
    puzzle_metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DungeonGenerationResult:
    """Result of generating a complete dungeon."""

    dungeon_grid: np.ndarray
    rooms: Dict[int, RoomGenerationResult]
    mission_graph: nx.Graph
    metrics: Dict[str, Any]
    map_elites_score: Optional[Dict[str, float]] = None
    generation_time: float = 0.0
    stitched_layout: Optional[StitchedRoomLayout] = None
    puzzle_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedDungeonGeneration:
    """Prepared graph/context bundle for multi-room generation."""

    mission_graph: nx.Graph
    mission_graph_physical: nx.Graph
    graph_data: Dict[str, Any]


@dataclass
class GeneratedRoomSet:
    """Partial generation result for room-only runs before stitching/evaluation."""

    rooms: Dict[Any, RoomGenerationResult]
    room_latents: Dict[Any, torch.Tensor]
    batch_runtime_diagnostics: list[Dict[str, Any]] = field(default_factory=list)


__all__ = [
    "MissingPipelineComponentError",
    "NeuralGenerationComponents",
    "SymbolicGenerationComponents",
    "PipelineComponents",
    "PipelineComponentFactory",
    "RoomGenerationResult",
    "DungeonGenerationResult",
    "PreparedDungeonGeneration",
    "GeneratedRoomSet",
]
