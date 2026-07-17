"""Facade for the neural-symbolic dungeon generation pipeline.

This module owns the public pipeline surface and delegates implementation to
focused runtime, model, sampling, graph-context, stitching, and evaluation
modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from src.core import ROOM_HEIGHT, ROOM_WIDTH
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
from src.pipeline.config import GraphConfig, ModelConfig, PipelineConfig, SamplerConfig
from src.pipeline.config_bridge import (
    generation_runtime_kwargs_from_resolved_config,
    pipeline_kwargs_from_resolved_config,
    topology_generation_kwargs_from_resolved_config,
)
from src.pipeline.generation import DiffusionSampler
from src.pipeline.generation.sampler import DEFAULT_ROOM_LATENT_HW
from src.pipeline.models import ModelManager
from src.pipeline.graph_features import (
    compute_tpe_features,
    encode_edge_feature_vector,
    extract_node_feature_vector,
)
from src.pipeline.room_stitching import (
    StitchedRoomLayout,
    compute_relaxed_room_placement,
    compute_strict_room_placement,
    solve_component_strict_adjacency,
)
from src.pipeline.room_topology_conditioning import ROOM_TOPOLOGY_CHANNEL_COUNT
from src.pipeline.spatial_utils import (
    clamp_room_coord,
    coerce_bool,
    coerce_difficulty,
    first_free_position,
    fit_room_grid,
    get_node_grid_position,
    infer_direction,
    parse_label_tokens,
    parse_room_coord,
    stable_node_sort_key,
)
from src.pipeline.stitching import DungeonAssembler
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
from src.simulation.map_elites import MAPElitesEvaluator
from src.utils.graph_utils import validate_graph_topology

_RUNTIME_POSITIONAL_ARGS = (
    "vqvae_checkpoint",
    "diffusion_checkpoint",
    "logic_net_checkpoint",
    "condition_encoder_checkpoint",
    "device",
)


def _config_from_constructor_args(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> PipelineConfig:
    """Normalize config-object and flat constructor calls into PipelineConfig."""
    if len(args) == 1 and isinstance(args[0], PipelineConfig):
        if kwargs:
            return args[0].with_overrides(**kwargs)
        return args[0]
    if len(args) > len(_RUNTIME_POSITIONAL_ARGS):
        raise TypeError(
            "NeuralSymbolicDungeonPipeline accepts PipelineConfig or "
            f"positional args {_RUNTIME_POSITIONAL_ARGS}; got {len(args)} positional args."
        )

    runtime_kwargs = dict(kwargs)
    for name, value in zip(_RUNTIME_POSITIONAL_ARGS, args):
        if name in runtime_kwargs:
            raise TypeError(f"Multiple values for constructor argument {name!r}.")
        runtime_kwargs[name] = value
    return PipelineConfig.from_kwargs(**runtime_kwargs)


class _PipelineFacadeMeta(type):
    """Adapter that keeps old keyword construction working while __init__ is config-only."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls is NeuralSymbolicDungeonPipeline:
            config = _config_from_constructor_args(args, kwargs)
            return super().__call__(config)
        return super().__call__(*args, **kwargs)


class NeuralSymbolicDungeonPipeline(metaclass=_PipelineFacadeMeta):
    """Facade/orchestrator for the refactored generation pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_manager = ModelManager(self)
        self.sampler = DiffusionSampler(self)
        self.assembler = DungeonAssembler(self)
        from src.pipeline.runtime import initialize_pipeline

        initialize_pipeline(self, config=config)

    @staticmethod
    def from_kwargs(**kwargs: Any) -> "NeuralSymbolicDungeonPipeline":
        """Explicit constructor for flat config kwargs."""
        return NeuralSymbolicDungeonPipeline(PipelineConfig.from_kwargs(**kwargs))

    @staticmethod
    def from_legacy_kwargs(**kwargs: Any) -> "NeuralSymbolicDungeonPipeline":
        """Explicit compatibility constructor for former long-kwargs callers."""
        return NeuralSymbolicDungeonPipeline.from_kwargs(**kwargs)

    @classmethod
    def from_components(
        cls,
        *,
        components: PipelineComponents,
        **kwargs: Any,
    ) -> "NeuralSymbolicDungeonPipeline":
        """Construct a pipeline from an explicit dependency bundle."""
        return cls(PipelineConfig.from_kwargs(components=components, **kwargs))

    @classmethod
    def create_symbolic_repair_pipeline(
        cls,
        *,
        device: str = "cpu",
        use_learned_refiner_rules: bool = True,
        symbolic_max_repair_attempts: int = 5,
        symbolic_repair_margin: int = 2,
        symbolic_adjacency_threshold: float = 0.01,
        enable_map_elites: bool = False,
        map_elites_resolution: int = 20,
        map_elites_archive_path: Optional[str] = None,
        map_elites_load_archive: bool = False,
        map_elites_autosave_archive: bool = False,
        enable_logging: bool = True,
        strict_checkpoint_mode: bool = False,
        stitcher: Optional[Any] = None,
        map_elites: Optional[MAPElitesEvaluator] = None,
    ) -> "NeuralSymbolicDungeonPipeline":
        """Create a lightweight pipeline for symbolic-only repair/stitch workflows."""
        symbolic = SymbolicGenerationComponents(
            refiner=cls._create_refiner(
                use_learned_refiner_rules,
                max_repair_attempts=symbolic_max_repair_attempts,
                margin=symbolic_repair_margin,
                adjacency_threshold=symbolic_adjacency_threshold,
            ),
            stitcher=stitcher,
            map_elites=(
                map_elites
                if map_elites is not None
                else (
                    MAPElitesEvaluator(
                        resolution=map_elites_resolution,
                        tie_breaker="quality_score",
                        descriptor_mode="hybrid",
                        archive_path=map_elites_archive_path,
                        load_existing_archive=map_elites_load_archive,
                        autosave_archive=map_elites_autosave_archive,
                    )
                    if enable_map_elites
                    else None
                )
            ),
        )
        return cls.from_components(
            components=PipelineComponents(symbolic=symbolic),
            device=device,
            use_learned_refiner_rules=use_learned_refiner_rules,
            map_elites_resolution=map_elites_resolution,
            enable_logging=enable_logging,
            strict_checkpoint_mode=strict_checkpoint_mode,
            symbolic_max_repair_attempts=symbolic_max_repair_attempts,
            symbolic_repair_margin=symbolic_repair_margin,
            symbolic_adjacency_threshold=symbolic_adjacency_threshold,
        )

    def _runtime_helper(self, name: str, *args: Any, **kwargs: Any) -> Any:
        from src.pipeline import runtime

        return getattr(runtime, name)(self, *args, **kwargs)

    def _load_checkpoint_and_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_load_checkpoint_and_metadata", *args, **kwargs)

    def _extract_checkpoint_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_extract_checkpoint_config", *args, **kwargs)

    def _extract_checkpoint_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_extract_checkpoint_state_dict", *args, **kwargs)

    def _bump_diagnostic(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_bump_diagnostic", *args, **kwargs)

    def _prepare_component(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_prepare_component", *args, **kwargs)

    def _bind_components(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_bind_components", *args, **kwargs)

    def component_status(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("component_status", *args, **kwargs)

    def supports_room_generation(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("supports_room_generation", *args, **kwargs)

    def supports_symbolic_repair(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("supports_symbolic_repair", *args, **kwargs)

    def _require_component(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_require_component", *args, **kwargs)

    def _require_room_generation_components(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_require_room_generation_components", *args, **kwargs)

    def _condition_feature_dims(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_condition_feature_dims", *args, **kwargs)

    def _fit_feature_vector(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_helper("_fit_feature_vector", *args, **kwargs)

    def _load_vqvae(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_manager.load_vqvae(*args, **kwargs)

    def _load_condition_encoder(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_manager.load_condition_encoder(*args, **kwargs)

    def _load_diffusion(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_manager.load_diffusion(*args, **kwargs)

    def _load_logic_net(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_manager.load_logic_net(*args, **kwargs)

    def _load_masked_room_model(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_manager.load_masked_room_model(*args, **kwargs)

    @staticmethod
    def _create_refiner(*args: Any, **kwargs: Any) -> Any:
        return ModelManager.create_refiner(*args, **kwargs)

    def _extract_node_feature_vector(self, attrs: Dict[str, Any]) -> torch.Tensor:
        node_dim, _ = self._condition_feature_dims()
        return extract_node_feature_vector(
            attrs,
            node_dim=node_dim,
            device=self.device,
            parse_label_tokens=self._parse_label_tokens,
            coerce_bool=self._coerce_bool,
            coerce_difficulty=self._coerce_difficulty,
        )

    def _encode_edge_feature_vector(self, edge_data: Dict[str, Any]) -> list[float]:
        _, edge_dim = self._condition_feature_dims()
        return encode_edge_feature_vector(edge_data, edge_dim=edge_dim)

    def _compute_tpe_features(
        self,
        graph: nx.Graph,
        node_order: list[Any],
        node_to_idx: Dict[Any, int],
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        return compute_tpe_features(
            graph,
            node_order,
            node_to_idx,
            node_features,
            device=self.device,
            parse_label_tokens=self._parse_label_tokens,
            coerce_bool=self._coerce_bool,
            coerce_difficulty=self._coerce_difficulty,
            on_shortest_path_fallback=lambda: self._bump_diagnostic("tpe_shortest_path_fallback"),
        )

    def _parse_label_tokens(self, label: Any) -> set[str]:
        return parse_label_tokens(label)

    def _coerce_bool(self, value: Any) -> bool:
        return coerce_bool(value)

    def _coerce_difficulty(self, value: Any) -> float:
        return coerce_difficulty(value)

    def _parse_room_coord(self, value: Any) -> Optional[Tuple[int, int]]:
        return parse_room_coord(value)

    def _clamp_room_coord(self, coord: Tuple[int, int]) -> Tuple[int, int]:
        return clamp_room_coord(coord)

    def _normalize_room_coord(self, coord: Any, *, field_name: str) -> Tuple[int, int]:
        if not isinstance(coord, (tuple, list)) or len(coord) != 2:
            raise ValueError(f"{field_name} must be a 2-item (row, col) coordinate, got {coord!r}.")
        try:
            row = int(coord[0])
            col = int(coord[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field_name} must contain integer-compatible row/col values.") from exc
        return self._clamp_room_coord((row, col))

    def _normalize_start_goal_coords(self, start_goal_coords: Any) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if not isinstance(start_goal_coords, (tuple, list)) or len(start_goal_coords) != 2:
            raise ValueError("start_goal_coords must be ((start_row, start_col), (goal_row, goal_col)).")
        start = self._normalize_room_coord(start_goal_coords[0], field_name="start")
        goal = self._normalize_room_coord(start_goal_coords[1], field_name="goal")
        return start, goal

    def _get_node_grid_position(self, graph: nx.Graph, node_id: Any) -> Optional[Tuple[int, int]]:
        return get_node_grid_position(graph, node_id)

    def _infer_direction(self, graph: nx.Graph, source_node: Any, target_node: Any) -> Optional[str]:
        return infer_direction(graph, source_node, target_node)

    def _first_free_position(self, start_pos: Tuple[int, int], occupied: set) -> Tuple[int, int]:
        return first_free_position(start_pos, occupied)

    def _fit_room_grid(self, room_grid: np.ndarray) -> np.ndarray:
        return fit_room_grid(room_grid)

    def _compute_strict_room_placement(
        self,
        graph: nx.Graph,
        room_ids: list[Any],
    ) -> Dict[Any, Tuple[int, int]]:
        return compute_strict_room_placement(
            graph=graph,
            room_ids=room_ids,
            sort_key=stable_node_sort_key,
            node_position_getter=self._get_node_grid_position,
            first_free_position_fn=self._first_free_position,
        )

    def _compute_relaxed_room_placement(
        self,
        graph: nx.Graph,
        room_ids: list[Any],
    ) -> Dict[Any, Tuple[int, int]]:
        return compute_relaxed_room_placement(
            graph=graph,
            room_ids=room_ids,
            sort_key=stable_node_sort_key,
            node_position_getter=self._get_node_grid_position,
            first_free_position_fn=self._first_free_position,
        )

    def _solve_component_strict_adjacency(
        self,
        comp_nodes: list[Any],
        adjacency: Dict[Any, set],
        explicit_pos: Dict[Any, Tuple[int, int]],
    ) -> Dict[Any, Tuple[int, int]]:
        return solve_component_strict_adjacency(
            comp_nodes=comp_nodes,
            adjacency=adjacency,
            explicit_pos=explicit_pos,
            sort_key=stable_node_sort_key,
        )

    @torch.no_grad()
    def _encode_room_grid_to_latent(
        self,
        room_grid: np.ndarray,
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        vqvae = self._require_component("vqvae", "_encode_room_grid_to_latent")
        resolved_num_classes = int(
            num_classes
            if num_classes is not None
            else getattr(vqvae, "num_classes", int(np.max(self._valid_semantic_tile_ids_np)) + 1)
        )
        grid = np.asarray(room_grid, dtype=np.int64)
        grid = np.clip(grid, 0, resolved_num_classes - 1)
        one_hot = np.eye(resolved_num_classes, dtype=np.float32)[grid]
        x_0 = (
            torch.from_numpy(one_hot)
            .to(self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        z_q, _ = vqvae.encode(x_0)
        diffusion = getattr(self, "diffusion", None)
        if diffusion is not None and hasattr(diffusion, "scale_first_stage_latent"):
            z_q = diffusion.scale_first_stage_latent(z_q)
        return z_q.detach()

    def prepare_dungeon_generation(self, *args: Any, **kwargs: Any) -> PreparedDungeonGeneration:
        return self.assembler.prepare_dungeon_generation(*args, **kwargs)

    def generate_rooms_for_graph(self, *args: Any, **kwargs: Any) -> GeneratedRoomSet:
        return self.assembler.generate_rooms_for_graph(*args, **kwargs)

    def generate_room(self, *args: Any, **kwargs: Any) -> RoomGenerationResult:
        return self.sampler.generate_room(*args, **kwargs)

    def _generate_room_batch(self, *args: Any, **kwargs: Any) -> Dict[Any, RoomGenerationResult]:
        return self.sampler.generate_room_batch(*args, **kwargs)

    def generate_dungeon(self, *args: Any, **kwargs: Any) -> DungeonGenerationResult:
        return self.assembler.generate_dungeon(*args, **kwargs)

    def stitch_rooms(self, *args: Any, **kwargs: Any) -> Any:
        return self.assembler.stitch_rooms(*args, **kwargs)

    def stitch_room_layout(self, *args: Any, **kwargs: Any) -> StitchedRoomLayout:
        return self.assembler.stitch_room_layout(*args, **kwargs)

    def _evaluation_helper(self, name: str, *args: Any, **kwargs: Any) -> Any:
        from src.pipeline import evaluation

        return getattr(evaluation, name)(self, *args, **kwargs)

    def repair_room(self, *args: Any, **kwargs: Any) -> Any:
        return self._evaluation_helper("repair_room", *args, **kwargs)

    def evaluate_generated_dungeon(self, *args: Any, **kwargs: Any) -> Any:
        return self._evaluation_helper("evaluate_generated_dungeon", *args, **kwargs)

    def evaluate_dungeon_solvability(self, *args: Any, **kwargs: Any) -> Any:
        return self._evaluation_helper("evaluate_dungeon_solvability", *args, **kwargs)

    def repair_and_stitch_dungeon(self, *args: Any, **kwargs: Any) -> DungeonGenerationResult:
        return self._evaluation_helper("repair_and_stitch_dungeon", *args, **kwargs)

    def _validate_dungeon(self, *args: Any, **kwargs: Any) -> Any:
        return self._evaluation_helper("_validate_dungeon", *args, **kwargs)

    def _graph_context_helper(self, name: str, *args: Any, **kwargs: Any) -> Any:
        from src.pipeline.generation import graph_context

        return getattr(graph_context, name)(self, *args, **kwargs)

    def _prepare_graph_context(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_prepare_graph_context", *args, **kwargs)

    def _get_neighbor_latents(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_get_neighbor_latents", *args, **kwargs)

    def _get_neighbor_reference_room_maps(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_get_neighbor_reference_room_maps", *args, **kwargs)

    def _extract_room_start_goal(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_extract_room_start_goal", *args, **kwargs)

    def _build_room_boundary_constraints(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_build_room_boundary_constraints", *args, **kwargs)

    def _room_role_flags(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_room_role_flags", *args, **kwargs)

    def _resolve_puzzle_room_scaffold_profile(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_resolve_puzzle_room_scaffold_profile", *args, **kwargs)

    def _extract_room_topology_semantics(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_extract_room_topology_semantics", *args, **kwargs)

    def _build_room_plan_trace(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_build_room_plan_trace", *args, **kwargs)

    def _resolve_validator_plan_state_budget(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_resolve_validator_plan_state_budget", *args, **kwargs)

    def _build_room_topology_condition_tensor(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_build_room_topology_condition_tensor", *args, **kwargs)

    def _extract_explicit_style_id(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_extract_explicit_style_id", *args, **kwargs)

    def _build_room_graph_context(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_build_room_graph_context", *args, **kwargs)

    def _edge_tokens_to_door_tile(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_edge_tokens_to_door_tile", *args, **kwargs)

    def _build_masked_room_fixed_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_build_masked_room_fixed_tokens", *args, **kwargs)

    def _build_room_position_tensor(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph_context_helper("_build_room_position_tensor", *args, **kwargs)

    def _room_processing_helper(self, name: str, *args: Any, **kwargs: Any) -> Any:
        from src.pipeline.generation import room_processing

        return getattr(room_processing, name)(self, *args, **kwargs)

    def _sanitize_semantic_grid(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_sanitize_semantic_grid", *args, **kwargs)

    def _strip_room_void_tiles(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_strip_room_void_tiles", *args, **kwargs)

    def _strip_volatile_room_semantics(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_strip_volatile_room_semantics", *args, **kwargs)

    def _apply_semantic_constrained_decoding(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_apply_semantic_constrained_decoding", *args, **kwargs)

    def _all_room_door_slots_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_all_room_door_slots_mask", *args, **kwargs)

    def _required_room_door_slots_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_required_room_door_slots_mask", *args, **kwargs)

    def _enforce_room_boundary_shell(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_enforce_room_boundary_shell", *args, **kwargs)

    def _strip_structural_room_artifacts(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_strip_structural_room_artifacts", *args, **kwargs)

    def _strip_room_block_structure(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_strip_room_block_structure", *args, **kwargs)

    def _dilate_room_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_dilate_room_mask", *args, **kwargs)

    def _paint_room_line_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_paint_room_line_mask", *args, **kwargs)

    def _build_puzzle_room_route_template(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_build_puzzle_room_route_template", *args, **kwargs)

    def _resolve_puzzle_interaction_sequence(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_resolve_puzzle_interaction_sequence", *args, **kwargs)

    def _evaluate_puzzle_candidate_interaction_sequence(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_evaluate_puzzle_candidate_interaction_sequence", *args, **kwargs)

    def _select_puzzle_room_scaffold_archetype(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_select_puzzle_room_scaffold_archetype", *args, **kwargs)

    def _classify_puzzle_gate_family(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_classify_puzzle_gate_family", *args, **kwargs)

    def _build_puzzle_room_variant_specs(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_build_puzzle_room_variant_specs", *args, **kwargs)

    def _summarize_puzzle_candidate_descriptor(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_summarize_puzzle_candidate_descriptor", *args, **kwargs)

    def _build_room_walkable_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_build_room_walkable_mask", *args, **kwargs)

    def _count_room_path_turns(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_count_room_path_turns", *args, **kwargs)

    def _nearest_walkable_room_coord(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_nearest_walkable_room_coord", *args, **kwargs)

    def _shortest_room_path(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_shortest_room_path", *args, **kwargs)

    def _evaluate_puzzle_candidate_route_quality(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_evaluate_puzzle_candidate_route_quality", *args, **kwargs)

    def _evaluate_puzzle_candidate_contract(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_evaluate_puzzle_candidate_contract", *args, **kwargs)

    def _evaluate_puzzle_candidate_interaction_geometry(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_evaluate_puzzle_candidate_interaction_geometry", *args, **kwargs)

    def _puzzle_descriptor_distance(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_puzzle_descriptor_distance", *args, **kwargs)

    def _score_puzzle_candidate(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_score_puzzle_candidate", *args, **kwargs)

    def _commit_puzzle_novelty_choice(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_commit_puzzle_novelty_choice", *args, **kwargs)

    def _build_puzzle_room_segments(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_build_puzzle_room_segments", *args, **kwargs)

    def _strip_small_interior_structure_components(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_strip_small_interior_structure_components", *args, **kwargs)

    def _apply_puzzle_room_scaffold(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_apply_puzzle_room_scaffold", *args, **kwargs)

    def _count_small_interior_structure_components(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_count_small_interior_structure_components", *args, **kwargs)

    def _should_retry_room_with_teacher(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_should_retry_room_with_teacher", *args, **kwargs)

    def _resolve_effective_sampling_guidance(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_resolve_effective_sampling_guidance", *args, **kwargs)

    def _resolve_room_graph_markers(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_resolve_room_graph_markers", *args, **kwargs)

    def _build_room_graph_marker_preferences(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_build_room_graph_marker_preferences", *args, **kwargs)

    def _find_room_graph_marker_slot(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_find_room_graph_marker_slot", *args, **kwargs)

    def _overlay_room_graph_markers(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_overlay_room_graph_markers", *args, **kwargs)

    def _plan_room_graph_marker_layout(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_plan_room_graph_marker_layout", *args, **kwargs)

    def _build_room_puzzle_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_build_room_puzzle_metadata", *args, **kwargs)

    def _globalize_room_puzzle_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_globalize_room_puzzle_metadata", *args, **kwargs)

    def _measure_room_graph_marker_alignment(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_measure_room_graph_marker_alignment", *args, **kwargs)

    @staticmethod
    def _aggregate_room_alignment_metrics(*args: Any, **kwargs: Any) -> Any:
        from src.pipeline.generation import room_processing

        return room_processing._aggregate_room_alignment_metrics(None, *args, **kwargs)

    @staticmethod
    def _aggregate_puzzle_stage_semantics_metrics(*args: Any, **kwargs: Any) -> Any:
        from src.pipeline.generation import room_processing

        return room_processing._aggregate_puzzle_stage_semantics_metrics(None, *args, **kwargs)

    def _build_latent_edit_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_build_latent_edit_mask", *args, **kwargs)

    def _logicnet_guided_inpaint_room(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_logicnet_guided_inpaint_room", *args, **kwargs)

    def _wfc_guided_inpaint_room(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_wfc_guided_inpaint_room", *args, **kwargs)

    def _compute_room_condition(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_compute_room_condition", *args, **kwargs)

    def _topological_generation_layers(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_topological_generation_layers", *args, **kwargs)

    def _infer_room_latent_shape(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_infer_room_latent_shape", *args, **kwargs)

    def _normalize_neighbor_latents(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_normalize_neighbor_latents", *args, **kwargs)

    def _cast_latent_for_vqvae_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_cast_latent_for_vqvae_decode", *args, **kwargs)

    def _synchronize_cuda_device(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_synchronize_cuda_device", *args, **kwargs)

    def _decode_latent_with_vqvae(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_decode_latent_with_vqvae", *args, **kwargs)

    def _estimate_safe_batch_size(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_estimate_safe_batch_size", *args, **kwargs)

    def _stack_room_topology_maps(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_stack_room_topology_maps", *args, **kwargs)

    def _slice_graph_guidance_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_slice_graph_guidance_batch", *args, **kwargs)

    def _bucket_room_ids_by_latent_shape(self, *args: Any, **kwargs: Any) -> Any:
        return self._room_processing_helper("_bucket_room_ids_by_latent_shape", *args, **kwargs)


def create_pipeline(
    checkpoint_dir: str = "./checkpoints",
    device: str = "auto",
    *,
    config: Optional[PipelineConfig] = None,
    **kwargs: Any,
) -> NeuralSymbolicDungeonPipeline:
    """Create the facade pipeline using checkpoint-dir conventions."""
    if config is None:
        checkpoint_dir_path = Path(checkpoint_dir)
        resolved_config = kwargs.pop("resolved_config", None)
        if resolved_config is None:
            try:
                from src.config_system import load_resolved_config_for_artifact

                resolved_config = load_resolved_config_for_artifact(checkpoint_dir_path)
            except (ImportError, RuntimeError, ValueError, TypeError):
                resolved_config = None
        config = PipelineConfig.from_checkpoint_dir(
            checkpoint_dir_path,
            device=device,
            resolved_config=resolved_config,
            **kwargs,
        )
    elif kwargs:
        config = config.with_overrides(**kwargs)
    return NeuralSymbolicDungeonPipeline(config)


__all__ = [
    "NeuralSymbolicDungeonPipeline",
    "PipelineConfig",
    "ModelConfig",
    "SamplerConfig",
    "GraphConfig",
    "ModelManager",
    "DiffusionSampler",
    "DungeonAssembler",
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
    "ROOM_HEIGHT",
    "ROOM_WIDTH",
    "ROOM_TOPOLOGY_CHANNEL_COUNT",
    "DEFAULT_ROOM_LATENT_HW",
    "EvolutionaryTopologyGenerator",
    "validate_graph_topology",
    "topology_generation_kwargs_from_resolved_config",
    "generation_runtime_kwargs_from_resolved_config",
    "pipeline_kwargs_from_resolved_config",
    "create_pipeline",
]
