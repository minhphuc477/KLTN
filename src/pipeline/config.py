"""Configuration objects for the neural-symbolic dungeon pipeline facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_VALIDATOR_PLAN_MAX_STATES,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
)
from src.pipeline.types import PipelineComponentFactory, PipelineComponents


def _pop_dataclass_kwargs(config_type: type[Any], values: Dict[str, Any]) -> Dict[str, Any]:
    names = {field_info.name for field_info in fields(config_type)}
    selected: Dict[str, Any] = {}
    for name in list(values):
        if name in names:
            selected[name] = values.pop(name)
    return selected


@dataclass
class ModelConfig:
    """Model checkpoints, device placement, and model-construction options."""

    vqvae_checkpoint: Optional[str] = None
    diffusion_checkpoint: Optional[str] = None
    logic_net_checkpoint: Optional[str] = None
    condition_encoder_checkpoint: Optional[str] = None
    masked_room_checkpoint: Optional[str] = None
    fast_sampling_checkpoint: Optional[str] = None
    device: str = "auto"
    strict_checkpoint_mode: bool = False
    condition_encoder_fallback_config: Optional[Dict[str, Any]] = None
    diffusion_fallback_config: Optional[Dict[str, Any]] = None
    logic_net_fallback_config: Optional[Dict[str, Any]] = None
    masked_room_fallback_config: Optional[Dict[str, Any]] = None
    condition_gnn_type: str = "gcn"
    condition_use_reference_room_maps: bool = False
    condition_reference_tile_vocab_size: int = 44
    condition_reference_embedding_dim: int = 32
    condition_reference_hidden_dim: int = 64
    topology_refinement_mode: str = "gat2"
    diffusion_attention_mode: str = "softmax"
    diffusion_hedgehog_feature_dim: int = 32


@dataclass
class SamplerConfig:
    """Runtime sampling defaults for room and dungeon generation."""

    room_generator_mode: str = "latent_diffusion"
    default_guidance_scale: float = 3.0
    default_logic_guidance_scale: float = 0.0
    default_logic_guidance_strategy: str = "late"
    default_logic_guidance_active_fraction: float = 0.2
    default_num_diffusion_steps: int = 50
    default_use_fast_sampling: bool = False
    default_latent_sampler: str = "diffusion"
    default_categorical_codebook_size: Optional[int] = None
    diffusion_cfg_schedule_mode: str = "constant"
    diffusion_cfg_schedule_min_scale: float = 1.0
    diffusion_cfg_schedule_power: float = 1.0
    masked_sampling_steps: int = 8
    fast_sampling_steps: int = 4
    default_fast_sampler_teacher_fallback_enabled: bool = False
    default_masked_room_teacher_fallback_enabled: bool = False
    default_masked_room_sampling_temperature: float = 1.0
    default_masked_room_sampling_schedule: str = "cosine"
    default_masked_room_sampling_stochastic: bool = True
    default_masked_room_corrector_steps: int = 1
    default_masked_room_corrector_mask_ratio: float = 0.1


@dataclass
class GraphConfig:
    """Graph conditioning, topology generation, repair, and assembly defaults."""

    use_learned_refiner_rules: bool = True
    map_elites_resolution: int = 20
    map_elites_archive_path: Optional[str] = None
    map_elites_load_archive: bool = False
    map_elites_autosave_archive: bool = False
    use_graph_node_cross_attention: bool = True
    use_latent_boundary_masking: bool = True
    use_current_node_distance_features: bool = True
    current_node_distance_max: int = 8
    default_use_topological_positional_encoding: bool = True
    default_apply_repair: bool = True
    default_enable_map_elites: bool = False
    default_start_goal_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = ((1, 5), (14, 5))
    default_semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH
    default_semantic_anchor_threshold: float = 0.5
    default_semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET
    default_semantic_constrained_decoding_enabled: bool = True
    default_semantic_marker_logit_bias: float = 10000.0
    default_semantic_marker_suppression_bias: float = 100.0
    default_puzzle_room_scaffold_enabled: bool = True
    default_puzzle_room_structure_enabled: bool = True
    default_puzzle_room_scaffold_min_structure_tiles: int = 10
    default_puzzle_room_archetype_mode: str = "auto"
    default_puzzle_room_branch_density: float = 0.75
    default_puzzle_room_block_budget: int = 28
    default_puzzle_room_preserve_route_margin: int = 0
    default_puzzle_room_switch_pocket_depth: int = 3
    default_puzzle_room_resource_bypass_offset: int = 2
    default_puzzle_room_key_pocket_depth: int = 3
    default_puzzle_room_item_slot_depth: int = 3
    default_puzzle_room_toggle_corridor_offset: int = 2
    default_puzzle_room_novelty_enabled: bool = True
    default_puzzle_room_candidate_count: int = 4
    default_puzzle_room_novelty_weight: float = 0.45
    default_puzzle_room_min_quality_gain: float = 0.5
    default_validator_plan_max_states: int = DEFAULT_VALIDATOR_PLAN_MAX_STATES
    default_puzzle_stage_topology_enabled: bool = False
    default_puzzle_stage_trace_decay: float = DEFAULT_PUZZLE_STAGE_TRACE_DECAY
    default_deterministic_graph_marker_overlay_enabled: bool = True
    topology_default_target_curve: Optional[List[float]] = None
    topology_num_rooms: int = 8
    topology_population_size: int = 50
    topology_generations: int = 100
    topology_mutation_rate: float = 0.15
    topology_crossover_rate: float = 0.7
    topology_genome_length: int = 0
    topology_rule_space: str = "full"
    topology_transition_mix: float = 0.7
    topology_search_strategy: str = "ga"
    topology_qd_archive_cells: int = 128
    topology_qd_init_random_fraction: float = 0.35
    topology_qd_emitter_mutation_rate: float = 0.18
    topology_qd_archive_path: Optional[str] = None
    topology_qd_load_archive: bool = False
    topology_qd_autosave_archive: bool = False
    topology_max_lock_key_rules: int = 3
    topology_enable_rule_credit_assignment: bool = False
    topology_enforce_generation_constraints: bool = False
    topology_allow_candidate_repairs: bool = False
    symbolic_max_repair_attempts: int = 5
    symbolic_repair_margin: int = 2
    symbolic_adjacency_threshold: float = 0.01


@dataclass
class PipelineConfig:
    """Top-level pipeline facade configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    enable_logging: bool = True
    components: Optional[PipelineComponents] = None
    component_factory: Optional[PipelineComponentFactory] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "PipelineConfig":
        """Build grouped config from flat pipeline constructor kwargs."""
        values = dict(kwargs)
        model = ModelConfig(**_pop_dataclass_kwargs(ModelConfig, values))
        sampler = SamplerConfig(**_pop_dataclass_kwargs(SamplerConfig, values))
        graph = GraphConfig(**_pop_dataclass_kwargs(GraphConfig, values))
        enable_logging = bool(values.pop("enable_logging", True))
        components = values.pop("components", None)
        component_factory = values.pop("component_factory", None)
        return cls(
            model=model,
            sampler=sampler,
            graph=graph,
            enable_logging=enable_logging,
            components=components,
            component_factory=component_factory,
            extra_kwargs=values,
        )

    @classmethod
    def from_legacy_kwargs(cls, **kwargs: Any) -> "PipelineConfig":
        """Compatibility alias for callers using the former flat constructor name."""
        return cls.from_kwargs(**kwargs)

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: str | Path = "./checkpoints",
        *,
        device: str = "auto",
        resolved_config: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> "PipelineConfig":
        """Build config using the historical checkpoint directory convention."""
        checkpoint_path = Path(checkpoint_dir)
        kwargs: Dict[str, Any] = {}
        if isinstance(resolved_config, dict):
            from src.pipeline.config_bridge import pipeline_kwargs_from_resolved_config

            kwargs.update(pipeline_kwargs_from_resolved_config(resolved_config))
        kwargs.update(
            {
                "vqvae_checkpoint": str(checkpoint_path / "vqvae_best.pth"),
                "diffusion_checkpoint": str(checkpoint_path / "diffusion_best.pth"),
                "logic_net_checkpoint": str(checkpoint_path / "logic_net_best.pth"),
                "condition_encoder_checkpoint": str(checkpoint_path / "condition_encoder_best.pth"),
                "device": device,
            }
        )
        kwargs.update(overrides)
        return cls.from_kwargs(**kwargs)

    def with_overrides(self, **overrides: Any) -> "PipelineConfig":
        """Return a copy with flat constructor overrides applied."""
        merged = self.to_runtime_kwargs()
        merged.update(overrides)
        return type(self).from_kwargs(**merged)

    def to_runtime_kwargs(self) -> Dict[str, Any]:
        """Flatten grouped config for runtime initialization."""
        values: Dict[str, Any] = {}
        values.update(asdict(self.model))
        values.update(asdict(self.sampler))
        values.update(asdict(self.graph))
        values["enable_logging"] = bool(self.enable_logging)
        if self.components is not None:
            values["components"] = self.components
        if self.component_factory is not None:
            values["component_factory"] = self.component_factory
        values.update(dict(self.extra_kwargs))
        return values

    def to_legacy_kwargs(self) -> Dict[str, Any]:
        """Compatibility alias for callers expecting the former method name."""
        return self.to_runtime_kwargs()

    def replace(self, **changes: Any) -> "PipelineConfig":
        """Dataclass-style replacement helper for callers that prefer immutability."""
        return replace(self, **changes)


TopologyConfig = GraphConfig

__all__ = [
    "PipelineConfig",
    "ModelConfig",
    "SamplerConfig",
    "GraphConfig",
    "TopologyConfig",
]
