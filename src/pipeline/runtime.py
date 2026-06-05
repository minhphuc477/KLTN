"""Runtime state initialization and component binding for the pipeline facade."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from src.core import SEMANTIC_PALETTE
from src.core.definitions import TileID
from src.core.domain import resolve_domain_schema
from src.pipeline.block_contracts import validate_checkpoint_metadata
from src.pipeline.graph_features import condition_feature_dims, fit_feature_vector
from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    DEFAULT_VALIDATOR_PLAN_MAX_STATES,
    TOPOLOGY_ANCHOR_POLICY_VERSION,
)
from src.pipeline.types import (
    MissingPipelineComponentError,
    PipelineComponentFactory,
    PipelineComponents,
)
from src.utils.checkpoint import safe_torch_load

logger = logging.getLogger(__name__)


def initialize_pipeline(pipeline, config=None, **legacy_kwargs):
    """
    Initialize a pipeline instance from a grouped PipelineConfig.

    Flat keyword arguments are still accepted as a compatibility path, but the
    public runtime entrypoint no longer exposes the former 100+ parameter
    signature.
    """
    from src.pipeline.config import PipelineConfig

    if config is not None and legacy_kwargs:
        raise ValueError("Pass either config or legacy keyword overrides, not both.")
    if config is None:
        config = PipelineConfig.from_kwargs(**legacy_kwargs)
    elif isinstance(config, dict):
        config = PipelineConfig.from_kwargs(**config)
    elif not isinstance(config, PipelineConfig):
        raise TypeError(
            "initialize_pipeline config must be a PipelineConfig, a flat kwargs dict, "
            f"or omitted; got {type(config).__name__}."
        )
    return _initialize_pipeline_from_flat_kwargs(pipeline, **config.to_runtime_kwargs())


def _initialize_pipeline_from_flat_kwargs(
    pipeline,
    vqvae_checkpoint: Optional[str] = None,
    diffusion_checkpoint: Optional[str] = None,
    logic_net_checkpoint: Optional[str] = None,
    condition_encoder_checkpoint: Optional[str] = None,
    device: str = 'auto',
    use_learned_refiner_rules: bool = True,
    map_elites_resolution: int = 20,
    map_elites_archive_path: Optional[str] = None,
    map_elites_load_archive: bool = False,
    map_elites_autosave_archive: bool = False,
    enable_logging: bool = True,
    strict_checkpoint_mode: bool = False,
    use_graph_node_cross_attention: bool = True,
    use_latent_boundary_masking: bool = True,
    condition_gnn_type: str = "gcn",
    condition_use_reference_room_maps: bool = False,
    condition_reference_tile_vocab_size: int = 44,
    condition_reference_embedding_dim: int = 32,
    condition_reference_hidden_dim: int = 64,
    topology_refinement_mode: str = "gat2",
    diffusion_attention_mode: str = "softmax",
    diffusion_hedgehog_feature_dim: int = 32,
    diffusion_cfg_schedule_mode: str = "constant",
    diffusion_cfg_schedule_min_scale: float = 1.0,
    diffusion_cfg_schedule_power: float = 1.0,
    use_current_node_distance_features: bool = True,
    current_node_distance_max: int = 8,
    room_generator_mode: str = "latent_diffusion",
    masked_room_checkpoint: Optional[str] = None,
    masked_sampling_steps: int = 8,
    fast_sampling_checkpoint: Optional[str] = None,
    fast_sampling_steps: int = 4,
    default_guidance_scale: float = 3.0,
    default_logic_guidance_scale: float = 0.0,
    default_logic_guidance_strategy: str = "late",
    default_logic_guidance_active_fraction: float = 0.2,
    default_num_diffusion_steps: int = 50,
    default_use_fast_sampling: bool = False,
    default_latent_sampler: str = "diffusion",
    default_categorical_codebook_size: Optional[int] = None,
    default_use_topological_positional_encoding: bool = True,
    default_apply_repair: bool = True,
    default_use_neural_guided_repair: bool = True,
    default_use_neural_repair_feedback: bool = True,
    default_repair_inpaint_noise_strength: float = 0.5,
    default_repair_inpaint_guidance_scale_multiplier: float = 1.0,
    default_enable_map_elites: bool = False,
    default_start_goal_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = ((1, 5), (14, 5)),
    default_semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    default_semantic_anchor_threshold: float = 0.5,
    default_semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    default_semantic_constrained_decoding_enabled: bool = True,
    default_semantic_marker_logit_bias: float = 10000.0,
    default_semantic_marker_suppression_bias: float = 100.0,
    default_puzzle_room_scaffold_enabled: bool = True,
    default_puzzle_room_structure_enabled: bool = True,
    default_puzzle_room_scaffold_min_structure_tiles: int = 10,
    default_puzzle_room_archetype_mode: str = "auto",
    default_puzzle_room_branch_density: float = 0.75,
    default_puzzle_room_block_budget: int = 28,
    default_puzzle_room_preserve_route_margin: int = 0,
    default_puzzle_room_switch_pocket_depth: int = 3,
    default_puzzle_room_resource_bypass_offset: int = 2,
    default_puzzle_room_key_pocket_depth: int = 3,
    default_puzzle_room_item_slot_depth: int = 3,
    default_puzzle_room_toggle_corridor_offset: int = 2,
    default_puzzle_room_novelty_enabled: bool = True,
    default_puzzle_room_candidate_count: int = 4,
    default_puzzle_room_novelty_weight: float = 0.45,
    default_puzzle_room_min_quality_gain: float = 0.5,
    default_validator_plan_max_states: int = DEFAULT_VALIDATOR_PLAN_MAX_STATES,
    default_puzzle_stage_topology_enabled: bool = False,
    default_puzzle_stage_trace_decay: float = DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    default_deterministic_graph_marker_overlay_enabled: bool = True,
    default_fast_sampler_teacher_fallback_enabled: bool = False,
    default_masked_room_teacher_fallback_enabled: bool = False,
    default_masked_room_sampling_temperature: float = 1.0,
    default_masked_room_sampling_schedule: str = "cosine",
    default_masked_room_sampling_stochastic: bool = True,
    default_masked_room_corrector_steps: int = 1,
    default_masked_room_corrector_mask_ratio: float = 0.1,
    condition_encoder_fallback_config: Optional[Dict[str, Any]] = None,
    diffusion_fallback_config: Optional[Dict[str, Any]] = None,
    logic_net_fallback_config: Optional[Dict[str, Any]] = None,
    masked_room_fallback_config: Optional[Dict[str, Any]] = None,
    topology_default_target_curve: Optional[List[float]] = None,
    topology_num_rooms: int = 8,
    topology_population_size: int = 50,
    topology_generations: int = 100,
    topology_mutation_rate: float = 0.15,
    topology_crossover_rate: float = 0.7,
    topology_genome_length: int = 0,
    topology_rule_space: str = "full",
    topology_transition_mix: float = 0.7,
    topology_search_strategy: str = "ga",
    topology_qd_archive_cells: int = 128,
    topology_qd_init_random_fraction: float = 0.35,
    topology_qd_emitter_mutation_rate: float = 0.18,
    topology_qd_archive_path: Optional[str] = None,
    topology_qd_load_archive: bool = False,
    topology_qd_autosave_archive: bool = False,
    topology_max_lock_key_rules: int = 3,
    topology_enable_rule_credit_assignment: bool = False,
    topology_enforce_generation_constraints: bool = False,
    topology_allow_candidate_repairs: bool = False,
    symbolic_max_repair_attempts: int = 5,
    symbolic_repair_margin: int = 2,
    symbolic_adjacency_threshold: float = 0.01,
    domain_schema: Optional[Any] = None,
    components: Optional[PipelineComponents] = None,
    component_factory: Optional[PipelineComponentFactory] = None,
):
    # Device setup
    if device == 'auto':
        pipeline.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        pipeline.device = torch.device(device)

    if enable_logging:
        logger.info(f"Initializing NeuralSymbolicDungeonPipeline on {pipeline.device}")

    pipeline.strict_checkpoint_mode = bool(strict_checkpoint_mode)
    pipeline.domain_schema = resolve_domain_schema(domain_schema)
    pipeline.domain_schema_name = str(getattr(pipeline.domain_schema, "name", type(pipeline.domain_schema).__name__))
    pipeline.use_graph_node_cross_attention = bool(use_graph_node_cross_attention)
    pipeline.use_latent_boundary_masking = bool(use_latent_boundary_masking)
    pipeline.topology_refinement_mode = str(topology_refinement_mode).strip().lower()
    pipeline.diffusion_attention_mode = str(diffusion_attention_mode).strip().lower()
    pipeline.diffusion_cfg_schedule_mode = str(diffusion_cfg_schedule_mode).strip().lower()
    pipeline.diffusion_cfg_schedule_min_scale = float(max(0.0, diffusion_cfg_schedule_min_scale))
    pipeline.diffusion_cfg_schedule_power = float(max(1e-6, diffusion_cfg_schedule_power))
    pipeline.use_current_node_distance_features = bool(use_current_node_distance_features)
    pipeline.current_node_distance_max = int(max(1, int(current_node_distance_max)))
    pipeline.room_generator_mode = str(room_generator_mode).strip().lower()
    pipeline.masked_room_checkpoint = (
        None if masked_room_checkpoint is None else str(masked_room_checkpoint).strip()
    ) or None
    pipeline.masked_sampling_steps = int(max(1, int(masked_sampling_steps)))
    pipeline.fast_sampling_checkpoint = (
        None if fast_sampling_checkpoint is None else str(fast_sampling_checkpoint).strip()
    ) or None
    pipeline.fast_sampling_steps = int(max(1, int(fast_sampling_steps)))
    pipeline.default_guidance_scale = float(max(0.0, float(default_guidance_scale)))
    pipeline.default_logic_guidance_scale = float(max(0.0, float(default_logic_guidance_scale)))
    pipeline.default_logic_guidance_strategy = str(default_logic_guidance_strategy or "late").strip().lower()
    if pipeline.default_logic_guidance_strategy not in {"none", "late", "full"}:
        raise ValueError(
            "default_logic_guidance_strategy must be 'none', 'late', or 'full', "
            f"got {default_logic_guidance_strategy!r}."
        )
    pipeline.default_logic_guidance_active_fraction = float(
        max(0.05, min(1.0, float(default_logic_guidance_active_fraction)))
    )
    pipeline.default_num_diffusion_steps = int(max(1, int(default_num_diffusion_steps)))
    pipeline.default_use_fast_sampling = bool(default_use_fast_sampling)
    pipeline.default_latent_sampler = str(default_latent_sampler or "diffusion").strip().lower()
    pipeline.default_categorical_codebook_size = (
        None
        if default_categorical_codebook_size is None
        else int(max(1, int(default_categorical_codebook_size)))
    )
    pipeline.default_use_topological_positional_encoding = bool(default_use_topological_positional_encoding)
    pipeline.default_apply_repair = bool(default_apply_repair)
    pipeline.default_use_neural_guided_repair = bool(default_use_neural_guided_repair)
    pipeline.default_use_neural_repair_feedback = bool(default_use_neural_repair_feedback)
    pipeline.default_repair_inpaint_noise_strength = float(max(0.0, min(1.0, default_repair_inpaint_noise_strength)))
    pipeline.default_repair_inpaint_guidance_scale_multiplier = float(max(0.0, default_repair_inpaint_guidance_scale_multiplier))
    pipeline.default_enable_map_elites = bool(default_enable_map_elites)
    pipeline.default_start_goal_coords = (
        None
        if default_start_goal_coords is None
        else pipeline._normalize_start_goal_coords(default_start_goal_coords)
    )
    pipeline.default_semantic_role_prior_strength = float(
        max(0.0, min(1.0, float(default_semantic_role_prior_strength)))
    )
    pipeline.default_semantic_anchor_threshold = float(
        max(0.0, min(1.0, float(default_semantic_anchor_threshold)))
    )
    pipeline.default_semantic_puzzle_offset = int(max(0, int(default_semantic_puzzle_offset)))
    pipeline.default_semantic_constrained_decoding_enabled = bool(default_semantic_constrained_decoding_enabled)
    pipeline.default_semantic_marker_logit_bias = float(max(0.0, float(default_semantic_marker_logit_bias)))
    pipeline.default_semantic_marker_suppression_bias = float(
        max(0.0, float(default_semantic_marker_suppression_bias))
    )
    pipeline.default_puzzle_room_scaffold_enabled = bool(default_puzzle_room_scaffold_enabled)
    pipeline.default_puzzle_room_structure_enabled = bool(default_puzzle_room_structure_enabled)
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = int(
        max(0, int(default_puzzle_room_scaffold_min_structure_tiles))
    )
    scaffold_mode = str(default_puzzle_room_archetype_mode or "auto").strip().lower()
    if scaffold_mode not in {"auto", "gate", "serpentine", "hub", "island", "combat"}:
        scaffold_mode = "auto"
    pipeline.default_puzzle_room_archetype_mode = scaffold_mode
    pipeline.default_puzzle_room_branch_density = float(
        max(0.0, min(1.0, float(default_puzzle_room_branch_density)))
    )
    pipeline.default_puzzle_room_block_budget = int(max(0, int(default_puzzle_room_block_budget)))
    pipeline.default_puzzle_room_preserve_route_margin = int(
        max(0, min(4, int(default_puzzle_room_preserve_route_margin)))
    )
    pipeline.default_puzzle_room_switch_pocket_depth = int(
        max(1, min(6, int(default_puzzle_room_switch_pocket_depth)))
    )
    pipeline.default_puzzle_room_resource_bypass_offset = int(
        max(1, min(5, int(default_puzzle_room_resource_bypass_offset)))
    )
    pipeline.default_puzzle_room_key_pocket_depth = int(
        max(1, min(6, int(default_puzzle_room_key_pocket_depth)))
    )
    pipeline.default_puzzle_room_item_slot_depth = int(
        max(1, min(6, int(default_puzzle_room_item_slot_depth)))
    )
    pipeline.default_puzzle_room_toggle_corridor_offset = int(
        max(1, min(5, int(default_puzzle_room_toggle_corridor_offset)))
    )
    pipeline.default_puzzle_room_novelty_enabled = bool(default_puzzle_room_novelty_enabled)
    pipeline.default_puzzle_room_candidate_count = int(
        max(1, min(6, int(default_puzzle_room_candidate_count)))
    )
    pipeline.default_puzzle_room_novelty_weight = float(
        max(0.0, min(2.0, float(default_puzzle_room_novelty_weight)))
    )
    pipeline.default_puzzle_room_min_quality_gain = float(
        max(0.0, min(10.0, float(default_puzzle_room_min_quality_gain)))
    )
    pipeline.default_validator_plan_max_states = int(max(32, int(default_validator_plan_max_states)))
    pipeline.default_puzzle_stage_topology_enabled = bool(default_puzzle_stage_topology_enabled)
    pipeline.default_puzzle_stage_trace_decay = float(
        max(0.05, min(1.0, float(default_puzzle_stage_trace_decay)))
    )
    pipeline.default_deterministic_graph_marker_overlay_enabled = bool(
        default_deterministic_graph_marker_overlay_enabled
    )
    pipeline.default_fast_sampler_teacher_fallback_enabled = bool(default_fast_sampler_teacher_fallback_enabled)
    pipeline.default_masked_room_teacher_fallback_enabled = bool(default_masked_room_teacher_fallback_enabled)
    pipeline.default_masked_room_sampling_temperature = float(
        max(1e-6, float(default_masked_room_sampling_temperature))
    )
    masked_room_schedule = str(default_masked_room_sampling_schedule or "cosine").strip().lower()
    if masked_room_schedule not in {"cosine", "linear"}:
        masked_room_schedule = "cosine"
    pipeline.default_masked_room_sampling_schedule = masked_room_schedule
    pipeline.default_masked_room_sampling_stochastic = bool(default_masked_room_sampling_stochastic)
    pipeline.default_masked_room_corrector_steps = int(max(0, min(4, int(default_masked_room_corrector_steps))))
    pipeline.default_masked_room_corrector_mask_ratio = float(
        max(0.0, min(1.0, float(default_masked_room_corrector_mask_ratio)))
    )
    pipeline._puzzle_novelty_history: List[Dict[str, Any]] = []
    pipeline._puzzle_variant_cache: Dict[Any, Dict[str, Any]] = {}
    pipeline._puzzle_novelty_committed: Set[Any] = set()
    pipeline.logic_net_checkpoint_loaded = False
    pipeline.topology_anchor_policy_version = TOPOLOGY_ANCHOR_POLICY_VERSION
    pipeline.condition_encoder_fallback_config = dict(condition_encoder_fallback_config or {})
    pipeline.diffusion_fallback_config = dict(diffusion_fallback_config or {})
    pipeline.logic_net_fallback_config = dict(logic_net_fallback_config or {})
    pipeline.masked_room_fallback_config = dict(masked_room_fallback_config or {})
    pipeline.diffusion_puzzle_structure_condition_enabled = bool(
        float(pipeline.diffusion_fallback_config.get("puzzle_structure_dropout_prob", 0.0)) > 0.0
    )
    pipeline.masked_room_puzzle_structure_condition_enabled = bool(
        float(pipeline.masked_room_fallback_config.get("puzzle_structure_dropout_prob", 0.0)) > 0.0
    )
    default_curve = topology_default_target_curve
    if default_curve is None:
        default_curve = [0.2, 0.4, 0.6, 0.8, 1.0]
    pipeline.topology_default_target_curve = [float(v) for v in default_curve]
    if not pipeline.topology_default_target_curve:
        raise ValueError("topology_default_target_curve must be non-empty.")
    pipeline.topology_num_rooms = int(max(1, int(topology_num_rooms)))
    pipeline.topology_population_size = int(max(1, int(topology_population_size)))
    pipeline.topology_generations = int(max(1, int(topology_generations)))
    pipeline.topology_mutation_rate = float(np.clip(float(topology_mutation_rate), 0.0, 1.0))
    pipeline.topology_crossover_rate = float(np.clip(float(topology_crossover_rate), 0.0, 1.0))
    pipeline.topology_genome_length = int(max(0, int(topology_genome_length)))
    pipeline.topology_rule_space = str(topology_rule_space).strip().lower()
    pipeline.topology_transition_mix = float(np.clip(float(topology_transition_mix), 0.0, 1.0))
    pipeline.topology_search_strategy = str(topology_search_strategy).strip().lower()
    pipeline.topology_qd_archive_cells = int(max(32, int(topology_qd_archive_cells)))
    pipeline.topology_qd_init_random_fraction = float(
        np.clip(float(topology_qd_init_random_fraction), 0.05, 0.95)
    )
    pipeline.topology_qd_emitter_mutation_rate = float(
        np.clip(float(topology_qd_emitter_mutation_rate), 0.01, 0.95)
    )
    pipeline.topology_qd_archive_path = str(topology_qd_archive_path) if topology_qd_archive_path else None
    pipeline.topology_qd_load_archive = bool(topology_qd_load_archive)
    pipeline.topology_qd_autosave_archive = bool(topology_qd_autosave_archive)
    pipeline.topology_max_lock_key_rules = int(max(0, int(topology_max_lock_key_rules)))
    pipeline.topology_enable_rule_credit_assignment = bool(topology_enable_rule_credit_assignment)
    pipeline.topology_enforce_generation_constraints = bool(topology_enforce_generation_constraints)
    pipeline.topology_allow_candidate_repairs = bool(topology_allow_candidate_repairs)
    pipeline.symbolic_max_repair_attempts = int(max(1, int(symbolic_max_repair_attempts)))
    pipeline.symbolic_repair_margin = int(max(0, int(symbolic_repair_margin)))
    pipeline.symbolic_adjacency_threshold = float(max(0.0, float(symbolic_adjacency_threshold)))
    if pipeline.room_generator_mode not in {"latent_diffusion", "discrete_masked"}:
        raise ValueError(
            f"Invalid room_generator_mode={room_generator_mode!r}. "
            "Expected 'latent_diffusion' or 'discrete_masked'."
        )
    if pipeline.diffusion_attention_mode not in {"softmax", "linear_hedgehog"}:
        raise ValueError(
            f"Invalid diffusion_attention_mode={diffusion_attention_mode!r}. "
            "Expected 'softmax' or 'linear_hedgehog'."
        )
    if pipeline.default_latent_sampler not in {"diffusion", "categorical"}:
        raise ValueError(
            f"Invalid default_latent_sampler={default_latent_sampler!r}. "
            "Expected 'diffusion' or 'categorical'."
        )
    if pipeline.diffusion_cfg_schedule_mode not in {"constant", "linear_decay", "cosine_decay"}:
        raise ValueError(
            f"Invalid diffusion_cfg_schedule_mode={diffusion_cfg_schedule_mode!r}. "
            "Expected 'constant', 'linear_decay', or 'cosine_decay'."
        )
    pipeline.diffusion_hedgehog_feature_dim = int(max(4, int(diffusion_hedgehog_feature_dim)))
    if pipeline.topology_rule_space not in {"core", "full"}:
        raise ValueError(
            f"Invalid topology_rule_space={topology_rule_space!r}. Expected 'core' or 'full'."
        )
    if pipeline.topology_search_strategy not in {"ga", "cvt_emitter", "map_elites", "cvt", "cvt_map_elites"}:
        raise ValueError(
            f"Invalid topology_search_strategy={topology_search_strategy!r}. "
            "Expected 'ga', 'cvt_emitter', 'map_elites', 'cvt', or 'cvt_map_elites'."
        )
    if pipeline.topology_refinement_mode == "upgraded":
        pipeline.topology_refinement_mode = "gat2"
    if pipeline.topology_refinement_mode not in {"none", "lightweight", "gat2", "graphormer"}:
        raise ValueError(
            f"Invalid topology_refinement_mode={topology_refinement_mode!r}. "
            "Expected 'none', 'lightweight', 'gat2', or 'graphormer'."
        )
    gnn_type = str(condition_gnn_type).strip().lower()
    if gnn_type not in {"gcn", "gat", "sage", "gps"}:
        raise ValueError(
            f"Invalid condition_gnn_type={condition_gnn_type!r}. Expected 'gcn', 'gat', 'sage', or 'gps'."
        )
    pipeline.condition_gnn_type = gnn_type
    pipeline.condition_use_reference_room_maps = bool(condition_use_reference_room_maps)
    pipeline.condition_reference_tile_vocab_size = int(max(2, int(condition_reference_tile_vocab_size)))
    pipeline.condition_reference_embedding_dim = int(max(4, int(condition_reference_embedding_dim)))
    pipeline.condition_reference_hidden_dim = int(max(4, int(condition_reference_hidden_dim)))

    # Runtime fallback diagnostics for auditability of best-effort paths.
    pipeline.runtime_diagnostics: Dict[str, int] = {}
    pipeline._valid_semantic_tile_ids_np = np.array(
        sorted({int(v) for v in SEMANTIC_PALETTE.values()}),
        dtype=np.int32,
    )
    pipeline._volatile_room_semantic_tile_ids_np = np.array(
        sorted(
            {
                int(TileID.ENEMY),
                int(TileID.START),
                int(TileID.TRIFORCE),
                int(TileID.BOSS),
                int(TileID.KEY_SMALL),
                int(TileID.KEY_BOSS),
                int(TileID.KEY_ITEM),
                int(TileID.ITEM_MINOR),
                int(TileID.ELEMENT),
                int(TileID.ELEMENT_FLOOR),
                int(TileID.STAIR),
                int(TileID.PUZZLE),
            }
        ),
        dtype=np.int32,
    )

    if components is not None and component_factory is not None:
        raise ValueError("Pass either components or component_factory, not both.")

    if components is None:
        factory = component_factory or PipelineComponentFactory(
            vqvae_checkpoint=vqvae_checkpoint,
            diffusion_checkpoint=diffusion_checkpoint,
            logic_net_checkpoint=logic_net_checkpoint,
            condition_encoder_checkpoint=condition_encoder_checkpoint,
            use_learned_refiner_rules=use_learned_refiner_rules,
            map_elites_resolution=map_elites_resolution,
            map_elites_archive_path=map_elites_archive_path,
            map_elites_load_archive=map_elites_load_archive,
            map_elites_autosave_archive=map_elites_autosave_archive,
            symbolic_max_repair_attempts=pipeline.symbolic_max_repair_attempts,
            symbolic_repair_margin=pipeline.symbolic_repair_margin,
            symbolic_adjacency_threshold=pipeline.symbolic_adjacency_threshold,
        )
        components = factory.build(pipeline)

    pipeline._bind_components(components)
    pipeline.masked_room_model = pipeline._load_masked_room_model(pipeline.masked_room_checkpoint)

    if enable_logging:
        logger.info(
            "Pipeline initialized successfully (components=%s)",
            pipeline.component_status(),
        )


def _load_checkpoint_and_metadata(
    pipeline,
    checkpoint_path: str,
    model_name: str,
    *,
    accepted_model_types: Optional[Tuple[str, ...]] = None,
) -> Tuple[dict, dict]:
    """Load checkpoint and optional sidecar metadata for strict validation."""
    checkpoint = safe_torch_load(checkpoint_path, map_location=pipeline.device)
    metadata_path = Path(f"{checkpoint_path}.meta.json")
    metadata: dict = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        validate_checkpoint_metadata(
            metadata=metadata,
            model_name=model_name,
            accepted_model_types=accepted_model_types,
        )
    elif pipeline.strict_checkpoint_mode:
        raise FileNotFoundError(
            f"Strict checkpoint mode enabled: metadata sidecar missing for {model_name} at {metadata_path}"
        )
    return checkpoint, metadata


def _extract_checkpoint_config(pipeline, checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return dict(checkpoint["config"])
    return {}


def _extract_checkpoint_state_dict(pipeline, checkpoint: Any, *candidate_keys: str) -> Optional[Dict[str, Any]]:
    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
        state = checkpoint.get("model_state_dict")
        if isinstance(state, dict):
            return state
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return checkpoint
    return None


def _bump_diagnostic(pipeline, key: str) -> None:
    """Increment a named runtime diagnostic counter."""
    k = str(key).strip().lower()
    if not k:
        return
    pipeline.runtime_diagnostics[k] = int(pipeline.runtime_diagnostics.get(k, 0)) + 1


def _prepare_component(
    pipeline,
    component: Optional[Any],
    *,
    eval_mode: bool = False,
) -> Optional[Any]:
    """Move module components to the pipeline device and switch them to eval mode."""
    if component is None:
        return None
    if isinstance(component, torch.nn.Module):
        component = component.to(pipeline.device)
        if eval_mode:
            component.eval()
    return component


def _bind_components(pipeline, components: PipelineComponents) -> None:
    """Bind an injected component bundle to legacy pipeline attributes."""
    pipeline.components = components

    pipeline.vqvae = pipeline._prepare_component(components.neural.vqvae, eval_mode=True)
    pipeline.condition_encoder = pipeline._prepare_component(components.neural.condition_encoder, eval_mode=True)
    pipeline.diffusion = pipeline._prepare_component(components.neural.diffusion, eval_mode=True)
    pipeline.logic_net = pipeline._prepare_component(components.neural.logic_net, eval_mode=True)
    pipeline.refiner = components.symbolic.refiner
    pipeline.stitcher = components.symbolic.stitcher
    pipeline.map_elites = components.symbolic.map_elites

    if pipeline.logic_net is not None:
        pipeline.logic_net_checkpoint_loaded = bool(
            getattr(pipeline.logic_net, "_hmolqd_checkpoint_loaded", True)
        )
    else:
        pipeline.logic_net_checkpoint_loaded = False

    pipeline.components.neural.vqvae = pipeline.vqvae
    pipeline.components.neural.condition_encoder = pipeline.condition_encoder
    pipeline.components.neural.diffusion = pipeline.diffusion
    pipeline.components.neural.logic_net = pipeline.logic_net
    pipeline.components.symbolic.refiner = pipeline.refiner
    pipeline.components.symbolic.stitcher = pipeline.stitcher
    pipeline.components.symbolic.map_elites = pipeline.map_elites


def component_status(pipeline) -> Dict[str, bool]:
    """Return which injectable components are currently configured."""
    return {
        'vqvae': pipeline.vqvae is not None,
        'condition_encoder': pipeline.condition_encoder is not None,
        'diffusion': pipeline.diffusion is not None,
        'masked_room_model': getattr(pipeline, 'masked_room_model', None) is not None,
        'logic_net': pipeline.logic_net is not None,
        'refiner': pipeline.refiner is not None,
        'stitcher': pipeline.stitcher is not None,
        'map_elites': pipeline.map_elites is not None,
    }


def supports_room_generation(pipeline) -> bool:
    """Whether the neural room-generation stack is configured."""
    if pipeline.room_generator_mode == "discrete_masked":
        return (
            pipeline.condition_encoder is not None
            and getattr(pipeline, "masked_room_model", None) is not None
        )
    if str(getattr(pipeline, "default_latent_sampler", "diffusion") or "diffusion").strip().lower() == "categorical":
        return (
            pipeline.vqvae is not None
            and pipeline.condition_encoder is not None
        )
    return (
        pipeline.vqvae is not None
        and pipeline.condition_encoder is not None
        and pipeline.diffusion is not None
    )


def supports_symbolic_repair(pipeline) -> bool:
    """Whether symbolic room repair is available."""
    return pipeline.refiner is not None


def _require_component(pipeline, component_name: str, operation: str) -> Any:
    """Return a configured component or raise a targeted capability error."""
    component = getattr(pipeline, component_name, None)
    if component is None:
        raise MissingPipelineComponentError(
            f"{operation} requires pipeline component '{component_name}', "
            "but it is not configured. Inject it via PipelineComponents or "
            "use a pipeline constructor that provides the full stack."
        )
    return component


def _require_room_generation_components(
    pipeline,
    operation: str,
    *,
    latent_sampler: Optional[str] = None,
    room_generator_mode: Optional[str] = None,
) -> None:
    """Ensure the core neural stack is available for room generation."""
    mode = (
        pipeline.room_generator_mode
        if room_generator_mode is None
        else str(room_generator_mode).strip().lower()
    )
    sampler_mode = str(
        pipeline.default_latent_sampler
        if latent_sampler is None
        else latent_sampler
    ).strip().lower()
    required = (
        ('condition_encoder', 'masked_room_model')
        if mode == "discrete_masked"
        else (
            ('vqvae', 'condition_encoder')
            if sampler_mode == "categorical"
            else ('vqvae', 'condition_encoder', 'diffusion')
        )
    )
    missing = [name for name in required if getattr(pipeline, name, None) is None]
    if missing:
        raise MissingPipelineComponentError(
            f"{operation} requires neural generation components {missing}, "
            "but this pipeline was initialized without them."
        )


def _condition_feature_dims(pipeline) -> Tuple[int, int]:
    """Get active (node_dim, edge_dim) expected by the condition encoder."""
    encoder = pipeline._require_component("condition_encoder", "_condition_feature_dims")
    return condition_feature_dims(encoder)


def _fit_feature_vector(pipeline, values: List[float], target_dim: int) -> List[float]:
    """Pad/truncate feature list to target dimension."""
    return fit_feature_vector(values, target_dim)

