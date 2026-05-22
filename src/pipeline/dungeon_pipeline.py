"""
H-MOLQD Master Pipeline - Neural-Symbolic Dungeon Generation
==============================================================

Complete 7-block pipeline for Legend of Zelda dungeon generation.

Pipeline Architecture:
    Block 0:   Data Adapter (zelda_core.py) [offline preprocessing]
    Block I:   Evolutionary Topology Director (evolutionary_director.py)
    Block II:  Semantic VQ-VAE (vqvae.py)
    Block III: Dual-Stream Condition Encoder (condition_encoder.py)
    Block IV:  Latent Diffusion with Guidance (latent_diffusion.py)
    Block V:   LogicNet (logic_net.py)
    Block VI:  Symbolic Refiner (symbolic_refiner.py)
    Block VII: MAP-Elites Validator (map_elites.py)

Usage:
    pipeline = NeuralSymbolicDungeonPipeline(
        vqvae_checkpoint="checkpoints/vqvae_best.pth",
        diffusion_checkpoint="checkpoints/diffusion_best.pth",
        logic_net_checkpoint="checkpoints/logic_net_best.pth",
    )
    
    # Generate single room
    result = pipeline.generate_room(
        neighbor_latents={'N': z_north, 'W': z_west},
        graph_context=graph_data,
        room_id=5,
        seed=42
    )
    
    # Generate full dungeon
    dungeon_result = pipeline.generate_dungeon(
        mission_graph=nx.Graph(...),
        guidance_scale=3.0,
        logic_guidance_scale=0.0,
        seed=42
    )
"""

import json
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Mapping, Sequence
from dataclasses import dataclass, field

import torch
import numpy as np
import networkx as nx

from src.core import (
    SemanticVQVAE,
    DualStreamConditionEncoder,
    LatentDiffusionModel,
    DiscreteMaskedRoomModel,
    LogicNet,
    SymbolicRefiner,
    LearnedTileStatistics,
    SEMANTIC_PALETTE,
    ROOM_HEIGHT,
    ROOM_WIDTH,
)
from src.core.definitions import (
    DOOR_POSITIONS,
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    TileID,
    parse_edge_type_tokens,
)
from src.core.symbolic_refiner import DEFAULT_ADJACENCY
from src.simulation.map_elites import MAPElitesEvaluator
# Block I: Evolutionary Topology Director
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator

# VGLC compliance imports
from src.zelda_data.vglc_utils import (
    filter_virtual_nodes,
    validate_room_dimensions,
    get_physical_start_node,
)
from src.utils.graph_utils import validate_graph_topology
from src.utils.stable_seed import stable_seed_offset
from src.utils.style_tokens import iter_style_metadata_candidates, resolve_style_token_id
from src.pipeline.repair_feedback import (
    build_latent_edit_mask,
    build_neighbor_boundary_inpaint_inputs,
    wfc_guided_inpaint_room,
)
from src.pipeline.spatial_utils import (
    carve_room_connection,
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
from src.pipeline.graph_features import (
    compute_current_node_distance_features,
    compute_tpe_features,
    condition_feature_dims,
    encode_edge_feature_vector,
    extract_node_feature_vector,
    fit_feature_vector,
)
from src.pipeline.room_stitching import (
    StitchedRoomLayout,
    build_stitched_room_layout,
    carve_room_connection_between_bboxes,
    compute_relaxed_room_placement,
    compute_strict_room_placement,
    solve_component_strict_adjacency,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_VALIDATOR_PLAN_MAX_STATES,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    TOPOLOGY_ANCHOR_POLICY_VERSION,
    apply_puzzle_structure_control_to_conditioning,
    build_puzzle_stage_condition_metadata,
    build_room_semantic_anchor_points,
    build_semantic_room_plan_trace,
    build_room_topology_condition_map,
)
from src.core.condition_encoder import build_boundary_constraints
from src.core.vqvae import canonical_latent_shape
from src.pipeline.block_contracts import (
    BlockShapeContract,
    summarize_missing_keys,
    validate_checkpoint_metadata,
    validate_feature_dims,
    validate_tensor_contract,
)

logger = logging.getLogger(__name__)

_stable_node_sort_key = stable_node_sort_key
DEFAULT_ROOM_LATENT_HW: Tuple[int, int] = canonical_latent_shape((ROOM_HEIGHT, ROOM_WIDTH))

def _stable_node_seed_offset(node: Any) -> int:
    """Deterministic integer seed offset for arbitrary node-id types."""
    return stable_seed_offset(node, digest_size=4)


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
    symbolic_max_repair_attempts: int = 5
    symbolic_repair_margin: int = 2
    symbolic_adjacency_threshold: float = 0.01

    def build(self, pipeline: "NeuralSymbolicDungeonPipeline") -> PipelineComponents:
        return PipelineComponents(
            neural=NeuralGenerationComponents(
                vqvae=pipeline._load_vqvae(self.vqvae_checkpoint),
                condition_encoder=pipeline._load_condition_encoder(self.condition_encoder_checkpoint),
                diffusion=pipeline._load_diffusion(self.diffusion_checkpoint),
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
                    tie_breaker='quality_score',
                    descriptor_mode='hybrid',
                ),
            ),
        )


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RoomGenerationResult:
    """Result of generating a single room."""
    room_id: int
    room_grid: np.ndarray  # (16, 11) discrete tile IDs
    latent: torch.Tensor  # (1, 64, 4, 3) detached CPU latent
    neural_grid: np.ndarray  # (16, 11) before symbolic repair
    was_repaired: bool
    repair_mask: Optional[np.ndarray] = None  # (16, 11) bool mask
    room_plan_mask: Optional[np.ndarray] = None  # (16, 11) float/bool traversability prior
    neural_probs: Optional[np.ndarray] = None  # (44, 16, 11) pre-repair tile probabilities
    puzzle_metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DungeonGenerationResult:
    """Result of generating a complete dungeon."""
    dungeon_grid: np.ndarray  # (H, W) stitched dungeon
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
    batch_runtime_diagnostics: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# MASTER PIPELINE
# =============================================================================

class NeuralSymbolicDungeonPipeline:
    """
    Complete 7-block neural-symbolic dungeon generation pipeline.
    
    Orchestrates:
    - VQ-VAE latent encoding/decoding
    - Dual-stream context conditioning
    - Latent diffusion with LogicNet guidance
    - Symbolic WFC repair
    - MAP-Elites quality-diversity evaluation
    
    Args:
        vqvae_checkpoint: Path to VQ-VAE checkpoint
        diffusion_checkpoint: Path to diffusion checkpoint
        logic_net_checkpoint: Path to LogicNet checkpoint
        condition_encoder_checkpoint: Optional condition encoder checkpoint
        device: Device to run on ('cuda', 'cpu', or 'auto')
        use_learned_refiner_rules: Use learned tile statistics for WFC
        map_elites_resolution: MAP-Elites grid resolution
    """
    
    def __init__(
        self,
        vqvae_checkpoint: Optional[str] = None,
        diffusion_checkpoint: Optional[str] = None,
        logic_net_checkpoint: Optional[str] = None,
        condition_encoder_checkpoint: Optional[str] = None,
        device: str = 'auto',
        use_learned_refiner_rules: bool = True,
        map_elites_resolution: int = 20,
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
        default_num_diffusion_steps: int = 50,
        default_use_fast_sampling: bool = False,
        default_latent_sampler: str = "diffusion",
        default_categorical_codebook_size: Optional[int] = None,
        default_use_topological_positional_encoding: bool = True,
        default_apply_repair: bool = True,
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
        default_fast_sampler_teacher_fallback_enabled: bool = True,
        default_masked_room_teacher_fallback_enabled: bool = True,
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
        topology_max_lock_key_rules: int = 3,
        topology_enable_rule_credit_assignment: bool = False,
        topology_enforce_generation_constraints: bool = False,
        topology_allow_candidate_repairs: bool = False,
        symbolic_max_repair_attempts: int = 5,
        symbolic_repair_margin: int = 2,
        symbolic_adjacency_threshold: float = 0.01,
        components: Optional[PipelineComponents] = None,
        component_factory: Optional[PipelineComponentFactory] = None,
    ):
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if enable_logging:
            logger.info(f"Initializing NeuralSymbolicDungeonPipeline on {self.device}")

        self.strict_checkpoint_mode = bool(strict_checkpoint_mode)
        self.use_graph_node_cross_attention = bool(use_graph_node_cross_attention)
        self.use_latent_boundary_masking = bool(use_latent_boundary_masking)
        self.topology_refinement_mode = str(topology_refinement_mode).strip().lower()
        self.diffusion_attention_mode = str(diffusion_attention_mode).strip().lower()
        self.diffusion_cfg_schedule_mode = str(diffusion_cfg_schedule_mode).strip().lower()
        self.diffusion_cfg_schedule_min_scale = float(max(0.0, diffusion_cfg_schedule_min_scale))
        self.diffusion_cfg_schedule_power = float(max(1e-6, diffusion_cfg_schedule_power))
        self.use_current_node_distance_features = bool(use_current_node_distance_features)
        self.current_node_distance_max = int(max(1, int(current_node_distance_max)))
        self.room_generator_mode = str(room_generator_mode).strip().lower()
        self.masked_room_checkpoint = (
            None if masked_room_checkpoint is None else str(masked_room_checkpoint).strip()
        ) or None
        self.masked_sampling_steps = int(max(1, int(masked_sampling_steps)))
        self.fast_sampling_checkpoint = (
            None if fast_sampling_checkpoint is None else str(fast_sampling_checkpoint).strip()
        ) or None
        self.fast_sampling_steps = int(max(1, int(fast_sampling_steps)))
        self.default_guidance_scale = float(max(0.0, float(default_guidance_scale)))
        self.default_logic_guidance_scale = float(max(0.0, float(default_logic_guidance_scale)))
        self.default_num_diffusion_steps = int(max(1, int(default_num_diffusion_steps)))
        self.default_use_fast_sampling = bool(default_use_fast_sampling)
        self.default_latent_sampler = str(default_latent_sampler or "diffusion").strip().lower()
        self.default_categorical_codebook_size = (
            None
            if default_categorical_codebook_size is None
            else int(max(1, int(default_categorical_codebook_size)))
        )
        self.default_use_topological_positional_encoding = bool(default_use_topological_positional_encoding)
        self.default_apply_repair = bool(default_apply_repair)
        self.default_enable_map_elites = bool(default_enable_map_elites)
        self.default_start_goal_coords = (
            None
            if default_start_goal_coords is None
            else self._normalize_start_goal_coords(default_start_goal_coords)
        )
        self.default_semantic_role_prior_strength = float(
            max(0.0, min(1.0, float(default_semantic_role_prior_strength)))
        )
        self.default_semantic_anchor_threshold = float(
            max(0.0, min(1.0, float(default_semantic_anchor_threshold)))
        )
        self.default_semantic_puzzle_offset = int(max(0, int(default_semantic_puzzle_offset)))
        self.default_semantic_constrained_decoding_enabled = bool(default_semantic_constrained_decoding_enabled)
        self.default_semantic_marker_logit_bias = float(max(0.0, float(default_semantic_marker_logit_bias)))
        self.default_semantic_marker_suppression_bias = float(
            max(0.0, float(default_semantic_marker_suppression_bias))
        )
        self.default_puzzle_room_scaffold_enabled = bool(default_puzzle_room_scaffold_enabled)
        self.default_puzzle_room_structure_enabled = bool(default_puzzle_room_structure_enabled)
        self.default_puzzle_room_scaffold_min_structure_tiles = int(
            max(0, int(default_puzzle_room_scaffold_min_structure_tiles))
        )
        scaffold_mode = str(default_puzzle_room_archetype_mode or "auto").strip().lower()
        if scaffold_mode not in {"auto", "gate", "serpentine", "hub", "island", "combat"}:
            scaffold_mode = "auto"
        self.default_puzzle_room_archetype_mode = scaffold_mode
        self.default_puzzle_room_branch_density = float(
            max(0.0, min(1.0, float(default_puzzle_room_branch_density)))
        )
        self.default_puzzle_room_block_budget = int(max(0, int(default_puzzle_room_block_budget)))
        self.default_puzzle_room_preserve_route_margin = int(
            max(0, min(4, int(default_puzzle_room_preserve_route_margin)))
        )
        self.default_puzzle_room_switch_pocket_depth = int(
            max(1, min(6, int(default_puzzle_room_switch_pocket_depth)))
        )
        self.default_puzzle_room_resource_bypass_offset = int(
            max(1, min(5, int(default_puzzle_room_resource_bypass_offset)))
        )
        self.default_puzzle_room_key_pocket_depth = int(
            max(1, min(6, int(default_puzzle_room_key_pocket_depth)))
        )
        self.default_puzzle_room_item_slot_depth = int(
            max(1, min(6, int(default_puzzle_room_item_slot_depth)))
        )
        self.default_puzzle_room_toggle_corridor_offset = int(
            max(1, min(5, int(default_puzzle_room_toggle_corridor_offset)))
        )
        self.default_puzzle_room_novelty_enabled = bool(default_puzzle_room_novelty_enabled)
        self.default_puzzle_room_candidate_count = int(
            max(1, min(6, int(default_puzzle_room_candidate_count)))
        )
        self.default_puzzle_room_novelty_weight = float(
            max(0.0, min(2.0, float(default_puzzle_room_novelty_weight)))
        )
        self.default_puzzle_room_min_quality_gain = float(
            max(0.0, min(10.0, float(default_puzzle_room_min_quality_gain)))
        )
        self.default_validator_plan_max_states = int(max(32, int(default_validator_plan_max_states)))
        self.default_puzzle_stage_topology_enabled = bool(default_puzzle_stage_topology_enabled)
        self.default_puzzle_stage_trace_decay = float(
            max(0.05, min(1.0, float(default_puzzle_stage_trace_decay)))
        )
        self.default_deterministic_graph_marker_overlay_enabled = bool(
            default_deterministic_graph_marker_overlay_enabled
        )
        self.default_fast_sampler_teacher_fallback_enabled = bool(default_fast_sampler_teacher_fallback_enabled)
        self.default_masked_room_teacher_fallback_enabled = bool(default_masked_room_teacher_fallback_enabled)
        self.default_masked_room_sampling_temperature = float(
            max(1e-6, float(default_masked_room_sampling_temperature))
        )
        masked_room_schedule = str(default_masked_room_sampling_schedule or "cosine").strip().lower()
        if masked_room_schedule not in {"cosine", "linear"}:
            masked_room_schedule = "cosine"
        self.default_masked_room_sampling_schedule = masked_room_schedule
        self.default_masked_room_sampling_stochastic = bool(default_masked_room_sampling_stochastic)
        self.default_masked_room_corrector_steps = int(max(0, min(4, int(default_masked_room_corrector_steps))))
        self.default_masked_room_corrector_mask_ratio = float(
            max(0.0, min(1.0, float(default_masked_room_corrector_mask_ratio)))
        )
        self._puzzle_novelty_history: List[Dict[str, Any]] = []
        self._puzzle_variant_cache: Dict[Any, Dict[str, Any]] = {}
        self._puzzle_novelty_committed: Set[Any] = set()
        self.topology_anchor_policy_version = TOPOLOGY_ANCHOR_POLICY_VERSION
        self.condition_encoder_fallback_config = dict(condition_encoder_fallback_config or {})
        self.diffusion_fallback_config = dict(diffusion_fallback_config or {})
        self.logic_net_fallback_config = dict(logic_net_fallback_config or {})
        self.masked_room_fallback_config = dict(masked_room_fallback_config or {})
        self.diffusion_puzzle_structure_condition_enabled = bool(
            float(self.diffusion_fallback_config.get("puzzle_structure_dropout_prob", 0.0)) > 0.0
        )
        self.masked_room_puzzle_structure_condition_enabled = bool(
            float(self.masked_room_fallback_config.get("puzzle_structure_dropout_prob", 0.0)) > 0.0
        )
        default_curve = topology_default_target_curve
        if default_curve is None:
            default_curve = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.topology_default_target_curve = [float(v) for v in default_curve]
        if not self.topology_default_target_curve:
            raise ValueError("topology_default_target_curve must be non-empty.")
        self.topology_num_rooms = int(max(1, int(topology_num_rooms)))
        self.topology_population_size = int(max(1, int(topology_population_size)))
        self.topology_generations = int(max(1, int(topology_generations)))
        self.topology_mutation_rate = float(np.clip(float(topology_mutation_rate), 0.0, 1.0))
        self.topology_crossover_rate = float(np.clip(float(topology_crossover_rate), 0.0, 1.0))
        self.topology_genome_length = int(max(0, int(topology_genome_length)))
        self.topology_rule_space = str(topology_rule_space).strip().lower()
        self.topology_transition_mix = float(np.clip(float(topology_transition_mix), 0.0, 1.0))
        self.topology_search_strategy = str(topology_search_strategy).strip().lower()
        self.topology_qd_archive_cells = int(max(32, int(topology_qd_archive_cells)))
        self.topology_qd_init_random_fraction = float(
            np.clip(float(topology_qd_init_random_fraction), 0.05, 0.95)
        )
        self.topology_qd_emitter_mutation_rate = float(
            np.clip(float(topology_qd_emitter_mutation_rate), 0.01, 0.95)
        )
        self.topology_max_lock_key_rules = int(max(0, int(topology_max_lock_key_rules)))
        self.topology_enable_rule_credit_assignment = bool(topology_enable_rule_credit_assignment)
        self.topology_enforce_generation_constraints = bool(topology_enforce_generation_constraints)
        self.topology_allow_candidate_repairs = bool(topology_allow_candidate_repairs)
        self.symbolic_max_repair_attempts = int(max(1, int(symbolic_max_repair_attempts)))
        self.symbolic_repair_margin = int(max(0, int(symbolic_repair_margin)))
        self.symbolic_adjacency_threshold = float(max(0.0, float(symbolic_adjacency_threshold)))
        if self.room_generator_mode not in {"latent_diffusion", "discrete_masked"}:
            raise ValueError(
                f"Invalid room_generator_mode={room_generator_mode!r}. "
                "Expected 'latent_diffusion' or 'discrete_masked'."
            )
        if self.diffusion_attention_mode not in {"softmax", "linear_hedgehog"}:
            raise ValueError(
                f"Invalid diffusion_attention_mode={diffusion_attention_mode!r}. "
                "Expected 'softmax' or 'linear_hedgehog'."
            )
        if self.default_latent_sampler not in {"diffusion", "categorical"}:
            raise ValueError(
                f"Invalid default_latent_sampler={default_latent_sampler!r}. "
                "Expected 'diffusion' or 'categorical'."
            )
        if self.diffusion_cfg_schedule_mode not in {"constant", "linear_decay", "cosine_decay"}:
            raise ValueError(
                f"Invalid diffusion_cfg_schedule_mode={diffusion_cfg_schedule_mode!r}. "
                "Expected 'constant', 'linear_decay', or 'cosine_decay'."
            )
        self.diffusion_hedgehog_feature_dim = int(max(4, int(diffusion_hedgehog_feature_dim)))
        if self.topology_rule_space not in {"core", "full"}:
            raise ValueError(
                f"Invalid topology_rule_space={topology_rule_space!r}. Expected 'core' or 'full'."
            )
        if self.topology_search_strategy not in {"ga", "cvt_emitter", "map_elites", "cvt", "cvt_map_elites"}:
            raise ValueError(
                f"Invalid topology_search_strategy={topology_search_strategy!r}. "
                "Expected 'ga', 'cvt_emitter', 'map_elites', 'cvt', or 'cvt_map_elites'."
            )
        if self.topology_refinement_mode == "upgraded":
            self.topology_refinement_mode = "gat2"
        if self.topology_refinement_mode not in {"none", "lightweight", "gat2"}:
            raise ValueError(
                f"Invalid topology_refinement_mode={topology_refinement_mode!r}. "
                "Expected 'none', 'lightweight', or 'gat2'."
            )
        gnn_type = str(condition_gnn_type).strip().lower()
        if gnn_type not in {"gcn", "gat", "sage", "gps"}:
            raise ValueError(
                f"Invalid condition_gnn_type={condition_gnn_type!r}. Expected 'gcn', 'gat', 'sage', or 'gps'."
            )
        self.condition_gnn_type = gnn_type
        self.condition_use_reference_room_maps = bool(condition_use_reference_room_maps)
        self.condition_reference_tile_vocab_size = int(max(2, int(condition_reference_tile_vocab_size)))
        self.condition_reference_embedding_dim = int(max(4, int(condition_reference_embedding_dim)))
        self.condition_reference_hidden_dim = int(max(4, int(condition_reference_hidden_dim)))

        # Runtime fallback diagnostics for auditability of best-effort paths.
        self.runtime_diagnostics: Dict[str, int] = {}
        self._valid_semantic_tile_ids_np = np.array(
            sorted({int(v) for v in SEMANTIC_PALETTE.values()}),
            dtype=np.int32,
        )
        self._volatile_room_semantic_tile_ids_np = np.array(
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
                symbolic_max_repair_attempts=self.symbolic_max_repair_attempts,
                symbolic_repair_margin=self.symbolic_repair_margin,
                symbolic_adjacency_threshold=self.symbolic_adjacency_threshold,
            )
            components = factory.build(self)

        self._bind_components(components)
        self.masked_room_model = self._load_masked_room_model(self.masked_room_checkpoint)
        
        if enable_logging:
            logger.info(
                "Pipeline initialized successfully (components=%s)",
                self.component_status(),
            )

    @classmethod
    def from_components(
        cls,
        *,
        components: PipelineComponents,
        **kwargs,
    ) -> "NeuralSymbolicDungeonPipeline":
        """Construct a pipeline from an explicit dependency bundle."""
        return cls(components=components, **kwargs)

    @classmethod
    def create_symbolic_repair_pipeline(
        cls,
        *,
        device: str = 'cpu',
        use_learned_refiner_rules: bool = True,
        symbolic_max_repair_attempts: int = 5,
        symbolic_repair_margin: int = 2,
        symbolic_adjacency_threshold: float = 0.01,
        enable_map_elites: bool = False,
        map_elites_resolution: int = 20,
        enable_logging: bool = True,
        strict_checkpoint_mode: bool = False,
        stitcher: Optional[Any] = None,
        map_elites: Optional[MAPElitesEvaluator] = None,
    ) -> "NeuralSymbolicDungeonPipeline":
        """
        Create a lightweight pipeline for symbolic-only repair and stitching workflows.

        This intentionally avoids constructing the neural stack so tests/tools can
        exercise repair logic without loading VQ-VAE, diffusion, or LogicNet.
        """
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
                        tie_breaker='quality_score',
                        descriptor_mode='hybrid',
                    )
                    if enable_map_elites
                    else None
                )
            ),
        )
        return cls(
            device=device,
            use_learned_refiner_rules=use_learned_refiner_rules,
            map_elites_resolution=map_elites_resolution,
            enable_logging=enable_logging,
            strict_checkpoint_mode=strict_checkpoint_mode,
            symbolic_max_repair_attempts=symbolic_max_repair_attempts,
            symbolic_repair_margin=symbolic_repair_margin,
            symbolic_adjacency_threshold=symbolic_adjacency_threshold,
            components=PipelineComponents(symbolic=symbolic),
        )

    def _load_checkpoint_and_metadata(
        self,
        checkpoint_path: str,
        model_name: str,
        *,
        accepted_model_types: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[dict, dict]:
        """Load checkpoint and optional sidecar metadata for strict validation."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
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
        elif self.strict_checkpoint_mode:
            raise FileNotFoundError(
                f"Strict checkpoint mode enabled: metadata sidecar missing for {model_name} at {metadata_path}"
            )
        return checkpoint, metadata

    @staticmethod
    def _extract_checkpoint_config(checkpoint: Any) -> Dict[str, Any]:
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
            return dict(checkpoint["config"])
        return {}

    @staticmethod
    def _extract_checkpoint_state_dict(checkpoint: Any, *candidate_keys: str) -> Optional[Dict[str, Any]]:
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

    def _bump_diagnostic(self, key: str) -> None:
        """Increment a named runtime diagnostic counter."""
        k = str(key).strip().lower()
        if not k:
            return
        self.runtime_diagnostics[k] = int(self.runtime_diagnostics.get(k, 0)) + 1

    def _prepare_component(
        self,
        component: Optional[Any],
        *,
        eval_mode: bool = False,
    ) -> Optional[Any]:
        """Move module components to the pipeline device and switch them to eval mode."""
        if component is None:
            return None
        if isinstance(component, torch.nn.Module):
            component = component.to(self.device)
            if eval_mode:
                component.eval()
        return component

    def _bind_components(self, components: PipelineComponents) -> None:
        """Bind an injected component bundle to legacy pipeline attributes."""
        self.components = components

        self.vqvae = self._prepare_component(components.neural.vqvae, eval_mode=True)
        self.condition_encoder = self._prepare_component(components.neural.condition_encoder, eval_mode=True)
        self.diffusion = self._prepare_component(components.neural.diffusion, eval_mode=True)
        self.logic_net = self._prepare_component(components.neural.logic_net, eval_mode=True)
        self.refiner = components.symbolic.refiner
        self.stitcher = components.symbolic.stitcher
        self.map_elites = components.symbolic.map_elites

        self.components.neural.vqvae = self.vqvae
        self.components.neural.condition_encoder = self.condition_encoder
        self.components.neural.diffusion = self.diffusion
        self.components.neural.logic_net = self.logic_net
        self.components.symbolic.refiner = self.refiner
        self.components.symbolic.stitcher = self.stitcher
        self.components.symbolic.map_elites = self.map_elites

    def component_status(self) -> Dict[str, bool]:
        """Return which injectable components are currently configured."""
        return {
            'vqvae': self.vqvae is not None,
            'condition_encoder': self.condition_encoder is not None,
            'diffusion': self.diffusion is not None,
            'masked_room_model': getattr(self, 'masked_room_model', None) is not None,
            'logic_net': self.logic_net is not None,
            'refiner': self.refiner is not None,
            'stitcher': self.stitcher is not None,
            'map_elites': self.map_elites is not None,
        }

    def supports_room_generation(self) -> bool:
        """Whether the neural room-generation stack is configured."""
        if self.room_generator_mode == "discrete_masked":
            return (
                self.condition_encoder is not None
                and getattr(self, "masked_room_model", None) is not None
            )
        return (
            self.vqvae is not None
            and self.condition_encoder is not None
            and self.diffusion is not None
        )

    def supports_symbolic_repair(self) -> bool:
        """Whether symbolic room repair is available."""
        return self.refiner is not None

    def _require_component(self, component_name: str, operation: str) -> Any:
        """Return a configured component or raise a targeted capability error."""
        component = getattr(self, component_name, None)
        if component is None:
            raise MissingPipelineComponentError(
                f"{operation} requires pipeline component '{component_name}', "
                "but it is not configured. Inject it via PipelineComponents or "
                "use a pipeline constructor that provides the full stack."
            )
        return component

    def _require_room_generation_components(self, operation: str) -> None:
        """Ensure the core neural stack is available for room generation."""
        required = (
            ('condition_encoder', 'masked_room_model')
            if self.room_generator_mode == "discrete_masked"
            else ('vqvae', 'condition_encoder', 'diffusion')
        )
        missing = [name for name in required if getattr(self, name, None) is None]
        if missing:
            raise MissingPipelineComponentError(
                f"{operation} requires neural generation components {missing}, "
                "but this pipeline was initialized without them."
            )
    
    def _load_vqvae(self, checkpoint_path: Optional[str]) -> SemanticVQVAE:
        """Load or create VQ-VAE model."""
        use_coordconv = True
        checkpoint: Optional[Dict[str, Any]] = None
        state_dict: Optional[Dict[str, Any]] = None
        metadata: Dict[str, Any] = {}
        checkpoint_config: Dict[str, Any] = {}
        num_classes = int(np.max(self._valid_semantic_tile_ids_np)) + 1
        latent_dim = 64
        codebook_size = 512
        hidden_dim = 128
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, metadata = self._load_checkpoint_and_metadata(
                checkpoint_path,
                "vqvae",
                accepted_model_types=("diffusion",),
            )
            if isinstance(checkpoint, dict):
                checkpoint_config = self._extract_checkpoint_config(checkpoint)
                declared_model_type = str(metadata.get("model_type", "")).strip().lower()
                explicit_vq_state = checkpoint.get("vqvae_state_dict")
                is_composite_generation_checkpoint = any(
                    isinstance(checkpoint.get(key), dict)
                    for key in ("diffusion_state_dict", "condition_encoder_state_dict", "logic_net_state_dict")
                )
                if isinstance(explicit_vq_state, dict):
                    state_dict = explicit_vq_state
                elif declared_model_type not in {"diffusion"} and not is_composite_generation_checkpoint:
                    state_dict = self._extract_checkpoint_state_dict(checkpoint)
            architecture = metadata.get("architecture", {}) if isinstance(metadata, dict) else {}
            num_classes = int(
                checkpoint_config.get(
                    "num_classes",
                    architecture.get("num_classes", num_classes),
                )
            )
            latent_dim = int(checkpoint_config.get("latent_dim", latent_dim))
            latent_dim = int(architecture.get("latent_dim", latent_dim))
            codebook_size = int(checkpoint_config.get("codebook_size", codebook_size))
            codebook_size = int(architecture.get("codebook_size", codebook_size))
            use_coordconv = bool(checkpoint_config.get("use_coordconv", use_coordconv))
            use_coordconv = bool(architecture.get("use_coordconv", use_coordconv))
            # Backward compatibility: older checkpoints may use plain Conv2d
            # keys (encoder.conv_in.weight) while newer CoordConv checkpoints
            # use encoder.conv_in.conv.weight.
            if isinstance(state_dict, dict):
                has_coordconv_keys = ('encoder.conv_in.conv.weight' in state_dict)
                has_plain_conv_keys = ('encoder.conv_in.weight' in state_dict)
                if has_plain_conv_keys and not has_coordconv_keys:
                    use_coordconv = False
                conv_key = 'encoder.conv_in.conv.weight' if has_coordconv_keys else 'encoder.conv_in.weight'
                conv_weight = state_dict.get(conv_key)
                if isinstance(conv_weight, torch.Tensor) and conv_weight.dim() == 4:
                    hidden_dim = int(max(1, int(conv_weight.shape[0])))

        model = SemanticVQVAE(
            num_classes=num_classes,
            codebook_size=codebook_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_coordconv=use_coordconv,
        ).to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            if checkpoint is None:
                checkpoint, _metadata = self._load_checkpoint_and_metadata(
                    checkpoint_path,
                    "vqvae",
                    accepted_model_types=("diffusion",),
                )
            if state_dict is None and isinstance(checkpoint, dict):
                declared_model_type = str(metadata.get("model_type", "")).strip().lower()
                explicit_vq_state = checkpoint.get("vqvae_state_dict")
                is_composite_generation_checkpoint = any(
                    isinstance(checkpoint.get(key), dict)
                    for key in ("diffusion_state_dict", "condition_encoder_state_dict", "logic_net_state_dict")
                )
                if isinstance(explicit_vq_state, dict):
                    state_dict = explicit_vq_state
                elif declared_model_type not in {"diffusion"} and not is_composite_generation_checkpoint:
                    state_dict = self._extract_checkpoint_state_dict(checkpoint)
            if isinstance(state_dict, dict):
                incompatible = model.load_state_dict(state_dict, strict=False)
                missing = [str(k) for k in getattr(incompatible, 'missing_keys', [])]
                unexpected = [str(k) for k in getattr(incompatible, 'unexpected_keys', [])]

                # Legacy checkpoints created before explicit legality buffer registration.
                allowed_missing = {"illegal_adjacency_matrix"}
                unexpected_missing = [k for k in missing if k not in allowed_missing]

                if unexpected_missing or unexpected:
                    logger.warning(
                        "VQ-VAE checkpoint key mismatch. missing=%s unexpected=%s",
                        unexpected_missing,
                        unexpected,
                    )
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded VQ-VAE from {checkpoint_path}")
        else:
            if self.strict_checkpoint_mode:
                raise FileNotFoundError(
                    f"Strict checkpoint mode enabled: missing VQ-VAE checkpoint at {checkpoint_path!r}"
                )
            logger.warning("No VQ-VAE checkpoint provided, using random initialization")
        
        return model
    
    def _load_condition_encoder(
        self, 
        checkpoint_path: Optional[str]
    ) -> DualStreamConditionEncoder:
        """
        Load or create condition encoder.

        Best-practice behavior:
        - default to richer graph-conditioning schema for fresh training,
        - auto-infer legacy schema from checkpoint weights for compatibility.
        """
        default_node_feature_dim = int(GRAPH_NODE_FEATURE_DIM)
        default_edge_feature_dim = int(GRAPH_EDGE_FEATURE_DIM)
        node_feature_dim = int(default_node_feature_dim)
        edge_feature_dim = int(default_edge_feature_dim)
        checkpoint_state: Optional[Dict[str, Any]] = None
        checkpoint_config: Dict[str, Any] = {}
        fallback_config = dict(self.condition_encoder_fallback_config)
        default_latent_dim = int(
            getattr(getattr(self, "vqvae", None), "latent_dim", 64)
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, _metadata = self._load_checkpoint_and_metadata(
                checkpoint_path,
                "condition_encoder",
                accepted_model_types=("diffusion", "masked_room_model"),
            )
            checkpoint_state = self._extract_checkpoint_state_dict(
                checkpoint,
                "condition_encoder_state_dict",
            )
            checkpoint_config = self._extract_checkpoint_config(checkpoint)
            if isinstance(checkpoint_state, dict):
                node_weight = checkpoint_state.get('global_encoder.node_encoder.weight')
                edge_weight = checkpoint_state.get('global_encoder.edge_encoder.weight')
                if isinstance(node_weight, torch.Tensor) and node_weight.dim() == 2:
                    node_feature_dim = int(max(1, int(node_weight.shape[1])))
                if isinstance(edge_weight, torch.Tensor) and edge_weight.dim() == 2:
                    edge_feature_dim = int(max(1, int(edge_weight.shape[1])))

        model = DualStreamConditionEncoder(
            latent_dim=int(checkpoint_config.get("latent_dim", fallback_config.get("latent_dim", default_latent_dim))),
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=int(checkpoint_config.get("condition_hidden_dim", fallback_config.get("condition_hidden_dim", 256))),
            output_dim=int(checkpoint_config.get("context_dim", fallback_config.get("context_dim", 256))),
            gnn_type=str(checkpoint_config.get("condition_gnn_type", fallback_config.get("condition_gnn_type", self.condition_gnn_type))),
            num_gnn_layers=int(checkpoint_config.get("condition_num_gnn_layers", fallback_config.get("condition_num_gnn_layers", 3))),
            num_attention_heads=int(checkpoint_config.get("condition_num_attention_heads", fallback_config.get("condition_num_attention_heads", 8))),
            dropout=float(checkpoint_config.get("condition_dropout", fallback_config.get("condition_dropout", 0.1))),
            use_current_node_distance_features=bool(
                checkpoint_config.get(
                    "use_current_node_distance_features",
                    fallback_config.get("use_current_node_distance_features", self.use_current_node_distance_features),
                )
            ),
            use_reference_room_maps=bool(
                checkpoint_config.get(
                    "condition_use_reference_room_maps",
                    fallback_config.get("condition_use_reference_room_maps", self.condition_use_reference_room_maps),
                )
            ),
            reference_num_tile_types=int(
                checkpoint_config.get(
                    "condition_reference_tile_vocab_size",
                    fallback_config.get("condition_reference_tile_vocab_size", self.condition_reference_tile_vocab_size),
                )
            ),
            reference_embedding_dim=int(
                checkpoint_config.get(
                    "condition_reference_embedding_dim",
                    fallback_config.get("condition_reference_embedding_dim", self.condition_reference_embedding_dim),
                )
            ),
            reference_hidden_dim=int(
                checkpoint_config.get(
                    "condition_reference_hidden_dim",
                    fallback_config.get("condition_reference_hidden_dim", self.condition_reference_hidden_dim),
                )
            ),
        ).to(self.device)
        
        if checkpoint_state is not None:
            if isinstance(checkpoint_state, dict):
                model_state = model.state_dict()
                filtered_state: Dict[str, Any] = {}
                dropped_shape: List[str] = []
                for k, v in checkpoint_state.items():
                    if k not in model_state:
                        continue
                    target_v = model_state[k]
                    if hasattr(v, 'shape') and hasattr(target_v, 'shape') and tuple(v.shape) != tuple(target_v.shape):
                        dropped_shape.append(k)
                        continue
                    filtered_state[k] = v
                checkpoint_state = filtered_state
            incompatible = model.load_state_dict(checkpoint_state, strict=False)
            missing = list(getattr(incompatible, 'missing_keys', []))
            unexpected = list(getattr(incompatible, 'unexpected_keys', []))
            logger.info(
                "Loaded condition encoder from %s (node_dim=%d edge_dim=%d, missing=%d unexpected=%d)",
                checkpoint_path,
                node_feature_dim,
                edge_feature_dim,
                len(missing),
                len(unexpected),
            )
            if 'dropped_shape' in locals() and dropped_shape:
                logger.warning(
                    "Condition encoder dropped incompatible checkpoint keys (shape mismatch): %s",
                    summarize_missing_keys(dropped_shape),
                )
            if missing or unexpected:
                msg = (
                    "Condition encoder checkpoint/schema mismatch: "
                    f"missing={summarize_missing_keys(missing)} unexpected={summarize_missing_keys(unexpected)}"
                )
                if self.strict_checkpoint_mode:
                    raise RuntimeError(msg)
                logger.warning(msg)
        else:
            if self.strict_checkpoint_mode:
                raise FileNotFoundError(
                    f"Strict checkpoint mode enabled: missing condition encoder checkpoint at {checkpoint_path!r}"
                )
            logger.warning(
                "No condition encoder checkpoint, using random initialization with enhanced schema (node_dim=%d edge_dim=%d)",
                node_feature_dim,
                edge_feature_dim,
            )
        
        return model

    def _condition_feature_dims(self) -> Tuple[int, int]:
        """Get active (node_dim, edge_dim) expected by the condition encoder."""
        encoder = self._require_component("condition_encoder", "_condition_feature_dims")
        return condition_feature_dims(encoder)

    @staticmethod
    def _fit_feature_vector(values: List[float], target_dim: int) -> List[float]:
        """Pad/truncate feature list to target dimension."""
        return fit_feature_vector(values, target_dim)
    
    def _load_diffusion(self, checkpoint_path: Optional[str]) -> LatentDiffusionModel:
        """Load or create latent diffusion model."""
        checkpoint_config: Dict[str, Any] = {}
        checkpoint_state: Optional[Dict[str, Any]] = None
        fallback_config = dict(self.diffusion_fallback_config)
        default_latent_dim = int(
            checkpoint_config.get(
                "latent_dim",
                fallback_config.get("latent_dim", getattr(getattr(self, "vqvae", None), "latent_dim", 64)),
            )
        )
        default_context_dim = int(
            checkpoint_config.get(
                "context_dim",
                fallback_config.get("context_dim", getattr(getattr(self, "condition_encoder", None), "output_dim", 256)),
            )
        )
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, _metadata = self._load_checkpoint_and_metadata(checkpoint_path, "diffusion")
            checkpoint_config = self._extract_checkpoint_config(checkpoint)
            checkpoint_state_key = "ema_diffusion_state_dict"
            checkpoint_state = self._extract_checkpoint_state_dict(
                checkpoint,
                "ema_diffusion_state_dict",
                "diffusion_state_dict",
            )
            if checkpoint_state is None:
                checkpoint_state_key = "diffusion_state_dict"
            elif not isinstance(checkpoint.get("ema_diffusion_state_dict"), dict):
                checkpoint_state_key = "diffusion_state_dict"
            default_latent_dim = int(
                checkpoint_config.get(
                    "latent_dim",
                    getattr(getattr(self, "vqvae", None), "latent_dim", default_latent_dim),
                )
            )
            default_context_dim = int(
                checkpoint_config.get(
                    "context_dim",
                    getattr(getattr(self, "condition_encoder", None), "output_dim", default_context_dim),
                )
            )
        model = LatentDiffusionModel(
            latent_dim=default_latent_dim,
            context_dim=default_context_dim,
            num_timesteps=int(checkpoint_config.get("num_timesteps", fallback_config.get("num_timesteps", 1000))),
            prediction_type=str(checkpoint_config.get("prediction_type", fallback_config.get("prediction_type", "epsilon"))),
            cfg_dropout_prob=float(checkpoint_config.get("cfg_dropout_prob", fallback_config.get("cfg_dropout_prob", 0.1))),
            cfg_scale=float(checkpoint_config.get("cfg_scale", fallback_config.get("cfg_scale", 3.0))),
            cfg_schedule_mode=str(checkpoint_config.get("cfg_schedule_mode", self.diffusion_cfg_schedule_mode)),
            cfg_schedule_min_scale=float(checkpoint_config.get("cfg_schedule_min_scale", self.diffusion_cfg_schedule_min_scale)),
            cfg_schedule_power=float(checkpoint_config.get("cfg_schedule_power", self.diffusion_cfg_schedule_power)),
            min_snr_gamma=float(checkpoint_config.get("min_snr_gamma", fallback_config.get("min_snr_gamma", 5.0))),
            model_channels=int(checkpoint_config.get("model_channels", fallback_config.get("model_channels", 128))),
            topology_refinement_mode=str(checkpoint_config.get("topology_refinement_mode", self.topology_refinement_mode)),
            attention_mode=str(checkpoint_config.get("attention_mode", self.diffusion_attention_mode)),
            topology_conditioning_mode=str(
                checkpoint_config.get("topology_conditioning_mode", fallback_config.get("topology_conditioning_mode", "additive"))
            ),
            hedgehog_feature_dim=int(checkpoint_config.get("hedgehog_feature_dim", self.diffusion_hedgehog_feature_dim)),
            unet_channel_mult=tuple(checkpoint_config.get("unet_channel_mult", fallback_config.get("unet_channel_mult", (1, 2, 4)))),
            unet_num_res_blocks=int(checkpoint_config.get("unet_num_res_blocks", fallback_config.get("unet_num_res_blocks", 2))),
            unet_attention_resolutions=tuple(
                checkpoint_config.get("unet_attention_resolutions", fallback_config.get("unet_attention_resolutions", (1, 2)))
            ),
            unet_num_heads=int(checkpoint_config.get("unet_num_heads", fallback_config.get("unet_num_heads", 8))),
            unet_dropout=float(checkpoint_config.get("unet_dropout", fallback_config.get("unet_dropout", 0.1))),
            graph_auto_linear_attention_nodes=int(
                checkpoint_config.get(
                    "graph_auto_linear_attention_nodes",
                    fallback_config.get("graph_auto_linear_attention_nodes", 128),
                )
            ),
            spatial_graph_gate_init=float(
                checkpoint_config.get("spatial_graph_gate_init", fallback_config.get("spatial_graph_gate_init", -2.0))
            ),
            spatial_topology_gate_init=float(
                checkpoint_config.get("spatial_topology_gate_init", fallback_config.get("spatial_topology_gate_init", -2.0))
            ),
            room_topology_channels=int(
                checkpoint_config.get("room_topology_channels", fallback_config.get("room_topology_channels", ROOM_TOPOLOGY_CHANNEL_COUNT))
            ),
        ).to(self.device)
        setattr(
            model,
            "training_cfg_scale",
            float(checkpoint_config.get("cfg_scale", fallback_config.get("cfg_scale", 3.0))),
        )
        setattr(
            model,
            "inference_checkpoint_state_key",
            str(locals().get("checkpoint_state_key", "random_init")),
        )
        self.diffusion_puzzle_structure_condition_enabled = bool(
            float(checkpoint_config.get("puzzle_structure_dropout_prob", fallback_config.get("puzzle_structure_dropout_prob", 0.0))) > 0.0
        )
        
        if checkpoint_path and Path(checkpoint_path).exists():
            if not isinstance(checkpoint_state, dict):
                raise ValueError(
                    f"Diffusion checkpoint at {checkpoint_path!r} does not contain a loadable state_dict."
                )
            if isinstance(checkpoint_state, dict):
                model_state = model.state_dict()
                filtered_state: Dict[str, Any] = {}
                dropped_shape: List[str] = []
                for k, v in checkpoint_state.items():
                    if k not in model_state:
                        continue
                    target_v = model_state[k]
                    if hasattr(v, 'shape') and hasattr(target_v, 'shape') and tuple(v.shape) != tuple(target_v.shape):
                        dropped_shape.append(k)
                        continue
                    filtered_state[k] = v
                checkpoint_state = filtered_state
            incompatible = model.load_state_dict(checkpoint_state, strict=False)
            missing = list(getattr(incompatible, 'missing_keys', []))
            unexpected = list(getattr(incompatible, 'unexpected_keys', []))
            logger.info(
                "Loaded diffusion model from %s using %s (missing=%d unexpected=%d)",
                checkpoint_path,
                getattr(model, "inference_checkpoint_state_key", "diffusion_state_dict"),
                len(missing),
                len(unexpected),
            )
            if 'dropped_shape' in locals() and dropped_shape:
                logger.warning(
                    "Diffusion dropped incompatible checkpoint keys (shape mismatch): %s",
                    summarize_missing_keys(dropped_shape),
                )
            if missing or unexpected:
                msg = (
                    "Diffusion checkpoint/schema mismatch: "
                    f"missing={summarize_missing_keys(missing)} unexpected={summarize_missing_keys(unexpected)}"
                )
                if self.strict_checkpoint_mode:
                    raise RuntimeError(msg)
                logger.warning(msg)
        else:
            if self.strict_checkpoint_mode:
                raise FileNotFoundError(
                    f"Strict checkpoint mode enabled: missing diffusion checkpoint at {checkpoint_path!r}"
                )
            logger.warning("No diffusion checkpoint, using random initialization")

        if self.fast_sampling_checkpoint:
            fast_ckpt_path = Path(self.fast_sampling_checkpoint)
            if fast_ckpt_path.exists():
                try:
                    model.enable_fast_sampling(
                        adapter_checkpoint=str(fast_ckpt_path),
                        num_inference_steps=int(self.fast_sampling_steps),
                        use_fp16=(self.device.type == "cuda"),
                        compile_model=False,
                        strict=self.strict_checkpoint_mode,
                    )
                    logger.info(
                        "Enabled distilled fast sampling from %s (%d steps).",
                        fast_ckpt_path,
                        int(self.fast_sampling_steps),
                    )
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    if self.strict_checkpoint_mode:
                        raise
                    logger.warning(
                        "Fast-sampling checkpoint rejected; using standard diffusion sampling: %s",
                        exc,
                    )
            elif self.strict_checkpoint_mode:
                raise FileNotFoundError(
                    f"Strict checkpoint mode enabled: missing fast-sampling checkpoint at {fast_ckpt_path}"
                )
            else:
                logger.warning(
                    "Fast-sampling checkpoint not found at %s; using standard diffusion sampling.",
                    fast_ckpt_path,
                )
        
        return model
    
    def _load_logic_net(self, checkpoint_path: Optional[str]) -> LogicNet:
        """Load or create LogicNet."""
        checkpoint_state: Optional[Dict[str, Any]] = None
        checkpoint_config: Dict[str, Any] = {}
        fallback_config = dict(self.logic_net_fallback_config)
        default_latent_dim = int(
            getattr(
                getattr(self, "diffusion", None),
                "latent_dim",
                fallback_config.get("latent_dim", getattr(getattr(self, "vqvae", None), "latent_dim", 64)),
            )
        )
        default_num_classes = int(
            fallback_config.get("num_classes", getattr(getattr(self, "vqvae", None), "num_classes", 44))
        )
        model = LogicNet(
            latent_dim=default_latent_dim,
            num_classes=default_num_classes,
            num_iterations=int(fallback_config.get("num_logic_iterations", 20)),
            topology_trace_weight=float(fallback_config.get("logic_topology_trace_weight", 0.25)),
            topology_anchor_weight=float(fallback_config.get("logic_topology_anchor_weight", 0.25)),
            global_reach_weight=float(fallback_config.get("logic_global_reach_weight", 1.0)),
            global_room_weight=float(fallback_config.get("logic_global_room_weight", 0.25)),
        ).to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, metadata = self._load_checkpoint_and_metadata(
                checkpoint_path,
                "logic_net",
                accepted_model_types=("diffusion",),
            )
            checkpoint_state = self._extract_checkpoint_state_dict(
                checkpoint,
                "logic_net_state_dict",
            )
            checkpoint_config = self._extract_checkpoint_config(checkpoint)
            architecture = metadata.get("architecture", {}) if isinstance(metadata, dict) else {}
            model = LogicNet(
                latent_dim=int(checkpoint_config.get("latent_dim", architecture.get("latent_dim", default_latent_dim))),
                num_classes=int(checkpoint_config.get("num_classes", architecture.get("num_classes", default_num_classes))),
                num_iterations=int(checkpoint_config.get("num_logic_iterations", 20)),
                topology_trace_weight=float(checkpoint_config.get("logic_topology_trace_weight", 0.25)),
                topology_anchor_weight=float(checkpoint_config.get("logic_topology_anchor_weight", 0.25)),
                global_reach_weight=float(checkpoint_config.get("logic_global_reach_weight", 1.0)),
                global_room_weight=float(checkpoint_config.get("logic_global_room_weight", 0.25)),
            ).to(self.device)
            if isinstance(checkpoint_state, dict):
                model.load_state_dict(checkpoint_state)
            else:
                raise ValueError(
                    f"LogicNet checkpoint at {checkpoint_path!r} does not contain a loadable state_dict."
                )
            logger.info(f"Loaded LogicNet from {checkpoint_path}")
        else:
            if self.strict_checkpoint_mode:
                raise FileNotFoundError(
                    f"Strict checkpoint mode enabled: missing LogicNet checkpoint at {checkpoint_path!r}"
                )
            logger.warning("No LogicNet checkpoint, using random initialization")
        
        return model

    def _load_masked_room_model(
        self,
        checkpoint_path: Optional[str],
    ) -> Optional[DiscreteMaskedRoomModel]:
        """Load or create the optional discrete masked room model."""
        if checkpoint_path is None and self.room_generator_mode != "discrete_masked":
            return None

        from src.core.discrete_masked_model import create_discrete_masked_model

        checkpoint_config: Dict[str, Any] = {}
        checkpoint_state: Optional[Dict[str, Any]] = None
        checkpoint: Optional[Dict[str, Any]] = None
        fallback_config = dict(self.masked_room_fallback_config)
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, _metadata = self._load_checkpoint_and_metadata(checkpoint_path, "masked_room_model")
            checkpoint_config = self._extract_checkpoint_config(checkpoint)
            checkpoint_state = self._extract_checkpoint_state_dict(checkpoint)

        model = create_discrete_masked_model(
            num_classes=int(
                checkpoint_config.get(
                    "num_classes",
                    fallback_config.get("num_classes", getattr(getattr(self, "vqvae", None), "num_classes", 44)),
                )
            ),
            hidden_dim=int(checkpoint_config.get("hidden_dim", fallback_config.get("hidden_dim", 48))),
            model_channels=int(checkpoint_config.get("model_channels", fallback_config.get("model_channels", 64))),
            context_dim=int(
                checkpoint_config.get(
                    "context_dim",
                    fallback_config.get("context_dim", getattr(getattr(self, "condition_encoder", None), "output_dim", 256)),
                )
            ),
            num_steps=int(checkpoint_config.get("masked_steps", self.masked_sampling_steps)),
            attention_mode=str(checkpoint_config.get("attention_mode", self.diffusion_attention_mode)),
            topology_conditioning_mode=str(
                checkpoint_config.get("topology_conditioning_mode", fallback_config.get("topology_conditioning_mode", "additive"))
            ),
            hedgehog_feature_dim=int(checkpoint_config.get("hedgehog_feature_dim", self.diffusion_hedgehog_feature_dim)),
            graph_auto_linear_attention_nodes=int(
                checkpoint_config.get(
                    "graph_auto_linear_attention_nodes",
                    fallback_config.get("graph_auto_linear_attention_nodes", 128),
                )
            ),
            spatial_graph_gate_init=float(
                checkpoint_config.get("spatial_graph_gate_init", fallback_config.get("spatial_graph_gate_init", -2.0))
            ),
            spatial_topology_gate_init=float(
                checkpoint_config.get("spatial_topology_gate_init", fallback_config.get("spatial_topology_gate_init", -2.0))
            ),
            unet_channel_mult=tuple(checkpoint_config.get("unet_channel_mult", fallback_config.get("unet_channel_mult", (1, 2)))),
            unet_num_res_blocks=int(checkpoint_config.get("unet_num_res_blocks", fallback_config.get("unet_num_res_blocks", 1))),
            unet_attention_resolutions=tuple(
                checkpoint_config.get("unet_attention_resolutions", fallback_config.get("unet_attention_resolutions", (0, 1)))
            ),
            unet_num_heads=int(checkpoint_config.get("unet_num_heads", fallback_config.get("unet_num_heads", 4))),
            unet_dropout=float(checkpoint_config.get("unet_dropout", fallback_config.get("unet_dropout", 0.1))),
            room_topology_channels=int(
                checkpoint_config.get("room_topology_channels", fallback_config.get("room_topology_channels", ROOM_TOPOLOGY_CHANNEL_COUNT))
            ),
        ).to(self.device)
        self.masked_room_puzzle_structure_condition_enabled = bool(
            float(checkpoint_config.get("puzzle_structure_dropout_prob", fallback_config.get("puzzle_structure_dropout_prob", 0.0))) > 0.0
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            assert checkpoint is not None
            state_dict = checkpoint_state
            if not isinstance(state_dict, dict):
                raise ValueError(
                    f"Masked-room checkpoint at {checkpoint_path!r} does not contain a loadable state_dict."
                )
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if missing or unexpected:
                msg = (
                    "Masked-room checkpoint/schema mismatch: "
                    f"missing={summarize_missing_keys(missing)} unexpected={summarize_missing_keys(unexpected)}"
                )
                if self.strict_checkpoint_mode:
                    raise RuntimeError(msg)
                logger.warning(msg)

            cond_state = checkpoint.get("condition_encoder_state_dict")
            if cond_state is not None and self.condition_encoder is not None:
                incompatible_cond = self.condition_encoder.load_state_dict(cond_state, strict=False)
                cond_missing = list(getattr(incompatible_cond, "missing_keys", []))
                cond_unexpected = list(getattr(incompatible_cond, "unexpected_keys", []))
                if cond_missing or cond_unexpected:
                    msg = (
                        "Masked-room checkpoint bundled condition encoder mismatch: "
                        f"missing={summarize_missing_keys(cond_missing)} unexpected={summarize_missing_keys(cond_unexpected)}"
                    )
                    if self.strict_checkpoint_mode:
                        raise RuntimeError(msg)
                    logger.warning(msg)
            logger.info("Loaded discrete masked room model from %s", checkpoint_path)
        else:
            if self.room_generator_mode == "discrete_masked" and self.strict_checkpoint_mode:
                raise FileNotFoundError(
                    f"Strict checkpoint mode enabled: missing discrete masked room checkpoint at {checkpoint_path!r}"
                )
            if self.room_generator_mode == "discrete_masked":
                logger.warning("No discrete masked room checkpoint provided, using random initialization")

        return model
    
    @staticmethod
    def _create_refiner(
        use_learned_rules: bool,
        *,
        max_repair_attempts: int = 5,
        margin: int = 2,
        adjacency_threshold: float = 0.01,
    ) -> SymbolicRefiner:
        """Create symbolic refiner with optional learned rules."""
        learned_stats = LearnedTileStatistics() if use_learned_rules else None

        # Align refiner vocabulary with project semantic palette.
        # Legacy SymbolicRefiner defaults include tile id 50, which is outside
        # this project's canonical semantic range and can leak invalid ids.
        valid_ids = set(int(v) for v in SEMANTIC_PALETTE.values())
        enemy_id = int(SEMANTIC_PALETTE.get("ENEMY", 20))
        legacy_to_canonical = {
            50: enemy_id,
        }
        canonical_adjacency: Dict[int, Set[int]] = {}
        for raw_src, raw_neighbors in dict(DEFAULT_ADJACENCY).items():
            src = int(legacy_to_canonical.get(int(raw_src), int(raw_src)))
            if src not in valid_ids:
                continue
            neigh_set: Set[int] = set()
            for raw_dst in set(raw_neighbors):
                dst = int(legacy_to_canonical.get(int(raw_dst), int(raw_dst)))
                if dst in valid_ids:
                    neigh_set.add(dst)
            if not neigh_set:
                neigh_set.add(src)
            neigh_set.add(src)
            canonical_adjacency[src] = neigh_set

        tile_types = sorted(int(t) for t in canonical_adjacency.keys())
        
        refiner = SymbolicRefiner(
            tile_types=tile_types,
            adjacency=(None if use_learned_rules else canonical_adjacency),
            learned_stats=learned_stats,
            max_repair_attempts=int(max(1, int(max_repair_attempts))),
            margin=int(max(0, int(margin))),
            adjacency_threshold=float(max(0.0, float(adjacency_threshold))),
        )
        
        logger.info(f"Created SymbolicRefiner (learned_rules={use_learned_rules})")
        return refiner

    def _sanitize_semantic_grid(
        self,
        grid: np.ndarray,
        *,
        fallback_grid: Optional[np.ndarray] = None,
        strip_void: bool = False,
    ) -> Tuple[np.ndarray, int, List[int]]:
        """
        Clamp/repair semantic IDs to the canonical palette.

        Invalid tile IDs are replaced with fallback_grid values when available;
        otherwise they are replaced with FLOOR.
        """
        out = np.asarray(grid, dtype=np.int32).copy()
        invalid_mask = ~np.isin(out, self._valid_semantic_tile_ids_np)
        if bool(strip_void):
            invalid_mask |= out == int(TileID.VOID)
        invalid_count = int(np.sum(invalid_mask))
        invalid_ids: List[int] = []
        if invalid_count <= 0:
            return out, 0, invalid_ids

        invalid_ids = [int(v) for v in np.unique(out[invalid_mask])]
        floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
        if fallback_grid is not None and np.shape(fallback_grid) == np.shape(out):
            fb = np.asarray(fallback_grid, dtype=np.int32)
            fb_invalid = ~np.isin(fb, self._valid_semantic_tile_ids_np)
            if bool(strip_void):
                fb_invalid |= fb == int(TileID.VOID)
            if bool(np.any(fb_invalid)):
                fb = fb.copy()
                fb[fb_invalid] = floor_id
            out[invalid_mask] = fb[invalid_mask]
        else:
            out[invalid_mask] = floor_id
        return out, invalid_count, invalid_ids

    def _strip_room_void_tiles(
        self,
        grid: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Remove VOID tiles from a generated room grid.

        Room-local semantic tensors should never retain stitched-world VOID.
        If VOID survives decoding or repair, exported rooms can show black
        holes inside otherwise valid geometry. Boundary VOID becomes WALL;
        interior VOID becomes FLOOR.
        """
        out = np.asarray(grid, dtype=np.int32).copy()
        void_mask = out == int(TileID.VOID)
        total_void = int(np.sum(void_mask))
        if total_void <= 0:
            return out, {
                "boundary_void_tiles_removed": 0,
                "interior_void_tiles_removed": 0,
            }

        boundary_mask = np.zeros_like(out, dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[ROOM_HEIGHT - 1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, ROOM_WIDTH - 1] = True

        boundary_void_mask = void_mask & boundary_mask
        interior_void_mask = void_mask & ~boundary_mask
        if bool(np.any(boundary_void_mask)):
            out[boundary_void_mask] = int(TileID.WALL)
        if bool(np.any(interior_void_mask)):
            out[interior_void_mask] = int(TileID.FLOOR)
        return out, {
            "boundary_void_tiles_removed": int(np.sum(boundary_void_mask)),
            "interior_void_tiles_removed": int(np.sum(interior_void_mask)),
        }

    def _strip_volatile_room_semantics(
        self,
        grid: np.ndarray,
        *,
        graph: Optional[nx.Graph] = None,
        room_id: Any = None,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        salvage_graph_markers: bool = True,
        salvage_max_manhattan_distance: int = 2,
    ) -> Tuple[np.ndarray, int, List[int], int, List[int]]:
        """
        Remove volatile gameplay semantics from a generated room.

        The neural room generator is responsible for layout/geometry. Graph-owned
        semantics such as keys, enemies, stairs, and puzzle markers are placed
        deterministically after room structure is finalized.
        """
        out = np.asarray(grid, dtype=np.int32).copy()
        preserved_count = 0
        preserved_ids: List[int] = []
        keep_mask = np.zeros_like(out, dtype=bool)
        if bool(salvage_graph_markers) and isinstance(graph, nx.Graph) and room_id in graph:
            structural_view = out.copy()
            structural_view[np.isin(structural_view, self._volatile_room_semantic_tile_ids_np)] = int(
                SEMANTIC_PALETTE.get("FLOOR", 1)
            )
            planned = self._plan_room_graph_marker_layout(
                structural_view,
                graph=graph,
                room_id=room_id,
                start_goal=start_goal,
            )
            max_distance = int(max(0, salvage_max_manhattan_distance))
            for tile_id, slot in planned:
                hits = np.argwhere(out == int(tile_id))
                if hits.size <= 0:
                    continue
                distances = np.abs(hits[:, 0] - int(slot[0])) + np.abs(hits[:, 1] - int(slot[1]))
                best_idx = int(np.argmin(distances))
                best_distance = int(distances[best_idx])
                if best_distance > max_distance:
                    continue
                best_row = int(hits[best_idx][0])
                best_col = int(hits[best_idx][1])
                if (best_row, best_col) != (int(slot[0]), int(slot[1])):
                    out[best_row, best_col] = int(SEMANTIC_PALETTE.get("FLOOR", 1))
                out[int(slot[0]), int(slot[1])] = int(tile_id)
                keep_mask[int(slot[0]), int(slot[1])] = True
                preserved_count += 1
                preserved_ids.append(int(tile_id))

        volatile_mask = np.isin(out, self._volatile_room_semantic_tile_ids_np) & ~keep_mask
        volatile_count = int(np.sum(volatile_mask))
        if volatile_count <= 0:
            return out, 0, [], preserved_count, preserved_ids

        volatile_ids = [int(v) for v in np.unique(out[volatile_mask])]
        out[volatile_mask] = int(SEMANTIC_PALETTE.get("FLOOR", 1))
        return out, volatile_count, volatile_ids, preserved_count, preserved_ids

    def _apply_semantic_constrained_decoding(
        self,
        logits: torch.Tensor,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Dict[str, int]:
        """
        Bias decode logits toward graph-owned semantic markers before argmax.

        This keeps semantic alignment inside the learned decode path instead of
        relying exclusively on post-hoc deterministic overlay. The bias is
        intentionally softer than hard door forcing: graph markers are nudged
        toward planned slots while stray volatile semantic channels are
        suppressed elsewhere.
        """
        if not bool(self.default_semantic_constrained_decoding_enabled):
            return {"planned_markers": 0, "biased_slots": 0}
        if not isinstance(graph, nx.Graph) or room_id not in graph:
            return {"planned_markers": 0, "biased_slots": 0}
        if not isinstance(logits, torch.Tensor) or logits.dim() != 4 or int(logits.shape[0]) != 1:
            return {"planned_markers": 0, "biased_slots": 0}

        try:
            preview_grid = logits.argmax(dim=1).detach().cpu().numpy()[0]
            preview_grid, _, _ = self._sanitize_semantic_grid(preview_grid, strip_void=True)
            preview_grid[np.isin(preview_grid, self._volatile_room_semantic_tile_ids_np)] = int(
                SEMANTIC_PALETTE.get("FLOOR", 1)
            )
            planned = self._plan_room_graph_marker_layout(
                preview_grid,
                graph=graph,
                room_id=room_id,
                start_goal=start_goal,
            )
            if not planned:
                return {"planned_markers": 0, "biased_slots": 0}

            suppression_bias = float(self.default_semantic_marker_suppression_bias)
            positive_bias = float(self.default_semantic_marker_logit_bias)
            marker_channels = [
                int(tile_id)
                for tile_id in self._volatile_room_semantic_tile_ids_np.tolist()
                if 0 <= int(tile_id) < int(logits.shape[1])
            ]
            if suppression_bias > 0.0 and marker_channels:
                logits[0, marker_channels, :, :] = logits[0, marker_channels, :, :] - suppression_bias

            biased_slots = 0
            if positive_bias > 0.0:
                for tile_id, slot in planned:
                    tile_index = int(tile_id)
                    if tile_index < 0 or tile_index >= int(logits.shape[1]):
                        continue
                    row, col = self._clamp_room_coord(slot)
                    logits[0, tile_index, row, col] = logits[0, tile_index, row, col] + positive_bias
                    biased_slots += 1

            return {"planned_markers": int(len(planned)), "biased_slots": int(biased_slots)}
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Semantic constrained decoding skipped for room %s: %s", room_id, exc)
            return {"planned_markers": 0, "biased_slots": 0}

    @staticmethod
    def _all_room_door_slots_mask() -> np.ndarray:
        """Return a boolean mask covering every canonical doorway slot in a room."""
        mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
        for direction, spec in DOOR_POSITIONS.items():
            if direction in {"N", "S"}:
                row = int(spec["row"])
                c0 = int(spec["col_start"])
                c1 = int(spec["col_end"]) + 1
                mask[row, c0:c1] = True
            else:
                col = int(spec["col"])
                r0 = int(spec["row_start"])
                r1 = int(spec["row_end"]) + 1
                mask[r0:r1, col] = True
        return mask

    def _required_room_door_slots_mask(
        self,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
    ) -> np.ndarray:
        """
        Return a mask of boundary door slots that are legal for this room.

        When graph metadata is unavailable, fall back to all canonical doorway
        slots so standalone room generation does not over-strip doors.
        """
        if not isinstance(graph, nx.Graph) or room_id not in graph:
            return self._all_room_door_slots_mask()

        semantics = self._extract_room_topology_semantics(graph, room_id)
        mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
        for direction, enabled in semantics["required_doors"].items():
            if not bool(enabled):
                continue
            spec = DOOR_POSITIONS[str(direction)]
            if direction in {"N", "S"}:
                row = int(spec["row"])
                c0 = int(spec["col_start"])
                c1 = int(spec["col_end"]) + 1
                mask[row, c0:c1] = True
            else:
                col = int(spec["col"])
                r0 = int(spec["row_start"])
                r1 = int(spec["row_end"]) + 1
                mask[r0:r1, col] = True
        return mask

    def _enforce_room_boundary_shell(
        self,
        grid: np.ndarray,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Seal room boundaries deterministically and reopen only valid doorway slots.

        Neural generation and symbolic repair can both leave boundary floor leaks,
        which later read as "rooms" expanding into stitched void. The exported
        contract should be simple: boundary tiles are walls unless the graph
        explicitly requires a doorway there.
        """
        out = np.asarray(grid, dtype=np.int32).copy()
        before = out.copy()
        wall_id = int(TileID.WALL)
        floor_id = int(TileID.FLOOR)
        door_tile_values = np.array(
            [
                int(TileID.DOOR_OPEN),
                int(TileID.DOOR_LOCKED),
                int(TileID.DOOR_BOMB),
                int(TileID.DOOR_PUZZLE),
                int(TileID.DOOR_BOSS),
                int(TileID.DOOR_SOFT),
            ],
            dtype=np.int32,
        )

        boundary_mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[ROOM_HEIGHT - 1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, ROOM_WIDTH - 1] = True

        # Default to a closed shell first, then reopen only legal canonical doors.
        out[boundary_mask] = wall_id

        required_doors: Dict[str, bool] = {}
        edge_constraints: Dict[str, Set[str]] = {}
        if isinstance(graph, nx.Graph) and room_id in graph:
            semantics = self._extract_room_topology_semantics(graph, room_id)
            required_doors = {str(k): bool(v) for k, v in semantics["required_doors"].items()}
            edge_constraints = {
                str(k): {str(tok) for tok in set(v)}
                for k, v in semantics["edge_constraints"].items()
            }

        # Standalone generation has no graph semantics; preserve any canonical
        # door tiles that the model placed in legal doorway strips.
        if not required_doors:
            for direction, spec in DOOR_POSITIONS.items():
                if direction in {"N", "S"}:
                    row = int(spec["row"])
                    c0 = int(spec["col_start"])
                    c1 = int(spec["col_end"]) + 1
                    strip = before[row, c0:c1]
                    if bool(np.any(np.isin(strip, door_tile_values))):
                        out[row, c0:c1] = strip
                else:
                    col = int(spec["col"])
                    r0 = int(spec["row_start"])
                    r1 = int(spec["row_end"]) + 1
                    strip = before[r0:r1, col]
                    if bool(np.any(np.isin(strip, door_tile_values))):
                        out[r0:r1, col] = strip
        else:
            for direction, enabled in required_doors.items():
                if not bool(enabled):
                    continue
                tile_id = int(self._edge_tokens_to_door_tile(edge_constraints.get(str(direction), set())))
                spec = DOOR_POSITIONS[str(direction)]
                if direction in {"N", "S"}:
                    row = int(spec["row"])
                    c0 = int(spec["col_start"])
                    c1 = int(spec["col_end"]) + 1
                    out[row, c0:c1] = tile_id
                    interior_row = 1 if direction == "N" else ROOM_HEIGHT - 2
                    out[interior_row, c0:c1] = floor_id
                else:
                    col = int(spec["col"])
                    r0 = int(spec["row_start"])
                    r1 = int(spec["row_end"]) + 1
                    out[r0:r1, col] = tile_id
                    interior_col = 1 if direction == "W" else ROOM_WIDTH - 2
                    out[r0:r1, interior_col] = floor_id

        boundary_wall_tiles_forced = int(np.sum(boundary_mask & (out == wall_id) & (before != wall_id)))
        boundary_door_tiles_forced = int(
            np.sum(boundary_mask & np.isin(out, door_tile_values) & (before != out))
        )
        interior_door_apron_tiles_forced = int(
            np.sum((~boundary_mask) & (out == floor_id) & (before != floor_id))
        )

        return out, {
            "boundary_wall_tiles_forced": int(boundary_wall_tiles_forced),
            "boundary_door_tiles_forced": int(boundary_door_tiles_forced),
            "interior_door_apron_tiles_forced": int(interior_door_apron_tiles_forced),
        }

    def _strip_structural_room_artifacts(
        self,
        grid: np.ndarray,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
        max_interior_component_tiles: int = 4,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Remove impossible structural artifacts from generated rooms.

        The diffusion branch is currently prone to producing:
        - door tiles floating inside the room interior or on the wrong wall slot
        - tiny isolated wall/block islands that read as decode noise rather than
          intentional room structure

        These artifacts are deterministic to detect from room topology and can
        be cleaned without changing mission-graph semantics.
        """
        out = np.asarray(grid, dtype=np.int32).copy()
        floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
        door_tiles = np.array(
            [
                int(TileID.DOOR_OPEN),
                int(TileID.DOOR_LOCKED),
                int(TileID.DOOR_BOMB),
                int(TileID.DOOR_PUZZLE),
                int(TileID.DOOR_BOSS),
                int(TileID.DOOR_SOFT),
            ],
            dtype=np.int32,
        )
        allowed_door_mask = self._required_room_door_slots_mask(graph=graph, room_id=room_id)
        invalid_door_mask = np.isin(out, door_tiles) & ~allowed_door_mask
        invalid_door_tiles_removed = int(np.sum(invalid_door_mask))
        if invalid_door_tiles_removed > 0:
            out[invalid_door_mask] = floor_id

        wall_like_mask = np.isin(out, np.array([int(TileID.WALL), int(TileID.BLOCK)], dtype=np.int32))
        visited = np.zeros_like(wall_like_mask, dtype=bool)
        interior_obstacle_tiles_removed = 0
        interior_obstacle_components_removed = 0

        for row in range(ROOM_HEIGHT):
            for col in range(ROOM_WIDTH):
                if not bool(wall_like_mask[row, col]) or bool(visited[row, col]):
                    continue

                component: List[Tuple[int, int]] = []
                stack: List[Tuple[int, int]] = [(row, col)]
                visited[row, col] = True
                touches_allowed_door = False

                while stack:
                    cur_r, cur_c = stack.pop()
                    component.append((cur_r, cur_c))
                    if bool(allowed_door_mask[cur_r, cur_c]):
                        touches_allowed_door = True
                    for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        next_r = cur_r + d_r
                        next_c = cur_c + d_c
                        if not (0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH):
                            continue
                        if not bool(wall_like_mask[next_r, next_c]) or bool(visited[next_r, next_c]):
                            continue
                        visited[next_r, next_c] = True
                        stack.append((next_r, next_c))

                if touches_allowed_door or len(component) > int(max_interior_component_tiles):
                    continue

                for comp_r, comp_c in component:
                    out[comp_r, comp_c] = floor_id
                interior_obstacle_tiles_removed += int(len(component))
                interior_obstacle_components_removed += 1

        return out, {
            "invalid_door_tiles_removed": int(invalid_door_tiles_removed),
            "interior_obstacle_tiles_removed": int(interior_obstacle_tiles_removed),
            "interior_obstacle_components_removed": int(interior_obstacle_components_removed),
        }

    def _strip_room_block_structure(
        self,
        grid: np.ndarray,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Strip interior BLOCK tiles for strict no-puzzle ablations.

        Disabling the runtime puzzle scaffold alone does not guarantee a true
        no-puzzle export because learned decode noise can still leave interior
        brown BLOCK tiles behind. This helper enforces the stronger ablation
        contract by removing those tiles deterministically.
        """
        out = np.asarray(grid, dtype=np.int32).copy()
        block_id = int(TileID.BLOCK)
        floor_id = int(TileID.FLOOR)
        block_mask = out == block_id
        removed_tiles = int(np.sum(block_mask))
        if removed_tiles <= 0:
            return out, {
                "applied": 0,
                "block_tiles_removed": 0,
                "block_components_removed": 0,
            }

        visited = np.zeros_like(block_mask, dtype=bool)
        removed_components = 0
        for row in range(ROOM_HEIGHT):
            for col in range(ROOM_WIDTH):
                if not bool(block_mask[row, col]) or bool(visited[row, col]):
                    continue
                removed_components += 1
                stack: List[Tuple[int, int]] = [(row, col)]
                visited[row, col] = True
                while stack:
                    cur_r, cur_c = stack.pop()
                    for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        next_r = cur_r + d_r
                        next_c = cur_c + d_c
                        if not (0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH):
                            continue
                        if not bool(block_mask[next_r, next_c]) or bool(visited[next_r, next_c]):
                            continue
                        visited[next_r, next_c] = True
                        stack.append((next_r, next_c))

        out[block_mask] = floor_id
        logger.debug(
            "Room %s stripped %d BLOCK tiles across %d components for strict no-puzzle cleanup.",
            room_id,
            removed_tiles,
            removed_components,
        )
        return out, {
            "applied": 1,
            "block_tiles_removed": int(removed_tiles),
            "block_components_removed": int(removed_components),
        }

    @staticmethod
    def _dilate_room_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
        """Lightweight 4-neighbour dilation for room-scale boolean masks."""
        out = np.asarray(mask, dtype=bool).copy()
        steps = int(max(0, int(radius)))
        for _ in range(steps):
            prev = out.copy()
            out[1:, :] |= prev[:-1, :]
            out[:-1, :] |= prev[1:, :]
            out[:, 1:] |= prev[:, :-1]
            out[:, :-1] |= prev[:, 1:]
        return out

    @staticmethod
    def _paint_room_line_mask(
        canvas: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        *,
        value: bool = True,
    ) -> None:
        """Paint a Manhattan polyline between two room-local points."""
        r0, c0 = int(start[0]), int(start[1])
        r1, c1 = int(end[0]), int(end[1])
        r, c = r0, c0
        while r != r1:
            canvas[r, c] = bool(value)
            r += 1 if r1 > r else -1
        while c != c1:
            canvas[r, c] = bool(value)
            c += 1 if c1 > c else -1
        canvas[r1, c1] = bool(value)

    def _build_puzzle_room_route_template(
        self,
        *,
        archetype: str,
        gate_family: str,
        variant_spec: Optional[Mapping[str, Any]] = None,
        stateful_anchor: Optional[Tuple[int, int]],
        interaction_sequence: Optional[Sequence[Tuple[str, Tuple[int, int]]]] = None,
        flow_is_horizontal: bool,
        source_anchor: Tuple[int, int],
        destination_anchor: Tuple[int, int],
        puzzle_anchor: Tuple[int, int],
        role_flags: Dict[str, bool],
        semantics: Dict[str, Any],
    ) -> np.ndarray:
        """
        Build an archetype-specific route skeleton for constructive puzzle rooms.

        The default semantic room trace is intentionally permissive; it keeps
        traversal valid, but it does not impose much puzzle readability. For
        puzzle rooms we instead reserve a more explicit route skeleton first,
        then place obstacles around it. This yields rooms that read like
        gates/hubs/serpentines instead of random block bars.
        """
        mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
        source = self._clamp_room_coord(source_anchor)
        destination = self._clamp_room_coord(destination_anchor)
        puzzle = self._clamp_room_coord(puzzle_anchor)
        stateful = self._clamp_room_coord(stateful_anchor) if stateful_anchor is not None else puzzle
        center = (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)
        switch_depth = int(max(1, getattr(self, "default_puzzle_room_switch_pocket_depth", 3)))
        resource_offset = int(max(1, getattr(self, "default_puzzle_room_resource_bypass_offset", 2)))
        key_depth = int(max(1, getattr(self, "default_puzzle_room_key_pocket_depth", 3)))
        item_slot_depth = int(max(1, getattr(self, "default_puzzle_room_item_slot_depth", 3)))
        toggle_offset = int(max(1, getattr(self, "default_puzzle_room_toggle_corridor_offset", 2)))
        variant = dict(variant_spec or {})
        variant_style = str(variant.get("style", "baseline") or "baseline").strip().lower()
        variant_side_bias = int(max(-1, min(1, int(variant.get("side_bias", 0) or 0))))

        def _pick_side_lane(anchor_value: int, reference_value: int, *, low: int, high: int, offset: int) -> int:
            anchor_value = int(anchor_value)
            reference_value = int(reference_value)
            if variant_side_bias != 0:
                direction = int(variant_side_bias)
            else:
                direction = 1 if anchor_value >= reference_value else -1
            candidates = []
            if abs(anchor_value - reference_value) >= 1:
                candidates.append(anchor_value)
            candidates.append(anchor_value + direction * max(1, offset))
            candidates.append(reference_value + direction * max(2, offset))
            candidates.append(reference_value + (2 if direction >= 0 else -2))
            for candidate in candidates:
                candidate = max(low, min(high, int(candidate)))
                if abs(candidate - reference_value) >= 1:
                    return candidate
            fallback = reference_value + (1 if reference_value < ((low + high) // 2) else -1)
            return max(low, min(high, int(fallback)))

        def _add_polyline(points: List[Tuple[int, int]]) -> None:
            if len(points) < 2:
                return
            for start, end in zip(points, points[1:]):
                self._paint_room_line_mask(mask, self._clamp_room_coord(start), self._clamp_room_coord(end))

        if archetype == "gate":
            if flow_is_horizontal:
                gate_col = max(3, min(ROOM_WIDTH - 4, int(puzzle[1])))
                pocket_row = max(
                    2,
                    min(
                        ROOM_HEIGHT - 3,
                        int(stateful[0]) if variant_side_bias == 0 else int(center[0] + variant_side_bias * max(2, abs(int(stateful[0]) - int(center[0])) or 2)),
                    ),
                )
                entry = (source[0], max(2, gate_col - 2))
                if gate_family == "switch":
                    pocket = (pocket_row, max(2, gate_col - switch_depth))
                    gate_open = (pocket_row, gate_col)
                    exit_point = (
                        center[0] if variant_style == "bridge" else pocket_row,
                        min(ROOM_WIDTH - 3, gate_col + (3 if variant_style == "bridge" else 2)),
                    )
                elif gate_family == "toggle":
                    toggle_row = max(
                        2,
                        min(
                            ROOM_HEIGHT - 3,
                            int(stateful[0]) + (-toggle_offset if stateful[0] > ROOM_HEIGHT // 2 else toggle_offset),
                        ),
                    )
                    if variant_style == "weave":
                        pocket = (toggle_row, max(2, gate_col - 3))
                    else:
                        pocket = (toggle_row, max(2, gate_col - 2))
                    gate_open = (stateful[0], gate_col)
                    exit_point = (toggle_row if variant_style == "weave" else stateful[0], min(ROOM_WIDTH - 3, gate_col + 2))
                elif gate_family == "bombable":
                    bypass_row = _pick_side_lane(
                        int(stateful[0]),
                        int(center[0]),
                        low=2,
                        high=ROOM_HEIGHT - 3,
                        offset=resource_offset,
                    )
                    pocket = (bypass_row, max(2, gate_col - (resource_offset + (2 if variant_style == "wrap" else 1))))
                    gate_open = (bypass_row, gate_col)
                    exit_point = (bypass_row, min(ROOM_WIDTH - 3, gate_col + (4 if variant_style == "wrap" else 2)))
                elif gate_family == "item_unlock":
                    item_row = max(2, min(ROOM_HEIGHT - 3, int(stateful[0])))
                    item_col = max(
                        2,
                        min(
                            ROOM_WIDTH - 3,
                            int(stateful[1]) + (
                                variant_side_bias * max(1, item_slot_depth - 1)
                                if variant_style in {"slot", "ring"} else 0
                            ),
                        ),
                    )
                    pocket = (item_row, item_col)
                    gate_open = (center[0], gate_col)
                    exit_point = (item_row if variant_style == "ring" else center[0], max(min(ROOM_WIDTH - 3, item_col), min(ROOM_WIDTH - 3, gate_col + 2)))
                elif gate_family == "key" and stateful_anchor is not None:
                    key_row = max(
                        2,
                        min(
                            ROOM_HEIGHT - 3,
                            int(stateful[0]) if variant_side_bias == 0 else int(center[0] + variant_side_bias * max(2, abs(int(stateful[0]) - int(center[0])) or 2)),
                        ),
                    )
                    pocket = (key_row, max(2, gate_col - key_depth))
                    gate_open = (center[0], gate_col)
                    exit_point = (key_row if variant_style == "split" else center[0], min(ROOM_WIDTH - 3, gate_col + 2))
                else:
                    pocket = (pocket_row, max(2, gate_col - 2))
                    gate_open = (pocket_row, gate_col)
                    exit_point = (pocket_row, min(ROOM_WIDTH - 3, gate_col + (4 if variant_style == "wrap" else 2)))
                destination_hook = (destination[0], int(exit_point[1]))
                _add_polyline([source, entry, pocket, gate_open, exit_point, destination_hook, destination])
            else:
                gate_row = max(3, min(ROOM_HEIGHT - 4, int(puzzle[0])))
                pocket_col = max(
                    2,
                    min(
                        ROOM_WIDTH - 3,
                        int(stateful[1]) if variant_side_bias == 0 else int(center[1] + variant_side_bias * max(2, abs(int(stateful[1]) - int(center[1])) or 2)),
                    ),
                )
                entry = (max(2, gate_row - 2), source[1])
                if gate_family == "switch":
                    pocket = (max(2, gate_row - switch_depth), pocket_col)
                    gate_open = (gate_row, pocket_col)
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + (3 if variant_style == "bridge" else 2)), center[1] if variant_style == "bridge" else pocket_col)
                elif gate_family == "toggle":
                    toggle_col = max(
                        2,
                        min(
                            ROOM_WIDTH - 3,
                            int(stateful[1]) + (-toggle_offset if stateful[1] > ROOM_WIDTH // 2 else toggle_offset),
                        ),
                    )
                    pocket = (max(2, gate_row - (3 if variant_style == "weave" else 2)), toggle_col)
                    gate_open = (gate_row, stateful[1])
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + 2), toggle_col if variant_style == "weave" else stateful[1])
                elif gate_family == "bombable":
                    bypass_col = _pick_side_lane(
                        int(stateful[1]),
                        int(center[1]),
                        low=2,
                        high=ROOM_WIDTH - 3,
                        offset=resource_offset,
                    )
                    pocket = (max(2, gate_row - (resource_offset + (2 if variant_style == "wrap" else 1))), bypass_col)
                    gate_open = (gate_row, bypass_col)
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + (4 if variant_style == "wrap" else 2)), bypass_col)
                elif gate_family == "item_unlock":
                    item_row = max(
                        2,
                        min(
                            ROOM_HEIGHT - 3,
                            int(stateful[0]) + (
                                variant_side_bias * max(1, item_slot_depth - 1)
                                if variant_style in {"slot", "ring"} else 0
                            ),
                        ),
                    )
                    item_col = max(2, min(ROOM_WIDTH - 3, int(stateful[1])))
                    pocket = (item_row, item_col)
                    gate_open = (gate_row, center[1])
                    exit_point = (max(min(ROOM_HEIGHT - 3, item_row), min(ROOM_HEIGHT - 3, gate_row + 2)), item_col if variant_style == "ring" else center[1])
                elif gate_family == "key" and stateful_anchor is not None:
                    key_col = max(
                        2,
                        min(
                            ROOM_WIDTH - 3,
                            int(stateful[1]) if variant_side_bias == 0 else int(center[1] + variant_side_bias * max(2, abs(int(stateful[1]) - int(center[1])) or 2)),
                        ),
                    )
                    pocket = (max(2, gate_row - key_depth), key_col)
                    gate_open = (gate_row, center[1])
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + 2), key_col if variant_style == "split" else center[1])
                else:
                    pocket = (max(2, gate_row - 2), pocket_col)
                    gate_open = (gate_row, pocket_col)
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + (4 if variant_style == "wrap" else 2)), pocket_col)
                destination_hook = (int(exit_point[0]), destination[1])
                _add_polyline([source, entry, pocket, gate_open, exit_point, destination_hook, destination])
        elif archetype == "hub":
            hub_row = int(round((puzzle[0] + center[0]) / 2.0))
            hub_col = int(round((puzzle[1] + center[1]) / 2.0))
            if variant_style == "offset":
                hub_row += int(variant_side_bias * 2)
            hub = self._clamp_room_coord((hub_row, hub_col))
            _add_polyline([source, hub, destination])
            for direction, enabled in semantics.get("required_doors", {}).items():
                if not bool(enabled):
                    continue
                door_anchor = self._clamp_room_coord(
                    build_room_semantic_anchor_points(
                        room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
                        required_doors={str(direction): True},
                        incoming_dirs=set(),
                        outgoing_dirs=set(),
                        room_role_flags={},
                        semantic_puzzle_offset=self.default_semantic_puzzle_offset,
                    ).get(f"door:{direction}", hub)
                )
                self._paint_room_line_mask(mask, hub, door_anchor)
            if variant_style == "cross":
                self._paint_room_line_mask(mask, hub, (hub[0], max(1, hub[1] - 3)))
                self._paint_room_line_mask(mask, hub, (hub[0], min(ROOM_WIDTH - 2, hub[1] + 3)))
            mask[max(1, hub[0] - 1): min(ROOM_HEIGHT - 1, hub[0] + 2), max(1, hub[1] - 1): min(ROOM_WIDTH - 1, hub[1] + 2)] = True
        elif archetype == "combat":
            arena_center = self._clamp_room_coord(
                (
                    int(round((source[0] + destination[0] + puzzle[0]) / 3.0)) + (variant_side_bias * 2 if variant_style == "offset" else 0),
                    int(round((source[1] + destination[1] + puzzle[1]) / 3.0)),
                )
            )
            _add_polyline([source, arena_center, destination])
            self._paint_room_line_mask(mask, arena_center, puzzle)
            if variant_style == "cross":
                self._paint_room_line_mask(mask, (arena_center[0], max(1, arena_center[1] - 3)), (arena_center[0], min(ROOM_WIDTH - 2, arena_center[1] + 3)))
            mask[max(1, arena_center[0] - 1): min(ROOM_HEIGHT - 1, arena_center[0] + 2), max(1, arena_center[1] - 1): min(ROOM_WIDTH - 1, arena_center[1] + 2)] = True
        elif archetype == "island":
            waypoint = self._clamp_room_coord(
                (
                    int(round((puzzle[0] + destination[0]) / 2.0)) + (variant_side_bias * 2 if variant_style == "bridge" else 0),
                    int(round((puzzle[1] + destination[1]) / 2.0)),
                )
            )
            _add_polyline([source, puzzle, waypoint, destination])
            if variant_style == "staggered":
                _add_polyline([source, (puzzle[0], max(1, puzzle[1] - 2)), puzzle])
            mask[max(1, puzzle[0] - 1): min(ROOM_HEIGHT - 1, puzzle[0] + 2), max(1, puzzle[1] - 1): min(ROOM_WIDTH - 1, puzzle[1] + 2)] = True
        else:  # serpentine
            if flow_is_horizontal:
                if variant_style == "split":
                    side_row = max(2, min(ROOM_HEIGHT - 3, center[0] + variant_side_bias * 3))
                    waypoints = [source, (2, 3), (side_row, 3), (side_row, ROOM_WIDTH - 4), (ROOM_HEIGHT - 4, ROOM_WIDTH - 4), destination]
                else:
                    left_first = variant_style != "mirror"
                    waypoints = [
                        source,
                        (2, 3 if left_first else ROOM_WIDTH - 4),
                        (4, 3 if left_first else ROOM_WIDTH - 4),
                        (4, ROOM_WIDTH - 4 if left_first else 3),
                        (8, ROOM_WIDTH - 4 if left_first else 3),
                        (8, 3 if left_first else ROOM_WIDTH - 4),
                        (12, 3 if left_first else ROOM_WIDTH - 4),
                        (12, ROOM_WIDTH - 4 if left_first else 3),
                        destination,
                    ]
            else:
                if variant_style == "split":
                    side_col = max(2, min(ROOM_WIDTH - 3, center[1] + variant_side_bias * 2))
                    waypoints = [source, (3, 2), (3, side_col), (ROOM_HEIGHT - 4, side_col), (ROOM_HEIGHT - 4, ROOM_WIDTH - 3), destination]
                else:
                    top_first = variant_style != "mirror"
                    waypoints = [
                        source,
                        (3 if top_first else ROOM_HEIGHT - 4, 2),
                        (3 if top_first else ROOM_HEIGHT - 4, 4),
                        (ROOM_HEIGHT - 4 if top_first else 3, 4),
                        (ROOM_HEIGHT - 4 if top_first else 3, 6),
                        (3 if top_first else ROOM_HEIGHT - 4, 6),
                        (3 if top_first else ROOM_HEIGHT - 4, ROOM_WIDTH - 3),
                        destination,
                    ]
            _add_polyline(waypoints)
            self._paint_room_line_mask(mask, puzzle, waypoints[min(len(waypoints) - 2, 3)])

        if role_flags.get("has_puzzle", False):
            mask[max(1, puzzle[0] - 1): min(ROOM_HEIGHT - 1, puzzle[0] + 2), max(1, puzzle[1] - 1): min(ROOM_WIDTH - 1, puzzle[1] + 2)] = True

        sequence = list(interaction_sequence or [])
        if len(sequence) >= 2:
            ordered_points: List[Tuple[int, int]] = [source]
            ordered_points.extend(self._clamp_room_coord(anchor) for _name, anchor in sequence)
            ordered_points.append(destination)
            _add_polyline(ordered_points)
            for _name, anchor in sequence:
                seq_r, seq_c = self._clamp_room_coord(anchor)
                mask[
                    max(1, seq_r - 1): min(ROOM_HEIGHT - 1, seq_r + 2),
                    max(1, seq_c - 1): min(ROOM_WIDTH - 1, seq_c + 2),
                ] = True

        return mask

    def _resolve_puzzle_interaction_sequence(
        self,
        *,
        archetype: str,
        gate_family: str,
        role_flags: Mapping[str, bool],
        semantic_anchors: Mapping[str, Tuple[int, int]],
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Resolve a staged in-room interaction sequence for complex puzzle rooms.

        This is intentionally a lightweight constructive grammar. It does not
        simulate a full state machine, but it gives the room generator a
        progression order across local anchors such as key, item, puzzle, and
        enemy instead of treating every puzzle room as a single interaction.
        """
        archetype = str(archetype or "serpentine").strip().lower()
        gate_family = str(gate_family or "generic").strip().lower()
        flags = {str(key): bool(value) for key, value in dict(role_flags or {}).items()}
        anchors = {
            str(name): self._clamp_room_coord(coord)
            for name, coord in dict(semantic_anchors or {}).items()
            if coord is not None
        }
        sequence: List[Tuple[str, Tuple[int, int]]] = []
        seen: Set[str] = set()

        def _add(name: str) -> None:
            key = str(name)
            if key in seen:
                return
            anchor = anchors.get(key)
            if anchor is None:
                return
            sequence.append((key, anchor))
            seen.add(key)

        multi_step_room = bool(flags.get("is_complex_puzzle", False))
        multi_step_room = multi_step_room or archetype in {"hub", "combat"}
        multi_step_room = multi_step_room or sum(
            int(flags.get(name, False))
            for name in ("has_key", "has_item", "has_enemy", "has_puzzle")
        ) >= 3
        if multi_step_room:
            for name, flag_key in (
                ("key", "has_key"),
                ("item", "has_item"),
                ("puzzle", "has_puzzle"),
                ("enemy", "has_enemy"),
                ("boss", "has_boss"),
            ):
                if bool(flags.get(flag_key, False)):
                    _add(name)
            return list(sequence)

        if gate_family == "key":
            _add("key")
            _add("puzzle")
        elif gate_family == "item_unlock":
            _add("item")
            _add("puzzle")
        elif gate_family in {"switch", "toggle", "bombable"}:
            _add("puzzle")
        elif gate_family == "combat":
            if flags.get("has_puzzle", False):
                _add("puzzle")
            _add("enemy")

        return list(sequence)

    def _evaluate_puzzle_candidate_interaction_sequence(
        self,
        *,
        grid: np.ndarray,
        route_mask: np.ndarray,
        source_anchor: Tuple[int, int],
        destination_anchor: Tuple[int, int],
        interaction_sequence: Sequence[Tuple[str, Tuple[int, int]]],
    ) -> Dict[str, Any]:
        """
        Validate simple staged room progression across multiple local anchors.

        This remains a room-local proxy for multi-step puzzle structure. A
        candidate is rewarded when the reserved route actually visits the
        intermediate anchors and when each consecutive stage is walkably
        connectable in the final geometry.
        """
        sequence = [
            (str(name), self._clamp_room_coord(anchor))
            for name, anchor in list(interaction_sequence or [])
            if anchor is not None
        ]
        if len(sequence) <= 1:
            return {
                "required": 0,
                "valid": 1,
                "score": 0.0,
                "failure_reasons": [],
                "sequence_length": int(len(sequence)),
                "route_anchor_coverage": 1.0 if sequence else 0.0,
                "pairwise_path_ratio": 1.0,
                "sequence_names": [name for name, _anchor in sequence],
            }

        walkable = self._build_room_walkable_mask(grid)
        route_arr = np.asarray(route_mask, dtype=bool)

        covered = 0
        for _name, anchor in sequence:
            anchor_r, anchor_c = anchor
            route_r0 = max(0, anchor_r - 1)
            route_r1 = min(ROOM_HEIGHT, anchor_r + 2)
            route_c0 = max(0, anchor_c - 1)
            route_c1 = min(ROOM_WIDTH, anchor_c + 2)
            if bool(np.any(route_arr[route_r0:route_r1, route_c0:route_c1])):
                covered += 1
        route_anchor_coverage = float(covered) / float(max(1, len(sequence)))

        path_nodes: List[Tuple[int, int]] = [self._clamp_room_coord(source_anchor)]
        path_nodes.extend(anchor for _name, anchor in sequence)
        path_nodes.append(self._clamp_room_coord(destination_anchor))
        pairwise_success = 0
        pairwise_total = max(0, len(path_nodes) - 1)
        for start, goal in zip(path_nodes, path_nodes[1:]):
            if self._shortest_room_path(walkable, start, goal):
                pairwise_success += 1
        pairwise_path_ratio = float(pairwise_success) / float(max(1, pairwise_total))

        failure_reasons: List[str] = []
        if route_anchor_coverage < 1.0:
            failure_reasons.append("incomplete_sequence_route")
        if pairwise_path_ratio < 1.0:
            failure_reasons.append("broken_sequence_connectivity")

        score = 0.0
        score += 0.90 * route_anchor_coverage
        score += 0.70 * pairwise_path_ratio
        score += 0.20 * float(min(1.0, float(len(sequence) - 1) / 3.0))
        valid = int(len(failure_reasons) == 0)
        if not valid:
            score -= 1.75 + (0.10 * len(failure_reasons))

        return {
            "required": 1,
            "valid": int(valid),
            "score": float(score),
            "failure_reasons": list(failure_reasons),
            "sequence_length": int(len(sequence)),
            "route_anchor_coverage": float(route_anchor_coverage),
            "pairwise_path_ratio": float(pairwise_path_ratio),
            "sequence_names": [name for name, _anchor in sequence],
        }

    def _select_puzzle_room_scaffold_archetype(
        self,
        *,
        role_flags: Dict[str, bool],
        semantics: Dict[str, Any],
        node_type: str,
    ) -> str:
        """Select the constructive puzzle archetype from graph-local semantics."""
        forced = str(getattr(self, "default_puzzle_room_archetype_mode", "auto") or "auto").strip().lower()
        valid = {"auto", "gate", "serpentine", "hub", "island", "combat"}
        if forced not in valid:
            forced = "auto"
        if forced != "auto":
            return forced

        edge_constraints = semantics.get("edge_constraints", {})
        flat_edge_tokens: Set[str] = set()
        for tokens in edge_constraints.values():
            flat_edge_tokens.update(str(token) for token in tokens)

        required_doors = semantics.get("required_doors", {})
        required_door_count = int(sum(1 for enabled in required_doors.values() if enabled))

        if node_type == "combat_puzzle" or bool(role_flags.get("has_enemy", False)):
            return "combat"
        if {"switch", "switch_locked", "state_block", "on_off_gate"} & flat_edge_tokens:
            return "gate"
        if node_type in {"item", "protection_item", "minor_item", "treasure", "stair", "stairs_up", "stairs_down", "warp"} or bool(role_flags.get("has_item", False)):
            return "island"
        if required_door_count >= 3:
            return "hub"
        return "serpentine"

    def _classify_puzzle_gate_family(
        self,
        *,
        role_flags: Dict[str, bool],
        semantics: Dict[str, Any],
        node_type: str,
    ) -> str:
        """Classify the local puzzle as a concrete gate semantic family."""
        edge_constraints = semantics.get("edge_constraints", {})
        flat_edge_tokens: Set[str] = set()
        for tokens in edge_constraints.values():
            flat_edge_tokens.update(str(token) for token in tokens)

        if {"state_block", "on_off_gate"} & flat_edge_tokens:
            return "toggle"
        if role_flags.get("is_switch_puzzle", False) or {"switch", "switch_locked"} & flat_edge_tokens:
            return "switch"
        if role_flags.get("is_combat_puzzle", False) or node_type == "combat_puzzle":
            return "combat"
        if {"bombable"} & flat_edge_tokens:
            return "bombable"
        if {"item_locked", "item_gate"} & flat_edge_tokens:
            return "item_unlock"
        if {"key_locked", "locked", "boss_locked"} & flat_edge_tokens:
            return "key"
        if {"shutter", "soft_locked"} & flat_edge_tokens:
            return "combat"
        return "generic"

    def _build_puzzle_room_variant_specs(
        self,
        *,
        archetype: str,
        gate_family: str,
    ) -> List[Dict[str, Any]]:
        """Enumerate small, valid scaffold variants for novelty-aware puzzle selection."""
        if not bool(getattr(self, "default_puzzle_room_novelty_enabled", True)):
            return [{"name": "baseline", "style": "baseline", "side_bias": 0, "branch_density_delta": 0.0, "block_budget_delta": 0}]

        specs: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        def _add(
            name: str,
            *,
            style: str,
            side_bias: int = 0,
            branch_density_delta: float = 0.0,
            block_budget_delta: int = 0,
        ) -> None:
            variant_name = str(name).strip().lower()
            if not variant_name or variant_name in seen:
                return
            seen.add(variant_name)
            specs.append(
                {
                    "name": variant_name,
                    "style": str(style).strip().lower(),
                    "side_bias": int(max(-1, min(1, int(side_bias)))),
                    "branch_density_delta": float(branch_density_delta),
                    "block_budget_delta": int(block_budget_delta),
                }
            )

        if archetype == "gate":
            if gate_family == "switch":
                _add("switch_upper_pocket", style="pocket", side_bias=-1, branch_density_delta=0.00, block_budget_delta=0)
                _add("switch_lower_pocket", style="pocket", side_bias=1, branch_density_delta=0.00, block_budget_delta=0)
                _add("switch_upper_bridge", style="bridge", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
                _add("switch_lower_bridge", style="bridge", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)
            elif gate_family == "toggle":
                _add("toggle_upper_corridor", style="corridor", side_bias=-1, branch_density_delta=0.04, block_budget_delta=1)
                _add("toggle_lower_corridor", style="corridor", side_bias=1, branch_density_delta=0.04, block_budget_delta=1)
                _add("toggle_upper_weave", style="weave", side_bias=-1, branch_density_delta=0.10, block_budget_delta=3)
                _add("toggle_lower_weave", style="weave", side_bias=1, branch_density_delta=0.10, block_budget_delta=3)
            elif gate_family == "bombable":
                _add("bomb_upper_bypass", style="bypass", side_bias=-1, branch_density_delta=0.02, block_budget_delta=0)
                _add("bomb_lower_bypass", style="bypass", side_bias=1, branch_density_delta=0.02, block_budget_delta=0)
                _add("bomb_upper_wrap", style="wrap", side_bias=-1, branch_density_delta=0.10, block_budget_delta=2)
                _add("bomb_lower_wrap", style="wrap", side_bias=1, branch_density_delta=0.10, block_budget_delta=2)
            elif gate_family == "item_unlock":
                _add("item_slot_left", style="slot", side_bias=-1, branch_density_delta=0.05, block_budget_delta=0)
                _add("item_slot_right", style="slot", side_bias=1, branch_density_delta=0.05, block_budget_delta=0)
                _add("item_ring_left", style="ring", side_bias=-1, branch_density_delta=0.12, block_budget_delta=3)
                _add("item_ring_right", style="ring", side_bias=1, branch_density_delta=0.12, block_budget_delta=3)
            elif gate_family == "key":
                _add("key_upper_alcove", style="alcove", side_bias=-1, branch_density_delta=0.00, block_budget_delta=0)
                _add("key_lower_alcove", style="alcove", side_bias=1, branch_density_delta=0.00, block_budget_delta=0)
                _add("key_upper_split", style="split", side_bias=-1, branch_density_delta=0.10, block_budget_delta=2)
                _add("key_lower_split", style="split", side_bias=1, branch_density_delta=0.10, block_budget_delta=2)
            else:
                _add("gate_upper_offset", style="offset", side_bias=-1, branch_density_delta=0.00, block_budget_delta=0)
                _add("gate_lower_offset", style="offset", side_bias=1, branch_density_delta=0.00, block_budget_delta=0)
                _add("gate_upper_wrap", style="wrap", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
                _add("gate_lower_wrap", style="wrap", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)
        elif archetype == "hub":
            _add("hub_ring", style="ring", branch_density_delta=0.00, block_budget_delta=0)
            _add("hub_cross", style="cross", branch_density_delta=0.06, block_budget_delta=2)
            _add("hub_upper_offset", style="offset", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
            _add("hub_lower_offset", style="offset", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)
        elif archetype == "combat":
            _add("combat_cross", style="cross", branch_density_delta=0.00, block_budget_delta=0)
            _add("combat_corners", style="corners", branch_density_delta=0.06, block_budget_delta=2)
            _add("combat_upper_lane", style="offset", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
            _add("combat_lower_lane", style="offset", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)
        elif archetype == "island":
            _add("island_quad", style="quad", branch_density_delta=0.00, block_budget_delta=0)
            _add("island_staggered", style="staggered", branch_density_delta=0.06, block_budget_delta=2)
            _add("island_upper_bridge", style="bridge", side_bias=-1, branch_density_delta=0.10, block_budget_delta=3)
            _add("island_lower_bridge", style="bridge", side_bias=1, branch_density_delta=0.10, block_budget_delta=3)
        else:
            _add("serpentine_classic", style="classic", branch_density_delta=0.00, block_budget_delta=0)
            _add("serpentine_mirror", style="mirror", branch_density_delta=0.02, block_budget_delta=0)
            _add("serpentine_upper_split", style="split", side_bias=-1, branch_density_delta=0.08, block_budget_delta=2)
            _add("serpentine_lower_split", style="split", side_bias=1, branch_density_delta=0.08, block_budget_delta=2)

        candidate_limit = int(max(1, min(6, int(getattr(self, "default_puzzle_room_candidate_count", 4)))))
        return specs[:candidate_limit]

    def _summarize_puzzle_candidate_descriptor(
        self,
        *,
        grid: np.ndarray,
        stats: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Compact numeric descriptor used for novelty-aware scaffold selection."""
        block_mask = np.asarray(grid, dtype=np.int32) == int(TileID.BLOCK)
        rows = np.sum(block_mask, axis=1)
        cols = np.sum(block_mask, axis=0)
        coords = np.argwhere(block_mask)
        if coords.size > 0:
            mean_row = float(np.mean(coords[:, 0]))
            mean_col = float(np.mean(coords[:, 1]))
        else:
            mean_row = float(ROOM_HEIGHT // 2)
            mean_col = float(ROOM_WIDTH // 2)
        half_r = ROOM_HEIGHT // 2
        half_c = ROOM_WIDTH // 2
        quadrants = [
            int(np.sum(block_mask[:half_r, :half_c])),
            int(np.sum(block_mask[:half_r, half_c:])),
            int(np.sum(block_mask[half_r:, :half_c])),
            int(np.sum(block_mask[half_r:, half_c:])),
        ]
        return {
            "variant_name": str(stats.get("variant_name", "") or ""),
            "archetype": str(stats.get("archetype", "") or ""),
            "gate_family": str(stats.get("gate_family", "") or ""),
            "tiles_added": int(stats.get("tiles_added", 0)),
            "segments_added": int(stats.get("segments_added", 0)),
            "row_coverage": int(np.sum(rows > 0)),
            "col_coverage": int(np.sum(cols > 0)),
            "center_row": float(mean_row),
            "center_col": float(mean_col),
            "quadrants": quadrants,
        }

    @staticmethod
    def _build_room_walkable_mask(grid: np.ndarray) -> np.ndarray:
        """Return walkable cells for room-local structural scoring."""
        grid_arr = np.asarray(grid, dtype=np.int32)
        blocked = np.isin(
            grid_arr,
            np.array([int(TileID.WALL), int(TileID.BLOCK)], dtype=np.int32),
        )
        return ~blocked

    @staticmethod
    def _count_room_path_turns(path: List[Tuple[int, int]]) -> int:
        """Count Manhattan direction changes along a discrete room path."""
        if len(path) < 3:
            return 0
        turns = 0
        prev_delta: Optional[Tuple[int, int]] = None
        for prev, cur in zip(path, path[1:]):
            delta = (int(cur[0]) - int(prev[0]), int(cur[1]) - int(prev[1]))
            if prev_delta is not None and delta != prev_delta:
                turns += 1
            prev_delta = delta
        return int(turns)

    def _nearest_walkable_room_coord(
        self,
        walkable: np.ndarray,
        anchor: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Project an anchor onto the nearest walkable room tile."""
        mask = np.asarray(walkable, dtype=bool)
        if mask.shape != (ROOM_HEIGHT, ROOM_WIDTH):
            return None
        target_r, target_c = self._clamp_room_coord(anchor)
        if bool(mask[target_r, target_c]):
            return (int(target_r), int(target_c))

        visited = np.zeros_like(mask, dtype=bool)
        queue: deque[Tuple[int, int]] = deque([(int(target_r), int(target_c))])
        visited[target_r, target_c] = True
        while queue:
            row, col = queue.popleft()
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_row = row + d_row
                next_col = col + d_col
                if not (0 <= next_row < ROOM_HEIGHT and 0 <= next_col < ROOM_WIDTH):
                    continue
                if bool(visited[next_row, next_col]):
                    continue
                if bool(mask[next_row, next_col]):
                    return (int(next_row), int(next_col))
                visited[next_row, next_col] = True
                queue.append((int(next_row), int(next_col)))
        return None

    def _shortest_room_path(
        self,
        walkable: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Compute a room-local 4-neighbour shortest path."""
        mask = np.asarray(walkable, dtype=bool)
        if mask.shape != (ROOM_HEIGHT, ROOM_WIDTH):
            return []
        start_cell = self._nearest_walkable_room_coord(mask, start)
        goal_cell = self._nearest_walkable_room_coord(mask, goal)
        if start_cell is None or goal_cell is None:
            return []
        if start_cell == goal_cell:
            return [start_cell]

        queue: deque[Tuple[int, int]] = deque([start_cell])
        parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_cell: None}
        while queue:
            row, col = queue.popleft()
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_row = row + d_row
                next_col = col + d_col
                next_cell = (int(next_row), int(next_col))
                if not (0 <= next_row < ROOM_HEIGHT and 0 <= next_col < ROOM_WIDTH):
                    continue
                if not bool(mask[next_row, next_col]) or next_cell in parents:
                    continue
                parents[next_cell] = (int(row), int(col))
                if next_cell == goal_cell:
                    queue.clear()
                    break
                queue.append(next_cell)

        if goal_cell not in parents:
            return []
        path: List[Tuple[int, int]] = []
        cursor: Optional[Tuple[int, int]] = goal_cell
        while cursor is not None:
            path.append((int(cursor[0]), int(cursor[1])))
            cursor = parents[cursor]
        path.reverse()
        return path

    def _evaluate_puzzle_candidate_route_quality(
        self,
        *,
        grid: np.ndarray,
        source_anchor: Tuple[int, int],
        destination_anchor: Tuple[int, int],
        stateful_anchor: Optional[Tuple[int, int]],
        route_mask: np.ndarray,
        gate_family: str,
        baseline_path_length: Optional[int],
    ) -> Dict[str, Any]:
        """
        Score puzzle readability from the actual walkable route, not novelty alone.

        The scaffold should not just be different; it should create a readable
        route with a visible detour or stateful pocket that matches the edge
        semantics implied by the topology.
        """
        walkable = self._build_room_walkable_mask(grid)
        path = self._shortest_room_path(walkable, source_anchor, destination_anchor)
        if not path:
            return {
                "path_exists": 0,
                "path_length": 0,
                "turn_count": 0,
                "route_overlap_ratio": 0.0,
                "detour_gain": 0.0,
                "stateful_distance_to_path": None,
                "stateful_via_path_length": None,
                "stateful_branch_gain": None,
                "stateful_on_path": 0,
                "score": -4.0,
            }

        path_length = max(0, len(path) - 1)
        turn_count = self._count_room_path_turns(path)
        route_arr = np.asarray(route_mask, dtype=bool)
        route_overlap_ratio = (
            float(np.mean([1.0 if bool(route_arr[row, col]) else 0.0 for row, col in path]))
            if path else 0.0
        )

        reference_path_length = (
            int(baseline_path_length)
            if baseline_path_length is not None and int(baseline_path_length) > 0
            else int(path_length)
        )
        detour_gain = max(0.0, float(path_length - reference_path_length))

        stateful_distance_to_path: Optional[int] = None
        stateful_via_path_length: Optional[int] = None
        stateful_branch_gain: Optional[float] = None
        stateful_on_path = 0
        stateful_required = gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"}

        if stateful_anchor is not None:
            projected_stateful = self._nearest_walkable_room_coord(walkable, stateful_anchor)
            if projected_stateful is not None:
                stateful_distance_to_path = int(
                    min(
                        abs(int(projected_stateful[0]) - int(cell[0])) + abs(int(projected_stateful[1]) - int(cell[1]))
                        for cell in path
                    )
                )
                stateful_on_path = int(stateful_distance_to_path == 0)
                source_to_stateful = self._shortest_room_path(walkable, source_anchor, projected_stateful)
                stateful_to_goal = self._shortest_room_path(walkable, projected_stateful, destination_anchor)
                if source_to_stateful and stateful_to_goal:
                    stateful_via_path_length = max(0, len(source_to_stateful) - 1) + max(0, len(stateful_to_goal) - 1)
                    stateful_branch_gain = float(max(0, stateful_via_path_length - path_length))

        route_quality_score = 0.0
        route_quality_score += 1.20 * float(max(0.0, min(1.0, route_overlap_ratio)))
        route_quality_score += 0.85 * float(min(1.0, detour_gain / 6.0))
        route_quality_score += 0.40 * float(min(1.0, turn_count / 4.0))

        if path_length > reference_path_length + 18:
            route_quality_score -= 0.35

        if stateful_required:
            if stateful_distance_to_path is None:
                route_quality_score -= 1.15
            else:
                proximity_score = max(0.0, 1.0 - (float(stateful_distance_to_path) / 5.0))
                route_quality_score += 1.10 * proximity_score

                desired_low = 0.0 if gate_family in {"switch", "toggle"} else 1.0
                desired_high = 4.0 if gate_family in {"switch", "toggle"} else 6.0
                branch_gain = float(stateful_branch_gain or 0.0)
                if desired_low <= branch_gain <= desired_high:
                    route_quality_score += 0.60
                elif branch_gain < desired_low:
                    route_quality_score += max(0.0, 0.60 - (0.35 * (desired_low - branch_gain)))
                else:
                    route_quality_score += max(0.0, 0.60 - (0.10 * (branch_gain - desired_high)))

        return {
            "path_exists": 1,
            "path_length": int(path_length),
            "turn_count": int(turn_count),
            "route_overlap_ratio": float(route_overlap_ratio),
            "detour_gain": float(detour_gain),
            "stateful_distance_to_path": (
                int(stateful_distance_to_path) if stateful_distance_to_path is not None else None
            ),
            "stateful_via_path_length": (
                int(stateful_via_path_length) if stateful_via_path_length is not None else None
            ),
            "stateful_branch_gain": (
                float(stateful_branch_gain) if stateful_branch_gain is not None else None
            ),
            "stateful_on_path": int(stateful_on_path),
            "score": float(route_quality_score),
        }

    def _evaluate_puzzle_candidate_contract(
        self,
        *,
        grid: np.ndarray,
        gate_family: str,
        source_anchor: Tuple[int, int],
        destination_anchor: Tuple[int, int],
        stateful_anchor: Optional[Tuple[int, int]],
        route_quality: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate that a puzzle scaffold expresses a readable local interaction contract.

        The scaffold should do more than add clutter. For stateful gate families,
        the room should expose a walkable interaction pocket near the stateful
        anchor, frame it with some local structure, and keep it legibly connected
        to the main route according to the intended gate semantics.
        """
        gate_family = str(gate_family or "generic").strip().lower()
        walkable = self._build_room_walkable_mask(grid)
        stateful_required = gate_family in {"switch", "toggle", "bombable", "item_unlock", "key", "combat"}
        projected_stateful: Optional[Tuple[int, int]] = None
        if stateful_anchor is not None:
            projected_stateful = self._nearest_walkable_room_coord(walkable, stateful_anchor)

        pocket_floor_tiles = 0
        frame_block_tiles = 0
        anchor_adjacent_walkable = 0
        if projected_stateful is not None:
            anchor_r, anchor_c = self._clamp_room_coord(projected_stateful)
            for row in range(max(1, anchor_r - 1), min(ROOM_HEIGHT - 1, anchor_r + 2)):
                for col in range(max(1, anchor_c - 1), min(ROOM_WIDTH - 1, anchor_c + 2)):
                    if bool(walkable[row, col]):
                        pocket_floor_tiles += 1
            for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_r = anchor_r + d_r
                next_c = anchor_c + d_c
                if 0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH and bool(walkable[next_r, next_c]):
                    anchor_adjacent_walkable += 1
            for row in range(max(1, anchor_r - 2), min(ROOM_HEIGHT - 1, anchor_r + 3)):
                for col in range(max(1, anchor_c - 2), min(ROOM_WIDTH - 1, anchor_c + 3)):
                    if abs(row - anchor_r) <= 1 and abs(col - anchor_c) <= 1:
                        continue
                    if int(grid[row, col]) in {int(TileID.BLOCK), int(TileID.WALL)}:
                        frame_block_tiles += 1

        path_exists = int(route_quality.get("path_exists", 0) or 0)
        stateful_distance = route_quality.get("stateful_distance_to_path", None)
        stateful_branch_gain = float(route_quality.get("stateful_branch_gain", 0.0) or 0.0)

        failure_reasons: List[str] = []
        if path_exists <= 0:
            failure_reasons.append("no_path")
        if stateful_required:
            if projected_stateful is None:
                failure_reasons.append("missing_stateful_anchor")
            if pocket_floor_tiles < 5:
                failure_reasons.append("weak_stateful_pocket")
            if anchor_adjacent_walkable < 1:
                failure_reasons.append("sealed_stateful_anchor")
            if gate_family in {"switch", "toggle", "combat", "key"}:
                min_frame_blocks = 2 if gate_family == "combat" else 4
                if frame_block_tiles < min_frame_blocks:
                    failure_reasons.append("weak_local_structure")
            if stateful_distance is None or int(stateful_distance) > 3:
                failure_reasons.append("stateful_anchor_too_far")
            if gate_family in {"key"} and stateful_branch_gain < 1.0:
                failure_reasons.append("missing_stateful_detour")

        contract_score = 0.0
        contract_score += 0.30 if path_exists > 0 else -1.50
        contract_score += 0.25 if projected_stateful is not None else -0.80
        contract_score += 0.35 * float(min(1.0, pocket_floor_tiles / 6.0))
        contract_score += 0.35 * float(min(1.0, frame_block_tiles / 6.0))
        contract_score += 0.20 * float(min(1.0, anchor_adjacent_walkable / 2.0))
        if stateful_distance is not None:
            contract_score += 0.25 * float(max(0.0, 1.0 - (float(stateful_distance) / 4.0)))
        if gate_family in {"item_unlock", "key", "bombable"}:
            contract_score += 0.25 * float(min(1.0, stateful_branch_gain / 2.0))
        valid = int(len(failure_reasons) == 0)
        if not valid:
            contract_score -= 1.5 + (0.10 * len(failure_reasons))

        return {
            "valid": int(valid),
            "score": float(contract_score),
            "failure_reasons": list(failure_reasons),
            "stateful_anchor_present": int(projected_stateful is not None),
            "projected_stateful_anchor": (
                [int(projected_stateful[0]), int(projected_stateful[1])]
                if projected_stateful is not None else None
            ),
            "pocket_floor_tiles": int(pocket_floor_tiles),
            "frame_block_tiles": int(frame_block_tiles),
            "anchor_adjacent_walkable": int(anchor_adjacent_walkable),
            "stateful_distance_to_path": (
                int(stateful_distance) if stateful_distance is not None else None
            ),
            "stateful_branch_gain": float(stateful_branch_gain),
        }

    def _evaluate_puzzle_candidate_interaction_geometry(
        self,
        *,
        grid: np.ndarray,
        gate_family: str,
        source_anchor: Tuple[int, int],
        destination_anchor: Tuple[int, int],
        stateful_anchor: Optional[Tuple[int, int]],
        route_mask: np.ndarray,
        route_quality: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate whether a candidate exposes a readable local interaction grammar.

        The constructive scaffold should not only add obstacles. For stateful
        gate families, the candidate should imply a concrete local action:

        - `switch` / `toggle`: a push-style interaction near the state anchor
        - `bombable`: a visible seam/barrier with a bypassed route
        - `item_unlock` / `key`: a gated alcove or pocket around the anchor

        This remains a local structural proxy, not a full causal simulator, but
        it is much closer to puzzle intent than generic block density.
        """
        gate_family = str(gate_family or "generic").strip().lower()
        interaction_required = gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"}
        walkable = self._build_room_walkable_mask(grid)
        projected_stateful = (
            self._nearest_walkable_room_coord(walkable, stateful_anchor)
            if stateful_anchor is not None
            else None
        )
        if not interaction_required:
            return {
                "valid": 1,
                "score": 0.0,
                "failure_reasons": [],
                "required": 0,
                "projected_stateful_anchor": (
                    [int(projected_stateful[0]), int(projected_stateful[1])]
                    if projected_stateful is not None
                    else None
                ),
                "push_slot_count": 0,
                "anchor_openings": 0,
                "local_block_tiles": 0,
                "barrier_axis_tiles": 0,
                "route_divergence": 0.0,
            }

        failure_reasons: List[str] = []
        if projected_stateful is None:
            return {
                "valid": 0,
                "score": -2.0,
                "failure_reasons": ["missing_stateful_anchor"],
                "required": 1,
                "projected_stateful_anchor": None,
                "push_slot_count": 0,
                "anchor_openings": 0,
                "local_block_tiles": 0,
                "barrier_axis_tiles": 0,
                "route_divergence": 0.0,
            }

        anchor_r, anchor_c = self._clamp_room_coord(projected_stateful)
        route_arr = np.asarray(route_mask, dtype=bool)
        block_like = np.isin(
            np.asarray(grid, dtype=np.int32),
            np.array([int(TileID.BLOCK), int(TileID.WALL)], dtype=np.int32),
        )
        flow_is_horizontal = abs(int(destination_anchor[1]) - int(source_anchor[1])) >= abs(
            int(destination_anchor[0]) - int(source_anchor[0])
        )

        anchor_openings = 0
        for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_r = anchor_r + d_r
            next_c = anchor_c + d_c
            if 0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH and bool(walkable[next_r, next_c]):
                anchor_openings += 1

        local_block_tiles = 0
        for row in range(max(1, anchor_r - 2), min(ROOM_HEIGHT - 1, anchor_r + 3)):
            for col in range(max(1, anchor_c - 2), min(ROOM_WIDTH - 1, anchor_c + 3)):
                if bool(block_like[row, col]):
                    local_block_tiles += 1

        barrier_axis_tiles = 0
        if flow_is_horizontal:
            for row in range(max(1, anchor_r - 2), min(ROOM_HEIGHT - 1, anchor_r + 3)):
                if bool(block_like[row, anchor_c]):
                    barrier_axis_tiles += 1
        else:
            for col in range(max(1, anchor_c - 2), min(ROOM_WIDTH - 1, anchor_c + 3)):
                if bool(block_like[anchor_r, col]):
                    barrier_axis_tiles += 1

        push_slot_count = 0
        seen_push_blocks: Set[Tuple[int, int]] = set()
        for row in range(max(1, anchor_r - 3), min(ROOM_HEIGHT - 1, anchor_r + 4)):
            for col in range(max(1, anchor_c - 3), min(ROOM_WIDTH - 1, anchor_c + 4)):
                if int(grid[row, col]) != int(TileID.BLOCK):
                    continue
                for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    push_dest_r = row + d_r
                    push_dest_c = col + d_c
                    player_r = row - d_r
                    player_c = col - d_c
                    if not (
                        1 <= push_dest_r < ROOM_HEIGHT - 1
                        and 1 <= push_dest_c < ROOM_WIDTH - 1
                        and 1 <= player_r < ROOM_HEIGHT - 1
                        and 1 <= player_c < ROOM_WIDTH - 1
                    ):
                        continue
                    if not bool(walkable[push_dest_r, push_dest_c]):
                        continue
                    if not bool(walkable[player_r, player_c]):
                        continue
                    route_r0 = max(0, min(player_r, push_dest_r) - 1)
                    route_r1 = min(ROOM_HEIGHT, max(player_r, push_dest_r) + 2)
                    route_c0 = max(0, min(player_c, push_dest_c) - 1)
                    route_c1 = min(ROOM_WIDTH, max(player_c, push_dest_c) + 2)
                    if not bool(np.any(route_arr[route_r0:route_r1, route_c0:route_c1])):
                        continue
                    seen_push_blocks.add((int(row), int(col)))
                    break
        push_slot_count = int(len(seen_push_blocks))

        route_overlap_ratio = float(route_quality.get("route_overlap_ratio", 1.0) or 1.0)
        route_divergence = float(max(0.0, 1.0 - route_overlap_ratio))
        stateful_branch_gain = float(route_quality.get("stateful_branch_gain", 0.0) or 0.0)

        score = 0.0
        score += 0.25 * float(min(1.0, local_block_tiles / 5.0))
        score += 0.20 * float(min(1.0, barrier_axis_tiles / 3.0))

        if gate_family in {"switch", "toggle"}:
            score += 0.70 * float(min(1.0, push_slot_count / 1.0))
            if push_slot_count < 1:
                failure_reasons.append("missing_push_interaction")
            if local_block_tiles < 3:
                failure_reasons.append("weak_interaction_geometry")
            if gate_family == "toggle":
                score += 0.20 * float(min(1.0, barrier_axis_tiles / 2.0))
        elif gate_family == "bombable":
            score += 0.70 * float(min(1.0, route_divergence / 0.20))
            if barrier_axis_tiles < 1 and local_block_tiles < 3:
                failure_reasons.append("weak_bomb_seam")
            if route_divergence < 0.10:
                failure_reasons.append("missing_bypass_divergence")
        elif gate_family in {"item_unlock", "key"}:
            if local_block_tiles < 4:
                failure_reasons.append("weak_alcove_frame")
            if gate_family == "key" and stateful_branch_gain < 1.0:
                failure_reasons.append("missing_key_detour")
            score += 0.45 * float(min(1.0, local_block_tiles / 6.0))
            score += 0.30 * float(max(0.0, 1.0 - (abs(anchor_openings - 1) / 2.0)))

        valid = int(len(failure_reasons) == 0)
        if not valid:
            score -= 1.75 + (0.10 * len(failure_reasons))

        return {
            "valid": int(valid),
            "score": float(score),
            "failure_reasons": list(failure_reasons),
            "required": 1,
            "projected_stateful_anchor": [int(anchor_r), int(anchor_c)],
            "push_slot_count": int(push_slot_count),
            "anchor_openings": int(anchor_openings),
            "local_block_tiles": int(local_block_tiles),
            "barrier_axis_tiles": int(barrier_axis_tiles),
            "route_divergence": float(route_divergence),
        }

    def _puzzle_descriptor_distance(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> float:
        """Lightweight diversity distance for puzzle scaffold descriptors."""
        left_quadrants = list(left.get("quadrants", [0, 0, 0, 0]))
        right_quadrants = list(right.get("quadrants", [0, 0, 0, 0]))
        distance = 0.0
        distance += 1.0 if str(left.get("variant_name", "")) != str(right.get("variant_name", "")) else 0.0
        distance += 0.35 if str(left.get("gate_family", "")) != str(right.get("gate_family", "")) else 0.0
        distance += 0.20 if str(left.get("archetype", "")) != str(right.get("archetype", "")) else 0.0
        distance += abs(int(left.get("tiles_added", 0)) - int(right.get("tiles_added", 0))) / 24.0
        distance += abs(int(left.get("segments_added", 0)) - int(right.get("segments_added", 0))) / 6.0
        distance += abs(int(left.get("row_coverage", 0)) - int(right.get("row_coverage", 0))) / float(max(1, ROOM_HEIGHT - 2))
        distance += abs(int(left.get("col_coverage", 0)) - int(right.get("col_coverage", 0))) / float(max(1, ROOM_WIDTH - 2))
        distance += abs(float(left.get("center_row", ROOM_HEIGHT // 2)) - float(right.get("center_row", ROOM_HEIGHT // 2))) / float(max(1, ROOM_HEIGHT - 1))
        distance += abs(float(left.get("center_col", ROOM_WIDTH // 2)) - float(right.get("center_col", ROOM_WIDTH // 2))) / float(max(1, ROOM_WIDTH - 1))
        distance += sum(abs(int(a) - int(b)) for a, b in zip(left_quadrants, right_quadrants)) / 24.0
        return float(distance)

    def _score_puzzle_candidate(
        self,
        *,
        descriptor: Mapping[str, Any],
        stats: Mapping[str, Any],
        room_id: Any,
    ) -> float:
        """Score one scaffold candidate by structural quality plus novelty."""
        history = list(getattr(self, "_puzzle_novelty_history", []) or [])
        novelty_weight = float(max(0.0, min(2.0, float(getattr(self, "default_puzzle_room_novelty_weight", 0.45)))))
        memory_window = history[-8:]
        if memory_window:
            novelty_score = min(self._puzzle_descriptor_distance(descriptor, prev) for prev in memory_window)
        else:
            novelty_score = 1.0

        same_variant_count = sum(
            1 for prev in memory_window if str(prev.get("variant_name", "")) == str(descriptor.get("variant_name", ""))
        )
        same_family_count = sum(
            1 for prev in memory_window if str(prev.get("gate_family", "")) == str(descriptor.get("gate_family", ""))
        )
        current_gate_family = str(descriptor.get("gate_family", "") or "")
        current_variant_name = str(descriptor.get("variant_name", "") or "")
        same_family_variants = {
            str(prev.get("variant_name", "") or "")
            for prev in memory_window
            if str(prev.get("gate_family", "") or "") == current_gate_family
        }
        repeat_family_variant = current_variant_name in same_family_variants
        unseen_family_variant_bonus = 0.35 if same_family_variants and not repeat_family_variant else 0.0
        block_budget = max(1, int(stats.get("profile_block_budget", 1)))
        structural_score = min(1.25, float(stats.get("tiles_added", 0)) / float(max(8, block_budget // 2)))
        structural_score += min(0.75, float(stats.get("segments_added", 0)) / 4.0)
        structural_score += min(0.35, float(stats.get("optional_segments_applied", 0)) / 3.0)
        structural_score += 0.15 * (
            float(descriptor.get("row_coverage", 0)) / float(max(1, ROOM_HEIGHT - 2))
            + float(descriptor.get("col_coverage", 0)) / float(max(1, ROOM_WIDTH - 2))
        )
        route_quality_score = float(stats.get("route_quality_score", 0.0) or 0.0)
        contract_score = float(stats.get("contract_score", 0.0) or 0.0)
        contract_valid = int(stats.get("contract_valid", 0) or 0)
        interaction_score = float(stats.get("interaction_score", 0.0) or 0.0)
        interaction_valid = int(stats.get("interaction_valid", 0) or 0)
        interaction_sequence_score = float(stats.get("interaction_sequence_score", 0.0) or 0.0)
        interaction_sequence_valid = int(stats.get("interaction_sequence_valid", 0) or 0)
        interaction_sequence_required = int(stats.get("interaction_sequence_required", 0) or 0)
        density_ratio = float(stats.get("tiles_added", 0)) / float(max(1, block_budget))
        optional_segments_applied = int(stats.get("optional_segments_applied", 0) or 0)
        route_overlap_ratio = float(stats.get("route_quality_overlap_ratio", 0.0) or 0.0)
        detour_gain = float(stats.get("route_quality_detour_gain", 0.0) or 0.0)
        stateful_branch_gain = float(stats.get("route_quality_stateful_branch_gain", 0.0) or 0.0)
        gate_family = str(descriptor.get("gate_family", "") or "")

        clutter_penalty = max(0.0, density_ratio - 0.55) * 1.5
        clutter_penalty += max(0, optional_segments_applied - 1) * 0.18
        if route_overlap_ratio > 0.90 and detour_gain < 1.0:
            clutter_penalty += 0.45
        if gate_family in {"switch", "toggle", "item_unlock", "key"} and stateful_branch_gain <= 0.0:
            clutter_penalty += 0.60

        tie_break = float(
            stable_seed_offset((room_id, str(descriptor.get("variant_name", ""))), modulo=1000)
        ) / 1000.0
        return float(
            structural_score
            + route_quality_score
            + contract_score
            + interaction_score
            + interaction_sequence_score
            + (min(0.25, novelty_weight) * novelty_score)
            + unseen_family_variant_bonus
            - (1.05 if repeat_family_variant else 0.0)
            - (0.30 * float(same_variant_count))
            - (0.10 * float(same_family_count))
            - clutter_penalty
            - (2.25 if contract_valid <= 0 else 0.0)
            - (2.50 if interaction_valid <= 0 and gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"} else 0.0)
            - (2.10 if interaction_sequence_required > 0 and interaction_sequence_valid <= 0 else 0.0)
            + (tie_break * 1e-3)
        )

    def _commit_puzzle_novelty_choice(
        self,
        *,
        room_id: Any,
        scaffold_stats: Mapping[str, Any],
    ) -> None:
        """Remember the selected scaffold descriptor once per room for later novelty scoring."""
        committed = getattr(self, "_puzzle_novelty_committed", None)
        if not isinstance(committed, set):
            self._puzzle_novelty_committed = set()
            committed = self._puzzle_novelty_committed
        if room_id in committed:
            return
        descriptor = scaffold_stats.get("novelty_descriptor")
        if isinstance(descriptor, dict) and descriptor:
            history = getattr(self, "_puzzle_novelty_history", None)
            if not isinstance(history, list):
                self._puzzle_novelty_history = []
                history = self._puzzle_novelty_history
            history.append(dict(descriptor))
        committed.add(room_id)

    def _build_puzzle_room_segments(
        self,
        *,
        archetype: str,
        gate_family: str,
        variant_spec: Optional[Mapping[str, Any]] = None,
        stateful_anchor: Optional[Tuple[int, int]],
        flow_is_horizontal: bool,
        puzzle_anchor: Tuple[int, int],
    ) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        """
        Return required and optional puzzle scaffold segments.

        Each segment is a list of room coordinates that will be painted as BLOCK
        tiles when they do not collide with reserved route/anchor cells.
        """
        center_r = max(3, min(ROOM_HEIGHT - 4, int(puzzle_anchor[0])))
        center_c = max(3, min(ROOM_WIDTH - 4, int(puzzle_anchor[1])))
        if stateful_anchor is not None:
            stateful_r = max(3, min(ROOM_HEIGHT - 4, int(stateful_anchor[0])))
            stateful_c = max(3, min(ROOM_WIDTH - 4, int(stateful_anchor[1])))
        else:
            stateful_r = center_r
            stateful_c = center_c
        left_col = 3
        right_col = ROOM_WIDTH - 4
        top_row = 3
        bottom_row = ROOM_HEIGHT - 4
        resource_offset = int(max(1, getattr(self, "default_puzzle_room_resource_bypass_offset", 2)))
        key_depth = int(max(1, getattr(self, "default_puzzle_room_key_pocket_depth", 3)))
        item_slot_depth = int(max(1, getattr(self, "default_puzzle_room_item_slot_depth", 3)))
        toggle_offset = int(max(1, getattr(self, "default_puzzle_room_toggle_corridor_offset", 2)))
        variant = dict(variant_spec or {})
        variant_style = str(variant.get("style", "baseline") or "baseline").strip().lower()
        variant_side_bias = int(max(-1, min(1, int(variant.get("side_bias", 0) or 0))))

        required: List[List[Tuple[int, int]]] = []
        optional: List[List[Tuple[int, int]]] = []

        if archetype == "gate":
            if flow_is_horizontal:
                gate_col = max(left_col + 1, min(right_col - 1, center_c))
                if gate_family == "bombable":
                    bypass_row = max(
                        2,
                        min(
                            ROOM_HEIGHT - 3,
                            stateful_r + (-resource_offset if stateful_r > ROOM_HEIGHT // 2 else resource_offset),
                        ),
                    )
                    gap_rows = {bypass_row}
                elif gate_family == "toggle":
                    gap_rows = {stateful_r}
                elif gate_family == "key" and stateful_anchor is not None:
                    gap_rows = {center_r}
                else:
                    gap_rows = {max(2, center_r - 1), center_r, min(ROOM_HEIGHT - 3, center_r + 1)}
                required.append([(row, gate_col) for row in range(2, ROOM_HEIGHT - 2) if row not in gap_rows])
                pocket_side = variant_side_bias if variant_side_bias != 0 else (-1 if center_r <= ROOM_HEIGHT // 2 else 1)
                pocket_row = max(2, min(ROOM_HEIGHT - 3, center_r + pocket_side * 2))
                if gate_family == "switch":
                    required.append([(stateful_r, col) for col in range(max(2, gate_col - 3), min(ROOM_WIDTH - 2, gate_col + 1))])
                    required.append([(row, max(2, gate_col - 3)) for row in range(min(center_r, stateful_r), max(center_r, stateful_r) + 1)])
                    if variant_style == "bridge":
                        required.append([(row, min(ROOM_WIDTH - 3, gate_col + 2)) for row in range(3, ROOM_HEIGHT - 3) if abs(row - center_r) > 0])
                    optional.append([(row, min(ROOM_WIDTH - 3, gate_col + 2)) for row in range(3, ROOM_HEIGHT - 3) if abs(row - center_r) > 1])
                elif gate_family == "toggle":
                    corridor_top = max(2, stateful_r - toggle_offset)
                    corridor_bottom = min(ROOM_HEIGHT - 3, stateful_r + toggle_offset)
                    required.append([(corridor_top, col) for col in range(max(2, gate_col - 3), min(ROOM_WIDTH - 2, gate_col + 2))])
                    required.append([(corridor_bottom, col) for col in range(max(2, gate_col - 3), min(ROOM_WIDTH - 2, gate_col + 2))])
                    if variant_style == "weave":
                        required.append([(row, max(2, gate_col - 1)) for row in range(corridor_top + 1, corridor_bottom)])
                    optional.append([(row, max(2, gate_col - 2)) for row in range(corridor_top + 1, corridor_bottom)])
                elif gate_family == "bombable":
                    bypass_row = max(
                        2,
                        min(
                            ROOM_HEIGHT - 3,
                            stateful_r + (-resource_offset if stateful_r > ROOM_HEIGHT // 2 else resource_offset),
                        ),
                    )
                    resource_row = stateful_r
                    required.append([(resource_row, col) for col in range(max(2, gate_col - 4), max(3, gate_col - 1))])
                    required.append([(row, max(2, gate_col - 4)) for row in range(min(resource_row, bypass_row), max(resource_row, bypass_row) + 1)])
                    if variant_style == "wrap":
                        required.append([(bypass_row, col) for col in range(max(2, gate_col - 1), min(ROOM_WIDTH - 2, gate_col + 4)) if col != gate_col])
                    optional.append([(row, max(2, gate_col - 2)) for row in range(min(resource_row, bypass_row), max(resource_row, bypass_row) + 1)])
                    optional.append([(bypass_row, col) for col in range(min(gate_col + 1, ROOM_WIDTH - 3), min(ROOM_WIDTH - 2, gate_col + 4))])
                elif gate_family == "item_unlock":
                    item_row = stateful_r
                    item_col = max(gate_col + 2, min(ROOM_WIDTH - 3, stateful_c))
                    left_col = max(2, item_col - 1)
                    right_col = min(ROOM_WIDTH - 3, item_col + 1)
                    top_row = max(2, item_row - 1)
                    bottom_row = min(ROOM_HEIGHT - 3, item_row + 1)
                    required.append([(row, left_col) for row in range(top_row, bottom_row + 1) if row != item_row])
                    required.append([(row, right_col) for row in range(top_row, bottom_row + 1) if row != item_row])
                    required.append([(top_row, col) for col in range(left_col, right_col + 1) if col != item_col])
                    required.append([(bottom_row, col) for col in range(left_col, right_col + 1) if col != item_col])
                    if variant_style == "ring":
                        required.append([(item_row, left_col)])
                        required.append([(item_row, right_col)])
                    optional.append([(center_r - 1, col) for col in range(min(gate_col + 1, item_col), max(gate_col + 1, item_col) + 1)])
                    optional.append([(center_r + 1, col) for col in range(min(gate_col + 1, item_col), max(gate_col + 1, item_col) + 1)])
                elif gate_family == "key" and stateful_anchor is not None:
                    key_row = stateful_r
                    required.append([(key_row, col) for col in range(max(2, gate_col - key_depth - 1), max(3, gate_col - 1))])
                    required.append([(row, max(2, gate_col - key_depth - 1)) for row in range(min(center_r, key_row), max(center_r, key_row) + 1)])
                    if variant_style == "split":
                        optional.append([(row, max(2, gate_col - key_depth + 1)) for row in range(min(center_r, key_row), max(center_r, key_row) + 1)])
                    optional.append([(row, min(ROOM_WIDTH - 3, gate_col + 2)) for row in range(3, ROOM_HEIGHT - 3) if abs(row - center_r) > 1])
                else:
                    required.append([(pocket_row, col) for col in range(max(2, gate_col - 2), min(ROOM_WIDTH - 2, gate_col + 1))])
                    optional.append([(row, max(2, gate_col - 2)) for row in range(min(center_r, pocket_row), max(center_r, pocket_row) + 1)])
                    optional.append([(row, min(ROOM_WIDTH - 3, gate_col + 2)) for row in range(3, ROOM_HEIGHT - 3) if abs(row - center_r) > 1])
            else:
                gate_row = max(top_row + 1, min(bottom_row - 1, center_r))
                if gate_family == "bombable":
                    bypass_col = max(
                        2,
                        min(
                            ROOM_WIDTH - 3,
                            stateful_c + (-resource_offset if stateful_c > ROOM_WIDTH // 2 else resource_offset),
                        ),
                    )
                    gap_cols = {bypass_col}
                elif gate_family == "toggle":
                    gap_cols = {stateful_c}
                elif gate_family == "key" and stateful_anchor is not None:
                    gap_cols = {center_c}
                else:
                    gap_cols = {max(2, center_c - 1), center_c, min(ROOM_WIDTH - 3, center_c + 1)}
                required.append([(gate_row, col) for col in range(2, ROOM_WIDTH - 2) if col not in gap_cols])
                pocket_side = variant_side_bias if variant_side_bias != 0 else (-1 if center_c <= ROOM_WIDTH // 2 else 1)
                pocket_col = max(2, min(ROOM_WIDTH - 3, center_c + pocket_side * 2))
                if gate_family == "switch":
                    required.append([(row, stateful_c) for row in range(max(2, gate_row - 3), min(ROOM_HEIGHT - 2, gate_row + 1))])
                    required.append([(max(2, gate_row - 3), col) for col in range(min(center_c, stateful_c), max(center_c, stateful_c) + 1)])
                    if variant_style == "bridge":
                        required.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(3, ROOM_WIDTH - 3) if abs(col - center_c) > 0])
                    optional.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(3, ROOM_WIDTH - 3) if abs(col - center_c) > 1])
                elif gate_family == "toggle":
                    corridor_left = max(2, stateful_c - toggle_offset)
                    corridor_right = min(ROOM_WIDTH - 3, stateful_c + toggle_offset)
                    required.append([(row, corridor_left) for row in range(max(2, gate_row - 3), min(ROOM_HEIGHT - 2, gate_row + 2))])
                    required.append([(row, corridor_right) for row in range(max(2, gate_row - 3), min(ROOM_HEIGHT - 2, gate_row + 2))])
                    if variant_style == "weave":
                        required.append([(max(2, gate_row - 1), col) for col in range(corridor_left + 1, corridor_right)])
                    optional.append([(max(2, gate_row - 2), col) for col in range(corridor_left + 1, corridor_right)])
                elif gate_family == "bombable":
                    bypass_col = max(
                        2,
                        min(
                            ROOM_WIDTH - 3,
                            stateful_c + (-resource_offset if stateful_c > ROOM_WIDTH // 2 else resource_offset),
                        ),
                    )
                    resource_col = stateful_c
                    ledge_row = max(2, min(ROOM_HEIGHT - 3, gate_row - 2))
                    required.append([(row, resource_col) for row in range(max(2, gate_row - 4), max(3, gate_row - 1))])
                    required.append([(max(2, gate_row - 4), col) for col in range(min(resource_col, bypass_col), max(resource_col, bypass_col) + 1)])
                    required.append([(ledge_row, col) for col in range(min(resource_col, bypass_col), max(resource_col, bypass_col) + 1)])
                    if variant_style == "wrap":
                        required.append([(row, bypass_col) for row in range(max(2, gate_row - 1), min(ROOM_HEIGHT - 2, gate_row + 4)) if row != gate_row])
                        required.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(min(resource_col, bypass_col), max(resource_col, bypass_col) + 1)])
                    optional.append([(max(2, gate_row - 2), col) for col in range(min(resource_col, bypass_col), max(resource_col, bypass_col) + 1)])
                    optional.append([(row, bypass_col) for row in range(min(gate_row + 1, ROOM_HEIGHT - 3), min(ROOM_HEIGHT - 2, gate_row + 4))])
                elif gate_family == "item_unlock":
                    item_row = max(gate_row + 2, min(ROOM_HEIGHT - 3, stateful_r))
                    item_col = stateful_c
                    top_row = max(2, item_row - 1)
                    bottom_row = min(ROOM_HEIGHT - 3, item_row + 1)
                    left_col = max(2, item_col - 1)
                    right_col = min(ROOM_WIDTH - 3, item_col + 1)
                    required.append([(row, left_col) for row in range(top_row, bottom_row + 1) if row != item_row])
                    required.append([(row, right_col) for row in range(top_row, bottom_row + 1) if row != item_row])
                    required.append([(top_row, col) for col in range(left_col, right_col + 1) if col != item_col])
                    required.append([(bottom_row, col) for col in range(left_col, right_col + 1) if col != item_col])
                    if variant_style == "ring":
                        required.append([(top_row, item_col)])
                        required.append([(bottom_row, item_col)])
                    optional.append([(row, center_c - 1) for row in range(min(gate_row + 1, item_row), max(gate_row + 1, item_row) + 1)])
                    optional.append([(row, center_c + 1) for row in range(min(gate_row + 1, item_row), max(gate_row + 1, item_row) + 1)])
                elif gate_family == "key" and stateful_anchor is not None:
                    key_col = stateful_c
                    required.append([(row, key_col) for row in range(max(2, gate_row - key_depth - 1), max(3, gate_row - 1))])
                    required.append([(max(2, gate_row - key_depth - 1), col) for col in range(min(center_c, key_col), max(center_c, key_col) + 1)])
                    if variant_style == "split":
                        optional.append([(max(2, gate_row - key_depth + 1), col) for col in range(min(center_c, key_col), max(center_c, key_col) + 1)])
                    optional.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(3, ROOM_WIDTH - 3) if abs(col - center_c) > 1])
                else:
                    required.append([(row, pocket_col) for row in range(max(2, gate_row - 2), min(ROOM_HEIGHT - 2, gate_row + 1))])
                    optional.append([(max(2, gate_row - 2), col) for col in range(min(center_c, pocket_col), max(center_c, pocket_col) + 1)])
                    optional.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(3, ROOM_WIDTH - 3) if abs(col - center_c) > 1])
        elif archetype == "hub":
            top = max(2, center_r - 3)
            bottom = min(ROOM_HEIGHT - 3, center_r + 3)
            left = max(2, center_c - 3)
            right = min(ROOM_WIDTH - 3, center_c + 3)
            required.append([(top, col) for col in range(left, right + 1) if abs(col - center_c) > 1])
            required.append([(bottom, col) for col in range(left, right + 1) if abs(col - center_c) > 1])
            required.append([(row, left) for row in range(top, bottom + 1) if abs(row - center_r) > 1])
            required.append([(row, right) for row in range(top, bottom + 1) if abs(row - center_r) > 1])
            if variant_style == "cross":
                required.append([(center_r, col) for col in range(left + 1, right) if abs(col - center_c) > 0])
                required.append([(row, center_c) for row in range(top + 1, bottom) if abs(row - center_r) > 0])
            elif variant_style == "offset":
                offset_row = max(2, min(ROOM_HEIGHT - 3, center_r + variant_side_bias * 2))
                required.append([(offset_row, col) for col in range(left + 1, right) if abs(col - center_c) > 1])
            optional.append([(center_r - 2, col) for col in range(left + 1, center_c - 1)])
            optional.append([(center_r - 2, col) for col in range(center_c + 2, right)])
            optional.append([(center_r + 2, col) for col in range(left + 1, center_c - 1)])
            optional.append([(center_r + 2, col) for col in range(center_c + 2, right)])
        elif archetype == "island":
            if variant_style == "staggered":
                optional.extend(
                    [
                        [(center_r - 3, center_c - 2), (center_r - 3, center_c - 1), (center_r - 2, center_c - 2)],
                        [(center_r - 1, center_c + 1), (center_r, center_c + 1), (center_r, center_c + 2)],
                        [(center_r + 2, center_c - 1), (center_r + 3, center_c - 2), (center_r + 3, center_c - 1)],
                        [(center_r + 1, center_c + 2), (center_r + 2, center_c + 1), (center_r + 2, center_c + 2)],
                    ]
                )
            else:
                optional.extend(
                    [
                        [(center_r - 3, center_c - 2), (center_r - 3, center_c - 1), (center_r - 2, center_c - 2), (center_r - 2, center_c - 1)],
                        [(center_r - 1, center_c + 1), (center_r - 1, center_c + 2), (center_r, center_c + 1), (center_r, center_c + 2)],
                        [(center_r + 2, center_c - 2), (center_r + 2, center_c - 1), (center_r + 3, center_c - 2), (center_r + 3, center_c - 1)],
                        [(center_r + 1, center_c + 1), (center_r + 1, center_c + 2), (center_r + 2, center_c + 1), (center_r + 2, center_c + 2)],
                    ]
                )
            if variant_style == "bridge":
                optional.append([(center_r + variant_side_bias * 2, col) for col in range(left_col + 1, right_col) if abs(col - center_c) > 1])
            if flow_is_horizontal:
                required.append([(center_r, col) for col in range(left_col + 1, right_col) if abs(col - center_c) > 2])
            else:
                required.append([(row, center_c) for row in range(top_row + 1, bottom_row) if abs(row - center_r) > 2])
        elif archetype == "combat":
            if variant_style == "corners":
                optional.extend(
                    [
                        [(center_r - 3, center_c - 2), (center_r - 3, center_c - 1), (center_r - 2, center_c - 2), (center_r - 2, center_c - 1)],
                        [(center_r - 3, center_c + 1), (center_r - 3, center_c + 2), (center_r - 2, center_c + 1), (center_r - 2, center_c + 2)],
                        [(center_r + 2, center_c - 2), (center_r + 2, center_c - 1), (center_r + 3, center_c - 2), (center_r + 3, center_c - 1)],
                        [(center_r + 2, center_c + 1), (center_r + 2, center_c + 2), (center_r + 3, center_c + 1), (center_r + 3, center_c + 2)],
                    ]
                )
            else:
                optional.extend(
                    [
                        [(center_r - 2, center_c - 1), (center_r - 2, center_c), (center_r - 1, center_c - 1)],
                        [(center_r - 2, center_c + 1), (center_r - 2, center_c + 2), (center_r - 1, center_c + 2)],
                        [(center_r + 1, center_c - 1), (center_r + 2, center_c - 1), (center_r + 2, center_c)],
                        [(center_r + 1, center_c + 2), (center_r + 2, center_c + 1), (center_r + 2, center_c + 2)],
                    ]
                )
            if variant_style == "cross":
                required.append([(center_r - 3, center_c), (center_r - 2, center_c)])
                required.append([(center_r + 2, center_c), (center_r + 3, center_c)])
            required.append([(center_r, center_c - 3), (center_r, center_c - 2)])
            required.append([(center_r, center_c + 2), (center_r, center_c + 3)])
        else:  # serpentine
            if flow_is_horizontal:
                rows = [3, 6, 9, 12] if variant_style != "split" else [4, 8, 12]
                for idx, row in enumerate(rows):
                    if row >= ROOM_HEIGHT - 2:
                        continue
                    gap_on_left = ((idx % 2) == 0) if variant_style != "mirror" else ((idx % 2) != 0)
                    segment = [
                        (row, col)
                        for col in range(2, ROOM_WIDTH - 2)
                        if not (2 <= col <= 4 and gap_on_left)
                        and not (ROOM_WIDTH - 5 <= col <= ROOM_WIDTH - 3 and not gap_on_left)
                    ]
                    if idx < 2:
                        required.append(segment)
                    else:
                        optional.append(segment)
                optional.append([(center_r + variant_side_bias, center_c - 1), (center_r + variant_side_bias, center_c + 1)])
            else:
                cols = [3, 5, 7] if variant_style != "split" else [3, 6, 8]
                for idx, col in enumerate(cols):
                    if col >= ROOM_WIDTH - 2:
                        continue
                    gap_on_top = ((idx % 2) == 0) if variant_style != "mirror" else ((idx % 2) != 0)
                    segment = [
                        (row, col)
                        for row in range(2, ROOM_HEIGHT - 2)
                        if not (2 <= row <= 4 and gap_on_top)
                        and not (ROOM_HEIGHT - 5 <= row <= ROOM_HEIGHT - 3 and not gap_on_top)
                    ]
                    if idx < 2:
                        required.append(segment)
                    else:
                        optional.append(segment)
                optional.append([(center_r - 1, center_c + variant_side_bias), (center_r + 1, center_c + variant_side_bias)])

        return required, optional

    def _strip_small_interior_structure_components(
        self,
        grid: np.ndarray,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
        max_component_tiles: int = 6,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Remove tiny isolated interior wall/block islands that read as noise."""
        out = np.asarray(grid, dtype=np.int32).copy()
        wall_like_mask = np.isin(out, np.array([int(TileID.WALL), int(TileID.BLOCK)], dtype=np.int32))
        allowed_door_mask = self._required_room_door_slots_mask(graph=graph, room_id=room_id)
        visited = np.zeros_like(wall_like_mask, dtype=bool)
        removed_components = 0
        removed_tiles = 0

        for row in range(ROOM_HEIGHT):
            for col in range(ROOM_WIDTH):
                if not bool(wall_like_mask[row, col]) or bool(visited[row, col]):
                    continue

                component: List[Tuple[int, int]] = []
                stack: List[Tuple[int, int]] = [(row, col)]
                visited[row, col] = True
                touches_boundary = False
                touches_allowed_door = False

                while stack:
                    cur_r, cur_c = stack.pop()
                    component.append((cur_r, cur_c))
                    if cur_r in {0, ROOM_HEIGHT - 1} or cur_c in {0, ROOM_WIDTH - 1}:
                        touches_boundary = True
                    if bool(allowed_door_mask[cur_r, cur_c]):
                        touches_allowed_door = True
                    for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        next_r = cur_r + d_r
                        next_c = cur_c + d_c
                        if not (0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH):
                            continue
                        if not bool(wall_like_mask[next_r, next_c]) or bool(visited[next_r, next_c]):
                            continue
                        visited[next_r, next_c] = True
                        stack.append((next_r, next_c))

                if touches_boundary or touches_allowed_door or len(component) > int(max_component_tiles):
                    continue
                for comp_r, comp_c in component:
                    out[comp_r, comp_c] = int(TileID.FLOOR)
                removed_components += 1
                removed_tiles += len(component)

        return out, {
            "removed_components": int(removed_components),
            "removed_tiles": int(removed_tiles),
        }

    def _apply_puzzle_room_scaffold(
        self,
        grid: np.ndarray,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
        room_plan_mask: Optional[np.ndarray] = None,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Inject a lightweight constructive scaffold into under-structured puzzle rooms.

        Research-backed room-generation systems tend to keep structure explicit for
        constrained layouts instead of expecting small-data generators to discover
        reliable puzzle geometry on their own. We follow the same pragmatic
        hybrid approach here: preserve the planned traversability route, then add a
        small deterministic block-maze scaffold only when the current room is still
        overly empty.
        """
        out = np.asarray(grid, dtype=np.int32).copy()
        stats: Dict[str, Any] = {
            "applied": 0,
            "tiles_added": 0,
            "segments_added": 0,
            "existing_structure_tiles": 0,
            "planned_route_pixels": 0,
        }
        if not bool(self.default_puzzle_room_scaffold_enabled):
            return out, stats
        if not isinstance(graph, nx.Graph) or room_id not in graph:
            return out, stats

        attrs = dict(graph.nodes[room_id])
        role_flags = self._room_role_flags(attrs)
        semantics = self._extract_room_topology_semantics(graph, room_id)
        has_puzzle_gate = any(
            {"switch", "switch_locked", "state_block", "on_off_gate", "puzzle"} & set(tokens)
            for tokens in semantics["edge_constraints"].values()
        )
        node_type = str(
            attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or ""
        ).strip().lower()
        if not (
            role_flags.get("has_puzzle", False)
            or has_puzzle_gate
            or node_type in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"}
        ):
            return out, stats

        out, structure_cleanup = self._strip_small_interior_structure_components(
            out,
            graph=graph,
            room_id=room_id,
        )
        stats["noise_components_removed"] = int(structure_cleanup["removed_components"])
        stats["noise_tiles_removed"] = int(structure_cleanup["removed_tiles"])

        block_id = int(TileID.BLOCK)
        wall_id = int(TileID.WALL)
        floor_id = int(TileID.FLOOR)
        structure_mask = np.isin(out, np.array([wall_id, block_id], dtype=np.int32))
        interior_mask = np.zeros_like(out, dtype=bool)
        interior_mask[2:ROOM_HEIGHT - 2, 2:ROOM_WIDTH - 2] = True
        existing_structure_tiles = int(np.sum(structure_mask & interior_mask))
        stats["existing_structure_tiles"] = existing_structure_tiles
        if (
            existing_structure_tiles >= int(self.default_puzzle_room_scaffold_min_structure_tiles)
            and int(structure_cleanup["removed_components"]) <= 0
        ):
            return out, stats

        normalized_start_goal = start_goal
        if normalized_start_goal is None:
            normalized_start_goal = self._extract_room_start_goal(graph, room_id)
        if normalized_start_goal is None:
            normalized_start_goal = ((ROOM_HEIGHT // 2, 1), (ROOM_HEIGHT // 2, ROOM_WIDTH - 2))
        start_coord, goal_coord = self._normalize_start_goal_coords(normalized_start_goal)

        if isinstance(room_plan_mask, np.ndarray) and room_plan_mask.shape == (ROOM_HEIGHT, ROOM_WIDTH):
            route_mask = np.asarray(room_plan_mask, dtype=np.float32) > 0.0
        else:
            try:
                route_mask = self._build_room_plan_trace(
                    graph,
                    room_id,
                    out,
                    start_goal=(start_coord, goal_coord),
                ) > 0.0
            except Exception:
                route_mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
        stats["planned_route_pixels"] = int(np.sum(route_mask))
        scaffold_profile = self._resolve_puzzle_room_scaffold_profile(
            attrs=attrs,
            role_flags=role_flags,
            semantics=semantics,
            node_type=node_type,
        )
        preserve_margin = int(max(0, scaffold_profile.get("preserve_route_margin", getattr(self, "default_puzzle_room_preserve_route_margin", 0))))

        semantic_anchors = build_room_semantic_anchor_points(
            room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
            start=start_coord,
            goal=goal_coord,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            room_role_flags=role_flags,
            semantic_puzzle_offset=self.default_semantic_puzzle_offset,
        )

        planned_markers = self._plan_room_graph_marker_layout(
            out,
            graph=graph,
            room_id=room_id,
            start_goal=(start_coord, goal_coord),
        )

        puzzle_anchor = semantic_anchors.get("puzzle", (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
        source_anchor = semantic_anchors.get("start", start_coord)
        destination_anchor = semantic_anchors.get("goal", goal_coord)
        flow_is_horizontal = abs(int(destination_anchor[1]) - int(source_anchor[1])) >= abs(
            int(destination_anchor[0]) - int(source_anchor[0])
        )
        archetype = str(scaffold_profile.get("archetype", "serpentine") or "serpentine").strip().lower()
        gate_family = str(scaffold_profile.get("gate_family", "generic") or "generic").strip().lower()
        if gate_family == "switch":
            stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
        elif gate_family == "toggle":
            stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
        elif gate_family == "bombable":
            stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
        elif gate_family == "item_unlock":
            stateful_anchor_name = "item" if "item" in semantic_anchors else ("puzzle" if "puzzle" in semantic_anchors else None)
        elif gate_family == "key":
            stateful_anchor_name = "key" if "key" in semantic_anchors else None
        elif gate_family == "combat":
            stateful_anchor_name = "enemy" if "enemy" in semantic_anchors else None
        else:
            stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
        stateful_anchor = semantic_anchors.get(stateful_anchor_name) if stateful_anchor_name is not None else None
        interaction_sequence = self._resolve_puzzle_interaction_sequence(
            archetype=archetype,
            gate_family=gate_family,
            role_flags=role_flags,
            semantic_anchors=semantic_anchors,
        )
        variant_specs = self._build_puzzle_room_variant_specs(
            archetype=archetype,
            gate_family=gate_family,
        )
        variant_cache = getattr(self, "_puzzle_variant_cache", None)
        cached_variant = variant_cache.get(room_id) if isinstance(variant_cache, dict) else None
        base_grid = out.copy()
        baseline_path_metrics = self._evaluate_puzzle_candidate_route_quality(
            grid=base_grid,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            stateful_anchor=stateful_anchor,
            route_mask=route_mask,
            gate_family=gate_family,
            baseline_path_length=None,
        )
        baseline_path_length = (
            int(baseline_path_metrics.get("path_length", 0))
            if int(baseline_path_metrics.get("path_exists", 0) or 0) > 0
            else None
        )

        def _mark_reserved(route_mask_candidate: np.ndarray) -> np.ndarray:
            reserved_candidate = (
                self._dilate_room_mask(route_mask_candidate, radius=preserve_margin)
                if preserve_margin > 0 else route_mask_candidate.copy()
            )
            for point in semantic_anchors.values():
                rr, cc = self._clamp_room_coord(point)
                reserved_candidate[int(rr), int(cc)] = True
            for _tile_id, slot in planned_markers:
                rr, cc = self._clamp_room_coord(slot)
                reserved_candidate[int(rr), int(cc)] = True
            for direction, enabled in semantics["required_doors"].items():
                if not bool(enabled):
                    continue
                spec = DOOR_POSITIONS[str(direction)]
                if direction in {"N", "S"}:
                    apron_row = 2 if direction == "N" else ROOM_HEIGHT - 3
                    c0 = int(spec["col_start"])
                    c1 = int(spec["col_end"]) + 1
                    reserved_candidate[apron_row, c0:c1] = True
                else:
                    apron_col = 2 if direction == "W" else ROOM_WIDTH - 3
                    r0 = int(spec["row_start"])
                    r1 = int(spec["row_end"]) + 1
                    reserved_candidate[r0:r1, apron_col] = True
            return reserved_candidate

        def _render_candidate(variant_spec: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
            candidate_grid = base_grid.copy()
            route_candidate = self._build_puzzle_room_route_template(
                archetype=archetype,
                gate_family=gate_family,
                variant_spec=variant_spec,
                stateful_anchor=stateful_anchor,
                interaction_sequence=interaction_sequence,
                flow_is_horizontal=flow_is_horizontal,
                source_anchor=source_anchor,
                destination_anchor=destination_anchor,
                puzzle_anchor=puzzle_anchor,
                role_flags=role_flags,
                semantics=semantics,
            )
            if bool(np.any(route_candidate)):
                route_mask_candidate = route_candidate
                route_template_used = 1
            else:
                route_mask_candidate = route_mask.copy()
                route_template_used = 0
            reserved_candidate = _mark_reserved(route_mask_candidate)

            branch_density = float(
                max(
                    0.0,
                    min(
                        1.0,
                        float(scaffold_profile.get("branch_density", getattr(self, "default_puzzle_room_branch_density", 0.75)))
                        + float(variant_spec.get("branch_density_delta", 0.0)),
                    ),
                )
            )
            block_budget = int(
                max(
                    0,
                    int(scaffold_profile.get("block_budget", getattr(self, "default_puzzle_room_block_budget", 28)))
                    + int(variant_spec.get("block_budget_delta", 0)),
                )
            )
            budget_remaining = int(block_budget)

            def _can_place(row: int, col: int) -> bool:
                if not (2 <= int(row) <= ROOM_HEIGHT - 3 and 2 <= int(col) <= ROOM_WIDTH - 3):
                    return False
                if bool(reserved_candidate[int(row), int(col)]):
                    return False
                return int(candidate_grid[int(row), int(col)]) == floor_id

            def _paint_block_line(points: List[Tuple[int, int]]) -> int:
                nonlocal budget_remaining
                added = 0
                for row, col in points:
                    if budget_remaining <= 0:
                        break
                    if _can_place(int(row), int(col)):
                        candidate_grid[int(row), int(col)] = block_id
                        added += 1
                        budget_remaining -= 1
                return int(added)

            def _apply_anchor_template(
                *,
                anchor: Optional[Tuple[int, int]],
                open_toward: Tuple[int, int],
            ) -> Tuple[int, int]:
                """
                Force a small readable puzzle pocket around the interaction anchor.

                This is the local grammar layer: before generic stochastic
                scaffold lines are added, stabilize the interaction site as a
                floor pocket framed by blocks with one intentional opening.
                """
                if anchor is None:
                    return 0, 0
                anchor_r, anchor_c = self._clamp_room_coord(anchor)
                open_r, open_c = self._clamp_room_coord(open_toward)

                pocket_tiles_forced = 0
                frame_tiles_added = 0
                row_start = max(2, int(anchor_r) - 1)
                row_end = min(ROOM_HEIGHT - 3, int(anchor_r) + 1)
                col_start = max(2, int(anchor_c) - 1)
                col_end = min(ROOM_WIDTH - 3, int(anchor_c) + 1)

                for row in range(row_start, row_end + 1):
                    for col in range(col_start, col_end + 1):
                        if int(candidate_grid[row, col]) != floor_id:
                            candidate_grid[row, col] = floor_id
                            pocket_tiles_forced += 1
                        reserved_candidate[row, col] = True

                delta_r = int(open_r) - int(anchor_r)
                delta_c = int(open_c) - int(anchor_c)
                if abs(delta_c) >= abs(delta_r):
                    opening_side = "E" if delta_c >= 0 else "W"
                else:
                    opening_side = "S" if delta_r >= 0 else "N"

                frame_cells: List[Tuple[int, int]] = []
                for row in range(row_start - 1, row_end + 2):
                    for col in range(col_start - 1, col_end + 2):
                        if not (2 <= row <= ROOM_HEIGHT - 3 and 2 <= col <= ROOM_WIDTH - 3):
                            continue
                        on_border = row in {row_start - 1, row_end + 1} or col in {col_start - 1, col_end + 1}
                        if not on_border:
                            continue
                        if opening_side == "N" and row == row_start - 1 and col == int(anchor_c):
                            continue
                        if opening_side == "S" and row == row_end + 1 and col == int(anchor_c):
                            continue
                        if opening_side == "W" and col == col_start - 1 and row == int(anchor_r):
                            continue
                        if opening_side == "E" and col == col_end + 1 and row == int(anchor_r):
                            continue
                        frame_cells.append((int(row), int(col)))

                frame_tiles_added += _paint_block_line(frame_cells)
                return int(pocket_tiles_forced), int(frame_tiles_added)

            def _place_push_block_prop() -> int:
                """
                Add one isolated BLOCK near the interaction zone.

                The scaffold already builds blocking structure, but dense bars
                often read as generic noise instead of an intentional pushable
                block puzzle. This helper adds a single readable BLOCK prop near
                the planned route when the local geometry supports a valid push
                interaction.
                """
                nonlocal budget_remaining
                if budget_remaining <= 0:
                    return 0
                if gate_family not in {"switch", "toggle", "key", "item_unlock", "generic"} and archetype not in {
                    "hub",
                    "serpentine",
                }:
                    return 0

                anchor_r, anchor_c = (
                    self._clamp_room_coord(stateful_anchor)
                    if stateful_anchor is not None
                    else self._clamp_room_coord(puzzle_anchor)
                )
                puzzle_r, puzzle_c = self._clamp_room_coord(puzzle_anchor)
                candidate_slots: List[Tuple[int, int]] = []
                if flow_is_horizontal:
                    candidate_slots.extend(
                        [
                            (anchor_r - 1, anchor_c - 2),
                            (anchor_r + 1, anchor_c - 2),
                            (anchor_r - 1, anchor_c + 2),
                            (anchor_r + 1, anchor_c + 2),
                            (puzzle_r - 2, puzzle_c - 1),
                            (puzzle_r + 2, puzzle_c + 1),
                        ]
                    )
                else:
                    candidate_slots.extend(
                        [
                            (anchor_r - 2, anchor_c - 1),
                            (anchor_r - 2, anchor_c + 1),
                            (anchor_r + 2, anchor_c - 1),
                            (anchor_r + 2, anchor_c + 1),
                            (puzzle_r - 1, puzzle_c - 2),
                            (puzzle_r + 1, puzzle_c + 2),
                        ]
                    )

                seen_slots: Set[Tuple[int, int]] = set()
                for raw_row, raw_col in candidate_slots:
                    row, col = self._clamp_room_coord((raw_row, raw_col))
                    slot = (int(row), int(col))
                    if slot in seen_slots:
                        continue
                    seen_slots.add(slot)
                    if not _can_place(int(row), int(col)):
                        continue
                    if bool(route_mask_candidate[int(row), int(col)]):
                        continue

                    route_r0 = max(0, int(row) - 1)
                    route_r1 = min(ROOM_HEIGHT, int(row) + 2)
                    route_c0 = max(0, int(col) - 1)
                    route_c1 = min(ROOM_WIDTH, int(col) + 2)
                    if not bool(np.any(route_mask_candidate[route_r0:route_r1, route_c0:route_c1])):
                        continue

                    for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        push_dest_r = int(row) + d_r
                        push_dest_c = int(col) + d_c
                        player_r = int(row) - d_r
                        player_c = int(col) - d_c
                        if not (
                            1 <= push_dest_r < ROOM_HEIGHT - 1
                            and 1 <= push_dest_c < ROOM_WIDTH - 1
                            and 1 <= player_r < ROOM_HEIGHT - 1
                            and 1 <= player_c < ROOM_WIDTH - 1
                        ):
                            continue
                        if bool(reserved_candidate[player_r, player_c]):
                            continue
                        if int(candidate_grid[push_dest_r, push_dest_c]) != floor_id:
                            continue
                        if int(candidate_grid[player_r, player_c]) != floor_id:
                            continue
                        candidate_grid[int(row), int(col)] = block_id
                        budget_remaining -= 1
                        return 1

                fallback_slots: List[Tuple[int, int]] = []
                if flow_is_horizontal:
                    fallback_slots.extend(
                        [
                            (anchor_r, anchor_c - 3),
                            (anchor_r, anchor_c + 3),
                            (anchor_r - 2, anchor_c),
                            (anchor_r + 2, anchor_c),
                        ]
                    )
                else:
                    fallback_slots.extend(
                        [
                            (anchor_r - 3, anchor_c),
                            (anchor_r + 3, anchor_c),
                            (anchor_r, anchor_c - 2),
                            (anchor_r, anchor_c + 2),
                        ]
                    )

                for raw_row, raw_col in fallback_slots:
                    row, col = self._clamp_room_coord((raw_row, raw_col))
                    if not _can_place(int(row), int(col)):
                        continue
                    if bool(route_mask_candidate[int(row), int(col)]):
                        continue
                    route_r0 = max(0, int(row) - 1)
                    route_r1 = min(ROOM_HEIGHT, int(row) + 2)
                    route_c0 = max(0, int(col) - 1)
                    route_c1 = min(ROOM_WIDTH, int(col) + 2)
                    if not bool(np.any(route_mask_candidate[route_r0:route_r1, route_c0:route_c1])):
                        continue
                    candidate_grid[int(row), int(col)] = block_id
                    budget_remaining -= 1
                    return 1
                return 0

            segments_added = 0
            tiles_added = 0
            optional_segments_applied = 0
            pocket_tiles_forced = 0
            required_segments, optional_segments = self._build_puzzle_room_segments(
                archetype=archetype,
                gate_family=gate_family,
                variant_spec=variant_spec,
                stateful_anchor=stateful_anchor,
                flow_is_horizontal=flow_is_horizontal,
                puzzle_anchor=puzzle_anchor,
            )

            grammar_anchor = stateful_anchor if stateful_anchor is not None else puzzle_anchor
            if gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"}:
                forced_floor_tiles, anchor_frame_tiles = _apply_anchor_template(
                    anchor=grammar_anchor,
                    open_toward=source_anchor,
                )
                pocket_tiles_forced += int(forced_floor_tiles)
                if anchor_frame_tiles > 0:
                    segments_added += 1
                    tiles_added += int(anchor_frame_tiles)

            for segment in required_segments:
                added = _paint_block_line(segment)
                if added > 0:
                    segments_added += 1
                    tiles_added += added

            optional_quota = int(round(branch_density * len(optional_segments)))
            if branch_density > 0.0 and optional_segments and optional_quota <= 0:
                optional_quota = 1
            optional_quota = min(len(optional_segments), max(0, optional_quota))
            for segment in optional_segments[:optional_quota]:
                added = _paint_block_line(segment)
                if added > 0:
                    segments_added += 1
                    optional_segments_applied += 1
                    tiles_added += added

            push_block_props_added = int(_place_push_block_prop())
            tiles_added += int(push_block_props_added)

            candidate_stats: Dict[str, Any] = {
                "applied": int(tiles_added > 0),
                "tiles_added": int(tiles_added),
                "segments_added": int(segments_added),
                "pocket_floor_tiles_forced": int(pocket_tiles_forced),
                "push_block_props_added": int(push_block_props_added),
                "optional_segments_requested": int(optional_quota),
                "optional_segments_applied": int(optional_segments_applied),
                "route_template_used": int(route_template_used),
                "planned_route_pixels": int(np.sum(route_mask_candidate)),
                "archetype": str(archetype),
                "gate_family": str(gate_family),
                "stateful_anchor_name": str(stateful_anchor_name or ""),
                "variant_name": str(variant_spec.get("name", "baseline") or "baseline"),
                "variant_style": str(variant_spec.get("style", "baseline") or "baseline"),
                "variant_side_bias": int(variant_spec.get("side_bias", 0) or 0),
                "profile_branch_density": float(branch_density),
                "profile_block_budget": int(block_budget),
                "profile_preserve_route_margin": int(preserve_margin),
            }
            route_quality = self._evaluate_puzzle_candidate_route_quality(
                grid=candidate_grid,
                source_anchor=source_anchor,
                destination_anchor=destination_anchor,
                stateful_anchor=stateful_anchor,
                route_mask=route_mask_candidate,
                gate_family=gate_family,
                baseline_path_length=baseline_path_length,
            )
            contract = self._evaluate_puzzle_candidate_contract(
                grid=candidate_grid,
                gate_family=gate_family,
                source_anchor=source_anchor,
                destination_anchor=destination_anchor,
                stateful_anchor=stateful_anchor,
                route_quality=route_quality,
            )
            interaction = self._evaluate_puzzle_candidate_interaction_geometry(
                grid=candidate_grid,
                gate_family=gate_family,
                source_anchor=source_anchor,
                destination_anchor=destination_anchor,
                stateful_anchor=stateful_anchor,
                route_mask=route_mask_candidate,
                route_quality=route_quality,
            )
            sequence_eval = self._evaluate_puzzle_candidate_interaction_sequence(
                grid=candidate_grid,
                route_mask=route_mask_candidate,
                source_anchor=source_anchor,
                destination_anchor=destination_anchor,
                interaction_sequence=interaction_sequence,
            )
            candidate_stats["route_quality_score"] = float(route_quality.get("score", 0.0) or 0.0)
            candidate_stats["route_quality_path_exists"] = int(route_quality.get("path_exists", 0) or 0)
            candidate_stats["route_quality_path_length"] = int(route_quality.get("path_length", 0) or 0)
            candidate_stats["route_quality_turn_count"] = int(route_quality.get("turn_count", 0) or 0)
            candidate_stats["route_quality_overlap_ratio"] = float(
                route_quality.get("route_overlap_ratio", 0.0) or 0.0
            )
            candidate_stats["route_quality_detour_gain"] = float(
                route_quality.get("detour_gain", 0.0) or 0.0
            )
            candidate_stats["route_quality_stateful_distance_to_path"] = route_quality.get(
                "stateful_distance_to_path",
                None,
            )
            candidate_stats["route_quality_stateful_via_path_length"] = route_quality.get(
                "stateful_via_path_length",
                None,
            )
            candidate_stats["route_quality_stateful_branch_gain"] = route_quality.get(
                "stateful_branch_gain",
                None,
            )
            candidate_stats["route_quality_stateful_on_path"] = int(
                route_quality.get("stateful_on_path", 0) or 0
            )
            candidate_stats["contract_valid"] = int(contract.get("valid", 0) or 0)
            candidate_stats["contract_score"] = float(contract.get("score", 0.0) or 0.0)
            candidate_stats["contract_failure_reasons"] = list(contract.get("failure_reasons", []) or [])
            candidate_stats["contract_stateful_anchor_present"] = int(
                contract.get("stateful_anchor_present", 0) or 0
            )
            candidate_stats["contract_projected_stateful_anchor"] = contract.get(
                "projected_stateful_anchor",
                None,
            )
            candidate_stats["contract_pocket_floor_tiles"] = int(
                contract.get("pocket_floor_tiles", 0) or 0
            )
            candidate_stats["contract_frame_block_tiles"] = int(
                contract.get("frame_block_tiles", 0) or 0
            )
            candidate_stats["contract_anchor_adjacent_walkable"] = int(
                contract.get("anchor_adjacent_walkable", 0) or 0
            )
            candidate_stats["interaction_valid"] = int(interaction.get("valid", 0) or 0)
            candidate_stats["interaction_score"] = float(interaction.get("score", 0.0) or 0.0)
            candidate_stats["interaction_failure_reasons"] = list(
                interaction.get("failure_reasons", []) or []
            )
            candidate_stats["interaction_push_slot_count"] = int(
                interaction.get("push_slot_count", 0) or 0
            )
            candidate_stats["interaction_anchor_openings"] = int(
                interaction.get("anchor_openings", 0) or 0
            )
            candidate_stats["interaction_local_block_tiles"] = int(
                interaction.get("local_block_tiles", 0) or 0
            )
            candidate_stats["interaction_barrier_axis_tiles"] = int(
                interaction.get("barrier_axis_tiles", 0) or 0
            )
            candidate_stats["interaction_route_divergence"] = float(
                interaction.get("route_divergence", 0.0) or 0.0
            )
            candidate_stats["interaction_sequence_valid"] = int(
                sequence_eval.get("valid", 0) or 0
            )
            candidate_stats["interaction_sequence_score"] = float(
                sequence_eval.get("score", 0.0) or 0.0
            )
            candidate_stats["interaction_sequence_required"] = int(
                sequence_eval.get("required", 0) or 0
            )
            candidate_stats["interaction_sequence_length"] = int(
                sequence_eval.get("sequence_length", 0) or 0
            )
            candidate_stats["interaction_sequence_route_anchor_coverage"] = float(
                sequence_eval.get("route_anchor_coverage", 0.0) or 0.0
            )
            candidate_stats["interaction_sequence_pairwise_path_ratio"] = float(
                sequence_eval.get("pairwise_path_ratio", 0.0) or 0.0
            )
            candidate_stats["interaction_sequence_names"] = list(
                sequence_eval.get("sequence_names", []) or []
            )
            candidate_stats["interaction_sequence_failure_reasons"] = list(
                sequence_eval.get("failure_reasons", []) or []
            )
            descriptor = self._summarize_puzzle_candidate_descriptor(
                grid=candidate_grid,
                stats=candidate_stats,
            )
            candidate_stats["novelty_descriptor"] = descriptor
            candidate_stats["novelty_score"] = float(
                self._score_puzzle_candidate(
                    descriptor=descriptor,
                    stats=candidate_stats,
                    room_id=room_id,
                )
            )
            return candidate_grid, candidate_stats

        selected_grid = base_grid
        selected_stats: Dict[str, Any] = {
            "applied": 0,
            "tiles_added": 0,
            "segments_added": 0,
            "pocket_floor_tiles_forced": 0,
            "push_block_props_added": 0,
            "optional_segments_requested": 0,
            "optional_segments_applied": 0,
            "route_template_used": 0,
            "planned_route_pixels": int(np.sum(route_mask)),
            "archetype": str(archetype),
            "gate_family": str(gate_family),
            "stateful_anchor_name": str(stateful_anchor_name or ""),
            "variant_name": "baseline",
            "variant_style": "baseline",
            "variant_side_bias": 0,
            "profile_branch_density": float(scaffold_profile.get("branch_density", getattr(self, "default_puzzle_room_branch_density", 0.75))),
            "profile_block_budget": int(scaffold_profile.get("block_budget", getattr(self, "default_puzzle_room_block_budget", 28))),
            "profile_preserve_route_margin": int(preserve_margin),
            "novelty_descriptor": {},
            "novelty_score": 0.0,
            "route_quality_score": float(baseline_path_metrics.get("score", 0.0) or 0.0),
            "route_quality_path_exists": int(baseline_path_metrics.get("path_exists", 0) or 0),
            "route_quality_path_length": int(baseline_path_metrics.get("path_length", 0) or 0),
            "route_quality_turn_count": int(baseline_path_metrics.get("turn_count", 0) or 0),
            "route_quality_overlap_ratio": float(
                baseline_path_metrics.get("route_overlap_ratio", 0.0) or 0.0
            ),
            "route_quality_detour_gain": float(
                baseline_path_metrics.get("detour_gain", 0.0) or 0.0
            ),
            "route_quality_stateful_distance_to_path": baseline_path_metrics.get(
                "stateful_distance_to_path",
                None,
            ),
            "route_quality_stateful_via_path_length": baseline_path_metrics.get(
                "stateful_via_path_length",
                None,
            ),
            "route_quality_stateful_branch_gain": baseline_path_metrics.get(
                "stateful_branch_gain",
                None,
            ),
            "route_quality_stateful_on_path": int(
                baseline_path_metrics.get("stateful_on_path", 0) or 0
            ),
        }
        baseline_contract = self._evaluate_puzzle_candidate_contract(
            grid=base_grid,
            gate_family=gate_family,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            stateful_anchor=stateful_anchor,
            route_quality=baseline_path_metrics,
        )
        baseline_interaction = self._evaluate_puzzle_candidate_interaction_geometry(
            grid=base_grid,
            gate_family=gate_family,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            stateful_anchor=stateful_anchor,
            route_mask=route_mask,
            route_quality=baseline_path_metrics,
        )
        baseline_sequence = self._evaluate_puzzle_candidate_interaction_sequence(
            grid=base_grid,
            route_mask=route_mask,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            interaction_sequence=interaction_sequence,
        )
        selected_stats["contract_valid"] = int(baseline_contract.get("valid", 0) or 0)
        selected_stats["contract_score"] = float(baseline_contract.get("score", 0.0) or 0.0)
        selected_stats["contract_failure_reasons"] = list(
            baseline_contract.get("failure_reasons", []) or []
        )
        selected_stats["contract_stateful_anchor_present"] = int(
            baseline_contract.get("stateful_anchor_present", 0) or 0
        )
        selected_stats["contract_projected_stateful_anchor"] = baseline_contract.get(
            "projected_stateful_anchor",
            None,
        )
        selected_stats["contract_pocket_floor_tiles"] = int(
            baseline_contract.get("pocket_floor_tiles", 0) or 0
        )
        selected_stats["contract_frame_block_tiles"] = int(
            baseline_contract.get("frame_block_tiles", 0) or 0
        )
        selected_stats["contract_anchor_adjacent_walkable"] = int(
            baseline_contract.get("anchor_adjacent_walkable", 0) or 0
        )
        selected_stats["interaction_valid"] = int(baseline_interaction.get("valid", 0) or 0)
        selected_stats["interaction_score"] = float(baseline_interaction.get("score", 0.0) or 0.0)
        selected_stats["interaction_failure_reasons"] = list(
            baseline_interaction.get("failure_reasons", []) or []
        )
        selected_stats["interaction_push_slot_count"] = int(
            baseline_interaction.get("push_slot_count", 0) or 0
        )
        selected_stats["interaction_anchor_openings"] = int(
            baseline_interaction.get("anchor_openings", 0) or 0
        )
        selected_stats["interaction_local_block_tiles"] = int(
            baseline_interaction.get("local_block_tiles", 0) or 0
        )
        selected_stats["interaction_barrier_axis_tiles"] = int(
            baseline_interaction.get("barrier_axis_tiles", 0) or 0
        )
        selected_stats["interaction_route_divergence"] = float(
            baseline_interaction.get("route_divergence", 0.0) or 0.0
        )
        selected_stats["interaction_sequence_valid"] = int(
            baseline_sequence.get("valid", 0) or 0
        )
        selected_stats["interaction_sequence_score"] = float(
            baseline_sequence.get("score", 0.0) or 0.0
        )
        selected_stats["interaction_sequence_required"] = int(
            baseline_sequence.get("required", 0) or 0
        )
        selected_stats["interaction_sequence_length"] = int(
            baseline_sequence.get("sequence_length", 0) or 0
        )
        selected_stats["interaction_sequence_route_anchor_coverage"] = float(
            baseline_sequence.get("route_anchor_coverage", 0.0) or 0.0
        )
        selected_stats["interaction_sequence_pairwise_path_ratio"] = float(
            baseline_sequence.get("pairwise_path_ratio", 0.0) or 0.0
        )
        selected_stats["interaction_sequence_names"] = list(
            baseline_sequence.get("sequence_names", []) or []
        )
        selected_stats["interaction_sequence_failure_reasons"] = list(
            baseline_sequence.get("failure_reasons", []) or []
        )
        baseline_descriptor = self._summarize_puzzle_candidate_descriptor(
            grid=base_grid,
            stats=selected_stats,
        )
        selected_stats["novelty_descriptor"] = baseline_descriptor
        baseline_selection_score = float(
            self._score_puzzle_candidate(
                descriptor=baseline_descriptor,
                stats=selected_stats,
                room_id=room_id,
            )
        )
        selected_stats["selection_score"] = baseline_selection_score
        baseline_stats = dict(selected_stats)
        selected_score = float("-inf")

        if not variant_specs:
            variant_specs = [{"name": "baseline", "style": "baseline", "side_bias": 0, "branch_density_delta": 0.0, "block_budget_delta": 0}]

        if isinstance(cached_variant, dict):
            selected_grid, selected_stats = _render_candidate(cached_variant)
            selected_stats["selection_score"] = float(selected_stats.get("novelty_score", 0.0))
        else:
            for variant_spec in variant_specs:
                candidate_grid, candidate_stats = _render_candidate(variant_spec)
                candidate_score = float(candidate_stats.get("novelty_score", 0.0))
                candidate_stats["selection_score"] = candidate_score
                if candidate_score > selected_score:
                    selected_score = candidate_score
                    selected_grid = candidate_grid
                    selected_stats = candidate_stats
                    if isinstance(variant_cache, dict):
                        variant_cache[room_id] = dict(variant_spec)

        min_quality_gain = float(
            max(0.0, float(getattr(self, "default_puzzle_room_min_quality_gain", 0.5)))
        )
        contract_required = gate_family in {"switch", "toggle", "item_unlock", "key", "combat"}
        interaction_required = gate_family in {"switch", "toggle", "bombable", "item_unlock", "key"}
        sequence_required = int(selected_stats.get("interaction_sequence_required", 0) or 0) > 0
        if contract_required and int(selected_stats.get("contract_valid", 0) or 0) <= 0:
            selected_grid = base_grid
            selected_stats = dict(baseline_stats)
            selected_stats["selection_score"] = float(baseline_selection_score)
            selected_stats["quality_gate_skipped"] = 0
            selected_stats["contract_gate_skipped"] = 1
            selected_stats["interaction_gate_skipped"] = 0
            selected_stats["sequence_gate_skipped"] = 0
            stats.update(selected_stats)
            return selected_grid, stats
        if interaction_required and int(selected_stats.get("interaction_valid", 0) or 0) <= 0:
            selected_grid = base_grid
            selected_stats = dict(baseline_stats)
            selected_stats["selection_score"] = float(baseline_selection_score)
            selected_stats["quality_gate_skipped"] = 0
            selected_stats["contract_gate_skipped"] = 0
            selected_stats["interaction_gate_skipped"] = 1
            selected_stats["sequence_gate_skipped"] = 0
            stats.update(selected_stats)
            return selected_grid, stats
        if sequence_required and int(selected_stats.get("interaction_sequence_valid", 0) or 0) <= 0:
            selected_grid = base_grid
            selected_stats = dict(baseline_stats)
            selected_stats["selection_score"] = float(baseline_selection_score)
            selected_stats["quality_gate_skipped"] = 0
            selected_stats["contract_gate_skipped"] = 0
            selected_stats["interaction_gate_skipped"] = 0
            selected_stats["sequence_gate_skipped"] = 1
            stats.update(selected_stats)
            return selected_grid, stats

        if (
            float(selected_stats.get("selection_score", float("-inf")))
            < (baseline_selection_score + min_quality_gain)
        ):
            selected_grid = base_grid
            selected_stats = dict(baseline_stats)
            selected_stats["selection_score"] = float(baseline_selection_score)
            selected_stats["quality_gate_skipped"] = 1
            selected_stats["contract_gate_skipped"] = 0
            selected_stats["interaction_gate_skipped"] = 0
            selected_stats["sequence_gate_skipped"] = 0
        else:
            selected_stats["quality_gate_skipped"] = 0
            selected_stats["contract_gate_skipped"] = 0
            selected_stats["interaction_gate_skipped"] = 0
            selected_stats["sequence_gate_skipped"] = 0

        stats.update(selected_stats)
        return selected_grid, stats

    def _count_small_interior_structure_components(
        self,
        grid: np.ndarray,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
        max_component_tiles: int = 6,
    ) -> int:
        """
        Count small interior wall/block islands that survive structural cleanup.

        These components are a good proxy for fast-sampler drift: they usually
        show up as isolated 1x1 / 2x2 block clusters inside otherwise plain
        rooms. We use the count as a quality gate for teacher fallback without
        touching topology semantics.
        """
        out = np.asarray(grid, dtype=np.int32)
        wall_like_mask = np.isin(out, np.array([int(TileID.WALL), int(TileID.BLOCK)], dtype=np.int32))
        allowed_door_mask = self._required_room_door_slots_mask(graph=graph, room_id=room_id)
        visited = np.zeros_like(wall_like_mask, dtype=bool)
        suspicious_components = 0

        for row in range(ROOM_HEIGHT):
            for col in range(ROOM_WIDTH):
                if not bool(wall_like_mask[row, col]) or bool(visited[row, col]):
                    continue

                component: List[Tuple[int, int]] = []
                stack: List[Tuple[int, int]] = [(row, col)]
                visited[row, col] = True
                touches_boundary = False
                touches_allowed_door = False

                while stack:
                    cur_r, cur_c = stack.pop()
                    component.append((cur_r, cur_c))
                    if cur_r in {0, ROOM_HEIGHT - 1} or cur_c in {0, ROOM_WIDTH - 1}:
                        touches_boundary = True
                    if bool(allowed_door_mask[cur_r, cur_c]):
                        touches_allowed_door = True
                    for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        next_r = cur_r + d_r
                        next_c = cur_c + d_c
                        if not (0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH):
                            continue
                        if not bool(wall_like_mask[next_r, next_c]) or bool(visited[next_r, next_c]):
                            continue
                        visited[next_r, next_c] = True
                        stack.append((next_r, next_c))

                if touches_boundary or touches_allowed_door:
                    continue
                if len(component) <= int(max_component_tiles):
                    suspicious_components += 1

        return int(suspicious_components)

    def _should_retry_room_with_teacher(
        self,
        *,
        final_grid: np.ndarray,
        graph: Optional[nx.Graph],
        room_id: Any,
        metrics: Dict[str, float],
        source_mode: str = "fast_sampler",
    ) -> bool:
        """
        Decide whether a non-teacher room should be regenerated with the full teacher.

        The goal is pragmatic: keep cheaper / weaker generators for normal rooms,
        but avoid exporting rooms that still contain obvious structural decode
        noise after cleanup/repair.
        """
        suspicious_components = self._count_small_interior_structure_components(
            final_grid,
            graph=graph,
            room_id=room_id,
        )
        if suspicious_components >= 1:
            return True

        block_id = int(TileID.BLOCK)
        interior_block_tiles = int(np.sum(np.asarray(final_grid, dtype=np.int32) == block_id))
        graph_markers = set(int(v) for v in self._resolve_room_graph_markers(graph, room_id))
        if interior_block_tiles > 0 and not graph_markers.intersection({int(TileID.PUZZLE), int(TileID.STAIR)}):
            return True

        door_like_tiles = np.isin(
            np.asarray(final_grid, dtype=np.int32),
            np.array(
                [
                    int(TileID.DOOR_OPEN),
                    int(TileID.DOOR_LOCKED),
                    int(TileID.DOOR_BOMB),
                    int(TileID.DOOR_PUZZLE),
                    int(TileID.DOOR_BOSS),
                    int(TileID.DOOR_SOFT),
                ],
                dtype=np.int32,
            ),
        )
        allowed_door_mask = self._required_room_door_slots_mask(graph=graph, room_id=room_id)
        unexpected_door_tiles = int(np.sum(door_like_tiles & ~allowed_door_mask))
        if unexpected_door_tiles >= 1:
            return True

        repair_tiles = int(metrics.get("tiles_changed", 0))
        repair_obstacles = int(metrics.get("repair_interior_obstacle_tiles_removed", 0))
        neural_obstacles = int(metrics.get("neural_interior_obstacle_tiles_removed", 0))
        if repair_tiles >= 18 and (repair_obstacles > 0 or neural_obstacles > 0):
            return True

        source_mode = str(source_mode or "fast_sampler").strip().lower()
        if source_mode == "masked_room":
            if float(metrics.get("final_graph_marker_overwrite_rate", 0.0)) > 0.34:
                return True
            if repair_tiles >= 12:
                return True

        return False

    def _resolve_effective_sampling_guidance(
        self,
        *,
        use_fast_sampling: bool,
        guidance_scale: float,
        logic_guidance_scale: float,
    ) -> Tuple[float, float]:
        """
        Clamp runtime guidance to the regime the distilled fast sampler was trained for.

        The fast-sampler student is distilled from the base diffusion model using
        the diffusion checkpoint's CFG setting and without LogicNet gradient
        guidance. Reusing more aggressive diffusion-time overrides than the
        teacher was trained with pushes the short-step student out of
        distribution and hurts room quality.
        """
        effective_guidance_scale = float(guidance_scale)
        effective_logic_guidance_scale = float(logic_guidance_scale)
        if not bool(use_fast_sampling) or not self.diffusion.supports_fast_sampling():
            return effective_guidance_scale, effective_logic_guidance_scale

        trained_cfg_scale = float(
            max(
                0.0,
                getattr(
                    self.diffusion,
                    "training_cfg_scale",
                    self.diffusion_fallback_config.get(
                        "cfg_scale",
                        getattr(self.diffusion, "cfg_scale", effective_guidance_scale),
                    ),
                ),
            )
        )
        if effective_guidance_scale > trained_cfg_scale + 1e-6:
            self._bump_diagnostic("fast_sampling_cfg_clamped")
            logger.debug(
                "Fast sampling clamped CFG from %.3f to %.3f to match the distilled teacher regime.",
                effective_guidance_scale,
                trained_cfg_scale,
            )
            effective_guidance_scale = trained_cfg_scale

        if effective_logic_guidance_scale > 0.0:
            self._bump_diagnostic("fast_sampling_logic_guidance_disabled")
            logger.debug(
                "Fast sampling disabled LogicNet runtime guidance (requested %.3f) because the student was "
                "not distilled with gradient guidance enabled.",
                effective_logic_guidance_scale,
            )
            effective_logic_guidance_scale = 0.0

        return effective_guidance_scale, effective_logic_guidance_scale

    def _resolve_room_graph_markers(
        self,
        graph: Optional[nx.Graph],
        room_id: Any,
    ) -> List[int]:
        """Infer deterministic per-room semantic markers from mission-graph metadata."""
        if not isinstance(graph, nx.Graph) or room_id not in graph:
            return []

        attrs = dict(graph.nodes[room_id])
        role_flags = self._room_role_flags(attrs)
        label_tokens = self._parse_label_tokens(attrs.get("label"))
        node_type = str(
            attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or ""
        ).strip().lower()

        markers: List[int] = []

        if role_flags.get("is_start", False) or node_type == "start":
            markers.append(int(TileID.START))
        if role_flags.get("has_goal", False) or node_type in {"goal", "triforce"}:
            markers.append(int(TileID.TRIFORCE))

        if node_type in {"boss", "mini_boss"} or role_flags.get("has_boss", False):
            markers.append(int(TileID.BOSS))
        elif (
            role_flags.get("has_enemy", False)
            or node_type in {"enemy", "arena", "combat_puzzle"}
            or "e" in label_tokens
        ):
            markers.append(int(TileID.ENEMY))

        if node_type in {"big_key", "boss_key"}:
            markers.append(int(TileID.KEY_BOSS))
        elif role_flags.get("has_key", False) or node_type == "key" or "k" in label_tokens:
            markers.append(int(TileID.KEY_SMALL))

        if node_type in {"item", "protection_item"} or role_flags.get("has_item", False):
            markers.append(int(TileID.KEY_ITEM))
        elif node_type in {"minor_item", "treasure"}:
            markers.append(int(TileID.ITEM_MINOR))

        if node_type in {"stairs_up", "stairs_down", "stair", "warp"}:
            markers.append(int(TileID.STAIR))

        if (
            role_flags.get("has_puzzle", False)
            or node_type in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"}
            or "p" in label_tokens
        ):
            markers.append(int(TileID.PUZZLE))

        return markers

    def _build_room_graph_marker_preferences(
        self,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Dict[int, Tuple[int, int]]:
        """
        Build gate-aware preferred semantic slots for deterministic room markers.

        Generic room-role anchors are often too weak for puzzle readability:
        the scaffold may create a clear key pocket or item alcove while the
        overlay still drops the marker near the room center. This helper keeps
        marker placement aligned with the same gate-family semantics used by the
        constructive scaffold.
        """
        if start_goal is None:
            start_goal = self._extract_room_start_goal(graph, room_id) if isinstance(graph, nx.Graph) else None
        start_coord, goal_coord = start_goal if start_goal is not None else (
            (ROOM_HEIGHT // 2, 0),
            (ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
        )
        attrs = dict(graph.nodes[room_id]) if isinstance(graph, nx.Graph) and room_id in graph else {}
        role_flags = self._room_role_flags(attrs)
        semantics = self._extract_room_topology_semantics(graph, room_id) if isinstance(graph, nx.Graph) and room_id in graph else {
            "required_doors": {},
            "incoming_dirs": set(),
            "outgoing_dirs": set(),
        }
        semantic_anchors = build_room_semantic_anchor_points(
            room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
            start=start_coord,
            goal=goal_coord,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            room_role_flags=role_flags,
            semantic_puzzle_offset=self.default_semantic_puzzle_offset,
        )
        node_type = str(
            attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or ""
        ).strip().lower()
        gate_family = self._classify_puzzle_gate_family(
            role_flags=role_flags,
            semantics=semantics,
            node_type=node_type,
        )
        if gate_family in {"switch", "toggle", "bombable"}:
            stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
        elif gate_family == "item_unlock":
            stateful_anchor_name = "item" if "item" in semantic_anchors else ("puzzle" if "puzzle" in semantic_anchors else None)
        elif gate_family == "key":
            stateful_anchor_name = "key" if "key" in semantic_anchors else None
        elif gate_family == "combat":
            stateful_anchor_name = "enemy" if "enemy" in semantic_anchors else None
        else:
            stateful_anchor_name = "puzzle" if "puzzle" in semantic_anchors else None
        stateful_anchor = self._clamp_room_coord(semantic_anchors.get(stateful_anchor_name)) if stateful_anchor_name and semantic_anchors.get(stateful_anchor_name) is not None else None
        puzzle_anchor = self._clamp_room_coord(semantic_anchors.get("puzzle", (max(1, ROOM_HEIGHT // 2 - 2), ROOM_WIDTH // 2)))
        enemy_anchor = self._clamp_room_coord(semantic_anchors.get("enemy", (ROOM_HEIGHT // 2 - 2, ROOM_WIDTH // 2)))
        item_anchor = self._clamp_room_coord(semantic_anchors.get("item", (min(ROOM_HEIGHT - 2, ROOM_HEIGHT // 2 + 2), ROOM_WIDTH // 2)))
        key_anchor = self._clamp_room_coord(semantic_anchors.get("key", (ROOM_HEIGHT // 2, max(1, ROOM_WIDTH // 2 - 2))))
        boss_anchor = self._clamp_room_coord(semantic_anchors.get("boss", (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)))

        preferred_positions: Dict[int, Tuple[int, int]] = {
            int(TileID.START): self._clamp_room_coord(semantic_anchors.get("start", start_coord)),
            int(TileID.TRIFORCE): self._clamp_room_coord(semantic_anchors.get("goal", goal_coord)),
            int(TileID.BOSS): boss_anchor,
            int(TileID.ENEMY): enemy_anchor,
            int(TileID.KEY_SMALL): key_anchor,
            int(TileID.KEY_BOSS): key_anchor,
            int(TileID.KEY_ITEM): item_anchor,
            int(TileID.ITEM_MINOR): item_anchor,
            int(TileID.STAIR): item_anchor,
            int(TileID.PUZZLE): puzzle_anchor,
        }

        if stateful_anchor is not None:
            if gate_family == "key":
                preferred_positions[int(TileID.KEY_SMALL)] = stateful_anchor
                preferred_positions[int(TileID.KEY_BOSS)] = stateful_anchor
                preferred_positions[int(TileID.PUZZLE)] = puzzle_anchor
            elif gate_family == "item_unlock":
                preferred_positions[int(TileID.KEY_ITEM)] = stateful_anchor
                preferred_positions[int(TileID.ITEM_MINOR)] = stateful_anchor
                preferred_positions[int(TileID.STAIR)] = stateful_anchor
                preferred_positions[int(TileID.PUZZLE)] = puzzle_anchor
            elif gate_family in {"switch", "toggle", "bombable"}:
                preferred_positions[int(TileID.PUZZLE)] = stateful_anchor
            elif gate_family == "combat":
                preferred_positions[int(TileID.ENEMY)] = stateful_anchor
                if role_flags.get("has_puzzle", False):
                    preferred_positions[int(TileID.PUZZLE)] = puzzle_anchor

        return preferred_positions

    def _find_room_graph_marker_slot(
        self,
        grid: np.ndarray,
        *,
        preferred: Tuple[int, int],
        occupied: Set[Tuple[int, int]],
        tile_id: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Find a stable in-room placement slot near a preferred coordinate."""
        floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
        preferred = self._clamp_room_coord(preferred)
        wanted_tile = None if tile_id is None else int(tile_id)

        if (
            wanted_tile is not None
            and preferred not in occupied
            and int(grid[preferred[0], preferred[1]]) == wanted_tile
        ):
            return preferred

        if wanted_tile is not None:
            for radius in range(0, max(ROOM_HEIGHT, ROOM_WIDTH)):
                row_min = max(1, preferred[0] - radius)
                row_max = min(ROOM_HEIGHT - 2, preferred[0] + radius)
                col_min = max(1, preferred[1] - radius)
                col_max = min(ROOM_WIDTH - 2, preferred[1] + radius)
                for r in range(row_min, row_max + 1):
                    for c in range(col_min, col_max + 1):
                        if max(abs(r - preferred[0]), abs(c - preferred[1])) != radius:
                            continue
                        if (r, c) in occupied:
                            continue
                        if int(grid[r, c]) == wanted_tile:
                            return (r, c)

        def _search(valid_tiles: Set[int]) -> Optional[Tuple[int, int]]:
            for radius in range(0, max(ROOM_HEIGHT, ROOM_WIDTH)):
                row_min = max(1, preferred[0] - radius)
                row_max = min(ROOM_HEIGHT - 2, preferred[0] + radius)
                col_min = max(1, preferred[1] - radius)
                col_max = min(ROOM_WIDTH - 2, preferred[1] + radius)
                for r in range(row_min, row_max + 1):
                    for c in range(col_min, col_max + 1):
                        if max(abs(r - preferred[0]), abs(c - preferred[1])) != radius:
                            continue
                        if (r, c) in occupied:
                            continue
                        if int(grid[r, c]) in valid_tiles:
                            return (r, c)
            return None

        slot = _search({floor_id})
        if slot is not None:
            return slot

        # Fallback: any non-boundary position that is not a hard wall.
        non_blocking_tiles = {
            int(TileID.FLOOR),
            int(TileID.VOID),
            int(TileID.BLOCK),
            int(TileID.DOOR_OPEN),
            int(TileID.DOOR_LOCKED),
            int(TileID.DOOR_BOMB),
            int(TileID.DOOR_PUZZLE),
            int(TileID.DOOR_BOSS),
            int(TileID.DOOR_SOFT),
        }
        slot = _search(non_blocking_tiles)
        if slot is not None:
            return slot

        if preferred not in occupied:
            return preferred

        for r in range(1, ROOM_HEIGHT - 1):
            for c in range(1, ROOM_WIDTH - 1):
                if (r, c) not in occupied:
                    return (r, c)

        return preferred

    def _overlay_room_graph_markers(
        self,
        grid: np.ndarray,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, int, List[int]]:
        """
        Reintroduce graph-owned semantics after structural room generation.

        This keeps the generator focused on layout while still producing rooms
        with the mission-critical markers the graph requires.
        """
        out = np.asarray(grid, dtype=np.int32).copy()
        markers = self._resolve_room_graph_markers(graph, room_id)
        if not markers:
            return out, 0, []

        if start_goal is None:
            start_goal = self._extract_room_start_goal(graph, room_id) if isinstance(graph, nx.Graph) else None
        start_coord, goal_coord = start_goal if start_goal is not None else (
            (ROOM_HEIGHT // 2, 0),
            (ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
        )
        role_flags = self._room_role_flags(dict(graph.nodes[room_id])) if isinstance(graph, nx.Graph) and room_id in graph else {}
        semantics = self._extract_room_topology_semantics(graph, room_id) if isinstance(graph, nx.Graph) and room_id in graph else {
            "required_doors": {},
            "incoming_dirs": set(),
            "outgoing_dirs": set(),
        }
        semantic_anchors = build_room_semantic_anchor_points(
            room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
            start=start_coord,
            goal=goal_coord,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            room_role_flags=role_flags,
            semantic_puzzle_offset=self.default_semantic_puzzle_offset,
        )
        preferred_positions = self._build_room_graph_marker_preferences(
            graph=graph,
            room_id=room_id,
            start_goal=(start_coord, goal_coord),
        )

        occupied: Set[Tuple[int, int]] = set()
        placed: List[int] = []

        for tile_id in markers:
            preferred = preferred_positions.get(int(tile_id), (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
            slot = self._find_room_graph_marker_slot(
                out,
                preferred=preferred,
                occupied=occupied,
                tile_id=int(tile_id),
            )
            occupied.add(slot)
            out[slot[0], slot[1]] = int(tile_id)
            placed.append(int(tile_id))

        return out, len(placed), placed

    def _plan_room_graph_marker_layout(
        self,
        grid: np.ndarray,
        *,
        graph: Optional[nx.Graph],
        room_id: Any,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> List[Tuple[int, Tuple[int, int]]]:
        """
        Compute the deterministic graph-marker placement plan for a room without
        mutating the room grid.

        The returned slots are the same positions `_overlay_room_graph_markers`
        would target on the same pre-overlay grid. This lets audits measure how
        much semantic placement is learned vs. forced by the symbolic overlay.
        """
        base_grid = np.asarray(grid, dtype=np.int32).copy()
        markers = self._resolve_room_graph_markers(graph, room_id)
        if not markers:
            return []

        if start_goal is None:
            start_goal = self._extract_room_start_goal(graph, room_id) if isinstance(graph, nx.Graph) else None
        start_coord, goal_coord = start_goal if start_goal is not None else (
            (ROOM_HEIGHT // 2, 0),
            (ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
        )
        role_flags = self._room_role_flags(dict(graph.nodes[room_id])) if isinstance(graph, nx.Graph) and room_id in graph else {}
        semantics = self._extract_room_topology_semantics(graph, room_id) if isinstance(graph, nx.Graph) and room_id in graph else {
            "required_doors": {},
            "incoming_dirs": set(),
            "outgoing_dirs": set(),
        }
        semantic_anchors = build_room_semantic_anchor_points(
            room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
            start=start_coord,
            goal=goal_coord,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            room_role_flags=role_flags,
            semantic_puzzle_offset=self.default_semantic_puzzle_offset,
        )
        preferred_positions = self._build_room_graph_marker_preferences(
            graph=graph,
            room_id=room_id,
            start_goal=(start_coord, goal_coord),
        )

        occupied: Set[Tuple[int, int]] = set()
        placements: List[Tuple[int, Tuple[int, int]]] = []
        for tile_id in markers:
            preferred = preferred_positions.get(int(tile_id), (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))
            slot = self._find_room_graph_marker_slot(
                base_grid,
                preferred=preferred,
                occupied=occupied,
                tile_id=int(tile_id),
            )
            occupied.add(slot)
            placements.append((int(tile_id), slot))
        return placements

    def _build_room_puzzle_metadata(
        self,
        *,
        grid: np.ndarray,
        graph: Optional[nx.Graph],
        room_id: Any,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        marker_plan: Optional[Sequence[Tuple[int, Tuple[int, int]]]] = None,
        scaffold_stats: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a structured room-local puzzle plan for validator-side state machines.

        The room scaffold already knows the intended local progression order.
        This helper serializes that order into explicit stateful stages so the
        stitched validator can enforce multi-step mechanics instead of treating
        puzzle rooms as purely visual clutter.
        """
        if not isinstance(graph, nx.Graph) or room_id not in graph:
            return {}

        room_grid = np.asarray(grid, dtype=np.int32)
        attrs = dict(graph.nodes[room_id])
        role_flags = self._room_role_flags(attrs)
        semantics = self._extract_room_topology_semantics(graph, room_id)
        node_type = str(
            attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or ""
        ).strip().lower()
        if start_goal is None:
            start_goal = self._extract_room_start_goal(graph, room_id)
        start_coord, goal_coord = start_goal if start_goal is not None else (
            (ROOM_HEIGHT // 2, 0),
            (ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
        )
        semantic_anchors = build_room_semantic_anchor_points(
            room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
            start=start_coord,
            goal=goal_coord,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            room_role_flags=role_flags,
            semantic_puzzle_offset=self.default_semantic_puzzle_offset,
        )
        gate_family = str(
            (scaffold_stats or {}).get(
                "gate_family",
                self._classify_puzzle_gate_family(
                    role_flags=role_flags,
                    semantics=semantics,
                    node_type=node_type,
                ),
            )
            or "generic"
        ).strip().lower()
        archetype = str(
            (scaffold_stats or {}).get(
                "archetype",
                self._select_puzzle_room_scaffold_archetype(
                    role_flags=role_flags,
                    semantics=semantics,
                    node_type=node_type,
                ),
            )
            or "serpentine"
        ).strip().lower()
        interaction_sequence = self._resolve_puzzle_interaction_sequence(
            archetype=archetype,
            gate_family=gate_family,
            role_flags=role_flags,
            semantic_anchors=semantic_anchors,
        )
        door_rows, door_cols = np.where(room_grid == int(TileID.DOOR_PUZZLE))
        controlled_doors_local = [
            [int(row), int(col)]
            for row, col in zip(door_rows.tolist(), door_cols.tolist())
        ]
        if not interaction_sequence and controlled_doors_local:
            fallback_anchor = self._clamp_room_coord(
                semantic_anchors.get("puzzle", semantic_anchors.get("goal", goal_coord))
            )
            interaction_sequence = [("puzzle", fallback_anchor)]
        if not interaction_sequence and not controlled_doors_local:
            return {}

        marker_slots: Dict[int, List[Tuple[int, int]]] = {}
        for tile_id, slot in list(marker_plan or []):
            marker_slots.setdefault(int(tile_id), []).append(self._clamp_room_coord(slot))

        def _resolve_anchor(
            name: str,
            fallback: Tuple[int, int],
        ) -> Tuple[Tuple[int, int], Optional[int]]:
            normalized = str(name).strip().lower()
            if normalized == "key":
                candidates = [int(TileID.KEY_SMALL), int(TileID.KEY_BOSS)]
            elif normalized == "item":
                candidates = [int(TileID.KEY_ITEM), int(TileID.ITEM_MINOR), int(TileID.STAIR)]
            elif normalized == "puzzle":
                candidates = [int(TileID.PUZZLE)]
            elif normalized == "enemy":
                candidates = [int(TileID.ENEMY), int(TileID.BOSS)]
            elif normalized == "boss":
                candidates = [int(TileID.BOSS)]
            else:
                candidates = []
            for tile_id in candidates:
                if marker_slots.get(int(tile_id)):
                    return marker_slots[int(tile_id)][0], int(tile_id)
            return self._clamp_room_coord(fallback), None

        def _stage_kind(name: str) -> str:
            normalized = str(name).strip().lower()
            if normalized == "key":
                return "collect_key"
            if normalized == "item":
                return "collect_item"
            if normalized in {"enemy", "boss"}:
                return "defeat_enemy"
            if normalized == "puzzle" and gate_family in {"switch", "toggle"}:
                return "push_block_to_switch"
            return "step_on_puzzle"

        stage_sequence: List[Dict[str, Any]] = []
        for stage_index, (name, anchor) in enumerate(interaction_sequence):
            resolved_anchor, tile_id = _resolve_anchor(str(name), anchor)
            stage_sequence.append(
                {
                    "stage_index": int(stage_index),
                    "name": str(name),
                    "kind": _stage_kind(str(name)),
                    "local_anchor": [int(resolved_anchor[0]), int(resolved_anchor[1])],
                    "trigger_tile_id": int(tile_id) if tile_id is not None else None,
                }
            )

        return {
            "plan_id": f"room_{room_id}",
            "room_id": room_id,
            "gate_family": str(gate_family),
            "archetype": str(archetype),
            "sequence_required": bool(int((scaffold_stats or {}).get("interaction_sequence_required", 0) or 0) > 0),
            "sequence_valid": bool(int((scaffold_stats or {}).get("interaction_sequence_valid", 0) or 0) > 0),
            "interaction_valid": bool(int((scaffold_stats or {}).get("interaction_valid", 0) or 0) > 0),
            "contract_valid": bool(int((scaffold_stats or {}).get("contract_valid", 0) or 0) > 0),
            "controlled_doors_local": list(controlled_doors_local),
            "stage_sequence": stage_sequence,
        }

    def _globalize_room_puzzle_metadata(
        self,
        *,
        rooms: Mapping[Any, RoomGenerationResult],
        stitched_layout: Optional[StitchedRoomLayout],
    ) -> Dict[str, Any]:
        """Lift room-local puzzle plans into stitched global coordinates."""
        if stitched_layout is None:
            return {"version": "stateful_v1", "plans": {}, "room_to_plan": {}}

        plans: Dict[str, Dict[str, Any]] = {}
        room_to_plan: Dict[str, str] = {}
        for room_id, room in dict(rooms or {}).items():
            local_meta = dict(getattr(room, "puzzle_metadata", {}) or {})
            if not local_meta:
                continue
            offset = stitched_layout.room_offsets.get(room_id)
            if offset is None:
                continue
            offset_r, offset_c = int(offset[0]), int(offset[1])
            plan_id = str(local_meta.get("plan_id", f"room_{room_id}"))
            room_to_plan[str(room_id)] = str(plan_id)

            global_stages: List[Dict[str, Any]] = []
            for raw_stage in list(local_meta.get("stage_sequence", []) or []):
                local_anchor = raw_stage.get("local_anchor")
                if not isinstance(local_anchor, (list, tuple)) or len(local_anchor) != 2:
                    continue
                local_r, local_c = int(local_anchor[0]), int(local_anchor[1])
                global_stages.append(
                    {
                        **dict(raw_stage),
                        "local_anchor": [local_r, local_c],
                        "global_anchor": [offset_r + local_r, offset_c + local_c],
                    }
                )

            global_doors: List[List[int]] = []
            for local_door in list(local_meta.get("controlled_doors_local", []) or []):
                if not isinstance(local_door, (list, tuple)) or len(local_door) != 2:
                    continue
                door_r, door_c = int(local_door[0]), int(local_door[1])
                global_doors.append([offset_r + door_r, offset_c + door_c])

            plans[str(plan_id)] = {
                **local_meta,
                "plan_id": str(plan_id),
                "room_id": room_id,
                "room_offset": [offset_r, offset_c],
                "stage_sequence": global_stages,
                "controlled_doors_global": global_doors,
            }

        return {
            "version": "stateful_v1",
            "plans": plans,
            "room_to_plan": room_to_plan,
        }

    def _measure_room_graph_marker_alignment(
        self,
        grid: np.ndarray,
        *,
        placements: List[Tuple[int, Tuple[int, int]]],
        prefix: str,
    ) -> Dict[str, float]:
        """Measure how well a room already matches the planned graph markers."""
        grid_np = np.asarray(grid, dtype=np.int32)
        expected = int(len(placements))
        if expected <= 0:
            return {
                f"{prefix}graph_marker_expected": 0.0,
                f"{prefix}graph_marker_exact_matches": 0.0,
                f"{prefix}graph_marker_tile_present": 0.0,
                f"{prefix}graph_marker_exact_match_rate": 1.0,
                f"{prefix}graph_marker_presence_rate": 1.0,
                f"{prefix}semantic_anchor_avg_manhattan_error": 0.0,
                f"{prefix}semantic_anchor_max_manhattan_error": 0.0,
            }

        exact_matches = 0
        tile_present = 0
        distances: List[float] = []
        for tile_id, slot in placements:
            sr, sc = int(slot[0]), int(slot[1])
            if int(grid_np[sr, sc]) == int(tile_id):
                exact_matches += 1
                tile_present += 1
                distances.append(0.0)
                continue

            positions = np.argwhere(grid_np == int(tile_id))
            if positions.size == 0:
                distances.append(float(ROOM_HEIGHT + ROOM_WIDTH))
                continue

            tile_present += 1
            min_dist = min(
                abs(int(pos[0]) - sr) + abs(int(pos[1]) - sc)
                for pos in positions
            )
            distances.append(float(min_dist))

        exact_rate = float(exact_matches) / float(expected)
        presence_rate = float(tile_present) / float(expected)
        avg_error = float(sum(distances) / len(distances)) if distances else 0.0
        max_error = float(max(distances)) if distances else 0.0
        return {
            f"{prefix}graph_marker_expected": float(expected),
            f"{prefix}graph_marker_exact_matches": float(exact_matches),
            f"{prefix}graph_marker_tile_present": float(tile_present),
            f"{prefix}graph_marker_exact_match_rate": exact_rate,
            f"{prefix}graph_marker_presence_rate": presence_rate,
            f"{prefix}semantic_anchor_avg_manhattan_error": avg_error,
            f"{prefix}semantic_anchor_max_manhattan_error": max_error,
        }

    @staticmethod
    def _aggregate_room_alignment_metrics(room_metric_dicts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate per-room semantic alignment metrics into dungeon-level summaries."""
        if not room_metric_dicts:
            return {
                "total_graph_marker_expected": 0.0,
                "total_graph_marker_overwrites": 0.0,
                "avg_neural_graph_marker_exact_match_rate": 1.0,
                "avg_final_pre_overlay_graph_marker_exact_match_rate": 1.0,
                "avg_final_post_overlay_graph_marker_exact_match_rate": 1.0,
                "avg_final_graph_marker_overwrite_rate": 0.0,
                "avg_neural_semantic_anchor_error": 0.0,
                "avg_final_pre_overlay_semantic_anchor_error": 0.0,
                "avg_final_post_overlay_semantic_anchor_error": 0.0,
            }

        def _mean(metric_key: str, default: float) -> float:
            return float(np.mean([float(m.get(metric_key, default)) for m in room_metric_dicts]))

        return {
            "total_graph_marker_expected": float(
                sum(float(m.get("final_pre_overlay_graph_marker_expected", 0.0)) for m in room_metric_dicts)
            ),
            "total_graph_marker_overwrites": float(
                sum(float(m.get("final_graph_marker_overwrites", 0.0)) for m in room_metric_dicts)
            ),
            "avg_neural_graph_marker_exact_match_rate": _mean(
                "neural_graph_marker_exact_match_rate",
                1.0,
            ),
            "avg_final_pre_overlay_graph_marker_exact_match_rate": _mean(
                "final_pre_overlay_graph_marker_exact_match_rate",
                1.0,
            ),
            "avg_final_post_overlay_graph_marker_exact_match_rate": _mean(
                "final_post_overlay_graph_marker_exact_match_rate",
                1.0,
            ),
            "avg_final_graph_marker_overwrite_rate": _mean(
                "final_graph_marker_overwrite_rate",
                0.0,
            ),
            "avg_neural_semantic_anchor_error": _mean(
                "neural_semantic_anchor_avg_manhattan_error",
                0.0,
            ),
            "avg_final_pre_overlay_semantic_anchor_error": _mean(
                "final_pre_overlay_semantic_anchor_avg_manhattan_error",
                0.0,
            ),
            "avg_final_post_overlay_semantic_anchor_error": _mean(
                "final_post_overlay_semantic_anchor_avg_manhattan_error",
                0.0,
            ),
        }

    def _build_latent_edit_mask(
        self,
        room_mask: np.ndarray,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        """Compatibility wrapper around extracted feedback helper."""
        return build_latent_edit_mask(
            room_mask,
            latent_h=latent_h,
            latent_w=latent_w,
            device=self.device,
        )

    @torch.no_grad()
    def _wfc_guided_inpaint_room(
        self,
        current_grid: np.ndarray,
        dead_end_mask: np.ndarray,
        condition: torch.Tensor,
        graph_data: Optional[Dict[str, Any]],
        num_diffusion_steps: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Compatibility wrapper around extracted feedback helper."""
        self._require_room_generation_components("_wfc_guided_inpaint_room")
        return wfc_guided_inpaint_room(
            current_grid=current_grid,
            dead_end_mask=dead_end_mask,
            condition=condition,
            graph_data=graph_data,
            num_diffusion_steps=num_diffusion_steps,
            seed=seed,
            device=self.device,
            vqvae=self.vqvae,
            diffusion=self.diffusion,
            num_classes=int(getattr(self.vqvae, "num_classes", int(np.max(self._valid_semantic_tile_ids_np)) + 1)),
        )

    def _compute_room_condition(
        self,
        *,
        neighbor_latents: Dict[str, Optional[torch.Tensor]],
        reference_room_maps: Optional[Dict[str, Optional[torch.Tensor]]] = None,
        graph_context: Dict[str, Any],
        boundary_constraints: Optional[torch.Tensor],
        position: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build Block-III conditioning tensor for a room."""
        if boundary_constraints is None:
            boundary_constraints = torch.zeros(1, 8, device=self.device)
        if position is None:
            position = torch.zeros(1, 2, device=self.device)

        node_tokens: Optional[torch.Tensor] = None
        condition_dim = int(getattr(self.condition_encoder, "output_dim", 256))
        style_id = graph_context.get("style_id")
        try:
            node_dim, edge_dim = self._condition_feature_dims()
            validate_feature_dims(
                node_features=graph_context.get('node_features'),
                edge_features=graph_context.get('edge_features'),
                expected_node_dim=node_dim,
                expected_edge_dim=edge_dim,
            )
            condition_out = self.condition_encoder(
                neighbor_latents=neighbor_latents,
                reference_room_maps=reference_room_maps,
                boundary_constraints=boundary_constraints,
                position=position,
                node_features=graph_context.get('node_features'),
                edge_index=graph_context.get('edge_index'),
                edge_features=graph_context.get('edge_features'),
                tpe=graph_context.get('tpe'),
                current_node_distance=graph_context.get('current_node_distance'),
                current_node_idx=graph_context.get('current_node_idx'),
                style_id=style_id,
                return_global_tokens=self.use_graph_node_cross_attention,
            )
            if self.use_graph_node_cross_attention:
                if not isinstance(condition_out, tuple) or len(condition_out) != 2:
                    raise ValueError(
                        "Condition encoder must return (conditioning, global_tokens) when "
                        "graph-node cross-attention is enabled."
                    )
                condition, node_tokens = condition_out
            else:
                condition = condition_out
            validate_tensor_contract(
                condition,
                BlockShapeContract(name='block_iii_condition_output', dims=2, batch_dim=1, channel_dim=condition_dim),
            )
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            self._bump_diagnostic("condition_encoder_fallback")
            if self.strict_checkpoint_mode:
                raise RuntimeError(
                    f"Condition encoding failed in strict mode: {e}"
                ) from e
            logger.warning(f"Condition encoding failed: {e}, using zero condition")
            if self.use_graph_node_cross_attention:
                num_nodes = 0
                node_features = graph_context.get("node_features")
                if isinstance(node_features, torch.Tensor) and node_features.dim() >= 2:
                    num_nodes = int(node_features.shape[0])
                required_seq_len = max(1, num_nodes + 1)
                condition = torch.zeros(1, required_seq_len, condition_dim, device=self.device)
            else:
                condition = torch.zeros(1, condition_dim, device=self.device)
            node_tokens = None

        if self.use_graph_node_cross_attention and isinstance(node_tokens, torch.Tensor):
            try:
                if node_tokens.dim() == 2:
                    node_tokens = node_tokens.unsqueeze(0)
                condition = torch.cat([condition.unsqueeze(1), node_tokens], dim=1)
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                self._bump_diagnostic("graph_node_cross_attention_fallback")
                logger.debug("Falling back to single conditioning vector: %s", e)

        puzzle_structure_condition_enabled = (
            self.masked_room_puzzle_structure_condition_enabled
            if self.room_generator_mode == "discrete_masked"
            else self.diffusion_puzzle_structure_condition_enabled
        )
        if puzzle_structure_condition_enabled and isinstance(condition, torch.Tensor):
            if condition.dim() == 3 and int(condition.shape[0]) == 1:
                condition = apply_puzzle_structure_control_to_conditioning(
                    condition.squeeze(0),
                    puzzle_structure_enabled=bool(graph_context.get("puzzle_room_structure_enabled", True)),
                    graph_conditioning_mode="node_sequence",
                ).unsqueeze(0)
            elif condition.dim() == 2:
                if bool(self.use_graph_node_cross_attention):
                    condition = apply_puzzle_structure_control_to_conditioning(
                        condition,
                        puzzle_structure_enabled=bool(graph_context.get("puzzle_room_structure_enabled", True)),
                        graph_conditioning_mode="node_sequence",
                    ).unsqueeze(0)
                else:
                    condition = apply_puzzle_structure_control_to_conditioning(
                        condition,
                        puzzle_structure_enabled=bool(graph_context.get("puzzle_room_structure_enabled", True)),
                        graph_conditioning_mode="pooled",
                    )

        return condition

    def _topological_generation_layers(self, graph: nx.Graph) -> List[List[Any]]:
        """Return dependency-safe topological layers for directed graphs."""
        if not graph.is_directed():
            return [sorted(list(graph.nodes()), key=_stable_node_sort_key)]
        try:
            layers = [
                sorted(list(layer), key=_stable_node_sort_key)
                for layer in nx.topological_generations(graph)
            ]
            return [layer for layer in layers if layer]
        except nx.NetworkXUnfeasible:
            logger.warning(
                "Mission graph contains cycles; disabling layer batching and using sorted node order."
            )
            return [sorted(list(graph.nodes()), key=_stable_node_sort_key)]

    def _infer_room_latent_shape(
        self,
        *,
        neighbor_latents: Dict[str, Optional[Any]],
    ) -> Tuple[int, int, int]:
        """Infer per-room latent (C,H,W) shape from neighbors or defaults."""
        if self.room_generator_mode == "discrete_masked":
            hidden_dim = int(getattr(self.masked_room_model, "hidden_dim", 64))
            default_shape = (hidden_dim, ROOM_HEIGHT, ROOM_WIDTH)
        else:
            diffusion = self._require_component("diffusion", "_infer_room_latent_shape")
            default_shape = (
                int(diffusion.latent_dim),
                int(DEFAULT_ROOM_LATENT_HW[0]),
                int(DEFAULT_ROOM_LATENT_HW[1]),
            )
        for latent in neighbor_latents.values():
            if isinstance(latent, torch.Tensor) and latent.dim() == 4:
                return (int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))
            if isinstance(latent, np.ndarray) and latent.ndim == 4:
                return (int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))
        return default_shape

    def _normalize_neighbor_latents(
        self,
        neighbor_latents: Dict[str, Optional[Any]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Normalize neighbor latents to detached float32 tensors on pipeline device.

        Accepts torch tensors and numpy arrays with expected shape [1, C, H, W].
        """
        normalized: Dict[str, Optional[torch.Tensor]] = {}
        for direction, latent in dict(neighbor_latents).items():
            direction_key = str(direction)
            if latent is None:
                normalized[direction_key] = None
                continue
            if isinstance(latent, torch.Tensor):
                if latent.dim() != 4:
                    raise ValueError(
                        f"Neighbor latent '{direction_key}' must be rank-4 tensor, got shape={tuple(latent.shape)}"
                    )
                normalized[direction_key] = latent.detach().to(self.device, dtype=torch.float32).contiguous()
                continue
            if isinstance(latent, np.ndarray):
                if latent.ndim != 4:
                    raise ValueError(
                        f"Neighbor latent '{direction_key}' must be rank-4 ndarray, got shape={tuple(latent.shape)}"
                    )
                normalized[direction_key] = (
                    torch.from_numpy(latent)
                    .detach()
                    .to(self.device, dtype=torch.float32)
                    .contiguous()
                )
                continue
            raise TypeError(
                f"Neighbor latent '{direction_key}' has unsupported type: {type(latent).__name__}"
            )
        return normalized

    def _cast_latent_for_vqvae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Match sampled latent dtype/device to the VQ-VAE decoder contract."""
        try:
            reference = next(self.vqvae.parameters())
            target_device = reference.device
            target_dtype = reference.dtype
        except StopIteration:
            target_device = self.device
            target_dtype = latent.dtype

        prepared = latent.contiguous()
        if prepared.device == target_device and prepared.dtype == target_dtype:
            return prepared
        return prepared.to(device=target_device, dtype=target_dtype)

    def _synchronize_cuda_device(self) -> None:
        """Conservatively drain queued CUDA work before cross-branch fallback handoffs."""
        if not torch.cuda.is_available():
            return
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        if device.type != "cuda":
            return
        try:
            torch.cuda.synchronize(device)
        except Exception:
            logger.debug("CUDA synchronize skipped during room-generation fallback handoff.", exc_info=True)

    def _decode_latent_with_vqvae(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent with a defensive retry for rare cuDNN stream-handoff failures."""
        prepared = self._cast_latent_for_vqvae_decode(latent)
        try:
            return self.vqvae.decode(prepared)
        except RuntimeError as exc:
            message = str(exc)
            if "stream_mismatch" not in message.lower() and "stream mismatch" not in message.lower():
                raise
            self._bump_diagnostic("vqvae_decode_stream_mismatch_retry")
            logger.warning(
                "VQ-VAE decode hit a CUDA stream mismatch; synchronizing device and retrying with cuDNN disabled."
            )
            self._synchronize_cuda_device()
            safe_latent = prepared.detach().clone().contiguous()
            with torch.backends.cudnn.flags(enabled=False):
                return self.vqvae.decode(safe_latent)

    def _estimate_safe_batch_size(
        self,
        *,
        requested_batch_size: int,
        latent_shape_chw: Tuple[int, int, int],
    ) -> int:
        """Estimate VRAM-safe batch size and clamp requested size accordingly."""
        requested = max(1, int(requested_batch_size))
        if not torch.cuda.is_available():
            return requested
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device=self.device)
            free_bytes = int(max(0, free_bytes))
        except Exception:
            return requested

        c, h, w = [max(1, int(v)) for v in latent_shape_chw]
        # Conservative estimate: latent + activations + guidance/intermediate buffers.
        bytes_per_sample = int(c * h * w * 4 * 18)
        reserve = int(1024 * 1024 * 256)  # 256MB reserve for fragmentation/runtime overhead
        usable = max(0, free_bytes - reserve)
        if bytes_per_sample <= 0 or usable <= 0:
            return 1
        safe = max(1, int(usable // bytes_per_sample))
        return max(1, min(requested, safe))

    def _stack_room_topology_maps(
        self,
        topology_maps: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Normalize per-room topology maps to a batched [B,C,H,W] tensor."""
        stacked: List[torch.Tensor] = []
        for topo in topology_maps:
            if not isinstance(topo, torch.Tensor):
                raise TypeError(f"room_topology_map must be a tensor, got {type(topo).__name__}")
            tensor = topo.to(self.device, dtype=torch.float32)
            if tensor.dim() == 4:
                if int(tensor.shape[0]) != 1:
                    raise ValueError(
                        f"Per-room room_topology_map must have batch size 1 when rank-4, got {tuple(tensor.shape)}."
                    )
                tensor = tensor.squeeze(0)
            elif tensor.dim() != 3:
                raise ValueError(
                    f"Per-room room_topology_map must have shape [1,C,H,W] or [C,H,W], got {tuple(tensor.shape)}."
                )
            stacked.append(tensor.contiguous())
        if not stacked:
            raise ValueError("Cannot batch room_topology_map for an empty room set.")
        return torch.stack(stacked, dim=0)

    @staticmethod
    def _slice_graph_guidance_batch(graph_ctx: Dict[str, Any], batch_index: int) -> Dict[str, Any]:
        """Slice a batched graph-guidance payload down to one room/sample."""
        sliced: Dict[str, Any] = {}
        for key, value in graph_ctx.items():
            if not isinstance(value, torch.Tensor):
                sliced[key] = value
                continue
            if value.dim() >= 1 and int(value.shape[0]) > batch_index:
                if key == "edge_index" and value.dim() == 2:
                    sliced[key] = value
                elif key == "node_features" and value.dim() == 2:
                    sliced[key] = value
                elif key == "edge_features" and value.dim() == 2:
                    sliced[key] = value
                else:
                    sliced[key] = value[batch_index:batch_index + 1]
            else:
                sliced[key] = value
        return sliced

    def _bucket_room_ids_by_latent_shape(
        self,
        *,
        room_ids: List[Any],
        mission_graph_physical: nx.Graph,
        room_latents: Dict[int, torch.Tensor],
    ) -> Dict[Tuple[int, int, int, int, int], List[Any]]:
        """Bucket independent rooms by latent shape and target room size."""
        buckets: Dict[Tuple[int, int, int, int, int], List[Any]] = {}
        for room_id in room_ids:
            neighbor_latents = self._get_neighbor_latents(room_id, mission_graph_physical, room_latents)
            shape_chw = self._infer_room_latent_shape(neighbor_latents=neighbor_latents)
            attrs = mission_graph_physical.nodes[room_id] if room_id in mission_graph_physical else {}
            target_h = int(attrs.get('room_height', attrs.get('height', ROOM_HEIGHT)))
            target_w = int(attrs.get('room_width', attrs.get('width', ROOM_WIDTH)))
            shape_key = (int(shape_chw[0]), int(shape_chw[1]), int(shape_chw[2]), target_h, target_w)
            if shape_key not in buckets:
                buckets[shape_key] = []
            buckets[shape_key].append(room_id)
        return buckets

    def _generate_room_batch(
        self,
        *,
        room_ids: List[Any],
        mission_graph_physical: nx.Graph,
        graph_data: Dict[str, Any],
        generated_rooms: Dict[Any, RoomGenerationResult],
        room_latents: Dict[int, torch.Tensor],
        guidance_scale: float,
        logic_guidance_scale: float,
        num_diffusion_steps: int,
        use_fast_sampling: bool,
        latent_sampler: str,
        categorical_codebook_size: Optional[int],
        apply_repair: bool,
        seed: Optional[int],
        layer_offset: int,
        latent_shape_chw: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[Any, RoomGenerationResult]:
        """Generate one dependency-safe room layer with batched diffusion decode."""
        self._require_room_generation_components("_generate_room_batch")
        if not room_ids:
            return {}

        sampler_mode = str(latent_sampler or "diffusion").strip().lower()
        batch_conditions: List[torch.Tensor] = []
        per_room_inputs: List[Dict[str, Any]] = []

        for j, room_id in enumerate(room_ids):
            neighbor_latents = self._normalize_neighbor_latents(
                self._get_neighbor_latents(room_id, mission_graph_physical, room_latents)
            )
            reference_room_maps = (
                self._get_neighbor_reference_room_maps(room_id, mission_graph_physical, generated_rooms)
                if bool(getattr(self.condition_encoder, "use_reference_room_maps", False))
                else None
            )
            boundary_constraints = self._build_room_boundary_constraints(
                graph=mission_graph_physical,
                room_id=room_id,
            )
            room_position = self._build_room_position_tensor(
                graph=mission_graph_physical,
                room_id=room_id,
                fallback_order_index=layer_offset + j,
            )
            room_seed = None
            if seed is not None:
                room_seed = int(seed) + int(_stable_node_seed_offset(room_id))
            start_goal = self._extract_room_start_goal(mission_graph_physical, room_id)

            room_graph_context = self._build_room_graph_context(
                graph_data=graph_data,
                mission_graph=mission_graph_physical,
                room_id=room_id,
                start_goal=start_goal,
            )
            condition = self._compute_room_condition(
                neighbor_latents=neighbor_latents,
                reference_room_maps=reference_room_maps,
                graph_context=room_graph_context,
                boundary_constraints=boundary_constraints,
                position=room_position,
            )
            batch_conditions.append(condition)
            per_room_inputs.append(
                {
                    'batch_index': int(j),
                    'room_id': room_id,
                    'neighbor_latents': neighbor_latents,
                    'reference_room_maps': reference_room_maps,
                    'graph_context': room_graph_context,
                    'boundary_constraints': boundary_constraints,
                    'position': room_position,
                    'start_goal': start_goal,
                    'seed': room_seed,
                }
            )

        if not batch_conditions or not per_room_inputs:
            return {}

        # Stack per-room conditions into one batch.
        expected_dim = int(batch_conditions[0].dim())
        if any(int(cond.dim()) != expected_dim for cond in batch_conditions):
            dims = [int(cond.dim()) for cond in batch_conditions]
            raise ValueError(f"Inconsistent condition tensor ranks inside batch: {dims}")
        condition_batch = torch.cat(batch_conditions, dim=0)
        first_room_graph_context = per_room_inputs[0]['graph_context']

        graph_ctx_for_guidance = {
            'graph_scope': 'dungeon',
            'node_features': graph_data.get('node_features'),
            'edge_index': graph_data.get('edge_index'),
            'edge_features': graph_data.get('edge_features'),
            'tpe': graph_data.get('tpe'),
            'node_positions': graph_data.get('node_positions'),
            'node_mask': graph_data.get('node_mask'),
            'start_node_id': graph_data.get(
                'start_node_id',
                first_room_graph_context.get('start_node_id', 0),
            ),
            'target_idx': graph_data.get(
                'target_idx',
                first_room_graph_context.get('target_idx', -1),
            ),
            'key_lock_pairs': graph_data.get(
                'key_lock_pairs',
                first_room_graph_context.get('key_lock_pairs', []),
            ),
            'boundary_constraints': torch.cat(
                [inp['boundary_constraints'].to(self.device, dtype=torch.float32) for inp in per_room_inputs],
                dim=0,
            ),
            'room_topology_map': self._stack_room_topology_maps(
                [inp['graph_context']['room_topology_map'] for inp in per_room_inputs]
            ),
        }

        # Map each sampled room latent back to its dungeon graph node.
        node_to_idx = graph_data.get('node_to_idx')
        if isinstance(node_to_idx, dict) and room_ids:
            current_node_idx_batch = []
            for room_id in room_ids:
                idx = node_to_idx.get(room_id, -1)
                current_node_idx_batch.append(int(idx))
            if all(idx >= 0 for idx in current_node_idx_batch):
                graph_ctx_for_guidance['current_node_idx'] = torch.tensor(
                    current_node_idx_batch,
                    device=self.device,
                    dtype=torch.long,
                )

        if self.use_current_node_distance_features:
            current_node_distance_batch: List[torch.Tensor] = []
            for inp in per_room_inputs:
                value = inp['graph_context']['current_node_distance']
                if not isinstance(value, torch.Tensor):
                    continue
                tensor = value.to(self.device, dtype=torch.float32)
                if tensor.dim() == 3 and int(tensor.shape[0]) == 1:
                    tensor = tensor.squeeze(0)
                current_node_distance_batch.append(tensor)
            if current_node_distance_batch:
                graph_ctx_for_guidance['current_node_distance'] = torch.stack(
                    current_node_distance_batch,
                    dim=0,
                )

        tokens_batch: Optional[torch.Tensor] = None
        if self.room_generator_mode == "discrete_masked":
            fixed_layouts = [
                self._build_masked_room_fixed_tokens(
                    mission_graph_physical,
                    inp['room_id'],
                    start_goal=inp['start_goal'],
                )
                for inp in per_room_inputs
            ]
            fixed_tokens = torch.cat([layout[0] for layout in fixed_layouts], dim=0)
            fixed_mask = torch.cat([layout[1] for layout in fixed_layouts], dim=0)
            tokens_batch, logits_batch, z_batch = self.masked_room_model.sample(
                context=condition_batch,
                graph_data=graph_ctx_for_guidance,
                fixed_tokens=fixed_tokens,
                fixed_mask=fixed_mask,
                num_steps=max(1, min(int(num_diffusion_steps), int(self.masked_sampling_steps))),
                temperature=float(self.default_masked_room_sampling_temperature),
                schedule_mode=self.default_masked_room_sampling_schedule,
                stochastic=bool(self.default_masked_room_sampling_stochastic),
                corrector_steps=int(self.default_masked_room_corrector_steps),
                corrector_mask_ratio=float(self.default_masked_room_corrector_mask_ratio),
                seed=seed,
            )
        elif sampler_mode == "categorical":
            self.diffusion.cfg_scale = float(guidance_scale)
            self.diffusion.guidance.logic_net = self.logic_net if logic_guidance_scale > 0 else None
            self.diffusion.guidance.guidance_scale = max(0.0, float(logic_guidance_scale))
            if hasattr(self.vqvae, "codebook_size"):
                num_embeddings = int(getattr(self.vqvae, "codebook_size"))
            else:
                num_embeddings = int(getattr(getattr(self.vqvae, "quantizer", object()), "num_embeddings", 512))
            active_codebook_size = int(max(1, min(num_embeddings, int(categorical_codebook_size or num_embeddings))))

            probs = np.ones(active_codebook_size, dtype=np.float64)
            try:
                usage = self.vqvae.get_codebook_usage()
                if isinstance(usage, torch.Tensor):
                    usage_np = usage.detach().float().cpu().numpy()
                    if usage_np.size >= active_codebook_size:
                        usage_np = np.asarray(usage_np[:active_codebook_size], dtype=np.float64)
                        if float(np.sum(usage_np)) > 0.0:
                            probs = usage_np
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
            probs = probs / max(float(np.sum(probs)), 1e-9)

            sampled = []
            for inp in per_room_inputs:
                local_rng = np.random.default_rng(inp['seed']) if inp['seed'] is not None else np.random.default_rng()
                sampled.append(
                    local_rng.choice(
                        active_codebook_size,
                        size=(latent_shape[2], latent_shape[3]),
                        p=probs,
                    )
                )
            indices_t = torch.from_numpy(np.stack(sampled, axis=0)).to(self.device, dtype=torch.long)
            logits_batch = self.vqvae.decode_indices(indices_t)
            z_batch = self.vqvae.quantizer.encode_indices(indices_t).permute(0, 3, 1, 2).contiguous()
        else:
            guidance_scale, logic_guidance_scale = self._resolve_effective_sampling_guidance(
                use_fast_sampling=use_fast_sampling,
                guidance_scale=float(guidance_scale),
                logic_guidance_scale=float(logic_guidance_scale),
            )
            self.diffusion.cfg_scale = float(guidance_scale)
            self.diffusion.guidance.logic_net = self.logic_net if logic_guidance_scale > 0 else None
            self.diffusion.guidance.guidance_scale = max(0.0, float(logic_guidance_scale))

            B = len(room_ids)
            if latent_shape_chw is None:
                latent_shape_chw = (
                    int(self.diffusion.latent_dim),
                    int(DEFAULT_ROOM_LATENT_HW[0]),
                    int(DEFAULT_ROOM_LATENT_HW[1]),
                )

            latent_shape: Tuple[int, int, int, int] = (
                B,
                int(latent_shape_chw[0]),
                int(latent_shape_chw[1]),
                int(latent_shape_chw[2]),
            )

            # Verify bucket uniformity for neighbor latent references.
            for inp in per_room_inputs:
                for latent in inp['neighbor_latents'].values():
                    shape_here: Optional[Tuple[int, int, int]] = None
                    if isinstance(latent, torch.Tensor) and latent.dim() == 4:
                        shape_here = (int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))
                    elif isinstance(latent, np.ndarray) and latent.ndim == 4:
                        shape_here = (int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))
                    if shape_here is not None and shape_here != tuple(latent_shape_chw):
                        raise ValueError(
                            f"Mixed latent shapes inside one batch: expected {latent_shape_chw}, got {shape_here}"
                        )
            if use_fast_sampling and self.diffusion.supports_fast_sampling():
                z_batch = self.diffusion.fast_sample(
                    context=condition_batch,
                    shape=latent_shape,
                    graph_data=graph_ctx_for_guidance,
                    guidance_scale=float(guidance_scale),
                    seed=seed,
                )
                self._bump_diagnostic("fast_sampling_used")
            else:
                if use_fast_sampling:
                    self._bump_diagnostic("fast_sampling_unavailable_fallback")
                z_batch = self.diffusion.ddim_sample(
                    context=condition_batch,
                    shape=latent_shape,
                    num_steps=max(1, int(num_diffusion_steps)),
                    graph_data=graph_ctx_for_guidance,
                )

            if self.use_latent_boundary_masking:
                for i, inp in enumerate(per_room_inputs):
                    try:
                        z_ref, boundary_edit_mask, has_boundary_constraints = build_neighbor_boundary_inpaint_inputs(
                            base_latent=z_batch[i:i + 1],
                            neighbor_latents=inp['neighbor_latents'],
                            band=1,
                        )
                        if has_boundary_constraints:
                            room_graph_guidance = self._slice_graph_guidance_batch(graph_ctx_for_guidance, i)
                            z_batch[i:i + 1] = self.diffusion.inpaint(
                                x_0=z_ref,
                                mask=boundary_edit_mask,
                                context=condition_batch[i:i + 1],
                                graph_data=room_graph_guidance,
                                num_steps=max(8, int(num_diffusion_steps) // 2),
                            )
                    except (AttributeError, RuntimeError, ValueError, TypeError):
                        continue
            logits_batch = self._decode_latent_with_vqvae(z_batch)

        out: Dict[Any, RoomGenerationResult] = {}
        for i, inp in enumerate(per_room_inputs):
            if int(inp['batch_index']) != int(i):
                raise RuntimeError(
                    f"Batch routing mismatch for room {inp['room_id']}: stored index={inp['batch_index']} actual={i}"
                )
            result_i = self.generate_room(
                neighbor_latents=inp['neighbor_latents'],
                graph_context=inp['graph_context'],
                room_id=inp['room_id'],
                boundary_constraints=inp['boundary_constraints'],
                position=inp['position'],
                reference_room_maps=inp['reference_room_maps'],
                guidance_scale=guidance_scale,
                logic_guidance_scale=logic_guidance_scale,
                num_diffusion_steps=num_diffusion_steps,
                use_fast_sampling=use_fast_sampling,
                latent_sampler=latent_sampler,
                categorical_codebook_size=categorical_codebook_size,
                apply_repair=apply_repair,
                start_goal_coords=inp['start_goal'],
                seed=inp['seed'],
                precomputed_condition=condition_batch[i:i + 1],
                precomputed_latent=z_batch[i:i + 1],
                precomputed_logits=logits_batch[i:i + 1],
                precomputed_tokens=(
                    tokens_batch[i:i + 1]
                    if isinstance(tokens_batch, torch.Tensor)
                    else None
                ),
            )
            out[inp['room_id']] = result_i

        return out
    
    @torch.no_grad()
    def generate_room(
        self,
        neighbor_latents: Dict[str, Optional[Any]],
        graph_context: Dict[str, Any],
        room_id: int,
        boundary_constraints: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        reference_room_maps: Optional[Dict[str, Optional[torch.Tensor]]] = None,
        guidance_scale: Optional[float] = None,
        logic_guidance_scale: Optional[float] = None,
        num_diffusion_steps: Optional[int] = None,
        use_fast_sampling: Optional[bool] = None,
        latent_sampler: Optional[str] = None,
        categorical_codebook_size: Optional[int] = None,
        use_ddim: bool = True,
        apply_repair: Optional[bool] = None,
        start_goal_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        seed: Optional[int] = None,
        precomputed_condition: Optional[torch.Tensor] = None,
        precomputed_latent: Optional[torch.Tensor] = None,
        precomputed_logits: Optional[torch.Tensor] = None,
        precomputed_tokens: Optional[torch.Tensor] = None,
        allow_teacher_fallback: Optional[bool] = None,
        room_generator_override: Optional[str] = None,
    ) -> RoomGenerationResult:
        """
        Generate a single room using the full 7-block pipeline.
        
        Args:
            neighbor_latents: Dict of neighboring room latents {'N': tensor, ...}
            graph_context: Graph data dict with:
                - node_features: (num_nodes, feature_dim)
                - edge_index: (2, num_edges)
                - tpe: Topological positional encoding
                - current_node_idx: Index of current room in graph
            room_id: Unique room identifier
            boundary_constraints: (1, 8) door mask tensor
            position: (1, 2) grid position
            guidance_scale: Classifier-free guidance scale
            logic_guidance_scale: LogicNet gradient guidance scale
            num_diffusion_steps: Number of DDIM/DDPM steps
            use_fast_sampling: Use a configured distilled fast sampler when available
            latent_sampler: "diffusion" (default) or "categorical"
            categorical_codebook_size: Optional cap for categorical sampling
            use_ddim: Use DDIM (deterministic) vs DDPM (stochastic)
            apply_repair: Apply symbolic WFC repair
            start_goal_coords: ((start_r, start_c), (goal_r, goal_c)) for repair
            seed: Random seed for reproducibility
            
        Returns:
            RoomGenerationResult with room grid, latents, and metrics
        """
        self._require_room_generation_components("generate_room")
        local_np_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if seed is not None:
            torch.manual_seed(seed)
        neighbor_latents = self._normalize_neighbor_latents(neighbor_latents)
        guidance_scale = self.default_guidance_scale if guidance_scale is None else float(guidance_scale)
        logic_guidance_scale = (
            self.default_logic_guidance_scale
            if logic_guidance_scale is None
            else float(logic_guidance_scale)
        )
        num_diffusion_steps = (
            self.default_num_diffusion_steps if num_diffusion_steps is None else int(num_diffusion_steps)
        )
        use_fast_sampling = (
            self.default_use_fast_sampling if use_fast_sampling is None else bool(use_fast_sampling)
        )
        latent_sampler = self.default_latent_sampler if latent_sampler is None else str(latent_sampler)
        if categorical_codebook_size is None and self.default_categorical_codebook_size is not None:
            categorical_codebook_size = int(self.default_categorical_codebook_size)
        apply_repair = self.default_apply_repair if apply_repair is None else bool(apply_repair)
        if start_goal_coords is None:
            start_goal_coords = self.default_start_goal_coords
        elif start_goal_coords is not None:
            start_goal_coords = self._normalize_start_goal_coords(start_goal_coords)
        effective_room_generator_mode = (
            self.room_generator_mode
            if room_generator_override is None
            else str(room_generator_override).strip().lower()
        )
        if allow_teacher_fallback is None:
            if effective_room_generator_mode == "discrete_masked":
                allow_teacher_fallback = self.default_masked_room_teacher_fallback_enabled
            else:
                allow_teacher_fallback = self.default_fast_sampler_teacher_fallback_enabled
        else:
            allow_teacher_fallback = bool(allow_teacher_fallback)

        if logic_guidance_scale > 0 and self.logic_net is None:
            self._bump_diagnostic("logic_guidance_disabled_missing_component")
            logger.warning(
                "Logic guidance requested for room %s but no logic_net component is configured; disabling guidance.",
                room_id,
            )
            logic_guidance_scale = 0.0
        if apply_repair and self.refiner is None:
            self._bump_diagnostic("repair_disabled_missing_component")
            logger.warning(
                "Symbolic repair requested for room %s but no refiner component is configured; using neural output.",
                room_id,
            )
            apply_repair = False
        
        if precomputed_condition is not None:
            condition = precomputed_condition.to(self.device)
        else:
            condition = self._compute_room_condition(
                neighbor_latents=neighbor_latents,
                reference_room_maps=reference_room_maps,
                graph_context=graph_context,
                boundary_constraints=boundary_constraints,
                position=position,
            )

        sampler_mode = str(latent_sampler or "diffusion").strip().lower()
        graph_data = graph_context if isinstance(graph_context, dict) else None
        if graph_data is not None and boundary_constraints is not None and "boundary_constraints" not in graph_data:
            graph_data = {
                **graph_data,
                "boundary_constraints": boundary_constraints.to(self.device, dtype=torch.float32),
            }
        mission_graph_for_room = graph_data.get("mission_graph") if isinstance(graph_data, dict) else None

        sampled_tokens: Optional[torch.Tensor] = None

        if precomputed_latent is not None and precomputed_logits is not None:
            z_latent = precomputed_latent.to(self.device)
            logits = precomputed_logits.to(self.device)
            if precomputed_tokens is not None:
                sampled_tokens = precomputed_tokens.to(self.device, dtype=torch.long)
        elif effective_room_generator_mode == "discrete_masked":
            fixed_tokens = None
            fixed_mask = None
            if mission_graph_for_room is not None:
                fixed_tokens, fixed_mask = self._build_masked_room_fixed_tokens(
                    mission_graph_for_room,
                    room_id,
                    start_goal=start_goal_coords,
                )
            sampled_tokens, logits, z_latent = self.masked_room_model.sample(
                context=condition,
                graph_data=graph_data,
                fixed_tokens=fixed_tokens,
                fixed_mask=fixed_mask,
                num_steps=max(1, min(int(num_diffusion_steps), int(self.masked_sampling_steps))),
                temperature=float(self.default_masked_room_sampling_temperature),
                schedule_mode=self.default_masked_room_sampling_schedule,
                stochastic=bool(self.default_masked_room_sampling_stochastic),
                corrector_steps=int(self.default_masked_room_corrector_steps),
                corrector_mask_ratio=float(self.default_masked_room_corrector_mask_ratio),
                seed=seed,
            )
        elif sampler_mode == "categorical":
            # Infer latent shape from neighbors when possible, otherwise use VQ-VAE spatial downsampling (x4).
            latent_shape: Tuple[int, int, int, int] = (
                1,
                int(self.diffusion.latent_dim),
                int(DEFAULT_ROOM_LATENT_HW[0]),
                int(DEFAULT_ROOM_LATENT_HW[1]),
            )
            for latent in neighbor_latents.values():
                if isinstance(latent, torch.Tensor) and latent.dim() == 4:
                    latent_shape = tuple(int(v) for v in latent.shape)  # type: ignore[assignment]
                    break
            logger.debug("Room %s: Sampling with categorical codebook path", room_id)
            latent_h = int(max(1, latent_shape[2]))
            latent_w = int(max(1, latent_shape[3]))
            if hasattr(self.vqvae, "codebook_size"):
                num_embeddings = int(getattr(self.vqvae, "codebook_size"))
            else:
                num_embeddings = int(getattr(getattr(self.vqvae, "quantizer", object()), "num_embeddings", 512))
            active_codebook_size = int(max(1, min(num_embeddings, int(categorical_codebook_size or num_embeddings))))

            probs = np.ones(active_codebook_size, dtype=np.float64)
            try:
                usage = self.vqvae.get_codebook_usage()
                if isinstance(usage, torch.Tensor):
                    usage_np = usage.detach().float().cpu().numpy()
                    if usage_np.size >= active_codebook_size:
                        usage_np = np.asarray(usage_np[:active_codebook_size], dtype=np.float64)
                        if float(np.sum(usage_np)) > 0.0:
                            probs = usage_np
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                self._bump_diagnostic("categorical_prior_fallback")
                logger.debug("Falling back to uniform categorical priors (codebook usage unavailable): %s", e)
            probs = np.asarray(probs, dtype=np.float64)
            probs = probs / max(float(np.sum(probs)), 1e-9)

            sampled_indices = local_np_rng.choice(
                active_codebook_size,
                size=(1, latent_h, latent_w),
                p=probs,
            )
            indices_t = torch.from_numpy(sampled_indices).to(self.device, dtype=torch.long)
            logits = self.vqvae.decode_indices(indices_t)  # (1, 44, 16, 11)
            with torch.no_grad():
                z_latent = self.vqvae.quantizer.encode_indices(indices_t).permute(0, 3, 1, 2).contiguous()
            validate_tensor_contract(
                z_latent,
                BlockShapeContract(name='block_iv_categorical_latent', dims=4, batch_dim=1),
            )
        else:
            # BLOCK V: Logic guidance configuration for diffusion sampler
            guidance_scale, logic_guidance_scale = self._resolve_effective_sampling_guidance(
                use_fast_sampling=use_fast_sampling,
                guidance_scale=float(guidance_scale),
                logic_guidance_scale=float(logic_guidance_scale),
            )
            self.diffusion.cfg_scale = float(guidance_scale)
            self.diffusion.guidance.logic_net = self.logic_net if logic_guidance_scale > 0 else None
            self.diffusion.guidance.guidance_scale = max(0.0, float(logic_guidance_scale))

            # Infer latent shape from neighbors when possible, otherwise use VQ-VAE spatial downsampling (x4).
            latent_shape = (
                1,
                int(self.diffusion.latent_dim),
                int(DEFAULT_ROOM_LATENT_HW[0]),
                int(DEFAULT_ROOM_LATENT_HW[1]),
            )
            for latent in neighbor_latents.values():
                if isinstance(latent, torch.Tensor) and latent.dim() == 4:
                    latent_shape = tuple(int(v) for v in latent.shape)  # type: ignore[assignment]
                    break

            # BLOCK IV: Latent Diffusion Sampling
            logger.debug(f"Room {room_id}: Sampling with {num_diffusion_steps} steps")
            if use_fast_sampling and self.diffusion.supports_fast_sampling():
                z_latent = self.diffusion.fast_sample(
                    context=condition,
                    shape=latent_shape,
                    graph_data=graph_data,
                    guidance_scale=float(guidance_scale),
                    seed=seed,
                )
                self._bump_diagnostic("fast_sampling_used")
            elif use_ddim:
                if use_fast_sampling:
                    self._bump_diagnostic("fast_sampling_unavailable_fallback")
                z_latent = self.diffusion.ddim_sample(
                    context=condition,
                    shape=latent_shape,
                    num_steps=max(1, int(num_diffusion_steps)),
                    graph_data=graph_data,
                )
            else:
                if use_fast_sampling:
                    self._bump_diagnostic("fast_sampling_unavailable_fallback")
                z_latent = self.diffusion.sample(
                    context=condition,
                    shape=latent_shape,
                    graph_data=graph_data,
                )

            validate_tensor_contract(
                z_latent,
                BlockShapeContract(
                    name='block_iv_diffusion_latent',
                    dims=4,
                    batch_dim=1,
                    channel_dim=int(self.diffusion.latent_dim),
                ),
            )

            # Autoregressive spatial generation: preserve known boundary latents from generated neighbors.
            if self.use_latent_boundary_masking:
                try:
                    z_ref, boundary_edit_mask, has_boundary_constraints = build_neighbor_boundary_inpaint_inputs(
                        base_latent=z_latent,
                        neighbor_latents=neighbor_latents,
                        band=1,
                    )
                    if has_boundary_constraints:
                        z_latent = self.diffusion.inpaint(
                            x_0=z_ref,
                            mask=boundary_edit_mask,
                            context=condition,
                            graph_data=graph_data,
                            num_steps=max(8, int(num_diffusion_steps) // 2),
                            noise_strength=0.25,  # Lower noise for boundary blending (not full regeneration)
                        )
                        self._bump_diagnostic("boundary_latent_masking_applied")
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    self._bump_diagnostic("boundary_latent_masking_fallback")
                    logger.debug("Boundary latent masking skipped due to error: %s", e)

            # BLOCK II: VQ-VAE Decoding
            logits = self._decode_latent_with_vqvae(z_latent)  # (1, 44, 16, 11)
        validate_tensor_contract(
            logits,
            BlockShapeContract(
                name='block_ii_decode_logits',
                dims=4,
                batch_dim=1,
                channel_dim=int(getattr(self.vqvae, "num_classes", logits.shape[1])),
                spatial_hw=(ROOM_HEIGHT, ROOM_WIDTH),
            ),
        )
        
        # BLOCK II.a: Topology-Enforced Constrained Decoding
        # Clamp doorway logits to the exact door type implied by graph semantics
        # before argmax. This keeps the topology constraint inside the decoder
        # instead of stamping a mismatched discrete tile after generation.
        door_tiles_forced = 0
        if isinstance(mission_graph_for_room, nx.Graph) and room_id in mission_graph_for_room:
            try:
                neg_large = float(-1e4)
                pos_large = float(1e4)
                semantics = self._extract_room_topology_semantics(mission_graph_for_room, room_id)
                required_doors = semantics.get("required_doors", {})
                for direction, is_required in required_doors.items():
                    if not is_required:
                        continue
                    spec = DOOR_POSITIONS.get(direction)
                    if spec is None:
                        continue
                    door_tile = int(
                        self._edge_tokens_to_door_tile(
                            semantics.get("edge_constraints", {}).get(direction, set())
                        )
                    )
                    
                    if direction in {"N", "S"}:
                        row = int(max(0, min(ROOM_HEIGHT - 1, spec["row"])))
                        col_start = int(max(0, min(ROOM_WIDTH - 1, spec["col_start"])))
                        col_end = int(max(0, min(ROOM_WIDTH - 1, spec["col_end"])))
                        for c in range(col_start, col_end + 1):
                            if int(logits[0, :, row, c].argmax()) != door_tile:
                                logits[0, :, row, c] = neg_large
                                logits[0, door_tile, row, c] = pos_large
                                door_tiles_forced += 1
                    else:
                        col = int(max(0, min(ROOM_WIDTH - 1, spec["col"])))
                        row_start = int(max(0, min(ROOM_HEIGHT - 1, spec["row_start"])))
                        row_end = int(max(0, min(ROOM_HEIGHT - 1, spec["row_end"])))
                        for r in range(row_start, row_end + 1):
                            if int(logits[0, :, r, col].argmax()) != door_tile:
                                logits[0, :, r, col] = neg_large
                                logits[0, door_tile, r, col] = pos_large
                                door_tiles_forced += 1
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                logger.debug("Topology door constrained decoding skipped for room %s: %s", room_id, e)
                
        if door_tiles_forced > 0:
            self._bump_diagnostic("topology_door_tiles_forced")
            logger.debug("Room %s: forced %d door tiles via constrained decoding", room_id, door_tiles_forced)

        semantic_decode_stats = self._apply_semantic_constrained_decoding(
            logits,
            graph=mission_graph_for_room,
            room_id=room_id,
            start_goal=start_goal_coords,
        )
        if int(semantic_decode_stats.get("biased_slots", 0)) > 0:
            self._bump_diagnostic("semantic_constrained_decode_applied")
            logger.debug(
                "Room %s: biased %d/%d planned graph-marker slots via semantic constrained decoding",
                room_id,
                int(semantic_decode_stats.get("biased_slots", 0)),
                int(semantic_decode_stats.get("planned_markers", 0)),
            )

        if effective_room_generator_mode == "discrete_masked" and sampled_tokens is not None:
            neural_grid = sampled_tokens.detach().cpu().numpy()[0].astype(np.int32, copy=False)
        else:
            neural_grid = logits.argmax(dim=1).detach().cpu().numpy()[0]  # (16, 11)
        neural_grid, neural_invalid_count, neural_invalid_ids = self._sanitize_semantic_grid(
            neural_grid,
            strip_void=True,
        )
        if neural_invalid_count > 0:
            self._bump_diagnostic("neural_invalid_tile_ids_sanitized")
            logger.warning(
                "Room %s neural decode produced invalid tile IDs %s (count=%d); sanitized.",
                room_id,
                neural_invalid_ids,
                neural_invalid_count,
            )
        neural_grid, neural_semantic_strip_count, neural_semantic_strip_ids, neural_semantic_preserved_count, neural_semantic_preserved_ids = self._strip_volatile_room_semantics(
            neural_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            start_goal=start_goal_coords,
        )
        if neural_semantic_strip_count > 0:
            self._bump_diagnostic("neural_room_semantics_stripped")
            logger.debug(
                "Room %s stripped %d volatile semantic tiles from neural output: %s",
                room_id,
                neural_semantic_strip_count,
                neural_semantic_strip_ids,
            )
        if neural_semantic_preserved_count > 0:
            self._bump_diagnostic("neural_graph_semantic_hints_salvaged")
            logger.debug(
                "Room %s preserved %d graph-owned semantic hints from neural output: %s",
                room_id,
                neural_semantic_preserved_count,
                neural_semantic_preserved_ids,
            )
        neural_structural_cleanup = {
            "invalid_door_tiles_removed": 0,
            "interior_obstacle_tiles_removed": 0,
            "interior_obstacle_components_removed": 0,
        }
        if effective_room_generator_mode == "latent_diffusion":
            neural_grid, neural_structural_cleanup = self._strip_structural_room_artifacts(
                neural_grid,
                graph=mission_graph_for_room,
                room_id=room_id,
            )
            if any(int(v) > 0 for v in neural_structural_cleanup.values()):
                self._bump_diagnostic("neural_structural_artifacts_stripped")
                logger.debug(
                    "Room %s stripped structural artifacts from neural output: %s",
                    room_id,
                    neural_structural_cleanup,
                )
        neural_probs = logits.softmax(dim=1).detach().cpu().numpy()[0]  # (44, 16, 11)
        
        # BLOCK III: Removed (Migrated to Block II.a Constrained Decoding)
        
        # BLOCK VI: Symbolic Repair (if enabled)
        was_repaired = False
        repair_mask = None
        room_plan_mask = None
        final_grid = neural_grid.copy()
        repaired_invalid_count = 0
        repaired_invalid_ids: List[int] = []
        repaired_semantic_strip_count = 0
        repaired_semantic_strip_ids: List[int] = []
        repaired_semantic_preserved_count = 0
        repaired_semantic_preserved_ids: List[int] = []
        neural_boundary_shell = {
            "boundary_wall_tiles_forced": 0,
            "boundary_door_tiles_forced": 0,
            "interior_door_apron_tiles_forced": 0,
        }
        repaired_boundary_shell = {
            "boundary_wall_tiles_forced": 0,
            "boundary_door_tiles_forced": 0,
            "interior_door_apron_tiles_forced": 0,
        }
        neural_puzzle_scaffold = {
            "applied": 0,
            "tiles_added": 0,
            "segments_added": 0,
            "existing_structure_tiles": 0,
            "planned_route_pixels": 0,
        }
        final_puzzle_scaffold = {
            "applied": 0,
            "tiles_added": 0,
            "segments_added": 0,
            "existing_structure_tiles": 0,
            "planned_route_pixels": 0,
        }
        neural_no_puzzle_structure_cleanup = {
            "applied": 0,
            "block_tiles_removed": 0,
            "block_components_removed": 0,
        }
        final_no_puzzle_structure_cleanup = {
            "applied": 0,
            "block_tiles_removed": 0,
            "block_components_removed": 0,
        }
        repaired_structural_cleanup = {
            "invalid_door_tiles_removed": 0,
            "interior_obstacle_tiles_removed": 0,
            "interior_obstacle_components_removed": 0,
        }
        repair_diag: Dict[str, Any] = {}
        normalized_start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        if start_goal_coords is not None:
            normalized_start_goal = self._normalize_start_goal_coords(start_goal_coords)
        
        if apply_repair and start_goal_coords is not None:
            start, goal = normalized_start_goal if normalized_start_goal is not None else self._normalize_start_goal_coords(start_goal_coords)
            try:
                if isinstance(mission_graph_for_room, nx.Graph) and room_id in mission_graph_for_room:
                    room_plan_mask = self._build_room_plan_trace(
                        mission_graph_for_room,
                        room_id,
                        neural_grid,
                        start_goal=(start, goal),
                    )
            except (AttributeError, RuntimeError, ValueError, TypeError):
                room_plan_mask = None
            try:
                def _feedback_callback(
                    current_grid_cb: np.ndarray,
                    dead_end_mask_cb: np.ndarray,
                    _start_cb: Tuple[int, int],
                    _goal_cb: Tuple[int, int],
                    attempt_idx: int,
                ) -> np.ndarray:
                    return self._wfc_guided_inpaint_room(
                        current_grid=current_grid_cb,
                        dead_end_mask=dead_end_mask_cb,
                        condition=condition,
                        graph_data=graph_data,
                        num_diffusion_steps=max(12, int(num_diffusion_steps) // 2),
                        seed=(None if seed is None else int(seed) + 1000 + int(attempt_idx)),
                    )

                repaired_grid, success, repair_diag = self.repair_room(
                    grid=neural_grid,
                    start=start,
                    goal=goal,
                    required_floor_mask=room_plan_mask,
                    feedback_callback=_feedback_callback,
                    max_feedback_rounds=2,
                )
                
                if success:
                    repaired_grid_raw = repaired_grid.copy()
                    repaired_grid, repaired_invalid_count, repaired_invalid_ids = self._sanitize_semantic_grid(
                        repaired_grid,
                        fallback_grid=neural_grid,
                        strip_void=True,
                    )
                    if repaired_invalid_count > 0:
                        self._bump_diagnostic("repair_invalid_tile_ids_sanitized")
                        logger.warning(
                            "Room %s repair produced invalid tile IDs %s (count=%d); replaced using neural fallback.",
                            room_id,
                            repaired_invalid_ids,
                            repaired_invalid_count,
                        )
                        logger.debug(
                            "Room %s neural grid before repair:\n%s",
                            room_id,
                            np.array2string(neural_grid, max_line_width=240),
                        )
                        logger.debug(
                            "Room %s repaired grid before sanitize:\n%s",
                            room_id,
                            np.array2string(repaired_grid_raw, max_line_width=240),
                        )
                    repaired_grid, repaired_semantic_strip_count, repaired_semantic_strip_ids, repaired_semantic_preserved_count, repaired_semantic_preserved_ids = (
                        self._strip_volatile_room_semantics(
                            repaired_grid,
                            graph=mission_graph_for_room,
                            room_id=room_id,
                            start_goal=normalized_start_goal if normalized_start_goal is not None else start_goal_coords,
                        )
                    )
                    if repaired_semantic_strip_count > 0:
                        self._bump_diagnostic("repair_room_semantics_stripped")
                        logger.debug(
                            "Room %s stripped %d volatile semantic tiles after repair: %s",
                            room_id,
                            repaired_semantic_strip_count,
                            repaired_semantic_strip_ids,
                        )
                    if repaired_semantic_preserved_count > 0:
                        self._bump_diagnostic("repair_graph_semantic_hints_salvaged")
                        logger.debug(
                            "Room %s preserved %d graph-owned semantic hints after repair: %s",
                            room_id,
                            repaired_semantic_preserved_count,
                            repaired_semantic_preserved_ids,
                        )
                    if effective_room_generator_mode == "latent_diffusion":
                        repaired_grid, repaired_structural_cleanup = self._strip_structural_room_artifacts(
                            repaired_grid,
                            graph=mission_graph_for_room,
                            room_id=room_id,
                        )
                        if any(int(v) > 0 for v in repaired_structural_cleanup.values()):
                            self._bump_diagnostic("repair_structural_artifacts_stripped")
                            logger.debug(
                                "Room %s stripped structural artifacts after repair: %s",
                                room_id,
                                repaired_structural_cleanup,
                            )
                    repair_mask = (repaired_grid != neural_grid)
                    final_grid = repaired_grid
                    was_repaired = bool(np.any(repair_mask))
                    logger.debug(f"Room {room_id}: Repair successful ({np.sum(repair_mask)} tiles changed)")
                else:
                    logger.warning(f"Room {room_id}: Repair failed, using neural output")
                self._bump_diagnostic("wfc_feedback_attempts")
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                self._bump_diagnostic("room_repair_exception")
                logger.error(f"Room {room_id}: Repair error: {e}")
        elif start_goal_coords is not None:
            start, goal = normalized_start_goal if normalized_start_goal is not None else self._normalize_start_goal_coords(start_goal_coords)
            try:
                if isinstance(mission_graph_for_room, nx.Graph) and room_id in mission_graph_for_room:
                    room_plan_mask = self._build_room_plan_trace(
                        mission_graph_for_room,
                        room_id,
                        neural_grid,
                        start_goal=(start, goal),
                    )
            except (AttributeError, RuntimeError, ValueError, TypeError):
                room_plan_mask = None

        neural_grid, neural_boundary_shell = self._enforce_room_boundary_shell(
            neural_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
        )
        final_grid, repaired_boundary_shell = self._enforce_room_boundary_shell(
            final_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
        )
        if any(int(v) > 0 for v in neural_boundary_shell.values()):
            self._bump_diagnostic("neural_boundary_shell_enforced")
            logger.debug(
                "Room %s enforced boundary shell on neural output: %s",
                room_id,
                neural_boundary_shell,
            )
        if any(int(v) > 0 for v in repaired_boundary_shell.values()):
            self._bump_diagnostic("final_boundary_shell_enforced")
            logger.debug(
                "Room %s enforced boundary shell on final output: %s",
                room_id,
                repaired_boundary_shell,
            )

        overlay_start_goal = normalized_start_goal
        if overlay_start_goal is None and isinstance(mission_graph_for_room, nx.Graph) and room_id in mission_graph_for_room:
            overlay_start_goal = self._extract_room_start_goal(mission_graph_for_room, room_id)

        neural_grid, _, _, neural_post_boundary_preserved_count, neural_post_boundary_preserved_ids = (
            self._strip_volatile_room_semantics(
                neural_grid,
                graph=mission_graph_for_room,
                room_id=room_id,
                start_goal=overlay_start_goal,
            )
        )
        if neural_post_boundary_preserved_count > 0:
            self._bump_diagnostic("neural_post_boundary_graph_semantic_hints_salvaged")
            logger.debug(
                "Room %s re-salvaged %d graph-owned semantic hints after boundary enforcement: %s",
                room_id,
                neural_post_boundary_preserved_count,
                neural_post_boundary_preserved_ids,
            )

        final_grid, _, _, final_post_boundary_preserved_count, final_post_boundary_preserved_ids = (
            self._strip_volatile_room_semantics(
                final_grid,
                graph=mission_graph_for_room,
                room_id=room_id,
                start_goal=overlay_start_goal,
            )
        )
        if final_post_boundary_preserved_count > 0:
            self._bump_diagnostic("final_post_boundary_graph_semantic_hints_salvaged")
            logger.debug(
                "Room %s re-salvaged %d graph-owned semantic hints on final grid after boundary enforcement: %s",
                room_id,
                final_post_boundary_preserved_count,
                final_post_boundary_preserved_ids,
            )

        neural_grid, neural_void_cleanup = self._strip_room_void_tiles(neural_grid)
        final_grid, final_void_cleanup = self._strip_room_void_tiles(final_grid)
        if any(int(v) > 0 for v in neural_void_cleanup.values()):
            self._bump_diagnostic("neural_void_tiles_stripped")
            logger.debug(
                "Room %s stripped VOID tiles from neural output: %s",
                room_id,
                neural_void_cleanup,
            )
        if any(int(v) > 0 for v in final_void_cleanup.values()):
            self._bump_diagnostic("final_void_tiles_stripped")
            logger.debug(
                "Room %s stripped VOID tiles from final output: %s",
                room_id,
                final_void_cleanup,
            )

        neural_grid, neural_puzzle_scaffold = self._apply_puzzle_room_scaffold(
            neural_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            room_plan_mask=room_plan_mask,
            start_goal=overlay_start_goal,
        )
        final_grid, final_puzzle_scaffold = self._apply_puzzle_room_scaffold(
            final_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            room_plan_mask=room_plan_mask,
            start_goal=overlay_start_goal,
        )
        if int(final_puzzle_scaffold.get("applied", 0)) > 0:
            self._commit_puzzle_novelty_choice(
                room_id=room_id,
                scaffold_stats=final_puzzle_scaffold,
            )
            self._bump_diagnostic("puzzle_room_scaffold_applied")
            puzzle_archetype = str(final_puzzle_scaffold.get("archetype", "")).strip().lower()
            if puzzle_archetype:
                self._bump_diagnostic(f"puzzle_room_scaffold_{puzzle_archetype}")
            puzzle_gate_family = str(final_puzzle_scaffold.get("gate_family", "")).strip().lower()
            if puzzle_gate_family:
                self._bump_diagnostic(f"puzzle_room_scaffold_gate_{puzzle_gate_family}")
            if str(final_puzzle_scaffold.get("variant_name", "")).strip():
                self._bump_diagnostic("puzzle_room_scaffold_novelty_selected")
            logger.debug(
                "Room %s applied puzzle scaffold: %s",
                room_id,
                final_puzzle_scaffold,
            )
        if int(final_puzzle_scaffold.get("contract_valid", 0) or 0) > 0:
            self._bump_diagnostic("puzzle_room_contract_valid")
        elif str(final_puzzle_scaffold.get("gate_family", "")).strip():
            self._bump_diagnostic("puzzle_room_contract_invalid")
        if int(final_puzzle_scaffold.get("contract_gate_skipped", 0) or 0) > 0:
            self._bump_diagnostic("puzzle_room_contract_gate_skipped")
        if int(final_puzzle_scaffold.get("interaction_valid", 0) or 0) > 0:
            self._bump_diagnostic("puzzle_room_interaction_valid")
        elif str(final_puzzle_scaffold.get("gate_family", "")).strip():
            self._bump_diagnostic("puzzle_room_interaction_invalid")
        if int(final_puzzle_scaffold.get("interaction_gate_skipped", 0) or 0) > 0:
            self._bump_diagnostic("puzzle_room_interaction_gate_skipped")
        if int(final_puzzle_scaffold.get("interaction_sequence_valid", 0) or 0) > 0:
            self._bump_diagnostic("puzzle_room_sequence_valid")
        elif int(final_puzzle_scaffold.get("interaction_sequence_required", 0) or 0) > 0:
            self._bump_diagnostic("puzzle_room_sequence_invalid")
        if int(final_puzzle_scaffold.get("sequence_gate_skipped", 0) or 0) > 0:
            self._bump_diagnostic("puzzle_room_sequence_gate_skipped")
        if int(final_puzzle_scaffold.get("quality_gate_skipped", 0) or 0) > 0:
            self._bump_diagnostic("puzzle_room_quality_gate_skipped")

        if not bool(self.default_puzzle_room_structure_enabled):
            neural_grid, neural_no_puzzle_structure_cleanup = self._strip_room_block_structure(
                neural_grid,
                graph=mission_graph_for_room,
                room_id=room_id,
            )
            final_grid, final_no_puzzle_structure_cleanup = self._strip_room_block_structure(
                final_grid,
                graph=mission_graph_for_room,
                room_id=room_id,
            )
            if int(final_no_puzzle_structure_cleanup.get("applied", 0)) > 0:
                self._bump_diagnostic("no_puzzle_block_structure_stripped")

        neural_pre_marker_grid = np.asarray(neural_grid, dtype=np.int32).copy()
        final_pre_marker_grid = np.asarray(final_grid, dtype=np.int32).copy()
        neural_marker_plan = self._plan_room_graph_marker_layout(
            neural_pre_marker_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            start_goal=overlay_start_goal,
        )
        final_marker_plan = self._plan_room_graph_marker_layout(
            final_pre_marker_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            start_goal=overlay_start_goal,
        )

        if bool(self.default_deterministic_graph_marker_overlay_enabled):
            neural_grid, neural_marker_count, neural_marker_ids = self._overlay_room_graph_markers(
                neural_grid,
                graph=mission_graph_for_room,
                room_id=room_id,
                start_goal=overlay_start_goal,
            )
            final_grid, final_marker_count, final_marker_ids = self._overlay_room_graph_markers(
                final_grid,
                graph=mission_graph_for_room,
                room_id=room_id,
                start_goal=overlay_start_goal,
            )
            if final_marker_count > 0:
                logger.debug(
                    "Room %s placed %d graph-owned semantic markers: %s",
                    room_id,
                    final_marker_count,
                    final_marker_ids,
                )
        else:
            neural_marker_count = 0
            final_marker_count = 0
            neural_marker_ids = []
            final_marker_ids = []
            self._bump_diagnostic("deterministic_graph_marker_overlay_disabled")

        neural_marker_alignment = self._measure_room_graph_marker_alignment(
            neural_pre_marker_grid,
            placements=neural_marker_plan,
            prefix="neural_",
        )
        final_pre_overlay_alignment = self._measure_room_graph_marker_alignment(
            final_pre_marker_grid,
            placements=final_marker_plan,
            prefix="final_pre_overlay_",
        )
        final_post_overlay_alignment = self._measure_room_graph_marker_alignment(
            final_grid,
            placements=final_marker_plan,
            prefix="final_post_overlay_",
        )
        final_marker_overwrites = sum(
            int(final_pre_marker_grid[int(slot[0]), int(slot[1])]) != int(tile_id)
            for tile_id, slot in final_marker_plan
        )
        final_marker_expected = max(1, len(final_marker_plan))
        final_marker_overwrite_rate = float(final_marker_overwrites) / float(final_marker_expected)
        room_puzzle_metadata = self._build_room_puzzle_metadata(
            grid=final_grid,
            graph=mission_graph_for_room,
            room_id=room_id,
            start_goal=overlay_start_goal,
            marker_plan=final_marker_plan,
            scaffold_stats=final_puzzle_scaffold,
        )

        # VGLC Compliance: Validate room dimensions
        valid_dims, dim_msg = validate_room_dimensions(final_grid)
        if not valid_dims:
            logger.error(f"Room {room_id}: VGLC dimension validation FAILED: {dim_msg}")
            raise ValueError(f"Generated room has invalid dimensions: {dim_msg}")
        else:
            logger.debug(f"Room {room_id}: VGLC dimension validation PASSED")
        
        # Compute metrics
        entropy_val = float(
            np.mean(
                -(
                    logits.softmax(dim=1).detach()
                    * logits.log_softmax(dim=1).detach()
                ).sum(dim=1).cpu().numpy()
            )
        )
        latent_cpu = z_latent.detach().to(device='cpu', dtype=torch.float32).contiguous()

        metrics = {
            'room_id': room_id,
            'neural_grid_entropy': entropy_val,
            'was_repaired': was_repaired,
            'tiles_changed': int(np.sum(repair_mask)) if repair_mask is not None else 0,
            'neural_invalid_tile_ids': int(neural_invalid_count),
            'repair_invalid_tile_ids': int(repaired_invalid_count),
            'neural_semantic_tiles_stripped': int(neural_semantic_strip_count),
            'neural_graph_semantic_hints_salvaged': int(neural_semantic_preserved_count),
            'repair_semantic_tiles_stripped': int(repaired_semantic_strip_count),
            'repair_graph_semantic_hints_salvaged': int(repaired_semantic_preserved_count),
            'neural_invalid_door_tiles_removed': int(neural_structural_cleanup['invalid_door_tiles_removed']),
            'neural_interior_obstacle_tiles_removed': int(neural_structural_cleanup['interior_obstacle_tiles_removed']),
            'neural_interior_obstacle_components_removed': int(neural_structural_cleanup['interior_obstacle_components_removed']),
            'neural_boundary_void_tiles_removed': int(neural_void_cleanup['boundary_void_tiles_removed']),
            'neural_interior_void_tiles_removed': int(neural_void_cleanup['interior_void_tiles_removed']),
            'neural_boundary_wall_tiles_forced': int(neural_boundary_shell['boundary_wall_tiles_forced']),
            'neural_boundary_door_tiles_forced': int(neural_boundary_shell['boundary_door_tiles_forced']),
            'neural_interior_door_apron_tiles_forced': int(neural_boundary_shell['interior_door_apron_tiles_forced']),
            'repair_invalid_door_tiles_removed': int(repaired_structural_cleanup['invalid_door_tiles_removed']),
            'repair_interior_obstacle_tiles_removed': int(repaired_structural_cleanup['interior_obstacle_tiles_removed']),
            'repair_interior_obstacle_components_removed': int(repaired_structural_cleanup['interior_obstacle_components_removed']),
            'final_boundary_void_tiles_removed': int(final_void_cleanup['boundary_void_tiles_removed']),
            'final_interior_void_tiles_removed': int(final_void_cleanup['interior_void_tiles_removed']),
            'repair_boundary_wall_tiles_forced': int(repaired_boundary_shell['boundary_wall_tiles_forced']),
            'repair_boundary_door_tiles_forced': int(repaired_boundary_shell['boundary_door_tiles_forced']),
            'repair_interior_door_apron_tiles_forced': int(repaired_boundary_shell['interior_door_apron_tiles_forced']),
            'neural_puzzle_scaffold_applied': int(neural_puzzle_scaffold['applied']),
            'neural_puzzle_scaffold_tiles_added': int(neural_puzzle_scaffold['tiles_added']),
            'neural_puzzle_scaffold_segments_added': int(neural_puzzle_scaffold['segments_added']),
            'neural_puzzle_scaffold_optional_segments_requested': int(neural_puzzle_scaffold.get('optional_segments_requested', 0)),
            'neural_puzzle_scaffold_optional_segments_applied': int(neural_puzzle_scaffold.get('optional_segments_applied', 0)),
            'neural_puzzle_scaffold_route_template_used': int(neural_puzzle_scaffold.get('route_template_used', 0)),
            'neural_puzzle_scaffold_noise_components_removed': int(neural_puzzle_scaffold.get('noise_components_removed', 0)),
            'neural_puzzle_scaffold_noise_tiles_removed': int(neural_puzzle_scaffold.get('noise_tiles_removed', 0)),
            'neural_puzzle_scaffold_novelty_score': float(neural_puzzle_scaffold.get('novelty_score', 0.0)),
            'neural_puzzle_scaffold_variant_name': str(neural_puzzle_scaffold.get('variant_name', '') or ''),
            'neural_puzzle_scaffold_variant_style': str(neural_puzzle_scaffold.get('variant_style', '') or ''),
            'neural_puzzle_scaffold_variant_side_bias': int(neural_puzzle_scaffold.get('variant_side_bias', 0) or 0),
            'neural_puzzle_scaffold_interaction_valid': int(neural_puzzle_scaffold.get('interaction_valid', 0) or 0),
            'neural_puzzle_scaffold_interaction_score': float(neural_puzzle_scaffold.get('interaction_score', 0.0) or 0.0),
            'neural_puzzle_scaffold_interaction_push_slot_count': int(neural_puzzle_scaffold.get('interaction_push_slot_count', 0) or 0),
            'neural_puzzle_scaffold_interaction_barrier_axis_tiles': int(neural_puzzle_scaffold.get('interaction_barrier_axis_tiles', 0) or 0),
            'neural_puzzle_scaffold_interaction_route_divergence': float(neural_puzzle_scaffold.get('interaction_route_divergence', 0.0) or 0.0),
            'neural_puzzle_scaffold_interaction_sequence_valid': int(neural_puzzle_scaffold.get('interaction_sequence_valid', 0) or 0),
            'neural_puzzle_scaffold_interaction_sequence_score': float(neural_puzzle_scaffold.get('interaction_sequence_score', 0.0) or 0.0),
            'neural_puzzle_scaffold_interaction_sequence_length': int(neural_puzzle_scaffold.get('interaction_sequence_length', 0) or 0),
            'neural_puzzle_scaffold_interaction_sequence_route_anchor_coverage': float(neural_puzzle_scaffold.get('interaction_sequence_route_anchor_coverage', 0.0) or 0.0),
            'neural_puzzle_scaffold_interaction_sequence_pairwise_path_ratio': float(neural_puzzle_scaffold.get('interaction_sequence_pairwise_path_ratio', 0.0) or 0.0),
            'final_puzzle_scaffold_applied': int(final_puzzle_scaffold['applied']),
            'final_puzzle_scaffold_tiles_added': int(final_puzzle_scaffold['tiles_added']),
            'final_puzzle_scaffold_segments_added': int(final_puzzle_scaffold['segments_added']),
            'final_puzzle_scaffold_optional_segments_requested': int(final_puzzle_scaffold.get('optional_segments_requested', 0)),
            'final_puzzle_scaffold_optional_segments_applied': int(final_puzzle_scaffold.get('optional_segments_applied', 0)),
            'final_puzzle_scaffold_route_template_used': int(final_puzzle_scaffold.get('route_template_used', 0)),
            'final_puzzle_scaffold_noise_components_removed': int(final_puzzle_scaffold.get('noise_components_removed', 0)),
            'final_puzzle_scaffold_noise_tiles_removed': int(final_puzzle_scaffold.get('noise_tiles_removed', 0)),
            'final_puzzle_scaffold_novelty_score': float(final_puzzle_scaffold.get('novelty_score', 0.0)),
            'final_puzzle_scaffold_variant_name': str(final_puzzle_scaffold.get('variant_name', '') or ''),
            'final_puzzle_scaffold_variant_style': str(final_puzzle_scaffold.get('variant_style', '') or ''),
            'final_puzzle_scaffold_variant_side_bias': int(final_puzzle_scaffold.get('variant_side_bias', 0) or 0),
            'final_puzzle_scaffold_interaction_valid': int(final_puzzle_scaffold.get('interaction_valid', 0) or 0),
            'final_puzzle_scaffold_interaction_score': float(final_puzzle_scaffold.get('interaction_score', 0.0) or 0.0),
            'final_puzzle_scaffold_interaction_push_slot_count': int(final_puzzle_scaffold.get('interaction_push_slot_count', 0) or 0),
            'final_puzzle_scaffold_interaction_barrier_axis_tiles': int(final_puzzle_scaffold.get('interaction_barrier_axis_tiles', 0) or 0),
            'final_puzzle_scaffold_interaction_route_divergence': float(final_puzzle_scaffold.get('interaction_route_divergence', 0.0) or 0.0),
            'final_puzzle_scaffold_interaction_sequence_valid': int(final_puzzle_scaffold.get('interaction_sequence_valid', 0) or 0),
            'final_puzzle_scaffold_interaction_sequence_score': float(final_puzzle_scaffold.get('interaction_sequence_score', 0.0) or 0.0),
            'final_puzzle_scaffold_interaction_sequence_length': int(final_puzzle_scaffold.get('interaction_sequence_length', 0) or 0),
            'final_puzzle_scaffold_interaction_sequence_route_anchor_coverage': float(final_puzzle_scaffold.get('interaction_sequence_route_anchor_coverage', 0.0) or 0.0),
            'final_puzzle_scaffold_interaction_sequence_pairwise_path_ratio': float(final_puzzle_scaffold.get('interaction_sequence_pairwise_path_ratio', 0.0) or 0.0),
            'puzzle_plan_stage_count': int(len(list(room_puzzle_metadata.get('stage_sequence', []) or []))),
            'puzzle_plan_controlled_door_count': int(len(list(room_puzzle_metadata.get('controlled_doors_local', []) or []))),
            'neural_no_puzzle_structure_cleanup_applied': int(neural_no_puzzle_structure_cleanup.get('applied', 0)),
            'neural_no_puzzle_block_tiles_removed': int(neural_no_puzzle_structure_cleanup.get('block_tiles_removed', 0)),
            'neural_no_puzzle_block_components_removed': int(neural_no_puzzle_structure_cleanup.get('block_components_removed', 0)),
            'final_no_puzzle_structure_cleanup_applied': int(final_no_puzzle_structure_cleanup.get('applied', 0)),
            'final_no_puzzle_block_tiles_removed': int(final_no_puzzle_structure_cleanup.get('block_tiles_removed', 0)),
            'final_no_puzzle_block_components_removed': int(final_no_puzzle_structure_cleanup.get('block_components_removed', 0)),
            'neural_graph_markers_placed': int(neural_marker_count),
            'final_graph_markers_placed': int(final_marker_count),
            'final_graph_marker_overwrites': int(final_marker_overwrites),
            'final_graph_marker_overwrite_rate': float(final_marker_overwrite_rate),
            'semantic_constrained_decode_planned_markers': float(semantic_decode_stats.get('planned_markers', 0)),
            'semantic_constrained_decode_biased_slots': float(semantic_decode_stats.get('biased_slots', 0)),
            'neural_post_boundary_graph_semantic_hints_salvaged': float(neural_post_boundary_preserved_count),
            'final_post_boundary_graph_semantic_hints_salvaged': float(final_post_boundary_preserved_count),
            'vglc_compliant': valid_dims,
            'wfc_feedback_rounds': float(repair_diag.get('feedback_rounds', 0)),
            'wfc_failures': float(repair_diag.get('wfc_failures', 0)),
            'planned_traversability_pixels': float(np.sum(room_plan_mask)) if isinstance(room_plan_mask, np.ndarray) else 0.0,
            'used_fast_sampling': float(bool(use_fast_sampling)),
            'masked_room_sampling_temperature': float(self.default_masked_room_sampling_temperature),
            'masked_room_sampling_stochastic': float(
                bool(self.default_masked_room_sampling_stochastic)
            ),
            'masked_room_corrector_steps': float(self.default_masked_room_corrector_steps),
            'masked_room_corrector_mask_ratio': float(self.default_masked_room_corrector_mask_ratio),
        }
        metrics.update(neural_marker_alignment)
        metrics.update(final_pre_overlay_alignment)
        metrics.update(final_post_overlay_alignment)

        teacher_fallback_source: Optional[str] = None
        if (
            bool(allow_teacher_fallback)
            and effective_room_generator_mode == "latent_diffusion"
            and bool(use_fast_sampling)
            and self.diffusion.supports_fast_sampling()
            and self._should_retry_room_with_teacher(
                final_grid=final_grid,
                graph=mission_graph_for_room,
                room_id=room_id,
                metrics=metrics,
                source_mode="fast_sampler",
            )
        ):
            teacher_fallback_source = "fast_sampler"
        elif (
            bool(allow_teacher_fallback)
            and effective_room_generator_mode == "discrete_masked"
            and self.diffusion is not None
            and self._should_retry_room_with_teacher(
                final_grid=final_grid,
                graph=mission_graph_for_room,
                room_id=room_id,
                metrics=metrics,
                source_mode="masked_room",
            )
        ):
            teacher_fallback_source = "masked_room"

        if teacher_fallback_source is not None:
            self._bump_diagnostic(f"{teacher_fallback_source}_teacher_fallback")
            logger.debug(
                "Room %s triggered %s teacher fallback; rerunning with full diffusion teacher.",
                room_id,
                teacher_fallback_source.replace("_", "-"),
            )
            if teacher_fallback_source == "masked_room":
                # The masked generator and diffusion teacher can use different CUDA kernels and
                # work queues. Flush queued masked-room work before the recursive teacher rerun
                # so VQ-VAE decode always sees tensors on a consistent stream.
                self._synchronize_cuda_device()
            teacher_result = self.generate_room(
                neighbor_latents=neighbor_latents,
                graph_context=graph_context,
                room_id=room_id,
                boundary_constraints=boundary_constraints,
                position=position,
                reference_room_maps=reference_room_maps,
                guidance_scale=guidance_scale,
                logic_guidance_scale=logic_guidance_scale,
                num_diffusion_steps=max(int(self.default_num_diffusion_steps), int(num_diffusion_steps)),
                use_fast_sampling=False,
                latent_sampler=latent_sampler,
                categorical_codebook_size=categorical_codebook_size,
                use_ddim=use_ddim,
                apply_repair=apply_repair,
                start_goal_coords=start_goal_coords,
                seed=seed,
                precomputed_condition=condition.detach().clone(),
                allow_teacher_fallback=False,
                room_generator_override="latent_diffusion",
            )
            teacher_result.metrics["teacher_fallback_used"] = 1.0
            teacher_result.metrics[f"teacher_fallback_source_{teacher_fallback_source}"] = 1.0
            teacher_result.metrics["original_fallback_candidate_neural_grid_entropy"] = float(metrics["neural_grid_entropy"])
            teacher_result.metrics["original_fallback_candidate_tiles_changed"] = float(metrics["tiles_changed"])
            return teacher_result

        return RoomGenerationResult(
            room_id=room_id,
            room_grid=final_grid,
            latent=latent_cpu,
            neural_grid=neural_grid,
            was_repaired=was_repaired,
            repair_mask=repair_mask,
            room_plan_mask=room_plan_mask,
            neural_probs=neural_probs,
            puzzle_metadata=room_puzzle_metadata,
            metrics=metrics,
        )

    def repair_room(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        *,
        required_floor_mask: Optional[np.ndarray] = None,
        feedback_callback: Optional[Any] = None,
        max_feedback_rounds: int = 0,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Public symbolic-only room repair entry point.

        This is intentionally usable on pipelines created via
        `create_symbolic_repair_pipeline()` without loading the neural stack.
        """
        refiner = self._require_component("refiner", "repair_room")
        repaired_grid, success, diagnostics = refiner.repair_room_with_feedback(
            grid=np.asarray(grid, dtype=np.int32),
            start=self._normalize_room_coord(start, field_name="start"),
            goal=self._normalize_room_coord(goal, field_name="goal"),
            required_floor_mask=(
                np.asarray(required_floor_mask, dtype=bool)
                if isinstance(required_floor_mask, np.ndarray)
                else None
            ),
            feedback_callback=feedback_callback,
            max_feedback_rounds=max_feedback_rounds,
        )
        return repaired_grid, bool(success), diagnostics

    def prepare_dungeon_generation(
        self,
        mission_graph: Optional[nx.Graph] = None,
        *,
        use_topological_positional_encoding: bool = True,
        generate_topology: bool = False,
        target_curve: Optional[List[float]] = None,
        num_rooms: Optional[int] = None,
        population_size: Optional[int] = None,
        generations: Optional[int] = None,
        mutation_rate: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        genome_length: Optional[int] = None,
        rule_space: Optional[str] = None,
        transition_mix: Optional[float] = None,
        search_strategy: Optional[str] = None,
        qd_archive_cells: Optional[int] = None,
        qd_init_random_fraction: Optional[float] = None,
        qd_emitter_mutation_rate: Optional[float] = None,
        max_lock_key_rules: Optional[int] = None,
        enable_rule_credit_assignment: Optional[bool] = None,
        enforce_generation_constraints: Optional[bool] = None,
        allow_candidate_repairs: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> PreparedDungeonGeneration:
        """
        Prepare mission-graph inputs for room generation.

        This isolates topology synthesis, virtual-node filtering, and graph
        tensor preparation so callers can run those phases independently.
        """
        graph = mission_graph
        if graph is None:
            if not generate_topology:
                raise ValueError(
                    "mission_graph is None but generate_topology=False. "
                    "Either provide a mission_graph or set generate_topology=True"
                )

            logger.info("Block I: Generating dungeon topology via evolutionary search")
            resolved_target_curve = (
                [float(v) for v in target_curve]
                if target_curve is not None
                else list(self.topology_default_target_curve)
            )
            resolved_num_rooms = int(self.topology_num_rooms if num_rooms is None else max(1, int(num_rooms)))
            resolved_population_size = int(
                self.topology_population_size if population_size is None else max(1, int(population_size))
            )
            resolved_generations = int(
                self.topology_generations if generations is None else max(1, int(generations))
            )
            resolved_mutation_rate = float(
                self.topology_mutation_rate if mutation_rate is None else np.clip(float(mutation_rate), 0.0, 1.0)
            )
            resolved_crossover_rate = float(
                self.topology_crossover_rate if crossover_rate is None else np.clip(float(crossover_rate), 0.0, 1.0)
            )
            resolved_genome_length = self.topology_genome_length if genome_length is None else int(max(0, int(genome_length)))
            resolved_rule_space = (
                self.topology_rule_space if rule_space is None else str(rule_space).strip().lower()
            )
            resolved_transition_mix = float(
                self.topology_transition_mix if transition_mix is None else np.clip(float(transition_mix), 0.0, 1.0)
            )
            resolved_search_strategy = (
                self.topology_search_strategy if search_strategy is None else str(search_strategy).strip().lower()
            )
            resolved_qd_archive_cells = int(
                self.topology_qd_archive_cells if qd_archive_cells is None else max(32, int(qd_archive_cells))
            )
            resolved_qd_init_random_fraction = float(
                self.topology_qd_init_random_fraction
                if qd_init_random_fraction is None
                else np.clip(float(qd_init_random_fraction), 0.05, 0.95)
            )
            resolved_qd_emitter_mutation_rate = float(
                self.topology_qd_emitter_mutation_rate
                if qd_emitter_mutation_rate is None
                else np.clip(float(qd_emitter_mutation_rate), 0.01, 0.95)
            )
            resolved_max_lock_key_rules = int(
                self.topology_max_lock_key_rules
                if max_lock_key_rules is None
                else max(0, int(max_lock_key_rules))
            )
            resolved_enable_rule_credit_assignment = bool(
                self.topology_enable_rule_credit_assignment
                if enable_rule_credit_assignment is None
                else enable_rule_credit_assignment
            )
            resolved_enforce_generation_constraints = bool(
                self.topology_enforce_generation_constraints
                if enforce_generation_constraints is None
                else enforce_generation_constraints
            )
            resolved_allow_candidate_repairs = bool(
                self.topology_allow_candidate_repairs
                if allow_candidate_repairs is None
                else allow_candidate_repairs
            )
            target_genome_length = int(resolved_genome_length)
            if target_genome_length <= 0:
                target_genome_length = max(10, int(resolved_num_rooms * 0.7))
            topology_generator = EvolutionaryTopologyGenerator(
                target_curve=resolved_target_curve,
                population_size=resolved_population_size,
                generations=resolved_generations,
                mutation_rate=resolved_mutation_rate,
                crossover_rate=resolved_crossover_rate,
                genome_length=target_genome_length,
                max_nodes=resolved_num_rooms,
                rule_space=resolved_rule_space,
                transition_mix=resolved_transition_mix,
                search_strategy=resolved_search_strategy,
                qd_archive_cells=resolved_qd_archive_cells,
                qd_init_random_fraction=resolved_qd_init_random_fraction,
                qd_emitter_mutation_rate=resolved_qd_emitter_mutation_rate,
                max_lock_key_rules=resolved_max_lock_key_rules,
                enable_rule_credit_assignment=resolved_enable_rule_credit_assignment,
                enforce_generation_constraints=resolved_enforce_generation_constraints,
                allow_candidate_repairs=resolved_allow_candidate_repairs,
                seed=seed,
            )

            graph = topology_generator.evolve(directed_output=True)
            logger.info("Block I: Generated topology with %d rooms", graph.number_of_nodes())

            is_valid, errors = validate_graph_topology(graph)
            if not is_valid:
                if self.strict_checkpoint_mode:
                    raise ValueError(
                        f"Block I: Generated topology failed validation in strict mode: {errors}"
                    )
                logger.warning("Block I: Generated topology has validation errors: %s", errors)

        if graph is None:
            raise ValueError("mission_graph is still None after topology generation attempt")

        logger.debug("Applying VGLC compliance: filtering virtual nodes from mission graph")
        mission_graph_physical = filter_virtual_nodes(graph)
        physical_start = get_physical_start_node(graph)
        if physical_start is not None:
            logger.debug("Physical start node: %s", physical_start)

        logger.info(
            "Generating dungeon with %d rooms (filtered %d virtual nodes)",
            len(mission_graph_physical.nodes),
            len(graph.nodes) - len(mission_graph_physical.nodes),
        )

        graph_data = self._prepare_graph_context(
            mission_graph_physical,
            use_tpe=use_topological_positional_encoding,
        )
        return PreparedDungeonGeneration(
            mission_graph=graph,
            mission_graph_physical=mission_graph_physical,
            graph_data=graph_data,
        )

    def generate_rooms_for_graph(
        self,
        prepared: PreparedDungeonGeneration,
        *,
        guidance_scale: Optional[float] = None,
        logic_guidance_scale: Optional[float] = None,
        num_diffusion_steps: Optional[int] = None,
        use_fast_sampling: Optional[bool] = None,
        latent_sampler: Optional[str] = None,
        categorical_codebook_size: Optional[int] = None,
        apply_repair: Optional[bool] = None,
        seed: Optional[int] = None,
        batch_independent_rooms: bool = True,
        max_batch_size: int = 8,
    ) -> GeneratedRoomSet:
        """
        Generate all room grids/latents for a prepared mission graph.

        This allows room generation to be tested or reused without immediately
        stitching the final dungeon grid.
        """
        self._require_room_generation_components("generate_rooms_for_graph")
        guidance_scale = self.default_guidance_scale if guidance_scale is None else float(guidance_scale)
        logic_guidance_scale = (
            self.default_logic_guidance_scale
            if logic_guidance_scale is None
            else float(logic_guidance_scale)
        )
        num_diffusion_steps = (
            self.default_num_diffusion_steps if num_diffusion_steps is None else int(num_diffusion_steps)
        )
        use_fast_sampling = (
            self.default_use_fast_sampling if use_fast_sampling is None else bool(use_fast_sampling)
        )
        latent_sampler = self.default_latent_sampler if latent_sampler is None else str(latent_sampler)
        if categorical_codebook_size is None and self.default_categorical_codebook_size is not None:
            categorical_codebook_size = int(self.default_categorical_codebook_size)
        apply_repair = self.default_apply_repair if apply_repair is None else bool(apply_repair)
        self._puzzle_novelty_history = []
        self._puzzle_variant_cache = {}
        self._puzzle_novelty_committed = set()

        mission_graph_physical = prepared.mission_graph_physical
        graph_data = prepared.graph_data
        rooms: Dict[Any, RoomGenerationResult] = {}
        room_latents: Dict[Any, torch.Tensor] = {}
        batch_runtime_diagnostics: List[Dict[str, Any]] = []

        if batch_independent_rooms and mission_graph_physical.is_directed():
            layers = self._topological_generation_layers(mission_graph_physical)
            offset = 0
            for layer_idx, layer in enumerate(layers):
                if not layer:
                    continue
                buckets = self._bucket_room_ids_by_latent_shape(
                    room_ids=list(layer),
                    mission_graph_physical=mission_graph_physical,
                    room_latents=room_latents,
                )
                for bucket_key, bucket_room_ids in buckets.items():
                    latent_shape_chw = (int(bucket_key[0]), int(bucket_key[1]), int(bucket_key[2]))
                    target_h = int(bucket_key[3])
                    target_w = int(bucket_key[4])

                    if (target_h, target_w) != (ROOM_HEIGHT, ROOM_WIDTH):
                        logger.warning(
                            "Room-size bucket (%d,%d) is non-canonical; using sequential fallback for %d room(s).",
                            target_h,
                            target_w,
                            len(bucket_room_ids),
                        )
                        for room_id in bucket_room_ids:
                            idx = offset
                            neighbor_latents = self._get_neighbor_latents(
                                room_id, mission_graph_physical, room_latents
                            )
                            reference_room_maps = (
                                self._get_neighbor_reference_room_maps(room_id, mission_graph_physical, rooms)
                                if bool(getattr(self.condition_encoder, "use_reference_room_maps", False))
                                else None
                            )
                            start_goal = self._extract_room_start_goal(mission_graph_physical, room_id)
                            boundary_constraints = self._build_room_boundary_constraints(
                                graph=mission_graph_physical,
                                room_id=room_id,
                            )
                            room_position = self._build_room_position_tensor(
                                graph=mission_graph_physical,
                                room_id=room_id,
                                fallback_order_index=idx,
                            )
                            room_seed = None
                            if seed is not None:
                                room_seed = int(seed) + int(_stable_node_seed_offset(room_id))
                            room_graph_context = self._build_room_graph_context(
                                graph_data=graph_data,
                                mission_graph=mission_graph_physical,
                                room_id=room_id,
                                start_goal=start_goal,
                            )
                            room_result = self.generate_room(
                                neighbor_latents=neighbor_latents,
                                graph_context=room_graph_context,
                                room_id=room_id,
                                boundary_constraints=boundary_constraints,
                                position=room_position,
                                reference_room_maps=reference_room_maps,
                                guidance_scale=guidance_scale,
                                logic_guidance_scale=logic_guidance_scale,
                                num_diffusion_steps=num_diffusion_steps,
                                use_fast_sampling=use_fast_sampling,
                                latent_sampler=latent_sampler,
                                categorical_codebook_size=categorical_codebook_size,
                                apply_repair=apply_repair,
                                start_goal_coords=start_goal,
                                seed=room_seed,
                            )
                            rooms[room_id] = room_result
                            room_latents[room_id] = self._encode_room_grid_to_latent(room_result.room_grid)
                            offset += 1
                        continue

                    requested = max(1, int(max_batch_size))
                    safe_chunk = self._estimate_safe_batch_size(
                        requested_batch_size=requested,
                        latent_shape_chw=latent_shape_chw,
                    )
                    cuda_free_mb = None
                    if torch.cuda.is_available():
                        try:
                            free_bytes, _total_bytes = torch.cuda.mem_get_info(device=self.device)
                            cuda_free_mb = float(free_bytes) / (1024.0 * 1024.0)
                        except Exception:
                            cuda_free_mb = None

                    logger.info(
                        "Batch planner: layer=%d bucket(latent=%s,target=%dx%d) rooms=%d requested=%d safe_chunk=%d cuda_free_mb=%s",
                        int(layer_idx),
                        str(latent_shape_chw),
                        int(target_h),
                        int(target_w),
                        int(len(bucket_room_ids)),
                        int(requested),
                        int(safe_chunk),
                        (f"{cuda_free_mb:.1f}" if isinstance(cuda_free_mb, float) else "n/a"),
                    )

                    batch_runtime_diagnostics.append(
                        {
                            'layer_index': int(layer_idx),
                            'latent_shape_chw': [int(latent_shape_chw[0]), int(latent_shape_chw[1]), int(latent_shape_chw[2])],
                            'target_room_size_hw': [int(target_h), int(target_w)],
                            'bucket_room_count': int(len(bucket_room_ids)),
                            'requested_batch_size': int(requested),
                            'safe_chunk_size': int(safe_chunk),
                            'cuda_free_mb': float(cuda_free_mb) if isinstance(cuda_free_mb, float) else float('nan'),
                        }
                    )

                    for k in range(0, len(bucket_room_ids), safe_chunk):
                        batch_room_ids = list(bucket_room_ids[k:k + safe_chunk])
                        logger.debug(
                            "Batch execute: layer=%d bucket=%s chunk_start=%d chunk_size=%d",
                            int(layer_idx),
                            str(bucket_key),
                            int(k),
                            int(len(batch_room_ids)),
                        )
                        batch_runtime_diagnostics.append(
                            {
                                'layer_index': int(layer_idx),
                                'bucket_key': str(bucket_key),
                                'chunk_start': int(k),
                                'actual_chunk_size': int(len(batch_room_ids)),
                            }
                        )
                        try:
                            batch_results = self._generate_room_batch(
                                room_ids=batch_room_ids,
                                mission_graph_physical=mission_graph_physical,
                                graph_data=graph_data,
                                generated_rooms=rooms,
                                room_latents=room_latents,
                                guidance_scale=guidance_scale,
                                logic_guidance_scale=logic_guidance_scale,
                                num_diffusion_steps=num_diffusion_steps,
                                use_fast_sampling=use_fast_sampling,
                                latent_sampler=latent_sampler,
                                categorical_codebook_size=categorical_codebook_size,
                                apply_repair=apply_repair,
                                seed=seed,
                                layer_offset=offset,
                                latent_shape_chw=latent_shape_chw,
                            )
                        except (RuntimeError, ValueError) as exc:
                            self._bump_diagnostic("batched_room_generation_fallback")
                            logger.warning(
                                "Batched room generation failed for chunk %s at layer %d; falling back to sequential generation. Error: %s",
                                batch_room_ids,
                                int(layer_idx),
                                exc,
                            )
                            if torch.cuda.is_available():
                                self._synchronize_cuda_device()
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    logger.debug("torch.cuda.empty_cache() skipped after batch failure.", exc_info=True)
                            batch_runtime_diagnostics.append(
                                {
                                    'layer_index': int(layer_idx),
                                    'bucket_key': str(bucket_key),
                                    'chunk_start': int(k),
                                    'actual_chunk_size': int(len(batch_room_ids)),
                                    'sequential_fallback_after_batch_error': 1,
                                    'batch_error': str(exc),
                                }
                            )
                            batch_results = {}
                            for seq_offset, room_id in enumerate(batch_room_ids):
                                idx = offset + seq_offset
                                neighbor_latents = self._get_neighbor_latents(
                                    room_id, mission_graph_physical, room_latents
                                )
                                reference_room_maps = (
                                    self._get_neighbor_reference_room_maps(room_id, mission_graph_physical, rooms)
                                    if bool(getattr(self.condition_encoder, "use_reference_room_maps", False))
                                    else None
                                )
                                start_goal = self._extract_room_start_goal(mission_graph_physical, room_id)
                                boundary_constraints = self._build_room_boundary_constraints(
                                    graph=mission_graph_physical,
                                    room_id=room_id,
                                )
                                room_position = self._build_room_position_tensor(
                                    graph=mission_graph_physical,
                                    room_id=room_id,
                                    fallback_order_index=idx,
                                )
                                room_seed = None
                                if seed is not None:
                                    room_seed = int(seed) + int(_stable_node_seed_offset(room_id))
                                room_graph_context = self._build_room_graph_context(
                                    graph_data=graph_data,
                                    mission_graph=mission_graph_physical,
                                    room_id=room_id,
                                    start_goal=start_goal,
                                )
                                batch_results[room_id] = self.generate_room(
                                    neighbor_latents=neighbor_latents,
                                    graph_context=room_graph_context,
                                    room_id=room_id,
                                    boundary_constraints=boundary_constraints,
                                    position=room_position,
                                    reference_room_maps=reference_room_maps,
                                    guidance_scale=guidance_scale,
                                    logic_guidance_scale=logic_guidance_scale,
                                    num_diffusion_steps=num_diffusion_steps,
                                    use_fast_sampling=use_fast_sampling,
                                    latent_sampler=latent_sampler,
                                    categorical_codebook_size=categorical_codebook_size,
                                    apply_repair=apply_repair,
                                    start_goal_coords=start_goal,
                                    seed=room_seed,
                                )
                        for room_id in batch_room_ids:
                            room_result = batch_results[room_id]
                            rooms[room_id] = room_result
                            room_latents[room_id] = self._encode_room_grid_to_latent(room_result.room_grid)
                        offset += len(batch_room_ids)
        else:
            generation_order = sorted(
                mission_graph_physical.nodes(),
                key=_stable_node_sort_key,
            )
            for idx, room_id in enumerate(generation_order):
                logger.debug("Generating room %s (%d/%d)", room_id, idx + 1, len(generation_order))
                neighbor_latents = self._get_neighbor_latents(
                    room_id, mission_graph_physical, room_latents
                )
                reference_room_maps = (
                    self._get_neighbor_reference_room_maps(room_id, mission_graph_physical, rooms)
                    if bool(getattr(self.condition_encoder, "use_reference_room_maps", False))
                    else None
                )
                start_goal = self._extract_room_start_goal(mission_graph_physical, room_id)
                boundary_constraints = self._build_room_boundary_constraints(
                    graph=mission_graph_physical,
                    room_id=room_id,
                )
                room_position = self._build_room_position_tensor(
                    graph=mission_graph_physical,
                    room_id=room_id,
                    fallback_order_index=idx,
                )
                room_seed = None
                if seed is not None:
                    room_seed = int(seed) + int(_stable_node_seed_offset(room_id))
                room_graph_context = self._build_room_graph_context(
                    graph_data=graph_data,
                    mission_graph=mission_graph_physical,
                    room_id=room_id,
                    start_goal=start_goal,
                )

                room_result = self.generate_room(
                    neighbor_latents=neighbor_latents,
                    graph_context=room_graph_context,
                    room_id=room_id,
                    boundary_constraints=boundary_constraints,
                    position=room_position,
                    reference_room_maps=reference_room_maps,
                    guidance_scale=guidance_scale,
                    logic_guidance_scale=logic_guidance_scale,
                    num_diffusion_steps=num_diffusion_steps,
                    use_fast_sampling=use_fast_sampling,
                    latent_sampler=latent_sampler,
                    categorical_codebook_size=categorical_codebook_size,
                    apply_repair=apply_repair,
                    start_goal_coords=start_goal,
                    seed=room_seed,
                )

                rooms[room_id] = room_result
                room_latents[room_id] = self._encode_room_grid_to_latent(room_result.room_grid)

        return GeneratedRoomSet(
            rooms=rooms,
            room_latents=room_latents,
            batch_runtime_diagnostics=batch_runtime_diagnostics,
        )

    def evaluate_generated_dungeon(
        self,
        dungeon_grid: np.ndarray,
        mission_graph_physical: nx.Graph,
        *,
        enable_map_elites: bool = True,
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate a stitched dungeon grid with MAP-Elites when available.

        Returns `None` when evaluation is disabled or not applicable.
        """
        if not enable_map_elites or self.map_elites is None:
            return None

        map_elites_score = None
        try:
            solver_result = self._validate_dungeon(dungeon_grid)
            if solver_result and solver_result.get('solvable'):
                self.map_elites.add_dungeon(
                    dungeon=dungeon_grid,
                    grid=dungeon_grid,
                    solver_result=solver_result,
                    mission_graph=mission_graph_physical,
                )
                map_elites_score = {
                    'linearity': solver_result.get('linearity', 0.0),
                    'leniency': solver_result.get('leniency', 0.0),
                    'progression_complexity': solver_result.get('progression_complexity', 0.0),
                    'topology_complexity': solver_result.get('topology_complexity', 0.0),
                    'path_length': solver_result.get('path_length', 0),
                }
                if hasattr(self.map_elites, 'advanced_archive_stats'):
                    advanced_stats = self.map_elites.advanced_archive_stats()
                    if advanced_stats is not None:
                        map_elites_score['advanced_archive'] = advanced_stats
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"MAP-Elites evaluation failed: {e}")
        return map_elites_score
    
    @torch.no_grad()
    def evaluate_dungeon_solvability(
        self,
        rooms: Dict[Any, 'RoomGenerationResult'],
        mission_graph_physical: nx.Graph,
    ) -> Dict[str, Any]:
        """
        Evaluate dungeon-level solvability using LogicNet at the correct scope.
        
        This is the graph-level evaluation that was previously (incorrectly) embedded
        in the per-room training loss. A single room's z [B, 64, 4, 3] cannot encode
        inter-room key-lock ordering; only a dungeon-level assessment with the actual
        generated room connectivity can meaningfully evaluate graph solvability.
        
        Args:
            rooms: Dict of room_id -> RoomGenerationResult from generate_rooms_for_graph
            mission_graph_physical: The physical mission graph with edge types
            
        Returns:
            Dict with solvability_score, graph_reach_loss, lock_loss, and
            failing_rooms (list of room IDs where local solvability failed)
        """
        result: Dict[str, Any] = {
            'solvability_score': 0.0,
            'room_solvability_score': 0.0,
            'graph_reach_loss': 0.0,
            'lock_loss': 0.0,
            'global_logic_loss': 0.0,
        }
        failing_rooms: List[Any] = []
        
        if self.logic_net is None:
            logger.debug("evaluate_dungeon_solvability skipped: no logic_net component")
            return result

        graph_context = self._prepare_graph_context(mission_graph_physical, use_tpe=True)
        node_to_idx = dict(graph_context.get('node_to_idx', {}) or {})

        def _first_matching_node(*keys: str) -> Optional[Any]:
            for node_id, attrs in mission_graph_physical.nodes(data=True):
                attrs_dict = dict(attrs)
                role_flags = self._room_role_flags(attrs_dict)
                if any(coerce_bool(attrs_dict.get(key)) for key in keys):
                    return node_id
                if {"is_start", "is_entry"} & set(keys) and role_flags.get("is_start", False):
                    return node_id
                if {"has_triforce", "is_triforce", "is_goal"} & set(keys) and role_flags.get("has_goal", False):
                    return node_id
            return None

        start_node = get_physical_start_node(mission_graph_physical)
        if start_node is None or start_node not in node_to_idx:
            start_node = _first_matching_node("is_start", "is_entry")
        if start_node is None or start_node not in node_to_idx:
            start_node = next(iter(node_to_idx.keys()), None)
        start_idx = int(node_to_idx.get(start_node, 0)) if start_node is not None else 0

        target_node = _first_matching_node("has_triforce", "is_triforce", "is_goal")
        target_idx = int(node_to_idx.get(target_node, -1)) if target_node is not None else -1

        dungeon_latents: List[torch.Tensor] = []
        current_node_indices: List[int] = []
        evaluated_room_ids: List[Any] = []
        reference_shape: Optional[Tuple[int, ...]] = None

        # Evaluate per-room walkability via LogicNet (grid-level only)
        total_grid_reach = 0.0
        num_rooms = 0
        for room_id, room_result in rooms.items():
            if room_result.latent is None:
                continue
            z = room_result.latent
            if not isinstance(z, torch.Tensor) or z.numel() == 0:
                continue
            z = z.to(self.device)
            if z.dim() == 3:
                z = z.unsqueeze(0)
            if z.dim() != 4:
                logger.debug("Room %s solvability eval skipped: latent shape %s is not rank-4", room_id, tuple(z.shape))
                continue
            if reference_shape is None:
                reference_shape = tuple(z.shape[1:])
            elif tuple(z.shape[1:]) != reference_shape:
                logger.debug(
                    "Room %s solvability eval skipped: latent shape %s does not match %s",
                    room_id,
                    tuple(z.shape[1:]),
                    reference_shape,
                )
                continue

            node_idx = node_to_idx.get(room_id)
            if node_idx is not None:
                dungeon_latents.append(z[:1])
                current_node_indices.append(int(node_idx))
                evaluated_room_ids.append(room_id)

            try:
                loss, info = self.logic_net(z, graph_data=None)
                grid_reach = float(info.get('grid_reachability', 0.0))
                total_grid_reach += grid_reach
                num_rooms += 1
                if grid_reach < 0.5:
                    failing_rooms.append(room_id)
            except (RuntimeError, ValueError) as e:
                logger.debug("Room %s solvability eval failed: %s", room_id, e)
                failing_rooms.append(room_id)
        
        if num_rooms > 0:
            result['room_solvability_score'] = total_grid_reach / num_rooms

        if dungeon_latents:
            z_dungeon = torch.cat(dungeon_latents, dim=0)
            graph_data = {
                'graph_scope': 'dungeon',
                'node_features': graph_context.get('node_features'),
                'edge_index': graph_context.get('edge_index'),
                'edge_features': graph_context.get('edge_features'),
                'tpe': graph_context.get('tpe'),
                'node_positions': graph_context.get('node_positions'),
                'node_mask': graph_context.get('node_mask'),
                'current_node_idx': torch.tensor(current_node_indices, device=self.device, dtype=torch.long),
                'start_node_id': int(start_idx),
                'target_idx': int(target_idx),
            }
            try:
                global_loss, global_info = self.logic_net(z_dungeon, graph_data=graph_data)
                global_logic_loss = global_info.get('global_logic_loss', global_loss)
                if isinstance(global_logic_loss, torch.Tensor):
                    global_logic_loss_value = float(global_logic_loss.detach().mean().item())
                else:
                    global_logic_loss_value = float(global_logic_loss)

                reachability_value = global_info.get('global_graph_reachability')
                if isinstance(reachability_value, torch.Tensor):
                    graph_score = float(reachability_value.detach().mean().item())
                elif reachability_value is not None:
                    graph_score = float(reachability_value)
                else:
                    graph_score = float(torch.exp(torch.tensor(-max(0.0, global_logic_loss_value))).item())

                result.update({
                    'solvability_score': max(0.0, min(1.0, graph_score)),
                    'graph_reach_loss': float(global_info.get('graph_reach_loss', global_info.get('global_graph_reach_loss', 0.0))) if not isinstance(global_info.get('graph_reach_loss'), torch.Tensor) else float(global_info['graph_reach_loss'].detach().mean().item()),
                    'lock_loss': float(global_info.get('lock_loss', 0.0)) if not isinstance(global_info.get('lock_loss'), torch.Tensor) else float(global_info['lock_loss'].detach().mean().item()),
                    'global_logic_loss': global_logic_loss_value,
                    'global_loss_score': float(torch.exp(torch.tensor(-max(0.0, global_logic_loss_value))).item()),
                    'global_room_passability': float(global_info.get('global_room_passability', 0.0)) if not isinstance(global_info.get('global_room_passability'), torch.Tensor) else float(global_info['global_room_passability'].detach().mean().item()),
                    'global_num_rooms_scored': float(len(evaluated_room_ids)),
                })
            except (RuntimeError, ValueError) as e:
                logger.debug("Dungeon-scope LogicNet solvability eval failed: %s", e)
                result['solvability_score'] = result['room_solvability_score']
        else:
            result['solvability_score'] = result['room_solvability_score']
        
        result['failing_rooms'] = failing_rooms  # type: ignore[assignment]
        result['num_rooms_evaluated'] = float(num_rooms)
        result['num_failing'] = float(len(failing_rooms))
        
        logger.info(
            "Dungeon solvability: %.3f global, %.3f room-local (%d/%d rooms passing)",
            result['solvability_score'],
            result['room_solvability_score'],
            num_rooms - len(failing_rooms),
            num_rooms,
        )
        return result
    
    @torch.no_grad()
    def generate_dungeon(
        self,
        mission_graph: Optional[nx.Graph] = None,
        guidance_scale: Optional[float] = None,
        logic_guidance_scale: Optional[float] = None,
        num_diffusion_steps: Optional[int] = None,
        use_fast_sampling: Optional[bool] = None,
        latent_sampler: Optional[str] = None,
        categorical_codebook_size: Optional[int] = None,
        use_topological_positional_encoding: Optional[bool] = None,
        apply_repair: Optional[bool] = None,
        seed: Optional[int] = None,
        enable_map_elites: Optional[bool] = None,
        # Block I: Evolutionary generation parameters
        generate_topology: bool = False,
        target_curve: Optional[List[float]] = None,
        num_rooms: Optional[int] = None,
        population_size: Optional[int] = None,
        generations: Optional[int] = None,
        mutation_rate: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        genome_length: Optional[int] = None,
        rule_space: Optional[str] = None,
        transition_mix: Optional[float] = None,
        search_strategy: Optional[str] = None,
        qd_archive_cells: Optional[int] = None,
        qd_init_random_fraction: Optional[float] = None,
        qd_emitter_mutation_rate: Optional[float] = None,
        max_lock_key_rules: Optional[int] = None,
        enable_rule_credit_assignment: Optional[bool] = None,
        enforce_generation_constraints: Optional[bool] = None,
        allow_candidate_repairs: Optional[bool] = None,
        batch_independent_rooms: bool = True,
        max_batch_size: int = 8,
    ) -> DungeonGenerationResult:
        """
        Generate a complete multi-room dungeon using graph-guided generation.
        
        This integrates all 7 blocks of the H-MOLQD pipeline:
        - Block I: Evolutionary Topology Director (optional, if generate_topology=True)
        - Block II: VQ-VAE latent encoding/decoding
        - Block III: Dual-stream condition encoding
        - Block IV: Latent diffusion with guidance
        - Block V: LogicNet differentiable solvability
        - Block VI: Symbolic WaveFunctionCollapse repair
        - Block VII: MAP-Elites quality-diversity evaluation
        
        VGLC Compliance: Filters virtual nodes before generation.
        
        Args:
            mission_graph: NetworkX graph with room nodes and door edges
                          If None and generate_topology=True, will generate automatically
            guidance_scale: Classifier-free guidance scale
            logic_guidance_scale: LogicNet gradient guidance scale
            num_diffusion_steps: Number of diffusion steps per room
            latent_sampler: "diffusion" (default) or "categorical"
            categorical_codebook_size: Optional cap for categorical codebook sampling
            use_topological_positional_encoding: Include TPE features in graph context
            apply_repair: Apply symbolic repair to each room
            seed: Random seed for reproducibility
            enable_map_elites: Compute MAP-Elites metrics
            generate_topology: Use Block I to evolve mission graph (if mission_graph=None)
            target_curve: Difficulty curve for evolutionary search [0.0-1.0]
            num_rooms: Number of rooms for generated topology
            population_size: Evolution population size
            generations: Number of evolutionary generations
            mutation_rate: Per-gene mutation probability for Block I
            crossover_rate: Crossover probability for Block I
            genome_length: Optional fixed genome length (0/None => auto)
            rule_space: Grammar rule-space (`core` or `full`)
            transition_mix: Mix between transition-biased and global rule priors
            search_strategy: Search backend (`ga` or `cvt_emitter` aliases)
            qd_archive_cells: CVT archive cells when using QD search
            qd_init_random_fraction: Bootstrap random fraction for QD search
            qd_emitter_mutation_rate: Emitter mutation rate for QD search
            max_lock_key_rules: Soft cap on InsertLockKey use per genome
            enable_rule_credit_assignment: Enable adaptive rule-credit assignment
            enforce_generation_constraints: Reject invalid intermediate candidates
            allow_candidate_repairs: Attempt local candidate repairs when constraints fail
            
        Returns:
            DungeonGenerationResult with complete dungeon and metrics
        """
        self._require_room_generation_components("generate_dungeon")
        import time
        start_time = time.time()
        guidance_scale = self.default_guidance_scale if guidance_scale is None else float(guidance_scale)
        logic_guidance_scale = (
            self.default_logic_guidance_scale
            if logic_guidance_scale is None
            else float(logic_guidance_scale)
        )
        num_diffusion_steps = (
            self.default_num_diffusion_steps if num_diffusion_steps is None else int(num_diffusion_steps)
        )
        use_fast_sampling = (
            self.default_use_fast_sampling if use_fast_sampling is None else bool(use_fast_sampling)
        )
        latent_sampler = self.default_latent_sampler if latent_sampler is None else str(latent_sampler)
        if categorical_codebook_size is None and self.default_categorical_codebook_size is not None:
            categorical_codebook_size = int(self.default_categorical_codebook_size)
        use_topological_positional_encoding = (
            self.default_use_topological_positional_encoding
            if use_topological_positional_encoding is None
            else bool(use_topological_positional_encoding)
        )
        apply_repair = self.default_apply_repair if apply_repair is None else bool(apply_repair)
        enable_map_elites = (
            self.default_enable_map_elites if enable_map_elites is None else bool(enable_map_elites)
        )

        if seed is not None:
            torch.manual_seed(seed)

        if apply_repair and self.refiner is None:
            self._bump_diagnostic("repair_disabled_missing_component")
            logger.warning(
                "Dungeon generation requested symbolic repair, but no refiner component is configured; disabling repair."
            )
            apply_repair = False
        if enable_map_elites and self.map_elites is None:
            self._bump_diagnostic("map_elites_disabled_missing_component")
            logger.warning(
                "Dungeon generation requested MAP-Elites evaluation, but no map_elites component is configured; disabling evaluation."
            )
            enable_map_elites = False
        
        prepared = self.prepare_dungeon_generation(
            mission_graph=mission_graph,
            use_topological_positional_encoding=use_topological_positional_encoding,
            generate_topology=generate_topology,
            target_curve=target_curve,
            num_rooms=num_rooms,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            genome_length=genome_length,
            rule_space=rule_space,
            transition_mix=transition_mix,
            search_strategy=search_strategy,
            qd_archive_cells=qd_archive_cells,
            qd_init_random_fraction=qd_init_random_fraction,
            qd_emitter_mutation_rate=qd_emitter_mutation_rate,
            max_lock_key_rules=max_lock_key_rules,
            enable_rule_credit_assignment=enable_rule_credit_assignment,
            enforce_generation_constraints=enforce_generation_constraints,
            allow_candidate_repairs=allow_candidate_repairs,
            seed=seed,
        )

        room_set = self.generate_rooms_for_graph(
            prepared,
            guidance_scale=guidance_scale,
            logic_guidance_scale=logic_guidance_scale,
            num_diffusion_steps=num_diffusion_steps,
            use_fast_sampling=use_fast_sampling,
            latent_sampler=latent_sampler,
            categorical_codebook_size=categorical_codebook_size,
            apply_repair=apply_repair,
            seed=seed,
            batch_independent_rooms=batch_independent_rooms,
            max_batch_size=max_batch_size,
        )

        stitched_layout = self.stitch_room_layout(room_set.rooms, prepared.mission_graph_physical)
        dungeon_grid = np.asarray(stitched_layout.dungeon_grid, dtype=np.int32)
        puzzle_metadata = self._globalize_room_puzzle_metadata(
            rooms=room_set.rooms,
            stitched_layout=stitched_layout,
        )
        map_elites_score = self.evaluate_generated_dungeon(
            dungeon_grid,
            prepared.mission_graph_physical,
            enable_map_elites=enable_map_elites,
        )
        try:
            logic_solvability = self.evaluate_dungeon_solvability(
                room_set.rooms,
                prepared.mission_graph_physical,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("LogicNet dungeon solvability metrics failed: %s", exc)
            logic_solvability = {}
        
        # Compute overall metrics
        generation_time = time.time() - start_time
        num_rooms_generated = len(room_set.rooms)
        room_metric_dicts = [dict(r.metrics) for r in room_set.rooms.values()]
        alignment_metrics = self._aggregate_room_alignment_metrics(room_metric_dicts)
        metrics = {
            'num_rooms': num_rooms_generated,
            'total_tiles_repaired': sum(r.metrics.get('tiles_changed', 0) for r in room_set.rooms.values()),
            'repair_rate': (
                sum(r.was_repaired for r in room_set.rooms.values()) / max(1, num_rooms_generated)
                if num_rooms_generated > 0
                else 0.0
            ),
            'dungeon_shape': dungeon_grid.shape,
            'generation_time_sec': generation_time,
            'batch_generation_diagnostics': room_set.batch_runtime_diagnostics,
            'puzzle_plan_count': int(len(dict(puzzle_metadata.get('plans', {}) or {}))),
            'puzzle_stage_count': int(
                sum(
                    len(list(dict(plan or {}).get('stage_sequence', []) or []))
                    for plan in dict(puzzle_metadata.get('plans', {}) or {}).values()
                )
            ),
            'logicnet_dungeon_solvability': float(logic_solvability.get('solvability_score', 0.0)),
            'logicnet_room_solvability': float(logic_solvability.get('room_solvability_score', 0.0)),
            'logicnet_graph_reach_loss': float(logic_solvability.get('graph_reach_loss', 0.0)),
            'logicnet_lock_loss': float(logic_solvability.get('lock_loss', 0.0)),
            'logicnet_global_logic_loss': float(logic_solvability.get('global_logic_loss', 0.0)),
            'logicnet_num_failing_rooms': float(logic_solvability.get('num_failing', 0.0)),
            **alignment_metrics,
        }
        
        logger.info(f"Dungeon generated in {generation_time:.2f}s "
                   f"(repair_rate={metrics['repair_rate']:.1%})")
        
        return DungeonGenerationResult(
            dungeon_grid=dungeon_grid,
            rooms=room_set.rooms,
            mission_graph=prepared.mission_graph,  # Preserve caller graph identity
            metrics=metrics,
            map_elites_score=map_elites_score,
            generation_time=generation_time,
            stitched_layout=stitched_layout,
            puzzle_metadata=puzzle_metadata,
        )

    @torch.no_grad()
    def repair_and_stitch_dungeon(
        self,
        *,
        rooms: Dict[Any, Any],
        mission_graph: nx.Graph,
        apply_repair: bool = True,
        enable_map_elites: bool = False,
    ) -> DungeonGenerationResult:
        """
        Public symbolic-only dungeon assembly entry point.

        This path accepts pre-existing room grids or `RoomGenerationResult`
        objects, optionally repairs each room, then stitches and evaluates the
        final dungeon without requiring the neural generation stack.
        """
        if mission_graph is None:
            raise ValueError("repair_and_stitch_dungeon requires a mission_graph.")
        if not rooms:
            raise ValueError("repair_and_stitch_dungeon requires at least one room.")

        if apply_repair and self.refiner is None:
            raise MissingPipelineComponentError(
                "repair_and_stitch_dungeon requires a refiner component when apply_repair=True."
            )

        normalized_rooms: Dict[Any, RoomGenerationResult] = {}
        for room_id, room_value in rooms.items():
            if isinstance(room_value, RoomGenerationResult):
                room_grid = fit_room_grid(room_value.room_grid)
                neural_grid = fit_room_grid(room_value.neural_grid)
                latent = room_value.latent.detach().cpu() if isinstance(room_value.latent, torch.Tensor) else torch.empty(0)
                puzzle_metadata = dict(room_value.puzzle_metadata or {})
            else:
                room_grid = fit_room_grid(room_value)
                neural_grid = room_grid.copy()
                latent = torch.empty(0)
                puzzle_metadata = {}

            room_plan_mask = None
            repaired_grid = room_grid.copy()
            was_repaired = False
            repair_diagnostics: Dict[str, Any] = {}

            if apply_repair:
                start_goal = self._extract_room_start_goal(mission_graph, room_id)
                room_plan_mask = self._build_room_plan_trace(
                    mission_graph,
                    room_id,
                    repaired_grid,
                    start_goal=start_goal,
                )
                repaired_grid, was_repaired, repair_diagnostics = self.repair_room(
                    repaired_grid,
                    start=start_goal[0],
                    goal=start_goal[1],
                    required_floor_mask=room_plan_mask,
                )

            normalized_rooms[room_id] = RoomGenerationResult(
                room_id=room_id,
                room_grid=np.asarray(repaired_grid, dtype=np.int32),
                latent=latent,
                neural_grid=np.asarray(neural_grid, dtype=np.int32),
                was_repaired=bool(was_repaired),
                repair_mask=None,
                room_plan_mask=room_plan_mask,
                neural_probs=None,
                puzzle_metadata=puzzle_metadata,
                metrics={
                    "room_id": room_id,
                    "was_repaired": bool(was_repaired),
                    "planned_traversability_pixels": float(np.sum(room_plan_mask)) if isinstance(room_plan_mask, np.ndarray) else 0.0,
                    **repair_diagnostics,
                },
            )

        stitched_layout = self.stitch_room_layout(normalized_rooms, mission_graph)
        dungeon_grid = np.asarray(stitched_layout.dungeon_grid, dtype=np.int32)
        puzzle_metadata = self._globalize_room_puzzle_metadata(
            rooms=normalized_rooms,
            stitched_layout=stitched_layout,
        )
        map_elites_score = self.evaluate_generated_dungeon(
            dungeon_grid,
            mission_graph,
            enable_map_elites=enable_map_elites,
        )
        try:
            logic_solvability = self.evaluate_dungeon_solvability(normalized_rooms, mission_graph)
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("LogicNet symbolic dungeon solvability metrics failed: %s", exc)
            logic_solvability = {}
        metrics = {
            "num_rooms": len(normalized_rooms),
            "total_tiles_repaired": sum(r.metrics.get("tiles_changed", 0) for r in normalized_rooms.values()),
            "repair_rate": (
                sum(bool(r.was_repaired) for r in normalized_rooms.values()) / max(1, len(normalized_rooms))
            ),
            "dungeon_shape": dungeon_grid.shape,
            "symbolic_only": True,
            "puzzle_plan_count": int(len(dict(puzzle_metadata.get("plans", {}) or {}))),
            "logicnet_dungeon_solvability": float(logic_solvability.get("solvability_score", 0.0)),
            "logicnet_room_solvability": float(logic_solvability.get("room_solvability_score", 0.0)),
            "logicnet_graph_reach_loss": float(logic_solvability.get("graph_reach_loss", 0.0)),
            "logicnet_lock_loss": float(logic_solvability.get("lock_loss", 0.0)),
            "logicnet_global_logic_loss": float(logic_solvability.get("global_logic_loss", 0.0)),
            "logicnet_num_failing_rooms": float(logic_solvability.get("num_failing", 0.0)),
        }
        return DungeonGenerationResult(
            dungeon_grid=dungeon_grid,
            rooms=normalized_rooms,
            mission_graph=mission_graph,
            metrics=metrics,
            map_elites_score=map_elites_score,
            generation_time=0.0,
            stitched_layout=stitched_layout,
            puzzle_metadata=puzzle_metadata,
        )
    
    def _prepare_graph_context(self, graph: nx.Graph, use_tpe: bool = True) -> Dict[str, Any]:
        """
        Prepare graph tensors for GNN conditioning with stable node indexing.

        Returns:
            Dict containing node_features, edge_index, edge_features, tpe,
            node_order, and node_to_idx.
        """
        node_dim, edge_dim = self._condition_feature_dims()

        if graph is None or len(graph.nodes) == 0:
            empty_nodes = torch.zeros(0, node_dim, device=self.device, dtype=torch.float32)
            empty_edges = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            empty_edge_feats = torch.zeros(0, edge_dim, device=self.device, dtype=torch.float32)
            empty_tpe = torch.zeros(0, 8, device=self.device, dtype=torch.float32)
            empty_positions = torch.zeros(0, 2, device=self.device, dtype=torch.float32)
            empty_mask = torch.zeros(0, device=self.device, dtype=torch.float32)
            return {
                'node_features': empty_nodes,
                'edge_index': empty_edges,
                'edge_features': empty_edge_feats,
                'tpe': empty_tpe,
                'node_positions': empty_positions,
                'node_mask': empty_mask,
                'node_order': [],
                'node_to_idx': {},
                'start_node_id': -1,
                'target_idx': -1,
            }

        # Deterministic order is required so room_id -> node_idx stays stable.
        if isinstance(graph, nx.DiGraph):
            try:
                node_order = list(nx.topological_sort(graph))
            except nx.NetworkXUnfeasible:
                node_order = sorted(graph.nodes(), key=_stable_node_sort_key)
        else:
            node_order = sorted(graph.nodes(), key=_stable_node_sort_key)

        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_order)}
        num_nodes = len(node_order)

        node_features = torch.zeros(num_nodes, node_dim, device=self.device, dtype=torch.float32)
        node_positions = torch.zeros(num_nodes, 2, device=self.device, dtype=torch.float32)
        for node_id, idx in node_to_idx.items():
            node_features[idx] = self._extract_node_feature_vector(graph.nodes[node_id])
            pos = self._get_node_grid_position(graph, node_id)
            if pos is None:
                node_positions[idx] = torch.tensor((float(idx), 0.0), device=self.device, dtype=torch.float32)
            else:
                node_positions[idx] = torch.tensor(
                    (float(pos[0]), float(pos[1])),
                    device=self.device,
                    dtype=torch.float32,
                )

        edge_pairs: List[Tuple[int, int]] = []
        edge_features_list: List[List[float]] = []
        for u, v, edge_data in graph.edges(data=True):
            if u not in node_to_idx or v not in node_to_idx:
                continue

            edge_pairs.append((node_to_idx[u], node_to_idx[v]))
            edge_features_list.append(self._encode_edge_feature_vector(edge_data))

            # For undirected graphs we add reverse edges explicitly for message passing.
            if not graph.is_directed() and u != v:
                edge_pairs.append((node_to_idx[v], node_to_idx[u]))
                edge_features_list.append(self._encode_edge_feature_vector(edge_data))

        if edge_pairs:
            edge_index = (
                torch.tensor(edge_pairs, dtype=torch.long, device=self.device)
                .t()
                .contiguous()
            )
            edge_features = torch.tensor(
                edge_features_list, dtype=torch.float32, device=self.device
            )
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            edge_features = torch.zeros(0, edge_dim, dtype=torch.float32, device=self.device)

        if use_tpe:
            tpe = self._compute_tpe_features(graph, node_order, node_to_idx, node_features)
        else:
            tpe = torch.zeros(num_nodes, 8, device=self.device, dtype=torch.float32)

        start_node = get_physical_start_node(graph)
        if start_node is None or start_node not in node_to_idx:
            start_node = next(
                (
                    node_id
                    for node_id in node_order
                    if self._room_role_flags(dict(graph.nodes[node_id])).get("is_start", False)
                ),
                None,
            )
        target_node = next(
            (
                node_id
                for node_id in node_order
                if self._room_role_flags(dict(graph.nodes[node_id])).get("has_goal", False)
            ),
            None,
        )

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'tpe': tpe,
            'node_positions': node_positions,
            'node_mask': torch.ones(num_nodes, device=self.device, dtype=torch.float32),
            'node_order': node_order,
            'node_to_idx': node_to_idx,
            'start_node_id': int(node_to_idx.get(start_node, 0)) if start_node is not None else 0,
            'target_idx': int(node_to_idx.get(target_node, -1)) if target_node is not None else -1,
        }
    
    def _get_neighbor_latents(
        self,
        room_id: int,
        graph: nx.Graph,
        generated_latents: Dict[int, torch.Tensor]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get directional neighbor latents for already-generated rooms.

        Best-effort direction inference uses node layout coordinates when present.
        """
        neighbor_dict: Dict[str, Optional[torch.Tensor]] = {
            'N': None, 'S': None, 'E': None, 'W': None
        }

        if room_id not in graph:
            return neighbor_dict

        if graph.is_directed():
            neighbor_ids = [n for n in graph.predecessors(room_id) if n in generated_latents]
        else:
            neighbor_ids = [n for n in graph.neighbors(room_id) if n in generated_latents]

        if not neighbor_ids:
            return neighbor_dict

        unresolved: List[int] = []
        for nid in sorted(neighbor_ids, key=_stable_node_sort_key):
            direction = self._infer_direction(graph, source_node=nid, target_node=room_id)
            if direction is not None and neighbor_dict[direction] is None:
                neighbor_dict[direction] = generated_latents[nid].to(self.device)
            else:
                unresolved.append(nid)

        # Stable fallback assignment when spatial metadata is missing or ambiguous.
        for direction, nid in zip(['N', 'W', 'E', 'S'], unresolved):
            if neighbor_dict[direction] is None:
                neighbor_dict[direction] = generated_latents[nid].to(self.device)

        return neighbor_dict

    def _get_neighbor_reference_room_maps(
        self,
        room_id: int,
        graph: nx.Graph,
        generated_rooms: Dict[Any, Any],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get directional neighboring room maps for already-generated rooms.

        This mirrors the teacher-forced room-map signal used during training
        and closes the exemplar-conditioning gap at inference time.
        """
        neighbor_dict: Dict[str, Optional[torch.Tensor]] = {
            'N': None, 'S': None, 'E': None, 'W': None
        }

        if room_id not in graph:
            return neighbor_dict

        if graph.is_directed():
            neighbor_ids = [n for n in graph.predecessors(room_id) if n in generated_rooms]
        else:
            neighbor_ids = [n for n in graph.neighbors(room_id) if n in generated_rooms]

        if not neighbor_ids:
            return neighbor_dict

        def _coerce_room_map(room_value: Any) -> Optional[torch.Tensor]:
            if isinstance(room_value, RoomGenerationResult):
                room_map = room_value.room_grid
            elif hasattr(room_value, "room_grid"):
                room_map = getattr(room_value, "room_grid")
            else:
                room_map = room_value
            if room_map is None:
                return None
            if isinstance(room_map, torch.Tensor):
                tensor = room_map.detach().to(self.device)
            else:
                tensor = torch.as_tensor(room_map, device=self.device)
            if tensor.dim() == 4 and int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == 1:
                tensor = tensor.squeeze(0).squeeze(0)
            elif tensor.dim() == 3 and int(tensor.shape[0]) == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 2:
                raise ValueError(
                    f"Reference room map must resolve to [H,W], got shape={tuple(tensor.shape)}."
                )
            return tensor.contiguous()

        unresolved: List[int] = []
        for nid in sorted(neighbor_ids, key=_stable_node_sort_key):
            room_map = _coerce_room_map(generated_rooms[nid])
            if room_map is None:
                continue
            direction = self._infer_direction(graph, source_node=nid, target_node=room_id)
            if direction is not None and neighbor_dict[direction] is None:
                neighbor_dict[direction] = room_map
            else:
                unresolved.append(nid)

        for direction, nid in zip(['N', 'W', 'E', 'S'], unresolved):
            if neighbor_dict[direction] is None:
                room_map = _coerce_room_map(generated_rooms[nid])
                if room_map is not None:
                    neighbor_dict[direction] = room_map

        return neighbor_dict
    
    def _extract_room_start_goal(
        self,
        graph: nx.Graph,
        room_id: int
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Extract room-local start/goal hints for symbolic repair.

        Uses node metadata when available, otherwise infers a sensible
        left-to-right flow based on graph predecessors/successors.
        """
        if room_id not in graph:
            return ((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1))

        attrs = graph.nodes[room_id]

        start = (
            self._parse_room_coord(attrs.get('start_pos'))
            or self._parse_room_coord(attrs.get('entry_pos'))
            or self._parse_room_coord(attrs.get('entrance'))
        )
        goal = (
            self._parse_room_coord(attrs.get('goal_pos'))
            or self._parse_room_coord(attrs.get('exit_pos'))
            or self._parse_room_coord(attrs.get('exit'))
        )

        if start is None:
            has_pred = graph.in_degree(room_id) > 0 if graph.is_directed() else graph.degree(room_id) > 0
            start = (ROOM_HEIGHT // 2, 0) if has_pred else (ROOM_HEIGHT // 2, ROOM_WIDTH // 4)

        if goal is None:
            has_succ = graph.out_degree(room_id) > 0 if graph.is_directed() else graph.degree(room_id) > 0
            goal = (ROOM_HEIGHT // 2, ROOM_WIDTH - 1) if has_succ else (ROOM_HEIGHT // 2, (3 * ROOM_WIDTH) // 4)

        start = self._clamp_room_coord(start)
        goal = self._clamp_room_coord(goal)

        if start == goal:
            goal = self._clamp_room_coord((goal[0], goal[1] + 1))

        return (start, goal)
    
    def _stitch_rooms(
        self,
        rooms: Dict[int, RoomGenerationResult],
        graph: nx.Graph
    ) -> np.ndarray:
        """
        Stitch generated rooms into a global dungeon grid.
        """
        if not rooms:
            return np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
        return self.stitch_room_layout(rooms, graph).dungeon_grid

    def stitch_room_layout(
        self,
        rooms: Dict[Any, Any],
        graph: nx.Graph,
        *,
        enforce_room_dimensions: Optional[Tuple[int, int]] = (ROOM_HEIGHT, ROOM_WIDTH),
        carve_connections: bool = True,
    ) -> StitchedRoomLayout:
        """
        Shared internal stitch core used by both the main and advanced pipelines.

        Returns both the stitched grid and bbox metadata so downstream systems
        can operate from one consistent room-layout realization.
        """
        return build_stitched_room_layout(
            rooms=rooms,
            graph=graph,
            fill_tile=int(SEMANTIC_PALETTE.get("VOID", 0)),
            sort_key=_stable_node_sort_key,
            node_position_getter=self._get_node_grid_position,
            first_free_position_fn=self._first_free_position,
            enforce_room_dimensions=enforce_room_dimensions,
            carve_connections=carve_connections,
            diagnostic_callback=self._bump_diagnostic,
        )

    def stitch_rooms(
        self,
        rooms: Dict[int, RoomGenerationResult],
        graph: nx.Graph,
    ) -> np.ndarray:
        """
        Public room stitching entry point.

        Custom stitchers may implement `stitch_rooms(rooms=..., graph=...)`.
        When absent, the built-in fallback stitcher is used.
        """
        stitcher = self.stitcher
        if stitcher is not None:
            stitch_fn = getattr(stitcher, "stitch_rooms", None)
            if callable(stitch_fn):
                try:
                    stitched = stitch_fn(rooms=rooms, graph=graph)
                except TypeError:
                    stitched = stitch_fn(rooms, graph)
                if isinstance(stitched, StitchedRoomLayout):
                    return stitched.dungeon_grid
                return stitched
        return self.stitch_room_layout(rooms=rooms, graph=graph).dungeon_grid

    def _build_room_boundary_constraints(
        self,
        graph: nx.Graph,
        room_id: Any,
    ) -> torch.Tensor:
        """Build [1, 8] boundary constraints from incident topology edges."""
        has_neighbor: Dict[str, bool] = {"N": False, "S": False, "E": False, "W": False}
        required_door: Dict[str, bool] = {"N": False, "S": False, "E": False, "W": False}

        if room_id not in graph:
            return torch.zeros(1, 8, device=self.device, dtype=torch.float32)

        incident: List[Any] = []
        if graph.is_directed():
            incident.extend(list(graph.predecessors(room_id)))
            incident.extend(list(graph.successors(room_id)))
        else:
            incident.extend(list(graph.neighbors(room_id)))

        incident_unique = sorted(set(incident), key=_stable_node_sort_key)
        unresolved: List[Any] = []
        for nid in incident_unique:
            direction = self._infer_direction(graph, source_node=nid, target_node=room_id)
            if direction is None:
                unresolved.append(nid)
                continue
            has_neighbor[direction] = True
            required_door[direction] = True

        for direction, _nid in zip(["N", "E", "S", "W"], unresolved):
            if not has_neighbor[direction]:
                has_neighbor[direction] = True
                required_door[direction] = True

        boundary = build_boundary_constraints(has_neighbor=has_neighbor, required_door=required_door)
        return boundary.to(device=self.device, dtype=torch.float32).unsqueeze(0)

    def _room_role_flags(self, attrs: Dict[str, Any]) -> Dict[str, bool]:
        """Extract high-level room-role booleans from graph node metadata."""
        tokens = self._parse_label_tokens(attrs.get("label"))
        raw_type = str(attrs.get("type", attrs.get("node_type", attrs.get("room_type", ""))) or "").strip().lower()
        role_tokens = set(tokens) | set(self._parse_label_tokens(raw_type))
        difficulty_rating = str(attrs.get("difficulty_rating", "") or "").strip().upper()

        def _hint(name: str, *aliases: str) -> bool:
            return self._coerce_bool(attrs.get(name)) or any(self._coerce_bool(attrs.get(alias)) for alias in aliases)

        return {
            "is_start": _hint("is_start", "is_entry") or raw_type in {"start", "entry"} or "start" in role_tokens or "entry" in role_tokens,
            "has_enemy": _hint("has_enemy") or "e" in role_tokens or "enemy" in role_tokens,
            "has_key": _hint("has_key") or "k" in role_tokens or "key" in role_tokens,
            "has_item": _hint("has_item", "has_macro_item", "has_minor_item") or "i" in role_tokens or "item" in role_tokens or "treasure" in role_tokens,
            "has_goal": _hint("has_triforce", "is_triforce", "is_goal") or raw_type in {"goal", "triforce"} or "t" in role_tokens or "goal" in role_tokens or "triforce" in role_tokens,
            "has_boss": _hint("has_boss", "is_boss") or "b" in role_tokens or "boss" in role_tokens,
            "has_puzzle": _hint("has_puzzle") or "p" in role_tokens or "puzzle" in role_tokens or raw_type in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"} or "puzzle" in raw_type,
            "is_tutorial_puzzle": bool(_hint("is_tutorial") or raw_type == "tutorial_puzzle" or difficulty_rating == "SAFE"),
            "is_combat_puzzle": bool(raw_type == "combat_puzzle"),
            "is_complex_puzzle": bool(raw_type == "complex_puzzle" or difficulty_rating in {"HARD", "EXTREME"}),
            "is_switch_puzzle": bool(raw_type == "switch"),
        }

    def _resolve_puzzle_room_scaffold_profile(
        self,
        *,
        attrs: Dict[str, Any],
        role_flags: Dict[str, bool],
        semantics: Dict[str, Any],
        node_type: str,
    ) -> Dict[str, Any]:
        """
        Derive a per-room constructive puzzle profile from topology semantics.

        Hybrid PCG systems work best when local room structure reflects explicit
        mission intent. This profile keeps the existing scaffold mechanism but
        adapts its density to pedagogical puzzle subtype information.
        """
        archetype = self._select_puzzle_room_scaffold_archetype(
            role_flags=role_flags,
            semantics=semantics,
            node_type=node_type,
        )
        gate_family = self._classify_puzzle_gate_family(
            role_flags=role_flags,
            semantics=semantics,
            node_type=node_type,
        )
        branch_density = float(max(0.0, min(1.0, getattr(self, "default_puzzle_room_branch_density", 0.75))))
        block_budget = int(max(0, getattr(self, "default_puzzle_room_block_budget", 28)))
        preserve_margin = int(max(0, getattr(self, "default_puzzle_room_preserve_route_margin", 0)))
        difficulty_rating = str(attrs.get("difficulty_rating", "") or "").strip().upper()

        edge_constraints = semantics.get("edge_constraints", {})
        flat_edge_tokens: Set[str] = set()
        for tokens in edge_constraints.values():
            flat_edge_tokens.update(str(token) for token in tokens)

        required_doors = semantics.get("required_doors", {})
        required_door_count = int(sum(1 for enabled in required_doors.values() if enabled))

        if role_flags.get("is_tutorial_puzzle", False):
            archetype = "gate"
            branch_density = min(branch_density, 0.25)
            block_budget = min(block_budget, 10)
        elif role_flags.get("is_combat_puzzle", False):
            archetype = "combat"
            branch_density = min(branch_density, 0.30)
            block_budget = min(block_budget, 12)
        elif role_flags.get("is_complex_puzzle", False):
            if archetype not in {"hub", "serpentine"}:
                archetype = "hub" if required_door_count >= 3 else "serpentine"
            branch_density = max(branch_density, 0.70)
            block_budget = max(block_budget, 26)
        elif gate_family in {"switch", "toggle"}:
            archetype = "gate"
            branch_density = max(branch_density, 0.50)
            block_budget = max(block_budget, 18)
        elif gate_family in {"bombable", "item_unlock", "key"} and archetype not in {"hub", "combat"}:
            archetype = "gate"
            if gate_family == "key":
                branch_density = max(branch_density, 0.40)
                block_budget = max(block_budget, 15)
            elif gate_family == "item_unlock":
                branch_density = max(branch_density, 0.42)
                block_budget = max(block_budget, 16)
            else:
                branch_density = max(branch_density, 0.45)
                block_budget = max(block_budget, 17)
        elif node_type in {"item", "protection_item", "minor_item", "treasure", "stair", "stairs_up", "stairs_down", "warp"}:
            archetype = "island"
            branch_density = min(max(branch_density, 0.40), 0.65)
            block_budget = max(block_budget, 16)
        elif difficulty_rating == "MODERATE" and archetype == "serpentine":
            branch_density = min(max(branch_density, 0.45), 0.65)

        return {
            "archetype": str(archetype),
            "gate_family": str(gate_family),
            "branch_density": float(max(0.0, min(1.0, branch_density))),
            "block_budget": int(max(0, block_budget)),
            "preserve_route_margin": int(max(0, preserve_margin)),
        }

    def _extract_room_topology_semantics(
        self,
        graph: nx.Graph,
        room_id: Any,
    ) -> Dict[str, Any]:
        """Extract doorway and edge semantics for one room from the mission graph."""
        required_doors: Dict[str, bool] = {"N": False, "S": False, "E": False, "W": False}
        edge_constraints: Dict[str, Set[str]] = {"N": set(), "S": set(), "E": set(), "W": set()}
        incoming_dirs: Set[str] = set()
        outgoing_dirs: Set[str] = set()

        if room_id not in graph:
            return {
                "required_doors": required_doors,
                "edge_constraints": edge_constraints,
                "incoming_dirs": incoming_dirs,
                "outgoing_dirs": outgoing_dirs,
            }

        incident: List[Tuple[Any, Dict[str, Any], str]] = []
        if graph.is_directed():
            incident.extend((nid, dict(graph.get_edge_data(nid, room_id, default={}) or {}), "incoming") for nid in graph.predecessors(room_id))
            incident.extend((nid, dict(graph.get_edge_data(room_id, nid, default={}) or {}), "outgoing") for nid in graph.successors(room_id))
        else:
            incident.extend((nid, dict(graph.get_edge_data(room_id, nid, default={}) or {}), "bidirectional") for nid in graph.neighbors(room_id))

        unresolved: List[Tuple[Any, Dict[str, Any], str]] = []
        for nid, edge_data, flow in sorted(incident, key=lambda item: _stable_node_sort_key(item[0])):
            direction = self._infer_direction(graph, source_node=nid, target_node=room_id)
            if direction is None:
                unresolved.append((nid, edge_data, flow))
                continue
            required_doors[direction] = True
            if flow in {"incoming", "bidirectional"}:
                incoming_dirs.add(direction)
            if flow in {"outgoing", "bidirectional"}:
                outgoing_dirs.add(direction)
            edge_constraints[direction].update(
                parse_edge_type_tokens(
                    label=str(edge_data.get("label", "") or ""),
                    edge_type=str(edge_data.get("edge_type", edge_data.get("type", "")) or ""),
                )
            )

        for direction, (_nid, edge_data, _flow) in zip(["N", "E", "S", "W"], unresolved):
            if required_doors[direction]:
                continue
            required_doors[direction] = True
            incoming_dirs.add(direction)
            outgoing_dirs.add(direction)
            edge_constraints[direction].update(
                parse_edge_type_tokens(
                    label=str(edge_data.get("label", "") or ""),
                    edge_type=str(edge_data.get("edge_type", edge_data.get("type", "")) or ""),
                )
            )

        return {
            "required_doors": required_doors,
            "edge_constraints": edge_constraints,
            "incoming_dirs": incoming_dirs,
            "outgoing_dirs": outgoing_dirs,
        }

    def _build_room_plan_trace(
        self,
        graph: nx.Graph,
        room_id: Any,
        room_grid: np.ndarray,
        *,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Build a concrete traversability plan mask from the current room grid."""
        if room_id not in graph:
            return np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
        attrs = graph.nodes[room_id]
        if start_goal is None:
            start_goal = self._extract_room_start_goal(graph, room_id)
        start, goal = start_goal if start_goal is not None else (None, None)
        semantics = self._extract_room_topology_semantics(graph, room_id)
        budget = self._resolve_validator_plan_state_budget(
            attrs=attrs,
            semantics=semantics,
        )
        return build_semantic_room_plan_trace(
            np.asarray(room_grid, dtype=np.int32),
            start=start,
            goal=goal,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            edge_constraint_tokens=semantics["edge_constraints"],
            room_role_flags=self._room_role_flags(attrs),
            validator_plan_max_states=budget,
        ).astype(np.float32, copy=False)

    def _resolve_validator_plan_state_budget(
        self,
        *,
        attrs: Mapping[str, Any],
        semantics: Mapping[str, Any],
    ) -> int:
        """
        Adapt the room-local validator plan budget to semantic complexity.

        A single tiny cap for every room is too brittle for richer puzzle rooms,
        but a truly unbounded local planner is not acceptable in this pipeline
        because it can explode memory/runtime during export. We therefore scale
        the budget with room complexity while keeping a hard cap.
        """
        base_budget = int(max(32, int(getattr(self, "default_validator_plan_max_states", DEFAULT_VALIDATOR_PLAN_MAX_STATES))))
        role_flags = self._room_role_flags(dict(attrs))
        active_roles = sum(1 for enabled in role_flags.values() if bool(enabled))
        required_doors = sum(
            1 for enabled in dict(semantics.get("required_doors", {})).values()
            if bool(enabled)
        )
        distinct_tokens: Set[str] = set()
        for token_set in dict(semantics.get("edge_constraints", {})).values():
            distinct_tokens.update(str(token).strip().lower() for token in set(token_set or set()))
        complex_gate_tokens = {
            "switch",
            "switch_locked",
            "state_block",
            "on_off_gate",
            "toggle",
            "bombable",
            "item_gate",
            "item_locked",
            "key_locked",
            "boss_locked",
            "combat",
        }
        complexity_score = int(active_roles)
        complexity_score += max(0, int(required_doors) - 2)
        complexity_score += int(len(distinct_tokens))
        if bool(distinct_tokens & complex_gate_tokens):
            complexity_score += 2
        if bool(role_flags.get("has_puzzle", False)):
            complexity_score += 1
        if bool(role_flags.get("has_enemy", False)):
            complexity_score += 1

        if complexity_score <= 0:
            return base_budget

        complexity_multiplier = 1.0 + min(3.0, 0.2 * float(complexity_score))
        hard_cap = max(base_budget, min(4096, int(base_budget * 4)))
        return int(max(base_budget, min(hard_cap, round(base_budget * complexity_multiplier))))

    def _build_room_topology_condition_tensor(
        self,
        graph: nx.Graph,
        room_id: Any,
        *,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """Build explicit per-room topology prior [1, C, H, W] for the diffusion U-Net."""
        if room_id not in graph:
            return torch.zeros(
                1,
                ROOM_TOPOLOGY_CHANNEL_COUNT,
                ROOM_HEIGHT,
                ROOM_WIDTH,
                device=self.device,
                dtype=torch.float32,
            )

        attrs = graph.nodes[room_id]
        if start_goal is None:
            start_goal = self._extract_room_start_goal(graph, room_id)
        start, goal = start_goal if start_goal is not None else (None, None)
        semantics = self._extract_room_topology_semantics(graph, room_id)
        budget = self._resolve_validator_plan_state_budget(
            attrs=attrs,
            semantics=semantics,
        )

        topo_np = build_room_topology_condition_map(
            start=start,
            goal=goal,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            edge_constraint_tokens=semantics["edge_constraints"],
            room_role_flags=self._room_role_flags(attrs),
            semantic_role_prior_strength=self.default_semantic_role_prior_strength,
            semantic_puzzle_offset=self.default_semantic_puzzle_offset,
            validator_plan_max_states=budget,
            puzzle_stage_topology_enabled=self.default_puzzle_stage_topology_enabled,
            puzzle_stage_trace_decay=self.default_puzzle_stage_trace_decay,
        )
        return torch.from_numpy(topo_np).unsqueeze(0).to(device=self.device, dtype=torch.float32)

    @staticmethod
    def _extract_explicit_style_id(graph: nx.Graph, *, room_id: Any) -> Optional[int]:
        """
        Extract an explicit style/theme token for one room.

        This prefers explicit numeric IDs, but it also accepts the repo's
        canonical symbolic sector-theme labels so generated mission graphs can
        drive the style path without inventing a broader visual taxonomy.
        """
        candidate_values: List[Any] = []
        if room_id in graph.nodes:
            node_attrs = dict(graph.nodes[room_id])
            candidate_values.extend(
                iter_style_metadata_candidates(
                    node_attrs,
                    keys=("style_id", "theme_id", "sector_theme_id", "sector_theme", "theme", "theme_name"),
                )
            )
        graph_attrs = getattr(graph, "graph", None)
        if isinstance(graph_attrs, dict):
            candidate_values.extend(
                iter_style_metadata_candidates(
                    graph_attrs,
                    keys=("style_id", "theme_id", "sector_theme_id", "sector_theme", "theme", "theme_name"),
                )
            )
        return resolve_style_token_id(*candidate_values)

    def _build_room_graph_context(
        self,
        *,
        graph_data: Dict[str, Any],
        mission_graph: nx.Graph,
        room_id: Any,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Build per-room graph context shared by condition encoding and diffusion."""
        node_to_idx = dict(graph_data.get('node_to_idx', {}) or {})
        current_node_idx = node_to_idx.get(room_id, 0)
        start_node = get_physical_start_node(mission_graph)
        if start_node is None or start_node not in node_to_idx:
            start_node = next(
                (
                    node_id
                    for node_id, attrs in mission_graph.nodes(data=True)
                    if self._room_role_flags(dict(attrs)).get("is_start", False)
                ),
                None,
            )
        target_node = next(
            (
                node_id
                for node_id, attrs in mission_graph.nodes(data=True)
                if self._room_role_flags(dict(attrs)).get("has_goal", False)
            ),
            None,
        )
        style_id = self._extract_explicit_style_id(mission_graph, room_id=room_id)
        current_node_distance = compute_current_node_distance_features(
            graph_data.get('edge_index'),
            int(graph_data.get('node_features').shape[0]) if isinstance(graph_data.get('node_features'), torch.Tensor) else 0,
            current_node_idx=current_node_idx,
            device=self.device,
            dtype=torch.float32,
            max_distance=self.current_node_distance_max,
        )
        attrs = dict(mission_graph.nodes[room_id]) if room_id in mission_graph else {}
        if start_goal is None:
            start_goal = self._extract_room_start_goal(mission_graph, room_id)
        start, goal = start_goal if start_goal is not None else (None, None)
        semantics = (
            self._extract_room_topology_semantics(mission_graph, room_id)
            if room_id in mission_graph
            else {
                "required_doors": {},
                "incoming_dirs": set(),
                "outgoing_dirs": set(),
                "edge_constraints": {},
            }
        )
        budget = self._resolve_validator_plan_state_budget(
            attrs=attrs,
            semantics=semantics,
        )
        puzzle_stage_condition = build_puzzle_stage_condition_metadata(
            room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
            start=start,
            goal=goal,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            edge_constraint_tokens=semantics["edge_constraints"],
            room_role_flags=self._room_role_flags(attrs),
            validator_plan_max_states=budget,
            semantic_puzzle_offset=self.default_semantic_puzzle_offset,
            stage_trace_decay=self.default_puzzle_stage_trace_decay,
        )
        return {
            'graph_scope': 'room',
            'node_features': graph_data.get('node_features'),
            'edge_index': graph_data.get('edge_index'),
            'edge_features': graph_data.get('edge_features'),
            'tpe': graph_data.get('tpe'),
            'node_positions': graph_data.get('node_positions'),
            'node_mask': graph_data.get('node_mask'),
            'has_room_anchor': True,
            'mission_graph': mission_graph,
            'current_node_idx': current_node_idx,
            'start_node_id': int(node_to_idx.get(start_node, 0)) if start_node is not None else 0,
            'target_idx': int(node_to_idx.get(target_node, -1)) if target_node is not None else -1,
            'puzzle_room_structure_enabled': bool(self.default_puzzle_room_structure_enabled),
            'puzzle_stage_condition': puzzle_stage_condition,
            **({'current_node_distance': current_node_distance} if self.use_current_node_distance_features else {}),
            **({'style_id': int(style_id)} if style_id is not None else {}),
            'room_topology_map': self._build_room_topology_condition_tensor(
                mission_graph,
                room_id,
                start_goal=start_goal,
            ),
        }

    @staticmethod
    def _edge_tokens_to_door_tile(tokens: Set[str]) -> int:
        """Map edge-semantic tokens to a concrete door tile ID."""
        normalized = {str(t).strip().lower() for t in set(tokens)}
        if {"boss_locked"} & normalized:
            return int(SEMANTIC_PALETTE["DOOR_BOSS"])
        if {"key_locked", "locked"} & normalized:
            return int(SEMANTIC_PALETTE["DOOR_LOCKED"])
        if {"bombable"} & normalized:
            return int(SEMANTIC_PALETTE["DOOR_BOMB"])
        if {"switch", "switch_locked", "puzzle"} & normalized:
            return int(SEMANTIC_PALETTE["DOOR_PUZZLE"])
        if {"soft_locked", "one_way", "shutter"} & normalized:
            return int(SEMANTIC_PALETTE["DOOR_SOFT"])
        return int(SEMANTIC_PALETTE["DOOR_OPEN"])

    def _build_masked_room_fixed_tokens(
        self,
        graph: nx.Graph,
        room_id: Any,
        *,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build hard-known token layout for the discrete masked room generator.

        This encodes exact doors and explicit start/goal hints as fixed tokens
        that are never re-masked during iterative sampling.
        """
        fixed_tokens = torch.zeros((1, ROOM_HEIGHT, ROOM_WIDTH), device=self.device, dtype=torch.long)
        fixed_mask = torch.zeros((1, ROOM_HEIGHT, ROOM_WIDTH), device=self.device, dtype=torch.bool)

        if room_id not in graph:
            return fixed_tokens, fixed_mask

        semantics = self._extract_room_topology_semantics(graph, room_id)
        for direction, enabled in semantics["required_doors"].items():
            if not bool(enabled):
                continue
            tile_id = self._edge_tokens_to_door_tile(semantics["edge_constraints"].get(direction, set()))
            spec = DOOR_POSITIONS[str(direction)]
            if direction in {"N", "S"}:
                row = int(spec["row"])
                c0 = int(spec["col_start"])
                c1 = int(spec["col_end"]) + 1
                fixed_tokens[0, row, c0:c1] = tile_id
                fixed_mask[0, row, c0:c1] = True
            else:
                col = int(spec["col"])
                r0 = int(spec["row_start"])
                r1 = int(spec["row_end"]) + 1
                fixed_tokens[0, r0:r1, col] = tile_id
                fixed_mask[0, r0:r1, col] = True

        if start_goal is None:
            start_goal = self._extract_room_start_goal(graph, room_id)
        if start_goal is not None:
            start, goal = start_goal
            role_flags = self._room_role_flags(dict(graph.nodes[room_id]))
            semantic_anchors = build_room_semantic_anchor_points(
                room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
                start=start,
                goal=goal,
                required_doors=semantics["required_doors"],
                incoming_dirs=semantics["incoming_dirs"],
                outgoing_dirs=semantics["outgoing_dirs"],
                room_role_flags=role_flags,
                semantic_puzzle_offset=self.default_semantic_puzzle_offset,
            )
            sr, sc = self._clamp_room_coord(semantic_anchors.get("start", start))
            gr, gc = self._clamp_room_coord(semantic_anchors.get("goal", goal))

            # These anchors primarily encode local traversability hints. Only the
            # actual start / goal rooms should receive semantic START / TRIFORCE
            # tiles; all other rooms keep these anchors walkable as plain floor.
            start_tile = (
                int(SEMANTIC_PALETTE["START"])
                if role_flags.get("is_start", False)
                else int(SEMANTIC_PALETTE["FLOOR"])
            )
            goal_tile = (
                int(SEMANTIC_PALETTE["TRIFORCE"])
                if role_flags.get("has_goal", False)
                else int(SEMANTIC_PALETTE["FLOOR"])
            )
            fixed_tokens[0, sr, sc] = start_tile
            fixed_mask[0, sr, sc] = True
            fixed_tokens[0, gr, gc] = goal_tile
            fixed_mask[0, gr, gc] = True

            marker_to_anchor = {
                int(TileID.KEY_SMALL): "key",
                int(TileID.KEY_BOSS): "key",
                int(TileID.KEY_ITEM): "item",
                int(TileID.ITEM_MINOR): "item",
                int(TileID.BOSS): "boss",
                int(TileID.PUZZLE): "puzzle",
                int(TileID.STAIR): "item",
            }
            for tile_id in self._resolve_room_graph_markers(graph, room_id):
                if int(tile_id) in {int(TileID.START), int(TileID.TRIFORCE), int(TileID.ENEMY)}:
                    continue
                anchor_name = marker_to_anchor.get(int(tile_id))
                if anchor_name is None:
                    continue
                point = semantic_anchors.get(anchor_name)
                if point is None:
                    continue
                rr, cc = self._clamp_room_coord(point)
                fixed_tokens[0, rr, cc] = int(tile_id)
                fixed_mask[0, rr, cc] = True

        return fixed_tokens, fixed_mask

    def _build_room_position_tensor(
        self,
        graph: nx.Graph,
        room_id: Any,
        fallback_order_index: int,
    ) -> torch.Tensor:
        """Build [1, 2] room position tensor from graph metadata."""
        pos = self._get_node_grid_position(graph, room_id)
        if pos is None:
            pos = (int(fallback_order_index), 0)
        return torch.tensor([[float(pos[0]), float(pos[1])]], device=self.device, dtype=torch.float32)

    def _compute_strict_room_placement(
        self,
        graph: nx.Graph,
        room_ids: List[Any],
    ) -> Dict[Any, Tuple[int, int]]:
        """Compatibility wrapper around shared strict room placement."""
        return compute_strict_room_placement(
            graph=graph,
            room_ids=room_ids,
            sort_key=_stable_node_sort_key,
            node_position_getter=self._get_node_grid_position,
            first_free_position_fn=self._first_free_position,
        )

    def _compute_relaxed_room_placement(
        self,
        graph: nx.Graph,
        room_ids: List[Any],
    ) -> Dict[Any, Tuple[int, int]]:
        """Compatibility wrapper around shared relaxed room placement."""
        return compute_relaxed_room_placement(
            graph=graph,
            room_ids=room_ids,
            sort_key=_stable_node_sort_key,
            node_position_getter=self._get_node_grid_position,
            first_free_position_fn=self._first_free_position,
        )

    def _solve_component_strict_adjacency(
        self,
        comp_nodes: List[Any],
        adjacency: Dict[Any, set],
        explicit_pos: Dict[Any, Tuple[int, int]],
    ) -> Dict[Any, Tuple[int, int]]:
        """Compatibility wrapper around shared component strict-adjacency solver."""
        return solve_component_strict_adjacency(
            comp_nodes=comp_nodes,
            adjacency=adjacency,
            explicit_pos=explicit_pos,
            sort_key=_stable_node_sort_key,
        )
    
    def _validate_dungeon(self, dungeon_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Validate dungeon solvability and compute MAP-Elites descriptors.

        Uses the project validator when available, with graceful fallback.
        """
        if self.map_elites is None:
            logger.debug("Skipping dungeon validation because no map_elites component is configured.")
            return None
        floor_id = int(SEMANTIC_PALETTE.get('FLOOR', 1))
        enemy_id = int(SEMANTIC_PALETTE.get('ENEMY', 7))
        key_id = int(SEMANTIC_PALETTE.get('KEY_SMALL', SEMANTIC_PALETTE.get('KEY', 8)))
        lock_id = int(SEMANTIC_PALETTE.get('DOOR_LOCKED', 11))
        playable_area = int((dungeon_grid == floor_id).sum())
        leniency = float(self.map_elites.calculate_leniency(dungeon_grid))
        enemy_count = int((dungeon_grid == enemy_id).sum())
        key_count = int((dungeon_grid == key_id).sum())
        lock_count = int((dungeon_grid == lock_id).sum())

        try:
            from src.simulation.validator import ZeldaValidator

            validator = ZeldaValidator()
            result = validator.validate_single(dungeon_grid)

            path_length = int(result.path_length) if result.is_solvable else 0
            linearity = float(self.map_elites.calculate_linearity(path_length, playable_area))
            backtracking = float(np.clip(getattr(result, 'backtracking_score', 0.0), 0.0, 1.0))
            reachability = float(np.clip(getattr(result, 'reachability', 0.0), 0.0, 1.0))
            lock_pressure = min(1.0, lock_count / max(1.0, float(max(1, key_count))))
            path_pressure = min(1.0, float(path_length) / max(1.0, np.sqrt(max(1, playable_area)) * 2.5))
            progression_complexity = float(np.clip(
                (0.45 * lock_pressure) + (0.35 * backtracking) + (0.20 * path_pressure),
                0.0,
                1.0,
            ))
            topology_complexity = float(np.clip(
                (0.55 * float(np.clip(linearity, 0.0, 1.0))) + (0.45 * progression_complexity),
                0.0,
                1.0,
            ))
            quality_score = float(np.clip(
                (0.35 * reachability) +
                (0.25 * (1.0 - abs(backtracking - 0.25) / 0.25 if backtracking <= 0.5 else 0.0)) +
                (0.20 * float(result.is_valid_syntax)) +
                (0.20 * path_pressure),
                0.0,
                1.0,
            ))
            return {
                'solvable': bool(result.is_solvable),
                'path_length': path_length,
                'linearity': linearity,
                'leniency': leniency,
                'progression_complexity': progression_complexity,
                'topology_complexity': topology_complexity,
                'quality_score': quality_score,
                'backtracking_score': backtracking,
                'reachability': reachability,
                'key_count': key_count,
                'lock_count': lock_count,
                'enemy_count': enemy_count,
                'is_valid_syntax': bool(result.is_valid_syntax),
                'error_message': str(result.error_message) if result.error_message else "",
            }
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            self._bump_diagnostic("dungeon_validation_fallback")
            logger.warning(f"Dungeon validation failed: {e}")
            return {
                'solvable': False,
                'path_length': 0,
                'linearity': 0.0,
                'leniency': leniency,
                'progression_complexity': 0.0,
                'topology_complexity': 0.0,
                'quality_score': 0.0,
                'backtracking_score': 0.0,
                'reachability': 0.0,
                'key_count': key_count,
                'lock_count': lock_count,
                'enemy_count': enemy_count,
                'is_valid_syntax': False,
                'error_message': f"validator_error: {e}",
            }

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

    def _encode_edge_feature_vector(self, edge_data: Dict[str, Any]) -> List[float]:
        _, edge_dim = self._condition_feature_dims()
        return encode_edge_feature_vector(edge_data, edge_dim=edge_dim)

    def _compute_tpe_features(
        self,
        graph: nx.Graph,
        node_order: List[int],
        node_to_idx: Dict[int, int],
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

    def _parse_label_tokens(self, label: Any) -> set:
        """Compatibility wrapper around extracted parsing helper."""
        return parse_label_tokens(label)

    def _coerce_bool(self, value: Any) -> bool:
        """Compatibility wrapper around extracted parsing helper."""
        return coerce_bool(value)

    def _coerce_difficulty(self, value: Any) -> float:
        """Compatibility wrapper around extracted parsing helper."""
        return coerce_difficulty(value)

    def _parse_room_coord(self, value: Any) -> Optional[Tuple[int, int]]:
        """Compatibility wrapper around extracted parsing helper."""
        return parse_room_coord(value)

    def _clamp_room_coord(self, coord: Tuple[int, int]) -> Tuple[int, int]:
        """Compatibility wrapper around extracted parsing helper."""
        return clamp_room_coord(coord)

    def _normalize_room_coord(
        self,
        coord: Any,
        *,
        field_name: str,
    ) -> Tuple[int, int]:
        """Validate and clamp a room-local coordinate expressed as (row, col)."""
        if not isinstance(coord, (tuple, list)) or len(coord) != 2:
            raise ValueError(
                f"{field_name} must be a 2-item (row, col) coordinate, got {coord!r}."
            )
        try:
            row = int(coord[0])
            col = int(coord[1])
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(
                f"{field_name} must contain integer-compatible row/col values."
            ) from e
        return self._clamp_room_coord((row, col))

    def _normalize_start_goal_coords(
        self,
        start_goal_coords: Any,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Normalize API-provided repair endpoints to internal (row, col) coordinates."""
        if not isinstance(start_goal_coords, (tuple, list)) or len(start_goal_coords) != 2:
            raise ValueError(
                "start_goal_coords must be ((start_row, start_col), (goal_row, goal_col))."
            )
        start = self._normalize_room_coord(start_goal_coords[0], field_name="start")
        goal = self._normalize_room_coord(start_goal_coords[1], field_name="goal")
        return start, goal

    def _get_node_grid_position(self, graph: nx.Graph, node_id: int) -> Optional[Tuple[int, int]]:
        """Compatibility wrapper around extracted parsing helper."""
        return get_node_grid_position(graph, node_id)

    def _infer_direction(
        self,
        graph: nx.Graph,
        source_node: int,
        target_node: int,
    ) -> Optional[str]:
        """Compatibility wrapper around extracted parsing helper."""
        return infer_direction(graph, source_node, target_node)

    def _first_free_position(
        self,
        start_pos: Tuple[int, int],
        occupied: set,
    ) -> Tuple[int, int]:
        """Compatibility wrapper around extracted spatial helper."""
        return first_free_position(start_pos, occupied)

    def _fit_room_grid(self, room_grid: np.ndarray) -> np.ndarray:
        """Compatibility wrapper around extracted spatial helper."""
        return fit_room_grid(room_grid)

    @torch.no_grad()
    def _encode_room_grid_to_latent(
        self,
        room_grid: np.ndarray,
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode finalized room grid back into latent space for neighbor conditioning."""
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
        return z_q.detach()

    def _carve_room_connection(
        self,
        global_grid: np.ndarray,
        src_pos: Tuple[int, int],
        dst_pos: Tuple[int, int],
        edge_data: Optional[Dict[str, Any]] = None,
        has_reverse_edge: bool = False,
    ) -> None:
        """Compatibility wrapper around extracted spatial helper."""
        carve_room_connection(
            global_grid,
            src_pos,
            dst_pos,
            edge_data=edge_data,
            has_reverse_edge=has_reverse_edge,
        )

    def _carve_room_connection_with_fallback(
        self,
        global_grid: np.ndarray,
        src_pos: Tuple[int, int],
        dst_pos: Tuple[int, int],
        edge_data: Optional[Dict[str, Any]] = None,
        has_reverse_edge: bool = False,
    ) -> None:
        """Compatibility wrapper around shared bbox-aware connection carving."""
        src_bbox = (
            int(src_pos[1] * ROOM_WIDTH),
            int(src_pos[0] * ROOM_HEIGHT),
            int((src_pos[1] + 1) * ROOM_WIDTH - 1),
            int((src_pos[0] + 1) * ROOM_HEIGHT - 1),
        )
        dst_bbox = (
            int(dst_pos[1] * ROOM_WIDTH),
            int(dst_pos[0] * ROOM_HEIGHT),
            int((dst_pos[1] + 1) * ROOM_WIDTH - 1),
            int((dst_pos[0] + 1) * ROOM_HEIGHT - 1),
        )
        carve_room_connection_between_bboxes(
            global_grid,
            src_bbox,
            dst_bbox,
            edge_data=edge_data,
            has_reverse_edge=has_reverse_edge,
            fill_tile=int(SEMANTIC_PALETTE.get("VOID", 0)),
            diagnostic_callback=self._bump_diagnostic,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def topology_generation_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build Block I generation kwargs from the validated global config payload."""
    stage = config["topology"]
    return {
        "target_curve": list(stage["default_target_curve"]),
        "num_rooms": stage["num_rooms"],
        "population_size": stage["population_size"],
        "generations": stage["generations"],
        "mutation_rate": stage["mutation_rate"],
        "crossover_rate": stage["crossover_rate"],
        "genome_length": stage["genome_length"],
        "rule_space": stage["rule_space"],
        "transition_mix": stage["transition_mix"],
        "search_strategy": stage["search_strategy"],
        "qd_archive_cells": stage["qd_archive_cells"],
        "qd_init_random_fraction": stage["qd_init_random_fraction"],
        "qd_emitter_mutation_rate": stage["qd_emitter_mutation_rate"],
        "max_lock_key_rules": stage["max_lock_key_rules"],
        "enable_rule_credit_assignment": stage["enable_rule_credit_assignment"],
        "enforce_generation_constraints": stage["enforce_generation_constraints"],
        "allow_candidate_repairs": stage["allow_candidate_repairs"],
    }


def generation_runtime_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build runtime room/dungeon generation defaults from the validated config payload."""
    stage = config["generation"]
    return {
        "default_guidance_scale": stage.get("guidance_scale", 3.0),
        "default_logic_guidance_scale": stage.get("logic_guidance_scale", 0.0),
        "default_num_diffusion_steps": stage.get("num_diffusion_steps", 50),
        "default_use_fast_sampling": stage.get("use_fast_sampling", False),
        "default_latent_sampler": stage.get("latent_sampler", "diffusion"),
        "default_categorical_codebook_size": stage.get("categorical_codebook_size", 256),
        "default_use_topological_positional_encoding": stage.get("use_topological_positional_encoding", True),
        "default_apply_repair": stage.get("apply_repair", True),
        "default_enable_map_elites": stage.get("enable_map_elites", False),
        "symbolic_max_repair_attempts": stage.get("symbolic_max_repair_attempts", 5),
        "symbolic_repair_margin": stage.get("symbolic_repair_margin", 2),
        "symbolic_adjacency_threshold": stage.get("symbolic_adjacency_threshold", 0.01),
        "default_start_goal_coords": (
            tuple(int(v) for v in stage.get("default_start_coord", (1, 5))),
            tuple(int(v) for v in stage.get("default_goal_coord", (14, 5))),
        ),
        "default_semantic_role_prior_strength": stage.get(
            "semantic_role_prior_strength",
            DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
        ),
        "default_semantic_anchor_threshold": stage.get("semantic_anchor_threshold", 0.5),
        "default_semantic_puzzle_offset": stage.get(
            "semantic_puzzle_offset",
            DEFAULT_SEMANTIC_PUZZLE_OFFSET,
        ),
        "default_semantic_constrained_decoding_enabled": stage.get(
            "semantic_constrained_decoding_enabled",
            True,
        ),
        "default_semantic_marker_logit_bias": stage.get(
            "semantic_marker_logit_bias",
            10000.0,
        ),
        "default_semantic_marker_suppression_bias": stage.get(
            "semantic_marker_suppression_bias",
            100.0,
        ),
        "default_puzzle_room_scaffold_enabled": stage.get(
            "puzzle_room_scaffold_enabled",
            True,
        ),
        "default_puzzle_room_structure_enabled": stage.get(
            "puzzle_room_structure_enabled",
            True,
        ),
        "default_puzzle_room_scaffold_min_structure_tiles": stage.get(
            "puzzle_room_scaffold_min_structure_tiles",
            10,
        ),
        "default_puzzle_room_archetype_mode": stage.get(
            "puzzle_room_archetype_mode",
            "auto",
        ),
        "default_puzzle_room_branch_density": stage.get(
            "puzzle_room_branch_density",
            0.75,
        ),
        "default_puzzle_room_block_budget": stage.get(
            "puzzle_room_block_budget",
            28,
        ),
        "default_puzzle_room_preserve_route_margin": stage.get(
            "puzzle_room_preserve_route_margin",
            0,
        ),
        "default_puzzle_room_switch_pocket_depth": stage.get(
            "puzzle_room_switch_pocket_depth",
            3,
        ),
        "default_puzzle_room_resource_bypass_offset": stage.get(
            "puzzle_room_resource_bypass_offset",
            2,
        ),
        "default_puzzle_room_key_pocket_depth": stage.get(
            "puzzle_room_key_pocket_depth",
            3,
        ),
        "default_puzzle_room_item_slot_depth": stage.get(
            "puzzle_room_item_slot_depth",
            3,
        ),
        "default_puzzle_room_toggle_corridor_offset": stage.get(
            "puzzle_room_toggle_corridor_offset",
            2,
        ),
        "default_puzzle_room_novelty_enabled": stage.get(
            "puzzle_room_novelty_enabled",
            True,
        ),
        "default_puzzle_room_candidate_count": stage.get(
            "puzzle_room_candidate_count",
            4,
        ),
        "default_puzzle_room_novelty_weight": stage.get(
            "puzzle_room_novelty_weight",
            0.45,
        ),
        "default_puzzle_room_min_quality_gain": stage.get(
            "puzzle_room_min_quality_gain",
            0.5,
        ),
        "default_validator_plan_max_states": stage.get(
            "validator_plan_max_states",
            DEFAULT_VALIDATOR_PLAN_MAX_STATES,
        ),
        "default_puzzle_stage_topology_enabled": stage.get(
            "puzzle_stage_topology_enabled",
            False,
        ),
        "default_puzzle_stage_trace_decay": stage.get(
            "puzzle_stage_trace_decay",
            DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
        ),
        "default_deterministic_graph_marker_overlay_enabled": stage.get(
            "deterministic_graph_marker_overlay_enabled",
            True,
        ),
        "default_fast_sampler_teacher_fallback_enabled": stage.get(
            "fast_sampler_teacher_fallback_enabled",
            True,
        ),
        "default_masked_room_teacher_fallback_enabled": stage.get(
            "masked_room_teacher_fallback_enabled",
            True,
        ),
        "default_masked_room_sampling_temperature": stage.get(
            "masked_room_sampling_temperature",
            1.0,
        ),
        "default_masked_room_sampling_schedule": stage.get(
            "masked_room_sampling_schedule",
            "cosine",
        ),
        "default_masked_room_sampling_stochastic": stage.get(
            "masked_room_sampling_stochastic",
            True,
        ),
        "default_masked_room_corrector_steps": stage.get(
            "masked_room_corrector_steps",
            1,
        ),
        "default_masked_room_corrector_mask_ratio": stage.get(
            "masked_room_corrector_mask_ratio",
            0.1,
        ),
    }


def pipeline_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build canonical pipeline constructor kwargs from the validated global config payload."""
    diffusion = config["diffusion"]
    fast_sampler = config["fast_sampler"]
    masked_room = config["masked_room"]
    topology_kwargs = topology_generation_kwargs_from_resolved_config(config)
    kwargs = {
        "topology_default_target_curve": list(topology_kwargs["target_curve"]),
        "topology_num_rooms": topology_kwargs["num_rooms"],
        "topology_population_size": topology_kwargs["population_size"],
        "topology_generations": topology_kwargs["generations"],
        "topology_mutation_rate": topology_kwargs["mutation_rate"],
        "topology_crossover_rate": topology_kwargs["crossover_rate"],
        "topology_genome_length": topology_kwargs["genome_length"],
        "topology_rule_space": topology_kwargs["rule_space"],
        "topology_transition_mix": topology_kwargs["transition_mix"],
        "topology_search_strategy": topology_kwargs["search_strategy"],
        "topology_qd_archive_cells": topology_kwargs["qd_archive_cells"],
        "topology_qd_init_random_fraction": topology_kwargs["qd_init_random_fraction"],
        "topology_qd_emitter_mutation_rate": topology_kwargs["qd_emitter_mutation_rate"],
        "topology_max_lock_key_rules": topology_kwargs["max_lock_key_rules"],
        "topology_enable_rule_credit_assignment": topology_kwargs["enable_rule_credit_assignment"],
        "topology_enforce_generation_constraints": topology_kwargs["enforce_generation_constraints"],
        "topology_allow_candidate_repairs": topology_kwargs["allow_candidate_repairs"],
    }
    kwargs.update(generation_runtime_kwargs_from_resolved_config(config))
    kwargs.update(
        {
            "condition_gnn_type": diffusion["condition_gnn_type"],
            "condition_use_reference_room_maps": diffusion["condition_use_reference_room_maps"],
            "condition_reference_tile_vocab_size": diffusion["condition_reference_tile_vocab_size"],
            "condition_reference_embedding_dim": diffusion["condition_reference_embedding_dim"],
            "condition_reference_hidden_dim": diffusion["condition_reference_hidden_dim"],
            "topology_refinement_mode": diffusion["topology_refinement_mode"],
            "diffusion_attention_mode": diffusion["attention_mode"],
            "diffusion_hedgehog_feature_dim": diffusion["hedgehog_feature_dim"],
            "diffusion_cfg_schedule_mode": diffusion["cfg_schedule_mode"],
            "diffusion_cfg_schedule_min_scale": diffusion["cfg_schedule_min_scale"],
            "diffusion_cfg_schedule_power": diffusion["cfg_schedule_power"],
            "use_current_node_distance_features": diffusion["use_current_node_distance_features"],
            "current_node_distance_max": diffusion["current_node_distance_max"],
            "masked_sampling_steps": masked_room["masked_steps"],
            "fast_sampling_steps": fast_sampler["num_inference_steps"],
            "condition_encoder_fallback_config": {
                "latent_dim": diffusion["latent_dim"],
                "condition_hidden_dim": diffusion["condition_hidden_dim"],
                "context_dim": diffusion["context_dim"],
                "condition_gnn_type": diffusion["condition_gnn_type"],
                "condition_num_gnn_layers": diffusion["condition_num_gnn_layers"],
                "condition_num_attention_heads": diffusion["condition_num_attention_heads"],
                "condition_dropout": diffusion["condition_dropout"],
                "use_current_node_distance_features": diffusion["use_current_node_distance_features"],
                "condition_use_reference_room_maps": diffusion["condition_use_reference_room_maps"],
                "condition_reference_tile_vocab_size": diffusion["condition_reference_tile_vocab_size"],
                "condition_reference_embedding_dim": diffusion["condition_reference_embedding_dim"],
                "condition_reference_hidden_dim": diffusion["condition_reference_hidden_dim"],
            },
            "diffusion_fallback_config": {
                "latent_dim": diffusion["latent_dim"],
                "context_dim": diffusion["context_dim"],
                "num_timesteps": diffusion["num_timesteps"],
                "prediction_type": diffusion["prediction_type"],
                "cfg_dropout_prob": diffusion["cfg_dropout_prob"],
                "cfg_scale": diffusion["cfg_scale"],
                "min_snr_gamma": diffusion["min_snr_gamma"],
                "model_channels": diffusion["model_channels"],
                "topology_conditioning_mode": diffusion["topology_conditioning_mode"],
                "unet_channel_mult": list(diffusion["unet_channel_mult"]),
                "unet_num_res_blocks": diffusion["unet_num_res_blocks"],
                "unet_attention_resolutions": list(diffusion["unet_attention_resolutions"]),
                "unet_num_heads": diffusion["unet_num_heads"],
                "unet_dropout": diffusion["unet_dropout"],
                "graph_auto_linear_attention_nodes": diffusion["graph_auto_linear_attention_nodes"],
                "spatial_graph_gate_init": diffusion["spatial_graph_gate_init"],
                "spatial_topology_gate_init": diffusion["spatial_topology_gate_init"],
                "room_topology_channels": diffusion["room_topology_channels"],
                "puzzle_structure_dropout_prob": diffusion.get("puzzle_structure_dropout_prob", 0.0),
            },
            "logic_net_fallback_config": {
                "latent_dim": diffusion["latent_dim"],
                "num_classes": config["dataset"]["num_classes"],
                "num_logic_iterations": diffusion["num_logic_iterations"],
                "logic_topology_trace_weight": diffusion["logic_topology_trace_weight"],
                "logic_topology_anchor_weight": diffusion["logic_topology_anchor_weight"],
                "logic_global_reach_weight": diffusion.get("logic_global_reach_weight", 1.0),
                "logic_global_room_weight": diffusion.get("logic_global_room_weight", 0.25),
            },
            "masked_room_fallback_config": {
                "num_classes": config["dataset"]["num_classes"],
                "hidden_dim": masked_room["hidden_dim"],
                "model_channels": masked_room["model_channels"],
                "context_dim": masked_room["context_dim"],
                "topology_conditioning_mode": masked_room["topology_conditioning_mode"],
                "graph_auto_linear_attention_nodes": masked_room["graph_auto_linear_attention_nodes"],
                "spatial_graph_gate_init": masked_room["spatial_graph_gate_init"],
                "spatial_topology_gate_init": masked_room["spatial_topology_gate_init"],
                "unet_channel_mult": list(masked_room["unet_channel_mult"]),
                "unet_num_res_blocks": masked_room["unet_num_res_blocks"],
                "unet_attention_resolutions": list(masked_room["unet_attention_resolutions"]),
                "unet_num_heads": masked_room["unet_num_heads"],
                "unet_dropout": masked_room["unet_dropout"],
                "room_topology_channels": masked_room["room_topology_channels"],
                "puzzle_structure_dropout_prob": masked_room.get("puzzle_structure_dropout_prob", 0.0),
            },
        }
    )
    return kwargs


def create_pipeline(
    checkpoint_dir: str = "./checkpoints",
    device: str = 'auto',
    **kwargs
) -> NeuralSymbolicDungeonPipeline:
    """
    Create pipeline with checkpoints from directory.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        device: Device to run on
        **kwargs: Additional arguments for pipeline
        
    Returns:
        Initialized pipeline
    """
    checkpoint_dir = Path(checkpoint_dir)
    resolved_config = kwargs.pop("resolved_config", None)
    if resolved_config is None:
        try:
            from src.config_system import load_resolved_config_for_artifact

            resolved_config = load_resolved_config_for_artifact(checkpoint_dir)
        except (ImportError, RuntimeError, ValueError, TypeError):
            resolved_config = None
    pipeline_kwargs: Dict[str, Any] = {}
    if isinstance(resolved_config, dict):
        pipeline_kwargs.update(pipeline_kwargs_from_resolved_config(resolved_config))
    pipeline_kwargs.update(kwargs)

    return NeuralSymbolicDungeonPipeline(
        vqvae_checkpoint=str(checkpoint_dir / "vqvae_best.pth"),
        diffusion_checkpoint=str(checkpoint_dir / "diffusion_best.pth"),
        logic_net_checkpoint=str(checkpoint_dir / "logic_net_best.pth"),
        condition_encoder_checkpoint=str(checkpoint_dir / "condition_encoder_best.pth"),
        device=device,
        **pipeline_kwargs
    )


__all__ = [
    'NeuralSymbolicDungeonPipeline',
    'MissingPipelineComponentError',
    'NeuralGenerationComponents',
    'SymbolicGenerationComponents',
    'PipelineComponents',
    'PipelineComponentFactory',
    'RoomGenerationResult',
    'DungeonGenerationResult',
    'PreparedDungeonGeneration',
    'GeneratedRoomSet',
    'topology_generation_kwargs_from_resolved_config',
    'generation_runtime_kwargs_from_resolved_config',
    'pipeline_kwargs_from_resolved_config',
    'create_pipeline',
]
