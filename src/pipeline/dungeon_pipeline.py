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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
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
    DEFAULT_VALIDATOR_PLAN_MAX_STATES,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    TOPOLOGY_ANCHOR_POLICY_VERSION,
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
        default_validator_plan_max_states: int = DEFAULT_VALIDATOR_PLAN_MAX_STATES,
        default_deterministic_graph_marker_overlay_enabled: bool = True,
        default_fast_sampler_teacher_fallback_enabled: bool = True,
        default_masked_room_teacher_fallback_enabled: bool = True,
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
        self.default_validator_plan_max_states = int(max(32, int(default_validator_plan_max_states)))
        self.default_deterministic_graph_marker_overlay_enabled = bool(
            default_deterministic_graph_marker_overlay_enabled
        )
        self.default_fast_sampler_teacher_fallback_enabled = bool(default_fast_sampler_teacher_fallback_enabled)
        self.default_masked_room_teacher_fallback_enabled = bool(default_masked_room_teacher_fallback_enabled)
        self.topology_anchor_policy_version = TOPOLOGY_ANCHOR_POLICY_VERSION
        self.condition_encoder_fallback_config = dict(condition_encoder_fallback_config or {})
        self.diffusion_fallback_config = dict(diffusion_fallback_config or {})
        self.logic_net_fallback_config = dict(logic_net_fallback_config or {})
        self.masked_room_fallback_config = dict(masked_room_fallback_config or {})
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
    ) -> Tuple[np.ndarray, int, List[int]]:
        """
        Clamp/repair semantic IDs to the canonical palette.

        Invalid tile IDs are replaced with fallback_grid values when available;
        otherwise they are replaced with FLOOR.
        """
        out = np.asarray(grid, dtype=np.int32).copy()
        invalid_mask = ~np.isin(out, self._valid_semantic_tile_ids_np)
        invalid_count = int(np.sum(invalid_mask))
        invalid_ids: List[int] = []
        if invalid_count <= 0:
            return out, 0, invalid_ids

        invalid_ids = [int(v) for v in np.unique(out[invalid_mask])]
        floor_id = int(SEMANTIC_PALETTE.get("FLOOR", 1))
        if fallback_grid is not None and np.shape(fallback_grid) == np.shape(out):
            fb = np.asarray(fallback_grid, dtype=np.int32)
            fb_invalid = ~np.isin(fb, self._valid_semantic_tile_ids_np)
            if bool(np.any(fb_invalid)):
                fb = fb.copy()
                fb[fb_invalid] = floor_id
            out[invalid_mask] = fb[invalid_mask]
        else:
            out[invalid_mask] = floor_id
        return out, invalid_count, invalid_ids

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
            preview_grid, _, _ = self._sanitize_semantic_grid(preview_grid)
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
        stateful_anchor: Optional[Tuple[int, int]],
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

        def _pick_side_lane(anchor_value: int, reference_value: int, *, low: int, high: int, offset: int) -> int:
            anchor_value = int(anchor_value)
            reference_value = int(reference_value)
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
                pocket_row = max(2, min(ROOM_HEIGHT - 3, int(stateful[0])))
                entry = (source[0], max(2, gate_col - 2))
                if gate_family == "switch":
                    pocket = (pocket_row, max(2, gate_col - switch_depth))
                    gate_open = (pocket_row, gate_col)
                    exit_point = (pocket_row, min(ROOM_WIDTH - 3, gate_col + 2))
                elif gate_family == "toggle":
                    toggle_row = max(
                        2,
                        min(
                            ROOM_HEIGHT - 3,
                            int(stateful[0]) + (-toggle_offset if stateful[0] > ROOM_HEIGHT // 2 else toggle_offset),
                        ),
                    )
                    pocket = (toggle_row, max(2, gate_col - 2))
                    gate_open = (stateful[0], gate_col)
                    exit_point = (stateful[0], min(ROOM_WIDTH - 3, gate_col + 2))
                elif gate_family == "bombable":
                    bypass_row = _pick_side_lane(
                        int(stateful[0]),
                        int(center[0]),
                        low=2,
                        high=ROOM_HEIGHT - 3,
                        offset=resource_offset,
                    )
                    pocket = (bypass_row, max(2, gate_col - (resource_offset + 1)))
                    gate_open = (bypass_row, gate_col)
                    exit_point = (bypass_row, min(ROOM_WIDTH - 3, gate_col + 2))
                elif gate_family == "item_unlock":
                    item_row = max(2, min(ROOM_HEIGHT - 3, int(stateful[0])))
                    item_col = max(2, min(ROOM_WIDTH - 3, int(stateful[1])))
                    pocket = (item_row, item_col)
                    gate_open = (center[0], gate_col)
                    exit_point = (center[0], max(min(ROOM_WIDTH - 3, item_col), min(ROOM_WIDTH - 3, gate_col + 2)))
                elif gate_family == "key" and stateful_anchor is not None:
                    key_row = max(2, min(ROOM_HEIGHT - 3, int(stateful[0])))
                    pocket = (key_row, max(2, gate_col - key_depth))
                    gate_open = (center[0], gate_col)
                    exit_point = (center[0], min(ROOM_WIDTH - 3, gate_col + 2))
                else:
                    pocket = (pocket_row, max(2, gate_col - 2))
                    gate_open = (pocket_row, gate_col)
                    exit_point = (pocket_row, min(ROOM_WIDTH - 3, gate_col + 2))
                destination_hook = (destination[0], int(exit_point[1]))
                _add_polyline([source, entry, pocket, gate_open, exit_point, destination_hook, destination])
            else:
                gate_row = max(3, min(ROOM_HEIGHT - 4, int(puzzle[0])))
                pocket_col = max(2, min(ROOM_WIDTH - 3, int(stateful[1])))
                entry = (max(2, gate_row - 2), source[1])
                if gate_family == "switch":
                    pocket = (max(2, gate_row - switch_depth), pocket_col)
                    gate_open = (gate_row, pocket_col)
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + 2), pocket_col)
                elif gate_family == "toggle":
                    toggle_col = max(
                        2,
                        min(
                            ROOM_WIDTH - 3,
                            int(stateful[1]) + (-toggle_offset if stateful[1] > ROOM_WIDTH // 2 else toggle_offset),
                        ),
                    )
                    pocket = (max(2, gate_row - 2), toggle_col)
                    gate_open = (gate_row, stateful[1])
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + 2), stateful[1])
                elif gate_family == "bombable":
                    bypass_col = _pick_side_lane(
                        int(stateful[1]),
                        int(center[1]),
                        low=2,
                        high=ROOM_WIDTH - 3,
                        offset=resource_offset,
                    )
                    pocket = (max(2, gate_row - (resource_offset + 1)), bypass_col)
                    gate_open = (gate_row, bypass_col)
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + 2), bypass_col)
                elif gate_family == "item_unlock":
                    item_row = max(2, min(ROOM_HEIGHT - 3, int(stateful[0])))
                    item_col = max(2, min(ROOM_WIDTH - 3, int(stateful[1])))
                    pocket = (item_row, item_col)
                    gate_open = (gate_row, center[1])
                    exit_point = (max(min(ROOM_HEIGHT - 3, item_row), min(ROOM_HEIGHT - 3, gate_row + 2)), center[1])
                elif gate_family == "key" and stateful_anchor is not None:
                    key_col = max(2, min(ROOM_WIDTH - 3, int(stateful[1])))
                    pocket = (max(2, gate_row - key_depth), key_col)
                    gate_open = (gate_row, center[1])
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + 2), center[1])
                else:
                    pocket = (max(2, gate_row - 2), pocket_col)
                    gate_open = (gate_row, pocket_col)
                    exit_point = (min(ROOM_HEIGHT - 3, gate_row + 2), pocket_col)
                destination_hook = (int(exit_point[0]), destination[1])
                _add_polyline([source, entry, pocket, gate_open, exit_point, destination_hook, destination])
        elif archetype == "hub":
            hub = self._clamp_room_coord(
                (
                    int(round((puzzle[0] + center[0]) / 2.0)),
                    int(round((puzzle[1] + center[1]) / 2.0)),
                )
            )
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
            mask[max(1, hub[0] - 1): min(ROOM_HEIGHT - 1, hub[0] + 2), max(1, hub[1] - 1): min(ROOM_WIDTH - 1, hub[1] + 2)] = True
        elif archetype == "combat":
            arena_center = self._clamp_room_coord(
                (
                    int(round((source[0] + destination[0] + puzzle[0]) / 3.0)),
                    int(round((source[1] + destination[1] + puzzle[1]) / 3.0)),
                )
            )
            _add_polyline([source, arena_center, destination])
            self._paint_room_line_mask(mask, arena_center, puzzle)
            mask[max(1, arena_center[0] - 1): min(ROOM_HEIGHT - 1, arena_center[0] + 2), max(1, arena_center[1] - 1): min(ROOM_WIDTH - 1, arena_center[1] + 2)] = True
        elif archetype == "island":
            waypoint = self._clamp_room_coord(
                (
                    int(round((puzzle[0] + destination[0]) / 2.0)),
                    int(round((puzzle[1] + destination[1]) / 2.0)),
                )
            )
            _add_polyline([source, puzzle, waypoint, destination])
            mask[max(1, puzzle[0] - 1): min(ROOM_HEIGHT - 1, puzzle[0] + 2), max(1, puzzle[1] - 1): min(ROOM_WIDTH - 1, puzzle[1] + 2)] = True
        else:  # serpentine
            if flow_is_horizontal:
                waypoints = [
                    source,
                    (2, 3),
                    (4, 3),
                    (4, ROOM_WIDTH - 4),
                    (8, ROOM_WIDTH - 4),
                    (8, 3),
                    (12, 3),
                    (12, ROOM_WIDTH - 4),
                    destination,
                ]
            else:
                waypoints = [
                    source,
                    (3, 2),
                    (3, 4),
                    (ROOM_HEIGHT - 4, 4),
                    (ROOM_HEIGHT - 4, 6),
                    (3, 6),
                    (3, ROOM_WIDTH - 3),
                    destination,
                ]
            _add_polyline(waypoints)
            self._paint_room_line_mask(mask, puzzle, waypoints[min(len(waypoints) - 2, 3)])

        if role_flags.get("has_puzzle", False):
            mask[max(1, puzzle[0] - 1): min(ROOM_HEIGHT - 1, puzzle[0] + 2), max(1, puzzle[1] - 1): min(ROOM_WIDTH - 1, puzzle[1] + 2)] = True

        return mask

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

    def _build_puzzle_room_segments(
        self,
        *,
        archetype: str,
        gate_family: str,
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
                pocket_side = -1 if center_r <= ROOM_HEIGHT // 2 else 1
                pocket_row = max(2, min(ROOM_HEIGHT - 3, center_r + pocket_side * 2))
                if gate_family == "switch":
                    required.append([(stateful_r, col) for col in range(max(2, gate_col - 3), min(ROOM_WIDTH - 2, gate_col + 1))])
                    required.append([(row, max(2, gate_col - 3)) for row in range(min(center_r, stateful_r), max(center_r, stateful_r) + 1)])
                    optional.append([(row, min(ROOM_WIDTH - 3, gate_col + 2)) for row in range(3, ROOM_HEIGHT - 3) if abs(row - center_r) > 1])
                elif gate_family == "toggle":
                    corridor_top = max(2, stateful_r - toggle_offset)
                    corridor_bottom = min(ROOM_HEIGHT - 3, stateful_r + toggle_offset)
                    required.append([(corridor_top, col) for col in range(max(2, gate_col - 3), min(ROOM_WIDTH - 2, gate_col + 2))])
                    required.append([(corridor_bottom, col) for col in range(max(2, gate_col - 3), min(ROOM_WIDTH - 2, gate_col + 2))])
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
                    optional.append([(center_r - 1, col) for col in range(min(gate_col + 1, item_col), max(gate_col + 1, item_col) + 1)])
                    optional.append([(center_r + 1, col) for col in range(min(gate_col + 1, item_col), max(gate_col + 1, item_col) + 1)])
                elif gate_family == "key" and stateful_anchor is not None:
                    key_row = stateful_r
                    required.append([(key_row, col) for col in range(max(2, gate_col - key_depth - 1), max(3, gate_col - 1))])
                    required.append([(row, max(2, gate_col - key_depth - 1)) for row in range(min(center_r, key_row), max(center_r, key_row) + 1)])
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
                pocket_side = -1 if center_c <= ROOM_WIDTH // 2 else 1
                pocket_col = max(2, min(ROOM_WIDTH - 3, center_c + pocket_side * 2))
                if gate_family == "switch":
                    required.append([(row, stateful_c) for row in range(max(2, gate_row - 3), min(ROOM_HEIGHT - 2, gate_row + 1))])
                    required.append([(max(2, gate_row - 3), col) for col in range(min(center_c, stateful_c), max(center_c, stateful_c) + 1)])
                    optional.append([(min(ROOM_HEIGHT - 3, gate_row + 2), col) for col in range(3, ROOM_WIDTH - 3) if abs(col - center_c) > 1])
                elif gate_family == "toggle":
                    corridor_left = max(2, stateful_c - toggle_offset)
                    corridor_right = min(ROOM_WIDTH - 3, stateful_c + toggle_offset)
                    required.append([(row, corridor_left) for row in range(max(2, gate_row - 3), min(ROOM_HEIGHT - 2, gate_row + 2))])
                    required.append([(row, corridor_right) for row in range(max(2, gate_row - 3), min(ROOM_HEIGHT - 2, gate_row + 2))])
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
                    required.append([(row, resource_col) for row in range(max(2, gate_row - 4), max(3, gate_row - 1))])
                    required.append([(max(2, gate_row - 4), col) for col in range(min(resource_col, bypass_col), max(resource_col, bypass_col) + 1)])
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
                    optional.append([(row, center_c - 1) for row in range(min(gate_row + 1, item_row), max(gate_row + 1, item_row) + 1)])
                    optional.append([(row, center_c + 1) for row in range(min(gate_row + 1, item_row), max(gate_row + 1, item_row) + 1)])
                elif gate_family == "key" and stateful_anchor is not None:
                    key_col = stateful_c
                    required.append([(row, key_col) for row in range(max(2, gate_row - key_depth - 1), max(3, gate_row - 1))])
                    required.append([(max(2, gate_row - key_depth - 1), col) for col in range(min(center_c, key_col), max(center_c, key_col) + 1)])
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
            optional.append([(center_r - 2, col) for col in range(left + 1, center_c - 1)])
            optional.append([(center_r - 2, col) for col in range(center_c + 2, right)])
            optional.append([(center_r + 2, col) for col in range(left + 1, center_c - 1)])
            optional.append([(center_r + 2, col) for col in range(center_c + 2, right)])
        elif archetype == "island":
            optional.extend(
                [
                    [(center_r - 3, center_c - 2), (center_r - 3, center_c - 1), (center_r - 2, center_c - 2), (center_r - 2, center_c - 1)],
                    [(center_r - 1, center_c + 1), (center_r - 1, center_c + 2), (center_r, center_c + 1), (center_r, center_c + 2)],
                    [(center_r + 2, center_c - 2), (center_r + 2, center_c - 1), (center_r + 3, center_c - 2), (center_r + 3, center_c - 1)],
                    [(center_r + 1, center_c + 1), (center_r + 1, center_c + 2), (center_r + 2, center_c + 1), (center_r + 2, center_c + 2)],
                ]
            )
            if flow_is_horizontal:
                required.append([(center_r, col) for col in range(left_col + 1, right_col) if abs(col - center_c) > 2])
            else:
                required.append([(row, center_c) for row in range(top_row + 1, bottom_row) if abs(row - center_r) > 2])
        elif archetype == "combat":
            optional.extend(
                [
                    [(center_r - 3, center_c - 2), (center_r - 3, center_c - 1), (center_r - 2, center_c - 2), (center_r - 2, center_c - 1)],
                    [(center_r - 3, center_c + 1), (center_r - 3, center_c + 2), (center_r - 2, center_c + 1), (center_r - 2, center_c + 2)],
                    [(center_r + 2, center_c - 2), (center_r + 2, center_c - 1), (center_r + 3, center_c - 2), (center_r + 3, center_c - 1)],
                    [(center_r + 2, center_c + 1), (center_r + 2, center_c + 2), (center_r + 3, center_c + 1), (center_r + 3, center_c + 2)],
                ]
            )
            required.append([(center_r, center_c - 3), (center_r, center_c - 2)])
            required.append([(center_r, center_c + 2), (center_r, center_c + 3)])
        else:  # serpentine
            if flow_is_horizontal:
                rows = [3, 6, 9, 12]
                for idx, row in enumerate(rows):
                    if row >= ROOM_HEIGHT - 2:
                        continue
                    gap_on_left = (idx % 2) == 0
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
                optional.append([(center_r, center_c - 1), (center_r, center_c + 1)])
            else:
                cols = [3, 5, 7]
                for idx, col in enumerate(cols):
                    if col >= ROOM_WIDTH - 2:
                        continue
                    gap_on_top = (idx % 2) == 0
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
                optional.append([(center_r - 1, center_c), (center_r + 1, center_c)])

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
        reserved = self._dilate_room_mask(route_mask, radius=preserve_margin) if preserve_margin > 0 else route_mask.copy()

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
        for point in semantic_anchors.values():
            rr, cc = self._clamp_room_coord(point)
            reserved[int(rr), int(cc)] = True

        planned_markers = self._plan_room_graph_marker_layout(
            out,
            graph=graph,
            room_id=room_id,
            start_goal=(start_coord, goal_coord),
        )
        for _tile_id, slot in planned_markers:
            rr, cc = self._clamp_room_coord(slot)
            reserved[int(rr), int(cc)] = True

        for direction, enabled in semantics["required_doors"].items():
            if not bool(enabled):
                continue
            spec = DOOR_POSITIONS[str(direction)]
            if direction in {"N", "S"}:
                apron_row = 2 if direction == "N" else ROOM_HEIGHT - 3
                c0 = int(spec["col_start"])
                c1 = int(spec["col_end"]) + 1
                reserved[apron_row, c0:c1] = True
            else:
                apron_col = 2 if direction == "W" else ROOM_WIDTH - 3
                r0 = int(spec["row_start"])
                r1 = int(spec["row_end"]) + 1
                reserved[r0:r1, apron_col] = True

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
        route_template = self._build_puzzle_room_route_template(
            archetype=archetype,
            gate_family=gate_family,
            stateful_anchor=stateful_anchor,
            flow_is_horizontal=flow_is_horizontal,
            source_anchor=source_anchor,
            destination_anchor=destination_anchor,
            puzzle_anchor=puzzle_anchor,
            role_flags=role_flags,
            semantics=semantics,
        )
        if bool(np.any(route_template)):
            route_mask = route_template
            reserved = self._dilate_room_mask(route_mask, radius=preserve_margin) if preserve_margin > 0 else route_mask.copy()
            for point in semantic_anchors.values():
                rr, cc = self._clamp_room_coord(point)
                reserved[int(rr), int(cc)] = True
            for _tile_id, slot in planned_markers:
                rr, cc = self._clamp_room_coord(slot)
                reserved[int(rr), int(cc)] = True
            for direction, enabled in semantics["required_doors"].items():
                if not bool(enabled):
                    continue
                spec = DOOR_POSITIONS[str(direction)]
                if direction in {"N", "S"}:
                    apron_row = 2 if direction == "N" else ROOM_HEIGHT - 3
                    c0 = int(spec["col_start"])
                    c1 = int(spec["col_end"]) + 1
                    reserved[apron_row, c0:c1] = True
                else:
                    apron_col = 2 if direction == "W" else ROOM_WIDTH - 3
                    r0 = int(spec["row_start"])
                    r1 = int(spec["row_end"]) + 1
                    reserved[r0:r1, apron_col] = True
            stats["planned_route_pixels"] = int(np.sum(route_mask))
            stats["route_template_used"] = 1
        else:
            stats["route_template_used"] = 0
        stats["archetype"] = str(archetype)
        stats["gate_family"] = str(gate_family)
        stats["stateful_anchor_name"] = str(stateful_anchor_name or "")
        stats["profile_branch_density"] = float(scaffold_profile.get("branch_density", getattr(self, "default_puzzle_room_branch_density", 0.75)))
        stats["profile_block_budget"] = int(scaffold_profile.get("block_budget", getattr(self, "default_puzzle_room_block_budget", 28)))
        stats["profile_preserve_route_margin"] = int(preserve_margin)

        def _can_place(row: int, col: int) -> bool:
            if not (2 <= int(row) <= ROOM_HEIGHT - 3 and 2 <= int(col) <= ROOM_WIDTH - 3):
                return False
            if bool(reserved[int(row), int(col)]):
                return False
            return int(out[int(row), int(col)]) == floor_id

        budget_remaining = int(max(0, scaffold_profile.get("block_budget", getattr(self, "default_puzzle_room_block_budget", 28))))

        def _paint_block_line(points: List[Tuple[int, int]]) -> int:
            nonlocal budget_remaining
            added = 0
            for row, col in points:
                if budget_remaining <= 0:
                    break
                if _can_place(int(row), int(col)):
                    out[int(row), int(col)] = block_id
                    added += 1
                    budget_remaining -= 1
            return int(added)

        segments_added = 0
        tiles_added = 0
        required_segments, optional_segments = self._build_puzzle_room_segments(
            archetype=archetype,
            gate_family=gate_family,
            stateful_anchor=stateful_anchor,
            flow_is_horizontal=flow_is_horizontal,
            puzzle_anchor=puzzle_anchor,
        )

        for segment in required_segments:
            added = _paint_block_line(segment)
            if added > 0:
                segments_added += 1
                tiles_added += added

        branch_density = float(max(0.0, min(1.0, scaffold_profile.get("branch_density", getattr(self, "default_puzzle_room_branch_density", 0.75)))))
        optional_quota = int(round(branch_density * len(optional_segments)))
        if branch_density > 0.0 and optional_segments and optional_quota <= 0:
            optional_quota = 1
        optional_quota = min(len(optional_segments), max(0, optional_quota))
        stats["optional_segments_requested"] = int(optional_quota)
        optional_segments_applied = 0
        for segment in optional_segments[:optional_quota]:
            added = _paint_block_line(segment)
            if added > 0:
                segments_added += 1
                optional_segments_applied += 1
                tiles_added += added

        stats["applied"] = int(tiles_added > 0)
        stats["tiles_added"] = int(tiles_added)
        stats["segments_added"] = int(segments_added)
        stats["optional_segments_applied"] = int(optional_segments_applied)
        return out, stats

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
        preferred_positions: Dict[int, Tuple[int, int]] = {
            int(TileID.START): self._clamp_room_coord(semantic_anchors.get("start", start_coord)),
            int(TileID.TRIFORCE): self._clamp_room_coord(semantic_anchors.get("goal", goal_coord)),
            int(TileID.BOSS): self._clamp_room_coord(semantic_anchors.get("boss", (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))),
            int(TileID.ENEMY): self._clamp_room_coord(semantic_anchors.get("enemy", (ROOM_HEIGHT // 2 - 2, ROOM_WIDTH // 2))),
            int(TileID.KEY_SMALL): self._clamp_room_coord(semantic_anchors.get("key", (ROOM_HEIGHT // 2, max(1, ROOM_WIDTH // 2 - 2)))),
            int(TileID.KEY_BOSS): self._clamp_room_coord(semantic_anchors.get("key", (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))),
            int(TileID.KEY_ITEM): self._clamp_room_coord(semantic_anchors.get("item", (ROOM_HEIGHT // 2, min(ROOM_WIDTH - 2, ROOM_WIDTH // 2 + 2)))),
            int(TileID.ITEM_MINOR): self._clamp_room_coord(semantic_anchors.get("item", (min(ROOM_HEIGHT - 2, ROOM_HEIGHT // 2 + 2), ROOM_WIDTH // 2))),
            int(TileID.STAIR): self._clamp_room_coord(semantic_anchors.get("item", (min(ROOM_HEIGHT - 2, ROOM_HEIGHT // 2 + 2), ROOM_WIDTH // 2))),
            int(TileID.PUZZLE): self._clamp_room_coord(semantic_anchors.get("puzzle", (max(1, ROOM_HEIGHT // 2 - 2), ROOM_WIDTH // 2))),
        }

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
        preferred_positions: Dict[int, Tuple[int, int]] = {
            int(TileID.START): self._clamp_room_coord(semantic_anchors.get("start", start_coord)),
            int(TileID.TRIFORCE): self._clamp_room_coord(semantic_anchors.get("goal", goal_coord)),
            int(TileID.BOSS): self._clamp_room_coord(semantic_anchors.get("boss", (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))),
            int(TileID.ENEMY): self._clamp_room_coord(semantic_anchors.get("enemy", (ROOM_HEIGHT // 2 - 2, ROOM_WIDTH // 2))),
            int(TileID.KEY_SMALL): self._clamp_room_coord(semantic_anchors.get("key", (ROOM_HEIGHT // 2, max(1, ROOM_WIDTH // 2 - 2)))),
            int(TileID.KEY_BOSS): self._clamp_room_coord(semantic_anchors.get("key", (ROOM_HEIGHT // 2, ROOM_WIDTH // 2))),
            int(TileID.KEY_ITEM): self._clamp_room_coord(semantic_anchors.get("item", (ROOM_HEIGHT // 2, min(ROOM_WIDTH - 2, ROOM_WIDTH // 2 + 2)))),
            int(TileID.ITEM_MINOR): self._clamp_room_coord(semantic_anchors.get("item", (min(ROOM_HEIGHT - 2, ROOM_HEIGHT // 2 + 2), ROOM_WIDTH // 2))),
            int(TileID.STAIR): self._clamp_room_coord(semantic_anchors.get("item", (min(ROOM_HEIGHT - 2, ROOM_HEIGHT // 2 + 2), ROOM_WIDTH // 2))),
            int(TileID.PUZZLE): self._clamp_room_coord(semantic_anchors.get("puzzle", (max(1, ROOM_HEIGHT // 2 - 2), ROOM_WIDTH // 2))),
        }

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

        if latent.device == target_device and latent.dtype == target_dtype:
            return latent
        return latent.to(device=target_device, dtype=target_dtype)

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

        graph_ctx_for_guidance = {
            'node_features': graph_data.get('node_features'),
            'edge_index': graph_data.get('edge_index'),
            'edge_features': graph_data.get('edge_features'),
            'tpe': graph_data.get('tpe'),
            'node_positions': graph_data.get('node_positions'),
            'node_mask': graph_data.get('node_mask'),
            'boundary_constraints': torch.cat(
                [inp['boundary_constraints'].to(self.device, dtype=torch.float32) for inp in per_room_inputs],
                dim=0,
            ),
            'room_topology_map': torch.cat(
                [inp['graph_context']['room_topology_map'] for inp in per_room_inputs],
                dim=0,
            ),
        }
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
                            z_batch[i:i + 1] = self.diffusion.inpaint(
                                x_0=z_ref,
                                mask=boundary_edit_mask,
                                context=condition_batch[i:i + 1],
                                graph_data=graph_ctx_for_guidance,
                                num_steps=max(8, int(num_diffusion_steps) // 2),
                            )
                    except (AttributeError, RuntimeError, ValueError, TypeError):
                        continue
            logits_batch = self.vqvae.decode(self._cast_latent_for_vqvae_decode(z_batch))

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
            logits = self.vqvae.decode(self._cast_latent_for_vqvae_decode(z_latent))  # (1, 44, 16, 11)
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
        neural_grid, neural_invalid_count, neural_invalid_ids = self._sanitize_semantic_grid(neural_grid)
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
            self._bump_diagnostic("puzzle_room_scaffold_applied")
            puzzle_archetype = str(final_puzzle_scaffold.get("archetype", "")).strip().lower()
            if puzzle_archetype:
                self._bump_diagnostic(f"puzzle_room_scaffold_{puzzle_archetype}")
            puzzle_gate_family = str(final_puzzle_scaffold.get("gate_family", "")).strip().lower()
            if puzzle_gate_family:
                self._bump_diagnostic(f"puzzle_room_scaffold_gate_{puzzle_gate_family}")
            logger.debug(
                "Room %s applied puzzle scaffold: %s",
                room_id,
                final_puzzle_scaffold,
            )

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
            'neural_boundary_wall_tiles_forced': int(neural_boundary_shell['boundary_wall_tiles_forced']),
            'neural_boundary_door_tiles_forced': int(neural_boundary_shell['boundary_door_tiles_forced']),
            'neural_interior_door_apron_tiles_forced': int(neural_boundary_shell['interior_door_apron_tiles_forced']),
            'repair_invalid_door_tiles_removed': int(repaired_structural_cleanup['invalid_door_tiles_removed']),
            'repair_interior_obstacle_tiles_removed': int(repaired_structural_cleanup['interior_obstacle_tiles_removed']),
            'repair_interior_obstacle_components_removed': int(repaired_structural_cleanup['interior_obstacle_components_removed']),
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
            'final_puzzle_scaffold_applied': int(final_puzzle_scaffold['applied']),
            'final_puzzle_scaffold_tiles_added': int(final_puzzle_scaffold['tiles_added']),
            'final_puzzle_scaffold_segments_added': int(final_puzzle_scaffold['segments_added']),
            'final_puzzle_scaffold_optional_segments_requested': int(final_puzzle_scaffold.get('optional_segments_requested', 0)),
            'final_puzzle_scaffold_optional_segments_applied': int(final_puzzle_scaffold.get('optional_segments_applied', 0)),
            'final_puzzle_scaffold_route_template_used': int(final_puzzle_scaffold.get('route_template_used', 0)),
            'final_puzzle_scaffold_noise_components_removed': int(final_puzzle_scaffold.get('noise_components_removed', 0)),
            'final_puzzle_scaffold_noise_tiles_removed': int(final_puzzle_scaffold.get('noise_tiles_removed', 0)),
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
    ) -> Dict[str, float]:
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
        result: Dict[str, float] = {
            'solvability_score': 0.0,
            'graph_reach_loss': 0.0,
            'lock_loss': 0.0,
        }
        failing_rooms: List[Any] = []
        
        if self.logic_net is None:
            logger.debug("evaluate_dungeon_solvability skipped: no logic_net component")
            return result
        
        # Evaluate per-room walkability via LogicNet (grid-level only)
        total_grid_reach = 0.0
        num_rooms = 0
        for room_id, room_result in rooms.items():
            if room_result.latent is None:
                continue
            z = room_result.latent.to(self.device)
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
            result['solvability_score'] = total_grid_reach / num_rooms
        
        result['failing_rooms'] = failing_rooms  # type: ignore[assignment]
        result['num_rooms_evaluated'] = float(num_rooms)
        result['num_failing'] = float(len(failing_rooms))
        
        logger.info(
            "Dungeon solvability: %.3f (%d/%d rooms passing)",
            result['solvability_score'],
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

        dungeon_grid = self.stitch_rooms(room_set.rooms, prepared.mission_graph_physical)
        map_elites_score = self.evaluate_generated_dungeon(
            dungeon_grid,
            prepared.mission_graph_physical,
            enable_map_elites=enable_map_elites,
        )
        
        # Compute overall metrics
        generation_time = time.time() - start_time
        num_rooms_generated = len(room_set.rooms)
        room_metric_dicts = [dict(r.metrics) for r in room_set.rooms.values()]
        total_graph_marker_expected = float(sum(m.get("final_pre_overlay_graph_marker_expected", 0.0) for m in room_metric_dicts))
        total_graph_marker_overwrites = float(sum(m.get("final_graph_marker_overwrites", 0.0) for m in room_metric_dicts))
        avg_neural_graph_marker_exact_match_rate = (
            float(np.mean([m.get("neural_graph_marker_exact_match_rate", 1.0) for m in room_metric_dicts]))
            if room_metric_dicts
            else 1.0
        )
        avg_final_pre_overlay_graph_marker_exact_match_rate = (
            float(np.mean([m.get("final_pre_overlay_graph_marker_exact_match_rate", 1.0) for m in room_metric_dicts]))
            if room_metric_dicts
            else 1.0
        )
        avg_final_post_overlay_graph_marker_exact_match_rate = (
            float(np.mean([m.get("final_post_overlay_graph_marker_exact_match_rate", 1.0) for m in room_metric_dicts]))
            if room_metric_dicts
            else 1.0
        )
        avg_final_graph_marker_overwrite_rate = (
            float(np.mean([m.get("final_graph_marker_overwrite_rate", 0.0) for m in room_metric_dicts]))
            if room_metric_dicts
            else 0.0
        )
        avg_neural_semantic_anchor_error = (
            float(np.mean([m.get("neural_semantic_anchor_avg_manhattan_error", 0.0) for m in room_metric_dicts]))
            if room_metric_dicts
            else 0.0
        )
        avg_final_pre_overlay_semantic_anchor_error = (
            float(np.mean([m.get("final_pre_overlay_semantic_anchor_avg_manhattan_error", 0.0) for m in room_metric_dicts]))
            if room_metric_dicts
            else 0.0
        )
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
            'total_graph_marker_expected': total_graph_marker_expected,
            'total_graph_marker_overwrites': total_graph_marker_overwrites,
            'avg_neural_graph_marker_exact_match_rate': avg_neural_graph_marker_exact_match_rate,
            'avg_final_pre_overlay_graph_marker_exact_match_rate': avg_final_pre_overlay_graph_marker_exact_match_rate,
            'avg_final_post_overlay_graph_marker_exact_match_rate': avg_final_post_overlay_graph_marker_exact_match_rate,
            'avg_final_graph_marker_overwrite_rate': avg_final_graph_marker_overwrite_rate,
            'avg_neural_semantic_anchor_error': avg_neural_semantic_anchor_error,
            'avg_final_pre_overlay_semantic_anchor_error': avg_final_pre_overlay_semantic_anchor_error,
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
            else:
                room_grid = fit_room_grid(room_value)
                neural_grid = room_grid.copy()
                latent = torch.empty(0)

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
                metrics={
                    "room_id": room_id,
                    "was_repaired": bool(was_repaired),
                    "planned_traversability_pixels": float(np.sum(room_plan_mask)) if isinstance(room_plan_mask, np.ndarray) else 0.0,
                    **repair_diagnostics,
                },
            )

        dungeon_grid = self.stitch_rooms(normalized_rooms, mission_graph)
        map_elites_score = self.evaluate_generated_dungeon(
            dungeon_grid,
            mission_graph,
            enable_map_elites=enable_map_elites,
        )
        metrics = {
            "num_rooms": len(normalized_rooms),
            "total_tiles_repaired": sum(r.metrics.get("tiles_changed", 0) for r in normalized_rooms.values()),
            "repair_rate": (
                sum(bool(r.was_repaired) for r in normalized_rooms.values()) / max(1, len(normalized_rooms))
            ),
            "dungeon_shape": dungeon_grid.shape,
            "symbolic_only": True,
        }
        return DungeonGenerationResult(
            dungeon_grid=dungeon_grid,
            rooms=normalized_rooms,
            mission_graph=mission_graph,
            metrics=metrics,
            map_elites_score=map_elites_score,
            generation_time=0.0,
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

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'tpe': tpe,
            'node_positions': node_positions,
            'node_mask': torch.ones(num_nodes, device=self.device, dtype=torch.float32),
            'node_order': node_order,
            'node_to_idx': node_to_idx,
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
        difficulty_rating = str(attrs.get("difficulty_rating", "") or "").strip().upper()

        def _hint(name: str, *aliases: str) -> bool:
            return self._coerce_bool(attrs.get(name)) or any(self._coerce_bool(attrs.get(alias)) for alias in aliases)

        return {
            "is_start": _hint("is_start", "is_entry") or "start" in tokens,
            "has_enemy": _hint("has_enemy") or "e" in tokens or "enemy" in tokens,
            "has_key": _hint("has_key") or "k" in tokens or "key" in tokens,
            "has_item": _hint("has_item", "has_macro_item", "has_minor_item") or "i" in tokens or "item" in tokens or "treasure" in tokens,
            "has_goal": _hint("has_triforce", "is_triforce", "is_goal") or "t" in tokens or "goal" in tokens or "triforce" in tokens,
            "has_boss": _hint("has_boss", "is_boss") or "b" in tokens or "boss" in tokens,
            "has_puzzle": _hint("has_puzzle") or "p" in tokens or "puzzle" in tokens or raw_type in {"switch", "puzzle", "tutorial_puzzle", "combat_puzzle", "complex_puzzle"} or "puzzle" in raw_type,
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
            branch_density = min(branch_density, 0.35)
            block_budget = min(block_budget, 14)
        elif role_flags.get("is_combat_puzzle", False):
            archetype = "combat"
            branch_density = min(branch_density, 0.45)
            block_budget = min(block_budget, 18)
        elif role_flags.get("is_complex_puzzle", False):
            if archetype not in {"hub", "serpentine"}:
                archetype = "hub" if required_door_count >= 3 else "serpentine"
            branch_density = max(branch_density, 0.9)
            block_budget = max(block_budget, 34)
        elif gate_family in {"switch", "toggle"}:
            archetype = "gate"
            branch_density = max(branch_density, 0.7)
            block_budget = max(block_budget, 24)
        elif gate_family in {"bombable", "item_unlock", "key"} and archetype not in {"hub", "combat"}:
            archetype = "gate"
            if gate_family == "key":
                branch_density = max(branch_density, 0.55)
                block_budget = max(block_budget, 20)
            elif gate_family == "item_unlock":
                branch_density = max(branch_density, 0.6)
                block_budget = max(block_budget, 22)
            else:
                branch_density = max(branch_density, 0.65)
                block_budget = max(block_budget, 24)
        elif node_type in {"item", "protection_item", "minor_item", "treasure", "stair", "stairs_up", "stairs_down", "warp"}:
            archetype = "island"
            branch_density = min(max(branch_density, 0.55), 0.8)
            block_budget = max(block_budget, 22)
        elif difficulty_rating == "MODERATE" and archetype == "serpentine":
            branch_density = min(max(branch_density, 0.6), 0.8)

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
        return build_semantic_room_plan_trace(
            np.asarray(room_grid, dtype=np.int32),
            start=start,
            goal=goal,
            required_doors=semantics["required_doors"],
            incoming_dirs=semantics["incoming_dirs"],
            outgoing_dirs=semantics["outgoing_dirs"],
            edge_constraint_tokens=semantics["edge_constraints"],
            room_role_flags=self._room_role_flags(attrs),
            validator_plan_max_states=self.default_validator_plan_max_states,
        ).astype(np.float32, copy=False)

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
        current_node_idx = graph_data.get('node_to_idx', {}).get(room_id, 0)
        style_id = self._extract_explicit_style_id(mission_graph, room_id=room_id)
        current_node_distance = compute_current_node_distance_features(
            graph_data.get('edge_index'),
            int(graph_data.get('node_features').shape[0]) if isinstance(graph_data.get('node_features'), torch.Tensor) else 0,
            current_node_idx=current_node_idx,
            device=self.device,
            dtype=torch.float32,
            max_distance=self.current_node_distance_max,
        )
        return {
            'node_features': graph_data.get('node_features'),
            'edge_index': graph_data.get('edge_index'),
            'edge_features': graph_data.get('edge_features'),
            'tpe': graph_data.get('tpe'),
            'node_positions': graph_data.get('node_positions'),
            'node_mask': graph_data.get('node_mask'),
            'has_room_anchor': True,
            'mission_graph': mission_graph,
            'current_node_idx': current_node_idx,
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
        "default_validator_plan_max_states": stage.get(
            "validator_plan_max_states",
            DEFAULT_VALIDATOR_PLAN_MAX_STATES,
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
            },
            "logic_net_fallback_config": {
                "latent_dim": diffusion["latent_dim"],
                "num_classes": config["dataset"]["num_classes"],
                "num_logic_iterations": diffusion["num_logic_iterations"],
                "logic_topology_trace_weight": diffusion["logic_topology_trace_weight"],
                "logic_topology_anchor_weight": diffusion["logic_topology_anchor_weight"],
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
