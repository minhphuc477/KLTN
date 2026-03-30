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
        guidance_scale=7.5,
        logic_guidance_scale=1.0,
        seed=42
    )
"""

import json
import logging
import hashlib
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
from src.core.definitions import DOOR_POSITIONS, parse_edge_type_tokens
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
    ROOM_TOPOLOGY_CHANNEL_COUNT,
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
    if isinstance(node, (int, np.integer)):
        return int(node) & 0xFFFFFFFF
    payload = repr(node).encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


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

    def build(self, pipeline: "NeuralSymbolicDungeonPipeline") -> PipelineComponents:
        return PipelineComponents(
            neural=NeuralGenerationComponents(
                vqvae=pipeline._load_vqvae(self.vqvae_checkpoint),
                condition_encoder=pipeline._load_condition_encoder(self.condition_encoder_checkpoint),
                diffusion=pipeline._load_diffusion(self.diffusion_checkpoint),
                logic_net=pipeline._load_logic_net(self.logic_net_checkpoint),
            ),
            symbolic=SymbolicGenerationComponents(
                refiner=pipeline._create_refiner(self.use_learned_refiner_rules),
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
        topology_refinement_mode: str = "gat2",
        diffusion_attention_mode: str = "softmax",
        diffusion_hedgehog_feature_dim: int = 32,
        diffusion_cfg_schedule_mode: str = "constant",
        diffusion_cfg_schedule_min_scale: float = 1.0,
        diffusion_cfg_schedule_power: float = 1.0,
        room_generator_mode: str = "latent_diffusion",
        masked_room_checkpoint: Optional[str] = None,
        masked_sampling_steps: int = 8,
        fast_sampling_checkpoint: Optional[str] = None,
        fast_sampling_steps: int = 4,
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
        self.room_generator_mode = str(room_generator_mode).strip().lower()
        self.masked_room_checkpoint = (
            None if masked_room_checkpoint is None else str(masked_room_checkpoint).strip()
        ) or None
        self.masked_sampling_steps = int(max(1, int(masked_sampling_steps)))
        self.fast_sampling_checkpoint = (
            None if fast_sampling_checkpoint is None else str(fast_sampling_checkpoint).strip()
        ) or None
        self.fast_sampling_steps = int(max(1, int(fast_sampling_steps)))
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
        if self.diffusion_cfg_schedule_mode not in {"constant", "linear_decay", "cosine_decay"}:
            raise ValueError(
                f"Invalid diffusion_cfg_schedule_mode={diffusion_cfg_schedule_mode!r}. "
                "Expected 'constant', 'linear_decay', or 'cosine_decay'."
            )
        self.diffusion_hedgehog_feature_dim = int(max(4, int(diffusion_hedgehog_feature_dim)))
        if self.topology_refinement_mode == "upgraded":
            self.topology_refinement_mode = "gat2"
        if self.topology_refinement_mode not in {"none", "lightweight", "gat2"}:
            raise ValueError(
                f"Invalid topology_refinement_mode={topology_refinement_mode!r}. "
                "Expected 'none', 'lightweight', or 'gat2'."
            )
        gnn_type = str(condition_gnn_type).strip().lower()
        if gnn_type not in {"gcn", "gat", "sage"}:
            raise ValueError(
                f"Invalid condition_gnn_type={condition_gnn_type!r}. Expected 'gcn', 'gat', or 'sage'."
            )
        self.condition_gnn_type = gnn_type

        # Runtime fallback diagnostics for auditability of best-effort paths.
        self.runtime_diagnostics: Dict[str, int] = {}
        self._valid_semantic_tile_ids_np = np.array(
            sorted({int(v) for v in SEMANTIC_PALETTE.values()}),
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
            refiner=cls._create_refiner(use_learned_refiner_rules),
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
            components=PipelineComponents(symbolic=symbolic),
        )

    def _load_checkpoint_and_metadata(self, checkpoint_path: str, model_name: str) -> Tuple[dict, dict]:
        """Load checkpoint and optional sidecar metadata for strict validation."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        metadata_path = Path(f"{checkpoint_path}.meta.json")
        metadata: dict = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            validate_checkpoint_metadata(metadata=metadata, model_name=model_name)
        elif self.strict_checkpoint_mode:
            raise FileNotFoundError(
                f"Strict checkpoint mode enabled: metadata sidecar missing for {model_name} at {metadata_path}"
            )
        return checkpoint, metadata

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
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, _metadata = self._load_checkpoint_and_metadata(checkpoint_path, "vqvae")
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Backward compatibility: older checkpoints may use plain Conv2d
            # keys (encoder.conv_in.weight) while newer CoordConv checkpoints
            # use encoder.conv_in.conv.weight.
            if isinstance(state_dict, dict):
                has_coordconv_keys = ('encoder.conv_in.conv.weight' in state_dict)
                has_plain_conv_keys = ('encoder.conv_in.weight' in state_dict)
                if has_plain_conv_keys and not has_coordconv_keys:
                    use_coordconv = False

        model = SemanticVQVAE(
            num_classes=44,
            codebook_size=512,
            latent_dim=64,
            hidden_dim=128,
            use_coordconv=use_coordconv,
        ).to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            if checkpoint is None:
                checkpoint, _metadata = self._load_checkpoint_and_metadata(checkpoint_path, "vqvae")
            if state_dict is None and isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
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
        default_node_feature_dim = 12
        default_edge_feature_dim = 14
        node_feature_dim = int(default_node_feature_dim)
        edge_feature_dim = int(default_edge_feature_dim)
        checkpoint_state: Optional[Dict[str, Any]] = None

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, _metadata = self._load_checkpoint_and_metadata(checkpoint_path, "condition_encoder")
            checkpoint_state = checkpoint.get('model_state_dict', checkpoint)
            if isinstance(checkpoint_state, dict):
                node_weight = checkpoint_state.get('global_encoder.node_encoder.weight')
                edge_weight = checkpoint_state.get('global_encoder.edge_encoder.weight')
                if isinstance(node_weight, torch.Tensor) and node_weight.dim() == 2:
                    node_feature_dim = int(max(1, int(node_weight.shape[1])))
                if isinstance(edge_weight, torch.Tensor) and edge_weight.dim() == 2:
                    edge_feature_dim = int(max(1, int(edge_weight.shape[1])))

        model = DualStreamConditionEncoder(
            latent_dim=64,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=256,
            output_dim=256,
            gnn_type=self.condition_gnn_type,
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
        model = LatentDiffusionModel(
            latent_dim=64,
            context_dim=256,
            num_timesteps=1000,
            model_channels=128,
            topology_refinement_mode=self.topology_refinement_mode,
            attention_mode=self.diffusion_attention_mode,
            hedgehog_feature_dim=self.diffusion_hedgehog_feature_dim,
            cfg_schedule_mode=self.diffusion_cfg_schedule_mode,
            cfg_schedule_min_scale=self.diffusion_cfg_schedule_min_scale,
            cfg_schedule_power=self.diffusion_cfg_schedule_power,
            room_topology_channels=ROOM_TOPOLOGY_CHANNEL_COUNT,
        ).to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, _metadata = self._load_checkpoint_and_metadata(checkpoint_path, "diffusion")
            checkpoint_state = checkpoint.get('model_state_dict', checkpoint)
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
                "Loaded diffusion model from %s (missing=%d unexpected=%d)",
                checkpoint_path,
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
        model = LogicNet(
            latent_dim=64,
            num_classes=44,
            num_iterations=20,
        ).to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, _metadata = self._load_checkpoint_and_metadata(checkpoint_path, "logic_net")
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
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

        model = create_discrete_masked_model(
            num_classes=44,
            hidden_dim=64,
            model_channels=128,
            context_dim=256,
            num_steps=self.masked_sampling_steps,
            attention_mode=self.diffusion_attention_mode,
            hedgehog_feature_dim=self.diffusion_hedgehog_feature_dim,
            room_topology_channels=ROOM_TOPOLOGY_CHANNEL_COUNT,
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint, _metadata = self._load_checkpoint_and_metadata(checkpoint_path, "masked_room_model")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
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
    def _create_refiner(use_learned_rules: bool) -> SymbolicRefiner:
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
            adjacency=canonical_adjacency,
            learned_stats=learned_stats,
            max_repair_attempts=5,
            margin=2,
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
            num_classes=44,
        )

    def _compute_room_condition(
        self,
        *,
        neighbor_latents: Dict[str, Optional[torch.Tensor]],
        graph_context: Dict[str, Any],
        boundary_constraints: Optional[torch.Tensor],
        position: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build Block-III conditioning tensor for a room."""
        if boundary_constraints is None:
            boundary_constraints = torch.zeros(1, 8, device=self.device)
        if position is None:
            position = torch.zeros(1, 2, device=self.device)

        try:
            node_dim, edge_dim = self._condition_feature_dims()
            validate_feature_dims(
                node_features=graph_context.get('node_features'),
                edge_features=graph_context.get('edge_features'),
                expected_node_dim=node_dim,
                expected_edge_dim=edge_dim,
            )
            condition = self.condition_encoder(
                neighbor_latents=neighbor_latents,
                boundary_constraints=boundary_constraints,
                position=position,
                node_features=graph_context.get('node_features'),
                edge_index=graph_context.get('edge_index'),
                edge_features=graph_context.get('edge_features'),
                tpe=graph_context.get('tpe'),
                current_node_idx=graph_context.get('current_node_idx'),
            )
            validate_tensor_contract(
                condition,
                BlockShapeContract(name='block_iii_condition_output', dims=2, batch_dim=1, channel_dim=256),
            )
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            self._bump_diagnostic("condition_encoder_fallback")
            if self.strict_checkpoint_mode:
                raise RuntimeError(
                    f"Condition encoding failed in strict mode: {e}"
                ) from e
            logger.warning(f"Condition encoding failed: {e}, using zero condition")
            condition = torch.zeros(1, 256, device=self.device)

        if self.use_graph_node_cross_attention:
            try:
                node_tokens = self.condition_encoder.encode_global_only(
                    node_features=graph_context.get('node_features'),
                    edge_index=graph_context.get('edge_index'),
                    edge_features=graph_context.get('edge_features'),
                    tpe=graph_context.get('tpe'),
                )
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
            'room_topology_map': torch.cat(
                [inp['graph_context']['room_topology_map'] for inp in per_room_inputs],
                dim=0,
            ),
        }

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

            logits_batch = self.vqvae.decode(z_batch)

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
        guidance_scale: float = 7.5,
        logic_guidance_scale: float = 1.0,
        num_diffusion_steps: int = 50,
        use_fast_sampling: bool = False,
        latent_sampler: str = "diffusion",
        categorical_codebook_size: Optional[int] = None,
        use_ddim: bool = True,
        apply_repair: bool = True,
        start_goal_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        seed: Optional[int] = None,
        precomputed_condition: Optional[torch.Tensor] = None,
        precomputed_latent: Optional[torch.Tensor] = None,
        precomputed_logits: Optional[torch.Tensor] = None,
        precomputed_tokens: Optional[torch.Tensor] = None,
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
                graph_context=graph_context,
                boundary_constraints=boundary_constraints,
                position=position,
            )

        sampler_mode = str(latent_sampler or "diffusion").strip().lower()
        graph_data = graph_context if isinstance(graph_context, dict) else None
        mission_graph_for_room = graph_data.get("mission_graph") if isinstance(graph_data, dict) else None

        sampled_tokens: Optional[torch.Tensor] = None

        if precomputed_latent is not None and precomputed_logits is not None:
            z_latent = precomputed_latent.to(self.device)
            logits = precomputed_logits.to(self.device)
            if precomputed_tokens is not None:
                sampled_tokens = precomputed_tokens.to(self.device, dtype=torch.long)
        elif self.room_generator_mode == "discrete_masked":
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
            logits = self.vqvae.decode(z_latent)  # (1, 44, 16, 11)
        validate_tensor_contract(
            logits,
            BlockShapeContract(
                name='block_ii_decode_logits',
                dims=4,
                batch_dim=1,
                channel_dim=44,
                spatial_hw=(ROOM_HEIGHT, ROOM_WIDTH),
            ),
        )
        if self.room_generator_mode == "discrete_masked" and sampled_tokens is not None:
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
        neural_probs = logits.softmax(dim=1).detach().cpu().numpy()[0]  # (44, 16, 11)
        
        # BLOCK VI: Symbolic Repair (if enabled)
        was_repaired = False
        repair_mask = None
        room_plan_mask = None
        final_grid = neural_grid.copy()
        repaired_invalid_count = 0
        repaired_invalid_ids: List[int] = []
        repair_diag: Dict[str, Any] = {}
        
        if apply_repair and start_goal_coords is not None:
            start, goal = self._normalize_start_goal_coords(start_goal_coords)
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
            start, goal = self._normalize_start_goal_coords(start_goal_coords)
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
            'vglc_compliant': valid_dims,
            'wfc_feedback_rounds': float(repair_diag.get('feedback_rounds', 0)),
            'wfc_failures': float(repair_diag.get('wfc_failures', 0)),
            'planned_traversability_pixels': float(np.sum(room_plan_mask)) if isinstance(room_plan_mask, np.ndarray) else 0.0,
        }
        
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
        num_rooms: int = 8,
        population_size: int = 50,
        generations: int = 100,
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
            if target_curve is None:
                target_curve = [0.2, 0.4, 0.6, 0.8, 1.0]

            target_genome_length = max(10, int(num_rooms * 0.7))
            topology_generator = EvolutionaryTopologyGenerator(
                target_curve=target_curve,
                population_size=population_size,
                generations=generations,
                genome_length=target_genome_length,
                max_nodes=num_rooms,
                seed=seed,
            )

            graph = topology_generator.evolve()
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
        guidance_scale: float = 7.5,
        logic_guidance_scale: float = 1.0,
        num_diffusion_steps: int = 50,
        use_fast_sampling: bool = False,
        latent_sampler: str = "diffusion",
        categorical_codebook_size: Optional[int] = None,
        apply_repair: bool = True,
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
    def generate_dungeon(
        self,
        mission_graph: Optional[nx.Graph] = None,
        guidance_scale: float = 7.5,
        logic_guidance_scale: float = 1.0,
        num_diffusion_steps: int = 50,
        use_fast_sampling: bool = False,
        latent_sampler: str = "diffusion",
        categorical_codebook_size: Optional[int] = None,
        use_topological_positional_encoding: bool = True,
        apply_repair: bool = True,
        seed: Optional[int] = None,
        enable_map_elites: bool = True,
        # Block I: Evolutionary generation parameters
        generate_topology: bool = False,
        target_curve: Optional[List[float]] = None,
        num_rooms: int = 8,
        population_size: int = 50,
        generations: int = 100,
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
            
        Returns:
            DungeonGenerationResult with complete dungeon and metrics
        """
        self._require_room_generation_components("generate_dungeon")
        import time
        start_time = time.time()
        
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

        def _hint(name: str, *aliases: str) -> bool:
            return self._coerce_bool(attrs.get(name)) or any(self._coerce_bool(attrs.get(alias)) for alias in aliases)

        return {
            "is_start": _hint("is_start", "is_entry") or "s" in tokens or "start" in tokens,
            "has_enemy": _hint("has_enemy") or "e" in tokens or "enemy" in tokens,
            "has_key": _hint("has_key") or "k" in tokens or "key" in tokens,
            "has_item": _hint("has_item", "has_macro_item", "has_minor_item") or "i" in tokens or "item" in tokens or "treasure" in tokens,
            "has_goal": _hint("has_triforce", "is_triforce", "is_goal") or "t" in tokens or "goal" in tokens or "triforce" in tokens,
            "has_boss": _hint("has_boss", "is_boss") or "b" in tokens or "boss" in tokens,
            "has_puzzle": _hint("has_puzzle") or "p" in tokens or "puzzle" in tokens,
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
            edge_constraint_tokens=semantics["edge_constraints"],
            room_role_flags=self._room_role_flags(attrs),
        )
        return torch.from_numpy(topo_np).unsqueeze(0).to(device=self.device, dtype=torch.float32)

    def _build_room_graph_context(
        self,
        *,
        graph_data: Dict[str, Any],
        mission_graph: nx.Graph,
        room_id: Any,
        start_goal: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Build per-room graph context shared by condition encoding and diffusion."""
        return {
            'node_features': graph_data.get('node_features'),
            'edge_index': graph_data.get('edge_index'),
            'edge_features': graph_data.get('edge_features'),
            'tpe': graph_data.get('tpe'),
            'node_positions': graph_data.get('node_positions'),
            'node_mask': graph_data.get('node_mask'),
            'has_room_anchor': True,
            'mission_graph': mission_graph,
            'current_node_idx': graph_data.get('node_to_idx', {}).get(room_id, 0),
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
            sr, sc = self._clamp_room_coord(start)
            gr, gc = self._clamp_room_coord(goal)
            fixed_tokens[0, sr, sc] = int(SEMANTIC_PALETTE["START"])
            fixed_mask[0, sr, sc] = True
            fixed_tokens[0, gr, gc] = int(SEMANTIC_PALETTE["TRIFORCE"])
            fixed_mask[0, gr, gc] = True

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
    def _encode_room_grid_to_latent(self, room_grid: np.ndarray, num_classes: int = 44) -> torch.Tensor:
        """Encode finalized room grid back into latent space for neighbor conditioning."""
        vqvae = self._require_component("vqvae", "_encode_room_grid_to_latent")
        grid = np.asarray(room_grid, dtype=np.int64)
        grid = np.clip(grid, 0, int(num_classes) - 1)
        one_hot = np.eye(int(num_classes), dtype=np.float32)[grid]
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
    
    return NeuralSymbolicDungeonPipeline(
        vqvae_checkpoint=str(checkpoint_dir / "vqvae_best.pth"),
        diffusion_checkpoint=str(checkpoint_dir / "diffusion_best.pth"),
        logic_net_checkpoint=str(checkpoint_dir / "logic_net_best.pth"),
        condition_encoder_checkpoint=str(checkpoint_dir / "condition_encoder_best.pth"),
        device=device,
        **kwargs
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
    'create_pipeline',
]

