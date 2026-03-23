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

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import torch
import numpy as np
import networkx as nx

from src.core import (
    SemanticVQVAE,
    DualStreamConditionEncoder,
    LatentDiffusionModel,
    LogicNet,
    SymbolicRefiner,
    LearnedTileStatistics,
    SEMANTIC_PALETTE,
    ROOM_HEIGHT,
    ROOM_WIDTH,
)
from src.simulation.map_elites import MAPElitesEvaluator
from src.zelda_data.zelda_core import DungeonStitcher

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
)
from src.pipeline.graph_features import (
    compute_tpe_features,
    condition_feature_dims,
    encode_edge_feature_vector,
    extract_node_feature_vector,
    fit_feature_vector,
)

logger = logging.getLogger(__name__)


def _stable_node_sort_key(node: Any) -> Tuple[int, Any]:
    """
    Deterministic sort key that supports mixed node-id types.

    Keeps numeric node IDs in numeric order while remaining robust for
    string/tuple/object node identifiers.
    """
    if isinstance(node, (int, np.integer)):
        return (0, int(node))
    if isinstance(node, str):
        return (1, node)
    return (2, str(node))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RoomGenerationResult:
    """Result of generating a single room."""
    room_id: int
    room_grid: np.ndarray  # (16, 11) discrete tile IDs
    latent: torch.Tensor   # (1, 64, 4, 3) VQ-VAE latent
    neural_grid: np.ndarray  # (16, 11) before symbolic repair
    was_repaired: bool
    repair_mask: Optional[np.ndarray] = None  # (16, 11) bool mask
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
    ):
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if enable_logging:
            logger.info(f"Initializing NeuralSymbolicDungeonPipeline on {self.device}")
        
        # Block II: VQ-VAE
        self.vqvae = self._load_vqvae(vqvae_checkpoint)
        self.vqvae.eval()
        
        # Block III: Condition Encoder
        self.condition_encoder = self._load_condition_encoder(condition_encoder_checkpoint)
        self.condition_encoder.eval()
        
        # Block IV: Latent Diffusion
        self.diffusion = self._load_diffusion(diffusion_checkpoint)
        self.diffusion.eval()
        
        # Block V: LogicNet
        self.logic_net = self._load_logic_net(logic_net_checkpoint)
        self.logic_net.eval()
        
        # Block VI: Symbolic Refiner
        self.refiner = self._create_refiner(use_learned_refiner_rules)
        
        # Block VII: MAP-Elites
        self.map_elites = MAPElitesEvaluator(
            resolution=map_elites_resolution,
            tie_breaker='quality_score',
            descriptor_mode='hybrid',
        )

        # Runtime fallback diagnostics for auditability of best-effort paths.
        self.runtime_diagnostics: Dict[str, int] = {}
        
        # Utilities
        self.stitcher = DungeonStitcher()
        
        if enable_logging:
            logger.info("Pipeline initialized successfully")

    def _bump_diagnostic(self, key: str) -> None:
        """Increment a named runtime diagnostic counter."""
        k = str(key).strip().lower()
        if not k:
            return
        self.runtime_diagnostics[k] = int(self.runtime_diagnostics.get(k, 0)) + 1
    
    def _load_vqvae(self, checkpoint_path: Optional[str]) -> SemanticVQVAE:
        """Load or create VQ-VAE model."""
        model = SemanticVQVAE(
            num_classes=44,
            codebook_size=512,
            latent_dim=64,
            hidden_dim=128,
        ).to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded VQ-VAE from {checkpoint_path}")
        else:
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
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
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
        ).to(self.device)
        
        if checkpoint_state is not None:
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
            if missing:
                logger.warning("Condition encoder missing keys (first 8): %s", missing[:8])
            if unexpected:
                logger.warning("Condition encoder unexpected keys (first 8): %s", unexpected[:8])
        else:
            logger.warning(
                "No condition encoder checkpoint, using random initialization with enhanced schema (node_dim=%d edge_dim=%d)",
                node_feature_dim,
                edge_feature_dim,
            )
        
        return model

    def _condition_feature_dims(self) -> Tuple[int, int]:
        """Get active (node_dim, edge_dim) expected by the condition encoder."""
        return condition_feature_dims(self.condition_encoder)

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
        ).to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded diffusion model from {checkpoint_path}")
        else:
            logger.warning("No diffusion checkpoint, using random initialization")
        
        return model
    
    def _load_logic_net(self, checkpoint_path: Optional[str]) -> LogicNet:
        """Load or create LogicNet."""
        model = LogicNet(
            latent_dim=64,
            num_classes=44,
            num_iterations=20,
        ).to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded LogicNet from {checkpoint_path}")
        else:
            logger.warning("No LogicNet checkpoint, using random initialization")
        
        return model
    
    def _create_refiner(self, use_learned_rules: bool) -> SymbolicRefiner:
        """Create symbolic refiner with optional learned rules."""
        learned_stats = LearnedTileStatistics() if use_learned_rules else None
        
        refiner = SymbolicRefiner(
            learned_stats=learned_stats,
            max_repair_attempts=5,
            margin=2,
        )
        
        logger.info(f"Created SymbolicRefiner (learned_rules={use_learned_rules})")
        return refiner

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
    
    @torch.no_grad()
    def generate_room(
        self,
        neighbor_latents: Dict[str, Optional[torch.Tensor]],
        graph_context: Dict[str, Any],
        room_id: int,
        boundary_constraints: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        logic_guidance_scale: float = 1.0,
        num_diffusion_steps: int = 50,
        latent_sampler: str = "diffusion",
        categorical_codebook_size: Optional[int] = None,
        use_ddim: bool = True,
        apply_repair: bool = True,
        start_goal_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        seed: Optional[int] = None,
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
            latent_sampler: "diffusion" (default) or "categorical"
            categorical_codebook_size: Optional cap for categorical sampling
            use_ddim: Use DDIM (deterministic) vs DDPM (stochastic)
            apply_repair: Apply symbolic WFC repair
            start_goal_coords: ((start_r, start_c), (goal_r, goal_c)) for repair
            seed: Random seed for reproducibility
            
        Returns:
            RoomGenerationResult with room grid, latents, and metrics
        """
        local_np_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if seed is not None:
            torch.manual_seed(seed)
        
        # Default boundary and position if not provided
        if boundary_constraints is None:
            boundary_constraints = torch.zeros(1, 8, device=self.device)
        if position is None:
            position = torch.zeros(1, 2, device=self.device)
        
        # BLOCK III: Dual-Stream Conditioning
        try:
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
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            self._bump_diagnostic("condition_encoder_fallback")
            logger.warning(f"Condition encoding failed: {e}, using zero condition")
            condition = torch.zeros(1, 256, device=self.device)
        
        # BLOCK V: Logic guidance configuration for diffusion sampler
        self.diffusion.cfg_scale = float(guidance_scale)
        self.diffusion.guidance.logic_net = self.logic_net if logic_guidance_scale > 0 else None
        self.diffusion.guidance.guidance_scale = max(0.0, float(logic_guidance_scale))

        # Infer latent shape from neighbors when possible, otherwise use VQ-VAE spatial downsampling (x4).
        latent_shape: Tuple[int, int, int, int] = (
            1,
            int(self.diffusion.latent_dim),
            max(1, ROOM_HEIGHT // 4),
            max(1, (ROOM_WIDTH + 3) // 4),
        )
        for latent in neighbor_latents.values():
            if isinstance(latent, torch.Tensor) and latent.dim() == 4:
                latent_shape = tuple(int(v) for v in latent.shape)  # type: ignore[assignment]
                break

        sampler_mode = str(latent_sampler or "diffusion").strip().lower()
        graph_data = graph_context if isinstance(graph_context, dict) else None

        if sampler_mode == "categorical":
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
        else:
            # BLOCK IV: Latent Diffusion Sampling
            logger.debug(f"Room {room_id}: Sampling with {num_diffusion_steps} steps")
            if use_ddim:
                z_latent = self.diffusion.ddim_sample(
                    context=condition,
                    shape=latent_shape,
                    num_steps=max(1, int(num_diffusion_steps)),
                    graph_data=graph_data,
                )
            else:
                z_latent = self.diffusion.sample(
                    context=condition,
                    shape=latent_shape,
                    graph_data=graph_data,
                )

            # BLOCK II: VQ-VAE Decoding
            logits = self.vqvae.decode(z_latent)  # (1, 44, 16, 11)
        neural_grid = logits.argmax(dim=1).cpu().numpy()[0]  # (16, 11)
        
        # BLOCK VI: Symbolic Repair (if enabled)
        was_repaired = False
        repair_mask = None
        final_grid = neural_grid.copy()
        repair_diag: Dict[str, Any] = {}
        
        if apply_repair and start_goal_coords is not None:
            start_rc, goal_rc = start_goal_coords
            # API/documentation uses (row, col), but SymbolicRefiner expects (x, y).
            start = (
                max(0, min(ROOM_WIDTH - 1, int(start_rc[1]))),
                max(0, min(ROOM_HEIGHT - 1, int(start_rc[0]))),
            )
            goal = (
                max(0, min(ROOM_WIDTH - 1, int(goal_rc[1]))),
                max(0, min(ROOM_HEIGHT - 1, int(goal_rc[0]))),
            )
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

                repaired_grid, success, repair_diag = self.refiner.repair_room_with_feedback(
                    grid=neural_grid,
                    start=start,
                    goal=goal,
                    feedback_callback=_feedback_callback,
                    max_feedback_rounds=2,
                )
                
                if success:
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
        
        # VGLC Compliance: Validate room dimensions
        valid_dims, dim_msg = validate_room_dimensions(final_grid)
        if not valid_dims:
            logger.error(f"Room {room_id}: VGLC dimension validation FAILED: {dim_msg}")
            raise ValueError(f"Generated room has invalid dimensions: {dim_msg}")
        else:
            logger.debug(f"Room {room_id}: VGLC dimension validation PASSED")
        
        # Compute metrics
        metrics = {
            'room_id': room_id,
            'neural_grid_entropy': float(np.mean(-(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1).cpu().numpy())),
            'was_repaired': was_repaired,
            'tiles_changed': int(np.sum(repair_mask)) if repair_mask is not None else 0,
            'vglc_compliant': valid_dims,
            'wfc_feedback_rounds': float(repair_diag.get('feedback_rounds', 0)),
            'wfc_failures': float(repair_diag.get('wfc_failures', 0)),
        }
        
        return RoomGenerationResult(
            room_id=room_id,
            room_grid=final_grid,
            latent=z_latent.cpu(),
            neural_grid=neural_grid,
            was_repaired=was_repaired,
            repair_mask=repair_mask,
            metrics=metrics,
        )
    
    @torch.no_grad()
    def generate_dungeon(
        self,
        mission_graph: Optional[nx.Graph] = None,
        guidance_scale: float = 7.5,
        logic_guidance_scale: float = 1.0,
        num_diffusion_steps: int = 50,
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
        import time
        start_time = time.time()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Block I: Evolutionary Topology Generation (if needed)
        if mission_graph is None:
            if not generate_topology:
                raise ValueError(
                    "mission_graph is None but generate_topology=False. "
                    "Either provide a mission_graph or set generate_topology=True"
                )
            
            logger.info("Block I: Generating dungeon topology via evolutionary search")
            if target_curve is None:
                # Default tension curve: gradual increase
                target_curve = [0.2, 0.4, 0.6, 0.8, 1.0]
            
            # Calculate genome_length to target desired room count
            # Empirical relationship: genome_length â‰ˆ num_rooms * 0.7 (rules don't always apply)
            target_genome_length = max(10, int(num_rooms * 0.7))
            
            topology_generator = EvolutionaryTopologyGenerator(
                target_curve=target_curve,
                population_size=population_size,
                generations=generations,
                genome_length=target_genome_length,
                max_nodes=num_rooms,  # Direct room count constraint
                seed=seed,
            )
            
            mission_graph = topology_generator.evolve()
            logger.info(f"Block I: Generated topology with {mission_graph.number_of_nodes()} rooms")
            
            # Validate generated topology
            is_valid, errors = validate_graph_topology(mission_graph)
            if not is_valid:
                logger.warning(f"Block I: Generated topology has validation errors: {errors}")
        
        # Continue with existing pipeline (Blocks II-VII)
        if mission_graph is None:
            raise ValueError("mission_graph is still None after topology generation attempt")
        
        # VGLC Compliance: Filter virtual nodes before processing
        logger.debug("Applying VGLC compliance: filtering virtual nodes from mission graph")
        mission_graph_physical = filter_virtual_nodes(mission_graph)
        physical_start = get_physical_start_node(mission_graph)
        
        if physical_start is not None:
            logger.debug(f"Physical start node: {physical_start}")
        
        logger.info(f"Generating dungeon with {len(mission_graph_physical.nodes)} rooms "
                   f"(filtered {len(mission_graph.nodes) - len(mission_graph_physical.nodes)} virtual nodes)")
        
        # Prepare graph context (use physical graph)
        graph_data = self._prepare_graph_context(
            mission_graph_physical,
            use_tpe=use_topological_positional_encoding,
        )
        
        rooms: Dict[int, RoomGenerationResult] = {}
        room_latents: Dict[int, torch.Tensor] = {}

        # Generate rooms in a deterministic order.
        if mission_graph_physical.is_directed():
            try:
                generation_order = list(nx.topological_sort(mission_graph_physical))
            except nx.NetworkXUnfeasible:
                logger.warning(
                    "Mission graph contains cycles; falling back to sorted node order for generation."
                )
                generation_order = sorted(
                    mission_graph_physical.nodes(),
                    key=_stable_node_sort_key,
                )
        else:
            generation_order = sorted(
                mission_graph_physical.nodes(),
                key=_stable_node_sort_key,
            )

        for idx, room_id in enumerate(generation_order):
            logger.debug(f"Generating room {room_id} ({idx+1}/{len(generation_order)})")
            
            # Get neighboring latents
            neighbor_latents = self._get_neighbor_latents(
                room_id, mission_graph_physical, room_latents
            )
            
            # Get start/goal for repair
            start_goal = self._extract_room_start_goal(mission_graph_physical, room_id)
            
            # Generate room
            room_result = self.generate_room(
                neighbor_latents=neighbor_latents,
                graph_context={
                    'node_features': graph_data.get('node_features'),
                    'edge_index': graph_data.get('edge_index'),
                    'edge_features': graph_data.get('edge_features'),
                    'tpe': graph_data.get('tpe'),
                    'current_node_idx': graph_data.get('node_to_idx', {}).get(room_id, idx),
                },
                room_id=room_id,
                guidance_scale=guidance_scale,
                logic_guidance_scale=logic_guidance_scale,
                num_diffusion_steps=num_diffusion_steps,
                latent_sampler=latent_sampler,
                categorical_codebook_size=categorical_codebook_size,
                apply_repair=apply_repair,
                start_goal_coords=start_goal,
                seed=seed + room_id if seed is not None else None,
            )
            
            rooms[room_id] = room_result
            room_latents[room_id] = room_result.latent
        
        # Stitch rooms into complete dungeon
        dungeon_grid = self._stitch_rooms(rooms, mission_graph_physical)
        
        # BLOCK VII: MAP-Elites Evaluation
        map_elites_score = None
        if enable_map_elites:
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
        
        # Compute overall metrics
        generation_time = time.time() - start_time
        metrics = {
            'num_rooms': len(rooms),
            'total_tiles_repaired': sum(r.metrics.get('tiles_changed', 0) for r in rooms.values()),
            'repair_rate': sum(r.was_repaired for r in rooms.values()) / len(rooms),
            'dungeon_shape': dungeon_grid.shape,
            'generation_time_sec': generation_time,
        }
        
        logger.info(f"Dungeon generated in {generation_time:.2f}s "
                   f"(repair_rate={metrics['repair_rate']:.1%})")
        
        return DungeonGenerationResult(
            dungeon_grid=dungeon_grid,
            rooms=rooms,
            mission_graph=mission_graph,  # Preserve caller graph identity
            metrics=metrics,
            map_elites_score=map_elites_score,
            generation_time=generation_time,
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
            return {
                'node_features': empty_nodes,
                'edge_index': empty_edges,
                'edge_features': empty_edge_feats,
                'tpe': empty_tpe,
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
        for node_id, idx in node_to_idx.items():
            node_features[idx] = self._extract_node_feature_vector(graph.nodes[node_id])

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
        for nid in sorted(neighbor_ids):
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

        If graph node positions are available, use them. Otherwise fallback to
        deterministic vertical stacking (keeps legacy behavior for tests/tools).
        """
        if not rooms:
            return np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)

        # Prefer explicit node positions when present.
        placement: Dict[int, Tuple[int, int]] = {}
        occupied = set()
        for room_id in rooms.keys():
            pos = self._get_node_grid_position(graph, room_id)
            if pos is None:
                continue
            pos = self._first_free_position(pos, occupied)
            placement[room_id] = pos
            occupied.add(pos)

        # Fallback placement for nodes without coordinates.
        if graph.is_directed():
            try:
                order = [n for n in nx.topological_sort(graph) if n in rooms]
            except nx.NetworkXUnfeasible:
                order = sorted(rooms.keys())
        else:
            order = sorted(rooms.keys())

        next_row = max((r for r, _ in occupied), default=-1) + 1
        for room_id in order:
            if room_id in placement:
                continue
            while (next_row, 0) in occupied:
                next_row += 1
            placement[room_id] = (next_row, 0)
            occupied.add((next_row, 0))
            next_row += 1

        # Normalize to non-negative origin.
        min_r = min(r for r, _ in placement.values())
        min_c = min(c for _, c in placement.values())
        placement = {
            rid: (r - min_r, c - min_c)
            for rid, (r, c) in placement.items()
        }

        max_r = max(r for r, _ in placement.values())
        max_c = max(c for _, c in placement.values())
        global_grid = np.zeros(
            ((max_r + 1) * ROOM_HEIGHT, (max_c + 1) * ROOM_WIDTH),
            dtype=np.int32,
        )

        for room_id, (grid_r, grid_c) in placement.items():
            room_grid = self._fit_room_grid(rooms[room_id].room_grid)
            r0 = grid_r * ROOM_HEIGHT
            c0 = grid_c * ROOM_WIDTH
            global_grid[r0:r0 + ROOM_HEIGHT, c0:c0 + ROOM_WIDTH] = room_grid

        # Carve door openings for adjacent graph-connected rooms.
        for u, v in graph.edges():
            if u in placement and v in placement:
                self._carve_room_connection(global_grid, placement[u], placement[v])

        return global_grid
    
    def _validate_dungeon(self, dungeon_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Validate dungeon solvability and compute MAP-Elites descriptors.

        Uses the project validator when available, with graceful fallback.
        """
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

    def _carve_room_connection(
        self,
        global_grid: np.ndarray,
        src_pos: Tuple[int, int],
        dst_pos: Tuple[int, int],
    ) -> None:
        """Compatibility wrapper around extracted spatial helper."""
        carve_room_connection(global_grid, src_pos, dst_pos)


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
    'RoomGenerationResult',
    'DungeonGenerationResult',
    'create_pipeline',
]

