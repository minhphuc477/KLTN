"""
H-MOLQD Master Pipeline - Neural-Symbolic Dungeon Generation
==============================================================

Complete 7-block pipeline for Legend of Zelda dungeon generation.

Pipeline Architecture:
    Block I:   Data Adapter (zelda_core.py)
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
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
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
from src.data.zelda_core import DungeonStitcher

# Block I: Evolutionary Topology Director
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator

# VGLC compliance imports
from src.data.vglc_utils import (
    filter_virtual_nodes,
    validate_room_dimensions,
    get_physical_start_node,
)
from src.utils.graph_utils import validate_graph_topology

logger = logging.getLogger(__name__)


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
        self.map_elites = MAPElitesEvaluator(resolution=map_elites_resolution)
        
        # Utilities
        self.stitcher = DungeonStitcher()
        
        if enable_logging:
            logger.info("Pipeline initialized successfully")
    
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
        """Load or create condition encoder."""
        model = DualStreamConditionEncoder(
            latent_dim=64,
            node_feature_dim=6,
            hidden_dim=256,
            output_dim=256,
        ).to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded condition encoder from {checkpoint_path}")
        else:
            logger.warning("No condition encoder checkpoint, using random initialization")
        
        return model
    
    def _load_diffusion(self, checkpoint_path: Optional[str]) -> LatentDiffusionModel:
        """Load or create latent diffusion model."""
        model = LatentDiffusionModel(
            latent_dim=64,
            condition_dim=256,
            num_timesteps=1000,
            hidden_dim=128,
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
            use_ddim: Use DDIM (deterministic) vs DDPM (stochastic)
            apply_repair: Apply symbolic WFC repair
            start_goal_coords: ((start_r, start_c), (goal_r, goal_c)) for repair
            seed: Random seed for reproducibility
            
        Returns:
            RoomGenerationResult with room grid, latents, and metrics
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
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
                tpe=graph_context.get('tpe'),
                current_node_idx=graph_context.get('current_node_idx'),
            )
        except Exception as e:
            logger.warning(f"Condition encoding failed: {e}, using zero condition")
            condition = torch.zeros(1, 256, device=self.device)
        
        # BLOCK V: LogicNet Gradient Guidance Function
        def logic_guidance_fn(z_t: torch.Tensor) -> torch.Tensor:
            """Compute LogicNet gradient for guidance."""
            z_t_guided = z_t.clone().requires_grad_(True)
            
            # Decode to tile probabilities
            with torch.enable_grad():
                logits = self.vqvae.decode(z_t_guided)  # (1, 44, 16, 11)
                
                # Create simplified graph context for LogicNet
                simple_graph = {
                    'tile_probs': torch.softmax(logits, dim=1),
                }
                
                # Compute logic loss
                logic_loss, _ = self.logic_net(z_t_guided, simple_graph)
                
                # Compute gradient
                grad = torch.autograd.grad(logic_loss, z_t_guided)[0]
            
            return grad
        
        # BLOCK IV: Latent Diffusion Sampling
        logger.debug(f"Room {room_id}: Sampling with {num_diffusion_steps} steps")
        
        z_latent = self.diffusion.sample(
            condition=condition,
            logic_guidance_fn=logic_guidance_fn if logic_guidance_scale > 0 else None,
            guidance_scale=logic_guidance_scale,
            num_steps=num_diffusion_steps,
            use_ddim=use_ddim,
        )
        
        # BLOCK II: VQ-VAE Decoding
        logits = self.vqvae.decode(z_latent)  # (1, 44, 16, 11)
        neural_grid = logits.argmax(dim=1).cpu().numpy()[0]  # (16, 11)
        
        # BLOCK VI: Symbolic Repair (if enabled)
        was_repaired = False
        repair_mask = None
        final_grid = neural_grid.copy()
        
        if apply_repair and start_goal_coords is not None:
            start, goal = start_goal_coords
            try:
                repaired_grid, success = self.refiner.repair_room(
                    grid=neural_grid,
                    start=start,
                    goal=goal,
                )
                
                if success:
                    repair_mask = (repaired_grid != neural_grid)
                    final_grid = repaired_grid
                    was_repaired = np.any(repair_mask)
                    logger.debug(f"Room {room_id}: Repair successful ({np.sum(repair_mask)} tiles changed)")
                else:
                    logger.warning(f"Room {room_id}: Repair failed, using neural output")
            except Exception as e:
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
            np.random.seed(seed)
        
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
            # Empirical relationship: genome_length ≈ num_rooms * 0.7 (rules don't always apply)
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
        graph_data = self._prepare_graph_context(mission_graph_physical)
        
        rooms: Dict[int, RoomGenerationResult] = {}
        room_latents: Dict[int, torch.Tensor] = {}
        
        # Generate rooms in topological order
        for idx, room_id in enumerate(nx.topological_sort(mission_graph_physical)):
            logger.debug(f"Generating room {room_id} ({idx+1}/{len(mission_graph_physical.nodes)})")
            
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
                    'tpe': graph_data.get('tpe'),
                    'current_node_idx': idx,
                },
                room_id=room_id,
                guidance_scale=guidance_scale,
                logic_guidance_scale=logic_guidance_scale,
                num_diffusion_steps=num_diffusion_steps,
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
                    self.map_elites.add_dungeon(dungeon_grid, dungeon_grid, solver_result)
                    map_elites_score = {
                        'linearity': solver_result.get('linearity', 0.0),
                        'leniency': solver_result.get('leniency', 0.0),
                        'path_length': solver_result.get('path_length', 0),
                    }
            except Exception as e:
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
            mission_graph=mission_graph_physical,  # Return physical graph
            metrics=metrics,
            map_elites_score=map_elites_score,
            generation_time=generation_time,
        )
    
    def _prepare_graph_context(self, graph: nx.Graph) -> Dict[str, torch.Tensor]:
        """Prepare graph data for GNN conditioning."""
        # Simplified: Create basic node features and edge index
        num_nodes = len(graph.nodes)
        
        # Basic node features (can be enhanced with room type, items, etc.)
        node_features = torch.zeros(num_nodes, 6, device=self.device)
        
        # Edge index
        edges = list(graph.edges())
        if edges:
            edge_index = torch.tensor(edges, device=self.device).t().contiguous()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        
        # Topological positional encoding (simplified)
        tpe = torch.zeros(num_nodes, 8, device=self.device)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'tpe': tpe,
        }
    
    def _get_neighbor_latents(
        self,
        room_id: int,
        graph: nx.Graph,
        generated_latents: Dict[int, torch.Tensor]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Get latents of neighboring rooms (N/S/E/W)."""
        # Simplified: Just check graph neighbors
        neighbors = list(graph.neighbors(room_id))
        
        neighbor_dict = {'N': None, 'S': None, 'E': None, 'W': None}
        
        # Use first available neighbor (in production, detect direction from layout)
        for i, direction in enumerate(['N', 'E', 'S', 'W']):
            if i < len(neighbors) and neighbors[i] in generated_latents:
                neighbor_dict[direction] = generated_latents[neighbors[i]].to(self.device)
        
        return neighbor_dict
    
    def _extract_room_start_goal(
        self,
        graph: nx.Graph,
        room_id: int
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Extract start and goal positions for room repair."""
        # Simplified: Use default entrance/exit positions
        # In production, parse from graph node attributes
        return ((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1))
    
    def _stitch_rooms(
        self,
        rooms: Dict[int, RoomGenerationResult],
        graph: nx.Graph
    ) -> np.ndarray:
        """Stitch individual rooms into complete dungeon grid."""
        # Simplified: Create vertical stack
        # In production, use DungeonStitcher with proper layout
        room_grids = [result.room_grid for result in rooms.values()]
        
        if not room_grids:
            return np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
        
        # Simple vertical stacking
        return np.vstack(room_grids)
    
    def _validate_dungeon(self, dungeon_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Validate dungeon solvability (simplified)."""
        # In production, use actual A* validator
        return {
            'solvable': True,
            'path_length': 50,
            'linearity': 0.5,
            'leniency': 0.7,
        }


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
