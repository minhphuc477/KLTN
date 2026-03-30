"""
Training Pipeline for Latent Diffusion Model
=============================================

Full training pipeline connecting:
- LatentDiffusionModel for generation
- VQ-VAE for latent encoding
- LogicNet for solvability guidance
- DualStreamConditionEncoder for conditioning

Usage:
    python -m src.train_diffusion --data-dir "Data/The Legend of Zelda" --epochs 100
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zelda_data.zelda_loader import create_dataloader, extract_start_goal
from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion
from src.core.vqvae import SemanticVQVAE as VQVAE, create_vqvae
from src.core.condition_encoder import DualStreamConditionEncoder, create_condition_encoder
from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH
# Use Block V LogicNet (with temperature annealing), not legacy src.ml.logic_net
from src.core.logic_net import LogicNet
from src.utils.checkpoint import MetricsLogger, write_checkpoint_metadata

logger = logging.getLogger(__name__)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class DiffusionTrainingConfig:
    """Training configuration for latent diffusion."""
    
    def __init__(
        self,
        # Data
        data_dir: str = "Data/The Legend of Zelda",
        batch_size: int = 4,
        use_vglc: bool = True,
        room_level: bool = True,
        
        # VQ-VAE (frozen encoder)
        vqvae_checkpoint: Optional[str] = None,
        
        # Diffusion Model
        latent_dim: int = 64,
        model_channels: int = 128,
        context_dim: int = 256,
        condition_gnn_type: str = "gcn",  # gcn | gat | sage
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        topology_refinement_mode: str = "gat2",  # none | lightweight | gat2
        
        # LogicNet
        num_logic_iterations: int = 30,
        guidance_scale: float = 1.0,
        
        # Training
        epochs: int = 100,
        learning_rate: float = 1e-4,
        alpha_visual: float = 1.0,   # Diffusion loss weight
        alpha_logic: float = 0.1,     # Solvability loss weight
        logic_loss_mode: str = "predicted_latent",  # predicted_latent | detached_real
        graph_conditioning_mode: str = "node_sequence",  # node_sequence | pooled
        warmup_epochs: int = 5,       # Epochs before adding logic loss
        
        # Checkpointing
        checkpoint_dir: str = "./checkpoints",
        save_every: int = 10,
        
        # Device
        device: str = "auto",
        
        # Quick mode
        quick: bool = False,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_vglc = use_vglc
        self.room_level = bool(room_level)
        
        self.vqvae_checkpoint = vqvae_checkpoint
        
        self.latent_dim = latent_dim
        self.model_channels = model_channels
        self.context_dim = context_dim
        gnn_type = str(condition_gnn_type).strip().lower()
        if gnn_type not in {"gcn", "gat", "sage"}:
            raise ValueError(
                f"Invalid condition_gnn_type={condition_gnn_type!r}. "
                "Expected 'gcn', 'gat', or 'sage'."
            )
        self.condition_gnn_type = gnn_type
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        trm = str(topology_refinement_mode).strip().lower()
        if trm == "upgraded":
            trm = "gat2"
        if trm not in {"none", "lightweight", "gat2"}:
            raise ValueError(
                f"Invalid topology_refinement_mode={topology_refinement_mode!r}. "
                "Expected 'none', 'lightweight', or 'gat2'."
            )
        self.topology_refinement_mode = trm
        
        self.num_logic_iterations = num_logic_iterations
        self.guidance_scale = guidance_scale
        
        self.epochs = epochs if not quick else 2
        self.learning_rate = learning_rate
        self.alpha_visual = alpha_visual
        self.alpha_logic = alpha_logic
        mode = str(logic_loss_mode).strip().lower()
        if mode not in {"predicted_latent", "detached_real"}:
            raise ValueError(
                f"Invalid logic_loss_mode={logic_loss_mode!r}. "
                "Expected 'predicted_latent' or 'detached_real'."
            )
        self.logic_loss_mode = mode
        gmode = str(graph_conditioning_mode).strip().lower()
        if gmode not in {"node_sequence", "pooled"}:
            raise ValueError(
                f"Invalid graph_conditioning_mode={graph_conditioning_mode!r}. "
                "Expected 'node_sequence' or 'pooled'."
            )
        self.graph_conditioning_mode = gmode
        self.warmup_epochs = warmup_epochs
        
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.quick = quick
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# =============================================================================
# INTEGRATED DIFFUSION TRAINER
# =============================================================================

class DiffusionTrainer:
    """
    Unified trainer for latent diffusion dungeon generation.
    
    Components:
    1. VQ-VAE: Encode real dungeons to latent space (frozen)
    2. ConditionEncoder: Process graph + spatial context
    3. LatentDiffusion: Generate dungeons in latent space
    4. LogicNet: Differentiable solvability (gradient guidance)
    """
    
    def __init__(
        self,
        config: DiffusionTrainingConfig,
        vqvae: Optional[VQVAE] = None,
        diffusion: Optional[LatentDiffusionModel] = None,
        condition_encoder: Optional[DualStreamConditionEncoder] = None,
        logic_net: Optional[LogicNet] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.vqvae = vqvae or self._create_vqvae()
        self.diffusion = diffusion or self._create_diffusion()
        self.condition_encoder = condition_encoder or self._create_condition_encoder()
        self.logic_net = logic_net or self._create_logic_net()
        
        # Move to device
        self.vqvae = self.vqvae.to(self.device)
        self.diffusion = self.diffusion.to(self.device)
        self.condition_encoder = self.condition_encoder.to(self.device)
        self.logic_net = self.logic_net.to(self.device)
        
        # Freeze VQ-VAE
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False
        
        # --- Wire LogicNet into diffusion model's GradientGuidance ---
        # This enables gradient guidance during sampling: at each denoising
        # step, âˆ‡_{x_t}L_logic nudges the sample toward solvable configs.
        self.diffusion.guidance.logic_net = self.logic_net
        self.diffusion.guidance.guidance_scale = config.guidance_scale
        
        # Setup optimizer: train diffusion + condition encoder
        # Note: LogicNet is now a submodule of diffusion.guidance, so its
        # parameters are already included in self.diffusion.parameters().
        self.optimizer = optim.AdamW(
            list(self.diffusion.parameters()) + 
            list(self.condition_encoder.parameters()),
            lr=config.learning_rate,
            weight_decay=1e-5,
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        
        # Metrics tracking
        self.epoch = 0
        self.global_step = 0
        
        # --- Phase 4A: EMA model weights ---
        import copy
        self.ema_diffusion = copy.deepcopy(self.diffusion)
        self.ema_diffusion.eval()
        for param in self.ema_diffusion.parameters():
            param.requires_grad = False
        self.ema_decay = 0.9999
    
    def _create_vqvae(self) -> VQVAE:
        """Create or load VQ-VAE."""
        # CRITICAL-1 fix: create_vqvae expects num_classes (default 44), not in_channels
        vqvae = create_vqvae(
            num_classes=44,
            latent_dim=self.config.latent_dim,
        )
        
        if self.config.vqvae_checkpoint:
            checkpoint = torch.load(self.config.vqvae_checkpoint, map_location='cpu')
            vqvae.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded VQ-VAE from {self.config.vqvae_checkpoint}")
        
        return vqvae
    
    def _create_diffusion(self) -> LatentDiffusionModel:
        """Create latent diffusion model."""
        return create_latent_diffusion(
            latent_dim=self.config.latent_dim,
            model_channels=self.config.model_channels,
            context_dim=self.config.context_dim,
            num_timesteps=self.config.num_timesteps,
            schedule_type=self.config.schedule_type,
            topology_refinement_mode=self.config.topology_refinement_mode,
        )
    
    def _create_condition_encoder(self) -> DualStreamConditionEncoder:
        """Create condition encoder."""
        return create_condition_encoder(
            latent_dim=self.config.latent_dim,
            output_dim=self.config.context_dim,
            gnn_type=self.config.condition_gnn_type,
        )

    def _stack_conditioning_vectors(self, cond_vectors: List[torch.Tensor]) -> torch.Tensor:
        """Stack per-sample conditioning vectors into batch tensor for diffusion."""
        if not cond_vectors:
            raise ValueError("cond_vectors must be non-empty")

        if self.config.graph_conditioning_mode == "node_sequence":
            max_nodes = max(int(c.shape[0]) for c in cond_vectors)
            padded = []
            for c in cond_vectors:
                if c.shape[0] < max_nodes:
                    pad = torch.zeros(
                        max_nodes - c.shape[0],
                        c.shape[1],
                        device=c.device,
                        dtype=c.dtype,
                    )
                    c = torch.cat([c, pad], dim=0)
                padded.append(c.unsqueeze(0))
            return torch.cat(padded, dim=0)  # [B, N_max, context_dim]

        return torch.cat(cond_vectors, dim=0)  # [B, context_dim]
    
    def _create_logic_net(self) -> LogicNet:
        """Create LogicNet for solvability."""
        return LogicNet(
            latent_dim=self.config.latent_dim,
            num_classes=44,
            num_iterations=self.config.num_logic_iterations,
        )
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to VQ-VAE latent space.
        
        Handles data format conversion:
        - Data loader returns [B, 1, H, W] normalized tile IDs in [0, 1]
        - VQ-VAE expects [B, C=44, H, W] one-hot encoded tiles
        
        Conversion: denormalize â†’ integer tile IDs â†’ one-hot â†’ VQ-VAE encode
        """
        import torch.nn.functional as F
        
        with torch.no_grad():
            num_classes = self.vqvae.num_classes  # 44
            
            if x.shape[1] == 1:
                # Data loader format: [B, 1, H, W] normalized [0, 1]
                # Step 1: Denormalize to integer tile IDs
                tile_ids = (x.squeeze(1) * (num_classes - 1)).round().long()
                tile_ids = tile_ids.clamp(0, num_classes - 1)
                
                # Step 2: One-hot encode â†’ [B, H, W, C] â†’ permute to [B, C, H, W]
                x_onehot = F.one_hot(tile_ids, num_classes=num_classes)
                x_onehot = x_onehot.permute(0, 3, 1, 2).float()
            elif x.shape[1] == num_classes:
                # Already one-hot: [B, C, H, W]
                x_onehot = x
            else:
                raise ValueError(
                    f"Unexpected input channels {x.shape[1]}. "
                    f"Expected 1 (normalized tile IDs) or {num_classes} (one-hot)."
                )
            
            # encode() returns (z_q, indices) â€” 2 values, not 3
            z_q, _indices = self.vqvae.encode(x_onehot)
        return z_q
    
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes back to tile logits.
        
        Returns:
            Tensor [B, C=44, H, W] of tile class logits
        """
        with torch.no_grad():
            return self.vqvae.decode(z, target_size=(ROOM_HEIGHT, ROOM_WIDTH))

    def _encode_edge_features(self, graph_dict: dict) -> Optional[torch.Tensor]:
        """Convert integer edge labels to one-hot features for the condition encoder."""
        edge_attr = graph_dict.get('edge_attr')
        if edge_attr is None:
            return None
        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        if edge_attr.numel() == 0:
            return None
        edge_attr = edge_attr.to(self.device)
        num_edge_types = 8
        edge_attr_clamped = edge_attr.clamp(0, num_edge_types - 1)
        return torch.nn.functional.one_hot(
            edge_attr_clamped, num_classes=num_edge_types
        ).float()
    
    def get_dummy_conditioning(self, batch_size: int) -> torch.Tensor:
        """
        Get fallback conditioning when graph data is unavailable.
        
        Used only as a fallback during validation or when graph loading fails.
        During training, real graph data from .dot files is used instead.
        """
        if self.config.graph_conditioning_mode == "node_sequence":
            return torch.randn(batch_size, 1, self.config.context_dim, device=self.device)
        return torch.randn(batch_size, self.config.context_dim, device=self.device)

    def _prediction_to_x0(
        self,
        prediction: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert diffusion model prediction to predicted clean latent x0."""
        sqrt_alpha_t = self.diffusion.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        if self.diffusion.prediction_type == 'v':
            # v-prediction: x0 = sqrt(alpha_bar_t) * x_t - sqrt(1-alpha_bar_t) * v
            return sqrt_alpha_t * x_t - sqrt_one_minus_alpha_t * prediction

        # epsilon-prediction: x0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        return (x_t - sqrt_one_minus_alpha_t * prediction) / (sqrt_alpha_t + 1e-8)
    
    def _encode_graph_conditioning(
        self,
        graph_dict: dict,
    ) -> torch.Tensor:
        """
        Encode a single graph dict into a conditioning vector using the GNN.
        
        Args:
            graph_dict: Dict from zelda_loader with:
                - node_features: [N, 6]
                - edge_index: [2, E]
                - edge_attr: [E] edge type labels
                
        Returns:
            If graph_conditioning_mode='pooled': [1, context_dim]
            If graph_conditioning_mode='node_sequence': [N, context_dim]
        """
        node_features = graph_dict['node_features'].to(self.device)
        edge_index = graph_dict['edge_index'].to(self.device)
        
        edge_features = self._encode_edge_features(graph_dict)
        c_global = self.condition_encoder.encode_global_only(
            node_features, edge_index,
            edge_features=edge_features,
            tpe=graph_dict.get('tpe').to(self.device) if isinstance(graph_dict.get('tpe'), torch.Tensor) else None,
        )

        boundary_constraints = graph_dict.get("boundary_constraints")
        room_position = graph_dict.get("room_position")
        current_node_idx = graph_dict.get("current_node_idx")
        has_room_anchor = bool(graph_dict.get("has_room_anchor", False)) or (
            isinstance(boundary_constraints, torch.Tensor)
            and isinstance(room_position, torch.Tensor)
        )
        if has_room_anchor:
            boundary_constraints = boundary_constraints.to(self.device, dtype=torch.float32)
            room_position = room_position.to(self.device, dtype=torch.float32)
            if boundary_constraints.dim() == 1:
                boundary_constraints = boundary_constraints.unsqueeze(0)
            if room_position.dim() == 1:
                room_position = room_position.unsqueeze(0)
            room_anchor = self.condition_encoder(
                neighbor_latents={'N': None, 'S': None, 'E': None, 'W': None},
                boundary_constraints=boundary_constraints,
                position=room_position,
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                tpe=graph_dict.get('tpe').to(self.device) if isinstance(graph_dict.get('tpe'), torch.Tensor) else None,
                current_node_idx=int(current_node_idx) if current_node_idx is not None else None,
            )
            if self.config.graph_conditioning_mode == "node_sequence":
                return torch.cat([room_anchor, c_global], dim=0)
            return room_anchor

        if self.config.graph_conditioning_mode == "node_sequence":
            return c_global

        # Pooled baseline.
        return c_global.mean(dim=0, keepdim=True)

    def _normalize_diffusion_graph_sample(self, graph_dict: dict) -> Dict[str, torch.Tensor]:
        """Prepare one graph sample for diffusion spatial/topological conditioning."""
        node_features = graph_dict["node_features"]
        edge_index = graph_dict["edge_index"]
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_features = node_features.to(self.device, dtype=torch.float32)
        edge_index = edge_index.to(self.device, dtype=torch.long)
        if node_features.dim() != 2:
            raise ValueError(f"node_features must have shape [N, F], got {tuple(node_features.shape)}")
        if edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

        num_nodes = int(node_features.shape[0])
        tpe = graph_dict.get("tpe")
        if not isinstance(tpe, torch.Tensor):
            tpe = torch.zeros(num_nodes, 8, dtype=torch.float32)
        tpe = tpe.to(self.device, dtype=torch.float32)
        if tpe.dim() == 1:
            tpe = tpe.unsqueeze(0)
        if tpe.dim() != 2 or int(tpe.shape[0]) != num_nodes:
            raise ValueError(
                f"tpe must have shape [N, D] matching node_features; got {tuple(tpe.shape)} for N={num_nodes}."
            )

        node_positions = graph_dict.get("node_positions")
        if not isinstance(node_positions, torch.Tensor):
            node_positions = torch.stack(
                [
                    torch.arange(num_nodes, dtype=torch.float32),
                    torch.zeros(num_nodes, dtype=torch.float32),
                ],
                dim=1,
            ) if num_nodes > 0 else torch.zeros((0, 2), dtype=torch.float32)
        node_positions = node_positions.to(self.device, dtype=torch.float32)
        if node_positions.dim() == 1:
            node_positions = node_positions.view(-1, 2)
        if node_positions.dim() != 2 or int(node_positions.shape[0]) != num_nodes or int(node_positions.shape[1]) != 2:
            raise ValueError(
                f"node_positions must have shape [N, 2] matching node_features; got {tuple(node_positions.shape)} for N={num_nodes}."
            )

        node_mask = graph_dict.get("node_mask")
        if not isinstance(node_mask, torch.Tensor):
            node_mask = torch.ones(num_nodes, dtype=torch.float32)
        node_mask = node_mask.to(self.device, dtype=torch.float32)
        if node_mask.dim() == 2:
            if int(node_mask.shape[0]) != 1:
                raise ValueError(f"Unbatched diffusion sample cannot provide multi-row node_mask: {tuple(node_mask.shape)}")
            node_mask = node_mask.squeeze(0)
        if node_mask.dim() != 1 or int(node_mask.shape[0]) != num_nodes:
            raise ValueError(
                f"node_mask must have shape [N] matching node_features; got {tuple(node_mask.shape)} for N={num_nodes}."
            )

        room_topology_map = graph_dict.get("room_topology_map")
        if isinstance(room_topology_map, torch.Tensor):
            room_topology_map = room_topology_map.to(self.device, dtype=torch.float32)
            if room_topology_map.dim() == 4:
                if int(room_topology_map.shape[0]) != 1:
                    raise ValueError(
                        f"Single graph sample room_topology_map must be [C,H,W] or [1,C,H,W], got {tuple(room_topology_map.shape)}."
                    )
                room_topology_map = room_topology_map.squeeze(0)
            if room_topology_map.dim() != 3:
                raise ValueError(
                    f"room_topology_map must have shape [C,H,W] for one sample, got {tuple(room_topology_map.shape)}."
                )

        has_room_anchor = bool(graph_dict.get("has_room_anchor", False)) or (
            isinstance(graph_dict.get("boundary_constraints"), torch.Tensor)
            and isinstance(graph_dict.get("room_position"), torch.Tensor)
        )

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "tpe": tpe,
            "node_positions": node_positions,
            "node_mask": node_mask,
            "has_room_anchor": has_room_anchor,
            **({"room_topology_map": room_topology_map} if isinstance(room_topology_map, torch.Tensor) else {}),
        }

    def _stack_diffusion_graph_batch(self, graph_list: List[dict]) -> Optional[Dict[str, torch.Tensor]]:
        """Pad a batch of variable-size graph tensors for diffusion conditioning."""
        if not graph_list:
            return None

        samples = [self._normalize_diffusion_graph_sample(graph_dict) for graph_dict in graph_list]
        if not samples:
            return None

        anchor_flags = {bool(sample.get("has_room_anchor", False)) for sample in samples}
        if len(anchor_flags) > 1:
            raise ValueError(
                "Mixed graph anchor semantics in one diffusion batch. "
                "All samples must either include a room anchor token or omit it."
            )

        max_nodes = max(int(sample["node_features"].shape[0]) for sample in samples)
        feat_dim = max(int(sample["node_features"].shape[1]) if sample["node_features"].dim() == 2 else 0 for sample in samples)
        tpe_dim = max(int(sample["tpe"].shape[1]) if sample["tpe"].dim() == 2 else 0 for sample in samples)
        pos_dim = max(int(sample["node_positions"].shape[1]) if sample["node_positions"].dim() == 2 else 0 for sample in samples)
        max_edges = max(int(sample["edge_index"].shape[1]) if sample["edge_index"].dim() == 2 else 0 for sample in samples)

        node_features_batch = torch.zeros(len(samples), max_nodes, max(1, feat_dim), device=self.device, dtype=torch.float32)
        tpe_batch = torch.zeros(len(samples), max_nodes, max(1, tpe_dim), device=self.device, dtype=torch.float32)
        node_positions_batch = torch.zeros(len(samples), max_nodes, max(1, pos_dim), device=self.device, dtype=torch.float32)
        node_mask_batch = torch.zeros(len(samples), max_nodes, device=self.device, dtype=torch.float32)
        edge_index_batch = torch.full((len(samples), 2, max_edges), -1, device=self.device, dtype=torch.long)

        topo_maps = []
        has_topology = [("room_topology_map" in sample) for sample in samples]
        if any(has_topology) and not all(has_topology):
            raise ValueError(
                "room_topology_map must be present for every graph in a diffusion batch or omitted for all of them."
            )
        can_stack_topology = all(has_topology)
        topo_shape = None

        for i, sample in enumerate(samples):
            num_nodes = int(sample["node_features"].shape[0])
            if num_nodes > 0:
                node_features_batch[i, :num_nodes, : sample["node_features"].shape[1]] = sample["node_features"]
                tpe_batch[i, :num_nodes, : sample["tpe"].shape[1]] = sample["tpe"]
                node_positions_batch[i, :num_nodes, : sample["node_positions"].shape[1]] = sample["node_positions"]
                node_mask_batch[i, :num_nodes] = sample["node_mask"]

            num_edges = int(sample["edge_index"].shape[1]) if sample["edge_index"].dim() == 2 else 0
            if num_edges > 0:
                edge_index_batch[i, :, :num_edges] = sample["edge_index"]

            if can_stack_topology:
                topo = sample["room_topology_map"]
                if topo.dim() == 3:
                    topo = topo.unsqueeze(0)
                current_shape = tuple(topo.shape[1:])
                if topo_shape is None:
                    topo_shape = current_shape
                if current_shape != topo_shape:
                    if not bool(getattr(self, "_topology_shape_mismatch_warning_emitted", False)):
                        logger.warning(
                            "Disabling batched room_topology_map stacking due to shape mismatch: "
                            "expected %s, got %s. Topology conditioning will be omitted for this batch.",
                            str(topo_shape),
                            str(current_shape),
                        )
                        self._topology_shape_mismatch_warning_emitted = True
                    can_stack_topology = False
                    topo_maps = []
                else:
                    topo_maps.append(topo)

        batch_graph = {
            "node_features": node_features_batch,
            "edge_index": edge_index_batch,
            "tpe": tpe_batch,
            "node_positions": node_positions_batch,
            "node_mask": node_mask_batch,
            "has_room_anchor": bool(next(iter(anchor_flags))) if anchor_flags else False,
        }
        if can_stack_topology and topo_maps:
            batch_graph["room_topology_map"] = torch.cat(topo_maps, dim=0)
        return batch_graph
    
    def _build_logic_graph_data(
        self,
        graph_dict: dict,
    ) -> dict:
        """
        Convert a dataset graph_dict to LogicNet's expected format.
        
        LogicNet expects:
            adjacency: [N, N] adjacency matrix
            edge_weights: [N, N] traversal costs (1.0 for open, 2.0 for locked)
            start_idx: int
            target_idx: int (triforce room)
            key_lock_pairs: List[(key_node, lock_target)]
            
        Args:
            graph_dict: Dict from zelda_loader
            
        Returns:
            Dict for LogicNet.forward() or None if graph is empty
        """
        num_nodes = graph_dict['num_nodes']
        if num_nodes == 0:
            return None
        
        edge_index = graph_dict['edge_index']
        edge_attr = graph_dict.get('edge_attr')
        node_features = graph_dict['node_features']
        
        # Build adjacency matrix
        adjacency = torch.zeros(num_nodes, num_nodes, device=self.device)
        edge_weights = torch.zeros(num_nodes, num_nodes, device=self.device)
        
        key_lock_pairs = []
        
        if edge_index.numel() > 0:
            edge_index_dev = edge_index.to(self.device)
            src_indices = edge_index_dev[0]
            dst_indices = edge_index_dev[1]
            
            # Vectorized adjacency construction
            adjacency[src_indices, dst_indices] = 1.0
            
            # Vectorized edge weights: default=1.0, key_locked=2.0, boss_locked=3.0
            edge_weights[src_indices, dst_indices] = 1.0
            if edge_attr is not None:
                edge_attr_dev = edge_attr.to(self.device) if isinstance(edge_attr, torch.Tensor) else torch.tensor(edge_attr, device=self.device)
                num_edges = min(len(edge_attr_dev), edge_index_dev.shape[1])
                key_locked_mask = edge_attr_dev[:num_edges] == 1
                boss_locked_mask = edge_attr_dev[:num_edges] == 4
                if key_locked_mask.any():
                    edge_weights[src_indices[:num_edges][key_locked_mask], dst_indices[:num_edges][key_locked_mask]] = 2.0
                if boss_locked_mask.any():
                    edge_weights[src_indices[:num_edges][boss_locked_mask], dst_indices[:num_edges][boss_locked_mask]] = 3.0
        
        # Find start and target nodes from node_features
        start_idx = graph_dict.get('start_node_id', 0)
        if start_idx < 0:
            start_idx = 0
        
        target_idx = None
        node_feats = node_features if isinstance(node_features, torch.Tensor) else torch.tensor(node_features)
        for i in range(num_nodes):
            # has_triforce is feature[3]
            if node_feats[i, 3] > 0.5:
                target_idx = i
                break
        
        # Find key-lock pairs:
        # Key nodes (feature[1] = has_key) should be reachable before locked doors
        key_nodes = [i for i in range(num_nodes) if node_feats[i, 1] > 0.5]
        # Lock targets: rooms behind key-locked edges
        lock_targets = set()
        if edge_index.numel() > 0 and edge_attr is not None:
            for e in range(edge_index.shape[1]):
                if edge_attr[e].item() == 1:  # key_locked
                    lock_targets.add(edge_index[1, e].item())
        
        # Pair keys to locks (simple: pair by order)
        lock_list = sorted(lock_targets)
        for i, key_node in enumerate(key_nodes):
            if i < len(lock_list):
                key_lock_pairs.append((key_node, lock_list[i]))
        
        return {
            'adjacency': adjacency,
            'edge_weights': edge_weights,
            'start_idx': start_idx,
            'target_idx': target_idx,
            'key_lock_pairs': key_lock_pairs,
        }

    @staticmethod
    def _logic_loss_to_solvability_proxy(logic_loss: torch.Tensor) -> torch.Tensor:
        """
        Convert unbounded non-negative LogicNet loss into a bounded proxy score.

        We use exp(-loss) so the proxy stays in (0, 1], decreases smoothly as
        constraints are violated, and never becomes negative.
        """
        if not isinstance(logic_loss, torch.Tensor):
            logic_loss = torch.tensor(float(logic_loss), dtype=torch.float32)
        return torch.exp(-logic_loss.detach().clamp_min(0.0))
    
    def _update_ema(self):
        """
        Update EMA model weights (Phase 4A).
        
        EMA provides more stable weights for sampling/validation.
        Standard in all modern diffusion training (Nichol & Dhariwal, 2021).
        """
        with torch.no_grad():
            for p_ema, p in zip(self.ema_diffusion.parameters(),
                                self.diffusion.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)
    
    def train_step(
        self,
        real_maps: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        include_logic_loss: bool = True,
        logic_graph_data: Optional[dict] = None,
        diffusion_graph_data: Optional[dict] = None,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Training strategy:
        1. Diffusion loss: standard Îµ-prediction on real encoded latents
        2. LogicNet loss: computed on REAL z_0 with graph_data from .dot files,
           enabling both grid-level AND graph-level pathfinding/key-lock checking.
        3. GradientGuidance (wired in __init__): applies âˆ‡_{x_t}L_logic
           during sampling/validation to steer generation toward solvable maps.
        
        Args:
            real_maps: [B, 1, H, W] real dungeon maps
            conditioning: [B, context_dim] conditioning vectors from real graphs
            include_logic_loss: Whether to train LogicNet on real data
            logic_graph_data: Graph data dict for LogicNet (from _build_logic_graph_data)
            
        Returns:
            Dict of loss values
        """
        self.diffusion.train()
        self.condition_encoder.train()
        
        batch_size = real_maps.shape[0]
        
        # Get conditioning (use fallback if not provided)
        if conditioning is None:
            conditioning = self.get_dummy_conditioning(batch_size)
        
        # Encode to latent space
        z_0 = self.encode_to_latent(real_maps)
        
        # === Part 1: Diffusion loss (standard noise prediction) ===
        diffusion_loss = self.diffusion.training_loss(z_0, conditioning, graph_data=diffusion_graph_data)
        
        # === Part 2: LogicNet loss on model-predicted latent WITH graph topology ===
        # IMPORTANT: computing logic loss on detached real z_0 does not train diffusion.
        # We instead denoise a noisy latent and apply LogicNet to predicted x0 so
        # logic gradients flow into diffusion + condition encoder.
        logic_loss = torch.tensor(0.0, device=self.device)
        solvability_proxy = torch.tensor(0.0, device=self.device)
        
        if include_logic_loss and self.config.alpha_logic > 0:
            if self.config.logic_loss_mode == "detached_real":
                # Legacy baseline: logic regularization on real latent only.
                z_for_logic = z_0.detach().requires_grad_(True)
                logic_loss, _logic_info = self.logic_net(z_for_logic, graph_data=logic_graph_data)
            else:
                # New default: logic supervision on predicted latent (trains diffusion).
                t_logic = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
                noise_logic = torch.randn_like(z_0)
                x_t_logic = self.diffusion.q_sample(z_0, t_logic, noise_logic)

                # Predict noise/velocity and convert to predicted clean latent x0.
                context_edge_index, context_node_mask = self.diffusion._extract_context_topology(
                    conditioning,
                    diffusion_graph_data,
                )
                spatial_graph_data = self.diffusion._extract_spatial_graph_context(
                    conditioning,
                    diffusion_graph_data,
                )
                pred_logic = self.diffusion.denoiser(
                    x_t_logic,
                    t_logic,
                    conditioning,
                    context_edge_index=context_edge_index,
                    context_node_mask=context_node_mask,
                    spatial_graph_data=spatial_graph_data,
                )
                pred_x0_logic = self._prediction_to_x0(pred_logic, x_t_logic, t_logic)

                # Keep latent range bounded similarly to sampling path.
                pred_x0_logic = torch.clamp(pred_x0_logic, -1.0, 1.0)

                # Pass predicted latent to LogicNet for graph-level pathfinding loss.
                logic_loss, _logic_info = self.logic_net(pred_x0_logic, graph_data=logic_graph_data)
            solvability_proxy = self._logic_loss_to_solvability_proxy(logic_loss)
        
        # Combined loss
        total_loss = (
            self.config.alpha_visual * diffusion_loss + 
            self.config.alpha_logic * logic_loss
        )
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.diffusion.parameters()) + 
            list(self.condition_encoder.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        # --- Phase 4A: Update EMA model weights ---
        self._update_ema()
        
        # --- Phase 1D: Anneal LogicNet temperature ---
        # Use estimated total steps from config instead of hardcoded epochs*100
        if hasattr(self.logic_net, 'update_temperature'):
            estimated_total_steps = max(1, getattr(self, '_estimated_total_steps', self.config.epochs * 100))
            progress = min(1.0, self.global_step / estimated_total_steps)
            self.logic_net.update_temperature(progress)
        
        self.global_step += 1
        
        return {
            'loss': total_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'logic_loss': logic_loss.item(),
            'solvability_proxy': solvability_proxy.item(),
            'solvability': solvability_proxy.item(),
            'logic_loss_mode_predicted': 1.0 if self.config.logic_loss_mode == 'predicted_latent' else 0.0,
        }
    
    def _extract_coords_from_maps(self, real_maps: torch.Tensor) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        """Extract start/goal coordinates from map tensors. Fallback to defaults."""
        start, goal = extract_start_goal(real_maps[0])
        return (start if start else (2, 2)), (goal if goal else (13, 8))

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch using real graph data from .dot files.
        
        The dataloader returns (images, graph_list) when load_graphs=True.
        Each graph in graph_list is a dict from zelda_loader._extract_graph()
        containing real mission topology from the VGLC .dot files.
        """
        metrics_sum = {'loss': 0, 'diffusion_loss': 0, 'logic_loss': 0, 'solvability_proxy': 0, 'solvability': 0}
        num_batches = 0
        
        # DESIGN-08: Compute actual total training steps for temperature annealing
        self._estimated_total_steps = max(1, self.config.epochs * len(dataloader))
        
        include_logic = self.epoch >= self.config.warmup_epochs
        total_epochs = int(getattr(self.config, "epochs", self.epoch + 1))
        logger.info(
            "Train epoch %d/%d: logic_loss_%s (warmup_epochs=%d)",
            int(self.epoch + 1),
            total_epochs,
            "enabled" if include_logic and self.config.alpha_logic > 0 else "disabled",
            int(self.config.warmup_epochs),
        )
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Handle (images, graph_list) from graph_collate_fn
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                real_maps, graph_list = batch_data
            else:
                real_maps = batch_data
                graph_list = None
            real_maps = real_maps.to(self.device)
            
            # === Build conditioning from REAL graph data ===
            conditioning = None
            logic_graph_data = None
            diffusion_graph_data = None
            
            if graph_list is not None and self.condition_encoder is not None:
                try:
                    # Encode each graph through GNN and stack.
                    cond_vectors = []
                    for graph_dict in graph_list:
                        c_i = self._encode_graph_conditioning(graph_dict)
                        cond_vectors.append(c_i)
                    conditioning = self._stack_conditioning_vectors(cond_vectors)
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    logger.debug(f"Graph conditioning failed: {e}")
                    conditioning = None

                try:
                    diffusion_graph_data = self._stack_diffusion_graph_batch(graph_list)
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    logger.debug(f"Diffusion graph-data build failed: {e}")
                    diffusion_graph_data = None
                
                # Build LogicNet graph data from first graph in batch
                # (LogicNet processes single graphs, not batches)
                if include_logic:
                    try:
                        logic_graph_data = self._build_logic_graph_data(graph_list[0])
                    except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                        logger.debug(f"Logic graph build failed: {e}")
                        logic_graph_data = None
            
            metrics = self.train_step(
                real_maps,
                conditioning=conditioning,
                include_logic_loss=include_logic,
                logic_graph_data=logic_graph_data,
                diffusion_graph_data=diffusion_graph_data,
            )
            
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v)
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx}: loss={metrics['loss']:.4f}, "
                    f"diffusion={metrics['diffusion_loss']:.4f}, "
                    f"train_solvability_proxy={metrics.get('solvability_proxy', metrics['solvability']):.4f}, "
                    f"logic_loss={metrics['logic_loss']:.4f}"
                )
        
        self.epoch += 1
        self.scheduler.step()
        
        return {k: v / max(num_batches, 1) for k, v in metrics_sum.items()}
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        num_samples: int = 4,
    ) -> Dict[str, float]:
        """Validate model using EMA weights and real graph conditioning."""
        eval_model = self.ema_diffusion if hasattr(self, 'ema_diffusion') else self.diffusion
        eval_model.eval()
        
        total_logic_loss = 0.0
        total_solvability_proxy = 0.0
        num_samples_eval = 0
        
        for batch_data in dataloader:
            # Handle (images, graph_list) from graph_collate_fn
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                real_maps = batch_data[0]
                graph_list = batch_data[1] if len(batch_data) > 1 else None
            else:
                real_maps = batch_data
                graph_list = None
            real_maps = real_maps.to(self.device)
            batch_size = real_maps.shape[0]
            
            # Build conditioning from real graphs if available
            conditioning = None
            diffusion_graph_data = None
            if graph_list is not None:
                cond_vectors = []
                for idx, graph_dict in enumerate(graph_list):
                    try:
                        c_i = self._encode_graph_conditioning(graph_dict)
                    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                        logger.debug(
                            "Graph conditioning encode failed for sample %d; using dummy conditioning: %s",
                            idx,
                            exc,
                        )
                        c_i = self.get_dummy_conditioning(1)
                    cond_vectors.append(c_i)
                if cond_vectors:
                    conditioning = self._stack_conditioning_vectors(cond_vectors)
                try:
                    diffusion_graph_data = self._stack_diffusion_graph_batch(graph_list)
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    logger.debug("Diffusion graph-data build failed during validation: %s", exc)
                    diffusion_graph_data = None
            
            if conditioning is None:
                conditioning = self.get_dummy_conditioning(batch_size)
            
            # Encode real maps to get latent shape
            z_0 = self.encode_to_latent(real_maps)
            
            # Generate samples using EMA model
            z_gen = eval_model.sample(conditioning, shape=z_0.shape, graph_data=diffusion_graph_data)
            
            # Build LogicNet graph data if available
            logic_graph_data = None
            if graph_list is not None:
                build_failures = 0
                for graph_dict in graph_list:
                    try:
                        logic_graph_data = self._build_logic_graph_data(graph_dict)
                        break
                    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                        build_failures += 1
                        logger.debug("LogicNet graph-data build failed for one sample: %s", exc)
                if logic_graph_data is None and build_failures > 0:
                    logger.debug(
                        "LogicNet graph-data unavailable for this validation batch (%d failures); proceeding without graph_data",
                        build_failures,
                    )
            
            # LogicNet: evaluate with graph topology
            logic_loss, _logic_info = self.logic_net(z_gen, graph_data=logic_graph_data)
            solvability_proxy = float(self._logic_loss_to_solvability_proxy(logic_loss).item())
            total_logic_loss += float(logic_loss.item()) * batch_size
            total_solvability_proxy += solvability_proxy * batch_size
            num_samples_eval += batch_size
            
            if num_samples_eval >= num_samples:
                break
        
        return {
            'val_logic_loss': total_logic_loss / max(num_samples_eval, 1),
            'val_solvability_proxy': total_solvability_proxy / max(num_samples_eval, 1),
            'val_solvability': total_solvability_proxy / max(num_samples_eval, 1),
        }
    
    def save_checkpoint(self, path: str, metrics: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'vqvae_state_dict': self.vqvae.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'ema_diffusion_state_dict': self.ema_diffusion.state_dict(),
            'condition_encoder_state_dict': self.condition_encoder.state_dict(),
            'logic_net_state_dict': self.logic_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            # Store schedule/prediction type for inference consistency
            'schedule_type': self.config.schedule_type,
        }
        torch.save(checkpoint, path)
        write_checkpoint_metadata(
            path,
            model_type="diffusion",
            architecture={
                "latent_dim": int(self.config.latent_dim),
                "context_dim": int(self.config.context_dim),
                "num_timesteps": int(self.config.num_timesteps),
                "schedule_type": str(self.config.schedule_type),
            },
            extra={
                "epoch": int(self.epoch),
                "global_step": int(self.global_step),
                "contains": ["vqvae", "diffusion", "condition_encoder", "logic_net"],
            },
        )
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        if 'ema_diffusion_state_dict' in checkpoint:
            self.ema_diffusion.load_state_dict(checkpoint['ema_diffusion_state_dict'])
        self.condition_encoder.load_state_dict(checkpoint['condition_encoder_state_dict'])
        if 'logic_net_state_dict' in checkpoint:
            self.logic_net.load_state_dict(checkpoint['logic_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Re-wire LogicNet into guidance after loading
        self.diffusion.guidance.logic_net = self.logic_net
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_diffusion(config: DiffusionTrainingConfig) -> DiffusionTrainer:
    """Main training function."""
    logger.info(f"Starting diffusion training with config: {config.to_dict()}")
    
    # Create data loaders WITH real graph data from .dot files.
    # graph_collate_fn handles variable-size graphs by returning a list.
    train_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=True,
        use_vglc=config.use_vglc,
        normalize=True,
        room_level=config.room_level,
        load_graphs=True,
    )
    
    val_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=False,
        use_vglc=config.use_vglc,
        normalize=True,
        room_level=config.room_level,
        load_graphs=True,
    )
    
    sample_kind = "rooms" if config.room_level else "dungeons"
    logger.info(f"Training samples: {len(train_loader.dataset)} {sample_kind}")
    
    # Create trainer
    trainer = DiffusionTrainer(config)
    
    # Checkpoint manager
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_logger = MetricsLogger(
        log_dir=str(checkpoint_dir / 'logs'),
        experiment_name='diffusion_training',
    )
    
    best_solvability = 0.0
    
    # Training loop
    for epoch in range(config.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Combine metrics
        metrics = {
            'epoch': epoch,
            'lr': trainer.scheduler.get_last_lr()[0],
            **train_metrics,
            **val_metrics,
        }
        
        metrics_logger.log(metrics)
        
        logger.info(
            f"Epoch {epoch+1}/{config.epochs}: "
            f"loss={train_metrics['loss']:.4f}, "
            f"diffusion={train_metrics['diffusion_loss']:.4f}, "
            f"val_logic_loss={val_metrics.get('val_logic_loss', 0.0):.4f}, "
            f"val_solvability_proxy={val_metrics.get('val_solvability_proxy', val_metrics['val_solvability']):.4f}, "
            f"logic_loss_{'enabled' if epoch >= config.warmup_epochs and config.alpha_logic > 0 else 'disabled'}"
        )
        
        # Save checkpoints
        if (epoch + 1) % config.save_every == 0:
            trainer.save_checkpoint(
                str(checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pth"),
                metrics,
            )
        
        # Save best model
        if val_metrics['val_solvability'] > best_solvability:
            best_solvability = val_metrics['val_solvability']
            trainer.save_checkpoint(
                str(checkpoint_dir / "best_model.pth"),
                metrics,
            )
    
    # Final save
    trainer.save_checkpoint(str(checkpoint_dir / "final_model.pth"), metrics)
    metrics_logger.save()
    
    return trainer


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Latent Diffusion for Dungeon Generation',
    )
    
    parser.add_argument('--data-dir', type=str, default='Data/The Legend of Zelda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--room-level', dest='room_level', action='store_true', help='Train the diffusion model on individual room samples.')
    parser.add_argument('--dungeon-level', dest='room_level', action='store_false', help='Train the diffusion model on whole-dungeon samples.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha-logic', type=float, default=0.1)
    parser.add_argument(
        '--logic-loss-mode',
        type=str,
        default='predicted_latent',
        choices=['predicted_latent', 'detached_real'],
        help='Logic-loss target mode for A/B: predicted_latent (new) or detached_real (legacy).',
    )
    parser.add_argument(
        '--graph-conditioning-mode',
        type=str,
        default='node_sequence',
        choices=['node_sequence', 'pooled'],
        help='Graph conditioning for diffusion cross-attention: node_sequence (GCN node tokens) or pooled baseline.',
    )
    parser.add_argument(
        '--condition-gnn-type',
        type=str,
        default='gcn',
        choices=['gcn', 'gat', 'sage'],
        help='GNN backbone for graph-node conditioning.',
    )
    parser.add_argument(
        '--topology-refinement-mode',
        type=str,
        default='gat2',
        choices=['none', 'lightweight', 'gat2', 'upgraded'],
        help='Topology preprocessing inside diffusion cross-attention (gat2 is explicit 2-layer GAT).',
    )
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--vqvae-checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.set_defaults(room_level=True)
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    
    config = DiffusionTrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        room_level=args.room_level,
        epochs=args.epochs,
        learning_rate=args.lr,
        alpha_logic=args.alpha_logic,
        logic_loss_mode=args.logic_loss_mode,
        graph_conditioning_mode=args.graph_conditioning_mode,
        condition_gnn_type=args.condition_gnn_type,
        topology_refinement_mode=args.topology_refinement_mode,
        guidance_scale=args.guidance_scale,
        checkpoint_dir=args.checkpoint_dir,
        vqvae_checkpoint=args.vqvae_checkpoint,
        device=args.device,
        quick=args.quick,
    )
    
    try:
        _trainer = train_diffusion(config)
        logger.info("Training complete!")
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    except (AttributeError, RuntimeError, ValueError, TypeError) as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()

