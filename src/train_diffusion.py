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
from typing import Optional, Dict, Any, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zelda_loader import create_dataloader, extract_start_goal
from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion
from src.core.vqvae import SemanticVQVAE as VQVAE, create_vqvae
from src.core.condition_encoder import DualStreamConditionEncoder, create_condition_encoder
# Use Block V LogicNet (with temperature annealing), not legacy src.ml.logic_net
from src.core.logic_net import LogicNet
from src.utils.checkpoint import MetricsLogger

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
        
        # VQ-VAE (frozen encoder)
        vqvae_checkpoint: Optional[str] = None,
        
        # Diffusion Model
        latent_dim: int = 64,
        model_channels: int = 128,
        context_dim: int = 256,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        
        # LogicNet
        num_logic_iterations: int = 30,
        guidance_scale: float = 1.0,
        
        # Training
        epochs: int = 100,
        learning_rate: float = 1e-4,
        alpha_visual: float = 1.0,   # Diffusion loss weight
        alpha_logic: float = 0.1,     # Solvability loss weight
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
        
        self.vqvae_checkpoint = vqvae_checkpoint
        
        self.latent_dim = latent_dim
        self.model_channels = model_channels
        self.context_dim = context_dim
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        self.num_logic_iterations = num_logic_iterations
        self.guidance_scale = guidance_scale
        
        self.epochs = epochs if not quick else 2
        self.learning_rate = learning_rate
        self.alpha_visual = alpha_visual
        self.alpha_logic = alpha_logic
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
        # step, ∇_{x_t}L_logic nudges the sample toward solvable configs.
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
        )
    
    def _create_condition_encoder(self) -> DualStreamConditionEncoder:
        """Create condition encoder."""
        return create_condition_encoder(
            latent_dim=self.config.latent_dim,
            output_dim=self.config.context_dim,
        )
    
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
        
        Conversion: denormalize → integer tile IDs → one-hot → VQ-VAE encode
        """
        import torch.nn.functional as F
        
        with torch.no_grad():
            num_classes = self.vqvae.num_classes  # 44
            
            if x.shape[1] == 1:
                # Data loader format: [B, 1, H, W] normalized [0, 1]
                # Step 1: Denormalize to integer tile IDs
                tile_ids = (x.squeeze(1) * (num_classes - 1)).round().long()
                tile_ids = tile_ids.clamp(0, num_classes - 1)
                
                # Step 2: One-hot encode → [B, H, W, C] → permute to [B, C, H, W]
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
            
            # encode() returns (z_q, indices) — 2 values, not 3
            z_q, _indices = self.vqvae.encode(x_onehot)
        return z_q
    
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes back to tile logits.
        
        Returns:
            Tensor [B, C=44, H, W] of tile class logits
        """
        with torch.no_grad():
            return self.vqvae.decode(z, target_size=(11, 16))
    
    def get_dummy_conditioning(self, batch_size: int) -> torch.Tensor:
        """
        Get fallback conditioning when graph data is unavailable.
        
        Used only as a fallback during validation or when graph loading fails.
        During training, real graph data from .dot files is used instead.
        """
        return torch.randn(batch_size, self.config.context_dim, device=self.device)
    
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
            Conditioning vector [1, context_dim]
        """
        node_features = graph_dict['node_features'].to(self.device)
        edge_index = graph_dict['edge_index'].to(self.device)
        
        # Convert edge_attr from integer labels to one-hot for edge_features
        edge_attr = graph_dict.get('edge_attr')
        edge_features = None
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = edge_attr.to(self.device)
            # 7 edge types: open, key_locked, bombable, soft_locked, boss_locked, item_locked, stair
            num_edge_types = 7
            edge_attr_clamped = edge_attr.clamp(0, num_edge_types - 1)
            edge_features = torch.nn.functional.one_hot(
                edge_attr_clamped, num_classes=num_edge_types
            ).float()
            # Project to edge_feature_dim=3 expected by GNN
            # Use first 3 dims: key_locked, bombable, soft_locked (most important)
            if edge_features.shape[1] > 3:
                edge_features = edge_features[:, 1:4]  # skip 'open', take key/bomb/soft
        
        # Encode through global stream
        c_global = self.condition_encoder.encode_global_only(
            node_features, edge_index,
            edge_features=edge_features,
        )
        # Pool node embeddings → single conditioning vector [1, context_dim]
        c = c_global.mean(dim=0, keepdim=True)
        return c
    
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
            for e in range(edge_index_dev.shape[1]):
                src, dst = edge_index_dev[0, e].item(), edge_index_dev[1, e].item()
                adjacency[src, dst] = 1.0
                
                # Edge cost: locked doors have higher cost
                edge_type = 0
                if edge_attr is not None and e < len(edge_attr):
                    edge_type = edge_attr[e].item()
                
                if edge_type == 1:   # key_locked
                    edge_weights[src, dst] = 2.0
                elif edge_type == 4:  # boss_locked
                    edge_weights[src, dst] = 3.0
                else:
                    edge_weights[src, dst] = 1.0
        
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
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Training strategy:
        1. Diffusion loss: standard ε-prediction on real encoded latents
        2. LogicNet loss: computed on REAL z_0 with graph_data from .dot files,
           enabling both grid-level AND graph-level pathfinding/key-lock checking.
        3. GradientGuidance (wired in __init__): applies ∇_{x_t}L_logic
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
        diffusion_loss = self.diffusion.training_loss(z_0, conditioning)
        
        # === Part 2: LogicNet loss on REAL data WITH graph topology ===
        logic_loss = torch.tensor(0.0, device=self.device)
        solvability_score = torch.tensor(0.0, device=self.device)
        
        if include_logic_loss and self.config.alpha_logic > 0:
            # Detach z_0 from VQ-VAE graph but enable gradients for LogicNet
            z_for_logic = z_0.detach().requires_grad_(True)
            # Pass real graph_data to LogicNet for graph-level pathfinding
            logic_loss, logic_info = self.logic_net(z_for_logic, graph_data=logic_graph_data)
            solvability_score = 1.0 - logic_loss.detach()
        
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
        if hasattr(self.logic_net, 'update_temperature'):
            progress = min(1.0, self.global_step / max(1, self.config.epochs * 100))
            self.logic_net.update_temperature(progress)
        
        self.global_step += 1
        
        return {
            'loss': total_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'logic_loss': logic_loss.item(),
            'solvability': solvability_score.item(),
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
        metrics_sum = {'loss': 0, 'diffusion_loss': 0, 'logic_loss': 0, 'solvability': 0}
        num_batches = 0
        
        include_logic = self.epoch >= self.config.warmup_epochs
        
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
            
            if graph_list is not None and self.condition_encoder is not None:
                try:
                    # Encode each graph through GNN and stack
                    cond_vectors = []
                    for graph_dict in graph_list:
                        c_i = self._encode_graph_conditioning(graph_dict)
                        cond_vectors.append(c_i)
                    conditioning = torch.cat(cond_vectors, dim=0)  # [B, context_dim]
                except Exception as e:
                    logger.debug(f"Graph conditioning failed: {e}")
                    conditioning = None
                
                # Build LogicNet graph data from first graph in batch
                # (LogicNet processes single graphs, not batches)
                if include_logic:
                    try:
                        logic_graph_data = self._build_logic_graph_data(graph_list[0])
                    except Exception as e:
                        logger.debug(f"Logic graph build failed: {e}")
                        logic_graph_data = None
            
            metrics = self.train_step(
                real_maps,
                conditioning=conditioning,
                include_logic_loss=include_logic,
                logic_graph_data=logic_graph_data,
            )
            
            for k, v in metrics.items():
                metrics_sum[k] += v
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx}: loss={metrics['loss']:.4f}, "
                    f"diffusion={metrics['diffusion_loss']:.4f}, "
                    f"solvability={metrics['solvability']:.4f}"
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
        
        total_solvability = 0.0
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
            if graph_list is not None:
                try:
                    cond_vectors = []
                    for graph_dict in graph_list:
                        c_i = self._encode_graph_conditioning(graph_dict)
                        cond_vectors.append(c_i)
                    conditioning = torch.cat(cond_vectors, dim=0)
                except Exception:
                    conditioning = None
            
            if conditioning is None:
                conditioning = self.get_dummy_conditioning(batch_size)
            
            # Encode real maps to get latent shape
            z_0 = self.encode_to_latent(real_maps)
            
            # Generate samples using EMA model
            z_gen = eval_model.sample(conditioning, shape=z_0.shape)
            
            # Build LogicNet graph data if available
            logic_graph_data = None
            if graph_list is not None:
                try:
                    logic_graph_data = self._build_logic_graph_data(graph_list[0])
                except Exception:
                    pass
            
            # LogicNet: evaluate with graph topology
            logic_loss, logic_info = self.logic_net(z_gen, graph_data=logic_graph_data)
            solvability = 1.0 - logic_loss.item()
            total_solvability += solvability * batch_size
            num_samples_eval += batch_size
            
            if num_samples_eval >= num_samples:
                break
        
        return {
            'val_solvability': total_solvability / max(num_samples_eval, 1),
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
        load_graphs=True,
    )
    
    val_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=False,
        use_vglc=config.use_vglc,
        normalize=True,
        load_graphs=True,
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    
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
            f"solvability={val_metrics['val_solvability']:.4f}"
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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha-logic', type=float, default=0.1)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--vqvae-checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    
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
        epochs=args.epochs,
        learning_rate=args.lr,
        alpha_logic=args.alpha_logic,
        guidance_scale=args.guidance_scale,
        checkpoint_dir=args.checkpoint_dir,
        vqvae_checkpoint=args.vqvae_checkpoint,
        device=args.device,
        quick=args.quick,
    )
    
    try:
        trainer = train_diffusion(config)
        logger.info("Training complete!")
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
