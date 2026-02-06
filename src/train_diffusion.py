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

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
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
from src.ml.logic_net import LogicNet, solvability_loss
from src.utils.checkpoint import CheckpointManager, EarlyStopping, MetricsLogger

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
        
        # Setup optimizer (only diffusion + condition encoder trainable)
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
    
    def _create_vqvae(self) -> VQVAE:
        """Create or load VQ-VAE."""
        vqvae = create_vqvae(
            in_channels=1,
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
            num_iterations=self.config.num_logic_iterations,
        )
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to VQ-VAE latent space."""
        with torch.no_grad():
            z, _, _ = self.vqvae.encode(x)
        return z
    
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes back to images."""
        with torch.no_grad():
            return self.vqvae.decode(z)
    
    def get_dummy_conditioning(self, batch_size: int) -> torch.Tensor:
        """
        Get dummy conditioning for unconditional training.
        
        In production, this should come from:
        - Graph structure (mission grammar output)
        - Neighbor room latents
        - Position encoding
        """
        # Simple random conditioning for initial training
        return torch.randn(batch_size, self.config.context_dim, device=self.device)
    
    def train_step(
        self,
        real_maps: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        include_logic_loss: bool = True,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            real_maps: [B, 1, H, W] real dungeon maps
            conditioning: [B, context_dim] conditioning vectors
            include_logic_loss: Whether to add solvability guidance
            
        Returns:
            Dict of loss values
        """
        self.diffusion.train()
        self.condition_encoder.train()
        
        batch_size = real_maps.shape[0]
        
        # Get conditioning (use dummy if not provided)
        if conditioning is None:
            conditioning = self.get_dummy_conditioning(batch_size)
        
        # Encode to latent space
        z_0 = self.encode_to_latent(real_maps)
        
        # Diffusion loss (predict noise)
        diffusion_loss = self.diffusion.training_loss(z_0, conditioning)
        
        # Logic loss (solvability)
        logic_loss = torch.tensor(0.0, device=self.device)
        solvability_score = torch.tensor(0.0, device=self.device)
        
        if include_logic_loss and self.config.alpha_logic > 0:
            # Sample from current model
            with torch.no_grad():
                z_sample = self.diffusion.sample(
                    conditioning,
                    shape=z_0.shape,
                )
            
            # Decode to probability map
            prob_maps = self.decode_from_latent(z_sample)
            prob_maps = torch.sigmoid(prob_maps)  # Ensure [0, 1]
            
            # Get start/goal coordinates
            start_coords = []
            goal_coords = []
            for i in range(batch_size):
                start, goal = extract_start_goal(prob_maps[i])
                start_coords.append(start if start else (2, 2))
                goal_coords.append(goal if goal else (13, 8))
            
            # Compute solvability - requires gradients through decoder
            prob_maps_grad = self.decode_from_latent(z_sample.detach().requires_grad_(True))
            prob_maps_grad = torch.sigmoid(prob_maps_grad)
            
            solvability_scores = self.logic_net(prob_maps_grad, start_coords, goal_coords)
            solvability_score = solvability_scores.mean()
            logic_loss = solvability_loss(solvability_scores)
        
        # Combined loss
        total_loss = (
            self.config.alpha_visual * diffusion_loss + 
            self.config.alpha_logic * logic_loss
        )
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.diffusion.parameters()) + list(self.condition_encoder.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        self.global_step += 1
        
        return {
            'loss': total_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'logic_loss': logic_loss.item(),
            'solvability': solvability_score.item(),
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        metrics_sum = {'loss': 0, 'diffusion_loss': 0, 'logic_loss': 0, 'solvability': 0}
        num_batches = 0
        
        include_logic = self.epoch >= self.config.warmup_epochs
        
        for batch_idx, real_maps in enumerate(dataloader):
            real_maps = real_maps.to(self.device)
            
            metrics = self.train_step(
                real_maps,
                include_logic_loss=include_logic,
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
        """Validate model."""
        self.diffusion.eval()
        
        total_solvability = 0.0
        num_samples_eval = 0
        
        for real_maps in dataloader:
            real_maps = real_maps.to(self.device)
            batch_size = real_maps.shape[0]
            
            # Sample conditioning
            conditioning = self.get_dummy_conditioning(batch_size)
            
            # Encode real maps to get latent shape
            z_0 = self.encode_to_latent(real_maps)
            
            # Generate samples
            z_gen = self.diffusion.sample(conditioning, shape=z_0.shape)
            gen_maps = self.decode_from_latent(z_gen)
            gen_maps = torch.sigmoid(gen_maps)
            
            # Evaluate solvability
            start_coords = [(2, 2)] * batch_size
            goal_coords = [(13, 8)] * batch_size
            
            solvability = self.logic_net(gen_maps, start_coords, goal_coords)
            total_solvability += solvability.sum().item()
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
            'diffusion_state_dict': self.diffusion.state_dict(),
            'condition_encoder_state_dict': self.condition_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        self.condition_encoder.load_state_dict(checkpoint['condition_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_diffusion(config: DiffusionTrainingConfig) -> DiffusionTrainer:
    """Main training function."""
    logger.info(f"Starting diffusion training with config: {config.to_dict()}")
    
    # Create data loaders
    train_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=True,
        use_vglc=config.use_vglc,
        normalize=True,
    )
    
    val_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=False,
        use_vglc=config.use_vglc,
        normalize=True,
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
