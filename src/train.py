"""
Training Pipeline for KLTN PCG System
=====================================

End-to-end training for Zelda dungeon generation with:
1. Visual Loss: Diffusion model reconstruction
2. Logic Loss: Differentiable solvability via LogicNet

Usage:
    python -m src.train --data-dir "Data/The Legend of Zelda" --epochs 100
    
    # Quick test
    python -m src.train --data-dir "Data/The Legend of Zelda" --epochs 2 --quick
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
from src.ml.logic_net import LogicNet, solvability_loss
from src.utils.checkpoint import CheckpointManager, EarlyStopping, MetricsLogger

logger = logging.getLogger(__name__)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class TrainingConfig:
    """Training configuration with sensible defaults."""
    
    def __init__(
        self,
        # Data
        data_dir: str = "Data/The Legend of Zelda",
        batch_size: int = 4,
        use_vglc: bool = True,
        
        # Model
        latent_dim: int = 64,
        num_logic_iterations: int = 30,
        
        # Training
        epochs: int = 100,
        learning_rate: float = 1e-4,
        alpha: float = 0.1,  # Logic loss weight
        
        # Checkpointing
        checkpoint_dir: str = "./checkpoints",
        save_every: int = 10,
        
        # Device
        device: str = "auto",
        
        # Quick mode (for testing)
        quick: bool = False,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_vglc = use_vglc
        
        self.latent_dim = latent_dim
        self.num_logic_iterations = num_logic_iterations
        
        self.epochs = epochs if not quick else 2
        self.learning_rate = learning_rate
        self.alpha = alpha
        
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.quick = quick
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# =============================================================================
# H-MOLQD INTEGRATED TRAINING SYSTEM
# =============================================================================

class IntegratedDungeonGenerator(nn.Module):
    """
    Integrated H-MOLQD architecture for dungeon generation.
    
    This combines all neural components for end-to-end training:
    - VQ-VAE for discrete latent representation
    - Condition Encoder for context
    - Latent Diffusion for generation
    - LogicNet for solvability guidance
    
    Production-ready implementation replacing the simple placeholder.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 44,
        hidden_dim: int = 128,
        condition_dim: int = 256,
        codebook_size: int = 512,
        output_size: Tuple[int, int] = (16, 11),  # Zelda room size
        use_logic_guidance: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.use_logic_guidance = use_logic_guidance
        
        # Import H-MOLQD components
        from src.core.vqvae import SemanticVQVAE
        from src.core.latent_diffusion import LatentDiffusionModel
        from src.core.condition_encoder import DualStreamConditionEncoder
        from src.core.logic_net import LogicNet
        
        # Block II: VQ-VAE
        self.vqvae = SemanticVQVAE(
            num_classes=num_classes,
            codebook_size=codebook_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        
        # Block III: Condition Encoder
        self.condition_encoder = DualStreamConditionEncoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=condition_dim,
        )
        
        # Block IV: Latent Diffusion
        self.diffusion = LatentDiffusionModel(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            num_timesteps=1000,
            hidden_dim=hidden_dim,
        )
        
        # Block V: LogicNet (optional)
        if use_logic_guidance:
            self.logic_net = LogicNet(
                latent_dim=latent_dim,
                num_classes=num_classes,
                num_iterations=20,
            )
        else:
            self.logic_net = None
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            x: (B, num_classes, H, W) input dungeon grids (one-hot)
            condition: Optional (B, condition_dim) conditioning vector
            
        Returns:
            Dict with reconstruction, latents, losses
        """
        # VQ-VAE encode-decode
        z_e, z_q, vq_loss = self.vqvae.encode_decode(x)
        x_recon = self.vqvae.decode(z_q)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total VQ-VAE loss
        vqvae_loss = recon_loss + vq_loss
        
        results = {
            'reconstruction': x_recon,
            'latent_continuous': z_e,
            'latent_quantized': z_q,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'vqvae_loss': vqvae_loss,
        }
        
        # Optional: Logic loss for solvability
        if self.logic_net is not None and self.use_logic_guidance:
            logic_loss, logic_info = self.logic_net(z_q, {'tile_probs': torch.softmax(x_recon, dim=1)})
            results['logic_loss'] = logic_loss
            results['logic_info'] = logic_info
        
        return results
    
    def sample(
        self,
        num_samples: int = 1,
        device: Optional[torch.device] = None,
        condition: Optional[torch.Tensor] = None,
        num_diffusion_steps: int = 50,
    ) -> torch.Tensor:
        """
        Sample random dungeon maps using diffusion.
        
        Args:
            num_samples: Number of maps to generate
            device: Device to generate on
            condition: Optional conditioning
            num_diffusion_steps: DDIM steps
            
        Returns:
            (num_samples, num_classes, H, W) generated maps (logits)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from diffusion model
        z_latent = self.diffusion.sample(
            condition=condition,
            num_steps=num_diffusion_steps,
            batch_size=num_samples if condition is None else condition.shape[0],
        )
        
        # Decode with VQ-VAE
        with torch.no_grad():
            x_logits = self.vqvae.decode(z_latent)
        
        return x_logits
        return self.forward(z)
    
    def add_noise(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise for diffusion-style training (simplified)."""
        # Simple linear interpolation
        alpha = (1000 - t.float()).view(-1, 1, 1, 1) / 1000
        return alpha * x + (1 - alpha) * noise
    
    def predict_start_from_noise(
        self,
        noisy: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate clean image from noisy input (simplified)."""
        alpha = (1000 - t.float()).view(-1, 1, 1, 1) / 1000
        alpha = torch.clamp(alpha, 0.1, 1.0)  # Prevent division by zero
        return (noisy - (1 - alpha) * predicted_noise) / alpha


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    logic_net: LogicNet,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha: float = 0.1,
    default_start: Tuple[int, int] = (2, 2),
    default_goal: Tuple[int, int] = (13, 8),
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        dataloader: Training data loader
        model: Generator model
        logic_net: LogicNet for solvability
        optimizer: Optimizer
        device: Training device
        alpha: Weight for logic loss
        default_start: Default start position if not found in map
        default_goal: Default goal position if not found in map
        
    Returns:
        Dictionary of metrics
    """
    model.train()
    
    total_loss = 0.0
    total_visual_loss = 0.0
    total_logic_loss = 0.0
    num_batches = 0
    
    for batch_idx, real_maps in enumerate(dataloader):
        real_maps = real_maps.to(device)
        batch_size = real_maps.size(0)
        
        # Generate noise and timesteps
        noise = torch.randn_like(real_maps)
        t = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Add noise to real maps
        noisy_maps = model.add_noise(real_maps, noise, t)
        
        # Predict noise (reconstruction)
        z = torch.randn(batch_size, model.latent_dim, device=device)
        predicted = model(z)
        
        # Visual loss: reconstruction
        visual_loss = nn.functional.mse_loss(predicted, real_maps)
        
        # Get estimated clean maps for solvability check
        # NOTE: Do NOT use .detach() here - we need gradients to flow
        # from the logic loss back to the generator
        estimated_clean = predicted  # Keep gradient connection
        
        # Extract start/goal from maps or use defaults
        start_coords = []
        goal_coords = []
        for i in range(batch_size):
            start, goal = extract_start_goal(estimated_clean[i])
            start_coords.append(start if start else default_start)
            goal_coords.append(goal if goal else default_goal)
        
        # Logic loss: maximize solvability
        solvability_scores = logic_net(estimated_clean, start_coords, goal_coords)
        logic_loss = solvability_loss(solvability_scores)
        
        # Combined loss
        loss = visual_loss + alpha * logic_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_visual_loss += visual_loss.item()
        total_logic_loss += logic_loss.item()
        num_batches += 1
        
        # Logging
        if batch_idx % 10 == 0:
            logger.debug(f"Batch {batch_idx}: loss={loss.item():.4f}, "
                        f"visual={visual_loss.item():.4f}, "
                        f"logic={logic_loss.item():.4f}, "
                        f"solvability={solvability_scores.mean().item():.4f}")
    
    return {
        'loss': total_loss / num_batches,
        'visual_loss': total_visual_loss / num_batches,
        'logic_loss': total_logic_loss / num_batches,
    }


def validate(
    dataloader: DataLoader,
    model: nn.Module,
    logic_net: LogicNet,
    device: torch.device,
    default_start: Tuple[int, int] = (2, 2),
    default_goal: Tuple[int, int] = (13, 8),
) -> Dict[str, float]:
    """
    Validate model on validation set.
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_solvability = 0.0
    total_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for real_maps in dataloader:
            real_maps = real_maps.to(device)
            batch_size = real_maps.size(0)
            
            # Generate maps
            generated = model.sample(batch_size, device)
            
            # MSE to real maps (diversity check)
            mse = nn.functional.mse_loss(generated, real_maps)
            
            # Solvability
            start_coords = [default_start] * batch_size
            goal_coords = [default_goal] * batch_size
            solvability = logic_net(generated, start_coords, goal_coords)
            
            total_solvability += solvability.mean().item()
            total_mse += mse.item()
            num_batches += 1
    
    return {
        'val_solvability': total_solvability / max(num_batches, 1),
        'val_mse': total_mse / max(num_batches, 1),
    }


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(config: TrainingConfig) -> nn.Module:
    """
    Main training function.
    
    Args:
        config: Training configuration
        
    Returns:
        Trained model
    """
    logger.info(f"Starting training with config: {config.to_dict()}")
    
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info(f"Loading data from {config.data_dir}")
    train_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=True,
        use_vglc=config.use_vglc,
        normalize=True,
    )
    
    # Use same data for validation in this simple setup
    val_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=False,
        use_vglc=config.use_vglc,
        normalize=True,
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    
    # Get data dimensions from dataset
    sample = train_loader.dataset[0]
    data_h, data_w = sample.shape[-2], sample.shape[-1]
    logger.info(f"Data dimensions: {data_h}x{data_w}")
    
    # Create models with matching output size
    model = SimpleDungeonGenerator(
        latent_dim=config.latent_dim,
        output_size=(data_h, data_w),
    ).to(device)
    
    logic_net = LogicNet(
        num_iterations=config.num_logic_iterations,
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-5,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6,
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        config.checkpoint_dir,
        max_checkpoints=5,
        metric_name='val_solvability',
        mode='max',
    )
    
    # Metrics logger
    metrics_logger = MetricsLogger(
        log_dir=str(Path(config.checkpoint_dir) / 'logs'),
        experiment_name='zelda_pcg',
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=20, mode='max')
    
    # Try to resume from checkpoint
    start_epoch = checkpoint_manager.load(model, optimizer, scheduler)
    if start_epoch > 0:
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training loop...")
    
    for epoch in range(start_epoch, config.epochs):
        # Train
        train_metrics = train_one_epoch(
            train_loader,
            model,
            logic_net,
            optimizer,
            device,
            alpha=config.alpha,
        )
        
        # Validate
        val_metrics = validate(val_loader, model, logic_net, device)
        
        # Update scheduler
        scheduler.step()
        
        # Combine metrics
        metrics = {
            'epoch': epoch,
            'lr': scheduler.get_last_lr()[0],
            **train_metrics,
            **val_metrics,
        }
        
        # Log metrics
        metrics_logger.log(metrics)
        
        logger.info(
            f"Epoch {epoch+1}/{config.epochs}: "
            f"loss={train_metrics['loss']:.4f}, "
            f"visual={train_metrics['visual_loss']:.4f}, "
            f"logic={train_metrics['logic_loss']:.4f}, "
            f"solvability={val_metrics['val_solvability']:.4f}"
        )
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0 or epoch == config.epochs - 1:
            checkpoint_manager.save(model, optimizer, epoch, metrics, scheduler)
        
        # Early stopping check
        if early_stopping(val_metrics['val_solvability'], epoch):
            logger.info("Early stopping triggered")
            break
    
    # Final save
    metrics_logger.save()
    logger.info(f"\n{metrics_logger.summary()}")
    
    return model


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train KLTN PCG Dungeon Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir', type=str, default='Data/The Legend of Zelda',
        help='Path to dungeon data directory'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--use-vglc', action='store_true', default=True,
        help='Use VGLC format via ZeldaDungeonAdapter'
    )
    
    # Model arguments
    parser.add_argument(
        '--latent-dim', type=int, default=64,
        help='Dimension of latent space'
    )
    parser.add_argument(
        '--logic-iterations', type=int, default=30,
        help='Number of LogicNet iterations'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='Weight for logic (solvability) loss'
    )
    
    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint-dir', type=str, default='./checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--save-every', type=int, default=10,
        help='Save checkpoint every N epochs'
    )
    
    # Device arguments
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to train on'
    )
    
    # Quick mode
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test mode (2 epochs)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    
    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_vglc=args.use_vglc,
        latent_dim=args.latent_dim,
        num_logic_iterations=args.logic_iterations,
        epochs=args.epochs,
        learning_rate=args.lr,
        alpha=args.alpha,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        device=args.device,
        quick=args.quick,
    )
    
    # Train
    try:
        model = train(config)
        logger.info("Training complete!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
