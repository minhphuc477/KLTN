"""
H-MOLQD Block II: Semantic VQ-VAE
=================================

Discrete Representation Learning for Zelda Dungeon Rooms.

This module implements a Vector Quantized Variational Autoencoder (VQ-VAE)
specialized for semantic dungeon grids. The model learns a discrete codebook
of room "vocabulary" that can be used by the diffusion model.

Mathematical Formulation:
-------------------------
Encoder: z_e = E(x) ∈ ℝ^{H'×W'×D}
Quantization: z_q = argmin_{e_k} ||z_e - e_k||_2
Decoder: x̂ = D(z_q)

Losses:
- Reconstruction: L_rec = ||x - x̂||_2^2 (weighted for rare tiles)
- Codebook: L_cb = ||sg[z_e] - e||_2^2
- Commitment: L_com = β||z_e - sg[e]||_2^2
- Total: L = L_rec + L_cb + βL_com

where sg[] is stop-gradient operator.

Architecture:
- Encoder: Conv2d → ResBlocks → Downsample → Latent
- Vector Quantizer: Learned codebook with K entries
- Decoder: Upsample → ResBlocks → Conv2d → Output

"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# VECTOR QUANTIZER
# ============================================================================

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with learnable codebook.
    
    Implements the VQ operation from VQ-VAE:
    z_q = e_k where k = argmin_j ||z_e - e_j||_2
    
    Uses straight-through estimator for backpropagation:
    Forward: z_q (discrete)
    Backward: gradient flows to z_e directly
    
    Args:
        num_embeddings: Size of codebook (K)
        embedding_dim: Dimension of embeddings (D)
        commitment_cost: Weight β for commitment loss
        decay: EMA decay for codebook updates (if using EMA)
        use_ema: Use exponential moving average for codebook
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        use_ema: bool = True,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.use_ema = use_ema
        self.epsilon = epsilon
        
        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 
            1.0 / num_embeddings
        )
        
        if use_ema:
            # EMA cluster counts and sums
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_embedding_sum', self.embedding.weight.data.clone())
        
        # Statistics tracking
        self.register_buffer('codebook_usage', torch.zeros(num_embeddings))
        
        # --- Phase 1B: Dead code reset tracking ---
        self._reset_counter = 0
        self._reset_interval = 100  # Reset dead codes every N batches
        self._dead_threshold = 2    # Usage below this = "dead"
    
    def forward(
        self, 
        z_e: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Quantize continuous latents to discrete codebook entries.
        
        Args:
            z_e: Encoder output [B, D, H, W] or [B, H, W, D]
            
        Returns:
            z_q: Quantized latents (same shape as input)
            indices: Codebook indices [B, H, W]
            losses: Dict with 'vq_loss', 'commitment_loss', 'perplexity'
        """
        # Handle both channel-first and channel-last
        if z_e.dim() == 4 and z_e.shape[1] == self.embedding_dim:
            # [B, D, H, W] -> [B, H, W, D]
            z_e = z_e.permute(0, 2, 3, 1).contiguous()
            channel_first = True
        else:
            channel_first = False
        
        B, H, W, D = z_e.shape
        
        # Flatten spatial dimensions
        z_flat = z_e.view(-1, D)  # [B*H*W, D]
        
        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )  # [B*H*W, K]
        
        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=1)  # [B*H*W]
        
        # Quantize
        z_q = self.embedding(indices)  # [B*H*W, D]
        z_q = z_q.view(B, H, W, D)
        
        # Compute losses
        losses = {}
        
        if self.training:
            if self.use_ema:
                # EMA codebook update
                self._ema_update(z_flat, indices)
                
                # Only commitment loss (codebook is updated via EMA)
                commitment_loss = F.mse_loss(z_e, z_q.detach())
                losses['commitment_loss'] = self.commitment_cost * commitment_loss
                losses['vq_loss'] = losses['commitment_loss']
            else:
                # Standard VQ-VAE loss
                codebook_loss = F.mse_loss(z_q, z_e.detach())
                commitment_loss = F.mse_loss(z_e, z_q.detach())
                
                losses['codebook_loss'] = codebook_loss
                losses['commitment_loss'] = self.commitment_cost * commitment_loss
                losses['vq_loss'] = codebook_loss + losses['commitment_loss']
        else:
            losses['vq_loss'] = torch.tensor(0.0, device=z_e.device)
            losses['commitment_loss'] = torch.tensor(0.0, device=z_e.device)
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        # Compute perplexity (measure of codebook usage)
        encodings = torch.one_hot(indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        losses['perplexity'] = perplexity
        
        # Update usage statistics
        if self.training:
            with torch.no_grad():
                self.codebook_usage += torch.bincount(
                    indices, minlength=self.num_embeddings
                ).float()
        
        # Reshape indices
        indices = indices.view(B, H, W)
        
        # Convert back to channel-first if needed
        if channel_first:
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, indices, losses
    
    def _ema_update(self, z_flat: Tensor, indices: Tensor):
        """Update codebook using exponential moving average."""
        with torch.no_grad():
            encodings = torch.one_hot(indices, self.num_embeddings).float()
            
            # Update cluster sizes
            cluster_size = torch.sum(encodings, dim=0)
            self.ema_cluster_size = (
                self.decay * self.ema_cluster_size 
                + (1 - self.decay) * cluster_size
            )
            
            # Update embedding sums
            embedding_sum = torch.matmul(encodings.t(), z_flat)
            self.ema_embedding_sum = (
                self.decay * self.ema_embedding_sum
                + (1 - self.decay) * embedding_sum
            )
            
            # Laplace smoothing
            n = torch.sum(self.ema_cluster_size)
            cluster_size_smoothed = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            
            # Update embeddings
            self.embedding.weight.data = (
                self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1)
            )
            
            # --- Phase 1B: Periodic dead code reset ---
            self._reset_counter += 1
            if self._reset_counter % self._reset_interval == 0:
                self._reset_dead_codes(z_flat)
    
    def _reset_dead_codes(self, z_flat: Tensor):
        """
        Reset dead codebook entries to random encoder outputs + noise.
        
        Dead codes (rarely or never selected) waste codebook capacity.
        Resetting them to actual encoder outputs ensures they capture
        useful patterns. (Dhariwal et al. 2020, Zeghidour et al. 2021)
        """
        with torch.no_grad():
            # Identify dead codes: those with very low EMA cluster size
            dead_mask = self.ema_cluster_size < self._dead_threshold
            num_dead = dead_mask.sum().item()
            
            if num_dead > 0 and z_flat.shape[0] > 0:
                # Replace dead codes with random encoder outputs + small noise
                random_indices = torch.randint(0, z_flat.shape[0], (int(num_dead),),
                                               device=z_flat.device)
                new_embeddings = z_flat[random_indices].detach()
                new_embeddings = new_embeddings + torch.randn_like(new_embeddings) * 0.01
                
                self.embedding.weight.data[dead_mask] = new_embeddings
                
                # Reset EMA stats for replaced codes
                self.ema_cluster_size[dead_mask] = 1.0
                self.ema_embedding_sum[dead_mask] = new_embeddings
                
                logger.debug(
                    f"VQ codebook: reset {num_dead}/{self.num_embeddings} dead codes "
                    f"(utilization: {(~dead_mask).sum().item()}/{self.num_embeddings})"
                )
    
    def get_codebook_usage(self) -> Tensor:
        """Get normalized codebook usage statistics."""
        total = self.codebook_usage.sum()
        if total > 0:
            return self.codebook_usage / total
        return self.codebook_usage
    
    def encode_indices(self, indices: Tensor) -> Tensor:
        """Convert indices to embeddings."""
        return self.embedding(indices)


# ============================================================================
# RESIDUAL BLOCK
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and SiLU activation.
    
    Architecture:
    x → GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv → + → out
    └─────────────────── (skip connection) ────────────────┘
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(
            num_groups=min(groups, in_channels), 
            num_channels=in_channels
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(
            num_groups=min(groups, out_channels), 
            num_channels=out_channels
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.activation = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


# ============================================================================
# ENCODER
# ============================================================================

class Encoder(nn.Module):
    """
    Convolutional encoder for semantic grids.
    
    Architecture:
    Input [B, C, H, W] → Conv → ResBlocks → Downsample → ... → Latent [B, D, H', W']
    
    For 11×16 rooms with 2 downsamples: output is [B, D, 2, 4] or similar
    """
    
    def __init__(
        self,
        in_channels: int = 44,
        hidden_channels: int = 128,
        latent_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        downsample_factor: int = 2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        
        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Build encoder blocks
        self.down_blocks = nn.ModuleList()
        
        ch = hidden_channels
        for i, mult in enumerate(channel_mult):
            out_ch = hidden_channels * mult
            
            block = nn.ModuleList()
            
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(ch, out_ch))
                ch = out_ch
            
            # Downsample (except last level)
            if i < len(channel_mult) - 1:
                block.append(
                    nn.Conv2d(ch, ch, 3, stride=downsample_factor, padding=1)
                )
            
            self.down_blocks.append(block)
        
        # Final layers
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, latent_channels, 3, padding=1)
        
        self.activation = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Latent tensor [B, D, H', W']
        """
        h = self.conv_in(x)
        
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
        
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        
        return h


# ============================================================================
# DECODER
# ============================================================================

class Decoder(nn.Module):
    """
    Transposed convolutional decoder for semantic grids.
    
    Architecture:
    Latent [B, D, H', W'] → Conv → Upsample → ResBlocks → ... → Output [B, C, H, W]
    """
    
    def __init__(
        self,
        out_channels: int = 44,
        hidden_channels: int = 128,
        latent_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (4, 2, 1),
        upsample_factor: int = 2,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Initial projection
        ch = hidden_channels * channel_mult[0]
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)
        
        # Build decoder blocks
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(channel_mult):
            out_ch = hidden_channels * mult
            
            block = nn.ModuleList()
            
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(ch, out_ch))
                ch = out_ch
            
            # Upsample (except last level)
            if i < len(channel_mult) - 1:
                block.append(
                    nn.ConvTranspose2d(
                        ch, ch, 4, stride=upsample_factor, padding=1
                    )
                )
            
            self.up_blocks.append(block)
        
        # Final layers
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        self.activation = nn.SiLU()
    
    def forward(self, z: Tensor, target_size: Optional[Tuple[int, int]] = None) -> Tensor:
        """
        Decode latent to output.
        
        Args:
            z: Latent tensor [B, D, H', W']
            target_size: Optional (H, W) to resize output
            
        Returns:
            Output tensor [B, C, H, W]
        """
        h = self.conv_in(z)
        
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)
        
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        
        # Resize to target if needed
        if target_size is not None:
            h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
        
        return h


# ============================================================================
# SEMANTIC VQ-VAE
# ============================================================================

class SemanticVQVAE(nn.Module):
    """
    Semantic Vector Quantized VAE for Zelda Dungeon Rooms.
    
    This model learns a discrete vocabulary of room patterns that can be
    used by the latent diffusion model for generation.
    
    Key Features:
    - Semantic-aware reconstruction loss (weighted for rare tiles)
    - Learnable codebook with EMA updates
    - Straight-through gradient estimation
    
    Architecture:
        Input (11×16×C) → Encoder → VQ → Decoder → Output (11×16×C)
    
    Args:
        num_classes: Number of semantic tile classes (default: 44)
        codebook_size: Number of codebook entries K (default: 512)
        latent_dim: Dimension of latent vectors D (default: 64)
        hidden_dim: Hidden channel dimension (default: 128)
        commitment_cost: VQ commitment loss weight β (default: 0.25)
        rare_tile_weight: Weight multiplier for rare tiles (default: 5.0)
    
    Usage:
        model = SemanticVQVAE(num_classes=44, codebook_size=512)
        
        # Forward pass
        x = room_tensor  # [B, C, H, W]
        recon, indices, losses = model(x)
        
        # Encode only
        z_q, indices = model.encode(x)
        
        # Decode only
        recon = model.decode(z_q)
    """
    
    # Tile IDs that are considered "rare" (keys, doors, items, etc.)
    RARE_TILES = [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
    
    def __init__(
        self,
        num_classes: int = 44,
        codebook_size: int = 512,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        commitment_cost: float = 0.25,
        rare_tile_weight: float = 5.0,
        use_ema: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.rare_tile_weight = rare_tile_weight
        
        # Encoder
        self.encoder = Encoder(
            in_channels=num_classes,
            hidden_channels=hidden_dim,
            latent_channels=latent_dim,
            num_res_blocks=num_res_blocks,
            channel_mult=(1, 2, 4),
        )
        
        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            use_ema=use_ema,
        )
        
        # Decoder
        self.decoder = Decoder(
            out_channels=num_classes,
            hidden_channels=hidden_dim,
            latent_channels=latent_dim,
            num_res_blocks=num_res_blocks,
            channel_mult=(4, 2, 1),
        )
        
        # Build rare tile mask for weighted loss
        self.register_buffer(
            'tile_weights',
            self._build_tile_weights()
        )
    
    def _build_tile_weights(self) -> Tensor:
        """Build weight tensor for semantic-aware reconstruction."""
        weights = torch.ones(self.num_classes)
        for tile_id in self.RARE_TILES:
            if tile_id < self.num_classes:
                weights[tile_id] = self.rare_tile_weight
        return weights
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to quantized latent.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            z_q: Quantized latent [B, D, H', W']
            indices: Codebook indices [B, H', W']
        """
        z_e = self.encoder(x)
        z_q, indices, _ = self.quantizer(z_e)
        return z_q, indices
    
    def decode(
        self, 
        z_q: Tensor, 
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Decode quantized latent to output.
        
        Args:
            z_q: Quantized latent [B, D, H', W']
            target_size: Optional output size (H, W)
            
        Returns:
            Output logits [B, C, H, W]
        """
        return self.decoder(z_q, target_size)
    
    def decode_indices(
        self, 
        indices: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Decode from codebook indices directly.
        
        Args:
            indices: Codebook indices [B, H', W']
            target_size: Optional output size (H, W)
            
        Returns:
            Output logits [B, C, H, W]
        """
        z_q = self.quantizer.encode_indices(indices)  # [B, H', W', D]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()   # [B, D, H', W']
        return self.decode(z_q, target_size)
    
    def forward(
        self, 
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Forward pass: encode, quantize, decode.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            recon: Reconstructed logits [B, C, H, W]
            indices: Codebook indices [B, H', W']
            losses: Dict containing all loss components
        """
        input_size = (x.shape[2], x.shape[3])
        
        # Encode
        z_e = self.encoder(x)
        
        # Quantize
        z_q, indices, vq_losses = self.quantizer(z_e)
        
        # Decode
        recon = self.decoder(z_q, target_size=input_size)
        
        # Compute reconstruction loss
        losses = vq_losses.copy()
        
        # Weighted cross-entropy for semantic preservation
        recon_loss = self._weighted_reconstruction_loss(recon, x)
        losses['recon_loss'] = recon_loss
        
        # Total loss
        losses['total_loss'] = recon_loss + losses['vq_loss']
        
        return recon, indices, losses
    
    def _weighted_reconstruction_loss(
        self, 
        recon: Tensor, 
        target: Tensor,
    ) -> Tensor:
        """
        Compute weighted reconstruction loss.
        
        Applies higher weight to rare tiles (keys, items, etc.)
        to ensure they are preserved during reconstruction.
        
        Args:
            recon: Predicted logits [B, C, H, W]
            target: Target one-hot or class indices
            
        Returns:
            Weighted reconstruction loss
        """
        # Handle one-hot vs class index targets
        if target.shape[1] == self.num_classes:
            # One-hot encoded - convert to class indices
            target_idx = target.argmax(dim=1)  # [B, H, W]
        else:
            target_idx = target.squeeze(1)
        
        # Get per-class weights
        weights = self.tile_weights[target_idx]  # [B, H, W]
        
        # Cross-entropy loss with class weights
        loss = F.cross_entropy(
            recon, 
            target_idx,
            reduction='none'
        )  # [B, H, W]
        
        # Apply tile weights
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss
    
    def get_codebook(self) -> Tensor:
        """Get the learned codebook embeddings."""
        return self.quantizer.embedding.weight.data
    
    def get_codebook_usage(self) -> Tensor:
        """Get codebook usage statistics."""
        return self.quantizer.get_codebook_usage()


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class VQVAETrainer:
    """
    Training utilities for Semantic VQ-VAE.
    
    Usage:
        model = SemanticVQVAE()
        trainer = VQVAETrainer(model)
        
        for batch in dataloader:
            loss, metrics = trainer.train_step(batch)
    """
    
    def __init__(
        self,
        model: SemanticVQVAE,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    
    def train_step(
        self, 
        batch: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            batch: Input batch [B, C, H, W]
            
        Returns:
            loss: Total loss tensor
            metrics: Dict of metric values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward
        recon, indices, losses = self.model(batch)
        
        # Backward
        loss = losses['total_loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'recon_loss': losses['recon_loss'].item(),
            'vq_loss': losses['vq_loss'].item(),
            'perplexity': losses['perplexity'].item(),
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def eval_step(self, batch: Tensor) -> Dict[str, float]:
        """Evaluation step."""
        self.model.eval()
        
        recon, indices, losses = self.model(batch)
        
        # Compute accuracy
        pred = recon.argmax(dim=1)
        target = batch.argmax(dim=1) if batch.shape[1] > 1 else batch.squeeze(1)
        accuracy = (pred == target).float().mean()
        
        return {
            'loss': losses['total_loss'].item(),
            'recon_loss': losses['recon_loss'].item(),
            'accuracy': accuracy.item(),
            'perplexity': losses['perplexity'].item(),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_vqvae(
    num_classes: int = 44,
    codebook_size: int = 512,
    latent_dim: int = 64,
    **kwargs,
) -> SemanticVQVAE:
    """
    Create a Semantic VQ-VAE model.
    
    Args:
        num_classes: Number of semantic tile classes
        codebook_size: Number of codebook entries
        latent_dim: Latent vector dimension
        **kwargs: Additional arguments for SemanticVQVAE
        
    Returns:
        SemanticVQVAE model
    """
    return SemanticVQVAE(
        num_classes=num_classes,
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        **kwargs,
    )
