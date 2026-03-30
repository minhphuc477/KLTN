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

import logging
import math
from typing import Dict, List, Tuple, Optional, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, TileID, normalize_room_shape

logger = logging.getLogger(__name__)


# ============================================================================
# SHAPE HELPERS
# ============================================================================

def canonical_latent_shape(
    room_shape: Sequence[int] = (ROOM_HEIGHT, ROOM_WIDTH),
    *,
    channel_mult: Tuple[int, ...] = (1, 2, 4),
    downsample_factor: int = 2,
) -> Tuple[int, int]:
    """
    Compute the default VQ-VAE latent `(rows, cols)` for a room grid.

    The encoder downsamples once per level except the last using a stride-2
    convolution with padding=1, which yields `ceil(size / 2)` each time.
    Keeping this calculation in one place avoids scattering hardcoded
    `(ROOM_HEIGHT // 4, (ROOM_WIDTH + 3) // 4)` assumptions across the codebase.
    """
    latent_h, latent_w = normalize_room_shape(room_shape)
    stride = max(1, int(downsample_factor))
    num_downsamples = max(0, len(channel_mult) - 1)

    for _ in range(num_downsamples):
        latent_h = int(math.ceil(latent_h / stride))
        latent_w = int(math.ceil(latent_w / stride))

    return latent_h, latent_w


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
        return_info: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Quantize continuous latents to discrete codebook entries.
        
        Args:
            z_e: Encoder output [B, D, H, W] or [B, H, W, D]
            
        Returns:
            If return_info=False (default):
                z_q: Quantized latents (same shape as input)
                vq_loss: Scalar quantization loss
                indices: Codebook indices [B, H, W]
            If return_info=True:
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
        encodings = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        losses['perplexity'] = perplexity
        
        # Update usage statistics (EMA-tracked, not forever-accumulating)
        if self.training:
            with torch.no_grad():
                batch_usage = torch.bincount(
                    indices, minlength=self.num_embeddings
                ).float()
                # EMA decay prevents stale early-training bias from
                # blocking dead-code resets in long training runs.
                self.codebook_usage = 0.99 * self.codebook_usage + 0.01 * batch_usage
        
        # Reshape indices
        indices = indices.view(B, H, W)
        
        # Convert back to channel-first if needed
        if channel_first:
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        if return_info:
            return z_q, indices, losses
        return z_q, losses['vq_loss'], indices
    
    def _ema_update(self, z_flat: Tensor, indices: Tensor):
        """Update codebook using exponential moving average (DDP-safe)."""
        with torch.no_grad():
            encodings = F.one_hot(indices, self.num_embeddings).float()
            
            # Update cluster sizes
            cluster_size = torch.sum(encodings, dim=0)
            embedding_sum = torch.matmul(encodings.t(), z_flat)
            
            # DDP synchronization: aggregate stats across all GPUs
            # before applying EMA so every replica sees the same update.
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
                    dist.all_reduce(embedding_sum, op=dist.ReduceOp.SUM)
            except (ImportError, RuntimeError):
                pass  # single-GPU fallback: no-op
            
            self.ema_cluster_size = (
                self.decay * self.ema_cluster_size 
                + (1 - self.decay) * cluster_size
            )
            
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
        
        DDP-safe: replacement vectors are determined on rank-0 and
        broadcast to all replicas.
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
                
                # DDP: broadcast from rank 0 so all replicas reset identically
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.broadcast(new_embeddings, src=0)
                except (ImportError, RuntimeError):
                    pass  # single-GPU fallback: no-op
                
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
# COORDCONV
# ============================================================================

class CoordConv2d(nn.Module):
    """
    Conv2d with explicit normalized coordinate channels.

    Appends x/y channels in [-1, 1] before applying convolution so the model
    can learn absolute boundary-aware behaviors on fixed-size room grids.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels + 2,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _c, h, w = x.shape
        device = x.device
        dtype = x.dtype

        y_coords = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)

        y_grid = y_coords.view(1, 1, h, 1).expand(b, 1, h, w)
        x_grid = x_coords.view(1, 1, 1, w).expand(b, 1, h, w)

        return self.conv(torch.cat([x, x_grid, y_grid], dim=1))


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
        hidden_dims: Optional[List[int]] = None,
        latent_dim: Optional[int] = None,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        downsample_factor: int = 2,
        use_coordconv: bool = True,
    ):
        super().__init__()

        if latent_dim is not None:
            latent_channels = int(latent_dim)

        if hidden_dims is not None and len(hidden_dims) > 0:
            block_channels = [int(max(1, c)) for c in hidden_dims]
            hidden_channels = int(block_channels[0])
        else:
            block_channels = [int(hidden_channels * mult) for mult in channel_mult]
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        
        # Initial projection (CoordConv improves absolute spatial awareness).
        if bool(use_coordconv):
            self.conv_in = CoordConv2d(in_channels, block_channels[0], 3, padding=1)
        else:
            self.conv_in = nn.Conv2d(in_channels, block_channels[0], 3, padding=1)
        
        # Build encoder blocks
        self.down_blocks = nn.ModuleList()
        
        ch = block_channels[0]
        for i, out_ch in enumerate(block_channels):
            
            block = nn.ModuleList()
            
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(ch, out_ch))
                ch = out_ch
            
            # Downsample (except last level)
            if i < len(block_channels) - 1:
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
            for layer in list(block.children()):
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
        hidden_dims: Optional[List[int]] = None,
        latent_dim: Optional[int] = None,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (4, 2, 1),
        upsample_factor: int = 2,
    ):
        super().__init__()

        if latent_dim is not None:
            latent_channels = int(latent_dim)

        if hidden_dims is not None and len(hidden_dims) > 0:
            block_channels = [int(max(1, c)) for c in hidden_dims]
            hidden_channels = int(block_channels[-1])
        else:
            block_channels = [int(hidden_channels * mult) for mult in channel_mult]
        
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Initial projection
        ch = block_channels[0]
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)
        
        # Build decoder blocks
        self.up_blocks = nn.ModuleList()
        
        for i, out_ch in enumerate(block_channels):
            
            block = nn.ModuleList()
            
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(ch, out_ch))
                ch = out_ch
            
            # Upsample (except last level)
            if i < len(block_channels) - 1:
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
            for layer in list(block.children()):
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
        num_tile_classes: Optional[int] = None,
        codebook_size: int = 512,
        num_embeddings: Optional[int] = None,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        commitment_cost: float = 0.25,
        rare_tile_weight: float = 5.0,
        use_ema: bool = True,
        use_coordconv: bool = True,
        mrf_penalty_weight: float = 0.05,
    ):
        super().__init__()

        if num_tile_classes is not None:
            num_classes = int(num_tile_classes)
        if num_embeddings is not None:
            codebook_size = int(num_embeddings)
        
        self.num_classes = num_classes
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.rare_tile_weight = rare_tile_weight
        self.mrf_penalty_weight = float(max(0.0, mrf_penalty_weight))
        
        # Encoder
        self.encoder = Encoder(
            in_channels=num_classes,
            hidden_channels=hidden_dim,
            latent_channels=latent_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            channel_mult=(1, 2, 4),
            use_coordconv=bool(use_coordconv),
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
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            channel_mult=(4, 2, 1),
        )
        
        # Build rare tile mask for weighted loss
        self.register_buffer(
            'tile_weights',
            self._build_tile_weights()
        )
        self.register_buffer('illegal_adjacency_matrix', self._build_illegal_adjacency_matrix())
    
    def _build_tile_weights(self) -> Tensor:
        """Build weight tensor for semantic-aware reconstruction."""
        weights = torch.ones(self.num_classes)
        for tile_id in self.RARE_TILES:
            if tile_id < self.num_classes:
                weights[tile_id] = self.rare_tile_weight
        return weights

    def _build_illegal_adjacency_matrix(self) -> Tensor:
        """
        Build static illegal tile-pair mask for soft MRF-style penalties.

        Mask value 1 means adjacency is illegal and should be penalized.
        """
        m = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32)

        door_ids = [
            int(TileID.DOOR_OPEN),
            int(TileID.DOOR_LOCKED),
            int(TileID.DOOR_BOMB),
            int(TileID.DOOR_PUZZLE),
            int(TileID.DOOR_BOSS),
            int(TileID.DOOR_SOFT),
        ]
        element_ids = [
            int(TileID.ELEMENT),
        ]

        # Example hard rule from user request: Water/Element next to doors is illegal.
        for a in element_ids:
            if a >= self.num_classes:
                continue
            for b in door_ids:
                if b >= self.num_classes:
                    continue
                m[a, b] = 1.0
                m[b, a] = 1.0

        return m

    def _illegal_adjacency_penalty(self, recon_logits: Tensor) -> Tensor:
        """
        Differentiable soft penalty over 3x3 neighborhood illegal adjacencies.
        """
        probs = F.softmax(recon_logits, dim=1)
        illegal = self.illegal_adjacency_matrix.to(dtype=probs.dtype, device=probs.device)

        # 8-neighborhood shifts (3x3 window excluding center).
        shifts = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        total = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        padded = F.pad(probs, (1, 1, 1, 1), mode='constant', value=0.0)
        h = probs.shape[2]
        w = probs.shape[3]

        for dy, dx in shifts:
            y0 = 1 + dy
            x0 = 1 + dx
            neighbor = padded[:, :, y0:y0 + h, x0:x0 + w]
            # Expected illegal pair probability per pixel.
            pair_score = torch.einsum('bchw,cd,bdhw->bhw', probs, illegal, neighbor)
            total = total + pair_score.mean()

        return total / float(len(shifts))
    
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
        z_q, _, indices = self.quantizer(z_e)
        return z_q, indices

    def quantize(self, z_e: Tensor | Tuple[Tensor, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """Backward-compatible quantize helper returning (z_q, vq_loss, indices)."""
        if isinstance(z_e, (tuple, list)):
            z_e = z_e[0]
        return self.quantizer(z_e)
    
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
        if target_size is None:
            # Internal room tensors use row-major `(rows, cols)`.
            target_size = (ROOM_HEIGHT, ROOM_WIDTH)
        else:
            target_size = normalize_room_shape(target_size)
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
    
    def forward_with_losses(
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
        z_q, indices, vq_losses = self.quantizer(z_e, return_info=True)
        
        # Decode
        recon = self.decoder(z_q, target_size=input_size)
        
        # Compute reconstruction loss
        losses = vq_losses.copy()
        
        # Weighted cross-entropy for semantic preservation
        recon_loss = self._weighted_reconstruction_loss(recon, x)
        losses['recon_loss'] = recon_loss

        illegal_penalty = self._illegal_adjacency_penalty(recon)
        losses['illegal_adjacency_penalty'] = illegal_penalty
        
        # Total loss
        losses['total_loss'] = recon_loss + losses['vq_loss'] + (self.mrf_penalty_weight * illegal_penalty)
        
        return recon, indices, losses

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Backward-compatible forward returning (recon, vq_loss, losses)."""
        recon, indices, losses = self.forward_with_losses(x)
        losses = dict(losses)
        losses['indices'] = indices
        return recon, losses['vq_loss'], losses

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Return detailed loss dictionary for training/evaluation code."""
        _, _, losses = self.forward_with_losses(x)
        return losses
    
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
        learning_rate: Optional[float] = None,
        weight_decay: float = 1e-5,
    ):
        if learning_rate is not None:
            lr = float(learning_rate)
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    
    def train_step(
        self, 
        batch: Tensor,
        return_metrics: bool = False,
    ) -> float | Tuple[float, Dict[str, float]]:
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
        losses = self.model.compute_loss(batch)
        
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
            'illegal_adjacency_penalty': losses.get('illegal_adjacency_penalty', torch.tensor(0.0)).item(),
            'perplexity': losses['perplexity'].item(),
        }
        
        loss_value = float(loss.item())
        if return_metrics:
            return loss_value, metrics
        return loss_value
    
    @torch.no_grad()
    def eval_step(self, batch: Tensor) -> Dict[str, float]:
        """Evaluation step."""
        self.model.eval()
        
        recon, _, _ = self.model(batch)
        losses = self.model.compute_loss(batch)
        
        # Compute accuracy
        pred = recon.argmax(dim=1)
        target = batch.argmax(dim=1) if batch.shape[1] > 1 else batch.squeeze(1)
        accuracy = (pred == target).float().mean()
        
        return {
            'loss': losses['total_loss'].item(),
            'recon_loss': losses['recon_loss'].item(),
            'illegal_adjacency_penalty': losses.get('illegal_adjacency_penalty', torch.tensor(0.0)).item(),
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
