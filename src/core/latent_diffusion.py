"""
H-MOLQD Block IV: Latent Diffusion with Gradient Guidance
==========================================================

Global Layout Generation (Macro-Optimization) for Zelda Dungeons.

This module implements a latent diffusion model that operates on VQ-VAE
latent tokens, guided by LogicNet gradients for solvability.

Mathematical Formulation:
-------------------------
Forward Diffusion:
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
    
Reverse Process (with guidance):
    x̂_{t-1} = μ_θ(x_t, t, c) - γ∇_{x_t}L_logic
    
where:
    μ_θ: U-Net predicted mean
    c: Conditioning from DualStreamConditionEncoder
    L_logic: LogicNet solvability loss
    γ: Guidance scale

Architecture:
- U-Net denoising network with cross-attention conditioning
- DDPM/DDIM sampling schedules
- Gradient guidance from LogicNet at inference

"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# NOISE SCHEDULES
# ============================================================================

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Tensor:
    """Linear noise schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """
    Cosine noise schedule from "Improved Denoising Diffusion Probabilistic Models".
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def get_noise_schedule(schedule_type: str, timesteps: int) -> Tensor:
    """Get noise schedule by name."""
    if schedule_type == "linear":
        return linear_beta_schedule(timesteps)
    elif schedule_type == "cosine":
        return cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# ============================================================================
# TIMESTEP EMBEDDING
# ============================================================================

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding.
    
    Uses positional encoding style embedding for diffusion timesteps.
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # MLP to project embeddings
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Embed timesteps.
        
        Args:
            t: Timesteps [B] or [B, 1]
            
        Returns:
            Embeddings [B, dim]
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1), mode='constant')
        
        return self.mlp(embedding)


# ============================================================================
# ATTENTION BLOCKS
# ============================================================================

class SelfAttention(nn.Module):
    """Self-attention block for U-Net."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, H*W, C]
        Returns:
            [B, H*W, C]
        """
        B, N, C = x.shape
        
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention for conditioning injection."""
    
    def __init__(
        self, 
        query_dim: int, 
        context_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.LayerNorm(query_dim)
        self.norm_context = nn.LayerNorm(context_dim)
        
        self.q = nn.Linear(query_dim, query_dim)
        self.k = nn.Linear(context_dim, query_dim)
        self.v = nn.Linear(context_dim, query_dim)
        self.proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x: Query [B, H*W, C]
            context: Context [B, context_dim] or [B, L, context_dim]
        Returns:
            [B, H*W, C]
        """
        B, N, C = x.shape
        
        x = self.norm(x)
        
        # Handle 2D context
        if context.dim() == 2:
            context = context.unsqueeze(1)
        context = self.norm_context(context)
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ============================================================================
# U-NET BLOCKS
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""
    
    @staticmethod
    def _num_groups(channels: int, max_groups: int = 32) -> int:
        """Find valid num_groups for GroupNorm: largest divisor of channels <= max_groups."""
        for g in range(min(max_groups, channels), 0, -1):
            if channels % g == 0:
                return g
        return 1
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(self._num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(self._num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W]
            t_emb: [B, time_dim]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Time conditioning (scale and shift)
        t_out = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Combined self-attention and cross-attention block."""
    
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attn = SelfAttention(dim, num_heads, dropout)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W]
            context: [B, context_dim]
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dims
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Self-attention
        x_flat = x_flat + self.self_attn(x_flat)
        
        # Cross-attention with context
        x_flat = x_flat + self.cross_attn(x_flat, context)
        
        # FFN
        x_flat = x_flat + self.ffn(x_flat)
        
        # Reshape back
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


class DownBlock(nn.Module):
    """Encoder block: ResBlock + Attention + Downsample."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        context_dim: int,
        num_res_blocks: int = 2,
        has_attention: bool = True,
        downsample: bool = True,
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(in_ch, out_channels, time_dim))
            
            if has_attention:
                self.attn_blocks.append(AttentionBlock(out_channels, context_dim))
            else:
                self.attn_blocks.append(nn.Identity())
        
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(
        self, 
        x: Tensor, 
        t_emb: Tensor, 
        context: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Returns:
            (output, list of skip connections)
        """
        skips = []
        
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, t_emb)
            if not isinstance(attn_block, nn.Identity):
                x = attn_block(x, context)
            skips.append(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, skips


class UpBlock(nn.Module):
    """Decoder block: Upsample + ResBlock + Attention.
    
    Each ResBlock receives a skip connection from the encoder (popped from
    all_skips in reverse order), following the standard U-Net pattern.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_dim: int,
        context_dim: int,
        num_res_blocks: int = 2,
        has_attention: bool = True,
        upsample: bool = True,
    ):
        super().__init__()
        
        self.upsample = None
        if upsample:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        self.num_res_blocks = num_res_blocks
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            # Every ResBlock receives a skip connection (concat on channel dim)
            in_ch = (in_channels if i == 0 else out_channels) + skip_channels
            self.res_blocks.append(ResBlock(in_ch, out_channels, time_dim))
            
            if has_attention:
                self.attn_blocks.append(AttentionBlock(out_channels, context_dim))
            else:
                self.attn_blocks.append(nn.Identity())
    
    def forward(
        self, 
        x: Tensor, 
        all_skips: List[Tensor], 
        t_emb: Tensor, 
        context: Tensor,
    ) -> Tensor:
        """Pop num_res_blocks skip connections from all_skips (from the end)."""
        if self.upsample is not None:
            x = self.upsample(x)
        
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            skip = all_skips.pop()  # consume from end (reverse order)
            # Handle size mismatch from downsampling rounding
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            
            x = res_block(x, t_emb)
            if not isinstance(attn_block, nn.Identity):
                x = attn_block(x, context)
        
        return x


# ============================================================================
# U-NET DENOISER
# ============================================================================

class UNetDenoiser(nn.Module):
    """
    U-Net denoising network for latent diffusion.
    
    Predicts the noise ε added to latent codes, conditioned on
    timestep t and context c from the condition encoder.
    
    Architecture:
        x_t, t, c → Encoder → Bottleneck → Decoder → ε_θ(x_t, t, c)
    
    Args:
        in_channels: Input latent channels
        model_channels: Base channel count
        out_channels: Output channels (same as input)
        context_dim: Conditioning dimension
        channel_mult: Channel multipliers per level
        num_res_blocks: ResBlocks per level
        attention_resolutions: Which downsampling levels have attention
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        model_channels: int = 128,
        out_channels: int = 64,
        context_dim: int = 256,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        
        time_dim = model_channels * 4
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_dim)
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Build encoder
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            has_attn = i in attention_resolutions
            downsample = i < len(channel_mult) - 1
            
            self.down_blocks.append(DownBlock(
                in_channels=channels[-1],
                out_channels=out_ch,
                time_dim=time_dim,
                context_dim=context_dim,
                num_res_blocks=num_res_blocks,
                has_attention=has_attn,
                downsample=downsample,
            ))
            channels.append(out_ch)
        
        # Bottleneck
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim)
        self.mid_attn = AttentionBlock(mid_ch, context_dim)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim)
        
        # Build decoder
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            has_attn = (len(channel_mult) - 1 - i) in attention_resolutions
            upsample = i > 0
            
            skip_ch = out_ch  # Simplified skip connection
            
            self.up_blocks.append(UpBlock(
                in_channels=channels[-1],
                out_channels=out_ch,
                skip_channels=skip_ch,
                time_dim=time_dim,
                context_dim=context_dim,
                num_res_blocks=num_res_blocks,
                has_attention=has_attn,
                upsample=upsample,
            ))
            channels.append(out_ch)
        
        # Output projection
        self.output_norm = nn.GroupNorm(ResBlock._num_groups(model_channels), model_channels)
        self.output_proj = nn.Conv2d(model_channels, out_channels, 3, padding=1)
    
    def forward(
        self, 
        x: Tensor, 
        t: Tensor, 
        context: Tensor,
    ) -> Tensor:
        """
        Predict noise in x_t.
        
        Args:
            x: Noisy latent [B, C, H, W]
            t: Timesteps [B]
            context: Conditioning [B, context_dim]
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Timestep embedding
        t_emb = self.time_embed(t)
        
        # Input projection
        h = self.input_proj(x)
        
        # Encoder with skip connections
        all_skips = []
        for down_block in self.down_blocks:
            h, skips = down_block(h, t_emb, context)
            all_skips.extend(skips)
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)
        
        # Decoder with skip connections
        for up_block in self.up_blocks:
            h = up_block(h, all_skips, t_emb, context)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_proj(h)
        
        return h


# ============================================================================
# GRADIENT GUIDANCE
# ============================================================================

class GradientGuidance(nn.Module):
    """
    Gradient guidance module for diffusion sampling.
    
    At each denoising step, computes gradient of LogicNet loss
    and adjusts the predicted mean accordingly:
    
        x̂_{t-1} = μ_θ(x_t) - γ∇_{x_t}L_logic
    
    Args:
        logic_net: LogicNet module (Block V)
        guidance_scale: Scale factor γ for gradients
        clamp_magnitude: Maximum gradient magnitude
    """
    
    def __init__(
        self,
        logic_net: Optional[nn.Module] = None,
        guidance_scale: float = 1.0,
        clamp_magnitude: float = 1.0,
    ):
        super().__init__()
        self.logic_net = logic_net
        self.guidance_scale = guidance_scale
        self.clamp_magnitude = clamp_magnitude
    
    def compute_guidance(
        self,
        x_t: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Compute guidance gradient from LogicNet.
        
        Args:
            x_t: Current noisy latent [B, C, H, W]
            graph_data: Graph information for LogicNet
            
        Returns:
            Gradient tensor [B, C, H, W]
        """
        if self.logic_net is None:
            return torch.zeros_like(x_t)
        
        # Use torch.enable_grad() so autograd works even inside @torch.no_grad() sampling
        with torch.enable_grad():
            # Enable gradient computation for x_t
            x_t_grad = x_t.detach().requires_grad_(True)
            
            # Compute LogicNet loss
            loss, _ = self.logic_net(x_t_grad, graph_data)
            
            # Compute gradient
            grad = torch.autograd.grad(
                loss,
                x_t_grad,
                create_graph=False,
                retain_graph=False,
            )[0]
        
        # Clamp gradient magnitude
        if self.clamp_magnitude > 0:
            # Flatten all dims except batch for norm, handles any tensor shape
            flat = grad.flatten(1)
            grad_norm = flat.norm(dim=1, keepdim=True)
            # Reshape norm back for broadcasting
            for _ in range(grad.dim() - 1):
                grad_norm = grad_norm.unsqueeze(-1)
            # Remove extra dims to match grad shape
            while grad_norm.dim() > grad.dim():
                grad_norm = grad_norm.squeeze(-1)
            grad = grad * torch.clamp(self.clamp_magnitude / (grad_norm + 1e-8), max=1.0)
        
        return self.guidance_scale * grad
    
    def apply_guidance(
        self,
        predicted_mean: Tensor,
        x_t: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Apply gradient guidance to predicted mean.
        
        Args:
            predicted_mean: μ_θ(x_t) from denoiser
            x_t: Current noisy latent
            graph_data: Graph information
            
        Returns:
            Guided mean
        """
        guidance = self.compute_guidance(x_t, graph_data)
        return predicted_mean - guidance


# ============================================================================
# LATENT DIFFUSION MODEL
# ============================================================================

class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model for Zelda Dungeon Generation.
    
    Operates on VQ-VAE latent tokens, using conditioning from the
    DualStreamConditionEncoder and optional gradient guidance from LogicNet.
    
    Training:
        loss = ||ε - ε_θ(x_t, t, c)||^2
    
    Sampling (with guidance):
        x̂_{t-1} = μ_θ(x_t, t, c) - γ∇_{x_t}L_logic + σ_t z
    
    Args:
        latent_dim: Dimension of VQ-VAE latent
        model_channels: U-Net base channels
        context_dim: Conditioning dimension
        num_timesteps: Number of diffusion timesteps
        schedule_type: Noise schedule ('linear' or 'cosine')
        logic_net: Optional LogicNet for gradient guidance
        guidance_scale: Gradient guidance scale γ
    
    Usage:
        model = LatentDiffusionModel(latent_dim=64, context_dim=256)
        
        # Training
        loss = model.training_loss(z_0, condition)
        
        # Sampling
        z_gen = model.sample(condition, shape=(1, 64, 4, 8))
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        model_channels: int = 128,
        context_dim: int = 256,
        num_timesteps: int = 1000,
        schedule_type: str = 'cosine',
        logic_net: Optional[nn.Module] = None,
        guidance_scale: float = 1.0,
        # --- Phase 1A: Classifier-Free Guidance (Ho & Salimans, 2022) ---
        cfg_dropout_prob: float = 0.1,
        cfg_scale: float = 3.0,
        # --- Phase 1C: v-prediction (Salimans & Ho, ICLR 2022) ---
        prediction_type: str = 'epsilon',  # 'epsilon' or 'v'
        # --- Phase 4B: Min-SNR-γ weighting (Hang et al., ICCV 2023) ---
        min_snr_gamma: float = 5.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_timesteps = num_timesteps
        self.cfg_dropout_prob = cfg_dropout_prob
        self.cfg_scale = cfg_scale
        self.prediction_type = prediction_type
        self.min_snr_gamma = min_snr_gamma
        
        # Denoising U-Net
        self.denoiser = UNetDenoiser(
            in_channels=latent_dim,
            model_channels=model_channels,
            out_channels=latent_dim,
            context_dim=context_dim,
        )
        
        # Gradient guidance
        self.guidance = GradientGuidance(
            logic_net=logic_net,
            guidance_scale=guidance_scale,
        )
        
        # Noise schedule
        betas = get_noise_schedule(schedule_type, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers for sampling
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(torch.clamp(posterior_variance, min=1e-20)))
    
    def q_sample(
        self, 
        x_0: Tensor, 
        t: Tensor, 
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward diffusion: sample x_t from x_0.
        
        q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
    
    def _predict_noise_cfg(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
    ) -> Tensor:
        """
        Predict noise with Classifier-Free Guidance (CFG).
        
        At inference: ε̃ = ε_uncond + s * (ε_cond - ε_uncond)
        where s = cfg_scale. When s=1 → standard conditional, s>1 → stronger conditioning.
        """
        pred_cond = self.denoiser(x_t, t, context)
        
        if self.cfg_scale > 1.0:
            # Unconditional prediction (zero context)
            pred_uncond = self.denoiser(x_t, t, torch.zeros_like(context))
            # CFG interpolation
            return pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
        
        return pred_cond
    
    def _convert_prediction(
        self,
        prediction: Tensor,
        x_t: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert model prediction to (pred_x0, pred_noise) based on prediction_type.
        
        Supports:
        - 'epsilon': model predicts noise ε (standard DDPM)
        - 'v': model predicts velocity v = √ᾱ_t·ε − √(1−ᾱ_t)·x_0 (Salimans & Ho, 2022)
        """
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        if self.prediction_type == 'v':
            # v-prediction: v = √ᾱ·ε − √(1−ᾱ)·x₀
            # Solve: x₀ = √ᾱ·x_t − √(1−ᾱ)·v
            #        ε  = √ᾱ·v + √(1−ᾱ)·x_t  (unused here, kept for reference)
            pred_x0 = sqrt_alpha_t * x_t - sqrt_one_minus_alpha_t * prediction
            pred_noise = sqrt_alpha_t * prediction + sqrt_one_minus_alpha_t * x_t
        else:
            # ε-prediction (standard)
            pred_noise = prediction
            pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / (sqrt_alpha_t + 1e-8)
        
        return pred_x0, pred_noise
    
    def p_mean_variance(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        clip_denoised: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute p(x_{t-1} | x_t) mean and variance.
        Now supports CFG and v-prediction.
        """
        # Predict with CFG
        prediction = self._predict_noise_cfg(x_t, t, context)
        
        # Convert to pred_x0 and pred_noise
        pred_x0, pred_noise = self._convert_prediction(prediction, x_t, t)
        
        # Compute predicted x_0
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        # pred_x0 already computed by _convert_prediction above
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute posterior mean
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t][:, None, None, None]
        beta_t = self.betas[t][:, None, None, None]
        
        posterior_mean = (
            torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1.0 - alpha_cumprod_t) * pred_x0
            + torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t) * x_t
        )
        
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        posterior_log_variance = self.posterior_log_variance[t][:, None, None, None]
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: Tensor,
        t: int,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Sample x_{t-1} from x_t (single step).
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        
        # Get mean and variance
        mean, variance, _ = self.p_mean_variance(x_t, t_tensor, context)
        
        # Apply gradient guidance
        if self.guidance.logic_net is not None and self.guidance.guidance_scale > 0:
            mean = self.guidance.apply_guidance(mean, x_t, graph_data)
        
        # Add noise (except at t=0)
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        
        return mean + torch.sqrt(variance) * noise
    
    @torch.no_grad()
    def sample(
        self,
        context: Tensor,
        shape: Tuple[int, ...],
        graph_data: Optional[Dict[str, Tensor]] = None,
        return_intermediates: bool = False,
    ) -> Tensor:
        """
        Generate samples via reverse diffusion.
        
        Args:
            context: Conditioning [B, context_dim]
            shape: Output shape [B, C, H, W]
            graph_data: Graph data for LogicNet guidance
            return_intermediates: Return all intermediate steps
            
        Returns:
            Generated latent codes
        """
        device = context.device
        B = context.shape[0]
        
        # Start from noise
        x_t = torch.randn(shape, device=device)
        
        intermediates = [x_t] if return_intermediates else None
        
        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            x_t = self.p_sample(x_t, t, context, graph_data)
            
            if return_intermediates:
                intermediates.append(x_t)
        
        if return_intermediates:
            return x_t, intermediates
        return x_t
    
    @torch.no_grad()
    def ddim_sample(
        self,
        context: Tensor,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        eta: float = 0.0,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        DDIM sampling for faster generation.
        
        Args:
            context: Conditioning [B, context_dim]
            shape: Output shape
            num_steps: Number of DDIM steps
            eta: DDIM stochasticity (0 = deterministic)
            graph_data: Graph data for guidance
            
        Returns:
            Generated latent codes
        """
        device = context.device
        
        # Create timestep schedule
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        # Start from noise
        x_t = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict with CFG
            prediction = self._predict_noise_cfg(x_t, t_tensor, context)
            
            # Convert to (pred_x0, pred_noise) supporting v-prediction
            pred_x0, pred_noise = self._convert_prediction(prediction, x_t, t_tensor)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Get next timestep
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_prev = torch.tensor(1.0, device=device)
            
            # CRITICAL-3 fix: compute alpha_t from current timestep
            alpha_t = self.alphas_cumprod[t]
            
            # DDIM update
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
            
            pred_dir = torch.sqrt(1 - alpha_prev - sigma**2) * pred_noise
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            
            x_t = torch.sqrt(alpha_prev) * pred_x0 + pred_dir + sigma * noise
            
            # Apply guidance
            if self.guidance.logic_net is not None and self.guidance.guidance_scale > 0:
                guidance_grad = self.guidance.compute_guidance(x_t, graph_data)
                x_t = x_t - guidance_grad
        
        return x_t
    
    def training_loss(
        self,
        x_0: Tensor,
        context: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute training loss with CFG dropout, v-prediction, and Min-SNR weighting.
        
        Enhancements over basic DDPM:
        - CFG dropout: randomly zero conditioning (Ho & Salimans, 2022)
        - v-prediction: predict velocity instead of noise (Salimans & Ho, 2022)
        - Min-SNR-γ: reweight loss by clamped SNR (Hang et al., ICCV 2023)
        
        Args:
            x_0: Clean latent codes [B, C, H, W]
            context: Conditioning [B, context_dim]
            noise: Optional noise tensor
            
        Returns:
            Weighted MSE loss
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # --- Phase 1A: CFG dropout during training ---
        # Randomly zero-out conditioning to train unconditional path
        if self.training and self.cfg_dropout_prob > 0:
            cfg_mask = (torch.rand(B, device=device) > self.cfg_dropout_prob).float()
            context = context * cfg_mask[:, None]  # Zero out entire conditioning
        
        # Get noisy samples
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict
        prediction = self.denoiser(x_t, t, context)
        
        # --- Phase 1C: Build target based on prediction_type ---
        if self.prediction_type == 'v':
            # v-target = √ᾱ_t · ε − √(1−ᾱ_t) · x₀
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            target = sqrt_alpha_t * noise - sqrt_one_minus_alpha_t * x_0
        else:
            target = noise
        
        # Per-sample MSE (not reduced yet)
        per_sample_loss = F.mse_loss(prediction, target, reduction='none')
        per_sample_loss = per_sample_loss.mean(dim=[1, 2, 3])  # [B]
        
        # --- Phase 4B: Min-SNR-γ weighting ---
        if self.min_snr_gamma > 0:
            snr = self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t] + 1e-8)
            # Clamp SNR and compute weight
            min_snr_weight = torch.clamp(snr, max=self.min_snr_gamma) / (snr + 1e-8)
            per_sample_loss = per_sample_loss * min_snr_weight
        
        loss = per_sample_loss.mean()
        
        return loss


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_latent_diffusion(
    latent_dim: int = 64,
    context_dim: int = 256,
    num_timesteps: int = 1000,
    prediction_type: str = 'epsilon',
    cfg_dropout_prob: float = 0.1,
    cfg_scale: float = 3.0,
    min_snr_gamma: float = 5.0,
    **kwargs,
) -> LatentDiffusionModel:
    """
    Create a LatentDiffusionModel.
    
    Args:
        latent_dim: VQ-VAE latent dimension
        context_dim: Conditioning dimension
        num_timesteps: Number of diffusion steps
        prediction_type: 'epsilon' or 'v' (v-prediction, Salimans & Ho 2022)
        cfg_dropout_prob: Conditioning dropout probability for CFG training
        cfg_scale: Classifier-free guidance scale for inference (1.0 = no CFG)
        min_snr_gamma: Min-SNR-γ clamp value (0 = disabled, 5.0 = recommended)
        **kwargs: Additional arguments
        
    Returns:
        LatentDiffusionModel instance
    """
    return LatentDiffusionModel(
        latent_dim=latent_dim,
        context_dim=context_dim,
        num_timesteps=num_timesteps,
        prediction_type=prediction_type,
        cfg_dropout_prob=cfg_dropout_prob,
        cfg_scale=cfg_scale,
        min_snr_gamma=min_snr_gamma,
        **kwargs,
    )
