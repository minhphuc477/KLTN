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
import os
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.attention_kernels import HedgehogFeatureMap, hedgehog_linear_attention
from src.core.definitions import ROOM_TOPOLOGY_CHANNEL_COUNT
from src.core.graph_grid_attention import SpatialGraphConditioner

logger = logging.getLogger(__name__)
HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


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

        first_linear = self.mlp[0]
        weight = getattr(first_linear, "weight", None)
        if isinstance(weight, torch.Tensor):
            embedding = embedding.to(dtype=weight.dtype)

        return self.mlp(embedding)


# ============================================================================
# ATTENTION BLOCKS
# ============================================================================

class SelfAttention(nn.Module):
    """Self-attention block for U-Net."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError(f"SelfAttention num_heads must be > 0, got {num_heads}.")
        if int(dim) % int(num_heads) != 0:
            raise ValueError(
                f"SelfAttention requires dim to be divisible by num_heads; got dim={dim}, num_heads={num_heads}."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.perturbation_mode = "none"

    def set_perturbation_mode(self, mode: str = "none") -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"none", "identity"}:
            raise ValueError(f"SelfAttention perturbation mode must be 'none' or 'identity', got {mode!r}.")
        self.perturbation_mode = normalized
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, H*W, C]
        Returns:
            [B, H*W, C]
        """
        residual = x
        B, N, C = x.shape
        
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.perturbation_mode == "identity":
            out = v.transpose(1, 2).reshape(B, N, C)
        elif HAS_SDPA:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
            ).transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return residual + self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention for conditioning injection."""

    def __init__(
        self, 
        query_dim: int, 
        context_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.0,
        topology_refinement_mode: str = "gat2",
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
    ):
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError(f"CrossAttention num_heads must be > 0, got {num_heads}.")
        if int(query_dim) % int(num_heads) != 0:
            raise ValueError(
                "CrossAttention requires query_dim to be divisible by num_heads; "
                f"got query_dim={query_dim}, num_heads={num_heads}."
            )
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
        self.hedgehog_q = HedgehogFeatureMap(
            num_heads=num_heads,
            head_dim=self.head_dim,
            feature_dim=hedgehog_feature_dim,
        )
        self.hedgehog_k = HedgehogFeatureMap(
            num_heads=num_heads,
            head_dim=self.head_dim,
            feature_dim=hedgehog_feature_dim,
        )
        self.attention_mode = "softmax"
        self.set_attention_mode(attention_mode)

        # Topology-aware context refinement (2-layer graph message passing).
        self.topology_norm = nn.LayerNorm(context_dim)
        self.topology_light_self = nn.Linear(context_dim, context_dim)
        self.topology_light_neigh = nn.Linear(context_dim, context_dim)
        self.topology_in = nn.ModuleList([nn.Linear(context_dim, context_dim), nn.Linear(context_dim, context_dim)])
        self.topology_neigh = nn.ModuleList([nn.Linear(context_dim, context_dim), nn.Linear(context_dim, context_dim)])
        self.topology_gat_q = nn.ModuleList([nn.Linear(context_dim, context_dim), nn.Linear(context_dim, context_dim)])
        self.topology_gat_k = nn.ModuleList([nn.Linear(context_dim, context_dim), nn.Linear(context_dim, context_dim)])
        self.topology_gat_v = nn.ModuleList([nn.Linear(context_dim, context_dim), nn.Linear(context_dim, context_dim)])
        self.topology_gat_o = nn.ModuleList([nn.Linear(context_dim, context_dim), nn.Linear(context_dim, context_dim)])
        self.topology_refinement_mode = "gat2"
        self.set_topology_refinement_mode(topology_refinement_mode)

    def set_attention_mode(self, mode: str) -> None:
        """Set cross-attention kernel: softmax or linear_hedgehog."""
        normalized = str(mode).strip().lower()
        if normalized not in {"softmax", "linear_hedgehog"}:
            raise ValueError(
                f"Invalid attention_mode={mode!r}. Expected 'softmax' or 'linear_hedgehog'."
            )
        self.attention_mode = normalized

    def set_topology_refinement_mode(self, mode: str) -> None:
        """Set topology refinement mode: none | lightweight | gat2 (alias: upgraded)."""
        m = str(mode).strip().lower()
        if m == "upgraded":
            m = "gat2"
        if m not in {"none", "lightweight", "gat2"}:
            raise ValueError(
                f"Invalid topology_refinement_mode={mode!r}. Expected 'none', 'lightweight', or 'gat2'."
            )
        self.topology_refinement_mode = m
    
    def _normalize_adjacency(self, num_nodes: int, edge_index: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Build normalized adjacency with self loops for lightweight GCN refinement."""
        adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
        if edge_index.numel() > 0:
            src = edge_index[0].long().clamp(0, max(0, num_nodes - 1))
            dst = edge_index[1].long().clamp(0, max(0, num_nodes - 1))
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0
        adj = adj + torch.eye(num_nodes, device=device, dtype=dtype)
        deg = adj.sum(dim=1).clamp(min=1.0)
        inv_sqrt = deg.pow(-0.5)
        return inv_sqrt[:, None] * adj * inv_sqrt[None, :]

    def _batched_normalized_adjacency(
        self,
        *,
        batch_size: int,
        seq_len: int,
        edge_index: Tensor,
        node_mask: Optional[Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """Build dense [B,L,L] normalized adjacency and valid-node mask without per-sample loops."""
        if node_mask is None:
            valid = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        else:
            valid = node_mask.to(device=device, dtype=torch.bool)
            if valid.dim() == 1:
                valid = valid.unsqueeze(0)
            if int(valid.shape[0]) == 1 and batch_size > 1:
                valid = valid.expand(batch_size, -1)
            if int(valid.shape[0]) != batch_size:
                raise ValueError(
                    f"CrossAttention node_mask batch size {int(valid.shape[0])} does not match context batch {batch_size}."
                )
            if int(valid.shape[1]) > seq_len:
                valid = valid[:, :seq_len]
            elif int(valid.shape[1]) < seq_len:
                valid = F.pad(valid, (0, seq_len - int(valid.shape[1])), value=False)

        adj = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=dtype)
        eye = torch.eye(seq_len, device=device, dtype=dtype).unsqueeze(0)
        valid_pair = valid[:, :, None] & valid[:, None, :]

        if edge_index.dim() == 2:
            ei = edge_index.to(device=device, dtype=torch.long)
            if int(ei.shape[0]) != 2:
                raise ValueError(f"CrossAttention edge_index must have first dimension 2, got {tuple(ei.shape)}.")
            if ei.numel() > 0:
                src = ei[0].clamp(0, max(0, seq_len - 1))
                dst = ei[1].clamp(0, max(0, seq_len - 1))
                edge_valid = valid[:, src] & valid[:, dst]
                batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, src.numel())
                adj[batch_idx[edge_valid], src.unsqueeze(0).expand(batch_size, -1)[edge_valid], dst.unsqueeze(0).expand(batch_size, -1)[edge_valid]] = 1.0
                adj[batch_idx[edge_valid], dst.unsqueeze(0).expand(batch_size, -1)[edge_valid], src.unsqueeze(0).expand(batch_size, -1)[edge_valid]] = 1.0
        elif edge_index.dim() == 3:
            ei = edge_index.to(device=device, dtype=torch.long)
            if int(ei.shape[1]) != 2:
                raise ValueError(f"CrossAttention edge_index must have shape [B,2,E], got {tuple(ei.shape)}.")
            if int(ei.shape[0]) == 1 and batch_size > 1:
                ei = ei.expand(batch_size, -1, -1)
            if int(ei.shape[0]) != batch_size:
                raise ValueError(
                    f"CrossAttention edge_index batch size {int(ei.shape[0])} does not match context batch {batch_size}."
                )
            if ei.numel() > 0:
                src = ei[:, 0].clamp(0, max(0, seq_len - 1))
                dst = ei[:, 1].clamp(0, max(0, seq_len - 1))
                edge_valid = valid.gather(1, src) & valid.gather(1, dst)
                batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(src)
                adj[batch_idx[edge_valid], src[edge_valid], dst[edge_valid]] = 1.0
                adj[batch_idx[edge_valid], dst[edge_valid], src[edge_valid]] = 1.0
        else:
            raise ValueError(
                f"CrossAttention edge_index must have shape [2,E] or [B,2,E], got {tuple(edge_index.shape)}."
            )

        adj = (adj + eye).masked_fill(~valid_pair, 0.0)
        deg = adj.sum(dim=-1).clamp(min=1.0)
        inv_sqrt = deg.pow(-0.5)
        norm_adj = inv_sqrt[:, :, None] * adj * inv_sqrt[:, None, :]
        return norm_adj, valid

    def _refine_context_topology(
        self,
        context: Tensor,
        edge_index: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Refine token context with topology-aware message passing before K/V projection.

        Args:
            context: [B, L, C]
            edge_index: [2, E] or [B, 2, E]
            node_mask: [B, L] optional valid-token mask
        """
        if edge_index is None or context.dim() != 3 or self.topology_refinement_mode == "none":
            return context

        bsz, seq_len, _ctx_dim = context.shape
        refined = self.topology_norm(context)

        if node_mask is not None and node_mask.dim() == 2 and node_mask.shape[0] == 1 and bsz > 1:
            node_mask = node_mask.expand(bsz, -1)

        norm_adj, valid = self._batched_normalized_adjacency(
            batch_size=bsz,
            seq_len=seq_len,
            edge_index=edge_index,
            node_mask=node_mask,
            device=context.device,
            dtype=context.dtype,
        )
        h = refined
        if self.topology_refinement_mode == "lightweight":
            neigh = torch.bmm(norm_adj, h)
            update = F.gelu(self.topology_light_self(h) + self.topology_light_neigh(neigh))
            h = h + self.dropout(update)
        else:
            attn_mask = norm_adj > 0.0
            for q_lin, k_lin, v_lin, o_lin in zip(
                self.topology_gat_q,
                self.topology_gat_k,
                self.topology_gat_v,
                self.topology_gat_o,
            ):
                q = q_lin(h)
                k = k_lin(h)
                v = v_lin(h)
                scores = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
                scores = scores.masked_fill(~attn_mask, float("-inf"))
                attn = torch.softmax(scores, dim=-1)
                attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
                attn = self.dropout(attn)
                update = o_lin(torch.bmm(attn, v))
                h = h + self.dropout(F.gelu(update))

        return torch.where(valid[:, :, None], h, refined)
    
    def forward(
        self,
        x: Tensor,
        context: Tensor,
        edge_index: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Query [B, H*W, C]
            context: Context [B, context_dim] or [B, L, context_dim]
            edge_index: Optional graph topology for context tokens [2, E] or [B, 2, E]
            node_mask: Optional valid-token mask [B, L]
        Returns:
            [B, H*W, C]
        """
        B, N, C = x.shape
        
        x = self.norm(x)
        
        # Handle 2D context
        if context.dim() == 2:
            context = context.unsqueeze(1)
        context = self._refine_context_topology(context, edge_index=edge_index, node_mask=node_mask)
        context = self.norm_context(context)
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if self.attention_mode == "linear_hedgehog":
            out = hedgehog_linear_attention(
                q,
                k,
                v,
                q_map=self.hedgehog_q,
                k_map=self.hedgehog_k,
                token_mask=node_mask,
            ).transpose(1, 2).reshape(B, N, C)
        else:
            attn_mask = None
            if node_mask is not None:
                mask = node_mask
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                if mask.shape[0] == 1 and B > 1:
                    mask = mask.expand(B, -1)
                attn_mask = torch.zeros(
                    B,
                    1,
                    1,
                    int(mask.shape[-1]),
                    device=q.device,
                    dtype=q.dtype,
                )
                attn_mask = attn_mask.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

            if HAS_SDPA:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                ).transpose(1, 2).reshape(B, N, C)
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale
                if attn_mask is not None:
                    attn = attn + attn_mask
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
    def num_groups(channels: int, max_groups: int = 32) -> int:
        """Public wrapper for GroupNorm group-count selection."""
        return ResBlock._num_groups(channels, max_groups)
    
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
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
        topology_map_channels: int = 18,
        topology_conditioning_mode: str = "additive",
        auto_linear_attention_nodes: int = 128,
        graph_gate_init: float = -2.0,
        topology_gate_init: float = -2.0,
    ):
        super().__init__()
        
        self.self_attn = SelfAttention(dim, num_heads, dropout)
        self.cross_attn = CrossAttention(
            dim,
            context_dim,
            num_heads,
            dropout,
            attention_mode=attention_mode,
            hedgehog_feature_dim=hedgehog_feature_dim,
        )
        self.spatial_graph_conditioner = SpatialGraphConditioner(
            grid_dim=dim,
            graph_dim=context_dim,
            topology_channels=topology_map_channels,
            topology_conditioning_mode=topology_conditioning_mode,
            num_heads=num_heads,
            dropout=dropout,
            attention_mode=attention_mode,
            hedgehog_feature_dim=hedgehog_feature_dim,
            auto_linear_attention_nodes=auto_linear_attention_nodes,
            graph_gate_init=graph_gate_init,
            topology_gate_init=topology_gate_init,
        )
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_edge_index: Optional[Tensor] = None,
        context_node_mask: Optional[Tensor] = None,
        spatial_graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Args:
            x: [B, C, H, W]
            context: [B, context_dim]
            context_edge_index: Optional topology for context tokens [2, E] or [B, 2, E]
            context_node_mask: Optional valid-token mask [B, L]
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dims
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Self-attention (SelfAttention owns its Pre-LN residual).
        x_flat = self.self_attn(x_flat)
        
        cross_context = context
        cross_edge_index = context_edge_index
        cross_node_mask = context_node_mask
        if (
            spatial_graph_data
            and isinstance(context, torch.Tensor)
            and context.dim() == 3
            and int(context.shape[1]) > 1
            and isinstance(spatial_graph_data.get("graph_nodes"), torch.Tensor)
            and int(spatial_graph_data["graph_nodes"].shape[1]) > 0
        ):
            # Avoid injecting the same graph-node sequence twice: keep the room
            # summary token on the generic cross-attention path and let the
            # dedicated spatial conditioner handle graph-node grounding.
            cross_context = context[:, :1, :]
            cross_edge_index = None
            cross_node_mask = None

        # Cross-attention with context
        x_flat = x_flat + self.cross_attn(
            x_flat,
            cross_context,
            edge_index=cross_edge_index,
            node_mask=cross_node_mask,
        )

        x = x_flat.permute(0, 2, 1).view(B, C, H, W)
        if spatial_graph_data:
            x = self.spatial_graph_conditioner(
                x,
                graph_nodes=spatial_graph_data.get("graph_nodes"),
                edge_index=spatial_graph_data.get("edge_index"),
                node_positions=spatial_graph_data.get("node_positions"),
                node_tpe=spatial_graph_data.get("node_tpe"),
                current_node_distance=spatial_graph_data.get("current_node_distance"),
                node_mask=spatial_graph_data.get("node_mask"),
                room_topology_map=spatial_graph_data.get("room_topology_map"),
            )
            x_flat = x.view(B, C, -1).permute(0, 2, 1)
        
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
        num_heads: int = 8,
        num_res_blocks: int = 2,
        has_attention: bool = True,
        downsample: bool = True,
        dropout: float = 0.1,
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
        topology_map_channels: int = 18,
        topology_conditioning_mode: str = "additive",
        auto_linear_attention_nodes: int = 128,
        graph_gate_init: float = -2.0,
        topology_gate_init: float = -2.0,
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(in_ch, out_channels, time_dim, dropout=dropout))
            
            if has_attention:
                self.attn_blocks.append(
                    AttentionBlock(
                        out_channels,
                        context_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        attention_mode=attention_mode,
                        hedgehog_feature_dim=hedgehog_feature_dim,
                        topology_map_channels=topology_map_channels,
                        topology_conditioning_mode=topology_conditioning_mode,
                        auto_linear_attention_nodes=auto_linear_attention_nodes,
                        graph_gate_init=graph_gate_init,
                        topology_gate_init=topology_gate_init,
                    )
                )
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
        context_edge_index: Optional[Tensor] = None,
        context_node_mask: Optional[Tensor] = None,
        spatial_graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Returns:
            (output, list of skip connections)
        """
        skips = []
        
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, t_emb)
            if not isinstance(attn_block, nn.Identity):
                x = attn_block(
                    x,
                    context,
                    context_edge_index=context_edge_index,
                    context_node_mask=context_node_mask,
                    spatial_graph_data=spatial_graph_data,
                )
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
        skip_channels: int | Tuple[int, ...] | List[int],
        time_dim: int,
        context_dim: int,
        num_heads: int = 8,
        num_res_blocks: int = 2,
        has_attention: bool = True,
        upsample: bool = True,
        dropout: float = 0.1,
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
        topology_map_channels: int = 18,
        topology_conditioning_mode: str = "additive",
        auto_linear_attention_nodes: int = 128,
        graph_gate_init: float = -2.0,
        topology_gate_init: float = -2.0,
    ):
        super().__init__()
        
        self.upsample = None
        if upsample:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        self.num_res_blocks = num_res_blocks
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        if isinstance(skip_channels, int):
            skip_channels_per_block = [int(skip_channels)] * int(num_res_blocks)
        else:
            skip_channels_per_block = [int(ch) for ch in skip_channels]
            if len(skip_channels_per_block) != int(num_res_blocks):
                raise ValueError(
                    "UpBlock skip_channels sequence must match num_res_blocks; "
                    f"got {len(skip_channels_per_block)} skip widths for {num_res_blocks} blocks."
                )

        for i, skip_ch in enumerate(skip_channels_per_block):
            # Every ResBlock receives a skip connection (concat on channel dim)
            in_ch = (in_channels if i == 0 else out_channels) + skip_ch
            self.res_blocks.append(ResBlock(in_ch, out_channels, time_dim, dropout=dropout))
            
            if has_attention:
                self.attn_blocks.append(
                    AttentionBlock(
                        out_channels,
                        context_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        attention_mode=attention_mode,
                        hedgehog_feature_dim=hedgehog_feature_dim,
                        topology_map_channels=topology_map_channels,
                        topology_conditioning_mode=topology_conditioning_mode,
                        auto_linear_attention_nodes=auto_linear_attention_nodes,
                        graph_gate_init=graph_gate_init,
                        topology_gate_init=topology_gate_init,
                    )
                )
            else:
                self.attn_blocks.append(nn.Identity())
    
    def forward(
        self, 
        x: Tensor, 
        all_skips: List[Tensor], 
        t_emb: Tensor, 
        context: Tensor,
        context_edge_index: Optional[Tensor] = None,
        context_node_mask: Optional[Tensor] = None,
        spatial_graph_data: Optional[Dict[str, Tensor]] = None,
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
                x = attn_block(
                    x,
                    context,
                    context_edge_index=context_edge_index,
                    context_node_mask=context_node_mask,
                    spatial_graph_data=spatial_graph_data,
                )
        
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
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
        topology_map_channels: int = 18,
        topology_conditioning_mode: str = "additive",
        auto_linear_attention_nodes: int = 128,
        graph_gate_init: float = -2.0,
        topology_gate_init: float = -2.0,
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
        skip_channel_stack: List[int] = []
        
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            has_attn = i in attention_resolutions
            downsample = i < len(channel_mult) - 1
            
            self.down_blocks.append(DownBlock(
                in_channels=channels[-1],
                out_channels=out_ch,
                time_dim=time_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                num_res_blocks=num_res_blocks,
                has_attention=has_attn,
                downsample=downsample,
                dropout=dropout,
                attention_mode=attention_mode,
                hedgehog_feature_dim=hedgehog_feature_dim,
                topology_map_channels=topology_map_channels,
                topology_conditioning_mode=topology_conditioning_mode,
                auto_linear_attention_nodes=auto_linear_attention_nodes,
                graph_gate_init=graph_gate_init,
                topology_gate_init=topology_gate_init,
            ))
            channels.append(out_ch)
            skip_channel_stack.extend([out_ch] * int(num_res_blocks))
        
        # Bottleneck
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(
            mid_ch,
            context_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_mode=attention_mode,
            hedgehog_feature_dim=hedgehog_feature_dim,
            topology_map_channels=topology_map_channels,
            topology_conditioning_mode=topology_conditioning_mode,
            auto_linear_attention_nodes=auto_linear_attention_nodes,
            graph_gate_init=graph_gate_init,
            topology_gate_init=topology_gate_init,
        )
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim, dropout=dropout)
        
        # Build decoder
        self.up_blocks = nn.ModuleList()
        
        current_ch = mid_ch
        decoder_skip_stack = list(skip_channel_stack)
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            has_attn = (len(channel_mult) - 1 - i) in attention_resolutions
            upsample = i > 0

            if len(decoder_skip_stack) < int(num_res_blocks):
                raise ValueError(
                    "UNetDenoiser decoder requested more skip connections than the encoder produced. "
                    f"remaining={len(decoder_skip_stack)}, required={num_res_blocks}."
                )
            skip_ch = [decoder_skip_stack.pop() for _ in range(int(num_res_blocks))]
            
            self.up_blocks.append(UpBlock(
                in_channels=current_ch,
                out_channels=out_ch,
                skip_channels=skip_ch,
                time_dim=time_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                num_res_blocks=num_res_blocks,
                has_attention=has_attn,
                upsample=upsample,
                dropout=dropout,
                attention_mode=attention_mode,
                hedgehog_feature_dim=hedgehog_feature_dim,
                topology_map_channels=topology_map_channels,
                topology_conditioning_mode=topology_conditioning_mode,
                auto_linear_attention_nodes=auto_linear_attention_nodes,
                graph_gate_init=graph_gate_init,
                topology_gate_init=topology_gate_init,
            ))
            current_ch = out_ch
        
        # Output projection
        self.output_norm = nn.GroupNorm(ResBlock.num_groups(current_ch), current_ch)
        self.output_proj = nn.Conv2d(current_ch, out_channels, 3, padding=1)
    
    def forward(
        self, 
        x: Tensor, 
        t: Tensor, 
        context: Tensor,
        context_edge_index: Optional[Tensor] = None,
        context_node_mask: Optional[Tensor] = None,
        spatial_graph_data: Optional[Dict[str, Tensor]] = None,
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
            h, skips = down_block(
                h,
                t_emb,
                context,
                context_edge_index=context_edge_index,
                context_node_mask=context_node_mask,
                spatial_graph_data=spatial_graph_data,
            )
            all_skips.extend(skips)
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(
            h,
            context,
            context_edge_index=context_edge_index,
            context_node_mask=context_node_mask,
            spatial_graph_data=spatial_graph_data,
        )
        h = self.mid_block2(h, t_emb)
        
        # Decoder with skip connections
        for up_block in self.up_blocks:
            h = up_block(
                h,
                all_skips,
                t_emb,
                context,
                context_edge_index=context_edge_index,
                context_node_mask=context_node_mask,
                spatial_graph_data=spatial_graph_data,
            )
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_proj(h)
        
        return h


class DiTBlock(nn.Module):
    """DiT transformer block with adaLN-style conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"DiTBlock hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(max(hidden_dim, round(hidden_dim * float(mlp_ratio))))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 9 * hidden_dim))
        self.perturbation_mode = "none"

    def set_perturbation_mode(self, mode: str = "none") -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"none", "identity"}:
            raise ValueError(f"DiTBlock perturbation mode must be 'none' or 'identity', got {mode!r}.")
        self.perturbation_mode = normalized

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        context_tokens: Optional[Tensor] = None,
        context_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(cond).chunk(9, dim=-1)
        attn_in = self._modulate(self.norm1(x), shift_msa, scale_msa)
        if self.perturbation_mode == "identity":
            attn_out = attn_in
        else:
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + torch.sigmoid(gate_msa).unsqueeze(1) * attn_out
        if context_tokens is not None:
            cross_in = self._modulate(self.norm_cross(x), shift_cross, scale_cross)
            cross_out, _ = self.cross_attn(
                cross_in,
                context_tokens,
                context_tokens,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
            x = x + torch.sigmoid(gate_cross).unsqueeze(1) * cross_out
        mlp_in = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + torch.sigmoid(gate_mlp).unsqueeze(1) * self.mlp(mlp_in)
        return x


class DiTDenoiser(nn.Module):
    """
    Diffusion Transformer denoiser for flow-matching/diffusion ablations.

    The module patchifies latent feature maps, injects global timestep/context
    through adaLN, lets latent patches cross-attend to graph/context tokens, and
    unpatchifies back to the latent tensor shape expected by the trainer.
    """

    def __init__(
        self,
        in_channels: int = 64,
        model_channels: int = 128,
        out_channels: int = 64,
        context_dim: int = 256,
        depth: int = 4,
        patch_size: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if int(patch_size) <= 0:
            raise ValueError(f"DiTDenoiser patch_size must be > 0, got {patch_size}.")
        if int(model_channels) % int(num_heads) != 0:
            raise ValueError(
                f"DiTDenoiser model_channels={model_channels} must be divisible by num_heads={num_heads}."
            )
        self.in_channels = int(in_channels)
        self.model_channels = int(model_channels)
        self.out_channels = int(out_channels)
        self.context_dim = int(context_dim)
        self.patch_size = int(patch_size)
        self.depth = int(max(1, depth))

        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.model_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.time_embed = TimestepEmbedding(self.model_channels)
        self.context_proj = nn.Linear(self.context_dim, self.model_channels)
        self.context_token_proj = nn.Linear(self.context_dim, self.model_channels)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.model_channels,
                    cond_dim=self.model_channels,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(self.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(self.model_channels, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(self.model_channels, 2 * self.model_channels))
        self.out_proj = nn.Linear(self.model_channels, self.patch_size * self.patch_size * self.out_channels)

    def _pool_context(self, context: Tensor, node_mask: Optional[Tensor] = None) -> Tensor:
        if context.dim() == 2:
            return context
        if context.dim() != 3:
            raise ValueError(f"DiTDenoiser context must be [B,C] or [B,N,C], got {tuple(context.shape)}.")
        if node_mask is None:
            return context.mean(dim=1)
        mask = node_mask.to(device=context.device, dtype=context.dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if int(mask.shape[0]) == 1 and int(context.shape[0]) > 1:
            mask = mask.expand(int(context.shape[0]), -1)
        if int(mask.shape[1]) != int(context.shape[1]):
            if int(mask.shape[1]) > int(context.shape[1]):
                mask = mask[:, : int(context.shape[1])]
            else:
                mask = F.pad(mask, (0, int(context.shape[1]) - int(mask.shape[1])), value=0.0)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (context * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _context_tokens(
        self,
        context: Tensor,
        node_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if context.dim() == 2:
            tokens = context.unsqueeze(1)
            return self.context_token_proj(tokens), None
        if context.dim() != 3:
            raise ValueError(f"DiTDenoiser context must be [B,C] or [B,N,C], got {tuple(context.shape)}.")
        mask = None
        if node_mask is not None:
            mask = node_mask.to(device=context.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if int(mask.shape[0]) == 1 and int(context.shape[0]) > 1:
                mask = mask.expand(int(context.shape[0]), -1)
            if int(mask.shape[1]) != int(context.shape[1]):
                if int(mask.shape[1]) > int(context.shape[1]):
                    mask = mask[:, : int(context.shape[1])]
                else:
                    mask = F.pad(mask, (0, int(context.shape[1]) - int(mask.shape[1])), value=False)
        return self.context_token_proj(context), None if mask is None else ~mask

    def _positional_encoding(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack([yy, xx, torch.sin(math.pi * yy), torch.cos(math.pi * xx)], dim=-1)
        base = base.reshape(1, height * width, 4)
        repeat = math.ceil(self.model_channels / 4)
        return base.repeat(1, 1, repeat)[..., : self.model_channels]

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor,
        context_edge_index: Optional[Tensor] = None,
        context_node_mask: Optional[Tensor] = None,
        spatial_graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        del context_edge_index, spatial_graph_data
        B, _C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"DiTDenoiser patch_size={self.patch_size} requires latent H/W divisible by patch size; got {(H, W)}."
            )
        h = self.patch_embed(x)
        ph, pw = int(h.shape[-2]), int(h.shape[-1])
        tokens = h.flatten(2).transpose(1, 2)
        tokens = tokens + self._positional_encoding(ph, pw, tokens.device, tokens.dtype)

        pooled_context = self._pool_context(context, node_mask=context_node_mask).to(dtype=tokens.dtype)
        context_tokens, context_key_padding_mask = self._context_tokens(context, node_mask=context_node_mask)
        context_tokens = context_tokens.to(dtype=tokens.dtype)
        cond = self.time_embed(t).to(dtype=tokens.dtype) + self.context_proj(pooled_context)
        for block in self.blocks:
            tokens = block(
                tokens,
                cond,
                context_tokens=context_tokens,
                context_key_padding_mask=context_key_padding_mask,
            )

        shift, scale = self.final_mod(cond).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        patches = self.out_proj(tokens)
        patches = patches.view(B, ph, pw, self.patch_size, self.patch_size, self.out_channels)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(B, self.out_channels, H, W)


# ============================================================================
# GRADIENT GUIDANCE
# ============================================================================

class GradientGuidance(nn.Module):
    """
    Gradient guidance module for diffusion sampling.
    
    At each denoising step, computes gradient of a LogicNet objective
    and adjusts the predicted mean accordingly. By default the objective is a
    loss, so guidance performs gradient descent:
    
        x̂_{t-1} = μ_θ(x_t) - γ∇_{x_t}L_logic
    
    Args:
        logic_net: LogicNet module (Block V)
        guidance_scale: Scale factor γ for gradients
        clamp_magnitude: Maximum gradient magnitude
        relative_norm_cap: Maximum guidance norm relative to x_t norm
        mean_relative_norm_cap: Maximum applied guidance norm relative to predicted mean norm
        objective_mode: "loss" minimizes the LogicNet output; "reward" maximizes it
    """
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "logic_net":
            object.__setattr__(self, "_logic_net_ref", value)
            modules = self.__dict__.get("_modules")
            if isinstance(modules, dict):
                modules.pop("logic_net", None)
            return
        super().__setattr__(name, value)

    @property
    def logic_net(self) -> Optional[nn.Module]:
        return getattr(self, "_logic_net_ref", None)

    def __init__(
        self,
        logic_net: Optional[nn.Module] = None,
        guidance_scale: float = 1.0,
        clamp_magnitude: float = 1.0,
        relative_norm_cap: float = 0.25,
        mean_relative_norm_cap: float = 2.0,
        mean_norm_floor_fraction: float = 0.02,
        schedule_enabled: bool = True,
        active_fraction: float = 1.0,
        decay_power: float = 1.0,
        max_graph_nodes: int = 512,
        max_key_lock_pairs: int = 2048,
        max_guidance_elements: int = 2_000_000,
        objective_mode: str = "loss",
    ):
        super().__init__()
        self.logic_net = logic_net
        self.guidance_scale = guidance_scale
        self.clamp_magnitude = clamp_magnitude
        self.relative_norm_cap = float(max(0.0, float(relative_norm_cap)))
        self.mean_relative_norm_cap = float(max(0.0, float(mean_relative_norm_cap)))
        self.mean_norm_floor_fraction = float(max(0.0, float(mean_norm_floor_fraction)))
        self.schedule_enabled = bool(schedule_enabled)
        self.active_fraction = float(max(0.05, min(1.0, float(active_fraction))))
        self.decay_power = float(max(0.25, float(decay_power)))
        self.max_graph_nodes = int(max(1, int(max_graph_nodes)))
        self.max_key_lock_pairs = int(max(0, int(max_key_lock_pairs)))
        self.max_guidance_elements = int(max(1, int(max_guidance_elements)))
        self.objective_mode = str(objective_mode).strip().lower()
        if self.objective_mode not in {"loss", "reward"}:
            raise ValueError(
                f"objective_mode must be 'loss' or 'reward', got {objective_mode!r}."
            )
        self._missing_logic_net_warning_emitted = False
        self._warning_counts: Dict[str, int] = {}
        self.failure_count = 0
        self.last_failure_type: Optional[str] = None
        self.last_failure_message: Optional[str] = None

    def _warn_rate_limited(self, key: str, message: str, *args: Any) -> None:
        """Rate-limit repeated guidance warnings to keep logs readable."""
        count = int(self._warning_counts.get(key, 0)) + 1
        self._warning_counts[key] = count
        if count <= 5 or count in {10, 20, 50} or count % 100 == 0:
            logger.warning(message, *args)
            if count == 5:
                logger.warning(
                    "Further `%s` guidance warnings will be rate-limited.",
                    key,
                )

    def _sanitize_graph_data(
        self,
        graph_data: Optional[Any],
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Validate and clamp graph_data passed into LogicNet to avoid expensive
        unbounded path-computation on malformed payloads.
        """
        if graph_data is None:
            return None
        if not isinstance(graph_data, dict):
            logger.warning(
                "Gradient guidance: ignoring non-dict graph_data of type %s",
                type(graph_data).__name__,
            )
            return None

        target_device = device
        target_dtype = dtype

        room_topology_map = graph_data.get("room_topology_map")
        if isinstance(room_topology_map, torch.Tensor):
            room_topology_map = room_topology_map.detach()
            if room_topology_map.dim() == 3:
                room_topology_map = room_topology_map.unsqueeze(0)
            if room_topology_map.dim() != 4:
                logger.warning(
                    "Gradient guidance: room_topology_map must have shape [B,C,H,W] or [C,H,W], got %s; ignoring topology map.",
                    tuple(room_topology_map.shape),
                )
                room_topology_map = None
            else:
                if target_device is None:
                    target_device = room_topology_map.device
                if target_dtype is None:
                    target_dtype = room_topology_map.dtype if torch.is_floating_point(room_topology_map) else torch.float32
                room_topology_map = room_topology_map.to(device=target_device, dtype=torch.float32).clamp_(0.0, 1.0)
                room_topology_map = room_topology_map.to(dtype=target_dtype)
        else:
            room_topology_map = None

        boundary_constraints = graph_data.get("boundary_constraints")
        if isinstance(boundary_constraints, torch.Tensor):
            boundary_constraints = boundary_constraints.detach()
            if boundary_constraints.dim() == 1:
                boundary_constraints = boundary_constraints.unsqueeze(0)
            if boundary_constraints.dim() != 2 or int(boundary_constraints.shape[1]) != 8:
                logger.warning(
                    "Gradient guidance: boundary_constraints must have shape [B,8] or [8], got %s; ignoring boundary constraints.",
                    tuple(boundary_constraints.shape),
                )
                boundary_constraints = None
            else:
                if target_device is None:
                    target_device = boundary_constraints.device
                if target_dtype is None:
                    target_dtype = (
                        boundary_constraints.dtype if torch.is_floating_point(boundary_constraints) else torch.float32
                    )
                boundary_constraints = boundary_constraints.to(device=target_device, dtype=torch.float32).clamp_(0.0, 1.0)
                boundary_constraints = boundary_constraints.to(dtype=target_dtype)
        else:
            boundary_constraints = None

        graph_passthrough: Dict[str, Any] = {}

        graph_scope = graph_data.get("graph_scope")
        if isinstance(graph_scope, str):
            graph_passthrough["graph_scope"] = graph_scope

        node_features = graph_data.get("node_features")
        if isinstance(node_features, torch.Tensor):
            nf = node_features.detach()
            if nf.dim() not in {2, 3}:
                logger.warning(
                    "Gradient guidance: node_features must have shape [N,F] or [B,N,F], got %s; ignoring node features.",
                    tuple(nf.shape),
                )
            else:
                node_count = int(nf.shape[0] if nf.dim() == 2 else nf.shape[1])
                if node_count > self.max_graph_nodes:
                    logger.warning(
                        "Gradient guidance: graph too large (%d nodes > %d cap); disabling edge-index graph guidance.",
                        node_count,
                        self.max_graph_nodes,
                    )
                else:
                    if target_device is None:
                        target_device = nf.device
                    if target_dtype is None:
                        target_dtype = nf.dtype if torch.is_floating_point(nf) else torch.float32
                    graph_passthrough["node_features"] = nf.to(device=target_device, dtype=target_dtype)

        edge_index = graph_data.get("edge_index")
        if isinstance(edge_index, torch.Tensor):
            ei = edge_index.detach()
            if ei.dim() not in {2, 3} or int(ei.shape[-2]) != 2:
                logger.warning(
                    "Gradient guidance: edge_index must have shape [2,E] or [B,2,E], got %s; ignoring edge index.",
                    tuple(ei.shape),
                )
            elif target_device is not None:
                graph_passthrough["edge_index"] = ei.to(device=target_device, dtype=torch.long)
            else:
                target_device = ei.device
                graph_passthrough["edge_index"] = ei.to(dtype=torch.long)

        node_mask = graph_data.get("node_mask")
        if isinstance(node_mask, torch.Tensor):
            nm = node_mask.detach()
            if nm.dim() not in {1, 2}:
                logger.warning(
                    "Gradient guidance: node_mask must have shape [N] or [B,N], got %s; ignoring node mask.",
                    tuple(nm.shape),
                )
            else:
                graph_passthrough["node_mask"] = nm.to(device=target_device or nm.device, dtype=torch.float32).clamp_(0.0, 1.0)

        edge_features = graph_data.get("edge_features")
        if isinstance(edge_features, torch.Tensor):
            ef = edge_features.detach()
            if ef.dim() not in {2, 3}:
                logger.warning(
                    "Gradient guidance: edge_features must have shape [E,F] or [B,E,F], got %s; ignoring edge features.",
                    tuple(ef.shape),
                )
            else:
                graph_passthrough["edge_features"] = ef.to(device=target_device or ef.device, dtype=target_dtype or torch.float32)

        edge_attr = graph_data.get("edge_attr")
        if isinstance(edge_attr, torch.Tensor):
            ea = edge_attr.detach()
            if ea.dim() not in {1, 2}:
                logger.warning(
                    "Gradient guidance: edge_attr must have shape [E] or [B,E], got %s; ignoring edge labels.",
                    tuple(ea.shape),
                )
            else:
                graph_passthrough["edge_attr"] = ea.to(device=target_device or ea.device, dtype=torch.long)

        for key in ("current_node_idx", "start_node_id", "start_idx", "target_idx"):
            value = graph_data.get(key)
            if isinstance(value, torch.Tensor):
                graph_passthrough[key] = value.detach().to(device=target_device or value.device, dtype=torch.long)
            elif value is not None:
                graph_passthrough[key] = value

        adjacency = graph_data.get("adjacency")
        edge_weights = graph_data.get("edge_weights")

        # No graph adjacency, edge-index graph, or room-topology context => nothing to sanitize.
        if (
            adjacency is None
            and edge_weights is None
            and room_topology_map is None
            and boundary_constraints is None
            and "edge_index" not in graph_passthrough
            and "node_features" not in graph_passthrough
        ):
            return None

        sanitized: Dict[str, Any] = dict(graph_passthrough)
        if room_topology_map is not None:
            sanitized["room_topology_map"] = room_topology_map
        if boundary_constraints is not None:
            sanitized["boundary_constraints"] = boundary_constraints

        if adjacency is None and edge_weights is None:
            return sanitized or None

        if not isinstance(adjacency, torch.Tensor) or adjacency.dim() != 2:
            logger.warning("Gradient guidance: invalid adjacency tensor; disabling graph guidance.")
            return sanitized or None
        if adjacency.shape[0] != adjacency.shape[1]:
            logger.warning(
                "Gradient guidance: adjacency must be square, got %s.",
                tuple(adjacency.shape),
            )
            return sanitized or None
        node_count = int(adjacency.shape[0])
        if node_count > self.max_graph_nodes:
            logger.warning(
                "Gradient guidance: graph too large (%d nodes > %d cap); disabling graph guidance.",
                node_count,
                self.max_graph_nodes,
            )
            return sanitized or None

        target_device = target_device if target_device is not None else adjacency.device
        target_dtype = target_dtype if target_dtype is not None else (
            adjacency.dtype if torch.is_floating_point(adjacency) else torch.float32
        )
        adjacency = adjacency.detach().to(device=target_device, dtype=torch.float32)
        adjacency = torch.nan_to_num(adjacency, nan=0.0, posinf=1.0, neginf=0.0)
        adjacency = (adjacency > 0).to(dtype=target_dtype)

        if edge_weights is not None:
            if (not isinstance(edge_weights, torch.Tensor)) or edge_weights.shape != adjacency.shape:
                logger.warning(
                    "Gradient guidance: edge_weights shape mismatch (%s vs %s); disabling graph guidance.",
                    tuple(edge_weights.shape) if isinstance(edge_weights, torch.Tensor) else type(edge_weights).__name__,
                    tuple(adjacency.shape),
                )
                return None
        else:
            edge_weights = adjacency

        edge_weights = edge_weights.detach().to(device=target_device, dtype=target_dtype)
        edge_weights = torch.nan_to_num(
            edge_weights,
            nan=float(self.max_graph_nodes),
            posinf=float(self.max_graph_nodes),
            neginf=0.0,
        ).clamp(min=0.0, max=float(self.max_graph_nodes))

        try:
            start_idx = int(graph_data.get("start_idx", 0))
        except (TypeError, ValueError, OverflowError):
            start_idx = 0
        if start_idx < 0 or start_idx >= node_count:
            start_idx = 0

        target_idx_raw = graph_data.get("target_idx")
        target_idx: Optional[int] = None
        if target_idx_raw is not None:
            try:
                target_idx = int(target_idx_raw)
            except (TypeError, ValueError, OverflowError):
                target_idx = None
            if target_idx is not None and (target_idx < 0 or target_idx >= node_count):
                target_idx = None

        key_lock_pairs_raw = graph_data.get("key_lock_pairs", [])
        key_lock_pairs: List[Tuple[int, int]] = []
        if isinstance(key_lock_pairs_raw, (list, tuple)):
            for pair in list(key_lock_pairs_raw)[: self.max_key_lock_pairs]:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                try:
                    key_idx = int(pair[0])
                    lock_idx = int(pair[1])
                except (TypeError, ValueError, OverflowError):
                    continue
                if 0 <= key_idx < node_count and 0 <= lock_idx < node_count:
                    key_lock_pairs.append((key_idx, lock_idx))

        sanitized.update({
            "adjacency": adjacency,
            "edge_weights": edge_weights,
            "start_idx": int(start_idx),
            "target_idx": target_idx,
            "key_lock_pairs": key_lock_pairs,
        })
        return sanitized

    @staticmethod
    def _extract_logic_loss(logic_out: Any) -> Optional[Tensor]:
        """Support both LogicNet return signatures: Tensor or (loss, info)."""
        if isinstance(logic_out, tuple):
            if len(logic_out) == 0:
                return None
            loss = logic_out[0]
        else:
            loss = logic_out

        if not isinstance(loss, torch.Tensor) or loss.numel() == 0:
            return None
        if loss.numel() != 1:
            loss = loss.mean()
        return loss

    def _scheduled_scale(self, *, t: Optional[int], num_timesteps: Optional[int]) -> float:
        """Scale LogicNet guidance toward the cleaner end of the reverse process."""
        if (not self.schedule_enabled) or t is None or num_timesteps is None or num_timesteps <= 1:
            return float(self.guidance_scale)

        total = max(1, int(num_timesteps) - 1)
        # Reverse progress: 0 at first denoise step, 1 at final step.
        reverse_progress = float(total - int(t)) / float(total)
        active = float(self.active_fraction)
        active_start = 1.0 - active
        if reverse_progress < active_start:
            return 0.0
        phase = (reverse_progress - active_start) / max(active, 1e-8)
        return float(self.guidance_scale) * float(max(0.0, phase) ** self.decay_power)
    
    def compute_guidance(
        self,
        x_t: Tensor,
        graph_data: Optional[Dict[str, Any]] = None,
        *,
        t: Optional[int] = None,
        num_timesteps: Optional[int] = None,
        scale_multiplier: float = 1.0,
    ) -> Tensor:
        """
        Compute guidance gradient from LogicNet.
        
        Args:
            x_t: Current noisy latent [B, C, H, W]
            graph_data: Graph information for LogicNet
            
        Returns:
            Gradient tensor [B, C, H, W]
        """
        # DEBUG LOGGING
        _debug_guidance = int(os.environ.get('KLTN_DEBUG_GUIDANCE', 0)) > 0
        if _debug_guidance:
            logger.info(f"[GUIDANCE] compute_guidance called: logic_net={self.logic_net is not None}, guidance_scale={self.guidance_scale}, t={t}")

        if self.logic_net is None:
            if (not self._missing_logic_net_warning_emitted) and float(self.guidance_scale) > 0.0:
                logger.warning(
                    "Gradient guidance requested but no LogicNet is configured; skipping guidance."
                )
                self._missing_logic_net_warning_emitted = True
            if _debug_guidance:
                logger.info("[GUIDANCE] LogicNet is None, returning zero guidance")
            return torch.zeros_like(x_t)
        scaled_gamma = self._scheduled_scale(t=t, num_timesteps=num_timesteps)
        try:
            scaled_gamma *= float(scale_multiplier)
        except (TypeError, ValueError, OverflowError):
            scaled_gamma = 0.0
        if _debug_guidance:
            logger.info(f"[GUIDANCE] scaled_gamma={scaled_gamma}")
        if scaled_gamma <= 0.0:
            if _debug_guidance:
                logger.info(f"[GUIDANCE] scaled_gamma <= 0, returning zero guidance")
            return torch.zeros_like(x_t)
        if int(x_t.numel()) > self.max_guidance_elements:
            logger.warning(
                "Gradient guidance: latent too large for safe autograd guidance (%d elements > %d cap); skipping guidance.",
                int(x_t.numel()),
                self.max_guidance_elements,
            )
            return torch.zeros_like(x_t)
        
        input_dtype = x_t.dtype
        logic_dtype = torch.float32
        try:
            logic_ref = next(self.logic_net.parameters(), None)
            if logic_ref is None:
                logic_ref = next(self.logic_net.buffers(), None)
            if logic_ref is not None and torch.is_floating_point(logic_ref):
                logic_dtype = logic_ref.dtype
        except (AttributeError, StopIteration, TypeError):
            logic_dtype = torch.float32

        # Use torch.enable_grad() so autograd works even inside @torch.no_grad() sampling.
        # The sampler may run in FP16 while LogicNet is stored in FP32; guidance is
        # computed in the LogicNet dtype and cast back to the sampler dtype at return.
        try:
            safe_graph_data = self._sanitize_graph_data(
                graph_data,
                device=x_t.device,
                dtype=logic_dtype,
            )
            with torch.enable_grad():
                # Enable gradient computation for x_t
                x_t_grad = x_t.detach().to(dtype=logic_dtype).requires_grad_(True)

                # Compute LogicNet loss (supports tuple or tensor return modes).
                logic_out = self.logic_net(x_t_grad, safe_graph_data)
                loss = self._extract_logic_loss(logic_out)
                if _debug_guidance:
                    logger.info(f"[GUIDANCE] LogicNet loss={float(loss) if loss is not None else 'None'}")
                if loss is None:
                    self._warn_rate_limited(
                        "invalid_loss",
                        "Gradient guidance: LogicNet returned invalid loss; skipping guidance step.",
                    )
                    if _debug_guidance:
                        logger.info("[GUIDANCE] Invalid loss, returning zero guidance")
                    return torch.zeros_like(x_t)
                if not bool(torch.isfinite(loss).all()):
                    self._warn_rate_limited(
                        "nonfinite_loss",
                        "Gradient guidance: non-finite loss detected; skipping guidance step.",
                    )
                    if _debug_guidance:
                        logger.info(f"[GUIDANCE] Non-finite loss {loss}, returning zero guidance")
                    return torch.zeros_like(x_t)

                # Compute gradient
                grad = torch.autograd.grad(
                    loss,
                    x_t_grad,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # DEBUG: Log gradient statistics before clamping
                if _debug_guidance:
                    grad_norm_raw = grad.view(grad.shape[0], -1).norm(dim=1)
                    logger.info(f"[GUIDANCE] grad_norm (before clamp): min={grad_norm_raw.min():.6f}, max={grad_norm_raw.max():.6f}, mean={grad_norm_raw.mean():.6f}")
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            self.failure_count += 1
            self.last_failure_type = type(e).__name__
            self.last_failure_message = str(e)
            if int(os.environ.get("KLTN_GUIDANCE_RAISE_ERRORS", "0") or "0") > 0:
                raise
            self._warn_rate_limited(
                "guidance_failure",
                "Gradient guidance failed (%s: %s); continuing without guidance.",
                type(e).__name__,
                str(e),
            )
            return torch.zeros_like(x_t)
        
        # Clamp gradient magnitude
        if self.clamp_magnitude > 0:
            # Per-batch scalar norm with deterministic broadcast shape [B,1,1,...].
            grad_norm = grad.view(grad.shape[0], -1).norm(dim=1)
            grad_norm = grad_norm.view(grad.shape[0], *([1] * (grad.dim() - 1)))
            grad = grad * torch.clamp(self.clamp_magnitude / (grad_norm + 1e-8), max=1.0)

        guidance_direction = 1.0 if self.objective_mode == "loss" else -1.0
        guidance = float(scaled_gamma) * float(guidance_direction) * grad
        if self.relative_norm_cap > 0:
            ref_norm = x_t.detach().to(dtype=guidance.dtype).view(x_t.shape[0], -1).norm(dim=1)
            ref_norm = ref_norm.view(x_t.shape[0], *([1] * (x_t.dim() - 1)))
            guidance_norm = guidance.view(guidance.shape[0], -1).norm(dim=1)
            guidance_norm = guidance_norm.view(guidance.shape[0], *([1] * (guidance.dim() - 1)))
            max_guidance_norm = torch.clamp(ref_norm * self.relative_norm_cap, min=1e-8)
            guidance = guidance * torch.clamp(max_guidance_norm / (guidance_norm + 1e-8), max=1.0)

        # DEBUG: Log final guidance
        if _debug_guidance:
            guidance_norm_final = guidance.view(guidance.shape[0], -1).norm(dim=1)
            guidance_is_nonzero = (guidance_norm_final > 1e-8).sum().item()
            logger.info(f"[GUIDANCE] final_guidance_norm: min={guidance_norm_final.min():.6f}, max={guidance_norm_final.max():.6f}, nonzero_samples={guidance_is_nonzero}/{guidance.shape[0]}")

        return guidance.to(dtype=input_dtype)

    def _cap_guidance_to_predicted_mean(
        self,
        guidance: Tensor,
        predicted_mean: Tensor,
        x_t: Tensor,
    ) -> Tensor:
        """Limit the final mean update so guidance cannot dominate denoising."""
        if self.mean_relative_norm_cap <= 0:
            return guidance

        guidance_norm = guidance.view(guidance.shape[0], -1).norm(dim=1)
        guidance_norm = guidance_norm.view(guidance.shape[0], *([1] * (guidance.dim() - 1)))

        mean_norm = predicted_mean.detach().to(dtype=guidance.dtype).view(
            predicted_mean.shape[0], -1
        ).norm(dim=1)
        mean_norm = mean_norm.view(predicted_mean.shape[0], *([1] * (predicted_mean.dim() - 1)))
        max_guidance_norm = mean_norm * self.mean_relative_norm_cap

        if self.mean_norm_floor_fraction > 0:
            latent_norm = x_t.detach().to(dtype=guidance.dtype).view(x_t.shape[0], -1).norm(dim=1)
            latent_norm = latent_norm.view(x_t.shape[0], *([1] * (x_t.dim() - 1)))
            floor_norm = latent_norm * self.mean_norm_floor_fraction
            max_guidance_norm = torch.maximum(max_guidance_norm, floor_norm)

        max_guidance_norm = torch.clamp(max_guidance_norm, min=1e-8)
        return guidance * torch.clamp(max_guidance_norm / (guidance_norm + 1e-8), max=1.0)
    
    def apply_guidance(
        self,
        predicted_mean: Tensor,
        x_t: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        *,
        t: Optional[int] = None,
        num_timesteps: Optional[int] = None,
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
        guidance = self.compute_guidance(
            x_t,
            graph_data,
            t=t,
            num_timesteps=num_timesteps,
        )
        guidance = self._cap_guidance_to_predicted_mean(guidance, predicted_mean, x_t)
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
        cfg_schedule_mode: str = "constant",
        cfg_schedule_min_scale: float = 1.0,
        cfg_schedule_power: float = 1.0,
        pag_scale: float = 0.0,
        # --- Phase 1C: v-prediction (Salimans & Ho, ICLR 2022) ---
        prediction_type: str = 'epsilon',  # 'epsilon' or 'v'
        # --- Phase 4B: Min-SNR-γ weighting (Hang et al., ICCV 2023) ---
        min_snr_gamma: float = 5.0,
        topology_refinement_mode: str = "gat2",
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
        room_topology_channels: int = ROOM_TOPOLOGY_CHANNEL_COUNT,
        topology_conditioning_mode: str = "additive",
        denoiser_backbone: str = "unet",
        unet_channel_mult: Tuple[int, ...] = (1, 2, 4),
        unet_num_res_blocks: int = 2,
        unet_attention_resolutions: Tuple[int, ...] = (1, 2),
        unet_num_heads: int = 8,
        unet_dropout: float = 0.1,
        dit_depth: int = 4,
        dit_patch_size: int = 1,
        dit_mlp_ratio: float = 4.0,
        graph_auto_linear_attention_nodes: int = 128,
        spatial_graph_gate_init: float = -2.0,
        spatial_topology_gate_init: float = -2.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_timesteps = num_timesteps
        self.cfg_dropout_prob = cfg_dropout_prob
        self.cfg_scale = cfg_scale
        self.pag_scale = float(max(0.0, pag_scale))
        self.prediction_type = prediction_type
        self.min_snr_gamma = min_snr_gamma
        self.attention_mode = str(attention_mode).strip().lower()
        self.hedgehog_feature_dim = int(max(4, int(hedgehog_feature_dim)))
        self.room_topology_channels = int(max(1, int(room_topology_channels)))
        self.topology_conditioning_mode = str(topology_conditioning_mode).strip().lower()
        self.denoiser_backbone = str(denoiser_backbone).strip().lower()
        if self.topology_conditioning_mode not in {"additive", "spade"}:
            raise ValueError(
                "topology_conditioning_mode must be 'additive' or 'spade'. "
                f"Got {topology_conditioning_mode!r}."
            )
        if self.denoiser_backbone not in {"unet", "dit"}:
            raise ValueError(f"denoiser_backbone must be 'unet' or 'dit', got {denoiser_backbone!r}.")
        self.fast_sampler = None
        self.fast_sampler_checkpoint: Optional[str] = None
        
        if self.denoiser_backbone == "dit":
            self.denoiser = DiTDenoiser(
                in_channels=latent_dim,
                model_channels=model_channels,
                out_channels=latent_dim,
                context_dim=context_dim,
                depth=int(dit_depth),
                patch_size=int(dit_patch_size),
                num_heads=int(unet_num_heads),
                dropout=float(unet_dropout),
                mlp_ratio=float(dit_mlp_ratio),
            )
        else:
            self.denoiser = UNetDenoiser(
                in_channels=latent_dim,
                model_channels=model_channels,
                out_channels=latent_dim,
                context_dim=context_dim,
                channel_mult=tuple(int(v) for v in unet_channel_mult),
                num_res_blocks=int(unet_num_res_blocks),
                attention_resolutions=tuple(int(v) for v in unet_attention_resolutions),
                num_heads=int(unet_num_heads),
                dropout=float(unet_dropout),
                attention_mode=self.attention_mode,
                hedgehog_feature_dim=self.hedgehog_feature_dim,
                topology_map_channels=self.room_topology_channels,
                topology_conditioning_mode=self.topology_conditioning_mode,
                auto_linear_attention_nodes=int(graph_auto_linear_attention_nodes),
                graph_gate_init=float(spatial_graph_gate_init),
                topology_gate_init=float(spatial_topology_gate_init),
            )
        self.set_topology_refinement_mode(topology_refinement_mode)
        self.set_attention_mode(self.attention_mode)
        self.set_cfg_schedule(
            mode=cfg_schedule_mode,
            min_scale=cfg_schedule_min_scale,
            power=cfg_schedule_power,
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
        self._compiled_for_inference = False

    def compile_for_inference(self, *, mode: str = "reduce-overhead") -> bool:
        """
        Compile the denoiser with `torch.compile` when the local PyTorch build supports it.

        This is intentionally opt-in because compilation can be slow on first use
        and some CPU/CUDA combinations still have unsupported graph fragments.
        """
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            logger.warning("torch.compile is unavailable in this PyTorch build; inference compile skipped.")
            return False
        if self._compiled_for_inference:
            return True
        try:
            self.denoiser = compile_fn(self.denoiser, mode=mode)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("torch.compile failed for latent diffusion denoiser; continuing uncompiled: %s", exc)
            return False
        self._compiled_for_inference = True
        return True

    def set_cfg_schedule(
        self,
        mode: str = "constant",
        min_scale: float = 1.0,
        power: float = 1.0,
    ) -> None:
        """Configure inference-time classifier-free guidance scheduling."""
        normalized = str(mode).strip().lower()
        if normalized not in {"constant", "linear_decay", "cosine_decay"}:
            raise ValueError(
                f"Invalid cfg_schedule_mode={mode!r}. "
                "Expected 'constant', 'linear_decay', or 'cosine_decay'."
            )
        self.cfg_schedule_mode = normalized
        self.cfg_schedule_min_scale = float(max(0.0, min_scale))
        self.cfg_schedule_power = float(max(1e-6, power))

    def _cfg_scale_for_timestep(self, t: Tensor) -> Tensor:
        """Return per-sample CFG scale for the current reverse-diffusion timestep."""
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, device=self.betas.device, dtype=torch.long)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.to(device=self.betas.device, dtype=torch.float32)

        base_scale = float(self.cfg_scale)
        min_scale = float(self.cfg_schedule_min_scale)
        lower = min(base_scale, min_scale)
        upper = max(base_scale, min_scale)

        if self.cfg_schedule_mode == "constant" or math.isclose(base_scale, min_scale, rel_tol=1e-6, abs_tol=1e-6):
            scale = torch.full_like(t, fill_value=base_scale, dtype=torch.float32)
        else:
            denom = float(max(1, int(self.num_timesteps) - 1))
            progress = torch.clamp(t / denom, 0.0, 1.0)
            shaped = progress.pow(float(self.cfg_schedule_power))
            if self.cfg_schedule_mode == "cosine_decay":
                shaped = torch.sin(shaped * (math.pi * 0.5))
            scale = min_scale + (base_scale - min_scale) * shaped

        return torch.clamp(scale, min=lower, max=upper)

    def set_topology_refinement_mode(self, mode: str) -> int:
        """Set topology refinement mode across all active CrossAttention layers."""
        updated = 0
        requested = str(mode).strip().lower()
        for module in self.modules():
            if isinstance(module, CrossAttention):
                module.set_topology_refinement_mode(mode)
                updated += 1
        self._topology_refinement_mode_alias = requested
        return updated

    def get_topology_refinement_mode(self) -> str:
        """Get current topology refinement mode from first CrossAttention layer."""
        alias = getattr(self, "_topology_refinement_mode_alias", None)
        if isinstance(alias, str) and alias:
            return alias
        for module in self.modules():
            if isinstance(module, CrossAttention):
                return str(getattr(module, "topology_refinement_mode", "gat2"))
        return "gat2"

    def set_attention_mode(self, mode: str) -> int:
        """Set attention kernel across all active token and graph-grid attention layers."""
        updated = 0
        normalized = str(mode).strip().lower()
        for module in self.modules():
            if isinstance(module, CrossAttention):
                module.set_attention_mode(normalized)
                updated += 1
            elif hasattr(module, "set_attention_mode") and module is not self:
                if isinstance(module, SpatialGraphConditioner):
                    module.set_attention_mode(normalized)
                    updated += 1
        self.attention_mode = normalized
        return updated

    def set_self_attention_perturbation(self, mode: str = "none") -> int:
        """Set self-attention perturbation mode for PAG-style inference guidance."""
        updated = 0
        for module in self.modules():
            if isinstance(module, SelfAttention):
                module.set_perturbation_mode(mode)
                updated += 1
            elif isinstance(module, DiTBlock):
                module.set_perturbation_mode(mode)
                updated += 1
        return updated

    def set_spatial_attention_capture(self, enabled: bool = True) -> int:
        """Enable differentiable graph-grid attention capture in every spatial conditioner."""
        updated = 0
        for module in self.modules():
            if isinstance(module, SpatialGraphConditioner):
                module.set_attention_capture(enabled)
                updated += 1
        return updated

    def spatial_alignment_loss(
        self,
        node_indices: Tensor,
        target_positions: Tensor,
        valid_mask: Optional[Tensor] = None,
        *,
        reduce_heads: str = "mean",
        eps: float = 1e-8,
    ) -> Tensor:
        """Aggregate spatial graph-node alignment losses from captured conditioner maps."""
        losses: List[Tensor] = []
        for module in self.modules():
            if not isinstance(module, SpatialGraphConditioner):
                continue
            graph_attn = module.graph_cross_attn
            if graph_attn.last_attention_weights_for_loss is None:
                continue
            losses.append(
                module.spatial_alignment_loss(
                    node_indices,
                    target_positions,
                    valid_mask=valid_mask,
                    reduce_heads=reduce_heads,
                    eps=eps,
                )
            )
        if not losses:
            raise RuntimeError(
                "Spatial alignment loss was requested, but no SpatialGraphConditioner captured softmax attention maps."
            )
        return torch.stack(losses).mean()

    def get_attention_mode(self) -> str:
        """Get current attention kernel from first CrossAttention layer."""
        for module in self.modules():
            if isinstance(module, CrossAttention):
                return str(getattr(module, "attention_mode", "softmax"))
        return "softmax"

    def supports_fast_sampling(self) -> bool:
        """Whether a validated fast-sampling adapter is currently configured."""
        return self.fast_sampler is not None

    def disable_fast_sampling(self) -> None:
        """Disable the optional fast-sampling adapter."""
        self.fast_sampler = None
        self.fast_sampler_checkpoint = None

    def enable_fast_sampling(
        self,
        *,
        adapter_checkpoint: str,
        num_inference_steps: Optional[int] = None,
        use_fp16: bool = False,
        compile_model: bool = False,
        strict: bool = True,
    ) -> None:
        """Enable metadata-gated fast sampling from a distilled adapter checkpoint."""
        from src.optimization.lcm_lora import GraphConditionedFastSampler

        self.fast_sampler = GraphConditionedFastSampler(
            self,
            adapter_checkpoint=adapter_checkpoint,
            num_inference_steps=num_inference_steps,
            use_fp16=use_fp16,
            compile_model=compile_model,
            strict=strict,
        )
        self.fast_sampler_checkpoint = str(adapter_checkpoint)

    @torch.no_grad()
    def fast_sample(
        self,
        *,
        context: Tensor,
        shape: Tuple[int, ...],
        graph_data: Optional[Dict[str, Tensor]] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tensor:
        """Run the configured fast-sampling backend, if available.

        Note: adapter-controlled fast sampling uses the adapter's configured
        inference-step count rather than the standard DDPM/DDIM timestep schedule.
        """
        if self.fast_sampler is None:
            raise RuntimeError("Fast sampling requested but no fast-sampler adapter is configured.")
        return self.fast_sampler.sample_fast(
            context=context,
            latent_shape=shape,
            graph_data=graph_data,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    
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
    
    def _extract_context_topology(
        self,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Prepare topology tensors for context-token cross-attention refinement."""
        if graph_data is None or context.dim() != 3:
            return None, None

        edge_index = graph_data.get("edge_index") if isinstance(graph_data, dict) else None
        if not isinstance(edge_index, torch.Tensor):
            return None, None

        seq_len = int(context.shape[1])
        adjusted = edge_index.to(context.device)

        def _batched_tensor(name: str, value: Optional[Tensor], *, min_dim: int = 3) -> Optional[Tensor]:
            if not isinstance(value, torch.Tensor):
                return None
            tensor = value.to(context.device)
            if min_dim == 3 and tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
                if int(context.shape[0]) > 1:
                    tensor = tensor.expand(int(context.shape[0]), -1, -1)
            if tensor.dim() < min_dim:
                raise ValueError(f"{name} has invalid shape {tuple(tensor.shape)}")
            if int(tensor.shape[0]) != int(context.shape[0]):
                raise ValueError(
                    f"{name} batch size {int(tensor.shape[0])} does not match context batch size {int(context.shape[0])}."
                )
            return tensor

        node_features = _batched_tensor("node_features", graph_data.get("node_features"))
        node_mask = graph_data.get("node_mask") if isinstance(graph_data, dict) else None
        if isinstance(node_mask, torch.Tensor):
            node_mask = node_mask.to(context.device)
            if node_mask.dim() == 1:
                node_mask = node_mask.unsqueeze(0)
                if int(context.shape[0]) > 1:
                    node_mask = node_mask.expand(int(context.shape[0]), -1)
            if node_mask.dim() != 2:
                raise ValueError(f"node_mask has invalid shape {tuple(node_mask.shape)}")
            if int(node_mask.shape[0]) != int(context.shape[0]):
                raise ValueError(
                    f"node_mask batch size {int(node_mask.shape[0])} does not match context batch size {int(context.shape[0])}."
                )
        else:
            node_mask = None

        has_room_anchor = bool(graph_data.get("has_room_anchor", False)) if isinstance(graph_data, dict) else False
        if node_features is not None:
            num_nodes = int(node_features.shape[1])
            if node_mask is not None and int(node_mask.shape[1]) != num_nodes:
                raise ValueError(
                    f"node_mask node count {int(node_mask.shape[1])} does not match node_features node count {num_nodes}."
                )
            required_seq_len = num_nodes + (1 if has_room_anchor else 0)
            if seq_len < required_seq_len:
                raise ValueError(
                    f"context sequence length {seq_len} is smaller than required graph token length {required_seq_len}."
                )
            if has_room_anchor:
                adjusted = adjusted + 1
                if node_mask is not None:
                    anchor = torch.ones(node_mask.shape[0], 1, device=node_mask.device, dtype=node_mask.dtype)
                    node_mask = torch.cat([anchor, node_mask], dim=1)
            elif node_mask is not None and int(node_mask.shape[1]) != seq_len:
                if int(node_mask.shape[1]) > seq_len:
                    raise ValueError(
                        f"node_mask length {int(node_mask.shape[1])} exceeds context sequence length {seq_len}."
                    )
                node_mask = F.pad(node_mask, (0, seq_len - int(node_mask.shape[1])), value=0)

        return adjusted, node_mask

    def _extract_spatial_graph_context(
        self,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Optional[Dict[str, Tensor]]:
        """Prepare graph-node tokens and room topology maps for spatial conditioning."""
        if not isinstance(graph_data, dict):
            return None

        spatial: Dict[str, Tensor] = {}
        batch_size = int(context.shape[0])

        def _normalize_node_tensor(name: str, value: Optional[Tensor], *, trailing_dim: Optional[int] = None) -> Optional[Tensor]:
            if not isinstance(value, torch.Tensor):
                return None
            tensor = value.to(context.device)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
                if batch_size > 1:
                    tensor = tensor.expand(batch_size, -1, -1)
            if tensor.dim() < 3:
                raise ValueError(f"{name} has invalid shape {tuple(tensor.shape)}")
            if int(tensor.shape[0]) != batch_size:
                raise ValueError(f"{name} batch size {int(tensor.shape[0])} does not match context batch size {batch_size}.")
            if trailing_dim is not None and int(tensor.shape[-1]) != trailing_dim:
                raise ValueError(f"{name} trailing dimension must be {trailing_dim}, got {int(tensor.shape[-1])}.")
            return tensor

        node_features = _normalize_node_tensor("node_features", graph_data.get("node_features"))
        if node_features is None:
            return None
        num_nodes = int(node_features.shape[1])
        has_room_anchor = bool(graph_data.get("has_room_anchor", False))

        if context.dim() == 3 and int(context.shape[1]) > 1:
            if num_nodes > 0:
                needed = num_nodes + (1 if has_room_anchor else 0)
                if int(context.shape[1]) < needed:
                    raise ValueError(
                        f"context sequence length {int(context.shape[1])} is too short for has_room_anchor={has_room_anchor} and num_nodes={num_nodes}."
                    )
                if has_room_anchor:
                    graph_nodes = context[:, 1:1 + num_nodes, :]
                else:
                    graph_nodes = context[:, :num_nodes, :]
                spatial["graph_nodes"] = graph_nodes

        node_mask = graph_data.get("node_mask")
        if isinstance(node_mask, torch.Tensor):
            node_mask = node_mask.to(context.device)
            if node_mask.dim() == 1:
                node_mask = node_mask.unsqueeze(0)
                if batch_size > 1:
                    node_mask = node_mask.expand(batch_size, -1)
            if node_mask.dim() != 2:
                raise ValueError(f"node_mask has invalid shape {tuple(node_mask.shape)}")
            if int(node_mask.shape[0]) != batch_size or int(node_mask.shape[1]) != num_nodes:
                raise ValueError(
                    f"node_mask shape {tuple(node_mask.shape)} must match [B, N] = ({batch_size}, {num_nodes})."
                )
            spatial["node_mask"] = node_mask

        for key in ("edge_index", "tpe", "node_positions", "current_node_distance", "room_topology_map"):
            value = graph_data.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(context.device)
            if key in {"tpe", "node_positions", "current_node_distance"} and value.dim() == 2:
                value = value.unsqueeze(0)
                if batch_size > 1:
                    value = value.expand(batch_size, -1, -1)
            if key == "tpe" and value.dim() >= 3 and int(value.shape[1]) != num_nodes:
                raise ValueError(f"tpe node count {int(value.shape[1])} does not match node_features node count {num_nodes}.")
            if key == "node_positions" and value.dim() >= 3:
                if int(value.shape[1]) != num_nodes or int(value.shape[-1]) != 2:
                    raise ValueError(
                        f"node_positions shape {tuple(value.shape)} must match [B, N, 2] with N={num_nodes}."
                    )
            if key == "current_node_distance" and value.dim() >= 3:
                if int(value.shape[1]) != num_nodes or int(value.shape[-1]) != 4:
                    raise ValueError(
                        "current_node_distance shape "
                        f"{tuple(value.shape)} must match [B, N, 4] with N={num_nodes}."
                    )
            if key == "room_topology_map" and value.dim() == 3:
                value = value.unsqueeze(0)
                if batch_size > 1:
                    value = value.expand(batch_size, -1, -1, -1)
            if key == "room_topology_map" and int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"room_topology_map batch size {int(value.shape[0])} does not match context batch size {batch_size}."
                )
            if key == "edge_index" and value.dim() == 3 and int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"edge_index batch size {int(value.shape[0])} does not match context batch size {batch_size}."
                )
            target_key = "node_tpe" if key == "tpe" else key
            spatial[target_key] = value

        return spatial or None

    def _sampling_dtype(self) -> torch.dtype:
        """Return the floating-point dtype currently used by the denoiser."""
        for param in self.denoiser.parameters():
            if param.is_floating_point():
                return param.dtype
        for buffer in self.denoiser.buffers():
            if buffer.is_floating_point():
                return buffer.dtype
        for param in self.parameters():
            if param.is_floating_point():
                return param.dtype
        for buffer in self.buffers():
            if buffer.is_floating_point():
                return buffer.dtype
        return torch.float32

    @staticmethod
    def _cast_tensor_for_sampling(
        value: Optional[Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if not isinstance(value, torch.Tensor):
            return value
        kwargs: Dict[str, Any] = {}
        if value.device != device:
            kwargs["device"] = device
        if value.is_floating_point() and value.dtype != dtype:
            kwargs["dtype"] = dtype
        return value.to(**kwargs) if kwargs else value

    def _cast_spatial_graph_context_for_sampling(
        self,
        spatial_graph_data: Optional[Dict[str, Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, Tensor]]:
        if spatial_graph_data is None:
            return None
        return {
            key: self._cast_tensor_for_sampling(value, device=device, dtype=dtype)
            for key, value in spatial_graph_data.items()
        }

    def _predict_noise_cfg(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        *,
        cached_topology: Optional[Tuple] = None,
        cached_spatial: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Predict noise with Classifier-Free Guidance (CFG).
        
        At inference: ε̃ = ε_uncond + s * (ε_cond - ε_uncond)
        where s = cfg_scale. When s=1 → standard conditional, s>1 → stronger conditioning.
        
        PERF-02: accepts pre-computed topology/spatial context to avoid
        redundant extraction at every sampling step.
        """
        if cached_topology is not None:
            context_edge_index, context_node_mask = cached_topology
        else:
            context_edge_index, context_node_mask = self._extract_context_topology(context, graph_data)
        
        if cached_spatial is not None:
            spatial_graph_data = cached_spatial
        else:
            spatial_graph_data = self._extract_spatial_graph_context(context, graph_data)

        context = self._cast_tensor_for_sampling(context, device=x_t.device, dtype=x_t.dtype)
        context_edge_index = self._cast_tensor_for_sampling(
            context_edge_index, device=x_t.device, dtype=x_t.dtype
        )
        context_node_mask = self._cast_tensor_for_sampling(
            context_node_mask, device=x_t.device, dtype=x_t.dtype
        )
        spatial_graph_data = self._cast_spatial_graph_context_for_sampling(
            spatial_graph_data, device=x_t.device, dtype=x_t.dtype
        )

        cfg_scale = self._cfg_scale_for_timestep(t).to(device=x_t.device, dtype=x_t.dtype)
        pred_cond = self.denoiser(
            x_t,
            t,
            context,
            context_edge_index=context_edge_index,
            context_node_mask=context_node_mask,
            spatial_graph_data=spatial_graph_data,
        )
        
        if torch.any(cfg_scale > (1.0 + 1e-6)):
            # Unconditional prediction (zero context)
            pred_uncond = self.denoiser(x_t, t, torch.zeros_like(context))
            # CFG interpolation
            prediction = pred_uncond + cfg_scale[:, None, None, None] * (pred_cond - pred_uncond)
        else:
            prediction = pred_cond

        pag_scale = float(getattr(self, "pag_scale", 0.0))
        if pag_scale > 0.0:
            updated = self.set_self_attention_perturbation("identity")
            if updated > 0:
                try:
                    pred_perturbed = self.denoiser(
                        x_t,
                        t,
                        context,
                        context_edge_index=context_edge_index,
                        context_node_mask=context_node_mask,
                        spatial_graph_data=spatial_graph_data,
                    )
                finally:
                    self.set_self_attention_perturbation("none")
                prediction = prediction + pag_scale * (prediction - pred_perturbed)

        return prediction
    
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

    def _apply_logic_guidance_to_prediction(
        self,
        *,
        x_t: Tensor,
        pred_x0: Tensor,
        pred_noise: Tensor,
        t: int,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply LogicNet guidance to a DDIM-style clean-latent prediction.

        LogicNet is trained/evaluated on clean VQ-VAE latents, not noisy x_t.
        Guiding pred_x0 directly keeps the symbolic gradient aligned with the
        object it scores, then pred_noise is recomputed for the DDIM update.
        """
        if self.guidance.logic_net is None or self.guidance.guidance_scale <= 0:
            return pred_x0, pred_noise

        self._sync_logic_temperature_to_timestep(int(t))

        guidance_grad = self.guidance.compute_guidance(
            pred_x0,
            graph_data,
            t=int(t),
            num_timesteps=int(self.num_timesteps),
            scale_multiplier=self._guidance_timestep_scale(int(t)),
        )
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
        pred_x0 = torch.clamp(pred_x0 - guidance_grad, -1.0, 1.0)
        if float(sqrt_one_minus_alpha_t.detach().abs().item()) > 1e-8:
            pred_noise = (x_t - sqrt_alpha_t * pred_x0) / torch.clamp(sqrt_one_minus_alpha_t, min=1e-8)
        return pred_x0, pred_noise

    def _sync_logic_temperature_to_timestep(self, t: int) -> None:
        """Anneal LogicNet sharpness from soft at high noise to sharp near x0."""
        logic_net = self.guidance.logic_net
        if logic_net is None or not hasattr(logic_net, "update_temperature"):
            return
        try:
            sqrt_alpha = float(self.sqrt_alphas_cumprod[int(t)].detach().clamp(0.0, 1.0).item())
            logic_net.update_temperature(sqrt_alpha)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            logger.debug("Skipping timestep-synced LogicNet temperature update", exc_info=True)

    def _guidance_timestep_scale(self, t: int) -> float:
        """DPS-style scale: guidance is weak at high noise and strong near clean x0."""
        try:
            return float(self.sqrt_alphas_cumprod[int(t)].detach().clamp(0.0, 1.0).item())
        except (AttributeError, RuntimeError, ValueError, TypeError):
            return 1.0

    def _q_posterior_from_pred_x0(
        self,
        pred_x0: Tensor,
        x_t: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute q(x_{t-1} | x_t, pred_x0) posterior parameters."""
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
    
    def p_mean_variance(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        clip_denoised: bool = True,
        *,
        cached_topology: Optional[Tuple] = None,
        cached_spatial: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute p(x_{t-1} | x_t) mean and variance.
        Now supports CFG, v-prediction, and topology caching.
        """
        # Predict with CFG
        prediction = self._predict_noise_cfg(
            x_t, t, context, graph_data=graph_data,
            cached_topology=cached_topology, cached_spatial=cached_spatial,
        )
        
        # Convert to pred_x0 and pred_noise
        pred_x0, _pred_noise = self._convert_prediction(prediction, x_t, t)
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        return self._q_posterior_from_pred_x0(pred_x0, x_t, t)
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: Tensor,
        t: int,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        *,
        cached_topology: Optional[Tuple] = None,
        cached_spatial: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Sample x_{t-1} from x_t (single step).
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        
        prediction = self._predict_noise_cfg(
            x_t,
            t_tensor,
            context,
            graph_data=graph_data,
            cached_topology=cached_topology,
            cached_spatial=cached_spatial,
        )
        pred_x0, pred_noise = self._convert_prediction(prediction, x_t, t_tensor)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        pred_x0, _pred_noise = self._apply_logic_guidance_to_prediction(
            x_t=x_t,
            pred_x0=pred_x0,
            pred_noise=pred_noise,
            t=int(t),
            graph_data=graph_data,
        )
        
        # DDPM and DDIM both guide the predicted clean latent, then rebuild the
        # posterior from that guided pred_x0.
        mean, variance, _ = self._q_posterior_from_pred_x0(pred_x0, x_t, t_tensor)
        
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
        _B = context.shape[0]
        
        # PERF-02: Cache topology extraction (static throughout sampling)
        cached_topology = self._extract_context_topology(context, graph_data)
        cached_spatial = self._extract_spatial_graph_context(context, graph_data)

        sample_dtype = self._sampling_dtype()
        context = self._cast_tensor_for_sampling(context, device=device, dtype=sample_dtype)
        work_dtype = torch.float32 if sample_dtype in {torch.float16, torch.bfloat16} else sample_dtype

        # Start from noise
        x_t = torch.randn(shape, device=device, dtype=sample_dtype)
        
        intermediates = [x_t] if return_intermediates else None
        
        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            x_t = self.p_sample(
                x_t, t, context, graph_data,
                cached_topology=cached_topology, cached_spatial=cached_spatial,
            )
            
            if return_intermediates:
                intermediates.append(x_t)
        
        if return_intermediates:
            return x_t, intermediates
        return x_t

    @staticmethod
    def _build_reverse_t_schedule(start_t: int, num_steps: int) -> List[int]:
        """
        Build a descending timestep schedule that always includes both endpoints.

        DDIM-style striding must start from the actual noised timestep and end at 0.
        Simple ``range(..., step)`` schedules can silently skip the start timestep,
        which misaligns reverse updates with the forward noise level.
        """
        start_t = max(0, int(start_t))
        if start_t == 0:
            return [0]

        # Include both start_t and 0. A single denoise step is not meaningful when
        # starting from a noisy latent, so we ensure both endpoints are present.
        num_steps = max(2, min(int(num_steps), start_t + 1))
        raw = torch.linspace(float(start_t), 0.0, steps=num_steps, dtype=torch.float64)

        schedule: List[int] = []
        last_t: Optional[int] = None
        for value in raw.tolist():
            t = max(0, min(start_t, int(round(value))))
            if last_t is None or t < last_t:
                schedule.append(t)
                last_t = t

        if not schedule or schedule[0] != start_t:
            schedule.insert(0, start_t)
        if schedule[-1] != 0:
            schedule.append(0)
        return schedule

    @staticmethod
    def _compute_ddim_sigma(
        alpha_t: Tensor,
        alpha_prev: Tensor,
        *,
        eta: float,
    ) -> Tensor:
        """
        Compute the DDIM stochasticity term safely.

        Deterministic DDIM (`eta == 0`) should yield sigma=0 exactly. In reduced
        precision, evaluating the full formula at the last step can produce 0/0
        when `alpha_t` rounds to 1, which then turns the latent into NaNs.
        """
        eta = float(eta)
        if eta <= 0.0:
            return torch.zeros_like(alpha_t)

        one = torch.ones_like(alpha_t)
        alpha_prev_safe = torch.clamp(alpha_prev, min=1e-8)
        denom = torch.clamp(one - alpha_t, min=0.0)
        numer = torch.clamp(one - alpha_prev, min=0.0)
        ratio = torch.where(
            denom > 1e-8,
            numer / denom,
            torch.zeros_like(alpha_t),
        )
        direction = torch.clamp(one - (alpha_t / alpha_prev_safe), min=0.0)
        eta_tensor = alpha_t.new_tensor(eta)
        return eta_tensor * torch.sqrt(torch.clamp(ratio, min=0.0)) * torch.sqrt(direction)

    @staticmethod
    def _sanitize_sampling_tensor(
        value: Tensor,
        *,
        fallback: Optional[Tensor] = None,
        clamp_range: Optional[Tuple[float, float]] = None,
    ) -> Tensor:
        """Replace non-finite values with a finite fallback during sampling."""
        if torch.isfinite(value).all():
            if clamp_range is not None:
                lo, hi = clamp_range
                return torch.clamp(value, lo, hi)
            return value

        safe_fallback = fallback if isinstance(fallback, torch.Tensor) else torch.zeros_like(value)
        safe = torch.where(torch.isfinite(value), value, safe_fallback)
        safe = torch.nan_to_num(safe, nan=0.0, posinf=1.0, neginf=-1.0)
        if clamp_range is not None:
            lo, hi = clamp_range
            safe = torch.clamp(safe, lo, hi)
        return safe
    
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
        timesteps = self._build_reverse_t_schedule(
            int(self.num_timesteps) - 1,
            num_steps,
        )
        
        # PERF-02: Cache topology extraction (static throughout sampling)
        cached_topology = self._extract_context_topology(context, graph_data)
        cached_spatial = self._extract_spatial_graph_context(context, graph_data)

        sample_dtype = self._sampling_dtype()
        context = self._cast_tensor_for_sampling(context, device=device, dtype=sample_dtype)
        work_dtype = torch.float32 if sample_dtype in {torch.float16, torch.bfloat16} else sample_dtype

        # Start from noise
        x_t = torch.randn(shape, device=device, dtype=sample_dtype)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict with CFG (using cached topology)
            prediction = self._predict_noise_cfg(
                x_t, t_tensor, context, graph_data=graph_data,
                cached_topology=cached_topology, cached_spatial=cached_spatial,
            )
            prediction = self._sanitize_sampling_tensor(
                prediction,
                fallback=torch.zeros_like(prediction),
            )
            
            # Convert to (pred_x0, pred_noise) supporting v-prediction
            pred_x0, pred_noise = self._convert_prediction(prediction, x_t, t_tensor)
            pred_x0 = self._sanitize_sampling_tensor(
                pred_x0,
                fallback=x_t,
                clamp_range=(-1.0, 1.0),
            )
            pred_noise = self._sanitize_sampling_tensor(
                pred_noise,
                fallback=torch.zeros_like(pred_noise),
            )
            
            # Apply LogicNet gradient guidance BEFORE the DDIM step
            # (Dhariwal & Nichol 2021 §3.2: guidance modifies score prediction)
            pred_x0, pred_noise = self._apply_logic_guidance_to_prediction(
                x_t=x_t,
                pred_x0=pred_x0,
                pred_noise=pred_noise,
                t=int(t),
                graph_data=graph_data,
            )
            pred_x0 = self._sanitize_sampling_tensor(
                pred_x0,
                fallback=x_t,
                clamp_range=(-1.0, 1.0),
            )
            pred_noise = self._sanitize_sampling_tensor(
                pred_noise,
                fallback=torch.zeros_like(pred_noise),
            )
            
            # Get next timestep
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev].to(device=device, dtype=work_dtype)
            else:
                alpha_prev = torch.tensor(1.0, device=device, dtype=work_dtype)
            
            # CRITICAL-3 fix: compute alpha_t from current timestep
            alpha_t = self.alphas_cumprod[t].to(device=device, dtype=work_dtype)
            pred_x0_work = pred_x0.to(dtype=work_dtype)
            pred_noise_work = pred_noise.to(dtype=work_dtype)

            if int(t) == 0:
                x_t = pred_x0_work.to(dtype=sample_dtype)
                continue
            
            # DDIM update
            sigma = self._compute_ddim_sigma(alpha_t, alpha_prev, eta=eta)

            pred_dir_scale = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0))
            noise = (
                torch.randn(shape, device=device, dtype=work_dtype)
                if t > 0
                else torch.zeros(shape, device=device, dtype=work_dtype)
            )

            x_next = torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * pred_x0_work
            x_next = x_next + pred_dir_scale * pred_noise_work + sigma * noise
            x_next = self._sanitize_sampling_tensor(
                x_next,
                fallback=pred_x0_work,
                clamp_range=(-1.0, 1.0),
            )

            x_t = x_next.to(dtype=sample_dtype)
        
        return x_t

    @torch.no_grad()
    def inpaint(
        self,
        x_0: Tensor,
        mask: Tensor,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        num_steps: int = 30,
        noise_strength: float = 0.35,
    ) -> Tensor:
        """
        Masked latent inpainting using reverse diffusion with hard constraint injection.

        Args:
            x_0: Reference latent to preserve on unmasked regions [B, C, H, W]
            mask: Binary editable mask [B, 1|C, H, W], 1=editable, 0=preserve
            context: Conditioning [B, context_dim]
            graph_data: Optional graph data for logic guidance
            num_steps: Number of reverse diffusion steps
            noise_strength: Fraction of diffusion schedule to noise (0.0-1.0).
                SDEdit (Meng et al. 2021) recommends 0.3-0.5 for partial edits.

        Returns:
            Inpainted latent tensor [B, C, H, W]
        """
        if num_steps <= 0:
            return x_0

        device = x_0.device
        B = int(x_0.shape[0])

        edit_mask = mask.to(device=device, dtype=x_0.dtype)
        if edit_mask.dim() == 3:
            edit_mask = edit_mask.unsqueeze(1)
        if edit_mask.shape[1] == 1 and x_0.shape[1] > 1:
            edit_mask = edit_mask.expand(-1, x_0.shape[1], -1, -1)
        edit_mask = torch.clamp(edit_mask, 0.0, 1.0)

        preserve_mask = 1.0 - edit_mask

        # Compute noise level from noise_strength (fraction of full schedule).
        # SDEdit (Meng et al. 2021): 30-50% strength is optimal for partial edits.
        # Decoupled from num_steps so callers can control both independently.
        noise_strength = max(0.01, min(1.0, float(noise_strength)))
        start_t = max(1, min(self.num_timesteps - 1, int(self.num_timesteps * noise_strength)))

        t_schedule = self._build_reverse_t_schedule(start_t, num_steps)
        t_start = torch.full((B,), t_schedule[0], device=device, dtype=torch.long)
        x_t = self.q_sample(x_0, t_start)

        # Cache graph-conditioned context just once; the topology is static through
        # the reverse trajectory.
        cached_topology = self._extract_context_topology(context, graph_data)
        cached_spatial = self._extract_spatial_graph_context(context, graph_data)

        for i, t in enumerate(t_schedule):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            prediction = self._predict_noise_cfg(
                x_t,
                t_tensor,
                context,
                graph_data=graph_data,
                cached_topology=cached_topology,
                cached_spatial=cached_spatial,
            )
            pred_x0, pred_noise = self._convert_prediction(prediction, x_t, t_tensor)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            pred_x0, pred_noise = self._apply_logic_guidance_to_prediction(
                x_t=x_t,
                pred_x0=pred_x0,
                pred_noise=pred_noise,
                t=int(t),
                graph_data=graph_data,
            )

            if i + 1 < len(t_schedule):
                t_prev = int(t_schedule[i + 1])
                alpha_prev = self.alphas_cumprod[t_prev]
                t_prev_tensor = torch.full((B,), t_prev, device=device, dtype=torch.long)
                known_prev = self.q_sample(x_0, t_prev_tensor)
            else:
                t_prev = -1
                alpha_prev = torch.tensor(1.0, device=device, dtype=x_t.dtype)
                known_prev = x_0

            pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * pred_noise
            x_prev = torch.sqrt(alpha_prev) * pred_x0 + pred_dir
            x_t = edit_mask * x_prev + preserve_mask * known_prev

        return x_t
    
    def training_loss(
        self,
        x_0: Tensor,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
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
            if int(context.shape[0]) != B:
                raise ValueError(
                    f"context batch size {int(context.shape[0])} does not match latent batch size {B}."
                )
            # Keep/drop one whole conditioning payload per sample and broadcast
            # across any trailing conditioning dimensions (e.g. [B, C] or [B, N, C]).
            keep_mask = (torch.rand(B, device=device) > self.cfg_dropout_prob).to(dtype=context.dtype)
            mask_shape = [B] + [1] * max(0, context.dim() - 1)
            context = context * keep_mask.view(*mask_shape)
        
        # Get noisy samples
        x_t = self.q_sample(x_0, t, noise)
        
        alignment_weight = 0.0
        alignment_node_indices = None
        alignment_positions = None
        alignment_valid_mask = None
        if isinstance(graph_data, dict):
            alignment_weight = float(graph_data.get("spatial_alignment_weight", 0.0) or 0.0)
            alignment_node_indices = graph_data.get("spatial_alignment_node_indices")
            alignment_positions = graph_data.get("spatial_alignment_positions")
            alignment_valid_mask = graph_data.get("spatial_alignment_valid_mask")
        use_spatial_alignment = (
            alignment_weight > 0.0
            and isinstance(alignment_node_indices, torch.Tensor)
            and isinstance(alignment_positions, torch.Tensor)
        )

        if use_spatial_alignment:
            updated = self.set_spatial_attention_capture(True)
            if updated <= 0:
                raise RuntimeError("Spatial alignment was requested, but the denoiser has no SpatialGraphConditioner modules.")

        # Predict
        context_edge_index, context_node_mask = self._extract_context_topology(context, graph_data)
        spatial_graph_data = self._extract_spatial_graph_context(context, graph_data)
        try:
            prediction = self.denoiser(
                x_t,
                t,
                context,
                context_edge_index=context_edge_index,
                context_node_mask=context_node_mask,
                spatial_graph_data=spatial_graph_data,
            )
        except Exception:
            if use_spatial_alignment:
                self.set_spatial_attention_capture(False)
            raise
        finally:
            if not use_spatial_alignment:
                self.set_spatial_attention_capture(False)
        
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
        if use_spatial_alignment:
            try:
                alignment_loss = self.spatial_alignment_loss(
                    alignment_node_indices.to(device=device),
                    alignment_positions.to(device=device),
                    valid_mask=(
                        alignment_valid_mask.to(device=device)
                        if isinstance(alignment_valid_mask, torch.Tensor)
                        else None
                    ),
                )
                loss = loss + float(alignment_weight) * alignment_loss
            finally:
                self.set_spatial_attention_capture(False)
        
        return loss

    def flow_matching_loss(
        self,
        x_0: Tensor,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute a rectified-flow / conditional flow-matching velocity loss.

        This is an opt-in training ablation. It reuses the existing U-Net
        conditioner/backbone and trains it to predict the straight-line
        velocity from clean latent data to Gaussian noise:

            x_t = (1 - t) * x_0 + t * eps
            target = eps - x_0

        The default sampler remains the DDPM/DDIM diffusion path; experiments
        using this objective should report it separately until a matching ODE
        sampler is added.
        """
        B = x_0.shape[0]
        device = x_0.device
        dtype = x_0.dtype

        if noise is None:
            noise = torch.randn_like(x_0)

        t_cont = torch.rand(B, device=device, dtype=dtype)
        view_shape = (B,) + (1,) * max(0, x_0.dim() - 1)
        t_view = t_cont.view(*view_shape)
        x_t = (1.0 - t_view) * x_0 + t_view * noise
        # Flow matching is continuous-time. Keep the same sinusoidal embedding
        # scale as diffusion timesteps, but avoid quantizing t into integer bins.
        t = t_cont * float(max(1, self.num_timesteps - 1))

        if self.training and self.cfg_dropout_prob > 0:
            if int(context.shape[0]) != B:
                raise ValueError(
                    f"context batch size {int(context.shape[0])} does not match latent batch size {B}."
                )
            keep_mask = (torch.rand(B, device=device) > self.cfg_dropout_prob).to(dtype=context.dtype)
            mask_shape = [B] + [1] * max(0, context.dim() - 1)
            context = context * keep_mask.view(*mask_shape)

        alignment_weight = 0.0
        alignment_node_indices = None
        alignment_positions = None
        alignment_valid_mask = None
        if isinstance(graph_data, dict):
            alignment_weight = float(graph_data.get("spatial_alignment_weight", 0.0) or 0.0)
            alignment_node_indices = graph_data.get("spatial_alignment_node_indices")
            alignment_positions = graph_data.get("spatial_alignment_positions")
            alignment_valid_mask = graph_data.get("spatial_alignment_valid_mask")
        use_spatial_alignment = (
            alignment_weight > 0.0
            and isinstance(alignment_node_indices, torch.Tensor)
            and isinstance(alignment_positions, torch.Tensor)
        )
        if use_spatial_alignment:
            updated = self.set_spatial_attention_capture(True)
            if updated <= 0:
                raise RuntimeError("Spatial alignment was requested, but the denoiser has no SpatialGraphConditioner modules.")

        context_edge_index, context_node_mask = self._extract_context_topology(context, graph_data)
        spatial_graph_data = self._extract_spatial_graph_context(context, graph_data)
        try:
            prediction = self.denoiser(
                x_t,
                t,
                context,
                context_edge_index=context_edge_index,
                context_node_mask=context_node_mask,
                spatial_graph_data=spatial_graph_data,
            )
        except Exception:
            if use_spatial_alignment:
                self.set_spatial_attention_capture(False)
            raise
        finally:
            if not use_spatial_alignment:
                self.set_spatial_attention_capture(False)
        target = noise - x_0
        loss = F.mse_loss(prediction, target)
        if use_spatial_alignment:
            try:
                alignment_loss = self.spatial_alignment_loss(
                    alignment_node_indices.to(device=device),
                    alignment_positions.to(device=device),
                    valid_mask=(
                        alignment_valid_mask.to(device=device)
                        if isinstance(alignment_valid_mask, torch.Tensor)
                        else None
                    ),
                )
                loss = loss + float(alignment_weight) * alignment_loss
            finally:
                self.set_spatial_attention_capture(False)
        return loss

    def denoising_preference_score(
        self,
        x_0: Tensor,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        *,
        noise: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        objective: str = "diffusion",
    ) -> Tensor:
        """
        Return a per-sample denoising score for preference optimization.

        Scores are negative prediction errors, so higher is better. The method
        supports both standard diffusion targets and the flow-matching velocity
        target, making it usable for Diffusion-DPO style pairwise losses.
        """
        B = int(x_0.shape[0])
        device = x_0.device
        objective = str(objective).strip().lower()
        if noise is None:
            noise = torch.randn_like(x_0)
        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps, (B,), device=device)
        else:
            timesteps = timesteps.to(device=device, dtype=torch.long)
            if timesteps.dim() == 0:
                timesteps = timesteps.expand(B)
            if int(timesteps.shape[0]) != B:
                raise ValueError(f"timesteps batch size {int(timesteps.shape[0])} must match x_0 batch size {B}.")

        if objective == "flow_matching":
            t_cont = timesteps.to(dtype=x_0.dtype) / float(max(1, self.num_timesteps - 1))
            t_view = t_cont.view((B,) + (1,) * max(0, x_0.dim() - 1))
            x_t = (1.0 - t_view) * x_0 + t_view * noise
            target = noise - x_0
        elif objective == "diffusion":
            x_t = self.q_sample(x_0, timesteps, noise)
            if self.prediction_type == 'v':
                sqrt_alpha_t = self.sqrt_alphas_cumprod[timesteps][:, None, None, None]
                sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None, None]
                target = sqrt_alpha_t * noise - sqrt_one_minus_alpha_t * x_0
            else:
                target = noise
        else:
            raise ValueError(f"objective must be 'diffusion' or 'flow_matching', got {objective!r}.")

        context_edge_index, context_node_mask = self._extract_context_topology(context, graph_data)
        spatial_graph_data = self._extract_spatial_graph_context(context, graph_data)
        prediction = self.denoiser(
            x_t,
            timesteps,
            context,
            context_edge_index=context_edge_index,
            context_node_mask=context_node_mask,
            spatial_graph_data=spatial_graph_data,
        )
        per_sample_error = F.mse_loss(prediction, target, reduction='none').mean(dim=[1, 2, 3])
        return -per_sample_error

    def diffusion_dpo_loss(
        self,
        chosen_x0: Tensor,
        rejected_x0: Tensor,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        *,
        rejected_context: Optional[Tensor] = None,
        rejected_graph_data: Optional[Dict[str, Tensor]] = None,
        reference_model: Optional["LatentDiffusionModel"] = None,
        beta: float = 0.1,
        objective: str = "diffusion",
        noise: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Pairwise Diffusion-DPO loss for solver/human preference pairs.

        The preferred sample should receive a higher denoising score than the
        rejected sample relative to an optional frozen reference model.
        """
        if chosen_x0.shape != rejected_x0.shape:
            raise ValueError(
                f"chosen_x0 and rejected_x0 must have the same shape, got {tuple(chosen_x0.shape)} and {tuple(rejected_x0.shape)}."
            )
        rejected_context = context if rejected_context is None else rejected_context
        rejected_graph_data = graph_data if rejected_graph_data is None else rejected_graph_data
        B = int(chosen_x0.shape[0])
        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps, (B,), device=chosen_x0.device)
        else:
            timesteps = timesteps.to(device=chosen_x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(chosen_x0)

        chosen_score = self.denoising_preference_score(
            chosen_x0,
            context,
            graph_data,
            noise=noise,
            timesteps=timesteps,
            objective=objective,
        )
        rejected_score = self.denoising_preference_score(
            rejected_x0,
            rejected_context,
            rejected_graph_data,
            noise=noise,
            timesteps=timesteps,
            objective=objective,
        )
        policy_logit = chosen_score - rejected_score

        if reference_model is not None:
            with torch.no_grad():
                ref_chosen = reference_model.denoising_preference_score(
                    chosen_x0,
                    context,
                    graph_data,
                    noise=noise,
                    timesteps=timesteps,
                    objective=objective,
                )
                ref_rejected = reference_model.denoising_preference_score(
                    rejected_x0,
                    rejected_context,
                    rejected_graph_data,
                    noise=noise,
                    timesteps=timesteps,
                    objective=objective,
                )
            reference_logit = ref_chosen - ref_rejected
        else:
            reference_logit = torch.zeros_like(policy_logit)

        margin = policy_logit - reference_logit
        loss = -F.logsigmoid(float(beta) * margin).mean()
        metrics = {
            "dpo_margin": margin.detach().mean(),
            "dpo_model_margin": policy_logit.detach().mean(),
            "dpo_reference_margin": reference_logit.detach().mean(),
            "dpo_accuracy": (margin.detach() > 0).float().mean(),
            "chosen_score": chosen_score.detach().mean(),
            "rejected_score": rejected_score.detach().mean(),
        }
        return loss, metrics

    def dpo_preference_loss(
        self,
        preferred_x0: Tensor,
        rejected_x0: Tensor,
        context: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
        *,
        reference_model: Optional["LatentDiffusionModel"] = None,
        beta: float = 0.1,
        objective: str = "diffusion",
        noise: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compatibility wrapper using explicit preferred/rejected naming."""
        loss, metrics = self.diffusion_dpo_loss(
            preferred_x0,
            rejected_x0,
            context,
            graph_data=graph_data,
            reference_model=reference_model,
            beta=beta,
            objective=objective,
            noise=noise,
            timesteps=timesteps,
        )
        renamed = {
            "dpo_model_margin": metrics["dpo_model_margin"],
            "dpo_reference_margin": metrics["dpo_reference_margin"],
            "dpo_accuracy": metrics["dpo_accuracy"],
            "dpo_preferred_score": metrics["chosen_score"],
            "dpo_rejected_score": metrics["rejected_score"],
        }
        return loss, renamed


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
    cfg_schedule_mode: str = "constant",
    cfg_schedule_min_scale: float = 1.0,
    cfg_schedule_power: float = 1.0,
    pag_scale: float = 0.0,
    min_snr_gamma: float = 5.0,
    topology_conditioning_mode: str = "additive",
    denoiser_backbone: str = "unet",
    unet_channel_mult: Tuple[int, ...] = (1, 2, 4),
    unet_num_res_blocks: int = 2,
    unet_attention_resolutions: Tuple[int, ...] = (1, 2),
    unet_num_heads: int = 8,
    unet_dropout: float = 0.1,
    dit_depth: int = 4,
    dit_patch_size: int = 1,
    dit_mlp_ratio: float = 4.0,
    graph_auto_linear_attention_nodes: int = 128,
    spatial_graph_gate_init: float = -2.0,
    spatial_topology_gate_init: float = -2.0,
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
        cfg_schedule_mode: Inference CFG scheduling mode
        cfg_schedule_min_scale: Minimum CFG scale near final denoising steps
        cfg_schedule_power: Shape parameter for non-constant CFG schedules
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
        cfg_schedule_mode=cfg_schedule_mode,
        cfg_schedule_min_scale=cfg_schedule_min_scale,
        cfg_schedule_power=cfg_schedule_power,
        pag_scale=pag_scale,
        min_snr_gamma=min_snr_gamma,
        topology_conditioning_mode=topology_conditioning_mode,
        denoiser_backbone=denoiser_backbone,
        unet_channel_mult=unet_channel_mult,
        unet_num_res_blocks=unet_num_res_blocks,
        unet_attention_resolutions=unet_attention_resolutions,
        unet_num_heads=unet_num_heads,
        unet_dropout=unet_dropout,
        dit_depth=dit_depth,
        dit_patch_size=dit_patch_size,
        dit_mlp_ratio=dit_mlp_ratio,
        graph_auto_linear_attention_nodes=graph_auto_linear_attention_nodes,
        spatial_graph_gate_init=spatial_graph_gate_init,
        spatial_topology_gate_init=spatial_topology_gate_init,
        **kwargs,
    )
