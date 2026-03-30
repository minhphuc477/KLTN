"""
Graph-to-Grid Cross-Attention Module
=====================================

Per-position graph attention for conditioning U-Net blocks.

Current Issue:
    The CrossAttention in latent_diffusion.py only uses a single 
    context vector [B, dim], which loses spatial graph information.

Solution:
    This module provides per-position attention where each grid
    position can attend to all graph nodes, with positional encoding
    to maintain spatial coherence.

Architecture:
    1. GraphNodeEncoder: Encode graph nodes with position
    2. SpatialPositionEncoder: Encode grid positions  
    3. GraphToGridCrossAttention: Per-position attention to graph

Usage:
    cross_attn = GraphToGridCrossAttention(
        grid_dim=128,
        graph_dim=256,
        num_heads=8,
    )
    
    # grid_features: [B, C, H, W] from U-Net
    # graph_nodes: [B, N_nodes, graph_dim] from GNN
    # node_positions: [B, N_nodes, 2] optional positional info
    
    output = cross_attn(grid_features, graph_nodes, node_positions=node_positions)
"""

import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.attention_kernels import HedgehogFeatureMap, hedgehog_linear_attention

logger = logging.getLogger(__name__)


# ============================================================================
# LIGHTWEIGHT GRAPH CONV
# ============================================================================

class LightweightGCNLayer(nn.Module):
    """Simple normalized A_hat X W graph convolution without external deps."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, D]
            edge_index: [2, E] or [B, 2, E]
        """
        b, n, _d = x.shape
        out = []
        for bi in range(b):
            xb = x[bi]  # [N, D]
            adj = torch.zeros(n, n, device=xb.device, dtype=xb.dtype)

            if edge_index.dim() == 3:
                ei = edge_index[bi]
            else:
                ei = edge_index

            if ei.numel() > 0:
                src = ei[0].long()
                dst = ei[1].long()
                valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
                src = src[valid]
                dst = dst[valid]
                adj[src, dst] = 1.0
                adj[dst, src] = 1.0

            # Add self-loops and normalize.
            adj = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
            deg = adj.sum(dim=1).clamp(min=1.0)
            inv_sqrt = deg.pow(-0.5)
            norm_adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]

            out.append(norm_adj @ self.linear(xb))

        return torch.stack(out, dim=0)


# ============================================================================
# POSITIONAL ENCODINGS
# ============================================================================

class SinusoidalPositionEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for grid positions.
    
    Encodes (row, col) positions using sine/cosine functions
    similar to transformer positional encoding.
    """
    
    def __init__(
        self,
        dim: int,
        max_size: Tuple[int, int] = (64, 64),
        temperature: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_h, self.max_w = max_size
        
        # Create position encoding buffer
        pe = self._create_encoding(dim, self.max_h, self.max_w, temperature)
        self.register_buffer('pe', pe)
    
    def _create_encoding(
        self, 
        dim: int, 
        max_h: int, 
        max_w: int, 
        temperature: float,
    ) -> Tensor:
        """Create 2D positional encoding (vectorized, no Python loops)."""
        half_dim = dim // 2
        half = max(1, half_dim // 2)
        
        # Create frequency bands
        omega = torch.arange(half, dtype=torch.float32)
        omega = 1.0 / (temperature ** (omega / max(1, half)))
        
        # Row and column positions
        rows = torch.arange(max_h, dtype=torch.float32)
        cols = torch.arange(max_w, dtype=torch.float32)
        
        # Compute encodings — [H, half] and [W, half]
        row_pe = rows.unsqueeze(1) * omega.unsqueeze(0)
        col_pe = cols.unsqueeze(1) * omega.unsqueeze(0)
        
        row_sin = torch.sin(row_pe)  # [H, half]
        row_cos = torch.cos(row_pe)  # [H, half]
        col_sin = torch.sin(col_pe)  # [W, half]
        col_cos = torch.cos(col_pe)  # [W, half]
        
        # Vectorized broadcast into [H, W, D] — no Python loops
        pe = torch.zeros(max_h, max_w, dim)
        pe[:, :, :half] = row_sin.unsqueeze(1).expand(max_h, max_w, half)
        pe[:, :, half:half*2] = row_cos.unsqueeze(1).expand(max_h, max_w, half)
        pe[:, :, half*2:half*3] = col_sin.unsqueeze(0).expand(max_h, max_w, half)
        remaining = dim - half * 3
        if remaining > 0:
            fill = min(remaining, half)
            pe[:, :, half*3:half*3 + fill] = col_cos[:, :fill].unsqueeze(0).expand(max_h, max_w, fill)
        
        return pe
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to grid features.
        
        Args:
            x: [B, C, H, W] grid features
            
        Returns:
            [B, C, H, W] with positional encoding added
        """
        _B, C, H, W = x.shape
        
        # Get relevant portion of encoding
        pe = self.pe[:H, :W, :C]  # [H, W, C]
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        return x + pe.to(x.device)


class GraphNodePositionEncoding(nn.Module):
    """
    Encode graph node positions in the dungeon layout.
    
    Nodes have positions like (room_row, room_col) in the
    overall dungeon grid.
    """
    
    def __init__(
        self,
        dim: int,
        max_nodes: int = 50,
    ):
        super().__init__()
        self.dim = dim
        
        # Learnable position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )
        
        # Learnable topological encoding
        self.topo_embed = nn.Sequential(
            nn.Linear(8, dim // 2),  # 8-dim TPE
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )
    
    def forward(
        self,
        node_features: Tensor,
        node_positions: Optional[Tensor] = None,
        tpe: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Add position encoding to node features.
        
        Args:
            node_features: [B, N, D] node features
            node_positions: [B, N, 2] optional (row, col) positions
            tpe: [B, N, 8] topological positional encoding
            
        Returns:
            [B, N, D] with position encoding added
        """
        output = node_features
        
        if node_positions is not None:
            pos_enc = self.pos_embed(node_positions.float())
            output = output + pos_enc
        
        if tpe is not None:
            topo_enc = self.topo_embed(tpe.float())
            output = output + topo_enc
        
        return output


# ============================================================================
# GRAPH-TO-GRID CROSS-ATTENTION
# ============================================================================

class GraphToGridCrossAttention(nn.Module):
    """
    Per-position cross-attention from grid to graph nodes.
    
    Each position (i, j) in the grid can attend to all N graph nodes,
    allowing spatial-aware conditioning from the mission graph.
    
    Architecture:
        Q: Grid features [B, H*W, C]
        K: Graph nodes [B, N, D]  
        V: Graph nodes [B, N, D]
        
        Attention: softmax(QK^T / sqrt(d)) V
        
    This replaces the single-context-vector approach with
    per-position graph awareness.
    
    Args:
        grid_dim: Grid feature dimension
        graph_dim: Graph node dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        grid_dim: int,
        graph_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
        allow_legacy_argument_swap: bool = False,
    ):
        super().__init__()
        
        self.grid_dim = grid_dim
        self.graph_dim = graph_dim
        self.num_heads = num_heads
        self.head_dim = grid_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_mode = "softmax"
        self.allow_legacy_argument_swap = bool(allow_legacy_argument_swap)

        # Grid position encoding
        self.grid_pe = SinusoidalPositionEncoding2D(grid_dim)

        # Optional graph convolution preprocessing before attention.
        self.graph_gcn = LightweightGCNLayer(graph_dim, graph_dim)
        
        # Graph node position encoding
        self.graph_pe = GraphNodePositionEncoding(graph_dim)
        
        # Normalization layers
        self.norm_grid = nn.LayerNorm(grid_dim)
        self.norm_graph = nn.LayerNorm(graph_dim)
        
        # Query from grid
        self.q_proj = nn.Linear(grid_dim, grid_dim)
        
        # Key/Value from graph (project to grid dim)
        self.k_proj = nn.Linear(graph_dim, grid_dim)
        self.v_proj = nn.Linear(graph_dim, grid_dim)
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

        # Output projection
        self.out_proj = nn.Linear(grid_dim, grid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(grid_dim),
            nn.Linear(grid_dim, grid_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(grid_dim * 4, grid_dim),
        )
        self.set_attention_mode(attention_mode)

    def set_attention_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"softmax", "linear_hedgehog"}:
            raise ValueError(
                f"Invalid attention_mode={mode!r}. Expected 'softmax' or 'linear_hedgehog'."
            )
        self.attention_mode = normalized
    
    def forward(
        self,
        grid_features: Tensor,
        graph_nodes: Tensor,
        edge_index: Optional[Tensor] = None,
        node_positions: Optional[Tensor] = None,
        node_tpe: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply graph-to-grid cross-attention.
        
        Args:
            grid_features: [B, C, H, W] grid features from U-Net
            graph_nodes: [B, N, graph_dim] graph node features
            edge_index: [2, E] or [B, 2, E] graph connectivity for GCN prepass
            node_positions: [B, N, 2] optional node positions
            node_tpe: [B, N, 8] topological positional encoding
            node_mask: [B, N] optional mask (1 = valid, 0 = padding)
            
        Returns:
            [B, C, H, W] conditioned grid features
        """
        legacy_argument_swap = (
            isinstance(edge_index, torch.Tensor)
            and edge_index.dim() >= 2
            and torch.is_floating_point(edge_index)
            and int(edge_index.shape[-1]) == 2
            and isinstance(node_positions, torch.Tensor)
            and node_positions.dim() >= 2
            and int(node_positions.shape[-1]) == 8
            and node_tpe is None
        )
        # Backward-compatibility for older positional calls:
        # module(x, graph_nodes, node_positions, node_tpe)
        if legacy_argument_swap and not self.allow_legacy_argument_swap:
            raise ValueError(
                "GraphToGridCrossAttention received legacy positional arguments that look like "
                "(node_positions, node_tpe). Pass node_positions=... and node_tpe=... as keywords, "
                "or construct the module with allow_legacy_argument_swap=True for explicit compatibility."
            )
        if legacy_argument_swap:
            node_tpe = node_positions
            node_positions = edge_index
            edge_index = None

        B, C, H, W = grid_features.shape
        if graph_nodes.dim() != 3:
            raise ValueError(
                f"GraphToGridCrossAttention graph_nodes must have shape [B, N, D], got {tuple(graph_nodes.shape)}."
            )
        if int(graph_nodes.shape[0]) != B:
            raise ValueError(
                "GraphToGridCrossAttention batch mismatch between "
                f"grid_features {tuple(grid_features.shape)} and graph_nodes {tuple(graph_nodes.shape)}."
            )
        N = graph_nodes.shape[1]

        def _normalize_batched(name: str, value: Optional[Tensor], expected_last: Optional[int] = None) -> Optional[Tensor]:
            if value is None:
                return None
            tensor = value
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
                if B > 1:
                    tensor = tensor.expand(B, -1, -1)
            if tensor.dim() < 3:
                raise ValueError(
                    f"GraphToGridCrossAttention {name} must have shape [B, N, ...], got {tuple(tensor.shape)}."
                )
            if int(tensor.shape[0]) != B:
                raise ValueError(
                    f"GraphToGridCrossAttention {name} batch size {int(tensor.shape[0])} "
                    f"does not match grid batch size {B}; shape={tuple(tensor.shape)}."
                )
            if int(tensor.shape[1]) != N:
                raise ValueError(
                    f"GraphToGridCrossAttention {name} node count {int(tensor.shape[1])} "
                    f"does not match graph_nodes N={N}; shape={tuple(tensor.shape)}."
                )
            if expected_last is not None and int(tensor.shape[-1]) != expected_last:
                raise ValueError(
                    f"GraphToGridCrossAttention {name} trailing dimension must be {expected_last}, "
                    f"got {int(tensor.shape[-1])}; shape={tuple(tensor.shape)}."
                )
            return tensor

        node_positions = _normalize_batched("node_positions", node_positions, expected_last=2)
        node_tpe = _normalize_batched("node_tpe", node_tpe)

        valid_rows = None
        if node_mask is not None:
            mask = node_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
                if B > 1:
                    mask = mask.expand(B, -1)
            if mask.dim() != 2:
                raise ValueError(
                    f"GraphToGridCrossAttention node_mask must have shape [B, N], got {tuple(mask.shape)}."
                )
            if int(mask.shape[0]) != B or int(mask.shape[1]) != N:
                raise ValueError(
                    "GraphToGridCrossAttention node_mask shape "
                    f"{tuple(mask.shape)} must match [B, N] = ({B}, {N})."
                )
            mask = mask.to(device=grid_features.device)
            valid_rows = mask.sum(dim=1) > 0
            if N > 0 and not torch.all(valid_rows):
                mask = mask.clone()
                mask[~valid_rows, 0] = 1
            node_mask = mask

        # Add positional encoding to grid
        grid_with_pe = self.grid_pe(grid_features)
        
        # Flatten grid to sequence: [B, H*W, C]
        grid_seq = grid_with_pe.view(B, C, -1).permute(0, 2, 1)
        
        # Optional GCN preprocessing preserves node-topology dependencies.
        if edge_index is not None:
            graph_nodes = self.graph_gcn(graph_nodes, edge_index)

        # Add position encoding to graph nodes
        graph_with_pe = self.graph_pe(graph_nodes, node_positions, node_tpe)
        
        # Normalize
        grid_normed = self.norm_grid(grid_seq)
        graph_normed = self.norm_graph(graph_with_pe)
        
        # Compute Q, K, V
        Q = self.q_proj(grid_normed)  # [B, H*W, C]
        K = self.k_proj(graph_normed)  # [B, N, C]
        V = self.v_proj(graph_normed)  # [B, N, C]
        
        # Reshape for multi-head attention
        Q = Q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, H*W, head_dim]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)      # [B, heads, N, head_dim]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)      # [B, heads, N, head_dim]
        
        if self.attention_mode == "linear_hedgehog":
            attn_output = hedgehog_linear_attention(
                Q,
                K,
                V,
                q_map=self.hedgehog_q,
                k_map=self.hedgehog_k,
                token_mask=node_mask,
            )
            if valid_rows is not None and not torch.all(valid_rows):
                attn_output = attn_output.clone()
                attn_output[~valid_rows] = 0.0
        else:
            # Attention scores: [B, heads, H*W, N]
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

            if node_mask is not None:
                attn_scores = attn_scores.masked_fill(node_mask[:, None, None, :] == 0, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
            if valid_rows is not None and not torch.all(valid_rows):
                attn_output = attn_output.clone()
                attn_output[~valid_rows] = 0.0
        
        # Reshape: [B, H*W, C]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H * W, C)
        
        # Project output
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        grid_seq = grid_seq + self.dropout(attn_output)
        
        # FFN with residual
        grid_seq = grid_seq + self.ffn(grid_seq)
        
        # Reshape back to grid: [B, C, H, W]
        output = grid_seq.permute(0, 2, 1).view(B, C, H, W)
        
        return output


class RoomTopologyConditioner(nn.Module):
    """Project explicit room topology maps into a spatial feature bias."""

    def __init__(
        self,
        topology_channels: int,
        grid_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hid = int(hidden_dim or max(32, grid_dim // 2))
        self.proj = nn.Sequential(
            nn.Conv2d(int(topology_channels), hid, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hid, grid_dim, kernel_size=1),
        )

    def forward(self, room_topology_map: Tensor, *, target_hw: Tuple[int, int]) -> Tensor:
        topo = room_topology_map
        if topo.shape[-2:] != target_hw:
            topo = F.interpolate(topo, size=target_hw, mode="bilinear", align_corners=False)
        return self.proj(topo)


class SpatialGraphConditioner(nn.Module):
    """
    Combine explicit room topology maps with graph-to-grid cross-attention.

    This keeps the graph tokens and the spatial room constraints on one
    conditioning path instead of treating them as two unrelated systems.
    """

    def __init__(
        self,
        grid_dim: int,
        graph_dim: int,
        topology_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
    ):
        super().__init__()
        self.topology_conditioner = RoomTopologyConditioner(
            topology_channels=topology_channels,
            grid_dim=grid_dim,
        )
        self.graph_cross_attn = GraphToGridCrossAttention(
            grid_dim=grid_dim,
            graph_dim=graph_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_mode=attention_mode,
            hedgehog_feature_dim=hedgehog_feature_dim,
        )
        self.topology_gate = nn.Parameter(torch.zeros(1))
        self.graph_gate = nn.Parameter(torch.zeros(1))

    def set_attention_mode(self, mode: str) -> None:
        self.graph_cross_attn.set_attention_mode(mode)

    def forward(
        self,
        x: Tensor,
        *,
        graph_nodes: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        node_positions: Optional[Tensor] = None,
        node_tpe: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
        room_topology_map: Optional[Tensor] = None,
    ) -> Tensor:
        if room_topology_map is not None and int(room_topology_map.shape[0]) != int(x.shape[0]):
            raise ValueError(
                f"room_topology_map batch size {int(room_topology_map.shape[0])} does not match x batch size {int(x.shape[0])}."
            )
        if room_topology_map is not None:
            topo_bias = self.topology_conditioner(
                room_topology_map,
                target_hw=(int(x.shape[-2]), int(x.shape[-1])),
            )
            x = x + torch.tanh(self.topology_gate) * topo_bias

        if graph_nodes is not None:
            conditioned = self.graph_cross_attn(
                x,
                graph_nodes,
                edge_index=edge_index,
                node_positions=node_positions,
                node_tpe=node_tpe,
                node_mask=node_mask,
            )
            x = x + torch.tanh(self.graph_gate) * (conditioned - x)

        return x


# ============================================================================
# ENHANCED ATTENTION BLOCK FOR U-NET
# ============================================================================

class EnhancedAttentionBlock(nn.Module):
    """
    Enhanced attention block with per-position graph conditioning.
    
    Replaces the standard AttentionBlock in U-Net with:
    1. Self-attention on grid features
    2. Graph-to-grid cross-attention (per-position)
    3. Optional single-context cross-attention (backward compat)
    
    This allows the U-Net to receive rich spatial conditioning
    from the mission graph structure.
    """
    
    def __init__(
        self,
        grid_dim: int,
        graph_dim: int,
        context_dim: int = 256,  # For backward compat with single context
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.grid_dim = grid_dim
        self.graph_dim = graph_dim
        
        # Self-attention on grid
        self.self_attn = nn.MultiheadAttention(
            embed_dim=grid_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(grid_dim)
        
        # Graph-to-grid cross-attention (new)
        self.graph_cross_attn = GraphToGridCrossAttention(
            grid_dim=grid_dim,
            graph_dim=graph_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Context cross-attention (for backward compatibility)
        self.context_cross_attn = nn.MultiheadAttention(
            embed_dim=grid_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(grid_dim)
        self.context_proj = nn.Linear(context_dim, grid_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(grid_dim),
            nn.Linear(grid_dim, grid_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(grid_dim * 4, grid_dim),
        )
    
    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        graph_nodes: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        node_positions: Optional[Tensor] = None,
        node_tpe: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with flexible conditioning.
        
        Args:
            x: [B, C, H, W] grid features
            context: [B, context_dim] single context vector (optional)
            graph_nodes: [B, N, graph_dim] graph node features (optional)
            edge_index: [2, E] or [B, 2, E] graph connectivity (optional)
            node_positions: [B, N, 2] node positions
            node_tpe: [B, N, 8] topological encoding
            node_mask: [B, N] node validity mask
            
        Returns:
            [B, C, H, W] conditioned features
        """
        B, C, H, W = x.shape
        
        # Flatten to sequence for self-attention
        x_seq = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Self-attention
        x_normed = self.self_attn_norm(x_seq)
        attn_out, _ = self.self_attn(x_normed, x_normed, x_normed)
        x_seq = x_seq + attn_out
        
        # Reshape back to grid
        x = x_seq.permute(0, 2, 1).view(B, C, H, W)
        
        # Graph cross-attention (new per-position attention)
        if graph_nodes is not None:
            x = self.graph_cross_attn(
                x, graph_nodes, edge_index, node_positions, node_tpe, node_mask
            )
        
        # Context cross-attention (backward compat)
        elif context is not None:
            x_seq = x.view(B, C, -1).permute(0, 2, 1)
            
            # Project context to grid dim and expand
            ctx = self.context_proj(context)  # [B, C]
            ctx = ctx.unsqueeze(1)  # [B, 1, C]
            
            x_normed = self.context_norm(x_seq)
            attn_out, _ = self.context_cross_attn(x_normed, ctx, ctx)
            x_seq = x_seq + attn_out
            
            x = x_seq.permute(0, 2, 1).view(B, C, H, W)
        
        # FFN
        x_seq = x.view(B, C, -1).permute(0, 2, 1)
        x_seq = x_seq + self.ffn(x_seq)
        x = x_seq.permute(0, 2, 1).view(B, C, H, W)
        
        return x


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def upgrade_attention_block(
    old_block: nn.Module,
    graph_dim: int = 256,
) -> EnhancedAttentionBlock:
    """
    Upgrade an existing AttentionBlock to EnhancedAttentionBlock.
    
    Args:
        old_block: Existing attention block
        graph_dim: Graph node dimension
        
    Returns:
        EnhancedAttentionBlock with weights transferred where possible
    """
    # Get dimensions from old block
    if hasattr(old_block, 'self_attn'):
        grid_dim = old_block.self_attn.embed_dim
        num_heads = old_block.self_attn.num_heads
    else:
        grid_dim = 128  # default
        num_heads = 8
    
    # Create new block
    new_block = EnhancedAttentionBlock(
        grid_dim=grid_dim,
        graph_dim=graph_dim,
        num_heads=num_heads,
    )
    
    # Transfer weights where shapes match
    old_state = old_block.state_dict()
    new_state = new_block.state_dict()
    
    for key in old_state:
        if key in new_state and old_state[key].shape == new_state[key].shape:
            new_state[key] = old_state[key]
    
    new_block.load_state_dict(new_state)
    
    return new_block


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Graph-to-Grid Cross-Attention...")
    
    # Test parameters
    B, C, H, W = 2, 128, 16, 11
    N_nodes = 10
    graph_dim = 256
    
    # Create module
    cross_attn = GraphToGridCrossAttention(
        grid_dim=C,
        graph_dim=graph_dim,
        num_heads=8,
    )
    
    # Create test inputs
    grid_features = torch.randn(B, C, H, W)
    graph_nodes = torch.randn(B, N_nodes, graph_dim)
    node_positions = torch.randint(0, 10, (B, N_nodes, 2)).float()
    node_tpe = torch.randn(B, N_nodes, 8)
    
    # Forward pass
    output = cross_attn(
        grid_features,
        graph_nodes,
        node_positions=node_positions,
        node_tpe=node_tpe,
    )
    
    print(f"Input shape: {grid_features.shape}")
    print(f"Graph nodes: {graph_nodes.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test gradient flow
    loss = output.mean()
    loss.backward()
    
    print("Gradient test passed!")
    
    # Test enhanced attention block
    print("\nTesting EnhancedAttentionBlock...")
    
    enhanced = EnhancedAttentionBlock(
        grid_dim=C,
        graph_dim=graph_dim,
        context_dim=256,
    )
    
    # With graph nodes
    out1 = enhanced(
        grid_features, 
        graph_nodes=graph_nodes,
        node_positions=node_positions,
    )
    print(f"With graph nodes: {out1.shape}")
    
    # With context vector (backward compat)
    context = torch.randn(B, 256)
    out2 = enhanced(grid_features, context=context)
    print(f"With context vector: {out2.shape}")
    
    print("\nAll tests passed!")
