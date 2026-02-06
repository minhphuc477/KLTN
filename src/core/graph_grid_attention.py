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
    
    output = cross_attn(grid_features, graph_nodes, node_positions)
"""

import math
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


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
        """Create 2D positional encoding."""
        half_dim = dim // 2
        
        # Create frequency bands
        omega = torch.arange(half_dim // 2, dtype=torch.float32)
        omega = 1.0 / (temperature ** (omega / (half_dim // 2)))
        
        # Row and column positions
        rows = torch.arange(max_h, dtype=torch.float32)
        cols = torch.arange(max_w, dtype=torch.float32)
        
        # Compute encodings
        row_pe = rows.unsqueeze(1) * omega.unsqueeze(0)  # [H, D/4]
        col_pe = cols.unsqueeze(1) * omega.unsqueeze(0)  # [W, D/4]
        
        # Sin and cos
        row_sin = torch.sin(row_pe)
        row_cos = torch.cos(row_pe)
        col_sin = torch.sin(col_pe)
        col_cos = torch.cos(col_pe)
        
        # Combine into [H, W, D] tensor
        pe = torch.zeros(max_h, max_w, dim)
        half = half_dim // 2
        
        for h in range(max_h):
            for w in range(max_w):
                pe[h, w, :half] = row_sin[h]
                pe[h, w, half:half*2] = row_cos[h]
                pe[h, w, half*2:half*3] = col_sin[w]
                pe[h, w, half*3:] = col_cos[w]
        
        return pe
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to grid features.
        
        Args:
            x: [B, C, H, W] grid features
            
        Returns:
            [B, C, H, W] with positional encoding added
        """
        B, C, H, W = x.shape
        
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
    ):
        super().__init__()
        
        self.grid_dim = grid_dim
        self.graph_dim = graph_dim
        self.num_heads = num_heads
        self.head_dim = grid_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Grid position encoding
        self.grid_pe = SinusoidalPositionEncoding2D(grid_dim)
        
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
    
    def forward(
        self,
        grid_features: Tensor,
        graph_nodes: Tensor,
        node_positions: Optional[Tensor] = None,
        node_tpe: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply graph-to-grid cross-attention.
        
        Args:
            grid_features: [B, C, H, W] grid features from U-Net
            graph_nodes: [B, N, graph_dim] graph node features
            node_positions: [B, N, 2] optional node positions
            node_tpe: [B, N, 8] topological positional encoding
            node_mask: [B, N] optional mask (1 = valid, 0 = padding)
            
        Returns:
            [B, C, H, W] conditioned grid features
        """
        B, C, H, W = grid_features.shape
        N = graph_nodes.shape[1]
        
        # Add positional encoding to grid
        grid_with_pe = self.grid_pe(grid_features)
        
        # Flatten grid to sequence: [B, H*W, C]
        grid_seq = grid_with_pe.view(B, C, -1).permute(0, 2, 1)
        
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
        
        # Attention scores: [B, heads, H*W, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if node_mask is not None:
            # Expand mask: [B, N] -> [B, 1, 1, N]
            mask = node_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention: [B, heads, H*W, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
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
                x, graph_nodes, node_positions, node_tpe, node_mask
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
    output = cross_attn(grid_features, graph_nodes, node_positions, node_tpe)
    
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
