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
import threading
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.attention_kernels import HedgehogFeatureMap, hedgehog_linear_attention

logger = logging.getLogger(__name__)
HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


# ============================================================================
# LIGHTWEIGHT GRAPH CONV
# ============================================================================

class LightweightGCNLayer(nn.Module):
    """Simple normalized A_hat X W graph convolution without external deps."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, node_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, N, D]
            edge_index: [2, E] or [B, 2, E]
            node_mask: Optional [B, N] mask where True marks real graph nodes.
        """
        b, n, _d = x.shape
        if edge_index.dim() not in {2, 3}:
            raise ValueError(
                f"LightweightGCNLayer edge_index must have shape [2, E] or [B, 2, E], got {tuple(edge_index.shape)}."
            )
        if node_mask is None:
            valid_nodes_all = torch.ones(b, n, device=x.device, dtype=torch.bool)
        else:
            valid_nodes_all = node_mask.to(device=x.device, dtype=torch.bool)
            if valid_nodes_all.dim() == 1:
                valid_nodes_all = valid_nodes_all.unsqueeze(0)
            if int(valid_nodes_all.shape[0]) == 1 and b > 1:
                valid_nodes_all = valid_nodes_all.expand(b, -1)
            if tuple(valid_nodes_all.shape) != (b, n):
                raise ValueError(
                    f"LightweightGCNLayer node_mask must have shape [B, N] = ({b}, {n}), "
                    f"got {tuple(valid_nodes_all.shape)}."
                )
        if n == 0:
            return self.linear(x)
        if edge_index.dim() == 2:
            if int(edge_index.shape[0]) != 2:
                raise ValueError(
                    f"LightweightGCNLayer edge_index first dimension must be 2, got {tuple(edge_index.shape)}."
                )
            batched_edges = edge_index.to(device=x.device).unsqueeze(0).expand(b, -1, -1)
        else:
            if int(edge_index.shape[1]) != 2:
                raise ValueError(
                    f"LightweightGCNLayer edge_index second dimension must be 2, got {tuple(edge_index.shape)}."
                )
            if int(edge_index.shape[0]) == b:
                batched_edges = edge_index.to(device=x.device)
            elif int(edge_index.shape[0]) == 1:
                batched_edges = edge_index.to(device=x.device).expand(b, -1, -1)
            else:
                raise ValueError(
                    f"LightweightGCNLayer edge_index batch size {int(edge_index.shape[0])} "
                    f"does not match graph batch size {b}."
                )

        z = self.linear(x) * valid_nodes_all.unsqueeze(-1).to(dtype=x.dtype)
        src_raw = batched_edges[:, 0, :].long()
        dst_raw = batched_edges[:, 1, :].long()
        in_range = (src_raw >= 0) & (src_raw < n) & (dst_raw >= 0) & (dst_raw < n)
        safe_src = src_raw.clamp(0, max(0, n - 1))
        safe_dst = dst_raw.clamp(0, max(0, n - 1))
        valid_edges = (
            in_range
            & valid_nodes_all.gather(1, safe_src)
            & valid_nodes_all.gather(1, safe_dst)
        )

        offsets = torch.arange(b, device=x.device, dtype=torch.long).unsqueeze(1) * n
        flat_src = (safe_src + offsets)[valid_edges]
        flat_dst = (safe_dst + offsets)[valid_edges]
        self_nodes = torch.nonzero(valid_nodes_all.reshape(-1), as_tuple=False).flatten()
        src_all = torch.cat([flat_src, flat_dst, self_nodes], dim=0)
        dst_all = torch.cat([flat_dst, flat_src, self_nodes], dim=0)

        z_flat = z.reshape(b * n, -1)
        if src_all.numel() == 0:
            return torch.zeros_like(z)

        deg = torch.zeros(b * n, device=x.device, dtype=x.dtype)
        deg.index_add_(
            0,
            src_all,
            torch.ones(src_all.shape[0], device=x.device, dtype=x.dtype),
        )
        norm = (
            deg[src_all].clamp(min=1.0).pow(-0.5)
            * deg[dst_all].clamp(min=1.0).pow(-0.5)
        )
        messages = z_flat[dst_all] * norm.unsqueeze(-1)
        aggregated = torch.zeros_like(z_flat)
        aggregated.index_add_(0, src_all, messages)
        return aggregated.reshape(b, n, -1)


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
        self.temperature = float(temperature)
        
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
        
        # Compute encodings -- [H, half] and [W, half]
        row_pe = rows.unsqueeze(1) * omega.unsqueeze(0)
        col_pe = cols.unsqueeze(1) * omega.unsqueeze(0)
        
        row_sin = torch.sin(row_pe)  # [H, half]
        row_cos = torch.cos(row_pe)  # [H, half]
        col_sin = torch.sin(col_pe)  # [W, half]
        col_cos = torch.cos(col_pe)  # [W, half]
        
        # Vectorized broadcast into [H, W, D] -- no Python loops
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
        if int(H) > int(self.pe.shape[0]) or int(W) > int(self.pe.shape[1]):
            new_h = max(int(H), int(self.pe.shape[0]))
            new_w = max(int(W), int(self.pe.shape[1]))
            self.max_h = new_h
            self.max_w = new_w
            self.pe = self._create_encoding(self.dim, new_h, new_w, self.temperature).to(
                device=self.pe.device,
                dtype=self.pe.dtype,
            )
        
        # Get relevant portion of encoding
        pe = self.pe[:H, :W, :C]  # [H, W, C]
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        return x + pe.to(device=x.device, dtype=x.dtype)


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
        hidden_dim = max(4, dim // 2)
        
        # Learnable position embedding
        self.feature_norm = nn.LayerNorm(dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        
        # Learnable topological encoding
        self.topo_embed = nn.Sequential(
            nn.Linear(8, hidden_dim),  # 8-dim TPE
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.distance_embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        # Degree-style structural features are lightweight but permutation aware.
        self.struct_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.pos_gate = nn.Parameter(torch.tensor(1.0))
        self.topo_gate = nn.Parameter(torch.tensor(1.0))
        self.distance_gate = nn.Parameter(torch.tensor(1.0))
        self.struct_gate = nn.Parameter(torch.tensor(1.0))
        self.output_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        node_features: Tensor,
        node_positions: Optional[Tensor] = None,
        tpe: Optional[Tensor] = None,
        current_node_distance: Optional[Tensor] = None,
        structure_features: Optional[Tensor] = None,
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
        output = self.feature_norm(node_features)
        target_dtype = output.dtype
        
        if node_positions is not None:
            pos_enc = self.pos_embed(node_positions.to(dtype=target_dtype))
            output = output + torch.sigmoid(self.pos_gate) * pos_enc
        
        if tpe is not None:
            topo_enc = self.topo_embed(tpe.to(dtype=target_dtype))
            output = output + torch.sigmoid(self.topo_gate) * topo_enc

        if current_node_distance is not None:
            distance_enc = self.distance_embed(current_node_distance.to(dtype=target_dtype))
            output = output + torch.sigmoid(self.distance_gate) * distance_enc

        if structure_features is not None:
            struct_enc = self.struct_embed(structure_features.to(dtype=target_dtype))
            output = output + torch.sigmoid(self.struct_gate) * struct_enc
        
        return self.output_norm(output)


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
        auto_linear_attention_nodes: int = 128,
        use_edge_semantics: bool = False,
        edge_type_vocab_size: int = 16,
        allow_legacy_argument_swap: bool = False,
    ):
        super().__init__()
        if int(grid_dim) <= 0 or int(graph_dim) <= 0:
            raise ValueError(
                f"GraphToGridCrossAttention expects positive dims, got grid_dim={grid_dim}, graph_dim={graph_dim}."
            )
        if int(num_heads) <= 0:
            raise ValueError(f"GraphToGridCrossAttention num_heads must be > 0, got {num_heads}.")
        if int(grid_dim) % int(num_heads) != 0:
            raise ValueError(
                "GraphToGridCrossAttention requires grid_dim to be divisible by num_heads; "
                f"got grid_dim={grid_dim}, num_heads={num_heads}."
            )
        
        self.grid_dim = grid_dim
        self.graph_dim = graph_dim
        self.num_heads = num_heads
        self.head_dim = grid_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_mode = "softmax"
        self.auto_linear_attention_nodes = int(max(0, int(auto_linear_attention_nodes)))
        self.use_edge_semantics = bool(use_edge_semantics)
        self.edge_type_vocab_size = int(max(1, int(edge_type_vocab_size)))
        self.allow_legacy_argument_swap = bool(allow_legacy_argument_swap)
        self._large_graph_fallback_warning_emitted = False
        self.capture_attention_maps = False
        self.last_attention_weights: Optional[Tensor] = None
        self.last_attention_weights_for_loss: Optional[Tensor] = None
        self.last_attention_grid_shape: Optional[Tuple[int, int]] = None
        self._attention_capture_lock = threading.RLock()

        # Grid position encoding
        self.grid_pe = SinusoidalPositionEncoding2D(grid_dim)

        # Optional graph convolution preprocessing before attention.
        self.graph_gcn = LightweightGCNLayer(graph_dim, graph_dim)
        
        # Graph node position encoding
        self.graph_pe = GraphNodePositionEncoding(graph_dim)
        self.node_struct_bias = nn.Sequential(
            nn.Linear(2, max(4, num_heads)),
            nn.SiLU(),
            nn.Linear(max(4, num_heads), num_heads),
        )
        self.node_distance_bias = nn.Sequential(
            nn.Linear(4, max(8, num_heads)),
            nn.SiLU(),
            nn.Linear(max(8, num_heads), num_heads),
        )
        self.edge_type_node_bias = (
            nn.Embedding(self.edge_type_vocab_size, num_heads)
            if self.use_edge_semantics
            else None
        )
        
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

    @staticmethod
    def _edge_attr_type_ids(
        edge_attr: Optional[Tensor],
        *,
        batch_size: int,
        num_edges: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if not isinstance(edge_attr, torch.Tensor) or int(num_edges) <= 0:
            return None
        attr = edge_attr.to(device=device)
        if attr.dim() == 1:
            ids = attr.long().view(1, -1).expand(batch_size, -1)
        elif attr.dim() == 2:
            if int(attr.shape[0]) == batch_size and int(attr.shape[1]) == num_edges:
                ids = attr.long()
            elif int(attr.shape[0]) == 1 and int(attr.shape[1]) == num_edges:
                ids = attr.long().expand(batch_size, -1)
            elif int(attr.shape[0]) == num_edges:
                ids = attr.argmax(dim=-1).long().view(1, -1).expand(batch_size, -1)
            else:
                raise ValueError(
                    f"GraphToGridCrossAttention edge_attr must align to E={num_edges}; got {tuple(attr.shape)}."
                )
        elif attr.dim() == 3:
            if int(attr.shape[0]) == 1 and batch_size > 1:
                attr = attr.expand(batch_size, -1, -1)
            if int(attr.shape[0]) != batch_size or int(attr.shape[1]) != num_edges:
                raise ValueError(
                    f"GraphToGridCrossAttention edge_attr must have shape [B,E,D] with B={batch_size}, E={num_edges}; "
                    f"got {tuple(attr.shape)}."
                )
            ids = attr.argmax(dim=-1).long()
        else:
            raise ValueError(
                f"GraphToGridCrossAttention edge_attr must have shape [E], [B,E], [E,D], or [B,E,D], got {tuple(attr.shape)}."
            )
        return ids[:, :num_edges]

    def _edge_semantic_node_bias(
        self,
        edge_index: Optional[Tensor],
        edge_attr: Optional[Tensor],
        *,
        batch_size: int,
        num_nodes: int,
        node_mask: Optional[Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if not self.use_edge_semantics or self.edge_type_node_bias is None or edge_index is None or num_nodes <= 0:
            return None
        if edge_index.dim() not in {2, 3}:
            raise ValueError(
                f"GraphToGridCrossAttention edge_index must have shape [2, E] or [B, 2, E], got {tuple(edge_index.shape)}."
            )
        valid_nodes_all = (
            torch.ones(batch_size, num_nodes, device=device, dtype=torch.bool)
            if node_mask is None
            else node_mask.to(device=device, dtype=torch.bool)
        )
        if valid_nodes_all.dim() == 1:
            valid_nodes_all = valid_nodes_all.unsqueeze(0)
        if int(valid_nodes_all.shape[0]) == 1 and batch_size > 1:
            valid_nodes_all = valid_nodes_all.expand(batch_size, -1)
        if tuple(valid_nodes_all.shape) != (batch_size, num_nodes):
            raise ValueError(
                f"GraphToGridCrossAttention node_mask must have shape [B, N] = ({batch_size}, {num_nodes}), "
                f"got {tuple(valid_nodes_all.shape)}."
            )

        if edge_index.dim() == 2:
            ei = edge_index.to(device=device, dtype=torch.long)
            if int(ei.shape[0]) != 2:
                raise ValueError(f"GraphToGridCrossAttention edge_index first dimension must be 2, got {tuple(ei.shape)}.")
            edge_count = int(ei.shape[1])
            edge_ids = self._edge_attr_type_ids(edge_attr, batch_size=batch_size, num_edges=edge_count, device=device)
            if edge_ids is None:
                return None
            src = ei[0].view(1, -1).expand(batch_size, -1)
            dst = ei[1].view(1, -1).expand(batch_size, -1)
        else:
            ei = edge_index.to(device=device, dtype=torch.long)
            if int(ei.shape[1]) != 2:
                raise ValueError(f"GraphToGridCrossAttention edge_index must have shape [B,2,E], got {tuple(ei.shape)}.")
            if int(ei.shape[0]) == 1 and batch_size > 1:
                ei = ei.expand(batch_size, -1, -1)
            if int(ei.shape[0]) != batch_size:
                raise ValueError(
                    f"GraphToGridCrossAttention edge_index batch size {int(ei.shape[0])} does not match graph batch size {batch_size}."
                )
            edge_count = int(ei.shape[2])
            edge_ids = self._edge_attr_type_ids(edge_attr, batch_size=batch_size, num_edges=edge_count, device=device)
            if edge_ids is None:
                return None
            src = ei[:, 0]
            dst = ei[:, 1]

        valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
        safe_src = src.clamp(0, max(0, num_nodes - 1))
        safe_dst = dst.clamp(0, max(0, num_nodes - 1))
        valid = valid & valid_nodes_all.gather(1, safe_src) & valid_nodes_all.gather(1, safe_dst)
        edge_ids = edge_ids.clamp(0, self.edge_type_vocab_size - 1)
        edge_bias = self.edge_type_node_bias(edge_ids).to(dtype=dtype)
        flat_offsets = (
            torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(1)
            * num_nodes
        )
        flat_dst = (safe_dst + flat_offsets)[valid]
        flat_values = edge_bias[valid]
        node_bias_flat = torch.zeros(
            batch_size * num_nodes,
            self.num_heads,
            device=device,
            dtype=dtype,
        )
        counts_flat = torch.zeros(
            batch_size * num_nodes,
            1,
            device=device,
            dtype=dtype,
        )
        if flat_dst.numel() > 0:
            node_bias_flat.index_add_(0, flat_dst, flat_values)
            counts_flat.index_add_(
                0,
                flat_dst,
                torch.ones(flat_dst.shape[0], 1, device=device, dtype=dtype),
            )
        node_bias = node_bias_flat.reshape(batch_size, num_nodes, self.num_heads)
        counts = counts_flat.reshape(batch_size, num_nodes, 1)
        node_bias = node_bias / counts.clamp_min(1.0)
        node_bias = node_bias * valid_nodes_all[:, :, None].to(dtype=dtype)
        return node_bias.permute(0, 2, 1).unsqueeze(2)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_attention_capture_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._attention_capture_lock = threading.RLock()

    def set_attention_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"softmax", "linear_hedgehog"}:
            raise ValueError(
                f"Invalid attention_mode={mode!r}. Expected 'softmax' or 'linear_hedgehog'."
            )
        self.attention_mode = normalized

    def set_attention_capture(self, enabled: bool = True) -> None:
        """Enable or disable storage of the latest softmax attention map."""
        with self._attention_capture_lock:
            self.capture_attention_maps = bool(enabled)
            if not self.capture_attention_maps:
                self.last_attention_weights = None
                self.last_attention_weights_for_loss = None
                self.last_attention_grid_shape = None

    def get_last_attention_map(self, reduce_heads: str = "mean") -> Optional[Tensor]:
        """
        Return the latest captured attention as [B, H, W, N] or [B, heads, H, W, N].

        Attention maps are populated only when set_attention_capture(True) is active
        and the module runs in softmax mode.
        """
        with self._attention_capture_lock:
            if self.last_attention_weights is None or self.last_attention_grid_shape is None:
                return None
            weights = self.last_attention_weights
            height, width = self.last_attention_grid_shape
        mode = str(reduce_heads).strip().lower()
        if mode in {"none", "heads", "per_head", "per-head"}:
            return weights.reshape(
                weights.shape[0],
                weights.shape[1],
                int(height),
                int(width),
                weights.shape[-1],
            )
        if mode == "mean":
            return weights.mean(dim=1).reshape(weights.shape[0], int(height), int(width), weights.shape[-1])
        if mode == "max":
            return weights.max(dim=1).values.reshape(weights.shape[0], int(height), int(width), weights.shape[-1])
        raise ValueError("reduce_heads must be 'mean', 'max', or 'none'.")

    def spatial_alignment_loss(
        self,
        node_indices: Tensor,
        target_positions: Tensor,
        valid_mask: Optional[Tensor] = None,
        *,
        reduce_heads: str = "mean",
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Penalize graph-to-grid attention that misses known node locations.

        Args:
            node_indices: [B, M] graph-node ids to supervise.
            target_positions: [B, M, 2] row/col grid positions at the latest
                attention resolution. Floating inputs are rounded.
            valid_mask: optional [B, M] mask for padded supervision targets.
            reduce_heads: mean or max head reduction before gathering.
        """
        with self._attention_capture_lock:
            weights = self.last_attention_weights_for_loss
            grid_shape = self.last_attention_grid_shape
            if weights is None or grid_shape is None:
                raise RuntimeError(
                    "spatial_alignment_loss requires set_attention_capture(True) and a softmax forward pass first."
                )
        if node_indices.dim() != 2:
            raise ValueError(f"node_indices must have shape [B, M], got {tuple(node_indices.shape)}.")
        if target_positions.dim() != 3 or int(target_positions.shape[-1]) != 2:
            raise ValueError(
                f"target_positions must have shape [B, M, 2], got {tuple(target_positions.shape)}."
            )
        if int(node_indices.shape[0]) != int(weights.shape[0]) or int(target_positions.shape[0]) != int(weights.shape[0]):
            raise ValueError("spatial alignment batch size must match the last captured attention batch.")
        if int(node_indices.shape[1]) != int(target_positions.shape[1]):
            raise ValueError("node_indices and target_positions must have the same target count.")

        height, width = grid_shape
        mode = str(reduce_heads).strip().lower()
        if mode == "mean":
            attn = weights.mean(dim=1).reshape(weights.shape[0], int(height), int(width), weights.shape[-1])
        elif mode == "max":
            attn = weights.max(dim=1).values.reshape(weights.shape[0], int(height), int(width), weights.shape[-1])
        else:
            raise ValueError("reduce_heads must be 'mean' or 'max' for spatial_alignment_loss.")

        bsz, target_count = node_indices.shape
        rows = target_positions[..., 0].round().long().clamp(0, int(height) - 1)
        cols = target_positions[..., 1].round().long().clamp(0, int(width) - 1)
        nodes = node_indices.long().clamp(0, int(attn.shape[-1]) - 1)
        batch_idx = torch.arange(bsz, device=attn.device).view(bsz, 1).expand(bsz, target_count)
        gathered = attn[batch_idx, rows.to(attn.device), cols.to(attn.device), nodes.to(attn.device)]

        if valid_mask is None:
            valid = torch.ones_like(gathered, dtype=torch.bool)
        else:
            if valid_mask.shape != node_indices.shape:
                raise ValueError(f"valid_mask must have shape {tuple(node_indices.shape)}, got {tuple(valid_mask.shape)}.")
            valid = valid_mask.to(device=attn.device, dtype=torch.bool)
        valid = valid & torch.isfinite(gathered)
        if not bool(valid.any()):
            return gathered.sum() * 0.0
        min_prob = max(float(eps), 1.0e-4)
        safe_prob = gathered.clamp(min=min_prob, max=1.0 - min_prob)
        logits = torch.logit(safe_prob)
        targets = torch.ones_like(logits)
        return F.binary_cross_entropy_with_logits(logits[valid], targets[valid])

    def _compute_degree_features(
        self,
        edge_index: Optional[Tensor],
        *,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute normalized in/out-degree features used as lightweight structural encodings."""
        degree = torch.zeros(batch_size, num_nodes, 2, device=device, dtype=dtype)
        if num_nodes == 0 or edge_index is None:
            return degree

        if edge_index.dim() not in {2, 3}:
            raise ValueError(
                f"GraphToGridCrossAttention edge_index must have shape [2, E] or [B, 2, E], got {tuple(edge_index.shape)}."
            )
        valid_nodes_all = None
        if node_mask is not None:
            valid_nodes_all = node_mask.to(device=device, dtype=torch.bool)
            if valid_nodes_all.dim() == 1:
                valid_nodes_all = valid_nodes_all.unsqueeze(0)
            if int(valid_nodes_all.shape[0]) == 1 and batch_size > 1:
                valid_nodes_all = valid_nodes_all.expand(batch_size, -1)
            if tuple(valid_nodes_all.shape) != (batch_size, num_nodes):
                raise ValueError(
                    f"GraphToGridCrossAttention node_mask must have shape [B, N] = ({batch_size}, {num_nodes}), "
                    f"got {tuple(valid_nodes_all.shape)}."
                )

        ones_dtype = degree.dtype
        degree_norm = float(max(1, num_nodes - 1))
        if edge_index.dim() == 2:
            if int(edge_index.shape[0]) != 2:
                raise ValueError(
                    f"GraphToGridCrossAttention edge_index first dimension must be 2, got {tuple(edge_index.shape)}."
                )
            batched_edges = edge_index.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if int(edge_index.shape[1]) != 2:
                raise ValueError(
                    f"GraphToGridCrossAttention edge_index second dimension must be 2, got {tuple(edge_index.shape)}."
                )
            if int(edge_index.shape[0]) == batch_size:
                batched_edges = edge_index.to(device=device)
            elif int(edge_index.shape[0]) == 1:
                batched_edges = edge_index.to(device=device).expand(batch_size, -1, -1)
            else:
                raise ValueError(
                    f"GraphToGridCrossAttention edge_index batch size {int(edge_index.shape[0])} "
                    f"does not match graph batch size {batch_size}."
                )

        src_raw = batched_edges[:, 0, :].long()
        dst_raw = batched_edges[:, 1, :].long()
        valid = (
            (src_raw >= 0)
            & (src_raw < num_nodes)
            & (dst_raw >= 0)
            & (dst_raw < num_nodes)
        )
        safe_src = src_raw.clamp(0, max(0, num_nodes - 1))
        safe_dst = dst_raw.clamp(0, max(0, num_nodes - 1))
        if valid_nodes_all is not None:
            valid = (
                valid
                & valid_nodes_all.gather(1, safe_src)
                & valid_nodes_all.gather(1, safe_dst)
            )
        offsets = (
            torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(1)
            * num_nodes
        )
        src = (safe_src + offsets)[valid]
        dst = (safe_dst + offsets)[valid]
        if src.numel() > 0:
            flat_degree = degree.reshape(batch_size * num_nodes, 2)
            ones = torch.ones(src.shape[0], device=device, dtype=ones_dtype)
            flat_degree[:, 0].index_add_(0, dst, ones)
            flat_degree[:, 1].index_add_(0, src, ones)

        degree = degree / degree_norm
        if valid_nodes_all is not None:
            degree = degree * valid_nodes_all.unsqueeze(-1).to(dtype=dtype)
        return degree
    
    def forward(
        self,
        grid_features: Tensor,
        graph_nodes: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        node_positions: Optional[Tensor] = None,
        node_tpe: Optional[Tensor] = None,
        current_node_distance: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply graph-to-grid cross-attention.
        
        Args:
            grid_features: [B, C, H, W] grid features from U-Net
            graph_nodes: [B, N, graph_dim] graph node features
            edge_index: [2, E] or [B, 2, E] graph connectivity for GCN prepass
            edge_attr: optional edge labels/features aligned with edge_index
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
        legacy_argument_swap_with_edge_attr_slot = (
            isinstance(edge_index, torch.Tensor)
            and edge_index.dim() >= 2
            and torch.is_floating_point(edge_index)
            and int(edge_index.shape[-1]) == 2
            and isinstance(edge_attr, torch.Tensor)
            and edge_attr.dim() >= 2
            and int(edge_attr.shape[-1]) == 8
            and node_positions is None
            and node_tpe is None
        )
        # Backward-compatibility for older positional calls:
        # module(x, graph_nodes, node_positions, node_tpe)
        if (legacy_argument_swap or legacy_argument_swap_with_edge_attr_slot) and not self.allow_legacy_argument_swap:
            raise ValueError(
                "GraphToGridCrossAttention received legacy positional arguments that look like "
                "(node_positions, node_tpe). Pass node_positions=... and node_tpe=... as keywords, "
                "or construct the module with allow_legacy_argument_swap=True for explicit compatibility."
            )
        if legacy_argument_swap:
            node_tpe = node_positions
            node_positions = edge_index
            edge_index = None
        elif legacy_argument_swap_with_edge_attr_slot:
            node_tpe = edge_attr
            node_positions = edge_index
            edge_attr = None
            edge_index = None

        B, C, H, W = grid_features.shape
        with self._attention_capture_lock:
            self.last_attention_weights = None
            self.last_attention_weights_for_loss = None
            self.last_attention_grid_shape = None
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
        if N == 0:
            logger.debug("GraphToGridCrossAttention received zero graph nodes; returning grid features unchanged.")
            return grid_features

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
                logger.warning(
                    "GraphToGridCrossAttention %s trailing dimension mismatch: got %d, expected %d. Applying pad/truncate.",
                    name,
                    int(tensor.shape[-1]),
                    int(expected_last),
                )
                if int(tensor.shape[-1]) > int(expected_last):
                    tensor = tensor[..., :expected_last]
                else:
                    pad = torch.zeros(
                        *tensor.shape[:-1],
                        int(expected_last) - int(tensor.shape[-1]),
                        device=tensor.device,
                        dtype=tensor.dtype,
                    )
                    tensor = torch.cat([tensor, pad], dim=-1)
            return tensor

        node_positions = _normalize_batched("node_positions", node_positions, expected_last=2)
        node_tpe = _normalize_batched("node_tpe", node_tpe, expected_last=8)
        current_node_distance = _normalize_batched(
            "current_node_distance",
            current_node_distance,
            expected_last=4,
        )

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
        grid_seq = grid_with_pe.reshape(B, C, -1).permute(0, 2, 1)

        degree_features = self._compute_degree_features(
            edge_index,
            batch_size=B,
            num_nodes=N,
            device=grid_features.device,
            dtype=graph_nodes.dtype,
            node_mask=node_mask,
        )
        
        # Optional GCN preprocessing preserves node-topology dependencies.
        if edge_index is not None:
            graph_nodes = self.graph_gcn(graph_nodes, edge_index, node_mask=node_mask)

        # Add position encoding to graph nodes
        graph_with_pe = self.graph_pe(
            graph_nodes,
            node_positions,
            node_tpe,
            current_node_distance=current_node_distance,
            structure_features=degree_features,
        )
        
        # Normalize
        grid_normed = self.norm_grid(grid_seq)
        graph_normed = self.norm_graph(graph_with_pe)
        
        # Compute Q, K, V
        Q = self.q_proj(grid_normed)  # [B, H*W, C]
        K = self.k_proj(graph_normed)  # [B, N, C]
        V = self.v_proj(graph_normed)  # [B, N, C]
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        attention_mode = self.attention_mode
        if (
            attention_mode == "softmax"
            and self.auto_linear_attention_nodes > 0
            and int(N) > self.auto_linear_attention_nodes
        ):
            if not self._large_graph_fallback_warning_emitted:
                logger.warning(
                    "GraphToGridCrossAttention received %d nodes, exceeding softmax threshold %d; switching to linear attention.",
                    int(N),
                    int(self.auto_linear_attention_nodes),
                )
                self._large_graph_fallback_warning_emitted = True
            attention_mode = "linear_hedgehog"

        if attention_mode == "linear_hedgehog":
            if self.capture_attention_maps:
                logger.debug(
                    "Attention capture requested, but linear_hedgehog mode does not expose node-wise softmax maps."
                )
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
            struct_bias = self.node_struct_bias(degree_features).permute(0, 2, 1).unsqueeze(2)
            attn_bias = struct_bias.to(dtype=Q.dtype)
            if current_node_distance is not None:
                distance_bias = self.node_distance_bias(current_node_distance).permute(0, 2, 1).unsqueeze(2)
                attn_bias = attn_bias + distance_bias.to(dtype=Q.dtype)
            edge_bias = self._edge_semantic_node_bias(
                edge_index,
                edge_attr,
                batch_size=B,
                num_nodes=N,
                node_mask=node_mask,
                device=grid_features.device,
                dtype=Q.dtype,
            )
            if edge_bias is not None:
                attn_bias = attn_bias + edge_bias

            if node_mask is not None:
                attn_bias = attn_bias.masked_fill(node_mask[:, None, None, :] == 0, -1.0e4)

            if HAS_SDPA and not self.capture_attention_maps:
                attn_output = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=attn_bias,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
            else:
                # Attention scores: [B, heads, H*W, N]
                attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
                attn_scores = attn_scores + attn_bias
                attn_weights = F.softmax(attn_scores.float(), dim=-1).to(dtype=attn_scores.dtype)
                if self.capture_attention_maps:
                    loss_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
                    captured = loss_weights.detach()
                    if valid_rows is not None and not torch.all(valid_rows):
                        loss_weights = loss_weights.clone()
                        loss_weights[~valid_rows] = 0.0
                        captured = loss_weights.detach()
                    with self._attention_capture_lock:
                        self.last_attention_weights_for_loss = loss_weights
                        self.last_attention_weights = captured.cpu()
                        self.last_attention_grid_shape = (int(H), int(W))
                attn_weights = self.dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, V)
            if valid_rows is not None and not torch.all(valid_rows):
                attn_output = attn_output.clone()
                attn_output[~valid_rows] = 0.0
        
        # Reshape: [B, H*W, C]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, H * W, C)
        
        # Project output
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        grid_seq = grid_seq + self.dropout(attn_output)
        
        # FFN with residual
        grid_seq = grid_seq + self.ffn(grid_seq)
        
        # Reshape back to grid: [B, C, H, W]
        output = grid_seq.permute(0, 2, 1).reshape(B, C, H, W)
        if valid_rows is not None and not torch.all(valid_rows):
            output = torch.where(
                valid_rows.to(device=output.device).view(B, 1, 1, 1),
                output,
                grid_features,
            )
        
        return output


class RoomTopologyConditioner(nn.Module):
    """Project explicit room topology maps into either an additive bias or SPADE-style modulation."""

    def __init__(
        self,
        topology_channels: int,
        grid_dim: int,
        hidden_dim: Optional[int] = None,
        conditioning_mode: str = "additive",
    ):
        super().__init__()
        mode = str(conditioning_mode).strip().lower()
        if mode not in {"additive", "spade"}:
            raise ValueError(
                f"Invalid conditioning_mode={conditioning_mode!r}. Expected 'additive' or 'spade'."
            )
        self.conditioning_mode = mode
        self.grid_dim = int(grid_dim)
        hid = int(hidden_dim or max(32, grid_dim // 2))

        if self.conditioning_mode == "additive":
            self.proj = nn.Sequential(
                nn.Conv2d(int(topology_channels), hid, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hid, grid_dim, kernel_size=1),
            )
        else:
            self.norm = nn.GroupNorm(self._num_groups(self.grid_dim), self.grid_dim, affine=False)
            self.to_scale_shift = nn.Sequential(
                nn.Conv2d(int(topology_channels), hid, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hid, grid_dim * 2, kernel_size=1),
            )

    @staticmethod
    def _num_groups(channels: int, max_groups: int = 32) -> int:
        for groups in range(min(max_groups, int(channels)), 0, -1):
            if int(channels) % groups == 0:
                return groups
        return 1

    def forward(
        self,
        room_topology_map: Tensor,
        *,
        target_hw: Tuple[int, int],
        features: Optional[Tensor] = None,
    ) -> Tensor:
        topo = room_topology_map
        if topo.shape[-2:] != target_hw:
            topo = F.interpolate(topo, size=target_hw, mode="bilinear", align_corners=False)

        if self.conditioning_mode == "additive":
            return self.proj(topo)

        if features is None:
            raise ValueError("SPADE-style topology conditioning requires the current grid features.")
        if tuple(features.shape[-2:]) != tuple(target_hw):
            raise ValueError(
                f"features spatial shape {tuple(features.shape[-2:])} does not match target_hw={tuple(target_hw)}."
            )

        gamma_beta = self.to_scale_shift(topo)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        normalized = self.norm(features)
        return normalized * (1.0 + gamma) + beta


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
        topology_conditioning_mode: str = "additive",
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_mode: str = "softmax",
        hedgehog_feature_dim: int = 32,
        auto_linear_attention_nodes: int = 128,
        graph_to_grid_edge_semantics: bool = False,
        edge_type_vocab_size: int = 16,
        graph_gate_init: float = -2.0,
        topology_gate_init: float = -2.0,
    ):
        super().__init__()
        self.topology_conditioner = RoomTopologyConditioner(
            topology_channels=topology_channels,
            grid_dim=grid_dim,
            conditioning_mode=topology_conditioning_mode,
        )
        self.graph_cross_attn = GraphToGridCrossAttention(
            grid_dim=grid_dim,
            graph_dim=graph_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_mode=attention_mode,
            hedgehog_feature_dim=hedgehog_feature_dim,
            auto_linear_attention_nodes=auto_linear_attention_nodes,
            use_edge_semantics=graph_to_grid_edge_semantics,
            edge_type_vocab_size=edge_type_vocab_size,
        )
        # GLIGEN-style gates are zero-initialized to protect frozen pretrained
        # backbones. H-MOLQD trains this path end-to-end, so we keep the initial
        # influence small but non-zero to avoid starving the conditioner weights
        # of gradients on the first optimization steps.
        self.topology_gate = nn.Parameter(torch.full((1,), float(topology_gate_init)))
        self.graph_gate = nn.Parameter(torch.full((1,), float(graph_gate_init)))

    def set_attention_mode(self, mode: str) -> None:
        self.graph_cross_attn.set_attention_mode(mode)

    def set_attention_capture(self, enabled: bool = True) -> None:
        self.graph_cross_attn.set_attention_capture(enabled)

    def spatial_alignment_loss(
        self,
        node_indices: Tensor,
        target_positions: Tensor,
        valid_mask: Optional[Tensor] = None,
        *,
        reduce_heads: str = "mean",
        eps: float = 1e-8,
    ) -> Tensor:
        return self.graph_cross_attn.spatial_alignment_loss(
            node_indices,
            target_positions,
            valid_mask=valid_mask,
            reduce_heads=reduce_heads,
            eps=eps,
        )

    def forward(
        self,
        x: Tensor,
        *,
        graph_nodes: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        node_positions: Optional[Tensor] = None,
        node_tpe: Optional[Tensor] = None,
        current_node_distance: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
        room_topology_map: Optional[Tensor] = None,
    ) -> Tensor:
        if room_topology_map is not None and int(room_topology_map.shape[0]) != int(x.shape[0]):
            raise ValueError(
                f"room_topology_map batch size {int(room_topology_map.shape[0])} does not match x batch size {int(x.shape[0])}."
            )
        if room_topology_map is not None:
            topo_out = self.topology_conditioner(
                room_topology_map,
                target_hw=(int(x.shape[-2]), int(x.shape[-1])),
                features=x,
            )
            gate = torch.sigmoid(self.topology_gate)
            if self.topology_conditioner.conditioning_mode == "additive":
                x = x + gate * topo_out
            else:
                x = x + gate * (topo_out - x)

        if graph_nodes is not None:
            conditioned = self.graph_cross_attn(
                x,
                graph_nodes,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_positions=node_positions,
                node_tpe=node_tpe,
                current_node_distance=current_node_distance,
                node_mask=node_mask,
            )
            x = x + torch.sigmoid(self.graph_gate) * (conditioned - x)

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
        current_node_distance: Optional[Tensor] = None,
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
        x_seq = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Self-attention
        x_normed = self.self_attn_norm(x_seq)
        attn_out, _ = self.self_attn(x_normed, x_normed, x_normed)
        x_seq = x_seq + attn_out
        
        # Reshape back to grid
        x = x_seq.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Graph cross-attention (new per-position attention)
        if graph_nodes is not None:
            x = self.graph_cross_attn(
                x,
                graph_nodes,
                edge_index,
                node_positions,
                node_tpe,
                current_node_distance,
                node_mask,
            )
        
        # Context cross-attention (backward compat)
        elif context is not None:
            x_seq = x.reshape(B, C, -1).permute(0, 2, 1)
            
            # Project context to grid dim and expand
            ctx = self.context_proj(context)  # [B, C]
            ctx = ctx.unsqueeze(1)  # [B, 1, C]
            
            x_normed = self.context_norm(x_seq)
            attn_out, _ = self.context_cross_attn(x_normed, ctx, ctx)
            x_seq = x_seq + attn_out
            
            x = x_seq.permute(0, 2, 1).reshape(B, C, H, W)
        
        # FFN
        x_seq = x.reshape(B, C, -1).permute(0, 2, 1)
        x_seq = x_seq + self.ffn(x_seq)
        x = x_seq.permute(0, 2, 1).reshape(B, C, H, W)
        
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
