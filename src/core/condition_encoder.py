"""
H-MOLQD Block III: Dual-Stream Condition Encoder
=================================================

Contextual Awareness for Conditioned Dungeon Generation.

This module implements a dual-stream architecture that combines:
- Stream A (Local): Spatial context from neighboring rooms
- Stream B (Global): Mission graph structure via GNN

Mathematical Formulation:
-------------------------
Local Stream:
    c_local = MLP(concat(z_N, z_W, boundary_feats))
    where z_N, z_W are latent tokens of North/West neighbors

Global Stream:
    h_v = GNN(node_features, edge_index)
    c_global = h_v for current node v

Cross-Attention Fusion:
    c = CrossAttn(Q=c_local, K=c_global, V=c_global)
    
Output: Conditioning vector c for diffusion model

Architecture:
- LocalStreamEncoder: Processes spatial context
- GlobalStreamEncoder: GNN for graph structure  
- CrossAttentionFusion: Merges both streams
- DualStreamConditionEncoder: Main interface

"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Try to import torch_geometric for GNN
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, MessagePassing
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logger.warning("torch_geometric not available. Using fallback GNN implementation.")


# ============================================================================
# LOCAL STREAM ENCODER (Stream A)
# ============================================================================

class LocalStreamEncoder(nn.Module):
    """
    Local context encoder for spatial awareness.
    
    Encodes information from neighboring rooms and boundary constraints
    to provide local spatial context for generation.
    
    Input Features:
    - North neighbor latent (if exists)
    - West neighbor latent (if exists)
    - Boundary constraint encoding
    - Position encoding
    
    Args:
        latent_dim: Dimension of VQ-VAE latent tokens
        hidden_dim: Hidden layer dimension
        output_dim: Output conditioning dimension
        num_neighbors: Number of neighbor directions (default: 4 for N,S,E,W)
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_neighbors: int = 4,
        boundary_dim: int = 16,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_neighbors = num_neighbors
        
        # Neighbor encoders (one per direction)
        self.neighbor_encoders = nn.ModuleDict({
            'N': nn.Linear(latent_dim, hidden_dim),
            'S': nn.Linear(latent_dim, hidden_dim),
            'E': nn.Linear(latent_dim, hidden_dim),
            'W': nn.Linear(latent_dim, hidden_dim),
        })
        
        # Boundary constraint encoder
        # Encodes which boundaries must have doors/connections
        self.boundary_encoder = nn.Sequential(
            nn.Linear(num_neighbors * 2, boundary_dim),  # 2 features per direction
            nn.ReLU(),
            nn.Linear(boundary_dim, hidden_dim),
        )
        
        # Position encoder (row, col in dungeon layout)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )
        
        # Fusion MLP
        # Input: 4 neighbors + boundary + position
        fusion_input_dim = hidden_dim * (num_neighbors + 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Learnable null token for missing neighbors
        self.null_token = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.null_token, std=0.02)
    
    def forward(
        self,
        neighbor_latents: Dict[str, Optional[Tensor]],
        boundary_constraints: Tensor,
        position: Tensor,
    ) -> Tensor:
        """
        Encode local spatial context.
        
        Args:
            neighbor_latents: Dict mapping direction to latent tensor or None
                {'N': [B, D], 'S': [B, D], 'E': [B, D], 'W': [B, D]}
            boundary_constraints: [B, num_neighbors * 2] boundary features
                Format: [has_door_N, required_N, has_door_S, required_S, ...]
            position: [B, 2] position encoding (row, col)
            
        Returns:
            Local conditioning vector [B, output_dim]
        """
        batch_size = position.shape[0]
        device = position.device
        
        # Encode each neighbor
        neighbor_features = []
        for direction in ['N', 'S', 'E', 'W']:
            latent = neighbor_latents.get(direction)
            
            if latent is None:
                # Use null token
                latent = self.null_token.unsqueeze(0).expand(batch_size, -1)
            
            # Pool spatial dimensions if needed
            if latent.dim() > 2:
                latent = latent.mean(dim=[-2, -1])  # Global average pooling
            
            encoded = self.neighbor_encoders[direction](latent)
            neighbor_features.append(encoded)
        
        # Encode boundary constraints
        boundary_feat = self.boundary_encoder(boundary_constraints)
        
        # Encode position
        position_feat = self.position_encoder(position.float())
        
        # Concatenate all features
        all_features = torch.cat(
            neighbor_features + [boundary_feat, position_feat],
            dim=-1
        )
        
        # Fuse into output conditioning
        c_local = self.fusion(all_features)
        
        return c_local


# ============================================================================
# GLOBAL STREAM ENCODER (Stream B) - GNN
# ============================================================================

class FallbackGNN(nn.Module):
    """
    Simple fallback GNN when torch_geometric is not available.
    Uses basic message passing with adjacency matrix.
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # Message passing layers
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            node_features: [N, node_dim]
            adjacency: [N, N] adjacency matrix
            
        Returns:
            Node embeddings [N, output_dim]
        """
        h = self.input_proj(node_features)
        
        for layer in self.layers:
            # Aggregate neighbor features
            neighbor_sum = torch.matmul(adjacency, h)
            # Concatenate self and neighbor
            combined = torch.cat([h, neighbor_sum], dim=-1)
            h = layer(combined) + h  # Residual
        
        return self.output_proj(h)


class GlobalStreamEncoder(nn.Module):
    """
    Global context encoder using Graph Neural Network.
    
    Encodes the mission graph structure to provide global context
    about required items, dependencies, and dungeon topology.
    
    Uses either torch_geometric GNN or fallback implementation.
    
    Args:
        node_feature_dim: Dimension of node features (default: 5)
        edge_feature_dim: Dimension of edge features (default: 3)
        hidden_dim: GNN hidden dimension
        output_dim: Output conditioning dimension
        num_layers: Number of GNN layers
        gnn_type: Type of GNN ('gcn', 'gat', or 'sage')
    """
    
    def __init__(
        self,
        node_feature_dim: int = 6,
        edge_feature_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        gnn_type: str = 'gat',
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_torch_geometric = HAS_TORCH_GEOMETRIC
        
        if HAS_TORCH_GEOMETRIC:
            self._build_torch_geometric_gnn(
                node_feature_dim, edge_feature_dim, 
                hidden_dim, output_dim, num_layers, 
                gnn_type, num_heads
            )
        else:
            self.gnn = FallbackGNN(
                node_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
            )
        
        # TPE (Topological Positional Encoding) projection
        self.tpe_proj = nn.Linear(8, hidden_dim)
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim + hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def _build_torch_geometric_gnn(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        gnn_type: str,
        num_heads: int,
    ):
        """
        Build GNN layers using torch_geometric.
        
        Phase 3A: Uses GATv2Conv with edge features to capture edge types
        (key_locked, boss_locked, bombable, stair, switch) which are critical
        for conditioning the dungeon generator.
        """
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)  # Phase 3A
        
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'gat':
                # Phase 3A: GATv2Conv supports edge features natively
                # Edge types (key_locked, boss_locked, etc.) are critical
                # for the model to understand dungeon structure
                layer = GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    edge_dim=hidden_dim,  # Encoded edge features
                )
            else:  # gcn
                layer = GCNConv(hidden_dim, hidden_dim)
            
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.node_output = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        tpe: Optional[Tensor] = None,
        batch_idx: Optional[Tensor] = None,
        node_idx: Optional[int] = None,
    ) -> Tensor:
        """
        Encode global graph context.
        
        Args:
            node_features: [N, node_feature_dim] node features
            edge_index: [2, E] edge indices
            edge_features: [E, edge_feature_dim] edge type features (Phase 3A)
            tpe: [N, 8] topological positional encoding (optional)
            batch_idx: [N] batch assignment for batched graphs
            node_idx: Target node index to return embedding for
            
        Returns:
            Global conditioning vector [B, output_dim] or [N, output_dim]
        """
        if self.use_torch_geometric:
            h = self._forward_torch_geometric(node_features, edge_index, edge_features)
        else:
            # Build adjacency matrix from edge_index
            N = node_features.shape[0]
            adj = torch.zeros(N, N, device=node_features.device)
            if edge_index.numel() > 0:
                adj[edge_index[0], edge_index[1]] = 1.0
            h = self.gnn(node_features, adj)
        
        # Add TPE if available
        if tpe is not None:
            tpe_feat = self.tpe_proj(tpe)
            h = self.output_proj(torch.cat([h, tpe_feat], dim=-1))
        else:
            # Pad with zeros of hidden_dim width to match output_proj input size
            pad = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
            h = self.output_proj(torch.cat([h, pad], dim=-1))
        
        # Return specific node embedding or all
        if node_idx is not None:
            return h[node_idx:node_idx+1]
        
        return h
    
    def _forward_torch_geometric(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward using torch_geometric layers.
        
        Phase 3A: Passes encoded edge features to GATv2Conv so the GNN
        can distinguish edge types (key_locked, boss_locked, stair, etc.).
        """
        h = self.node_encoder(node_features)
        
        # Encode edge features if available
        edge_attr = None
        if edge_features is not None and hasattr(self, 'edge_encoder'):
            edge_attr = self.edge_encoder(edge_features)
        
        for layer, norm in zip(self.gnn_layers, self.layer_norms):
            if edge_attr is not None and isinstance(layer, GATv2Conv):
                h_new = layer(h, edge_index, edge_attr=edge_attr)
            else:
                h_new = layer(h, edge_index)
            h_new = norm(h_new)
            h = F.relu(h_new) + h  # Residual connection
        
        return self.node_output(h)


# ============================================================================
# CROSS-ATTENTION FUSION
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention layer to fuse local and global contexts.
    
    Uses local context as queries and global context as keys/values
    to produce a unified conditioning vector.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Args:
        local_dim: Dimension of local stream output
        global_dim: Dimension of global stream output
        output_dim: Output conditioning dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        local_dim: int = 256,
        global_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Query from local
        self.query_proj = nn.Linear(local_dim, output_dim)
        
        # Key and Value from global
        self.key_proj = nn.Linear(global_dim, output_dim)
        self.value_proj = nn.Linear(global_dim, output_dim)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        c_local: Tensor,
        c_global: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Fuse local and global conditioning via cross-attention.
        
        Args:
            c_local: Local conditioning [B, local_dim]
            c_global: Global conditioning [B, N, global_dim] or [N, global_dim]
            mask: Optional attention mask [B, N]
            
        Returns:
            Fused conditioning [B, output_dim]
        """
        B = c_local.shape[0]
        
        # Ensure c_global has batch dimension
        if c_global.dim() == 2:
            c_global = c_global.unsqueeze(0).expand(B, -1, -1)
        
        N = c_global.shape[1]
        
        # Add sequence dimension to local (single query)
        c_local = c_local.unsqueeze(1)  # [B, 1, local_dim]
        
        # Project Q, K, V
        Q = self.query_proj(c_local)  # [B, 1, output_dim]
        K = self.key_proj(c_global)    # [B, N, output_dim]
        V = self.value_proj(c_global)  # [B, N, output_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, 1, N]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, 1, D]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.output_dim)
        attn_output = self.output_proj(attn_output)
        
        # Add & Norm
        c_out = self.layer_norm(c_local + attn_output)
        
        # FFN + Add & Norm
        c_out = self.ffn_norm(c_out + self.ffn(c_out))
        
        # Remove sequence dimension
        c_out = c_out.squeeze(1)  # [B, output_dim]
        
        return c_out


# ============================================================================
# DUAL-STREAM CONDITION ENCODER (Main Interface)
# ============================================================================

class DualStreamConditionEncoder(nn.Module):
    """
    Dual-Stream Condition Encoder for H-MOLQD Block III.
    
    Combines local spatial context (neighboring rooms) with global
    mission graph structure to produce conditioning for the diffusion model.
    
    Architecture:
        Stream A (Local):  neighbor_latents → LocalStreamEncoder → c_local
        Stream B (Global): graph_data → GlobalStreamEncoder → c_global
        Fusion:            CrossAttention(c_local, c_global) → c
    
    Args:
        latent_dim: VQ-VAE latent dimension
        node_feature_dim: Graph node feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Final conditioning dimension
        num_gnn_layers: Number of GNN layers
        num_attention_heads: Number of cross-attention heads
    
    Usage:
        encoder = DualStreamConditionEncoder()
        
        # Prepare inputs
        neighbor_latents = {'N': z_north, 'W': z_west, 'S': None, 'E': None}
        boundary = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 0]])  # [B, 8]
        position = torch.tensor([[2, 3]])  # [B, 2]
        
        # Graph data
        node_features = ...  # [N, 5]
        edge_index = ...     # [2, E]
        tpe = ...            # [N, 8]
        current_node = 5
        
        # Get conditioning
        c = encoder(
            neighbor_latents, boundary, position,
            node_features, edge_index, tpe, current_node
        )
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        node_feature_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_gnn_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Stream A: Local context
        self.local_encoder = LocalStreamEncoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        
        # Stream B: Global context
        self.global_encoder = GlobalStreamEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_gnn_layers,
        )
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            local_dim=output_dim,
            global_dim=output_dim,
            output_dim=output_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(
        self,
        neighbor_latents: Dict[str, Optional[Tensor]],
        boundary_constraints: Tensor,
        position: Tensor,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        tpe: Optional[Tensor] = None,
        current_node_idx: Optional[int] = None,
    ) -> Tensor:
        """
        Compute conditioning vector from local and global context.
        
        Args:
            neighbor_latents: Dict of neighboring room latents
            boundary_constraints: [B, 8] boundary features
            position: [B, 2] room position
            node_features: [N, node_feature_dim] graph node features
            edge_index: [2, E] graph edges
            edge_features: [E, edge_feature_dim] edge type features (Phase 3A)
            tpe: [N, 8] topological positional encoding
            current_node_idx: Index of current node in graph
            
        Returns:
            Conditioning vector [B, output_dim]
        """
        # Stream A: Local context
        c_local = self.local_encoder(
            neighbor_latents,
            boundary_constraints,
            position,
        )
        
        # Stream B: Global context
        c_global = self.global_encoder(
            node_features,
            edge_index,
            edge_features=edge_features,
            tpe=tpe,
            node_idx=current_node_idx,
        )
        
        # Ensure c_global has proper shape for cross-attention
        if c_global.dim() == 2:
            c_global = c_global.unsqueeze(0).expand(c_local.shape[0], -1, -1)
        
        # Cross-attention fusion
        c_fused = self.fusion(c_local, c_global)
        
        # Final projection
        c = self.output_proj(c_fused)
        
        return c
    
    def encode_local_only(
        self,
        neighbor_latents: Dict[str, Optional[Tensor]],
        boundary_constraints: Tensor,
        position: Tensor,
    ) -> Tensor:
        """Encode only local context (for inference without graph)."""
        return self.local_encoder(
            neighbor_latents,
            boundary_constraints,
            position,
        )
    
    def encode_global_only(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        tpe: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode only global context (all nodes)."""
        return self.global_encoder(
            node_features,
            edge_index,
            edge_features=edge_features,
            tpe=tpe,
        )


# ============================================================================
# UTILITIES
# ============================================================================

def build_boundary_constraints(
    has_neighbor: Dict[str, bool],
    required_door: Dict[str, bool],
) -> Tensor:
    """
    Build boundary constraint tensor from neighbor/door info.
    
    Args:
        has_neighbor: Dict of {direction: has_neighbor_room}
        required_door: Dict of {direction: must_have_door}
        
    Returns:
        [8] tensor: [has_N, req_N, has_S, req_S, has_E, req_E, has_W, req_W]
    """
    constraints = []
    for direction in ['N', 'S', 'E', 'W']:
        constraints.append(float(has_neighbor.get(direction, False)))
        constraints.append(float(required_door.get(direction, False)))
    return torch.tensor(constraints, dtype=torch.float32)


def graph_to_edge_index(adj_matrix: Tensor) -> Tensor:
    """
    Convert adjacency matrix to edge_index format.
    
    Args:
        adj_matrix: [N, N] adjacency matrix
        
    Returns:
        [2, E] edge index tensor
    """
    edges = torch.nonzero(adj_matrix, as_tuple=False)
    return edges.t().contiguous()


def create_condition_encoder(
    latent_dim: int = 64,
    output_dim: int = 256,
    **kwargs,
) -> DualStreamConditionEncoder:
    """
    Create a DualStreamConditionEncoder.
    
    Args:
        latent_dim: VQ-VAE latent dimension
        output_dim: Conditioning output dimension
        **kwargs: Additional arguments
        
    Returns:
        DualStreamConditionEncoder instance
    """
    return DualStreamConditionEncoder(
        latent_dim=latent_dim,
        output_dim=output_dim,
        **kwargs,
    )
