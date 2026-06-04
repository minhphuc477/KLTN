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
    c_local = MLP(concat(z_N, z_S, z_E, z_W, boundary_feats, position_feats))
    where z_* are latent tokens of the four cardinal neighbors and
    boundary/position features anchor room-local geometry

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
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.definitions import GRAPH_EDGE_FEATURE_DIM, GRAPH_NODE_FEATURE_DIM, GRAPH_TPE_DIM

logger = logging.getLogger(__name__)
CARDINAL_DIRECTIONS = ("N", "S", "E", "W")

# Try to import torch_geometric for GNN
try:
    from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
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

    def _pool_neighbor_latent(self, latent: Tensor, direction: str) -> Tensor:
        """Pool the neighbor edge that faces the current room."""
        if latent.dim() <= 2:
            return latent

        direction = str(direction).upper()
        channel_first = latent.dim() == 4 and int(latent.shape[1]) == self.latent_dim
        channel_last = latent.dim() == 4 and int(latent.shape[-1]) == self.latent_dim

        if channel_first:
            # Neighbor N touches this room with its south edge; S with north;
            # E with west; W with east.
            if direction == "N":
                edge = latent[:, :, -1, :]
            elif direction == "S":
                edge = latent[:, :, 0, :]
            elif direction == "E":
                edge = latent[:, :, :, 0]
            elif direction == "W":
                edge = latent[:, :, :, -1]
            else:
                edge = latent.flatten(2)
            return edge.mean(dim=-1)

        if channel_last:
            if direction == "N":
                edge = latent[:, -1, :, :]
            elif direction == "S":
                edge = latent[:, 0, :, :]
            elif direction == "E":
                edge = latent[:, :, 0, :]
            elif direction == "W":
                edge = latent[:, :, -1, :]
            else:
                edge = latent.flatten(1, -2)
            return edge.mean(dim=1)

        return latent.flatten(1, -2).mean(dim=1)
    
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
        
        # Encode each neighbor
        neighbor_features = []
        for direction in ['N', 'S', 'E', 'W']:
            latent = neighbor_latents.get(direction)
            
            if latent is None:
                # Use null token
                latent = self.null_token.unsqueeze(0).expand(batch_size, -1)
            
            # Pool the boundary-facing edge of spatial latents if needed.
            if latent.dim() > 2:
                latent = self._pool_neighbor_latent(latent, direction)
            
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


class ReferenceRoomMapEncoder(nn.Module):
    """
    Lightweight exemplar encoder over discrete neighboring room maps.

    This mirrors the "reference-derived context" lesson from recent prompt-free
    and single-example diffusion work: preserve domain-native local structure
    instead of forcing style/control through a heavyweight generic encoder.
    """

    def __init__(
        self,
        num_tile_types: int = 44,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 256,
    ):
        super().__init__()
        self.num_tile_types = int(max(2, num_tile_types))
        self.embedding_dim = int(max(4, embedding_dim))
        self.hidden_dim = int(max(4, hidden_dim))
        self.output_dim = int(max(4, output_dim))

        self.tile_embedding = nn.Embedding(self.num_tile_types, self.embedding_dim)
        self.direction_embedding = nn.Embedding(len(CARDINAL_DIRECTIONS), self.embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.embedding_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        nn.init.normal_(self.tile_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.direction_embedding.weight, mean=0.0, std=0.02)

    def _coerce_room_map(
        self,
        room_map: Union[Tensor, torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if not isinstance(room_map, torch.Tensor):
            room_map = torch.as_tensor(room_map)

        if room_map.dim() == 2:
            room_map = room_map.unsqueeze(0)
        elif room_map.dim() == 4:
            if int(room_map.shape[1]) != 1:
                raise ValueError(
                    f"Reference room maps must have a singleton channel dimension, got shape={tuple(room_map.shape)}."
                )
            room_map = room_map.squeeze(1)
        elif room_map.dim() != 3:
            raise ValueError(
                f"Reference room maps must be rank-2/3/4, got shape={tuple(room_map.shape)}."
            )

        room_map = room_map.to(device)
        if int(room_map.shape[0]) == 1 and batch_size > 1:
            room_map = room_map.expand(batch_size, -1, -1)
        elif int(room_map.shape[0]) != batch_size:
            raise ValueError(
                f"Reference room map batch mismatch: expected batch_size={batch_size}, got shape={tuple(room_map.shape)}."
            )

        if room_map.is_floating_point():
            max_value = float(room_map.detach().max().item()) if room_map.numel() > 0 else 0.0
            if max_value <= 1.0 + 1e-6:
                room_map = room_map * float(self.num_tile_types - 1)
            room_map = room_map.round()

        room_map = room_map.long().clamp_(0, self.num_tile_types - 1)
        return room_map

    def forward(
        self,
        reference_room_maps: Optional[Dict[str, Optional[Tensor]]],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if not isinstance(reference_room_maps, dict):
            return torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

        encoded_maps = []
        for direction_index, direction in enumerate(CARDINAL_DIRECTIONS):
            room_map = reference_room_maps.get(direction)
            if room_map is None:
                continue
            room_ids = self._coerce_room_map(room_map, batch_size=batch_size, device=device)
            room_embed = self.tile_embedding(room_ids).permute(0, 3, 1, 2).contiguous()
            direction_bias = self.direction_embedding.weight[direction_index].view(1, -1, 1, 1)
            pooled = self.encoder(room_embed + direction_bias).flatten(1)
            encoded_maps.append(pooled)

        if not encoded_maps:
            return torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

        stacked = torch.stack(encoded_maps, dim=0).mean(dim=0)
        return self.output_proj(stacked).to(dtype=dtype)


# ============================================================================
# GLOBAL STREAM ENCODER (Stream B) - GNN
# ============================================================================

class FallbackGNN(nn.Module):
    """
    Simple fallback GNN when torch_geometric is not available.
    Uses sparse edge-index message passing.
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
        edge_index: Tensor,
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            node_features: [N, node_dim]
            edge_index: [2, E] sparse edge indices
            
        Returns:
            Node embeddings [N, output_dim]
        """
        h = self.input_proj(node_features)

        num_nodes = int(node_features.shape[0])
        if edge_index is None or edge_index.numel() == 0:
            src = torch.empty(0, dtype=torch.long, device=node_features.device)
            dst = torch.empty(0, dtype=torch.long, device=node_features.device)
            norm = torch.empty(0, dtype=h.dtype, device=h.device)
        else:
            edge_index = edge_index.to(device=node_features.device, dtype=torch.long)
            src = torch.cat([edge_index[0], edge_index[1]], dim=0)
            dst = torch.cat([edge_index[1], edge_index[0]], dim=0)
            valid = (
                (src >= 0)
                & (src < num_nodes)
                & (dst >= 0)
                & (dst < num_nodes)
            )
            src = src[valid]
            dst = dst[valid]
            degree = torch.zeros(num_nodes, device=h.device, dtype=h.dtype)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            deg_inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
            norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        for layer in self.layers:
            neighbor_sum = torch.zeros_like(h)
            if src.numel() > 0:
                neighbor_sum.index_add_(0, dst, h[src] * norm.unsqueeze(-1))
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
        node_feature_dim: Dimension of node features (default: 14)
        edge_feature_dim: Dimension of edge features (default: 16)
        hidden_dim: GNN hidden dimension
        output_dim: Output conditioning dimension
        num_layers: Number of GNN layers
        gnn_type: Type of GNN ('gcn', 'gat', 'sage', or 'gps')
    """
    
    def __init__(
        self,
        node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        gnn_type: str = 'gcn',
        num_heads: int = 4,
        dropout: float = 0.1,
        use_current_node_distance_features: bool = True,
        current_node_distance_dim: int = 4,
        use_rrwp_edge_features: bool = False,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_current_node_distance_features = bool(use_current_node_distance_features)
        self.use_rrwp_edge_features = bool(use_rrwp_edge_features)
        self.current_node_distance_dim = int(max(1, current_node_distance_dim))
        self.gnn_type = str(gnn_type).strip().lower()
        if self.gnn_type not in {"gcn", "gat", "sage", "gps"}:
            raise ValueError(
                f"Unsupported gnn_type={gnn_type!r}. Expected 'gcn', 'gat', 'sage', or 'gps'."
            )
        if self.use_rrwp_edge_features and self.gnn_type in {"gcn", "sage"}:
            raise ValueError(
                "RRWP edge features require gnn_type='gat' or gnn_type='gps'. "
                f"Got gnn_type={gnn_type!r} with use_rrwp_edge_features=True."
            )
        self.use_torch_geometric = HAS_TORCH_GEOMETRIC and self.gnn_type != "gps"
        self._warned_shape_mismatches = set()

        if self.gnn_type == "gps":
            self._build_gps_gnn(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif self.use_torch_geometric:
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
        self.tpe_proj = nn.Linear(int(GRAPH_TPE_DIM), hidden_dim)
        self.edge_rrwp_proj = nn.Linear(int(GRAPH_TPE_DIM), hidden_dim) if self.use_rrwp_edge_features else None
        self.current_node_distance_proj = nn.Linear(self.current_node_distance_dim, hidden_dim)
        self.current_node_distance_gate = nn.Parameter(torch.tensor(0.0))

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
        
        for _i in range(num_layers):
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
            elif gnn_type == 'sage':
                layer = SAGEConv(hidden_dim, hidden_dim)
            else:  # gcn
                layer = GCNConv(hidden_dim, hidden_dim)
            
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.node_output = nn.Linear(hidden_dim, output_dim)

    def _build_gps_gnn(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        """Build a GraphGPS-style hybrid encoder with local MPNN plus global attention."""
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        self.gps_layers = nn.ModuleList(
            [
                GPSLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.node_output = nn.Linear(hidden_dim, output_dim)

    def _warn_once(self, key: str, message: str) -> None:
        """Log noisy tensor-shape warnings only once per mismatch signature."""
        if key in self._warned_shape_mismatches:
            return
        self._warned_shape_mismatches.add(key)
        logger.warning(message)

    def _align_feature_dim(
        self,
        features: Tensor,
        expected_dim: int,
        feature_name: str,
    ) -> Tensor:
        """
        Align feature width to model expectation using truncate/pad.

        This keeps older datasets (e.g., 5-D node features) and newer schemas
        (e.g., 6-D) compatible with a single checkpoint/API surface.
        """
        if features.dim() != 2:
            raise ValueError(
                f"{feature_name} must be 2D [N, D], got shape {tuple(features.shape)}"
            )

        current_dim = features.shape[1]
        if current_dim == expected_dim:
            return features

        self._warn_once(
            f"{feature_name}:{current_dim}->{expected_dim}",
            (
                f"{feature_name} feature dim mismatch: got {current_dim}, "
                f"expected {expected_dim}. Applying automatic pad/truncate."
            ),
        )

        if current_dim > expected_dim:
            return features[:, :expected_dim]

        pad = torch.zeros(
            features.shape[0],
            expected_dim - current_dim,
            device=features.device,
            dtype=features.dtype,
        )
        return torch.cat([features, pad], dim=-1)

    def _prepare_node_features(self, node_features: Tensor) -> Tensor:
        """Normalize node features to float [N, node_feature_dim]."""
        if node_features.dim() == 1:
            node_features = node_features.unsqueeze(-1)
        if not torch.is_floating_point(node_features):
            node_features = node_features.float()
        return self._align_feature_dim(
            node_features,
            expected_dim=self.node_feature_dim,
            feature_name="node_features",
        )

    def _prepare_edge_features(
        self,
        edge_features: Optional[Tensor],
        edge_index: Tensor,
    ) -> Optional[Tensor]:
        """Normalize edge features to float [E, edge_feature_dim]."""
        if edge_features is None:
            return None

        if edge_features.dim() == 1:
            # Integer labels -> one-hot (preferred), scalar values -> column vector.
            if edge_features.dtype in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                num_classes = max(1, int(self.edge_feature_dim))
                labels = edge_features.long()
                if labels.numel() > 0:
                    min_label = int(labels.min().item())
                    max_label = int(labels.max().item())
                    if min_label < 0 or max_label >= num_classes:
                        self._warn_once(
                            f"edge_label_clamp:{min_label}:{max_label}->{num_classes}",
                            (
                                f"edge_features integer labels out of range for fixed width {num_classes}. "
                                "Clamping to preserve a stable one-hot schema."
                            ),
                        )
                edge_features = F.one_hot(
                    labels.clamp(min=0, max=num_classes - 1),
                    num_classes=num_classes,
                ).float()
            else:
                edge_features = edge_features.unsqueeze(-1)
        elif edge_features.dim() != 2:
            raise ValueError(
                f"edge_features must be 1D labels or 2D [E, D], got {tuple(edge_features.shape)}"
            )

        if not torch.is_floating_point(edge_features):
            edge_features = edge_features.float()

        if edge_index.dim() == 2 and edge_index.shape[0] == 2:
            num_edges = int(edge_index.shape[1])
            if edge_features.shape[0] != num_edges:
                self._warn_once(
                    f"edge_rows:{edge_features.shape[0]}->{num_edges}",
                    (
                        f"edge_features edge count mismatch: got {edge_features.shape[0]}, "
                        f"expected {num_edges}. Applying automatic row align."
                    ),
                )
                if edge_features.shape[0] > num_edges:
                    edge_features = edge_features[:num_edges]
                else:
                    row_pad = torch.zeros(
                        num_edges - edge_features.shape[0],
                        edge_features.shape[1],
                        device=edge_features.device,
                        dtype=edge_features.dtype,
                    )
                    edge_features = torch.cat([edge_features, row_pad], dim=0)

        return self._align_feature_dim(
            edge_features,
            expected_dim=self.edge_feature_dim,
            feature_name="edge_features",
        )

    def _align_projected_edge_attr(self, edge_attr: Tensor, *, feature_name: str) -> Tensor:
        """Align projected edge attributes to the GNN edge hidden width."""
        if edge_attr.dim() != 2:
            raise ValueError(f"{feature_name} must be [E,D], got {tuple(edge_attr.shape)}.")
        expected_dim = int(self.hidden_dim)
        current_dim = int(edge_attr.shape[1])
        if current_dim == expected_dim:
            return edge_attr
        self._warn_once(
            f"{feature_name}_projected_width:{current_dim}->{expected_dim}",
            (
                f"{feature_name} projected width {current_dim} does not match hidden_dim={expected_dim}. "
                "Aligning width instead of dropping edge-conditioning signal."
            ),
        )
        return self._align_feature_dim(
            edge_attr,
            expected_dim=expected_dim,
            feature_name=feature_name,
        )

    def _prepare_tpe(
        self,
        tpe: Optional[Tensor],
        num_nodes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        """Normalize topological positional encodings to [N, D]."""
        if tpe is None:
            return None
        tensor = tpe.to(device=device, dtype=dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise ValueError(f"tpe must be 2D [N, D], got shape {tuple(tensor.shape)}")

        expected_tpe_dim = int(GRAPH_TPE_DIM)
        if int(tensor.shape[0]) != int(num_nodes) or int(tensor.shape[1]) != expected_tpe_dim:
            self._warn_once(
                f"tpe:{tuple(tensor.shape)}->{(int(num_nodes), expected_tpe_dim)}",
                (
                    f"tpe shape mismatch: got {tuple(tensor.shape)}, expected {(int(num_nodes), expected_tpe_dim)}. "
                    "Applying automatic pad/truncate."
                ),
            )
            aligned = torch.zeros(int(num_nodes), expected_tpe_dim, device=device, dtype=dtype)
            rows = min(int(num_nodes), int(tensor.shape[0]))
            cols = min(expected_tpe_dim, int(tensor.shape[1]))
            if rows > 0 and cols > 0:
                aligned[:rows, :cols] = tensor[:rows, :cols]
            tensor = aligned
        return tensor

    def _prepare_edge_rrwp(
        self,
        edge_rrwp: Optional[Tensor],
        edge_index: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        """Normalize RRWP edge features to [E, GRAPH_TPE_DIM]."""
        if not self.use_rrwp_edge_features or edge_rrwp is None:
            return None
        tensor = edge_rrwp.to(device=device, dtype=dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.dim() != 2:
            raise ValueError(f"edge_rrwp must be 2D [E,D], got shape {tuple(tensor.shape)}")

        num_edges = int(edge_index.shape[1]) if edge_index.dim() == 2 and edge_index.shape[0] == 2 else 0
        expected_dim = int(GRAPH_TPE_DIM)
        if int(tensor.shape[0]) != num_edges or int(tensor.shape[1]) != expected_dim:
            self._warn_once(
                f"edge_rrwp:{tuple(tensor.shape)}->{(num_edges, expected_dim)}",
                (
                    f"edge_rrwp shape mismatch: got {tuple(tensor.shape)}, expected {(num_edges, expected_dim)}. "
                    "Applying automatic pad/truncate."
                ),
            )
            aligned = torch.zeros(num_edges, expected_dim, device=device, dtype=dtype)
            rows = min(num_edges, int(tensor.shape[0]))
            cols = min(expected_dim, int(tensor.shape[1]))
            if rows > 0 and cols > 0:
                aligned[:rows, :cols] = tensor[:rows, :cols]
            tensor = aligned
        return tensor

    def _prepare_current_node_distance(
        self,
        current_node_distance: Optional[Tensor],
        num_nodes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        """Normalize current-room distance features to [N, D]."""
        if not self.use_current_node_distance_features or current_node_distance is None:
            return None
        tensor = current_node_distance.to(device=device, dtype=dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise ValueError(
                "current_node_distance must be 2D [N, D], "
                f"got shape {tuple(tensor.shape)}"
            )

        expected_shape = (int(num_nodes), int(self.current_node_distance_dim))
        if tuple(tensor.shape) != expected_shape:
            self._warn_once(
                f"current_node_distance:{tuple(tensor.shape)}->{expected_shape}",
                (
                    "current_node_distance shape mismatch: got "
                    f"{tuple(tensor.shape)}, expected {expected_shape}. "
                    "Applying automatic pad/truncate."
                ),
            )
            aligned = torch.zeros(*expected_shape, device=device, dtype=dtype)
            rows = min(int(num_nodes), int(tensor.shape[0]))
            cols = min(int(self.current_node_distance_dim), int(tensor.shape[1]))
            if rows > 0 and cols > 0:
                aligned[:rows, :cols] = tensor[:rows, :cols]
            tensor = aligned
        return tensor

    def _prepare_node_mask(
        self,
        node_mask: Optional[Tensor],
        num_nodes: int,
        *,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Normalize valid-node masks to bool [N]."""
        if node_mask is None:
            return None
        tensor = node_mask.to(device=device)
        if tensor.dim() == 2 and int(tensor.shape[0]) == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 1:
            raise ValueError(f"node_mask must be 1D [N] or [1,N], got {tuple(tensor.shape)}.")
        expected_shape = (int(num_nodes),)
        if tuple(tensor.shape) != expected_shape:
            self._warn_once(
                f"node_mask:{tuple(tensor.shape)}->{expected_shape}",
                (
                    f"node_mask shape mismatch: got {tuple(tensor.shape)}, expected {expected_shape}. "
                    "Applying automatic pad/truncate."
                ),
            )
            aligned = torch.zeros(int(num_nodes), device=device, dtype=torch.bool)
            rows = min(int(num_nodes), int(tensor.shape[0]))
            if rows > 0:
                aligned[:rows] = tensor[:rows].to(dtype=torch.bool)
            return aligned
        return tensor.to(dtype=torch.bool)
    
    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        edge_rrwp: Optional[Tensor] = None,
        tpe: Optional[Tensor] = None,
        current_node_distance: Optional[Tensor] = None,
        batch_idx: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
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
            node_mask: [N] valid-node mask for padded graph batches
            node_idx: Target node index to return embedding for
            
        Returns:
            Global conditioning vector [B, output_dim] or [N, output_dim]
        """
        node_features = self._prepare_node_features(node_features)
        edge_features = self._prepare_edge_features(edge_features, edge_index)
        edge_rrwp = self._prepare_edge_rrwp(
            edge_rrwp,
            edge_index,
            device=node_features.device,
            dtype=node_features.dtype,
        )
        tpe = self._prepare_tpe(
            tpe,
            num_nodes=int(node_features.shape[0]),
            device=node_features.device,
            dtype=node_features.dtype,
        )
        current_node_distance = self._prepare_current_node_distance(
            current_node_distance,
            num_nodes=int(node_features.shape[0]),
            device=node_features.device,
            dtype=node_features.dtype,
        )
        prepared_node_mask = self._prepare_node_mask(
            node_mask,
            int(node_features.shape[0]),
            device=node_features.device,
        )

        if self.use_torch_geometric:
            h = self._forward_torch_geometric(node_features, edge_index, edge_features, edge_rrwp)
        elif self.gnn_type == "gps":
            h = self._forward_gps(
                node_features,
                edge_index,
                edge_features,
                edge_rrwp,
                batch_idx=batch_idx,
                node_mask=prepared_node_mask,
            )
        else:
            h = self.gnn(node_features, edge_index)
        
        aux_features = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        if tpe is not None:
            aux_features = aux_features + self.tpe_proj(tpe)
        if current_node_distance is not None:
            aux_features = aux_features + (
                torch.sigmoid(self.current_node_distance_gate)
                * self.current_node_distance_proj(current_node_distance)
            )
        h = self.output_proj(torch.cat([h, aux_features], dim=-1))
        if prepared_node_mask is not None:
            h = h * prepared_node_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
        
        # Return specific node embedding or all
        if node_idx is not None:
            return h[node_idx:node_idx+1]
        
        return h
    
    def _forward_torch_geometric(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        edge_rrwp: Optional[Tensor] = None,
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
            edge_attr = self._align_projected_edge_attr(
                self.edge_encoder(edge_features),
                feature_name="edge_attr",
            )
        if edge_rrwp is not None and self.edge_rrwp_proj is not None:
            rrwp_attr = self._align_projected_edge_attr(
                self.edge_rrwp_proj(edge_rrwp),
                feature_name="edge_rrwp",
            )
            if edge_attr is not None and int(rrwp_attr.shape[0]) != int(edge_attr.shape[0]):
                self._warn_once(
                    f"edge_rrwp_attr_mismatch:{tuple(rrwp_attr.shape)}!={tuple(edge_attr.shape)}",
                    (
                        "Skipping RRWP edge attributes because projected edge_rrwp rows "
                        f"{int(rrwp_attr.shape[0])} do not match edge_attr rows {int(edge_attr.shape[0])}."
                    ),
                )
                rrwp_attr = None
            if rrwp_attr is not None:
                edge_attr = rrwp_attr if edge_attr is None else edge_attr + rrwp_attr
        
        for layer, norm in zip(self.gnn_layers, self.layer_norms):
            if edge_attr is not None and isinstance(layer, GATv2Conv):
                h_new = layer(h, edge_index, edge_attr=edge_attr)
            else:
                h_new = layer(h, edge_index)
            h_new = norm(h_new)
            h = F.relu(h_new) + h  # Residual connection
        
        return self.node_output(h)

    def _forward_gps(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        edge_rrwp: Optional[Tensor] = None,
        batch_idx: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward through a GraphGPS-style encoder.

        GraphGPS couples local message passing with a global attention branch.
        This improves long-range information flow without discarding the graph
        inductive bias that the mission-conditioning path relies on.
        """
        h = self.node_encoder(node_features)
        edge_attr = (
            self._align_projected_edge_attr(self.edge_encoder(edge_features), feature_name="gps_edge_attr")
            if edge_features is not None else None
        )
        if edge_rrwp is not None and self.edge_rrwp_proj is not None:
            rrwp_attr = self._align_projected_edge_attr(
                self.edge_rrwp_proj(edge_rrwp),
                feature_name="gps_edge_rrwp",
            )
            if edge_attr is not None and int(rrwp_attr.shape[0]) != int(edge_attr.shape[0]):
                self._warn_once(
                    f"gps_edge_rrwp_attr_mismatch:{tuple(rrwp_attr.shape)}!={tuple(edge_attr.shape)}",
                    (
                        "Skipping GPS RRWP edge attributes because projected edge_rrwp rows "
                        f"{int(rrwp_attr.shape[0])} do not match edge_attr rows {int(edge_attr.shape[0])}."
                    ),
                )
                rrwp_attr = None
            if rrwp_attr is not None:
                edge_attr = rrwp_attr if edge_attr is None else edge_attr + rrwp_attr

        for layer in self.gps_layers:
            h = layer(
                h,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch_idx=batch_idx,
                node_mask=node_mask,
            )

        return self.node_output(h)


class GPSLayer(nn.Module):
    """GraphGPS-style block: local message passing, global attention, then FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError(f"GPSLayer num_heads must be > 0, got {num_heads}.")
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError(
                f"GPSLayer requires hidden_dim divisible by num_heads; got hidden_dim={hidden_dim}, num_heads={num_heads}."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = nn.Dropout(float(dropout))
        self.local_norm = nn.LayerNorm(self.hidden_dim)
        self.global_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)

        if HAS_TORCH_GEOMETRIC:
            self.local_gnn = GATv2Conv(
                self.hidden_dim,
                self.hidden_dim // self.num_heads,
                heads=self.num_heads,
                concat=True,
                edge_dim=self.hidden_dim,
            )
        else:
            self.local_gnn = None

        self.local_self = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.local_neighbor = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.local_edge = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

    def _fallback_local_message(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        num_nodes = int(h.shape[0])
        if num_nodes == 0:
            return h

        update = self.local_self(h)
        if edge_index.numel() <= 0:
            return update

        src = edge_index[0].long().clamp(0, max(0, num_nodes - 1))
        dst = edge_index[1].long().clamp(0, max(0, num_nodes - 1))
        messages = self.local_neighbor(h[src])
        if edge_attr is not None:
            messages = messages + self.local_edge(edge_attr)

        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, messages)
        degree = torch.zeros(num_nodes, 1, device=h.device, dtype=h.dtype)
        degree.index_add_(
            0,
            dst,
            torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype),
        )
        aggregated = aggregated / degree.clamp(min=1.0)
        return update + aggregated

    def forward(
        self,
        h: Tensor,
        *,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch_idx: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        local_in = self.local_norm(h)
        if self.local_gnn is not None:
            if edge_attr is not None:
                local_out = self.local_gnn(local_in, edge_index, edge_attr=edge_attr)
            else:
                local_out = self.local_gnn(local_in, edge_index)
        else:
            local_out = self._fallback_local_message(local_in, edge_index=edge_index, edge_attr=edge_attr)
        h = h + self.dropout(F.gelu(local_out))

        global_normed = self.global_norm(h)
        valid_nodes = None
        if node_mask is not None:
            if node_mask.dim() != 1 or int(node_mask.shape[0]) != int(h.shape[0]):
                raise ValueError(
                    f"GPSLayer node_mask must have shape [N] with N={int(h.shape[0])}, got {tuple(node_mask.shape)}."
                )
            valid_nodes = node_mask.to(device=h.device, dtype=torch.bool)
        if batch_idx is None:
            if valid_nodes is not None and not bool(valid_nodes.any()):
                global_out = torch.zeros_like(h)
            else:
                global_in = global_normed.unsqueeze(0)
                key_padding_mask = (~valid_nodes).unsqueeze(0) if valid_nodes is not None else None
                global_out, _ = self.global_attn(
                    global_in,
                    global_in,
                    global_in,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                global_out = global_out.squeeze(0)
                if valid_nodes is not None:
                    global_out = global_out * valid_nodes.to(dtype=global_out.dtype).unsqueeze(-1)
        else:
            if batch_idx.dim() != 1 or int(batch_idx.shape[0]) != int(h.shape[0]):
                raise ValueError(
                    f"GPSLayer batch_idx must have shape [N] with N={int(h.shape[0])}, got {tuple(batch_idx.shape)}."
                )
            assignments = batch_idx.to(device=h.device, dtype=torch.long)
            global_out = torch.zeros_like(h)
            for graph_id in torch.unique(assignments, sorted=True):
                mask = assignments == graph_id
                if not bool(torch.any(mask)):
                    continue
                seq = global_normed[mask].unsqueeze(0)
                local_valid = valid_nodes[mask] if valid_nodes is not None else None
                if local_valid is not None and not bool(local_valid.any()):
                    continue
                key_padding_mask = (~local_valid).unsqueeze(0) if local_valid is not None else None
                out, _ = self.global_attn(
                    seq,
                    seq,
                    seq,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                out = out.squeeze(0)
                if local_valid is not None:
                    out = out * local_valid.to(dtype=out.dtype).unsqueeze(-1)
                global_out[mask] = out
        h = h + self.dropout(global_out)

        h = h + self.dropout(self.ffn(self.ffn_norm(h)))
        return h


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
        self.residual_proj = (
            nn.Identity()
            if local_dim == output_dim
            else nn.Linear(local_dim, output_dim)
        )
        
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
        
        valid_mask = None
        if mask is not None:
            valid_mask = mask.to(device=attn_scores.device, dtype=torch.bool)
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0).expand(B, -1)
            if valid_mask.dim() != 2 or int(valid_mask.shape[0]) != B or int(valid_mask.shape[1]) != N:
                raise ValueError(
                    f"CrossAttentionFusion mask must have shape [B,N] with B={B}, N={N}; got {tuple(valid_mask.shape)}."
                )
            expanded_mask = valid_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~expanded_mask, -1.0e4)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        if valid_mask is not None:
            expanded_mask = valid_mask.unsqueeze(1).unsqueeze(2).to(dtype=attn_weights.dtype)
            attn_weights = attn_weights * expanded_mask
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, 1, D]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.output_dim)
        attn_output = self.output_proj(attn_output)
        
        # Add & Norm
        c_out = self.layer_norm(self.residual_proj(c_local) + attn_output)
        
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
        edge_feature_dim: Graph edge feature dimension
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
        node_features = ...  # [N, D]
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
        node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_gnn_layers: int = 3,
        gnn_type: str = 'gcn',
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        num_style_tokens: int = 6,
        style_dim: int = 128,
        use_current_node_distance_features: bool = True,
        use_reference_room_maps: bool = False,
        reference_num_tile_types: int = 44,
        reference_embedding_dim: int = 32,
        reference_hidden_dim: int = 64,
        use_rrwp_edge_features: bool = False,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.style_dim = style_dim
        self.use_reference_room_maps = bool(use_reference_room_maps)
        
        # Stream A: Local context
        self.local_encoder = LocalStreamEncoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        
        # Stream B: Global context
        self.global_encoder = GlobalStreamEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            use_current_node_distance_features=use_current_node_distance_features,
            use_rrwp_edge_features=use_rrwp_edge_features,
        )
        
        # GLOBAL STYLE TOKEN (Theme Consistency)
        # Embeds the repo's canonical sector themes:
        # fire, water, ice, forest, shadow, spirit.
        # Injected into cross-attention to anchor all room generations to a
        # consistent visual style, preventing "telephone game" drift.
        self.style_embedding = nn.Embedding(
            num_embeddings=num_style_tokens,
            embedding_dim=style_dim
        )
        # Initialize with small values
        nn.init.normal_(self.style_embedding.weight, mean=0.0, std=0.02)
        
        # Style projection to match output_dim for fusion
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.reference_room_encoder = (
            ReferenceRoomMapEncoder(
                num_tile_types=reference_num_tile_types,
                embedding_dim=reference_embedding_dim,
                hidden_dim=reference_hidden_dim,
                output_dim=output_dim,
            )
            if self.use_reference_room_maps
            else None
        )
        
        # Cross-attention fusion (now receives style-augmented global context)
        self.fusion = CrossAttentionFusion(
            local_dim=output_dim,
            global_dim=output_dim,
            output_dim=output_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Final projection keeps style and reference features disentangled for
        # ablation and gradient attribution.
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
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
        edge_rrwp: Optional[Tensor] = None,
        tpe: Optional[Tensor] = None,
        current_node_distance: Optional[Tensor] = None,
        batch_idx: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
        current_node_idx: Optional[int] = None,
        reference_room_maps: Optional[Dict[str, Optional[Tensor]]] = None,
        style_id: Optional[Tensor] = None,
        return_global_tokens: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute conditioning vector from local, global, and style context.
        
        Args:
            neighbor_latents: Dict of neighboring room latents
            boundary_constraints: [B, 8] boundary features
            position: [B, 2] room position
            node_features: [N, node_feature_dim] graph node features
            edge_index: [2, E] graph edges
            edge_features: [E, edge_feature_dim] edge type features (Phase 3A)
            tpe: [N, 8] topological positional encoding
            current_node_idx: Index of current node in graph
            reference_room_maps: Optional neighboring room-grid exemplars keyed by
                cardinal direction. Each value can be [H,W], [1,H,W], or [B,1,H,W].
            style_id: [B] or scalar - Global style token ID for dungeon theme
                     (0=fire, 1=water, 2=ice, 3=forest, 4=shadow, 5=spirit)
            return_global_tokens: When True, also return the batched graph-token
                sequence so callers do not have to re-run the global encoder.
            
        Returns:
            Conditioning vector [B, output_dim], optionally paired with global
            graph tokens [B, N, output_dim].
        """
        batch_size = boundary_constraints.shape[0]
        device = boundary_constraints.device
        
        # Stream A: Local context
        c_local = self.local_encoder(
            neighbor_latents,
            boundary_constraints,
            position,
        )
        
        # Stream B: Global context. When callers need both the current-room
        # summary token and the full graph-token sequence, run the global
        # encoder once and slice the current node locally.
        global_tokens = self.global_encoder(
            node_features,
            edge_index,
            edge_features=edge_features,
            edge_rrwp=edge_rrwp,
            tpe=tpe,
            current_node_distance=current_node_distance,
            batch_idx=batch_idx,
            node_mask=node_mask,
            node_idx=None,
        )

        # Ensure global tokens have proper shape for cross-attention.
        if global_tokens.dim() == 2:
            global_tokens = global_tokens.unsqueeze(0).expand(c_local.shape[0], -1, -1)

        global_mask = None
        if node_mask is not None:
            global_mask = node_mask.to(device=device, dtype=torch.bool)
            if global_mask.dim() == 1:
                global_mask = global_mask.unsqueeze(0).expand(batch_size, -1)
            elif global_mask.dim() == 2 and int(global_mask.shape[0]) == 1 and batch_size > 1:
                global_mask = global_mask.expand(batch_size, -1)
            if global_mask.dim() != 2 or int(global_mask.shape[0]) != batch_size:
                raise ValueError(
                    f"node_mask must broadcast to [B,N] with B={batch_size}; got {tuple(global_mask.shape)}."
                )

        if current_node_idx is not None:
            num_tokens = int(global_tokens.shape[1])
            if current_node_idx < 0 or current_node_idx >= num_tokens:
                raise IndexError(
                    f"current_node_idx={current_node_idx} is out of range for {num_tokens} graph tokens"
                )
            c_global = global_tokens[:, current_node_idx:current_node_idx + 1, :]
            if global_mask is not None:
                global_mask = global_mask[:, current_node_idx:current_node_idx + 1]
        else:
            c_global = global_tokens
        
        # GLOBAL STYLE TOKEN: Inject theme consistency
        if style_id is not None:
            # Convert to tensor if scalar
            if not isinstance(style_id, torch.Tensor):
                style_id = torch.tensor([style_id], dtype=torch.long, device=device)
            
            # Expand to batch size if needed
            if style_id.dim() == 0:
                style_id = style_id.unsqueeze(0).expand(batch_size)
            elif style_id.shape[0] == 1 and batch_size > 1:
                style_id = style_id.expand(batch_size)
            
            # Embed style token (fixed for entire dungeon)
            style_token = self.style_embedding(style_id)  # [B, style_dim]
            style_feat = self.style_proj(style_token)      # [B, output_dim]
        else:
            # No style specified: use zero vector
            style_feat = torch.zeros(batch_size, self.output_dim, 
                                    device=device, dtype=c_local.dtype)

        if self.reference_room_encoder is not None:
            reference_feat = self.reference_room_encoder(
                reference_room_maps,
                batch_size=batch_size,
                device=device,
                dtype=c_local.dtype,
            )
        else:
            reference_feat = torch.zeros(
                batch_size,
                self.output_dim,
                device=device,
                dtype=c_local.dtype,
            )
        
        # Cross-attention fusion (local queries global with style-augmented keys)
        c_fused = self.fusion(c_local, c_global, mask=global_mask)

        # Final projection: fuse graph-aware room context with the global style token
        # and any discrete reference-room exemplar features.
        c_combined = torch.cat([c_fused, style_feat, reference_feat], dim=-1)
        c = self.output_proj(c_combined)

        if return_global_tokens:
            return c, global_tokens
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
        edge_rrwp: Optional[Tensor] = None,
        tpe: Optional[Tensor] = None,
        current_node_distance: Optional[Tensor] = None,
        batch_idx: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode only global context (all nodes)."""
        return self.global_encoder(
            node_features,
            edge_index,
            edge_features=edge_features,
            edge_rrwp=edge_rrwp,
            tpe=tpe,
            current_node_distance=current_node_distance,
            batch_idx=batch_idx,
            node_mask=node_mask,
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
