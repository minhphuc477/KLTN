"""
H-MOLQD Block V: LogicNet - Differentiable Solvability Teacher
===============================================================

Differentiable Pathfinding for Gradient-Based Guidance.

This module implements a differentiable approximation of dungeon solvability,
allowing gradients to flow back to the diffusion model during inference
for logic-guided generation.

Mathematical Formulation:
-------------------------
Differentiable Bellman-Ford:
    d^{(k+1)}(v) = min_{u∈N(v)} [d^{(k)}(u) + c(u,v)]
    
Soft-min approximation:
    d̃^{(k+1)}(v) = -τ log Σ_u exp(-(d̃^{(k)}(u) + c(u,v))/τ)
    
Reachability Score:
    R(v) = σ(α(d_max - d(start, v)))
    where σ is sigmoid, α is temperature
    
Key-Lock Dependency:
    L_lock = Σ_doors max(0, d(key_room) - d(lock_room) + margin)

Output:
    L_logic = L_reach + λL_lock
    ∇L_logic for gradient guidance

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
# DIFFERENTIABLE OPERATIONS
# ============================================================================

def soft_min(x: Tensor, dim: int, temperature: float = 1.0) -> Tensor:
    """
    Differentiable soft-min operation.
    
    soft_min(x) = -τ * log(Σ exp(-x/τ))
    
    As τ → 0, this approaches the hard min.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce
        temperature: Softness parameter τ
        
    Returns:
        Soft minimum values
    """
    return -temperature * torch.logsumexp(-x / temperature, dim=dim)


def soft_max(x: Tensor, dim: int, temperature: float = 1.0) -> Tensor:
    """
    Differentiable soft-max operation (max, not softmax).
    
    soft_max(x) = τ * log(Σ exp(x/τ))
    """
    return temperature * torch.logsumexp(x / temperature, dim=dim)


def soft_threshold(x: Tensor, threshold: float, temperature: float = 1.0) -> Tensor:
    """
    Differentiable thresholding.
    
    Approximates: 1 if x < threshold else 0
    """
    return torch.sigmoid((threshold - x) / temperature)


# ============================================================================
# DIFFERENTIABLE PATHFINDER
# ============================================================================

class DifferentiablePathfinder(nn.Module):
    """
    Differentiable approximation of shortest path computation.
    
    Uses a soft Bellman-Ford algorithm that propagates distance estimates
    through the graph while maintaining differentiability.
    
    The key insight is to replace hard min operations with soft-min,
    allowing gradients to flow through the path computation.
    
    Args:
        num_iterations: Number of Bellman-Ford iterations (should be ≥ diameter)
        temperature: Soft-min temperature (lower = closer to hard min)
        inf_distance: Value representing infinity
    """
    
    def __init__(
        self,
        num_iterations: int = 20,
        temperature: float = 0.1,
        inf_distance: float = 100.0,
    ):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.inf_distance = inf_distance
    
    def forward(
        self,
        adjacency: Tensor,
        edge_weights: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        """
        Compute differentiable shortest distances from sources.
        
        Args:
            adjacency: [N, N] adjacency matrix (1 = connected)
            edge_weights: [N, N] edge costs (higher = harder to traverse)
            source_mask: [N] binary mask for source nodes
            
        Returns:
            distances: [N] soft distances from nearest source
        """
        N = adjacency.shape[0]
        device = adjacency.device
        
        # Initialize distances
        distances = torch.where(
            source_mask.bool(),
            torch.zeros(N, device=device),
            torch.full((N,), self.inf_distance, device=device),
        )
        
        # Create effective edge weights (inf for non-edges)
        effective_weights = torch.where(
            adjacency > 0,
            edge_weights,
            torch.full_like(edge_weights, self.inf_distance),
        )
        
        # Bellman-Ford iterations
        for _ in range(self.num_iterations):
            # For each node, compute distance through each neighbor
            # candidate[v] = min_{u} (distances[u] + weight[u,v])
            
            # distances[u] + weight[u,v] for all u, v
            candidates = distances.unsqueeze(1) + effective_weights  # [N, N]
            
            # Soft-min over incoming edges
            new_distances = soft_min(candidates, dim=0, temperature=self.temperature)
            
            # Keep better of current and new
            distances = torch.minimum(distances, new_distances)
        
        return distances


class ConvolutionalPathfinder(nn.Module):
    """
    CNN-based differentiable pathfinder for grid-based rooms.
    
    Uses convolutions to propagate distance information across the grid,
    approximating a flood-fill pathfinding algorithm.
    
    Args:
        num_layers: Number of propagation layers
        hidden_dim: Hidden channel dimension
    """
    
    def __init__(
        self,
        num_layers: int = 10,
        hidden_dim: int = 32,
        input_channels: int = 44,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Conv2d(input_channels + 1, hidden_dim, 3, padding=1)
        
        # Propagation layers
        self.prop_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, 1, 1)
    
    def forward(
        self,
        room_grid: Tensor,
        source_mask: Tensor,
        walkability: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute distance field from sources within a room.
        
        Args:
            room_grid: [B, C, H, W] semantic room grid (one-hot or logits)
            source_mask: [B, 1, H, W] binary source mask
            walkability: [B, 1, H, W] optional walkability mask
            
        Returns:
            distances: [B, 1, H, W] distance field
        """
        # Concatenate input with source mask
        x = torch.cat([room_grid, source_mask], dim=1)
        
        # Project to hidden
        h = self.input_proj(x)
        
        # Propagation layers with residual connections
        for layer in self.prop_layers:
            h = h + layer(h)
        
        # Output distance field
        distances = self.output_proj(h)
        
        # Apply walkability mask if provided
        if walkability is not None:
            distances = distances + (1 - walkability) * 100.0
        
        return distances


# ============================================================================
# REACHABILITY SCORER
# ============================================================================

class ReachabilityScorer(nn.Module):
    """
    Computes differentiable reachability scores for dungeon rooms.
    
    A room is "reachable" if there exists a valid path from the start
    that satisfies all key-lock dependencies.
    
    Score formulation:
        R(v) = σ(α(d_max - d(v)))
        
    where d(v) is the distance from start to v,
    d_max is the maximum acceptable distance,
    and α controls the sharpness.
    
    Args:
        max_distance: Maximum expected distance
        temperature: Sharpness of sigmoid
    """
    
    def __init__(
        self,
        max_distance: float = 50.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.max_distance = max_distance
        self.temperature = temperature
    
    def forward(
        self,
        distances: Tensor,
        target_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute reachability scores.
        
        Args:
            distances: [N] or [B, N] distances from start
            target_mask: [N] or [B, N] mask for target nodes (optional)
            
        Returns:
            scores: [N] or [B, N] reachability scores in [0, 1]
            loss: Scalar loss (1 - mean reachability of targets)
        """
        # Compute reachability scores
        scores = torch.sigmoid(
            (self.max_distance - distances) / self.temperature
        )
        
        # Compute loss
        if target_mask is not None:
            # Focus on target nodes
            target_scores = scores * target_mask
            num_targets = target_mask.sum() + 1e-6
            mean_reachability = target_scores.sum() / num_targets
        else:
            mean_reachability = scores.mean()
        
        # Loss: want high reachability
        loss = 1.0 - mean_reachability
        
        return scores, loss


# ============================================================================
# KEY-LOCK DEPENDENCY CHECKER
# ============================================================================

class KeyLockChecker(nn.Module):
    """
    Verifies key-lock dependencies are satisfiable.
    
    For each locked door, checks that the key room is reachable
    before the door needs to be opened.
    
    Loss formulation:
        L_lock = Σ_doors max(0, d(key) - d(lock) + margin)
        
    This penalizes configurations where keys are farther than their doors.
    
    Args:
        margin: Required distance margin between key and door
        temperature: Soft-max temperature
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        distances: Tensor,
        key_nodes: Tensor,
        lock_nodes: Tensor,
        key_lock_pairs: List[Tuple[int, int]],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Check key-lock dependencies.
        
        Args:
            distances: [N] distances from start
            key_nodes: [N] binary mask of key-containing nodes
            lock_nodes: [N] binary mask of locked door nodes
            key_lock_pairs: List of (key_node_idx, lock_node_idx) pairs
            
        Returns:
            loss: Scalar dependency violation loss
            info: Dict with per-pair violation info
        """
        violations = []
        
        for key_idx, lock_idx in key_lock_pairs:
            key_dist = distances[key_idx]
            lock_dist = distances[lock_idx]
            
            # Violation if key is farther than lock + margin
            violation = F.relu(key_dist - lock_dist + self.margin)
            violations.append(violation)
        
        if violations:
            loss = torch.stack(violations).mean()
        else:
            loss = torch.tensor(0.0, device=distances.device)
        
        info = {
            'num_violations': sum(1 for v in violations if v > 0),
            'total_violation': loss,
        }
        
        return loss, info


# ============================================================================
# TILE CLASSIFIER
# ============================================================================

class TileClassifier(nn.Module):
    """
    Classifies latent features to semantic tile predictions.
    
    Used to convert VQ-VAE latents to soft tile predictions
    for differentiable pathfinding.
    
    Args:
        latent_dim: Input latent dimension
        num_classes: Number of tile classes
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 44,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
    
    def forward(self, z: Tensor) -> Tensor:
        """
        Classify latent to tile logits.
        
        Args:
            z: Latent tensor [B, D, H, W]
            
        Returns:
            Tile logits [B, num_classes, H, W]
        """
        return self.classifier(z)


class WalkabilityPredictor(nn.Module):
    """
    Predicts walkability mask from tile logits.
    
    Walkable tiles: FLOOR, DOOR_*, STAIR
    Non-walkable: WALL, BLOCK, VOID
    """
    
    # Walkable tile IDs (from definitions.py)
    WALKABLE_IDS = [1, 10, 11, 12, 13, 14, 15, 42]  # FLOOR, DOORs, STAIR
    
    def __init__(self, num_classes: int = 44):
        super().__init__()
        
        # Create walkability weight vector
        walkability = torch.zeros(num_classes)
        for tid in self.WALKABLE_IDS:
            if tid < num_classes:
                walkability[tid] = 1.0
        
        self.register_buffer('walkability_weights', walkability)
    
    def forward(self, tile_logits: Tensor) -> Tensor:
        """
        Predict walkability from tile logits.
        
        Args:
            tile_logits: [B, C, H, W] tile class logits
            
        Returns:
            walkability: [B, 1, H, W] soft walkability mask
        """
        # Soft assignment via softmax
        probs = F.softmax(tile_logits, dim=1)
        
        # Weighted sum with walkability
        walkability = torch.einsum(
            'bchw,c->bhw',
            probs,
            self.walkability_weights,
        ).unsqueeze(1)
        
        return walkability


# ============================================================================
# LOGIC NET (Main Module)
# ============================================================================

class LogicNet(nn.Module):
    """
    LogicNet: Differentiable Solvability Approximation for H-MOLQD Block V.
    
    Provides differentiable loss and gradients for dungeon solvability,
    enabling gradient-guided generation during diffusion sampling.
    
    Components:
    1. Tile Classifier: Convert latents to tile predictions
    2. Walkability Predictor: Determine traversable regions
    3. Differentiable Pathfinder: Compute soft distances
    4. Reachability Scorer: Score room accessibility
    5. Key-Lock Checker: Verify item dependencies
    
    Output:
        L_logic = λ_reach * L_reach + λ_lock * L_lock
        ∇L_logic w.r.t. input latents
    
    Args:
        latent_dim: VQ-VAE latent dimension
        num_classes: Number of tile classes
        num_iterations: Pathfinder iterations
        temperature: Soft-min temperature
        reach_weight: Weight for reachability loss
        lock_weight: Weight for key-lock loss
    
    Usage:
        logic_net = LogicNet(latent_dim=64)
        
        # Forward pass
        loss, info = logic_net(z_latent, graph_data)
        
        # Compute gradient for guidance
        grad = torch.autograd.grad(loss, z_latent)[0]
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 44,
        num_iterations: int = 20,
        temperature: float = 0.1,
        reach_weight: float = 1.0,
        lock_weight: float = 0.5,
        # --- Phase 1D: Temperature annealing (Jang et al., 2017) ---
        initial_temperature: float = 1.0,
        final_temperature: float = 0.05,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.reach_weight = reach_weight
        self.lock_weight = lock_weight
        
        # --- Phase 1D: Temperature annealing state ---
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.register_buffer('current_temperature', torch.tensor(initial_temperature))
        
        # Tile classification
        self.tile_classifier = TileClassifier(
            latent_dim=latent_dim,
            num_classes=num_classes,
        )
        
        # Walkability prediction
        self.walkability = WalkabilityPredictor(num_classes)
        
        # Grid-level pathfinder
        self.grid_pathfinder = ConvolutionalPathfinder(
            num_layers=10,
            hidden_dim=32,
            input_channels=num_classes,
        )
        
        # Graph-level pathfinder
        self.graph_pathfinder = DifferentiablePathfinder(
            num_iterations=num_iterations,
            temperature=temperature,
        )
        
        # Reachability scoring
        self.reachability = ReachabilityScorer(
            max_distance=50.0,
            temperature=temperature,
        )
        
        # Key-lock checking
        self.key_lock = KeyLockChecker(
            margin=1.0,
            temperature=temperature,
        )
    
    def update_temperature(self, progress: float):
        """
        Anneal soft-min temperature during training.
        
        Uses exponential decay from initial_temperature → final_temperature.
        High temperature (start): smooth gradients, easy optimization.
        Low temperature (end): sharp soft-min ≈ true shortest path.
        
        Follows Gumbel-Softmax annealing (Jang et al., 2017; Maddison et al., 2017).
        
        Args:
            progress: Training progress in [0, 1] (0=start, 1=end)
        """
        progress = max(0.0, min(1.0, progress))
        tau = self.initial_temperature * (
            self.final_temperature / self.initial_temperature
        ) ** progress
        
        self.current_temperature.fill_(tau)
        
        # Propagate to sub-modules
        self.graph_pathfinder.temperature = tau
        self.reachability.temperature = tau
        self.key_lock.temperature = tau
    
    def forward(
        self,
        z: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Compute solvability loss for latent codes.
        
        Args:
            z: Latent codes [B, D, H, W]
            graph_data: Optional dict with:
                - 'adjacency': [N, N] room adjacency
                - 'edge_weights': [N, N] traversal costs
                - 'start_idx': Index of start room
                - 'target_idx': Index of target room
                - 'key_lock_pairs': List of (key_idx, lock_idx)
            
        Returns:
            loss: Scalar solvability loss
            info: Dict with detailed metrics
        """
        B = z.shape[0]
        device = z.device
        
        info = {}
        
        # 1. Classify tiles
        tile_logits = self.tile_classifier(z)
        info['tile_logits'] = tile_logits
        
        # 2. Predict walkability
        walkability = self.walkability(tile_logits)
        info['walkability'] = walkability
        
        # 3. Compute within-room pathability
        # Create source mask at typical door positions
        source_mask = self._create_door_source_mask(z.shape, device)
        
        grid_distances = self.grid_pathfinder(
            tile_logits,
            source_mask,
            walkability,
        )
        info['grid_distances'] = grid_distances
        
        # Grid-level reachability: can we traverse the room?
        grid_reach_scores, grid_reach_loss = self.reachability(
            grid_distances.view(B, -1),
            (walkability > 0.5).view(B, -1).float(),
        )
        info['grid_reachability'] = grid_reach_scores.mean()
        
        # 4. Graph-level pathfinding (if graph data provided)
        graph_reach_loss = torch.tensor(0.0, device=device)
        lock_loss = torch.tensor(0.0, device=device)
        
        if graph_data is not None:
            adjacency = graph_data.get('adjacency')
            edge_weights = graph_data.get('edge_weights')
            start_idx = graph_data.get('start_idx', 0)
            target_idx = graph_data.get('target_idx')
            key_lock_pairs = graph_data.get('key_lock_pairs', [])
            
            if adjacency is not None and edge_weights is not None:
                N = adjacency.shape[0]
                
                # Create source mask for start node
                source_mask = torch.zeros(N, device=device)
                source_mask[start_idx] = 1.0
                
                # Compute graph distances
                graph_distances = self.graph_pathfinder(
                    adjacency,
                    edge_weights,
                    source_mask,
                )
                info['graph_distances'] = graph_distances
                
                # Target reachability
                if target_idx is not None:
                    target_mask = torch.zeros(N, device=device)
                    target_mask[target_idx] = 1.0
                    _, graph_reach_loss = self.reachability(
                        graph_distances,
                        target_mask,
                    )
                
                # Key-lock dependencies
                if key_lock_pairs:
                    key_mask = torch.zeros(N, device=device)
                    lock_mask = torch.zeros(N, device=device)
                    for k, l in key_lock_pairs:
                        key_mask[k] = 1.0
                        lock_mask[l] = 1.0
                    
                    lock_loss, lock_info = self.key_lock(
                        graph_distances,
                        key_mask,
                        lock_mask,
                        key_lock_pairs,
                    )
                    info.update(lock_info)
        
        # 5. Combine losses
        loss = (
            self.reach_weight * (grid_reach_loss + graph_reach_loss)
            + self.lock_weight * lock_loss
        )
        
        info['grid_reach_loss'] = grid_reach_loss
        info['graph_reach_loss'] = graph_reach_loss
        info['lock_loss'] = lock_loss
        info['total_loss'] = loss
        
        return loss, info
    
    def _create_door_source_mask(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
    ) -> Tensor:
        """Create source mask at door positions for grid pathfinding."""
        B, _, H, W = shape
        mask = torch.zeros(B, 1, H, W, device=device)
        
        # Place sources at typical door positions
        # North door (top center)
        if H > 0:
            mask[:, :, 0, W//2-1:W//2+2] = 1.0
        # South door (bottom center)
        if H > 1:
            mask[:, :, -1, W//2-1:W//2+2] = 1.0
        # East door (right middle)
        if W > 0:
            mask[:, :, H//2-1:H//2+2, -1] = 1.0
        # West door (left middle)
        if W > 1:
            mask[:, :, H//2-1:H//2+2, 0] = 1.0
        
        return mask
    
    def get_gradient(
        self,
        z: Tensor,
        graph_data: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Compute gradient of solvability loss w.r.t. latents.
        
        Args:
            z: Latent codes [B, D, H, W]
            graph_data: Optional graph information
            
        Returns:
            Gradient tensor [B, D, H, W]
        """
        z_grad = z.detach().requires_grad_(True)
        loss, _ = self.forward(z_grad, graph_data)
        
        grad = torch.autograd.grad(
            loss,
            z_grad,
            create_graph=False,
            retain_graph=False,
        )[0]
        
        return grad


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_logic_net(
    latent_dim: int = 64,
    num_classes: int = 44,
    **kwargs,
) -> LogicNet:
    """
    Create a LogicNet module.
    
    Args:
        latent_dim: VQ-VAE latent dimension
        num_classes: Number of tile classes
        **kwargs: Additional arguments
        
    Returns:
        LogicNet instance
    """
    return LogicNet(
        latent_dim=latent_dim,
        num_classes=num_classes,
        **kwargs,
    )


def build_graph_data(
    adjacency: Tensor,
    edge_weights: Optional[Tensor] = None,
    start_idx: int = 0,
    target_idx: Optional[int] = None,
    key_lock_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Build graph_data dict for LogicNet.
    
    Args:
        adjacency: [N, N] adjacency matrix
        edge_weights: [N, N] edge costs (defaults to 1s)
        start_idx: Start node index
        target_idx: Target node index
        key_lock_pairs: List of (key, lock) node pairs
        
    Returns:
        Dict for LogicNet.forward()
    """
    if edge_weights is None:
        edge_weights = adjacency.float()
    
    return {
        'adjacency': adjacency,
        'edge_weights': edge_weights,
        'start_idx': start_idx,
        'target_idx': target_idx,
        'key_lock_pairs': key_lock_pairs or [],
    }
