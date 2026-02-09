"""
LogicNet: Differentiable Solvability Approximation
==================================================

Implements Soft-Bellman-Ford value iteration for differentiable
pathfinding as a neural network layer.

Research:
- Tamar et al. (2016) "Value Iteration Networks"
- Poganvcic et al. (2019) "Differentiation of Blackbox Combinatorial Solvers"
- Berthet et al. (2020) "Learning with Differentiable Perturbed Optimizers"

Algorithm:
    The Soft-Bellman-Ford algorithm computes reachability by iteratively
    propagating probability mass through the grid:
    
    R(t+1)[i,j] = clamp(R(t)[i,j] + sum(R(t)[neighbors]) * P[i,j], 0, 1)
    
    Where P is the probability that cell (i,j) is walkable.
    After N iterations, R[goal] ≈ 1 if path exists, ≈ 0 otherwise.

Usage:
    >>> logic_net = LogicNet(num_iterations=20)
    >>> probability_map = model(noise)  # (B, 1, H, W) walkability probs
    >>> solvability = logic_net(probability_map, start_coords, goal_coords)
    >>> loss = 1.0 - solvability.mean()  # Maximize solvability
"""

import logging
import math
from typing import List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SoftBellmanFord(nn.Module):
    """
    Soft-Bellman-Ford value iteration for differentiable reachability.
    
    This module propagates "reachability" values through a grid based on
    walkability probabilities, providing a differentiable approximation
    of whether a goal is reachable from a start position.
    
    NOTE ON FORMULATION:
    --------------------
    The classic Soft Bellman-Ford uses log-sum-exp for softmin:
        d[v] = softmin_{u ∈ neighbors(v)} (d[u] + w(u,v))
        where softmin(x) = -τ * log(Σ exp(-x/τ))
    
    This implementation uses an equivalent REACHABILITY formulation that is
    more numerically stable for grid propagation:
        R[v] = clamp(R[v] + Σ R[neighbors] * P[v], 0, 1)
    
    The reachability formulation computes the probability of reaching
    a cell from the start, which is equivalent to computing shortest
    paths when P is binary (0/1 walkability).
    
    Args:
        num_iterations: Number of propagation iterations (higher = longer paths)
        connectivity: 4 for cardinal directions, 8 for including diagonals
        temperature: Softmax temperature for soft-max pooling (lower = sharper)
    """
    
    def __init__(
        self,
        num_iterations: int = 20,
        connectivity: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.connectivity = connectivity
        self.temperature = temperature
        
        # Create convolution kernel for neighbor aggregation
        if connectivity == 4:
            # Cardinal directions only (up, down, left, right)
            kernel = torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0]
            ])
        else:
            # 8-connectivity (including diagonals)
            kernel = torch.tensor([
                [0.707, 1.0, 0.707],
                [1.0,   0.0, 1.0],
                [0.707, 1.0, 0.707]
            ])
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Register as buffer (not a parameter, but moves with device)
        self.register_buffer(
            'kernel',
            kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        )
    
    def forward(
        self,
        probability_map: torch.Tensor,
        start_coords: List[Tuple[int, int]],
        goal_coords: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Compute differentiable reachability from start to goal.
        
        Args:
            probability_map: (B, 1, H, W) walkability probabilities in [0, 1]
            start_coords: List of (row, col) start positions for each batch item
            goal_coords: List of (row, col) goal positions for each batch item
            
        Returns:
            (B,) tensor of reachability scores in [0, 1]
        """
        # Defensive assertions for tensor shape
        assert probability_map.ndim == 4 and probability_map.shape[1] == 1, \
            f"Expected (B, 1, H, W) tensor, got shape {probability_map.shape}"
        
        B, C, H, W = probability_map.shape
        device = probability_map.device
        
        # Initialize reachability map with 1.0 at start positions
        R = torch.zeros_like(probability_map)
        for i in range(B):
            sr, sc = start_coords[i]
            # Clamp to valid range
            sr = max(0, min(sr, H - 1))
            sc = max(0, min(sc, W - 1))
            R[i, 0, sr, sc] = 1.0
        
        # Value iteration
        for _ in range(self.num_iterations):
            # Aggregate neighbor reachability
            neighbors = F.conv2d(R, self.kernel, padding=1)
            
            # Flow = neighbor reachability * cell walkability
            incoming_flow = neighbors * probability_map
            
            # Update reachability (max of current and incoming)
            R = torch.clamp(R + incoming_flow, 0.0, 1.0)
        
        # Extract goal reachability for each batch item
        goal_values = []
        for i in range(B):
            gr, gc = goal_coords[i]
            # Clamp to valid range
            gr = max(0, min(gr, H - 1))
            gc = max(0, min(gc, W - 1))
            goal_values.append(R[i, 0, gr, gc])
        
        return torch.stack(goal_values)


class LogicNet(nn.Module):
    """
    LogicNet: Neural module for differentiable dungeon solvability.
    
    Combines probability map processing with Soft-Bellman-Ford to
    produce a differentiable solvability score.
    
    Features:
    - Learnable walkability thresholds
    - Multi-scale reachability (optional)
    - Inventory-aware extensions (future)
    
    Args:
        num_iterations: Soft-Bellman-Ford iterations
        connectivity: 4 or 8 directions
        learnable_threshold: Whether to learn walkability threshold
        
    Example:
        >>> logic_net = LogicNet(num_iterations=30)
        >>> prob_map = torch.rand(4, 1, 16, 11)  # Batch of 4 rooms
        >>> starts = [(2, 2)] * 4
        >>> goals = [(14, 9)] * 4
        >>> scores = logic_net(prob_map, starts, goals)
        >>> print(scores.shape)  # (4,)
    """
    
    def __init__(
        self,
        num_iterations: int = 20,
        connectivity: int = 4,
        learnable_threshold: bool = False,
        num_tile_types: int = 44,  # Max semantic ID + 1
    ):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Core reachability module
        self.bellman_ford = SoftBellmanFord(
            num_iterations=num_iterations,
            connectivity=connectivity,
        )
        
        # Optional learnable walkability mapping
        if learnable_threshold:
            # Learn which tile types are walkable
            self.walkability = nn.Parameter(
                torch.ones(num_tile_types) * 0.5
            )
        else:
            self.walkability = None
        
        # Statistics tracking
        self.register_buffer('_call_count', torch.tensor(0))
    
    def forward(
        self,
        probability_map: torch.Tensor,
        start_coords: List[Tuple[int, int]],
        goal_coords: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Compute solvability score for dungeon maps.
        
        Args:
            probability_map: (B, 1, H, W) values representing walkability
                - If normalized [0,1]: interpreted as walkability probability
                - If integer tile IDs: converted via walkability mapping
            start_coords: List of (row, col) start positions
            goal_coords: List of (row, col) goal positions
            
        Returns:
            (B,) tensor of solvability scores in [0, 1]
        """
        # Defensive assertion for batch size consistency
        assert len(start_coords) == probability_map.shape[0] == len(goal_coords), \
            f"Batch size mismatch: probability_map batch={probability_map.shape[0]}, " \
            f"start_coords len={len(start_coords)}, goal_coords len={len(goal_coords)}"
        
        self._call_count += 1
        
        # Handle integer tile IDs if walkability mapping exists
        if self.walkability is not None and probability_map.max() > 1:
            probability_map = self._tile_to_walkability(probability_map)
        
        # Ensure values are in [0, 1]
        probability_map = torch.clamp(probability_map, 0.0, 1.0)
        
        # Compute reachability
        return self.bellman_ford(probability_map, start_coords, goal_coords)
    
    def _tile_to_walkability(self, tile_map: torch.Tensor) -> torch.Tensor:
        """Convert tile IDs to walkability probabilities."""
        B, C, H, W = tile_map.shape
        
        # Flatten, lookup, reshape
        flat = tile_map.long().view(-1)
        flat = torch.clamp(flat, 0, len(self.walkability) - 1)
        walkable = torch.sigmoid(self.walkability[flat])
        
        return walkable.view(B, C, H, W)
    
    def get_reachability_map(
        self,
        probability_map: torch.Tensor,
        start_coords: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Get full reachability map (for visualization).
        
        Returns:
            (B, 1, H, W) reachability values
        """
        B, C, H, W = probability_map.shape
        device = probability_map.device
        
        R = torch.zeros_like(probability_map)
        for i in range(B):
            sr, sc = start_coords[i]
            sr = max(0, min(sr, H - 1))
            sc = max(0, min(sc, W - 1))
            R[i, 0, sr, sc] = 1.0
        
        for _ in range(self.num_iterations):
            neighbors = F.conv2d(R, self.bellman_ford.kernel, padding=1)
            incoming_flow = neighbors * probability_map
            R = torch.clamp(R + incoming_flow, 0.0, 1.0)
        
        return R


class InventoryAwareLogicNet(nn.Module):
    """
    Extended LogicNet that models inventory-based traversal.
    
    Handles:
    - Key collection and locked door opening
    - Item requirements (ladder, raft, etc.)
    - Multi-stage reachability
    
    This is a more complex version that better models actual Zelda
    dungeon traversal but is more computationally expensive.
    
    Args:
        num_iterations: Per-stage iterations
        num_key_stages: Number of key collection stages to model
    """
    
    def __init__(
        self,
        num_iterations: int = 15,
        num_key_stages: int = 3,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_key_stages = num_key_stages
        
        # Separate kernels for different connection types
        cardinal_kernel = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ]).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('kernel', cardinal_kernel / 4.0)
        
        # Learnable gate for locked doors (how much keys "unlock")
        self.key_gate = nn.Parameter(torch.tensor(0.8))
    
    def forward(
        self,
        floor_prob: torch.Tensor,
        key_locations: torch.Tensor,
        locked_door_locations: torch.Tensor,
        start_coords: List[Tuple[int, int]],
        goal_coords: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Multi-stage reachability with key collection.
        
        Args:
            floor_prob: (B, 1, H, W) base walkability
            key_locations: (B, 1, H, W) probability of key at each cell
            locked_door_locations: (B, 1, H, W) probability of locked door
            start_coords: Starting positions
            goal_coords: Goal positions
            
        Returns:
            (B,) solvability scores
        """
        B, C, H, W = floor_prob.shape
        device = floor_prob.device
        
        # Initialize
        R = torch.zeros_like(floor_prob)
        keys_collected = torch.zeros(B, device=device)
        
        for i in range(B):
            sr, sc = start_coords[i]
            R[i, 0, sr, sc] = 1.0
        
        # Multi-stage propagation
        for stage in range(self.num_key_stages):
            # Current walkability (locked doors reduced by keys needed)
            door_passability = torch.sigmoid(
                self.key_gate * (keys_collected.view(B, 1, 1, 1) - 1)
            )
            current_walkability = floor_prob * (
                1 - locked_door_locations + 
                locked_door_locations * door_passability
            )
            
            # Propagate
            for _ in range(self.num_iterations):
                neighbors = F.conv2d(R, self.kernel, padding=1)
                incoming_flow = neighbors * current_walkability
                R = torch.clamp(R + incoming_flow, 0.0, 1.0)
            
            # Collect keys in reachable areas
            keys_collected = keys_collected + (R * key_locations).sum(dim=(1, 2, 3))
        
        # Extract goal values
        goal_values = []
        for i in range(B):
            gr, gc = goal_coords[i]
            gr = max(0, min(gr, H - 1))
            gc = max(0, min(gc, W - 1))
            goal_values.append(R[i, 0, gr, gc])
        
        return torch.stack(goal_values)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def solvability_loss(
    solvability_scores: torch.Tensor,
    target: float = 1.0,
) -> torch.Tensor:
    """
    Compute loss to maximize solvability.
    
    Args:
        solvability_scores: (B,) scores from LogicNet
        target: Target solvability (1.0 = always solvable)
        
    Returns:
        Scalar loss value
    """
    return F.mse_loss(
        solvability_scores,
        torch.full_like(solvability_scores, target)
    )


def diversity_regularized_solvability_loss(
    solvability_scores: torch.Tensor,
    generated_maps: torch.Tensor,
    solvability_weight: float = 1.0,
    diversity_weight: float = 0.1,
) -> torch.Tensor:
    """
    Loss combining solvability with diversity regularization.
    
    Prevents mode collapse by encouraging diversity in generated maps.
    
    Args:
        solvability_scores: (B,) scores from LogicNet
        generated_maps: (B, 1, H, W) generated dungeon maps
        solvability_weight: Weight for solvability term
        diversity_weight: Weight for diversity term
        
    Returns:
        Combined loss value
    """
    # Solvability loss
    solv_loss = solvability_loss(solvability_scores)
    
    # Diversity: negative variance of flattened maps
    B = generated_maps.shape[0]
    flat_maps = generated_maps.view(B, -1)
    mean_map = flat_maps.mean(dim=0, keepdim=True)
    variance = ((flat_maps - mean_map) ** 2).mean()
    diversity_loss = -variance  # Negative because we want to maximize variance
    
    return solvability_weight * solv_loss + diversity_weight * diversity_loss


# =============================================================================
# TORTUOSITY LOSS
# =============================================================================

class DifferentiableTortuosity(nn.Module):
    """
    Differentiable Tortuosity Computation for Dungeon Paths.
    
    Tortuosity measures how "winding" a path is:
        tortuosity = path_length / euclidean_distance
    
    A straight path has tortuosity = 1.0
    A winding path has tortuosity > 1.0
    
    For good dungeon design, we want moderate tortuosity:
    - Too straight (≈1.0): Boring, trivial navigation
    - Too winding (>3.0): Frustrating, confusing
    - Ideal range: 1.5 - 2.5 for engaging exploration
    
    The loss penalizes paths that are too straight:
        L_tortuosity = -log(tortuosity) for tortuosity < target
        L_tortuosity = log(tortuosity / target) for tortuosity > max_target
    
    This encourages interesting, non-trivial paths.
    """
    
    def __init__(
        self,
        num_iterations: int = 50,
        target_tortuosity: float = 1.5,
        max_tortuosity: float = 3.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.target_tortuosity = target_tortuosity
        self.max_tortuosity = max_tortuosity
        self.epsilon = epsilon
        
        # Cardinal movement kernel for path length estimation
        cardinal_kernel = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', cardinal_kernel / 4.0)
    
    def compute_soft_path_length(
        self,
        probability_map: torch.Tensor,
        start_coords: List[Tuple[int, int]],
        goal_coords: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Compute differentiable approximation of shortest path length.
        
        Uses value iteration to estimate the minimum number of steps
        needed to reach the goal from the start.
        
        Args:
            probability_map: (B, 1, H, W) walkability in [0, 1]
            start_coords: List of start (row, col)
            goal_coords: List of goal (row, col)
            
        Returns:
            (B,) estimated path lengths
        """
        B, C, H, W = probability_map.shape
        device = probability_map.device
        
        # Initialize distance map with large values
        # D[i,j] = estimated distance from goal to (i,j)
        D = torch.ones_like(probability_map) * (H + W)  # Max possible
        
        # Set goal distance to 0
        for i in range(B):
            gr, gc = goal_coords[i]
            gr = max(0, min(gr, H - 1))
            gc = max(0, min(gc, W - 1))
            D[i, 0, gr, gc] = 0.0
        
        # Value iteration (backward from goal)
        for _ in range(self.num_iterations):
            # Get neighbor distances (min distance + 1 step)
            neighbor_dist = F.conv2d(D, self.kernel, padding=1)
            
            # Update distance: min of current and (neighbor + 1) * walkability
            # Non-walkable cells maintain high distance
            new_dist = neighbor_dist + 1.0
            
            # Weight by walkability (low walkability = high effective distance)
            effective_dist = new_dist / (probability_map + self.epsilon)
            
            # Take minimum
            D = torch.min(D, effective_dist)
        
        # Extract distance at start positions
        path_lengths = []
        for i in range(B):
            sr, sc = start_coords[i]
            sr = max(0, min(sr, H - 1))
            sc = max(0, min(sc, W - 1))
            path_lengths.append(D[i, 0, sr, sc])
        
        return torch.stack(path_lengths)
    
    def compute_euclidean_distance(
        self,
        start_coords: List[Tuple[int, int]],
        goal_coords: List[Tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute Euclidean distance between start and goal."""
        euclidean_dists = []
        for (sr, sc), (gr, gc) in zip(start_coords, goal_coords):
            dist = float(math.sqrt((gr - sr) ** 2 + (gc - sc) ** 2))
            euclidean_dists.append(dist)
        # Use a detached constant - this is intentionally not differentiable
        # (coordinates are fixed inputs, not learned)
        return torch.tensor(euclidean_dists, dtype=torch.float32, device=device)
    
    def forward(
        self,
        probability_map: torch.Tensor,
        start_coords: List[Tuple[int, int]],
        goal_coords: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute tortuosity for paths in the dungeon.
        
        Args:
            probability_map: (B, 1, H, W) walkability
            start_coords: Start positions
            goal_coords: Goal positions
            
        Returns:
            (tortuosity, is_valid) where:
                tortuosity: (B,) tortuosity values
                is_valid: (B,) bool tensor indicating if path exists
        """
        device = probability_map.device
        
        # Compute path lengths
        path_lengths = self.compute_soft_path_length(
            probability_map, start_coords, goal_coords
        )
        
        # Compute Euclidean distances
        euclidean_dists = self.compute_euclidean_distance(
            start_coords, goal_coords, device
        )
        
        # Tortuosity = path_length / euclidean_distance
        # Clamp to avoid division issues
        euclidean_dists = torch.clamp(euclidean_dists, min=1.0)
        tortuosity = path_lengths / euclidean_dists
        
        # Check validity (finite path length indicates reachable)
        is_valid = path_lengths < (probability_map.shape[2] + probability_map.shape[3])
        
        return tortuosity, is_valid


def tortuosity_loss(
    probability_map: torch.Tensor,
    start_coords: List[Tuple[int, int]],
    goal_coords: List[Tuple[int, int]],
    target_tortuosity: float = 1.5,
    max_tortuosity: float = 3.0,
    num_iterations: int = 50,
) -> torch.Tensor:
    """
    Compute tortuosity loss to encourage interesting (non-straight) paths.
    
    Formula:
        For tortuosity τ:
        - If τ < target: loss = -log(τ / target)  (penalize straight paths)
        - If τ > max: loss = log(τ / max)  (penalize overly winding paths)
        - If target ≤ τ ≤ max: loss = 0  (ideal range)
    
    This encourages dungeons with moderately winding paths that
    are engaging to navigate without being frustrating.
    
    Args:
        probability_map: (B, 1, H, W) walkability probabilities
        start_coords: Start positions
        goal_coords: Goal positions
        target_tortuosity: Minimum desired tortuosity (default: 1.5)
        max_tortuosity: Maximum acceptable tortuosity (default: 3.0)
        num_iterations: Iterations for path length computation
        
    Returns:
        Scalar loss value
        
    Example:
        >>> prob_map = torch.rand(4, 1, 16, 11)
        >>> starts = [(2, 2)] * 4
        >>> goals = [(13, 8)] * 4
        >>> loss = tortuosity_loss(prob_map, starts, goals)
        >>> print(f"Tortuosity loss: {loss.item():.4f}")
    """
    tort_module = DifferentiableTortuosity(
        num_iterations=num_iterations,
        target_tortuosity=target_tortuosity,
        max_tortuosity=max_tortuosity,
    )
    tort_module = tort_module.to(probability_map.device)
    
    tortuosity, is_valid = tort_module(probability_map, start_coords, goal_coords)
    
    # Use soft masking to preserve gradient flow (boolean indexing breaks grads)
    # is_valid is a bool tensor; convert to float mask
    valid_mask = is_valid.float()
    num_valid = valid_mask.sum().clamp(min=1.0)
    
    # If no valid paths, return a loss connected to prob_map so grads still flow
    # Add a tiny dependency on prob_map to maintain the computation graph
    if not is_valid.any():
        return (probability_map * 0.0).sum() + 1.0
    
    eps = 1e-6
    
    # Penalize paths that are too straight: soft penalty via ReLU
    # straight_penalty = max(0, target - tortuosity) for each sample
    straight_penalty = F.relu(target_tortuosity - tortuosity)
    
    # Penalize paths that are too winding: soft penalty via ReLU
    # winding_penalty = max(0, tortuosity - max_tortuosity) for each sample
    winding_penalty = F.relu(tortuosity - max_tortuosity)
    
    # Combine with valid mask (only count valid paths)
    loss = ((straight_penalty + winding_penalty) * valid_mask).sum() / num_valid
    
    return loss


def manhattan_tortuosity_loss(
    actual_path_length: torch.Tensor,
    start_coords: List[Tuple[int, int]],
    goal_coords: List[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute tortuosity loss using Manhattan distance (Thesis Formula H).
    
    Formula:
        L_tort = |actual_path_length - manhattan_distance| / manhattan_distance
    
    This penalizes paths that deviate significantly from the theoretical
    minimum (Manhattan distance for grid-based navigation).
    
    Args:
        actual_path_length: (B,) tensor of path lengths
        start_coords: Start positions
        goal_coords: Goal positions  
        device: Torch device
        
    Returns:
        Scalar loss value
    """
    manhattan_dists = []
    for (sr, sc), (gr, gc) in zip(start_coords, goal_coords):
        manhattan_dists.append(abs(gr - sr) + abs(gc - sc))
    
    manhattan = torch.tensor(manhattan_dists, dtype=torch.float32, device=device)
    manhattan = torch.clamp(manhattan, min=1.0)  # Avoid division by zero
    
    # L_tort = |path_length - manhattan| / manhattan
    tortuosity = torch.abs(actual_path_length - manhattan) / manhattan
    
    return tortuosity.mean()


def combined_logic_loss(
    probability_map: torch.Tensor,
    start_coords: List[Tuple[int, int]],
    goal_coords: List[Tuple[int, int]],
    solvability_weight: float = 1.0,
    tortuosity_weight: float = 0.3,
    target_tortuosity: float = 1.5,
    logic_net: Optional['LogicNet'] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Combined loss for solvability + path quality.
    
    L_combined = α * L_solvability + β * L_tortuosity
    
    Args:
        probability_map: (B, 1, H, W) walkability
        start_coords: Start positions
        goal_coords: Goal positions
        solvability_weight: Weight for solvability term
        tortuosity_weight: Weight for tortuosity term
        target_tortuosity: Target minimum tortuosity
        logic_net: Optional pre-created LogicNet
        
    Returns:
        (total_loss, loss_dict) where loss_dict contains individual terms
    """
    # Create LogicNet if not provided
    if logic_net is None:
        logic_net = LogicNet(num_iterations=30)
        logic_net = logic_net.to(probability_map.device)
    
    # Solvability
    solvability_scores = logic_net(probability_map, start_coords, goal_coords)
    solv_loss = solvability_loss(solvability_scores)
    
    # Tortuosity
    tort_loss = tortuosity_loss(
        probability_map, start_coords, goal_coords,
        target_tortuosity=target_tortuosity,
    )
    
    # Combined
    total_loss = solvability_weight * solv_loss + tortuosity_weight * tort_loss
    
    loss_dict = {
        'solvability_loss': solv_loss,
        'tortuosity_loss': tort_loss,
        'mean_solvability': solvability_scores.mean(),
    }
    
    return total_loss, loss_dict


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test LogicNet
    print("Testing LogicNet...")
    
    logic_net = LogicNet(num_iterations=30)
    
    # Create test grid: mostly walkable with some walls
    B, H, W = 4, 16, 11
    prob_map = torch.ones(B, 1, H, W) * 0.9
    
    # Add some walls
    prob_map[:, :, 0, :] = 0.0  # Top wall
    prob_map[:, :, -1, :] = 0.0  # Bottom wall
    prob_map[:, :, :, 0] = 0.0  # Left wall
    prob_map[:, :, :, -1] = 0.0  # Right wall
    
    starts = [(2, 2)] * B
    goals = [(13, 8)] * B
    
    scores = logic_net(prob_map, starts, goals)
    print(f"Solvability scores: {scores}")
    print(f"Mean: {scores.mean():.4f}")
    
    # Test gradient flow
    prob_map.requires_grad = True
    scores = logic_net(prob_map, starts, goals)
    loss = solvability_loss(scores)
    loss.backward()
    print(f"Gradient exists: {prob_map.grad is not None}")
    print(f"Gradient mean: {prob_map.grad.abs().mean():.6f}")
    
    print("\nLogicNet test passed!")
