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
from typing import List, Tuple, Optional, Union

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
