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

import logging
import math
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor

from src.core.definitions import (
    DOOR_POSITIONS,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNELS,
    ROOM_TOPOLOGY_DIRECTIONAL_CHANNEL_GROUPS,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
)
from src.core.perturb_and_map import perturb_and_map_distance

logger = logging.getLogger(__name__)


# Canonical graph-feature channel positions emitted by
# ``src.pipeline.graph_features.encode_*_feature_vector``. These names keep
# logic semantics explicit without changing the checkpoint feature widths.
GRAPH_NODE_KEY_FEATURE_INDEX = 1
GRAPH_NODE_GOAL_FEATURE_INDEX = 3

GRAPH_EDGE_OPEN_FEATURE_INDEX = 0
GRAPH_EDGE_KEY_LOCK_FEATURE_INDEX = 1
GRAPH_EDGE_BOMBABLE_FEATURE_INDEX = 2
GRAPH_EDGE_SOFT_LOCK_FEATURE_INDEX = 3
GRAPH_EDGE_BOSS_LOCK_FEATURE_INDEX = 4
GRAPH_EDGE_ITEM_LOCK_FEATURE_INDEX = 5
GRAPH_EDGE_STAIR_FEATURE_INDEX = 6
GRAPH_EDGE_SWITCH_FEATURE_INDEX = 7

GRAPH_EDGE_LOCK_FEATURE_INDICES = (
    GRAPH_EDGE_KEY_LOCK_FEATURE_INDEX,
    GRAPH_EDGE_BOSS_LOCK_FEATURE_INDEX,
    GRAPH_EDGE_ITEM_LOCK_FEATURE_INDEX,
)
GRAPH_EDGE_TRAVERSAL_PENALTIES = {
    GRAPH_EDGE_KEY_LOCK_FEATURE_INDEX: 1.0,
    GRAPH_EDGE_BOMBABLE_FEATURE_INDEX: 0.5,
    GRAPH_EDGE_SOFT_LOCK_FEATURE_INDEX: 0.25,
    GRAPH_EDGE_BOSS_LOCK_FEATURE_INDEX: 2.0,
    GRAPH_EDGE_ITEM_LOCK_FEATURE_INDEX: 1.0,
    GRAPH_EDGE_SWITCH_FEATURE_INDEX: 0.5,
}


class LogicGraphContractError(Exception):
    """Raised when graph metadata cannot define the requested logic objective."""

CANONICAL_LOGIC_WALKABLE_IDS = sorted(
    {
        int(SEMANTIC_PALETTE[name])
        for name in (
            "FLOOR",
            "DOOR_OPEN",
            "DOOR_LOCKED",
            "DOOR_BOMB",
            "DOOR_PUZZLE",
            "DOOR_BOSS",
            "DOOR_SOFT",
            "START",
            "TRIFORCE",
            "KEY_SMALL",
            "KEY_BOSS",
            "KEY_ITEM",
            "ITEM_MINOR",
            "ELEMENT_FLOOR",
            "STAIR",
            "ENEMY",
            "BOSS",
            "PUZZLE",
        )
        if name in SEMANTIC_PALETTE
    }
)


# ============================================================================
# DIFFERENTIABLE OPERATIONS
# ============================================================================

def soft_min(x: Tensor, dim: Optional[int] = None, temperature: float = 1.0) -> Tensor:
    """
    Differentiable soft-min operation.
    
    soft_min(x) = -τ * log(Σ exp(-x/τ))
    
    As τ -> 0, this approaches the hard min.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce
        temperature: Softness parameter τ
        
    Returns:
        Soft minimum values
    """
    reduce_dim = int(dim) if dim is not None else -1
    tau = max(float(temperature), 1e-6)
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return (-tau * torch.logsumexp((-x / tau).float(), dim=reduce_dim)).to(dtype=x.dtype)


def soft_max(x: Tensor, dim: Optional[int] = None, temperature: float = 1.0) -> Tensor:
    """
    Differentiable soft-max operation (max, not softmax).
    
    soft_max(x) = τ * log(Σ exp(x/τ))
    """
    reduce_dim = int(dim) if dim is not None else -1
    tau = max(float(temperature), 1e-6)
    return (tau * torch.logsumexp((x / tau).float(), dim=reduce_dim)).to(dtype=x.dtype)


def soft_threshold(x: Tensor, threshold: float, temperature: float = 1.0) -> Tensor:
    """
    Differentiable thresholding.
    
    Approximates: 1 if x < threshold else 0
    """
    return torch.sigmoid((threshold - x) / temperature)


def conservative_soft_min(
    x: Tensor,
    *,
    dim: int,
    temperature: float,
    finite_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Smooth minimum that does not undercut the smallest finite candidate.

    A raw ``-tau * logsumexp(-x / tau)`` is lower than the hard minimum when
    several alternatives have similar cost. Reapplying that operator inside an
    undiscounted Bellman loop creates an artificial negative cycle. Using the
    log-mean-exp over finite candidates preserves equality for duplicate paths
    and converges to the hard minimum as ``tau`` approaches zero.
    """
    tau = max(float(temperature), 1e-6)
    values = x.float()
    if finite_mask is None:
        finite_mask = torch.isfinite(values)
    else:
        finite_mask = finite_mask.to(device=values.device, dtype=torch.bool)
    masked = torch.where(finite_mask, values, torch.full_like(values, float("inf")))
    count = finite_mask.sum(dim=dim).clamp_min(1).to(dtype=values.dtype)
    result = -tau * torch.logsumexp(-masked / tau, dim=dim)
    result = result + tau * torch.log(count)
    return result.to(dtype=x.dtype)


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
        num_iterations: Number of Bellman-Ford iterations (should be >= diameter)
        temperature: Soft-min temperature (lower = closer to hard min)
        inf_distance: Value representing infinity
    """
    
    def __init__(
        self,
        num_iterations: int = 20,
        iterations: Optional[int] = None,
        temperature: float = 0.1,
        inf_distance: float = 1_000_000.0,
        full_coverage: bool = True,
        convergence_tolerance: float = 1e-5,
    ):
        super().__init__()

        if iterations is not None:
            num_iterations = int(iterations)

        self.num_iterations = num_iterations
        self.temperature = temperature
        self.inf_distance = inf_distance
        self.wall_penalty_scale = min(10.0, max(1.0, float(inf_distance) * 0.5))
        self.full_coverage = bool(full_coverage)
        self.convergence_tolerance = float(max(0.0, convergence_tolerance))

    def _iteration_limit(self, node_count: int) -> int:
        requested = int(max(1, self.num_iterations))
        if not self.full_coverage:
            return requested
        return max(requested, max(1, int(node_count) - 1))

    def _has_converged(self, previous: Tensor, current: Tensor) -> bool:
        if self.convergence_tolerance <= 0.0:
            return False
        # A Python boolean forces a device synchronization. Keep training
        # iterations fixed and deterministic; inference can stop early.
        if torch.is_grad_enabled() and bool(current.requires_grad):
            return False
        with torch.no_grad():
            delta = torch.nan_to_num(
                (current.detach().float() - previous.detach().float()).abs(),
                nan=float("inf"),
                posinf=float("inf"),
                neginf=float("inf"),
            )
            return bool(delta.max().item() <= self.convergence_tolerance)

    def _should_checkpoint_relaxation(self, *tensors: Tensor) -> bool:
        dynamic_problem_size = max(
            (
                int(t.numel())
                for t in tensors
                if isinstance(t, torch.Tensor)
            ),
            default=0,
        )
        return (
            bool(torch.is_grad_enabled())
            and (
                int(self.num_iterations) > 32
                or (self.full_coverage and dynamic_problem_size > 64)
            )
            and any(isinstance(t, torch.Tensor) and bool(t.requires_grad) for t in tensors)
        )

    def _grid_relax_step(
        self,
        dist: Tensor,
        walkability: Tensor,
        traversal_cost: Tensor,
        start: Tensor,
    ) -> Tensor:
        inf = float(self.inf_distance)
        dist = torch.nan_to_num(dist, nan=inf, posinf=inf, neginf=-inf).clamp(-inf, inf)
        up = torch.full_like(dist, inf)
        down = torch.full_like(dist, inf)
        left = torch.full_like(dist, inf)
        right = torch.full_like(dist, inf)
        up[:, 1:, :] = dist[:, :-1, :]
        down[:, :-1, :] = dist[:, 1:, :]
        left[:, :, 1:] = dist[:, :, :-1]
        right[:, :, :-1] = dist[:, :, 1:]
        wall_cost = (1.0 - walkability).clamp(0.0, 1.0).pow(2) * float(self.wall_penalty_scale)
        step_cost = traversal_cost + wall_cost
        candidates = torch.stack(
            [
                up + step_cost,
                down + step_cost,
                left + step_cost,
                right + step_cost,
            ],
            dim=0,
        ).clamp(-inf, inf)
        finite_candidates = candidates < (inf - 1e-6)
        has_finite_candidate = finite_candidates.any(dim=0)
        relaxed = conservative_soft_min(
            candidates,
            dim=0,
            temperature=max(self.temperature, 1e-4),
            finite_mask=finite_candidates,
        )
        relaxed = torch.where(has_finite_candidate, relaxed, torch.full_like(relaxed, inf))
        dist = torch.minimum(dist, relaxed).clamp(0.0, inf)
        dist = torch.where(start > 0.5, torch.zeros_like(dist), dist)
        return dist.clamp(-inf, inf)

    def _grid_relax_chunk(
        self,
        dist: Tensor,
        walkability: Tensor,
        traversal_cost: Tensor,
        start: Tensor,
        steps: int,
    ) -> Tensor:
        """Run an exact sequence of local relaxations inside one checkpoint."""
        for _ in range(int(max(1, steps))):
            dist = self._grid_relax_step(dist, walkability, traversal_cost, start)
        return dist

    def _graph_relax_step(
        self,
        distances: Tensor,
        effective_weights: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        inf = float(self.inf_distance)
        distances = torch.nan_to_num(
            distances,
            nan=inf,
            posinf=inf,
            neginf=-inf,
        ).clamp(-inf, inf)
        if distances.ndim == 1:
            candidates = (distances.unsqueeze(1) + effective_weights).clamp(-inf, inf)
            reduce_dim = 0
        elif distances.ndim == 2:
            candidates = (distances.unsqueeze(2) + effective_weights).clamp(-inf, inf)
            reduce_dim = 1
        else:
            raise ValueError(f"Graph relaxation expects distances [N] or [B,N], got {tuple(distances.shape)}.")
        has_finite_candidate = (candidates < (inf - 1e-6)).any(dim=reduce_dim)
        new_distances = conservative_soft_min(
            candidates,
            dim=reduce_dim,
            temperature=self.temperature,
            finite_mask=candidates < (inf - 1e-6),
        )
        new_distances = torch.where(
            has_finite_candidate,
            new_distances,
            torch.full_like(new_distances, inf),
        )
        distances = torch.minimum(distances, new_distances).clamp(0.0, inf)
        return torch.where(source_mask.bool(), torch.zeros_like(distances), distances)
    
    def forward(
        self,
        adjacency: Tensor,
        edge_weights: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        """
        Compute differentiable shortest distances from sources.

        Supports two explicit modes:

        1) Graph mode (Bellman-Ford over adjacency matrix):
           - adjacency: [N, N]
           - edge_weights: [N, N]
           - source_mask: [N]

        2) Grid compatibility mode (room flood-fill approximation):
           - adjacency: walkability [B, H, W]
           - edge_weights: traversal weights [B, H, W]
           - source_mask: source/start mask [B, H, W]
        
        Args:
            adjacency: Graph adjacency or grid walkability tensor
            edge_weights: Graph edge weights or grid start-mask tensor
            source_mask: Graph source mask or grid goal-mask tensor
            
        Returns:
            distances: [N] soft distances from nearest source
        """
        if not isinstance(adjacency, torch.Tensor) or not isinstance(edge_weights, torch.Tensor) or not isinstance(source_mask, torch.Tensor):
            raise TypeError("DifferentiablePathfinder.forward expects tensor inputs.")

        # Grid mode:
        #   adjacency -> walkability [B, H, W]
        #   edge_weights -> traversal weights [B, H, W]
        #   source_mask -> source/start mask [B, H, W]
        batched_graph_mode = (
            adjacency.ndim == 3
            and edge_weights.ndim == 3
            and source_mask.ndim == 2
        )
        grid_mode = (
            not batched_graph_mode
            and isinstance(adjacency, torch.Tensor)
            and isinstance(edge_weights, torch.Tensor)
            and isinstance(source_mask, torch.Tensor)
            and any(t.ndim == 3 for t in (adjacency, edge_weights, source_mask))
        )
        if grid_mode:
            if not all(t.ndim == 3 for t in (adjacency, edge_weights, source_mask)):
                raise ValueError(
                    "DifferentiablePathfinder grid mode expects adjacency, edge_weights, and "
                    "source_mask to all be rank-3 [B,H,W] tensors. "
                    f"Got adjacency={tuple(adjacency.shape)}, edge_weights={tuple(edge_weights.shape)}, "
                    f"source_mask={tuple(source_mask.shape)}. "
                    "Update callers to pass explicit batched [B,H,W] walkability/weights/source tensors."
                )
            if adjacency.shape != edge_weights.shape or adjacency.shape != source_mask.shape:
                raise ValueError(
                    f"Grid mode shape mismatch: adjacency={tuple(adjacency.shape)}, "
                    f"edge_weights={tuple(edge_weights.shape)}, source_mask={tuple(source_mask.shape)}."
                )
            walkability = adjacency.float().clamp(0.0, 1.0)
            traversal_cost = edge_weights.float().clamp_min(0.0)
            start = source_mask.float().clamp(0.0, 1.0)
            B, H, W = walkability.shape
            device = walkability.device

            dist = torch.full(
                (B, H, W),
                float(self.inf_distance),
                device=device,
                dtype=torch.float32,
            )
            dist = torch.where(start > 0.5, torch.zeros_like(dist), dist)

            # Soft Bellman-style relaxation over 4-neighborhood. Historical
            # callers passed their start mask in the second slot and a goal mask
            # in the third. The current contract uses the third tensor as the
            # source mask; this still computes a valid distance field from the
            # requested source while avoiding ambiguous argument inversion.
            iteration_limit = self._iteration_limit(H * W)
            use_checkpoint = self._should_checkpoint_relaxation(walkability, traversal_cost, start)
            if use_checkpoint:
                # Segmenting K relaxations into about sqrt(K) checkpoints keeps
                # the exact K-step result while reducing checkpoint-node depth
                # from O(K) to O(sqrt(K)).
                chunk_size = int(max(1, math.ceil(math.sqrt(iteration_limit))))
                completed = 0
                while completed < iteration_limit:
                    steps = min(chunk_size, iteration_limit - completed)
                    dist = checkpoint(
                        lambda d, w, c, s, step_count=steps: self._grid_relax_chunk(
                            d, w, c, s, step_count
                        ),
                        dist,
                        walkability,
                        traversal_cost,
                        start,
                        use_reentrant=False,
                    )
                    completed += steps
            else:
                for _ in range(iteration_limit):
                    previous = dist
                    dist = self._grid_relax_step(dist, walkability, traversal_cost, start)
                    if self._has_converged(previous, dist):
                        break

            return dist

        if batched_graph_mode:
            if adjacency.shape != edge_weights.shape:
                raise ValueError(
                    f"Batched graph mode shape mismatch: adjacency={tuple(adjacency.shape)} edge_weights={tuple(edge_weights.shape)}."
                )
            if adjacency.shape[1] != adjacency.shape[2]:
                raise ValueError(f"Batched graph adjacency must be square, got {tuple(adjacency.shape)}.")
            if source_mask.shape != adjacency.shape[:2]:
                raise ValueError(
                    f"Batched graph source_mask shape {tuple(source_mask.shape)} must match [B,N]={tuple(adjacency.shape[:2])}."
                )
            adjacency = adjacency.float()
            edge_weights = edge_weights.float()
            source_mask = source_mask.float()
            B, N, _ = adjacency.shape
            device = adjacency.device
            distances = torch.where(
                source_mask.bool(),
                torch.zeros((B, N), device=device, dtype=adjacency.dtype),
                torch.full((B, N), self.inf_distance, device=device, dtype=adjacency.dtype),
            )
            effective_weights = torch.where(
                adjacency > 0,
                edge_weights,
                torch.full_like(edge_weights, self.inf_distance),
            ).clamp(-float(self.inf_distance), float(self.inf_distance))
            use_checkpoint = self._should_checkpoint_relaxation(effective_weights, source_mask)
            for _ in range(self._iteration_limit(N)):
                previous = distances
                if use_checkpoint:
                    distances = checkpoint(
                        self._graph_relax_step,
                        distances,
                        effective_weights,
                        source_mask,
                        use_reentrant=False,
                    )
                else:
                    distances = self._graph_relax_step(distances, effective_weights, source_mask)
                if self._has_converged(previous, distances):
                    break
            return distances

        if adjacency.ndim != 2 or edge_weights.ndim != 2 or source_mask.ndim != 1:
            raise ValueError(
                "Graph mode requires adjacency [N,N], edge_weights [N,N], source_mask [N], "
                "or batched adjacency [B,N,N], edge_weights [B,N,N], source_mask [B,N]."
            )
        if adjacency.shape != edge_weights.shape:
            raise ValueError(
                f"Graph mode shape mismatch: adjacency={tuple(adjacency.shape)} edge_weights={tuple(edge_weights.shape)}."
            )
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(
                f"Graph mode adjacency must be square, got {tuple(adjacency.shape)}."
            )
        if source_mask.shape[0] != adjacency.shape[0]:
            raise ValueError(
                f"Graph mode source_mask length {source_mask.shape[0]} must equal N={adjacency.shape[0]}."
            )

        adjacency = adjacency.float()
        edge_weights = edge_weights.float()
        source_mask = source_mask.float()
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
        ).clamp(-float(self.inf_distance), float(self.inf_distance))
        
        # Bellman-Ford iterations
        use_checkpoint = self._should_checkpoint_relaxation(effective_weights, source_mask)
        for _ in range(self._iteration_limit(N)):
            previous = distances
            if use_checkpoint:
                distances = checkpoint(
                    self._graph_relax_step,
                    distances,
                    effective_weights,
                    source_mask,
                    use_reentrant=False,
                )
            else:
                distances = self._graph_relax_step(distances, effective_weights, source_mask)
            if self._has_converged(previous, distances):
                break
        
        return distances


class SparseDifferentiablePathfinder(DifferentiablePathfinder):
    """Edge-sparse Bellman-Ford relaxation for large mission graphs.

    Grid inputs retain the canonical dense implementation. Graph inputs gather
    only real edges and reduce incoming candidates by destination, reducing a
    full-coverage pass from ``O(N^3)`` dense work to ``O(N E)`` for ``N`` nodes
    and ``E`` directed edges. The conservative log-mean-exp reduction matches
    :func:`conservative_soft_min` without introducing entropy-driven negative
    cycles.
    """

    def _sparse_graph_relax_step(
        self,
        distances: Tensor,
        src: Tensor,
        dst: Tensor,
        edge_weights: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        node_count = int(distances.shape[0])
        inf = float(self.inf_distance)
        if src.numel() == 0:
            return torch.where(source_mask.bool(), torch.zeros_like(distances), distances)

        candidates = (distances.index_select(0, src) + edge_weights).clamp(-inf, inf)
        hard_min = torch.full_like(distances, inf).scatter_reduce(
            0,
            dst,
            candidates,
            reduce="amin",
            include_self=True,
        )
        finite_edge = candidates < (inf - 1e-6)
        centered = torch.where(
            finite_edge,
            candidates - hard_min.index_select(0, dst),
            torch.zeros_like(candidates),
        )
        stabilized = (
            torch.exp(-centered / max(float(self.temperature), 1e-6))
            * finite_edge.to(dtype=candidates.dtype)
        )
        exp_sum = torch.zeros_like(distances).scatter_add(0, dst, stabilized)
        counts = torch.bincount(dst[finite_edge], minlength=node_count).to(
            device=distances.device,
            dtype=distances.dtype,
        )
        has_candidate = counts > 0
        mean_exp = exp_sum / counts.clamp_min(1.0)
        relaxed = hard_min - max(float(self.temperature), 1e-6) * torch.log(
            mean_exp.clamp_min(torch.finfo(mean_exp.dtype).tiny)
        )
        relaxed = torch.where(has_candidate, relaxed, torch.full_like(relaxed, inf))
        updated = torch.minimum(distances, relaxed).clamp(0.0, inf)
        return torch.where(source_mask.bool(), torch.zeros_like(updated), updated)

    def _forward_sparse_graph(
        self,
        adjacency: Tensor,
        edge_weights: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Sparse graph adjacency must be square [N,N], got {tuple(adjacency.shape)}.")
        if edge_weights.shape != adjacency.shape:
            raise ValueError(
                f"Sparse graph weight shape {tuple(edge_weights.shape)} must match adjacency {tuple(adjacency.shape)}."
            )
        node_count = int(adjacency.shape[0])
        if source_mask.shape != (node_count,):
            raise ValueError(
                f"Sparse graph source mask must have shape ({node_count},), got {tuple(source_mask.shape)}."
            )

        adjacency = adjacency.float()
        weights = edge_weights.float()
        source = source_mask.float()
        edge_index = torch.nonzero(adjacency > 0, as_tuple=False)
        src = edge_index[:, 0].to(dtype=torch.long)
        dst = edge_index[:, 1].to(dtype=torch.long)
        sparse_weights = weights[src, dst]
        distances = torch.where(
            source.bool(),
            torch.zeros(node_count, device=adjacency.device, dtype=weights.dtype),
            torch.full((node_count,), float(self.inf_distance), device=adjacency.device, dtype=weights.dtype),
        )
        use_checkpoint = self._should_checkpoint_relaxation(sparse_weights, source)
        for _ in range(self._iteration_limit(node_count)):
            previous = distances
            if use_checkpoint:
                distances = checkpoint(
                    self._sparse_graph_relax_step,
                    distances,
                    src,
                    dst,
                    sparse_weights,
                    source,
                    use_reentrant=False,
                )
            else:
                distances = self._sparse_graph_relax_step(
                    distances,
                    src,
                    dst,
                    sparse_weights,
                    source,
                )
            if self._has_converged(previous, distances):
                break
        return distances

    def forward(
        self,
        adjacency: Tensor,
        edge_weights: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        batched_graph = (
            adjacency.ndim == 3
            and edge_weights.ndim == 3
            and source_mask.ndim == 2
        )
        if batched_graph:
            if adjacency.shape != edge_weights.shape or source_mask.shape != adjacency.shape[:2]:
                raise ValueError(
                    "Batched sparse graph inputs require adjacency/weights [B,N,N] "
                    "and source_mask [B,N]."
                )
            return torch.stack(
                [
                    self._forward_sparse_graph(adjacency[i], edge_weights[i], source_mask[i])
                    for i in range(int(adjacency.shape[0]))
                ],
                dim=0,
            )
        if adjacency.ndim == 2 and edge_weights.ndim == 2 and source_mask.ndim == 1:
            return self._forward_sparse_graph(adjacency, edge_weights, source_mask)
        return super().forward(adjacency, edge_weights, source_mask)


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


class SoftBellmanFordGridPathfinder(nn.Module):
    """
    Explicit soft Bellman-Ford room pathfinder for ablations.

    This shares the graph pathfinder's relaxation rule but applies it to the
    4-neighborhood room grid, making the pathfinding inductive bias auditable
    against the learned convolutional approximation.
    """

    WALKABLE_IDS = CANONICAL_LOGIC_WALKABLE_IDS

    def __init__(
        self,
        num_iterations: int = 20,
        temperature: float = 0.1,
        num_classes: int = 44,
        full_coverage: bool = True,
    ):
        super().__init__()
        self.num_classes = int(max(1, num_classes))
        if not any(tile_id < self.num_classes for tile_id in self.WALKABLE_IDS):
            raise ValueError(
                "SoftBellmanFordGridPathfinder num_classes is too small for the configured walkable tile IDs."
            )
        skipped_walkable_ids = [tile_id for tile_id in self.WALKABLE_IDS if tile_id >= self.num_classes]
        if skipped_walkable_ids:
            logger.warning(
                "SoftBellmanFordGridPathfinder skipping walkable tile IDs outside num_classes=%d: %s",
                self.num_classes,
                skipped_walkable_ids,
            )
        self.pathfinder = DifferentiablePathfinder(
            num_iterations=num_iterations,
            temperature=temperature,
            full_coverage=full_coverage,
        )

        walkability = torch.zeros(self.num_classes)
        for tile_id in self.WALKABLE_IDS:
            if tile_id < self.num_classes:
                walkability[tile_id] = 1.0
        self.register_buffer("walkability_weights", walkability)

    def _derive_walkability(self, room_grid: Tensor) -> Tensor:
        if room_grid.dim() != 4:
            raise ValueError(f"room_grid must be [B,C,H,W], got {tuple(room_grid.shape)}.")
        if int(room_grid.shape[1]) == 1:
            return torch.sigmoid(room_grid[:, :1])
        if int(room_grid.shape[1]) != self.num_classes:
            raise ValueError(
                f"SoftBellmanFordGridPathfinder expected {self.num_classes} tile channels, "
                f"got {int(room_grid.shape[1])}."
            )
        probs = F.softmax(room_grid.float(), dim=1).to(dtype=room_grid.dtype)
        walkability = torch.einsum("bchw,c->bhw", probs, self.walkability_weights)
        return walkability.unsqueeze(1)

    def forward(
        self,
        room_grid: Tensor,
        source_mask: Tensor,
        walkability: Optional[Tensor] = None,
    ) -> Tensor:
        if source_mask.dim() != 4 or int(source_mask.shape[1]) != 1:
            raise ValueError(f"source_mask must be [B,1,H,W], got {tuple(source_mask.shape)}.")
        if walkability is None:
            walkability = self._derive_walkability(room_grid)
        if walkability.dim() != 4 or int(walkability.shape[1]) != 1:
            raise ValueError(f"walkability must be [B,1,H,W], got {tuple(walkability.shape)}.")
        if walkability.shape[0] != source_mask.shape[0] or walkability.shape[-2:] != source_mask.shape[-2:]:
            raise ValueError(
                f"walkability/source shape mismatch: walkability={tuple(walkability.shape)}, "
                f"source_mask={tuple(source_mask.shape)}."
            )

        distances = self.pathfinder(
            walkability[:, 0].clamp(0.0, 1.0),
            torch.ones_like(walkability[:, 0]),
            source_mask[:, 0].clamp(0.0, 1.0),
        )
        return distances.unsqueeze(1)


class LearnableGridPathfinder(nn.Module):
    """
    Learnable grid value-propagation pathfinder ablation.

    The module keeps the same distance-field contract as the fixed
    SoftBellmanFordGridPathfinder, but learns the local transition kernel used
    to propagate value estimates. The legacy config name "vin" is retained as
    a compatibility alias, but reports should describe this as a learnable grid
    pathfinder unless the full VIN policy head/objective is added.
    """

    WALKABLE_IDS = CANONICAL_LOGIC_WALKABLE_IDS

    def __init__(
        self,
        num_iterations: int = 20,
        temperature: float = 0.1,
        num_classes: int = 44,
        num_actions: int = 4,
        inf_distance: float = 20.0,
        full_coverage: bool = True,
    ) -> None:
        super().__init__()
        self.num_iterations = int(max(1, num_iterations))
        self.temperature = float(max(1e-4, temperature))
        self.num_classes = int(max(1, num_classes))
        self.num_actions = int(max(1, num_actions))
        self.inf_distance = float(max(1.0, inf_distance))
        self.full_coverage = bool(full_coverage)

        self.reward_proj = nn.Sequential(
            nn.Conv2d(self.num_classes + 2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.transition = nn.Conv2d(
            2,
            self.num_actions,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        with torch.no_grad():
            self.transition.weight.zero_()
            centers = [
                (0, 1),  # north reads value above
                (2, 1),  # south
                (1, 0),  # west
                (1, 2),  # east
            ]
            for action, (row, col) in enumerate(centers[: self.num_actions]):
                self.transition.weight[action, 1, row, col] = 1.0

        walkability = torch.zeros(self.num_classes)
        for tile_id in self.WALKABLE_IDS:
            if tile_id < self.num_classes:
                walkability[tile_id] = 1.0
        self.register_buffer("walkability_weights", walkability)

    def _derive_walkability(self, room_grid: Tensor) -> Tensor:
        if room_grid.dim() != 4:
            raise ValueError(f"room_grid must be [B,C,H,W], got {tuple(room_grid.shape)}.")
        if int(room_grid.shape[1]) == 1:
            return torch.sigmoid(room_grid[:, :1])
        if int(room_grid.shape[1]) != self.num_classes:
            raise ValueError(
                f"LearnableGridPathfinder expected {self.num_classes} tile channels, got {int(room_grid.shape[1])}."
            )
        probs = F.softmax(room_grid.float(), dim=1).to(dtype=room_grid.dtype)
        walkability = torch.einsum("bchw,c->bhw", probs, self.walkability_weights)
        return walkability.unsqueeze(1)

    def forward(
        self,
        room_grid: Tensor,
        source_mask: Tensor,
        walkability: Optional[Tensor] = None,
    ) -> Tensor:
        if source_mask.dim() != 4 or int(source_mask.shape[1]) != 1:
            raise ValueError(f"source_mask must be [B,1,H,W], got {tuple(source_mask.shape)}.")
        if walkability is None:
            walkability = self._derive_walkability(room_grid)
        if walkability.dim() != 4 or int(walkability.shape[1]) != 1:
            raise ValueError(f"walkability must be [B,1,H,W], got {tuple(walkability.shape)}.")

        tile_probs = (
            F.softmax(room_grid.float(), dim=1).to(dtype=room_grid.dtype)
            if int(room_grid.shape[1]) == self.num_classes
            else room_grid.expand(-1, self.num_classes, -1, -1)
        )
        walk = walkability.clamp(0.0, 1.0)
        source = source_mask.clamp(0.0, 1.0)
        reward = self.reward_proj(torch.cat([tile_probs, walk, source], dim=1))
        reward = reward + 5.0 * source - 5.0 * (1.0 - walk)

        value = torch.zeros_like(source)
        spatial_nodes = int(room_grid.shape[-2]) * int(room_grid.shape[-1])
        iteration_limit = (
            max(self.num_iterations, max(1, spatial_nodes - 1))
            if self.full_coverage
            else self.num_iterations
        )
        for _ in range(iteration_limit):
            q_values = self.transition(torch.cat([reward, value], dim=1))
            updated = soft_max(q_values, dim=1, temperature=max(self.temperature, 1e-4)).unsqueeze(1)
            value = torch.maximum(value, reward + updated)
            # NOTE: Do NOT zero the source cell here. The large source reward
            # drives value to its maximum at the source through the max() update.
            # Per-step zeroing would reset the source to 0 every iteration,
            # reporting maximum distance (distance_scale) at the start cell.
            # The output already zeroes source via: `distances * (1.0 - source)`.

        value = value - value.amin(dim=(2, 3), keepdim=True)
        value = value / value.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        semantic_walkability = torch.einsum("bchw,c->bhw", tile_probs, self.walkability_weights).unsqueeze(1)
        semantic_cost = (1.0 - semantic_walkability.clamp(0.0, 1.0)) * 5.0
        distance_scale = max(1.0, self.inf_distance - 5.0)
        distances = ((1.0 - value) * distance_scale + semantic_cost).clamp(0.0, self.inf_distance)
        return distances * (1.0 - source)


ValueIterationGridPathfinder = LearnableGridPathfinder


class PerturbAndMAPGridPathfinder(nn.Module):
    """
    Hard stochastic grid pathfinder with straight-through gradients.

    Forward solves perturbed hard shortest paths over predicted walkability.
    Backward routes a surrogate gradient through the expected hard-solver
    support. This is intentionally an ablation path, not a default replacement
    for the smoother Bellman-Ford teacher.
    """

    WALKABLE_IDS = CANONICAL_LOGIC_WALKABLE_IDS

    def __init__(
        self,
        num_iterations: int = 20,
        temperature: float = 0.1,
        num_classes: int = 44,
        num_samples: Optional[int] = None,
        obstacle_penalty: float = 8.0,
    ) -> None:
        super().__init__()
        self.num_iterations = int(max(1, num_iterations))
        self.temperature = float(max(1e-4, temperature))
        self.num_classes = int(max(1, num_classes))
        self.num_samples = int(max(1, num_samples if num_samples is not None else min(8, self.num_iterations)))
        self.obstacle_penalty = float(max(0.0, obstacle_penalty))

        walkability = torch.zeros(self.num_classes)
        for tile_id in self.WALKABLE_IDS:
            if tile_id < self.num_classes:
                walkability[tile_id] = 1.0
        self.register_buffer("walkability_weights", walkability)

    def _derive_walkability(self, room_grid: Tensor) -> Tensor:
        if room_grid.dim() != 4:
            raise ValueError(f"room_grid must be [B,C,H,W], got {tuple(room_grid.shape)}.")
        if int(room_grid.shape[1]) == 1:
            return torch.sigmoid(room_grid[:, :1])
        if int(room_grid.shape[1]) != self.num_classes:
            raise ValueError(
                f"PerturbAndMAPGridPathfinder expected {self.num_classes} tile channels, got {int(room_grid.shape[1])}."
            )
        probs = F.softmax(room_grid.float(), dim=1).to(dtype=room_grid.dtype)
        walkability = torch.einsum("bchw,c->bhw", probs, self.walkability_weights)
        return walkability.unsqueeze(1)

    def forward(
        self,
        room_grid: Tensor,
        source_mask: Tensor,
        walkability: Optional[Tensor] = None,
    ) -> Tensor:
        if source_mask.dim() != 4 or int(source_mask.shape[1]) != 1:
            raise ValueError(f"source_mask must be [B,1,H,W], got {tuple(source_mask.shape)}.")
        if walkability is None:
            walkability = self._derive_walkability(room_grid)
        if walkability.dim() != 4 or int(walkability.shape[1]) != 1:
            raise ValueError(f"walkability must be [B,1,H,W], got {tuple(walkability.shape)}.")
        if walkability.shape != source_mask.shape:
            raise ValueError(
                f"walkability/source shape mismatch: walkability={tuple(walkability.shape)}, "
                f"source_mask={tuple(source_mask.shape)}."
            )
        return perturb_and_map_distance(
            walkability.clamp(0.0, 1.0),
            source_mask.clamp(0.0, 1.0),
            num_samples=self.num_samples,
            noise_scale=self.temperature,
            obstacle_penalty=self.obstacle_penalty,
        )


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
        return_loss: bool = False,
    ) -> Any:
        """
        Compute reachability scores.
        
        Args:
            distances: [N] or [B, N] distances from start
            target_mask: [N] or [B, N] mask for target nodes (optional)
            
        Returns:
            scores: [N] or [B, N] reachability scores in [0, 1]
            loss: Scalar loss (1 - mean reachability of targets)
        """
        # Distances are semantically non-negative. The CNN room pathfinder can emit
        # unconstrained values early in training, so softly project them back into
        # the valid domain before computing reachability. This prevents scores > 1
        # and negative "losses" that would otherwise destabilize diffusion training.
        distances = F.softplus(distances)

        # Compute reachability scores -- smooth, no saturation or clamp dead zones.
        # Use exponential decay for the primary score (always has gradient).
        # Temperature controls the sharpness: high temp = smooth gradients early,
        # annealed low temp = sharp scores at convergence.
        effective_temp = max(float(self.temperature), 1e-4)
        scores = torch.exp(-distances / (effective_temp * self.max_distance + 1e-8))
        
        # Mix with a linear component for stable early-training gradients.
        # The linear part uses softplus instead of clamp to avoid dead zones.
        normalized = distances / (self.max_distance + 1e-8)
        linear_scores = torch.sigmoid(2.0 * (1.0 - normalized))  # smooth [0, 1]
        scores = 0.5 * scores + 0.5 * linear_scores
        
        # Compute loss
        if target_mask is not None:
            # If mask shape matches distances shape and includes batch dimensions,
            # compute per-batch target-reachability scores for compatibility.
            if distances.ndim >= 2 and target_mask.shape == distances.shape:
                reduce_dims = tuple(range(1, distances.ndim))
                target_scores = scores * target_mask
                num_targets = target_mask.sum(dim=reduce_dims).clamp_min(1e-6)
                per_batch = target_scores.sum(dim=reduce_dims) / num_targets
                mean_reachability = per_batch.mean()
                scores_out = per_batch
            else:
                # Focus on target nodes
                target_scores = scores * target_mask
                num_targets = target_mask.sum() + 1e-6
                mean_reachability = target_scores.sum() / num_targets
                scores_out = mean_reachability
        else:
            mean_reachability = scores.mean()
            scores_out = scores
        
        # Loss: want high reachability
        loss = 1.0 - mean_reachability
        
        if return_loss:
            return scores_out, loss
        return scores_out


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
        legacy_probability_mode: bool = False,
    ):
        super().__init__()
        
        self.margin = margin
        self.temperature = temperature
        self.legacy_probability_mode = bool(legacy_probability_mode)
    
    def forward(
        self,
        distances: Tensor,
        key_nodes: Tensor,
        lock_nodes: Optional[Tensor] = None,
        key_lock_pairs: Optional[List[Tuple[int, int]]] = None,
        mode: Optional[str] = None,
    ) -> Any:
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
        # Explicit legacy probability mode:
        # checker(key_probs, lock_probs, mode="legacy_probability") -> score in [0, 1]
        if lock_nodes is None and key_lock_pairs is None and distances.ndim == 1 and key_nodes.ndim == 1:
            requested_mode = str(mode or "").strip().lower()
            legacy_requested = requested_mode in {"legacy", "legacy_probability", "probability"}
            if not (legacy_requested or self.legacy_probability_mode):
                raise ValueError(
                    "Ambiguous two-tensor KeyLockChecker call. Pass lock_nodes/key_lock_pairs for "
                    "distance-based checking, or mode='legacy_probability' for the old "
                    "checker(key_probs, lock_probs) score."
                )
            key_mean = distances.mean() if distances.numel() > 0 else torch.tensor(0.0, device=distances.device)
            lock_mean = key_nodes.mean() if key_nodes.numel() > 0 else torch.tensor(0.0, device=distances.device)
            return torch.sigmoid(key_mean - lock_mean)

        if lock_nodes is None:
            lock_nodes = torch.zeros_like(key_nodes)
        if key_lock_pairs is None:
            key_lock_pairs = []

        valid_pairs = [
            (int(key_idx), int(lock_idx))
            for key_idx, lock_idx in key_lock_pairs
            if 0 <= int(key_idx) < int(distances.shape[0]) and 0 <= int(lock_idx) < int(distances.shape[0])
        ]
        if valid_pairs:
            pair_tensor = torch.tensor(valid_pairs, device=distances.device, dtype=torch.long)
            key_dists = distances.index_select(0, pair_tensor[:, 0])
            lock_dists = distances.index_select(0, pair_tensor[:, 1])
            violations_t = F.relu(key_dists - lock_dists + self.margin)
            loss = violations_t.mean()
        else:
            violations_t = torch.empty(0, device=distances.device, dtype=distances.dtype)
            loss = torch.tensor(0.0, device=distances.device, dtype=distances.dtype)
        
        info = {
            'num_violations': int((violations_t > 0).sum().detach().item()),
            'total_violation': loss,
        }
        
        return loss, info


class SemanticEdgeEncoder(nn.Module):
    """Learnable traversal penalties for graph edge labels."""

    def __init__(self, num_edge_types: int = 16):
        super().__init__()
        base = torch.zeros(num_edge_types, dtype=torch.float32)
        for idx, value in GRAPH_EDGE_TRAVERSAL_PENALTIES.items():
            if idx < num_edge_types:
                base[idx] = float(value)
        self.register_buffer("base_penalty", base)
        self.residual_logits = nn.Parameter(torch.zeros(num_edge_types, dtype=torch.float32))

    def forward(self, edge_attr: Tensor) -> Tensor:
        attr = edge_attr.to(dtype=torch.long).clamp(0, int(self.base_penalty.numel()) - 1)
        residual = F.softplus(self.residual_logits) - F.softplus(
            torch.zeros((), device=self.residual_logits.device, dtype=self.residual_logits.dtype)
        )
        penalties = (
            self.base_penalty.to(device=edge_attr.device, dtype=residual.dtype)
            + residual.to(device=edge_attr.device)
        ).clamp_min(0.0)
        return penalties.index_select(0, attr.flatten()).view_as(attr).to(dtype=torch.float32)


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
        in_channels: Optional[int] = None,
        num_classes: int = 44,
        hidden_dim: int = 128,
        output_mode: str = "logits",
    ):
        super().__init__()

        if in_channels is not None:
            latent_dim = int(in_channels)
        self.output_mode = str(output_mode).strip().lower()
        if self.output_mode not in {"logits", "probs"}:
            raise ValueError(f"TileClassifier output_mode must be 'logits' or 'probs', got {output_mode!r}.")
        
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
        logits = self.classifier(z)
        if self.output_mode == "logits":
            return logits
        return F.softmax(logits.float(), dim=1).to(dtype=logits.dtype)


class WalkabilityPredictor(nn.Module):
    """
    Predicts walkability mask from tile logits.
    
    Walkable tiles: FLOOR, DOOR_*, STAIR
    Non-walkable: WALL, BLOCK, VOID
    """
    
    WALKABLE_IDS = CANONICAL_LOGIC_WALKABLE_IDS
    
    def __init__(self, num_classes: int = 44, num_tile_classes: Optional[int] = None, keep_channel_dim: bool = False):
        super().__init__()

        if num_tile_classes is not None:
            num_classes = int(num_tile_classes)
        self.keep_channel_dim = bool(keep_channel_dim)
        
        # Create walkability weight vector
        walkability = torch.zeros(num_classes)
        for tid in self.WALKABLE_IDS:
            if tid < num_classes:
                walkability[tid] = 1.0
        
        self.register_buffer('walkability_weights', walkability)
    
    def forward(self, tile_logits: Tensor, *, is_probs: Optional[bool] = None) -> Tensor:
        """
        Predict walkability from tile logits.
        
        Args:
            tile_logits: [B, C, H, W] tile class logits
            
        Returns:
            walkability: [B, 1, H, W] soft walkability mask
        """
        # Prefer an explicit contract. The legacy heuristic is retained only
        # when callers omit is_probs.
        if is_probs is True:
            probs = tile_logits
        elif is_probs is False:
            probs = F.softmax(tile_logits.float(), dim=1).to(dtype=tile_logits.dtype)
        else:
            if (
                torch.all(tile_logits >= 0)
                and torch.all(tile_logits <= 1)
                and torch.allclose(
                    tile_logits.sum(dim=1),
                    torch.ones_like(tile_logits[:, :1, :, :]).squeeze(1),
                    atol=1e-4,
                )
            ):
                probs = tile_logits
            else:
                probs = F.softmax(tile_logits.float(), dim=1).to(dtype=tile_logits.dtype)
        
        # Weighted sum with walkability
        walkability = torch.einsum(
            'bchw,c->bhw',
            probs,
            self.walkability_weights,
        )

        if self.keep_channel_dim:
            return walkability.unsqueeze(1)
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
        hidden_dim: int = 128,
        num_tile_classes: Optional[int] = None,
        num_classes: int = 44,
        num_iterations: int = 20,
        temperature: float = 0.1,
        reach_weight: float = 1.0,
        lock_weight: float = 0.5,
        topology_trace_weight: float = 0.25,
        topology_anchor_weight: float = 0.25,
        global_reach_weight: float = 1.0,
        global_room_weight: float = 0.25,
        grid_pathfinder_type: str = "bellman_ford",
        graph_pathfinder_type: str = "dense_bellman_ford",
        resource_gate_mode: str = "hard_ordered",
        full_coverage: bool = True,
        # --- Phase 1D: Temperature annealing (Jang et al., 2017) ---
        initial_temperature: float = 1.0,
        final_temperature: float = 0.05,
        latent_scale_factor: float = 1.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.latent_scale_factor = float(latent_scale_factor)
        if not math.isfinite(self.latent_scale_factor) or self.latent_scale_factor <= 0.0:
            raise ValueError("latent_scale_factor must be finite and greater than zero.")
        if num_tile_classes is not None:
            num_classes = int(num_tile_classes)
        self.num_classes = num_classes
        self.reach_weight = reach_weight
        self.lock_weight = lock_weight
        self.topology_trace_weight = float(max(0.0, topology_trace_weight))
        self.topology_anchor_weight = float(max(0.0, topology_anchor_weight))
        self.global_reach_weight = float(max(0.0, global_reach_weight))
        self.global_room_weight = float(max(0.0, global_room_weight))
        self.grid_pathfinder_type = str(grid_pathfinder_type).strip().lower()
        self.graph_pathfinder_type = str(graph_pathfinder_type).strip().lower().replace("-", "_")
        if self.graph_pathfinder_type in {"dense", "bellman_ford"}:
            self.graph_pathfinder_type = "dense_bellman_ford"
        elif self.graph_pathfinder_type in {"sparse", "edge_sparse", "sparse_edge"}:
            self.graph_pathfinder_type = "sparse_bellman_ford"
        if self.graph_pathfinder_type not in {"dense_bellman_ford", "sparse_bellman_ford"}:
            raise ValueError(
                "graph_pathfinder_type must be 'dense_bellman_ford' or 'sparse_bellman_ford'."
            )
        self.resource_gate_mode = str(resource_gate_mode).strip().lower()
        if self.resource_gate_mode not in {"hard_ordered", "soft_ordered"}:
            raise ValueError(
                "resource_gate_mode must be 'hard_ordered' or 'soft_ordered'."
            )
        self.full_coverage = bool(full_coverage)
        if self.grid_pathfinder_type in {"bellman-ford", "soft_bellman_ford", "soft-bellman-ford"}:
            self.grid_pathfinder_type = "bellman_ford"
        if self.grid_pathfinder_type in {"value_iteration", "value-iteration"}:
            self.grid_pathfinder_type = "vin"
        if self.grid_pathfinder_type in {"perturb-and-map", "perturb_map", "pmap"}:
            self.grid_pathfinder_type = "perturb_and_map"
        if self.grid_pathfinder_type not in {"cnn", "bellman_ford", "vin", "perturb_and_map"}:
            raise ValueError("grid_pathfinder_type must be 'cnn', 'bellman_ford', 'vin', or 'perturb_and_map'.")
        
        # --- Phase 1D: Temperature annealing state ---
        for name, value in (
            ("temperature", temperature),
            ("initial_temperature", initial_temperature),
            ("final_temperature", final_temperature),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero.")
        self.initial_temperature = float(initial_temperature)
        self.final_temperature = float(final_temperature)
        self.register_buffer(
            'current_temperature',
            torch.tensor(self.initial_temperature, dtype=torch.float32),
        )
        self.register_buffer(
            "locked_edge_role_ids",
            torch.tensor(GRAPH_EDGE_LOCK_FEATURE_INDICES, dtype=torch.long),
            persistent=False,
        )
        
        # Tile classification
        self.tile_classifier = TileClassifier(
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            output_mode="logits",
        )
        
        # Walkability prediction
        self.walkability = WalkabilityPredictor(num_classes=num_classes, keep_channel_dim=True)
        
        # Grid-level pathfinder
        if self.grid_pathfinder_type == "bellman_ford":
            self.grid_pathfinder = SoftBellmanFordGridPathfinder(
                num_iterations=num_iterations,
                temperature=temperature,
                num_classes=num_classes,
                full_coverage=self.full_coverage,
            )
        elif self.grid_pathfinder_type == "vin":
            self.grid_pathfinder = LearnableGridPathfinder(
                num_iterations=num_iterations,
                temperature=temperature,
                num_classes=num_classes,
                full_coverage=self.full_coverage,
            )
        elif self.grid_pathfinder_type == "perturb_and_map":
            self.grid_pathfinder = PerturbAndMAPGridPathfinder(
                num_iterations=num_iterations,
                temperature=temperature,
                num_classes=num_classes,
            )
        else:
            self.grid_pathfinder = ConvolutionalPathfinder(
                num_layers=10,
                hidden_dim=32,
                input_channels=num_classes,
            )
        
        # Graph-level pathfinder
        graph_pathfinder_cls = (
            SparseDifferentiablePathfinder
            if self.graph_pathfinder_type == "sparse_bellman_ford"
            else DifferentiablePathfinder
        )
        self.graph_pathfinder = graph_pathfinder_cls(
            num_iterations=num_iterations,
            temperature=temperature,
            full_coverage=self.full_coverage,
        )
        self.semantic_edge_encoder = SemanticEdgeEncoder()
        
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
        # The schedule buffer and the actual solver temperatures must agree
        # before the first optimizer step and after checkpoint restoration.
        self._sync_temperature_modules(self.initial_temperature)
        self.register_load_state_dict_post_hook(self._restore_temperature_after_load)

    @staticmethod
    def _project_tile_logits_to_room(tile_logits: Tensor) -> Tensor:
        """Project latent-scale room logits to canonical room size when needed."""
        if tile_logits.dim() != 4:
            raise ValueError(f"tile_logits must be [B,C,H,W], got {tuple(tile_logits.shape)}.")
        height, width = int(tile_logits.shape[-2]), int(tile_logits.shape[-1])
        if (height, width) != (ROOM_HEIGHT, ROOM_WIDTH) and height <= ROOM_HEIGHT and width <= ROOM_WIDTH:
            return F.interpolate(
                tile_logits,
                size=(ROOM_HEIGHT, ROOM_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
        return tile_logits

    @staticmethod
    def _normalize_room_topology_map(
        room_topology_map: Optional[Any],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tensor]:
        if not isinstance(room_topology_map, torch.Tensor):
            return None
        topo = room_topology_map.to(device=device, dtype=dtype)
        if topo.dim() == 3:
            topo = topo.unsqueeze(0)
        if topo.dim() != 4:
            raise ValueError(
                f"LogicNet graph_data['room_topology_map'] must have shape [B,C,H,W] or [C,H,W], got {tuple(topo.shape)}."
            )
        if int(topo.shape[0]) == 1 and batch_size > 1:
            topo = topo.expand(batch_size, -1, -1, -1)
        elif int(topo.shape[0]) != batch_size:
            raise ValueError(
                f"LogicNet graph_data['room_topology_map'] batch {int(topo.shape[0])} does not match latent batch {batch_size}."
            )
        target_hw = (int(spatial_hw[0]), int(spatial_hw[1]))
        if tuple(topo.shape[-2:]) != target_hw:
            topo = F.interpolate(topo, size=target_hw, mode="bilinear", align_corners=False)
        return topo

    @staticmethod
    def _door_slices_for_shape(direction: str, height: int, width: int) -> Tuple[slice, slice]:
        """Return relative door spans for arbitrary room/map sizes."""
        h = int(max(1, height))
        w = int(max(1, width))
        direction = str(direction).upper()
        horizontal_span = max(1, min(w, int(round(w * 3.0 / max(1, ROOM_WIDTH)))))
        vertical_span = max(1, min(h, int(round(h * 3.0 / max(1, ROOM_HEIGHT)))))
        col_start = max(0, (w - horizontal_span) // 2)
        row_start = max(0, (h - vertical_span) // 2)
        if direction == "N":
            return slice(0, 1), slice(col_start, col_start + horizontal_span)
        if direction == "S":
            return slice(h - 1, h), slice(col_start, col_start + horizontal_span)
        if direction == "E":
            return slice(row_start, row_start + vertical_span), slice(w - 1, w)
        if direction == "W":
            return slice(row_start, row_start + vertical_span), slice(0, 1)
        raise ValueError(f"Unknown door direction {direction!r}.")

    @staticmethod
    def _normalize_boundary_constraints(
        boundary_constraints: Optional[Any],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if not isinstance(boundary_constraints, torch.Tensor):
            return None
        boundary = boundary_constraints.to(device=device, dtype=dtype)
        if boundary.dim() == 1:
            boundary = boundary.unsqueeze(0)
        if boundary.dim() != 2 or int(boundary.shape[1]) != 8:
            raise ValueError(
                "LogicNet graph_data['boundary_constraints'] must have shape [B,8] or [8]."
            )
        if int(boundary.shape[0]) == 1 and batch_size > 1:
            boundary = boundary.expand(batch_size, -1)
        elif int(boundary.shape[0]) != batch_size:
            raise ValueError(
                f"LogicNet graph_data['boundary_constraints'] batch {int(boundary.shape[0])} does not match latent batch {batch_size}."
            )
        return boundary

    @staticmethod
    def _build_boundary_door_mask(
        boundary_constraints: Optional[Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        spatial_hw: Tuple[int, int],
    ) -> Optional[Tensor]:
        if boundary_constraints is None:
            return None
        active = torch.maximum(
            boundary_constraints[:, 0::2],
            boundary_constraints[:, 1::2],
        ).clamp(0.0, 1.0)

        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        mask = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
        for idx, direction in enumerate(("N", "S", "E", "W")):
            values = active[:, idx]
            row_slice, col_slice = LogicNet._door_slices_for_shape(direction, height, width)
            target = mask[:, 0, row_slice, col_slice]
            expanded = values.view(batch_size, 1, 1).expand_as(target)
            mask[:, 0, row_slice, col_slice] = torch.maximum(target, expanded)
        return mask

    def _resolve_room_logic_targets(
        self,
        graph_data: Optional[Any],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Optional[Tensor]]:
        spatial_hw = spatial_hw if spatial_hw is not None else (ROOM_HEIGHT, ROOM_WIDTH)
        targets: Dict[str, Optional[Tensor]] = {
            "source_mask": None,
            "target_mask": None,
            "trace_target": None,
            "anchor_target": None,
        }
        if not isinstance(graph_data, dict):
            return targets

        def _explicit_mask(name: str) -> Optional[Tensor]:
            value = graph_data.get(name)
            if not isinstance(value, torch.Tensor):
                return None
            mask = value.to(device=device, dtype=dtype)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.dim() != 4 or int(mask.shape[1]) != 1:
                raise ValueError(
                    f"LogicNet graph_data[{name!r}] must have shape [B,1,H,W] or [B,H,W], "
                    f"got {tuple(mask.shape)}."
                )
            if int(mask.shape[0]) == 1 and batch_size > 1:
                mask = mask.expand(batch_size, -1, -1, -1)
            elif int(mask.shape[0]) != batch_size:
                raise ValueError(
                    f"LogicNet graph_data[{name!r}] batch {int(mask.shape[0])} "
                    f"does not match latent batch {batch_size}."
                )
            if tuple(mask.shape[-2:]) != tuple(spatial_hw):
                mask = F.interpolate(mask.float(), size=spatial_hw, mode="nearest")
            return mask.clamp(0.0, 1.0)

        explicit_source = _explicit_mask("logic_source_mask")
        explicit_target = _explicit_mask("logic_target_mask")

        topology_map = self._normalize_room_topology_map(
            graph_data.get("room_topology_map"),
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            spatial_hw=spatial_hw if spatial_hw is not None else (ROOM_HEIGHT, ROOM_WIDTH),
        )
        boundary = self._normalize_boundary_constraints(
            graph_data.get("boundary_constraints"),
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        def _channel(name: str) -> Optional[Tensor]:
            if topology_map is None:
                return None
            index = int(ROOM_TOPOLOGY_CHANNELS.get(name, -1))
            if index < 0 or index >= int(topology_map.shape[1]):
                return None
            return topology_map[:, index:index + 1].clamp(0.0, 1.0)

        trace_target = _channel("traversability")
        start_target = _channel("start")
        goal_target = _channel("goal")

        door_channel_names: List[str] = []
        for direction in ("N", "S", "E", "W"):
            door_channel_names.extend(ROOM_TOPOLOGY_DIRECTIONAL_CHANNEL_GROUPS.get(direction, ()))
        door_parts = [
            maybe
            for maybe in (_channel(name) for name in door_channel_names)
            if maybe is not None
        ]
        door_target = None
        if door_parts:
            door_target = torch.clamp(torch.sum(torch.cat(door_parts, dim=1), dim=1, keepdim=True), 0.0, 1.0)

        boundary_door_target = self._build_boundary_door_mask(
            boundary,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            spatial_hw=spatial_hw if spatial_hw is not None else (ROOM_HEIGHT, ROOM_WIDTH),
        )
        if door_target is None:
            door_target = boundary_door_target
        elif boundary_door_target is not None:
            door_target = torch.maximum(door_target, boundary_door_target)

        def _first_nonempty_per_sample(*candidates: Optional[Tensor]) -> Optional[Tensor]:
            available = [candidate for candidate in candidates if candidate is not None]
            if not available:
                return None
            selected = torch.zeros_like(available[0])
            assigned = torch.zeros(
                batch_size,
                1,
                1,
                1,
                device=device,
                dtype=torch.bool,
            )
            for candidate in available:
                has_signal = candidate.amax(dim=(1, 2, 3), keepdim=True) > 0.0
                choose = has_signal & ~assigned
                selected = torch.where(choose, candidate, selected)
                assigned = assigned | has_signal
            return selected

        targets["source_mask"] = _first_nonempty_per_sample(
            explicit_source,
            start_target,
            door_target,
        )
        targets["target_mask"] = _first_nonempty_per_sample(
            explicit_target,
            goal_target,
            trace_target,
            door_target,
        )

        anchor_parts = [
            maybe
            for maybe in (start_target, goal_target, door_target)
            if maybe is not None
        ]
        if anchor_parts:
            targets["anchor_target"] = torch.clamp(
                torch.sum(torch.cat(anchor_parts, dim=1), dim=1, keepdim=True),
                0.0,
                1.0,
            )
        if trace_target is not None:
            targets["trace_target"] = trace_target

        return targets

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().flatten()[0].item()
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _select_batch_value(value: Any, batch_idx: int) -> Any:
        """Select one item from an optionally batched graph tensor/list."""
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value
            if value.dim() >= 1 and int(value.shape[0]) > batch_idx:
                return value[batch_idx]
            return value
        if isinstance(value, (list, tuple)) and len(value) > batch_idx:
            return value[batch_idx]
        return value

    @staticmethod
    def _current_indices_for_graph(
        current_node_idx: Any,
        *,
        batch_size: int,
        node_count: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(current_node_idx, torch.Tensor):
            indices = current_node_idx.detach().to(device=device, dtype=torch.long).flatten()
        elif isinstance(current_node_idx, (list, tuple)):
            indices = torch.tensor([int(v) for v in current_node_idx], device=device, dtype=torch.long)
        elif current_node_idx is None:
            if int(batch_size) == int(node_count):
                indices = torch.arange(node_count, device=device, dtype=torch.long)
            else:
                indices = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            indices = torch.full((batch_size,), int(current_node_idx), device=device, dtype=torch.long)

        if indices.numel() == 1 and batch_size > 1:
            indices = indices.expand(batch_size)
        if indices.numel() < batch_size:
            pad = torch.zeros(batch_size - int(indices.numel()), device=device, dtype=torch.long)
            indices = torch.cat([indices, pad], dim=0)
        indices = indices[:batch_size]
        return indices.clamp(0, max(0, int(node_count) - 1))

    @staticmethod
    def _infer_target_idx(
        node_features: Optional[Tensor],
        explicit_target_idx: Any,
        *,
        node_count: int,
    ) -> Optional[int]:
        target_idx = LogicNet._coerce_optional_int(explicit_target_idx)
        if target_idx is not None and 0 <= target_idx < node_count:
            return target_idx
        if (
            isinstance(node_features, torch.Tensor)
            and node_features.dim() == 2
            and node_features.shape[1] > GRAPH_NODE_GOAL_FEATURE_INDEX
        ):
            hits = torch.nonzero(
                node_features[:, GRAPH_NODE_GOAL_FEATURE_INDEX] > 0.5,
                as_tuple=False,
            ).flatten()
            if hits.numel() > 0:
                return int(hits[0].item())
        return None

    def _edge_feature_penalty(
        self,
        edge_features: Optional[Tensor],
        edge_attr: Optional[Tensor],
        num_edges: int,
    ) -> Optional[Tensor]:
        penalty: Optional[Tensor] = None
        if isinstance(edge_features, torch.Tensor) and edge_features.numel() > 0:
            ef = edge_features
            if ef.dim() == 1:
                ef = ef.unsqueeze(-1)
            ef = ef[:num_edges].float()
            penalty = torch.zeros(num_edges, device=ef.device, dtype=ef.dtype)
            for feature_index, scale in GRAPH_EDGE_TRAVERSAL_PENALTIES.items():
                if ef.shape[1] > feature_index:
                    penalty = penalty + ef[:, feature_index].clamp(0.0, 1.0) * scale

        if isinstance(edge_attr, torch.Tensor) and edge_attr.numel() > 0:
            attr = edge_attr[:num_edges].to(dtype=torch.long)
            attr_penalty = self.semantic_edge_encoder(attr)
            penalty = attr_penalty if penalty is None else torch.maximum(penalty.to(attr_penalty.device), attr_penalty)

        return penalty

    def _locked_edge_mask(
        self,
        *,
        node_count: int,
        edge_index: Optional[Tensor],
        adjacency: Optional[Tensor],
        edge_features: Optional[Tensor],
        edge_attr: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """Return a dense [N,N] mask of resource-gated lock edges."""
        n = int(max(0, node_count))
        locked = torch.zeros(n, n, device=device, dtype=torch.bool)
        if n <= 0:
            return locked

        def _locked_from_features(features: Tensor) -> Tensor:
            ef = features
            if ef.dim() == 1:
                ef = ef.unsqueeze(-1)
            if ef.shape[-1] <= min(GRAPH_EDGE_LOCK_FEATURE_INDICES):
                return torch.zeros(ef.shape[:-1], device=ef.device, dtype=torch.bool)
            mask = torch.zeros(ef.shape[:-1], device=ef.device, dtype=torch.bool)
            for feature_index in GRAPH_EDGE_LOCK_FEATURE_INDICES:
                if ef.shape[-1] > feature_index:
                    mask = mask | (ef[..., feature_index] > 0.5)
            return mask

        if isinstance(adjacency, torch.Tensor):
            if isinstance(edge_features, torch.Tensor) and edge_features.numel() > 0:
                ef = edge_features.to(device=device)
                if ef.dim() == 3 and int(ef.shape[0]) >= n and int(ef.shape[1]) >= n:
                    locked |= _locked_from_features(ef[:n, :n])
                elif ef.dim() == 2 and int(ef.shape[0]) >= n and int(ef.shape[1]) >= n:
                    locked |= ef[:n, :n] > 0.5
            if isinstance(edge_attr, torch.Tensor) and edge_attr.numel() > 0:
                ea = edge_attr.to(device=device)
                if ea.dim() == 2 and int(ea.shape[0]) >= n and int(ea.shape[1]) >= n:
                    locked |= torch.isin(
                        ea[:n, :n].to(dtype=torch.long),
                        self.locked_edge_role_ids.to(device=device),
                    )
            return locked

        if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
            return locked
        if edge_index.numel() <= 0:
            return locked
        src_all = edge_index[0].to(device=device, dtype=torch.long)
        dst_all = edge_index[1].to(device=device, dtype=torch.long)
        valid = (src_all >= 0) & (src_all < n) & (dst_all >= 0) & (dst_all < n)
        if not torch.any(valid):
            return locked
        src = src_all[valid]
        dst = dst_all[valid]
        edge_locked = torch.zeros_like(src, dtype=torch.bool)
        if isinstance(edge_features, torch.Tensor) and edge_features.numel() > 0:
            ef = edge_features.to(device=device)
            if ef.dim() >= 1 and int(ef.shape[0]) == int(valid.numel()):
                ef = ef[valid]
            if ef.dim() >= 1 and int(ef.shape[0]) >= int(src.numel()):
                edge_locked |= _locked_from_features(ef[: int(src.numel())]).flatten().to(device=device)
        if isinstance(edge_attr, torch.Tensor) and edge_attr.numel() > 0:
            ea = edge_attr.to(device=device)
            if ea.dim() >= 1 and int(ea.shape[0]) == int(valid.numel()):
                ea = ea[valid]
            if ea.dim() >= 1 and int(ea.shape[0]) >= int(src.numel()):
                edge_locked |= torch.isin(
                    ea[: int(src.numel())].flatten().to(dtype=torch.long),
                    self.locked_edge_role_ids.to(device=device),
                )
        if torch.any(edge_locked):
            locked[src[edge_locked], dst[edge_locked]] = True
        return locked

    def _build_adjacency_and_weights(
        self,
        *,
        node_count: int,
        device: torch.device,
        dtype: torch.dtype,
        edge_index: Optional[Tensor] = None,
        adjacency: Optional[Tensor] = None,
        edge_weights: Optional[Tensor] = None,
        edge_features: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        node_passability: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        n = int(max(0, node_count))
        if n <= 0:
            return None, None

        if isinstance(adjacency, torch.Tensor):
            adj = adjacency.to(device=device, dtype=dtype)
            if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
                return None, None
            adj = (adj[:n, :n] > 0).to(dtype=dtype)
            weights = (
                edge_weights.to(device=device, dtype=dtype)[:n, :n]
                if isinstance(edge_weights, torch.Tensor) and edge_weights.shape == adjacency.shape
                else adj.clone()
            )
            dense_penalty: Optional[Tensor] = None
            if isinstance(edge_features, torch.Tensor) and edge_features.numel() > 0:
                ef = edge_features.to(device=device)
                if ef.dim() == 3 and int(ef.shape[0]) >= n and int(ef.shape[1]) >= n:
                    ef = ef[:n, :n].to(dtype=dtype)
                    dense_penalty = torch.zeros(n, n, device=device, dtype=dtype)
                    for feature_index, scale in GRAPH_EDGE_TRAVERSAL_PENALTIES.items():
                        if ef.shape[2] > feature_index:
                            dense_penalty = (
                                dense_penalty
                                + ef[:, :, feature_index].clamp(0.0, 1.0) * scale
                            )
                elif ef.dim() == 2 and int(ef.shape[0]) >= n and int(ef.shape[1]) >= n:
                    dense_penalty = ef[:n, :n].to(dtype=dtype).clamp_min(0.0)
            if isinstance(edge_attr, torch.Tensor) and edge_attr.numel() > 0:
                ea = edge_attr.to(device=device)
                if ea.dim() == 2 and int(ea.shape[0]) >= n and int(ea.shape[1]) >= n:
                    attr_penalty = self.semantic_edge_encoder(ea[:n, :n]).to(device=device, dtype=dtype)
                    dense_penalty = attr_penalty if dense_penalty is None else torch.maximum(dense_penalty, attr_penalty)
            if dense_penalty is not None:
                weights = torch.where(adj > 0, weights + dense_penalty.to(device=device, dtype=dtype) * adj, weights)
        else:
            adj = torch.zeros(n, n, device=device, dtype=dtype)
            weights = torch.zeros(n, n, device=device, dtype=dtype)
            if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
                return None, None
            if edge_index.numel() > 0:
                src = edge_index[0].to(device=device, dtype=torch.long)
                dst = edge_index[1].to(device=device, dtype=torch.long)
                valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
                src = src[valid]
                dst = dst[valid]
                if src.numel() > 0:
                    num_edges = int(src.numel())
                    adj[src, dst] = 1.0
                    base = torch.ones(num_edges, device=device, dtype=dtype)
                    edge_features_valid = None
                    if isinstance(edge_features, torch.Tensor):
                        ef = edge_features.to(device=device)
                        if ef.dim() >= 1 and int(ef.shape[0]) == int(valid.numel()):
                            edge_features_valid = ef[valid]
                        else:
                            edge_features_valid = ef
                    edge_attr_valid = None
                    if isinstance(edge_attr, torch.Tensor):
                        ea = edge_attr.to(device=device)
                        if ea.dim() >= 1 and int(ea.shape[0]) == int(valid.numel()):
                            edge_attr_valid = ea[valid]
                        else:
                            edge_attr_valid = ea
                    feature_penalty = self._edge_feature_penalty(
                        edge_features_valid,
                        edge_attr_valid,
                        num_edges,
                    )
                    if feature_penalty is not None:
                        base = base + feature_penalty.to(device=device, dtype=dtype)
                    weights[src, dst] = base

        if node_passability is not None:
            node_pass = node_passability.to(device=device, dtype=dtype).flatten()[:n].clamp(0.0, 1.0)
            if node_pass.numel() < n:
                node_pass = F.pad(node_pass, (0, n - int(node_pass.numel())), value=1.0)
            # Treat passability as a traversal probability. Multiplying
            # (1-p) by the 1e6 infinity sentinel saturated ReachabilityScorer
            # for every ordinary prediction below ~1.0 and removed the graph
            # gradient. A negative-log barrier is finite and informative for
            # p in (0,1], while p~0 still behaves as effectively blocked.
            entry_penalty = (
                -torch.log(node_pass.clamp_min(1e-6)).reshape(1, n)
                * float(self.reachability.max_distance)
            )
            weights = torch.where(adj > 0, weights + entry_penalty, weights)

        return adj, weights

    def _room_passability_from_local_scores(
        self,
        *,
        walkability: Tensor,
        grid_reach_scores: Tensor,
        trace_target: Optional[Tensor],
        anchor_target: Optional[Tensor],
    ) -> Tensor:
        terms = [grid_reach_scores.reshape(walkability.shape[0]).clamp(0.0, 1.0)]
        if trace_target is not None:
            mass = trace_target.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            terms.append(((walkability * trace_target).sum(dim=(1, 2, 3)) / mass).clamp(0.0, 1.0))
        if anchor_target is not None:
            mass = anchor_target.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            terms.append(((walkability * anchor_target).sum(dim=(1, 2, 3)) / mass).clamp(0.0, 1.0))
        return torch.stack(terms, dim=0).mean(dim=0).clamp(0.0, 1.0)

    def _resource_gated_rollout(
        self,
        *,
        adjacency: Tensor,
        weights: Tensor,
        locked_edges: Tensor,
        source_mask: Tensor,
        key_lock_pairs: List[Tuple[int, int]],
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Evaluate ordered key acquisition while unlocking one gate at a time."""
        if self.resource_gate_mode == "soft_ordered":
            return self._soft_resource_gated_rollout(
                adjacency=adjacency,
                weights=weights,
                locked_edges=locked_edges,
                source_mask=source_mask,
                key_lock_pairs=key_lock_pairs,
                dtype=dtype,
            )
        device = adjacency.device
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        unlocked_edges = torch.zeros_like(locked_edges, dtype=torch.bool)
        pair_losses: List[Tensor] = []
        key_scores: List[Tensor] = []
        lock_scores: List[Tensor] = []
        remaining_providers_by_target: Dict[int, int] = {}
        for _provider_idx, gated_target_idx in key_lock_pairs:
            remaining_providers_by_target[int(gated_target_idx)] = (
                remaining_providers_by_target.get(int(gated_target_idx), 0) + 1
            )
        matched_edge_count = 0
        blocked_stage_count = 0

        def _distances() -> Tensor:
            active_locked = locked_edges & ~unlocked_edges
            staged_adjacency = torch.where(
                active_locked,
                torch.zeros_like(adjacency),
                adjacency,
            )
            return self.graph_pathfinder(staged_adjacency, weights, source_mask)

        distances = _distances()
        for key_idx, lock_idx in key_lock_pairs:
            key_target = torch.zeros_like(source_mask)
            key_target[int(key_idx)] = 1.0
            key_score, key_loss = self.reachability(
                distances,
                key_target,
                return_loss=True,
            )

            # key_lock_pairs identify the room entered through the lock. Keep
            # unrelated and later locks closed; this is what permits valid
            # key1 -> lock1 -> key2 -> lock2 progressions.
            matching_edges = locked_edges & (
                torch.arange(
                    int(adjacency.shape[0]),
                    device=device,
                ).view(1, -1)
                == int(lock_idx)
            )
            key_is_reachable = bool(
                distances[int(key_idx)].detach().item()
                < (float(self.graph_pathfinder.inf_distance) - 1e-6)
            )
            unlocks_target = False
            if key_is_reachable:
                target_key = int(lock_idx)
                remaining_providers_by_target[target_key] = max(
                    0,
                    remaining_providers_by_target.get(target_key, 1) - 1,
                )
                unlocks_target = remaining_providers_by_target[target_key] == 0
                if unlocks_target:
                    matched_edge_count += int(matching_edges.sum().detach().item())
                    unlocked_edges = unlocked_edges | matching_edges
            else:
                blocked_stage_count += 1
            distances = _distances()

            if unlocks_target:
                lock_target = torch.zeros_like(source_mask)
                lock_target[int(lock_idx)] = 1.0
                lock_score, lock_loss = self.reachability(
                    distances,
                    lock_target,
                    return_loss=True,
                )
                pair_losses.append(0.5 * (key_loss + lock_loss))
                lock_scores.append(lock_score.mean())
            else:
                pair_losses.append(key_loss)
            key_scores.append(key_score.mean())

        pair_loss_tensor = (
            torch.stack(pair_losses)
            if pair_losses
            else torch.empty(0, device=device, dtype=dtype)
        )
        resource_loss = pair_loss_tensor.mean() if pair_loss_tensor.numel() > 0 else zero
        remaining_locked = locked_edges & ~unlocked_edges
        info: Dict[str, Any] = {
            "num_violations": (
                int((pair_loss_tensor > 0.25).sum().detach().item())
                if pair_loss_tensor.numel() > 0
                else 0
            ),
            "total_violation": resource_loss,
            "key_lock_mode": "resource_gated",
            "resource_gate_ordering": "ordered",
            "locked_edge_count": float(locked_edges.float().sum().detach().item()),
            "matched_lock_edge_count": float(matched_edge_count),
            "remaining_locked_edge_count": float(
                remaining_locked.float().sum().detach().item()
            ),
            "resource_stage_count": float(len(key_lock_pairs)),
            "blocked_resource_stage_count": float(blocked_stage_count),
            "key_reach_before_lock": (
                torch.stack(key_scores).mean() if key_scores else zero
            ),
            "lock_reach_after_key": (
                torch.stack(lock_scores).mean() if lock_scores else zero
            ),
        }
        return distances, resource_loss, info

    def _soft_resource_gated_rollout(
        self,
        *,
        adjacency: Tensor,
        weights: Tensor,
        locked_edges: Tensor,
        source_mask: Tensor,
        key_lock_pairs: List[Tuple[int, int]],
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Differentiable ordered resource rollout with probabilistic lock opening.

        Providers assigned to the same target are conjunctive: their reachability
        probabilities are multiplied, and the edge is softened only after every
        declared provider has been evaluated. This preserves multi-token locks
        while allowing gradients to flow from later rooms back through earlier
        key-room passability.
        """
        device = adjacency.device
        zero = torch.zeros((), device=device, dtype=dtype)
        unlock_probability = torch.zeros_like(weights, dtype=dtype)
        remaining_by_target: Dict[int, int] = {}
        product_by_target: Dict[int, Tensor] = {}
        for _provider_idx, target_idx in key_lock_pairs:
            target = int(target_idx)
            remaining_by_target[target] = remaining_by_target.get(target, 0) + 1
            product_by_target.setdefault(target, torch.ones((), device=device, dtype=dtype))

        def _distances() -> Tensor:
            probability = torch.where(
                locked_edges,
                unlock_probability.clamp(1e-6, 1.0),
                torch.ones_like(unlock_probability),
            )
            gate_penalty = -torch.log(probability) * float(self.reachability.max_distance)
            staged_weights = weights + torch.where(
                locked_edges,
                gate_penalty.to(dtype=weights.dtype),
                torch.zeros_like(weights),
            )
            return self.graph_pathfinder(adjacency, staged_weights, source_mask)

        pair_losses: List[Tensor] = []
        key_scores: List[Tensor] = []
        lock_scores: List[Tensor] = []
        distances = _distances()
        matched_edge_count = 0
        for key_idx, lock_idx in key_lock_pairs:
            target = int(lock_idx)
            key_target = torch.zeros_like(source_mask)
            key_target[int(key_idx)] = 1.0
            key_score, key_loss = self.reachability(distances, key_target, return_loss=True)
            key_probability = key_score.mean().clamp(0.0, 1.0)
            product_by_target[target] = product_by_target[target] * key_probability
            remaining_by_target[target] -= 1
            key_scores.append(key_probability)

            lock_loss = zero
            if remaining_by_target[target] == 0:
                matching_edges = locked_edges & (
                    torch.arange(int(adjacency.shape[0]), device=device).view(1, -1)
                    == target
                )
                unlock_probability = torch.where(
                    matching_edges,
                    product_by_target[target].expand_as(unlock_probability),
                    unlock_probability,
                )
                matched_edge_count += int(matching_edges.sum().detach().item())
                distances = _distances()
                lock_target = torch.zeros_like(source_mask)
                lock_target[target] = 1.0
                lock_score, lock_loss = self.reachability(
                    distances,
                    lock_target,
                    return_loss=True,
                )
                lock_scores.append(lock_score.mean())
            pair_losses.append(0.5 * (key_loss + lock_loss))

        pair_loss_tensor = (
            torch.stack(pair_losses)
            if pair_losses
            else torch.empty(0, device=device, dtype=dtype)
        )
        resource_loss = pair_loss_tensor.mean() if pair_loss_tensor.numel() else zero
        unresolved = locked_edges & (unlock_probability < 0.5)
        info: Dict[str, Any] = {
            "num_violations": (
                int((pair_loss_tensor > 0.25).sum().detach().item())
                if pair_loss_tensor.numel()
                else 0
            ),
            "total_violation": resource_loss,
            "key_lock_mode": "resource_gated",
            "resource_gate_ordering": "soft_ordered",
            "locked_edge_count": float(locked_edges.float().sum().detach().item()),
            "matched_lock_edge_count": float(matched_edge_count),
            "remaining_locked_edge_count": float(unresolved.float().sum().detach().item()),
            "resource_stage_count": float(len(key_lock_pairs)),
            "blocked_resource_stage_count": float(
                sum(float(score.detach().item()) < 0.5 for score in key_scores)
            ),
            "key_reach_before_lock": torch.stack(key_scores).mean() if key_scores else zero,
            "lock_reach_after_key": torch.stack(lock_scores).mean() if lock_scores else zero,
            "mean_lock_open_probability": (
                unlock_probability[locked_edges].mean() if torch.any(locked_edges) else zero
            ),
        }
        return distances, resource_loss, info

    def _compute_one_global_graph_loss(
        self,
        *,
        node_count: int,
        edge_index: Optional[Tensor],
        adjacency: Optional[Tensor],
        edge_weights: Optional[Tensor],
        edge_features: Optional[Tensor],
        edge_attr: Optional[Tensor],
        node_features: Optional[Tensor],
        node_mask: Optional[Tensor],
        start_idx: Any,
        target_idx: Any,
        key_lock_pairs: Any,
        current_node_idx: Any,
        room_passability: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        n = int(max(0, node_count))
        if n <= 0:
            return zero, zero, zero, {}

        if isinstance(node_mask, torch.Tensor):
            mask = node_mask.to(device=device, dtype=torch.bool).flatten()
            if mask.numel() >= n:
                n = int(mask[:n].sum().item())
        if n <= 0:
            return zero, zero, zero, {}

        room_pass = room_passability.to(device=device, dtype=dtype).flatten()
        if room_pass.numel() != n and current_node_idx is None:
            return zero, zero, zero, {
                "global_graph_skipped": "room_passability_node_count_mismatch",
                "global_graph_room_passability_count": float(room_pass.numel()),
                "global_graph_node_count": float(n),
            }
        current_indices = self._current_indices_for_graph(
            current_node_idx,
            batch_size=int(room_pass.numel()),
            node_count=n,
            device=device,
        )
        node_passability = torch.ones(n, device=device, dtype=dtype)
        node_passability = node_passability.index_copy(0, current_indices, room_pass[: current_indices.numel()])

        adj, weights = self._build_adjacency_and_weights(
            node_count=n,
            device=device,
            dtype=dtype,
            edge_index=edge_index,
            adjacency=adjacency,
            edge_weights=edge_weights,
            edge_features=edge_features,
            edge_attr=edge_attr,
            node_passability=node_passability,
        )
        if adj is None or weights is None:
            return zero, zero, zero, {}
        locked_edges = self._locked_edge_mask(
            node_count=n,
            edge_index=edge_index,
            adjacency=adjacency,
            edge_features=edge_features,
            edge_attr=edge_attr,
            device=device,
        ) & (adj > 0)

        start = self._coerce_optional_int(start_idx)
        if start is None or start < 0 or start >= n:
            start = 0
        source_mask = torch.zeros(n, device=device, dtype=dtype)
        source_mask[start] = 1.0

        distances = self.graph_pathfinder(adj, weights, source_mask)

        pairs_were_supplied = key_lock_pairs is not None
        resource_pair_inference_ambiguous = False
        pairs: List[Tuple[int, int]] = []
        if isinstance(key_lock_pairs, (list, tuple)):
            for pair in key_lock_pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                k = self._coerce_optional_int(pair[0])
                lock_idx = self._coerce_optional_int(pair[1])
                if (
                    k is not None
                    and lock_idx is not None
                    and 0 <= k < n
                    and 0 <= lock_idx < n
                ):
                    pairs.append((k, lock_idx))
        if (
            not pairs_were_supplied
            and not pairs
            and isinstance(node_features, torch.Tensor)
            and node_features.dim() == 2
            and node_features.shape[1] > GRAPH_NODE_KEY_FEATURE_INDEX
        ):
            key_nodes = torch.nonzero(
                node_features[:n, GRAPH_NODE_KEY_FEATURE_INDEX] > 0.5,
                as_tuple=False,
            ).flatten().tolist()
            locked_targets: List[int] = []
            if isinstance(edge_features, torch.Tensor) and isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
                ef = edge_features
                if ef.dim() == 1:
                    ef = ef.unsqueeze(-1)
                if ef.shape[1] > GRAPH_EDGE_KEY_LOCK_FEATURE_INDEX:
                    locked = torch.nonzero(
                        ef[:, GRAPH_EDGE_KEY_LOCK_FEATURE_INDEX] > 0.5,
                        as_tuple=False,
                    ).flatten()
                    for edge_i in locked.tolist():
                        if edge_i < edge_index.shape[1]:
                            dst = int(edge_index[1, edge_i].item())
                            if 0 <= dst < n:
                                locked_targets.append(dst)
            unique_locked_targets = sorted(set(locked_targets))
            # Inference is only defensible when the assignment is unique. An
            # index-order zip across multiple keys and locks invents causal
            # supervision that is not present in the graph metadata.
            if len(key_nodes) == 1 and len(unique_locked_targets) == 1:
                pairs.append((int(key_nodes[0]), int(unique_locked_targets[0])))
            elif key_nodes or unique_locked_targets:
                resource_pair_inference_ambiguous = True

        lock_loss = zero
        lock_info: Dict[str, Any] = {}
        if torch.any(locked_edges):
            if not pairs:
                reason = (
                    "ambiguous provider-to-lock assignment"
                    if resource_pair_inference_ambiguous
                    else "missing provider-to-lock assignment"
                )
                raise LogicGraphContractError(
                    "LogicNet cannot supervise a graph with locked edges and "
                    f"{reason}. Supply explicit key_lock_pairs from the mission graph."
                )
            distances, lock_loss, lock_info = self._resource_gated_rollout(
                adjacency=adj,
                weights=weights,
                locked_edges=locked_edges,
                source_mask=source_mask,
                key_lock_pairs=pairs,
                dtype=dtype,
            )
        elif pairs:
            key_mask = torch.zeros(n, device=device, dtype=dtype)
            lock_mask = torch.zeros(n, device=device, dtype=dtype)
            for key_idx, lock_idx in pairs:
                key_mask[key_idx] = 1.0
                lock_mask[lock_idx] = 1.0
            lock_loss, lock_info = self.key_lock(distances, key_mask, lock_mask, pairs)
            lock_info["key_lock_mode"] = "distance_ordering"

        target = self._infer_target_idx(node_features, target_idx, node_count=n)
        graph_reach_loss = zero
        graph_reach_score = torch.tensor(0.0, device=device, dtype=dtype)
        if target is not None:
            target_mask = torch.zeros(n, device=device, dtype=dtype)
            target_mask[target] = 1.0
            graph_reach_score, graph_reach_loss = self.reachability(
                distances,
                target_mask,
                return_loss=True,
            )
            if isinstance(graph_reach_score, torch.Tensor) and graph_reach_score.numel() != 1:
                graph_reach_score = graph_reach_score.mean()

        room_loss = (1.0 - room_pass[: current_indices.numel()].clamp(0.0, 1.0)).mean()
        total = (
            self.global_reach_weight * graph_reach_loss
            + self.lock_weight * lock_loss
            + self.global_room_weight * room_loss
        )
        info = {
            "global_graph_distances": distances,
            "global_graph_reachability": graph_reach_score,
            "global_room_passability": room_pass.mean(),
            "global_room_loss": room_loss,
            "global_num_key_lock_pairs": float(len(pairs)),
            "resource_pairs_inferred": float(not pairs_were_supplied and bool(pairs)),
            "resource_pair_inference_ambiguous": float(resource_pair_inference_ambiguous),
            **lock_info,
        }
        return total, graph_reach_loss, lock_loss, info

    def _compute_global_graph_losses(
        self,
        graph_data: Optional[Any],
        *,
        room_passability: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        if not isinstance(graph_data, dict):
            return zero, zero, zero, {}

        adjacency = graph_data.get("adjacency")
        edge_index = graph_data.get("edge_index")
        node_features = graph_data.get("node_features")
        node_mask = graph_data.get("node_mask")
        graph_scope = str(graph_data.get("graph_scope", "") or "").strip().lower()

        if isinstance(adjacency, torch.Tensor) and adjacency.dim() == 2:
            n = int(adjacency.shape[0])
            if graph_scope == "dungeon" and int(room_passability.numel()) != n:
                return zero, zero, zero, {
                    "global_graph_skipped": "dungeon_scope_requires_full_room_passability",
                    "global_graph_room_passability_count": float(room_passability.numel()),
                    "global_graph_node_count": float(n),
                }
            node_feats = node_features.to(device=device, dtype=dtype) if isinstance(node_features, torch.Tensor) and node_features.dim() == 2 else None
            return self._compute_one_global_graph_loss(
                node_count=n,
                edge_index=None,
                adjacency=adjacency.to(device=device, dtype=dtype),
                edge_weights=graph_data.get("edge_weights"),
                edge_features=graph_data.get("edge_features"),
                edge_attr=graph_data.get("edge_attr"),
                node_features=node_feats,
                node_mask=node_mask if isinstance(node_mask, torch.Tensor) else None,
                start_idx=graph_data.get("start_idx", graph_data.get("start_node_id", 0)),
                target_idx=graph_data.get("target_idx"),
                key_lock_pairs=graph_data.get("key_lock_pairs"),
                current_node_idx=graph_data.get("current_node_idx"),
                room_passability=room_passability,
                device=device,
                dtype=dtype,
            )

        if not isinstance(edge_index, torch.Tensor):
            return zero, zero, zero, {}

        current_node_idx = graph_data.get("current_node_idx")
        room_batch = int(room_passability.numel())
        if isinstance(current_node_idx, torch.Tensor):
            idx_shape_valid = int(current_node_idx.numel()) == room_batch
        elif isinstance(current_node_idx, (list, tuple)):
            idx_shape_valid = len(current_node_idx) == room_batch
        else:
            idx_shape_valid = current_node_idx is not None and room_batch == 1

        if edge_index.dim() == 2 and isinstance(node_features, torch.Tensor) and node_features.dim() == 2:
            n = int(node_features.shape[0])
            full_node_batch = room_batch == n
            if graph_scope == "dungeon":
                if not full_node_batch:
                    return zero, zero, zero, {
                        "global_graph_skipped": "dungeon_scope_requires_full_room_passability",
                        "global_graph_room_passability_count": float(room_batch),
                        "global_graph_node_count": float(n),
                    }
            elif not idx_shape_valid:
                return zero, zero, zero, {}
            return self._compute_one_global_graph_loss(
                node_count=n,
                edge_index=edge_index.to(device=device, dtype=torch.long),
                adjacency=None,
                edge_weights=None,
                edge_features=graph_data.get("edge_features"),
                edge_attr=graph_data.get("edge_attr"),
                node_features=node_features.to(device=device, dtype=dtype) if isinstance(node_features, torch.Tensor) else None,
                node_mask=node_mask if isinstance(node_mask, torch.Tensor) else None,
                start_idx=graph_data.get("start_idx", graph_data.get("start_node_id", 0)),
                target_idx=graph_data.get("target_idx"),
                key_lock_pairs=graph_data.get("key_lock_pairs"),
                current_node_idx=current_node_idx,
                room_passability=room_passability,
                device=device,
                dtype=dtype,
            )

        # Batched room-scope: each item has its own graph and one current room.
        if edge_index.dim() != 3:
            return zero, zero, zero, {}

        losses: List[Tensor] = []
        reach_losses: List[Tensor] = []
        lock_losses: List[Tensor] = []
        infos: List[Dict[str, Any]] = []
        batch_size = min(int(edge_index.shape[0]), int(room_passability.numel()))
        for bi in range(batch_size):
            nf_i = self._select_batch_value(node_features, bi)
            if not isinstance(nf_i, torch.Tensor) or nf_i.dim() != 2:
                continue
            nm_i = self._select_batch_value(node_mask, bi)
            ei_i = edge_index[bi].to(device=device, dtype=torch.long)
            ef_i = self._select_batch_value(graph_data.get("edge_features"), bi)
            ea_i = self._select_batch_value(graph_data.get("edge_attr"), bi)
            total_i, reach_i, lock_i, info_i = self._compute_one_global_graph_loss(
                node_count=int(nf_i.shape[0]),
                edge_index=ei_i,
                adjacency=None,
                edge_weights=None,
                edge_features=ef_i if isinstance(ef_i, torch.Tensor) else None,
                edge_attr=ea_i if isinstance(ea_i, torch.Tensor) else None,
                node_features=nf_i.to(device=device, dtype=dtype),
                node_mask=nm_i if isinstance(nm_i, torch.Tensor) else None,
                start_idx=self._select_batch_value(graph_data.get("start_node_id", graph_data.get("start_idx", 0)), bi),
                target_idx=self._select_batch_value(graph_data.get("target_idx"), bi),
                key_lock_pairs=self._select_batch_value(graph_data.get("key_lock_pairs"), bi),
                current_node_idx=self._select_batch_value(graph_data.get("current_node_idx"), bi),
                room_passability=room_passability[bi:bi + 1],
                device=device,
                dtype=dtype,
            )
            losses.append(total_i)
            reach_losses.append(reach_i)
            lock_losses.append(lock_i)
            infos.append(info_i)

        if not losses:
            return zero, zero, zero, {}
        merged_info: Dict[str, Any] = {
            "global_graph_count": float(len(losses)),
            "global_room_passability": room_passability[:batch_size].mean(),
        }
        scalar_keys = (
            "global_graph_reachability",
            "global_room_loss",
            "global_num_key_lock_pairs",
            "locked_edge_count",
            "matched_lock_edge_count",
            "remaining_locked_edge_count",
            "resource_stage_count",
            "blocked_resource_stage_count",
            "key_reach_before_lock",
            "lock_reach_after_key",
            "mean_lock_open_probability",
            "missing_resource_pair_loss",
            "resource_pair_inference_ambiguous",
        )
        for key in scalar_keys:
            values = [info[key] for info in infos if key in info]
            if values:
                if isinstance(values[0], torch.Tensor):
                    merged_info[key] = torch.stack([v if v.dim() == 0 else v.mean() for v in values]).mean()
                else:
                    merged_info[key] = float(sum(float(v) for v in values) / len(values))
        return torch.stack(losses).mean(), torch.stack(reach_losses).mean(), torch.stack(lock_losses).mean(), merged_info
    
    def _sync_temperature_modules(self, temperature: float) -> None:
        """Apply one temperature to every differentiable planning component."""
        tau = float(temperature)
        self.graph_pathfinder.temperature = tau
        self.reachability.temperature = tau
        self.key_lock.temperature = tau
        if hasattr(self.grid_pathfinder, "temperature"):
            self.grid_pathfinder.temperature = tau
        nested_pathfinder = getattr(self.grid_pathfinder, "pathfinder", None)
        if nested_pathfinder is not None and hasattr(nested_pathfinder, "temperature"):
            nested_pathfinder.temperature = tau

    def _restore_temperature_after_load(self, module: nn.Module, _incompatible_keys: Any) -> None:
        """Restore plain-Python solver attributes from the checkpointed buffer."""
        if module is self:
            self._sync_temperature_modules(float(self.current_temperature.detach().item()))

    def update_temperature(self, progress: float):
        """
        Anneal soft-min temperature during training.
        
        Uses exponential decay from initial_temperature -> final_temperature.
        High temperature (start): smooth gradients, easy optimization.
        Low temperature (end): sharp soft-min ≈ true shortest path.
        
        Follows Gumbel-Softmax annealing (Jang et al., 2017; Maddison et al., 2017).
        
        Args:
            progress: Training progress in [0, 1] (0=start, 1=end)
        """
        progress = float(progress)
        if not math.isfinite(progress):
            raise ValueError("LogicNet temperature progress must be finite.")
        progress = max(0.0, min(1.0, progress))
        tau = self.initial_temperature * (
            self.final_temperature / self.initial_temperature
        ) ** progress
        
        self.current_temperature.fill_(tau)
        
        self._sync_temperature_modules(tau)

    def anneal_temperature(self, step_fraction: float):
        """Alias used by training scripts and experiment protocols."""
        self.update_temperature(step_fraction)
    
    def forward(
        self,
        z: Tensor,
        graph_data: Optional[Any] = None,
        goal_mask: Optional[Tensor] = None,
    ) -> Any:
        """
        Compute solvability loss for latent codes.
        
        Args:
            z: Latent codes [B, D, H, W]
            graph_data:
                - Compatibility mode: start mask tensor [B, H, W], paired with `goal_mask`
                - Graph mode dict with optional keys:
                  'adjacency' [N,N], 'edge_weights' [N,N], 'start_idx', 'target_idx',
                  and 'key_lock_pairs'
            goal_mask: Compatibility mode goal mask [B, H, W]
            
        Returns:
            loss: Scalar solvability loss
            info: Dict with detailed metrics
        """
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"LogicNet.forward expected z tensor, got {type(z).__name__}.")
        if z.dim() != 4:
            raise ValueError(f"LogicNet.forward expected z shape [B,D,H,W], got {tuple(z.shape)}.")
        if goal_mask is not None and not isinstance(goal_mask, torch.Tensor):
            raise TypeError(f"LogicNet.forward expected goal_mask tensor, got {type(goal_mask).__name__}.")
        if graph_data is not None and not isinstance(graph_data, (dict, torch.Tensor)):
            raise TypeError(
                f"LogicNet.forward expected graph_data dict/tensor/None, got {type(graph_data).__name__}."
            )

        # Backward-compatible inference mode:
        #   logic_net(tile_probs, start_mask, goal_mask) -> solvability scores [B]
        if isinstance(graph_data, torch.Tensor) and isinstance(goal_mask, torch.Tensor):
            start_mask = graph_data.float()
            goal = goal_mask.float()

            # BUG-06 fix: detect whether z contains tile probs (num_classes
            # channels) or latent codes (latent_dim channels). Route through
            # tile_classifier if needed.
            if z.shape[1] == self.num_classes:
                # z is already tile probs/logits -- use directly
                tile_logits = self._project_tile_logits_to_room(z)
                walkability = self.walkability(tile_logits, is_probs=None)
            else:
                # z is latent codes -- classify first, then lift to room size
                tile_logits = self.tile_classifier(z / self.latent_scale_factor)
                tile_logits = self._project_tile_logits_to_room(tile_logits)
                walkability = self.walkability(tile_logits, is_probs=False)
            if start_mask.dim() == 3:
                start_mask = start_mask.unsqueeze(1)
            if goal.dim() == 3:
                goal = goal.unsqueeze(1)
            distances = self.grid_pathfinder(tile_logits, start_mask, walkability)
            reach_scores = self.reachability(distances.flatten(1), goal.flatten(1))

            # Add a direct goal-region walkability term to keep gradients informative
            # in compatibility mode where inputs are already categorical probabilities.
            goal_mass = goal.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            goal_walkability = (walkability * goal).sum(dim=(1, 2, 3)) / goal_mass
            return (reach_scores + goal_walkability) * 0.5

        B = z.shape[0]
        device = z.device
        
        info = {}
        
        # 1. Resolve tile logits before walkability/pathfinding. Diffusion
        # training can pass decoder logits directly; raw continuous latents
        # still go through the learned classifier fallback.
        if int(z.shape[1]) == int(self.num_classes):
            latent_tile_logits = z
            tile_logits = self._project_tile_logits_to_room(z)
            info["logic_input_space"] = "tile_logits"
        else:
            latent_tile_logits = self.tile_classifier(z / self.latent_scale_factor)
            tile_logits = self._project_tile_logits_to_room(latent_tile_logits)
            info["logic_input_space"] = "latent"
        info['latent_tile_logits'] = latent_tile_logits
        info['tile_logits'] = tile_logits

        # 2. Predict walkability
        walkability = self.walkability(tile_logits, is_probs=False)
        info['walkability'] = walkability

        # 3. Compute within-room pathability
        room_logic_targets = self._resolve_room_logic_targets(
            graph_data,
            batch_size=B,
            device=device,
            dtype=walkability.dtype,
            spatial_hw=(int(walkability.shape[-2]), int(walkability.shape[-1])),
        )
        source_mask = room_logic_targets.get("source_mask")
        if source_mask is None:
            source_mask = self._create_single_cell_source_mask(walkability)
            info["source_mask_mode"] = "shape_center_fallback"
        else:
            info["source_mask_mode"] = "topology"

        grid_distances = self.grid_pathfinder(
            tile_logits,
            source_mask,
            walkability,
        )
        info['grid_distances'] = grid_distances
        
        # Grid-level reachability: can we traverse the room?
        # Use smooth sigmoid approximation of the hard threshold (walkability > 0.5)
        # to maintain differentiability. k=10 gives a steep but smooth step function.
        # (Bengio et al. 2013: smooth estimators for discrete latents)
        soft_walkable_mask = torch.sigmoid(10.0 * (walkability - 0.5))
        grid_target_mask = room_logic_targets.get("target_mask")
        if grid_target_mask is None:
            # Without topology there is no defensible start-goal claim. Use a
            # fixed coverage objective instead of letting the prediction define
            # its own target mask, which made all-wall rooms nearly loss-free.
            grid_target_mask = torch.ones_like(soft_walkable_mask)
            info["target_mask_mode"] = "room_coverage_fallback"
        else:
            info["target_mask_mode"] = "topology"
        grid_reach_scores, grid_reach_loss = self.reachability(
            grid_distances.reshape(B, -1),
            grid_target_mask.reshape(B, -1),
            return_loss=True,
        )
        info['grid_reachability'] = grid_reach_scores.mean()

        topology_trace_loss = torch.tensor(0.0, device=device)
        trace_target = room_logic_targets.get("trace_target")
        if trace_target is not None:
            trace_mass = trace_target.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            topology_trace_loss = (
                ((1.0 - walkability) * trace_target).sum(dim=(1, 2, 3)) / trace_mass
            ).mean()

        topology_anchor_loss = torch.tensor(0.0, device=device)
        anchor_target = room_logic_targets.get("anchor_target")
        if anchor_target is not None:
            anchor_mass = anchor_target.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            topology_anchor_loss = (
                ((1.0 - walkability) * anchor_target).sum(dim=(1, 2, 3)) / anchor_mass
            ).mean()
        
        # 4. Graph-level pathfinding. Room predictions are lifted to node
        # passability, then used as differentiable entry costs on the mission
        # graph. This gives dungeon-scope reachability and key-lock losses a
        # real gradient path back to the room latents in z.
        room_passability = self._room_passability_from_local_scores(
            walkability=walkability,
            grid_reach_scores=grid_reach_scores,
            trace_target=trace_target,
            anchor_target=anchor_target,
        )
        graph_total_loss, graph_reach_loss, lock_loss, global_info = self._compute_global_graph_losses(
            graph_data,
            room_passability=room_passability,
            device=device,
            dtype=walkability.dtype,
        )

        # 5. Combine losses
        loss = (
            self.reach_weight * grid_reach_loss
            + self.topology_trace_weight * topology_trace_loss
            + self.topology_anchor_weight * topology_anchor_loss
            + graph_total_loss
        )

        info['grid_reach_loss'] = grid_reach_loss
        info['graph_reach_loss'] = graph_reach_loss
        info['lock_loss'] = lock_loss
        info['global_logic_loss'] = graph_total_loss
        info['room_passability'] = room_passability.mean()
        info['topology_trace_loss'] = topology_trace_loss
        info['topology_anchor_loss'] = topology_anchor_loss
        info.update(global_info)
        info['total_loss'] = loss
        
        return loss, info
    
    def _create_door_source_mask(
        self,
        batch_size: int,
        device: torch.device,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """Create source mask at door positions for grid pathfinding."""
        height, width = spatial_hw if spatial_hw is not None else (ROOM_HEIGHT, ROOM_WIDTH)
        height = int(max(1, height))
        width = int(max(1, width))
        mask = torch.zeros(batch_size, 1, height, width, device=device)

        for direction, spec in DOOR_POSITIONS.items():
            row_slice, col_slice = self._door_slices_for_shape(direction, height, width)
            mask[:, :, row_slice, col_slice] = 1.0

        return mask

    def _create_single_cell_source_mask(
        self,
        walkability: Tensor,
    ) -> Tensor:
        """Create a deterministic fallback source when topology is unavailable."""
        if walkability.dim() != 4 or int(walkability.shape[1]) != 1:
            raise ValueError(f"walkability must be [B,1,H,W], got {tuple(walkability.shape)}.")
        B, _C, H, W = walkability.shape
        mask = torch.zeros(B, 1, H, W, device=walkability.device, dtype=walkability.dtype)
        mask[:, 0, H // 2, W // 2] = 1.0
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
