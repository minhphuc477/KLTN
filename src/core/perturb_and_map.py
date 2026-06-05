"""Core Perturb-and-MAP pathfinding surrogate for LogicNet ablations."""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

import torch
from torch import Tensor


def _grid_neighbors(row: int, col: int, height: int, width: int):
    if row > 0:
        yield row - 1, col
    if row + 1 < height:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < width:
        yield row, col + 1


def _dijkstra_tree(cost: Tensor, traversable: Tensor, source: Tensor) -> Tuple[Tensor, Tensor]:
    """Hard Dijkstra solve returning distances and shortest-tree support."""
    height, width = int(cost.shape[0]), int(cost.shape[1])
    inf = 1e6
    distances = torch.full((height, width), inf, dtype=torch.float32)
    support = torch.zeros((height, width), dtype=torch.float32)
    source_cells = torch.nonzero((source > 0.5) & traversable, as_tuple=False)
    if source_cells.numel() == 0:
        return distances, support

    frontier: List[Tuple[float, int, int]] = []
    for cell in source_cells:
        row, col = int(cell[0]), int(cell[1])
        distances[row, col] = 0.0
        support[row, col] = 1.0
        heapq.heappush(frontier, (0.0, row, col))

    while frontier:
        current_cost, row, col = heapq.heappop(frontier)
        if current_cost > float(distances[row, col]) + 1e-6:
            continue
        for nr, nc in _grid_neighbors(row, col, height, width):
            if not bool(traversable[nr, nc]):
                continue
            next_cost = current_cost + float(cost[nr, nc])
            if next_cost < float(distances[nr, nc]):
                distances[nr, nc] = float(next_cost)
                support[nr, nc] = 1.0
                heapq.heappush(frontier, (next_cost, nr, nc))
    return distances, support


class DifferentiablePerturbedAStar(torch.autograd.Function):
    """
    Hard stochastic shortest-path forward with a straight-through support gradient.

    This is a biased surrogate by design. It gives LogicNet a hard-solver
    ablation without pretending that Python A*/Dijkstra is exactly
    differentiable.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        walkability: Tensor,
        source_mask: Tensor,
        num_samples: int,
        noise_scale: float,
        obstacle_penalty: float,
        blocked_threshold: float,
    ) -> Tensor:
        if walkability.dim() != 4 or int(walkability.shape[1]) != 1:
            raise ValueError(f"walkability must be [B,1,H,W], got {tuple(walkability.shape)}.")
        if source_mask.shape != walkability.shape:
            raise ValueError(
                f"source_mask shape {tuple(source_mask.shape)} must match walkability {tuple(walkability.shape)}."
            )

        device = walkability.device
        dtype = walkability.dtype
        walk_cpu = walkability.detach().float().cpu().clamp(0.0, 1.0)
        source_cpu = source_mask.detach().float().cpu().clamp(0.0, 1.0)
        B, _C, H, W = walk_cpu.shape
        samples = int(max(1, num_samples))
        noise = float(max(0.0, noise_scale))
        penalty = float(max(0.0, obstacle_penalty))
        threshold = float(blocked_threshold)

        all_distances = torch.zeros((B, H, W), dtype=torch.float32)
        all_support = torch.zeros((B, H, W), dtype=torch.float32)
        for batch_idx in range(B):
            walk = walk_cpu[batch_idx, 0]
            source = source_cpu[batch_idx, 0]
            traversable = walk > threshold
            base_cost = 1.0 + penalty * (1.0 - walk)
            sample_dist = []
            sample_support = []
            for _ in range(samples):
                if noise > 0.0:
                    uniform = torch.rand((H, W), dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
                    gumbel = -torch.log(-torch.log(uniform)) * noise
                    cost = (base_cost + gumbel).clamp_min(1e-4)
                else:
                    cost = base_cost
                distances, support = _dijkstra_tree(cost, traversable, source)
                sample_dist.append(distances)
                sample_support.append(support)
            mean_dist = torch.stack(sample_dist, dim=0).mean(dim=0)
            mean_support = torch.stack(sample_support, dim=0).mean(dim=0)
            all_distances[batch_idx] = mean_dist
            all_support[batch_idx] = mean_support

        ctx.save_for_backward(all_support.to(device=device, dtype=dtype))
        return all_distances.unsqueeze(1).to(device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: Optional[Tensor]):  # type: ignore[override]
        (support,) = ctx.saved_tensors
        if grad_output is None:
            return None, None, None, None, None, None
        # Lower queried distances should be achieved by increasing walkability
        # on cells that the hard solver actually used across perturbed solves.
        # grad_output is usually non-zero only at target/goal cells, so using it
        # elementwise would zero out every intermediate path cell. Route the
        # per-sample scalar distance signal onto the support tree instead.
        routed_signal = grad_output.sum(dim=(1, 2, 3), keepdim=True)
        grad_walkability = -routed_signal * support.unsqueeze(1)
        return grad_walkability, None, None, None, None, None


def perturb_and_map_distance(
    walkability: Tensor,
    source_mask: Tensor,
    *,
    num_samples: int = 8,
    noise_scale: float = 0.25,
    obstacle_penalty: float = 8.0,
    blocked_threshold: float = 0.05,
) -> Tensor:
    """Return a hard stochastic distance field with straight-through gradients."""
    return DifferentiablePerturbedAStar.apply(
        walkability,
        source_mask,
        int(num_samples),
        float(noise_scale),
        float(obstacle_penalty),
        float(blocked_threshold),
    )


PerturbAndMAPDistanceFunction = DifferentiablePerturbedAStar
