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
    """Hard Dijkstra solve returning distances and predecessor indices."""
    height, width = int(cost.shape[0]), int(cost.shape[1])
    inf = 1e6
    distances = torch.full((height, width), inf, dtype=torch.float32)
    parents = torch.full((height, width), -1, dtype=torch.long)
    source_cells = torch.nonzero((source > 0.5) & traversable, as_tuple=False)
    if source_cells.numel() == 0:
        return distances, parents

    frontier: List[Tuple[float, int, int]] = []
    for cell in source_cells:
        row, col = int(cell[0]), int(cell[1])
        distances[row, col] = 0.0
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
                parents[nr, nc] = row * width + col
                heapq.heappush(frontier, (next_cost, nr, nc))
    return distances, parents


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
        parent_samples: List[List[Tensor]] = []
        distance_samples: List[List[Tensor]] = []
        active_cost_samples: List[List[Tensor]] = []
        for batch_idx in range(B):
            walk = walk_cpu[batch_idx, 0]
            source = source_cpu[batch_idx, 0]
            traversable = walk > threshold
            base_cost = 1.0 + penalty * (1.0 - walk)
            sample_dist = []
            sample_parents: List[Tensor] = []
            sample_active_cost: List[Tensor] = []
            for _ in range(samples):
                if noise > 0.0:
                    uniform = torch.rand((H, W), dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
                    gumbel = -torch.log(-torch.log(uniform)) * noise
                    raw_cost = base_cost + gumbel
                    cost = raw_cost.clamp_min(1e-4)
                    active_cost = raw_cost > 1e-4
                else:
                    cost = base_cost
                    active_cost = torch.ones_like(cost, dtype=torch.bool)
                distances, parents = _dijkstra_tree(cost, traversable, source)
                sample_dist.append(distances)
                sample_parents.append(parents)
                sample_active_cost.append(active_cost)
            mean_dist = torch.stack(sample_dist, dim=0).mean(dim=0)
            all_distances[batch_idx] = mean_dist
            parent_samples.append(sample_parents)
            distance_samples.append(sample_dist)
            active_cost_samples.append(sample_active_cost)

        ctx.parent_samples = parent_samples
        ctx.distance_samples = distance_samples
        ctx.active_cost_samples = active_cost_samples
        ctx.num_samples = samples
        ctx.obstacle_penalty = penalty
        ctx.input_device = device
        ctx.input_dtype = dtype
        return all_distances.unsqueeze(1).to(device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: Optional[Tensor]):  # type: ignore[override]
        if grad_output is None:
            return None, None, None, None, None, None
        grad_cpu = grad_output.detach().float().cpu()
        batch_size, _channels, height, width = grad_cpu.shape
        grad_walkability = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
        sample_scale = 1.0 / float(ctx.num_samples)

        for batch_idx in range(batch_size):
            for sample_idx in range(ctx.num_samples):
                parents = ctx.parent_samples[batch_idx][sample_idx].reshape(-1)
                distances = ctx.distance_samples[batch_idx][sample_idx].reshape(-1)
                active_cost = ctx.active_cost_samples[batch_idx][sample_idx].reshape(-1)
                accumulated = grad_cpu[batch_idx, 0].reshape(-1).clone() * sample_scale
                grad_cost = torch.zeros_like(accumulated)

                # A predecessor always has lower distance, so reverse distance
                # order propagates every target's adjoint back to its source.
                for flat_idx in torch.argsort(distances, descending=True).tolist():
                    if float(distances[flat_idx]) >= 1e6:
                        continue
                    parent_idx = int(parents[flat_idx])
                    if parent_idx < 0:
                        continue
                    if bool(active_cost[flat_idx]):
                        grad_cost[flat_idx] += accumulated[flat_idx]
                    accumulated[parent_idx] += accumulated[flat_idx]

                grad_walkability[batch_idx, 0] -= (
                    float(ctx.obstacle_penalty) * grad_cost.reshape(height, width)
                )

        grad_walkability = grad_walkability.to(
            device=ctx.input_device,
            dtype=ctx.input_dtype,
        )
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
