#!/usr/bin/env python
"""Benchmark dense and edge-sparse LogicNet graph planning on paired graphs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logic_net import (  # noqa: E402
    DifferentiablePathfinder,
    SparseDifferentiablePathfinder,
)


def _build_graph(
    node_count: int,
    extra_edges_per_node: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacency = torch.zeros(node_count, node_count, device=device)
    chain = torch.arange(node_count - 1, device=device)
    adjacency[chain, chain + 1] = 1.0
    if extra_edges_per_node > 0 and node_count > 2:
        src = torch.arange(node_count - 1, device=device).repeat_interleave(extra_edges_per_node)
        jumps = torch.randint(
            1,
            node_count,
            (int(src.numel()),),
            generator=generator,
            device=device,
        )
        dst = torch.minimum(src + jumps, torch.full_like(src, node_count - 1))
        valid = src != dst
        adjacency[src[valid], dst[valid]] = 1.0
    weights = adjacency * (
        0.5
        + torch.rand(
            node_count,
            node_count,
            generator=generator,
            device=device,
        )
    )
    source = torch.zeros(node_count, device=device)
    source[0] = 1.0
    return adjacency, weights, source


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    repeats: int,
    warmup: int,
    device: torch.device,
) -> Dict[str, Any]:
    for _ in range(warmup):
        with torch.no_grad():
            model(*inputs)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.no_grad():
            output = model(*inputs)
        _synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    assert output is not None
    return {
        "median_ms": float(statistics.median(timings)),
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
        "peak_allocated_bytes": peak_bytes,
        "output": output.detach().cpu(),
    }


def _gradient_deviation(
    dense: DifferentiablePathfinder,
    sparse: SparseDifferentiablePathfinder,
    adjacency: torch.Tensor,
    weights: torch.Tensor,
    source: torch.Tensor,
) -> float:
    dense_weights = weights.detach().clone().requires_grad_(True)
    sparse_weights = weights.detach().clone().requires_grad_(True)
    dense(adjacency, dense_weights, source)[-1].backward()
    sparse(adjacency, sparse_weights, source)[-1].backward()
    edge_mask = adjacency.bool()
    return float(
        (dense_weights.grad[edge_mask] - sparse_weights.grad[edge_mask])
        .abs()
        .max()
        .detach()
        .cpu()
        .item()
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    rows = []
    for node_count in args.node_counts:
        adjacency, weights, source = _build_graph(
            int(node_count),
            int(args.extra_edges_per_node),
            generator=generator,
            device=device,
        )
        common = {
            "num_iterations": int(args.num_iterations),
            "temperature": float(args.temperature),
            "full_coverage": True,
            "convergence_tolerance": 0.0,
        }
        dense = DifferentiablePathfinder(**common).to(device)
        sparse = SparseDifferentiablePathfinder(**common).to(device)
        dense_result = _measure(
            dense,
            (adjacency, weights, source),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            device=device,
        )
        sparse_result = _measure(
            sparse,
            (adjacency, weights, source),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            device=device,
        )
        output_delta = float(
            (dense_result.pop("output") - sparse_result.pop("output")).abs().max().item()
        )
        rows.append(
            {
                "node_count": int(node_count),
                "directed_edge_count": int(adjacency.sum().item()),
                "dense": dense_result,
                "sparse": sparse_result,
                "max_output_abs_delta": output_delta,
                "max_edge_gradient_abs_delta": _gradient_deviation(
                    dense,
                    sparse,
                    adjacency,
                    weights,
                    source,
                ),
            }
        )
    return {
        "device": str(device),
        "seed": int(args.seed),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "temperature": float(args.temperature),
        "num_iterations_requested": int(args.num_iterations),
        "full_coverage": True,
        "results": rows,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-counts", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--extra-edges-per-node", type=int, default=2)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/logicnet_graph_pathfinder_benchmark.json"),
    )
    args = parser.parse_args(argv)
    if any(int(value) < 2 for value in args.node_counts):
        parser.error("Every node count must be at least 2.")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be non-negative.")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
