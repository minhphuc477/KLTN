"""Benchmark graph-to-grid attention and topology-refinement ablation costs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.common import json_ready, set_reproducible_seed
from src.core import ROOM_HEIGHT, ROOM_WIDTH
from src.core.graph_grid_attention import GraphToGridCrossAttention
from src.core.latent_diffusion import CrossAttention


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    return 0.0


def _parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in str(value).split(",") if str(item).strip()]


def _parse_csv_strings(value: str) -> List[str]:
    return [str(item).strip() for item in str(value).split(",") if str(item).strip()]


@torch.no_grad()
def bench_mode(
    *,
    mode: str,
    nodes: int,
    repeats: int,
    device: torch.device,
    grid_dim: int,
    graph_dim: int,
    heads: int,
) -> Dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    module = GraphToGridCrossAttention(
        grid_dim=grid_dim,
        graph_dim=graph_dim,
        num_heads=heads,
        dropout=0.0,
        attention_mode=mode,
        auto_linear_attention_nodes=0,
    ).to(device).eval()
    grid = torch.randn(1, grid_dim, ROOM_HEIGHT, ROOM_WIDTH, device=device)
    graph = torch.randn(1, int(nodes), graph_dim, device=device)
    pos = torch.rand(1, int(nodes), 2, device=device)
    tpe = torch.rand(1, int(nodes), 8, device=device)
    mask = torch.ones(1, int(nodes), device=device)
    edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(int(repeats)):
        y = module(grid, graph, edge_index=edge_index, node_positions=pos, node_tpe=tpe, node_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - start) / max(1, int(repeats))
    return {
        "attention_mode": mode,
        "nodes": int(nodes),
        "grid_tokens": int(ROOM_HEIGHT * ROOM_WIDTH),
        "score_pairs": int(ROOM_HEIGHT * ROOM_WIDTH * int(nodes)) if mode == "softmax" else 0,
        "linear_tokens": int(ROOM_HEIGHT * ROOM_WIDTH + int(nodes)) if mode == "linear_hedgehog" else 0,
        "seconds_per_forward": float(seconds),
        "peak_memory_mb": _peak_memory_mb(device),
        "finite_output": bool(torch.isfinite(y).all().item()),
        "output_shape": list(y.shape),
    }


def topology_cost_rows(*, nodes: int, edges: int, modes: List[str]) -> List[Dict[str, Any]]:
    """Static topology-refinement cost metrics used to screen large-graph ablations."""
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        metrics = CrossAttention.topology_refinement_metrics(
            num_nodes=int(nodes),
            num_edges=int(edges),
            mode=str(mode),
        )
        rows.append(
            {
                "topology_refinement_mode": str(mode),
                "nodes": int(nodes),
                "edges": int(edges),
                **{key: float(value) for key, value in metrics.items()},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=str, default="10,50,128,300")
    parser.add_argument("--topology-edges", type=str, default="9,49,127,299")
    parser.add_argument(
        "--topology-modes",
        type=str,
        default="gat2,sparse_edge,sparse_directed,sparse_directed_semantic,graphormer",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid-dim", type=int, default=64)
    parser.add_argument("--graph-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/outputs/ablation_gat_attention.json")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    node_counts = [10, 32] if args.dry_run else [int(item) for item in args.nodes.split(",") if item]
    edge_counts = [9, 31] if args.dry_run else _parse_csv_ints(args.topology_edges)
    if len(edge_counts) == 1 and len(node_counts) > 1:
        edge_counts = edge_counts * len(node_counts)
    if len(edge_counts) != len(node_counts):
        raise ValueError("--topology-edges must have one value or the same number of values as --nodes.")
    topology_modes = _parse_csv_strings(args.topology_modes)
    repeats = 1 if args.dry_run else int(args.repeats)
    rows: List[Dict[str, Any]] = []
    topology_rows: List[Dict[str, Any]] = []
    for n in node_counts:
        for mode in ("softmax", "linear_hedgehog"):
            rows.append(
                bench_mode(
                    mode=mode,
                    nodes=n,
                    repeats=repeats,
                    device=device,
                    grid_dim=args.grid_dim,
                    graph_dim=args.graph_dim,
                    heads=args.heads,
                )
            )
    for n, e in zip(node_counts, edge_counts):
        topology_rows.extend(topology_cost_rows(nodes=n, edges=e, modes=topology_modes))
    result = {"config": vars(args), "results": rows, "topology_costs": topology_rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    print(json.dumps(json_ready(result), indent=2))


if __name__ == "__main__":
    main()
